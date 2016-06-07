import os, pickle
import os.path
import pkg_resources
import sys
print(sys.version_info)
SCRIPT_DIR = os.path.join(os.getcwd(), 'Script Bundle')
#SCRIPT_DIR = os.path.join(os.getcwd())
SITE_DIR = os.path.join(SCRIPT_DIR, 'Python27', 'site-packages')
siteEnv = pkg_resources.Environment([SITE_DIR])
packages, pkgErrors = pkg_resources.working_set.find_plugins(siteEnv)
for pkg in packages:
    pkg_resources.working_set.add(pkg) # add plugins+libs to sys.path
if pkgErrors:
   print ("Couldn't load", pkgErrors) # display errors
# Set Theano variable to disable warning messages
os.environ['THEANO_FLAGS'] = 'cxx='
os.environ['OMP_NUM_THREADS'] = '4'

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.regularizers import l2
import pandas as pd, numpy as np
from keras.layers.noise import GaussianNoise
from scipy.stats import gmean

sys.path.append(r".\\Script Bundle\\")

def preprocessing(data):
    patients_data = {p: {} for p in ['p1', 'p2', 'p3', 'p4']}
    data = data[data.Stimulus_ID != -1]

    for p in ['p1', 'p2', 'p3', 'p4']:
        pat = data[data.PatientID == p]
        pat = pat.select(lambda x: pat[x].iloc[0]!=-999999, axis=1)

        patients_data[p]['ids'] = list(pat[pat.Stimulus_Type != 101].Stimulus_ID.unique())
        num_epoch = len(patients_data[p]['ids'])
        
        channels = pat.shape[1] - 3
        time = 800
        pat_data = patients_data[p]['data'] = np.zeros((num_epoch, 1, channels, time)) 

        for epoch, id in enumerate(patients_data[p]['ids']):
            if id>200:
                epoch_data = pat[pat.Stimulus_ID == id].iloc[:time, 1:-2]
                pat_data[epoch, 0] = ((epoch_data - np.mean(epoch_data, axis=0))/np.std(epoch_data, axis=0)).T
            
    return patients_data
            
def load_model(filename, shape):
    model = Sequential()
    model.add(Convolution2D(5, shape[1], 1, input_shape=shape, activation='tanh', init='normal'))
    model.add(Convolution2D(5, 1, 10, subsample=(1, 5), activation='tanh', init='normal'))
    model.add(Flatten())
    model.add(Dense(10, activation='sigmoid', init='normal', W_regularizer=l2(0.01), 
                    b_regularizer=l2(1.0)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', init='normal'))
    
    with open(os.path.join(SCRIPT_DIR, filename+'.pkl'), 'rb') as f:
        weights = pickle.load(f) 
    # weights = list(np.load(os.path.join(SCRIPT_DIR, patient+'.npz'))['data'])
    model.set_weights(weights)
    # print('MODEL WAS BUILT')
    # model.load_weights(os.path.join(SCRIPT_DIR, patient+'.model'))
    # print('WEIGHTS')
    return model
    
windows = {'p1': [[200, 800], [300, 800], [600, 700], [0, 800]],
           'p2': [[100, 800], [200, 700], [600, 700], [0, 800]],
           'p3': [[200, 700], [200, 800], [400, 700], [500, 800], [0, 800]],
           'p4': [[100, 800], [300, 700], [500, 800], [0, 800]]}
           
def get_models(patient, shape):
    models = []
    for i, w in enumerate(windows[patient]):
        model = load_model(patient+'_'+str(i), (shape[0], shape[1], w[1]-w[0]))
        models.append(model)
    return models

def predict(patient, models, data):
    probs = np.zeros((data.shape[0], len(windows[patient])))
    for i, w in enumerate(windows[patient]):
        probs[:, i] = models[i].predict_proba(data[:, :, :, w[0]:w[1]], verbose=0, batch_size=data.shape[0]).flatten()
            
    return (gmean(probs, axis=1)>0.5).astype(int)
        
def get_predictions(data):
    patients_data = preprocessing(data)
    
    res = pd.DataFrame(columns=['PatientID', 'Stimulus_ID', 'Scored Labels'])
    for p in ['p1', 'p2', 'p3', 'p4']:
        pat_data = patients_data[p]
        models = get_models(p, pat_data['data'].shape[1:-1])
        print(models[2].summary())
        ans = predict(p, models, pat_data['data'])+1
        res = res.append(pd.DataFrame.from_dict({
                'PatientID': [p] *  len(pat_data['ids']),
                'Stimulus_ID': pat_data['ids'],
                'Scored Labels': list(ans)
            }), ignore_index=True)
        
    res.Stimulus_ID = res.Stimulus_ID.astype(int)
    res['Scored Labels'] = res['Scored Labels'].astype(int)
    return res
    
def azureml_main(dataset1):
    res = get_predictions(dataset1)
    return res