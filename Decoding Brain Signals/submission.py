import sys, os, shutil, numpy as np
import pickle

mode = sys.argv[1]
folder = sys.argv[2]
if mode == 'create':
    os.mkdir(folder)
        
    shutil.copytree('Python27', os.path.join(folder, 'Python27'))
    
    # Since it's not possible to use Python 3 in Azure ML experiments and
    # there are some differences in pickling between Python 2 and 3,
    # we need a workaround to transform weights matrices in suitable format
    
    for p in ['p1', 'p2', 'p3', 'p4']:
        i=0
        while True:
            try:
                weights = list(np.load(p+'_'+str(i)+'.npz')['data'])
                with open(os.path.join(folder, p+'_'+str(i)+'.pkl'), 'wb') as f:
                    pickle.dump(weights, f)
                i+=1
            except:
                break
                  
    # -----
    
    shutil.make_archive(folder, 'zip', folder)
    shutil.rmtree(folder)
    
elif mode == 'remove':
    os.remove(folder+'.zip')
        
                  
    
        
    

