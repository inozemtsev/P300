def load_trials_from_mat(path, mode='remove_empty'):
    from scipy.io import loadmat
    from multiprocessing import Pool
    import numpy as np

    data = loadmat(path)
    ntrials = data['number_of_trials'][0, 0]
    keys = [key for key in data.keys() if 'shape' in dir(data[key]) and data[key].shape[1] != 1]

    trial_nums = [num for num in range(ntrials) if np.prod(np.in1d(data['spike_channel'][0, num], data['channels'][0, num])) != 0]
                  
    if mode == 'all':
        pass
    elif mode == 'remove_empty':
        trial_nums = [num for num in trial_nums if data['spike_channel'][0, num].shape[0] != 0]

    raw_trials = [{key: data[key][0, num].flatten() for key in keys} for num in trial_nums]

    pool = Pool(8)
    trials = pool.map(Trial.from_dict, raw_trials)
    pool.close()
    pool.join()
    return Trials(trials)

def load_trials_from_pickle(path):
    import pickle
    with open(path, 'rb') as f:
        trials = pickle.load(f)
    return trials

class Trials:
    def __init__(self, trials_list):
        self._trials = trials_list
        self.ntrials = len(trials_list)
        
    def __getitem__(self, index):
        return self._trials[index]
    
    def __add__(self, other):
        return Trials(self._trials + other._trials)
        
    def to_pickle(self, path):
        import pickle, copy
        
        self_copy = copy.copy(self)
        for trial in self_copy._trials:
            trial.spikes = trial.spikes.to_sparse(fill_value=0)
            
        with open(path, 'wb') as f:
            pickle.dump(self_copy._trials, f)
            
    def get_trials_by_tasks(self, tasks):
        new_trials = [trial for trial in self._trials if trial.task in tasks]
        return Trials(new_trials)
    
    def get_trials_by_state(self, states):
        new_trials = [trial for trial in self._trials if trial.state in states]
        return Trials(new_trials)
    
    def _get_spikes(trial, begin, end):
        return trial.get_spikes_between_events(begin=begin, end=end)
    
    def get_spikes_between_events(self, begin=None, end=None):
        from multiprocessing import Pool
        from itertools import repeat
        
        pool = Pool(8)
        params = zip(self._trials, repeat(begin), repeat(end))
        new_trials = pool.starmap(Trials._get_spikes, params)
        pool.close()
        pool.join()
        trials = [trial for trial in new_trials if trial is not None]
        return Trials(trials)
    
class Trial:  
    def __init__(self, spikes,  rotation, events, expected_location, initial_location, direction, 
                 task, user_interfered, state):
        self.spikes = spikes
        self.events = events
        self.expected_location = expected_location
        self.initial_location = initial_location
        self.direction = direction
        self.task = task
        self.user_interfered = user_interfered
        self.rotation = rotation   
        self.state = state
    
    def get_spikes_between_events(self, begin=None, end=None):
        import numpy as np
        
        if begin is None:
            start = self.spikes.index[0]
        else:
            start = self.events.loc[begin][0]
        
        if end is None:
            end = self.spikes.index[-1]
        else:
            end = self.events.loc[end][0]
            
        if not np.isnan(start) and not np.isnan(end):
            import copy
            trial_copy = copy.copy(self)
            try:
                trial_copy.spikes = self.spikes.loc[start:end]
            except Exception as e:
                print(e)
                print(self.spikes, start, end, self.events, self.spikes.index)
            return trial_copy
        else:
            return None
    
    def from_dict(trial_dict):
        import pandas as pd
        import numpy as np
        
        spikes_num = trial_dict['spike_channel'].shape[0]
        spikes = np.zeros((spikes_num, trial_dict['channels'].shape[0]))
        spikes = pd.DataFrame(spikes, index=trial_dict['spike_time'], columns=trial_dict['channels']).sort_index()
        
        for index, col in enumerate(trial_dict['spike_channel']):
            col_index = spikes.columns.tolist().index(col)
            spikes.iloc[index, col_index] = 1

        event_names = ['bar_press_onset', 'fix_spot_on','fixation1_onset','target_on','target_start_move',
                       'target_stop_move', 'target_blink','saccade_detected','fixation2_onset',
                       'fix_spot_dim','bar_press_offset', 'reward']
        events = {ev_name: trial_dict[ev_name][0] for ev_name in event_names}
        events = pd.DataFrame.from_dict(events, orient='index').loc[event_names]
        events.columns = ['time']
        
        rotation = np.column_stack((trial_dict['increment_time'], trial_dict['increment']))
        rotation = pd.Series(trial_dict['increment'], index=trial_dict['increment_time'])
        
        expected_location = trial_dict['expected_location'][0]
        initial_location = trial_dict['cond_num'][0]
        direction = trial_dict['direction'][0]
        
        task_types = ['stimulus','memory','stimulus & memory','central dim','stim-nm','mem-nm','stim&mem-nm']
        task = int(trial_dict['stimulus_type'][0])
        if task < len(task_types):
            task = task_types[task]
        
        user_interfered = trial_dict['user_interfered'][0]
        state = trial_dict['the_state'][0]

        return Trial(spikes, rotation, events, expected_location, initial_location, direction, 
                     task, user_interfered, state)