# -*- coding: utf-8 -*-

'''
Main file to train models

TRAIN WITH 5 FOLD CROSS VALIDATION
'''

# Import functions
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import os
import sys
path = 'C:/directory/'
sys.path.insert(0, path) # insert path
import import_data as impt


##### SETTINGS
'''
    Load data and data settings
'''

outcomes = ['Intracranial_hemorrhage','ADMCIDementia','Ischemic_stroke', 
            'Depression','Death']

for outcome in outcomes:
        
    full_feat_list, data_aux_tr, data_aux_test = impt.import_datasets(path, outcome)
     
    file_path = path + outcome
    if not os.path.exists(file_path):
            os.makedirs(file_path)
    
    data_aux_tr[['MRN','PatientEncounterID','event','duration']].to_csv(file_path  + '/tr_data_full.csv', index=False) 
    
    # features list         
    pd.DataFrame(full_feat_list).to_csv(file_path  + '/full_feat_list.csv', index=False)
    
    # Survival curves
    full_feat_list = list(full_feat_list)
    full_feat_list.append('MRN') 
    full_feat_list.append('event')
    full_feat_list.append('duration') 

    size = 1        # sample size
    replace = False  # with replacement
    np.random.seed(0)
    fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
    np.random.seed(0)
    X_train = data_aux_tr[full_feat_list].groupby('MRN', as_index=False).apply(fn)
    np.random.seed(0)
    X_test = data_aux_test[full_feat_list].groupby('MRN', as_index=False).apply(fn)
 
    X_train.to_csv(file_path  + '/tr_curves.csv', index=False) 
    X_test.to_csv(file_path  + '/te_curves.csv', index=False) 
    
    print(outcome)
    np.random.seed(0)
    X_train = data_aux_tr.groupby('MRN', as_index=False).apply(fn)
    np.random.seed(0)
    X_test = data_aux_test.groupby('MRN', as_index=False).apply(fn)
    
    X_train.Visit_year.hist()
    X_test.Visit_year.hist()
    
    # # Save full external test set
    te_data = data_aux_test[full_feat_list]
       
    te_data.to_csv(file_path  + '/te_data_full.csv', index=False) 
  

    # '''
    #     Random number generation settings
    # '''
    seed                        = 1234 # Random seed for dataset splits (training, validation, testing)
    rs_seed                     = 1234 # Random seed for generating parameter sets for random search
    
    # ######### START 5-FOLD CV 
    
    # Randomly select one entry per MRN from train/test split (seed per iteration)
    def cv_get_minibatch(data_aux, x, seed):
    
        np.random.seed(seed)
    
        data_a = data_aux
        data_a = data_a.reset_index()
        data_a = data_a[data_a.index.isin(x)]
        data_a = data_a.drop(columns='index')
        
        size = 1        # sample size
        replace = False  # with replacement
        fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
        data_a = data_a.groupby('MRN', as_index=False).apply(fn)
        
        data_a = data_a.reset_index()
        idx = data_a.level_1
    
        return idx
    
    ###########################################################################
    # Bootstrapping
    #--------------------------

    # configure bootstrap
    n_iterations = 1000
    
    # Create batches
    kf = KFold(n_splits=n_iterations, shuffle=True, random_state=seed)
    
    # run bootstrap
    iteration = 0
    for set_index1, set_index2 in kf.split(data_aux_test):
        
        #### PREPARE TEST BATCH    
        set_index = pd.concat([pd.Series(set_index1), pd.Series(set_index2)],axis=0)
        index = cv_get_minibatch(data_aux_test, set_index, seed = seed + iteration)
    
        pd.DataFrame(index).to_csv(file_path + '/index_' + str(iteration) + '.csv', index=False) 
         
        iteration += 1
        
        print(iteration)
    
    
    # Test final model with bootstrapping hold-out test set in R

##############################################################################  

