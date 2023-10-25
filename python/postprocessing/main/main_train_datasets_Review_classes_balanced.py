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
        
    file_path = path + outcome

    data_aux_test =  pd.read_csv(file_path + '/te_data_full.csv')

    data_aux_test = data_aux_test.reset_index(drop=True)

    class0 = data_aux_test[data_aux_test['event'] == 0].reset_index().rename(columns={'index':'ind'})
    class1 = data_aux_test[data_aux_test['event'] == 1].reset_index().rename(columns={'index':'ind'})
    
    if outcome != 'Death':
        
        class2 = data_aux_test[data_aux_test['event'] == 2].reset_index().rename(columns={'index':'ind'})
          
    # '''
    #     Random number generation settings
    # '''
    seed                        = 1234 # Random seed for dataset splits (training, validation, testing)
    rs_seed                     = 1234 # Random seed for generating parameter sets for random search
    
    # ######### START 5-FOLD CV 
    
    # Randomly select one entry per MRN from train/test split (seed per iteration)
    def cv_get_minibatch(data_aux, seed):
    
        np.random.seed(seed)
    
        data_a = data_aux

        size = 1        # sample size
        replace = False  # with replacement
        fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
        data_a = data_a.groupby('MRN', as_index=False).apply(fn)
        
        idx = data_a.ind
        return idx
    
  
    ###########################################################################
    # Bootstrapping
    #--------------------------

    # configure bootstrap
    n_iterations = 1000
     
     
    # run bootstrap
    iteration = 0
    
    
    for i in range(0,1000):
        

        #### PREPARE TEST BATCH    
        index0 = cv_get_minibatch(class0, seed = seed + iteration)
        index1 = cv_get_minibatch(class1, seed = seed + iteration)
    
        # select random indexes from majority classes

        
    
        if outcome != 'Death':
            index2 = cv_get_minibatch(class2, seed = seed + iteration)
            
            if len(index1) < len(index2):
                size = len(index1)  # sample size
            else:
                size = len(index2)  # sample size
        else:
            size = len(index1)  # sample size
            
        replace = False     # with replacement
        np.random.seed(0)
        
        index0 = np.random.choice(pd.DataFrame(index0).ind, size, replace)
        
        if outcome != 'Death':
            if len(index1) < len(index2):
                np.random.seed(0)
                index2 = np.random.choice(pd.DataFrame(index2).ind, size, replace)
            else:    
                index1 = np.random.choice(pd.DataFrame(index1).ind, size, replace)
            
        if outcome != 'Death':
            index = pd.concat([pd.Series(index0),  pd.Series(index1), pd.Series(index2)], axis = 0).reset_index(drop=True)
            index = pd.DataFrame(index, columns=['ind'])
        else:
            index = pd.concat([pd.Series(index0),  pd.Series(index1)], axis = 0).reset_index(drop=True)
        
        file_path_save = file_path + '/index_review/'
        if not os.path.exists(file_path_save):
                os.makedirs(file_path_save)
                
        pd.DataFrame(index).to_csv(file_path_save +'index_' + str(iteration) + '.csv', index=False) 
         
        iteration += 1
        
        print(iteration)
    
    
    # Test final model with bootstrapping hold-out test set in R

##############################################################################  

