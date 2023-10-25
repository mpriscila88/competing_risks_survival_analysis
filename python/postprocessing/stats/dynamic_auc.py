# -*- coding: utf-8 -*-

import os
import scipy.io as sio
import numpy as np
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc

outcomes = ['Intracranial_hemorrhage','ADMCIDementia','Ischemic_stroke', 
            'Depression','Death']


for outcome in outcomes:    
    
    result_folder = "C:/directory/" + outcome
    
    result_mat = sio.loadmat(f'{result_folder}/coxph_scores_{outcome}.mat')

    
    if outcome != 'Death':
        cox_outcome_idx = [result_mat['survstates'][i,0][0] for i in range(len(result_mat['survstates']))].index('event1')
        survprob = pd.DataFrame(result_mat['survprob'][len(result_mat['survprob'])-1,:,cox_outcome_idx])[0]
    else:
        survprob = pd.DataFrame(1-result_mat['survprob'][len(result_mat['survprob'])-1,:])[0]
     
   
    y_test =  pd.read_csv(os.path.join(result_folder+ '/te_curves.csv'), sep=',')

    y_test.event[y_test.event == 1] = True
    y_test.event[y_test.event == 0] = False
    y_test.event[y_test.event == 2] = False
     
    y_train = pd.read_csv(os.path.join(result_folder+ '/tr_curves.csv'), sep=',')
    
    y_train.event[y_train.event == 1] = True
    y_train.event[y_train.event == 0] = False
    y_train.event[y_train.event == 2] = False

    y_train.duration[y_train.duration>np.max(y_test.duration)] = np.max(y_test.duration) #- 1e-10
    y_train.duration[y_train.duration<np.min(y_test.duration)] = np.min(y_test.duration) #+ 1e-10
    
    
    # Times <t> at which to calculate the AUC
    va_times = [1,5,10]
    
    # where max(<t>) is chosen arbitrarily and < of follow-up time# Risk scores <f(xi)> on test data
    cph_risk_scores = survprob
    
    
    # Use a compound data type for structured arrays
    tr = np.zeros(len(y_train), dtype={'names':('event', 'duration'),
                              'formats':('bool', '<f8')})
    
    tr['event'] = y_train.event
    tr['duration'] = y_train.duration
    
    te = np.zeros(len(y_test), dtype={'names':('event', 'duration'),
                              'formats':('bool', '<f8')})
    te['event'] = y_test.event
    te['duration'] = y_test.duration
    
    
    # AUC at times <t> and its average
    cph_auc, tpr, fpr = cumulative_dynamic_auc(tr, te, cph_risk_scores, va_times)
    
    print(outcome)
    print(cph_auc)
    print(tpr)
    print(fpr)
