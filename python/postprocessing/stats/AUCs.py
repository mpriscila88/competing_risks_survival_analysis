# -*- coding: utf-8 -*-

import os
import scipy.io as sio
import numpy as np
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc

outcomes = ['Intracranial_hemorrhage','ADMCIDementia','Ischemic_stroke', 
            'Depression','Death']


# for outcome in outcomes:    

outcome = 'ADMCIDementia'
    
result_folder = "C:/directory/" + outcome

data_aux_train =  pd.read_csv(result_folder + '/tr_curves.csv')

data_aux_train = data_aux_train.reset_index(drop=True)

data_aux_test =  pd.read_csv(result_folder + '/te_data_full.csv')

data_aux_test = data_aux_test.reset_index(drop=True)

cph_aucs = []
tprs = [] 
fprs = []

for boot in range(0,200):
    
    print(boot)
    
    result_mat = sio.loadmat(f'{result_folder}/coxph_scores_boot{boot}.mat')

    
    if outcome != 'Death':
        cox_outcome_idx = [result_mat['survstates'][i,0][0] for i in range(len(result_mat['survstates']))].index('event1')
        survprob = pd.DataFrame(result_mat['survprob'][len(result_mat['survprob'])-1,:,cox_outcome_idx])[0]
    else:
        survprob = pd.DataFrame(1-result_mat['survprob'][len(result_mat['survprob'])-1,:])[0]
     
    # load test data for boot
    
    te_indexes =  pd.read_csv(result_folder + '/index/index_' + str(boot) + '.csv') #.rename(columns={'0':'ind'})
     
    y_test = data_aux_test.loc[te_indexes.level_1,:]

    y_test.event[y_test.event == 1] = True
    y_test.event[y_test.event == 0] = False
    y_test.event[y_test.event == 2] = False
     
    y_train = data_aux_train 
    
    y_train.event[y_train.event == 1] = True
    y_train.event[y_train.event == 0] = False
    y_train.event[y_train.event == 2] = False

    y_train.duration[y_train.duration>np.max(y_test.duration)] = np.max(y_test.duration)
    y_train.duration[y_train.duration<np.min(y_test.duration)] = np.min(y_test.duration)
    
    
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
    
    # if len(cph_risk_scores) != len(te):
        
    #     if len(cph_risk_scores) > len(te):
    #         cph_risk_scores = cph_risk_scores.iloc[:-1]
    #     else:
    #         te = te[:-1]
    
    if len(cph_risk_scores) == len(te):
        # AUC at times <t> and its average
        cph_auc, tpr, fpr = cumulative_dynamic_auc(tr, te, cph_risk_scores, va_times)
        
        if cph_auc != 'error':
            cph_aucs.append(cph_auc)
            tprs.append(tpr)
            fprs.append(fpr)

cph_aucs = pd.DataFrame(cph_aucs)        
tprs = pd.DataFrame(tprs) 
fprs = pd.DataFrame(fprs) 

cph_aucs = pd.DataFrame(cph_aucs)        
tprs = pd.DataFrame(tprs) 
fprs = pd.DataFrame(fprs) 

#cph_aucs.to_csv(os.path.join(result_folder,'cph_aucs.csv'))
#tprs.to_csv(os.path.join(result_folder,'tprs.csv'))        
#fprs.to_csv(os.path.join(result_folder,'fprs.csv')) 

# confidence intervals
def ci(stats):
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    # lower = max(0.0, np.percentile(stats, p))
    lower = np.percentile(stats, p)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    # upper = min(1.0, np.percentile(stats, p))
    upper = np.percentile(stats, p)
    # print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower, upper))
    return lower, upper


for i in range(0,3):
    lower, upper = ci(cph_aucs.loc[:,i])  
    print(np.mean(cph_aucs.loc[:,i]))
    print(lower, upper)
    
for i in range(0,3):
    lower, upper = ci(tprs.loc[:,i])  
    print(np.mean(tprs.loc[:,i]))
    print(lower, upper)
       
for i in range(0,3):
    lower, upper = ci(fprs.loc[:,i])  
    print(np.mean(fprs.loc[:,i]))
    print(lower, upper)
    # plt.xlabel('Follow-up Time')
    # plt.ylabel('Dynamic AUC')# Draw textbox with average AUC
    # textbox = 'Average: {:.3f}'.format(cph_mean_auc)
    # props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.9)
    # # plt.text(1100, 0.55, textbox, fontsize = 18, bbox = props)
    # plt.grid(True)
