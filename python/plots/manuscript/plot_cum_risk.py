# -*- coding: utf-8 -*-

import os
# import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 30})
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


outcomes = ['Intracranial_hemorrhage','ADMCIDementia','Ischemic_stroke', 
            'Depression','Death']

tr = 0
for outcome in outcomes:    
    
    result_folder = "C:/directory/" + outcome

    if tr == 1:
        result_mat = sio.loadmat(f'{result_folder}/survival_curves_tr_{outcome}.mat')
        df = pd.read_csv(os.path.join(result_folder, 'tr_curves.csv'), sep=',')   
        
    else:
        result_mat = sio.loadmat(f'{result_folder}/survival_curves_te_{outcome}.mat')
        df = pd.read_csv(os.path.join(result_folder, 'te_curves.csv'), sep=',')   
     
    predicted_curve_names = ['mean(z)', 'mean(z)-stdev(z)', 'mean(z)+stdev(z)', '0%', '0.5%','1%', '2.5%', '10%', '25%', '50%', '75%', '90%', '97.5%', '99%', '99.5%', '100%']
    survtime = result_mat['survtime']
    
    if outcome != 'Death':
        cox_outcome_idx = [result_mat['survstates'][i,0][0] for i in range(len(result_mat['survstates']))].index('event1')
        survprob_mean = result_mat['survprob'][:,predicted_curve_names.index('mean(z)'),cox_outcome_idx]
        survprob_mean_m_std = result_mat['survprob'][:,predicted_curve_names.index('mean(z)-stdev(z)'),cox_outcome_idx]
        survprob_mean_p_std = result_mat['survprob'][:,predicted_curve_names.index('mean(z)+stdev(z)'),cox_outcome_idx]
    else:
        survprob_mean = 1-result_mat['survprob'][:,predicted_curve_names.index('mean(z)')]
        survprob_mean_m_std = 1-result_mat['survprob'][:,predicted_curve_names.index('mean(z)-stdev(z)')]
        survprob_mean_p_std = 1-result_mat['survprob'][:,predicted_curve_names.index('mean(z)+stdev(z)')]
       
    
    if outcome != 'Death':
         label_mean='Cox PH with competing risk: mean(z)'
         label_m='Cox PH with competing risk: mean(z)-stdev(z)'
         label_p='Cox PH with competing risk: mean(z)+stdev(z)'
    else:
         label_mean='Cox PH: mean(z)'
         label_m='Cox PH: mean(z)-stdev(z)'
         label_p='Cox PH: mean(z)+stdev(z)'
         aj_outcome_idx = 1
     
    

    fig = plt.figure(figsize=(10,7))
    
    ax = fig.add_subplot(111)
    if tr == 1:
        aj_mat = sio.loadmat(f'{result_folder}/AJ_output_tr_{outcome}_mean.mat')
    else:
        aj_mat = sio.loadmat(f'{result_folder}/AJ_output_te_{outcome}_mean.mat')
   
    if outcome != 'Death':
        aj_outcome_idx = [aj_mat['states'][i,0][0] for i in range(len(aj_mat['states']))].index('event1')
        ax.step(aj_mat['time'], aj_mat['val'][:,aj_outcome_idx]*100, ls='--', c='k', label='Aalen-Johansen: medium risk')
    else:
        ax.step(aj_mat['time'], aj_mat['val'][:,aj_outcome_idx]*100, ls='--', c='k', label='Aalen-Johansen: medium risk')   
   
    ax.step(survtime, survprob_mean*100, c='k', lw = 4, label='CoxPH: medium risk')
   
    if tr == 1:
          aj_mat = sio.loadmat(f'{result_folder}/AJ_output_tr_{outcome}_mean-std.mat')
    else:
          aj_mat = sio.loadmat(f'{result_folder}/AJ_output_te_{outcome}_mean-std.mat')
    
    if outcome != 'Death':
        aj_outcome_idx = [aj_mat['states'][i,0][0] for i in range(len(aj_mat['states']))].index('event1')
        ax.step(aj_mat['time'], aj_mat['val'][:,aj_outcome_idx]*100, ls='--', c='steelblue', label='Aalen-Johansen: low risk')
    else: 
        ax.step(aj_mat['time'], aj_mat['val'][:,aj_outcome_idx]*100, ls='--', c='steelblue', label='Aalen-Johansen: low risk')
    
    ax.step(survtime, survprob_mean_m_std*100, c='steelblue', lw = 4, label='CoxPH: low risk')
   
    if tr == 1:
         aj_mat = sio.loadmat(f'{result_folder}/AJ_output_tr_{outcome}_mean+std.mat')
    else:
         aj_mat = sio.loadmat(f'{result_folder}/AJ_output_te_{outcome}_mean+std.mat')
    aj_outcome_idx = [aj_mat['states'][i,0][0] for i in range(len(aj_mat['states']))].index('event1')
    ax.step(aj_mat['time'], aj_mat['val'][:,aj_outcome_idx]*100, ls='--', c='indianred', label='Aalen-Johansen: high risk')
    ax.step(survtime, survprob_mean_p_std*100, c='indianred', lw = 4, label='CoxPH: high risk')
        
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_xlim([0, 11])
    ax.set_ylim([0, 55])
    ax.legend(frameon=False, loc='upper left',  prop={'size': 22,'family': 'Cambria'})
    #plt.legend(prop={'family': 'Cambria'})
    ax.set_xlabel('Time (years)', fontname='Cambria')
    ax.set_ylabel(f'Cumulative risk (%)',fontname='Cambria')
    plt.rcParams["font.family"] = "Cambria"
    seaborn.despine()
    
   
