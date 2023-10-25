# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 30})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


outcomes = ['Intracranial_hemorrhage','ADMCIDementia','Ischemic_stroke', 
            'Depression','Death']


for outcome in outcomes:    

    result_folder = "C:/directory/" + outcome
    result_mat = sio.loadmat(f'{result_folder}/coxph_scores_{outcome}.mat')
    df = pd.read_csv(os.path.join(result_folder+ '/te_curves.csv'), sep=',')  
    
    survtime = result_mat['survtime']
    
    if outcome != 'Death':
        cox_outcome_idx = [result_mat['survstates'][i,0][0] for i in range(len(result_mat['survstates']))].index('event1')
        survprob = pd.DataFrame(result_mat['survprob'][len(result_mat['survprob'])-1,:,cox_outcome_idx])[0]
    else:
        survprob = pd.DataFrame(1-result_mat['survprob'][len(result_mat['survprob'])-1,:])[0]
        
    # hfont = {'fontname':'Cambria'}
    # plt.figure()
    # ax = survprob.plot.hist(color='#607c8e', grid=True, alpha=0.7) #, rwidth=0.85)
    # ax.grid(axis='y', alpha=0.75)
    # ax.set_xlabel('Score (%)',**hfont)
    # ax.set_ylabel('Frequency',**hfont)
    # ax.set_title(outcome,**hfont) # Cumulative incidence risk score at the end of the study period from baseline visit
    # ax.set_xlim([0,1])
    # ax.set_xticklabels((0, 50, 100),fontsize=30,fontname='Cambria')
    
    hfont = {'fontname':'Cambria'}
    plt.rcParams["font.family"] = "Cambria"
    plt.figure(figsize=(9,6))
    sns.histplot(survprob.loc[df['event'] == 1]*100, color="k", label='Non-censored',alpha=0.5, kde_kws={'linewidth':3})
    sns.histplot(survprob.loc[df['event'] == 0]*100, color="steelblue", label='Censored', alpha=0.3, kde_kws={'linewidth':3})
    if outcome != 'Death':
        sns.histplot(survprob.loc[df['event'] == 2]*100, color="indianred", label="Deceased",alpha=0.3, kde_kws={'linewidth':3})
    plt.legend(prop={'size': 22},frameon=True, loc='upper right')
    # plt.xlim([0, 100])
    # plt.xticks((0, 50, 100),fontsize=30,fontname='Cambria')
    plt.xlabel('Score (%)',**hfont) 
    #plt.title(outcome)
    plt.rcParams["font.family"] = "Cambria"
  
    plt.show()
 
