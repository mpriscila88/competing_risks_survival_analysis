# -*- coding: utf-8 -*-

# Import functions
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
path = 'C:/directory/'
sys.path.insert(0, path) # insert path

##### SETTINGS
'''
    Load data and data settings
'''

outcomes = ['Intracranial_hemorrhage','ADMCIDementia','Ischemic_stroke', 
            'Depression','Death']

for outcome in outcomes:
        
    df_train = pd.read_csv(os.path.join(path, outcome + '/tr_curves.csv'),sep=',')   
    df_test = pd.read_csv(os.path.join(path, outcome + '/te_curves.csv'), sep=',')  
    
    df = pd.concat([df_train,df_test], axis=0)


    plt.figure(figsize=(9,6))

    sns.histplot(df.loc[df['event'] == 1, "duration"], color="k", label='Non-censored',alpha=0.5, kde_kws={'linewidth':3})
    sns.histplot(df.loc[df['event'] == 0, "duration"], color="steelblue", label='Censored', alpha=0.3, kde_kws={'linewidth':3})
    if outcome != 'Death':
        sns.histplot(df.loc[df['event'] == 2, "duration"], color="indianred", label="Deceased",alpha=0.3, kde_kws={'linewidth':3})
    plt.legend(prop={'size': 22},frameon=True, loc='upper right')
    plt.xlim([0, 11])
    plt.xticks((0, 2,4,6,8,10),fontsize=30,fontname='Cambria')
    plt.xlabel('Survival Time (years)')   
    plt.rcParams["font.family"] = "Cambria"
  
    plt.show()
