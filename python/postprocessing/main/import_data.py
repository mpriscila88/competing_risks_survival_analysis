'''
Functions to import dataset 
'''

import numpy as np
import pandas as pd
import random


# IMPORT DATASET FUNCTIONS

def import_datasets(path, outcome):
    
    import sys
    import os
    sys.path.insert(0, path) # insert path

    # Import the dataset # removed events with duration inferior to 1 day for each outcome - thus, different encounters
   
    df_train = pd.read_csv(os.path.join(path, outcome + '_X_train_mice.csv'), index_col=0, sep=',')   
    df_test = pd.read_csv(os.path.join(path, outcome + '_X_test_mice.csv'), index_col=0, sep=',')    
     
    # Competing risk
    
    if outcome != 'Death':
        #censored if death occurs
        df_train[outcome][(df_train[outcome] == 0) & (df_train['Death'] == 1)] = 2
        df_test[outcome][(df_test[outcome] == 0) & (df_test['Death'] == 1)] = 2
     
    vars_ = ['A1C', 'ALT', 
             'AST', 'HDL', 'LDL', 
             'Albumin',
             'SystolicBP', 'DiastolicBP', 'Temperature', 'HeartRate', 'Weight',
             'RespirationRate', 'Height', 
             'BMI',
             'vitals_missing', 'labs_missing',
             'Union', 
             'Active',
             'Smoker',
             'AlcoholDrinksPerWeek', 
             'Diabetes_comorbidity', 'Hypertension_comorbidity', 
             'AGE_STRATA', 'Female',
             'event','duration','Age']
    
    if outcome == 'ADMCIDementia':
        vars_.append('Ischemic_stroke_comorbidity')
        vars_.append('Intracranial_hemorrhage_comorbidity')
        vars_.append('Depression_comorbidity')
      
         
    elif outcome == 'Depression':
        vars_.append('ADMCIDementia_comorbidity')
        vars_.append('Ischemic_stroke_comorbidity')
        vars_.append('Intracranial_hemorrhage_comorbidity')
    
         
    elif outcome == 'Ischemic_stroke':
        vars_.append('Intracranial_hemorrhage_comorbidity')
        vars_.append('ADMCIDementia_comorbidity')
        vars_.append('Depression_comorbidity')
        
     
    elif outcome == 'Intracranial_hemorrhage':
        vars_.append('Ischemic_stroke_comorbidity')
        vars_.append('ADMCIDementia_comorbidity')
        vars_.append('Depression_comorbidity')
   
         
    elif outcome == 'Death':
        vars_.append('Ischemic_stroke_comorbidity')
        vars_.append('Intracranial_hemorrhage_comorbidity')
        vars_.append('ADMCIDementia_comorbidity')
        vars_.append('Depression_comorbidity')
        
       
   
    data_aux_tr = df_train.rename(columns={outcome:"event","Survival_days_"+outcome:"duration"})
       
    data_aux_test = df_test.rename(columns={outcome:"event","Survival_days_"+outcome:"duration"})
    
    # Time in years
    data_aux_tr['duration'] = data_aux_tr['duration']/365.25
    
    data_aux_test['duration'] = data_aux_test['duration']/365.25


    # Input dimension
    feat_list = data_aux_tr[vars_].drop(columns=['event','duration','AGE_STRATA']).columns.values


    
    return  feat_list, data_aux_tr, data_aux_test
