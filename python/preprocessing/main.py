# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys

path = 'C:/directory/'
sys.path.insert(0, path) # insert path


########################################
# Cohort Sleep Lab Patients preprocessed
#---------------------------------------

e = pd.read_csv(os.path.join(path,'SleepLabCohort_IBCS.csv')) 

###################################
# Extend event
#----------------------------------

from ProlongEvent import  prolong_event

cols = ['Ischemic_stroke', 'Intracranial_hemorrhage', 'Depression', 
        'AD', 'MCI', 'Dementia']

e = prolong_event(e, cols)

###################################
# Time to Event
#----------------------------------

from Time2Event import define_dataset

e = define_dataset(e)

################################################
# Comorbidities in correspondent encounter
#-----------------------------------------------

outcomes = ['Intracranial_hemorrhage','Depression', 'Ischemic_stroke',
             'ADMCIDementia','Diabetes','Hypertension']
 
# Assign based on first diagnosis
for out in outcomes:
     e[out+'_comorbidity'] = e[out]
     e[out+'_comorbidity'][(e[out] == 1) & (e.ContactDTS < e[out+'_Event'])] = 0
          
################################################
# Patients with survival days > 1 day
#-----------------------------------------------

outcomes_ = ['Intracranial_hemorrhage','Depression',
             'Ischemic_stroke','ADMCIDementia','Death']
mrns = e
for outcome in outcomes_:
     mrns = mrns[mrns['Survival_days_' + outcome] > 1]
     e = e[e.MRN.isin(mrns.MRN)]
     mrns = e

e = mrns
 
################################################
# Train/test splitting
#-----------------------------------------------

from GetTrainTest import train_test_norm

train_test_norm(e, path)

###############################################
# Missing data imputation with MICE
#-----------------------------------------------

from MICE import create_mice_model

create_mice_model(path)

###################################
# Survival analysis - R studio
#----------------------------------

# Run
# main_train_datasets.py
# R studio survival modeling

###################################
# Plots for appendix
#----------------------------------

outcomes = ['Death','Intracranial_hemorrhage','Depression',
            'Ischemic_stroke','ADMCIDementia','Diabetes', 'Hypertension']

from scipy.stats import iqr
from scipy import stats
import matplotlib.pyplot as plt
hfont = {'fontname':'Cambria'}
fsize = 26

for o in outcomes:
    b = e[e[o] == 1].drop_duplicates()
    b = b[['MRN','Age']].drop_duplicates()
    b = b.groupby('MRN').Age.min()
    print(o)
    plt.figure()
    ax = b.hist()
    ax.set_ylabel('Frequency',**hfont, fontsize=fsize)
    ax.set_xlabel('Age at baseline (years)',**hfont, fontsize=fsize)

    c = e[e[o] == 0].drop_duplicates()
    c = c[['MRN','Age']].drop_duplicates()
    c = c.groupby('MRN').Age.min()
    plt.figure()
    ax = c.hist()
    ax.set_ylabel('Frequency',**hfont, fontsize=fsize)
    ax.set_xlabel('Age at baseline (years)',**hfont, fontsize=fsize)


####

data = e[['A1C', 'ALT', 'AST', 'HDL', 'LDL', 'Albumin', 'SystolicBP',
       'DiastolicBP', 'Temperature', 'HeartRate', 'RespirationRate', 'BMI',
       'AlcoholDrinksPerWeek', 'Female', 'Age', 'vitals_missing',
       'labs_missing', 'Union', 'Active', 'Smoker', 'Diabetes_comorbidity',
       'Hypertension_comorbidity']]

data = data.rename(columns={"SystolicBP":"Systolic BP",
                                "DiastolicBP":"Diastolic BP",
                                "HeartRate":"Heart Rate",
                                "RespirationRate":"Respiration Rate",
                                "AlcoholDrinksPerWeek":'Alcohol drinks per week', 
                                "vitals_missing":'Vital signs not measured', 
                                "labs_missing":'Laboratory values not measured', 
                                "Diabetes_comorbidity":"Diabetes",
                                "Hypertension_comorbidity":"Hypertension"
                                })
    
import missingno as msno
import matplotlib.pyplot as plt
    
hfont = {'fontname':'Cambria'}
ax=msno.bar(data, figsize=(12, 6), fontsize=12, color='steelblue')
ax.set_ylabel('Data availability (%)',**hfont)
ax.set_yticklabels((0, 20, 40, 60, 80, 100),fontsize=12,fontname='Cambria')
ax.set_xticklabels((data.columns), fontname='Cambria')

ax2 = ax.twinx()
ax2.set_yticklabels(labels='',fontsize=12,fontname='Cambria')
ax2.set_ylabel('Number of visits',**hfont)
ax2.yaxis.set_label_coords(1.1,0.5)


##############################################################################
