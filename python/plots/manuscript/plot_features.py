# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 22})



outcomes = ['Intracranial_hemorrhage','ADMCIDementia','Ischemic_stroke', 
            'Depression','Death']

for outcome in outcomes:

  path = "C:/directory/" + outcome
  
  a = pd.read_csv(os.path.join(path, 'coef_' + outcome + '_CoxPH_CompetingRisk.csv'), index_col=0, sep=',')   
      
  if outcome != 'Death':
      a = a[a.index.astype(str).str.contains('1:2')]
      a.index = a.index.astype(str).str.replace('_1:2','')
  
  if outcome == 'Death':
      a = a[~(a.index.astype(str).str.contains('Smoker'))]
    
  min_val, max_val = min(a.coef), max(a.coef)


  from operator import itemgetter
  indices, L_sorted = zip(*sorted(enumerate(a.coef), key=itemgetter(1)))
  
  var = pd.DataFrame(a.index[pd.DataFrame(indices)[0]],columns=['Features'])
                                
  var = pd.DataFrame(var, columns=['Features'])
  # cb.ax.set_yticklabels(var.Features)
 
  var[var.Features == 'AlcoholDrinksPerWeek'] = 'Alcohol drinks per week'
  var[var.Features == 'HeartRate'] = 'Heart rate'
  var[var.Features == 'RespirationRate'] = 'Respiration rate'
  var[var.Features == 'SystolicBP'] = 'Systolic blood pressure'
  var[var.Features == 'ADMCIDementia_comorbidity'] = 'Composite of dementia'
  var[var.Features == 'Ischemic_stroke_comorbidity'] = 'Ischemic stroke'
  var[var.Features == 'Intracranial_hemorrhage_comorbidity'] = 'Intracranial_hemorrhage'
  var[var.Features == 'Hypertension_comorbidity'] = 'Hypertension'
  var[var.Features == 'Diabetes_comorbidity'] = 'Diabetes'
  var[var.Features == 'Depression_comorbidity'] = 'Depression'
  var[var.Features == 'vitals_missing'] = 'Vital signs not measured'
  var[var.Features == 'labs_missing'] = 'Laboratory values not measured'
  var[var.Features == 'Ischemic_stroke'] = 'Ischemic stroke'
  var[var.Features == 'Intracranial_hemorrhage'] = 'Intracranial hemorrhage'
  var[var.Features == 'ADMCIDementia'] = 'Composite of Dementia'
  
  
  data_x = var.Features #[0,1,2,3]
  data_hight = sorted(a.coef) #[60,60,80,100]

  data_hight_normalized = [x / max(data_hight) for x in data_hight]
   
  if (outcome == 'Intracranial_hemorrhage') | (outcome == 'Ischemic_stroke'):
      size = 4
  elif outcome == 'Depression':
      size = 5
  else:
      size = 7
       
  
  fig, ax = plt.subplots(figsize=(12, size))

  my_cmap = plt.cm.get_cmap('coolwarm')
  colors = my_cmap(data_hight_normalized)

  rects = ax.barh(data_x, data_hight, color='steelblue')
  
  plt.grid(color='grey', linestyle=':', linewidth=1)
  plt.xlabel('CoxPH coefficients')
 # plt.title(outcome)
  
  plt.rcParams["font.family"] = "Cambria"
  plt.savefig(os.path.join(path,f'features_{outcome}.png'), bbox_inches='tight', pad_inches=0.01)
          
   
  
  
