
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
from lifelines import KaplanMeierFitter
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


outcomes = ['Intracranial_hemorrhage', 'Depression',
            'Ischemic_stroke','ADMCIDementia','Death']

for outcome in outcomes:    

    result_folder = "C:/directory/" + outcome
    
    df_p = []
    df_m = []
    
    for boot in range(0,1000):
        
        result_mat = sio.loadmat(f'{result_folder}/boot/bootstrapp_te_review{boot}.mat')
    
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
            
            
        years = [1,2,3,4,5,6,7,8,9,10,11]
        rr_m_stds = []
        rr_p_stds = []
        for year in years:
            idx = np.argmin(np.abs(survtime-year))
            rr_m_stds.append( survprob_mean_m_std[idx] / survprob_mean[idx] )
            rr_p_stds.append( survprob_mean_p_std[idx] / survprob_mean[idx] )
      
       
        df_p.append(rr_p_stds)
        df_m.append(rr_m_stds)
        
        if boot == 999:
            
            df_p = pd.DataFrame(df_p)
            df_m = pd.DataFrame(df_m)
         
            # confidence intervals
            def ci(stats):
                alpha = 0.95
                p = ((1.0-alpha)/2.0) * 100
                # lower = max(0.0, np.percentile(stats, p))
                lower = np.percentile(stats, p)
                p = (alpha+((1.0-alpha)/2.0)) * 100
                # upper = min(1.0, np.percentile(stats, p))
                upper = np.percentile(stats, p)
                print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower, upper))
                return lower, upper
    

            lower_p = []
            lower_m = []
            upper_p = []
            upper_m = []
            mean_p = []
            mean_m = []
            
            for year in years:
                
                year = year-1
                
                lower, upper = ci(df_p.loc[:,year])    
                lower_p.append(lower)
                upper_p.append(upper)
                mean_p.append(np.mean(df_p.loc[:,year]))
               
                lower, upper = ci(df_m.loc[:,year])    
                lower_m.append(lower)
                upper_m.append(upper)
                mean_m.append(np.mean(df_m.loc[:,year]))
     
            df_p = pd.DataFrame(data={
               'year':years,
                'mean':mean_p,
                'lower':lower_p,
                'upper': upper_p,})
                          
            print(df_p[df_p['mean'].astype(float) < df_p.lower.astype(float)])
            print(df_p[df_p['mean'].astype(float) > df_p.upper.astype(float)])
        
        
            df_m = pd.DataFrame(data={
               'year':years,
                'mean':mean_m,
                'lower':lower_m,
                'upper': upper_m,})
                          
            df_m[df_m['mean'].astype(float) < df_m.lower.astype(float)]
            df_m[df_m['mean'].astype(float) > df_m.upper.astype(float)]
        

            df_m.to_csv(os.path.join(result_folder,f'risk_ratio_table_{outcome}_m_review.csv'), index=False)
            df_p.to_csv(os.path.join(result_folder,f'risk_ratio_table_{outcome}_p_review.csv'), index=False)   
            
            print(outcome)
           
