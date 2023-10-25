# -*- coding: utf-8 -*-

def prolong_event(e, outcomes):
    
   
    import pandas as pd
    import numpy as np
    
    e.ContactDTS = e.ContactDTS.astype("datetime64[s]")
    
    for outcome in outcomes:
            
        # create outcome event
            
        col = outcome + '_Event'
        
        a = e[e[outcome] == 1].groupby('MRN').ContactDTS.min().reset_index() # first diagnosis
    
        a[col] = a.ContactDTS  # Event time
        
        a = a.drop(columns='ContactDTS')

        e = pd.merge(e,a, on=['MRN'], how = 'outer') # add Event column       
        
        e['Time2Event_days'] = np.nan
        e['Time2Event_days'] = e[col] - e['ContactDTS']
            
        e['Time2Event_days'] = e['Time2Event_days'].astype(str).str.extract('(-?\d+) days')
        
        # Assign -1 to no events: will stay 0 in outcome
        
        e['Time2Event_days'][e['Time2Event_days'].astype(str) == 'NaT'] = 1
        e['Time2Event_days'][e['Time2Event_days'].astype(str) == 'nan'] = 1
        
        e['Time2Event_days'] = e['Time2Event_days'].astype(int)
        e[outcome][(e['Time2Event_days'] <= 0)] = 1
     
        e[outcome][(e['Time2Event_days'] > 0)] = 0
        
        e = e.drop(columns=[col,'Time2Event_days'])  

    return e
