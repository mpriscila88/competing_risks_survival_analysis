# -*- coding: utf-8 -*-

def define_dataset(e):
    
    import pandas as pd
    # import numpy as np
    
    # For more than one encounter in same day, select that with more data
    e['n'] = e.isnull().sum(axis=1)
    
    # Select first known encounter for each patient (MRN) 
    a = e.groupby(['MRN','ContactDTS']).n.min().reset_index()
        
    e = pd.merge(e, a, on=['MRN','ContactDTS','n'], how='inner')
        
    # for cases where same n, select randomly one (essentially what differs is PatientEncounterID)
    
    e = e.drop_duplicates(subset=['MRN','ContactDTS']).reset_index().drop(columns='index')


    # Assign time to event
    
    outcomes = ['Intracranial_hemorrhage','Death','Depression',
                'Ischemic_stroke','ADMCIDementia','Diabetes','Hypertension']


    for outcome in outcomes:
        e = time2event(e, outcome)

    # Remove variables
    e = e.drop(columns=['n'])   
    
    return e


def time2event(e, outcome):
    
    import pandas as pd
     
    if outcome == 'Death':
     
        col = 'DeathDTS'   

    else: # create outcome event
        
        col = outcome + '_Event'
        
        a = e[e[outcome] == 1].groupby('MRN').ContactDTS.min().reset_index() # first diagnosis
    
        a[col] = a.ContactDTS  # Event time
        
        a = a.drop(columns='ContactDTS')

        e = pd.merge(e,a, on=['MRN'], how = 'outer') # add Event column

        # Assign outcome 1 to all patients who have this outcome
        
        e[outcome][((e[col].astype(str) !='nan') & (e[col].astype(str) !='NaT'))] = 1

    # Where no date for event, indicate last visit date
    
    e[col][((e[col].astype(str) =='nan') | (e[col].astype(str) =='NaT'))] = e['last_visit']


    return time2event_fnc(e, col, outcome)


def time2event_fnc(e, col, outcome):
    
    import numpy as np    
    
    e.ContactDTS = e.ContactDTS.astype("datetime64[s]")
    e[col] = e[col].astype("datetime64[s]")
        
    e['Time2Event_days'] = np.nan
    e['Time2Event_days'] = e[col] - e['ContactDTS']
        
    e['Time2Event_days'] = e['Time2Event_days'].astype(str).str.extract('(-?\d+) days')
        
    # Correct events study end date imputed for this outcome
    
    e[col][e[outcome] == 0] = np.nan
        
    # Assign nan to any encounters after Time of Event 
    
    e.Time2Event_days = e.Time2Event_days.astype(float)
    
    e.Time2Event_days[e.Time2Event_days < 0] = np.nan
    
    e = e.rename(columns={"Time2Event_days":"Survival_days_" + outcome})
        
    return e
