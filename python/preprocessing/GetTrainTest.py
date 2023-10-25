
# -*- coding: utf-8 -*-

def train_test_norm(e, path):
    
    import os
    import sys
    sys.path.insert(0, path) # insert path

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    e0 = e 
    
    e_aux = e[['MRN','Age']].drop_duplicates()

    e = e_aux.groupby(['MRN']).Age.max().reset_index()
    
 
    #%% Data division #############################################################
       
    age_strata = pd.qcut(x=e['Age'], q=[0, .25, .5, .75, 1.])
    
    print(age_strata.drop_duplicates())
     
    e['AGE_STRATA'] = 0
    
    e['AGE_STRATA'][(e['Age'] > 17) & (e['Age'] <= 47)] = 1
    e['AGE_STRATA'][(e['Age'] > 47) & (e['Age'] <= 60)] = 2
    e['AGE_STRATA'][(e['Age'] > 60) & (e['Age'] <= 71)] = 3
    e['AGE_STRATA'][(e['Age'] > 71)] = 4

    X_train, X_test = train_test_split(e, test_size=0.3, random_state=42, stratify=e['AGE_STRATA'])  
  
    # Check age distribution
    # indt = X_test[(X_test['Age'] > np.max(X_train['Age'])) & (X_test['Age'] == np.max(e.Age))].index
  
   
    # Assign all respective MRNs encounters in train and test
    
    X_train =  pd.merge(e0,pd.DataFrame(X_train['MRN']), on=['MRN'])
    
    X_test =  pd.merge(e0,pd.DataFrame(X_test['MRN']), on=['MRN']) 
    
     
    def age_strata(df):
        df['AGE_STRATA'] = 0
        df['AGE_STRATA'][(df['Age'] > 17) & (df['Age'] <= 47)] = 1
        df['AGE_STRATA'][(df['Age'] > 47) & (df['Age'] <= 60)] = 2
        df['AGE_STRATA'][(df['Age'] > 60) & (df['Age'] <= 71)] = 3
        df['AGE_STRATA'][(df['Age'] > 71)] = 4
        return df
    
    X_train = age_strata(X_train)
    X_test = age_strata(X_test)
    
    X_train.to_csv(os.path.join(path, 'X_train.csv'), sep=',')
    X_test.to_csv(os.path.join(path, 'X_test.csv'), sep=',')
        
    # For each outcome
    outcomes = ['Intracranial_hemorrhage','Depression',
                'Ischemic_stroke','ADMCIDementia','Death']

    X_train_ = X_train
    X_test_ = X_test
    
    for outcome in outcomes:
          
        print(len(X_train_))
        print(len(X_test_))
     
        X_train = X_train_[X_train_['Survival_days_' + outcome] > 1] # remove events with duration inferior to 1 day 
        X_test = X_test_[X_test_['Survival_days_' + outcome] > 1] # remove events with duration inferior to 1 day 
    
        print(len(X_train.MRN.drop_duplicates()))
        print(len(X_test.MRN.drop_duplicates()))
        
        # import matplotlib.pyplot as plt
      
        #%% log normalization for numerical variables
               
        cols_norm = ['A1C', 'HDL', 'Albumin', 'SystolicBP', 'DiastolicBP', 
                     'HeartRate', 'Weight','RespirationRate', 'Height','BMI',
                     'Temperature', 'AlcoholDrinksPerWeek','Age']
    
        cols_log = ['ALT', 'AST', 'LDL']
    
        def norm(data, col):
            median = np.nanmedian(data[col])
            q25 = np.nanpercentile(data[col], 25, axis=0)
            q75 = np.nanpercentile(data[col], 75, axis=0)
            iqr = q75 - q25
            y = (data[col] - median)/iqr
            return y, median, iqr
    
        for col in cols_norm:
            X_test[col] = np.interp(X_test[col], (X_test[col].min(), X_test[col].max()), (X_train[col].min(), X_train[col].max()))
            X_train[col], median, iqr = norm(X_train,col)
            X_test[col] = (X_test[col] - median)/iqr
       
            # plt.figure()
            # X_train[col].plot.hist(color='#607c8e', grid=True, alpha=0.7) #, rwidth=0.85)
            # plt.grid(axis='y', alpha=0.75)
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.title(col + ' in train set: (x-median)/iqr - ' + outcome)
            
            # plt.figure()
            # X_test[col].plot.hist(color='#607c8e', grid=True, alpha=0.7) #, rwidth=0.85)
            # plt.grid(axis='y', alpha=0.75)
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.title(col + ' in test set: (x-median)/iqr - ' + outcome)
        
        for col in cols_log:
            X_test[col] = np.interp(X_test[col], (X_test[col].min(), X_test[col].max()), (X_train[col].min(), X_train[col].max()))
            X_train[col] = np.log(X_train[col])
            X_test[col] = np.log(X_test[col])
    
            # plt.figure()
            # X_train[col].plot.hist(color='#607c8e', grid=True, alpha=0.7) #, rwidth=0.85)
            # plt.grid(axis='y', alpha=0.75)
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.title(col + ' in train set: log(x) - ' + outcome)
            
            # plt.figure()
            # X_test[col].plot.hist(color='#607c8e', grid=True, alpha=0.7) #, rwidth=0.85)
            # plt.grid(axis='y', alpha=0.75)
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.title(col + ' in test set: log(x) - ' + outcome)
    
            X_train.to_csv(os.path.join(path, outcome + '_X_train.csv'), sep=',')
            X_test.to_csv(os.path.join(path, outcome + '_X_test.csv'), sep=',')
        
     ##########################################################################
