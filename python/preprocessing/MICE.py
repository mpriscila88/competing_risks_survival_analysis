# -*- coding: utf-8 -*-

def create_mice_model(path):
    
    import pandas as pd
    import numpy as np
    import os
    import sys
    sys.path.insert(0, path) # insert path
    
    ##################################################################
    # Get the training data common to all outcomes train sets
    #-----------------------------------------------------------------

    X_train0 = pd.read_csv(os.path.join(path, 'Intracranial_hemorrhage' + '_X_train.csv'), index_col=0, sep=',') 
    all_ids = []
    ind = 0
    
    outcomes = ['Death','Depression', 'Ischemic_stroke', 'ADMCIDementia']

    for outcome in outcomes:
        X_train = pd.read_csv(os.path.join(path, outcome + '_X_train.csv'), index_col=0, sep=',') 
        if ind == 0:
            X_train = pd.merge(X_train[['MRN','PatientEncounterID']], X_train0['PatientEncounterID'], on ='PatientEncounterID', how='inner').drop_duplicates()
            ind = 1
        else:
            ids = pd.concat(all_ids).drop_duplicates() 
            X_train = pd.merge(X_train[['MRN','PatientEncounterID']], ids['PatientEncounterID'], on ='PatientEncounterID', how='inner').drop_duplicates()
        all_ids.append(X_train)
     
    # common training ids used to create the model    
    all_ids = pd.concat(all_ids).drop_duplicates()
    
    cols = ['A1C', 'ALT', 'AST', 'HDL', 'LDL', 'Albumin', 'SystolicBP', 
            'DiastolicBP', 'Temperature', 'HeartRate', 'Weight', 'RespirationRate', 
            'Height', 'BMI', 'AlcoholDrinksPerWeek', 'Female', 'Age', 
            'Race','Visit_year', 'vitals_missing', 'labs_missing', 
            'Union', 'Active', 'Smoker', 'Diabetes', 'Hypertension', 'AGE_STRATA'] 
    
    outcomes = ['Intracranial_hemorrhage','Death','Depression',
                'Ischemic_stroke','ADMCIDementia']

    # Encounters data used to create the model
    data_enc = []
    for outcome in outcomes:
        X_train = pd.read_csv(os.path.join(path, outcome + '_X_train.csv'), index_col=0, sep=',')   
        X_train = X_train[X_train.PatientEncounterID.isin(all_ids.PatientEncounterID)] #[cols]
        data_enc.append(X_train)
     
    data_ids = pd.concat(data_enc).drop_duplicates()     
    data = data_ids[cols].drop_duplicates()    
    
    data.to_csv(os.path.join(path,'train_data_common_for_all.csv'), sep=',') 
    
    ##################################################################
    # Select the best modeling technique
    #-----------------------------------------------------------------
    
    data =  pd.read_csv(os.path.join(path,'train_data_common_for_all.csv'), index_col=0, sep=',') 

    cols = ['Weight','Height','Visit_year',
            'AlcoholConsumption', 'Race', 'AGE_STRATA']
    
    for col in cols:
        data = data.drop(columns=col)
    
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
    ax=msno.bar(data, figsize=(12, 6), fontsize=18, color='steelblue')
    ax.set_ylabel('Data availability (%)',**hfont)
    ax.set_yticklabels((0, 20, 40, 60, 80, 100),fontsize=18,fontname='Cambria')
    ax.set_xticklabels((data.columns), fontname='Cambria')
    
    ax2 = ax.twinx()
    ax2.set_yticklabels(labels='',fontsize=18,fontname='Cambria')
    ax2.set_ylabel('Number of visits',**hfont)
    ax2.yaxis.set_label_coords(1.17,0.5)

    for pos in ['top']: #, 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
        plt.show()
        
        
        
    data = data.rename(columns={"SystolicBP":"Systolic BP",
                                "DiastolicBP":"Diastolic BP",
                                "HeartRate":"Heart Rate",
                                "RespirationRate":"Respiration Rate",
                                "AlcoholDrinksPerWeek":'Alcohol drinks/week', 
                                "vitals_missing":'Vitals not measured', 
                                "labs_missing":'Labs not measured', 
                                "Diabetes_comorbidity":"Diabetes",
                                "Hypertension_comorbidity":"Hypertension"
                                })
    
    import missingno as msno
    import matplotlib.pyplot as plt
    
    hfont = {'fontname':'Cambria'}
    ax=msno.bar(data, figsize=(12, 6), fontsize=22, color='steelblue')
    ax.set_ylabel('Data availability (%)',**hfont)
    ax.set_yticklabels((0, 20, 40, 60, 80, 100),fontsize=22,fontname='Cambria')
    ax.set_xticklabels((data.columns), fontname='Cambria')
    
    ax2 = ax.twinx()
    ax2.set_yticklabels(labels='',fontsize=22,fontname='Cambria')
    ax2.set_ylabel('Number of visits',**hfont)
    ax2.yaxis.set_label_coords(1.17,0.5)

    for pos in ['top']: #, 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
        plt.show()

    cs = ['A1C', 'ALT', 'AST', 'HDL', 'LDL', 'Albumin', 'Systolic BP',
       'Diastolic BP', 'Temperature', 'Heart Rate', 'Respiration Rate', 'BMI',
       'Alcohol drinks per week'] # 

    for col in cs:
        print(len(data[data[col].astype(str) !='nan'])/len(data)*100)
        


    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # To use this experimental feature, we need to explicitly ask for it:
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import SimpleImputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score

    def choose_alg(X_full, y_full, var):
        
        N_SPLITS = 5
        
        rng = np.random.RandomState(0)
        
        n_samples, n_features = X_full.shape
        
        # Estimate the score on the entire dataset, with no missing values
        br_estimator = BayesianRidge()
        score_full_data = pd.DataFrame(
            cross_val_score(
                br_estimator, X_full, y_full, scoring='neg_mean_squared_error',
                cv=N_SPLITS
            ),
            columns=['Full Data']
        )
        
        # Add a single missing value to each row
        X_missing = X_full.copy()
        y_missing = y_full
        missing_samples = np.arange(n_samples)
        missing_features = rng.choice(n_features, n_samples, replace=True)
        X_missing[missing_samples, missing_features] = np.nan
        
        # Estimate the score after imputation (mean and median strategies)
        score_simple_imputer = pd.DataFrame()
        for strategy in ('mean', 'median'):
            estimator = make_pipeline(
                SimpleImputer(missing_values=np.nan, strategy=strategy),
                br_estimator
            )
            score_simple_imputer[strategy] = cross_val_score(
                estimator, X_missing, y_missing, scoring='neg_mean_squared_error',
                cv=N_SPLITS
            )
        
        # Estimate the score after iterative imputation of the missing values
        # with different estimators
        estimators = [
            BayesianRidge(),
            DecisionTreeRegressor(max_features='sqrt', random_state=0),
            ExtraTreesRegressor(n_estimators=10, random_state=0),
            KNeighborsRegressor(n_neighbors=15)
        ]
        score_iterative_imputer = pd.DataFrame()
        for impute_estimator in estimators:
            estimator = make_pipeline(
                IterativeImputer(random_state=0, estimator=impute_estimator),
                br_estimator
            )
            score_iterative_imputer[impute_estimator.__class__.__name__] = \
                cross_val_score(
                    estimator, X_missing, y_missing, scoring='neg_mean_squared_error',
                    cv=N_SPLITS
                )
        
        scores = pd.concat(
            [score_full_data, score_simple_imputer, score_iterative_imputer],
           # keys=['Original', 'SimpleImputer', 'IterativeImputer'], 
           axis=1
        )
        
        # plot results
        plt.rcParams.update({'font.sans-serif':'Cambria'})
        fig, ax = plt.subplots(figsize=(13, 6))
        means = -scores.mean()
        errors = scores.std()
        means.plot.barh(xerr=errors, ax=ax)
        ax.set_title('Regression with Different Imputation Methods for ' + var)
        ax.set_xlabel('MSE')
        ax.set_yticks(np.arange(means.shape[0]))
       # ax.set_yticklabels([" w/ ".join(label) for label in means.index.tolist()])
        plt.tight_layout(pad=1)
        plt.show()
    
    
    
    vars_ = ['A1C', 'ALT', 'AST', 'HDL', 'LDL', 'Albumin', 'SystolicBP',
            'DiastolicBP', 'Temperature', 'HeartRate','RespirationRate',
            'Height', 'AlcoholDrinksPerWeek'] # ExtraTreesRegressor
    
    
    for var in vars_:
        
        X_full_ = data.drop(columns=['BMI', 'Weight']).dropna()
               
        y_full_ = np.array(X_full_[[var]])
        
        X_full_ = np.array(X_full_.drop(columns=var))
        
        choose_alg(X_full_, y_full_, var)
    
    
    vars__ = ['BMI', 'Weight']
    
    for var in vars__:
        
        X_full_ = data.drop(columns=['A1C', 'ALT', 'AST', 'HDL', 'LDL', 'Albumin']).dropna()
               
        y_full_ = np.array(X_full_[[var]])
        
        X_full_ = np.array(X_full_.drop(columns=var))
        
        choose_alg(X_full_, y_full_, var)
        
        
    ##################################################################
    # Save the ExtraTreesRegressor for each variable
    #-----------------------------------------------------------------

    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.pipeline import make_pipeline
    
    
    def var_alg(X_full, y_full, var):

        br_estimator = BayesianRidge()
             
        impute_estimator = ExtraTreesRegressor(n_estimators=10, random_state=0)
        
        estimator = make_pipeline(
                IterativeImputer(random_state=0, estimator=impute_estimator),
                br_estimator)   
    
        estimator.fit(X_full,y_full.ravel())
    
        return estimator
    

    vars_ = ['A1C', 'ALT', 'AST', 'HDL', 'LDL', 'Albumin', 'SystolicBP',
            'DiastolicBP', 'Temperature', 'HeartRate','RespirationRate',
            'Height', 'AlcoholDrinksPerWeek'] # ExtraTreesRegressor
    
    
    save_model = []
    
    for var in vars_:
        
        X_full_ = data.drop(columns=['BMI', 'Weight']).dropna()
               
        y_full_ = np.array(X_full_[[var]])
        
        X_full_ = np.array(X_full_.drop(columns=var))
        
        estimator = var_alg(X_full_, y_full_, var)
        
        save_model.append(estimator)
       
       
    vars__ = ['BMI', 'Weight']
    
    for var in vars__:
        
        X_full_ = data.drop(columns=['A1C', 'ALT', 'AST', 'HDL', 'LDL', 'Albumin']).dropna()
               
        y_full_ = np.array(X_full_[[var]])
        
        X_full_ = np.array(X_full_.drop(columns=var))
           
        estimator = var_alg(X_full_, y_full_, var)
        
        save_model.append(estimator)
    
    ########################################################################
    # Impute the data with the ExtraTreesRegressor for train and test sets
    #---------------------------------------------------------------------

    outcomes = ['Intracranial_hemorrhage', 'Death','Depression', 
                'Ischemic_stroke', 'ADMCIDementia'] 
    
    vars_ = ['A1C', 'ALT', 'AST', 'HDL', 'LDL', 'Albumin', 'SystolicBP',
           'DiastolicBP', 'Temperature', 'HeartRate','RespirationRate',
           'Height', 'AlcoholDrinksPerWeek', 'BMI', 'Weight'] 
    
    for outcome in outcomes:
        
        X_train = pd.read_csv(os.path.join(path, outcome + '_X_train.csv'), index_col=0, sep=',')   
        X_test = pd.read_csv(os.path.join(path, outcome + '_X_test.csv'), index_col=0, sep=',')  
        
        cat1 = np.max(X_train['BMI'][X_train['BMI_cat'] == 0])
        cat2 = np.min(X_train['BMI'][X_train['BMI_cat'] == 2])  
   
        ind = 0
        
        for var in vars_:
            
            estimator = save_model[ind]
            
            if ((var != 'BMI') & (var != 'Weight')): 
            
                X_full_ = data.drop(columns=['BMI', 'Weight'])
             
            else:           
 
                X_full_ = data.drop(columns=['A1C', 'ALT', 'AST', 'HDL', 'LDL', 'Albumin'])
 
            X_train_imputation = X_train[X_full_.columns].drop(columns=var)
        
            X_test_imputation = X_test[X_full_.columns].drop(columns=var)
    
            X_train[var] = estimator.predict(X_train_imputation)
        
            X_test[var] = estimator.predict(X_test_imputation)
     
            ind += 1
            
            
        # Assign BMI categories
           
        def bmi(df, cat1, cat2):
            df['BMI_cat'] = np.nan
            df['BMI_cat'][(df.BMI < cat1)] = 0
            df['BMI_cat'][(df.BMI >= cat1) & (df.BMI < cat2)] = 1
            df['BMI_cat'][(df.BMI >= cat2)] = 2
            return df
      
        X_train = bmi(X_train, cat1, cat2)
        X_test = bmi(X_test, cat1, cat2)
        
        
        #Save datasets   
        
        X_test.to_csv(os.path.join(path, outcome + '_X_test_mice.csv'), sep=',')
        X_train.to_csv(os.path.join(path, outcome + '_X_train_mice.csv'), sep=',')
    
        print(outcome)
        
    # import missingno as msno
    # import matplotlib.pyplot as plt
    # msno.bar(X_train, figsize=(5, 20), fontsize=12, color='steelblue')
    # msno.bar(X_test, figsize=(5, 20), fontsize=12, color='steelblue')
        
