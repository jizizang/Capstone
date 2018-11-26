#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from create_features_DF1125 import create_df, clean_data, create_features_df, roc_curve, plot_roc, div_count_pos_neg, undersample
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
import pandas as pd

class Model(object):
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.columns = None
    
    def get_data(self):      
        '''
        Create dataframe from cvs file
        Create features set X
        Create targets set y
        '''
        raw_df = create_df(self.data_path)
        df = clean_data(raw_df, False)
#split after clean
        y = df.CASE_STATUS.values
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30, random_state=67, stratify=y)
#        X = create_features_df(df)
#        y = X.pop('CASE_STATUS').values
#        self.columns = X.columns
        
        return X_train, X_test, y_train, y_test
#        return X,y

    def fit(self, X_train, y_train):
        '''
        Fit RandomForest Classifier with training data
        '''
#        self.model = GradientBoostingClassifier(n_estimators=500, max_depth=8, subsample=0.5, max_features='auto', learning_rate=0.05)
        
        self.model=RandomForestClassifier(n_estimators=500, n_jobs=-1)
        self.model.fit(X_train, y_train)
     
    def predict_proba(self, X_test):
        '''
        Returns predicted probabilities for not fraud / fraud
        '''
        return self.model.predict_proba(X_test)[:,1]
    
    def predict(self, X_test):
        '''
        Returns predicted class ( 0= not fraud / 1 = fraud)
        '''
        return self.model.predict(X_test)
    

if __name__ == '__main__':
    data_path = "H-1B_2017.csv"
    model = Model(data_path)
    X_train, X_test, y_train, y_test = model.get_data()

    ''' 
    undersample
    '''
#    X_train, y_train = undersample(X_train, y_train, 0.1)
    
    ''' 
    df_save_em, df_save_so rejection rate for each employer and SOC name
    '''
    X, df_save_em, df_save_so = create_features_df(X_train, predict=False)
    
    ''' 
    dummy 
    '''
    X_test=create_features_df(X_test)
    
    ''' add rejection rate for each employer and SOC name to test data
        replace nan with mean of train???                             
    '''
    group=dict(zip(df_save_em.EMPLOYER_NAME,df_save_em.EMPLOYER_RATE))
    X_test['EMPLOYER_RATE'] = X_test['EMPLOYER_NAME'].map(group, na_action=None)
    X_test.fillna(y_train.mean(), inplace=True)
    
    group=dict(zip(df_save_so.SOC_NAME,df_save_so.SOC_RATE))
    X_test['SOC_RATE'] = X_test['SOC_NAME'].map(group, na_action=None)
    X_test.fillna(y_train.mean(), inplace=True)

    missing_cols = set(X.columns)-set(X_test.columns)
# Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        X_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
    X_test= X_test[X.columns]
    X_test.drop(['EMPLOYER_NAME','SOC_NAME','CASE_STATUS'], inplace=True, axis=1)
    X.drop(['EMPLOYER_NAME','SOC_NAME', 'CASE_STATUS'], inplace=True, axis=1)

    model.fit(X, y_train)
    
    print('roc_auc_score:')
    print(skm.roc_auc_score(y_test,model.predict_proba(X_test)))
    print()
    pd.crosstab(y_test, model.predict(X_test))
    
    v_probs = model.predict_proba(X_test)[:, 1]
    plot_roc(v_probs, y_test, "ROC plot of H1-B prediction", 
         "False Positive Rate (1 - Specificity)", "True Positive Rate (Sensitivity, Recall)")

