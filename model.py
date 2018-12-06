#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from create_features_DF import clean_data, create_features_df, roc_curve, plot_roc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
import pandas as pd

class Model(object):
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.columns = None
        self.model=None
        
    def get_data(self):      
        '''
        Create dataframe from csv file
        Create features set X
        Create targets set y
        Output: X_train, X_test, y_train, y_test
        '''
        raw_df=pd.read_csv(self.data_path)
        
        #    fillna with mode or new category name and change Y/N to 1/0  
        df = clean_data(raw_df, False)
        y = df.CASE_STATUS.values
        #    split after clean
        #    split first then feature engineering, case status is still in df, need it to calculate rejection rate
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30, random_state=67, stratify=y)
        
        return X_train, X_test, y_train, y_test

    def fit(self, X_train, y_train):
        '''
        Fit RandomForest Classifier with training data
        '''
#        self.model = GradientBoostingClassifier(n_estimators=500, max_depth=8, subsample=0.5, max_features='auto', learning_rate=0.05)
        
        self.model=RandomForestClassifier(n_estimators=500, n_jobs=-1)
        self.model.fit(X_train, y_train)
     
    def predict_proba(self, X_test):
        '''
        Returns predicted probabilities for aprroveal/rejection
        '''
        return self.model.predict_proba(X_test)
    
    def predict(self, X_test):
        '''
        Input: 
        
        Returns predicted class ( 0= approvl / 1 = rejection )
        '''
        return self.model.predict(X_test)
    

if __name__ == '__main__':
    data_path = "H-1B_2017.csv"
    model = Model(data_path)
    X_train, X_test, y_train, y_test = model.get_data()

    #    df_save_em, df_save_so rejection rate for each employer and SOC name
    X, df_save_em, df_save_so = create_features_df(X_train, predict=False)
    #    same transformation for training and testing data
    X_test=create_features_df(X_test)
    
    #    add rejection rate for each employer and SOC name to test data, only use parameters learned from training data
    #    replace nan with mean of train???                             
    group=dict(zip(df_save_em.EMPLOYER_NAME,df_save_em.EMPLOYER_RATE))
    X_test['EMPLOYER_RATE'] = X_test['EMPLOYER_NAME'].map(group, na_action=None)
    X_test.fillna(y_train.mean(), inplace=True)
    
    group=dict(zip(df_save_so.SOC_NAME,df_save_so.SOC_RATE))
    X_test['SOC_RATE'] = X_test['SOC_NAME'].map(group, na_action=None)
    X_test.fillna(y_train.mean(), inplace=True)

    #    Add a missing column in test set with default value equal to 0
    missing_cols = set(X.columns)-set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    #    Ensure the order of column in the test set is in the same order than in train set
    X_test= X_test[X.columns]
    #    Drop columds that won't be used
    X_test.drop(['EMPLOYER_NAME','SOC_NAME','CASE_STATUS'], inplace=True, axis=1)
    X.drop(['EMPLOYER_NAME','SOC_NAME', 'CASE_STATUS'], inplace=True, axis=1)

    model.fit(X, y_train)
    
    #    get ROC_AUC_SCORE
    print('roc_auc_score:')
    print(skm.roc_auc_score(y_test,model.predict_proba(X_test)[:, 1]))
    print()
    #    Plot ROC curve
    v_probs = model.predict_proba(X_test)[:, 1]
    plot_roc(v_probs, y_test, "ROC plot of H1-B prediction", 
         "False Positive Rate (1 - Specificity)", "True Positive Rate (Sensitivity, Recall)")

