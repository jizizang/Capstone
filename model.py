#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from create_features_DF import create_df, clean_data, create_features_df
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
        X = create_features_df(df)
        y = X.pop('CASE_STATUS').values
        self.columns = X.columns
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=67, stratify=y)
#        return X_train, X_test, y_train, y_test
        return X,y

    def fit(self, X_train, y_train):
        '''
        Fit Gradient Boosted Classifier with training data
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
    X,y = model.get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=67, stratify=y)
    model.fit(X_train, y_train)
    
    
