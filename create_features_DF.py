#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_df(data_path):
    '''
    load csv data into pandas dataframe
    input: file path of csv data
    output: pandas dataframe
    '''

    df = pd.read_csv(data_path)
    
    return df

def clean_data(df, new_data=False):

    '''
    Input:  Pandas dataframe, new_data boolean (False for training data, True for 
                    data on which to make predictions)
    OUtput: Formatted Pandas dataframe
    
    '''
    
#    if new_data:
#        return df

    ''' CASE_STATUS: drop 'CERTIFIED-WITHDRAWN', 'WITHDRAWN'
        DENIED=1, CERTIFIED=0 
        only keep VISA_CLASS == 'H-1B'    
        drop 'E-3 Australian', 'H-1B1 Singapore', 'H-1B1 Chile'  '''
    
    
    df=df[ (df.CASE_STATUS=='DENIED') | (df.CASE_STATUS=='CERTIFIED')]
    df['CASE_STATUS'] = (df['CASE_STATUS'] == 'DENIED').astype(int)
    df= df[df.VISA_CLASS == 'H-1B']
    

    '''
0    533621
1      7481
Name: CASE_STATUS, dtype: int64
0

Y    309938
N    148486
Name: AGENT_REPRESENTING_EMPLOYER, dtype: int64
82678

Level I      190661
Level II     158218
Level III     55184
Level IV      31105
Name: PW_WAGE_LEVEL, dtype: int64
105934

Y    208698
N      6962
Name: SUPPORT_H1B, dtype: int64
325442

Y    205517
N      9824
Name: LABOR_CON_AGREE, dtype: int64
325761'''
    ''' very few NAN replaced with mode '''
    df.H1B_DEPENDENT.fillna('N', inplace=True)
    df.WILLFUL_VIOLATOR.fillna('N', inplace=True)
    df.FULL_TIME_POSITION.fillna('Y', inplace=True)
    
    df['H1B_DEPENDENT'] = (df['H1B_DEPENDENT'] == 'Y').astype(int)
    df['WILLFUL_VIOLATOR'] = (df['WILLFUL_VIOLATOR'] == 'Y').astype(int)
    df['FULL_TIME_POSITION'] = (df['FULL_TIME_POSITION'] == 'Y').astype(int)

    ''' too many NAN, dummy!!! '''
    df.AGENT_REPRESENTING_EMPLOYER.fillna('M', inplace=True)
    df.SUPPORT_H1B.fillna('M', inplace=True)
    df.LABOR_CON_AGREE.fillna('M', inplace=True)
    
#    df['AGENT_REPRESENTING_EMPLOYER'] = (df['AGENT_REPRESENTING_EMPLOYER'] == 'Y').astype(int)
#    df['SUPPORT_H1B'] = (df['SUPPORT_H1B'] == 'Y').astype(int)
#    df['LABOR_CON_AGREE'] = (df['LABOR_CON_AGREE'] == 'Y').astype(int)

    return df

def create_features_df(df, predict=True):
    
    '''
    dummy-ize variables: ???
    input: dataframe
    returns: features dataframe for use in prediction model
    predict: bool - is the transformation for prediction or fitting
    '''

    ''' Columns to keep '''
    
#    columns_to_keep_00=['CASE_STATUS', 'AGENT_REPRESENTING_EMPLOYER', 'FULL_TIME_POSITION', 'PW_WAGE_LEVEL', 'H1B_DEPENDENT', 'WILLFUL_VIOLATOR', 'SUPPORT_H1B', 'LABOR_CON_AGREE', 'WORKSITE_STATE']
#    columns_to_keep_01=['CASE_STATUS', 'AGENT_REPRESENTING_EMPLOYER', 'FULL_TIME_POSITION', 'H1B_DEPENDENT', 'WILLFUL_VIOLATOR', 'SUPPORT_H1B', 'LABOR_CON_AGREE', 'WORKSITE_STATE']

    if not predict:
        df['per_employer']=df.groupby('EMPLOYER_NAME')['EMPLOYER_NAME'].transform('count')
        df['per_employer_deny']=df.groupby('EMPLOYER_NAME')['CASE_STATUS'].transform('sum')
        df['EMPLOYER_RATE']=df['per_employer_deny']/df['per_employer']
    
        mask = df.per_employer < 5
        column_name = 'EMPLOYER_RATE'
        df.loc[mask, column_name] = df[(df.per_employer < 5)].CASE_STATUS.mean()
    
        df['EMPLOYER_RATE'].fillna(df.CASE_STATUS.mean(), inplace=True)
    
    
        df['per_soc']=df.groupby('SOC_NAME')['SOC_NAME'].transform('count')
        df['per_soc_deny']=df.groupby('SOC_NAME')['CASE_STATUS'].transform('sum')
        df['SOC_RATE']=df['per_soc_deny']/df['per_soc']
    
        mask = df.per_soc < 5
        column_name = 'SOC_RATE'
        df.loc[mask, column_name] = df[(df.per_soc < 5)].CASE_STATUS.mean()

        df_save_em=df[['EMPLOYER_NAME','EMPLOYER_RATE']]
        df_save_so=df[['SOC_NAME','SOC_RATE']]
        
#    columns_to_keep_02=['CASE_STATUS', 'AGENT_REPRESENTING_EMPLOYER', 'FULL_TIME_POSITION', 'H1B_DEPENDENT', 'WILLFUL_VIOLATOR', 'SUPPORT_H1B', 'LABOR_CON_AGREE', 'WORKSITE_STATE', 'EMPLOYER_RATE','SOC_RATE']
        columns_to_keep_03=['CASE_STATUS', 'AGENT_REPRESENTING_EMPLOYER', 'FULL_TIME_POSITION', 'H1B_DEPENDENT', 'WILLFUL_VIOLATOR', 'SUPPORT_H1B', 'LABOR_CON_AGREE', 'WORKSITE_STATE', 'EMPLOYER_NAME', 'EMPLOYER_RATE', 'SOC_NAME', 'SOC_RATE']
        df=df[columns_to_keep_03]
    
    if predict:
        columns_to_keep_03=['CASE_STATUS', 'AGENT_REPRESENTING_EMPLOYER', 'FULL_TIME_POSITION', 'H1B_DEPENDENT', 'WILLFUL_VIOLATOR', 'SUPPORT_H1B', 'LABOR_CON_AGREE', 'WORKSITE_STATE', 'EMPLOYER_NAME', 'SOC_NAME']
        df=df[columns_to_keep_03]
    
    df_dum_ST = pd.get_dummies(df.WORKSITE_STATE,dummy_na=True)
    df_dum_ST.apply(lambda x: x.value_counts())
    df_dum_ST.columns = map(lambda x: 'STATE_' + str(x), df_dum_ST.columns)

    df_dum_AGENT = pd.get_dummies(df.AGENT_REPRESENTING_EMPLOYER)
    df_dum_AGENT.apply(lambda x: x.value_counts())
    df_dum_AGENT.columns = map(lambda x: 'AGENT_' + str(x), df_dum_AGENT.columns)
    
    df_dum_SUPPORT = pd.get_dummies(df.SUPPORT_H1B)
    df_dum_SUPPORT.apply(lambda x: x.value_counts())
    df_dum_SUPPORT.columns = map(lambda x: 'SUPPORT_' + str(x), df_dum_SUPPORT.columns)
    
    df_dum_LABOR = pd.get_dummies(df.LABOR_CON_AGREE)
    df_dum_LABOR.apply(lambda x: x.value_counts())
    df_dum_LABOR.columns = map(lambda x: 'LABOR_' + str(x), df_dum_LABOR.columns)
    
    df_for_model = pd.concat([df, df_dum_ST, df_dum_AGENT, df_dum_SUPPORT, df_dum_LABOR], axis=1)

    df_for_model.drop(['STATE_CA','WORKSITE_STATE','AGENT_REPRESENTING_EMPLOYER',
                       'SUPPORT_H1B','LABOR_CON_AGREE','AGENT_M','SUPPORT_M','LABOR_M'], inplace=True, axis=1, errors='ignore')
    
    if predict:
        return df_for_model
    return df_for_model, df_save_em, df_save_so

def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''

    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []

    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels)
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)

        fprs.append(fpr)
        tprs.append(tpr)
    
    return tprs, fprs, thresholds.tolist()

def plot_roc(v_probs, y_test, title, xlabel, ylabel):
    # ROC
    tpr, fpr, thresholds = roc_curve(v_probs, y_test)

    plt.hold(True)
    plt.plot(fpr, tpr)

    # 45 degree line
    xx = np.linspace(0, 1.0, 20)
    plt.plot(xx, xx, color='red')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()
    
if __name__ == '__main__':
    cvs_path = "H-1B_2017.csv"
    new_df = create_df(cvs_path)
    new_df = clean_data(new_df)
    get_dummies = create_features_df(new_df)
    
    
    

