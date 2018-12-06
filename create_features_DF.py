#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def clean_data(df, new_data=False):

    '''
    Input:  Pandas dataframe, new_data boolean (False for training data, True for 
            new raw data on which to make predictions)
    OUtput: Formatted Pandas dataframe
    
    '''
#        CASE_STATUS: drop 'CERTIFIED-WITHDRAWN', 'WITHDRAWN'
#        DENIED=1, CERTIFIED=0 
#        only keep VISA_CLASS == 'H-1B'    
#        drop 'E-3 Australian', 'H-1B1 Singapore', 'H-1B1 Chile'  

    df=df[ (df.CASE_STATUS=='DENIED') | (df.CASE_STATUS=='CERTIFIED')]
    df['CASE_STATUS'] = (df['CASE_STATUS'] == 'DENIED').astype(int)
    df= df[df.VISA_CLASS == 'H-1B']
    
#    very few NAN, replaced with mode
    df.H1B_DEPENDENT.fillna('N', inplace=True)
    df.WILLFUL_VIOLATOR.fillna('N', inplace=True)
    df.FULL_TIME_POSITION.fillna('Y', inplace=True)
    
#    Y/N to 1/0
    df['H1B_DEPENDENT'] = (df['H1B_DEPENDENT'] == 'Y').astype(int)
    df['WILLFUL_VIOLATOR'] = (df['WILLFUL_VIOLATOR'] == 'Y').astype(int)
    df['FULL_TIME_POSITION'] = (df['FULL_TIME_POSITION'] == 'Y').astype(int)

#    too many NAN, dummy!!!
    df.AGENT_REPRESENTING_EMPLOYER.fillna('M', inplace=True)
    df.SUPPORT_H1B.fillna('M', inplace=True)
    df.LABOR_CON_AGREE.fillna('M', inplace=True)
    df.PW_WAGE_LEVEL.fillna('M', inplace=True)
    df.WAGE_UNIT_OF_PAY.fillna('Week', inplace=True)

#    None for missing EMPLOYER_NAME and SOC_NAME
    df.EMPLOYER_NAME.fillna('None', inplace=True)
    df.SOC_NAME.fillna('None', inplace=True)

    return df

def create_features_df(df, predict=True):
    
    '''
    dummy-ize variables:
    input: dataframe
    returns: features dataframe for use in prediction model
    predict: bool - is the transformation for prediction or fitting
    '''

    if not predict:        #    applied on training data
        #    calculate rejection rate for each EMPLOYER
        df['per_employer']=df.groupby('EMPLOYER_NAME')['EMPLOYER_NAME'].transform('count')
        df['per_employer_deny']=df.groupby('EMPLOYER_NAME')['CASE_STATUS'].transform('sum')
        df['EMPLOYER_RATE']=df['per_employer_deny']/df['per_employer']
    
        mask = df.per_employer < 5
        column_name = 'EMPLOYER_RATE'
        df.loc[mask, column_name] = df[(df.per_employer < 5)].CASE_STATUS.mean()
        df['EMPLOYER_RATE'].fillna(df.CASE_STATUS.mean(), inplace=True)

        #    calculate rejection rate for each SOC
        df['per_soc']=df.groupby('SOC_NAME')['SOC_NAME'].transform('count')
        df['per_soc_deny']=df.groupby('SOC_NAME')['CASE_STATUS'].transform('sum')
        df['SOC_RATE']=df['per_soc_deny']/df['per_soc']
    
        mask = df.per_soc < 5
        column_name = 'SOC_RATE'
        df.loc[mask, column_name] = df[(df.per_soc < 5)].CASE_STATUS.mean()
        df['SOC_RATE'].fillna(df.CASE_STATUS.mean(), inplace=True)

        #    save EMPLOYER_RATE, SOC_RATE table to apply on test data later
        df_save_em=df[['EMPLOYER_NAME','EMPLOYER_RATE']]
        df_save_so=df[['SOC_NAME','SOC_RATE']]
        
        #    columns to keep
        columns_to_keep_03=['CASE_STATUS', 'AGENT_REPRESENTING_EMPLOYER', 'FULL_TIME_POSITION', 'H1B_DEPENDENT', 'WILLFUL_VIOLATOR', 'SUPPORT_H1B', 'LABOR_CON_AGREE', 'WORKSITE_STATE', 'EMPLOYER_NAME', 'EMPLOYER_RATE', 'SOC_NAME', 'SOC_RATE','PW_WAGE_LEVEL','WAGE_UNIT_OF_PAY', 'WAGE_RATE_OF_PAY_FROM']
        df=df[columns_to_keep_03]
    
    if predict:        #    applied on test data, no EMPLOYER_RATE, SOC_RATE
        columns_to_keep_03=['CASE_STATUS', 'AGENT_REPRESENTING_EMPLOYER', 'FULL_TIME_POSITION', 'H1B_DEPENDENT', 'WILLFUL_VIOLATOR', 'SUPPORT_H1B', 'LABOR_CON_AGREE', 'WORKSITE_STATE', 'EMPLOYER_NAME', 'SOC_NAME','PW_WAGE_LEVEL','WAGE_UNIT_OF_PAY','WAGE_RATE_OF_PAY_FROM']
        df=df[columns_to_keep_03]

    #    applied on both trainning and test data!!! dummy    
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
    
    df_dum_LEVEL = pd.get_dummies(df.PW_WAGE_LEVEL)
    df_dum_LEVEL.apply(lambda x: x.value_counts())
    df_dum_LEVEL.columns = map(lambda x: 'LEVEL_' + str(x), df_dum_LEVEL.columns)
    
    df_dum_UNIT = pd.get_dummies(df.WAGE_UNIT_OF_PAY)
    df_dum_UNIT.apply(lambda x: x.value_counts())
    df_dum_UNIT.columns = map(lambda x: 'UNIT_' + str(x), df_dum_UNIT.columns)
    
    #    wage to wage per year based on WAGE_UNIT_OF_PAY and WAGE_RATE_OF_PAY_FROM, then binned to 7 labels
    unitpay_to_num = {"Year":1, "Hour": 2080, "Month": 12, "Bi-Weekly": 26, "Week": 52}
    df["MULTIPLIER"] = df["WAGE_UNIT_OF_PAY"].map(unitpay_to_num)
    df["ACTUAL_SALARY"] = df["WAGE_RATE_OF_PAY_FROM"] * df["MULTIPLIER"]
    bins = [10000, 20000, 30000, 40000, 50000, 60000, 200000,1000000000]
    labels = [1,2,3,4,5,6,7]
    df['binned'] = pd.cut(df['ACTUAL_SALARY'], bins=bins, labels=labels)
    
    df_dum_WAGE = pd.get_dummies(df.binned)
    df_dum_WAGE.apply(lambda x: x.value_counts())
    df_dum_WAGE.columns = map(lambda x: 'WAGE_' + str(x), df_dum_WAGE.columns)
    
    #    merge dummy and drop features that won't be used
    df_for_model = pd.concat([df, df_dum_ST, df_dum_AGENT, df_dum_SUPPORT, df_dum_LABOR, df_dum_LEVEL, df_dum_UNIT, df_dum_WAGE], axis=1)
    
    df_for_model.drop(['STATE_CA','WORKSITE_STATE','AGENT_REPRESENTING_EMPLOYER','SUPPORT_H1B','LABOR_CON_AGREE','AGENT_M','SUPPORT_M','LABOR_M','LEVEL_M','PW_WAGE_LEVEL','WAGE_UNIT_OF_PAY', 'MULTIPLIER', 'ACTUAL_SALARY','binned','WAGE_RATE_OF_PAY_FROM'], inplace=True, axis=1, errors='ignore')
    
    if predict:    #    test data
        return df_for_model
    return df_for_model, df_save_em, df_save_so    #    return training data with EMPLOYER_RATE, SOC_RATE table

def roc_curve(probabilities, labels):    #for ROC curve plot
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

def plot_roc(v_probs, y_test, title, xlabel, ylabel):    #    for ROC curve plot
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
    
    
    

