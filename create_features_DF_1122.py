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
    df.H1B_DEPENDENT.fillna('M', inplace=True)
    df.WILLFUL_VIOLATOR.fillna('M', inplace=True)
    df.FULL_TIME_POSITION.fillna('M', inplace=True)
    
#    df['H1B_DEPENDENT'] = (df['H1B_DEPENDENT'] == 'Y').astype(int)
#    df['WILLFUL_VIOLATOR'] = (df['WILLFUL_VIOLATOR'] == 'Y').astype(int)
#    df['FULL_TIME_POSITION'] = (df['FULL_TIME_POSITION'] == 'Y').astype(int)

    ''' too many NAN, dummy!!! '''
    df.AGENT_REPRESENTING_EMPLOYER.fillna('M', inplace=True)
    df.SUPPORT_H1B.fillna('M', inplace=True)
    df.LABOR_CON_AGREE.fillna('M', inplace=True)
    df.PW_WAGE_LEVEL.fillna('M', inplace=True)

    df.WAGE_UNIT_OF_PAY.fillna('M', inplace=True)
    
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
        columns_to_keep_03=['CASE_STATUS', 'AGENT_REPRESENTING_EMPLOYER', 'FULL_TIME_POSITION', 'H1B_DEPENDENT', 'WILLFUL_VIOLATOR', 'SUPPORT_H1B', 'LABOR_CON_AGREE', 'WORKSITE_STATE', 'EMPLOYER_NAME', 'EMPLOYER_RATE', 'SOC_NAME', 'SOC_RATE','PW_WAGE_LEVEL','WAGE_UNIT_OF_PAY']
        df=df[columns_to_keep_03]
    
    if predict:
        columns_to_keep_03=['CASE_STATUS', 'AGENT_REPRESENTING_EMPLOYER', 'FULL_TIME_POSITION', 'H1B_DEPENDENT', 'WILLFUL_VIOLATOR', 'SUPPORT_H1B', 'LABOR_CON_AGREE', 'WORKSITE_STATE', 'EMPLOYER_NAME', 'SOC_NAME','PW_WAGE_LEVEL','WAGE_UNIT_OF_PAY']
        df=df[columns_to_keep_03]
    
    df_dum_ST = pd.get_dummies(df.WORKSITE_STATE,dummy_na=True)
    df_dum_ST.apply(lambda x: x.value_counts())
    df_dum_ST.columns = map(lambda x: 'STATE_' + str(x), df_dum_ST.columns)

#    df_dum_AGENT = pd.get_dummies(df.AGENT_REPRESENTING_EMPLOYER)
#    df_dum_AGENT.apply(lambda x: x.value_counts())
#    df_dum_AGENT.columns = map(lambda x: 'AGENT_' + str(x), df_dum_AGENT.columns)
    
#    df_dum_SUPPORT = pd.get_dummies(df.SUPPORT_H1B)
#    df_dum_SUPPORT.apply(lambda x: x.value_counts())
#    df_dum_SUPPORT.columns = map(lambda x: 'SUPPORT_' + str(x), df_dum_SUPPORT.columns)
    
#    df_dum_LABOR = pd.get_dummies(df.LABOR_CON_AGREE)
#    df_dum_LABOR.apply(lambda x: x.value_counts())
#    df_dum_LABOR.columns = map(lambda x: 'LABOR_' + str(x), df_dum_LABOR.columns)
    
#    df_dum_LEVEL = pd.get_dummies(df.PW_WAGE_LEVEL)
#    df_dum_LEVEL.apply(lambda x: x.value_counts())
#    df_dum_LEVEL.columns = map(lambda x: 'LEVEL_' + str(x), df_dum_LEVEL.columns)
    
#    df_dum_UNIT = pd.get_dummies(df.WAGE_UNIT_OF_PAY)
#    df_dum_UNIT.apply(lambda x: x.value_counts())
#    df_dum_UNIT.columns = map(lambda x: 'UNIT_' + str(x), df_dum_UNIT.columns)
    
#    df_for_model = pd.concat([df, df_dum_ST, df_dum_AGENT, df_dum_SUPPORT, df_dum_LABOR, df_dum_LEVEL, df_dum_UNIT], axis=1)
    df_for_model = pd.concat([df, df_dum_ST], axis=1)
    df_for_model.drop(['STATE_CA','WORKSITE_STATE'], inplace=True, axis=1, errors='ignore')
#,'AGENT_REPRESENTING_EMPLOYER', 'SUPPORT_H1B','LABOR_CON_AGREE','AGENT_M','SUPPORT_M','LABOR_M','LEVEL_M','PW_WAGE_LEVEL','WAGE_UNIT_OF_PAY']
    
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
    
def div_count_pos_neg(X, y):
    """Helper function to divide X & y into positive and negative classes
    and counts up the number in each.
    Parameters
    ----------
    X : ndarray - 2D
    y : ndarray - 1D
    Returns
    -------
    negative_count : Int
    positive_count : Int
    X_positives    : ndarray - 2D
    X_negatives    : ndarray - 2D
    y_positives    : ndarray - 1D
    y_negatives    : ndarray - 1D
    """
    negatives, positives = y == 0, y == 1
    negative_count, positive_count = np.sum(negatives), np.sum(positives)
    X_positives, y_positives = X[positives], y[positives]
    X_negatives, y_negatives = X[negatives], y[negatives]
    return negative_count, positive_count, X_positives, \
           X_negatives, y_positives, y_negatives


def undersample(X, y, tp):
    """Randomly discards negative observations from X & y to achieve the
    target proportion of positive to negative observations.
    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - range [0.5, 1], target proportion of positive class observations
    Returns
    -------
    X_undersampled : ndarray - 2D
    y_undersampled : ndarray - 1D
    """
    if tp < np.mean(y):
        return X, y
    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    negative_sample_rate = (pos_count * (1 - tp)) / (neg_count * tp)
    negative_keepers = np.random.choice(a=[False, True], size=neg_count,
                                        p=[1 - negative_sample_rate,
                                           negative_sample_rate])
    X_negative_undersampled = X_neg[negative_keepers]
    y_negative_undersampled = y_neg[negative_keepers]
    X_undersampled = np.vstack((X_negative_undersampled, X_pos))
    y_undersampled = np.concatenate((y_negative_undersampled, y_pos))

    return X_undersampled, y_undersampled
    
if __name__ == '__main__':
    cvs_path = "H-1B_2017.csv"
    new_df = create_df(cvs_path)
    new_df = clean_data(new_df)
    get_dummies = create_features_df(new_df)
    
    
    

