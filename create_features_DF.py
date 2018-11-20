#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

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
#    '''use most common delivery method for missing values'''
    
#    df['delivery_method'].fillna(0, inplace=True)
    
#    if new_data:
#        return df

    ''' CASE_STATUS: drop 'CERTIFIED-WITHDRAWN', 'WITHDRAWN'
        DENIED=1, CERTIFIED=0 
        only keep VISA_CLASS == 'H-1B'    
        drop 'E-3 Australian', 'H-1B1 Singapore', 'H-1B1 Chile'  '''
    
    
    df=df[ (df.CASE_STATUS=='DENIED') | (df.CASE_STATUS=='CERTIFIED')]
    df['CASE_STATUS'] = (df['CASE_STATUS'] == 'DENIED').astype(int)
    df= df[df.VISA_CLASS == 'H-1B']
    
    ''' Columns to keep '''
    
    columns_to_keep_00=['CASE_STATUS', 'AGENT_REPRESENTING_EMPLOYER', 'FULL_TIME_POSITION', 'PW_WAGE_LEVEL', 'H1B_DEPENDENT', 'WILLFUL_VIOLATOR', 'SUPPORT_H1B', 'LABOR_CON_AGREE']
    df=df[columns_to_keep_00]
    
    
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

    ''' too many NAN, dummy? '''
    df.AGENT_REPRESENTING_EMPLOYER.fillna('Y', inplace=True)
    df.SUPPORT_H1B.fillna('Y', inplace=True)
    df.LABOR_CON_AGREE.fillna('Y', inplace=True)
    
    df['AGENT_REPRESENTING_EMPLOYER'] = (df['AGENT_REPRESENTING_EMPLOYER'] == 'Y').astype(int)
    df['SUPPORT_H1B'] = (df['SUPPORT_H1B'] == 'Y').astype(int)
    df['LABOR_CON_AGREE'] = (df['LABOR_CON_AGREE'] == 'Y').astype(int)

    return df

def create_features_df(df):
    
    '''
    dummy-ize variables: user_type, channels, delivery_method
    input: dataframe
    returns: features dataframe for use in prediction model, warning accrual
                note warnings accrual used to flag data with insufficient features
                for predictions
    
    '''

        df_dum_user = pd.get_dummies(df.user_type)
        df_dum_user.apply(lambda x: x.value_counts())
        df_dum_user.columns = map(lambda x: 'user_type_' + str(x), df_dum_user.columns)
   
        df_dum_channels = pd.get_dummies(df.channels)
        df_dum_channels.apply(lambda x: x.value_counts())
        df_dum_channels.columns = map(lambda x: 'channels_' + str(x), df_dum_channels.columns)

        df_dum_delivery = pd.get_dummies(df.delivery_method)
        df_dum_delivery.apply(lambda x: x.value_counts())
        df_dum_delivery.columns = map(lambda x: 'delivery_' + str(x), df_dum_delivery.columns)


        df_for_model = pd.concat([df_dum_user, df_dum_channels, df_dum_delivery], axis=1)
   
        df_for_model.drop(['user_type_103', 'channels_13', 'delivery_3.0'], inplace=True, axis=1, errors='ignore')
    
        return df_for_model


if __name__ == '__main__':
    cvs_path = "H-1B_2017.csv"
    new_df = create_df(cvs_path)
    new_df = clean_data(new_df)
#    get_dummies = create_features_df(new_df)
    
    
    

