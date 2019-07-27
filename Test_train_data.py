#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 04:46:48 2019

@author: andres
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from standar_normal_ization import caracteristicas_preprocessadas, classe_preprocessamento

def train_teste_dados(df):
    df_class = pd.DataFrame(df['y'])
    df_features = df.loc[:, df.columns != 'y']
    
    df_features_train, df_features_test, df_class_train,  df_class_test = train_test_split(df_features, df_class)
    
    df_train = pd.concat([df_features_train, df_class_train], axis=1)
    df_test = pd.concat([df_features_test, df_class_test], axis=1)

    return df_train, df_test

df = pd.read_csv('bank-full.csv',sep=';')
df_train, df_test = train_teste_dados(df)
df_train, df_test = caracteristicas_preprocessadas(df_train, df_test,"Standardize")
df_train, df_test = classe_preprocessamento(df_train, df_test)

df_train.to_csv('Data/bank_prepro_standardize_train.csv')
df_test.to_csv('Data/bank_prepro_standardize_test.csv')
