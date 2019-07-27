#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:52:36 2019

@author: andres
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from Avaliacao_Performance_Clasificador import avaliacao_PerformanceC

df_train = pd.read_csv('Data/bank_prepro_standardize_train.csv')
df_test = pd.read_csv('Data/bank_prepro_standardize_test.csv')

del df_train['Unnamed: 0']
del df_test['Unnamed: 0']

df_train_class = pd.DataFrame(df_train['y'])    
df_train_features = df_train.loc[:, df_train.columns != 'y']

df_test_class = pd.DataFrame(df_test['y'])
df_test_features = df_test.loc[:, df_test.columns != 'y']
    
# Reducao usando Analise de Componentes Principais
modelo_PCA = PCA(n_components=5)
modelo_PCA.fit(df_train_features)
df_train_features = pd.DataFrame(modelo_PCA.transform(df_train_features))
df_test_features = pd.DataFrame(modelo_PCA.transform(df_test_features))

#Modelo de Regressao Logistica

C_list = np.linspace(0.1, 1, 10)
lista_punicao = ['l1','l2']

skf_model = StratifiedKFold(n_splits=5,shuffle=True)

# minimo de 3 iteracoes
numero_iteracoes = 3
for t in range(0,numero_iteracoes):
    print ("---Iteration: ",t)
    AVG_ACC = np.zeros(shape=[len(C_list),len(lista_punicao)])
    STD_ACC = np.zeros(shape=[len(C_list),len(lista_punicao)])
    
    x_count = 0
    for c_value in C_list:
        
        y_count = 0
        for punicao in lista_punicao:
            
            temp_accuracy_list = []
            for train_subset_index, cv_index in skf_model.split(df_train_features,df_train_class):
                df_train_features_subset = df_train_features.loc[train_subset_index]
                df_train_class_subset = df_train_class.loc[train_subset_index]
                df_train_features_cv = df_train_features.loc[cv_index]
                df_train_class_cv = df_train_class.loc[cv_index]
                
                lr_model = LogisticRegression(penalty=punicao, C=c_value, class_weight= 'balanced')
                lr_model.fit(df_train_features_subset, df_train_class_subset)
                score_value = lr_model.score(df_train_features_cv, df_train_class_cv)
                temp_accuracy_list.append(score_value)
            
            AVG_ACC[x_count,y_count] = np.mean(temp_accuracy_list)
            STD_ACC[x_count,y_count] = np.std(temp_accuracy_list)
            y_count += 1
            
        x_count += 1
    
    if t==0:
        final_AVG_ACC = AVG_ACC
        final_STD_ACC = STD_ACC
    else:
        final_AVG_ACC = np.dstack([final_AVG_ACC, AVG_ACC])
        final_STD_ACC = np.dstack([final_STD_ACC, STD_ACC])
             
final_accuracy_mean_list = np.mean(final_AVG_ACC, axis=2)
max_ind = np.unravel_index(np.argmax(final_accuracy_mean_list, axis=None), final_accuracy_mean_list.shape)

Escolha_C = C_list[max_ind[0]]
Escolha_punicao = lista_punicao[max_ind[1]]
print ("Cross Validation - C pela Regressao Logistica: ",Escolha_C)
print ("Cross Validation - Punicao pela Regressao Logistica: ",Escolha_punicao)

RL_modelo_F = LogisticRegression(penalty=Escolha_punicao, C=Escolha_C, class_weight= 'balanced')
RL_modelo_F.fit(df_train_features, df_train_class)
                     
Predicao_train = RL_modelo_F.predict(df_train_features)
Predicao_test = RL_modelo_F.predict(df_test_features)

Predicao_prob_train = RL_modelo_F.predict_proba(df_train_features)
Predicao_prob_test = RL_modelo_F.predict_proba(df_test_features)

avaliacao_PerformanceC(df_train_class, Predicao_train, Predicao_prob_train, df_test_class, Predicao_test, Predicao_prob_test, 'y')
