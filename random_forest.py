#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:59:58 2019

@author: andres
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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

# Classificador Aleatorio de floresta
n_estimador_lista = range(10, 50, 5)

skf_model = StratifiedKFold(n_splits=5,shuffle=True)

Numero_iteracoes = 1
for t in range(0,Numero_iteracoes):
    print ("---Iteration: ",t)
    AVG_ACC = np.zeros(shape=[len(n_estimador_lista)])
    STD_ACC = np.zeros(shape=[len(n_estimador_lista)])
    
    x_count = 0
    for k_val in n_estimador_lista:
        Lista_Acuracia_Temp = []
        
        for Indice_subCindice, cv_index in skf_model.split(df_train_features,df_train_class):
            df_train_features_subset = df_train_features.loc[Indice_subCindice]
            df_train_class_subset = df_train_class.loc[Indice_subCindice]
            df_train_features_cv = df_train_features.loc[cv_index]
            df_train_class_cv = df_train_class.loc[cv_index]
        
            BA_modelo = RandomForestClassifier(n_estimators=k_val, class_weight='balanced')
            BA_modelo.fit(df_train_features_subset, df_train_class_subset)
            contagem = BA_modelo.score(df_train_features_cv, df_train_class_cv)
            Lista_Acuracia_Temp.append(contagem)
                
        AVG_ACC[x_count] = np.mean(Lista_Acuracia_Temp)
        STD_ACC[x_count] = np.std(Lista_Acuracia_Temp)
        x_count += 1
    
    if t==0:
        final_AVG_ACC = AVG_ACC
        final_STD_ACC = STD_ACC
    else:
        final_AVG_ACC = np.vstack([final_AVG_ACC, AVG_ACC])
        final_STD_ACC = np.vstack([final_STD_ACC, STD_ACC])
    
Lista_Acc_meia_final = np.mean(final_AVG_ACC, axis=0)
final_k_indice = np.argmax(Lista_Acc_meia_final)

Escolha_k= n_estimador_lista[final_k_indice]
print ("Cross Validation - Numero de Estimadores pela Floresta A : ",Escolha_k)

BA_modelo_final = RandomForestClassifier(n_estimators=Escolha_k, class_weight='balanced')
BA_modelo_final.fit(df_train_features, df_train_class)
                     
Predicao_train = BA_modelo_final.predict(df_train_features)
Predicao_test = BA_modelo_final.predict(df_test_features)

Predicao_prob_train = BA_modelo_final.predict_proba(df_train_features)
Predicao_prob_test = BA_modelo_final.predict_proba(df_test_features)

avaliacao_PerformanceC(df_train_class, Predicao_train, Predicao_prob_train, df_test_class, Predicao_test, Predicao_prob_test, 'y')


