#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:59:58 2019

@author: andres
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
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
    

# Perceptrao
lista_alpha = np.linspace(0.00001, 1, 15)
lista_punicao = ['l1','l2','elasticnet']

skf_model = StratifiedKFold(n_splits=5,shuffle=True)

numero_iteracoes = 2
for t in range(0,numero_iteracoes):
    print ("---Iteration: ",t)
    AVG_ACC = np.zeros(shape=[len(lista_alpha),len(lista_punicao)])
    STD_ACC = np.zeros(shape=[len(lista_alpha),len(lista_punicao)])
    
    x_count = 0
    for valor_alpha in lista_alpha:
        
        y_count = 0
        for punicao in lista_punicao:
            
            lista_temp_acuidade = []
            for indice_suCbtrain, cv_indice in skf_model.split(df_train_features,df_train_class):
                df_train_features_subset = df_train_features.loc[indice_suCbtrain]
                df_train_class_subset = df_train_class.loc[indice_suCbtrain]
                df_train_features_cv = df_train_features.loc[cv_indice]
                df_train_class_cv = df_train_class.loc[cv_indice]
                
                modelo_perceptrao = Perceptron(penalty=punicao, alpha=valor_alpha, class_weight= 'balanced')
                modelo_perceptrao.fit(df_train_features_subset, df_train_class_subset)
                pontagem = modelo_perceptrao.score(df_train_features_cv, df_train_class_cv)
                lista_temp_acuidade.append(pontagem)
            
            AVG_ACC[x_count,y_count] = np.mean(lista_temp_acuidade)
            STD_ACC[x_count,y_count] = np.std(lista_temp_acuidade)
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

Escolha_alpha = lista_alpha[max_ind[0]]
Escolha_Punicao = lista_punicao[max_ind[1]]
print ("Cross Validation - alpha pelo Perceptron: ",Escolha_alpha)
print ("Cross Validation - Punicao pelo Perceptron: ",Escolha_Punicao)

modelo_perceptrao_final = Perceptron(penalty=Escolha_Punicao, alpha=Escolha_alpha, class_weight= 'balanced')
modelo_perceptrao_final = CalibratedClassifierCV(base_estimator=modelo_perceptrao_final,cv=10, method='isotonic')
modelo_perceptrao_final.fit(df_train_features, df_train_class)
                     
Predicao_train = modelo_perceptrao_final.predict(df_train_features)
Predicao_test = modelo_perceptrao_final.predict(df_test_features)

Predicao_prob_train = modelo_perceptrao_final.predict_proba(df_train_features)
Predicao_prob_test  = modelo_perceptrao_final.predict_proba(df_test_features)

avaliacao_PerformanceC(df_train_class, Predicao_train, Predicao_prob_train, df_test_class, Predicao_test, Predicao_prob_test, 'y')
