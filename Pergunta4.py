#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:46:06 2019

@author: andres
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.close("all")

train_col_df = pd.read_csv('/media/andres/dados1/semantix_teste/desafio/Data/bank_prepro_standardize_train.csv')
col_df = pd.read_csv('/media/andres/dados1/semantix_teste/desafio/bank-full.csv',sep=';')
class Dados_iniciais2:

    def __init__(self, col_df, var,val_n,val_p):
        self.col_df = col_df
        self.col_df_grouped = var
        self.nome_classe_nao = val_n
        self.nome_classe_sim = val_p
        self.col_df_grouped_nao = self.col_df_grouped.get_group(self.nome_classe_nao)
        self.col_df_grouped_sim = self.col_df_grouped.get_group(self.nome_classe_sim)


    def plot_histograma_continuo(self, nome_carateristica, bin_tamanho):
        plt.figure()
        plt.hist(self.col_df_grouped_nao[nome_carateristica], bins=bin_tamanho, label=self.nome_classe_nao)
        plt.hist(self.col_df_grouped_sim[nome_carateristica], bins=bin_tamanho, label=self.nome_classe_sim)
        plt.legend()
        plt.title("Histograma de Carateristicas - "+nome_carateristica)
        plt.xlabel("Valores Carateristicos")
        plt.ylabel("grouped")
        
y=col_df.groupby("y")
val_n=col_df.nome_classe_nao = "no"
val_p=col_df.nome_classe_sim = "yes"        
        
Dados_iniciais_obj = Dados_iniciais2(col_df,y,val_n,val_p)

Dados_iniciais_obj.plot_histograma_continuo("campaign", 50)
plt.axis([ 0, 10, 0, 25000])
plt.title("Histograma de Carateristicas  Campaign")
plt.ylabel("Adesão à Campanha Atual")
plt.xlabel("Numero de Contactos ao Cliente")

Dados_iniciais_obj.plot_histograma_continuo("campaign", 50)
plt.axis([ 0, 2.5, 0, 4500])
plt.title("Histograma de Carateristicas  Campaign")
plt.ylabel("Adesão à Campanha Atual")
plt.xlabel("Numero de Contactos ao Cliente")

y=train_col_df.groupby("y")
val_n=train_col_df.nome_classe_nao = 0
val_p=train_col_df.nome_classe_sim = 1
Dados_iniciais_obj = Dados_iniciais2(train_col_df,y,val_n,val_p)

Dados_iniciais_obj.plot_histograma_continuo("campaign", 50)
plt.axis([ 0, 1., 0, 5000])
plt.title("Histograma de Carateristicas  Campaign Normalizado")
plt.ylabel("Adesão à Campanha Atual")
plt.xlabel("Numero de Contactos ao Cliente")

Camp=train_col_df['campaign']
descrip_stat=Camp.describe()
print(descrip_stat)

Camp=col_df.groupby("campaign")
Camp['y'].count()



Camp=col_df.groupby(["campaign",'y'])
#fre=Camp["y"].corr().unstack()

descrip_stat=Camp.describe()
print(descrip_stat)


#from nltk.tokenize import word_tokenize
#from nltk.corpus import gutenberg

#Camp=col_df['campaign']

#token = word_tokenize(sample)
#wlist = []

#for i in range(50):
#    wlist.append(token[i])

#wordfreq = [wlist.count(w) for w in wlist]
#print("Pairs\n" + str(zip(token, wordfreq)))