import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.close("all")

class Dados_iniciais:
    
    def __init__(self, col_df):
        self.col_df = col_df
        self.col_df_grouped = col_df.groupby("y")
        self.nome_classe_nao = "no"
        self.nome_classe_sim = "yes"
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

    def plot_histogram_categorical(self, nome_carateristica):
        carateristica_df = pd.DataFrame()
        carateristica_df["no"] = self.col_df_grouped_nao[nome_carateristica].value_counts()
        carateristica_df["yes"] = self.col_df_grouped_sim[nome_carateristica].value_counts()
        
        carateristica_df.plot(kind='bar')
        plt.title("Histograma de Carateristicas - "+nome_carateristica)
        plt.ylabel("grouped")
        plt.xlabel("Valores unicos carateristicos")
        plt.tight_layout()
    
    

### Read csv 
col_df = pd.read_csv('bank-full.csv',sep=';')
#print(col_df)



Dados_iniciais_obj = Dados_iniciais(col_df)
# 1 age	

Dados_iniciais_obj.plot_histograma_continuo("age", 50)
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/age.eps')

# 2 job	

Dados_iniciais_obj.plot_histogram_categorical("job")
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/job.eps')
# 3 marital

Dados_iniciais_obj.plot_histogram_categorical("marital")
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/marital.eps')
# 4 education

Dados_iniciais_obj.plot_histogram_categorical("education")
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/education.eps')
# 5 default	

Dados_iniciais_obj.plot_histogram_categorical("default")
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/default.eps')
# 6 balance	

Dados_iniciais_obj.plot_histograma_continuo("balance", 50)
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/balance.eps')

# 7 housing	
Dados_iniciais_obj.plot_histogram_categorical("housing")
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/housing.eps')
# 8 loan	
	

Dados_iniciais_obj.plot_histogram_categorical("loan")
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/loan.eps')
# 9 contact	

Dados_iniciais_obj.plot_histogram_categorical("contact")
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/contact.eps')
# 10 day	

Dados_iniciais_obj.plot_histogram_categorical("day")
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/day.eps')
# 11 month	

Dados_iniciais_obj.plot_histogram_categorical("month")
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/month.eps')
# 12 duration	

Dados_iniciais_obj.plot_histograma_continuo("duration",50)
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/duration.eps')
# 13 campaign	

Dados_iniciais_obj.plot_histograma_continuo("campaign", 50)
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/campaign.eps')
# 14 pdays	
Dados_iniciais_obj.plot_histograma_continuo("pdays", 50)
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/pdays.eps')
# 15 previous	

Dados_iniciais_obj.plot_histogram_categorical("previous")
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/previous.eps')
# 16 poutcome

Dados_iniciais_obj.plot_histogram_categorical("poutcome")
plt.savefig('/media/andres/dados1/semantix_teste/desafio/figuras/poutcome.eps')



