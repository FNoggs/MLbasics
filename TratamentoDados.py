import pandas as pd #Biblioteca para manipulação de dados
import numpy as np #Biblioteca para cálculos matemáticos

from sklearn.model_selection import train_test_split #Biblioteca para dividir os dados em treino e teste
from sklearn.preprocessing import MinMaxScaler #Biblioteca para normalização dos dados
from sklearn.neighbors import KNeighborsClassifier #Biblioteca para o modelo KNN
from sklearn.metrics import accuracy_score, confusion_matrix #Biblioteca para calcular a acurácia do modelo

import matplotlib.pyplot as plt #Biblioteca para visualização de dados
import seaborn as sns #Biblioteca  suporte para visualização de dados

dados = pd.read_csv('tested.csv') #Carregando os dados do arquivo CSV
dados.info() #Exibindo informações sobre os dados

print(dados.isnull().sum()) #Verificando se há valores nulos nos dados

# Preprocessamento dos dados

def preprocessamento_dados(df):
    #removendo colunas desnecessárias
    df.drop(["PassengerId", "Name" , "Ticket", "Cabin"], axis=1, inplace=True)

    df["Embarked"] = df["Embarked"].fillna("S", inplace=True) #Preenchendo valores nulos com a moda da coluna(ultimo porto para garantir que todos os passageiros tenham embarcado)
    df.drop(["Embarked"]) #Removendo a coluna Embarked, pois não é relevante para prever a sobrevivência dos passageiros

    completar_idades(df) #Chamando a função para completar os valores nulos da coluna Age

    df["Sex"] = df["Sex"].map({"male": 1, "female": 0}) #Convertendo a coluna Sex para valores numéricos

    #Tabelas auxiliares para o modelo
    df["Tamanho_Familia"] = df["SibSp"] + df["Parch"]  #Criando uma nova coluna com o tamanho da família do passageiro
    df['Sozinho'] = np.where(df['Tamanho_Familia'] == 0, 1, 0) #Criando uma nova coluna para indicar se o passageiro estava sozinho ou não
    df["faixa_etaria"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100], labels=False) #Criando uma nova coluna com a faixa etária do passageiro

    return df
#A idade dos passageiros variam conforme a classe em que se encontram, sendo assim, é necessário completar os valores nulos da coluna Age com a média da idade dos passageiros de cada classe.
def completar_idades(df):
    mapaidades= {}
    for pclass in df["Pclass"].unique():
        if pclass not in mapaidades:
            mapaidades[pclass] = df[df["Pclass"] == pclass]["Age"].median() #Calculando a mediana da idade dos passageiros de cada classe

    df["Age"].fillna(df["Pclass"].map(mapaidades), inplace=True) #Preenchendo os valores nulos da coluna Age com a mediana da idade dos passageiros da mesma classe
