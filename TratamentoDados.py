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
    # Removendo colunas desnecessárias (Ajustado)
    df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

    # 1. Preenchendo nulos sem usar inplace junto com atribuição (Corrigido)
    df["Embarked"] = df["Embarked"].fillna("S") 

    # 2. Tratando o valor nulo do FARE (Adicione esta linha!)
    # Preenche o valor nulo da tarifa com a mediana de todos os passageiros
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    
    # 3. Removendo a coluna Embarked indicando que é uma coluna (Corrigido)
    df.drop(["Embarked"], axis=1, inplace=True)

    # 4. Chamando a função para completar as idades
    completar_idades(df) 

    # 5. Convertendo a coluna Sex para valores numéricos
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0}) 

    # Tabelas auxiliares para o modelo
    df["Tamanho_Familia"] = df["SibSp"] + df["Parch"]  
    df['Sozinho'] = np.where(df['Tamanho_Familia'] == 0, 1, 0) 
    df["faixa_etaria"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100], labels=False) 

    return df

def completar_idades(df):
    mapaidades = {}
    for pclass in df["Pclass"].unique():
        if pclass not in mapaidades:
            mapaidades[pclass] = df[df["Pclass"] == pclass]["Age"].median() 

    # Removido o inplace=True para seguir as regras do Copy-on-Write (Corrigido)
    df["Age"] = df["Age"].fillna(df["Pclass"].map(mapaidades)) 
