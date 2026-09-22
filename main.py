from pyexpat import model
import pandas as pd #Biblioteca para manipulação de dados
import numpy as np #Biblioteca para cálculos matemáticos

from sklearn.model_selection import GridSearchCV, train_test_split #Biblioteca para dividir os dados em treino e teste
from sklearn.preprocessing import MinMaxScaler #Biblioteca para normalização dos dados
from sklearn.neighbors import KNeighborsClassifier #Biblioteca para o modelo KNN
from sklearn.metrics import accuracy_score, confusion_matrix #Biblioteca para calcular a acurácia do modelo

import matplotlib.pyplot as plt #Biblioteca para visualização de dados
import seaborn as sns #Biblioteca  suporte para visualização de dados

import TratamentoDados as td #Importando o arquivo TratamentoDados.py

dados = pd.read_csv('tested.csv') #Carregando os dados do arquivo CSV
dados.info() #Exibindo informações sobre os dados

print(dados.isnull().sum()) #Verificando se há valores nulos nos dados

dados = td.preprocessamento_dados(dados) #Chamando a função de preprocessamento

X = dados.drop("Survived", axis=1) #Separando os sobreviventes dos não sobreviventes
y = dados["Survived"] #Separando a variável alvo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #Dividindo os dados em treino e teste

scaler = MinMaxScaler() #Instanciando o objeto MinMaxScaler
X_train = scaler.fit_transform(X_train) #Normalizando os dados de treino
X_test = scaler.transform(X_test) #Normalizando os dados de teste

def hiperparametros_knn(X_train, y_train):
    param_grid = {
    "n_neighbors":range(1, 21), 

    # Corrigido para "metric" e trocado minkowski por cosine
    "metric": ["euclidean", "manhattan", "cosine"],
    
    "weights": ["uniform", "distance"]
}

    model = KNeighborsClassifier() #Instanciando o modelo KNN
    gridsearch = GridSearchCV(model, param_grid, cv=5) #Instanciando o objeto GridSearchCV
    gridsearch.fit(X_train, y_train) #Treinando o modelo com os dados de treino

    return gridsearch.best_params_ #Retornando os melhores parâmetros encontrados

