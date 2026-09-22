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

X = dados.drop(["Survived", "Sex"], axis=1) # Remove o Sex também!
y = dados["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #Dividindo os dados em treino e teste

scaler = MinMaxScaler() #Instanciando o objeto MinMaxScaler
X_train = scaler.fit_transform(X_train) #Normalizando os dados de treino
X_test = scaler.transform(X_test) #Normalizando os dados de teste

def hiperparametros_knn(X_train, y_train):
  # No seu main.py, mude o param_grid para isso:
    param_grid = {
    # Começa em 3 e pula de 2 em 2 (3, 5, 7, 9... até 19)
    "n_neighbors": list(range(3, 21, 2)), 
    
    "metric": ["euclidean", "manhattan", "minkowski"],
    
    # Força a votação ser uniforme para evitar peso infinito no próprio ponto
    "weights": ["uniform"] 
}


    model = KNeighborsClassifier() #Instanciando o modelo KNN
    gridsearch = GridSearchCV(model, param_grid, cv=5) #Instanciando o objeto GridSearchCV
    gridsearch.fit(X_train, y_train) #Treinando o modelo com os dados de treino

    return gridsearch.best_estimator_ #Retornando os melhores parâmetros encontrados

melhor_modelo = hiperparametros_knn(X_train, y_train) 

def avaliar_modelo(X_test, y_test, model):
    predicoes = model.predict(X_test) #Fazendo previsões com os dados de teste 
    acuracia = accuracy_score(y_test, predicoes) #Calculando a acurácia do modelo
    matriz_confusao = confusion_matrix(y_test, predicoes) #Calculando a matriz de confusão
    return acuracia, matriz_confusao #Retornando a acurácia e a matriz de confusão

# Chamando a função passando o objeto do modelo
acuracia, matriz_confusao = avaliar_modelo(X_test, y_test, melhor_modelo) 

print(f'\n(Acurácia: {acuracia*100:.2f}%)') #Exibindo a acurácia do modelo

# Adicione isso para ver como ele se comporta nos dados que ele usou para treinar
acuracia_treino, _ = avaliar_modelo(X_train, y_train, melhor_modelo)
print(f"Acurácia no Treino: {acuracia_treino*100:.2f}%")

# Adicione isso para ver a importância das variáveis (ou ver o que o KNN escolheu)
print(f"Melhores parâmetros escolhidos: {melhor_modelo.get_params()}")

print(f'Matriz de Confusão:\n{matriz_confusao}') #Exibindo a matriz de confusão

# Exibindo as configurações específicas do melhor modelo encontrado
print(f'Melhores Parâmetros: {melhor_modelo.get_params()}') 