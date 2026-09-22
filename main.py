import pandas as pd #Biblioteca para manipulação de dados
import numpy as np #Biblioteca para cálculos matemáticos

from sklearn.model_selection import train_test_split #Biblioteca para dividir os dados em treino e teste
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

