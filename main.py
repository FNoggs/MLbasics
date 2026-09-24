import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregando dados
dados = pd.read_csv('tested.csv')

# 2. Pré-processamento refinado
def preprocessamento_dados(df):
    # Extrair título dos nomes ANTES de apagar a coluna Name
    df['Titulo'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    
    # Mapear títulos raros
    titulos_comuns = ['Mr', 'Miss', 'Mrs', 'Master']
    df['Titulo'] = df['Titulo'].apply(lambda x: x if x in titulos_comuns else 'Raro')
    df = pd.get_dummies(df, columns=['Titulo'], drop_first=True)

    # Preenchimento de nulos
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    
    # Preencher idades pela mediana do título/classe
    df["Age"] = df["Age"].fillna(df.groupby("Pclass")["Age"].transform("median"))

    # Mapear Sex (0 e 1)
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    # Engenharia de recursos
    df["Tamanho_Familia"] = df["SibSp"] + df["Parch"] + 1
    df['Sozinho'] = np.where(df['Tamanho_Familia'] == 1, 1, 0)
    
    # Remover colunas que não ajudam na predição
    df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1, inplace=True)
    
    return df

dados = preprocessamento_dados(dados)

# 3. Divisão de treino e teste
X = dados.drop(["Survived", "Sex"], axis=1) 
y = dados["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Modelo Random Forest com Regularização (Combate ao Overfitting)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 4, 5],            # Limitar a profundidade evita que o modelo decore o Sexo
    'min_samples_split': [5, 10],       # Exige mais amostras para criar uma regra
    'min_samples_leaf': [2, 4]          # Evita nós folha com pouquíssimos dados
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

melhor_rf = grid_search.best_estimator_

# 5. Avaliação
y_pred_treino = melhor_rf.predict(X_train)
y_pred_teste = melhor_rf.predict(X_test)

print(f"Acurácia no Treino: {accuracy_score(y_train, y_pred_treino)*100:.2f}%")
print(f"Acurácia no Teste:  {accuracy_score(y_test, y_pred_teste)*100:.2f}%")

# Ver a importância de cada coluna no modelo
importancias = pd.Series(melhor_rf.feature_importances_, index=X.columns)
print("\nImportância das Variáveis:")
print(importancias.sort_values(ascending=False))