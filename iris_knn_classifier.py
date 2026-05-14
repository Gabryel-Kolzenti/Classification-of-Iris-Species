# %% [markdown]
# ## Importação de bibliotecas

# %%
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

# %% [markdown]
# ## Carregamento dos Dados

# %%
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

# %% [markdown]
# ## Análise Exploratória (EDA)

# %%
print(df.info())
df.describe()

# %%
print(df.nunique())
df.head()

# %%
df['species'].value_counts()

# %%
df.corr(numeric_only=True)

# %%
sns.pairplot(df, hue='species')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, gap=0.2)
plt.show()

# %% [markdown]
# ## Conclusão EDA:
# * Dataset com 150 amostras balanceadas entre 3 classes;
# * Possui 4 variáveis numéricas;
# * Não apresenta valores nulos;
# * Dados limpos e estruturados;
# * Possíveis outliers em sepal width;
# * Forte correlação entre variáveis relacionadas às pétalas;
# * petal length e petal width demonstram maior poder preditivo;
# * A espécie 0 apresenta clara separação das demais;
# * Existe leve sobreposição entre as espécies 1 e 2;
# ### O dataset demonstra boa separabilidade, indicando potencial para alta performance em modelos de classificação.

# %% [markdown]
# ## Separação dos Dados

# %%
X = df.drop(["species"], axis=1)
y = df["species"]
SEED = 32

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=32,
    stratify=y
)

# %% [markdown]
# ## Escolha do Modelo
# O modelo escolhido inicialmente foi o K-Nearest Neighbors (KNN).
# 
# Motivos da escolha:
# - simplicidade;
# - boa performance em datasets pequenos;
# - excelente para problemas com separação relativamente clara;
# - fácil interpretação inicial

# %% [markdown]
# ## Pipeline de Pré-processamento e Modelagem
# 
# Foi utilizado um Pipeline para:
# - organizar o fluxo de transformação;
# - evitar vazamento de dados;
# - tornar o processo reproduzível;
# - facilitar deploy futuro.
# 
# O pipeline contém:
# 1. Padronização dos dados;
# 2. Modelo KNN.

# %%
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# %% [markdown]
# # Ajuste de Hiperparâmetros
# 
# Foi utilizado Grid Search para buscar a melhor combinação de hiperparâmetros porque:
# - o espaço de busca é pequeno;
# - queremos avaliar todas as combinações possíveis;
# - o custo computacional é baixo neste problema;

# %%
param_grid = {
    "knn__n_neighbors": [3,5,7,9],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)

# %% [markdown]
# ## Avaliação Final do Modelo
# 
# Após encontrar a melhor configuração do modelo, é necessário avaliar sua capacidade de generalização.
# 
# Para isso, utilizaremos o conjunto de teste separado no início do projeto.
# 
# As métricas escolhidas foram:
# 
# - Accuracy
# - Classification Report
# - Confusion Matrix
# 
# A acurácia é adequada neste cenário porque:
# - o dataset está balanceado;
# - todas as classes possuem importância semelhante;
# - queremos medir a taxa geral de acertos do modelo.

# %%
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Acurácia do modelo: {accuracy:.4f}")

# %%
print(classification_report(y_test, y_pred))

# %%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")
plt.title("Matriz de Confusão")

plt.show()

# %% [markdown]
# ## Interpretação dos resultados
# 
# Os resultados indicam que o modelo KNN conseguiu separar muito bem as espécies do dataset Iris.
# 
# Isso acontece porque:
# - as classes possuem características relativamente distintas;
# - o dataset possui baixo ruído;
# - as variáveis apresentam boa capacidade discriminatória.

# %% [markdown]
# ## Salvando o modelo

# %%
joblib.dump(
    best_model,
    r"C:\Users\990211\Desktop\Classification-of-Iris-Species\models\iris_knn_model.pkl"
)

print("Processo finalizado, modelo criado e salvo!")


