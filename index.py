import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import pickle

# Cenário - Por meio de um SONAR, estai prevendo se é um rocha (objeto) ou uma mina.

# Estruturação dos Dados
df = pd.read_csv('sonar_data.csv', header=None) # carregar o dataset sem cabeçalho

print(df.shape)

print(df.dtypes)

print(df.describe())

print(df.isna().sum())

# Separa os dados dependentes e independentes
X = df.drop(columns=60, axis=1)
y = df[60]

# Treinar os dados
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Escolha do Modelo
model = RandomForestClassifier(n_jobs=1, random_state=42)

# Treinar o modelo
training = model.fit(X_train, y_train)

# Fazendo previsões
y_preds = model.predict(X_test)

# Avaliação do Modelo por meio de Métricas
test_accuracy = cross_val_score(model,
                                 X_train,
                                 y_train,
                                 cv=10,
                                 scoring='accuracy')

print("A acurácia média inicial é: {}" .format(np.mean(test_accuracy)))

# Melhorar os hiperparâmetros
grid = {
    "max_depth": [None, 5, 10, 20, 30],
    "max_features": ['sqrt', 'log2'],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 4, 6],
    "n_estimators": [10, 100, 200, 500]
}

rs_model = RandomizedSearchCV(model,
                              param_distributions=grid,
                              n_iter=10,
                              cv=5,
                              verbose=2)

# Modelo parametrizado treinado
rs_training = rs_model.fit(X_train, y_train)

# Melhor model
best_model = rs_model.best_estimator_
print("Melhores hiperparâmetros: {}" .format(rs_model.best_params_))

# Fazendo previsão com o modelo parametrizado
rs_y_preds = best_model.predict(X_test)

# Avaliação com o modelo treinado
rs_metrics = cross_val_score(best_model,
                             X_train,
                             y_train,
                             cv=10,
                             scoring='accuracy')

print("A acurácia média do melhor modelo é: {}" .format(np.mean(rs_metrics)))

# Salvar modelo
pickle.dump(best_model, open("final_model.pkl", 'wb'))