# 🔍 Projeto Classificação SONAR – Rocha ou Mina

Este projeto utiliza técnicas de **aprendizado de máquina** para prever se um objeto detectado por um sonar é uma **rocha (R)** ou uma **mina submarina (M)**, com base em 60 atributos numéricos extraídos de sinais sonoros.

---

## ⚙️ Tecnologias Utilizadas

- **Python** – Linguagem de programação
- **Pandas** – Manipulação e análise de dados
- **NumPy** – Cálculos numéricos e vetoriais
- **Matplotlib** – Visualização de dados
- **Scikit-learn (sklearn)** – Algoritmos de Machine Learning e ferramentas de avaliação
- **Pickle** – Serialização do modelo treinado

---

## 📁 Dados Utilizados

- **Arquivo**: `sonar_data.csv`
- **Formato**: CSV (sem cabeçalho)
- **Descrição**:
  - 60 colunas com valores numéricos representando a intensidade do retorno do sonar.
  - 1 coluna com o rótulo: `R` (Rocha) ou `M` (Mina).

---

## 🚀 Como Executar o Projeto

### 1. Clonar o Repositório

```bash
git clone https://https://github.com/HeitorDalla/ml_rock_mine
```

### 2. Instalar Dependências

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 3. Executar o Script

Certifique-se de que o arquivo `sonar_data.csv` está no mesmo diretório do script.

```bash
python index.py
```

---

## 🔎 Lógica do Projeto

### 📥 1. Leitura e Análise dos Dados

- Leitura do arquivo CSV sem cabeçalho.
- Verificação de tipos, ausência de valores nulos e descrição estatística dos dados.

### 🔀 2. Pré-processamento

- Separação dos dados:
  - `X` → características (colunas 0 a 59)
  - `y` → rótulo (coluna 60)

- Divisão em conjunto de treino e teste (`train_test_split`)

### 🌲 3. Treinamento com Random Forest

- Treinamento inicial com parâmetros padrão.
- Avaliação usando validação cruzada (`cross_val_score` com `cv=10`).

### 🎯 4. Otimização com RandomizedSearchCV

- Busca pelos melhores hiperparâmetros entre:
  - `max_depth`
  - `max_features`
  - `min_samples_leaf`
  - `min_samples_split`
  - `n_estimators`
- Avaliação final com novo `cross_val_score`.

### 💾 5. Salvamento do Modelo

- O melhor modelo é salvo como `final_model.pkl` utilizando `pickle`.

---

## ✅ Exemplo de Saídas no Terminal

```bash
A acurácia média inicial é: 0.80
Melhores hiperparâmetros: {'n_estimators': 100, 'max_depth': None, ...}
A acurácia média do melhor modelo é: 0.84
```

---

## 📈 Avaliação

- **Métrica principal**: Acurácia
- **Técnica de avaliação**: Validação cruzada (K-Fold)

---

## 🧠 Possíveis Melhorias Futuras

- Testar outros algoritmos como SVM, KNN ou XGBoost.
- Aplicar normalização/padronização nos dados.
- Visualizar os dados com PCA ou t-SNE.
- Implementar interface web com Streamlit.

---

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma **issue** ou enviar um **pull request** com sugestões, melhorias ou correções.

---

## 👨‍💻 Autor

- **Heitor Giussani Dalla Villa**  
- 📧 [heitorvillavilla@gmail.com](mailto:heitorvillavilla@gmail.com)  
- 🔗 [LinkedIn](https://www.linkedin.com/in/heitordallavilla)

---

## 📝 Observações Finais

> Este projeto é uma demonstração prática de como aplicar algoritmos de aprendizado de máquina em dados reais.  
> O dataset foi retirado do repositório da UCI (University of California, Irvine):  
> [https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar,+Mines+vs.+Rocks%29](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar,+Mines+vs.+Rocks%29)
