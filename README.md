# 🍷 Análise de Dados – Classificação de Vinhos com Machine Learning

Este projeto foi desenvolvido como parte do **Projeto Integrador** do curso de Ciência de Dados e Inteligência Artificial da **Faculdade UniDomBosco**. O objetivo é realizar uma análise exploratória e aplicar técnicas de machine learning para classificar diferentes tipos de vinho tinto com base em suas características físico-químicas.

---

## 🚀 Tecnologias e Bibliotecas Utilizadas

- **Python 3.x** – Linguagem principal  
- **VS Code** – Ambiente de desenvolvimento  
- **Git & GitHub** – Controle de versão  
- **Bibliotecas Python:**
  - [`os`](https://docs.python.org/3/library/os.html) – Manipulação de caminhos e comandos do sistema  
  - [`pandas`](https://pandas.pydata.org/) – Leitura e manipulação de dados  
  - [`matplotlib`](https://matplotlib.org/) – Visualização gráfica  
  - [`seaborn`](https://seaborn.pydata.org/) – Visualização estatística com gráficos mais elegantes  
  - [`scikit-learn`](https://scikit-learn.org/stable/) –  
    - `train_test_split` – Divisão dos dados para treino e teste  
    - `StandardScaler` – Normalização dos dados  
    - `RandomForestClassifier` – Modelo de classificação por floresta aleatória  
    - `classification_report`, `confusion_matrix` – Avaliação de desempenho dos modelos

---

## 📂 Dataset

O dataset utilizado é **público** e está disponível no repositório. Ele contém informações físico-químicas de diferentes tipos de vinho tinto, com a respectiva **classificação** de qualidade.

- **Nome do arquivo:** `winequality-red.csv`  
- **Fonte dos dados:** [Wine Quality Data Set – UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## 🧪 Funcionalidades

- Importação e pré-processamento de dados  
- Treinamento de modelo de classificação  
- Avaliação do desempenho com métricas como matriz de confusão e relatório de classificação  
- Verificação automática de dependências (instalação via `pip` se necessário)

---

## 📁 Estrutura do Projeto

```bash
📦 analise-vinhos
 ┣ 📄 analise_vinho.py
 ┣ 📄 winequality-red.csv
 ┣ 📄 README.md
