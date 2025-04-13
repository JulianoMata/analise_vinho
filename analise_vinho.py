# Faculdade UniDombosco - Projeto Integrador em Ciências de Dados e Inteligência Artificial V A
# Aluno: Juliano França da Mata
# Professora: Priscila Louise Leyser Santin
# Data: 13/04/2025

# Importação das bibliotecas necessárias para análise de dados e visualização
import os  # Manipulação de caminhos e comandos do sistema

# Verificação e instalação automática das bibliotecas, se necessário
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split  # Para dividir os dados
    from sklearn.preprocessing import StandardScaler  # Para normalizar os dados
    from sklearn.ensemble import RandomForestClassifier  # Modelo de machine learning
    from sklearn.metrics import classification_report, confusion_matrix  # Avaliação do modelo
except ModuleNotFoundError:
    os.system('pip install pandas matplotlib seaborn scikit-learn')
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix

# Definir caminho do arquivo CSV com os dados dos vinhos como uma única string raw
caminho = r"D:\FACULDADE_DOMBOSCO\Disciplinas\5_MODULAR\06-Projeto_Integrador_em_Ciências_de_Dados_e_Inteligencia_Artificial_V\winequality-red.csv"

# 1. Carregar os dados do arquivo CSV (separador ';' usado nesse dataset)
df = pd.read_csv(caminho, sep=';')
df.columns = df.columns.str.strip()  # Remover espaços em branco dos nomes das colunas

# 2. Visualizar as primeiras linhas do dataset para ter uma ideia da estrutura dos dados
print("Visualizando as primeiras linhas do dataset:")
print(df.head())

# 3. Exibir informações gerais sobre os dados: tipo de dado, valores não nulos etc.
print("\nInformações gerais do dataset:")
print(df.info())

# 4. Estatísticas descritivas básicas: média, desvio padrão, mínimo, máximo etc.
print("\nEstatísticas descritivas:")
print(df.describe())

# 5. Verificar se há valores ausentes em alguma coluna
total_nulos = df.isnull().sum()
print("\nValores ausentes por coluna:") 
print(total_nulos)
if total_nulos.sum() == 0:
    print("\nNão há valores ausentes no dataset.")

# 6. Visualizar a distribuição da variável alvo 'quality' (qualidade dos vinhos)
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df, hue='quality', palette='viridis', legend=False)
plt.title("Distribuição da Qualidade do Vinho")
plt.xlabel("Qualidade")
plt.ylabel("Contagem")
plt.show()

# 7. Matriz de correlação: identificar relações entre variáveis numéricas
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matriz de Correlação das Variáveis")
plt.show()

# 8. Distribuição da acidez fixa para entender sua densidade
plt.figure(figsize=(8, 5))
df['fixed acidity'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title("Distribuição da Acidez Fixa")
plt.xlabel("Acidez Fixa")
plt.ylabel("Frequência")
plt.show()

# 9. Boxplots para detectar outliers em variáveis importantes
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['alcohol', 'volatile acidity', 'residual sugar', 'pH']], palette='Set2')
plt.title("Boxplot de Variáveis para Identificação de Outliers")
plt.show()

# 10. Relação entre teor alcoólico e qualidade do vinho
plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='alcohol', data=df, hue='quality', palette='magma', legend=False)
plt.title("Relação entre Teor Alcoólico e Qualidade")
plt.xlabel("Qualidade")
plt.ylabel("Teor Alcoólico")
plt.show()

# 11. Agrupar a variável 'quality' em categorias: baixa (0), média (1), alta (2)
def categorizar_qualidade(valor):
    if valor <= 4:
        return 0  # baixa
    elif valor <= 6:
        return 1  # média
    else:
        return 2  # alta

df['qualidade_cat'] = df['quality'].apply(categorizar_qualidade)

# Visualizar nova distribuição após o agrupamento
plt.figure(figsize=(8, 5))
sns.countplot(x='qualidade_cat', data=df, hue='qualidade_cat', palette='plasma', legend=False)
plt.title("Distribuição Agrupada da Qualidade do Vinho")
plt.xlabel("Categorias de Qualidade (0=Baixa, 1=Média, 2=Alta)")
plt.ylabel("Contagem")
plt.show()

# 12. Preparação dos dados para o modelo de machine learning
X = df.drop(['quality', 'qualidade_cat'], axis=1)  # Atributos (features)
y = df['qualidade_cat']  # Variável alvo categorizada

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 13. Treinar um modelo de classificação (Random Forest)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 14. Avaliar o modelo
y_pred = modelo.predict(X_test)
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, zero_division=0)) 

# Isso significa:
# - Dos 11 vinhos realmente da Classe 0, todos foram classificados erroneamente como Classe 1.
# - Dos 262 vinhos da Classe 1, 250 foram corretamente classificados e 12 foram confundidos com Classe 2.
# - Dos 47 vinhos da Classe 2, 26 foram corretamente classificados, mas 21 foram confundidos com Classe 1.

"""Embora a Classe 0 não tenha sido corretamente classificada pelo modelo, esse comportamento é aceitável dentro do escopo educacional do projeto. Futuramente, técnicas de balanceamento de dados ou ajuste de hiperparâmetros poderiam ser exploradas para melhorar a performance."""

# Relatório de Classificação:
# Mostra o desempenho do modelo em cada classe (0 = Ruim, 1 = Regular, 2 = Boa)
# Métricas:
# - precision: acertos entre os que foram previstos como aquela classe
# - recall: acertos entre os que realmente pertencem àquela classe
# - f1-score: equilíbrio entre precision e recall
# - support: total real de exemplos daquela classe

# Observações:
# - O modelo teve ótimo desempenho na classe "Regular" (classe 1)
# - Teve dificuldades nas classes "Ruim" (0) e "Boa" (2), possivelmente por poucos exemplos (desequilíbrio de classes)
# - A acurácia geral foi de 86%, o que é um bom resultado para um projeto educacional
# - A média ponderada das métricas também indica desempenho consistente no geral

""" Por ser um exercício inicial, não faremos ajustes finos no modelo neste momento"""

# Conclusão:
# A análise inicial forneceu uma visão geral do conjunto de dados, incluindo a estrutura e estatísticas básicas.
# A matriz de correlação identificou variáveis com maior influência na qualidade do vinho.
# A distribuição da acidez e da qualidade revelou padrões importantes.
# Boxplots ajudaram a detectar outliers que podem impactar modelos futuros.
# A relação entre teor alcoólico e qualidade mostrou uma tendência clara de maior qualidade com mais álcool.
# 
# Também foi aplicado um modelo de Random Forest para prever a qualidade do vinho com base nas variáveis do dataset.
# Neste caso, a qualidade foi agrupada em três categorias (baixa, média e alta) para facilitar a predição.
# As métricas de avaliação ajudam a entender a performance e precisão do modelo com essa nova abordagem.
# O modelo pode ser melhorado com ajustes de hiperparâmetros e validação cruzada. 