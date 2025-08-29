
# Análise de Dados do Airbnb no Rio de Janeiro

## 📜 Visão Geral do Projeto

Este projeto tem como objetivo principal a **limpeza, transformação e análise exploratória de dados (EDA)** de anúncios do Airbnb no Rio de Janeiro. Utilizando Python e bibliotecas como Pandas e Matplotlib, exploramos o mercado de hospedagem para identificar padrões, tendências de preços, tipos de acomodação e a relação entre variáveis. O resultado é um conjunto de insights valiosos que podem ser utilizados para futuras tomadas de decisão ou para a construção de modelos preditivos.

## 📊 Estrutura dos Dados

O projeto utiliza dois conjuntos de dados principais:

  * **`listings.csv`**: Contém informações detalhadas sobre os anúncios, como localização, tipo de acomodação, capacidade, número de quartos e preço.
  * **`reviews.csv`**: Contém as avaliações dos hóspedes, incluindo o número total de avaliações e a pontuação média.

## ⚙️ Ferramentas e Metodologia

  * **Linguagem:** Python
  * **Bibliotecas:**
      * `pandas` e `numpy` para manipulação de dados.
      * `matplotlib` e `seaborn` para visualização e análise de dados.
      * `sklearn.preprocessing` para transformação de variáveis categóricas.
  * **Metodologia:**
    1.  **Entendimento dos Dados:** Importação e inspeção inicial dos datasets.
    2.  **Mesclagem (Merge):** Combinação das tabelas de anúncios e avaliações em um único DataFrame para análise integrada.
    3.  **Limpeza de Dados:**
          * Conversão de tipos de dados.
          * Tratamento de valores ausentes e duplicados.
          * Identificação e remoção de *outliers* na variável `price` usando o método **IQR (Intervalo Interquartil)**.
    4.  **Análise Exploratória (EDA):** Geração de visualizações (boxplots, gráficos de barras e histogramas) e matriz de correlação para extrair insights.

## 💻 Código e Análises

### Configuração Inicial e Carregamento dos Dados

```python
# Data Wrangling
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Padronizando visualizações
sns.set_theme(
    style = 'whitegrid',
    font_scale = 1.1,
    rc= {'figure.figsize': (8, 6),
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'axes.formatter.useoffset': False,
        'axes.formatter.limits': (-4, 4)
        }
    )

# Importando os datasets
df_listing = pd.read_csv('../Data/Raw/listings_cleaned.csv')
df_reviews = pd.read_csv('../Data/Raw/reviews.csv')

# Mesclando os dataframes
df = pd.merge(df_listing, df_reviews)

# Convertendo colunas para tipo inteiro
convert_cols = ['bathrooms', 'bedrooms', 'beds']
df[convert_cols] = df[convert_cols].astype(int)

# Codificando a variável 'room_type'
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['room_type_encoder'] = encoder.fit_transform(df['room_type'])

# Removendo outliers da variável 'price'
q1 = df['price'].quantile(0.25)
q3 = df['price'].quantile(0.75)
IQR = q3 - q1
upper_bound = q3 + 1.5 * IQR
lower_bound = q1 - 1.5 * IQR
df_filtred = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
```

### Análise Exploratória e Visualizações

#### 1\. Distribuição de Preços (Após Remoção de Outliers)

**Código:**

```python
plt.figure(figsize=(10, 6))
sns.histplot(df_filtred['price'], bins=50, kde=True, color='skyblue')
plt.title('Distribuição de Preços')
plt.xlabel('Preço')
plt.ylabel('Frequência')
plt.show()
```

**Insights:**

  * A maioria dos estabelecimentos se concentra na faixa de preço entre **R$ 250,00 e R$ 1.000,00**.
  * O pico de concentração de anúncios está em torno de **R$ 500,00**, indicando o preço mais comum no mercado.

-----

#### 2\. Top 10 Bairros com as Melhores Avaliações

**Código:**

```python
melhores_imoveis_por_bairro = df_filtred[df_filtred['review_scores_rating'] == 5.0]
contagem_por_bairro = melhores_imoveis_por_bairro['neighbourhood_cleansed'].value_counts().head(10)

sns.barplot(x=contagem_por_bairro.values, y=contagem_por_bairro.index, palette='viridis')
plt.title('Top 10 Bairros com Mais Imóveis Avaliados com Nota 5.0')
plt.xlabel('Número de Imóveis')
plt.ylabel('Bairro')
plt.show()
```

**Insights:**

  * **Copacabana** e **Barra da Tijuca** se destacam, indicando que a qualidade das acomodações e a experiência do hóspede nessas áreas são consistentemente altas.



#### 3\. Tipos de Acomodação por Preço Médio

**Código:**

```python
precos_por_quarto = df_filtred.groupby('room_type')['price'].mean().sort_values(ascending=False)
sns.barplot(precos_por_quarto, palette='pastel')
plt.title('Tipos de quartos com maiores preços médios')
plt.xlabel('Tipo de quarto')
plt.ylabel('Preço')
plt.show()
```

**Insights:**

  * Aluguéis de **casas inteiras ou apartamentos** são os mais caros, seguidos por quartos de hotéis.
  * **Quartos privativos** e **compartilhados** tendem a ter um valor significativamente menor, refletindo sua oferta de espaço e privacidade.

-----

#### 4\. Matriz de Correlação

**Código:**

```python
numeric_cols = ['price', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 'number_of_reviews']
correlation_matrix = df_filtred[numeric_cols].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação entre Variáveis Numéricas')
plt.show()
```

**Insights:**

  * A variável `price` possui uma correlação moderada com o número de `accommodates`, `bedrooms` e `beds`, sugerindo que o preço está mais relacionado à capacidade do imóvel do que com a pontuação de avaliações.
  * O `review_scores_rating` não possui uma alta correlação com o preço ou número de reviews, indicando que a qualidade da avaliação não é diretamente proporcional a um preço mais alto ou ao volume de avaliações.