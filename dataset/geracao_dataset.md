# Documentação: Geração do Dataset Sintético de Consumo de Energia

**Arquivo gerado:** `dataset_consumo_energia.csv` / `dataset_consumo_energia.arff`  
**Total de instâncias:** 520  
**Total de colunas:** 7  
**Objetivo:** Treinamento de modelos de Machine Learning para previsão de demanda energética multissetorial

---

## 1. Visão Geral

O dataset simula o consumo de energia elétrica em três setores de atividade (Residencial, Comercial e Industrial), incorporando variáveis contextuais como temperatura, área da unidade e número de ocupantes. O processo de geração foi inteiramente feito em Python com as bibliotecas `pandas` e `numpy`, seguindo especificações rigorosas de distribuição estatística, injeção de ruído e inserção de outliers.

---

## 2. Estrutura das Colunas

| # | Coluna | Tipo | Descrição |
|---|--------|------|-----------|
| 1 | `Setor_Atividade` | Categórico | Setor da unidade consumidora |
| 2 | `Dia_da_Semana` | Categórico | Dia da semana da medição |
| 3 | `Temperatura_Media` | Numérico (float) | Temperatura média do dia em °C |
| 4 | `Area_Metragem` | Numérico (float) | Área da unidade em m² |
| 5 | `Ocupantes_Ativos` | Numérico (inteiro) | Pessoas ou máquinas em operação |
| 6 | `Num_Extintores` | Numérico (inteiro) | Atributo irrelevante (ruído de domínio) |
| 7 | `Consumo_Energia` | Numérico (float) | **Variável-alvo** — consumo em kWh |

---

## 3. Geração de Cada Coluna

### 3.1 `Setor_Atividade`

Sorteio aleatório uniforme entre os três setores com `numpy.random.choice`, sem peso diferenciado, resultando em distribuição aproximadamente igual (~173 instâncias por setor).

```python
setores = np.random.choice(["Residencial", "Comercial", "Industrial"], size=N)
```

Valores possíveis: `Residencial`, `Comercial`, `Industrial`  
Permite NaN: **Não**

---

### 3.2 `Dia_da_Semana`

Sorteio aleatório uniforme entre os sete dias da semana.

```python
dias = np.random.choice(
    ["Segunda","Terça","Quarta","Quinta","Sexta","Sábado","Domingo"],
    size=N
)
```

Distribuição esperada: ~74 instâncias por dia  
Permite NaN: **Não**

---

### 3.3 `Temperatura_Media`

Gerada com distribuição **normal** (gaussiana), com média de 22°C e desvio padrão de 5°C. Após a geração, os valores são clipados ao intervalo realista de 15°C a 38°C.

```python
temp = np.random.normal(22, 5, N)
temp = np.clip(temp, 15, 38).round(1)
```

| Parâmetro | Valor |
|-----------|-------|
| Distribuição | Normal |
| Média | 22°C |
| Desvio padrão | 5°C |
| Mínimo (clip) | 15°C |
| Máximo (clip) | 38°C |
| Casas decimais | 1 |

Permite NaN: **Sim** (5%)

---

### 3.4 `Area_Metragem`

Gerada com distribuição uniforme dentro de faixas específicas por setor, refletindo o porte típico de cada tipo de unidade.

```python
for i, s in enumerate(setores):
    if s == "Residencial": area[i] = np.random.uniform(60, 200)
    elif s == "Comercial":  area[i] = np.random.uniform(100, 800)
    else:                   area[i] = np.random.uniform(500, 5000)
```

| Setor | Faixa (m²) |
|-------|-----------|
| Residencial | 60 – 200 |
| Comercial | 100 – 800 |
| Industrial | 500 – 5.000 |

Permite NaN: **Sim** (5%)

---

### 3.5 `Ocupantes_Ativos`

Número inteiro gerado com distribuição uniforme discreta por setor. Representa pessoas (residencial/comercial) ou máquinas em operação (industrial).

```python
for i, s in enumerate(setores):
    if s == "Residencial": ocup[i] = np.random.randint(2, 7)
    elif s == "Comercial":  ocup[i] = np.random.randint(10, 81)
    else:                   ocup[i] = np.random.randint(20, 201)
```

| Setor | Faixa |
|-------|-------|
| Residencial | 2 – 6 |
| Comercial | 10 – 80 |
| Industrial | 20 – 200 |

Permite NaN: **Sim** (5%)

---

### 3.6 `Num_Extintores`

Atributo **deliberadamente irrelevante**, inserido para simular ruído de domínio e testar a capacidade dos modelos de ML de descartar features sem poder preditivo. Gerado com distribuição uniforme discreta entre 1 e 20, sem qualquer relação com o consumo.

```python
extin = np.random.randint(1, 21, size=N).astype(float)
```

Permite NaN: **Sim** (5%)

---

### 3.7 `Consumo_Energia` (Variável-Alvo)

O consumo é calculado em três etapas: **cálculo base determinístico**, **adição de ruído gaussiano** e **injeção de outliers**.

#### Etapa 1 — Fórmula base por setor

Cada setor possui uma equação linear que combina área, ocupantes e temperatura:

| Setor | Fórmula |
|-------|---------|
| Residencial | `100 + 0.8 × área + 15 × ocupantes + 2 × (T − 22)` |
| Comercial | `200 + 0.5 × área + 5 × ocupantes + 3 × (T − 22)` |
| Industrial | `500 + 0.3 × área + 8 × ocupantes + 5 × (T − 22)` |

O termo `(T − 22)` captura o efeito da temperatura em relação à média histórica de 22°C, com sensitividade crescente do setor residencial ao industrial.

#### Etapa 2 — Ruído gaussiano

Após o cálculo base, é somado ruído gaussiano proporcional ao consumo de cada instância (5% de desvio padrão), simulando erros de medição realistas:

```python
consumo += np.random.normal(0, consumo * 0.05)
consumo = np.abs(consumo).round(1)
```

#### Etapa 3 — Outliers (ver Seção 4)

Permite NaN: **Não** — a variável-alvo não possui valores ausentes.

---

## 4. Outliers

Foram injetados outliers em aproximadamente **2% das instâncias** (~10 registros), com consumo igual a **10× a média típica do setor**, mais variação gaussiana. O objetivo é simular consumos anômalos para teste de robustez dos modelos.

```python
n_outliers = int(N * 0.02)
outlier_idx = np.random.choice(N, n_outliers, replace=False)
for i in outlier_idx:
    s = setores[i]
    if s == "Residencial": mean_s = 350
    elif s == "Comercial":  mean_s = 800
    else:                   mean_s = 3000
    consumo[i] = mean_s * 10 + np.random.normal(0, mean_s)
```

| Setor | Média típica (kWh) | Consumo outlier (~10×) |
|-------|-------------------|------------------------|
| Residencial | ~350 | ~3.500 |
| Comercial | ~800 | ~8.000 |
| Industrial | ~3.000 | ~30.000 |

Os outliers são distribuídos aleatoriamente entre as linhas e setores, sem concentração intencional.

---

## 5. Valores Ausentes (NaN)

Exatamente **5% de NaN** foi inserido aleatoriamente nas quatro colunas numéricas que permitem valores ausentes. A distribuição é independente por coluna, ou seja, diferentes linhas recebem NaN em cada atributo.

```python
nan_cols = ["Temperatura_Media", "Area_Metragem", "Ocupantes_Ativos", "Num_Extintores"]
n_nan_per_col = int(N * 0.05)  # = 26 por coluna
for col in nan_cols:
    nan_idx = np.random.choice(N, n_nan_per_col, replace=False)
    df.loc[nan_idx, col] = np.nan
```

| Coluna | NaN inseridos | % do total |
|--------|--------------|-----------|
| `Temperatura_Media` | 26 | 5,0% |
| `Area_Metragem` | 26 | 5,0% |
| `Ocupantes_Ativos` | 26 | 5,0% |
| `Num_Extintores` | 26 | 5,0% |
| `Setor_Atividade` | 0 | 0% |
| `Dia_da_Semana` | 0 | 0% |
| `Consumo_Energia` | 0 | 0% |

---

## 6. Formato de Saída

### 6.1 CSV (`dataset_consumo_energia.csv`)

| Especificação | Valor |
|---------------|-------|
| Separador | vírgula (`,`) |
| Encoding | UTF-8 |
| Separador decimal | ponto (`.`) |
| Cabeçalho | Sim (primeira linha) |
| Índice de linha | Não |
| NaN representado por | célula vazia |

### 6.2 ARFF (`dataset_consumo_energia.arff`)

Formato compatível com **Weka** e ferramentas de ML que utilizam o padrão ARFF (Attribute-Relation File Format).

Adaptações realizadas para compatibilidade ARFF:
- Acentos removidos nas categorias: `Terça → Terca`, `Sábado → Sabado`
- Valores ausentes representados por `?` (padrão ARFF)
- Colunas categóricas declaradas com `{valor1,valor2,...}`
- Colunas numéricas declaradas como `NUMERIC`

Exemplo do cabeçalho ARFF gerado:

```
@RELATION dataset_consumo_energia

@ATTRIBUTE Setor_Atividade {Residencial,Comercial,Industrial}
@ATTRIBUTE Dia_da_Semana {Segunda,Terca,Quarta,Quinta,Sexta,Sabado,Domingo}
@ATTRIBUTE Temperatura_Media NUMERIC
@ATTRIBUTE Area_Metragem NUMERIC
@ATTRIBUTE Ocupantes_Ativos NUMERIC
@ATTRIBUTE Num_Extintores NUMERIC
@ATTRIBUTE Consumo_Energia NUMERIC

@DATA
...
```

---

## 7. Reprodutibilidade

A semente aleatória foi fixada em `42` via `numpy.random.seed(42)`, garantindo que o dataset possa ser reproduzido identicamente em qualquer execução com o mesmo código.

```python
np.random.seed(42)
```

---

## 8. Dependências

| Biblioteca | Versão recomendada | Uso |
|------------|--------------------|-----|
| `numpy` | ≥ 1.23 | Geração de números aleatórios e operações vetoriais |
| `pandas` | ≥ 1.5 | Construção do DataFrame, NaN e exportação CSV |

---

## 9. Exemplo das Primeiras Linhas

```
Setor_Atividade,Dia_da_Semana,Temperatura_Media,Area_Metragem,Ocupantes_Ativos,Num_Extintores,Consumo_Energia
Industrial,Segunda,20.1,3736.4,51,14,1931.6
Residencial,Segunda,18.7,92.0,,15,208.1
Industrial,Terça,29.0,4983.5,31,15,2142.6
Industrial,Quarta,15.5,4886.6,194,9,3771.8
Residencial,Quinta,15.0,151.0,,19,240.5
```

---

## 10. Considerações para Uso em ML

- A coluna `Num_Extintores` é um **atributo irrelevante** intencional — modelos com seleção de features devem descartá-la.
- Os **~10 outliers** podem ser tratados ou removidos dependendo da estratégia adotada (robustez vs. acurácia em dados limpos).
- Os **NaN** demandam estratégia de imputação antes do treinamento (média, mediana, KNN imputer, etc.).
- A variável `Consumo_Energia` é contínua, portanto o problema é de **regressão**.
- As variáveis categóricas (`Setor_Atividade`, `Dia_da_Semana`) precisam de encoding (One-Hot, Label ou Ordinal) conforme o algoritmo utilizado.
