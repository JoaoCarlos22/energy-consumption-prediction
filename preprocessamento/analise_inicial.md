# Análise Inicial e Seleção de Filtros

Este documento apresenta de forma detalhada o processo de análise inicial do dataset de Consumo de Energia Elétrica e o
raciocínio técnico utilizado para selecionar os filtros de pré-processamento aplicados no software Weka.

A etapa de pré-processamento é fundamental em projetos de Mineração de Dados e Machine Learning, pois a qualidade dos
dados influencia diretamente o desempenho dos modelos preditivos. Dados inconsistentes, atributos em escalas
incompatíveis ou distribuições inadequadas podem comprometer a capacidade do algoritmo de aprender padrões relevantes.

Antes da aplicação dos algoritmos de aprendizado, foi realizada uma análise exploratória do conjunto de dados (descrita
em [AED](../analise/analise_exploratoria.md)) com o objetivo de identificar:

- presença de valores faltantes;
- distribuição estatística dos atributos;
- existência de atributos categóricos;
- atributos potencialmente irrelevantes;
- diferenças de escala entre variáveis.

Com base nessa análise, foram definidos os filtros descritos a seguir.

## 1. Tratamento de Valores Faltantes (ReplaceMissingValues)

### Objetivo

Corrigir problemas causados pela presença de valores ausentes no dataset.

### Problema Identificado

Durante a inspeção inicial dos dados no Weka, verificou-se a existência de aproximadamente 104 valores nulos (NaN)
distribuídos em atributos numéricos do conjunto de dados.

A presença de valores faltantes pode causar diversos problemas:

* falha na execução de alguns algoritmos;
* redução da qualidade das previsões;
* geração de vieses estatísticos;
* perda de amostras caso as linhas sejam removidas manualmente.

### Solução Aplicada

Foi utilizado o filtro:

```
weka.filters.unsupervised.attribute.ReplaceMissingValues
```

Esse filtro substitui:

* valores numéricos pela média do atributo;
* valores categóricos pela moda (categoria mais frequente).

### Impacto no Dataset

A substituição dos valores faltantes permitiu:

* preservar todas as instâncias do dataset;
* evitar perda de informação;
* manter a consistência estatística dos atributos;
* garantir compatibilidade com algoritmos de Machine Learning.

Além disso, essa abordagem é considerada adequada quando a quantidade de valores ausentes é relativamente pequena em
relação ao volume total do conjunto de dados.

## 2. Transformação Logarítmica (MathExpression - log(A))

### Objetivo

Reduzir a assimetria dos dados e minimizar o impacto de valores extremos.

### Problema Identificado

As variáveis:

* Area_Metragem
* Consumo_Energia

apresentavam distribuição com cauda longa à direita, indicando forte assimetria positiva.

Esse comportamento é comum em dados de consumo, onde poucas amostras possuem valores muito elevados quando comparadas à
maioria das observações.

### Solução Aplicada

Foi utilizado o filtro:

```
weka.filters.unsupervised.attribute.MathExpression
```

com a experessão:

```
log(A)
```

A transformação logarítmica reduz proporcionalmente os valores altos sem alterar a relação entre as observações.

### Impacto no Dataset

A aplicação do logaritmo proporcionou:

* redução do efeito de outliers;
* distribuição mais próxima da normalidade;
* estabilização da variância;
* melhoria na capacidade de generalização dos modelos.

Esse tipo de transformação é particularmente importante em algoritmos de regressão e redes neurais, que podem ser
sensíveis a distribuições muito assimétricas.

## 3. Codificação de Variáveis Categóricas (NominalToBinary)

### Objetivo

Transformar atributos categóricos em formato adequado para algoritmos numéricos.

### Problema Identificado

Os atributos:

* Setor_Atividade
* Dia_da_Semana

são variáveis nominais, ou seja, representam categorias sem relação de ordem natural.

Algoritmos de Machine Learning trabalham predominantemente com valores numéricos. Converter categorias diretamente em
números inteiros poderia induzir interpretações incorretas.

Por exemplo:

| Setor_Atividade | Codificação Numérica |
|-----------------|----------------------|
| Industrial      | 0                    |
| Comecial        | 1                    |
| Residencial     | 2                    |

Nesse caso, o modelo poderia interpretar que “Residencial” possui valor maior que “Industrial”, o que não possui
significado real.

### Solução Aplicada

Foi utilizado o filtro:

```
weka.filters.unsupervised.attribute.NominalToBinary
```

Esse procedimento realiza a técnica conhecida como One-Hot Encoding, criando uma coluna binária para cada categoria.

Exemplo:

| Setor_Atividade | Industrial | Comercial | Residencial |
|-----------------|------------|-----------|-------------|
| Industrial      | 1          | 0         | 0           |
| Comercial       | 0          | 1         | 0           |
| Residencial     | 0          | 0         | 1           |

### Impacto no Dataset

A transformação:

* elimina falsa relação ordinal;
* melhora interpretação do modelo;
* permite aprendizado independente de cada categoria;
* aumenta compatibilidade com modelos supervisionados.

## 4. Remoção de Atributos Irrelevantes (Remove)

### Objetivo

Eliminar atributos que não contribuem significativamente para o problema analisado.

### Problema Identificado

Os atributos:

* Num_Extintores

não apresentou relação lógica ou física direta com o consumo de energia elétrica.

Além disso, atributos irrelevantes podem:

* introduzir ruído;
* aumentar a complexidade do modelo;
* prejudicar a interpretação;
* causar overfitting.

### Solução Aplicada

Foi utilizado o filtro:

```
weka.filters.unsupervised.attribute.Remove
```

para excluir o atributo do conjunto de dados.

### Impacto no Dataset

A remoção do atributo:

* reduziu a dimensionalidade;
* simplificou o modelo;
* melhorou a eficiência computacional;
* diminuiu o risco de aprendizado de padrões espúrios.

Essa prática segue o princípio da parcimônia, no qual modelos mais simples tendem a generalizar melhor.

## 5. Padronização (Standardize)

### Objetivo

Colocar todos os atributos numéricos em uma mesma escala.

### Problema Identificado

Os atributos do dataset apresentavam escalas muito diferentes, por exemplo:

* temperatura variando entre 15 e 35;
* área em centenas de metros quadrados;
* consumo energético em milhares.

Algoritmos baseados em distância ou gradiente podem ser fortemente influenciados por atributos de maior magnitude.

### Solução Aplicada

Foi utilizado o filtro:

```
weka.filters.unsupervised.attribute.Standardize
```

Esse filtro transforma os dados para que cada atributo apresente:

* média igual a 0;
* desvio padrão igual a 1.

A transformação segue a fórmula:
$z = \frac{x - \mu}{\sigma}$

Onde:

- $x$ representa o valor original;
- $μ$ representa a média;
- $σ$ representa o desvio padrão.

### Impacto no Dataset

A padronização:

* evita dominância de atributos com valores maiores;
* melhora convergência de algoritmos;
* aumenta estabilidade numérica;
* favorece modelos como:
    * Redes Neurais;
    * SVM;
    * KNN;
    * Regressão Linear.