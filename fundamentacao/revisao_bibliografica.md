# Revisão Bibliográfica: Processo Metodológico de Pesquisa

## 1. Contextualização da Pesquisa

Esta revisão bibliográfica foi conduzida com o objetivo de mapear sistematicamente a aplicação de técnicas de aprendizado de máquina na análise e previsão do consumo de energia elétrica, com foco especial em ferramentas computacionais, modelos de aprendizagem supervisionada e contextos de aplicação (industrial versus urbano/residencial).

---

## 2. Metodologia de Pesquisa

### 2.1 Ferramenta Utilizada

A pesquisa foi conduzida através do **NotebookLM** (Google's AI-powered research tool), que possibilita buscas profundas (*Deep Research*) em múltiplas bases de dados académicas e científicas simultaneamente.

### 2.2 Estrutura da Busca

A estratégia de pesquisa foi guiada pelo seguinte prompt inserido no NotebookLM:

> Realize uma pesquisa profunda sobre a aplicação de tarefas de aprendizado de máquina (regressão, classificação, agrupamento e regras de associação) na análise e previsão do consumo de energia elétrica. Durante a pesquisa, aborde os seguintes pontos:
> 
> 1. **Ferramentas e Algoritmos**
> 2. **Contextos de Aplicação**

O prompt está presente em [prompt_pesquisa.txt](../prompts/prompt_pesquisa.txt). Além disso, ele foi decomposto em **três pilares temáticos principais** para estruturar a busca:

#### **Pilar 1: Tarefas de Aprendizado de Máquina**
- Regressão para séries temporais
- Classificação de padrões de consumo
- Agrupamento (*clustering*) e *load profiling*
- Mineração de regras de associação

#### **Pilar 2: Ferramentas e Ambientes**
- Weka (Waikato Environment for Knowledge Analysis)
- Implementações em Python (scikit-learn, TensorFlow, Keras)
- Ambientes integrados para análise comparativa

#### **Pilar 3: Contextos de Aplicação**
- Ambientes industriais (manufatura, fundições, indústrias eletrônicas)
- Contextos urbanos (edifícios inteligentes, redes distribuídas)
- Consumo residencial (casas inteligentes, smart meters)
- Sistemas SCADA e infraestrutura de rede

### 2.3 Strings de Busca

A ferramenta NotebookLM propôs as seguintes strings de busca booleana para operacionalizar a pesquisa profunda:

```
("Machine Learning" OR "Deep Learning" OR "Artificial Intelligence") 
AND 
("Electricity Consumption" OR "Energy Load Forecasting" OR "Power Demand Prediction") 
AND 
("Regression" OR "Classification" OR "Clustering" OR "Association Rules") 
AND 
("Weka" OR "Python" OR "TensorFlow" OR "scikit-learn") 
AND 
("Industrial" OR "Urban" OR "Residential" OR "Building" OR "Manufacturing")
```

### 2.4 Fontes Consultadas

A pesquisa abrangeu múltiplas bases de dados e plataformas:

- **Bases de Dados Científicas**: PubMed Central (PMC), IEEE Xplore, arXiv
- **Repositórios Acadêmicos**: ResearchGate, MDPI, IJCSNS, IJATES
- **Plataformas Especializadas**: DergiPark, Chemical Engineering Transactions, Taylor & Francis
- **Publicações Técnicas**: Medium (artigos técnicos), Scribd (estudos metodológicos)
- **Instituições de Pesquisa**: Polimi (Politecnico di Milano), Drexel University
- **Plataformas de Código Aberto**: Documentação oficial do Weka

---

## 3. Estrutura Analítica do Relatório

A feramenta gerou um relatório de síntese das fontes consultadas, organizado conforme a seguinte hierarquia:

### **Fase 1: Identificação de Tarefas Primárias**
Mapeamento da prevalência relativa de cada tarefa de ML (regressão > classificação ≈ clustering > associação)

### **Fase 2: Análise do Ecossistema Weka**
Consolidação de estudos sobre a aplicação prática do Weka em cenários industriais reais (GECOL, Líbia; produções de aço, fundições)

### **Fase 3: Aprofundamento em Regressão**
- Algoritmos específicos (M5P, XGBoost, LSTM, SARIMAX)
- Métricas de desempenho (RMSE, R², MAE, Skill Score)
- Comparação entre abordagens estadísticas e neurais

### **Fase 4: Exploração de Classificação**
- Árvores de decisão e interpretabilidade (J48)
- Classificadores probabilísticos (SVM, Random Forest)
- Aplicações em detecção de anomalias e resposta à demanda

### **Fase 5: Investigação de Agrupamento**
- Algoritmos não supervisionados (K-Means, EM, Fuzzy C-Means)
- *Load profiling* e segmentação de consumidores
- Abordagens híbridas (clustering + regressão)

### **Fase 6: Análise de Associação**
- Mineração de regras (Apriori, FP-Growth)
- Descoberta de padrões causal-temporais
- Impacto de variáveis climáticas

### **Fase 7: Contraste Dinâmico**
- Dados industriais: estruturados, cíclicos, previsíveis
- Dados urbanos: estocásticos, irregulares, ruidosos
- Seleção de modelos apropriados por contexto

### **Fase 8: Considerações Sustentáveis**
- Pegada de carbono de modelos ML
- Eficiência computacional versus acurácia
- *Green Software Engineering*



---

## 4. Desafios Metodológicos Encontrados

### 4.1 Fragmentação do Conhecimento
A pesquisa inicial revelou uma distribuição dispersa do conhecimento:
- Alguns algoritmos dominam subdomínios específicos
- Falta de estudos comparativos holísticos (parcialmente suprida pelo relatório)
- Termos variáveis em diferentes comunidades (ex: *load profiling* vs. *consumption profiling*)

### 4.2 Heterogeneidade de Métodos de Validação
Diferentes estudos utilizavam:
- RMSE, MAE, MAPE (regressão)
- Acurácia, Precisão, Recall, F1 (classificação)
- Silhueta, Davies-Bouldin Index (clustering)
- Suporte, Confiança, Lift (associação)

---

## 5. Resultados 

### 5.1 Estatísticas Globais da Pesquisa

A pesquisa inicial identificou um total de **38 fontes** relevantes, publicadas entre 2012 e 2024, com uma concentração crescente a partir de 2018, refletindo o aumento do interesse em ML para energia elétrica.

### 5.2 Distribuição por Tipo de Publicação

A maioria das fontes (65%) são artigos publicados em journals científicos de alto impacto, seguidos por repositórios preprint (18%), conferências e anais (12%) e uma pequena fração de documentação técnica e blogs (5%).

| Tipo de Publicação           | Percentual (%) | Número de Fontes |
|-----------------------------|----------------|------------------|
| Journals Científicos (MDPI, PMC, IEEE, etc.)        | 65%            | 25               |
| Repositórios Preprint (arXiv, ResearchGate)        | 18%            | 7                |
| Conferências e Anais         | 12%            | 4                |
| Documentação Técnica/Blogs   | 5%             | 2                |

### 5.3 Distribuição por Foco Temático

A análise temática revelou a seguinte distribuição de foco entre as fontes:

| Foco Temático                                      | Percentual (%) | Número de Fontes |
|----------------------------------------------------|----------------|------------------|
| Regressão e Previsão                               | 42%            | 16               |
| Classificação                                      | 16%            | 6                |
| Agrupamento/Clustering                            | 18%            | 7                |
| Associação/Mineração de Regras                    | 11%            | 4                |
| Sustentabilidade e Eficiência Computacional        | 8%             | 3                |
| Meta-análises e Revisões Sistemáticas              | 5%             | 2                |

### 5.4 Distribuição por Contexto

A maioria das fontes (45%) se concentra em contextos urbanos e edifícios inteligentes, seguidos por contextos industriais/manufatura (35%), residenciais/domésticos (13%) e redes inteligentes (7%). Evidencia-se uma tendência de aplicação crescente em ambientes urbanos, refletindo a demanda por soluções de eficiência energética em edifícios e infraestrutura urbana.

| Contexto de Aplicação                               | Percentual (%) | Número de Fontes |
|----------------------------------------------------|----------------|------------------|
| Edifícios Urbanos                     | 45%            | 17               |
| Industrial/Manufatura                             | 35%            | 13               |
| Residencial/Doméstico                             | 13%            | 5                |
| Rede Inteligente (Smart Grid)                     | 7%             | 3                |

---

## 6. Conclusões

### 6.1 Estudos Identificadas

Da pesquisa inicial conduzida no NotebookLM, foram identificadas **38 fontes** relacionadas ao tema de machine learning aplicado ao consumo de energia elétrica. Após leitura crítica e aplicação de critérios de inclusão específicos para o trabalho, **8 fontes foram selecionadas** para composição do relatório final.

#### **Critério de Seleção das 8 Fontes:**

As fontes aceitas foram aquelas que:
- Abordavam especificamente a **tarefa de regressão** para previsão de consumo de energia
- Utilizavam ou comparavam explicitamente a **ferramenta Weka** ou Python com bibliotecas equivalentes
- Apresentavam resultados empíricos validados com dados reais
- Continham comparações de múltiplos algoritmos

### 6.2 Estudos Selecionadas

As 8 fontes aceitas para o relatório final são:

1. **BILAL, M.; KIM, H.; FAYAZ, M.; PAWAR, P.** Comparative Analysis of Time Series Forecasting Approaches for Household Electricity Consumption Prediction. 2022. DOI: [10.48550/arXiv.2207.01019](https://doi.org/10.48550/arXiv.2207.01019).

2. **CORDON, D.; PITA, A.; JUAN, A. A.** Classifying and Predicting Household Energy Consumption Using Data Analytics and Machine Learning. Algorithms, v. 19, n. 2, art. 114, 2026. DOI: [10.3390/a19020114](https://doi.org/10.3390/a19020114).

3. **DING, Z. et al.** A Comprehensive Study on Integrating Clustering with Regression for Short-Term Forecasting of Building Energy Consumption: Case Study of a Green Building. Buildings, v. 12, n. 10, art. 1701, 2022. DOI: [10.3390/buildings12101701](https://doi.org/10.3390/buildings12101701).

4. **LEE, M. H. L. et al.** A Comparative Study of Forecasting Electricity Consumption Using Machine Learning Models. Mathematics, v. 10, n. 8, art. 1329, 2022. DOI: [10.3390/math10081329](https://doi.org/10.3390/math10081329).

5. **LEÓN-MUNIZAGA, N.; AGUIRRE-MUNIZAGA, M.; LAGOS-ORTIZ, K.; DEL CIOPPO-MORSTADT, J.** Prediction of Energy Consumption in an Electric Arc Furnace Using Weka. In: VALENCIA-GARCÍA, R. et al. (org.). CITI 2020, CCIS 1309. Cham: Springer Nature Switzerland, 2020. p. 58-70. DOI: [10.1007/978-3-030-62015-8_5](https://doi.org/10.1007/978-3-030-62015-8_5).

6. **SARSWATULA, S. A.; PUGH, T.; PRABHU, V.** Modeling Energy Consumption Using Machine Learning. Frontiers in Manufacturing Technology, v. 2, art. 855208, 2022. DOI: [10.3389/fmtec.2022.855208](https://doi.org/10.3389/fmtec.2022.855208).

7. **WU, J. et al.** A Comparative Analysis of Machine Learning-Based Energy Baseline Models across Multiple Building Types. Energies, v. 17, n. 6, art. 1285, 2024. DOI: [10.3390/en17061285](https://doi.org/10.3390/en17061285).

8. **ZHAO, X.; HUANG, X.; DING, J.; ZHANG, Y.** A Comparative Study of Machine Learning Algorithms for Electricity Price Forecasting with LIME-Based Interpretability. arXiv preprint, arXiv:2512.01212, 2025. DOI: [10.1109/ICEIEC65904.2025.11273147](https://doi.org/10.1109/ICEIEC65904.2025.11273147).

### 6.3 Síntese Consolidada

A pesquisa bibliográfica inicial identificou fontes de alta qualidade que cobrem o panorama completo de machine learning aplicado ao consumo de energia elétrica. A seleção das fontes fornece uma base sólida e focada para o desenvolvimento do trabalho, garantindo ênfase em:

- Metodologias de regressão robustas e validadas
- Aplicação prática da ferramenta Weka
- Contextos industriais, urbanos e residenciais
- Comparações empíricas de múltiplos algoritmos
- Validação com dados reais e públicos

Estas estudos formam a base teórica e empírica sólida para o desenvolvimento do trabalho prático, representando cerca de **21% das fontes identificadas** e priorizando a qualidade e especificidade temática sobre a quantidade bruta.
