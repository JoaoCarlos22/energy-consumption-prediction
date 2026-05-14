# Análise Sistêmica e Investigativa das Tarefas de Aprendizado de Máquina Aplicadas ao Consumo de Energia Elétrica: Paradigmas, Ferramentas e Desempenho em Contextos Industriais e Urbanos

> Relatório gerado pela ferramenta de IA NotebookLM, com base em uma revisão bibliográfica de 38 fontes.

A infraestrutura energética global atravessa uma fase de metamorfose digital, impulsionada pela necessidade urgente de eficiência operacional e sustentabilidade ambiental. No epicentro dessa transformação, o aprendizado de máquina (*Machine Learning* - ML) emerge não apenas como uma ferramenta acessória, mas como o motor fundamental para a interpretação de volumes massivos de dados gerados por redes inteligentes e sistemas de automação industrial. A análise da literatura técnica e científica revela uma distribuição estratégica das tarefas de aprendizado — classificação, regressão, agrupamento e associação — cada uma desempenhando um papel vital na mitigação de perdas, otimização de recursos e estabilização da rede elétrica.[1, 2, 3]

---

# Prevalência e Dinâmica das Tarefas de Aprendizado de Máquina

O cenário da pesquisa em consumo de energia elétrica é dominado por uma hierarquia funcional de tarefas de aprendizado de máquina. A regressão mantém-se como a tarefa mais frequente e crítica, dada a natureza contínua e temporal da demanda energética. No entanto, a complexidade crescente dos dados de consumo, marcados por comportamentos humanos estocásticos e ciclos industriais rígidos, exige a integração de outras tarefas.[4, 5, 6]

A regressão é a espinha dorsal do planejamento de curto e longo prazo, permitindo que concessionárias e gestores industriais antecipem valores exatos de carga.[7, 8] O agrupamento (*clustering*), por sua vez, atua como um facilitador de inteligência, permitindo a segmentação de bases de dados heterogêneas em perfis de carga compreensíveis, muitas vezes servindo como um passo preparatório para melhorar a precisão dos modelos de regressão.[9, 10, 11]

A classificação desempenha um papel tático, sendo utilizada para rotular níveis de demanda (baixa, média, alta), identificar tipos de consumidores ou detectar anomalias operacionais que podem indicar falhas ou furtos.[1, 12, 13] Por fim, a mineração de regras de associação, embora menos prevalente em termos de frequência bruta de publicações, oferece uma profundidade semântica única ao descobrir dependências lógicas entre fatores ambientais, horários e o comportamento do consumo.[14, 15, 16]

| Tarefa de Aprendizado | Frequência Relativa           | Objetivo Primário                                    | Contexto de Aplicação                             |
| --------------------- | ----------------------------- | ---------------------------------------------------- | ------------------------------------------------- |
| Regressão             | Muito Alta                    | Previsão de valores numéricos contínuos (kWh, MW)    | Planejamento de despacho e estabilidade da rede   |
| Classificação         | Alta                          | Categorização de perfis, níveis de carga e anomalias | Resposta à demanda e segurança operacional        |
| Agrupamento           | Alta (frequentemente híbrida) | Segmentação de padrões de consumo sem rótulos        | Identificação de perfis típicos e “Cold Start”    |
| Associação            | Moderada                      | Descoberta de relações lógicas e “Se-Então”          | Análise de causalidade e comportamento do usuário |

---

# O Ecossistema Weka na Modelagem Energética

A ferramenta Weka (*Waikato Environment for Knowledge Analysis*) consolidou-se como um recurso indispensável para pesquisadores devido à sua interface amigável e vasta coleção de algoritmos de mineração de dados implementados em Java.[1, 17, 18]

No contexto do consumo elétrico, o Weka é utilizado para realizar desde a limpeza e pré-processamento de dados brutos de medidores inteligentes até a execução de simulações complexas de desempenho preditivo.[1, 19, 20, 21]

A versatilidade do Weka permite que engenheiros comparem rapidamente modelos de diferentes famílias (árvores, funções, regras) em um único ambiente. Estudos focados na redução de perdas em empresas elétricas, como a GECOL na Líbia, demonstram como o Weka pode processar dados de sistemas SCADA (*Supervisory Control and Data Acquisition*) para prever níveis de consumo baseando-se em variáveis como temperatura, umidade e velocidade do vento.[1]

A capacidade de visualização da ferramenta também auxilia na interpretação de *clusters* de países ou regiões, facilitando a identificação de disparidades globais no uso de energia.[14, 22]

---

# Regressão: Previsão de Valores Contínuos e Séries Temporais

A tarefa de regressão é, indiscutivelmente, a mais pesquisada no domínio da energia elétrica. O objetivo central é mapear um conjunto de variáveis independentes (preditores) para uma saída numérica contínua que representa o consumo real. A complexidade dessa tarefa reside na natureza não linear e altamente volátil das séries temporais de carga.[5, 6, 7]

## Modelos de Regressão e o Papel do Algoritmo M5P no Weka

Dentro do ambiente Weka, o modelo M5P tem se destacado como uma das abordagens mais precisas para a regressão em cenários industriais complexos, como o consumo de Fornos de Arco Elétrico (*Electric Arc Furnace* - EAF).[20, 21]

O M5P é um algoritmo que reconstrói árvores de modelo, onde os nós folha não são apenas valores constantes, mas funções de regressão linear. Isso permite que o modelo capture variações locais com alta precisão, superando frequentemente redes neurais tradicionais e regressões lineares simples.[20, 21]

Em experimentos realizados com dados de produção de aço, o M5P demonstrou um coeficiente de correlação superior (0.8701) comparado a outros modelos como o J48 (usado para classificação via regressão) e o *Multilayer Perceptron* (MLP).[21]

Essa superioridade decorre da capacidade do M5P de lidar com a hipótese de que o consumo final depende diretamente da composição da sucata e de variáveis de processo como o tempo de *power on* e a injeção de carbono.[20, 21]

## Desempenho de Modelos de Conjunto e Deep Learning

Além das ferramentas tradicionais do Weka, a literatura aponta para a eficácia crescente de modelos de conjunto (*ensembles*) e aprendizado profundo (*Deep Learning*) para a regressão.

O *Extreme Gradient Boosting* (XGBoost) é frequentemente citado como o modelo de linha de base mais preciso para diversos tipos de edifícios em cidades inteligentes.[6, 23, 24]

No entanto, para dados sequenciais, as redes LSTM (*Long Short-Term Memory*) e GRU (*Gated Recurrent Units*) são as favoritas por sua capacidade de reter informações de longo prazo e lidar com o problema do desaparecimento do gradiente em sequências temporais longas.[7, 25, 26]

A precisão destes modelos é frequentemente medida através de métricas como o Erro Quadrático Médio (RMSE), definido matematicamente como:

RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}

Onde:

* ( y_i ) é o valor real;
* ( \hat{y}_i ) é o valor predito.

Estudos mostram que em cenários industriais, onde os padrões de carga são mais regulares, o RMSE tende a ser significativamente menor do que em cenários residenciais/urbanos.[27]

| Modelo        | Tarefa    | Ferramenta/Contexto         | Destaque de Desempenho                                         |
| ------------- | --------- | --------------------------- | -------------------------------------------------------------- |
| M5P           | Regressão | Weka / Industrial (EAF)     | Melhor correlação (0.87) em dados de fundição                  |
| LSTM-EMD      | Regressão | Python / Industrial         | *Skill Score* de 81.5% para previsão de pico                   |
| XGBoost       | Regressão | Python / Urbano (Edifícios) | Modelo *baseline* mais preciso para múltiplos tipos de prédios |
| SARIMAX       | Regressão | Python / Residencial        | Supera modelos de *Deep Learning* em dados ruidosos            |
| Random Forest | Regressão | Weka / Geral                | Melhor precisão em *benchmarking* de manufatura (R² 0.869)     |

---

# Classificação: Categorização e Detecção de Padrões Operacionais

A classificação transforma a medição numérica em conhecimento categórico. Esta tarefa é essencial para a tomada de decisão rápida, onde o gestor não precisa necessariamente do valor exato de consumo, mas sim de saber se a rede está sob estresse ou se um perfil de consumidor mudou.[1, 12]

## O Uso de Árvores de Decisão (J48) no Weka

O algoritmo J48, a implementação do Weka para o clássico C4.5, é amplamente utilizado por sua transparência e interpretabilidade.

Em sistemas SCADA de empresas elétricas, o J48 é empregado para classificar períodos de consumo em níveis (Baixo, Médio, Alto) com base em variáveis ambientais e horários de pico.[1]

A construção de árvores de decisão permite que os operadores entendam os gatilhos exatos que levam a picos de demanda, facilitando o planejamento de ações preventivas para manter a estabilidade da grade.[1, 13]

Além do J48, outros classificadores disponíveis no Weka, como o *Random Forest* e o *Multi-layer Perceptron* (MLP), são testados para validar a qualidade dos processos de produção. Em indústrias alimentícias e eletrônicas, esses modelos classificam se os parâmetros de consumo estão dentro dos limites de eficiência esperados, ajudando a identificar oportunidades de melhoria energética.[28]

## Classificação de Sistemas de Aquecimento Urbanos

Um caso de uso inovador de classificação em áreas urbanas é a distinção entre tipos de aquecimento residencial.

Ao analisar dados de medidores inteligentes com *Support Vector Machines* (SVM), pesquisadores conseguiram classificar residências com aquecimento distrital versus aquecimento elétrico com uma acurácia de 97.1%.[12]

Esse tipo de inteligência é vital para concessionárias urbanas, pois permite estimar a flexibilidade da demanda e planejar a infraestrutura sem a necessidade de auditorias físicas dispendiosas.[12, 26]

---

# Agrupamento: Descoberta de Perfis sem Rótulos

O agrupamento (*clustering*) é a tarefa de aprendizado não supervisionado que busca encontrar estruturas naturais nos dados. No tema do consumo elétrico, o agrupamento é a técnica fundamental para o *load profiling* (perfilamento de carga).[4, 9, 14, 29]

## K-Means e Expectation Maximization no Weka

O *SimpleKMeans* é o algoritmo de agrupamento mais frequente no Weka para analisar padrões de consumo.

Ele funciona agrupando curvas de carga diárias em *clusters* com base na similaridade euclidiana de suas formas. Isso permite identificar, por exemplo, que um grupo de consumidores tem picos matinais (perfil residencial clássico), enquanto outro consome mais durante a noite (perfil de carregamento de veículos elétricos ou indústrias noturnas).[4, 14]

Outro algoritmo relevante no Weka é o *Expectation Maximization* (EM), que utiliza uma abordagem probabilística para definir a pertinência de um dado a um *cluster*.

O EM é particularmente útil quando as fronteiras entre os perfis de consumo não são claras, permitindo uma segmentação mais matizada de consumidores urbanos baseada em fatores como distância de rodovias ou áreas agrícolas.[14, 29]

## Abordagens Híbridas: Agrupamento como Pilar da Previsão

Uma das tendências mais fortes na pesquisa atual é a integração do agrupamento com a regressão. O fluxo de trabalho consiste em:

1. Agrupar os dados históricos de consumo para separar padrões heterogêneos;
2. Treinar um modelo de regressão específico para cada *cluster*;
3. Utilizar um classificador para determinar a qual *cluster* os novos dados pertencem e aplicar o regressor correspondente.[9, 10, 11, 30]

Esta estratégia, conhecida como *ensemble baseado em clustering*, demonstrou reduzir erros de previsão (MAE e RMSE) de forma significativa ao tratar separadamente comportamentos de consumo que, se misturados, causariam ruído e imprecisão no modelo global.[9, 11, 30]

| Algoritmo de Agrupamento        | Ferramenta | Aplicação               | Impacto                                                |
| ------------------------------- | ---------- | ----------------------- | ------------------------------------------------------ |
| SimpleKMeans                    | Weka       | *Load Profiling* Urbano | Identificação de 6–8 perfis típicos de residências     |
| EM (*Expectation Maximization*) | Weka       | Segmentação Geográfica  | Correlação entre consumo e proximidade de rodovias     |
| Fuzzy C-Means                   | Geral      | Pré-regressão           | Redução de erros em SVR ao criar grupos homogêneos     |
| DBSCAN                          | Geral      | Manufatura Aditiva      | Detecção de ruído e anomalias em processos industriais |

---

# Mineração de Regras de Associação: A Semântica do Consumo

Diferente das outras tarefas que buscam prever ou agrupar, a mineração de regras de associação busca explicar. Ela descobre correlações e dependências lógicas que regem o comportamento energético.[13, 14, 16, 18]

## Algoritmos Apriori e FP-Growth no Weka

O Weka oferece o algoritmo Apriori, que é o padrão ouro para encontrar associações.

Ele identifica conjuntos de itens frequentes baseados em métricas de suporte (frequência relativa) e confiança (probabilidade condicional).[16, 18, 31]

No contexto elétrico, o Apriori gera regras como:

> Se (Temperatura > 30°C) E (Umidade > 80%) ENTÃO (Consumo = Muito Alto).

Embora o Apriori seja fácil de interpretar, ele pode ser computacionalmente caro para grandes volumes de dados.

Para mitigar isso, o Weka também implementa o FP-Growth, que utiliza uma estrutura de árvore compacta para evitar a geração exaustiva de candidatos, sendo mais escalável para dados massivos de medidores inteligentes.[17, 18, 32]

## Aplicações de Regras de Associação entre Clima e Energia

A extração de regras de associação paralela (usando MapReduce/Hadoop) permitiu descobrir que condições climáticas específicas, como neve combinada com alta umidade, desencadeiam modos de consumo elétrico previsíveis.[15]

Essas regras ajudam os tomadores de decisão a antecipar o impacto de frentes frias ou ondas de calor na estabilidade da rede, permitindo uma automação mais inteligente do despacho de carga e a prevenção de quedas de energia (*load shedding*).[13, 29]

---

# Contraste Dinâmico: Indústria versus Cidades

A eficácia das tarefas de ML é profundamente influenciada pelo ambiente onde os dados são gerados. A previsibilidade não é uniforme entre os domínios industrial e urbano.[27, 33]

## Consumo Industrial: O Domínio da Regularidade

Nas indústrias, a energia é um insumo de produção. Portanto, o consumo é intrinsecamente ligado a cronogramas de máquinas e metas de manufatura.

Isso resulta em dados mais sazonais, orientados a eventos e, crucialmente, mais “bem comportados” do ponto de vista estatístico.[27, 34]

Modelos complexos como o LSTM-EMD florescem neste ambiente, alcançando melhorias de precisão drásticas (até 81.5% de ganho de performance) porque conseguem mapear com precisão as variações periódicas dos turnos de trabalho.[27]

## Consumo Urbano e Residencial: O Desafio da Estocasticidade

Em cidades, o consumo é o reflexo da liberdade humana.

O uso de eletrodomésticos, iluminação e climatização residencial é altamente irregular e estocástico. Pequenas variações no comportamento individual podem gerar ruído massivo nos dados agregados.[27, 33]

Surpreendentemente, em cenários residenciais, modelos estatísticos robustos como o SARIMAX muitas vezes superam redes neurais profundas, que tendem a sofrer de *overfitting* (sobreajuste) ao tentar encontrar padrões em dados que possuem uma alta dose de aleatoriedade.[27]

| Fator de Comparação     | Contexto Industrial                      | Contexto Urbano/Residencial                 |
| ----------------------- | ---------------------------------------- | ------------------------------------------- |
| Natureza dos Dados      | Estruturada, cíclica, previsível         | Estocástica, irregular, ruidosa             |
| Principais Drivers      | Metas de produção, máquinas, turnos      | Comportamento humano, clima, estilo de vida |
| Melhor Modelo Preditivo | LSTM, Deep Learning, M5P                 | SARIMAX, Random Forest, XGBoost             |
| Desafio Principal       | Complexidade dos equipamentos de suporte | Variabilidade extrema e padrões fracos      |
| Erro de Previsão (RMSE) | Geralmente mais baixo e estável          | Geralmente mais alto e volátil              |

---

# Sustentabilidade e Eficiência do Próprio Aprendizado de Máquina

Um paradoxo emerge na pesquisa contemporânea: o uso de aprendizado de máquina para economizar energia consome, por si só, quantidades massivas de eletricidade.

O treinamento de modelos de larga escala, especialmente os baseados em *Deep Learning*, exige infraestruturas de computação intensivas que contribuem para as emissões globais de CO₂.[2, 35]

A literatura destaca que modelos menores e específicos para tarefas (*task-specific*), como os implementados no Weka (J48, M5P), são ordens de magnitude mais eficientes do que grandes modelos de linguagem (LLMs) ou redes neurais genéricas adaptadas para predição de séries temporais.[35, 36]

Além disso, a escolha da localização dos *data centers* influencia diretamente a pegada de carbono; treinar um modelo em regiões com matriz energética hidrelétrica resulta em emissões até 30 vezes menores do que em regiões dependentes de carvão.[35]

Esta consciência ambiental está moldando uma nova era de *Green Software Engineering*, onde a acurácia do modelo é balanceada contra o seu custo energético de treinamento e inferência.[2, 37, 38]

---

# Síntese de Achados e Conclusões

A investigação detalhada das tarefas de aprendizado de máquina aplicadas ao consumo de energia elétrica permite consolidar uma visão clara sobre o estado da arte e as ferramentas preferenciais.

A Regressão permanece como a tarefa soberana, essencial para a continuidade operacional da rede elétrica. No Weka, o modelo M5P é a referência para precisão industrial, enquanto fora dele, o XGBoost e as redes LSTM dominam a fronteira tecnológica.

O Agrupamento evoluiu de uma tarefa descritiva para uma ferramenta funcional de pré-processamento, permitindo que a heterogeneidade urbana seja “domada” antes da predição.

A Classificação oferece o diagnóstico necessário para a resposta à demanda, com o J48 e o SVM liderando em interpretabilidade e precisão técnica.

Finalmente, a Associação preenche a lacuna de conhecimento causal, transformando dados brutos em regras lógicas acionáveis para políticas de eficiência.[1, 4, 5, 9, 11, 21]

Em suma, a escolha da tarefa e do modelo não deve ser isolada, mas sim guiada pela natureza do dado — industrial ou urbano — e pelo equilíbrio necessário entre precisão, custo computacional e explicabilidade.

O futuro da gestão energética reside na orquestração dessas quatro tarefas, integrando a robustez dos modelos estatísticos com a flexibilidade da inteligência artificial moderna para criar redes elétricas resilientes, eficientes e genuinamente sustentáveis.

---

# Referências

1. [Improvement of Classification Algorithms for Energy Saving in Lost Energy Data of Libya Electricity Company Using Weka Model - DergiPark](https://dergipark.org.tr/tr/download/article-file/3441240)

2. [Evaluating the Energy Consumption of Machine Learning: Systematic Literature Review and Experiments - arXiv](https://arxiv.org/html/2408.15128v1)

3. [Artificial Intelligence and Machine Learning for Energy Consumption and Production in Emerging Markets: A Review - MDPI](https://www.mdpi.com/1996-1073/16/2/745)

4. [CLUSTERING-BASED PREDICTION OF RESIDENTIAL ELECTRICITY CONSUMPTION - Proceedings.com](https://www.proceedings.com/content/077/077185-0062open.pdf)

5. [Evaluating Machine Learning Algorithms for Energy Consumption Prediction in Electric Vehicles: A Comparative Study - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12062255/)

6. [A Comparative Study of Machine Learning Algorithms for Electricity Price Forecasting with LIME-Based Interpretability - arXiv](https://arxiv.org/html/2512.01212v1)

7. [Comparing 7 Machine Learning Models for Electricity Consumption Forecasting: A Practical Implementation Guide - Medium](https://medium.com/@itsmesut/comparing-7-machine-learning-models-for-electricity-consumption-forecasting-a-practical-2cc4633e0de0)

8. [Review of Methods and Models for Forecasting Electricity Consumption - MDPI](https://www.mdpi.com/1996-1073/18/15/4032)

9. [A Machine Learning Ensemble Framework Based on a Clustering Algorithm for Improving Electric Power Consumption Performance - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12623421/)

10. [Classifying and Predicting Household Energy Consumption Using Data Analytics and Machine Learning - MDPI](https://www.mdpi.com/1999-4893/19/2/114)

11. [A Comprehensive Study on Integrating Clustering with Regression for Short-Term Forecasting of Building Energy Consumption: Case Study of a Green Building - MDPI](https://www.mdpi.com/2075-5309/12/10/1701)

12. [ELECTRICITY CONSUMER CLASSIFICATION USING SUPERVISED MACHINE LEARNING - Energiforsk](https://energiforsk.se/media/29288/electricity-consumer-classification-using-supervised-machine-learning-energiforskrapport-2021-729.pdf)

13. [Analysis and Prediction of Energy Consumption Using Supervised Machine Learning Techniques: A Study of Libyan Electricity Company](http://paper.ijcsns.org/07_book/202303/20230302.pdf)

14. [Predicting Global Energy Consumption Through Data Mining - IIETA](https://www.iieta.org/journals/ijdne/paper/10.18280/ijdne.190205)

15. [Research on Association Analysis Between Electricity Consumption - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12059065/)

16. [Apriori Algorithm – Knowledge and References - Taylor & Francis](https://taylorandfrancis.com/knowledge/Engineering_and_technology/Computer_science/Apriori_algorithm/)

17. [Association Rule Mining with Apriori and FP-Growth Using Weka - ResearchGate](https://www.researchgate.net/publication/306037466_Association_rule_mining_with_apriori_and_fpgrowth_using_weka)

18. [ASSOCIATION RULE MINING WITH APRIORI AND FPGROWTH USING WEKA - IJATES](http://www.ijates.com/ADMIN/admin/postimages/images/fullpdf/1444756057_987D.pdf)

19. [Comparative Analysis of Time Series Forecasting Approaches for Household Electricity Consumption Prediction - arXiv](https://arxiv.org/pdf/2207.01019)

20. [Prediction of Energy Consumption in an Electric Arc Furnace Using Weka - ResearchGate](https://www.researchgate.net/publication/346246479_Prediction_of_Energy_Consumption_in_an_Electric_Arc_Furnace_Using_Weka)

21. [Prediction of Energy Consumption in an Electric Arc Furnace Using Weka - PDF Publication](https://www.researchgate.net/profile/Maritza-Aguirre-Munizaga/publication/352491794_Prediction_of_Energy_Consumptionpdf/data/60cba259a6fdcc01d47dc5b2/Prediction-of-Energy-Consumption.pdf?origin=publication_list)

22. [Mapping the Association Between Energy Use and ESG Dimensions: Evidence from Panel Econometrics, Clustering, and Machine Learning - MDPI](https://www.mdpi.com/1996-1073/19/3/828)

23. [A Comparative Analysis of Machine Learning-Based Energy Baseline Models Across Multiple Building Types - MDPI](https://www.mdpi.com/1996-1073/17/6/1285)

24. [Optimizing Urban Energy Efficiency Through a Machine Learning-Driven Framework: A Case Study in Reykjavik - Drexel University](https://researchdiscovery.drexel.edu/esploro/outputs/bookChapter/Optimizing-Urban-Energy-Efficiency-Through-a/991022055076604721)

25. [Beginner's Guide to Comparing Six Machine Learning Models for Energy Consumption Forecasting: A Five-Month Study (Part 1) - Python in Plain English](https://python.plainenglish.io/comparing-six-machine-learning-models-for-energy-consumption-forecasting-a-five-month-study-part-1fe6ebbe9388)

26. [A Comparative Study of Energy Consumption Forecasting Machine Learning Models - Chemical Engineering Transactions](https://www.cetjournal.it/cet/23/103/116.pdf)

27. [Comparing Peak Electricity Load Forecasting Models for an Industrial Environment](https://re.public.polimi.it/retrieve/d6eb0346-79cc-4bae-994b-67519b16461b/1-s2.0-S0378475425002629-main.pdf)

28. [Modeling Energy Consumption Using Machine Learning - Frontiers](https://www.frontiersin.org/journals/manufacturing-technology/articles/10.3389/fmtec.2022.855208/full)

29. [Survey on Electricity Consumption Using Data Mining Techniques](https://oaji.net/pdf.html?n=2019/2698-1549883942.pdf)

30. [A Comprehensive Study on Integrating Clustering with Regression for Short-Term Forecasting of Building Energy Consumption: Case Study of a Green Building - ResearchGate](https://www.researchgate.net/publication/364445659_A_Comprehensive_Study_on_Integrating_Clustering_with_Regression_for_Short-Term_Forecasting_of_Building_Energy_Consumption_Case_Study_of_a_Green_Building)

31. [WEKA 3.8.6 Association Rule Analysis - Scribd](https://www.scribd.com/document/794826116/exp-5-dwdm-merged)

32. [ASSOCIATION RULE MINING WITH APRIORI AND FPGROWTH USING WEKA - ConferenceWorld](http://data.conferenceworld.in/ICSTM2/P2837-2845.pdf)

33. [Estimating Electricity Consumption at City-Level Through Advanced Machine Learning Methods](https://www.tandfonline.com/doi/full/10.1080/09540091.2024.2313852)

34. [Predicting Industrial Building Energy Consumption with Statistical and Machine-Learning Models Informed by Physical System Parameters - OSTI](https://www.osti.gov/servlets/purl/2418832)

35. [Energy-Aware Machine Learning Models—A Review of Recent Techniques and Perspectives - MDPI](https://www.mdpi.com/1996-1073/18/11/2810)

36. [Comparing Energy Consumption and Accuracy in Text Classification Inference - arXiv](https://arxiv.org/pdf/2508.14170)

37. [Who Wins the Race? (R Vs Python) - An Exploratory Study on Energy Consumption of Machine Learning Algorithms - arXiv](https://arxiv.org/html/2508.17344v1)

38. [WEKA Sustainable AI Initiative](https://www.weka.io/company/sustainable-ai/)