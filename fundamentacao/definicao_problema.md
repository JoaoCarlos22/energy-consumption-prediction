# Previsão de Demanda Energética Multissetorial usando Machine Learning

## 1. Definição do Problema

O contínuo crescimento populacional e o desenvolvimento econômico aumentaram drasticamente a demanda global por energia, tornando a sua gestão eficiente um pré-requisito essencial para o planejamento sustentável e a redução das emissões de gases de efeito estufa. No setor de edificações, que é responsável por uma parcela imensa do consumo e da pegada de carbono no mundo inteiro, controlar o uso de eletricidade tornou-se uma prioridade para os gestores a fim de minimizar desperdícios ambientais e financeiros . A ineficiência na predição e na produção pode causar tanto uma escassez (que afeta diretamente o desenvolvimento de uma região) quanto um superávit (que representa um enorme desperdício de recursos que não podem ser estocados facilmente).

A grande complexidade na resolução deste problema reside na natureza heterogênea e multissetorial da demanda, que exige modelagens capazes de lidar com perfis de consumo drasticamente diferentes, como os dos setores residencial, comercial, industrial e acadêmico. A literatura científica demonstra que as instalações comerciais e acadêmicas tendem a apresentar padrões de uso de energia mais regulares e previsíveis, fortemente governados por horários fixos de funcionamento e dias úteis definidos. Em contrapartida, prever o consumo no setor residencial é notoriamente mais difícil, devido à alta estocasticidade gerada pelas variações no estilo de vida e pelos hábitos individuais imprevisíveis dos moradores. Além disso, o próprio setor industrial impõe o desafio de apresentar curvas diárias de comportamento bastante instáveis e flutuantes, o que prejudica a estabilidade dos algoritmos de predição, principalmente em curtos intervalos de tempo.

O presente trabalho aborda o desenvolvimento, validação e avaliação sistemática em tarefas de  aprendizagem supervisionada para previsão de demanda energética em contexto multissetorial pela ferramenta Weka. A escolha de tarefas de aprendizado máquina é fundamentada em múltiplos estudos que demonstram sua efetividade em cenários de previsão de consumo, onde o objetivo é estimar valores contínuos de consumo energético futuro. Diferenciando-se de estudos anteriores que tipicamente se concentram em um único tipo de edifício, este trabalho propõe uma abordagem integrada que reconhece e modela explicitamente os padrões de consumo distintos entre os setores residencial, comercial e industrial.

## 2. Fundamentação Teórica

A previsão de demanda energética tem migrado de modelos físicos tradicionais para abordagens guiadas por dados baseadas em Machine Learning, devido à capacidade desses algoritmos de mapear interações complexas e extrair padrões sem a necessidade de modelagem termodinâmica exaustiva. A literatura científica contemporânea investiga a aplicação dessas técnicas sob diversas óticas, abrangendo desde algoritmos preditivos específicos até a análise de comportamento multissetorial e a importância de fatores temporais.

### 2.1 Algoritmos Preditivos e Relações Não-Lineares

No que tange aos algoritmos preditivos, há um consenso empírico sobre a superioridade de modelos não-lineares em detrimento das abordagens lineares clássicas na previsão energética. Zhao et al. (2023) constataram que modelos baseados em instâncias (como KNN) e ensembles baseados em árvores (como Random Forest e XGBoost) capturam com excelência as dependências locais e os padrões não-lineares inerentes aos dados de eletricidade.

Em estudos focados no consumo industrial, Sarswatula et al. (2022) também concluíram que o algoritmo Random Forest Regressor supera técnicas tradicionais, oferecendo previsões robustas acompanhadas de alto coeficiente de determinação. Adicionalmente, Lee et al. (2022) evidenciaram a eficácia de Redes Neurais Artificiais (ANN) e Sistemas de Inferência Neuro-Fuzzy (ANFIS), notando que métodos avançados baseados em inteligência artificial respondem de maneira distinta dependendo dos horizontes de previsão (curto vs. longo prazo) e das especificidades dos dados.

### 2.2 Comportamento Multissetorial e Heterogeneidade

A literatura demonstra que a precisão preditiva varia substancialmente dependendo do setor analisado. Em análises comparativas, Wu et al. (2024) indicam que edifícios comerciais e industriais possuem perfis de consumo mais regulares e previsíveis, fortemente governados por regulamentações e horários operacionais fixos. Em contrapartida, o setor residencial apresenta alta estocasticidade, tornando a modelagem notoriamente mais desafiadora devido à variação nos comportamentos humanos, estilos de vida e à dependência de variáveis contextuais estáticas.

Cordon et al. (2026) corroboram essa visão ao destacarem que a integração explícita de atributos contextuais e físicos (como composição familiar, área da residência e presença de sistemas específicos) é fundamental para gerar uma previsão acurada da demanda. A ausência de variáveis contextuais resulta em uma degradação substancial da previsão, reforçando a necessidade de mapear características estruturais multissetoriais.

### 2.3 Ferramenta WEKA

Para conduzir experimentalmente este ciclo de Descoberta de Conhecimento em Bases de Dados (KDD), ferramentas de mineração de dados robustas como o WEKA desempenham um papel metodológico indispensável. Segundo León-Munizaga et al. (2020), o WEKA fornece um ambiente de software coeso que engloba a seleção de atributos, pré-processamento de ruídos, e algoritmos nativos de classificação e regressão, facilitando enormemente a visualização de desvios e a validação cruzada do modelo proposto.

Bilal et al. (2022) demonstram o uso eficaz do WEKA para a execução ágil de análises comparativas de tempo computacional e erro absoluto (RMSE, MAE) entre os principais algoritmos (como MLP, KNN, Processos Gaussianos e SVR) para dados de séries temporais de consumo, consolidando a ferramenta como o padrão analítico primário para experimentos práticos e otimização de Machine Learning no setor de energia.

## 3. Atributos Relevantes

A seleção dos atributos que compõem o conjunto de dados é amplamente justificada pelos achados empíricos dos autores. Abaixo está a fundamentação da relevância de cada atributo, comprovando sua necessidade para a modelagem:

### 3.1 Setor de Atividade

Wu et al. (2024) defendem a necessidade de informações abrangentes sobre o consumo de energia em diversos tipos de edifícios (como residenciais, comerciais e industriais). A inclusão desse atributo é vital porque os preditores e a própria acurácia dos modelos divergem drasticamente dependendo do setor analisado.

### 3.2 Variáveis Temporais

Fatores relacionados ao tempo são determinantes universais do consumo. Wu et al. (2024) concluíram que o "dia da semana" é uma das variáveis mais importantes e com maior impacto no consumo de energia em praticamente todos os tipos de edifícios. Ding et al. (2022) também ratificam essa importância, relatando que a distinção entre o tipo de dia (ex: dia de semana vs. fim de semana) dita os padrões de ocupação e afeta diretamente a demanda.

### 3.3 Condições Climáticas

As condições meteorológicas são os fatores exógenos de maior peso. Wu et al. (2024) apontam que as condições climáticas, como a temperatura ambiente, são o principal fator de previsão que afeta as condições internas de um edifício. Essa premissa é reforçada por Zhao et al. (2025), que, por meio de análises de interpretabilidade, confirmaram que variáveis meteorológicas atuam como contribuintes dominantes nas flutuações e previsões de energia elétrica.

### 3.4 Atributos Físicos

As características físicas da estrutura são indispensáveis para a definição da carga de base. Cordon et al. (2026) estabelecem a "área do piso aquecido/resfriado" (tamanho da habitação) como uma variável estática e estrutural essencial. A área em metros quadrados introduz uma heterogeneidade estrutural realista na modelagem, servindo como indicador para o limite máximo da demanda de energia e climatização de um local.

### 3.5 Ocupação de Ativos

A dimensão humana e o nível de atividade em um edifício moldam a curva diária de carga. Cordon et al. (2026) destacam que a composição do local, representada pelo número de residentes ou ocupantes, afeta diretamente o uso de equipamentos básicos e o comportamento de consumo.

### 3.6 Consumo de Energia

É a variável alvo quantitativa padrão (geralmente expressa em kWh ou MWh) a ser predita pelo modelo. A literatura estabelece este atributo como a métrica final cujo valor reflete a complexa interação das variáveis descritas acima, sendo usado para treinar e validar os erros (como RMSE e MAE) dos modelos de Machine Learning aplicados.

## Referências

BILAL, M.; KIM, H.; FAYAZ, M.; PAWAR, P. Comparative Analysis of Time Series Forecasting Approaches for Household Electricity Consumption Prediction. 2022. DOI: [10.48550/arXiv.2207.01019](https://doi.org/10.48550/arXiv.2207.01019).

CORDON, D.; PITA, A.; JUAN, A. A. Classifying and Predicting Household Energy Consumption Using Data Analytics and Machine Learning. Algorithms, v. 19, n. 2, art. 114, 2026. DOI: [10.3390/a19020114](https://doi.org/10.3390/a19020114).

DING, Z. et al. A Comprehensive Study on Integrating Clustering with Regression for Short-Term Forecasting of Building Energy Consumption: Case Study of a Green Building. Buildings, v. 12, n. 10, art. 1701, 2022. DOI: [10.3390/buildings12101701](https://doi.org/10.3390/buildings12101701).

LEE, M. H. L. et al. A Comparative Study of Forecasting Electricity Consumption Using Machine Learning Models. Mathematics, v. 10, n. 8, art. 1329, 2022. DOI: [10.3390/math10081329](https://doi.org/10.3390/math10081329).

LEÓN-MUNIZAGA, N.; AGUIRRE-MUNIZAGA, M.; LAGOS-ORTIZ, K.; DEL CIOPPO-MORSTADT, J. Prediction of Energy Consumption in an Electric Arc Furnace Using Weka. In: VALENCIA-GARCÍA, R. et al. (org.). CITI 2020, CCIS 1309. Cham: Springer Nature Switzerland, 2020. p. 58-70. DOI: [10.1007/978-3-030-62015-8_5](https://doi.org/10.1007/978-3-030-62015-8_5).

SARSWATULA, S. A.; PUGH, T.; PRABHU, V. Modeling Energy Consumption Using Machine Learning. Frontiers in Manufacturing Technology, v. 2, art. 855208, 2022. DOI: [10.3389/fmtec.2022.855208](https://doi.org/10.3389/fmtec.2022.855208).

WU, J. et al. A Comparative Analysis of Machine Learning-Based Energy Baseline Models across Multiple Building Types. Energies, v. 17, n. 6, art. 1285, 2024. DOI: [10.3390/en17061285](https://doi.org/10.3390/en17061285).

ZHAO, X.; HUANG, X.; DING, J.; ZHANG, Y. A Comparative Study of Machine Learning Algorithms for Electricity Price Forecasting with LIME-Based Interpretability. arXiv preprint, arXiv:2512.01212, 2025. DOI: [10.1109/ICEIEC65904.2025.11273147](https://doi.org/10.1109/ICEIEC65904.2025.11273147).
