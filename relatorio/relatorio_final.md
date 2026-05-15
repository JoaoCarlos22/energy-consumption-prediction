# Relatório Final

## Previsão de Consumo de Energia Elétrica Multissetorial com Machine Learning no Weka

Trabalho prático de Inteligência Artificial aplicado à previsão de demanda energética em ambientes residenciais, comerciais e industriais. O projeto usa um dataset sintético controlado, executa análise exploratória e pré-processamento no Weka, compara algoritmos supervisionados de regressão e seleciona o modelo com melhor equilíbrio entre erro, estabilidade e capacidade de generalização.

**Repositório:** `energy-consumption-prediction`  
**Ferramenta principal:** Weka  
**Tarefa de aprendizado:** Regressão supervisionada  
**Variável-alvo:** `Consumo_Energia`  
**Ano:** 2026

---

## Como Ler Este Relatório

Este arquivo é a versão resumida e explicativa do trabalho. Ele concentra o objetivo, as decisões metodológicas, os principais resultados e a conclusão final. Os documentos completos continuam separados por etapa e estão linkados abaixo, permitindo consultar detalhes, prints, tabelas, prompts, justificativas e interpretações sem deixar o relatório principal excessivamente longo.

---

## Documentação Completa Linkada

| Área | Documento | O que contém |
|---|---|---|
| Visão geral | [README.md](../README.md) | Apresentação do projeto, estrutura do repositório, principais resultados e instruções de reprodução. |
| Relatório | [relatorio_final.md](relatorio_final.md) | Este resumo consolidado do trabalho, com os links para todos os demais documentos. |
| Fundamentação | [definicao_problema.md](../fundamentacao/definicao_problema.md) | Formulação do problema, justificativa do uso de regressão e descrição dos atributos relevantes. |
| Fundamentação | [revisao_bibliografica.md](../fundamentacao/revisao_bibliografica.md) | Processo de pesquisa bibliográfica, critérios de busca, fontes consultadas e síntese metodológica. |
| Fundamentação | [relatorio_pesquisa.md](../fundamentacao/relatorio_pesquisa.md) | Discussão ampliada sobre tarefas de aprendizado de máquina, Weka e aplicações em energia. |
| Dataset | [geracao_dataset.md](../dataset/geracao_dataset.md) | Geração do dataset sintético, prompts utilizados, atributos, valores ausentes, ruído, outliers e arquivos de saída. |
| Análise | [analise_exploratoria.md](../analise/analise_exploratoria.md) | Análise inicial do dataset no Weka, distribuição dos atributos, valores faltantes, outliers e recomendações. |
| Análise | [correlacoes.md](../analise/correlacoes.md) | Visualizações e interpretação das relações entre atributos e consumo de energia. |
| Pré-processamento | [analise_inicial.md](../preprocessamento/analise_inicial.md) | Justificativa dos filtros escolhidos e problemas identificados antes do tratamento dos dados. |
| Pré-processamento | [descricao_etapas.md](../preprocessamento/descricao_etapas.md) | Procedimento aplicado no Weka: imputação, logaritmo, remoção, binarização e padronização. |
| Modelagem | [modelagem_treinamento_testes_weka.md](../treino/teste/modelagem_treinamento_testes_weka.md) | Treinamento dos modelos, configurações, resultados completos, gráficos, discussão e limitações. |
| Resultados | [somente_tabelas_resultados_weka.md](../treino/teste/somente_tabelas_resultados_weka.md) | Tabelas consolidadas das métricas obtidas no Weka. |

---

## Sumário

1. [Resumo Executivo](#resumo-executivo)
2. [Problema e Objetivo](#problema-e-objetivo)
3. [Fundamentação Teórica](#fundamentação-teórica)
4. [Dataset](#dataset)
5. [Análise Exploratória](#análise-exploratória)
6. [Pré-processamento](#pré-processamento)
7. [Modelagem e Avaliação](#modelagem-e-avaliação)
8. [Resultados Principais](#resultados-principais)
9. [Discussão e Modelo Final](#discussão-e-modelo-final)
10. [Limitações](#limitações)
11. [Conclusão](#conclusão)

---

## Resumo Executivo

O projeto desenvolveu um fluxo completo de mineração de dados para prever consumo de energia elétrica a partir de características contextuais e estruturais de unidades consumidoras. O consumo foi tratado como uma variável contínua, portanto a abordagem correta foi a regressão supervisionada.

O dataset possui **520 instâncias** e **7 atributos**, incluindo setor de atividade, dia da semana, temperatura média, área, ocupantes ativos, número de extintores e consumo de energia. A base foi construída de forma sintética e controlada, com valores ausentes, ruído estatístico e outliers planejados para simular desafios comuns em bases reais.

Após a análise exploratória, os dados foram tratados no Weka com imputação de valores faltantes, transformação logarítmica, remoção de atributo irrelevante, conversão de variáveis categóricas e padronização. Em seguida, foram comparados algoritmos como ZeroR, RandomTree, RandomForest, LinearRegression, IBk, SMOreg e MultilayerPerceptron.

Os melhores resultados se concentraram em **LinearRegression**, **SMOreg** e **RandomForest**. O modelo escolhido como melhor alternativa final foi o **RandomForest**, porque apresentou o menor RMSE no cenário Percentage Split 70/30, manteve alta correlação e demonstrou bom equilíbrio entre erro baixo e estabilidade.

---

## Problema e Objetivo

A previsão de consumo de energia elétrica é relevante para planejamento energético, controle operacional e redução de desperdícios. Em um cenário multissetorial, o desafio aumenta porque os setores residencial, comercial e industrial possuem padrões de consumo distintos.

O objetivo do trabalho foi construir e avaliar modelos capazes de estimar `Consumo_Energia` a partir de atributos como setor, dia da semana, temperatura, área e ocupação. A formulação completa do problema, junto com a justificativa dos atributos, está em [definicao_problema.md](../fundamentacao/definicao_problema.md).

---

## Fundamentação Teórica

A revisão bibliográfica indicou que aprendizado de máquina é amplamente aplicado à previsão de demanda energética, identificação de padrões de consumo, detecção de anomalias e apoio à eficiência energética. Para prever valores contínuos, como consumo em energia, a tarefa de regressão é a escolha mais adequada.

Também foi identificado que variáveis estruturais, operacionais e climáticas costumam influenciar o consumo. Por isso, o dataset considerou área física, temperatura, setor de atividade e ocupação. O Weka foi adotado por integrar inspeção de dados, aplicação de filtros, validação cruzada e comparação de algoritmos em um mesmo ambiente.

Os detalhes da pesquisa estão em [revisao_bibliografica.md](../fundamentacao/revisao_bibliografica.md) e a discussão ampliada das tarefas de aprendizado de máquina está em [relatorio_pesquisa.md](../fundamentacao/relatorio_pesquisa.md).

---

## Dataset

O dataset foi gerado de forma sintética com apoio de modelo de linguagem, seguindo regras de domínio para representar consumo em setores residencial, comercial e industrial. A documentação completa da geração está em [geracao_dataset.md](../dataset/geracao_dataset.md).

| Atributo | Tipo | Papel no problema |
|---|---|---|
| `Setor_Atividade` | Categórico | Identifica se a unidade é residencial, comercial ou industrial. |
| `Dia_da_Semana` | Categórico | Representa o contexto temporal da medição. |
| `Temperatura_Media` | Numérico | Captura influência climática sobre o consumo. |
| `Area_Metragem` | Numérico | Representa o porte físico da unidade consumidora. |
| `Ocupantes_Ativos` | Numérico | Representa pessoas ou máquinas em operação. |
| `Num_Extintores` | Numérico | Atributo propositalmente irrelevante, usado para testar remoção de ruído. |
| `Consumo_Energia` | Numérico | Variável-alvo da regressão. |

Foram inseridos valores ausentes em atributos preditores numéricos, ruído gaussiano e outliers planejados. A variável-alvo foi mantida sem valores ausentes para preservar a coerência da tarefa supervisionada.

---

## Análise Exploratória

A análise exploratória foi feita no Weka para verificar a qualidade geral da base, a distribuição dos atributos, a presença de faltantes e a coerência das relações entre variáveis. A análise completa está em [analise_exploratoria.md](../analise/analise_exploratoria.md), e as relações visuais estão em [correlacoes.md](../analise/correlacoes.md).

Principais achados:

- O dataset possui **520 instâncias**, sendo **416 completas** e **104 com algum valor ausente**.
- A variável-alvo `Consumo_Energia` não possui valores faltantes.
- Os setores residencial, comercial e industrial ficaram aproximadamente equilibrados.
- `Consumo_Energia` apresentou assimetria positiva e outliers planejados, comportamento coerente com consumo multissetorial.
- Área, temperatura e setor mostraram relação relevante com o consumo, indicando que o dataset possui padrões aprendíveis.

---

## Pré-processamento

O pré-processamento foi necessário para preparar os dados para algoritmos sensíveis a escala, valores ausentes e atributos categóricos. As justificativas estão em [analise_inicial.md](../preprocessamento/analise_inicial.md) e o passo a passo aplicado no Weka está em [descricao_etapas.md](../preprocessamento/descricao_etapas.md).

Etapas aplicadas:

1. `ReplaceMissingValues`: substituiu valores faltantes sem remover instâncias.
2. `MathExpression`: aplicou transformação logarítmica em atributos assimétricos.
3. `Remove`: removeu `Num_Extintores`, por não ter relação lógica direta com o consumo.
4. `NominalToBinary`: converteu `Setor_Atividade` e `Dia_da_Semana` para formato adequado aos modelos.
5. `Standardize`: padronizou atributos preditores, preservando `Consumo_Energia` como classe.

Essa sequência reduziu ruído, melhorou a distribuição dos dados e deixou o conjunto mais adequado para comparação entre diferentes algoritmos.

---

## Modelagem e Avaliação

A modelagem foi realizada na aba **Classify** do Weka. Como `Consumo_Energia` é numérico, os modelos foram avaliados como regressão, e não como classificação. Por isso, métricas como acurácia, precisão, recall, F1-score e matriz de confusão não foram usadas como critério principal.

As estratégias de avaliação foram:

| Estratégia | Finalidade |
|---|---|
| 10-fold cross-validation | Estimar desempenho médio de forma mais robusta. |
| Percentage Split 70/30 | Avaliar o comportamento em uma divisão simples entre treino e teste. |

As métricas consideradas foram correlação, MAE, RMSE, RAE e RRSE. A documentação completa da modelagem está em [modelagem_treinamento_testes_weka.md](../treino/teste/modelagem_treinamento_testes_weka.md), e as tabelas consolidadas estão em [somente_tabelas_resultados_weka.md](../treino/teste/somente_tabelas_resultados_weka.md).

---

## Resultados Principais

Na validação cruzada 10-fold, os três melhores modelos ficaram muito próximos:

| Modelo | Correlação | MAE | RMSE | Destaque |
|---|---:|---:|---:|---|
| LinearRegression | 0.9183 | 0.1673 | **0.3906** | Menor RMSE no 10-fold. |
| SMOreg | **0.9187** | **0.1512** | 0.3921 | Melhor correlação e menor MAE. |
| RandomForest | 0.9177 | 0.1627 | 0.3920 | Desempenho alto e estável. |

No Percentage Split 70/30, o RandomForest apresentou o melhor equilíbrio:

| Modelo | Correlação | MAE | RMSE | Destaque |
|---|---:|---:|---:|---|
| RandomForest | **0.9208** | 0.1516 | **0.3745** | Melhor correlação e menor RMSE no split. |
| LinearRegression | 0.9204 | 0.1587 | 0.3775 | Modelo simples e competitivo. |
| SMOreg | 0.9182 | **0.1480** | 0.3831 | Menor MAE no split. |

O ZeroR funcionou como baseline e teve os maiores erros, confirmando que os demais modelos aprenderam relações úteis entre os atributos preditores e o consumo.

---

## Discussão e Modelo Final

O LinearRegression teve ótimo desempenho, sugerindo que há relações aproximadamente lineares fortes no dataset após o pré-processamento. O SMOreg obteve erros absolutos baixos, indicando boa precisão média. O RandomForest, por sua vez, combinou alta correlação, baixo RMSE e maior robustez no cenário de teste separado.

O modelo final escolhido foi o **RandomForest**. A decisão considera que ele:

- obteve o menor RMSE no Percentage Split 70/30;
- manteve a maior correlação no mesmo cenário;
- ficou muito próximo dos melhores modelos na validação cruzada;
- é menos dependente de uma única estrutura de decisão, pois combina várias árvores;
- oferece bom equilíbrio entre desempenho e estabilidade.

LinearRegression e SMOreg permanecem alternativas fortes, especialmente se o critério principal for simplicidade interpretável ou menor erro absoluto médio.

---

## Limitações

As principais limitações do trabalho são:

- O dataset é sintético, embora tenha sido criado com regras e problemas realistas.
- O tamanho de 520 instâncias é adequado para experimentação didática, mas limitado para generalização em produção.
- Os erros foram avaliados após transformação logarítmica, então não representam diretamente valores na escala original de consumo.
- O resultado pode mudar com outros métodos de imputação, tratamento de outliers, seleção de atributos ou ajuste de hiperparâmetros.
- O MultilayerPerceptron pode exigir parametrização mais cuidadosa para uma comparação mais justa.

Como continuidade, seria recomendável testar dados reais, incluir variáveis temporais mais detalhadas, avaliar modelos adicionais e comparar os resultados com novas estratégias de validação.

---

## Conclusão

O trabalho cumpriu o objetivo de construir, tratar, avaliar e documentar um pipeline de aprendizado supervisionado para previsão de consumo de energia elétrica multissetorial.

O pré-processamento foi decisivo para melhorar a qualidade dos dados e permitir comparação justa entre os modelos. A modelagem confirmou que regressão é adequada ao problema, e os resultados mostraram desempenho consistente para LinearRegression, SMOreg e RandomForest.

Considerando desempenho, estabilidade e robustez, o **RandomForest** foi selecionado como melhor modelo final. O relatório resumido fica neste arquivo, enquanto os documentos completos linkados acima preservam toda a rastreabilidade do projeto.
