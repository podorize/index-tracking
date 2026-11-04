# Metodologia do modelo final

Este documento descreve o modelo do relatório final e como ele é reproduzido pelo código (`python -m src.pipeline`). O caminho até
ele, contado no notebook, está resumido no fim.

## Pergunta

Quantas ações a Finor precisa para acompanhar o S&P 100 (101 ações, em dólar) e o Ibovespa (76 ações, em reais), dado o risco que
aceita? O risco é a chance de a diferença acumulada entre a carteira e o índice passar de 4 p.p. em 12 meses, o limite que a CVM
aplica aos fundos de índice listados (Resolução CVM 175, Anexo Normativo V, Art. 27), contada nos dois sentidos.

## Dados

Preços diários do Yahoo Finance de 30/09/2019 a 30/09/2026, com a lista de ações de 30/09/2026 (`data/cache_dados_finor.pkl`,
montado e auditado no notebook). Usar a lista atual em todo o período cria viés de sobrevivência; ver "Limites".

## Como cada carteira é montada

Em cada rodada, a heurística com controle de giro (`src/heuristica_giro.py`) escolhe até K ações e os pesos que minimizam o erro
quadrático diário contra o índice no histórico de treino, mais uma penalidade λ sobre o giro anualizado. A carteira é mantida até a
rodada seguinte e então refeita (walk-forward, `src/walk_forward.py`). Custos por operação: 0,05% no S&P 100 e 0,15% no Ibovespa,
descontados dos retornos.

## Períodos

- Escolha: rodadas que terminam até 01/10/2024 (out/2021 a set/2024).
- Teste final: rodadas seguintes (out/2024 a set/2026), nunca usadas na escolha. As carteiras do teste são simuladas desde o início,
  com os parâmetros congelados, e só as rodadas do teste entram na avaliação.

## A grade (`src/grade.py`)

448 carteiras por índice e configuração: K ∈ {5, 10, 15, 20, 30, 50, sem limite} × treino de 1 ou 2 anos × rebalanceamento mensal ou
trimestral × λ ∈ {0; 0,5; 1,5; 3; 6; 12; 25; 50} × 2 índices.

## A chance (`src/metricas.py`)

Em cada janela de 12 meses seguidos, a diferença acumulada é tratada como normal:

- largura: o erro de rastreamento da carteira contra o índice na janela;
- centro: o desvio médio da carteira menos o da carteira sem limite montada da mesma forma (mesmo histórico, frequência e λ). Isso
  desconta o viés de sobrevivência, que infla igualmente o rendimento de todas as carteiras.

A chance da janela é P(diferença < −4) + P(diferença > +4); a chance da carteira é a média das janelas. A mesma conta, com a mesma
referência, é usada na escolha e no teste.

## A regra (`src/regra.py`), separada por índice

1. **Forma de montar.** Para cada uma das 32 combinações de histórico, frequência e λ, a chance média das sete carteiras (5 a 50 ações
   e sem limite); fica a menor. Resultado: S&P 100 com 1 ano, trimestral, λ = 6; Ibovespa com 1 ano, mensal, λ = 3.
2. **Número de ações.** Com a forma fixa, o menor K cuja chance mais um erro padrão cabe no risco máximo aceito. O erro padrão vem de
   um bootstrap circular em blocos de janelas vizinhas (12 mensais ou 4 trimestrais, 2.000 sorteios, semente 0), e a curva de chance é
   forçada a não subir com o K. Se nenhum K couber, fica a carteira sem limite, com aviso.

| Cenário | Risco máximo | S&P 100 | Ibovespa |
|---|---|---|---|
| Conservador | 10% | 50 ações | sem limite (nenhuma cabe) |
| Moderado (recomendado) | 20% | 30 ações | 30 ações |
| Arrojado | 30% | 20 ações | 30 ações |

## Avaliação no teste (`src/avaliacao.py`)

Para cada carteira dos cenários, e para a de 20 ações do Ibovespa, que a regra deixou de fora, a avaliação calcula:

- a chance com a mesma conta;
- o erro de rastreamento;
- as janelas de 12 meses inteiras no teste que passaram de 4 p.p. (4 no S&P 100, trimestral; 13 no Ibovespa, mensal);
- o giro e o custo;
- a comparação com uma alternativa simples (`src/alternativa_simples.py`): as 30 ações mais correlacionadas com o índice no último ano,
  com pesos de mínimos quadrados, refeita todo mês.

Também calcula a decomposição de Shapley da mudança da chance entre largura e centro. Todos os números citados no relatório ficam em
`results/numeros_relatorio.csv`.

## App (`app.py`, `src/app_dados.py`)

O app aplica a etapa 2 da regra a qualquer risco máximo, de 5% a 40%, e não só aos três cenários. Para isso, `src/app_dados.py`
simula na forma de montar de cada índice as sete carteiras da grade (5 a 50 ações e sem limite) do início da escolha ao fim dos
dados, com os pesos de cada rodada, e calcula os números do teste com a mesma conta de `src/avaliacao.py`. As carteiras dos cenários
coincidem com as do teste final.

## Limites

- **Viés de sobrevivência.** As empresas que saíram dos índices não estão nos dados. O centro da chance desconta a parte comum a todas
  as carteiras, mas a largura não: parte do erro do Ibovespa vem de uma composição antiga que não temos. Em operação, com a lista
  vigente, o viés some.
- **A curva normal.** Ela subestima eventos raros.
- **Amostra curta.** O teste é curto e as janelas se sobrepõem.
- **Escolha do S&P 100 apertada.** A forma de montar do S&P 100 venceu por 0,25 p.p., menos que o erro padrão.

## Como chegamos aqui

O notebook `notebooks/finor_index_tracking.ipynb` conta o caminho numa narrativa só, com os módulos correspondentes de `src/`:

1. **Dados e EDA** (seções 1 a 6 e apêndice A): coleta, correções auditadas (NATU3, UGPA3), qualidade, outliers, viés de
   sobrevivência, correlação, concentração e liquidez.
2. **Modelo de otimização** (seções 7 a 12): MIQP com restrição de cardinalidade (`src/optimizer.py`), resolvido no Gurobi a partir da
   heurística, em walk-forward de 20 trimestres; tamanho da janela de treino e número de ações.
3. **Comparação, giro e robustez** (seção 13, `src/compare_all_models.py`, `src/validacao_lambda.py`,
   `src/rebalanceamento_experiment.py`, `src/robustness.py`, `src/estatistica.py`): sete modelos com K = 10. A heurística com
   penalidade de giro erra quase o mesmo que a melhor alternativa e negocia de 13 a 32 vezes menos. Ela é determinística, ao
   contrário do Gurobi, que para por tempo, e por isso é o motor do modelo final. Seguem a escolha de λ, a comparação de 1 × 2 anos de
   treino e de rebalanceamento mensal × trimestral, e o modelo nulo, os regimes e o estresse.
4. **Modelo final** (seção 14): o que está descrito acima.

Até a seção 13, K, λ e a configuração eram comparados pelo erro de rastreamento médio, nos mesmos trimestres em que eram medidos. O
modelo final os escolhe pela chance de passar do limite da CVM, separadamente por índice e por apetite ao risco, num período separado
do teste.
