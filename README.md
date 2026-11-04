# Index Tracking: S&P 100 e Ibovespa (Projeto Finor)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Solver](https://img.shields.io/badge/solver-Gurobi-red)

A Finor quer acompanhar o S&P 100 (101 ações) e o Ibovespa (76 ações) sem comprar todas as ações. Este repositório monta carteiras com
poucas ações que andam junto com cada índice e escolhe quantas usar a partir do risco que a Finor aceita. O risco é a chance de a
diferença acumulada entre carteira e índice passar de 4 p.p. em 12 meses, o limite que a CVM aplica aos fundos de índice listados.

## Resultado

| Cenário | Risco máximo aceito | Ações (S&P 100 / Ibovespa) | Erro no teste, % a.a. | Janelas acima de 4 p.p. no teste | Giro, % a.a. |
|---|---|---|---|---|---|
| Conservador | 10% | 50 / Sem limite | 2,1 / 1,7 | 1 de 4 / 0 de 13 | 12 / 6 |
| **Moderado (recomendado)** | **20%** | **30 / 30** | **2,7 / 2,2** | **0 de 4 / 0 de 13** | **15 / 8** |
| Arrojado | 30% | 20 / 30 | 3,6 / 2,2 | 0 de 4 / 0 de 13 | 15 / 8 |

O teste vai de out/2024 a set/2026, meses que não entraram na escolha. A carteira de 30 ações ficou dentro do limite em todas as
janelas de 12 meses. Ela oscilou menos da metade de uma alternativa simples, as 30 ações mais correlacionadas com o índice: 2,7% contra
5,6% no S&P 100 e 2,2% contra 6,9% no Ibovespa. A forma de montar a carteira é escolhida para cada índice: no S&P 100, 1 ano de
histórico, carteira refeita a cada trimestre e penalidade de giro λ = 6; no Ibovespa, 1 ano, todo mês e λ = 3.

O método completo está em [METODOLOGIA.md](METODOLOGIA.md).

## Como reproduzir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
```

O pipeline parte de `data/cache_dados_finor.pkl` e segue grade → regra → teste final → alternativa simples → avaliação → figuras.
Com os resultados já gravados em `results/`, ele roda em cerca de 30 segundos e regrava:

- as tabelas da escolha;
- `results/numeros_relatorio.csv`, com todos os números citados no relatório;
- as nove figuras do relatório, em `figures/final/`.

As simulações só rodam para arquivos ausentes em `results/selecao/` e `results/teste_final/`. Refazê-las precisa da licença
acadêmica gratuita do Gurobi; a licença que vem com o `pip install gurobipy` limita o tamanho do problema.

O notebook [`notebooks/finor_index_tracking.ipynb`](notebooks/finor_index_tracking.ipynb) conta o projeto inteiro, dos dados ao
modelo final. Com os caches do repositório, ele roda de ponta a ponta em cerca de 30 segundos, sem Gurobi, com pandas 2.x ou 3.x.
As chaves da célula de setup refazem cada parte; os tempos medidos estão no início do notebook.

**No Google Colab:** em *Arquivo › Abrir notebook › GitHub*, marque *Incluir repositórios particulares* e escolha
`podorize/index-tracking`. Como o repositório é privado, o notebook precisa de um token do GitHub para clonar os dados: crie um token
com leitura do repositório (GitHub › Settings › Developer settings › Personal access tokens) e guarde-o nos *Secrets* do Colab (ícone
de chave) com o nome `GITHUB_TOKEN`. Depois, é só rodar as células de cima para baixo.

## O problema de otimização

Em cada rodada, a carteira minimiza o erro quadrático diário contra o índice no histórico de treino, com uma penalidade sobre o giro:

$$\min_{w} \; 252 \cdot \frac{1}{T} \sum_{t=1}^{T} \Big( R_t - \sum_i w_i r_{t,i} \Big)^2 + \lambda \cdot \text{giro anualizado}$$

sujeito a $\sum_i w_i = 1$, $w_i \ge 0$ (sem venda a descoberto) e no máximo $K$ ações com peso. A formulação MIQP está em
`src/optimizer.py`. O modelo final a resolve com uma heurística determinística (`src/heuristica_giro.py`) em walk-forward
(`src/walk_forward.py`): a carteira é montada só com dados passados, mantida até a rodada seguinte e então refeita.

## Estrutura

```
data/
  cache_dados_finor.pkl       bases limpas e tabelas de auditoria (gravadas pelo notebook)
  cache_modelo_preliminar.pkl resultados do walk-forward da seção 11 do notebook
  yahoo_raw.pkl               preços brutos do Yahoo Finance, para refazer a limpeza
  fontes/                     fonte Libre Franklin usada nas figuras
notebooks/
  finor_index_tracking.ipynb  o projeto inteiro, dos dados ao modelo final
src/
  data_loader.py              universo de ações e leitura das bases
  optimizer.py                MIQP com cardinalidade e giro (Gurobi)
  heuristica_giro.py          heurística gulosa + trocas com penalidade de giro
  walk_forward.py             simulação rodada a rodada
  config.py                   modelo final: períodos, custos, grade e cenários
  metricas.py                 erro, diferença acumulada e chance de passar de 4 p.p.
  grade.py                    448 carteiras no período de escolha
  regra.py                    regra de escolha por índice e cenários
  teste_final.py              carteiras escolhidas no teste final
  alternativa_simples.py      top 30 por correlação, para comparar
  avaliacao.py                avaliação no teste e números do relatório
  figuras.py                  figuras do relatório
  pipeline.py                 reproduz tudo
  features.py, models_benchmark.py, compare_all_models.py   seção 13: comparação de modelos
  validacao_lambda.py, rebalanceamento_experiment.py         seção 13: λ, treino e frequência
  robustness.py, estatistica.py, pesos_aproximados.py        seção 13: robustez
  figuras_comparacao.py       figuras da seção 13
results/
  selecao/                    as 448 carteiras da grade (rodadas até 01/10/2024)
  teste_final/                carteiras escolhidas e alternativa simples, até set/2026
  cenarios.csv, teste_carteiras.csv, numeros_relatorio.csv, ...   saídas do modelo final
  demais arquivos             saídas do notebook (seções 4 a 13)
figures/
  final/                      figuras do relatório final
  *.png                       figuras do notebook
docs/
  historico_criterios.md      regras de escolha na ordem em que foram fixadas
  relatorio_anterior_overleaf/ versão anterior do relatório (Overleaf)
```

## Como o projeto chegou ao modelo final

O notebook segue uma narrativa só:

1. **Dados e EDA** (seções 1 a 6). Correções auditadas (NATU3, UGPA3), outliers, viés de sobrevivência, correlação, concentração
   e liquidez.
2. **Modelo** (seções 7 a 12). MIQP com cardinalidade, partindo de uma heurística, em 20 trimestres fora da amostra: quantas ações
   são precisas e quanto histórico usar.
3. **Comparação, giro e robustez** (seção 13). Sete modelos com K = 10, de baselines a Lasso, clustering e heurísticas. A heurística
   com penalidade de giro erra quase o mesmo que a melhor alternativa e negocia de 13 a 32 vezes menos. Seguem a escolha de λ, a
   comparação de 1 × 2 anos de treino e de rebalanceamento mensal × trimestral, e os testes de robustez.
4. **Modelo final** (seção 14). K, λ, histórico e frequência passam a ser escolhidos pela chance de passar do limite da CVM,
   separadamente por índice e por apetite ao risco, num período separado do teste.

## Limites

O principal é o viés de sobrevivência: usamos a lista de ações de 30/09/2026 para todo o período. O centro da chance desconta a parte
comum a todas as carteiras, mas parte do erro do Ibovespa vem de uma composição antiga que não temos. Corrigir isso exige a composição
histórica dos índices, proposta como segunda fase do projeto.

## Referências

- Cornuéjols, G.; Tütüncü, R. (2006). *Optimization Methods in Finance*. Cambridge University Press.
- Sant'Anna, L.; Filomena, T. P.; Borenstein, D. (2014). Index Tracking com Controle do Número de Ativos. *Revista Brasileira de Finanças*, 12, 89-119.
- Sant'Anna, L.; Filomena, T. P.; Borenstein, D.; Guedes, P. (2017). Index tracking with controlled number of assets using a hybrid heuristic combining genetic algorithm and non-linear programming. *Annals of Operations Research*, 258, 849-867.
- Comissão de Valores Mobiliários. Resolução CVM 175, Anexo Normativo V, Art. 27.
