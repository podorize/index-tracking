# FINOR - insumos para documentação e apresentação (v7.7)

- Período dos dados: 2019-09-30 a 2026-09-30; universo fixo de 2026-09-30.
- Modo executado: preliminar; 20 trimestres fora da amostra (set/2021 a set/2026).
- Carteira de referência: Heurística com K=10 e treino de 504 pregões.
- MIQP: limite de 10 s por carteira, partida a quente pela heurística; tempo total do modelo 38.6 min.

## Resultados do modelo

- **S&P 100:** erro fora da amostra de 7,4% com 5, 5,1% com 10, 3,6% com 20 ações; a carteira sem limite (~83 ações) erra 2,3%. Escolher as 10 mais correlacionadas erra 8,2%.
- **Ibovespa:** erro fora da amostra de 6,5% com 5, 4,5% com 10, 3,4% com 20 ações; a carteira sem limite (~62 ações) erra 2,4%. Escolher as 10 mais correlacionadas erra 10,0%.
- **MIQP × heurística:** em 10 s por carteira, o Gurobi reduziu o erro de treino da heurística em 0,00% na mediana (máximo de 15,6%) e terminou com gap mediano de 63%. Fora da amostra, a diferença média de TE foi de −0,06 p.p. A heurística já entrega quase todo o ganho; o MIQP serve como verificação.
- **Dentro × fora da amostra (K=10):** o erro fora é 1,4 vez o de treino no S&P 100 e 1,1 no Ibovespa.
- **Diferença acumulada:** a carteira Heurística com 10 ações ficou +2,1 p.p. por ano em relação ao S&P 100 e +2,0 p.p. em relação ao Ibovespa. A carteira sem limite também fica acima dos índices (+1,6 e +2,5 p.p. por ano): parte da diferença vem do viés de sobrevivência, não do modelo.
- **Giro:** com 10 ações, 36% da carteira é trocada por trimestre no S&P 100 e 29% no Ibovespa, contra 12% e 11% sem limite de ações.
- **Janela de treino:** com K=10, treinar com 252 pregões leva o TE de 5,1% para 4,8% no S&P 100 e de 4,5% para 4,6% no Ibovespa.

## Resumo (treino de 504 pregões)

indice  janela treino           método  K  ações (média)  TE dentro (% a.a.)  TE fora (% a.a.)  TD (p.p. a.a.)  razão fora/dentro  RMSE fora (% a.a.)  giro médio (%)  tempo médio (s)  gap médio (%)
S&P100            504             MIQP  5           5.00                5.48              7.16            2.09               1.31                7.16           23.96            10.11          73.87
S&P100            504             MIQP 10          10.00                3.50              4.96            3.00               1.42                4.97           28.92            10.29          82.19
S&P100            504             MIQP 15          15.00                2.55              3.95            3.77               1.55                3.95           30.02            10.50          75.59
S&P100            504             MIQP 20          20.00                2.05              3.57            3.48               1.74                3.58           30.98            11.41          64.77
S&P100            504             MIQP 30          30.00                1.55              2.80            2.40               1.80                2.81           25.07            12.12          41.41
S&P100            504       Heurística  5           5.00                5.53              7.44            1.86               1.34                7.43           24.55             0.06            NaN
S&P100            504       Heurística 10          10.00                3.55              5.15            2.12               1.45                5.15           36.46             0.27            NaN
S&P100            504       Heurística 15          15.00                2.58              4.05            3.76               1.57                4.06           33.96             0.47            NaN
S&P100            504       Heurística 20          20.00                2.05              3.57            3.52               1.74                3.58           30.48             1.38            NaN
S&P100            504       Heurística 30          30.00                1.55              2.80            2.40               1.80                2.81           25.07             2.07            NaN
S&P100            504 Top-K correlação  5           4.65                9.78             10.17           -3.12               1.04               10.17           21.95             0.00            NaN
S&P100            504 Top-K correlação 10           8.45                7.43              8.25           -0.95               1.11                8.25           25.32             0.01            NaN
S&P100            504 Top-K correlação 15          11.30                6.17              7.34            0.82               1.19                7.33           20.45             0.03            NaN
S&P100            504 Top-K correlação 20          14.70                5.27              6.47            3.46               1.23                6.47           21.10             0.03            NaN
S&P100            504 Top-K correlação 30          21.90                3.88              4.88            4.39               1.26                4.89           17.94             0.06            NaN
S&P100            504       Sem limite  0          83.35                1.18              2.32            1.62               1.98                2.33           12.38             0.03            NaN
  IBOV            504             MIQP  5           5.00                6.28              6.43            5.32               1.02                6.44           25.83            10.10          63.98
  IBOV            504             MIQP 10          10.00                4.03              4.52            1.75               1.12                4.51           26.63            10.20          68.07
  IBOV            504             MIQP 15          15.00                3.15              3.70            2.37               1.17                3.70           19.67            11.00          55.56
  IBOV            504             MIQP 20          20.00                2.68              3.36            2.37               1.25                3.36           24.38            10.49          42.61
  IBOV            504             MIQP 30          30.00                2.21              2.71            2.37               1.23                2.71           18.04            13.59          19.13
  IBOV            504       Heurística  5           5.00                6.28              6.48            5.70               1.03                6.49           22.88             0.08            NaN
  IBOV            504       Heurística 10          10.00                4.04              4.54            1.97               1.12                4.54           28.51             0.17            NaN
  IBOV            504       Heurística 15          15.00                3.17              3.77            2.60               1.19                3.77           21.46             0.70            NaN
  IBOV            504       Heurística 20          20.00                2.68              3.36            2.25               1.25                3.36           24.10             0.48            NaN
  IBOV            504       Heurística 30          30.00                2.21              2.72            2.23               1.23                2.73           19.15             3.31            NaN
  IBOV            504 Top-K correlação  5           4.55               12.44             11.65            6.91               0.94               11.66           13.79             0.00            NaN
  IBOV            504 Top-K correlação 10           8.70               10.69              9.95            4.57               0.93                9.95           20.59             0.01            NaN
  IBOV            504 Top-K correlação 15          12.50                9.69              9.15            3.12               0.94                9.15           18.11             0.01            NaN
  IBOV            504 Top-K correlação 20          15.40                8.79              8.59            3.82               0.98                8.59           25.54             0.05            NaN
  IBOV            504 Top-K correlação 30          23.05                6.24              6.41            3.91               1.03                6.42           33.48             0.04            NaN
  IBOV            504       Sem limite  0          61.80                1.97              2.40            2.48               1.22                2.40           10.72             0.03            NaN

## Alertas para a interpretação
- O universo final é conhecido antecipadamente: viés de sobrevivência (diferença acumulada positiva).
- O S&P 100 usa retorno de preços; o Ibovespa, retorno total aproximado.
- MIQP com gap alto: soluções factíveis, sem prova de ótimo global.
- Giro penalizado só na seção 13; custos simplificados (taxa fixa).

## Figuras para os slides (pasta figuras/)
- 01_cobertura_dados.png
- 01b_outliers_por_mes.png
- 01c_mercado_ou_empresa.png
- 02_performance.png
- 02b_distribuicao_retornos.png
- 03_retornos_mensais.png
- 03b_volatilidade.png
- 04_comparacao_benchmarks.png
- 04b_retornos_anuais.png
- 05_heatmaps_correlacao.png
- 06_correlacao_por_setor.png
- 11_correlacao_por_regime.png
- 12_consistencia_universo_indice.png
- 12b_vies_universo_fora_amostra.png
- 13_risco_retorno.png
- 14_correlacao_com_indice_ranqueada.png
- 15_beta_risco_idiossincratico.png
- 16_estabilidade_ranking.png
- 17_assimetria_correlacao.png
- 18_pca_variancia_acumulada.png
- 19_liquidez.png
- 20_concentracao_indices.png
- 21_sensibilidade_janela_treino.png
- 22_fronteira_erro_por_k.png
- 24_erro_por_trimestre.png
- 25_carteira_vs_indice.png
- 29_giro_por_k.png
- 31_comparativo_modelos.png
- 33_modelo_nulo.png
- 34_te_por_regime.png
- 35_diferenca_acumulada.png
- 36_rebalanceamento_janela.png
- 38_validacao_lambda.png
- 41_erro_trimestral_modelo_final.png
- comparacao_todos_modelos_K10.png
- modelo_nulo_monte_carlo.png
- modelo_nulo_monte_carlo_K10.png