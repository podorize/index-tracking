# Index Tracking: S&P 100 & IBOVESPA (Projeto Finor)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/podorize/index-tracking/blob/main/notebooks/index_tracking_colab_otimizado.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Solver](https://img.shields.io/badge/solver-Gurobi%20MIQP-red)
![License](https://img.shields.io/badge/license-MIT-green)

Repositório quantitativo do projeto de **Index Tracking com Restrição de Cardinalidade ($K$) e Controle de Turnover** desenvolvido para o Bootcamp Finor 2026.

O projeto avalia a replicação passiva do **S&P 100 (^OEX)** e do **IBOVESPA (^BVSP)** ao longo de sete anos de histórico (2019 a 2026), testado fora da amostra (*out-of-sample*) em **20 trimestres consecutivos** via *walk-forward*.

---

## ⚡ Otimização para o Google Colab

O notebook original demorava mais de **25 a 30 minutos** para rodar do zero em cada sessão do Google Colab devido ao download repetido de 179 tickers e re-otimização sem cache.

Este repositório foi reestruturado para eliminar esse gargalo:
1. **Zero Download Repetido:** O dataset limpo, auditado e corrigido (Natura, Ultrapar, Copasa) fica versionado em [`data/bases_limpas.pkl`](data/bases_limpas.pkl) e carrega em **0,1 segundo**.
2. **Resultados e Figuras Pré-computados:** Todas as tabelas consolidadas e gráficos da Fronteira de Pareto estão salvos em [`results/`](results/) e [`figures/`](figures/), permitindo visualização instantânea no Colab sem esperar o solver.
3. **Formulação MIQP Compacta:** O problema de turnover foi reformulado para adicionar apenas $K \le 10$ variáveis contínuas de venda (em vez de $2N \approx 400$), reduzindo o tempo de resolução do Gurobi em mais de 70% e tornando-o 100% compatível com licenças demo/acadêmicas.
4. **Modos de Execução:**
   * `MODO = "instantaneo"`: Carrega os resultados pré-computados e gráficos em **2 segundos**.
   * `MODO = "recomputar"`: Executa a otimização interativa fora da amostra em tempo recorde (~1 a 2 minutos).

---

## 📐 O Problema Matemático

O modelo resolve uma **Programação Quadrática Inteira Mista (MIQP)** minimizando o erro quadrático médio diário com penalidade de turnover anualizada:

$$\min_{w, z, s} \; \underbrace{252 \cdot \left[ \frac{1}{T} \sum_{t=1}^T \left( R_t - \sum_{i \in I} w_i r_{t,i} \right)^2 \right]}_{\approx \text{TE}^2 \text{ anual}} + \lambda_{\text{anual}} \cdot \underbrace{\left( 4 \sum_{j \in S_{\text{prev}}} s_j + 4 w_{\text{deslistadas}} \right)}_{\text{Turnover anualizado}}$$

**Sujeito a:**
1. $\sum_{i \in I} w_i = 1$ *(orçamento 100% investido)*
2. $0 \le w_i \le z_i \quad \forall i \in I$ *(sem venda a descoberto)*
3. $\sum_{i \in I} z_i \le K$ *(máximo de $K$ ações na carteira)*
4. $s_j \ge w_j^{\text{drift}} - w_j, \quad s_j \ge 0 \quad \forall j \in S_{\text{prev}}$ *(conservação de fluxo de vendas)*
5. $z_i \in \{0, 1\}$

### Relação Euclidiana entre RMSE, Tracking Error e Viés
$$\text{RMSE}_{\text{anual}}^2 = \text{TE}^2 + \frac{\text{viés}^2}{252}$$
Pelo Teorema de Pitágoras no $\mathbb{R}^T$, o erro total decompõe-se ortogonalmente entre o **viés médio** (paralelo ao vetor unitário $\mathbf{1}$) e o **tracking error** (resíduo centrado ortogonal a $\mathbf{1}$).

---

## 📊 Comparativo Geral de Modelos (Guia Parte 2 - Tabela 4.3)

Abaixo está o benchmark consolidado dos 7 modelos avaliados em **20 trimestres fora da amostra (set/2021 a set/2026)** com $K = 10$, considerando custos explícitos de transação:

### S&P 100 ($K = 10$, $c = 5\text{ bps}$)
| Modelo | Família | Tracking Error (% a.a.) | Correlação | $R^2$ | Giro Médio (% trim) | Retorno Líquido (% a.a.) | Alfa Líquido (p.p.) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Top K Peso** | Heurística Ingênua | 10,29% | 0,91 | 0,67 | 7,85% | 25,44% | +11,49 |
| **Baseline: Top K Corr (1/K)** | Heurística Ingênua | 10,90% | 0,93 | 0,63 | 18,04% | 19,12% | +5,17 |
| **Top K Corr (QP Otimizado)** | Seleção + Alocação | 8,25% | 0,93 | 0,79 | 25,32% | 11,98% | -1,97 |
| **Lasso (L1 Positivo + QP)** | Regularização $L_1$ | 14,74% | 0,89 | 0,32 | 37,30% | 20,62% | +6,68 |
| **Clustering Hierárquico + Repr.** | Não-Supervisionado | 8,96% | 0,87 | 0,75 | 37,67% | 13,04% | -0,90 |
| **Heurística Gulosa + Swaps** | Busca Combinatória | 5,11% | 0,96 | 0,92 | 34,98% | 17,18% | +3,24 |
| **MIQP Gurobi (Turnover $\lambda=6.0$)** | Otimização Global | **4,91%** | **0,96** | **0,92** | **4,52%** | **18,18%** | **+4,23** |

### IBOVESPA ($K = 10$, $c = 15\text{ bps}$)
| Modelo | Família | Tracking Error (% a.a.) | Correlação | $R^2$ | Giro Médio (% trim) | Retorno Líquido (% a.a.) | Alfa Líquido (p.p.) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Top K Peso** | Heurística Ingênua | 12,19% | 0,83 | 0,51 | 13,09% | 8,50% | -2,12 |
| **Baseline: Top K Corr (1/K)** | Heurística Ingênua | 11,20% | 0,89 | 0,58 | 17,60% | 16,17% | +5,55 |
| **Top K Corr (QP Otimizado)** | Seleção + Alocação | 9,95% | 0,90 | 0,67 | 20,59% | 14,51% | +3,90 |
| **Lasso (L1 Positivo + QP)** | Regularização $L_1$ | 16,53% | 0,87 | 0,10 | 28,11% | 10,54% | -0,08 |
| **Clustering Hierárquico + Repr.** | Não-Supervisionado | 7,47% | 0,90 | 0,81 | 30,67% | 20,38% | +9,77 |
| **Heurística Gulosa + Swaps** | Busca Combinatória | 4,56% | 0,97 | 0,93 | 29,07% | 12,15% | +1,53 |
| **MIQP Gurobi (Turnover $\lambda=3.0$)** | Otimização Global | **4,52%** | **0,97** | **0,93** | **6,38%** | **12,87%** | **+2,26** |

---

## 🎲 Testes de Robustez e Estabilidade

1. **Modelo Nulo (Monte Carlo com 1.000 Carteiras Aleatórias):**
   * **S&P 100:** O modelo ótimo obteve $\text{TE} = 4,91\%$ vs. mediana aleatória de $11,72\%$. **Superou 100,0% das 1.000 carteiras sorteadas ao acaso ($p < 0,001$)**.
   * **IBOVESPA:** O modelo ótimo obteve $\text{TE} = 4,52\%$ vs. mediana aleatória de $7,89\%$. **Superou 99,0% das carteiras aleatórias ($p = 0,01$)**.
2. **Teste de Estresse (Remoção da Maior Ação):**
   * No Ibovespa, remover a maior ação (ITSA4) elevou o TE de apenas $2,92\%$ para $3,08\%$, demonstrando que o modelo redistribui os pesos em ativos correlatos sem perder a aderência estrutural ($R^2 = 0,96$).

---

## 📁 Estrutura do Repositório

```
index-tracking/
├── data/
│   └── bases_limpas.pkl          # Base pré-processada e auditada (carrega em 0,1s)
├── figures/
│   └── curva_pareto_turnover.png # Gráfico da Fronteira de Pareto e Retorno Líquido
├── notebooks/
│   ├── index_tracking_colab_otimizado.ipynb # Notebook pronto para Google Colab
│   └── experimento_turnover_pareto.ipynb    # Estudo da fronteira de giro
├── results/
│   ├── tabela_pareto_ibov.csv    # Resultados detalhados do Ibovespa
│   └── tabela_pareto_sp100.csv   # Resultados detalhados do S&P 100
├── src/
│   ├── data_loader.py            # Coleta e higienização dos dados do Yahoo Finance
│   ├── optimizer.py              # Formulação MIQP no Gurobi com cardinalidade e giro
│   ├── walk_forward.py           # Simulação sequencial em 20 trimestres OOS
│   └── pareto_experiment.py      # Grade de varredura da curva de Pareto
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🚀 Como Executar Localmente

### 1. Clonar e Instalar Dependências
```bash
git clone https://github.com/podorize/index-tracking.git
cd index-tracking
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Rodar o Experimento da Fronteira de Pareto
```bash
python src/pareto_experiment.py
```

---

## 📚 Referências Bibliográficas

* **Cornuéjols, G.; Tütüncü, R.** (2006). *Optimization Methods in Finance*. Cambridge University Press.
* **Sant’Anna, L.; Filomena, T. P.; Borenstein, D.** (2014). *Index Tracking com Controle do Número de Ativos*. Revista Brasileira de Finanças, v. 12, p. 89-119.
* **Sant’Anna, L.; Filomena, T. P.; Borenstein, D.; Guedes, P.** (2017). *Index tracking with controlled number of assets using a hybrid heuristic combining genetic algorithm and non-linear programming*. Annals of Operations Research, v. 258, p. 449-867.
