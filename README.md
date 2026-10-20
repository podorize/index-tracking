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

## 📊 Principais Resultados (Fronteira de Pareto e Custos Reais)

Avaliação em 20 trimestres fora da amostra com $K = 10$, considerando custos de transação explícitos:
- **B3 (Ibovespa):** $c = 15\text{ bps}$ ($0,0015$)
- **EUA (S&P 100):** $c = 5\text{ bps}$ ($0,0005$)

### IBOVESPA ($K = 10$, $c = 15\text{ bps}$)
| $\lambda_{\text{anual}}$ | Tracking Error (% a.a.) | Giro Médio (% trim) | Retorno Anual Líquido (%) | Alfa Líquido (p.p. a.a.) |
| :---: | :---: | :---: | :---: | :---: |
| **0.0 (sem penalidade)** | 4,50% | 21,51% | 13,32% | +2,70 |
| **1.5** | 4,46% | 9,09% | 13,06% | +2,45 |
| **3.0 ⭐ (Ótimo)** | **4,37%** | **5,56%** | **12,42%** | **+1,81** |
| **6.0** | 4,50% | 3,79% | 13,32% | +2,70 |
| **25.0** | 4,74% | 1,03% | 12,53% | +1,91 |

> **Descoberta:** No Ibovespa, $\lambda = 3.0$ atuou como um **regularizador estatístico**: o Tracking Error caiu de 4,50% para **4,37%**, enquanto o giro trimestral despencou **74%** (de 21,5% para 5,56%).

### S&P 100 ($K = 10$, $c = 5\text{ bps}$)
| $\lambda_{\text{anual}}$ | Tracking Error (% a.a.) | Giro Médio (% trim) | Retorno Anual Líquido (%) | Alfa Líquido (p.p. a.a.) |
| :---: | :---: | :---: | :---: | :---: |
| **0.0 (sem penalidade)** | 4,88% | 26,36% | 17,17% | +3,23 |
| **3.0** | 5,11% | 7,14% | 17,79% | +3,85 |
| **6.0 ⭐ (Ótimo)** | **4,93%** | **4,37%** | **18,20%** | **+4,26** |
| **25.0** | 4,98% | 1,24% | 16,50% | +2,55 |

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
