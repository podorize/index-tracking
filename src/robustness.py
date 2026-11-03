"""
robustness.py
Testes de Robustez e Estabilidade para Index Tracking (seção 13.7 do notebook):
  1. Modelo Nulo (Monte Carlo com 1.000 carteiras aleatórias)
  2. Teste de Estresse: Remoção do Ativo Dominante
  3. Desempenho por Regimes de Mercado (COVID, Juros 2022, Calmo)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models_benchmark import resolver_pesos_qp, avaliar_modelo
from src.optimizer import ProblemaTracking, ANUAL

PASTA_FIGURAS = Path("figures")
PASTA_RESULTADOS = Path("results")
PASTA_FIGURAS.mkdir(exist_ok=True)
PASTA_RESULTADOS.mkdir(exist_ok=True)


def rodar_modelo_nulo_monte_carlo(
    R_treino, y_treino, R_teste, y_teste, te_nosso_modelo, k=10, n_simulacoes=1000, seed=42
):
    """
    Seção 5.4 do Guia: Sorteia N carteiras aleatórias de K ativos, otimiza os pesos
    no treino e calcula o Tracking Error fora da amostra (teste).
    Mede a significância estatística do modelo de otimização frente ao acaso.
    """
    rng = np.random.default_rng(seed)
    tickers_disponiveis = list(R_treino.columns)
    te_aleatorios = []

    for _ in range(n_simulacoes):
        escolhidas = rng.choice(tickers_disponiveis, size=k, replace=False)
        w_rand = resolver_pesos_qp(R_treino, y_treino, escolhidas)
        sub_teste = R_teste[w_rand.index].fillna(0.0)
        r_cart = sub_teste @ w_rand
        dif = r_cart - y_teste
        te = float(dif.std() * np.sqrt(ANUAL) * 100.0)
        te_aleatorios.append(te)

    te_aleatorios = np.array(te_aleatorios)
    pct_superado = float((te_aleatorios > te_nosso_modelo).mean() * 100.0)
    te_mediano = float(np.median(te_aleatorios))
    te_pior = float(np.max(te_aleatorios))
    te_melhor = float(np.min(te_aleatorios))

    resumo_nulo = {
        "K": k,
        "n_simulacoes": n_simulacoes,
        "TE Nosso Modelo (% a.a.)": te_nosso_modelo,
        "TE Aleatório Mediano (% a.a.)": te_mediano,
        "TE Aleatório Mínimo (% a.a.)": te_melhor,
        "TE Aleatório Máximo (% a.a.)": te_pior,
        "Percentual Superado (%)": pct_superado,
    }

    return resumo_nulo, te_aleatorios


def rodar_modelo_nulo_walk_forward(nome_indice, bases, te_nosso_modelo, k=10, n_simulacoes=1000, seed=42):
    """
    Modelo nulo comparável ao walk-forward: cada carteira aleatória percorre os MESMOS 20 trimestres
    fora da amostra do modelo. A cada rebalanceamento sorteia K candidatas, otimiza os pesos no treino
    (mesmo QP dos demais modelos) e mantém a carteira sem rebalancear no trimestre de teste.
    O TE é agregado sobre todos os dias de teste, exatamente como o TE do modelo.
    """
    from scipy.optimize import nnls
    from src.models_benchmark import PENALIDADE_SOMA
    from src.walk_forward import construir_janelas

    b = bases[nome_indice]
    trimestres = []
    for jan in construir_janelas(nome_indice, bases):
        cands = list(jan["candidatos"])
        X = 100.0 * b["R"].reindex(jan["datas_treino"])[cands].fillna(0.0).to_numpy()
        a = 100.0 * b["y"].reindex(jan["datas_treino"]).fillna(0.0).to_numpy()
        R_ts = b["R"].reindex(jan["datas_teste"])[cands].fillna(0.0).to_numpy()
        y_ts = b["y"].reindex(jan["datas_teste"]).to_numpy()
        trimestres.append((len(cands), X, a, np.cumprod(1.0 + R_ts, axis=0), y_ts))

    rng = np.random.default_rng(seed)
    te_aleatorios = np.empty(n_simulacoes)
    for s in range(n_simulacoes):
        difs = []
        for n, X, a, G, y_ts in trimestres:
            idx = rng.choice(n, size=k, replace=False)
            A = np.vstack([X[:, idx], PENALIDADE_SOMA * np.ones(k)])
            w, _ = nnls(A, np.r_[a, PENALIDADE_SOMA])
            w = w / w.sum() if w.sum() > 0 else np.ones(k) / k
            patrimonio = G[:, idx] @ w
            r_cart = patrimonio / np.r_[1.0, patrimonio[:-1]] - 1.0
            dif = r_cart - y_ts
            difs.append(dif[~np.isnan(dif)])
        te_aleatorios[s] = np.concatenate(difs).std(ddof=1) * np.sqrt(ANUAL) * 100.0

    resumo_nulo = {
        "K": k,
        "n_simulacoes": n_simulacoes,
        "trimestres": len(trimestres),
        "TE Nosso Modelo (% a.a.)": te_nosso_modelo,
        "TE Aleatório Mediano (% a.a.)": float(np.median(te_aleatorios)),
        "TE Aleatório Mínimo (% a.a.)": float(te_aleatorios.min()),
        "TE Aleatório Máximo (% a.a.)": float(te_aleatorios.max()),
        "Percentual Superado (%)": float((te_aleatorios > te_nosso_modelo).mean() * 100.0),
    }
    return resumo_nulo, te_aleatorios


def plotar_modelo_nulo(te_aleatorios_sp, te_modelo_sp, te_aleatorios_ib, te_modelo_ib, K=10, nome="modelo_nulo_monte_carlo.png"):
    """Gera o gráfico comparativo da distribuição do modelo nulo vs. modelos ótimos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Painel 1: S&P 100
    ax1 = axes[0]
    ax1.hist(te_aleatorios_sp, bins=40, color="#1f77b4", alpha=0.6, edgecolor="white", density=True)
    ax1.axvline(te_modelo_sp, color="red", lw=2.5, linestyle="--",
                label=f"Nosso Modelo ({te_modelo_sp:.2f}% a.a.)\nSupera {(te_aleatorios_sp > te_modelo_sp).mean()*100:.1f}%")
    ax1.set_title(f"S&P 100: Modelo Ótimo vs. 1.000 Carteiras Aleatórias (K = {K})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Tracking Error Fora da Amostra (% a.a.)", fontsize=10)
    ax1.set_ylabel("Densidade de Probabilidade", fontsize=10)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Painel 2: IBOVESPA
    ax2 = axes[1]
    ax2.hist(te_aleatorios_ib, bins=40, color="#2ca02c", alpha=0.6, edgecolor="white", density=True)
    ax2.axvline(te_modelo_ib, color="red", lw=2.5, linestyle="--",
                label=f"Nosso Modelo ({te_modelo_ib:.2f}% a.a.)\nSupera {(te_aleatorios_ib > te_modelo_ib).mean()*100:.1f}%")
    ax2.set_title(f"IBOVESPA: Modelo Ótimo vs. 1.000 Carteiras Aleatórias (K = {K})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Tracking Error Fora da Amostra (% a.a.)", fontsize=10)
    ax2.set_ylabel("Densidade de Probabilidade", fontsize=10)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    caminho = PASTA_FIGURAS / nome
    plt.savefig(caminho)
    plt.close()
    print(f"Gráfico do modelo nulo salvo em {caminho}")


def teste_estresse_remocao_ativo(R_treino, y_treino, R_teste, y_teste, k=10, ticker_remover=None):
    """
    Seção 5.4 do Guia: retira a ação de maior peso e reotimiza, com a heurística (determinística, a mesma da Etapa 11).
    Verifica se a carteira depende exclusivamente de um ativo ou se diversifica de forma resiliente.
    """
    from src.heuristica_giro import heuristica_giro

    def carteira(R):
        prob = ProblemaTracking(R, y_treino)
        w, sel, _ = heuristica_giro(prob, k, None, 0.0, partidas=("vazio",))
        pesos = pd.Series(w[sel], index=[prob.tickers[i] for i in sel])
        return pesos / pesos.sum()

    pesos_comp = carteira(R_treino)
    m_comp = avaliar_modelo(pesos_comp, R_teste, y_teste, "Completo")
    if ticker_remover is None:
        ticker_remover = pesos_comp.idxmax()
    candidatos_sem = [t for t in R_treino.columns if t != ticker_remover]
    pesos_sem = carteira(R_treino[candidatos_sem])
    m_sem = avaliar_modelo(pesos_sem, R_teste, y_teste, f"Sem {ticker_remover}")

    resumo = pd.DataFrame([m_comp, m_sem])
    resumo["Ativo Removido"] = ["Nenhum", ticker_remover]
    return resumo
