"""
compare_all_models.py
Executa a comparação completa de todos os algoritmos previstos no Guia Parte 2:
  1. Baselines (Top K peso, Top K correlação 1/K, Top K correlação QP)
  2. Machine Learning (Lasso positivo, Clustering Hierárquico)
  3. Otimização Avançada (Heurística Gulosa + Swaps, MIQP Gurobi com Turnover)
  4. Testes de Robustez (Modelo Nulo Monte Carlo 1.000 carteiras e Remoção de Ativo)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import preparar_bases
from src.features import construir_features
from src.models_benchmark import (
    baseline_top_k_peso,
    baseline_top_k_corr_equal,
    baseline_top_k_corr_otimizado,
    modelo_lasso_k,
    modelo_clustering_k,
    modelo_heuristica_gulosa,
    avaliar_modelo
)
from src.optimizer import ProblemaTracking, resolver_miqp_turnover, ANUAL
from src.robustness import rodar_modelo_nulo_monte_carlo, plotar_modelo_nulo, teste_estresse_remocao_ativo
from src.walk_forward import construir_janelas

PASTA_RESULTADOS = Path("results")
PASTA_FIGURAS = Path("figures")
PASTA_RESULTADOS.mkdir(exist_ok=True)
PASTA_FIGURAS.mkdir(exist_ok=True)

K_PADRAO = 10
CUSTOS = {"S&P100": 0.0005, "IBOV": 0.0015}
LAMBDA_OTIMO = {"S&P100": 6.0, "IBOV": 3.0}


def simular_modelo_walk_forward(nome_indice, bases, nome_metodo, k=K_PADRAO):
    """
    Executa o walk-forward de 20 trimestres para um método específico,
    rastreando turnover e calculando métricas líquidas fora da amostra.
    """
    janelas = construir_janelas(nome_indice, bases)
    b = bases[nome_indice]
    custo_transacao = CUSTOS[nome_indice]
    lam_anual = LAMBDA_OTIMO[nome_indice] if nome_metodo == "MIQP Gurobi (Turnover)" else 0.0
    gamma = (4.0 * lam_anual) / ANUAL if lam_anual > 0 else 0.0

    w_drift_prev = None
    resultados_janelas = []
    series_retornos_carteira = []
    series_retornos_indice = []

    for jan in janelas:
        cands = list(jan["candidatos"])
        R_tr = b["R"].reindex(jan["datas_treino"])[cands].fillna(0.0)
        y_tr = b["y"].reindex(jan["datas_treino"]).fillna(0.0)
        feats = construir_features(R_tr, y_tr)

        # Seleção e ponderação pelo método
        if nome_metodo == "Baseline: Top K Peso":
            pesos = baseline_top_k_peso(cands, k)
        elif nome_metodo == "Baseline: Top K Corr (1/K)":
            pesos = baseline_top_k_corr_equal(R_tr, y_tr, k, feats)
        elif nome_metodo == "Top K Corr (QP Otimizado)":
            pesos = baseline_top_k_corr_otimizado(R_tr, y_tr, k, feats)
        elif nome_metodo == "Lasso (L1 Positivo + QP)":
            pesos = modelo_lasso_k(R_tr, y_tr, k)
        elif nome_metodo == "Clustering Hierárquico + Repr.":
            pesos = modelo_clustering_k(R_tr, y_tr, k, feats)
        elif nome_metodo == "Heurística Gulosa + Swaps":
            prob = ProblemaTracking(R_tr, y_tr)
            pesos = modelo_heuristica_gulosa(prob, k)
        elif nome_metodo == "MIQP Gurobi (Turnover)":
            # Garante compatibilidade de licença demo (<= 93 candidatos)
            if len(cands) > 93:
                ativos_prev = list(w_drift_prev.keys()) if w_drift_prev is not None else []
                corrs = R_tr.corrwith(y_tr).fillna(-1.0).sort_values(ascending=False).index.tolist()
                cands_sel = list(dict.fromkeys(ativos_prev + [t for t in corrs if t not in ativos_prev]))[:93]
                R_tr = R_tr[cands_sel]
            prob = ProblemaTracking(R_tr, y_tr)
            w_drift_dict = w_drift_prev.to_dict() if w_drift_prev is not None else None
            w_sol, _ = resolver_miqp_turnover(prob, K=k, w_drift_dict=w_drift_dict, gamma=gamma, tempo_max=1.5)
            pesos = pd.Series(w_sol, index=R_tr.columns)
            pesos = pesos[pesos > 1e-4]
            pesos = pesos / pesos.sum()
        else:
            raise ValueError(f"Método desconhecido: {nome_metodo}")

        # Cálculo de giro
        if w_drift_prev is None:
            giro_trim = 1.0
        else:
            uni = pesos.index.union(w_drift_prev.index)
            giro_trim = 0.5 * (
                pesos.reindex(uni, fill_value=0.0) - w_drift_prev.reindex(uni, fill_value=0.0)
            ).abs().sum()

        datas_ts = jan["datas_teste"]
        R_ts = b["R"].reindex(datas_ts)[pesos.index]
        y_ts = b["y"].reindex(datas_ts).rename("indice")

        valores = (1 + R_ts.fillna(0.0)).cumprod().mul(pesos, axis=1)
        patrimonio = valores.sum(axis=1)
        rp_diario = patrimonio.pct_change()
        rp_diario.iloc[0] = patrimonio.iloc[0] - 1.0

        par_diario = pd.concat([rp_diario.rename("carteira"), y_ts], axis=1).dropna()
        series_retornos_carteira.append(par_diario["carteira"])
        series_retornos_indice.append(par_diario["indice"])

        r_bruto = float(patrimonio.iloc[-1] - 1.0)
        custo_fric = float(custo_transacao * giro_trim)
        r_liq = r_bruto - custo_fric

        resultados_janelas.append({
            "giro": giro_trim,
            "custo_friccao": custo_fric,
            "retorno_bruto": r_bruto,
            "retorno_liq": r_liq,
            "retorno_indice": float((1 + y_ts.dropna()).prod() - 1.0),
        })

        w_drift_prev = valores.iloc[-1] / valores.iloc[-1].sum()

    df_j = pd.DataFrame(resultados_janelas)
    r_cart_all = pd.concat(series_retornos_carteira)
    r_idx_all = pd.concat(series_retornos_indice)
    dif = r_cart_all - r_idx_all

    te = float(dif.std() * np.sqrt(ANUAL) * 100.0)
    vies = float(dif.mean() * ANUAL * 100.0)
    corr = float(r_cart_all.corr(r_idx_all))

    ss_res = float(((r_idx_all - r_cart_all) ** 2).sum())
    ss_tot = float(((r_idx_all - r_idx_all.mean()) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    giro_medio = float(df_j["giro"].iloc[1:].mean() * 100.0)
    custo_anual = 4.0 * custo_transacao * (giro_medio / 100.0) * 100.0

    ret_acum_liq = float((1 + df_j["retorno_liq"]).prod() - 1.0)
    ret_acum_idx = float((1 + df_j["retorno_indice"]).prod() - 1.0)

    n_anos = len(r_cart_all) / ANUAL
    ret_anual_liq = ((1 + ret_acum_liq) ** (1 / n_anos) - 1.0) * 100.0
    ret_anual_idx = ((1 + ret_acum_idx) ** (1 / n_anos) - 1.0) * 100.0
    alfa_liq = ret_anual_liq - ret_anual_idx
    ir_liq = (alfa_liq / te) if te > 0 else np.nan

    return {
        "Modelo": nome_metodo,
        "K": k,
        "TE (% a.a.)": te,
        "Viés (p.p. a.a.)": vies,
        "Correlação": corr,
        "R²": r2,
        "Giro Médio (% trim)": giro_medio,
        "Custo Transação (% a.a.)": custo_anual,
        "Retorno Anual Líq (%)": ret_anual_liq,
        "Alfa Líquido (p.p.)": alfa_liq,
        "Info Ratio Líq": ir_liq,
    }


def executar_comparativo_completo():
    bases = preparar_bases()
    metodos = [
        "Baseline: Top K Peso",
        "Baseline: Top K Corr (1/K)",
        "Top K Corr (QP Otimizado)",
        "Lasso (L1 Positivo + QP)",
        "Clustering Hierárquico + Repr.",
        "Heurística Gulosa + Swaps",
        "MIQP Gurobi (Turnover)",
    ]

    print("=================================================================")
    print("1. Comparando todos os 7 modelos no S&P 100 (20 trimestres OOS)...")
    print("=================================================================")
    res_sp = []
    for m in metodos:
        print(f"  -> Rodando {m}...")
        r = simular_modelo_walk_forward("S&P100", bases, m, k=K_PADRAO)
        res_sp.append(r)
    df_comp_sp = pd.DataFrame(res_sp)
    df_comp_sp.to_csv(PASTA_RESULTADOS / "tabela_comparativa_modelos_sp100.csv", index=False)

    print("\n=================================================================")
    print("2. Comparando todos os 7 modelos no IBOVESPA (20 trimestres OOS)...")
    print("=================================================================")
    res_ib = []
    for m in metodos:
        print(f"  -> Rodando {m}...")
        r = simular_modelo_walk_forward("IBOV", bases, m, k=K_PADRAO)
        res_ib.append(r)
    df_comp_ib = pd.DataFrame(res_ib)
    df_comp_ib.to_csv(PASTA_RESULTADOS / "tabela_comparativa_modelos_ibov.csv", index=False)

    print("\n=== TABELA COMPARATIVA S&P 100 (K = 10, c = 5 bps) ===")
    print(df_comp_sp[["Modelo", "TE (% a.a.)", "Correlação", "R²", "Giro Médio (% trim)", "Retorno Anual Líq (%)", "Alfa Líquido (p.p.)"]].to_string(index=False))

    print("\n=== TABELA COMPARATIVA IBOVESPA (K = 10, c = 15 bps) ===")
    print(df_comp_ib[["Modelo", "TE (% a.a.)", "Correlação", "R²", "Giro Médio (% trim)", "Retorno Anual Líq (%)", "Alfa Líquido (p.p.)"]].to_string(index=False))

    # Gráfico comparativo de barras (TE e Retorno Líquido)
    gerar_grafico_comparativo_modelos(df_comp_sp, df_comp_ib)

    # Teste de Modelo Nulo (Monte Carlo) no período mais recente
    print("\n=================================================================")
    print("3. Executando Teste de Modelo Nulo (Monte Carlo 1.000 carteiras)...")
    print("=================================================================")
    jan_sp = construir_janelas("S&P100", bases)[-1]
    R_tr_sp = bases["S&P100"]["R"].reindex(jan_sp["datas_treino"])[jan_sp["candidatos"]].fillna(0.0)
    y_tr_sp = bases["S&P100"]["y"].reindex(jan_sp["datas_treino"]).fillna(0.0)
    R_ts_sp = bases["S&P100"]["R"].reindex(jan_sp["datas_teste"])[jan_sp["candidatos"]].fillna(0.0)
    y_ts_sp = bases["S&P100"]["y"].reindex(jan_sp["datas_teste"]).fillna(0.0)

    te_mod_sp = float(df_comp_sp.loc[df_comp_sp["Modelo"] == "MIQP Gurobi (Turnover)", "TE (% a.a.)"].iloc[0])
    res_nulo_sp, te_rand_sp = rodar_modelo_nulo_monte_carlo(R_tr_sp, y_tr_sp, R_ts_sp, y_ts_sp, te_nosso_modelo=te_mod_sp, k=K_PADRAO)

    jan_ib = construir_janelas("IBOV", bases)[-1]
    R_tr_ib = bases["IBOV"]["R"].reindex(jan_ib["datas_treino"])[jan_ib["candidatos"]].fillna(0.0)
    y_tr_ib = bases["IBOV"]["y"].reindex(jan_ib["datas_treino"]).fillna(0.0)
    R_ts_ib = bases["IBOV"]["R"].reindex(jan_ib["datas_teste"])[jan_ib["candidatos"]].fillna(0.0)
    y_ts_ib = bases["IBOV"]["y"].reindex(jan_ib["datas_teste"]).fillna(0.0)

    te_mod_ib = float(df_comp_ib.loc[df_comp_ib["Modelo"] == "MIQP Gurobi (Turnover)", "TE (% a.a.)"].iloc[0])
    res_nulo_ib, te_rand_ib = rodar_modelo_nulo_monte_carlo(R_tr_ib, y_tr_ib, R_ts_ib, y_ts_ib, te_nosso_modelo=te_mod_ib, k=K_PADRAO)

    plotar_modelo_nulo(te_rand_sp, te_mod_sp, te_rand_ib, te_mod_ib)
    print("Modelo Nulo S&P 100:", res_nulo_sp)
    print("Modelo Nulo IBOVESPA:", res_nulo_ib)

    # Teste de Estresse: Remoção da Maior Ação
    print("\n=================================================================")
    print("4. Executando Teste de Estresse (Remoção da Maior Ação)...")
    print("=================================================================")
    estresse_sp = teste_estresse_remocao_ativo(R_tr_sp, y_tr_sp, R_ts_sp, y_ts_sp, k=K_PADRAO)
    estresse_ib = teste_estresse_remocao_ativo(R_tr_ib, y_tr_ib, R_ts_ib, y_ts_ib, k=K_PADRAO)
    estresse_sp.to_csv(PASTA_RESULTADOS / "estresse_remocao_maior_acao_sp100.csv", index=False)
    estresse_ib.to_csv(PASTA_RESULTADOS / "estresse_remocao_maior_acao_ibov.csv", index=False)
    print("Estresse S&P 100:\n", estresse_sp[["Ativo Removido", "Tracking Error (% a.a.)", "R²", "Correlação"]])
    print("Estresse IBOVESPA:\n", estresse_ib[["Ativo Removido", "Tracking Error (% a.a.)", "R²", "Correlação"]])


def gerar_grafico_comparativo_modelos(df_sp, df_ib):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    modelos = df_sp["Modelo"].tolist()
    y_pos = np.arange(len(modelos))

    # Painel 1: Tracking Error (% a.a.)
    ax1 = axes[0]
    bars_sp = ax1.barh(y_pos - 0.2, df_sp["TE (% a.a.)"], height=0.38, color="#1f77b4", label="S&P 100")
    bars_ib = ax1.barh(y_pos + 0.2, df_ib["TE (% a.a.)"], height=0.38, color="#2ca02c", label="IBOVESPA")

    for bar in bars_sp:
        w = bar.get_width()
        ax1.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=8.5, color="#1f77b4")
    for bar in bars_ib:
        w = bar.get_width()
        ax1.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=8.5, color="#2ca02c")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(modelos, fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Tracking Error Fora da Amostra (% a.a.) - Menor é Melhor", fontsize=10)
    ax1.set_title("Comparativo de Tracking Error entre Modelos (K = 10)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9.5, loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="x")

    # Painel 2: Giro Médio Trimestral (% da Carteira)
    ax2 = axes[1]
    bars_g_sp = ax2.barh(y_pos - 0.2, df_sp["Giro Médio (% trim)"], height=0.38, color="#1f77b4", label="S&P 100")
    bars_g_ib = ax2.barh(y_pos + 0.2, df_ib["Giro Médio (% trim)"], height=0.38, color="#2ca02c", label="IBOVESPA")

    for bar in bars_g_sp:
        w = bar.get_width()
        ax2.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5, color="#1f77b4")
    for bar in bars_g_ib:
        w = bar.get_width()
        ax2.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5, color="#2ca02c")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Giro Médio Trimestral (% da Carteira) - Menor é Melhor", fontsize=10)
    ax2.set_title("Turnover Médio por Rebalanceamento Trimestral", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9.5, loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.tight_layout()
    caminho = PASTA_FIGURAS / "comparacao_todos_modelos.png"
    plt.savefig(caminho)
    plt.close()
    print(f"Gráfico comparativo salvo em {caminho}")


if __name__ == "__main__":
    executar_comparativo_completo()
