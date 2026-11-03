"""
compare_all_models.py
Comparação completa dos modelos do Guia Parte 2 (Etapa 13 do relatório), com K e janela de treino parametrizados:
  1. Baselines (Top K peso, Top K correlação 1/K, Top K correlação com pesos otimizados)
  2. Machine learning (Lasso positivo, clustering hierárquico)
  3. Otimização: heurística gulosa + trocas (a da Etapa 11) e heurística com controle de giro (o modelo final)
  4. Testes de robustez (modelo nulo com 1.000 carteiras aleatórias e remoção da maior ação)

Cada índice roda em um processo próprio e grava seus arquivos; `montar` combina os dois índices (tabelas por regime, gráficos).
Os arquivos levam o sufixo `_K{K}` e, se a janela não for de 2 anos, `_J{janela}`.
"""

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import preparar_bases
from src.features import construir_features
from src.heuristica_giro import heuristica_giro
from src.models_benchmark import (
    baseline_top_k_peso,
    baseline_top_k_corr_equal,
    baseline_top_k_corr_otimizado,
    modelo_lasso_k,
    modelo_clustering_k,
)
from src.optimizer import ProblemaTracking, ANUAL
from src.robustness import rodar_modelo_nulo_walk_forward, plotar_modelo_nulo, teste_estresse_remocao_ativo
from src.validacao_lambda import lambda_escolhido
from src.walk_forward import construir_janelas, simular_walk_forward

PASTA_RESULTADOS = Path("results")
PASTA_FIGURAS = Path("figures")
PASTA_RESULTADOS.mkdir(exist_ok=True)
PASTA_FIGURAS.mkdir(exist_ok=True)

K_PADRAO = 20
JANELA_PADRAO = 504
TEMPO_GUROBI = 10.0
CUSTOS = {"S&P100": 0.0005, "IBOV": 0.0015}
NOME_ARQ = {"S&P100": "sp100", "IBOV": "ibov"}

METODOS = [
    "Baseline: Top K Peso",
    "Baseline: Top K Corr (1/K)",
    "Top K Corr (QP Otimizado)",
    "Lasso (L1 Positivo + QP)",
    "Clustering Hierárquico + Repr.",
    "Heurística Gulosa + Swaps",
    "Heurística com controle de giro",
]
METODOS_GIRO = {"Heurística com controle de giro": "heuristica"}
MODELO_FINAL = "Heurística com controle de giro"

# Regimes do período fora da amostra (Seção 5.5 do Guia). A Covid (fev-abr/2020) fica antes
# do primeiro trimestre de teste (set/2021) e só aparece na Parte 1.
REGIMES_OOS = {
    "Fim de 2021": ("2021-09-01", "2021-12-31"),
    "Alta de juros 2022": ("2022-01-01", "2022-12-31"),
    "Calmo 2023-2024": ("2023-01-01", "2024-12-31"),
    "2025-2026": ("2025-01-01", "2026-09-30"),
    "Total": ("2021-09-01", "2026-09-30"),
}


def sufixo(K, janela):
    return f"_K{K}" + ("" if janela == JANELA_PADRAO else f"_J{janela}")


def tabela_te_por_regime(series):
    """TE anualizado de cada modelo em cada regime, a partir das séries diárias fora da amostra."""
    linhas = []
    for (indice, modelo), s in series.items():
        for regime, (inicio, fim) in REGIMES_OOS.items():
            trecho = s.loc[inicio:fim]
            dif = trecho["carteira"] - trecho["indice"]
            linhas.append({
                "indice": indice,
                "modelo": modelo,
                "regime": regime,
                "pregões": len(trecho),
                "TE (% a.a.)": float(dif.std() * np.sqrt(ANUAL) * 100.0),
                "TD (p.p. a.a.)": float(dif.mean() * ANUAL * 100.0),
                "Volatilidade do índice (% a.a.)": float(trecho["indice"].std() * np.sqrt(ANUAL) * 100.0),
            })
    return pd.DataFrame(linhas)


def _metricas(nome, k, serie_c, serie_i, df_j, custo_transacao):
    """Métricas da Etapa 13 a partir das séries diárias e do resultado por rodada (giro e retornos trimestrais)."""
    dif = serie_c - serie_i
    te = float(dif.std() * np.sqrt(ANUAL) * 100.0)
    vies = float(dif.mean() * ANUAL * 100.0)
    corr = float(serie_c.corr(serie_i))
    ss_res = float(((serie_i - serie_c) ** 2).sum())
    ss_tot = float(((serie_i - serie_i.mean()) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    giro_medio = float(df_j["giro"].iloc[1:].mean() * 100.0)
    custo_anual = 4.0 * custo_transacao * (giro_medio / 100.0) * 100.0
    ret_acum_liq = float((1 + df_j["retorno_liq"]).prod() - 1.0)
    ret_acum_idx = float((1 + df_j["retorno_indice"]).prod() - 1.0)
    n_anos = len(serie_c) / ANUAL
    ret_anual_liq = ((1 + ret_acum_liq) ** (1 / n_anos) - 1.0) * 100.0
    ret_anual_idx = ((1 + ret_acum_idx) ** (1 / n_anos) - 1.0) * 100.0
    alfa_liq = ret_anual_liq - ret_anual_idx
    return {
        "Modelo": nome, "K": k, "TE (% a.a.)": te, "TD (p.p. a.a.)": vies, "Correlação": corr, "R²": r2,
        "Giro Médio (% trim)": giro_medio, "Custo Transação (% a.a.)": custo_anual,
        "Retorno Anual Líq (%)": ret_anual_liq, "Alfa Líquido (p.p.)": alfa_liq,
        "Info Ratio Líq": (alfa_liq / te) if te > 0 else np.nan,
    }


def simular_modelo(nome_indice, bases, nome_metodo, k, janela, lam_anual):
    """Walk-forward de 20 trimestres de um método, com giro e custos medidos fora da amostra."""
    jan_ref = construir_janelas(nome_indice, bases)
    inicio, fim = jan_ref[0]["datas_teste"][0], jan_ref[-1]["datas_teste"][-1]
    custo = CUSTOS[nome_indice]

    if nome_metodo in METODOS_GIRO:
        # Mesmo motor da Etapa 12: heurística com penalidade
        metricas, df, serie = simular_walk_forward(
            nome_indice=nome_indice, bases=bases, K=k, lambda_anual=lam_anual, custo_transacao=custo, tempo_max=TEMPO_GUROBI,
            janela_treino=janela, horizonte_teste=63, inicio_teste=inicio, fim_teste=fim,
            metodo=METODOS_GIRO[nome_metodo], retornar_series=True)
        df_j = df.rename(columns={"retorno_liq_carteira": "retorno_liq"})
        return serie, _metricas(nome_metodo, k, serie["carteira"], serie["indice"], df_j, custo), df

    janelas = construir_janelas(nome_indice, bases, J=janela, H=63, inicio_teste=inicio, fim_teste=fim)
    b = bases[nome_indice]
    w_drift_prev = None
    linhas, series_c, series_i = [], [], []
    for jan in janelas:
        cands = list(jan["candidatos"])
        R_tr = b["R"].reindex(jan["datas_treino"])[cands].fillna(0.0)
        y_tr = b["y"].reindex(jan["datas_treino"]).fillna(0.0)
        feats = construir_features(R_tr, y_tr)

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
            # a heurística da Etapa 11 (gulosa a partir do zero e melhor troca), sem penalidade de giro
            prob = ProblemaTracking(R_tr, y_tr)
            w, sel, _ = heuristica_giro(prob, k, None, 0.0, partidas=("vazio",))
            pesos = pd.Series(w[sel], index=[prob.tickers[i] for i in sel])
            pesos = pesos / pesos.sum()
        else:
            raise ValueError(f"Método desconhecido: {nome_metodo}")

        if w_drift_prev is None:
            giro_trim = 1.0
        else:
            uni = pesos.index.union(w_drift_prev.index)
            giro_trim = 0.5 * (pesos.reindex(uni, fill_value=0.0) - w_drift_prev.reindex(uni, fill_value=0.0)).abs().sum()

        R_ts = b["R"].reindex(jan["datas_teste"])[pesos.index]
        y_ts = b["y"].reindex(jan["datas_teste"]).rename("indice")
        valores = (1 + R_ts.fillna(0.0)).cumprod().mul(pesos, axis=1)
        patrimonio = valores.sum(axis=1)
        rp = patrimonio.pct_change()
        rp.iloc[0] = patrimonio.iloc[0] - 1.0
        par = pd.concat([rp.rename("carteira"), y_ts], axis=1).dropna()
        series_c.append(par["carteira"])
        series_i.append(par["indice"])
        r_bruto = float(patrimonio.iloc[-1] - 1.0)
        linhas.append({"numero": jan["numero"], "giro": giro_trim, "retorno_liq": r_bruto - custo * giro_trim,
                       "retorno_indice": float((1 + y_ts.dropna()).prod() - 1.0)})
        w_drift_prev = valores.iloc[-1] / valores.iloc[-1].sum()

    df_j = pd.DataFrame(linhas)
    sc, si = pd.concat(series_c), pd.concat(series_i)
    serie = pd.DataFrame({"carteira": sc, "indice": si})
    return serie, _metricas(nome_metodo, k, sc, si, df_j, custo), df_j


def executar_indice(indice, K=K_PADRAO, janela=JANELA_PADRAO, bases=None, robustez=True):
    bases = bases if bases is not None else preparar_bases()
    suf = f"{sufixo(K, janela)}_{NOME_ARQ[indice]}"
    lam = lambda_escolhido(K)[indice]
    print(f"=== {indice}: K = {K}, janela {janela}, λ = {lam:g} ===", flush=True)
    series, linhas, rodadas = {}, [], []
    for m in METODOS:
        t0 = time.time()
        serie, r, df = simular_modelo(indice, bases, m, K, janela, lam)
        series[m] = serie
        linhas.append(r)
        rodadas.append(df.assign(modelo=m, indice=indice))
        print(f"  {m}: TE {r['TE (% a.a.)']:.2f}%, giro {r['Giro Médio (% trim)']:.1f}% ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(linhas).to_csv(PASTA_RESULTADOS / f"tabela_comparativa_modelos{suf}.csv", index=False)
    pd.concat([s.assign(indice_nome=indice, modelo=m) for m, s in series.items()]).rename_axis("data").to_csv(
        PASTA_RESULTADOS / f"series_oos_modelos{suf}.csv")
    pd.concat(rodadas, ignore_index=True).to_csv(PASTA_RESULTADOS / f"rodadas_modelos{suf}.csv", index=False)

    if not robustez:
        return
    # modelo nulo: mesmos trimestres, K ações sorteadas, pesos otimizados no treino
    te_modelo = float(pd.DataFrame(linhas).query("Modelo == 'Heurística com controle de giro'")["TE (% a.a.)"].iloc[0])
    t0 = time.time()
    resumo, te_aleatorios = rodar_modelo_nulo_walk_forward(indice, bases, te_nosso_modelo=te_modelo, k=K)
    pd.DataFrame([{"indice": indice, **resumo}]).to_csv(PASTA_RESULTADOS / f"modelo_nulo_monte_carlo{suf}.csv", index=False)
    np.save(PASTA_RESULTADOS / f"modelo_nulo_te{suf}.npy", te_aleatorios)
    print(f"  modelo nulo: {resumo} ({time.time() - t0:.0f} s)", flush=True)
    # estresse: remover a ação de maior peso no último trimestre
    jan = construir_janelas(indice, bases)[-1]
    R_tr = bases[indice]["R"].reindex(jan["datas_treino"])[jan["candidatos"]].fillna(0.0)
    y_tr = bases[indice]["y"].reindex(jan["datas_treino"]).fillna(0.0)
    R_ts = bases[indice]["R"].reindex(jan["datas_teste"])[jan["candidatos"]].fillna(0.0)
    y_ts = bases[indice]["y"].reindex(jan["datas_teste"]).fillna(0.0)
    estresse = teste_estresse_remocao_ativo(R_tr, y_tr, R_ts, y_ts, k=K)
    estresse.to_csv(PASTA_RESULTADOS / f"estresse_remocao_maior_acao_{NOME_ARQ[indice]}{suf[:suf.rfind('_')]}.csv", index=False)


def executar_sensibilidade_k(indice, K=K_PADRAO, bases=None):
    """Heurística (sem penalidade) com K − 5, K e K + 5 ações, para o teste de robustez ao tamanho da carteira."""
    bases = bases if bases is not None else preparar_bases()
    linhas = []
    for k in (K - 5, K, K + 5):
        t0 = time.time()
        _, r, _ = simular_modelo(indice, bases, "Heurística Gulosa + Swaps", k, JANELA_PADRAO, 0.0)
        linhas.append(r)
        print(f"  {indice} K={k}: TE {r['TE (% a.a.)']:.2f}% ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(linhas).to_csv(PASTA_RESULTADOS / f"robustez_tamanho_K{K}_{NOME_ARQ[indice]}.csv", index=False)


def reexecutar_giro(indice, K=K_PADRAO, janela=JANELA_PADRAO, bases=None):
    """Refaz só os modelos com controle de giro (por exemplo, depois de mudar o λ escolhido) e atualiza os arquivos do índice."""
    bases = bases if bases is not None else preparar_bases()
    suf = f"{sufixo(K, janela)}_{NOME_ARQ[indice]}"
    lam = lambda_escolhido(K)[indice]
    print(f"=== {indice}: K = {K}, só os modelos com giro, λ = {lam:g} ===", flush=True)
    tabela = pd.read_csv(PASTA_RESULTADOS / f"tabela_comparativa_modelos{suf}.csv")
    series = pd.read_csv(PASTA_RESULTADOS / f"series_oos_modelos{suf}.csv", index_col="data", parse_dates=True)
    rodadas = pd.read_csv(PASTA_RESULTADOS / f"rodadas_modelos{suf}.csv")
    for m in METODOS_GIRO:
        t0 = time.time()
        serie, r, df = simular_modelo(indice, bases, m, K, janela, lam)
        tabela = tabela[tabela["Modelo"] != m]
        tabela = pd.concat([tabela, pd.DataFrame([r])], ignore_index=True)
        series = pd.concat([series[series["modelo"] != m], serie.assign(indice_nome=indice, modelo=m)])
        rodadas = pd.concat([rodadas[rodadas["modelo"] != m], df.assign(modelo=m, indice=indice)], ignore_index=True)
        print(f"  {m}: TE {r['TE (% a.a.)']:.2f}%, giro {r['Giro Médio (% trim)']:.1f}% ({time.time() - t0:.0f} s)", flush=True)
    ordem = {m: i for i, m in enumerate(METODOS)}
    tabela["_o"] = tabela["Modelo"].map(ordem)
    tabela.sort_values("_o").drop(columns="_o").to_csv(PASTA_RESULTADOS / f"tabela_comparativa_modelos{suf}.csv", index=False)
    series.sort_index().rename_axis("data").to_csv(PASTA_RESULTADOS / f"series_oos_modelos{suf}.csv")
    rodadas.to_csv(PASTA_RESULTADOS / f"rodadas_modelos{suf}.csv", index=False)
    # o modelo nulo compara o TE do modelo final com as carteiras aleatórias: atualiza a porcentagem superada
    npy = PASTA_RESULTADOS / f"modelo_nulo_te{suf}.npy"
    arq_nulo = PASTA_RESULTADOS / f"modelo_nulo_monte_carlo{suf}.csv"
    if npy.exists() and arq_nulo.exists():
        te_nulo = np.load(npy)
        te_modelo = float(tabela.loc[tabela["Modelo"] == MODELO_FINAL, "TE (% a.a.)"].iloc[0])
        nulo = pd.read_csv(arq_nulo)
        nulo["TE Nosso Modelo (% a.a.)"] = te_modelo
        nulo["Percentual Superado (%)"] = 100 * float((te_nulo > te_modelo).mean())
        nulo.to_csv(arq_nulo, index=False)


def montar(K=K_PADRAO, janela=JANELA_PADRAO, robustez=True):
    """Combina os dois índices: séries, TE por regime e gráficos."""
    suf = sufixo(K, janela)
    series, comp = {}, {}
    for indice in ("S&P100", "IBOV"):
        s = pd.read_csv(PASTA_RESULTADOS / f"series_oos_modelos{suf}_{NOME_ARQ[indice]}.csv", index_col="data", parse_dates=True)
        for m, g in s.groupby("modelo"):
            series[(indice, m)] = g[["carteira", "indice"]]
        comp[indice] = pd.read_csv(PASTA_RESULTADOS / f"tabela_comparativa_modelos{suf}_{NOME_ARQ[indice]}.csv")
    pd.concat([s.assign(indice_nome=i, modelo=m) for (i, m), s in series.items()]).rename_axis("data").to_csv(
        PASTA_RESULTADOS / f"series_oos_modelos{suf}.csv")
    tabela_te_por_regime(series).to_csv(PASTA_RESULTADOS / f"te_por_regime_modelos{suf}.csv", index=False)
    gerar_grafico_comparativo_modelos(comp["S&P100"], comp["IBOV"], K, f"comparacao_todos_modelos{suf}.png")
    if robustez:
        nulo = {i: np.load(PASTA_RESULTADOS / f"modelo_nulo_te{suf}_{NOME_ARQ[i]}.npy") for i in comp}
        te = {i: float(comp[i].query("Modelo == 'Heurística com controle de giro'")["TE (% a.a.)"].iloc[0]) for i in comp}
        plotar_modelo_nulo(nulo["S&P100"], te["S&P100"], nulo["IBOV"], te["IBOV"], K=K, nome=f"modelo_nulo_monte_carlo{suf}.png")
        pd.concat([pd.read_csv(PASTA_RESULTADOS / f"modelo_nulo_monte_carlo{suf}_{NOME_ARQ[i]}.csv") for i in comp]).to_csv(
            PASTA_RESULTADOS / f"modelo_nulo_monte_carlo{suf}.csv", index=False)


def gerar_grafico_comparativo_modelos(df_sp, df_ib, K, nome):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), dpi=150)
    modelos = df_sp["Modelo"].tolist()
    y_pos = np.arange(len(modelos))
    for ax, col, titulo, xlabel, casas in (
            (axes[0], "TE (% a.a.)", f"Comparativo de Tracking Error entre Modelos (K = {K})", "Tracking Error Fora da Amostra (% a.a.)", 2),
            (axes[1], "Giro Médio (% trim)", "Turnover Médio por Rebalanceamento Trimestral", "Giro Médio Trimestral (% da Carteira)", 1)):
        b1 = ax.barh(y_pos - 0.2, df_sp[col], height=0.38, color="#1f77b4", label="S&P 100")
        b2 = ax.barh(y_pos + 0.2, df_ib[col], height=0.38, color="#2ca02c", label="IBOVESPA")
        for barras, cor in ((b1, "#1f77b4"), (b2, "#2ca02c")):
            for bar in barras:
                w = bar.get_width()
                ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{w:.{casas}f}%", va="center", fontsize=8.5, color=cor)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(modelos if ax is axes[0] else [], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel + " - Menor é Melhor", fontsize=10)
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9.5, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / nome)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=K_PADRAO)
    ap.add_argument("--janela", type=int, default=JANELA_PADRAO)
    ap.add_argument("--indice", default=None)
    ap.add_argument("--montar", action="store_true")
    ap.add_argument("--sensibilidade-k", action="store_true")
    ap.add_argument("--so-giro", action="store_true")
    ap.add_argument("--sem-robustez", action="store_true")
    a = ap.parse_args()
    if a.so_giro:
        reexecutar_giro(a.indice, K=a.K)
    elif a.sensibilidade_k:
        executar_sensibilidade_k(a.indice, K=a.K)
    elif a.montar:
        montar(a.K, a.janela, robustez=not a.sem_robustez)
    else:
        executar_indice(a.indice, K=a.K, janela=a.janela, robustez=not a.sem_robustez)
