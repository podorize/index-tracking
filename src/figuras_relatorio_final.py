"""Figuras do relatório da apresentação final, no mesmo estilo das figuras do relatório original (src/figuras_parte2.py)."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figuras_parte2 import (CORES, CORES_CLARAS, NOMES, TINTA, TINTA_2, APAGADO, FAIXA, GRADE,
                                aplicar_estilo, titulos, salvar, num)
import src.figuras_parte2 as base

SAIDA = Path("/private/tmp/claude-501/-Users-pedroodorize-index-tracking/6e97a2d6-56f7-42bc-a5ae-35345d0758f3/scratchpad")
base.PASTA_FIGURAS = SAIDA
FONTE_ESCOLHA = "Fonte: Yahoo Finance, set/2019 a set/2026. Período de escolha: set/2021 a set/2024. Elaboração própria."
LIMITE = 4.0


def estilo():
    aplicar_estilo()
    mpl.rcParams["font.family"] = [mpl.rcParams["font.family"][0], "DejaVu Sans"]   # λ não existe na Libre Franklin


def fig_erro_por_K():
    estilo()
    d = pd.read_csv("results/selecao_com_erro_padrao.csv")
    Ks = [5, 10, 15, 20, 30]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), sharey=True)
    for ax, (i, indice) in zip(axes, [("S", "S&P100"), ("I", "IBOV")]):
        ax.axvspan(3 - 0.5, 3 + 0.5, color=FAIXA, zorder=0)
        for (J, H, lab, cor, off, lw) in [(252, 63, "1 ano de treino, trimestral", CORES[indice], -0.1, 9),
                                         (504, 63, "2 anos de treino, trimestral", CORES_CLARAS[indice], 0.1, 9)]:
            g = d[(d.ind == i) & (d.J == J) & (d.H == H)].groupby("K").TE.agg(["min", "max"]).reindex(Ks)
            x = np.arange(len(Ks)) + off
            ax.vlines(x, g["min"], g["max"], color=cor, lw=lw, zorder=3, label=lab)
        ax.axhline(LIMITE, color=TINTA_2, lw=1.1, ls=(0, (1, 2)), zorder=1)
        ax.text(len(Ks) - 0.55, LIMITE + 0.12, "limite de 4 p.p.", ha="right", fontsize=10, color=TINTA_2)
        ax.set_xticks(range(len(Ks))); ax.set_xticklabels(Ks); ax.set_xlabel("Número de ações (K)")
        ax.set_title(NOMES[indice]); ax.set_ylim(1.5, 9.5)
        ax.legend(loc="upper right", fontsize=10)
    axes[0].set_ylabel("Erro de rastreamento (% ao ano)")
    titulos(fig, "A partir de 20 ações, o erro fica abaixo de 4 p.p. com qualquer penalidade de giro",
            "Cada barra vai do melhor ao pior valor de λ (a penalidade sobre a negociação) para aquele número de ações.",
            fonte=FONTE_ESCOLHA)
    salvar(fig, "rf_erro_por_K.png")


def curva_lambda(K, J, H, indice):
    nome = "sp100" if indice == "S&P100" else "ibov"
    grade = [0.0, 0.5, 1.5, 3.0, 6.0, 12.0, 25.0, 50.0]
    out = []
    for L in grade:
        r = pd.read_csv(f"results/selecao/K{K}_{nome}_J{J}_H{H}_L{L:g}.csv")
        r = r[r["dias"] == H]
        te = r["TE_trimestre (% a.a.)"].to_numpy()
        g = r.loc[r["numero"] > r["numero"].min(), "giro"].to_numpy() * 252 / H * 100
        out.append((L, np.sqrt((te ** 2).mean()), te.std(ddof=1), g.mean(), g.std(ddof=1)))
    return pd.DataFrame(out, columns=["lam", "te", "te_dp", "giro", "giro_dp"])


def fig_lambda(K, J, H, escolhidos, nome_arq, titulo, subtitulo):
    estilo()
    grade = [0.0, 0.5, 1.5, 3.0, 6.0, 12.0, 25.0, 50.0]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    for ax, indice in zip(axes, ["S&P100", "IBOV"]):
        d = curva_lambda(K, J, H, indice); x = np.arange(len(d))
        ax.axvspan(grade.index(escolhidos[indice]) - 0.5, grade.index(escolhidos[indice]) + 0.5, color=FAIXA, zorder=0)
        ax.errorbar(x - 0.07, d.te, yerr=d.te_dp, fmt="-o", color=CORES[indice], lw=2, ms=6.5, mec="white", mew=1.2, capsize=3, zorder=3)
        ax.axhline(LIMITE, color=TINTA_2, lw=1.1, ls=(0, (1, 2)), zorder=1)
        ax.text(2.5, LIMITE + 0.08, "limite de 4 p.p.", ha="center", fontsize=10, color=TINTA_2)
        ax.set_xticks(x); ax.set_xticklabels([f"{l:g}".replace(".", ",") for l in d.lam]); ax.set_xlabel("λ (penalidade sobre a negociação)")
        ax.set_ylabel("Erro de rastreamento (% ao ano)", color=CORES[indice]); ax.set_title(NOMES[indice])
        ax.set_ylim(0, max(5.6, (d.te + d.te_dp).max() + 0.3))
        ax2 = ax.twinx()
        ax2.plot(x + 0.07, d.giro, "--s", color=TINTA_2, lw=1.6, ms=5.5, mec="white", mew=1, zorder=2)
        ax2.set_ylim(bottom=0); ax2.grid(False); ax2.spines["right"].set_visible(False)
        ax2.set_ylabel("Giro (% ao ano)", color=TINTA_2)
        ax2.tick_params(axis="y", colors=TINTA_2)
    titulos(fig, titulo, subtitulo, fonte=FONTE_ESCOLHA, base_pol=1.0, espaco_w=0.42)
    salvar(fig, nome_arq)


FONTE_TESTE = "Fonte: Yahoo Finance. Teste final: out/2024 a set/2026, período que não influenciou nenhuma escolha. Elaboração própria."
NOMES_MODELOS = {"escolhido": "Modelo escolhido pela regra", "A_1ano_trimestral": "Variante: 1 ano, trimestral", "B_original": "Modelo do relatório original"}
CORTE = pd.Timestamp("2024-10-01")


def rodadas_de_teste(modelo, nome):
    r = pd.read_csv(f"results/teste_final/{modelo}_{nome}.csv"); r["data_fim"] = pd.to_datetime(r["data_fim"]); r["data_inicio"] = pd.to_datetime(r["data_inicio"])
    H = int(r.horizonte.iloc[0]); ultimo = r[(r.dias == H) & (r.data_fim <= CORTE)].numero.max()
    return r[r.numero > ultimo]


def fig_carteira_vs_indice():
    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0))
    for ax, (indice, nome) in zip(axes, [("S&P100", "sp100"), ("IBOV", "ibov")]):
        t = rodadas_de_teste("escolhido", nome)
        datas = [t.data_inicio.iloc[0]] + list(t.data_fim)
        cart = np.concatenate([[100], 100 * np.cumprod(1 + t["retorno_liq_carteira"].to_numpy())])
        idx = np.concatenate([[100], 100 * np.cumprod(1 + t["retorno_indice"].to_numpy())])
        ax.plot(datas, idx, color=APAGADO, lw=2, label="Índice", zorder=2)
        ax.plot(datas, cart, color=CORES[indice], lw=2.4, label="Carteira de 20 ações", zorder=3)
        ax.set_title(NOMES[indice]); ax.set_ylabel("Valor de 100 investidos no início do teste")
        ax.text(datas[-1], cart[-1], f" {num(cart[-1] - 100, 0, True)}%", va="center", fontsize=10.5, color=CORES[indice], fontweight="bold")
        ax.text(datas[-1], idx[-1] + (4 if cart[-1] < idx[-1] else -4), f" {num(idx[-1] - 100, 0, True)}%", va="center", fontsize=10.5, color=TINTA_2)
        ax.legend(loc="upper left"); ax.margins(x=0.12)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10], interval=1)); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    titulos(fig, "No teste, a carteira de 20 ações acompanha o índice nos dois mercados",
            "Modelo escolhido pela regra, montado só com dados anteriores. A carteira quase não negocia (giro de 0% a 0,1% ao ano).", fonte=FONTE_TESTE)
    salvar(fig, "rf_carteira_vs_indice.png")


def fig_erro_no_teste():
    estilo()
    d = pd.read_csv("results/teste_final_resumo.csv"); d = d[d.periodo == "teste"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), sharey=True)
    modelos = ["escolhido", "A_1ano_trimestral", "B_original"]
    for ax, (indice, nome) in zip(axes, [("S&P100", "S&P100"), ("IBOV", "IBOV")]):
        cores = [CORES[indice], CORES_CLARAS[indice], "#bfbdb4"]
        for k, m in enumerate(modelos):
            r = d[(d.modelo == m) & (d.ind == indice)].iloc[0]
            ax.bar(k, r.TE, 0.62, color=cores[k], yerr=r.SE, error_kw=dict(ecolor=TINTA_2, lw=1.4, capsize=4), zorder=3)
            ax.text(k, r.TE - 0.32, num(r.TE, 2), ha="center", fontsize=12, color="white" if k == 0 else TINTA, fontweight="bold")
        ax.axhline(LIMITE, color=TINTA_2, lw=1.1, ls=(0, (1, 2)), zorder=4)
        ax.text(2.48, LIMITE + 0.08, "4 p.p.", ha="right", fontsize=10, color=TINTA_2)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Escolhido\npela regra", "Variante:\n1 ano, trim.", "Original\n(2 anos, trim.)"], fontsize=10.5)
        ax.set_title(NOMES[indice]); ax.set_ylim(0, 6)
    axes[0].set_ylabel("Erro de rastreamento no teste (% ao ano)")
    titulos(fig, "No teste, o modelo escolhido passa no Ibovespa e fica acima de 4 p.p. no S&P 100",
            "K = 20 nos três modelos. A barra é a média; a linha vertical é o erro padrão da média. Linha pontilhada: limite de 4 p.p.", fonte=FONTE_TESTE)
    salvar(fig, "rf_erro_no_teste.png")


if __name__ == "__main__":
    fig_erro_por_K()
    fig_carteira_vs_indice()
    fig_erro_no_teste()
    fig_lambda(20, 504, 21, {"S&P100": 50.0, "IBOV": 25.0}, "rf_lambda_escolhido.png",
               "Quanto maior o λ, menos a carteira negocia, com pouco custo em erro até λ = 25",
               "K = 20, 2 anos de treino, rebalanceamento mensal. Linha colorida: erro, com média e ±1 desvio dos meses. Linha cinza tracejada: giro médio. Faixa: λ escolhido.")
