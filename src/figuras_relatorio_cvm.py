"""Figuras do relatório com o critério da CVM (CRITERIOS_SELECAO_FINAL.md, revisão 2), no estilo de src/figuras_parte2.py."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figuras_parte2 import CORES, CORES_CLARAS, NOMES, TINTA, TINTA_2, APAGADO, FAIXA, titulos, salvar, num
from src.figuras_relatorio_final import estilo, SAIDA, FONTE_ESCOLHA, FONTE_TESTE, CORTE, rodadas_de_teste
from src.criterio_cvm import janelas
import src.figuras_parte2 as base

base.PASTA_FIGURAS = SAIDA
TETO = 0.20
GRADE_L = [0.0, 0.5, 1.5, 3.0, 6.0, 12.0, 25.0, 50.0]


def dados_escolha():
    d = pd.read_csv("results/selecao_diferenca_acumulada.csv")
    d["f12"] = d.viol_12m / d.jan_12m
    d["f60"] = d.viol_60d / d.jan_60d
    d["pior"] = d[["f12", "f60"]].max(axis=1)
    return d


def fig_janelas_por_K():
    estilo()
    d = dados_escolha()
    Ks = [5, 10, 15, 20, 30]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), sharey=True)
    for ax, indice in zip(axes, ["S&P100", "IBOV"]):
        g = d[d.indice == indice].groupby("K").pior.min().reindex(Ks) * 100
        ax.axvspan(3 - 0.5, 3 + 0.5, color=FAIXA, zorder=0)
        ax.bar(range(len(Ks)), g, 0.62, color=CORES[indice], zorder=3)
        for k, v in enumerate(g):
            ax.text(k, v + 1.2, f"{num(v, 0)}%", ha="center", fontsize=11.5, color=TINTA, fontweight="bold")
        ax.axhline(TETO * 100, color=TINTA_2, lw=1.1, ls=(0, (1, 2)), zorder=4)
        ax.set_xticks(range(len(Ks))); ax.set_xticklabels(Ks); ax.set_xlabel("Número de ações (K)")
        ax.set_title(NOMES[indice]); ax.set_ylim(0, 85)
    axes[0].set_ylabel("Janelas acima do limite da CVM (%)")
    titulos(fig, "Com 20 ações, alguma configuração chega ao teto de 20% nos dois índices",
            "Para cada K, o melhor resultado entre configurações e λ: fração das janelas acima do limite (pior entre 12 meses e 60 pregões). Linha pontilhada: teto de 20%. \"Sem limite\": a otimização escolhe sozinha quantas ações usar.",
            fonte=FONTE_ESCOLHA)
    salvar(fig, "rc_janelas_por_K.png")


def fig_lambda_cvm():
    estilo()
    d = dados_escolha()
    d = d[(d.K == 20) & (d.janela == 252) & (d.horizonte == 21)]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    for ax, indice in zip(axes, ["S&P100", "IBOV"]):
        g = d[d.indice == indice].set_index("lam").reindex(GRADE_L)
        x = np.arange(len(g))
        ax.axvspan(GRADE_L.index(3.0) - 0.5, GRADE_L.index(3.0) + 0.5, color=FAIXA, zorder=0)
        ax.plot(x - 0.05, g.f12 * 100, "-o", color=CORES[indice], lw=2.2, ms=6.5, mec="white", mew=1.2, zorder=3, label="12 meses (> 4 p.p.)")
        ax.plot(x + 0.05, g.f60 * 100, "-o", color=CORES_CLARAS[indice], lw=2.2, ms=6.5, mec="white", mew=1.2, zorder=3, label="60 pregões (> 2 p.p.)")
        ax.axhline(TETO * 100, color=TINTA_2, lw=1.1, ls=(0, (1, 2)), zorder=1)
        ax.set_xticks(x); ax.set_xticklabels([f"{l:g}".replace(".", ",") for l in GRADE_L]); ax.set_xlabel("λ (penalidade sobre a negociação)")
        ax.set_ylabel("Janelas acima do limite (%)"); ax.set_title(NOMES[indice]); ax.set_ylim(0, 60)
        ax.legend(loc="upper left", fontsize=10)
        ax2 = ax.twinx()
        ax2.plot(x, g.giro_anual, "--s", color=TINTA_2, lw=1.6, ms=5.5, mec="white", mew=1, zorder=2)
        ax2.set_ylim(bottom=0); ax2.grid(False); ax2.spines["right"].set_visible(False)
        ax2.set_ylabel("Giro (% ao ano)", color=TINTA_2); ax2.tick_params(axis="y", colors=TINTA_2)
    titulos(fig, "Com λ = 3, as janelas acima do limite ficam em até 18% e o giro cai de 295% para 10% ao ano",
            "K = 20, 1 ano de treino, rebalanceamento mensal. Linhas coloridas: janelas acima do limite. Linha cinza tracejada: giro médio. Linha pontilhada: teto de 20%. \"Sem limite\": a otimização escolhe sozinha quantas ações usar. Faixa: λ escolhido.",
            fonte=FONTE_ESCOLHA, base_pol=1.0, espaco_w=0.42)
    salvar(fig, "rc_lambda_cvm.png")


def serie_12m(modelo, nome):
    r = pd.read_csv(f"results/teste_final/{modelo}_{nome}.csv", parse_dates=["data_fim"])
    H = int(r.horizonte.iloc[0])
    r = r[r.dias == H].sort_values("numero").reset_index(drop=True)
    n = 252 // H
    return r["data_fim"].iloc[n - 1:].reset_index(drop=True), np.array(janelas(r, n))


def fig_diferenca_12m():
    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), sharey=True)
    for ax, (indice, nome) in zip(axes, [("S&P100", "sp100"), ("IBOV", "ibov")]):
        datas, g = serie_12m("cvm20", nome)
        ax.axvspan(CORTE, datas.iloc[-1] + pd.Timedelta(days=20), color=FAIXA, zorder=0)
        for lim in (-4, 4):
            ax.axhline(lim, color=TINTA_2, lw=1.1, ls=(0, (1, 2)), zorder=1)
        ax.plot(datas, g, color=CORES[indice], lw=2.3, zorder=3)
        ax.axhline(0, color=APAGADO, lw=0.8, zorder=2)
        ax.set_title(NOMES[indice]); ax.set_ylim(-9, 9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
        ax.text(CORTE + pd.Timedelta(days=15), -8.4, "teste", fontsize=10, color=TINTA_2)
    axes[0].set_ylabel("Carteira menos índice, acumulado em 12 meses (p.p.)")
    titulos(fig, "A diferença contra o índice sai da faixa de 4 p.p. em alguns períodos, mais no Ibovespa",
            "Modelo da regra (K = 20, 1 ano, mensal, λ = 3). Cada ponto é a diferença de retorno acumulada nos 12 meses até aquela data. Linhas pontilhadas: ±4 p.p. Área cinza: teste.",
            fonte=FONTE_TESTE)
    salvar(fig, "rc_diferenca_12m.png")


def fig_erro_no_teste():
    estilo()
    d = pd.read_csv("results/teste_final_cvm.csv")
    ordem = list(dict.fromkeys(d.modelo))
    rot = ["Regra\nCVM", "Regra\nanterior", "A: 1 ano,\ntrim.", "B: 2 anos,\ntrim."]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), sharey=True)
    for ax, indice in zip(axes, ["S&P100", "IBOV"]):
        cores = [CORES[indice], "#bfbdb4", CORES_CLARAS[indice], "#bfbdb4"]
        for k, m in enumerate(ordem):
            r = d[(d.modelo == m) & (d.indice == indice)].iloc[0]
            ax.bar(k, r.TE, 0.62, color=cores[k], yerr=r.ep_TE, error_kw=dict(ecolor=TINTA_2, lw=1.4, capsize=4), zorder=3)
            ax.text(k, r.TE / 2, num(r.TE, 2), ha="center", va="center", fontsize=12.5, color="white" if k == 0 else TINTA, fontweight="bold")
        ax.axhline(4.0, color=TINTA_2, lw=1.1, ls=(0, (1, 2)), zorder=4)
        ax.text(3.48, 4.08, "4 p.p.", ha="right", fontsize=10, color=TINTA_2)
        ax.set_xticks(range(4)); ax.set_xticklabels(rot, fontsize=10.5); ax.set_title(NOMES[indice]); ax.set_ylim(0, 6)
    axes[0].set_ylabel("Erro de rastreamento no teste (% ao ano)")
    titulos(fig, "No teste, o modelo da regra tem o menor erro dos quatro nos dois índices",
            "K = 20 em todos. A barra é a média; a linha vertical é o erro padrão da média dos meses ou trimestres. Linha pontilhada: 4 p.p.",
            fonte=FONTE_TESTE)
    salvar(fig, "rc_erro_no_teste.png")


def fig_carteira_vs_indice():
    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0))
    for ax, (indice, nome) in zip(axes, [("S&P100", "sp100"), ("IBOV", "ibov")]):
        t = rodadas_de_teste("cvm20", nome)
        datas = [t.data_inicio.iloc[0]] + list(t.data_fim)
        cart = np.concatenate([[100], 100 * np.cumprod(1 + t["retorno_liq_carteira"].to_numpy())])
        idx = np.concatenate([[100], 100 * np.cumprod(1 + t["retorno_indice"].to_numpy())])
        ax.plot(datas, idx, color=APAGADO, lw=2, label="Índice", zorder=2)
        ax.plot(datas, cart, color=CORES[indice], lw=2.4, label="Carteira de 20 ações", zorder=3)
        ax.set_title(NOMES[indice]); ax.set_ylabel("Valor de 100 investidos no início do teste")
        ax.text(datas[-1], cart[-1] + (3 if cart[-1] >= idx[-1] else -3), f" {num(cart[-1] - 100, 0, True)}%", va="center", fontsize=10.5, color=CORES[indice], fontweight="bold")
        ax.text(datas[-1], idx[-1] + (3 if cart[-1] < idx[-1] else -3), f" {num(idx[-1] - 100, 0, True)}%", va="center", fontsize=10.5, color=TINTA_2)
        ax.legend(loc="upper left"); ax.margins(x=0.12)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    dif = {}
    for indice, nome in [("S&P100", "sp100"), ("IBOV", "ibov")]:
        t = rodadas_de_teste("cvm20", nome)
        dif[indice] = 100 * ((1 + t["retorno_liq_carteira"]).prod() - (1 + t["retorno_indice"]).prod())
    titulos(fig, f"No teste, a carteira segue o índice e termina {num(dif['S&P100'], 0, True)} p.p. no S&P 100 e {num(dif['IBOV'], 0, True)} p.p. no Ibovespa",
            "Modelo da regra (1 ano, mensal, λ = 3), montado só com dados anteriores a cada mês. Os valores finais são o retorno acumulado no teste; a diferença vem em boa parte do viés de sobrevivência.",
            fonte=FONTE_TESTE)
    salvar(fig, "rc_carteira_vs_indice.png")


if __name__ == "__main__":
    fig_janelas_por_K()
    fig_lambda_cvm()
    fig_diferenca_12m()
    fig_erro_no_teste()
    fig_carteira_vs_indice()


def melhores_por_K():
    """Para cada índice e K, a célula (configuração e λ) de menor chance estimada de passar do limite da CVM."""
    d = pd.read_csv("results/selecao_prob_estimada.csv")
    o = dados_escolha().rename(columns={"indice": "ind", "janela": "J", "horizonte": "H"})
    d = d.merge(o[["K", "ind", "J", "H", "lam", "pior"]], on=["K", "ind", "J", "H", "lam"])
    return d.sort_values("p").groupby(["ind", "K"]).head(1)


def fig_comparacao_K():
    estilo()
    b = melhores_por_K()
    Ks = [5, 10, 15, 20, 30, 50, 999]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.8), sharex=True)
    for col, indice in enumerate(["S&P100", "IBOV"]):
        g = b[b.ind == ("S&P100" if indice == "S&P100" else "IBOV")].set_index("K").reindex(Ks)
        x = np.arange(len(Ks))
        ax = axes[0, col]
        ax.axvspan(3 - 0.5, 3 + 0.5, color=FAIXA, zorder=0)
        ax.bar(x - 0.16, g.p * 100, 0.32, color=CORES[indice], zorder=3, label="Chance estimada (TE e viés)")
        ax.bar(x + 0.16, g.pior * 100, 0.32, color=CORES_CLARAS[indice], zorder=3, label="Fração observada nas janelas")
        for k, v in enumerate(g.p * 100):
            ax.text(k - 0.16, v + 1.5, f"{num(v, 0)}%", ha="center", fontsize=10.5, color=TINTA, fontweight="bold")
        ax.axhline(TETO * 100, color=TINTA_2, lw=1.1, ls=(0, (1, 2)), zorder=4)
        ax.set_title(NOMES[indice]); ax.set_ylim(0, 100)
        if col == 0:
            ax.set_ylabel("Janelas acima do limite da CVM (%)")
        ax.legend(loc="upper right", fontsize=9.5)
        ax = axes[1, col]
        ax.axvspan(3 - 0.5, 3 + 0.5, color=FAIXA, zorder=0)
        ax.bar(x, g.TE, 0.5, color=CORES[indice], zorder=3, label="Erro de rastreamento (TE)")
        ax.plot(x, g.vies, "D", color=TINTA_2, ms=8, mec="white", mew=1.2, zorder=5, label="Viés (carteira menos índice)")
        for k, v in enumerate(g.TE):
            ax.text(k, v + 0.2, num(v, 1), ha="center", va="bottom", fontsize=11, color=TINTA, fontweight="bold")
        ax.axhline(0, color=APAGADO, lw=0.8, zorder=2)
        ax.set_ylim(-2, 8.5); ax.set_xticks(x); ax.set_xticklabels(["sem\nlimite" if k == 999 else k for k in Ks]); ax.set_xlabel("Número de ações (K)")
        if col == 0:
            ax.set_ylabel("% ao ano (TE) e p.p. ao ano (viés)")
        ax.legend(loc="upper right", fontsize=9.5)
    titulos(fig, "Com mais ações o erro cai, mas o risco de passar do limite da CVM não cai junto no Ibovespa",
            "Para cada K, a combinação de configuração e λ com a menor chance estimada de passar do limite, sem olhar o giro. Em cima: chance estimada e fração observada. Embaixo: TE e viés dessa combinação. Linha pontilhada: teto de 20%. \"Sem limite\": a otimização escolhe sozinha quantas ações usar.",
            fonte=FONTE_ESCOLHA)
    salvar(fig, "rc_comparacao_K.png")


if __name__ == "__main__" and False:
    pass
