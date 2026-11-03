"""Figuras do relatório com a chance estimada só pelo TE e os três cenários de apetite ao risco (CRITERIOS_SELECAO_FINAL.md, decisões 3 a 6).
Cenários: conservador (risco máximo aceito de 10%), moderado (20%) e arrojado (30%), com K por índice, 1 ano de treino, rebalanceamento mensal e λ = 3."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figuras_parte2 import CORES, CORES_CLARAS, NOMES, TINTA, TINTA_2, APAGADO, FAIXA, CONTEXTO, titulos, salvar, num
from src.figuras_relatorio_final import estilo, SAIDA, FONTE_ESCOLHA, CORTE, rodadas_de_teste
from src.criterio_cvm import janelas
import src.figuras_parte2 as base

base.PASTA_FIGURAS = SAIDA
GRADE_L = [0.0, 0.5, 1.5, 3.0, 6.0, 12.0, 25.0, 50.0]
KS = [5, 10, 15, 20, 30, 50, 999]
LAM = 3.0
ARQ = {"S&P100": "sp100", "IBOV": "ibov"}
INDICES = ["S&P100", "IBOV"]
CENARIOS = {"Conservador": (10, {"S&P100": 50, "IBOV": 999}),   # Ibovespa: nenhum K cabe em 10% com a margem; usa a carteira sem limite
            "Moderado": (20, {"S&P100": 30, "IBOV": 30}),
            "Arrojado": (30, {"S&P100": 20, "IBOV": 30})}
# regra final (src/selecao_janelas_12m.configuracao): S&P 100 com 1 ano, trimestral, λ = 6; Ibovespa com 1 ano, mensal, λ = 3
CARTEIRAS = {("S&P100", 50): (252, 63, 6.0, "k50_tri_l6"), ("S&P100", 30): (252, 63, 6.0, "k30_tri_l6"), ("S&P100", 20): (252, 63, 6.0, "k20_tri_l6"),
             ("IBOV", 999): (252, 21, 3.0, "sem_limite_l3"), ("IBOV", 30): (252, 21, 3.0, "k30_l3")}
ALFA = {"Conservador": 1.0, "Moderado": 0.62, "Arrojado": 0.35}
FONTE_TESTE_TE = "Fonte: Yahoo Finance. Teste: out/2024 a set/2026. Elaboração própria."
FONTE_12M = "Fonte: Yahoo Finance. Período de escolha: out/2021 a set/2024. Elaboração própria."


def modelo(K, indice):
    """Modelo do teste final (results/teste_final) para um K e um índice, com λ = 3 (sem limite: λ = 1,5)."""
    if K == 999:
        return "sem_limite"
    return {15: "k15", 20: "cvm20", 30: "k30_l3", 50: "k50_l3"}[K]


def ks_distintos(indice):
    """K distintos dos cenários de um índice, do menor ao maior, com os cenários que usam cada um."""
    ks = sorted({v[1][indice] for v in CENARIOS.values()})
    alfas = np.linspace(0.4, 1.0, len(ks))
    return [(K, f"{' e '.join(c.lower() for c, v in CENARIOS.items() if v[1][indice] == K).capitalize()} (K = {K})", a) for K, a in zip(ks, alfas)]


def rotulo_k(K):
    return "Sem\nlimite" if K == 999 else str(K)


def fig_chance_por_K():
    """Etapa 2 da regra final: chance por número de ações na configuração do índice, com a margem de um erro padrão."""
    from src.selecao_janelas_12m import janelas_todas, chance_por_K
    estilo()
    w = janelas_todas()
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    x = np.arange(len(KS))
    for teto in (v[0] for v in CENARIOS.values()):
        ax.axhline(teto, color=APAGADO, lw=1.0, ls=(0, (2, 2)), zorder=1)
    ch = {i: chance_por_K(w, i, *CARTEIRAS[(i, CENARIOS["Moderado"][1][i])][:3]) for i in INDICES}
    for indice in ("IBOV", "S&P100"):
        g = ch[indice].chance.to_numpy()
        outro = ch["IBOV" if indice == "S&P100" else "S&P100"].chance.to_numpy()
        usados = {v[1][indice] for v in CENARIOS.values()}
        ax.plot(x, g, "-", color=CORES[indice], lw=2.4, zorder=3)
        for k, K in enumerate(KS):
            grande = K in usados
            ax.plot(k, g[k], "o", color=CORES[indice], ms=12 if grande else 6, mec="white", mew=1.5, zorder=4)
            txt = f"{num(g[k], 0)}%"
            peso = "bold" if grande else "normal"
            if g[k] >= outro[k]:
                ax.text(k, g[k] + 2.6, txt, ha="center", fontsize=10.5, color=TINTA, fontweight=peso)
            elif g[k] >= 8:
                ax.text(k, g[k] - 3.0, txt, ha="center", va="top", fontsize=10.5, color=TINTA, fontweight=peso)
            else:
                ax.text(k + 0.12, g[k] + 1.0, txt, ha="left", va="bottom", fontsize=10.5, color=TINTA, fontweight=peso)
        ax.text(-0.18, g[0], NOMES[indice], ha="right", va="center", fontsize=12, color=CORES[indice], fontweight="bold")
    for c, (teto, ks) in CENARIOS.items():
        ax.text(1.01, teto + 0.6, f"{c}: até {teto}%", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=10.5, color=TINTA, fontweight="bold")
        ib = "nenhuma cabe" if ks["IBOV"] == 999 else f"{ks['IBOV']}"
        ax.text(1.01, teto - 0.6, f"S&P 100: {ks['S&P100']} ações · Ibovespa: {ib}", transform=ax.get_yaxis_transform(),
                ha="left", va="top", fontsize=10, color=TINTA_2)
    ax.set_xticks(x); ax.set_xticklabels([rotulo_k(K) for K in KS], fontsize=11)
    ax.set_xlim(-1.1, len(KS) - 0.6); ax.set_ylim(0, 80)
    ax.set_ylabel("Chance de passar de 4 p.p. em 12 meses (%)")
    titulos(fig, "Quanto maior o apetite ao risco, menos ações a carteira precisa ter; no Ibovespa, nenhuma carteira cabe no conservador",
            "Chance estimada no período de escolha, na configuração de cada índice (S&P 100: 1 ano, trimestral, λ = 6; Ibovespa: 1 ano, mensal, λ = 3). Cada cenário usa o menor "
            "número de ações cuja chance, com uma margem de segurança, fica até o seu limite (tracejado). Pontos grandes: números escolhidos.",
            fonte=FONTE_12M, direita=0.80)
    salvar(fig, "rt_chance_por_K.png")


def rotulo_cfg(J, H, lam):
    return ("1 ano" if J == 252 else "2 anos") + (", mensal" if H == 21 else ", trimestral") + f", λ = {lam:g}".replace(".", ",")


def fig_indices():
    """Os dados: os dois índices (base 100 em 30/09/2019), com o período de escolha e o de teste marcados."""
    from src.data_loader import preparar_bases
    estilo()
    bases = preparar_bases()
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    fim = {}
    for indice in INDICES:
        y = bases[indice]["y"].fillna(0.0)
        v = 100 * (1 + y).cumprod()
        ax.plot(v.index, v.values, color=CORES[indice], lw=2.0, zorder=3)
        fim[indice] = v.iloc[-1]
        ax.text(v.index[-1] + pd.Timedelta(days=20), v.iloc[-1], f"{NOMES[indice]}\n{num(v.iloc[-1] - 100, 0, True)}%", va="center",
                fontsize=10.5, color=CORES[indice], fontweight="bold")
    ax.text(pd.Timestamp("2020-03-23"), 55, "Covid", ha="center", fontsize=10, color=TINTA_2)
    ax.set_ylabel("Valor de 100 investidos em 30/09/2019")
    ax.set_xlim(pd.Timestamp("2019-09-01"), pd.Timestamp("2027-04-01"))
    ax.set_xticks(pd.to_datetime([f"{a}-01-01" for a in range(2020, 2027)]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    titulos(fig, "Os dois índices viveram mercados diferentes: o S&P 100 quase triplicou, e o Ibovespa subiu menos da metade disso",
            "Índices com base 100 em 30/09/2019. Retornos diários do Yahoo Finance, em dólar no S&P 100 e em reais no Ibovespa.",
            fonte="Fonte: Yahoo Finance, 30/09/2019 a 30/09/2026. Elaboração própria.")
    salvar(fig, "rt_indices.png")


def fig_sobrevivencia():
    """Os dados: diferença acumulada em 12 meses da carteira com todas as ações disponíveis contra o índice. Ela deveria ficar perto
    de zero; o que sobra acima de zero é, em boa parte, o viés de sobrevivência da lista de ações de 2026."""
    estilo()
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.axhline(0, color=TINTA_2, lw=1.0, zorder=1)
    ax.axvline(CORTE, color=APAGADO, lw=1.0, ls=(0, (2, 2)), zorder=1)
    for indice in INDICES:
        r = pd.read_csv(f"results/teste_final/sem_limite_{ARQ[indice]}.csv", parse_dates=["data_fim"])
        r = r[r.dias == 21].sort_values("numero").reset_index(drop=True)
        g = np.array(janelas(r, 12)); f = r.data_fim.iloc[11:].reset_index(drop=True)
        ax.plot(f, g, color=CORES[indice], lw=2.4, zorder=3)
        antes = (f <= CORTE).to_numpy()
        for ini, fim, m in ((f.iloc[0], CORTE, g[antes].mean()), (CORTE, f.iloc[-1], g[~antes].mean())):
            ax.hlines(m, ini, fim, color=CORES[indice], lw=1.4, ls=(0, (4, 3)), zorder=2)
            ax.text(ini + pd.Timedelta(days=20), m + (0.3 if indice == "IBOV" else -0.3), f"média: {num(m, 1)} p.p.",
                    ha="left", va="bottom" if indice == "IBOV" else "top", fontsize=10, color=CORES[indice], fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85), zorder=5)
        ax.text(f.iloc[-1] + pd.Timedelta(days=20), g[-1], NOMES[indice], va="center", fontsize=11, color=CORES[indice], fontweight="bold")
    ax.text(CORTE + pd.Timedelta(days=15), 9.6, "início do teste", fontsize=9.5, color=TINTA_2, va="top")
    ax.set_ylabel("Carteira com todas as ações menos o índice,\nem 12 meses (p.p.)")
    ax.set_xlim(pd.Timestamp("2022-08-01"), pd.Timestamp("2027-02-01"))
    ax.set_xticks(pd.to_datetime([f"{a}-01-01" for a in range(2023, 2027)]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    titulos(fig, "O viés de sobrevivência aparece no Ibovespa: a carteira com todas as ações rendeu cerca de 5 p.p. acima do índice em 12 meses até 2024",
            "Diferença acumulada em cada janela de 12 meses entre a carteira com todas as ações da lista de 2026 e o índice. Sem viés, ela ficaria perto "
            "de zero. Tracejado: média das janelas antes e depois do início do teste. O efeito diminui à medida que o período se aproxima da data da lista.",
            fonte="Fonte: Yahoo Finance. Carteira refeita todo mês com o último ano de dados. Elaboração própria.")
    salvar(fig, "rt_sobrevivencia.png")


def fig_walk_forward():
    """Método: esquema do walk-forward, com rodadas de exemplo (treino de 1 ou 2 anos antes, avaliação no período seguinte),
    o período de escolha e o teste."""
    from matplotlib.patches import Patch
    estilo()
    T = pd.Timestamp
    ini, esc, tst, fim = T("2019-10-01"), T("2021-10-01"), CORTE, T("2026-10-01")
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.axvspan(esc, tst, color=FAIXA, zorder=0)
    ax.axvspan(tst, fim, color="#dcdad2", zorder=0)
    rodadas = [T("2021-10-01"), T("2022-10-01"), T("2023-10-01"), T("2024-10-01"), T("2025-10-01")]
    for k, r in enumerate(rodadas):
        y = len(rodadas) - k
        ax.barh(y, (r - (r - pd.DateOffset(years=2))).days, left=r - pd.DateOffset(years=2), height=0.5, color=CONTEXTO, zorder=2)
        ax.barh(y, (r - (r - pd.DateOffset(years=1))).days, left=r - pd.DateOffset(years=1), height=0.5, color=APAGADO, zorder=2)
        ax.barh(y, 91, left=r, height=0.5, color=TINTA, zorder=3)
        ax.text(r + pd.Timedelta(days=110), y, f"rodada de {r:%m/%Y}", va="center", fontsize=10, color=TINTA_2)
    ax.text(pd.Timestamp("2022-01-01") + (tst - esc) / 2 - pd.Timedelta(days=90), len(rodadas) + 1.05,
            "Período de escolha\nrodadas avaliadas de out/2021 a set/2024", ha="center", fontsize=10.5, color=TINTA_2)
    ax.text(tst + (fim - tst) / 2, len(rodadas) + 1.05, "Teste, com a regra congelada\nrodadas de out/2024 a set/2026",
            ha="center", fontsize=10.5, color=TINTA_2)
    ax.text(ini + (esc - ini) / 2, len(rodadas) + 1.05, "Só treino\nout/2019 a set/2021", ha="center", fontsize=10.5, color=TINTA_2)
    ax.text(T("2022-11-15"), 0.25, "… e assim por diante: a janela avança um mês (ou um trimestre) por vez", fontsize=10, color=TINTA_2)
    ax.set_yticks([]); ax.set_ylim(-0.2, len(rodadas) + 1.9)
    ax.set_xlim(ini, fim)
    ax.set_xticks(pd.to_datetime([f"{a}-01-01" for a in range(2020, 2027)]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", visible=False)
    for lado in ("left",):
        ax.spines[lado].set_visible(False)
    ax.legend(handles=[Patch(color=APAGADO, label="Treino com 1 ano de histórico"),
                       Patch(color=CONTEXTO, label="Ano a mais, nas configurações com 2 anos"),
                       Patch(color=TINTA, label="Mês ou trimestre avaliado")],
              loc="lower left", bbox_to_anchor=(0.0, 0.02), fontsize=9.5, ncol=1, frameon=False)
    titulos(fig, "Cada carteira é montada só com o passado e medida no período seguinte; a janela avança e o processo se repete",
            "Walk-forward com cinco rodadas de exemplo, uma por ano (na prática, há uma rodada por mês ou por trimestre). O mesmo "
            "procedimento roda nas duas fases: no período de escolha, os resultados decidem a forma de montar e o número de ações; no "
            "teste, a regra fica congelada e os resultados só medem.",
            fonte="Elaboração própria.")
    salvar(fig, "rt_walk_forward.png")


def fig_etapa1(n=8):
    """Método, primeira etapa: as n combinações de menor chance média em cada índice, com o erro padrão de cada média."""
    from src.selecao_janelas_12m import configuracao
    estilo()
    c = pd.read_csv("results/selecao_configuracao_indice.csv")
    cfg = configuracao(c)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0))
    for ax, indice in zip(axes, INDICES):
        x = c[c.ind == indice].sort_values("chance").head(n).iloc[::-1].reset_index(drop=True)
        y = np.arange(len(x))
        esc = [tuple(r) == tuple(cfg[indice]) for r in x[["J", "H", "lam"]].values]
        for k, r in x.iterrows():
            cor = CORES[indice] if esc[k] else APAGADO
            ax.errorbar(r.chance, k, xerr=r.ep, fmt="o", color=cor, ecolor=cor, elinewidth=2.2 if esc[k] else 1.4, capsize=4,
                        ms=10 if esc[k] else 6, mec="white", mew=1.2, zorder=4)
        for k, r in x.iterrows():
            ax.text(r.chance + r.ep + 0.4, k, f"{num(r.chance, 1)}% ± {num(r.ep, 1)}", va="center", fontsize=10,
                    color=TINTA, fontweight="bold" if esc[k] else "normal")
        ax.set_yticks(y); ax.set_yticklabels([rotulo_cfg(*r) for r in x[["J", "H", "lam"]].values], fontsize=10)
        ax.set_xlim((x.chance - x.ep).min() - 1, (x.chance + x.ep).max() + 5)
        ax.set_ylim(-0.6, len(x) - 0.4); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Chance média das sete carteiras (%)")
        ax.set_title(NOMES[indice])
    titulos(fig, "Primeira etapa: a combinação de menor chance média vence, mas as primeiras ficam dentro do erro padrão umas das outras",
            "Período de escolha. As oito combinações de histórico, frequência e λ com menor chance média das sete carteiras (5 a 50 ações e "
            "sem limite). Ponto colorido: escolhida. Traço: um erro padrão para cada lado, quanto a média varia por acaso.",
            fonte=FONTE_12M, esquerda=0.17, espaco_w=0.55)
    salvar(fig, "rt_etapa1.png")


def fig_etapa2():
    """Método, segunda etapa: para cada número de ações, chance + margem de um erro padrão contra o limite de cada cenário."""
    from src.selecao_janelas_12m import janelas_todas, chance_por_K
    estilo()
    w = janelas_todas()
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.6), sharex=True)
    for ax, indice in zip(axes, INDICES):
        t = chance_por_K(w, indice, *CARTEIRAS[(indice, CENARIOS["Moderado"][1][indice])][:3])
        y = np.arange(len(t))[::-1]
        usados = {K: [c for c, v in CENARIOS.items() if v[1][indice] == K] for K in t.K}
        for k, r in t.iterrows():
            esc = bool(usados[r.K])
            ax.barh(y[k], r.chance, 0.62, color=CORES[indice] if esc else CONTEXTO, zorder=3)
            ax.barh(y[k], r.ep, 0.62, left=r.chance, color=CORES_CLARAS[indice] if esc else "#ecebe5", zorder=3,
                    hatch="////" if esc else None, edgecolor="white", lw=0)
            ax.text(r.limite + 1.0, y[k], f"{num(r.chance, 1)}% + {num(r.ep, 1)} = {num(r.limite, 1)}%", va="center", fontsize=10,
                    color=TINTA if esc else TINTA_2, fontweight="bold" if esc else "normal", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
            if esc:
                nomes = " e ".join(usados[r.K])
                if indice == "IBOV" and r.K == 999:
                    nomes += " (nenhuma cabe;\na mais próxima)"
                ax.text(1.02, y[k], nomes, transform=ax.get_yaxis_transform(), va="center", fontsize=10.5,
                        color=CORES[indice], fontweight="bold")
        for c, (teto, _) in CENARIOS.items():
            ax.axvline(teto, color=TINTA, lw=1.2, ls=(0, (3, 2)), zorder=4)
            ax.text(teto, len(t) - 0.35, f"{c}\naté {teto}%", ha="center", va="bottom", fontsize=9.5, color=TINTA, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
        ax.set_yticks(y); ax.set_yticklabels(["Sem limite" if K == 999 else f"{K} ações" for K in t.K], fontsize=10.5)
        ax.set_ylim(-0.6, len(t) + 0.6); ax.set_xlim(0, 80); ax.grid(axis="y", visible=False)
        cfg = "1 ano, trimestral, λ = 6" if indice == "S&P100" else "1 ano, mensal, λ = 3"
        ax.set_title(f"{NOMES[indice]} ({cfg})", fontsize=12, loc="left", pad=28)
        ax.text(1.02, len(t) + 0.15, "Escolhida no cenário", transform=ax.get_yaxis_transform(), fontsize=9.5, color=TINTA_2)
    axes[1].set_xlabel("Chance de passar de 4 p.p. em 12 meses (%)")
    titulos(fig, "Segunda etapa: cada cenário fica com o menor número de ações cuja chance, somada à margem, cabe no seu limite",
            "Período de escolha. Parte cheia da barra: chance estimada; parte listrada: margem de segurança de um erro padrão. "
            "Uma carteira serve ao cenário quando a barra inteira termina antes da linha do limite; entre as que servem, fica a de menos ações.",
            fonte=FONTE_12M, esquerda=0.1, direita=0.8)
    fig.subplots_adjust(hspace=0.35)
    salvar(fig, "rt_etapa2.png")


def fig_piso_te():
    """Achado: erro de rastreamento fora da amostra por número de ações (o menor entre as 32 combinações) e o erro da carteira com
    todas as ações medido dentro do próprio ano de ajuste (1 ano, mensal, λ = 0), que é o piso do que a otimização consegue."""
    estilo()
    d = pd.read_csv("results/selecao_chance_12m.csv")
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    x = np.arange(len(KS))
    for indice in INDICES:
        te = d[d.ind == indice].groupby("K").TE.min().reindex(KS).to_numpy()
        r = pd.read_csv(f"results/selecao/K999_{ARQ[indice]}_J252_H21_L0.csv")
        dentro = float(np.sqrt(252 * r[r.dias == 21].objetivo_heuristica.mean()))
        ax.plot(x, te, "-o", color=CORES[indice], lw=2.4, ms=7, mec="white", mew=1.2, zorder=3)
        acima = indice == "IBOV"
        for k, v in enumerate(te):
            peso = "bold" if KS[k] == 999 else "normal"
            if not acima and KS[k] == 999:  # à direita, para não cair sobre o tracejado
                ax.text(k + 0.12, v, f"{num(v, 1)}%", ha="left", va="center", fontsize=10.5, color=TINTA, fontweight=peso)
            else:
                ax.text(k, v + (0.22 if acima else -0.22), f"{num(v, 1)}%", ha="center", va="bottom" if acima else "top", fontsize=10.5,
                        color=TINTA, fontweight=peso)
        ax.axhline(dentro, color=CORES[indice], lw=1.3, ls=(0, (3, 2)), zorder=2)
        ax.text(-1.0, dentro + 0.08, f"{NOMES[indice]}: ajuste dentro do próprio ano, {num(dentro, 1)}%", ha="left", va="bottom",
                fontsize=10, color=CORES[indice])
        ax.text(-0.18, te[0], NOMES[indice], ha="right", va="center", fontsize=12, color=CORES[indice], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([rotulo_k(K) for K in KS], fontsize=11)
    ax.set_xlim(-1.1, len(KS) - 0.3); ax.set_ylim(0, 7.5)
    ax.set_ylabel("Erro de rastreamento (% ao ano)")
    titulos(fig, "O Ibovespa erra mais que o S&P 100 em todos os tamanhos, e o piso do erro é o dobro",
            "Período de escolha. Linhas: erro de rastreamento fora da amostra, o menor entre as 32 combinações de histórico, frequência e λ. "
            "Tracejado: erro da carteira com todas as ações medido no mesmo ano usado para ajustar os pesos, o mínimo que a otimização "
            "consegue mesmo sem prever nada.",
            fonte=FONTE_12M, direita=0.97)
    salvar(fig, "rt_piso_te.png")


def fig_lambda_te():
    """Etapa 1 da regra final: chance média das sete carteiras por combinação de histórico, frequência e λ; fica a menor em cada índice."""
    from src.selecao_janelas_12m import configuracao
    estilo()
    c = pd.read_csv("results/selecao_configuracao_indice.csv")
    cfg = configuracao(c)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4))
    x = np.arange(len(GRADE_L))
    for ax, indice in zip(axes, INDICES):
        J0, H0, L0 = cfg[indice]
        fins = c[(c.ind == indice) & (c.lam == GRADE_L[-1])].sort_values("chance")
        ys = list(fins.chance)
        for k in range(1, len(ys)):
            ys[k] = max(ys[k], ys[k - 1] + 1.3)  # rótulos do fim das linhas sem sobreposição
        y_rot = {(r.J, r.H): y for (_, r), y in zip(fins.iterrows(), ys)}
        for (J, H), g in c[c.ind == indice].groupby(["J", "H"]):
            g = g.set_index("lam").reindex(GRADE_L)
            esc = (J, H) == (J0, H0)
            rot = ("1 ano" if J == 252 else "2 anos") + (", mensal" if H == 21 else ", trimestral")
            ax.plot(x, g.chance, "-o", color=CORES[indice] if esc else APAGADO, lw=2.6 if esc else 1.2, ms=6 if esc else 3.5,
                    zorder=3 if esc else 2, alpha=1 if esc else 0.8)
            ax.text(len(GRADE_L) - 0.85, y_rot[(J, H)], f" {rot}", va="center", fontsize=9.5,
                    color=CORES[indice] if esc else TINTA_2, fontweight="bold" if esc else "normal")
            if esc:
                k = GRADE_L.index(L0)
                ax.plot(k, g.chance.iloc[k], "o", ms=12, color=CORES[indice], mec="white", mew=1.5, zorder=4)
                ax.text(k, g.chance.iloc[k] - 1.0, f"{num(g.chance.iloc[k], 1)}%", ha="center", va="top", fontsize=10.5, fontweight="bold")
                giro = g.giro
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l:g}".replace(".", ",") + f"\n{num(gi, 0)}%" for l, gi in zip(GRADE_L, giro)], fontsize=10)
        ax.set_xlabel("λ e, abaixo, giro por ano na configuração escolhida")
        ax.set_title(NOMES[indice]); ax.set_xlim(-0.4, len(GRADE_L) + 1.6)
        ci = c[c.ind == indice].chance
        ax.set_ylim(ci.min() - 2.5, max(ci.max(), max(ys)) + 1.5)
        ax.set_ylabel("Chance média das sete carteiras (%)" if indice == "S&P100" else "")
    titulos(fig, "Controlar a negociação derruba o giro sem aumentar a chance, até certo ponto: com λ alto demais, a chance volta a subir",
            "Período de escolha. Chance média das sete carteiras (5 a 50 ações e Sem limite) por λ; abaixo de cada λ, o giro por ano na "
            "configuração escolhida (linha colorida). As linhas cinza são as outras combinações de histórico e frequência.",
            fonte=FONTE_12M, espaco_w=0.12, base_pol=1.1)
    salvar(fig, "rt_lambda.png")


def serie_12m(mod, nome):
    """Diferença acumulada em 12 meses (p.p.) só das janelas inteiras dentro do período de teste (12 no S&P 100 e 13 no Ibovespa)."""
    r = pd.read_csv(f"results/teste_final/{mod}_{nome}.csv", parse_dates=["data_fim"])
    r = r[r.dias == 21].sort_values("numero").reset_index(drop=True)
    g = np.array(janelas(r, 12))
    teste = (r["data_fim"] > CORTE).to_numpy()
    inteira = np.array([teste[k:k + 12].all() for k in range(len(g))])
    return r["data_fim"].iloc[11:].reset_index(drop=True)[inteira].reset_index(drop=True), g[inteira]


def fig_erro_no_teste():
    estilo()
    d = pd.read_csv("results/teste_final_cvm.csv")
    rot = {"Regra CVM (K=20, 1a/mensal, λ 3/3)": 20, "Arrojado: K=15 (1a/mensal, λ 3/3)": 15, "Sensibilidade: K=30 (1a/mensal, λ 3/3)": 30,
           "K=50 (1a/mensal, λ 50/3)": 50, "Referência: sem limite (1a/mensal, λ 1,5)": 999}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.6), sharey=True)
    for ax, indice in zip(axes, INDICES):
        barras = [(c, CENARIOS[c][1][indice]) for c in CENARIOS] + [("Sem limite", 999)]
        for k, (c, K) in enumerate(barras):
            nome = next(m for m, kk in rot.items() if kk == K)
            r = d[(d.modelo == nome) & (d.indice == indice)].iloc[0]
            cor = "#bfbdb4" if K == 999 else CORES[indice]
            ax.bar(k, r.TE, 0.62, color=cor, alpha=1.0 if K == 999 else ALFA[c], yerr=r.ep_TE, error_kw=dict(ecolor=TINTA_2, lw=1.4, capsize=4), zorder=3)
            ax.text(k, r.TE / 2, num(r.TE, 1), ha="center", va="center", fontsize=12.5, color=TINTA if (K == 999 or ALFA[c] < 0.5) else "white", fontweight="bold")
        ax.axhline(4.0, color=TINTA_2, lw=1.1, ls=(0, (1, 2)), zorder=4)
        ax.text(len(barras) - 0.52, 4.1, "4 p.p.", ha="right", fontsize=10, color=TINTA_2)
        ax.set_xticks(range(len(barras)))
        ax.set_xticklabels([f"{c}\nK = {K}" if K != 999 else "Sem limite\n(65 a 87)" for c, K in barras], fontsize=10)
        ax.set_title(NOMES[indice]); ax.set_ylim(0, 6)
    axes[0].set_ylabel("Erro de rastreamento no teste (% ao ano)")
    titulos(fig, "No teste, quanto menor o apetite ao risco, menor o erro de rastreamento, e todos ficam abaixo de 4 p.p.",
            "1 ano de treino e rebalanceamento mensal (λ = 3; sem limite com λ = 1,5). A barra é a média; a linha vertical é o erro padrão da média dos meses. Linha pontilhada: 4 p.p.", fonte=FONTE_TESTE_TE)
    salvar(fig, "rt_erro_no_teste.png")


def fig_diferenca_12m():
    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True)
    for ax, indice in zip(axes, INDICES):
        for lim in (-4, 4):
            ax.axhline(lim, color=TINTA_2, lw=1.1, ls=(0, (1, 2)), zorder=1)
        ax.axhline(0, color=APAGADO, lw=0.8, zorder=2)
        for K, rot, alfa in ks_distintos(indice):
            datas, g = serie_12m(modelo(K, indice), ARQ[indice])
            ax.plot(datas, g, color=CORES[indice], alpha=alfa, lw=2.4, zorder=3, label=rot)
        ax.set_title(NOMES[indice]); ax.set_ylim(-7, 8); ax.legend(loc="upper left", fontsize=10)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    axes[0].set_ylabel("Carteira menos índice, acumulado em 12 meses (p.p.)")
    titulos(fig, "Nos cenários mais conservadores, a diferença em relação ao índice fica dentro da faixa de 4 p.p.",
            "Cada ponto é a diferença de retorno acumulada nos 12 meses até aquela data, só nas janelas inteiras dentro do período de teste (λ = 3). Linhas pontilhadas: ±4 p.p.", fonte=FONTE_TESTE_TE)
    salvar(fig, "rt_diferenca_12m.png")


def fig_carteira_vs_indice(cenario="Moderado"):
    """Índice, carteira do cenário recomendado e a alternativa simples com o mesmo K (src/baseline_teste.py), base 100 no teste."""
    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0))
    fim = {}
    for ax, indice in zip(axes, INDICES):
        K = CENARIOS[cenario][1][indice]
        _, H, _, mod = CARTEIRAS[(indice, K)]
        t0 = rodadas_de_teste(mod, ARQ[indice])
        t0 = t0[t0.dias == H]
        # a alternativa é mensal; na carteira trimestral, vai até o fim da última rodada da carteira
        alt = rodadas_de_teste(f"corrqp{K}", ARQ[indice])
        alt = alt[(alt.dias == 21) & (alt.data_fim <= t0.data_fim.max())]
        linhas = [("Índice", t0, "retorno_indice", TINTA_2, "-", 2.2),
                  ("Alternativa simples", alt, "retorno_liq_carteira", APAGADO, (0, (3, 2)), 1.8),
                  (f"Carteira de {K} ações", t0, "retorno_liq_carteira", CORES[indice], "-", 2.6)]
        for nome, r, col, cor, ls, lw in linhas:
            datas_r = [r.data_inicio.iloc[0]] + list(r.data_fim)
            v = np.concatenate([[100], 100 * np.cumprod(1 + r[col].to_numpy())])
            fim[(indice, nome)] = v[-1]
            ax.plot(datas_r, v, color=cor, ls=ls, lw=lw, zorder=3 if nome.startswith("Carteira") else 2,
                    marker="o" if H == 63 and r is t0 else None, ms=4)
        datas = [t0.data_inicio.iloc[0]] + list(t0.data_fim)
        # rótulos no fim das linhas, afastados para não se sobreporem
        ordem = sorted([(n, fim[(indice, n)]) for n, *_ in linhas], key=lambda p: p[1])
        ys = [p[1] for p in ordem]
        passo = 0.075 * (max(ys) - 90)  # espaço mínimo proporcional à escala do painel
        for k in range(1, len(ys)):
            ys[k] = max(ys[k], ys[k - 1] + passo)
        cores = {n: c for n, _, _, c, *_ in linhas}
        for (n, v), y in zip(ordem, ys):
            ax.text(datas[-1] + pd.Timedelta(days=12), y, f"{n}: {num(v - 100, 0, True)}%", va="center", fontsize=10,
                    color=cores[n], fontweight="bold" if n.startswith("Carteira") else "normal")
        ax.set_title(NOMES[indice]); ax.set_ylabel("Valor de 100 investidos no início do teste")
        ax.set_xlim(datas[0] - pd.Timedelta(days=10), datas[-1] + pd.Timedelta(days=250))
        ax.set_xticks(pd.to_datetime(["2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01"]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ks = CENARIOS[cenario][1]
    tam = f"{ks['S&P100']} ações em cada índice" if ks["S&P100"] == ks["IBOV"] else f"{ks['S&P100']} ações no S&P 100 e {ks['IBOV']} no Ibovespa"
    titulos(fig, f"Com {tam}, a carteira anda junto com o índice; a alternativa simples se afasta",
            "Teste, de out/2024 a set/2026, carteiras montadas só com dados anteriores a cada rodada. No S&P 100, a carteira é refeita a cada trimestre "
            "(pontos) e a comparação vai até a última rodada completa, em jul/2026. Alternativa simples: as mesmas quantidades de ações mais "
            "correlacionadas com o índice no último ano, com os pesos que melhor o reproduziam, refeita todo mês. "
            "A carteira termina acima do índice em parte pelo viés de sobrevivência.",
            fonte=FONTE_TESTE_TE, espaco_w=0.3)
    salvar(fig, "rt_carteira_vs_indice.png")
    return fim


ROTULO_TESTE = {"k15": "Arrojado: K=15 (1a/mensal, λ 3/3)", "cvm20": "Regra CVM (K=20, 1a/mensal, λ 3/3)",
                "k30_l3": "Sensibilidade: K=30 (1a/mensal, λ 3/3)", "k50": "K=50 (1a/mensal, λ 50/3)"}


def fig_cenarios():
    """Os três cenários lado a lado: ações, erro de rastreamento no teste e janelas de 12 meses acima de 4 p.p."""
    estilo()
    d = pd.read_csv("results/teste_final_cvm.csv")
    nomes = list(CENARIOS)
    x = np.arange(len(nomes)); w = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.9))
    for j, indice in enumerate(INDICES):
        off = (j - 0.5) * w
        acoes, erros, frac, rot = [], [], [], []
        for c in nomes:
            K = CENARIOS[c][1][indice]
            acoes.append(K)
            erros.append(d[(d.modelo == ROTULO_TESTE[modelo(K, indice)]) & (d.indice == indice)].TE.iloc[0])
            _, g = serie_12m(modelo(K, indice), ARQ[indice])
            n4 = int((np.abs(g) > 4).sum()); frac.append(100 * n4 / len(g)); rot.append(f"{n4}/{len(g)}")
        for ax, vals, fmt in ((axes[0], acoes, lambda v: f"{v}"), (axes[1], erros, lambda v: num(v, 1)), (axes[2], frac, None)):
            ax.bar(x + off, vals, w, color=CORES[indice], zorder=3, label=NOMES[indice])
            for k, v in enumerate(vals):
                txt = rot[k] if fmt is None else fmt(v)
                ax.text(x[k] + off, v + (1.2 if ax is axes[0] else 0.08 if ax is axes[1] else 1.6), txt, ha="center", fontsize=10 if fmt else 9.5, color=TINTA, fontweight="bold")
    for k, c in enumerate(nomes):
        teto = CENARIOS[c][0]
        axes[2].hlines(teto, k - 0.42, k + 0.42, color=TINTA_2, lw=1.4, ls=(0, (2, 2)), zorder=4)
        axes[2].text(k + 0.44, teto + 2.2, f"máx. {teto}%", ha="right", va="bottom", fontsize=8.5, color=TINTA_2)
    titulos_eixo = ["Ações na carteira", "Erro de rastreamento no teste (% ao ano)", "Janelas de 12 meses acima de 4 p.p. (%)"]
    for ax, t in zip(axes, titulos_eixo):
        ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=10.5); ax.set_title(t, fontsize=12)
    axes[0].set_ylim(0, 58); axes[1].set_ylim(0, 4.6); axes[2].set_ylim(0, 80)
    axes[0].legend(loc="upper right", fontsize=10)
    titulos(fig, "Quanto mais conservador o cenário, mais ações e menor o desvio em relação ao índice",
            "Teste de out/2024 a set/2026, com 1 ano de treino, rebalanceamento mensal e λ = 3. No painel da direita, o tracejado é o risco máximo aceito de cada cenário; os rótulos mostram janelas acima de 4 p.p. sobre o total de janelas inteiras dentro do teste (12 no S&P 100 e 13 no Ibovespa).",
            fonte=FONTE_TESTE_TE)
    salvar(fig, "rt_cenarios.png")


if __name__ == "__main__":
    fig_cenarios()
    fig_chance_por_K()
    fig_lambda_te()
    fig_erro_no_teste()
    fig_diferenca_12m()
    print(fig_carteira_vs_indice())
