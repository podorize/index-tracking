"""As figuras do relatório final, gravadas em figures/final/ (título = conclusão, subtítulo = o que se vê; azul S&P 100, laranja Ibovespa).
Leem os resultados de src/regra.py, src/teste_final.py, src/alternativa_simples.py e src/avaliacao.py."""
import textwrap
import urllib.request
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARQ, CORTE, FIGURAS, GRADE_K as KS, GRADE_LAMBDAS as GRADE_L, INDICES, RECOMENDADO, RESULTADOS
from src.regra import cenarios, configuracoes

PASTA_FONTES = Path("data/fontes")
URL_FONTES = "https://raw.githubusercontent.com/impallari/Libre-Franklin/master/fonts/TTF/"
CORES = {"S&P100": "#2a78d6", "IBOV": "#eb6834"}
CORES_CLARAS = {"S&P100": "#9ec5f4", "IBOV": "#f6b79c"}
NOMES = {"S&P100": "S&P 100", "IBOV": "Ibovespa"}
TINTA, TINTA_2, APAGADO = "#121212", "#52514e", "#898781"
GRADE, BASE, FUNDO = "#e1e0d9", "#c3c2b7", "#ffffff"
CONTEXTO, FAIXA = "#d6d4cd", "#ecebe5"
FONTE_12M = "Fonte: Yahoo Finance. Período de escolha: out/2021 a set/2024. Elaboração própria."
FONTE_TESTE = "Fonte: Yahoo Finance. Teste: out/2024 a set/2026. Elaboração própria."
FONTE_CURVA = ("Fonte: Yahoo Finance. Período de escolha: out/2021 a set/2024; teste: out/2024 a set/2026 (janelas de 12 meses inteiras "
               "no teste). Elaboração própria.")


def preparar_fonte():
    PASTA_FONTES.mkdir(parents=True, exist_ok=True)
    try:
        for peso in ["Regular", "Medium", "SemiBold", "Bold"]:
            arquivo = PASTA_FONTES / f"LibreFranklin-{peso}.ttf"
            if not arquivo.exists():
                urllib.request.urlretrieve(URL_FONTES + arquivo.name, arquivo)
            fm.fontManager.addfont(str(arquivo))
        return "Libre Franklin"
    except Exception as exc:
        print(f"Libre Franklin indisponível ({type(exc).__name__}); usando DejaVu Sans.")
        return "DejaVu Sans"


def aplicar_estilo():
    mpl.rcParams.update({
        "font.family": preparar_fonte(), "font.size": 11.5,
        "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
        "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": TINTA_2, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRADE, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.titlesize": 12.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 12, "axes.titlecolor": TINTA,
        "xtick.color": BASE, "ytick.color": BASE, "xtick.labelcolor": TINTA_2, "ytick.labelcolor": TINTA_2,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 10.5, "legend.labelcolor": TINTA,
        "figure.dpi": 110, "savefig.dpi": 200, "text.parse_math": False,
    })


def num(x, casas=1, sinal=False):
    texto = f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("+" + texto) if (sinal and x > 0) else texto.replace("-", "−")


def quebrar(texto, largura_pol, tamanho, fator):
    maximo = max(int(largura_pol * 72 / (tamanho * fator)), 30)
    return "\n".join(textwrap.fill(p, maximo) if len(p) > maximo else p for p in texto.split("\n"))


def titulos(fig, titulo, subtitulo=None, fonte=None, base_pol=0.75, esquerda=0.07, direita=0.97, espaco_w=None):
    largura, altura = fig.get_size_inches()
    titulo = quebrar(titulo, 0.97 * largura, 17, 0.56)
    subtitulo = quebrar(subtitulo, 0.97 * largura, 12, 0.50) if subtitulo else None
    n_t = titulo.count("\n") + 1
    n_s = 0 if not subtitulo else subtitulo.count("\n") + 1
    topo = 0.22 + 0.34 * n_t + (0.1 + 0.25 * n_s if subtitulo else 0) + 0.55
    ajustes = dict(top=1 - topo / altura, bottom=base_pol / altura, left=esquerda, right=direita)
    if espaco_w is not None:
        ajustes["wspace"] = espaco_w
    fig.subplots_adjust(**ajustes)
    y0 = 1 - 0.2 / altura
    fig.text(0.01, y0, titulo, ha="left", va="top", fontsize=17, fontweight="bold", color=TINTA, linespacing=1.15)
    if subtitulo:
        fig.text(0.01, y0 - (0.34 * n_t + 0.1) / altura, subtitulo, ha="left", va="top", fontsize=12,
                 color=TINTA_2, linespacing=1.35)
    if fonte:
        fig.text(0.01, 0.1 / altura, fonte, ha="left", va="bottom", fontsize=9, color=APAGADO)


def salvar(fig, nome):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / nome, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Figura salva: {FIGURAS / nome}")


def estilo():
    aplicar_estilo()
    mpl.rcParams["font.family"] = [mpl.rcParams["font.family"][0], "DejaVu Sans"]  # λ não existe na Libre Franklin


def rotulo_k(K):
    return "Sem\nlimite" if K == 999 else str(K)


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


def fig_etapa1(n=8):
    """Método, primeira etapa: as n combinações de menor chance média em cada índice, com o erro padrão de cada média."""
    estilo()
    c = pd.read_csv(RESULTADOS / "selecao_configuracao_indice.csv")
    cfg = configuracoes()
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
    from src.regra import janelas_todas, chance_por_K
    CENARIOS, cfg = cenarios(), configuracoes()
    estilo()
    w = janelas_todas()
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.6), sharex=True)
    for ax, indice in zip(axes, INDICES):
        t = chance_por_K(w, indice, *cfg[indice])
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
        ax.set_title(f"{NOMES[indice]} ({rotulo_cfg(*cfg[indice])})", fontsize=12, loc="left", pad=28)
        ax.text(1.02, len(t) + 0.15, "Escolhida no cenário", transform=ax.get_yaxis_transform(), fontsize=9.5, color=TINTA_2)
    axes[1].set_xlabel("Chance de passar de 4 p.p. em 12 meses (%)")
    titulos(fig, "Segunda etapa: cada cenário fica com o menor número de ações cuja chance, somada à margem, cabe no seu limite",
            "Período de escolha. Parte cheia da barra: chance estimada; parte listrada: margem de segurança de um erro padrão. "
            "Uma carteira serve ao cenário quando a barra inteira termina antes da linha do limite; entre as que servem, fica a de menos ações.",
            fonte=FONTE_12M, esquerda=0.1, direita=0.8)
    fig.subplots_adjust(hspace=0.35)
    salvar(fig, "rt_etapa2.png")


def fig_chance_por_K():
    """Etapa 2 da regra final: chance por número de ações na configuração do índice, com a margem de um erro padrão."""
    from src.regra import janelas_todas, chance_por_K
    CENARIOS, cfg = cenarios(), configuracoes()
    estilo()
    w = janelas_todas()
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    x = np.arange(len(KS))
    for teto in (v[0] for v in CENARIOS.values()):
        ax.axhline(teto, color=APAGADO, lw=1.0, ls=(0, (2, 2)), zorder=1)
    ch = {i: chance_por_K(w, i, *cfg[i]) for i in INDICES}
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
            f"Chance estimada no período de escolha, na configuração de cada índice (S&P 100: {rotulo_cfg(*cfg['S&P100'])}; "
            f"Ibovespa: {rotulo_cfg(*cfg['IBOV'])}). Cada cenário usa o menor "
            "número de ações cuja chance, com uma margem de segurança, fica até o seu limite (tracejado). Pontos grandes: números escolhidos.",
            fonte=FONTE_12M, direita=0.80)
    salvar(fig, "rt_chance_por_K.png")


def fig_lambda_te():
    """Etapa 1 da regra final: chance média das sete carteiras por combinação de histórico, frequência e λ; fica a menor em cada índice."""
    estilo()
    c = pd.read_csv(RESULTADOS / "selecao_configuracao_indice.csv")
    cfg = configuracoes()
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


def fig_piso_te():
    """Achado: erro de rastreamento fora da amostra por número de ações (o menor entre as 32 combinações) e o erro da carteira com
    todas as ações medido dentro do próprio ano de ajuste (1 ano, mensal, λ = 0), que é o piso do que a otimização consegue."""
    estilo()
    d = pd.read_csv(RESULTADOS / "selecao_chance_12m.csv")
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    x = np.arange(len(KS))
    for indice in INDICES:
        te = d[d.ind == indice].groupby("K").TE.min().reindex(KS).to_numpy()
        r = pd.read_csv(RESULTADOS / f"selecao/K999_{ARQ[indice]}_J252_H21_L0.csv")
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


def fig_carteira_vs_indice(cenario=RECOMENDADO):
    """Índice, carteira do cenário recomendado e a alternativa simples com o mesmo K (src/alternativa_simples.py), base 100 no teste."""
    from src.alternativa_simples import arquivo as arq_alt
    from src.avaliacao import ler
    from src.metricas import rodadas
    CENARIOS, cfg = cenarios(), configuracoes()
    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0))
    fim = {}
    for ax, indice in zip(axes, INDICES):
        K = CENARIOS[cenario][1][indice]
        J, H, lam = cfg[indice]
        t0 = ler(indice, K, J, H, lam)
        t0 = t0[t0.data_fim > CORTE]
        # a alternativa é mensal; na carteira trimestral, vai até o fim da última rodada da carteira
        alt = rodadas(arq_alt(indice, K), 21)
        alt = alt[(alt.data_fim > CORTE) & (alt.data_fim <= t0.data_fim.max())]
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
            fonte=FONTE_TESTE, espaco_w=0.3)
    salvar(fig, "rt_carteira_vs_indice.png")
    return fim


def figura_curva(c):
    """Uma seta por carteira de cenário: da chance prevista no período de escolha até a mesma conta no teste, com o limite do cenário
    (o mais exigente, quando a carteira serve a dois) e as janelas que de fato passaram de 4 p.p."""
    from matplotlib.lines import Line2D
    CENARIOS = cenarios()
    estilo()
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    c = c[c.cenario.fillna("") != ""].reset_index(drop=True)
    y = np.arange(len(c))[::-1]
    for k, s in c.iterrows():
        cor = CORES[s.indice]
        teto = min(v[0] for v in CENARIOS.values() if v[1][s.indice] == s.K)
        ax.plot([teto, teto], [y[k] - 0.32, y[k] + 0.32], color=TINTA, lw=2.2, zorder=2)
        ax.annotate("", xy=(s.chance_teste, y[k]), xytext=(s.chance_val, y[k]),
                    arrowprops=dict(arrowstyle="-|>", color=cor, lw=2.2, mutation_scale=16), zorder=3)
        ax.plot(s.chance_val, y[k], "o", ms=9, color="white", mec=cor, mew=2, zorder=4)
        ax.plot(s.chance_teste, y[k], "o", ms=10, color=cor, mec="white", mew=1.2, zorder=4)
        esq, dir_ = sorted([(s.chance_val, "prev"), (s.chance_teste, "teste")])
        ax.text(esq[0] - 0.8, y[k], f"{num(esq[0], 0)}%", ha="right", va="center", fontsize=10.5,
                color=TINTA if esq[1] == "teste" else TINTA_2, fontweight="bold" if esq[1] == "teste" else "normal")
        afasta = 1.6 if 0 <= teto - dir_[0] < 1.5 else 0.8  # não escrever sobre a linha do limite
        ax.text(dir_[0] + afasta, y[k], f"{num(dir_[0], 0)}%", ha="left", va="center", fontsize=10.5,
                color=TINTA if dir_[1] == "teste" else TINTA_2, fontweight="bold" if dir_[1] == "teste" else "normal")
        alerta = s.n4 > 0
        ax.text(1.02, y[k], f"{s.n4} de {s.n} janelas" + (f" (por {num(s.pior - 4, 1)} p.p.)" if alerta else ""),
                transform=ax.get_yaxis_transform(), va="center", fontsize=10.5,
                color=cor if alerta else TINTA_2, fontweight="bold" if alerta else "normal")
    ax.text(1.02, len(c) - 0.4, "Acima de 4 p.p.\nno teste", transform=ax.get_yaxis_transform(), va="center", fontsize=10, color=TINTA_2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{NOMES[s.indice]}, " + ("Sem limite" if s.K == 999 else f"{s.K} ações") + f"\n{s.cenario}" for _, s in c.iterrows()],
                       fontsize=10.5)
    ax.set_xlim(0, 36); ax.set_ylim(-0.7, len(c) - 0.1); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Chance de passar de 4 p.p. em 12 meses (%)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, color="white", mec=TINTA_2, mew=2, label="Previsto no período de escolha"),
                       Line2D([], [], marker="o", ls="", ms=10, color=TINTA_2, label="Mesma conta no teste"),
                       Line2D([], [], color=TINTA, lw=2.2, label="Limite do cenário")],
              loc="upper right", fontsize=10, frameon=True, framealpha=1, edgecolor="white")
    titulos(fig, "No teste, a chance ficou perto da prevista no S&P 100 e caiu no Ibovespa; só a carteira de 50 ações do S&P 100 passou do limite",
            "Cada seta vai da chance prevista no período de escolha até a mesma conta feita com os dados do teste. À direita, quantas janelas "
            "de 12 meses de fato passaram de 4 p.p. No S&P 100, as carteiras são refeitas a cada trimestre e têm só 4 janelas no teste.",
            fonte=FONTE_CURVA, direita=0.84, esquerda=0.17)
    salvar(fig, "rt_chance_por_K_regra_final.png")


def figura_decomposicao():
    """Por que a chance mudou no teste: decomposição de Shapley da mudança entre o erro de rastreamento (largura da curva) e o
    desvio médio (centro), a partir de results/decomposicao_chance.csv. As partes vêm da curva com o erro e o desvio médios; a soma
    delas difere da mudança observada (traço) pela variação entre as janelas."""
    from matplotlib.patches import Patch
    estilo()
    d = pd.read_csv(RESULTADOS / "decomposicao_chance.csv")
    d["eTE"], d["emu"] = d.dTE, d.dmu
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    y = np.arange(len(d))[::-1]
    for k, r in d.iterrows():
        cor, clara = CORES[r.ind], CORES_CLARAS[r.ind]
        for val, c, hatch in ((r.eTE, cor, None), (r.emu, clara, "////")):
            base = 0 if (val >= 0) != (r.eTE >= 0) or c == cor else r.eTE
            ax.barh(y[k], val, 0.55, left=base, color=c, hatch=hatch, edgecolor="white", lw=0, zorder=3)
        ax.plot(r.delta, y[k], "|", ms=22, mew=2.4, color=TINTA, zorder=4)
        ax.text(1.02, y[k] + 0.17, f"chance {num(r.chance_esc, 0)}% → {num(r.chance_teste, 0)}%", transform=ax.get_yaxis_transform(),
                va="center", fontsize=10.5, color=TINTA, fontweight="bold")
        ax.text(1.02, y[k] - 0.17, f"erro {num(r.TE_esc, 1)} → {num(r.TE_teste, 1)}% a.a. · desvio {num(r.mu_esc, 1)} → {num(r.mu_teste, 1)} p.p.",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9.5, color=TINTA_2)
    ax.axvline(0, color=TINTA_2, lw=1.0, zorder=2)
    nomes = {50: "50 ações", 30: "30 ações", 20: "20 ações", 999: "Sem limite"}
    ax.set_yticks(y); ax.set_yticklabels([f"{NOMES[r.ind]}, {nomes[r.K]}\n{r.cen}" for _, r in d.iterrows()], fontsize=10.5)
    ax.set_xlim(-8, 8); ax.set_ylim(-0.6, len(d) - 0.4); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mudança da chance entre o período de escolha e o teste (p.p.)")
    ax.legend(handles=[Patch(color=APAGADO, label="Parte explicada pelo erro de rastreamento (largura da curva)"),
                       Patch(facecolor="#dcdad2", hatch="////", edgecolor="white", label="Parte explicada pelo desvio médio (centro da curva)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9.5, frameon=False)
    titulos(fig, "No Ibovespa, a chance caiu porque o erro de rastreamento diminuiu; no S&P 100, mudou pouco, com efeitos que se compensam",
            "Decomposição de Shapley da mudança da chance entre os dois ingredientes da conta: o erro de rastreamento e o desvio médio em relação "
            "à carteira com todas as ações. Traço preto: mudança total. À direita, a chance e os dois ingredientes no período de escolha e no teste.",
            fonte=FONTE_CURVA, direita=0.66, esquerda=0.17)
    salvar(fig, "rt_decomposicao.png")


def gerar():
    """Gera as nove figuras do relatório."""
    from src.avaliacao import curva
    for f in (fig_indices, fig_etapa1, fig_etapa2, fig_chance_por_K, fig_lambda_te, fig_piso_te, fig_carteira_vs_indice):
        f()
    figura_curva(curva())
    figura_decomposicao()
