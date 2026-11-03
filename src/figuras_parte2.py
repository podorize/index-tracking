"""
figuras_parte2.py
Figuras da Parte 2 no mesmo estilo das figuras da Parte 1 (título = conclusão, subtítulo = o que se vê,
cores fixas por índice: azul S&P 100, laranja Ibovespa). Lê os resultados de compare_all_models.py,
validacao_lambda.py e rebalanceamento_experiment.py (arquivos com sufixo _K{K}).
  31_comparativo_modelos.png   TE e giro dos 8 modelos (Tabela 4.3 do Guia)
  33_modelo_nulo.png           modelo final × 1.000 carteiras aleatórias nos mesmos 20 trimestres
  34_te_por_regime.png         TE por regime de mercado fora da amostra
  35_diferenca_acumulada.png   Diferença acumulada carteira − índice por modelo
  36_rebalanceamento_janela.png TE × giro anualizado por frequência de rebalanceamento e janela de treino
                               (rebalanceamento_experiment.py; não faz parte de gerar_figuras)
  41_erro_trimestral_modelo_final.png TE e diferença por trimestre do modelo final
  37_walk_forward.png          As 20 rodadas do walk-forward: treino de 504 pregões e teste de 63 (não faz parte de gerar_figuras)
"""

from pathlib import Path
import textwrap
import urllib.request

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

PASTA_RESULTADOS = Path("results")
PASTA_FIGURAS = Path("figures")
PASTA_FONTES = Path("data/fontes")
URL_FONTES = "https://raw.githubusercontent.com/impallari/Libre-Franklin/master/fonts/TTF/"

CORES = {"S&P100": "#2a78d6", "IBOV": "#eb6834"}
CORES_CLARAS = {"S&P100": "#9ec5f4", "IBOV": "#f6b79c"}
NOMES = {"S&P100": "S&P 100", "IBOV": "Ibovespa"}
TINTA, TINTA_2, APAGADO = "#121212", "#52514e", "#898781"
GRADE, BASE, FUNDO = "#e1e0d9", "#c3c2b7", "#ffffff"
CONTEXTO = "#d6d4cd"
MODELO_FINAL = "Heurística com controle de giro"   # K = 20, λ escolhido na validação, 2 anos de treino, trimestral
FAIXA = "#ecebe5"
FONTE_DADOS = "Fonte: Yahoo Finance, out/2019 a set/2026. 20 trimestres fora da amostra (set/2021 a set/2026). Elaboração própria."

NOMES_CURTOS = {
    "Baseline: Top K Peso": "Top K por peso (baseline)",
    "Baseline: Top K Corr (1/K)": "Top K por correlação, 1/K (baseline)",
    "Top K Corr (QP Otimizado)": "Top K por correlação + pesos otimizados",
    "Lasso (L1 Positivo + QP)": "Lasso positivo + pesos otimizados",
    "Clustering Hierárquico + Repr.": "Clustering + representantes",
    "Heurística Gulosa + Swaps": "Heurística gulosa + trocas",
    "Heurística com controle de giro": "Heurística com controle de giro",
}


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


def pct(x, casas=0, sinal=False):
    return num(x, casas, sinal) + "%"


def eixo_pct(ax, eixo="y", casas=0):
    alvo = ax.yaxis if eixo == "y" else ax.xaxis
    alvo.set_major_formatter(FuncFormatter(lambda v, _: pct(v, casas)))


def quebrar(texto, largura_pol, tamanho, fator):
    maximo = max(int(largura_pol * 72 / (tamanho * fator)), 30)
    return "\n".join(textwrap.fill(p, maximo) if len(p) > maximo else p for p in texto.split("\n"))


def titulos(fig, titulo, subtitulo=None, fonte=FONTE_DADOS, base_pol=0.75, esquerda=0.07, direita=0.97, espaco_w=None):
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
    fig.savefig(PASTA_FIGURAS / nome, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Figura salva: {PASTA_FIGURAS / nome}")


ROTULOS_LINHAS = {
    "Heurística com controle de giro": "Modelo final",
    "Heurística Gulosa + Swaps": "Heurística",
    "Top K Corr (QP Otimizado)": "Top K correlação",
    "Baseline: Top K Peso": "Top K peso",
}


def rotular_fim_das_linhas(ax, finais, distancia):
    """Rótulos no fim de cada linha, afastados verticalmente para não se sobreporem."""
    finais = sorted(finais, key=lambda f: f[1])
    posicoes = []
    for x, y, texto in finais:
        y_rot = max(y, posicoes[-1] + distancia) if posicoes else y
        posicoes.append(y_rot)
        ax.annotate(texto, (x, y), xytext=(x, y_rot), textcoords="data", va="center", fontsize=10,
                    color=TINTA_2, annotation_clip=False)


def carregar_comparativo(K=20):
    sp = pd.read_csv(PASTA_RESULTADOS / f"tabela_comparativa_modelos_K{K}_sp100.csv").assign(indice="S&P100")
    ib = pd.read_csv(PASTA_RESULTADOS / f"tabela_comparativa_modelos_K{K}_ibov.csv").assign(indice="IBOV")
    return pd.concat([sp, ib], ignore_index=True)


def figura_comparativo(comp, K=20):
    ordem = (comp.groupby("Modelo")["TE (% a.a.)"].mean().sort_values(ascending=False).index.tolist())
    y = np.arange(len(ordem))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
    for ax, (coluna, rotulo_eixo, casas) in zip(axes, [("TE (% a.a.)", "Tracking error fora da amostra (% a.a.)", 1),
                                                        ("Giro Médio (% trim)", "Giro médio por trimestre (% da carteira)", 0)]):
        for desloc, indice in [(-0.19, "S&P100"), (0.19, "IBOV")]:
            v = comp[comp["indice"] == indice].set_index("Modelo").loc[ordem, coluna]
            barras = ax.barh(y + desloc, v, height=0.36, color=CORES[indice], label=NOMES[indice])
            for b, val in zip(barras, v):
                ax.text(b.get_width() + 0.25, b.get_y() + b.get_height() / 2, pct(val, casas), va="center",
                        fontsize=9.5, color=TINTA_2)
        ax.grid(axis="x", color=GRADE)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(rotulo_eixo)
        eixo_pct(ax, "x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([NOMES_CURTOS[m] for m in ordem], fontsize=11, color=TINTA)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    hg = comp[comp["Modelo"] == "Heurística com controle de giro"].set_index("indice")
    heur = comp[comp["Modelo"] == "Heurística Gulosa + Swaps"].set_index("indice")
    titulos(fig, "O controle de giro mantém o erro da heurística e leva o giro de 24–30% para menos de 10% por trimestre",
            f"Mesmos 20 trimestres, {K} ações. Heurística com controle de giro: TE de {pct(hg.loc['S&P100', 'TE (% a.a.)'], 1)} no S&P 100 e "
            f"{pct(hg.loc['IBOV', 'TE (% a.a.)'], 1)} no Ibovespa (sem controle: {pct(heur.loc['S&P100', 'TE (% a.a.)'], 1)} e "
            f"{pct(heur.loc['IBOV', 'TE (% a.a.)'], 1)}), trocando {pct(hg.loc['S&P100', 'Giro Médio (% trim)'], 1)} e "
            f"{pct(hg.loc['IBOV', 'Giro Médio (% trim)'], 1)} da carteira por trimestre, contra {pct(heur.loc['S&P100', 'Giro Médio (% trim)'])} e "
            f"{pct(heur.loc['IBOV', 'Giro Médio (% trim)'])}.",
            esquerda=0.27, espaco_w=0.12)
    salvar(fig, "31_comparativo_modelos.png")


def figura_fronteira():
    """Etapa 12: erro (TE) × giro por penalidade λ, um painel por índice. Cada ponto é a média de três sementes do solver nos 20 trimestres."""
    medias = pd.read_csv(PASTA_RESULTADOS / "validacao_lambda.csv")
    medias = medias[medias["periodo"] == "tudo"]
    escolhido = {"S&P100": 6.0, "IBOV": 3.0}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), sharey=True)
    dados = {}
    for ax, indice in zip(axes, ["S&P100", "IBOV"]):
        d = medias[medias["indice"] == indice].sort_values("lambda_anual")
        dados[indice] = d.set_index("lambda_anual")
        giro, te = d["Giro médio (% trim)"].to_numpy(), d["TE (% a.a.)"].to_numpy()
        ax.vlines(giro, d["TE_min"].to_numpy(), d["TE_max"].to_numpy(), color=CORES[indice], lw=1.6, alpha=0.55, zorder=2)
        ax.plot(giro, te, marker="o", ms=7, lw=2, color=CORES[indice], mec=FUNDO, mew=1.5, zorder=3)
        for _, l in d.iterrows():
            lam = l["lambda_anual"]
            if lam in (0.0, 3.0, 6.0, 12.0, 50.0):
                destaque = lam == escolhido[indice]
                ax.annotate(f"λ = {num(lam, 0)}", (l["Giro médio (% trim)"], l["TE (% a.a.)"]),
                            xytext={50.0: (9, 7), 0.0: (0, 11), 12.0: (-4, 10) if indice == "IBOV" else (0, -17)}.get(lam, (0, -17)),
                            textcoords="offset points", ha="left" if lam == 50.0 else "center",
                            fontsize=10 if destaque else 9.5, color=TINTA if destaque else TINTA_2,
                            fontweight="bold" if destaque else "normal")
                if destaque:
                    ax.scatter([l["Giro médio (% trim)"]], [l["TE (% a.a.)"]], s=190, facecolor="none", edgecolor=TINTA, linewidth=1.3, zorder=4)
        ax.set_xlim(-1, 32)
        ax.set_title(NOMES[indice])
        ax.set_xlabel("Giro médio por trimestre (% da carteira)")
        eixo_pct(ax, "x")
    axes[0].set_ylim(4.0, 6.1)
    axes[0].set_ylabel("Erro fora da amostra (TE, % ao ano)")
    eixo_pct(axes[0], "y", 1)
    sp, ib = dados["S&P100"], dados["IBOV"]
    titulos(fig, "Penalizar o giro corta as trocas em três quartos ou mais e muda pouco o erro até λ = 12",
            f"Cada ponto é uma penalidade λ (de 0 a 50), média de três execuções do solver nos mesmos 20 trimestres; a barra vertical vai do menor ao maior TE entre as execuções. "
            f"Sem penalidade, o MIQP troca {pct(sp.loc[0.0, 'Giro médio (% trim)'])} da carteira por trimestre no S&P 100 e {pct(ib.loc[0.0, 'Giro médio (% trim)'])} no Ibovespa; "
            f"com a penalidade escolhida (círculo), {pct(sp.loc[6.0, 'Giro médio (% trim)'])} e {pct(ib.loc[3.0, 'Giro médio (% trim)'])}.", base_pol=1.05)
    fig.subplots_adjust(top=fig.subplotpars.top - 0.06, wspace=0.12)
    salvar(fig, "32_fronteira_giro.png")


def figura_validacao_lambda(K=20):
    """Etapa 12: TE e giro por λ nos 20 trimestres, com a heurística com penalidade (determinística), o λ escolhido e o limite da regra."""
    from src.validacao_lambda import GRADE_LAMBDAS, TOLERANCIA_TE, escolher_lambda
    medias = pd.read_csv(PASTA_RESULTADOS / f"validacao_lambda_K{K}.csv")
    escolhido = escolher_lambda(medias)
    fmt = lambda v: f"{v:g}".replace(".", ",")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True)
    x = np.arange(len(GRADE_LAMBDAS))
    g = {}
    for j, indice in enumerate(["S&P100", "IBOV"]):
        d = medias[(medias["indice"] == indice) & (medias["periodo"] == "tudo")].sort_values("lambda_anual").set_index("lambda_anual")
        g[indice] = d["Giro médio (% trim)"]
        k = GRADE_LAMBDAS.index(escolhido[indice])
        for i, (col, ylabel) in enumerate([("TE (% a.a.)", "Erro (TE, % ao ano)"), ("Giro médio (% trim)", "Giro por trimestre (%)")]):
            ax = axes[i, j]
            ax.axvspan(k - 0.5, k + 0.5, color=FAIXA, zorder=0)
            ax.plot(x, d[col].to_numpy(), marker="o", ms=6.5, lw=2, color=CORES[indice], mec=FUNDO, mew=1.2, zorder=3)
            ax.plot([k], [d[col].iloc[k]], marker="o", ms=11, color="none", mec=TINTA, mew=2, zorder=4)
            if col.startswith("TE"):
                limite = d[col].min() + TOLERANCIA_TE
                ax.axhline(limite, color=TINTA_2, lw=1.1, ls=(0, (1, 2)), zorder=1)
                ax.text(len(x) - 0.55, limite + 0.02, f"menor TE + {fmt(TOLERANCIA_TE)} p.p.", ha="right", va="bottom", fontsize=10, color=TINTA_2)
                ax.set_ylim(min(d[col].min() - 0.15, 3.0), d[col].max() + 0.25)
                eixo_pct(ax, "y", 1)
                ax.set_title(NOMES[indice])
            else:
                ax.set_xlabel("Penalidade λ")
                eixo_pct(ax)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels([fmt(v) for v in GRADE_LAMBDAS])
    titulos(fig, f"Com \u03bb escolhido nos 20 trimestres, o giro cai de {g['S&P100'][0.0]:.0f}% e {g['IBOV'][0.0]:.0f}% para {g['S&P100'][escolhido['S&P100']]:.0f}% e {g['IBOV'][escolhido['IBOV']]:.0f}% por trimestre, e o erro quase n\u00e3o muda",
            f"Heur\u00edstica com penalidade, K = {K}. O c\u00edrculo e a faixa cinza marcam o \u03bb escolhido: {fmt(escolhido['S&P100'])} no S&P 100 e {fmt(escolhido['IBOV'])} no Ibovespa. "
            f"A regra escolhe o maior \u03bb (o que mais reduz o giro) com TE a at\u00e9 {fmt(TOLERANCIA_TE)} p.p. do menor da grade (abaixo da linha do painel de cima).", base_pol=1.05)
    fig.subplots_adjust(top=fig.subplotpars.top - 0.04, hspace=0.18, wspace=0.16)
    salvar(fig, "38_validacao_lambda.png")


def figura_erro_treino_limite():
    """Etapa 11, 'Como testamos': erro de treino da heurística e do MIQP, limite inferior do Gurobi e carteira sem limite, por K (mediana das 20 rodadas)."""
    q = pd.read_csv(PASTA_RESULTADOS / "resultados_por_trimestre.csv")
    q = q[q["janela treino"] == 504]
    Ks = [5, 10, 15, 20, 30]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True)
    LARG = 0.36
    resumo = {}
    for ax, indice in zip(axes, ["S&P100", "IBOV"]):
        g = q[q["indice"] == indice]
        heur = [g[(g["método"] == "Heurística") & (g["K"] == k)]["erro dentro (% a.a.)"].median() for k in Ks]
        miqp = [g[(g["método"] == "MIQP") & (g["K"] == k)]["erro dentro (% a.a.)"].median() for k in Ks]
        lim = [np.sqrt(g[(g["método"] == "MIQP") & (g["K"] == k)]["limite inferior"].astype(float).clip(lower=0) * 252).median() for k in Ks]
        piso = g[g["método"] == "Sem limite"]["erro dentro (% a.a.)"].median()
        x = np.arange(len(Ks))
        ax.bar(x - LARG / 2, heur, LARG, color=CORES[indice], zorder=3, label="Heurística")
        ax.bar(x + LARG / 2, miqp, LARG, color="#a39f97", zorder=3, label="MIQP (Gurobi por 10 s, a partir da heurística)")
        ax.scatter(x, lim, marker="v", s=75, color=TINTA, zorder=5, label="Limite inferior que o Gurobi prova")
        ax.axhline(piso, color=TINTA, lw=1.3, ls=(0, (4, 3)), zorder=4, label="Carteira com todas as ações (linha tracejada)")
        for xi, v in zip(x, heur):
            ax.text(xi - LARG / 2, v + 0.06, num(v, 1), ha="center", va="bottom", fontsize=9.5, color=CORES[indice], fontweight="semibold")
        for xi, v in zip(x, miqp):
            ax.text(xi + LARG / 2, v + 0.06, num(v, 1), ha="center", va="bottom", fontsize=9.5, color=TINTA_2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"K = {k}" for k in Ks])
        ax.set_title(NOMES[indice])
        ax.set_ylim(0, 6.6)
        ax.set_xlim(-0.6, len(Ks) - 0.4)
        eixo_pct(ax)
        ax.grid(axis="x", visible=False)
        resumo[indice] = dict(heur=heur[3], miqp=miqp[3], lim=lim[3], piso=piso)
    axes[0].set_ylabel("Erro de treino (TE, % ao ano)")
    axes[0].legend(loc="upper right", fontsize=9.5)
    r = resumo
    titulos(fig, "O Gurobi não melhora o que a heurística achou, e o limite que ele prova fica colado na carteira com todas as ações",
            f"Erro de treino, mediana das 20 rodadas. Com 20 ações, a heurística e o MIQP têm {num(r['S&P100']['heur'], 1)}% no S&P 100 e {num(r['IBOV']['heur'], 1)}% no Ibovespa, "
            f"enquanto o limite inferior provado é {num(r['S&P100']['lim'], 1)}% e {num(r['IBOV']['lim'], 1)}%, quase igual à carteira com todas as ações ({num(r['S&P100']['piso'], 1)}% e {num(r['IBOV']['piso'], 1)}%). "
            "Esse limite não enxerga a restrição de K, por isso o gap é grande mesmo quando não há melhora possível.", base_pol=1.05)
    fig.subplots_adjust(top=fig.subplotpars.top - 0.06, wspace=0.1)
    salvar(fig, "40_erro_treino_limite.png")


def figura_modelo_nulo(comp, K=20):
    """Lê as 1.000 carteiras aleatórias já sorteadas por `compare_all_models` (arquivos modelo_nulo_*)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    resumos = {}
    for ax, indice in zip(axes, ["S&P100", "IBOV"]):
        nome = "sp100" if indice == "S&P100" else "ibov"
        te_rand = np.load(PASTA_RESULTADOS / f"modelo_nulo_te_K{K}_{nome}.npy")
        resumos[indice] = pd.read_csv(PASTA_RESULTADOS / f"modelo_nulo_monte_carlo_K{K}_{nome}.csv").iloc[0].copy()
        te_mod = float(comp[(comp["indice"] == indice) & (comp["Modelo"] == MODELO_FINAL)]["TE (% a.a.)"].iloc[0])
        resumos[indice]["Percentual Superado (%)"] = 100 * float((te_rand > te_mod).mean())
        ax.hist(te_rand, bins=35, color=CORES_CLARAS[indice], edgecolor=FUNDO, linewidth=0.6)
        ax.axvline(te_mod, color=CORES[indice], lw=2.4)
        ax.axvline(np.median(te_rand), color=APAGADO, lw=1, ls="--")
        topo = ax.get_ylim()[1]
        ax.text(te_mod + 0.12, topo * 0.92, f"modelo final\n{pct(te_mod, 1)}", color=CORES[indice], fontsize=11,
                fontweight="semibold", va="top")
        ax.text(np.median(te_rand) + 0.12, topo * 0.92, f"mediana aleatória\n{pct(np.median(te_rand), 1)}",
                color=TINTA_2, fontsize=10, va="top")
        ax.set_title(NOMES[indice])
        ax.set_xlabel("Tracking error fora da amostra (% a.a.)")
        eixo_pct(ax, "x")
        ax.set_xlim(0, None)
    axes[0].set_ylabel("Carteiras aleatórias")
    titulos(fig, "Nenhuma das 1.000 carteiras sorteadas erra menos que o modelo",
            f"Cada carteira aleatória sorteia {K} ações a cada trimestre, otimiza os pesos no treino e é testada nos mesmos "
            f"20 trimestres do modelo final (heurística com controle de giro). O modelo supera {pct(resumos['S&P100']['Percentual Superado (%)'])} das carteiras "
            f"no S&P 100 e {pct(resumos['IBOV']['Percentual Superado (%)'])} no Ibovespa.", espaco_w=0.08)
    salvar(fig, "33_modelo_nulo.png")
    return resumos


def figura_regimes(K=20):
    r = pd.read_csv(PASTA_RESULTADOS / f"te_por_regime_modelos_K{K}.csv")
    regimes = ["Fim de 2021", "Alta de juros 2022", "Calmo 2023-2024", "2025-2026"]
    modelos = [MODELO_FINAL, "Heurística Gulosa + Swaps", "Top K Corr (QP Otimizado)", "Baseline: Top K Peso"]
    estilos = {MODELO_FINAL: dict(lw=2.6, marker="o"),
               "Heurística Gulosa + Swaps": dict(lw=1.4, marker="o", ls="--"),
               "Top K Corr (QP Otimizado)": dict(lw=1.4, marker="s", alpha=0.55),
               "Baseline: Top K Peso": dict(lw=1.4, marker="^", alpha=0.35)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), sharey=True)
    x = np.arange(len(regimes))
    for ax, indice in zip(axes, ["S&P100", "IBOV"]):
        finais = []
        for m in modelos:
            v = r[(r["indice"] == indice) & (r["modelo"] == m)].set_index("regime").loc[regimes, "TE (% a.a.)"]
            ax.plot(x, v, color=CORES[indice], ms=6, mec=FUNDO, **estilos[m])
            finais.append((x[-1] + 0.12, v.iloc[-1], ROTULOS_LINHAS[m]))
        rotular_fim_das_linhas(ax, finais, distancia=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(["set–dez\n2021", "Alta de juros\n2022", "Calmo\n2023–2024", "2025–\nset/2026"])
        ax.set_title(NOMES[indice])
        eixo_pct(ax, "y")
        ax.set_xlim(-0.3, len(regimes) - 0.2)
    axes[0].set_ylabel("Tracking error (% a.a.)")
    miqp = r[r["modelo"] == MODELO_FINAL].set_index(["indice", "regime"])["TE (% a.a.)"]
    titulos(fig, "Em 2022 o erro sobe no Ibovespa e fica estável no S&P 100; os modelos otimizados lideram em todos os regimes",
            f"Tracking error de cada modelo por regime, {K} ações. Modelo final no S&P 100: {pct(miqp.loc[('S&P100', 'Alta de juros 2022')], 1)} em 2022 × "
            f"{pct(miqp.loc[('S&P100', 'Calmo 2023-2024')], 1)} em 2023–2024; no Ibovespa, {pct(miqp.loc[('IBOV', 'Alta de juros 2022')], 1)} × "
            f"{pct(miqp.loc[('IBOV', 'Calmo 2023-2024')], 1)}. A Covid (2020) fica antes do primeiro trimestre de teste.",
            direita=0.88, espaco_w=0.30)
    salvar(fig, "34_te_por_regime.png")


def figura_diferenca_acumulada(K=20):
    s = pd.read_csv(PASTA_RESULTADOS / f"series_oos_modelos_K{K}.csv", parse_dates=["data"])
    modelos = [MODELO_FINAL, "Heurística Gulosa + Swaps", "Top K Corr (QP Otimizado)", "Baseline: Top K Peso"]
    estilos = {MODELO_FINAL: dict(lw=2.4),
               "Heurística Gulosa + Swaps": dict(lw=1.3, ls="--"),
               "Top K Corr (QP Otimizado)": dict(lw=1.3, alpha=0.55),
               "Baseline: Top K Peso": dict(lw=1.3, alpha=0.35)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), sharey=True)
    for ax, indice in zip(axes, ["S&P100", "IBOV"]):
        finais = []
        for m in modelos:
            d = s[(s["indice_nome"] == indice) & (s["modelo"] == m)].set_index("data").sort_index()
            dif = 100 * ((1 + d["carteira"]).cumprod() - (1 + d["indice"]).cumprod())
            ax.plot(dif.index, dif, color=CORES[indice], **estilos[m])
            finais.append((dif.index[-1] + pd.Timedelta(days=20), dif.iloc[-1], ROTULOS_LINHAS[m]))
        rotular_fim_das_linhas(ax, finais, distancia=11)
        ax.axhline(0, color=BASE, lw=1)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(NOMES[indice])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, 0, sinal=True) + " p.p."))
    axes[0].set_ylabel("Carteira − índice, retorno acumulado (p.p.)")
    titulos(fig, "No acumulado, as carteiras terminam acima do índice; o modelo final é o que oscila menos no caminho",
            "Diferença entre o retorno acumulado de cada carteira de " + str(K) + " ações e o do índice, antes de custos, desde set/2021. "
            "Acima de zero: a carteira rendeu mais que o índice, em boa parte por causa do viés de sobrevivência do universo. O Top K por peso usa o valor de mercado de hoje e por isso "
            "escolhe as vencedoras do período: a distância dele é viés de olhar o futuro, não mérito.",
            direita=0.88, espaco_w=0.30)
    salvar(fig, "35_diferenca_acumulada.png")


def figura_rebalanceamento(K=20):
    """Etapa 12: TE × giro anualizado das quatro configurações (janela × frequência), com a penalidade λ escolhida na validação."""
    from src.rebalanceamento_experiment import REFERENCIA, ler_resultados
    from src.validacao_lambda import lambda_escolhido
    todos = ler_resultados(K)
    escolhido = lambda_escolhido(K)
    nome_janela = {504: "2 anos", 252: "1 ano"}
    nome_freq = {63: "trimestral", 21: "mensal"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, indice in zip(axes, ["S&P100", "IBOV"]):
        g = todos[(todos["indice"] == indice) & (todos["lambda_anual"] == escolhido[indice])]
        for _, l in g.iterrows():
            atual = (int(l["janela_treino"]), int(l["horizonte_teste"])) == REFERENCIA
            x, y = l["Giro anualizado (%)"], l["TE (% a.a.)"]
            ax.plot(x, y, ls="", marker="o" if l["horizonte_teste"] == 63 else "s", ms=13 if atual else 10,
                    color=CORES_CLARAS[indice] if l["janela_treino"] == 252 else CORES[indice],
                    mec=TINTA if atual else CORES[indice], mew=2 if atual else 1.4)
            ax.annotate(f"{nome_janela[int(l['janela_treino'])]}, {nome_freq[int(l['horizonte_teste'])]}" + (" (base)" if atual else ""),
                        (x, y), xytext=(9, 7), textcoords="offset points", fontsize=10, color=TINTA)
        ax.set_title(f"{NOMES[indice]} (λ = {num(escolhido[indice], 1)})")
        ax.set_xlabel("Giro anualizado (% da carteira por ano)")
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        eixo_pct(ax, "x")
        eixo_pct(ax, "y", 1)
        ax.margins(x=0.45, y=0.25)
    axes[0].set_ylabel("Tracking error fora da amostra (% a.a.)")
    titulos(fig, "Rebalanceamento e janela de treino: o que cada escolha custa em erro e em giro",
            f"Cada ponto é uma configuração da heurística com controle de giro ({K} ações) nos mesmos pregões de teste (set/2021 a set/2026). "
            "Círculo: trimestral; quadrado: mensal. Cor escura: 2 anos de treino; clara: 1 ano. Contorno escuro: configuração base.",
            espaco_w=0.12)
    salvar(fig, "36_rebalanceamento_janela.png")


def figura_walk_forward(bases=None):
    """Etapa 10: como funciona o walk-forward, com as datas reais das 20 rodadas do S&P 100."""
    from src.data_loader import preparar_bases
    from src.walk_forward import construir_janelas
    bases = bases if bases is not None else preparar_bases()
    janelas = construir_janelas("S&P100", bases)
    fig, ax = plt.subplots(figsize=(12, 7.2))
    n = len(janelas)
    for i, j in enumerate(janelas):
        y = n - i
        ini_tr, fim_tr = j["datas_treino"][0], j["datas_treino"][-1]
        ini_te, fim_te = j["datas_teste"][0], j["datas_teste"][-1]
        ax.barh(y, mdates.date2num(fim_tr) - mdates.date2num(ini_tr), left=mdates.date2num(ini_tr), height=0.62,
                color=CONTEXTO, edgecolor=FUNDO, linewidth=0.6, zorder=2)
        ax.barh(y, mdates.date2num(fim_te) - mdates.date2num(ini_te) + 1, left=mdates.date2num(ini_te), height=0.62,
                color=CORES["S&P100"], edgecolor=FUNDO, linewidth=0.6, zorder=3)
    primeira, ultima = janelas[0], janelas[-1]
    x_tr = mdates.date2num(primeira["datas_treino"][0]) + (mdates.date2num(primeira["datas_treino"][-1]) - mdates.date2num(primeira["datas_treino"][0])) / 2
    ax.text(x_tr, n + 0.95, "treino: 504 pregões\n(cerca de 2 anos)", ha="center", va="bottom", fontsize=10.5, color=TINTA_2)
    x_te = mdates.date2num(primeira["datas_teste"][0]) + 30
    ax.text(x_te, n + 0.95, "teste: 63 pregões\n(1 trimestre)", ha="left", va="bottom", fontsize=10.5, color=CORES["S&P100"],
            fontweight="semibold")
    ax.text(mdates.date2num(pd.Timestamp("2020-02-01")), 4.2,
            "A cada rodada, a janela anda um trimestre:\no teste de uma rodada passa a fazer parte\ndo treino da seguinte.",
            ha="left", va="center", fontsize=10.5, color=TINTA_2, linespacing=1.4)
    ax.set_yticks([n - i for i in (0, 4, 9, 14, 19)])
    ax.set_yticklabels([f"Rodada {k}" for k in (1, 5, 10, 15, 20)])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(0.3, n + 2.6)
    ax.grid(axis="x", color=GRADE, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Data")
    titulos(fig, "Em cada rodada, o modelo aprende com os 2 anos anteriores e é testado no trimestre seguinte",
            "20 rodadas, de set/2021 a set/2026. As barras azuis são trimestres que o modelo não viu ao escolher a carteira; "
            "as cinzas são o histórico usado para escolhê-la. Nada do teste entra no treino.",
            fonte="Fonte: Yahoo Finance, out/2019 a set/2026. Datas do S&P 100; o Ibovespa começa duas semanas depois. Elaboração própria.")
    salvar(fig, "37_walk_forward.png")


def figura_erro_trimestral(K=20):
    """Etapa 14: TE e diferença de rentabilidade de cada trimestre do modelo final, nos dois índices."""
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11.4), sharex=True, gridspec_kw={"height_ratios": [1.5, 0.8, 0.8]})
    ax3.sharey(ax2)
    medias = {}
    for indice, axb in [("S&P100", ax2), ("IBOV", ax3)]:
        d = pd.read_csv(PASTA_RESULTADOS / f"rodadas_modelos_K{K}_{'sp100' if indice == 'S&P100' else 'ibov'}.csv",
                        parse_dates=["data_inicio", "data_fim"])
        d = d[d["modelo"] == MODELO_FINAL].sort_values("numero")
        meio = d["data_inicio"] + (d["data_fim"] - d["data_inicio"]) / 2
        te = d["TE_trimestre (% a.a.)"]
        dif = 100 * (d["retorno_bruto_carteira"] - d["retorno_indice"])
        medias[indice] = (te.mean(), dif.mean())
        ax.plot(meio, te, marker="o", ms=6, lw=2.2, color=CORES[indice], mec=FUNDO, mew=1.2, label=NOMES[indice])
        ax.axhline(te.mean(), color=CORES[indice], lw=1.2, ls=(0, (5, 3)))
        axb.bar(meio, dif, width=pd.Timedelta(days=38), color=CORES[indice], zorder=2)
        axb.axhline(dif.mean(), color=TINTA, lw=1.2, ls=(0, (5, 3)), zorder=3)
        axb.text(meio.iloc[-1] + pd.Timedelta(days=45), dif.mean(), f"média: {num(dif.mean(), 1, sinal=True)} p.p.", fontsize=9.5, va="center")
        axb.axhline(0, color=BASE, lw=0.8)
        axb.set_title(f"{NOMES[indice]}: carteira − índice em cada trimestre (acima de zero, a carteira rendeu mais)", fontsize=11.5)
        axb.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, 0, sinal=True) + " p.p."))
    ax.set_title("TE de cada trimestre: quanto a carteira oscila em torno do índice (quanto maior, pior a réplica)", fontsize=11.5)
    ax.set_ylabel("TE do trimestre (% a.a.)")
    eixo_pct(ax, "y")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    titulos(fig, "O erro do modelo final varia de trimestre a trimestre, e a direção da diferença muda",
            f"Heurística com controle de giro, {K} ações, treino de 2 anos. TE médio por trimestre (linhas tracejadas) de {pct(medias['S&P100'][0], 1)} no S&P 100 e "
            f"{pct(medias['IBOV'][0], 1)} no Ibovespa. Cada trimestre é uma carteira nova, testada sem ter visto os dias seguintes.")
    salvar(fig, "41_erro_trimestral_modelo_final.png")


def gerar_figuras(bases=None, K=20):
    from src.data_loader import preparar_bases
    bases = bases if bases is not None else preparar_bases()
    aplicar_estilo()
    PASTA_FIGURAS.mkdir(exist_ok=True)
    comp = carregar_comparativo(K)
    figura_comparativo(comp, K)
    figura_modelo_nulo(comp, K)
    figura_regimes(K)
    figura_diferenca_acumulada(K)
    figura_erro_trimestral(K)


if __name__ == "__main__":
    gerar_figuras()
