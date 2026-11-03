"""Regra final (chance média das janelas de 12 meses, λ fixo) (src/selecao_janelas_12m.py) e o comportamento no teste:
chance média das janelas de 12 meses inteiras no teste (mesmo cálculo) e fração observada dessas janelas acima de 4 p.p."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.criterio_cvm import janelas as diferencas
from src.figuras_relatorio_te import CENARIOS, INDICES, ARQ, estilo, titulos, salvar, num, CORES, CORES_CLARAS, NOMES, TINTA, TINTA_2, APAGADO
from src.selecao_janelas_12m import janelas, selecionar_fixo, CORTE

FONTE = "Fonte: Yahoo Finance. Período de escolha: out/2021 a set/2024; teste: out/2024 a set/2026 (janelas de 12 meses inteiras no teste). Elaboração própria."
MODELO = {(30, 252, 21, 25.0): "k30_l25", (50, 252, 21, 3.0): "k50_l3", (20, 252, 63, 6.0): "k20_tri_l6", (30, 252, 21, 3.0): "k30_l3",
          (15, 252, 63, 25.0): "k15_tri_l25", (20, 252, 21, 0.5): "k20_l05", (20, 252, 21, 3.0): "cvm20", (15, 252, 21, 3.0): "k15",
          (20, 504, 21, 3.0): "k20_2a", (30, 252, 21, 50.0): "k30", (30, 504, 21, 6.0): "k30_2a_l6", (20, 252, 63, 0.5): "k20_tri_l05", (999, 252, 21, 3.0): "sem_limite_l3", (30, 252, 63, 6.0): "k30_tri_l6", (50, 252, 63, 6.0): "k50_tri_l6"}


def ler(mod, indice, H):
    r = pd.read_csv(f"results/teste_final/{mod}_{ARQ[indice]}.csv", parse_dates=["data_fim"])
    return r[r.dias == H].sort_values("numero").reset_index(drop=True)


def no_teste(s, indice):
    H = int(s.H)
    mod = MODELO[(int(s.K), int(s.J), H, float(s.lam))]
    r = ler(mod, indice, H)
    assert float(r["lam"].iloc[0]) == float(s.lam) and int(r["janela"].iloc[0]) == int(s.J), (mod, indice)
    todas = ler("sem_limite", indice, 21)
    if H != 21:  # viés da carteira com todas as ações (mensal) na mesma janela trimestral: média dos três meses
        todas = pd.DataFrame({"data_fim": r.data_fim, "TD_trimestre (p.p. a.a.)": [
            todas.loc[(todas.data_fim > a) & (todas.data_fim <= b), "TD_trimestre (p.p. a.a.)"].mean()
            for a, b in zip(r.data_inicio.astype("datetime64[ns]") - pd.Timedelta(days=1), r.data_fim)]})
    n = 252 // H
    w = janelas(r, todas, H)
    teste = (r.data_fim > CORTE).to_numpy()
    inteira = np.array([teste[k:k + n].all() for k in range(len(r) - n + 1)])
    g = np.array(diferencas(r, n))[inteira]
    giro = r.loc[(r.data_fim > CORTE) & (r.numero > r.numero.min()), "giro"].mean() * 252 / H * 100
    return {"modelo": mod, "chance_teste": w.chance.to_numpy()[inteira].mean(), "obs": 100 * (np.abs(g) > 4).mean(),
            "n4": int((np.abs(g) > 4).sum()), "n": len(g), "giro_teste": giro, "pior": np.abs(g).max()}


def tabela():
    d = pd.read_csv("results/selecao_chance_12m.csv")
    linhas = []
    for c, (teto, _) in CENARIOS.items():
        for ind in INDICES:
            s = selecionar_fixo(d, ind, teto)
            linhas.append({"cenario": c, "teto": teto, "indice": ind, "K": int(s.K), "J": int(s.J), "H": int(s.H), "lam": s.lam,
                           "chance_val": s.chance, "giro_val": s.giro, **no_teste(s, ind)})
    return pd.DataFrame(linhas)


def rotulo(s):
    cfg = ("1 ano" if s.J == 252 else "2 anos") + (", mensal" if s.H == 21 else ", trimestral")
    return f"{s.cenario}\nK = {s.K}, λ = {num(s.lam, 1).replace(',0', '')}\n{cfg}"


def figura(t):
    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.0), sharey=True)
    w = 0.26
    for ax, ind in zip(axes, INDICES):
        s = t[t.indice == ind].reset_index(drop=True)
        x = np.arange(len(s))
        series = [("chance_val", "Chance média na validação", CORES_CLARAS[ind]),
                  ("chance_teste", "Chance média no teste (mesmo cálculo)", CORES[ind]),
                  ("obs", "Janelas acima de 4 p.p. no teste", "#5a5a55")]
        for j, (col, lab, cor) in enumerate(series):
            ax.bar(x + (j - 1) * w, s[col], w, color=cor, zorder=3, label=lab)
            for k, v in enumerate(s[col]):
                txt = f"{s.n4[k]}/{s.n[k]}" if col == "obs" else f"{num(v, 0)}%"
                ax.text(x[k] + (j - 1) * w, v + 1.0, txt, ha="center", fontsize=9, color=TINTA, fontweight="bold")
        for k in range(len(s)):
            ax.hlines(s.teto[k], k - 0.45, k + 0.45, color=TINTA_2, lw=1.3, ls=(0, (2, 2)), zorder=4)
            ax.text(k + 0.47, s.teto[k], f"{s.teto[k]}%", va="center", fontsize=8.5, color=TINTA_2)
        ax.set_xticks(x); ax.set_xticklabels([rotulo(r) for _, r in s.iterrows()], fontsize=9.5)
        ax.set_title(NOMES[ind]); ax.set_ylim(0, 80)
    axes[0].set_ylabel("%"); axes[0].legend(loc="upper left", fontsize=9.5)
    titulos(fig, "Com 1 ano de treino, rebalanceamento mensal e λ = 3, o risco máximo define o K",
            "Em cada janela de 12 meses, chance de passar de 4 p.p. com o erro e o viés próprio daquela janela; a chance da combinação é a média. "
            "Treino, rebalanceamento e λ fixos (menor chance média sobre K = 10 a 50); K = menor com chance dentro do risco máximo aceito (tracejado).",
            fonte=FONTE)
    salvar(fig, "rt_selecao_12m.png")


def curva():
    """Carteiras dos cenários pela regra final: chance prevista no período de escolha, a mesma conta no teste e janelas acima de 4 p.p."""
    from src.figuras_relatorio_te import CARTEIRAS
    d = pd.read_csv("results/selecao_chance_12m.csv")
    linhas = []
    for ind in INDICES:
        for K in sorted({v[1][ind] for v in CENARIOS.values()}, reverse=True):
            J, H, lam, _ = CARTEIRAS[(ind, K)]
            s = d[(d.ind == ind) & (d.K == K) & (d.J == J) & (d.H == H) & (d.lam == lam)].iloc[0]
            cen = " e ".join(c.lower() for c, v in CENARIOS.items() if v[1][ind] == K)
            linhas.append({"indice": ind, "K": K, "cenario": cen, "J": J, "H": H, "lam": lam, "chance_val": s.chance, **no_teste(s, ind)})
    return pd.DataFrame(linhas)


def figura_curva(c):
    """Uma seta por carteira de cenário: da chance prevista no período de escolha até a mesma conta no teste, com o limite do cenário
    (o mais exigente, quando a carteira serve a dois) e as janelas que de fato passaram de 4 p.p."""
    from matplotlib.lines import Line2D
    estilo()
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    c = c.reset_index(drop=True)
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
    titulos(fig, "No teste, a chance subiu no S&P 100 e caiu no Ibovespa; só a carteira de 50 ações do S&P 100 passou do limite",
            "Cada seta vai da chance prevista no período de escolha até a mesma conta feita com os dados do teste. À direita, quantas janelas "
            "de 12 meses de fato passaram de 4 p.p. No S&P 100, as carteiras são refeitas a cada trimestre e têm só 4 janelas no teste.",
            fonte=FONTE, direita=0.84, esquerda=0.17)
    salvar(fig, "rt_chance_por_K_regra_final.png")


def figura_decomposicao():
    """Por que a chance mudou no teste: decomposição de Shapley da mudança entre o erro de rastreamento (largura da curva) e o
    desvio médio (centro), a partir de results/decomposicao_chance.csv; as duas partes são escaladas para somar a mudança observada."""
    from matplotlib.patches import Patch
    estilo()
    d = pd.read_csv("results/decomposicao_chance.csv")
    ordem = [("S&P100", 50), ("S&P100", 30), ("S&P100", 20), ("IBOV", 999), ("IBOV", 30)]
    d = d.set_index(["ind", "K"]).loc[ordem].reset_index()
    esc = d.delta / d.aprox
    d["eTE"], d["emu"] = d.dTE * esc, d.dmu * esc
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
    ax.set_xlim(-8, 8); ax.set_ylim(-0.6, len(d) + 0.3); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mudança da chance entre o período de escolha e o teste (p.p.)")
    ax.legend(handles=[Patch(color=APAGADO, label="Parte explicada pelo erro de rastreamento (largura da curva)"),
                       Patch(facecolor="#dcdad2", hatch="////", edgecolor="white", label="Parte explicada pelo desvio médio (centro da curva)")],
              loc="upper left", fontsize=9.5, frameon=False)
    titulos(fig, "No Ibovespa, a chance caiu porque o erro de rastreamento diminuiu; no S&P 100, subiu porque as carteiras oscilaram e se afastaram mais do índice",
            "Decomposição de Shapley da mudança da chance entre os dois ingredientes da conta: o erro de rastreamento e o desvio médio em relação "
            "ao índice. Traço preto: mudança total. À direita, a chance e os dois ingredientes no período de escolha e no teste.",
            fonte=FONTE, direita=0.66, esquerda=0.17)
    salvar(fig, "rt_decomposicao.png")


if __name__ == "__main__":
    t = tabela()
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
    figura(t)
    c = curva()
    print(c.round(2).to_string(index=False))
    figura_curva(c)
