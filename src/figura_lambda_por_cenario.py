"""λ por cenário pela regra original (menor giro que cabe no risco máximo aceito, na validação) e o comportamento das chances no teste."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.figuras_relatorio_te import (CENARIOS, INDICES, ARQ, GRADE_L, estilo, titulos, salvar, num, serie_12m,
                                      CORES, CORES_CLARAS, NOMES, TINTA, TINTA_2)

FONTE = "Fonte: Yahoo Finance. Validação: set/2021 a set/2024; teste: out/2024 a set/2026 (janelas inteiras no teste). Elaboração própria."
# modelo do teste final para (K, λ); o arquivo traz o λ de cada índice
MODELO = {(30, 50.0): "k30", (30, 12.0): "k30", (30, 3.0): "k30_l3", (50, 3.0): "k50", (20, 3.0): "cvm20",
          (15, 12.0): "k15_l12", (15, 3.0): "k15"}


def lambda_da_regra(d, K, indice, teto):
    g = d[(d.ind == indice) & (d.K == K) & (d.J == 252) & (d.H == 21)].set_index("lam").reindex(GRADE_L)
    ok = g[g.p12_te <= teto / 100 + 1e-9]
    lam = ok.sort_values("giro").index[0]
    return lam, ok.loc[lam, "p12_te"] * 100, ok.loc[lam, "giro"]


def no_teste(K, lam, indice):
    mod = MODELO[(K, lam)]
    r = pd.read_csv(f"results/teste_final/{mod}_{ARQ[indice]}.csv", parse_dates=["data_fim"])
    assert float(r["lam"].iloc[0]) == lam, (mod, indice, r["lam"].iloc[0], lam)
    r = r[r.dias == 21].sort_values("numero")
    te_rodadas = r.loc[r.data_fim > pd.Timestamp("2024-10-01"), "TE_trimestre (% a.a.)"]
    te = float(np.sqrt((te_rodadas ** 2).mean()))
    _, g = serie_12m(mod, ARQ[indice])
    giro = r.loc[(r.data_fim > pd.Timestamp("2024-10-01")), "giro"].mean() * 12 * 100
    return {"TE": te, "chance_te": 200 * norm.sf(4 / te), "obs": 100 * (np.abs(g) > 4).mean(),
            "n4": int((np.abs(g) > 4).sum()), "n": len(g), "giro": giro}


def tabela():
    d = pd.read_csv("results/selecao_prob_estimada.csv")
    linhas = []
    for c, (teto, ks) in CENARIOS.items():
        for ind in INDICES:
            K = ks[ind]
            lam, ch, giro_v = lambda_da_regra(d, K, ind, teto)
            t = no_teste(K, lam, ind)
            t3 = no_teste(K, 3.0, ind)
            linhas.append({"cenario": c, "teto": teto, "indice": ind, "K": K, "lam": lam, "chance_val": ch, "giro_val": giro_v,
                           **{f"teste_{k}": v for k, v in t.items()}, "obs_l3": t3["obs"], "TE_l3": t3["TE"]})
    return pd.DataFrame(linhas)


def figura(t):
    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
    w = 0.26
    for ax, ind in zip(axes, INDICES):
        s = t[t.indice == ind].reset_index(drop=True)
        x = np.arange(len(s))
        series = [("chance_val", "Chance estimada na validação", CORES_CLARAS[ind]),
                  ("teste_chance_te", "Chance estimada com o erro do teste", CORES[ind]),
                  ("teste_obs", "Janelas acima de 4 p.p. no teste", "#5a5a55")]
        for j, (col, lab, cor) in enumerate(series):
            ax.bar(x + (j - 1) * w, s[col], w, color=cor, zorder=3, label=lab)
            for k, v in enumerate(s[col]):
                txt = f"{s.teste_n4[k]}/{s.teste_n[k]}" if col == "teste_obs" else f"{num(v, 0)}%"
                ax.text(x[k] + (j - 1) * w, v + 1.0, txt, ha="center", fontsize=9, color=TINTA, fontweight="bold")
        ax.scatter(x + w, s.obs_l3, marker="D", s=46, facecolor="white", edgecolor=TINTA, zorder=5, label="Janelas acima de 4 p.p. com λ = 3")
        for k in range(len(s)):
            ax.hlines(s.teto[k], k - 0.45, k + 0.45, color=TINTA_2, lw=1.3, ls=(0, (2, 2)), zorder=4)
            ax.text(k + 0.47, s.teto[k], f"{s.teto[k]}%", va="center", fontsize=8.5, color=TINTA_2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\nK = {K}, λ = {num(l, 1).replace(',0', '')}" for c, K, l in zip(s.cenario, s.K, s.lam)], fontsize=10)
        ax.set_title(NOMES[ind]); ax.set_ylim(0, 80)
    axes[0].set_ylabel("%")
    axes[0].legend(loc="upper left", fontsize=9.5)
    titulos(fig, "Com o λ de menor giro, a carteira conservadora do S&P 100 é a que mais se afasta do previsto",
            "λ de cada cenário pela regra original: o de menor giro entre os que cabem no risco máximo aceito, na validação (1 ano de treino, mensal). "
            "Barras: chance estimada na validação, chance estimada com o erro de rastreamento do teste e fração de janelas de 12 meses acima de 4 p.p. no teste. "
            "Losango: fração observada com λ = 3. Tracejado (com o valor à direita): risco máximo aceito.",
            fonte=FONTE)
    salvar(fig, "rt_lambda_por_cenario.png")


if __name__ == "__main__":
    t = tabela()
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
    figura(t)
