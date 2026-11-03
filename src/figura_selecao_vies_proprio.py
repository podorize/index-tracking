"""Seleção por cenário e índice com o viés próprio (viés da carteira menos o da carteira com todas as ações, na validação):
K = menor número de ações com alguma combinação dentro do risco máximo aceito; configuração e λ = a de menor giro entre as que cabem.
Compara a chance estimada na validação com a chance pelo erro e pelo viés próprio do teste e com a fração observada de janelas acima de 4 p.p."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.figuras_relatorio_te import (CENARIOS, INDICES, ARQ, estilo, titulos, salvar, num, serie_12m,
                                      CORES, CORES_CLARAS, NOMES, TINTA, TINTA_2)

CORTE = pd.Timestamp("2024-10-01")
FONTE = "Fonte: Yahoo Finance. Validação: set/2021 a set/2024; teste: out/2024 a set/2026 (janelas inteiras no teste). Elaboração própria."
# modelo do teste final para (K, janela, horizonte, λ); o λ de cada índice é conferido no arquivo
MODELO = {(30, 252, 21, 50.0): "k30", (30, 252, 21, 12.0): "k30", (30, 252, 21, 3.0): "k30_l3", (50, 252, 21, 3.0): "k50",
          (20, 252, 21, 3.0): "cvm20", (15, 252, 21, 12.0): "k15_l12", (15, 252, 21, 3.0): "k15", (20, 504, 21, 3.0): "k20_2a"}


def chance(mu, te):
    return 100 * (norm.cdf((-4 - mu) / te) + norm.sf((4 - mu) / te))


def grade_com_vies_proprio():
    d = pd.read_csv("results/selecao_prob_estimada.csv")
    todas = d[d.K == 999][["ind", "J", "H", "lam", "vies"]].rename(columns={"vies": "vies_todas"})
    d = d.merge(todas, on=["ind", "J", "H", "lam"])
    d["mu"] = d.vies - d.vies_todas
    d["chance"] = [chance(m, t) for m, t in zip(d.mu, d.TE)]
    return d


def selecionar(d, indice, teto):
    for K in (5, 10, 15, 20, 30, 50, 999):
        ok = d[(d.ind == indice) & (d.K == K) & (d.chance <= teto + 1e-9)]
        if len(ok):
            return ok.sort_values(["giro", "chance"]).iloc[0]
    return None


def rodadas_teste(mod, indice):
    r = pd.read_csv(f"results/teste_final/{mod}_{ARQ[indice]}.csv", parse_dates=["data_fim"])
    r = r[r.dias == 21].sort_values("numero")
    return r, r[r.data_fim > CORTE]


def no_teste(sel, indice):
    mod = MODELO[(int(sel.K), int(sel.J), int(sel.H), float(sel.lam))]
    r, t = rodadas_teste(mod, indice)
    assert float(r["lam"].iloc[0]) == float(sel.lam) and int(r["janela"].iloc[0]) == int(sel.J), (mod, indice)
    _, todas = rodadas_teste("sem_limite", indice)
    te = float(np.sqrt((t["TE_trimestre (% a.a.)"] ** 2).mean()))
    mu = t["TD_trimestre (p.p. a.a.)"].mean() - todas["TD_trimestre (p.p. a.a.)"].mean()
    _, g = serie_12m(mod, ARQ[indice])
    return {"modelo": mod, "TE_teste": te, "mu_teste": mu, "chance_teste": chance(mu, te),
            "obs": 100 * (np.abs(g) > 4).mean(), "n4": int((np.abs(g) > 4).sum()), "n": len(g),
            "giro_teste": t.loc[t.numero > r.numero.min(), "giro"].mean() * 12 * 100}


def tabela():
    d = grade_com_vies_proprio()
    linhas = []
    for c, (teto, _) in CENARIOS.items():
        for ind in INDICES:
            s = selecionar(d, ind, teto)
            linhas.append({"cenario": c, "teto": teto, "indice": ind, "K": int(s.K), "J": int(s.J), "H": int(s.H), "lam": s.lam,
                           "chance_val": s.chance, "mu_val": s.mu, "TE_val": s.TE, "giro_val": s.giro, **no_teste(s, ind)})
    return pd.DataFrame(linhas)


def rotulo(s):
    janela = "1 ano" if s.J == 252 else "2 anos"
    return f"{s.cenario}\nK = {s.K}, {janela}, λ = {num(s.lam, 1).replace(',0', '')}"


def figura(t):
    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), sharey=True)
    w = 0.26
    for ax, ind in zip(axes, INDICES):
        s = t[t.indice == ind].reset_index(drop=True)
        x = np.arange(len(s))
        series = [("chance_val", "Chance estimada na validação", CORES_CLARAS[ind]),
                  ("chance_teste", "Chance com o erro e o viés próprio do teste", CORES[ind]),
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
    titulos(fig, "Com o viés próprio e o λ de menor giro, a chance da validação subestima o risco no S&P 100",
            "Seleção por cenário e índice na validação: menor K com alguma combinação dentro do risco máximo aceito e, entre as que cabem, a de menor giro. "
            "Chance com o viés próprio (viés da carteira menos o da carteira com todas as ações). Tracejado: risco máximo aceito.",
            fonte=FONTE)
    salvar(fig, "rt_selecao_vies_proprio.png")


if __name__ == "__main__":
    t = tabela()
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
    figura(t)
