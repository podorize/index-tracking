"""Chance estimada de a diferença acumulada passar do limite da CVM (aprox. normal com TE e viés), por célula da grade de escolha."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.validacao_lambda import agregar_periodo

CORTE = pd.Timestamp("2024-10-01")


def calcular():
    linhas = []
    for f in Path("results/selecao").glob("K*.csv"):
        K, i, J, H, L = re.match(r"K(\d+)_(sp100|ibov)_J(\d+)_H(\d+)_L([\d.]+)\.csv", f.name).groups()
        H = int(H)
        r = pd.read_csv(f, parse_dates=["data_fim"])
        r = r[(r["dias"] == H)].sort_values("numero")
        a = agregar_periodo(r)
        te, mu = a["TE (% a.a.)"], r["TD_trimestre (p.p. a.a.)"].mean()
        p12 = norm.sf(4, mu, te) + norm.cdf(-4, mu, te)
        m6, s6 = mu * 60 / 252, te * np.sqrt(60 / 252)
        p60 = norm.sf(2, m6, s6) + norm.cdf(-2, m6, s6)
        p12_te = 2 * norm.sf(4, 0, te)                    # viés zero: só o TE
        p60_te = 2 * norm.sf(2, 0, s6)
        giro = r.loc[r["numero"] > r["numero"].min(), "giro"].mean() * 252 / H * 100
        linhas.append({"K": int(K), "ind": "S&P100" if i == "sp100" else "IBOV", "J": int(J), "H": H, "lam": float(L),
                       "TE": te, "vies": mu, "giro": giro, "p12": p12, "p60": p60, "p": max(p12, p60), "p12_te": p12_te, "p60_te": p60_te,
                       "K_efetivo": r["K_efetivo"].mean()})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    d = calcular()
    d.to_csv("results/selecao_prob_estimada.csv", index=False)
    old = pd.read_csv("results/selecao_prob_estimada.csv")
    m = old.merge(d, on=["K", "ind", "J", "H", "lam"], suffixes=("_o", "_n"))
    print(len(old), len(m), "max dif p12:", (m.p12_o - m.p12_n).abs().max(), "TE:", (m.TE_o - m.TE_n).abs().max(), "giro:", (m.giro_o - m.giro_n).abs().max())
