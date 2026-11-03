"""Aplica a regra de CRITERIOS_SELECAO_FINAL.md aos resultados de results/selecao/ (período de escolha). Só rodadas completas até 01/10/2024."""
import re
from pathlib import Path

import pandas as pd

from src.validacao_lambda import agregar_periodo, GRADE_LAMBDAS

SAIDA = Path("results/selecao")
TOL, LIMITE = 0.2, 4.0


def carregar():
    linhas = []
    for f in SAIDA.glob("K*.csv"):
        K, i, J, H, L = re.match(r"K(\d+)_(sp100|ibov)_J(\d+)_H(\d+)_L([\d.]+)\.csv", f.name).groups()
        r = pd.read_csv(f)
        r = r[r["dias"] == int(H)]                        # só rodadas completas
        giro = r.loc[r["numero"] > r["numero"].min(), "giro"].mean()
        linhas.append({"K": int(K), "indice": "S&P100" if i == "sp100" else "IBOV", "janela": int(J), "horizonte": int(H), "lam": float(L),
                       "TE": agregar_periodo(r)["TE (% a.a.)"], "giro_anual": 100 * giro * 252 / int(H), "rodadas": len(r)})
    return pd.DataFrame(linhas)


def aplicar(d):
    op = []
    for (K, i, J, H), g in d.groupby(["K", "indice", "janela", "horizonte"]):
        g = g.sort_values("lam")
        ok = g[g["TE"] <= g["TE"].min() + TOL]
        op.append(ok.iloc[-1])                           # maior λ dentro da tolerância
    op = pd.DataFrame(op).reset_index(drop=True)
    cfg = op.pivot_table(index=["K", "janela", "horizonte"], columns="indice", values=["TE", "giro_anual", "lam"])
    return op, cfg


if __name__ == "__main__":
    d = carregar()
    print("rodadas por célula:", d.groupby(["janela", "horizonte", "indice"])["rodadas"].agg(["min", "max"]).to_string(), sep="\n")
    op, cfg = aplicar(d)
    pd.set_option("display.width", 200)
    print("\nPonto de operação por K e configuração:\n", cfg.round(2).to_string())
    passa = (cfg["TE"] < LIMITE).all(axis=1)
    Ks = sorted(passa[passa].reset_index()["K"].unique())
    Kstar = Ks[0] if Ks else 30
    print("\nK* =", Kstar, "(passam:", Ks, ")")
    c = cfg.loc[Kstar].copy()
    minimo = c["TE"].min()
    c["candidata"] = ((c["TE"] - minimo) <= TOL).all(axis=1)
    c["TE_medio"] = c["TE"].mean(axis=1)
    c["giro_medio"] = c["giro_anual"].mean(axis=1)
    print(c.round(2).to_string())
    cand = c[c["candidata"]]
    escolhida = cand["giro_medio"].idxmin() if len(cand) else c["TE_medio"].idxmin()
    print("\nConfiguração escolhida (janela, horizonte):", escolhida, "| candidatas:", list(cand.index))
    print("λ por índice:", c.loc[escolhida, "lam"].to_dict())
    op.to_csv("results/selecao_pontos_de_operacao.csv", index=False)
