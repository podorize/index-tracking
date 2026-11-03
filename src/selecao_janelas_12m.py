"""Seleção por cenário e índice com a chance média das janelas de 12 meses da validação.

Em cada janela móvel de 12 meses (12 rodadas mensais ou 4 trimestrais), a chance de a diferença acumulada passar de 4 p.p. usa o erro
de rastreamento e o viés próprio daquela janela (viés da carteira menos o da carteira com todas as ações, mesma configuração e λ).
A chance da combinação é a média dessas chances. Regra: menor K com alguma combinação dentro do risco máximo aceito; entre as que
cabem, a de menor giro."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.validacao_lambda import agregar_periodo
from src.figura_selecao_vies_proprio import chance

CORTE = pd.Timestamp("2024-10-01")


def rodadas(f, H):
    r = pd.read_csv(f, parse_dates=["data_fim"])
    return r[r["dias"] == H].sort_values("numero").reset_index(drop=True)


def janelas(r, todas, H):
    """Chance de cada janela móvel de 12 meses; `todas` é a carteira com todas as ações nas mesmas rodadas."""
    n = 252 // H
    m = r.merge(todas[["data_fim", "TD_trimestre (p.p. a.a.)"]], on="data_fim", suffixes=("", "_todas"))
    out = []
    for i in range(len(m) - n + 1):
        w = m.iloc[i:i + n]
        a = agregar_periodo(w)
        mu = a["TD (p.p. a.a.)"] - w["TD_trimestre (p.p. a.a.)_todas"].mean()
        out.append({"fim": w.data_fim.iloc[-1], "TE": a["TE (% a.a.)"], "mu": mu, "chance": chance(mu, a["TE (% a.a.)"])})
    return pd.DataFrame(out)


def grade():
    linhas = []
    for f in Path("results/selecao").glob("K*.csv"):
        K, i, J, H, L = re.match(r"K(\d+)_(sp100|ibov)_J(\d+)_H(\d+)_L([\d.]+)\.csv", f.name).groups()
        H = int(H)
        r = rodadas(f, H)
        todas = rodadas(f"results/selecao/K999_{i}_J{J}_H{H}_L{L}.csv", H)
        w = janelas(r, todas, H)
        giro = r.loc[r["numero"] > r["numero"].min(), "giro"].mean() * 252 / H * 100
        linhas.append({"K": int(K), "ind": "S&P100" if i == "sp100" else "IBOV", "J": int(J), "H": H, "lam": float(L),
                       "chance": w.chance.mean(), "chance_max": w.chance.max(), "TE": w.TE.mean(), "mu": w.mu.mean(),
                       "giro": giro, "n_janelas": len(w)})
    return pd.DataFrame(linhas)


def selecionar_fixo(d, indice, teto, J=252, H=21, lam=3.0):
    """Regra final: treino, rebalanceamento e λ fixos (menor chance média sobre K = 10 a 50); K = menor com chance dentro do risco."""
    for K in (5, 10, 15, 20, 30, 50, 999):
        ok = d[(d.ind == indice) & (d.K == K) & (d.J == J) & (d.H == H) & (d.lam == lam) & (d.chance <= teto + 1e-9)]
        if len(ok):
            return ok.iloc[0]
    return None


def janelas_todas():
    """Chance de cada janela de 12 meses em todas as combinações da grade (cache em results/janelas_12m.csv)."""
    arq = Path("results/janelas_12m.csv")
    if arq.exists():
        return pd.read_csv(arq, parse_dates=["fim"])
    partes = []
    for f in Path("results/selecao").glob("K*.csv"):
        K, i, J, H, L = re.match(r"K(\d+)_(sp100|ibov)_J(\d+)_H(\d+)_L([\d.]+)\.csv", f.name).groups()
        H = int(H)
        w = janelas(rodadas(f, H), rodadas(f"results/selecao/K999_{i}_J{J}_H{H}_L{L}.csv", H), H)
        partes.append(w.assign(K=int(K), ind="S&P100" if i == "sp100" else "IBOV", J=int(J), H=H, lam=float(L)))
    w = pd.concat(partes)
    w.to_csv(arq, index=False)
    return w


def erro_padrao(x, bloco, n=2000, semente=0):
    """Erro padrão da média por bootstrap circular em blocos de janelas vizinhas (que se sobrepõem)."""
    rng = np.random.default_rng(semente)
    x = np.asarray(x); T = len(x); nb = int(np.ceil(T / bloco))
    medias = [x[np.concatenate([(s + np.arange(bloco)) % T for s in rng.integers(0, T, nb)])[:T]].mean() for _ in range(n)]
    return float(np.std(medias))


def combinacoes(w, d):
    """Etapa 1, por índice: chance média das sete carteiras (5 a 50 ações e sem limite) em cada combinação de histórico,
    frequência e λ, com o erro padrão dessa média e o giro médio."""
    linhas = []
    for (ind, J, H, lam), g in w.groupby(["ind", "J", "H", "lam"]):
        s = g.groupby("fim").chance.mean()
        linhas.append({"ind": ind, "J": J, "H": H, "lam": lam, "chance": s.mean(), "ep": erro_padrao(s.values, 12 if H == 21 else 4)})
    giro = d.groupby(["ind", "J", "H", "lam"]).giro.mean().rename("giro").reset_index()
    return pd.DataFrame(linhas).merge(giro, on=["ind", "J", "H", "lam"])


def configuracao(c):
    """Regra final (04/11/2026), etapa 1: em cada índice, a combinação de histórico, frequência e λ de menor chance média das
    sete carteiras. Devolve {índice: (J, H, λ)}."""
    return {ind: tuple(x.sort_values("chance").iloc[0][["J", "H", "lam"]]) for ind, x in c.groupby("ind")}


def chance_por_K(w, ind, J, H, lam):
    """Etapa 2: chance de cada número de ações na configuração do índice, com erro padrão; a curva é forçada a não subir com o K."""
    linhas = []
    for K in (5, 10, 15, 20, 30, 50, 999):
        s = w[(w.ind == ind) & (w.K == K) & (w.J == J) & (w.H == H) & (w.lam == lam)].sort_values("fim").chance
        linhas.append({"K": K, "chance": s.mean(), "ep": erro_padrao(s.values, 12 if H == 21 else 4)})
    t = pd.DataFrame(linhas)
    t["chance"] = np.minimum.accumulate(t.chance.values)  # curva que só cai: um tamanho maior nunca tem chance maior
    t["limite"] = t.chance + t.ep
    return t


def selecionar_regra(t, teto):
    """Menor número de ações cuja chance mais um erro padrão cabe no risco; se nenhum couber, a carteira sem limite (cabe = False)."""
    ok = t[t.limite <= teto + 1e-9]
    s = (ok.iloc[0] if len(ok) else t[t.K == 999].iloc[0]).copy()
    s["cabe"] = bool(len(ok))
    return s


def selecionar(d, indice, teto):
    for K in (5, 10, 15, 20, 30, 50, 999):
        ok = d[(d.ind == indice) & (d.K == K) & (d.chance <= teto + 1e-9)]
        if len(ok):
            return ok.sort_values(["giro", "chance"]).iloc[0]
    return None


if __name__ == "__main__":
    from src.figuras_relatorio_te import CENARIOS, INDICES
    d = grade()
    d.to_csv("results/selecao_chance_12m.csv", index=False)
    pd.set_option("display.width", 250)
    for ind in INDICES:
        print(ind, "chance média 12m (1 ano/mensal) por K × λ")
        print(d[(d.ind == ind) & (d.J == 252) & (d.H == 21)].pivot(index="K", columns="lam", values="chance").round(1).to_string())
    for c, (teto, _) in CENARIOS.items():
        for ind in INDICES:
            s = selecionar(d, ind, teto)
            print(c, teto, ind, s[["K", "J", "H", "lam", "chance", "chance_max", "TE", "mu", "giro"]].round(2).to_dict())
