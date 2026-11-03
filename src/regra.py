"""Regra final de escolha, separada por índice, sobre a grade do período de escolha.

Etapa 1: a forma de montar a carteira (histórico, frequência e λ) de menor chance média das sete carteiras (5 a 50 ações e sem limite).
Etapa 2: com essa forma fixa, o menor K cuja chance mais um erro padrão cabe no risco máximo aceito do cenário; se nenhum couber,
a carteira sem limite. A chance de cada carteira é a média das chances das janelas de 12 meses (src/metricas.py)."""
import re

import numpy as np
import pandas as pd

from src.config import CENARIOS, GRADE_K, INDICES, RESULTADOS, SEM_LIMITE
from src.grade import SAIDA as GRADE
from src.metricas import janelas, rodadas

PADRAO = re.compile(r"K(\d+)_(sp100|ibov)_J(\d+)_H(\d+)_L([\d.]+)\.csv")


def _grade():
    """(K, índice, J, H, λ, rodadas da carteira, rodadas da carteira sem limite na mesma configuração) de cada arquivo da grade."""
    for f in sorted(GRADE.glob("K*.csv")):
        K, i, J, H, L = PADRAO.match(f.name).groups()
        H = int(H)
        yield (int(K), "S&P100" if i == "sp100" else "IBOV", int(J), H, float(L), rodadas(f, H),
               rodadas(GRADE / f"K{SEM_LIMITE}_{i}_J{J}_H{H}_L{L}.csv", H))


def janelas_todas(refazer=False):
    """Chance de cada janela de 12 meses em toda a grade (cache em results/janelas_12m.csv)."""
    arq = RESULTADOS / "janelas_12m.csv"
    if arq.exists() and not refazer:
        return pd.read_csv(arq, parse_dates=["fim"])
    w = pd.concat([janelas(r, todas, H).assign(K=K, ind=ind, J=J, H=H, lam=lam) for K, ind, J, H, lam, r, todas in _grade()])
    w.to_csv(arq, index=False)
    return w


def grade(w):
    """Uma linha por carteira da grade: chance média, erro e desvio médios das janelas e giro anual."""
    linhas = []
    for K, ind, J, H, lam, r, _ in _grade():
        x = w[(w.K == K) & (w.ind == ind) & (w.J == J) & (w.H == H) & (w.lam == lam)]
        giro = r.loc[r["numero"] > r["numero"].min(), "giro"].mean() * 252 / H * 100
        linhas.append({"K": K, "ind": ind, "J": J, "H": H, "lam": lam, "chance": x.chance.mean(), "chance_max": x.chance.max(),
                       "TE": x.TE.mean(), "mu": x.mu.mean(), "giro": giro, "n_janelas": len(x)})
    d = pd.DataFrame(linhas)
    d.to_csv(RESULTADOS / "selecao_chance_12m.csv", index=False)
    return d


def erro_padrao(x, bloco, n=2000, semente=0):
    """Erro padrão da média por bootstrap circular em blocos de janelas vizinhas (que se sobrepõem)."""
    rng = np.random.default_rng(semente)
    x = np.asarray(x); T = len(x); nb = int(np.ceil(T / bloco))
    medias = [x[np.concatenate([(s + np.arange(bloco)) % T for s in rng.integers(0, T, nb)])[:T]].mean() for _ in range(n)]
    return float(np.std(medias))


def combinacoes(w, d):
    """Etapa 1: chance média das sete carteiras em cada combinação de histórico, frequência e λ, com erro padrão e giro médio."""
    linhas = []
    for (ind, J, H, lam), g in w.groupby(["ind", "J", "H", "lam"]):
        s = g.groupby("fim").chance.mean()
        linhas.append({"ind": ind, "J": J, "H": H, "lam": lam, "chance": s.mean(), "ep": erro_padrao(s.values, 12 if H == 21 else 4)})
    giro = d.groupby(["ind", "J", "H", "lam"]).giro.mean().rename("giro").reset_index()
    c = pd.DataFrame(linhas).merge(giro, on=["ind", "J", "H", "lam"])
    c.to_csv(RESULTADOS / "selecao_configuracao_indice.csv", index=False)
    return c


def configuracao(c):
    """Etapa 1, por índice: a combinação de menor chance média. Devolve {índice: (J, H, λ)}."""
    return {ind: tuple(x.sort_values("chance").iloc[0][["J", "H", "lam"]]) for ind, x in c.groupby("ind")}


def chance_por_K(w, ind, J, H, lam):
    """Etapa 2: chance de cada K na configuração do índice, com erro padrão; a curva é forçada a não subir com o K."""
    linhas = []
    for K in GRADE_K:
        s = w[(w.ind == ind) & (w.K == K) & (w.J == J) & (w.H == H) & (w.lam == lam)].sort_values("fim").chance
        linhas.append({"K": K, "chance": s.mean(), "ep": erro_padrao(s.values, 12 if H == 21 else 4)})
    t = pd.DataFrame(linhas)
    t["chance"] = np.minimum.accumulate(t.chance.values)  # um tamanho maior nunca tem chance maior
    t["limite"] = t.chance + t.ep
    return t


def selecionar_regra(t, teto):
    """Menor K cuja chance mais um erro padrão cabe no risco; se nenhum couber, a carteira sem limite (cabe = False)."""
    ok = t[t.limite <= teto + 1e-9]
    s = (ok.iloc[0] if len(ok) else t[t.K == SEM_LIMITE].iloc[0]).copy()
    s["cabe"] = bool(len(ok))
    return s


def escolher(refazer=False):
    """Aplica a regra e grava results/cenarios.csv (uma linha por cenário e índice)."""
    w = janelas_todas(refazer)
    c = combinacoes(w, grade(w))
    cfg = configuracao(c)
    linhas = []
    for ind in INDICES:
        J, H, lam = cfg[ind]
        t = chance_por_K(w, ind, J, H, lam)
        for cen, teto in CENARIOS.items():
            s = selecionar_regra(t, teto)
            linhas.append({"cenario": cen, "teto": teto, "ind": ind, "K": int(s.K), "J": int(J), "H": int(H), "lam": lam,
                           "chance": s.chance, "ep": s.ep, "limite": s.limite, "cabe": s.cabe})
    e = pd.DataFrame(linhas)
    e.to_csv(RESULTADOS / "cenarios.csv", index=False)
    return e


def cenarios():
    """{cenário: (risco máximo, {índice: K})}, lido de results/cenarios.csv."""
    e = pd.read_csv(RESULTADOS / "cenarios.csv")
    return {cen: (teto, dict(zip(g.ind, g.K))) for (cen, teto), g in e.groupby(["cenario", "teto"], sort=False)}


def configuracoes():
    """{índice: (J, H, λ)} da regra, lido de results/cenarios.csv."""
    e = pd.read_csv(RESULTADOS / "cenarios.csv").drop_duplicates("ind")
    return {r.ind: (int(r.J), int(r.H), float(r.lam)) for r in e.itertuples()}


def carteiras():
    """Carteiras dos cenários e extras, como (índice, K, J, H, λ), do maior para o menor K em cada índice."""
    from src.config import EXTRAS
    cen, cfg = cenarios(), configuracoes()
    return [(ind, K, *cfg[ind]) for ind in INDICES
            for K in sorted({v[1][ind] for v in cen.values()} | set(EXTRAS.get(ind, [])), reverse=True)]
