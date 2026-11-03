"""Medidas comuns: erro e desvio de um período, diferença acumulada em janelas e a chance de passar do limite em 12 meses."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import LIMITE_12M
from src.optimizer import ANUAL


def rodadas(arquivo, H):
    """Rodadas de uma simulação, só as completas (H pregões), em ordem."""
    r = pd.read_csv(arquivo, parse_dates=["data_inicio", "data_fim"])
    return r[r["dias"] == H].sort_values("numero").reset_index(drop=True)


def agregar_periodo(r):
    """TE e TD de rodadas consecutivas, reconstruídos exatamente a partir da média e da variância de cada rodada."""
    n = r["dias"].to_numpy(dtype=float)
    sigma = r["TE_trimestre (% a.a.)"].to_numpy() / (100 * np.sqrt(ANUAL))
    mu = r["TD_trimestre (p.p. a.a.)"].to_numpy() / (100 * ANUAL)
    media = (n * mu).sum() / n.sum()
    soma_quadrados = ((n - 1) * sigma ** 2).sum() + (n * (mu - media) ** 2).sum()
    return {"TE (% a.a.)": 100 * np.sqrt(soma_quadrados / (n.sum() - 1)) * np.sqrt(ANUAL), "TD (p.p. a.a.)": 100 * media * ANUAL}


def diferencas(r, n):
    """Diferença acumulada carteira − índice (p.p.) em cada janela de n rodadas consecutivas."""
    p, i = 1 + r["retorno_liq_carteira"].to_numpy(), 1 + r["retorno_indice"].to_numpy()
    return np.array([100 * (p[k:k + n].prod() - i[k:k + n].prod()) for k in range(len(r) - n + 1)])


def chance(mu, te):
    """Chance (%) de a diferença de 12 meses passar de ±4 p.p., supondo uma normal de centro mu e largura te."""
    return 100 * (norm.cdf((-LIMITE_12M - mu) / te) + norm.sf((LIMITE_12M - mu) / te))


def janelas(r, todas, H):
    """Chance de cada janela móvel de 12 meses. Largura: erro de rastreamento contra o índice. Centro: desvio da carteira menos
    o da carteira com todas as ações (`todas`, mesma configuração e mesmas rodadas), que desconta o viés de sobrevivência."""
    n = 252 // H
    m = r.merge(todas[["data_fim", "TD_trimestre (p.p. a.a.)"]], on="data_fim", suffixes=("", "_todas"))
    out = []
    for i in range(len(m) - n + 1):
        w = m.iloc[i:i + n]
        a = agregar_periodo(w)
        mu = a["TD (p.p. a.a.)"] - w["TD_trimestre (p.p. a.a.)_todas"].mean()
        out.append({"fim": w.data_fim.iloc[-1], "TE": a["TE (% a.a.)"], "mu": mu, "chance": chance(mu, a["TE (% a.a.)"])})
    return pd.DataFrame(out)
