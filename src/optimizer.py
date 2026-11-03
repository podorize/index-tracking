"""
optimizer.py
Implementação do modelo de otimização de Index Tracking com:
  1. Restrição de cardinalidade (máximo K ativos)
  2. Penalidade de Turnover na função objetivo (Pareto TE vs. Giro)
  3. Formulação MIQP exata e compacta via Gurobi (compatível com qualquer licença)
  4. Heurística e Warm Start integrados
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # o modo instantâneo do notebook não precisa do Gurobi
    gp = GRB = None

_AMBIENTE_GUROBI = None


def ambiente_gurobi():
    """Cria o ambiente do Gurobi na primeira vez que o MIQP é resolvido."""
    global _AMBIENTE_GUROBI
    if gp is None:
        raise ImportError("gurobipy não instalado: execute `pip install gurobipy` para resolver o MIQP.")
    if _AMBIENTE_GUROBI is None:
        _AMBIENTE_GUROBI = gp.Env(empty=True)
        _AMBIENTE_GUROBI.setParam("OutputFlag", 0)
        _AMBIENTE_GUROBI.start()
    return _AMBIENTE_GUROBI

PENALIDADE_SOMA = 1e3
LIMIAR_PESO = 1e-4
ANUAL = 252


class ProblemaTracking:
    """
    Representa a matriz de retornos de treino e a função de perda quadrática (MSE diário em %²):
        f(w) = w' Q w - 2 c' w + k0
    """
    def __init__(self, R_treino, y_treino):
        X = 100.0 * R_treino.to_numpy(dtype=float)
        a = 100.0 * y_treino.to_numpy(dtype=float)
        self.tickers = list(R_treino.columns)
        self.n, self.T = X.shape[1], len(a)
        self.X, self.a = X, a
        self.Q = X.T @ X / self.T
        self.c = X.T @ a / self.T
        self.k0 = float(a @ a / self.T)
        self.G = X.T @ X + (PENALIDADE_SOMA ** 2)
        self.h = X.T @ a + (PENALIDADE_SOMA ** 2)

    def erro(self, idx, w):
        idx = list(idx)
        Qs = self.Q[np.ix_(idx, idx)]
        return max(float(w @ Qs @ w - 2 * self.c[idx] @ w + self.k0), 0.0)

    def qp(self, idx):
        idx = list(idx)
        Gs, hs = self.G[np.ix_(idx, idx)], self.h[idx]
        try:
            w = np.linalg.solve(Gs, hs)
            if w.min() < 0:
                A = np.vstack([self.X[:, idx], PENALIDADE_SOMA * np.ones(len(idx))])
                b = np.r_[self.a, PENALIDADE_SOMA]
                w, _ = nnls(A, b)
        except np.linalg.LinAlgError:
            A = np.vstack([self.X[:, idx], PENALIDADE_SOMA * np.ones(len(idx))])
            b = np.r_[self.a, PENALIDADE_SOMA]
            w, _ = nnls(A, b)

        soma = w.sum()
        w = w / soma if soma > 0 else np.ones(len(idx)) / len(idx)
        return w, self.erro(idx, w)


def resolver_miqp_turnover(
    prob,
    K,
    w_drift_dict=None,
    gamma=0.0,
    w_inicial=None,
    tempo_max=10.0,
    threads=4,
    mip_gap=0.01,
    seed=42
):
    """
    Resolve o MIQP com restrição de cardinalidade K e penalidade de turnover.
    Usa formulação compacta exata:
        Apenas as ações da carteira anterior podem ser vendidas.
        Portanto, Turnover = sum_{j in S_prev} s_j + w_deslistadas,
        onde s_j >= w_drift_j - w_j, com s_j >= 0.
        Essa formulação adiciona apenas |S_prev| <= K variáveis contínuas!
    """
    m = gp.Model(env=ambiente_gurobi())
    m.Params.TimeLimit = float(tempo_max)
    m.Params.MIPGap = mip_gap
    m.Params.Threads = threads
    m.Params.Seed = seed

    w = m.addMVar(prob.n, lb=0.0, ub=1.0, name="w")
    z = m.addMVar(prob.n, vtype=GRB.BINARY, name="z")

    m.addConstr(w.sum() == 1.0, name="investimento_total")
    m.addConstr(w <= z, name="liga_peso_e_selecao")
    m.addConstr(z.sum() <= K, name="cardinalidade")

    obj_expr = w @ prob.Q @ w - 2 * prob.c @ w + prob.k0

    if gamma > 0.0 and w_drift_dict is not None and len(w_drift_dict) > 0:
        # Identifica ações prévias presentes nos candidatos atuais
        idx_prev = []
        w_drift_prev = []
        w_deslistadas = 0.0

        for t_nome, peso in w_drift_dict.items():
            if t_nome in prob.tickers:
                idx_prev.append(prob.tickers.index(t_nome))
                w_drift_prev.append(float(peso))
            else:
                w_deslistadas += float(peso)

        if len(idx_prev) > 0:
            k_prev = len(idx_prev)
            s = m.addMVar(k_prev, lb=0.0, ub=1.0, name="s")
            w_drift_vec = np.array(w_drift_prev)
            # s_j >= w_drift_j - w_j
            m.addConstr(s >= w_drift_vec - w[idx_prev], name="venda_parcial")
            obj_expr = obj_expr + gamma * (s.sum() + w_deslistadas)
        elif w_deslistadas > 0:
            obj_expr = obj_expr + gamma * w_deslistadas

    m.setObjective(obj_expr, GRB.MINIMIZE)

    # Warm-start
    if w_inicial is not None:
        w0 = np.where(w_inicial > LIMIAR_PESO, w_inicial, 0.0)
        soma = w0.sum()
        if soma > 0:
            w0 = w0 / soma
            w.Start = w0
            z.Start = (w0 > 0).astype(float)

    m.optimize()

    status_str = "OPTIMAL" if m.Status == GRB.OPTIMAL else ("TIME_LIMIT" if m.Status == GRB.TIME_LIMIT else str(m.Status))
    info = {
        "status": status_str,
        "tempo_s": m.Runtime,
        "nos": m.NodeCount,
        "gap (%)": np.nan,
        "obj_val": np.nan,
        "erro_treino_pct2": np.nan,
    }

    if m.SolCount == 0:
        return None, info

    info["gap (%)"] = 100 * max(m.ObjVal - m.ObjBound, 0.0) / max(abs(m.ObjVal), 1e-12)
    info["obj_val"] = m.ObjVal

    w_otimo = np.asarray(w.X)
    idx_ativos = np.where(w_otimo > LIMIAR_PESO)[0]
    w_limpo = np.zeros(prob.n)
    if len(idx_ativos) > 0:
        w_limpo[idx_ativos] = w_otimo[idx_ativos] / w_otimo[idx_ativos].sum()
    else:
        w_limpo = w_otimo

    info["erro_treino_pct2"] = prob.erro(range(prob.n), w_limpo)
    return w_limpo, info
