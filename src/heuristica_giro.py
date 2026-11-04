"""
heuristica_giro.py
Heurística gulosa + trocas (a mesma da Etapa 11) com a penalidade de giro do MIQP da Etapa 12.

O objetivo avaliado para um conjunto S de ações é o do MIQP:
    erro de réplica no treino  +  γ · (fração vendida da carteira anterior),
com os pesos de S obtidos por um QP contínuo (pesos ≥ 0, soma 1). Com γ = 0 o QP é o da heurística base, e o resultado coincide
com `selecao_gulosa` + `busca_trocas` do notebook (ver `tests`/validação no relatório).

Duas partidas são tentadas e vence a de menor objetivo penalizado:
  - "vazio": seleção gulosa a partir da carteira vazia, como na heurística base, seguida da busca por trocas;
  - "anterior": busca por trocas a partir das ações da carteira anterior (completada, se faltar, pela seleção gulosa).

Um único modelo do Gurobi (contínuo) é reaproveitado: o conjunto de ações liberadas muda só pelos limites superiores dos pesos,
o que leva cerca de 0,4 ms por QP.
"""

import time

import numpy as np
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # só é preciso para recalcular os modelos; ler os resultados funciona sem o Gurobi
    gp = GRB = None

from src.optimizer import ambiente_gurobi


class QPPenalizado:
    """Pesos ótimos de um conjunto de ações sob o erro de réplica mais a penalidade de giro."""

    def __init__(self, prob, w_ant, gamma):
        self.prob, self.gamma = prob, float(gamma)
        n = prob.n
        w_ant = w_ant or {}
        self.p = np.array([float(w_ant.get(t, 0.0)) for t in prob.tickers])
        # peso das ações da carteira anterior que não são mais candidatas: vendido de qualquer forma
        self.fora = float(sum(v for t, v in w_ant.items() if t not in set(prob.tickers)))
        m = gp.Model(env=ambiente_gurobi())
        m.Params.Threads = 1
        self.w = m.addMVar(n, lb=0.0, ub=0.0)
        self.idx = np.where(self.p > 0)[0]
        self.s = m.addMVar(len(self.idx), lb=0.0)
        m.addConstr(self.w.sum() == 1.0)
        if len(self.idx):
            m.addConstr(self.s >= self.p[self.idx] - self.w[self.idx])
        penalidade = self.gamma * (self.s.sum() + self.fora) if self.gamma > 0 else 0.0
        m.setObjective(self.w @ prob.Q @ self.w - 2 * prob.c @ self.w + prob.k0 + penalidade, GRB.MINIMIZE)
        self.m = m
        self.avaliacoes = 0

    def resolver(self, conjunto):
        ub = np.zeros(self.prob.n)
        ub[list(conjunto)] = 1.0
        self.w.UB = ub
        self.m.optimize()
        self.avaliacoes += 1
        if self.m.Status != GRB.OPTIMAL:
            return None, np.inf
        return np.asarray(self.w.X), float(self.m.ObjVal)


def _gulosa(qp, inicio, k):
    sel = list(inicio)
    while len(sel) < k:
        melhor_obj, melhor_j = np.inf, None
        for j in range(qp.prob.n):
            if j in sel:
                continue
            _, obj = qp.resolver(sel + [j])
            if obj < melhor_obj:
                melhor_obj, melhor_j = obj, j
        if melhor_j is None:
            break
        sel.append(melhor_j)
    return sel


def _trocas(qp, sel, max_passos=50):
    sel = list(sel)
    _, obj = qp.resolver(sel)
    for _ in range(max_passos):
        melhor_obj, melhor_sel = obj, None
        fora = [j for j in range(qp.prob.n) if j not in sel]
        for i in range(len(sel)):
            for j in fora:
                cand = sel[:i] + [j] + sel[i + 1:]
                _, o = qp.resolver(cand)
                if o < melhor_obj - 1e-12:
                    melhor_obj, melhor_sel = o, cand
        if melhor_sel is None:
            break
        obj, sel = melhor_obj, melhor_sel
    w, obj = qp.resolver(sel)
    return sel, w, obj


def heuristica_giro(prob, K, w_ant=None, gamma=0.0, partidas=("vazio", "anterior")):
    """Devolve (pesos sobre prob.tickers, informações). `w_ant` é o dicionário {ticker: peso depois de derivar}."""
    t0 = time.perf_counter()
    qp = QPPenalizado(prob, w_ant, gamma)
    if K is None:                      # sem limite de cardinalidade: um único QP com todas as candidatas liberadas
        w, obj = qp.resolver(range(prob.n))
        return w, list(range(prob.n)), {"partida": "sem limite", "objetivo": obj, "tempo_s": time.perf_counter() - t0,
                                        "avaliacoes_qp": qp.avaliacoes, "objetivos": {"sem limite": obj}}
    resultados = {}
    if "vazio" in partidas:
        resultados["vazio"] = _trocas(qp, _gulosa(qp, [], K))
    if "anterior" in partidas and len(qp.idx):
        previas = [int(j) for j in qp.idx]
        previas = sorted(previas, key=lambda j: -qp.p[j])[:K]
        resultados["anterior"] = _trocas(qp, _gulosa(qp, previas, K))
    partida = min(resultados, key=lambda a: resultados[a][2])
    sel, w, obj = resultados[partida]
    info = {"partida": partida, "objetivo": obj, "tempo_s": time.perf_counter() - t0, "avaliacoes_qp": qp.avaliacoes,
            "objetivos": {a: r[2] for a, r in resultados.items()}}
    return w, sel, info
