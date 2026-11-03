"""
selecao_final.py
Grade K × configuração × λ no PERÍODO DE ESCOLHA (rodadas que terminam até 01/10/2024), conforme CRITERIOS_SELECAO_FINAL.md.
Reaproveita rodadas já calculadas (o walk-forward é determinístico e uma execução mais curta é o prefixo da longa) e roda só o que falta,
sem calcular nada depois do corte. Um arquivo por (K, índice, janela, horizonte, λ) em results/selecao/.
"""

import argparse
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from src.data_loader import preparar_bases
from src.validacao_lambda import GRADE_LAMBDAS, CUSTOS
from src.rebalanceamento_experiment import intervalo_de_teste
from src.walk_forward import simular_walk_forward

PASTA = Path("results")
SAIDA = PASTA / "selecao"
CORTE = pd.Timestamp("2024-10-01")
SEM_LIMITE = 999                  # K = 999 nos nomes de arquivo: sem restrição de cardinalidade (K=None no modelo)
GRADE_K = [5, 10, 15, 20, 30, 50, SEM_LIMITE]
CONFIGURACOES = [(504, 63), (504, 21), (252, 63), (252, 21)]
INDICES = ("S&P100", "IBOV")


def nome(indice):
    return "sp100" if indice == "S&P100" else "ibov"


def arquivo(K, indice, janela, horizonte, lam):
    return SAIDA / f"K{K}_{nome(indice)}_J{janela}_H{horizonte}_L{lam:g}.csv"


def existente(K, indice, janela, horizonte, lam):
    """Rodadas já calculadas por outro experimento, ou None."""
    if (janela, horizonte) == (504, 63):
        f = PASTA / f"validacao_lambda_K{K}_{nome(indice)}_rodadas.csv"
        if f.exists():
            r = pd.read_csv(f)
            r = r[r["lambda_anual"] == lam]
            return r if len(r) else None
        return None
    f = PASTA / f"rebalanceamento_K{K}_rodadas_{nome(indice)}_J{janela}_H{horizonte}_L{lam:g}.csv"
    return pd.read_csv(f) if f.exists() else None


def recortar(r):
    r = r.copy()
    r["data_fim"] = pd.to_datetime(r["data_fim"])
    return r[r["data_fim"] <= CORTE].sort_values("numero")


def tarefa(args):
    K, indice, janela, horizonte, lam = args
    arq = arquivo(K, indice, janela, horizonte, lam)
    if arq.exists():
        return "pronto", args
    r = existente(K, indice, janela, horizonte, lam)
    if r is not None:
        recortar(r).to_csv(arq, index=False)
        return "reaproveitado", args
    bases = preparar_bases()
    inicio, _ = intervalo_de_teste(indice, bases)   # mesmo primeiro pregão de teste da referência, para todas as janelas
    _, rodadas = simular_walk_forward(
        nome_indice=indice, bases=bases, K=None if K == SEM_LIMITE else K, lambda_anual=lam, custo_transacao=CUSTOS[indice],
        janela_treino=janela, horizonte_teste=horizonte, inicio_teste=inicio, fim_teste=CORTE, metodo="heuristica",
    )
    rodadas = rodadas.assign(indice=indice, lambda_anual=lam, K=K, janela_treino=janela, horizonte_teste=horizonte)
    recortar(rodadas).to_csv(arq, index=False)
    return "calculado", args


def todas():
    return [(K, i, j, h, lam) for K, (j, h), i, lam in product(GRADE_K, CONFIGURACOES, INDICES, GRADE_LAMBDAS)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processos", type=int, default=7)
    ap.add_argument("--contar", action="store_true")
    a = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)
    tarefas = [t for t in todas() if not arquivo(*t).exists()]
    falta = [t for t in tarefas if existente(*t) is None]
    print(f"{len(todas())} células; {len(todas()) - len(tarefas)} prontas; {len(tarefas) - len(falta)} a reaproveitar; {len(falta)} a calcular.", flush=True)
    if a.contar:
        raise SystemExit
    with Pool(a.processos) as p:
        for n, (estado, t) in enumerate(p.imap_unordered(tarefa, tarefas), 1):
            print(f"[{n}/{len(tarefas)}] {estado} {t}", flush=True)
