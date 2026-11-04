"""Dados do app (app.py): para cada índice, na forma de montar escolhida pela regra, as carteiras de todos os tamanhos da grade
(5 a 50 ações e sem limite) simuladas do início do período de escolha ao fim dos dados, com retornos diários e pesos de cada rodada,
mais a curva de chance por tamanho da escolha e os números de cada carteira no teste. Grava results/app/dados_app.pkl.

    python -m src.app_dados                 # 14 carteiras; refaz só as que faltam no cache
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import cache
from src.config import CORTE, CUSTOS, GRADE_K, INDICES, RESULTADOS, SEM_LIMITE
from src.data_loader import preparar_bases
from src.grade import intervalo_de_teste
from src.metricas import agregar_periodo, diferencas, janelas
from src.walk_forward import simular_walk_forward

SAIDA = RESULTADOS / "app"
ARQUIVO = SAIDA / "dados_app.pkl"


def simular(c):
    """Walk-forward de uma carteira, como em src/grade.py, guardando os retornos diários e os pesos."""
    ind, K, J, H, lam = c
    bases = preparar_bases()
    inicio, ultimo = intervalo_de_teste(ind, bases)
    _, r, serie, pesos = simular_walk_forward(nome_indice=ind, bases=bases, K=None if K == SEM_LIMITE else K, lambda_anual=lam,
                                              custo_transacao=CUSTOS[ind], janela_treino=J, horizonte_teste=H, inicio_teste=inicio,
                                              fim_teste=ultimo, metodo="heuristica", retornar_series=True, retornar_pesos=True)
    return c, {"rodadas": r, "diario": diario_liquido(r, serie), "pesos": pesos}


def diario_liquido(r, serie):
    """Retornos diários da carteira com o custo de cada rebalanceamento descontado no primeiro pregão da rodada, de modo que
    o retorno acumulado de cada rodada bata com o retorno líquido da tabela de rodadas."""
    d = serie.copy()
    for x in r.itertuples():
        dias = d.loc[x.data_inicio:x.data_fim].index
        if len(dias):
            fator = 1 - x.custo_friccao / (1 + x.retorno_bruto_carteira)
            d.loc[dias[0], "carteira"] = (1 + d.loc[dias[0], "carteira"]) * fator - 1
    return d


def completas(r, H):
    return r[r["dias"] == H].sort_values("numero").reset_index(drop=True)


def no_teste(r, todas, ind, H):
    """Os números da carteira no teste, com a mesma conta de src/avaliacao.py."""
    r, todas, n = completas(r, H), completas(todas, H), 252 // H
    t = r[r.data_fim > CORTE]
    teste = (r.data_fim > CORTE).to_numpy()
    ok = np.array([teste[k:k + n].all() for k in range(len(r) - n + 1)])
    w = janelas(r, todas, H)[ok]
    g = np.abs(diferencas(r, n)[ok])
    giro = t.loc[t.numero > r.numero.min(), "giro"].mean() * 252 / H * 100
    return {"chance_teste": w.chance.mean(), "TE_teste": agregar_periodo(t)["TE (% a.a.)"], "n4": int((g > 4).sum()),
            "n_janelas": len(g), "pior": g.max(), "giro_teste": giro, "custo_por_milhao": giro / 100 * CUSTOS[ind] * 1e6,
            "ret_carteira": 100 * ((1 + t.retorno_liq_carteira).prod() - 1), "ret_indice": 100 * ((1 + t.retorno_indice).prod() - 1)}


def gerar(processos=6):
    from src.regra import chance_por_K, configuracoes, janelas_todas
    cfg = configuracoes()
    tarefas = [(ind, K, *cfg[ind]) for ind in INDICES for K in GRADE_K]
    feitas = {}
    if ARQUIVO.exists():
        for ind, d in cache.ler(ARQUIVO)["indices"].items():
            c = d["config"]
            feitas.update({(ind, K, c["J"], c["H"], c["lam"]): v for K, v in d["carteiras"].items()})
    feitas = {c: v for c, v in feitas.items() if c in tarefas}
    falta = [c for c in tarefas if c not in feitas]
    print(f"App: {len(feitas)} de {len(tarefas)} carteiras prontas.", flush=True)
    if falta:
        with Pool(processos) as p:
            for c, v in p.imap_unordered(simular, falta):
                feitas[c] = v
                print("pronto", c, flush=True)
    w = janelas_todas()
    indices = {}
    for ind in INDICES:
        J, H, lam = cfg[ind]
        curva = chance_por_K(w, ind, J, H, lam)
        todas = feitas[(ind, SEM_LIMITE, J, H, lam)]["rodadas"]
        teste = pd.DataFrame([{"K": K, **no_teste(feitas[(ind, K, J, H, lam)]["rodadas"], todas, ind, H)} for K in GRADE_K])
        indices[ind] = {"config": {"J": J, "H": H, "lam": lam}, "curva": curva.merge(teste, on="K"),
                        "carteiras": {K: feitas[(ind, K, J, H, lam)] for K in GRADE_K}}
    SAIDA.mkdir(parents=True, exist_ok=True)
    cache.gravar({"corte": CORTE, "indices": indices}, ARQUIVO)
    print(f"Gravado {ARQUIVO}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processos", type=int, default=6)
    gerar(ap.parse_args().processos)
