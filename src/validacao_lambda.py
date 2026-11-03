"""
validacao_lambda.py
Escolha da penalidade λ do controle de giro (Etapa 12 do relatório).

Desenho declarado antes dos resultados:
  - o modelo com controle de giro (heurística gulosa + trocas com a penalidade, determinística) roda nos 20 trimestres de teste de sempre,
    para cada λ da grade e para cada K da sensibilidade (K = 10, 20 e 30; K = 20 é o principal);
  - λ é escolhido com os 20 trimestres, pela regra: o MAIOR λ da grade (o que mais reduz o giro) cujo TE fica a até 0,2 p.p. do menor TE
    da grade. Como o λ é escolhido nos mesmos trimestres em que o modelo é medido, o efeito é pequeno (o erro varia pouco com λ);
  - os períodos de validação (8 primeiros trimestres) e de teste (12 últimos) ficam no resumo só como verificação.
Os resultados por rodada são gravados a cada execução (um arquivo por K e índice), então uma interrupção não perde o que já foi feito.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data_loader import preparar_bases
from src.walk_forward import simular_walk_forward

PASTA_RESULTADOS = Path("results")

CUSTOS = {"S&P100": 0.0005, "IBOV": 0.0015}
GRADE_LAMBDAS = [0.0, 0.5, 1.5, 3.0, 6.0, 12.0, 25.0, 50.0]
LAMBDA_ATUAL = {"S&P100": 6.0, "IBOV": 6.0}   # usados se ainda não houver λ escolhido (valores finais para K = 20)
K_PADRAO = 20
N_VALIDACAO = 8          # primeiras rodadas = validação; as demais = teste
TOLERANCIA_TE = 0.2      # p.p. a.a. acima do menor TE da grade
ANUAL = 252


def arquivo_rodadas(K, indice):
    nome = "sp100" if indice == "S&P100" else "ibov"
    return PASTA_RESULTADOS / f"validacao_lambda_K{K}_{nome}_rodadas.csv"


def executar_validacao(K=K_PADRAO, indices=("S&P100", "IBOV"), bases=None, grade=None, metodo="heuristica", tempo_max=10.0):
    bases = bases if bases is not None else preparar_bases()
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    grade = grade or GRADE_LAMBDAS
    for indice in indices:
        arq = arquivo_rodadas(K, indice)
        feitas = pd.read_csv(arq) if arq.exists() else pd.DataFrame()
        ja = set(feitas["lambda_anual"].unique()) if len(feitas) else set()
        blocos = [feitas] if len(feitas) else []
        pendentes = [lam for lam in grade if lam not in ja]
        print(f"K={K} {indice}: {len(grade)} valores de λ, {len(grade) - len(pendentes)} já feitos, {len(pendentes)} a fazer.", flush=True)
        for n, lam in enumerate(pendentes, 1):
            t0 = time.time()
            metricas, rodadas = simular_walk_forward(
                nome_indice=indice, bases=bases, K=K, lambda_anual=lam, custo_transacao=CUSTOS[indice],
                tempo_max=tempo_max, metodo=metodo,
            )
            rodadas = rodadas.assign(indice=indice, lambda_anual=lam, K=K, metodo=metodo)
            blocos.append(rodadas)
            pd.concat(blocos, ignore_index=True).to_csv(arq, index=False)
            print(f"[{n}/{len(pendentes)}] K={K} {indice} λ={lam:g}: TE {metricas['TE (% a.a.)']:.2f}%, "
                  f"giro {metricas['Giro médio (% trim)']:.1f}% por trimestre ({time.time() - t0:.0f} s)", flush=True)


def ler_rodadas(K=K_PADRAO):
    partes = [pd.read_csv(arquivo_rodadas(K, i)) for i in ("S&P100", "IBOV") if arquivo_rodadas(K, i).exists()]
    return pd.concat(partes, ignore_index=True)


def agregar_periodo(rodadas):
    """TE, TD e giro de um conjunto de rodadas consecutivas de uma mesma execução.

    O TE do período é o desvio-padrão do erro diário de todos os dias do período, reconstruído exatamente a partir da média e da
    variância de cada rodada (que a simulação já guarda, anualizadas). O giro exclui a primeira rodada da simulação, em que a carteira
    é montada do zero (giro de 100%), como em `simular_walk_forward`.
    """
    n = rodadas["dias"].to_numpy(dtype=float)
    sigma = rodadas["TE_trimestre (% a.a.)"].to_numpy() / (100 * np.sqrt(ANUAL))
    mu = rodadas["TD_trimestre (p.p. a.a.)"].to_numpy() / (100 * ANUAL)
    media = (n * mu).sum() / n.sum()
    soma_quadrados = ((n - 1) * sigma ** 2).sum() + (n * (mu - media) ** 2).sum()
    te = 100 * np.sqrt(soma_quadrados / (n.sum() - 1)) * np.sqrt(ANUAL)
    giro = rodadas.loc[rodadas["numero"] > 1, "giro"]
    return {"TE (% a.a.)": te, "TD (p.p. a.a.)": 100 * media * ANUAL,
            "Giro médio (% trim)": 100 * giro.mean() if len(giro) else np.nan, "trimestres": len(rodadas)}


def resumo_por_periodo(rodadas):
    """Uma linha por índice, λ, semente e período (validação, teste ou tudo)."""
    chaves = ["indice", "lambda_anual"] + (["seed"] if "seed" in rodadas.columns else [])
    linhas = []
    for chave, g in rodadas.groupby(chaves):
        g = g.sort_values("numero")
        base = dict(zip(chaves, chave if isinstance(chave, tuple) else (chave,)))
        for periodo, parte in (("validação", g[g["numero"] <= N_VALIDACAO]), ("teste", g[g["numero"] > N_VALIDACAO]),
                               ("tudo", g)):
            linhas.append({**base, "periodo": periodo, **agregar_periodo(parte)})
    return pd.DataFrame(linhas)


def media_entre_sementes(resumo):
    cols = ["TE (% a.a.)", "TD (p.p. a.a.)", "Giro médio (% trim)"]
    g = resumo.groupby(["indice", "periodo", "lambda_anual"])
    medias = g[cols].mean()
    extremos = g["TE (% a.a.)"].agg(TE_min="min", TE_max="max")
    return medias.join(extremos).reset_index()


def escolher_lambda(medias, tolerancia=TOLERANCIA_TE, periodo="tudo"):
    """Regra declarada: o MAIOR λ da grade (o que mais reduz o giro) com TE a até `tolerancia` do menor TE da grade, nos 20 trimestres."""
    escolhidos = {}
    for indice, g in medias[medias["periodo"] == periodo].groupby("indice"):
        g = g.sort_values("lambda_anual")
        ok = g[g["TE (% a.a.)"] <= g["TE (% a.a.)"].min() + tolerancia]
        escolhidos[indice] = float(ok.iloc[-1]["lambda_anual"])
    return escolhidos


def resumir_e_escolher(K=K_PADRAO):
    """Grava o resumo por período (`validacao_lambda_K{K}.csv`) e o λ escolhido (`lambda_escolhido_K{K}.json`)."""
    rodadas = ler_rodadas(K)
    medias = media_entre_sementes(resumo_por_periodo(rodadas))
    medias.to_csv(PASTA_RESULTADOS / f"validacao_lambda_K{K}.csv", index=False)
    escolhido = escolher_lambda(medias)
    (PASTA_RESULTADOS / f"lambda_escolhido_K{K}.json").write_text(json.dumps(escolhido, ensure_ascii=False, indent=1))
    return medias, escolhido


def lambda_escolhido(K=K_PADRAO):
    arq = PASTA_RESULTADOS / f"lambda_escolhido_K{K}.json"
    return json.loads(arq.read_text()) if arq.exists() else dict(LAMBDA_ATUAL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=K_PADRAO)
    ap.add_argument("--indice", default=None, help="S&P100 ou IBOV (padrão: os dois, em sequência)")
    ap.add_argument("--resumir", action="store_true", help="só grava o resumo e o λ escolhido a partir dos arquivos existentes")
    args = ap.parse_args()
    if args.resumir:
        medias, escolhido = resumir_e_escolher(args.K)
        print("λ escolhido:", escolhido)
    else:
        executar_validacao(K=args.K, indices=(args.indice,) if args.indice else ("S&P100", "IBOV"))
