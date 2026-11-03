"""
rebalanceamento_experiment.py
Seção de rebalanceamento e janela de treino da Etapa 12 do relatório: frequência (trimestral × mensal) e janela de treino (2 anos × 1 ano)
com o modelo de controle de giro (heurística com penalidade, determinística), nos mesmos pregões de teste (set/2021 a set/2026).

Desenho declarado antes dos resultados (30/10/2026):
  - quatro configurações por índice: (janela de treino, horizonte de teste) em {504, 252} × {63, 21} pregões (2 anos ou 1 ano de treino;
    rebalanceamento trimestral ou mensal), com K = 20, nos mesmos pregões de teste (set/2021 a set/2026);
  - o modelo é o final (heurística com controle de giro), com λ = 6 em todas as configurações, o valor escolhido na Etapa 12; λ é a penalidade
    sobre o giro anualizado, então tem o mesmo significado nas duas frequências. λ em {0, 3, 12} entra só como sensibilidade;
  - a referência é a configuração atual (504, 63), refeita aqui com o mesmo código;
  - o motor é determinístico, então não há repetição por semente;
  - regra de decisão, fixada antes de olhar os resultados: uma alternativa substitui a referência só se, NOS DOIS ÍNDICES, (a) o TE ficar
    pelo menos 0,2 p.p. abaixo do da referência, ou (b) o TE ficar a até 0,2 p.p. acima do da referência e o giro anualizado cair pelo menos
    20%; e, além disso, se o sinal da diferença de TE for o mesmo nas duas metades do período de teste (10 trimestres cada). Os 0,2 p.p. são
    a mesma tolerância usada na escolha de λ. Se nenhuma alternativa cumprir a regra, a referência fica.
Os resultados são gravados a cada execução (um arquivo por K, índice e configuração), então uma interrupção não perde o que já foi feito.
"""

import argparse
from pathlib import Path
import time

import pandas as pd

from src.data_loader import preparar_bases
from src.validacao_lambda import agregar_periodo, lambda_escolhido
from src.walk_forward import construir_janelas, simular_walk_forward

PASTA_RESULTADOS = Path("results")

CUSTOS = {"S&P100": 0.0005, "IBOV": 0.0015}
GRADE_LAMBDAS = [0.0, 3.0, 6.0, 12.0]
CONFIGURACOES = [(504, 63), (504, 21), (252, 63), (252, 21)]   # (janela de treino, horizonte de teste), em pregões
REFERENCIA = (504, 63)
K_PADRAO = 20


def arquivo(K, indice, janela, horizonte):
    nome = "sp100" if indice == "S&P100" else "ibov"
    return PASTA_RESULTADOS / f"rebalanceamento_K{K}_{nome}_J{janela}_H{horizonte}.csv"


def intervalo_de_teste(nome_indice, bases):
    """Primeiro e último pregão de teste da configuração de referência (20 trimestres)."""
    janelas = construir_janelas(nome_indice, bases)
    return janelas[0]["datas_teste"][0], janelas[-1]["datas_teste"][-1]


def grade_de_lambdas(K, indice):
    escolhido = lambda_escolhido(K)[indice]
    return sorted(set(GRADE_LAMBDAS) | {escolhido})


def executar(K=K_PADRAO, indice="S&P100", janela=504, horizonte=63, bases=None):
    bases = bases if bases is not None else preparar_bases()
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    inicio, fim = intervalo_de_teste(indice, bases)
    arq = arquivo(K, indice, janela, horizonte)
    feitas = pd.read_csv(arq) if arq.exists() else pd.DataFrame()
    ja = set(feitas["lambda_anual"]) if len(feitas) else set()
    grade = grade_de_lambdas(K, indice)
    pendentes = [lam for lam in grade if lam not in ja]
    print(f"K={K} {indice} janela {janela} rebal. {horizonte}: {len(grade)} valores de λ, {len(pendentes)} a fazer.", flush=True)
    linhas = feitas.to_dict("records") if len(feitas) else []
    for n, lam in enumerate(pendentes, 1):
        t0 = time.time()
        metricas, rodadas = simular_walk_forward(
            nome_indice=indice, bases=bases, K=K, lambda_anual=lam, custo_transacao=CUSTOS[indice],
            janela_treino=janela, horizonte_teste=horizonte, inicio_teste=inicio, fim_teste=fim, metodo="heuristica",
        )
        metricas = {**metricas, "K": K, "metodo": "heuristica", "partidas_anterior": int((rodadas.get("partida") == "anterior").sum())}
        linhas.append(metricas)
        pd.DataFrame(linhas).to_csv(arq, index=False)
        print(f"[{n}/{len(pendentes)}] K={K} {indice} janela {janela} rebal. {horizonte} λ={lam:g}: TE {metricas['TE (% a.a.)']:.2f}%, "
              f"giro anualizado {metricas['Giro anualizado (%)']:.0f}% ({time.time() - t0:.0f} s)", flush=True)
        # rodadas por trimestre/mês, para a análise por período
        rodadas.assign(indice=indice, lambda_anual=lam, K=K, janela_treino=janela, horizonte_teste=horizonte).to_csv(
            PASTA_RESULTADOS / f"rebalanceamento_K{K}_rodadas_{'sp100' if indice == 'S&P100' else 'ibov'}_J{janela}_H{horizonte}_L{lam:g}.csv",
            index=False)


def ler_resultados(K=K_PADRAO):
    partes = [pd.read_csv(a) for a in sorted(PASTA_RESULTADOS.glob(f"rebalanceamento_K{K}_*_J*_H*.csv")) if "_rodadas_" not in a.name]
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def rodadas_da_configuracao(K, indice, janela, horizonte, lam):
    """Rodadas por trimestre/mês de uma configuração. A de referência (504, 63) vem da grade de λ da Etapa 12, que usa o mesmo código e
    os mesmos pregões de teste."""
    nome = "sp100" if indice == "S&P100" else "ibov"
    if (janela, horizonte) == REFERENCIA:
        r = pd.read_csv(PASTA_RESULTADOS / f"validacao_lambda_K{K}_{nome}_rodadas.csv")
        return r[r["lambda_anual"] == lam]
    return pd.read_csv(PASTA_RESULTADOS / f"rebalanceamento_K{K}_rodadas_{nome}_J{janela}_H{horizonte}_L{lam:g}.csv")


def medidas(rodadas, horizonte):
    """TE, TD e giro anualizado de um conjunto de rodadas consecutivas (mesma fórmula da Etapa 12; o giro exclui a primeira rodada)."""
    m = agregar_periodo(rodadas)
    giro = rodadas.loc[rodadas["numero"] > 1, "giro"].mean()
    return {"TE (% a.a.)": m["TE (% a.a.)"], "TD (p.p. a.a.)": m["TD (p.p. a.a.)"],
            "Giro anualizado (%)": 100 * giro * (252 / horizonte)}


def avaliar_regra(K=K_PADRAO, lam=6.0, tolerancia=0.2, queda_minima_giro=0.20):
    """Aplica a regra de decisão declarada no topo do módulo. Devolve a tabela por índice e configuração, com o veredicto de cada alternativa."""
    linhas = []
    for indice in ("S&P100", "IBOV"):
        lam_i = lam[indice] if isinstance(lam, dict) else lam      # λ pode ser um por índice (o escolhido para cada K)
        ref_r = rodadas_da_configuracao(K, indice, *REFERENCIA, lam_i).sort_values("data_inicio")
        meio = ref_r["data_inicio"].iloc[len(ref_r) // 2]          # as duas metades do período de teste, por data
        ref = {"tudo": medidas(ref_r, REFERENCIA[1]),
               "1ª metade": medidas(ref_r[ref_r["data_inicio"] < meio], REFERENCIA[1]),
               "2ª metade": medidas(ref_r[ref_r["data_inicio"] >= meio], REFERENCIA[1])}
        for janela, horizonte in CONFIGURACOES:
            r = rodadas_da_configuracao(K, indice, janela, horizonte, lam_i).sort_values("data_inicio")
            m = {"tudo": medidas(r, horizonte), "1ª metade": medidas(r[r["data_inicio"] < meio], horizonte),
                 "2ª metade": medidas(r[r["data_inicio"] >= meio], horizonte)}
            d_te = m["tudo"]["TE (% a.a.)"] - ref["tudo"]["TE (% a.a.)"]
            d_giro = m["tudo"]["Giro anualizado (%)"] / ref["tudo"]["Giro anualizado (%)"] - 1
            sinais = [m[h]["TE (% a.a.)"] - ref[h]["TE (% a.a.)"] for h in ("1ª metade", "2ª metade")]
            criterio = (d_te <= -tolerancia) or (d_te <= tolerancia and d_giro <= -queda_minima_giro)
            linhas.append({"indice": indice, "janela": janela, "rebalanceamento": horizonte, "lambda": lam_i,
                           "TE (% a.a.)": m["tudo"]["TE (% a.a.)"], "TD (p.p. a.a.)": m["tudo"]["TD (p.p. a.a.)"],
                           "Giro anualizado (%)": m["tudo"]["Giro anualizado (%)"],
                           "dTE vs ref (p.p.)": d_te, "dGiro vs ref (%)": 100 * d_giro,
                           "dTE 1ª metade": sinais[0], "dTE 2ª metade": sinais[1],
                           "passa o criterio no indice": bool(criterio and (janela, horizonte) != REFERENCIA),
                           "mesmo sinal nas metades": bool(sinais[0] * sinais[1] > 0)})
    t = pd.DataFrame(linhas)
    t["substitui a referência"] = False
    for (janela, horizonte), g in t.groupby(["janela", "rebalanceamento"]):
        ok = bool(g["passa o criterio no indice"].all() and g["mesmo sinal nas metades"].all()) and (janela, horizonte) != REFERENCIA
        t.loc[g.index, "substitui a referência"] = ok
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=K_PADRAO)
    ap.add_argument("--indice", required=True)
    ap.add_argument("--janela", type=int, required=True)
    ap.add_argument("--horizonte", type=int, required=True)
    a = ap.parse_args()
    executar(K=a.K, indice=a.indice, janela=a.janela, horizonte=a.horizonte)
