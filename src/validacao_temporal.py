"""
validacao_temporal.py
Defesa das escolhas feitas nos mesmos cinco anos do teste: refaz a escolha de K e de λ usando só os primeiros 12 trimestres (set/2021 a ~set/2024)
e mede o modelo escolhido nos 8 últimos (~set/2024 a set/2026), sem que nada do período final entre na escolha.

Desenho declarado antes dos resultados (30/10/2026):
  - período de escolha: as 12 primeiras rodadas do walk-forward; período de medida: as 8 últimas. Cada rodada propaga a carteira da anterior, então o
    período de medida usa a carteira que a escolha feita só com o início teria produzido desde o começo;
  - K: o menor K da grade {5, 10, 15, 20, 30} cujo TE (heurística sem penalidade, pooled nas 12 rodadas) fica abaixo de 4% ao ano nos dois índices;
  - λ: para esse K, o maior λ da grade cujo TE nas 12 rodadas fica a até 0,2 p.p. do menor TE da grade (a mesma regra da Etapa 12);
  - comparação: o TE nas 8 últimas rodadas desse modelo contra o do modelo escolhido com os 20 trimestres (K = 20, λ = 6). Se os dois ficarem perto,
    a escolha não depende de ter olhado o período de medida; a diferença é o tamanho do otimismo da escolha.
"""

import numpy as np
import pandas as pd

from src.validacao_lambda import agregar_periodo, arquivo_rodadas, GRADE_LAMBDAS, TOLERANCIA_TE

N_ESCOLHA = 12
GRADE_K = [5, 10, 15, 20, 30]
LIMITE_TE = 4.0
INDICES = ("S&P100", "IBOV")


def te_por_periodo(K, indice, lam):
    r = pd.read_csv(arquivo_rodadas(K, indice))
    r = r[r["lambda_anual"] == lam].sort_values("numero")
    return agregar_periodo(r[r["numero"] <= N_ESCOLHA])["TE (% a.a.)"], agregar_periodo(r[r["numero"] > N_ESCOLHA])


def escolher_K():
    tabela = []
    for K in GRADE_K:
        linha = {"K": K}
        for i in INDICES:
            linha[i] = te_por_periodo(K, i, 0.0)[0]
        tabela.append(linha)
    t = pd.DataFrame(tabela)
    ok = t[(t[list(INDICES)] < LIMITE_TE).all(axis=1)]
    return (int(ok.iloc[0]["K"]) if len(ok) else None), t


def escolher_lambda(K):
    out = {}
    for i in INDICES:
        te = {lam: te_por_periodo(K, i, lam)[0] for lam in GRADE_LAMBDAS}
        menor = min(te.values())
        out[i] = max(lam for lam, v in te.items() if v <= menor + TOLERANCIA_TE)
    return out


def medir(K, lams):
    linhas = []
    for i in INDICES:
        escolha, medida = te_por_periodo(K, i, lams[i])
        linhas.append({"indice": i, "K": K, "lambda": lams[i], "TE escolha, 12 rodadas (% a.a.)": escolha,
                       "TE medida, 8 últimas (% a.a.)": medida["TE (% a.a.)"], "TD medida (p.p. a.a.)": medida["TD (p.p. a.a.)"],
                       "Giro médio medida (% trim)": medida["Giro médio (% trim)"]})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    K, t = escolher_K()
    print("TE nas 12 primeiras rodadas, sem penalidade:\n", t.round(2).to_string(index=False))
    print("K escolhido:", K)
    lams = escolher_lambda(K)
    print("λ escolhido:", lams)
    res = pd.concat([medir(K, lams), medir(20, {i: 6.0 for i in INDICES})], ignore_index=True)
    res["modelo"] = ["escolhido só com as 12 primeiras rodadas"] * 2 + ["escolhido com as 20 rodadas (K = 20, λ = 6)"] * 2
    res.to_csv("results/validacao_temporal_K_lambda.csv", index=False)
    print(res.round(2).to_string(index=False))
