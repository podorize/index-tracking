"""Alternativa simples para comparar no teste final: as K ações mais correlacionadas com o índice no último ano, com os pesos que
melhor reproduziam o índice nesse ano (mínimos quadrados sem venda a descoberto), refeita todo mês com os mesmos custos e período."""
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src.config import ARQ, CUSTOS, INDICES, RECOMENDADO
from src.data_loader import preparar_bases
from src.grade import intervalo_de_teste
from src.optimizer import ANUAL
from src.teste_final import SAIDA
from src.walk_forward import construir_janelas

PENALIDADE_SOMA = 1e3  # peso da linha extra que força a soma dos pesos a 1


def pesos(R, y, K):
    top = R.corrwith(y).sort_values(ascending=False).head(K).index
    X, a = 100.0 * R[top].to_numpy(dtype=float), 100.0 * y.to_numpy(dtype=float)
    w, _ = nnls(np.vstack([X, PENALIDADE_SOMA * np.ones(len(top))]), np.r_[a, PENALIDADE_SOMA])
    w = w / w.sum() if w.sum() > 0 else np.ones(len(top)) / len(top)
    return pd.Series(w, index=top)


def simular(indice, bases, K, J=252, H=21):
    ini, fim = intervalo_de_teste(indice, bases)
    b, custo = bases[indice], CUSTOS[indice]
    w_prev, linhas = None, []
    for jan in construir_janelas(indice, bases, J=J, H=H, inicio_teste=ini, fim_teste=fim):
        R_tr = b["R"].reindex(jan["datas_treino"])[list(jan["candidatos"])].fillna(0.0)
        w = pesos(R_tr, b["y"].reindex(jan["datas_treino"]).fillna(0.0), K)
        w = w[w > 0]
        if w_prev is None:
            giro = 1.0
        else:
            uni = w.index.union(w_prev.index)
            giro = 0.5 * (w.reindex(uni, fill_value=0.0) - w_prev.reindex(uni, fill_value=0.0)).abs().sum()
        R_ts = b["R"].reindex(jan["datas_teste"])[w.index].fillna(0.0)
        y_ts = b["y"].reindex(jan["datas_teste"]).fillna(0.0)
        valores = (1 + R_ts).cumprod().mul(w, axis=1)
        patrimonio = valores.sum(axis=1)
        rp = patrimonio.pct_change()
        rp.iloc[0] = patrimonio.iloc[0] - 1.0
        dif = rp - y_ts
        r_bruto = float(patrimonio.iloc[-1] - 1.0)
        linhas.append({"numero": jan["numero"], "data_inicio": jan["datas_teste"][0], "data_fim": jan["datas_teste"][-1],
                       "dias": len(dif), "K_efetivo": K, "giro": giro, "retorno_bruto_carteira": r_bruto,
                       "retorno_liq_carteira": r_bruto - custo * giro, "retorno_indice": float((1 + y_ts).prod() - 1.0),
                       "TE_trimestre (% a.a.)": float(dif.std() * np.sqrt(ANUAL) * 100),
                       "TD_trimestre (p.p. a.a.)": float(dif.mean() * ANUAL * 100)})
        w_prev = valores.iloc[-1] / valores.iloc[-1].sum()
    return pd.DataFrame(linhas).assign(indice=indice, K=K, janela=J, horizonte=H)


def arquivo(indice, K):
    return SAIDA / f"alternativa_K{K}_{ARQ[indice]}.csv"


def rodar():
    """Alternativa com o mesmo número de ações da carteira do cenário recomendado, em cada índice."""
    from src.regra import cenarios
    ks = cenarios()[RECOMENDADO][1]
    bases = None
    for indice in INDICES:
        if not arquivo(indice, ks[indice]).exists():
            bases = bases if bases is not None else preparar_bases()
            simular(indice, bases, ks[indice]).to_csv(arquivo(indice, ks[indice]), index=False)
    print("Alternativa simples pronta.", flush=True)
