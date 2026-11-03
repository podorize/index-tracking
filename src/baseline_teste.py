"""Alternativa simples no teste final: as K ações mais correlacionadas com o índice no último ano, em pesos iguais,
ou com os pesos que melhor reproduzem o índice nesse ano, refeita todo mês (mesmo calendário, custos e período do teste final). Grava rodadas no formato de src/teste_final.py."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.compare_all_models import CUSTOS
from src.data_loader import preparar_bases
from src.features import construir_features
from src.models_benchmark import baseline_top_k_corr_equal, baseline_top_k_corr_otimizado
from src.optimizer import ANUAL
from src.rebalanceamento_experiment import intervalo_de_teste
from src.walk_forward import construir_janelas

SAIDA = Path("results/teste_final")


def simular(indice, bases, K, J=252, H=21, otimizado=False):
    ini, fim = intervalo_de_teste(indice, bases)
    b, custo = bases[indice], CUSTOS[indice]
    w_prev, linhas = None, []
    for jan in construir_janelas(indice, bases, J=J, H=H, inicio_teste=ini, fim_teste=fim):
        cands = list(jan["candidatos"])
        R_tr = b["R"].reindex(jan["datas_treino"])[cands].fillna(0.0)
        y_tr = b["y"].reindex(jan["datas_treino"]).fillna(0.0)
        escolher = baseline_top_k_corr_otimizado if otimizado else baseline_top_k_corr_equal
        pesos = escolher(R_tr, y_tr, K, construir_features(R_tr, y_tr))
        pesos = pesos[pesos > 0]
        if w_prev is None:
            giro = 1.0
        else:
            uni = pesos.index.union(w_prev.index)
            giro = 0.5 * (pesos.reindex(uni, fill_value=0.0) - w_prev.reindex(uni, fill_value=0.0)).abs().sum()
        R_ts = b["R"].reindex(jan["datas_teste"])[pesos.index].fillna(0.0)
        y_ts = b["y"].reindex(jan["datas_teste"]).fillna(0.0)
        valores = (1 + R_ts).cumprod().mul(pesos, axis=1)
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
    return pd.DataFrame(linhas).assign(indice=indice, modelo=f"corr{'qp' if otimizado else ''}{K}", K=K, janela=J, horizonte=H, lam=np.nan)


if __name__ == "__main__":
    bases = preparar_bases()
    for indice, arq in (("S&P100", "sp100"), ("IBOV", "ibov")):
        for K in (15, 30, 50):
            simular(indice, bases, K).to_csv(SAIDA / f"corr{K}_{arq}.csv", index=False)
            simular(indice, bases, K, otimizado=True).to_csv(SAIDA / f"corrqp{K}_{arq}.csv", index=False)
            print("pronto", indice, K, flush=True)
