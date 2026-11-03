"""
features.py
Feature Engineering e Feature Selection para Index Tracking (Guia Parte 2).
Calcula indicadores estatísticos das ações exclusivamente no conjunto de treino:
  - Correlação com o índice
  - Beta em relação ao índice
  - Volatilidade anualizada
  - Correlação média com os pares (redundância)
  - Drawdown máximo
  - Setor econômico
  - Filtro de redundância (pares > 0.90)
"""

from pathlib import Path
import numpy as np
import pandas as pd

ANUAL = 252


def carregar_setores():
    """Carrega o mapa de setores a partir dos arquivos salvos ou dicionário padrão."""
    arquivo_setores = Path("results/setores_yahoo.csv")
    if arquivo_setores.exists():
        df_set = pd.read_csv(arquivo_setores)
        return dict(zip(df_set["ticker"], df_set["setor"]))
    return {}


def construir_features(ret_acoes, ret_indice, mapa_setores=None):
    """
    Tabela ação x feature, calculada estritamente com dados de treino.
    Evita qualquer vazamento de informação (look-ahead bias).
    """
    if mapa_setores is None:
        mapa_setores = carregar_setores()

    feats = pd.DataFrame(index=ret_acoes.columns)
    var_idx = float(ret_indice.var())

    feats["corr_idx"] = ret_acoes.corrwith(ret_indice)
    feats["beta"] = ret_acoes.apply(lambda s: float(s.cov(ret_indice) / var_idx) if var_idx > 0 else 1.0)
    feats["vol_anual"] = ret_acoes.std() * np.sqrt(ANUAL)
    feats["setor"] = feats.index.map(mapa_setores).fillna("Outros")

    # Correlação entre pares (redundância)
    matriz_corr = ret_acoes.corr()
    n_ativos = len(matriz_corr)
    if n_ativos > 1:
        feats["corr_media_pares"] = (matriz_corr.sum(axis=1) - 1.0) / (n_ativos - 1)
    else:
        feats["corr_media_pares"] = 0.0

    # Drawdown máximo no treino
    niveis = (1 + ret_acoes.fillna(0.0)).cumprod()
    picos = niveis.cummax()
    drawdowns = (niveis - picos) / picos
    feats["max_drawdown"] = drawdowns.min()

    return feats


def selecionar_acoes_filtro_redundancia(ret_acoes, ret_indice, top_n=40, limite_corr_par=0.90, mapa_setores=None):
    """
    Abordagem de seleção descrita na Seção 2.3 do Guia Parte 2:
      1. Filtra as top_n ações com maior correlação com o índice
      2. Remove ações com correlação bilateral > limite_corr_par mantendo a de maior correlação com o índice
    """
    feats = construir_features(ret_acoes, ret_indice, mapa_setores=mapa_setores)
    candidatas = feats.sort_values("corr_idx", ascending=False).head(top_n).index.tolist()

    matriz_corr = ret_acoes[candidatas].corr()
    selecionadas = []

    for a in candidatas:
        # Se for o primeiro ou tiver correlação < limite com todas as já selecionadas
        if all(matriz_corr.loc[a, s] < limite_corr_par for s in selecionadas):
            selecionadas.append(a)

    return selecionadas, feats.loc[selecionadas]
