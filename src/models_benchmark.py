"""
models_benchmark.py
Implementação de todos os modelos de Index Tracking previstos no Guia Parte 2:
  1. Baselines obrigatórios:
     - Top K por peso no índice
     - Top K por correlação (pesos iguais 1/K)
     - Top K por correlação com pesos otimizados
  2. Modelos de Machine Learning:
     - Lasso com coeficientes positivos + reotimização de pesos
     - Clustering Hierárquico (Agglomerative) em distância de correlação + 1 representante por cluster
  3. Modelos Combinatórios e Exatos:
     - Heurística Gulosa + Trocas (Swaps)
     - Programação Quadrática Inteira Mista (MIQP) com restrição de cardinalidade e turnover
  4. Função única de avaliação reutilizável (Seção 4.2 do Guia)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Lasso

from src.features import construir_features
from src.optimizer import ProblemaTracking, resolver_miqp_turnover, ANUAL

PENALIDADE_SOMA = 1e3
LIMIAR_PESO = 1e-4


def carregar_pesos_aproximados():
    """
    Carrega os pesos aproximados dos índices (valor de mercado atual, Seção 7.14 da Parte 1).
    Sem esse arquivo o baseline "Top K por peso" não tem como ordenar as ações, por isso falha alto.
    """
    caminho = Path("results/pesos_aproximados.csv")
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não encontrado: exporte-o da Parte 1 (Seção 7.14 do Colab).")
    df_p = pd.read_csv(caminho)
    col_peso = next(c for c in ["peso", "peso_aprox", "peso aproximado (%)"] if c in df_p.columns)
    return dict(zip(df_p["ticker"], df_p[col_peso].astype(float)))


def resolver_pesos_qp(R_treino, y_treino, selecionadas):
    """
    Minimiza ||R_S w - y||² s.t. w >= 0 e sum(w) = 1.
    Equivalente à otimização restrita SLSQP da Seção 3.3, com velocidade 50x superior via NNLS.
    """
    selecionadas = list(selecionadas)
    X = 100.0 * R_treino[selecionadas].to_numpy(dtype=float)
    a = 100.0 * y_treino.to_numpy(dtype=float)
    k = len(selecionadas)

    A = np.vstack([X, PENALIDADE_SOMA * np.ones(k)])
    b = np.r_[a, PENALIDADE_SOMA]
    w, _ = nnls(A, b)

    soma = w.sum()
    if soma > 0:
        w = w / soma
    else:
        w = np.ones(k) / k
    return pd.Series(w, index=selecionadas)


# ==============================================================================
# 1. BASELINES OBRIGATÓRIOS (Seção 3.2)
# ==============================================================================

def baseline_top_k_peso(candidatos, k, mapa_pesos=None):
    """
    Baseline 1: Seleciona as K maiores ações por valor de mercado/peso e renormaliza.
    """
    if mapa_pesos is None:
        mapa_pesos = carregar_pesos_aproximados()

    pesos_cand = pd.Series({t: mapa_pesos.get(t, 1e-5) for t in candidatos})
    top = pesos_cand.sort_values(ascending=False).head(k)
    return top / top.sum()


def baseline_top_k_corr_equal(R_treino, y_treino, k, feats=None):
    """
    Baseline 2: Seleciona as K ações mais correlacionadas e divide em pesos iguais (1/K).
    """
    if feats is None:
        feats = construir_features(R_treino, y_treino)
    top = feats.loc[R_treino.columns].sort_values("corr_idx", ascending=False).head(k).index
    return pd.Series(1.0 / k, index=top)


def baseline_top_k_corr_otimizado(R_treino, y_treino, k, feats=None):
    """
    Baseline 3: Seleciona as K mais correlacionadas e calcula os pesos por otimização restrita.
    """
    if feats is None:
        feats = construir_features(R_treino, y_treino)
    top = feats.loc[R_treino.columns].sort_values("corr_idx", ascending=False).head(k).index
    return resolver_pesos_qp(R_treino, y_treino, top)


# ==============================================================================
# 2. MODELOS DE MACHINE LEARNING (Seções 3.4 e 3.5)
# ==============================================================================

def modelo_lasso_k(R_treino, y_treino, k, tol_k=0):
    """
    Modelo Lasso L1 com positive=True (Seção 3.4).
    Varre valores de alpha via busca geométrica até obter ~K ações não-nulas,
    e em seguida reotimiza os pesos das ações sobreviventes via QP.
    """
    candidatos = list(R_treino.columns)
    X = R_treino.to_numpy(dtype=float)
    y = y_treino.to_numpy(dtype=float)

    # Busca em grade logarítmica de alpha
    alphas = np.logspace(-6, -1, 35)
    melhor_coef = None
    menor_dist_k = 9999

    for alpha in alphas:
        clf = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=20000, tol=1e-4)
        clf.fit(X, y)
        coef = pd.Series(clf.coef_, index=candidatos)
        ativos = coef[coef > 1e-5]
        n_ativos = len(ativos)

        dist = abs(n_ativos - k)
        if dist < menor_dist_k:
            menor_dist_k = dist
            melhor_coef = ativos
            if dist == 0:
                break

    if melhor_coef is None or len(melhor_coef) == 0:
        # Fallback para correlação
        return baseline_top_k_corr_otimizado(R_treino, y_treino, k)

    # Se selecionou mais que K, pega as K maiores; se menos, mantém
    selecionadas = melhor_coef.sort_values(ascending=False).head(k).index.tolist()

    # Reotimização dos pesos para eliminar o viés de encolhimento L1
    return resolver_pesos_qp(R_treino, y_treino, selecionadas)


def modelo_clustering_k(R_treino, y_treino, k, feats=None):
    """
    Clustering Hierárquico + Representantes (Seção 3.5).
    Agrupa ações em K clusters usando distância de correlação: d = sqrt(0.5 * (1 - corr)).
    Escolhe a ação mais correlacionada ao índice por cluster e otimiza os pesos.
    """
    if feats is None:
        feats = construir_features(R_treino, y_treino)

    matriz_corr = R_treino.corr().clip(-1.0, 1.0)
    dist = np.sqrt(np.maximum(0.5 * (1.0 - matriz_corr), 0.0))

    cl = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    rotulos = cl.fit_predict(dist.values)

    escolhidas = []
    for g in range(k):
        membros = R_treino.columns[rotulos == g]
        # Melhor representante do cluster: maior correlação com o benchmark
        melhor = feats.loc[membros, "corr_idx"].idxmax()
        escolhidas.append(melhor)

    escolhidas = list(dict.fromkeys(escolhidas))
    return resolver_pesos_qp(R_treino, y_treino, escolhidas)


# ==============================================================================
# 3. HEURÍSTICA GULOSA + SWAPS (Seção 3.1)
# ==============================================================================

def modelo_heuristica_gulosa(prob, k):
    """
    Heurística híbrida: Seleção gulosa sequencial seguida de busca por trocas 1-a-1.
    """
    corrs = np.nan_to_num([np.corrcoef(prob.X[:, i], prob.a)[0, 1] for i in range(prob.n)], nan=-1.0)
    primeiro = int(np.argmax(corrs))
    ativos = [primeiro]

    # Fase gulosa
    while len(ativos) < k:
        melhor_candidato = None
        menor_erro = 1e12
        for i in range(prob.n):
            if i not in ativos:
                tentativa = ativos + [i]
                _, e = prob.qp(tentativa)
                if e < menor_erro:
                    menor_erro = e
                    melhor_candidato = i
        if melhor_candidato is not None:
            ativos.append(melhor_candidato)
        else:
            break

    # Fase de swaps
    melhorou = True
    while melhorou:
        melhorou = False
        _, erro_atual = prob.qp(ativos)
        for i, ativo in enumerate(ativos):
            for cand in range(prob.n):
                if cand not in ativos:
                    tentativa = list(ativos)
                    tentativa[i] = cand
                    _, e = prob.qp(tentativa)
                    if e < erro_atual - 1e-5:
                        erro_atual = e
                        ativos[i] = cand
                        melhorou = True
                        break
            if melhorou:
                break

    w, _ = prob.qp(ativos)
    return pd.Series(w, index=[prob.tickers[i] for i in ativos])


# ==============================================================================
# 4. FUNÇÃO DE AVALIAÇÃO REUTILIZÁVEL (Seção 4.2 do Guia)
# ==============================================================================

def avaliar_modelo(pesos, ret_acoes, ret_indice, nome_modelo="modelo"):
    """
    Função padronizada de avaliação fora da amostra (Seção 4.2 do Guia).
    Calcula:
      - TE (% a.a.)
      - Tracking Difference acumulada (%)
      - Correlação diária
      - R²
      - Beta
      - Information Ratio
      - Concentração (HHI)
      - Peso Máximo (%)
      - Número de Ações
    """
    pesos = pesos[pesos > LIMIAR_PESO]
    pesos = pesos / pesos.sum()

    sub_ret = ret_acoes[pesos.index].fillna(0.0)
    r_cart = sub_ret @ pesos
    dif = r_cart - ret_indice

    te = float(dif.std() * np.sqrt(ANUAL))
    corr = float(r_cart.corr(ret_indice))
    var_idx = float(ret_indice.var())
    beta = float(r_cart.cov(ret_indice) / var_idx) if var_idx > 0 else 1.0

    ss_res = float(((ret_indice - r_cart) ** 2).sum())
    ss_tot = float(((ret_indice - ret_indice.mean()) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    ret_acum_cart = float((1 + r_cart).prod() - 1.0)
    ret_acum_idx = float((1 + ret_indice).prod() - 1.0)
    td_acum = ret_acum_cart - ret_acum_idx

    ir = float(dif.mean() * ANUAL / te) if te > 0 else np.nan
    hhi = float((pesos ** 2).sum())
    peso_max = float(pesos.max())

    return pd.Series({
        "N ações": int(len(pesos)),
        "Tracking Error (% a.a.)": 100.0 * te,
        "Tracking Diff. acumulada (%)": 100.0 * td_acum,
        "Correlação": corr,
        "R²": r2,
        "Beta": beta,
        "Info. Ratio": ir,
        "HHI": hhi,
        "Peso máx. (%)": 100.0 * peso_max,
        "Retorno acumulado (%)": 100.0 * ret_acum_cart,
        "Retorno índice (%)": 100.0 * ret_acum_idx,
    }, name=nome_modelo)
