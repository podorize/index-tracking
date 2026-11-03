"""
walk_forward.py
Motor de simulação Walk-Forward (fora da amostra) com:
  - Rastreamento dinâmico de pesos derivados (w_drift) entre trimestres
  - Cálculo de Turnover unilateral
  - Desconto explícito de custos de transação (retorno líquido)
  - Métricas agregadas de Tracking Error, Tracking Difference (TD), RMSE e Information Ratio
  - Compatibilidade universal com limites de licença demo do Gurobi (<= 200 variáveis)
"""

import numpy as np
import pandas as pd
from src.data_loader import ENTRADAS_NO_INDICE
from src.heuristica_giro import heuristica_giro
from src.optimizer import ProblemaTracking, resolver_miqp_turnover, ANUAL

JANELA_TREINO_PADRAO = 504
HORIZONTE_TESTE_PADRAO = 63
DIAS_MINIMOS_TESTE = 21
COBERTURA_MINIMA_TREINO = 0.98
MAX_CANDIDATOS_GUROBI = 93  # Garante 2*N + K <= 196 <= 200 vars para licenças demo (com K = 10)


def max_candidatos(K):
    """No MIQP com giro há N pesos, N binárias e até K variáveis de venda: 2N + K <= 200 na licença gratuita."""
    if K is None:                      # sem limite de cardinalidade: QP contínuo, N pesos + até N variáveis de venda
        return MAX_CANDIDATOS_GUROBI
    return min(MAX_CANDIDATOS_GUROBI, (200 - K) // 2)


def construir_janelas(nome_indice, bases, J=JANELA_TREINO_PADRAO, H=HORIZONTE_TESTE_PADRAO, n_janelas=None,
                      inicio_teste=None, fim_teste=None):
    """
    Gera a lista de rodadas de teste com as ações candidatas em cada período.
    Com inicio_teste (e fim_teste), as rodadas cobrem exatamente esse intervalo de pregões, qualquer que seja
    a janela de treino J ou o horizonte H: é o que permite comparar janelas e frequências nos mesmos dias de teste.
    """
    b = bases[nome_indice]
    y, R = b["y"], b["R"]
    if inicio_teste is not None:
        primeiro = y.index.get_loc(pd.Timestamp(inicio_teste))
        fim = len(y) if fim_teste is None else y.index.get_loc(pd.Timestamp(fim_teste)) + 1
        inicios = list(range(primeiro, fim, H))   # a última rodada pode ser mais curta, para cobrir os mesmos pregões
    else:
        fim = len(y)
        inicios = [s for s in range(J, len(y), H) if len(y) - s >= DIAS_MINIMOS_TESTE]
    if n_janelas:
        inicios = inicios[-int(n_janelas):]

    janelas = []
    for numero, s in enumerate(inicios, 1):
        datas_treino = y.index[s - J:s]
        datas_teste = y.index[s:min(s + H, fim)]
        Rt = R.reindex(datas_treino)
        cobertura = Rt.notna().mean()
        elegiveis = cobertura[(cobertura >= COBERTURA_MINIMA_TREINO) & Rt.tail(5).notna().all()].index.tolist()
        fora_do_indice = [
            t for t in elegiveis
            if datas_teste[0] < ENTRADAS_NO_INDICE.get((nome_indice, t), pd.Timestamp.min)
        ]
        candidatos = [t for t in elegiveis if t not in fora_do_indice]
        janelas.append({
            "indice": nome_indice,
            "numero": numero,
            "rebalanceamento": datas_teste[0],
            "fim_teste": datas_teste[-1],
            "datas_treino": datas_treino,
            "datas_teste": datas_teste,
            "candidatos": candidatos,
        })
    return janelas


def simular_walk_forward(
    nome_indice,
    bases,
    K=10,
    lambda_anual=0.0,
    custo_transacao=0.0015,
    tempo_max=2.0,
    n_janelas=None,
    janela_treino=JANELA_TREINO_PADRAO,
    horizonte_teste=HORIZONTE_TESTE_PADRAO,
    inicio_teste=None,
    fim_teste=None,
    seed=42,
    metodo="miqp",
    retornar_series=False,
):
    """
    `metodo`: "miqp" (Gurobi, partindo da carteira anterior), "heuristica" (heurística gulosa + trocas com a penalidade de giro,
    determinística) ou "heuristica+miqp" (a heurística e, a partir dela, o Gurobi até `tempo_max`).
    Executa o walk-forward sequencial propagando w_drift de rodada em rodada.
    A frequência de rebalanceamento vem do horizonte de teste (63 pregões = 4 vezes por ano; 21 = 12 vezes).
    lambda_anual é a penalidade sobre o giro anualizado: o multiplicador do solver é rebal_por_ano × λ ÷ 252,
    então λ tem o mesmo significado em qualquer frequência.
    """
    janelas = construir_janelas(nome_indice, bases, J=janela_treino, H=horizonte_teste, n_janelas=n_janelas,
                                inicio_teste=inicio_teste, fim_teste=fim_teste)
    rebal_por_ano = int(round(ANUAL / horizonte_teste))
    b = bases[nome_indice]

    w_drift_prev = None
    resultados_janelas = []
    series_retornos_carteira = []
    series_retornos_indice = []

    # Conversão de lambda anual para o multiplicador gamma do solver
    gamma = (rebal_por_ano * lambda_anual) / ANUAL if lambda_anual > 0 else 0.0

    for jan in janelas:
        candidatos = list(jan["candidatos"])
        R_tr_full = b["R"].reindex(jan["datas_treino"])[candidatos].fillna(0.0)
        y_tr = b["y"].reindex(jan["datas_treino"]).fillna(0.0)

        # Se houver mais candidatos que o teto da licença demo, preserva ativos da carteira prévia e seleciona por correlação
        if len(candidatos) > max_candidatos(K):
            ativos_prev = list(w_drift_prev.keys()) if w_drift_prev is not None else []
            # Calcula correlações com o índice
            corrs = R_tr_full.corrwith(y_tr).fillna(-1.0)
            # Ordena por maior correlação
            ordenados = corrs.sort_values(ascending=False).index.tolist()
            selecionados = list(dict.fromkeys(ativos_prev + [t for t in ordenados if t not in ativos_prev]))
            candidatos = selecionados[:max_candidatos(K)]

        R_tr = R_tr_full[candidatos]
        prob = ProblemaTracking(R_tr, y_tr)

        w_drift_dict = w_drift_prev.to_dict() if w_drift_prev is not None else None

        # Warm start usando carteira prévia se disponível
        w_inicial = None
        if w_drift_prev is not None:
            w_inicial = np.zeros(prob.n)
            for t_nome, peso in w_drift_prev.items():
                if t_nome in candidatos:
                    w_inicial[candidatos.index(t_nome)] = peso

        extra = {}
        if metodo == "miqp":
            w_sol, info = resolver_miqp_turnover(
                prob,
                K=K,
                w_drift_dict=w_drift_dict,
                gamma=gamma,
                w_inicial=w_inicial,
                tempo_max=tempo_max,
                seed=seed
            )
        else:
            w_heur, _, info_h = heuristica_giro(prob, K, w_drift_dict, gamma)
            extra = {"partida": info_h["partida"], "objetivo_heuristica": info_h["objetivo"],
                     "tempo_heuristica_s": info_h["tempo_s"]}
            if metodo == "heuristica":
                w_sol, info = w_heur, {"status": "HEURISTICA", "tempo_s": info_h["tempo_s"]}
            else:
                w_sol, info = resolver_miqp_turnover(
                    prob,
                    K=K,
                    w_drift_dict=w_drift_dict,
                    gamma=gamma,
                    w_inicial=w_heur,
                    tempo_max=tempo_max,
                    seed=seed
                )
                extra["objetivo_final"] = info.get("obj_val", np.nan)
                info["tempo_s"] = info["tempo_s"] + info_h["tempo_s"]

        if w_sol is None:
            if w_inicial is not None and w_inicial.sum() > 0:
                pesos_novos = pd.Series(w_inicial / w_inicial.sum(), index=candidatos)
            else:
                pesos_novos = pd.Series(1.0 / (K or len(candidatos)), index=candidatos[:K])
        else:
            pesos_novos = pd.Series(w_sol, index=candidatos)
            pesos_novos = pesos_novos[pesos_novos > 1e-4]
            pesos_novos = pesos_novos / pesos_novos.sum()

        if w_drift_prev is None:
            giro_trimestre = 1.0
        else:
            uni = pesos_novos.index.union(w_drift_prev.index)
            giro_trimestre = 0.5 * (
                pesos_novos.reindex(uni, fill_value=0.0) - w_drift_prev.reindex(uni, fill_value=0.0)
            ).abs().sum()

        datas_ts = jan["datas_teste"]
        R_ts = b["R"].reindex(datas_ts)[pesos_novos.index]
        y_ts = b["y"].reindex(datas_ts).rename("indice")

        valores = (1 + R_ts.fillna(0.0)).cumprod().mul(pesos_novos, axis=1)
        patrimonio = valores.sum(axis=1)
        rp_diario = patrimonio.pct_change()
        rp_diario.iloc[0] = patrimonio.iloc[0] - 1.0

        par_diario = pd.concat([rp_diario.rename("carteira"), y_ts], axis=1).dropna()
        series_retornos_carteira.append(par_diario["carteira"])
        series_retornos_indice.append(par_diario["indice"])

        r_cart_trim_bruto = float(patrimonio.iloc[-1] - 1.0)
        r_idx_trim = float((1 + y_ts.dropna()).prod() - 1.0)
        # custo por unidade de giro: a taxa já cobre a compra e a venda (o giro conta um lado só)
        custo_friccao = float(custo_transacao * giro_trimestre)
        r_cart_trim_liq = r_cart_trim_bruto - custo_friccao

        erro_diario = par_diario["carteira"] - par_diario["indice"]

        resultados_janelas.append({
            "numero": jan["numero"],
            "data_inicio": datas_ts[0],
            "data_fim": datas_ts[-1],
            "dias": len(par_diario),
            "K_efetivo": len(pesos_novos),
            "giro": giro_trimestre,
            "custo_friccao": custo_friccao,
            "retorno_bruto_carteira": r_cart_trim_bruto,
            "retorno_liq_carteira": r_cart_trim_liq,
            "retorno_indice": r_idx_trim,
            "TE_trimestre (% a.a.)": 100 * erro_diario.std() * np.sqrt(ANUAL),
            "TD_trimestre (p.p. a.a.)": 100 * erro_diario.mean() * ANUAL,
            "status_solver": info["status"],
            "tempo_s": info["tempo_s"],
            **extra,
        })

        w_drift_prev = valores.iloc[-1] / valores.iloc[-1].sum()

    df_janelas = pd.DataFrame(resultados_janelas)
    r_cart_todos = pd.concat(series_retornos_carteira)
    r_idx_todos = pd.concat(series_retornos_indice)

    erro_total = r_cart_todos - r_idx_todos
    te_anual = 100 * erro_total.std() * np.sqrt(ANUAL)
    vies_anual = 100 * erro_total.mean() * ANUAL
    rmse_anual = 100 * np.sqrt((erro_total ** 2).mean() * ANUAL)

    giro_medio_trimestre = df_janelas["giro"].iloc[1:].mean() if len(df_janelas) > 1 else df_janelas["giro"].mean()
    custo_anual_friccao = rebal_por_ano * custo_transacao * giro_medio_trimestre

    ret_acum_bruto = (1 + df_janelas["retorno_bruto_carteira"]).prod() - 1.0
    ret_acum_liq = (1 + df_janelas["retorno_liq_carteira"]).prod() - 1.0
    ret_acum_idx = (1 + df_janelas["retorno_indice"]).prod() - 1.0

    n_anos = len(r_cart_todos) / ANUAL
    ret_anual_liq = (1 + ret_acum_liq) ** (1 / n_anos) - 1.0
    ret_anual_idx = (1 + ret_acum_idx) ** (1 / n_anos) - 1.0
    alfa_liq = 100 * (ret_anual_liq - ret_anual_idx)
    ir_liq = alfa_liq / te_anual if te_anual > 0 else np.nan

    metricas_globais = {
        "indice": nome_indice,
        "K": K,
        "lambda_anual": lambda_anual,
        "gamma": gamma,
        "custo_transacao": custo_transacao,
        "janela_treino": janela_treino,
        "horizonte_teste": horizonte_teste,
        "seed": seed,
        "rodadas": len(df_janelas),
        "dias_teste": int(df_janelas["dias"].sum()),
        "TE (% a.a.)": te_anual,
        "TD (p.p. a.a.)": vies_anual,
        "RMSE (% a.a.)": rmse_anual,
        "Giro médio (% trim)": 100 * giro_medio_trimestre,
        "Giro anualizado (%)": 100 * rebal_por_ano * giro_medio_trimestre,
        "Custo fricção (% a.a.)": 100 * custo_anual_friccao,
        "Retorno acumulado bruto (%)": 100 * ret_acum_bruto,
        "Retorno acumulado líquido (%)": 100 * ret_acum_liq,
        "Retorno acumulado índice (%)": 100 * ret_acum_idx,
        "Retorno anual líquido (%)": 100 * ret_anual_liq,
        "Alfa líquido (p.p. a.a.)": alfa_liq,
        "Information Ratio Líquido": ir_liq,
    }

    if retornar_series:
        return metricas_globais, df_janelas, pd.DataFrame({"carteira": r_cart_todos, "indice": r_idx_todos})
    return metricas_globais, df_janelas
