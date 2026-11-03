"""Avaliação das carteiras escolhidas no teste final (rodadas depois de 01/10/2024), com a mesma conta da escolha, e os números
citados no relatório (results/numeros_relatorio.csv)."""
import numpy as np
import pandas as pd

from src.config import CORTE, CUSTOS, INDICES, RECOMENDADO, RESULTADOS, SEM_LIMITE
from src.grade import arquivo
from src.metricas import agregar_periodo, chance, diferencas, janelas, rodadas
from src.teste_final import SAIDA


def ler(ind, K, J, H, lam):
    return rodadas(arquivo(SAIDA, K, ind, J, H, lam), H)


def rodadas_de_teste(r):
    return r[r.data_fim > CORTE]


def inteiras(r, n):
    """Janelas de n rodadas que ficam inteiras dentro do teste."""
    t = (r.data_fim > CORTE).to_numpy()
    return np.array([t[k:k + n].all() for k in range(len(r) - n + 1)])


def janelas_teste(ind, K, J, H, lam):
    """Janelas de 12 meses inteiras no teste: chance (centro contra a carteira sem limite da mesma configuração) e diferença acumulada."""
    r, n = ler(ind, K, J, H, lam), 252 // H
    ok = inteiras(r, n)
    w = janelas(r, ler(ind, SEM_LIMITE, J, H, lam), H)[ok].reset_index(drop=True)
    return w.assign(diferenca=diferencas(r, n)[ok])


def no_teste(ind, K, J, H, lam):
    r = ler(ind, K, J, H, lam)
    t, w = rodadas_de_teste(r), janelas_teste(ind, K, J, H, lam)
    g = np.abs(w.diferenca)
    giro = t.loc[t.numero > r.numero.min(), "giro"].mean() * 252 / H * 100
    return {"chance_teste": w.chance.mean(), "TE_janelas_teste": w.TE.mean(), "mu_teste": w.mu.mean(),
            "n4": int((g > 4).sum()), "n3": int((g > 3).sum()), "n": len(g), "pior": g.max(),
            "TE_teste": agregar_periodo(t)["TE (% a.a.)"], "giro_teste": giro, "custo_por_milhao": giro / 100 * CUSTOS[ind] * 1e6,
            "ret_carteira": 100 * ((1 + t.retorno_liq_carteira).prod() - 1), "ret_indice": 100 * ((1 + t.retorno_indice).prod() - 1),
            "fim": t.data_fim.max()}


def curva():
    """Uma linha por carteira avaliada: chance e erro previstos na escolha e os mesmos números no teste."""
    from src.regra import carteiras, cenarios
    cen = cenarios()
    d = pd.read_csv(RESULTADOS / "selecao_chance_12m.csv")
    linhas = []
    for ind, K, J, H, lam in carteiras():
        s = d[(d.ind == ind) & (d.K == K) & (d.J == J) & (d.H == H) & (d.lam == lam)].iloc[0]
        nomes = " e ".join(c.lower() for c, v in cen.items() if v[1][ind] == K)
        linhas.append({"indice": ind, "K": K, "cenario": nomes, "J": J, "H": H, "lam": lam, "chance_val": s.chance,
                       "TE_janelas_val": s.TE, **no_teste(ind, K, J, H, lam)})
    c = pd.DataFrame(linhas)
    c.to_csv(RESULTADOS / "teste_carteiras.csv", index=False)
    return c


def decomposicao(c):
    """Decomposição de Shapley da mudança da chance (escolha → teste) entre o erro de rastreamento (largura) e o desvio (centro)."""
    from src.regra import janelas_todas
    W = janelas_todas()
    linhas = []
    for s in c[c.cenario != ""].itertuples():
        e = W[(W.ind == s.indice) & (W.K == s.K) & (W.J == s.J) & (W.H == s.H) & (W.lam == s.lam)]
        t = janelas_teste(s.indice, s.K, s.J, s.H, s.lam)
        T0, T1, m0, m1 = e.TE.mean(), t.TE.mean(), e.mu.abs().mean(), t.mu.abs().mean()
        dT = 0.5 * ((chance(m0, T1) - chance(m0, T0)) + (chance(m1, T1) - chance(m1, T0)))
        dm = 0.5 * ((chance(m1, T0) - chance(m0, T0)) + (chance(m1, T1) - chance(m0, T1)))
        linhas.append({"ind": s.indice, "K": s.K, "cen": s.cenario, "chance_esc": e.chance.mean(), "chance_teste": t.chance.mean(),
                       "TE_esc": T0, "TE_teste": T1, "mu_esc": m0, "mu_teste": m1, "dTE": dT, "dmu": dm,
                       "delta": t.chance.mean() - e.chance.mean(), "aprox": dT + dm})
    d = pd.DataFrame(linhas)
    d.to_csv(RESULTADOS / "decomposicao_chance.csv", index=False)
    return d


def diagnostico_mercado():
    """Volatilidade do índice, dispersão das ações em torno dele e correlação mediana com ele, na escolha e no teste."""
    from src.data_loader import preparar_bases
    b, linhas = preparar_bases(), []
    for ind in INDICES:
        R, y = b[ind]["R"], b[ind]["y"]
        for periodo, a, z in (("escolha", "2021-10-01", "2024-09-30"), ("teste", "2024-10-01", "2026-09-30")):
            r, yy = R.loc[a:z], y.loc[a:z]
            linhas.append({"ind": ind, "periodo": periodo, "vol_indice": yy.std() * np.sqrt(252) * 100,
                           "dispersao": r.sub(yy, axis=0).std(axis=1).mean() * np.sqrt(252) * 100,
                           "corr_mediana": r.corrwith(yy).median()})
    d = pd.DataFrame(linhas)
    d.to_csv(RESULTADOS / "diagnostico_mercado.csv", index=False)
    return d


def alternativa(ind, K):
    """A alternativa simples no teste: erro do período, giro e janelas de 12 meses acima de 4 p.p."""
    from src.alternativa_simples import arquivo as arq_alt
    r = rodadas(arq_alt(ind, K), 21)
    t, ok = rodadas_de_teste(r), inteiras(r, 12)
    g = np.abs(diferencas(r, 12)[ok])
    return {"TE_teste": agregar_periodo(t)["TE (% a.a.)"], "giro_teste": t.giro.mean() * 12 * 100,
            "n4": int((g > 4).sum()), "n": len(g)}


def vies_sobrevivencia(ind, J, H, lam):
    """Diferença de 12 meses da carteira sem limite contra o índice, antes e depois do corte: mede o viés de sobrevivência."""
    n = 252 // H
    r = ler(ind, SEM_LIMITE, J, H, lam)
    g = pd.Series(diferencas(r, n), index=r.data_fim.iloc[n - 1:].to_numpy())
    antes, depois = g[g.index <= CORTE], g[g.index > CORTE]
    return {"media_ate_corte": antes.mean(), "max_ate_corte": antes.max(), "media_depois": depois.mean()}


def numeros():
    """Os números do relatório, recalculados a partir dos resultados."""
    from src.regra import cenarios, configuracoes
    cen, cfg = cenarios(), configuracoes()
    c = curva()
    dec = decomposicao(c)
    diag = diagnostico_mercado()
    sel = pd.read_csv(RESULTADOS / "selecao_configuracao_indice.csv")
    esc = pd.read_csv(RESULTADOS / "cenarios.csv")
    linhas = []
    add = lambda secao, item, valor: linhas.append({"secao": secao, "item": item, "valor": round(float(valor), 2)})
    for ind in INDICES:
        x = sel[sel.ind == ind].sort_values("chance")
        add("Etapa 1", f"{ind}: chance média da combinação escolhida {cfg[ind]}", x.chance.iloc[0])
        add("Etapa 1", f"{ind}: distância da segunda colocada (p.p.)", x.chance.iloc[1] - x.chance.iloc[0])
    for e in esc.itertuples():
        add("Etapa 2", f"{e.cenario} ({e.teto}%), {e.ind}: K = {e.K}, chance + erro padrão", e.limite)
    for s in c.itertuples():
        nome = f"{s.indice}, K = {s.K}" + (f" ({s.cenario})" if s.cenario else " (fora dos cenários)")
        add("Teste", f"{nome}: chance prevista", s.chance_val)
        add("Teste", f"{nome}: chance no teste", s.chance_teste)
        add("Teste", f"{nome}: erro médio das janelas, previsto", s.TE_janelas_val)
        add("Teste", f"{nome}: erro médio das janelas, teste", s.TE_janelas_teste)
        add("Teste", f"{nome}: erro do período de teste (% a.a.)", s.TE_teste)
        add("Teste", f"{nome}: janelas acima de 4 p.p. (de {s.n})", s.n4)
        add("Teste", f"{nome}: janelas acima de 3 p.p. (de {s.n})", s.n3)
        add("Teste", f"{nome}: maior diferença de 12 meses (p.p.)", s.pior)
        add("Teste", f"{nome}: giro (% a.a.)", s.giro_teste)
        add("Teste", f"{nome}: custo (R$ por ano a cada R$ 1 milhão)", s.custo_por_milhao)
        add("Teste", f"{nome}: retorno acumulado da carteira (%)", s.ret_carteira)
        add("Teste", f"{nome}: retorno acumulado do índice (%)", s.ret_indice)
    for ind in INDICES:
        K = cen[RECOMENDADO][1][ind]
        for k, v in alternativa(ind, K).items():
            add("Alternativa simples", f"{ind}, K = {K}: {k}", v)
        for k, v in vies_sobrevivencia(ind, *cfg[ind]).items():
            add("Viés de sobrevivência", f"{ind}, carteira sem limite: {k} (p.p.)", v)
        r = rodadas_de_teste(ler(ind, K, *cfg[ind]))
        p = r.loc[r.retorno_indice.idxmin()]
        add("Pior rodada do teste", f"{ind}, K = {K}: queda do índice até {p.data_fim:%d/%m/%Y} (%)", 100 * p.retorno_indice)
        add("Pior rodada do teste", f"{ind}, K = {K}: diferença carteira − índice (p.p.)", 100 * (p.retorno_liq_carteira - p.retorno_indice))
    for d in dec.itertuples():
        add("Decomposição", f"{d.ind}, K = {d.K}: parte do erro de rastreamento (p.p.)", d.dTE * d.delta / d.aprox)
        add("Decomposição", f"{d.ind}, K = {d.K}: parte do desvio (p.p.)", d.dmu * d.delta / d.aprox)
    for d in diag.itertuples():
        add("Mercado", f"{d.ind}, {d.periodo}: correlação mediana com o índice", d.corr_mediana)
        add("Mercado", f"{d.ind}, {d.periodo}: dispersão das ações em torno do índice (% a.a.)", d.dispersao)
    n = pd.DataFrame(linhas)
    n.to_csv(RESULTADOS / "numeros_relatorio.csv", index=False)
    return n
