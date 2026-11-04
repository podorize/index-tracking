"""App da Finor: escolha o índice e o risco aceito (ou o número de ações) e veja a carteira contra o índice e o desvio.

    streamlit run app.py

Lê results/app/dados_app.pkl, gravado por `python -m src.app_dados`.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import cache

RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "results" / "app" / "dados_app.pkl"
SEM_LIMITE = 999
LIMITE = 4.0
NOMES = {"S&P100": "S&P 100", "IBOV": "Ibovespa"}
CORES = {"S&P100": "#2a78d6", "IBOV": "#eb6834"}
CINZA, VERMELHO = "#898781", "#c62828"
CENARIOS = {"Conservador": 10, "Moderado": 20, "Arrojado": 30}

st.set_page_config(page_title="Finor: index tracking", page_icon="📈", layout="wide")


@st.cache_data
def carregar():
    return cache.ler(DADOS)


def nome_K(K, efetivo=None):
    if K == SEM_LIMITE:
        return f"todas ({efetivo})" if efetivo else "todas"
    return str(K)


def fmt(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


def mes(d):
    return f"{['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez'][d.month - 1]}/{d.year}"


if not DADOS.exists():
    st.error("Faltam os dados do app. Gere-os com `python -m src.app_dados`.")
    st.stop()

dados = carregar()
CORTE = pd.Timestamp(dados["corte"])

# ---------------------------------------------------------------- escolhas
with st.sidebar:
    st.header("Escolhas")
    ind = st.radio("Índice", list(NOMES), format_func=NOMES.get, horizontal=True)
    d = dados["indices"][ind]
    curva = d["curva"]
    modo = st.radio("Como escolher a carteira", ["Pelo risco aceito", "Pelo número de ações"])
    if modo == "Pelo risco aceito":
        teto = st.slider("Risco máximo aceito (%)", 5, 40, 20, 1,
                         help="Chance máxima aceita de a diferença entre carteira e índice passar de 4 p.p. em 12 meses.")
        st.caption("Cenários do relatório: " + " · ".join(f"{c} {v}%" for c, v in CENARIOS.items()))
        ok = curva[curva.limite <= teto + 1e-9]
        cabe = len(ok) > 0
        K = int(ok.K.iloc[0]) if cabe else SEM_LIMITE
    else:
        K = st.select_slider("Número de ações", options=list(curva.K), value=30, format_func=nome_K)
        teto, cabe = None, True
    c = d["config"]
    st.divider()
    st.caption(f"Forma de montar escolhida para o {NOMES[ind]}: {int(c['J']) // 252} ano de histórico, carteira refeita "
               f"{'a cada trimestre' if c['H'] == 63 else 'todo mês'}, penalidade de giro λ = {fmt(c['lam'], 0)}.")
    st.caption(f"Escolha: rodadas até {CORTE:%d/%m/%Y}. Teste: o período seguinte, que não entrou na escolha.")

cart = d["carteiras"][K]
linha = curva[curva.K == K].iloc[0]
rod, diario, pesos = cart["rodadas"], cart["diario"], cart["pesos"]
efetivo = int(rod.K_efetivo.iloc[-1])

# ---------------------------------------------------------------- cabeçalho
st.title(f"{NOMES[ind]}: carteira de {efetivo} ações" + (" (sem limite)" if K == SEM_LIMITE else ""))
if teto is not None:
    if cabe:
        st.markdown(f"Com risco máximo de **{teto}%**, a regra escolhe **{nome_K(K, efetivo)} ações**: é o menor tamanho cuja "
                    f"chance prevista mais um erro padrão ({fmt(linha.chance)}% + {fmt(linha.ep)} = {fmt(linha.limite)}%) "
                    f"cabe no risco.")
    else:
        st.warning(f"Nenhum tamanho cabe em {teto}% de risco: a menor chance prevista mais um erro padrão é "
                   f"{fmt(curva.limite.min())}%. Fica a carteira com todas as ações.")

m = st.columns(6)
m[0].metric("Chance prevista", f"{fmt(linha.chance)}%", help="Chance média de passar de 4 p.p. em 12 meses, no período de escolha.")
m[1].metric("Chance no teste", f"{fmt(linha.chance_teste)}%", help="A mesma conta, nas janelas de 12 meses inteiras no teste.")
m[2].metric("Erro no teste", f"{fmt(linha.TE_teste)}% a.a.", help="Erro de rastreamento diário anualizado no teste.")
m[3].metric("Janelas > 4 p.p.", f"{int(linha.n4)} de {int(linha.n_janelas)}",
            help="Janelas de 12 meses inteiras no teste em que a diferença acumulada passou de 4 p.p.")
m[4].metric("Maior desvio 12 meses", f"{fmt(linha.pior)} p.p.")
m[5].metric("Giro no teste", f"{fmt(linha.giro_teste, 0)}% a.a.",
            help=f"Custo de cerca de {fmt(linha.custo_por_milhao, 0)} por ano a cada 1 milhão aplicado.")

# ---------------------------------------------------------------- séries
s = diario.copy()
s.index = pd.to_datetime(s.index)
acum = (1 + s).cumprod()
doze = 100 * (acum / acum.shift(252) - 1)
desvio = (doze.carteira - doze.indice).dropna()

so_teste = st.toggle("Mostrar só o período de teste", value=False)
inicio = CORTE if so_teste else s.index[0]
base = acum.loc[inicio:]
base = 100 * (base / base.iloc[0] - 1)
ret = base.rename_axis("data").reset_index().melt("data", var_name="serie", value_name="retorno")
ret["serie"] = ret.serie.map({"carteira": "Carteira", "indice": NOMES[ind]})
cores = alt.Scale(domain=["Carteira", NOMES[ind]], range=[CORES[ind], CINZA])

faixa_teste = alt.Chart(pd.DataFrame({"a": [max(CORTE, inicio)], "b": [s.index[-1]]})).mark_rect(opacity=0.07, color="black").encode(
    x="a:T", x2="b:T")

g1 = alt.Chart(ret).mark_line(strokeWidth=2).encode(
    x=alt.X("data:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
    y=alt.Y("retorno:Q", title="Retorno acumulado (%)"),
    color=alt.Color("serie:N", scale=cores, legend=alt.Legend(title=None, orient="top-left")),
    tooltip=[alt.Tooltip("data:T", format="%d/%m/%Y"), "serie:N", alt.Tooltip("retorno:Q", format=".1f")])

dv = desvio.loc[inicio:].rename("desvio").rename_axis("data").reset_index()
dv["fora"] = dv.desvio.abs() > LIMITE
regras = alt.Chart(pd.DataFrame({"y": [-LIMITE, LIMITE]})).mark_rule(color=VERMELHO, strokeDash=[5, 4]).encode(y="y:Q")
zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=CINZA, strokeWidth=0.8).encode(y="y:Q")
g2 = alt.Chart(dv).mark_line(strokeWidth=1.8, color=CORES[ind]).encode(
    x=alt.X("data:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
    y=alt.Y("desvio:Q", title="Carteira − índice em 12 meses (p.p.)",
            scale=alt.Scale(domain=[min(-6, dv.desvio.min() - 0.5), max(6, dv.desvio.max() + 0.5)])),
    tooltip=[alt.Tooltip("data:T", format="%d/%m/%Y", title="12 meses até"), alt.Tooltip("desvio:Q", format=".2f", title="p.p.")])
pontos = alt.Chart(dv[dv.fora]).mark_circle(color=VERMELHO, size=14).encode(x="data:T", y="desvio:Q")

e, dir_ = st.columns(2)
with e:
    st.subheader("Carteira contra o índice")
    st.altair_chart((faixa_teste + g1).properties(height=340), use_container_width=True)
    ri, rc = linha.ret_indice, linha.ret_carteira
    st.caption(f"Área sombreada: teste. No teste, a carteira rendeu {fmt(rc)}% e o índice {fmt(ri)}%, já descontados os custos "
               f"das operações.")
with dir_:
    st.subheader("Desvio em 12 meses")
    st.altair_chart((faixa_teste + zero + regras + g2 + pontos).properties(height=340), use_container_width=True)
    fora = dv[dv.data >= CORTE].fora.mean() * 100 if (dv.data >= CORTE).any() else np.nan
    st.caption(f"Diferença acumulada nos 12 meses até cada dia. Linhas vermelhas: o limite de ±4 p.p. da CVM. No teste, a "
               f"diferença ficou fora do limite em {fmt(fora, 0)}% dos dias. Antes do teste, parte do desvio vem do viés de "
               f"sobrevivência (empresas que saíram do índice não estão nos dados).")

# ---------------------------------------------------------------- chance por tamanho
st.subheader("Chance de passar de 4 p.p. por número de ações")
cv = curva.assign(rotulo=[nome_K(k) for k in curva.K])
ordem = list(cv.rotulo)
cv_l = pd.concat([cv.assign(medida="Prevista (escolha)", valor=cv.chance),
                  cv.assign(medida="No teste", valor=cv.chance_teste)])
g3 = alt.Chart(cv_l).mark_line(point=True).encode(
    x=alt.X("rotulo:N", sort=ordem, title="Número de ações", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("valor:Q", title="Chance (%)"),
    color=alt.Color("medida:N", scale=alt.Scale(domain=["Prevista (escolha)", "No teste"], range=[CORES[ind], CINZA]),
                    legend=alt.Legend(title=None, orient="top-right")),
    tooltip=["rotulo:N", "medida:N", alt.Tooltip("valor:Q", format=".1f")])
barra = alt.Chart(cv).mark_errorbar(color=CORES[ind], opacity=0.5).encode(
    x=alt.X("rotulo:N", sort=ordem), y="chance:Q", y2="limite:Q")
escolhido = alt.Chart(cv[cv.K == K]).mark_point(size=260, color=CORES[ind], shape="circle", filled=False, strokeWidth=2).encode(
    x=alt.X("rotulo:N", sort=ordem), y="chance:Q")
camadas = barra + g3 + escolhido
if teto is not None:
    camadas += alt.Chart(pd.DataFrame({"y": [teto]})).mark_rule(color=VERMELHO, strokeDash=[5, 4]).encode(y="y:Q")
st.altair_chart(camadas.properties(height=280), use_container_width=True)
st.caption("A barra vai da chance prevista até ela mais um erro padrão; a regra compara esse valor com o risco aceito "
           "(linha vermelha). Círculo: a carteira escolhida.")

# ---------------------------------------------------------------- composição
st.subheader("Composição da carteira")
datas = sorted(pd.to_datetime(pesos.data).unique())
quando = st.select_slider("Rodada", options=datas, value=datas[-1], format_func=lambda x: f"{pd.Timestamp(x):%d/%m/%Y}")
p = pesos[pd.to_datetime(pesos.data) == quando].set_index("acao").peso
i = datas.index(quando)
antes = pesos[pd.to_datetime(pesos.data) == datas[i - 1]].set_index("acao").peso if i else pd.Series(dtype=float)
tab = pd.DataFrame({"Peso (%)": 100 * p, "Antes (%)": 100 * antes.reindex(p.index).fillna(0)})
tab["Mudança (p.p.)"] = tab["Peso (%)"] - tab["Antes (%)"]
tab = tab.sort_values("Peso (%)", ascending=False)
tab.index = [a.removesuffix(".SA") for a in tab.index]
tab.index.name = "Ação"
saem = sorted(a.removesuffix(".SA") for a in antes.index.difference(p.index))

a, b = st.columns([3, 2])
with a:
    g4 = alt.Chart(tab.reset_index()).mark_bar(color=CORES[ind]).encode(
        x=alt.X("Peso (%):Q"), y=alt.Y("Ação:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
        tooltip=["Ação", alt.Tooltip("Peso (%):Q", format=".2f")])
    st.altair_chart(g4.properties(height=max(200, 18 * len(tab))), use_container_width=True)
with b:
    st.markdown(f"**{len(tab)} ações** na rodada que começa em {pd.Timestamp(quando):%d/%m/%Y}. As 10 maiores somam "
                f"{fmt(tab['Peso (%)'].head(10).sum())}%.")
    if i:
        entram = [x for x in tab.index if tab.loc[x, "Antes (%)"] == 0]
        st.markdown(f"Entram: {', '.join(entram) or 'nenhuma'}  \nSaem: {', '.join(saem) or 'nenhuma'}  \n"
                    f"Giro da rodada: {fmt(100 * rod.loc[rod.data_inicio == quando, 'giro'].iloc[0])}%")
    st.dataframe(tab.round(2), use_container_width=True, height=360)
    st.download_button("Baixar a carteira (CSV)", tab.round(4).to_csv().encode("utf-8"),
                       file_name=f"carteira_{ind.lower().replace('&', '')}_{nome_K(K)}_{pd.Timestamp(quando):%Y%m%d}.csv")

st.divider()
st.caption("Carteiras simuladas em walk-forward: em cada rodada, a heurística escolhe as ações e os pesos que minimizam o erro "
           "contra o índice no histórico de treino, com penalidade sobre o giro, e a carteira fica parada até a rodada seguinte. "
           "A chance trata a diferença de 12 meses como normal, com largura igual ao erro de rastreamento e centro descontado do "
           "viés de sobrevivência. Método completo em METODOLOGIA.md.")
