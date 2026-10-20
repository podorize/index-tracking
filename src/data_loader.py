"""
data_loader.py
Carrega e trata os dados históricos do S&P 100 e IBOVESPA para o Projeto Finor.
Inclui as correções auditadas:
  - Recorte de código reaproveitado da Natura (NATU3)
  - Correção de escala da Ultrapar (UGPA3) em 28/06/2021
  - Correção de data ex do dividendo da Copasa (CSMG3) em 19/11/2020
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import yfinance as yf

DATA_INICIAL = pd.Timestamp("2019-09-30")
DATA_FINAL = pd.Timestamp("2026-09-30")
DATA_FINAL_EXCLUSIVA = DATA_FINAL + pd.Timedelta(days=1)
LIMITE_FFILL_DIAS = 2
LACUNA_REUTILIZACAO = 60
LIMIAR_SALTO_PRECO = 0.40
LIMIAR_INDICE_ESTAVEL = 0.02
LIMIAR_SALTO_FATOR = 0.25
TOLERANCIA_OFICIAL = 0.02

TICKERS_SP100 = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AMAT', 'AMD', 'AMGN', 'AMT', 'AMZN',
    'ANET', 'AVGO', 'AXP', 'BA', 'BAC', 'BKNG', 'BLK', 'BMY', 'BNY', 'BRK-B',
    'C', 'CAT', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX',
    'DE', 'DELL', 'DHR', 'DIS', 'DUK', 'EMR', 'FDX', 'GD', 'GE', 'GEV',
    'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'IBM', 'INTC', 'INTU', 'ISRG',
    'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'LRCX', 'MA', 'MCD',
    'MDLZ', 'MDT', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE',
    'NFLX', 'NOW', 'NVDA', 'ORCL', 'PANW', 'PEP', 'PFE', 'PG', 'PLTR', 'PM',
    'QCOM', 'RTX', 'SBUX', 'SCHW', 'SNDK', 'SO', 'T', 'TMO', 'TMUS', 'TSLA',
    'TXN', 'UBER', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
]

TICKERS_IBOV = [
    'ALOS3.SA', 'ABEV3.SA', 'ASAI3.SA', 'AURE3.SA', 'AXIA3.SA', 'AZZA3.SA', 'B3SA3.SA',
    'BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BRAP4.SA', 'BBAS3.SA', 'BRAV3.SA', 'BPAC11.SA',
    'CXSE3.SA', 'CEAB3.SA', 'CMIG4.SA', 'COGN3.SA', 'CSMG3.SA', 'CPLE3.SA', 'CSAN3.SA',
    'CPFE3.SA', 'CMIN3.SA', 'CURY3.SA', 'CYRE3.SA', 'DIRR3.SA', 'EMBJ3.SA', 'ENGI11.SA',
    'ENEV3.SA', 'EGIE3.SA', 'EQTL3.SA', 'FLRY3.SA', 'GGBR4.SA', 'GOAU4.SA', 'HAPV3.SA',
    'HYPE3.SA', 'IGTI11.SA', 'ISAE4.SA', 'ITSA4.SA', 'ITUB4.SA', 'KLBN11.SA', 'RENT3.SA',
    'LREN3.SA', 'MGLU3.SA', 'POMO4.SA', 'MBRF3.SA', 'BEEF3.SA', 'MOTV3.SA', 'MRVE3.SA',
    'MULT3.SA', 'NATU3.SA', 'PETR3.SA', 'PETR4.SA', 'PSSA3.SA', 'PRIO3.SA', 'RADL3.SA',
    'RDOR3.SA', 'RAIL3.SA', 'SBSP3.SA', 'SANB11.SA', 'CSNA3.SA', 'SMFT3.SA', 'SUZB3.SA',
    'TAEE11.SA', 'VIVT3.SA', 'TEND3.SA', 'TIMS3.SA', 'TOTS3.SA', 'UGPA3.SA', 'USIM5.SA',
    'VALE3.SA', 'VAMO3.SA', 'VBBR3.SA', 'VIVA3.SA', 'WEGE3.SA', 'YDUQ3.SA'
]

BENCHMARKS = {"S&P100": "^OEX", "IBOV": "^BVSP"}

MUDANCAS_COMPOSICAO = pd.DataFrame([
    {"indice": "S&P100", "data efetiva": "2026-09-21", "movimento": "entra", "tickers": "DELL, PANW, ANET, SNDK"},
    {"indice": "S&P100", "data efetiva": "2026-09-21", "movimento": "sai", "tickers": "NKE, HONA, SPG, CL"},
])

ENTRADAS_NO_INDICE = {
    (linha["indice"], t.strip()): pd.Timestamp(linha["data efetiva"])
    for _, linha in MUDANCAS_COMPOSICAO[MUDANCAS_COMPOSICAO["movimento"] == "entra"].iterrows()
    for t in linha["tickers"].split(",")
}

DIVIDENDO_CSMG3 = 6.4876594827
F_CSMG3 = 1 - DIVIDENDO_CSMG3 / 52.80
COTACOES_OFICIAIS = {
    "UGPA3.SA": {"fonte": "B3 (COTAHIST jun/2021)",
                 "fechamentos": {"2021-06-25": 18.99, "2021-06-28": 19.19}},
    "CSMG3.SA": {"fonte": "B3 (COTAHIST nov/2020)",
                 "fechamentos": {"2020-11-18": 51.49 * F_CSMG3, "2020-11-19": 52.80 * F_CSMG3,
                                 "2020-11-20": 44.86, "2020-11-23": 46.23}},
}


def baixar_dados_yahoo(cache_path=Path("data/yahoo_raw.pkl"), tamanho_lote=40):
    if cache_path.exists():
        print(f"Carregando dados brutos do cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    todos_tickers = list(dict.fromkeys(TICKERS_SP100 + TICKERS_IBOV + ["^OEX", "^BVSP"]))
    campos = {"Close": [], "Adj Close": [], "Volume": []}

    print(f"Iniciando download de {len(todos_tickers)} tickers do Yahoo Finance...")
    for inicio in range(0, len(todos_tickers), tamanho_lote):
        lote = todos_tickers[inicio:inicio + tamanho_lote]
        print(f"  Baixando lote {inicio // tamanho_lote + 1}: {len(lote)} ativos...")
        raw = yf.download(
            lote, start=DATA_INICIAL, end=DATA_FINAL_EXCLUSIVA,
            auto_adjust=False, actions=False, progress=False,
            group_by="column", threads=True, timeout=40
        )
        for campo in campos:
            if campo in raw.columns.levels[0]:
                df_campo = raw[campo].copy()
            else:
                df_campo = pd.DataFrame(index=raw.index)
            campos[campo].append(df_campo)

    yahoo = {}
    for campo, lista in campos.items():
        combinado = pd.concat(lista, axis=1)
        combinado = combinado.loc[:, ~combinado.columns.duplicated()]
        combinado.index = pd.to_datetime(combinado.index).tz_localize(None)
        yahoo[campo] = combinado.sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(yahoo, f)
    print(f"Download concluído e salvo em {cache_path}.")
    return yahoo


def recortar_tickers_reutilizados(painel, nome_indice, calendario, lacuna=LACUNA_REUTILIZACAO):
    saida = painel.copy()
    for ticker in painel.columns:
        s = painel[ticker].reindex(calendario)
        if s.notna().sum() == 0:
            continue
        trecho = s.loc[s.first_valid_index():s.last_valid_index()]
        grupos = trecho.isna().ne(trecho.isna().shift()).cumsum()
        tamanhos = trecho.isna().groupby(grupos).agg(["all", "size"])
        lacunas = tamanhos[tamanhos["all"] & (tamanhos["size"] > lacuna)]
        if lacunas.empty:
            continue
        ultimo_grupo = lacunas.index.max()
        inicio_novo = trecho[grupos > ultimo_grupo].first_valid_index()
        novo = trecho.loc[inicio_novo:].dropna()
        repetidos_iniciais = int(novo.eq(novo.iloc[0]).cummin().sum())
        primeiro_real = novo.index[repetidos_iniciais] if 1 < repetidos_iniciais < len(novo) else novo.index[0]
        saida.loc[:primeiro_real - pd.Timedelta(days=1), ticker] = np.nan
    return saida


def corrigir_fator_ajuste(close, adj, r_indice, limiar=LIMIAR_SALTO_FATOR):
    corrigido = adj.copy()
    r_adj = adj.pct_change(fill_method=None)
    for ticker in adj.columns:
        f = (adj[ticker] / close[ticker]).dropna()
        if len(f) < 2:
            continue
        dlog = np.log(f).diff().fillna(0.0)
        saltos = dlog.index[dlog.abs() > np.log1p(limiar)]
        anomalos = [d for d in saltos
                    if abs(r_adj[ticker].get(d, 0.0)) > LIMIAR_SALTO_PRECO
                    and abs(r_indice.get(d, 0.0)) < LIMIAR_INDICE_ESTAVEL]
        if len(anomalos) == 0:
            continue
        dlog_corr = dlog.copy()
        dlog_corr.loc[anomalos] = 0.0
        soma_futura = dlog_corr[::-1].cumsum()[::-1] - dlog_corr
        log_f_corr = np.log(f.iloc[-1]) - soma_futura
        corrigido.loc[f.index, ticker] = close.loc[f.index, ticker] * np.exp(log_f_corr)
    return corrigido


def ffill_interno_limitado(painel, limite=2):
    saida = painel.copy()
    for col in saida:
        primeiro = saida[col].first_valid_index()
        ultimo = saida[col].last_valid_index()
        if primeiro is not None and ultimo is not None:
            saida.loc[primeiro:ultimo, col] = saida.loc[primeiro:ultimo, col].ffill(limit=limite)
    return saida


def preparar_bases(cache_bases_path=Path("data/bases_limpas.pkl")):
    if cache_bases_path.exists():
        print(f"Carregando bases preparadas de {cache_bases_path}")
        with open(cache_bases_path, "rb") as f:
            return pickle.load(f)

    yahoo = baixar_dados_yahoo()
    yahoo_bruto = {c: p.copy() for c, p in yahoo.items()}

    # Etapa 2: códigos reaproveitados
    yahoo["Adj Close"].loc[:, TICKERS_IBOV] = recortar_tickers_reutilizados(
        yahoo["Adj Close"][TICKERS_IBOV], "IBOV", yahoo["Close"]["^BVSP"].dropna().index
    )
    yahoo["Close"].loc[:, TICKERS_SP100] = recortar_tickers_reutilizados(
        yahoo["Close"][TICKERS_SP100], "S&P100", yahoo["Close"]["^OEX"].dropna().index
    )

    # Etapa 3: fator de ajuste
    r_ibov_bench = yahoo["Close"]["^BVSP"].pct_change(fill_method=None)
    yahoo["Adj Close"].loc[:, TICKERS_IBOV] = corrigir_fator_ajuste(
        yahoo_bruto["Close"][TICKERS_IBOV], yahoo["Adj Close"][TICKERS_IBOV], r_ibov_bench
    )

    # Etapa 3: cotações oficiais
    calendario_ibov = yahoo["Close"]["^BVSP"].dropna().index
    for ticker, caso in COTACOES_OFICIAIS.items():
        niveis = pd.Series(caso["fechamentos"])
        niveis.index = pd.to_datetime(niveis.index)
        datas = [d for d in niveis.index if d in calendario_ibov]
        for d0, d1 in zip(datas[:-1], datas[1:]):
            serie = yahoo["Adj Close"][ticker]
            r_oficial = niveis[d1] / niveis[d0] - 1
            r_yahoo = serie.get(d1, np.nan) / serie.get(d0, np.nan) - 1
            if pd.notna(r_yahoo) and abs(r_yahoo - r_oficial) > TOLERANCIA_OFICIAL:
                yahoo["Adj Close"].loc[:d1 - pd.Timedelta(days=1), ticker] *= (1 + r_yahoo) / (1 + r_oficial)

    bases = {}
    especificacao = {
        "S&P100": {"tickers": TICKERS_SP100, "campo_componentes": "Close", "benchmark": "^OEX", "tipo_retorno": "preço"},
        "IBOV": {"tickers": TICKERS_IBOV, "campo_componentes": "Adj Close", "benchmark": "^BVSP", "tipo_retorno": "retorno total aproximado"},
    }

    for nome, spec in especificacao.items():
        p_bench = yahoo["Close"][spec["benchmark"]].dropna().rename(nome)
        calendario = p_bench.index
        p_ativos = ffill_interno_limitado(yahoo[spec["campo_componentes"]][spec["tickers"]].reindex(calendario), LIMITE_FFILL_DIAS)
        r_ativos = p_ativos.pct_change(fill_method=None)
        r_bench = p_bench.pct_change(fill_method=None).rename(nome)
        bases[nome] = {
            "precos": p_ativos,
            "R": r_ativos,
            "y": r_bench,
            "tipo_retorno": spec["tipo_retorno"],
            "campo_yahoo": spec["campo_componentes"],
        }

    cache_bases_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_bases_path, "wb") as f:
        pickle.dump(bases, f)
    print(f"Bases limpas salvas em {cache_bases_path}.")
    return bases


if __name__ == "__main__":
    b = preparar_bases()
    print("S&P100:", b["S&P100"]["R"].shape, "pregoes:", len(b["S&P100"]["y"]))
    print("IBOV:  ", b["IBOV"]["R"].shape, "pregoes:", len(b["IBOV"]["y"]))
