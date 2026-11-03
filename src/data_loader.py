"""
data_loader.py
Constantes do universo (S&P 100 e Ibovespa, fotografia de 30/09/2026) e carregamento das bases limpas.

A coleta no Yahoo Finance e as correções auditadas (Natura, Ultrapar) ficam no notebook
`notebooks/finor_index_tracking.ipynb`, que grava o resultado em `data/cache_dados_finor.pkl`.
Os módulos de modelagem (`src/`) leem esse mesmo arquivo, então notebook e código usam sempre as mesmas bases.
"""

from pathlib import Path
import pandas as pd

DATA_INICIAL = pd.Timestamp("2019-09-30")
DATA_FINAL = pd.Timestamp("2026-09-30")

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

CACHE_DADOS = Path("data/cache_dados_finor.pkl")


def preparar_bases(cache=CACHE_DADOS):
    """Carrega as bases limpas (retornos de cada ação e do índice) gravadas pelo notebook."""
    if not Path(cache).exists():
        raise FileNotFoundError(
            f"{cache} não encontrado. Execute a seção 'Dados' de notebooks/finor_index_tracking.ipynb "
            "(com RECALCULAR_DADOS = True) para gerar o cache."
        )
    return pd.read_pickle(cache)["bases"]


if __name__ == "__main__":
    b = preparar_bases()
    print("S&P100:", b["S&P100"]["R"].shape, "pregões:", len(b["S&P100"]["y"]))
    print("IBOV:  ", b["IBOV"]["R"].shape, "pregões:", len(b["IBOV"]["y"]))
