"""
pesos_aproximados.py
Gera results/pesos_aproximados.csv, usado pelo baseline "Top K por peso" (Seção 3.2 do Guia Parte 2).
Mesma aproximação da Parte 1 (Seção 7.14 do Colab): peso = valor de mercado atual do Yahoo / soma do índice.
  - Classes da mesma empresa (GOOG/GOOGL, PETR3/PETR4, BBDC3/BBDC4) contam uma vez: o Yahoo informa,
    para cada classe, o valor de mercado da empresa inteira.
  - No Ibovespa a aproximação ignora o free float e superestima empresas com controlador.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import yfinance as yf

from src.data_loader import TICKERS_SP100, TICKERS_IBOV

PARES_MESMA_EMPRESA = [("GOOG", "GOOGL"), ("PETR3.SA", "PETR4.SA"), ("BBDC3.SA", "BBDC4.SA")]
CAMINHO = Path("results/pesos_aproximados.csv")


def valor_de_mercado(ticker):
    try:
        return ticker, yf.Ticker(ticker).get_info().get("marketCap")
    except Exception:
        return ticker, None


def gerar_pesos_aproximados(caminho=CAMINHO):
    linhas = []
    for nome, tickers in [("S&P100", TICKERS_SP100), ("IBOV", TICKERS_IBOV)]:
        with ThreadPoolExecutor(max_workers=5) as executor:
            m = pd.Series(dict(executor.map(valor_de_mercado, tickers)), dtype=float).dropna()
        m = m[m > 0]
        for a, b in PARES_MESMA_EMPRESA:
            if a in m and b in m:
                m = m.drop(a if m[a] < m[b] else b)
        w = (m / m.sum()).sort_values(ascending=False)
        linhas.append(pd.DataFrame({"indice": nome, "ticker": w.index, "peso": w.values}))
    pesos = pd.concat(linhas, ignore_index=True)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    pesos.to_csv(caminho, index=False)
    print(f"Pesos aproximados salvos em {caminho} ({len(pesos)} ações).")
    return pesos


if __name__ == "__main__":
    p = gerar_pesos_aproximados()
    for nome, g in p.groupby("indice"):
        print(nome, "10 maiores:", ", ".join(f"{t} {100 * w:.1f}%" for t, w in zip(g["ticker"].head(10), g["peso"].head(10))))
