"""Reproduz o modelo final a partir de data/cache_dados_finor.pkl: grade → regra → teste final → alternativa simples → avaliação →
figuras. As simulações pulam os arquivos já gravados em results/selecao/ e results/teste_final/; apague-os para refazê-las.

    python -m src.pipeline              # tudo (com os resultados do repositório, cerca de 1 minuto)
    python -m src.pipeline --sem-figuras
"""
import argparse

import pandas as pd

from src import alternativa_simples, avaliacao, grade, regra, teste_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processos", type=int, default=6)
    ap.add_argument("--sem-figuras", action="store_true")
    a = ap.parse_args()
    grade.rodar(a.processos)
    e = regra.escolher(refazer=True)
    print(e[["cenario", "teto", "ind", "K", "J", "H", "lam", "chance", "limite", "cabe"]].round(2).to_string(index=False))
    teste_final.rodar(a.processos)
    alternativa_simples.rodar()
    n = avaliacao.numeros()
    with pd.option_context("display.width", 200, "display.max_colwidth", 110, "display.max_rows", 500):
        print(n.to_string(index=False))
    if not a.sem_figuras:
        from src import figuras
        figuras.gerar()


if __name__ == "__main__":
    main()
