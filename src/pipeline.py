"""Reproduz o modelo final a partir de data/cache_dados_finor.pkl: grade → regra → teste final → alternativa simples → avaliação →
figuras → dados do app. As simulações pulam os arquivos já gravados em results/selecao/, results/teste_final/ e results/app/; com --refazer, elas são apagadas e
refeitas do zero, a partir dos dados.

    python -m src.pipeline                      # reaproveita as simulações gravadas (cerca de 30 segundos)
    python -m src.pipeline --refazer            # refaz as 448 carteiras da grade e o teste final a partir dos dados
    python -m src.pipeline --saida resultados   # copia os resultados e as figuras para uma pasta
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from src import alternativa_simples, app_dados, avaliacao, grade, regra, teste_final
from src.config import FIGURAS, RESULTADOS

SAIDAS = ["cenarios.csv", "teste_carteiras.csv", "numeros_relatorio.csv", "selecao_chance_12m.csv",
          "selecao_configuracao_indice.csv", "decomposicao_chance.csv", "diagnostico_mercado.csv"]


def apagar_simulacoes():
    for pasta in (grade.SAIDA, teste_final.SAIDA):
        for arq in pasta.glob("*.csv"):
            arq.unlink()
    (RESULTADOS / "janelas_12m.csv").unlink(missing_ok=True)
    app_dados.ARQUIVO.unlink(missing_ok=True)


def copiar(destino):
    destino = Path(destino)
    (destino / "figuras").mkdir(parents=True, exist_ok=True)
    for nome in SAIDAS:
        shutil.copy(RESULTADOS / nome, destino / nome)
    for fig in FIGURAS.glob("*.png"):
        shutil.copy(fig, destino / "figuras" / fig.name)
    print(f"Resultados copiados para {destino}/", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processos", type=int, default=6)
    ap.add_argument("--sem-figuras", action="store_true")
    ap.add_argument("--refazer", action="store_true", help="apaga as simulações gravadas e refaz tudo a partir dos dados")
    ap.add_argument("--saida", help="pasta para onde copiar os resultados e as figuras")
    a = ap.parse_args()
    if a.refazer:
        apagar_simulacoes()
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
    app_dados.gerar(a.processos)
    if a.saida:
        copiar(a.saida)


if __name__ == "__main__":
    main()
