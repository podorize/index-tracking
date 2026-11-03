"""Parâmetros do modelo final: períodos, custos, grade da escolha e cenários de apetite ao risco."""
from pathlib import Path

import pandas as pd

RESULTADOS = Path("results")
FIGURAS = Path("figures/final")

CORTE = pd.Timestamp("2024-10-01")       # rodadas até aqui escolhem os parâmetros; as seguintes são o teste final
INDICES = ("S&P100", "IBOV")
ARQ = {"S&P100": "sp100", "IBOV": "ibov"}
CUSTOS = {"S&P100": 0.0005, "IBOV": 0.0015}  # custo por operação, sobre o valor negociado

SEM_LIMITE = 999                          # K = 999 nos nomes de arquivo: sem restrição de cardinalidade (K = None no modelo)
GRADE_K = [5, 10, 15, 20, 30, 50, SEM_LIMITE]
CONFIGURACOES = [(504, 63), (504, 21), (252, 63), (252, 21)]  # (pregões de treino, pregões entre rebalanceamentos)
GRADE_LAMBDAS = [0.0, 0.5, 1.5, 3.0, 6.0, 12.0, 25.0, 50.0]

LIMITE_12M = 4.0                          # p.p. em 12 meses (Resolução CVM 175, Anexo Normativo V, Art. 27)
CENARIOS = {"Conservador": 10, "Moderado": 20, "Arrojado": 30}  # chance máxima aceita de passar do limite (%)
RECOMENDADO = "Moderado"
# carteiras avaliadas no teste além das dos cenários: a de 20 ações do Ibovespa, que a regra deixou de fora
EXTRAS = {"IBOV": [20]}
