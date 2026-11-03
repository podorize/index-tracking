"""Teste final dos modelos congelados (CRITERIOS_SELECAO_FINAL.md, revisão de 01/11/2026): walk-forward completo; reporta só as rodadas após o período de escolha."""
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

from src.data_loader import preparar_bases
from src.validacao_lambda import CUSTOS
from src.rebalanceamento_experiment import intervalo_de_teste
from src.walk_forward import simular_walk_forward

SAIDA = Path("results/teste_final"); CORTE = pd.Timestamp("2024-10-01")
MODELOS = {  # nome: (K, janela, horizonte, λ S&P100, λ IBOV)
    "k50_l3": (50, 252, 21, 3.0, 3.0),                # regra final: 1 ano, mensal, λ = 3; K pelo risco máximo aceito
    "k30": (30, 252, 21, 50.0, 12.0),                  # regra final: chance só pelo TE, teto de 20%
    "k20_2a": (20, 504, 21, 3.0, 3.0),                 # Ibovespa arrojado: 2 anos de treino, mensal, λ = 3
    "k15_l12": (15, 252, 21, 12.0, 3.0),              # arrojado com λ da regra (S&P 100: λ = 12)
    "k15": (15, 252, 21, 3.0, 3.0),                   # cenário arrojado (teto de 30%)
    "k30_l3": (30, 252, 21, 3.0, 3.0),                # sensibilidade pós-teste: λ moderado, não é escolha
    "k50": (50, 252, 21, 50.0, 3.0),                  # regra com teto de 10%
    "sem_limite": (None, 252, 21, 1.5, 1.5),           # referência: todas as ações disponíveis
    "k30_2a_l6": (30, 504, 21, 6.0, 6.0),             # regra λ = 0 → maior λ sem piorar a chance: S&P 100 conservador e moderado
    "k20_tri_l05": (20, 252, 63, 0.5, 0.5),           # mesma regra: Ibovespa arrojado
    "sem_limite_l3": (None, 252, 21, 3.0, 3.0),        # regra final: Ibovespa conservador (nenhum K cabe em 10% com λ = 0)
    "k30_tri_l6": (30, 252, 63, 6.0, 6.0),            # regra final sem faixa de empate: S&P 100 moderado (1 ano, trimestral, λ = 6)
    "k50_tri_l6": (50, 252, 63, 6.0, 6.0),            # idem: S&P 100 conservador
    "cvm20": (20, 252, 21, 3.0, 3.0),
    "escolhido": (20, 504, 21, 50.0, 25.0),
    "A_1ano_trimestral": (20, 252, 63, 50.0, 50.0),
    "B_original": (20, 504, 63, 6.0, 6.0),
}

def tarefa(a):
    nome, indice = a; K, J, H, ls, li = MODELOS[nome]; lam = ls if indice == "S&P100" else li
    arq = SAIDA / f"{nome}_{'sp100' if indice == 'S&P100' else 'ibov'}.csv"
    if arq.exists(): return a
    bases = preparar_bases(); ini, fim = intervalo_de_teste(indice, bases)
    _, r = simular_walk_forward(nome_indice=indice, bases=bases, K=K, lambda_anual=lam, custo_transacao=CUSTOS[indice],
                                janela_treino=J, horizonte_teste=H, inicio_teste=ini, fim_teste=fim, metodo="heuristica")
    r.assign(indice=indice, modelo=nome, K=K, janela=J, horizonte=H, lam=lam).to_csv(arq, index=False)
    return a

if __name__ == "__main__":
    SAIDA.mkdir(parents=True, exist_ok=True)
    with Pool(6) as p:
        for a in p.imap_unordered(tarefa, [(m, i) for m in MODELOS for i in ("S&P100", "IBOV")]): print("pronto", a, flush=True)
