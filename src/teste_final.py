"""Teste final das carteiras escolhidas pela regra: walk-forward completo, do início do período de escolha ao fim dos dados, com os
parâmetros congelados. Só as rodadas depois de 01/10/2024 entram na avaliação (src/avaliacao.py). Cada carteira é acompanhada da
carteira sem limite na mesma configuração, que é o centro da chance, como na escolha. Arquivos já gravados não são refeitos."""
from multiprocessing import Pool

from src.config import RESULTADOS, SEM_LIMITE
from src.grade import arquivo, simular
from src.regra import carteiras

SAIDA = RESULTADOS / "teste_final"


def lista():
    """(K, índice, J, H, λ) a simular: as carteiras dos cenários, as extras e a sem limite de cada configuração."""
    c = {(K, ind, J, H, lam) for ind, K, J, H, lam in carteiras()}
    return sorted(c | {(SEM_LIMITE, ind, J, H, lam) for _, ind, J, H, lam in c})


def tarefa(c):
    arq = arquivo(SAIDA, *c)
    if not arq.exists():
        simular(*c).to_csv(arq, index=False)
    return c


def rodar(processos=6):
    SAIDA.mkdir(parents=True, exist_ok=True)
    falta = [c for c in lista() if not arquivo(SAIDA, *c).exists()]
    print(f"Teste final: {len(lista()) - len(falta)} de {len(lista())} carteiras prontas.", flush=True)
    if falta:
        with Pool(processos) as p:
            for c in p.imap_unordered(tarefa, falta):
                print("pronto", c, flush=True)
