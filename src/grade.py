"""Grade da escolha: 448 carteiras (K × histórico × frequência × λ × índice) simuladas no período de escolha, só com rodadas que
terminam até 01/10/2024. Um arquivo por carteira em results/selecao/; arquivos já gravados não são refeitos."""
from itertools import product
from multiprocessing import Pool

from src.config import ARQ, CONFIGURACOES, CORTE, CUSTOS, GRADE_K, GRADE_LAMBDAS, INDICES, RESULTADOS, SEM_LIMITE
from src.data_loader import preparar_bases
from src.walk_forward import construir_janelas, simular_walk_forward

SAIDA = RESULTADOS / "selecao"


def intervalo_de_teste(indice, bases):
    """Primeiro e último pregão avaliados, os mesmos em todas as configurações (os da referência de 2 anos e rebalanceamento trimestral)."""
    j = construir_janelas(indice, bases)
    return j[0]["datas_teste"][0], j[-1]["datas_teste"][-1]


def arquivo(pasta, K, indice, J, H, lam):
    return pasta / f"K{K}_{ARQ[indice]}_J{J}_H{H}_L{lam:g}.csv"


def simular(K, indice, J, H, lam, fim=None):
    """Walk-forward de uma carteira, do primeiro pregão avaliado até `fim` (padrão: o último dos dados)."""
    bases = preparar_bases()
    inicio, ultimo = intervalo_de_teste(indice, bases)
    _, r = simular_walk_forward(nome_indice=indice, bases=bases, K=None if K == SEM_LIMITE else K, lambda_anual=lam,
                                custo_transacao=CUSTOS[indice], janela_treino=J, horizonte_teste=H, inicio_teste=inicio,
                                fim_teste=fim if fim is not None else ultimo, metodo="heuristica")
    return r.assign(indice=indice, K=K, janela=J, horizonte=H, lam=lam)


def tarefa(c):
    arq = arquivo(SAIDA, *c)
    if not arq.exists():
        r = simular(*c, fim=CORTE)
        r[(r["data_fim"] <= CORTE) & (r["dias"] == c[3])].to_csv(arq, index=False)  # só rodadas completas (H pregões)
    return c


def celulas():
    return [(K, i, J, H, lam) for K, (J, H), i, lam in product(GRADE_K, CONFIGURACOES, INDICES, GRADE_LAMBDAS)]


def rodar(processos=7):
    SAIDA.mkdir(parents=True, exist_ok=True)
    falta = [c for c in celulas() if not arquivo(SAIDA, *c).exists()]
    print(f"Grade: {len(celulas()) - len(falta)} de {len(celulas())} carteiras prontas.", flush=True)
    if falta:
        with Pool(processos) as p:
            for n, c in enumerate(p.imap_unordered(tarefa, falta), 1):
                print(f"[{n}/{len(falta)}] {c}", flush=True)
