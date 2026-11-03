"""
estatistica.py
Incerteza e testes pareados da comparação de modelos (Etapa 13 do relatório), a partir das séries diárias fora da amostra.

Os 20 trimestres são os blocos: cada reamostragem sorteia 20 trimestres com reposição (os dias de um trimestre ficam juntos, porque
o erro diário de um trimestre é muito correlacionado) e recalcula o TE sobre os dias reunidos. Para comparar dois modelos, o mesmo sorteio
de trimestres vale para os dois, o que isola a diferença entre eles (teste pareado).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PASTA_RESULTADOS = Path("results")
ANUAL = 252
HORIZONTE = 63
REFERENCIA = "Heurística com controle de giro"   # o modelo final


def _blocos(dif):
    pos = np.arange(len(dif))
    return [dif[(pos // HORIZONTE) == b] for b in range(int(np.ceil(len(dif) / HORIZONTE)))]


def _te(blocos, indices):
    x = np.concatenate([blocos[i] for i in indices])
    return 100 * x.std(ddof=1) * np.sqrt(ANUAL)


def analisar(K=20, B=2000, semente=42, referencia=REFERENCIA):
    s = pd.read_csv(PASTA_RESULTADOS / f"series_oos_modelos_K{K}.csv", index_col="data", parse_dates=True)
    rng = np.random.default_rng(semente)
    linhas = []
    for indice, g in s.groupby("indice_nome"):
        difs = {m: _blocos((h["carteira"] - h["indice"]).to_numpy()) for m, h in g.groupby("modelo")}
        n = len(next(iter(difs.values())))
        sorteios = rng.integers(0, n, size=(B, n))
        te_boot = {m: np.array([_te(b, idx) for idx in sorteios]) for m, b in difs.items()}
        te_q = {m: np.array([100 * b.std(ddof=1) * np.sqrt(ANUAL) for b in bl]) for m, bl in difs.items()}
        for m in difs:
            d = te_boot[m] - te_boot[referencia]
            try:
                p = wilcoxon(te_q[m], te_q[referencia]).pvalue if m != referencia else np.nan
            except ValueError:
                p = np.nan
            linhas.append({
                "indice": indice, "modelo": m, "TE (% a.a.)": _te(difs[m], range(n)),
                "TE IC95 inf": np.percentile(te_boot[m], 2.5), "TE IC95 sup": np.percentile(te_boot[m], 97.5),
                "TE − referência": _te(difs[m], range(n)) - _te(difs[referencia], range(n)),
                "dif IC95 inf": np.percentile(d, 2.5), "dif IC95 sup": np.percentile(d, 97.5),
                "referência melhor (% dos sorteios)": 100 * float((d > 0).mean()),
                "Wilcoxon p (TE por trimestre)": p,
            })
    out = pd.DataFrame(linhas)
    out.to_csv(PASTA_RESULTADOS / f"estatistica_modelos_K{K}.csv", index=False)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=20)
    print(analisar(ap.parse_args().K).round(3).to_string(index=False))
