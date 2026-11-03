"""
criterio_cvm.py
Diferença de rentabilidade acumulada entre carteira e índice em janelas móveis de rodadas consecutivas,
medida como no Art. 27 do Anexo Normativo V da Resolução CVM 175: 12 meses (limite 4 p.p.) e 60 pregões (limite 2 p.p.).
Usa o retorno líquido de cada rodada (252 pregões = 4 rodadas trimestrais ou 12 mensais; 63 pregões = 1 trimestral ou 3 mensais, ~60).
Período de escolha: results/selecao/. Sem reescolher nada; só mede.
"""
import re
from pathlib import Path

import pandas as pd

SAIDA = Path("results/selecao")
LIM_12M, LIM_60D = 4.0, 2.0


def janelas(r, n):
    """Diferença acumulada (p.p.) em cada janela de n rodadas consecutivas."""
    p, i = 1 + r["retorno_liq_carteira"].to_numpy(), 1 + r["retorno_indice"].to_numpy()
    return [100 * (p[k:k + n].prod() - i[k:k + n].prod()) for k in range(len(r) - n + 1)]


def medir(r, H):
    g12, g60 = janelas(r, 252 // H), janelas(r, 63 // H)
    return {"max_12m": max(abs(x) for x in g12), "viol_12m": sum(abs(x) > LIM_12M for x in g12), "jan_12m": len(g12),
            "max_60d": max(abs(x) for x in g60), "viol_60d": sum(abs(x) > LIM_60D for x in g60), "jan_60d": len(g60)}


def carregar():
    linhas = []
    for f in SAIDA.glob("K*.csv"):
        K, i, J, H, L = re.match(r"K(\d+)_(sp100|ibov)_J(\d+)_H(\d+)_L([\d.]+)\.csv", f.name).groups()
        H = int(H)
        r = pd.read_csv(f)
        r = r[r["dias"] == H].sort_values("numero")
        giro = r.loc[r["numero"] > r["numero"].min(), "giro"].mean()
        linhas.append({"K": int(K), "indice": "S&P100" if i == "sp100" else "IBOV", "janela": int(J), "horizonte": H,
                       "lam": float(L), "giro_anual": 100 * giro * 252 / H, **medir(r, H)})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    d = carregar()
    d["passa"] = (d["viol_12m"] == 0) & (d["viol_60d"] == 0)
    d.to_csv("results/selecao_diferenca_acumulada.csv", index=False)
    pd.set_option("display.width", 250)
    print(d.groupby(["janela", "horizonte", "indice"])[["jan_12m", "jan_60d"]].agg(["min", "max"]).to_string())
    print("\nFração de células que passam, por K, configuração e índice:")
    print(d.pivot_table(index=["K", "janela", "horizonte"], columns="indice", values="passa", aggfunc="mean").round(2).to_string())


def regra(d, teto):
    """Menor K com alguma configuração em que, nos dois índices, existe λ com fração de janelas acima do limite <= teto
    (12 meses e 60 pregões); depois a configuração de menor giro médio (λ por índice = o de menor giro que passa)."""
    d = d.assign(f12=d["viol_12m"] / d["jan_12m"], f60=d["viol_60d"] / d["jan_60d"])
    ok = d[(d["f12"] <= teto + 1e-9) & (d["f60"] <= teto + 1e-9)]
    melhor = ok.sort_values(["giro_anual", "lam"]).groupby(["K", "janela", "horizonte", "indice"]).head(1)
    if set(melhor["indice"]) != {"S&P100", "IBOV"}:
        return None                                      # um dos índices não tem nenhuma célula que passe
    cfg = melhor.pivot_table(index=["K", "janela", "horizonte"], columns="indice", values=["giro_anual", "lam", "f12", "f60"])
    cfg = cfg.dropna()
    if cfg.empty:
        return None
    giro_medio = cfg["giro_anual"].mean(axis=1)
    K = cfg.index.get_level_values("K").min()
    c = cfg.loc[K]
    escolhida = giro_medio.loc[K].idxmin()
    return K, escolhida, c.loc[escolhida]


def regra_estimada(d, teto):
    """Regra pela chance estimada de a diferença acumulada passar de 4 p.p. em 12 meses (TE e viés): menor K cuja melhor
    configuração tem, nos dois índices, ao menos um λ com chance <= teto; depois, a configuração de menor giro médio."""
    d = d.rename(columns={"ind": "indice"})
    ok = d[d["p12"] <= teto + 1e-9]
    if set(ok["indice"]) != {"S&P100", "IBOV"}:
        return None
    melhor = ok.sort_values(["giro", "lam"]).groupby(["K", "J", "H", "indice"]).head(1)
    cfg = melhor.pivot_table(index=["K", "J", "H"], columns="indice", values=["giro", "lam", "TE", "p12"]).dropna()
    if cfg.empty:
        return None
    K = cfg.index.get_level_values("K").min()
    c = cfg.loc[K]
    escolhida = c["giro"].mean(axis=1).idxmin()
    return K, escolhida, c.loc[escolhida]
