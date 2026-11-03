"""Resultado do teste final: TE, giro e fração de janelas acima dos limites da CVM (janelas que terminam após 01/10/2024)."""
import numpy as np
import pandas as pd

from src.criterio_cvm import LIM_12M, LIM_60D, janelas

CORTE = pd.Timestamp("2024-10-01")
MODELOS = ["k15_l12", "k15", "k30", "k30_l3", "k50", "sem_limite", "cvm20", "escolhido", "A_1ano_trimestral", "B_original"]
ROTULO = {"k15_l12": "Arrojado com λ da regra (K=15, λ 12/3)", "k15": "Arrojado: K=15 (1a/mensal, λ 3/3)", "k30_l3": "Sensibilidade: K=30 (1a/mensal, λ 3/3)", "k50": "K=50 (1a/mensal, λ 50/3)", "k30": "Regra final (K=30, 1a/mensal, λ 50/12)", "sem_limite": "Referência: sem limite (1a/mensal, λ 1,5)", "cvm20": "Regra CVM (K=20, 1a/mensal, λ 3/3)", "escolhido": "Regra anterior (2a/mensal, λ 50/25)",
          "A_1ano_trimestral": "A (1a/trimestral, λ 50/50)", "B_original": "B (2a/trimestral, λ 6/6)"}


def linha(m, i):
    r = pd.read_csv(f"results/teste_final/{m}_{'sp100' if i == 'S&P100' else 'ibov'}.csv", parse_dates=["data_fim"])
    H = int(r["horizonte"].iloc[0])
    r = r[r["dias"] == H].sort_values("numero").reset_index(drop=True)
    teste = r["data_fim"] > CORTE
    te = r.loc[teste, "TE_trimestre (% a.a.)"]
    giro = r.loc[teste & (r["numero"] > 1), "giro"].mean() * 252 / H * 100
    out = {"modelo": ROTULO[m], "indice": i, "rodadas_teste": int(teste.sum()),
           "TE": float(np.sqrt((te ** 2).mean())), "ep_TE": float(te.std(ddof=1) / np.sqrt(len(te)) ) if len(te) > 1 else np.nan,
           "giro_anual": float(giro)}
    for nome, n, lim in (("12m", 252 // H, LIM_12M), ("60d", 63 // H, LIM_60D)):
        g = np.array(janelas(r, n))
        fim = r["data_fim"].iloc[n - 1:].reset_index(drop=True)          # data de término de cada janela
        t = fim > CORTE
        out[f"jan_{nome}"] = int(t.sum()); out[f"viol_{nome}"] = int((np.abs(g[t.to_numpy()]) > lim).sum())
        out[f"max_{nome}"] = float(np.abs(g[t.to_numpy()]).max())
    return out


if __name__ == "__main__":
    d = pd.DataFrame([linha(m, i) for m in MODELOS for i in ("S&P100", "IBOV")])
    d.to_csv("results/teste_final_cvm.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(d.round(2).to_string(index=False))
