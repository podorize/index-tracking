"""Tabela do teto de chance estimada: qual modelo a regra escolhe para cada teto."""
import pandas as pd

from src.criterio_cvm import regra_estimada

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    d = pd.read_csv("results/selecao_prob_estimada.csv")
    print(d.groupby("K").size().to_dict())
    print(d[d.K == 999].groupby("ind").K_efetivo.mean())
    b = d.sort_values("p").groupby(["ind", "K"]).head(1).sort_values(["ind", "K"])
    print(b[["ind", "K", "J", "H", "lam", "TE", "vies", "giro", "p12"]].round(2).to_string(index=False))
    for t in list(range(0, 55, 5)) + [22, 23, 24, 26, 27, 28]:
        r = regra_estimada(d, t / 100)
        if r is None:
            print(t, "nenhum"); continue
        K, cfg, c = r
        print(t, "K", K, "cfg", cfg, "lam", c["lam"].to_dict(), "giro", round(c["giro"].mean(), 1), "TE", c["TE"].round(2).to_dict(), "p12", c["p12"].round(2).to_dict())
