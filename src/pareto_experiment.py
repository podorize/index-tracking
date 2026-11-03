"""
pareto_experiment.py
Executa a varredura da fronteira de Pareto (Tracking Error vs. Turnover)
para o S&P 100 e IBOVESPA, avaliando custos de transação explícitos:
  - B3: 15 bps (0.0015)
  - EUA: 5 bps (0.0005)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import preparar_bases
from src.walk_forward import simular_walk_forward

PASTA_RESULTADOS = Path("results")
PASTA_FIGURAS = Path("figures")
PASTA_RESULTADOS.mkdir(exist_ok=True)
PASTA_FIGURAS.mkdir(exist_ok=True)

CUSTO_B3 = 0.0015    # 15 bps
CUSTO_EUA = 0.0005   # 5 bps

GRADE_LAMBDAS = [0.0, 0.5, 1.5, 3.0, 6.0, 12.0, 25.0, 50.0]
K_ESCOLHIDO = 10


def executar_grade(bases=None):
    bases = bases if bases is not None else preparar_bases()
    resultados_todos = []

    print("==========================================================")
    print("1. Executando grade de Pareto para o S&P 100 (K = 10)...")
    print("==========================================================")
    for lam in GRADE_LAMBDAS:
        print(f"  -> S&P 100 com lambda_anual = {lam:.1f}...")
        m, _ = simular_walk_forward(
            nome_indice="S&P100",
            bases=bases,
            K=K_ESCOLHIDO,
            lambda_anual=lam,
            custo_transacao=CUSTO_EUA,
            tempo_max=1.5
        )
        resultados_todos.append(m)

    print("\n==========================================================")
    print("2. Executando grade de Pareto para o IBOVESPA (K = 10)...")
    print("==========================================================")
    for lam in GRADE_LAMBDAS:
        print(f"  -> IBOV com lambda_anual = {lam:.1f}...")
        m, _ = simular_walk_forward(
            nome_indice="IBOV",
            bases=bases,
            K=K_ESCOLHIDO,
            lambda_anual=lam,
            custo_transacao=CUSTO_B3,
            tempo_max=1.5
        )
        resultados_todos.append(m)

    df_res = pd.DataFrame(resultados_todos)
    df_sp = df_res[df_res["indice"] == "S&P100"].copy()
    df_ib = df_res[df_res["indice"] == "IBOV"].copy()

    df_sp.to_csv(PASTA_RESULTADOS / "tabela_pareto_sp100.csv", index=False)
    df_ib.to_csv(PASTA_RESULTADOS / "tabela_pareto_ibov.csv", index=False)

    print("\nResultados S&P 100:")
    cols_exibir = ["lambda_anual", "TE (% a.a.)", "Giro médio (% trim)", "Custo fricção (% a.a.)", "Retorno anual líquido (%)", "Alfa líquido (p.p. a.a.)"]
    print(df_sp[cols_exibir].to_string(index=False))

    print("\nResultados IBOVESPA:")
    print(df_ib[cols_exibir].to_string(index=False))

    gerar_graficos_pareto(df_sp, df_ib)
    return df_sp, df_ib


def gerar_graficos_pareto(df_sp, df_ib):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=150)

    # Gráfico 1: Fronteira de Pareto (TE vs Giro)
    ax1 = axes[0]
    ax1.plot(df_sp["Giro médio (% trim)"], df_sp["TE (% a.a.)"], marker="o", color="#1f77b4", lw=2.2, label="S&P 100 (c = 5 bps)")
    ax1.plot(df_ib["Giro médio (% trim)"], df_ib["TE (% a.a.)"], marker="s", color="#2ca02c", lw=2.2, label="IBOVESPA (c = 15 bps)")

    for _, row in df_sp.iterrows():
        ax1.annotate(f"λ={row['lambda_anual']:.1f}", (row["Giro médio (% trim)"], row["TE (% a.a.)"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8, color="#1f77b4")
    for _, row in df_ib.iterrows():
        ax1.annotate(f"λ={row['lambda_anual']:.1f}", (row["Giro médio (% trim)"], row["TE (% a.a.)"]),
                     textcoords="offset points", xytext=(5, -12), fontsize=8, color="#2ca02c")

    ax1.set_title("Fronteira de Pareto: Tracking Error vs. Giro Médio (K = 10)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Giro Médio Trimestral (% da carteira)", fontsize=11)
    ax1.set_ylabel("Tracking Error Fora da Amostra (% a.a.)", fontsize=11)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Gráfico 2: Retorno Líquido Anualizado vs. Lambda
    ax2 = axes[1]
    ax2.plot(df_sp["lambda_anual"], df_sp["Retorno anual líquido (%)"], marker="o", color="#1f77b4", lw=2.2, label="S&P 100 Líquido")
    ax2.plot(df_ib["lambda_anual"], df_ib["Retorno anual líquido (%)"], marker="s", color="#2ca02c", lw=2.2, label="IBOVESPA Líquido")

    # Benchmarks de referência (linha tracejada)
    ret_idx_sp = df_sp["Retorno acumulado índice (%)"].iloc[0]
    ret_idx_ib = df_ib["Retorno acumulado índice (%)"].iloc[0]
    # anualizar aprox por 5 anos
    ret_idx_sp_a = (1 + ret_idx_sp / 100) ** (1 / 5.0) - 1.0
    ret_idx_ib_a = (1 + ret_idx_ib / 100) ** (1 / 5.0) - 1.0

    ax2.axhline(ret_idx_sp_a * 100, color="#1f77b4", linestyle=":", alpha=0.7, label="S&P 100 Índice")
    ax2.axhline(ret_idx_ib_a * 100, color="#2ca02c", linestyle=":", alpha=0.7, label="IBOVESPA Índice")

    ax2.set_title("Retorno Líquido Anualizado vs. Penalidade de Giro (λ)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Penalidade Anual de Giro (λ)", fontsize=11)
    ax2.set_ylabel("Retorno Anualizado Líquido de Custos (%)", fontsize=11)
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    caminho_fig = PASTA_FIGURAS / "curva_pareto_turnover.png"
    plt.savefig(caminho_fig)
    plt.close()
    print(f"Gráfico da Fronteira de Pareto salvo em {caminho_fig}")


if __name__ == "__main__":
    executar_grade()
