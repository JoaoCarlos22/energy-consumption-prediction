
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path("tabela_resultados_weka_regressao.csv")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
ibk = df[df["Algoritmo"] == "IBk"].copy()
ibk["k"] = ibk["Configuração"].str.extract(r"k=(\d+)").astype(int)

ibk_cv = ibk[ibk["Validação"] == "10-fold cross-validation"].sort_values("k")
ibk_split = ibk[ibk["Validação"] == "Percentage split 70/30"].sort_values("k")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 19,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(13.33, 7.5))

ax.plot(ibk_cv["k"], ibk_cv["RMSE"], marker="o", linewidth=3, markersize=9,
        label="RMSE — 10-fold CV", color="#E76F51")
ax.plot(ibk_split["k"], ibk_split["RMSE"], marker="o", linewidth=3, markersize=9,
        label="RMSE — Split 70/30", color="#F4A261")
ax.plot(ibk_cv["k"], ibk_cv["MAE"], marker="s", linewidth=3, markersize=8,
        label="MAE — 10-fold CV", color="#2A9D8F")
ax.plot(ibk_split["k"], ibk_split["MAE"], marker="s", linewidth=3, markersize=8,
        label="MAE — Split 70/30", color="#457B9D")

ax.set_title("Impacto do valor de k no IBk", pad=16, fontweight="bold")
ax.set_xlabel("Número de vizinhos mais próximos (k)")
ax.set_ylabel("Erro")
ax.set_xticks([1, 3, 5])
ax.set_ylim(0, max(ibk["RMSE"].max(), ibk["MAE"].max()) * 1.22)
ax.grid(True, linestyle="--", alpha=0.25)
ax.legend(frameon=True, ncol=2, loc="upper right")

for data, metric, color in [
    (ibk_cv, "RMSE", "#E76F51"),
    (ibk_split, "RMSE", "#F4A261"),
    (ibk_cv, "MAE", "#2A9D8F"),
    (ibk_split, "MAE", "#457B9D"),
]:
    for _, row in data.iterrows():
        ax.text(row["k"], row[metric] + 0.015, f"{row[metric]:.4f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=color)

fig.text(0.01, 0.015, "Fonte: resultados de regressão exportados do Weka.",
         ha="left", va="bottom", fontsize=9, color="#555555")

fig.savefig(OUT_DIR / "impacto_k_ibk.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_DIR / "impacto_k_ibk.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Gráfico do IBk gerado com sucesso.")
