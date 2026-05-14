
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = Path("tabela_resultados_weka_regressao.csv")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

df["Modelo"] = df["Algoritmo"].where(
    df["Algoritmo"] != "IBk",
    df["Algoritmo"] + " " + df["Configuração"]
)

df["Validação_curta"] = df["Validação"].replace({
    "10-fold cross-validation": "10-fold CV",
    "Percentage split 70/30": "Split 70/30"
})

cv = df[df["Validação"] == "10-fold cross-validation"].copy()
split = df[df["Validação"] == "Percentage split 70/30"].copy()

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 19,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 20,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

algorithm_colors = {
    "ZeroR": "#8D99AE",
    "RandomForest": "#2A9D8F",
    "RandomTree": "#F4A261",
    "IBk": "#457B9D",
    "SMOreg": "#E76F51",
    "MultilayerPerceptron": "#9B5DE5",
}

def model_color(model_name):
    if model_name.startswith("IBk"):
        return algorithm_colors["IBk"]
    if model_name.startswith("MultilayerPerceptron"):
        return algorithm_colors["MultilayerPerceptron"]
    for key, color in algorithm_colors.items():
        if model_name.startswith(key):
            return color
    return "#6C757D"

def add_source_note(fig):
    fig.text(
        0.01, 0.015,
        "Fonte: resultados de regressão exportados do Weka.",
        ha="left", va="bottom", fontsize=9, color="#555555"
    )

def add_value_labels_barh(ax, values, fmt="{:.4f}", offset_ratio=0.015):
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    for i, v in enumerate(values):
        ax.text(
            v + span * offset_ratio,
            i,
            fmt.format(v),
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold"
        )

def save_fig(fig, filename):
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{filename}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# 1. Correlação — 10-fold CV
cv_corr = cv.sort_values("Correlação", ascending=True)
fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.barh(
    cv_corr["Modelo"], cv_corr["Correlação"],
    color=[model_color(m) for m in cv_corr["Modelo"]],
    edgecolor="#222222", linewidth=0.5
)
ax.set_title("Comparação da Correlação dos Modelos — 10-fold CV", pad=16, fontweight="bold")
ax.set_xlabel("Coeficiente de correlação")
ax.set_xlim(min(-0.15, cv_corr["Correlação"].min() - 0.05), 1.03)
ax.axvline(0, color="#333333", linewidth=0.8)
ax.grid(axis="x", linestyle="--", alpha=0.25)
add_value_labels_barh(ax, cv_corr["Correlação"])
ax.text(
    0.99, 0.04, "Quanto mais próximo de 1, melhor",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD")
)
add_source_note(fig)
save_fig(fig, "01_comparacao_correlacao_10fold_cv")

# 2. RMSE — 10-fold CV
cv_rmse = cv.sort_values("RMSE", ascending=False)
fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.barh(
    cv_rmse["Modelo"], cv_rmse["RMSE"],
    color=[model_color(m) for m in cv_rmse["Modelo"]],
    edgecolor="#222222", linewidth=0.5
)
ax.set_title("Comparação do RMSE dos Modelos — 10-fold CV", pad=16, fontweight="bold")
ax.set_xlabel("RMSE")
ax.set_xlim(0, cv_rmse["RMSE"].max() * 1.18)
ax.grid(axis="x", linestyle="--", alpha=0.25)
add_value_labels_barh(ax, cv_rmse["RMSE"])
ax.text(
    0.99, 0.04, "Quanto menor, melhor",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD")
)
add_source_note(fig)
save_fig(fig, "02_comparacao_rmse_10fold_cv")

# 3. MAE — 10-fold CV
cv_mae = cv.sort_values("MAE", ascending=False)
fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.barh(
    cv_mae["Modelo"], cv_mae["MAE"],
    color=[model_color(m) for m in cv_mae["Modelo"]],
    edgecolor="#222222", linewidth=0.5
)
ax.set_title("Comparação do MAE dos Modelos — 10-fold CV", pad=16, fontweight="bold")
ax.set_xlabel("MAE")
ax.set_xlim(0, cv_mae["MAE"].max() * 1.18)
ax.grid(axis="x", linestyle="--", alpha=0.25)
add_value_labels_barh(ax, cv_mae["MAE"])
ax.text(
    0.99, 0.04, "Quanto menor, melhor",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD")
)
add_source_note(fig)
save_fig(fig, "03_comparacao_mae_10fold_cv")

# 4. RMSE — 10-fold CV x Split 70/30
pivot_rmse = df.pivot(index="Modelo", columns="Validação_curta", values="RMSE")
pivot_rmse = pivot_rmse.loc[cv.sort_values("RMSE")["Modelo"]]

fig, ax = plt.subplots(figsize=(14.2, 7.8))
x = np.arange(len(pivot_rmse.index))
width = 0.38
bars1 = ax.bar(x - width / 2, pivot_rmse["10-fold CV"], width, label="10-fold CV", color="#264653")
bars2 = ax.bar(x + width / 2, pivot_rmse["Split 70/30"], width, label="Split 70/30", color="#E9C46A")
ax.set_title("RMSE por Modelo: 10-fold CV x Split 70/30", pad=16, fontweight="bold")
ax.set_ylabel("RMSE")
ax.set_xticks(x)
ax.set_xticklabels(pivot_rmse.index, rotation=28, ha="right")
ax.set_ylim(0, max(pivot_rmse.max()) * 1.22)
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.legend(frameon=True, loc="upper left")
for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + max(pivot_rmse.max()) * 0.018,
            f"{h:.3f}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold"
        )
ax.text(
    0.99, 0.93, "Menor RMSE indica menor erro quadrático",
    transform=ax.transAxes, ha="right", va="top", fontsize=11, color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD")
)
add_source_note(fig)
save_fig(fig, "04_rmse_10fold_vs_split")

# 5. Correlação — 10-fold CV x Split 70/30
pivot_corr = df.pivot(index="Modelo", columns="Validação_curta", values="Correlação")
pivot_corr = pivot_corr.loc[cv.sort_values("Correlação", ascending=False)["Modelo"]]

fig, ax = plt.subplots(figsize=(14.2, 7.8))
x = np.arange(len(pivot_corr.index))
width = 0.38
bars1 = ax.bar(x - width / 2, pivot_corr["10-fold CV"], width, label="10-fold CV", color="#1D3557")
bars2 = ax.bar(x + width / 2, pivot_corr["Split 70/30"], width, label="Split 70/30", color="#A8DADC")
ax.set_title("Correlação por Modelo: 10-fold CV x Split 70/30", pad=16, fontweight="bold")
ax.set_ylabel("Coeficiente de correlação")
ax.set_xticks(x)
ax.set_xticklabels(pivot_corr.index, rotation=28, ha="right")
ax.set_ylim(min(-0.15, pivot_corr.min().min() - 0.05), 1.08)
ax.axhline(0, color="#333333", linewidth=0.8)
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.legend(frameon=True, loc="lower right")
for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        yoff = 0.025 if h >= 0 else -0.035
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + yoff,
            f"{h:.3f}",
            ha="center", va=va, fontsize=9.5, fontweight="bold"
        )
ax.text(
    0.99, 0.05, "Maior correlação indica maior associação entre previsto e real",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD")
)
add_source_note(fig)
save_fig(fig, "05_correlacao_10fold_vs_split")

# 6. Impacto do k no IBk
ibk = df[df["Algoritmo"] == "IBk"].copy()
ibk["k"] = ibk["Configuração"].str.extract(r"k=(\d+)").astype(int)
ibk_cv = ibk[ibk["Validação"] == "10-fold cross-validation"].sort_values("k")
ibk_split = ibk[ibk["Validação"] == "Percentage split 70/30"].sort_values("k")

fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.plot(ibk_cv["k"], ibk_cv["RMSE"], marker="o", linewidth=3, markersize=9, label="RMSE — 10-fold CV", color="#E76F51")
ax.plot(ibk_split["k"], ibk_split["RMSE"], marker="o", linewidth=3, markersize=9, label="RMSE — Split 70/30", color="#F4A261")
ax.plot(ibk_cv["k"], ibk_cv["MAE"], marker="s", linewidth=3, markersize=8, label="MAE — 10-fold CV", color="#2A9D8F")
ax.plot(ibk_split["k"], ibk_split["MAE"], marker="s", linewidth=3, markersize=8, label="MAE — Split 70/30", color="#457B9D")
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
        ax.text(
            row["k"], row[metric] + 0.015, f"{row[metric]:.4f}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=color
        )
ax.text(
    0.03, 0.05, "No 10-fold CV, k=5 apresentou o menor RMSE entre os valores testados.",
    transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD")
)
add_source_note(fig)
save_fig(fig, "06_impacto_k_ibk")

# 7. Mapa de desempenho — 10-fold CV
metrics = ["Correlação", "MAE", "RMSE", "RAE (%)", "RRSE (%)"]
heat = cv.set_index("Modelo")[metrics].copy()
score = pd.DataFrame(index=heat.index)
score["Correlação"] = (heat["Correlação"] - heat["Correlação"].min()) / (heat["Correlação"].max() - heat["Correlação"].min())
for metric in ["MAE", "RMSE", "RAE (%)", "RRSE (%)"]:
    score[metric] = 1 - ((heat[metric] - heat[metric].min()) / (heat[metric].max() - heat[metric].min()))
score = score.loc[cv.sort_values("RMSE")["Modelo"]]

fig, ax = plt.subplots(figsize=(13.33, 7.5))
im = ax.imshow(score.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_title("Mapa de Desempenho dos Modelos — 10-fold CV", pad=16, fontweight="bold")
ax.set_xticks(np.arange(len(metrics)))
ax.set_xticklabels(metrics)
ax.set_yticks(np.arange(len(score.index)))
ax.set_yticklabels(score.index)
for i in range(score.shape[0]):
    for j in range(score.shape[1]):
        original_value = heat.loc[score.index[i], metrics[j]]
        ax.text(
            j, i, f"{original_value:.3f}",
            ha="center", va="center",
            color="white" if score.iloc[i, j] < 0.55 else "#111111",
            fontsize=9.5, fontweight="bold"
        )
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
cbar.set_label("Pontuação visual normalizada\n(mais claro = melhor)", fontsize=10)
ax.text(
    0.99, -0.12,
    "Valores exibidos são as métricas originais; a cor representa o desempenho normalizado.",
    transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#444444"
)
add_source_note(fig)
save_fig(fig, "07_mapa_desempenho_10fold_cv")

# 8. Ranking visual — 10-fold CV
ranking = cv.copy()
corr_norm = (ranking["Correlação"] - ranking["Correlação"].min()) / (ranking["Correlação"].max() - ranking["Correlação"].min())
rmse_inv = 1 - ((ranking["RMSE"] - ranking["RMSE"].min()) / (ranking["RMSE"].max() - ranking["RMSE"].min()))
ranking["Score_visual"] = 0.55 * corr_norm + 0.45 * rmse_inv
ranking = ranking.sort_values("Score_visual", ascending=True)

fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.barh(
    ranking["Modelo"], ranking["Score_visual"],
    color=[model_color(m) for m in ranking["Modelo"]],
    edgecolor="#222222", linewidth=0.5
)
ax.set_title("Ranking Visual dos Modelos — 10-fold CV", pad=16, fontweight="bold")
ax.set_xlabel("Score visual combinado")
ax.set_xlim(0, 1.10)
ax.grid(axis="x", linestyle="--", alpha=0.25)
for i, (_, row) in enumerate(ranking.iterrows()):
    ax.text(
        row["Score_visual"] + 0.015, i, f"{row['Score_visual']:.3f}",
        va="center", ha="left", fontsize=10.5, fontweight="bold"
    )
ax.text(
    0.02, 0.96, "Score = 55% correlação + 45% RMSE invertido",
    transform=ax.transAxes, ha="left", va="top", fontsize=11, color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD")
)
add_source_note(fig)
save_fig(fig, "08_ranking_visual_modelos_10fold_cv")

print(f"Gráficos gerados com sucesso em: {OUT_DIR.resolve()}")
