from pathlib import Path
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DATA_DIR.parent.parent

CSV_PATH = DATA_DIR / "tabela_resultados_weka_regressao.csv"
OUT_DIR = PROJECT_ROOT / "imagens" / "visualizacoes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_CV = "10-fold cross-validation"
VALIDATION_SPLIT = "Percentage split 70/30"
VALIDATION_LABELS = {
    VALIDATION_CV: "10-fold CV",
    VALIDATION_SPLIT: "Split 70/30",
}


def fix_mojibake(value):
    text = str(value)
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def remove_accents(value):
    text = unicodedata.normalize("NFKD", value)
    return "".join(char for char in text if not unicodedata.combining(char))


def normalize_column(value):
    text = remove_accents(fix_mojibake(value)).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_results(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns={column: normalize_column(column) for column in df.columns})

    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].map(lambda value: fix_mojibake(value).strip())

    text_columns = {"algoritmo", "configuracao", "validacao"}
    for column in df.columns:
        if column in text_columns:
            continue
        converted = pd.to_numeric(
            df[column].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if converted.notna().any():
            df[column] = converted

    required = text_columns
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV: {', '.join(missing)}")

    df["modelo"] = np.where(
        df["algoritmo"].eq("IBk"),
        df["algoritmo"] + " " + df["configuracao"],
        df["algoritmo"],
    )
    df["validacao_curta"] = df["validacao"].replace(VALIDATION_LABELS)
    return df


df = load_results(CSV_PATH)
cv = df[df["validacao"] == VALIDATION_CV].copy()
split = df[df["validacao"] == VALIDATION_SPLIT].copy()

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
        0.01,
        0.015,
        "Fonte: resultados exportados do Weka.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#555555",
    )


def add_value_labels_barh(ax, values, fmt="{:.4f}", offset_ratio=0.015):
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    for index, value in enumerate(values):
        offset = span * offset_ratio if value >= 0 else -span * offset_ratio
        ax.text(
            value + offset,
            index,
            fmt.format(value),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10.5,
            fontweight="bold",
        )


def label_grouped_bars(ax, bars, offset, fmt="{:.3f}"):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )


def save_fig(fig, filename):
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{filename}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def choose_accuracy_metric(data):
    candidates = [
        "acuracia",
        "accuracy",
        "percentual_acertos",
        "corretamente_classificadas",
    ]
    for candidate in candidates:
        if candidate in data.columns:
            return {
                "column": candidate,
                "label": "Acurácia (%)",
                "title_metric": "Acurácia",
                "hint": "Maior acurácia indica melhor classificação",
                "is_accuracy": True,
            }

    return {
        "column": "correlacao",
        "label": "Correlação",
        "title_metric": "Correlação",
        "hint": "Tabela atual é de regressão; correlação substitui a acurácia",
        "is_accuracy": False,
    }


def metric_values_for_plot(data, metric_info):
    values = data[metric_info["column"]].astype(float).copy()
    if metric_info["is_accuracy"] and values.max() <= 1:
        values = values * 100
    return values


def plot_general_accuracy(data, validation_label, filename, metric_info):
    chart = data.copy()
    chart["valor_plot"] = metric_values_for_plot(chart, metric_info)
    chart = chart.sort_values("valor_plot", ascending=True)

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.barh(
        chart["modelo"],
        chart["valor_plot"],
        color=[model_color(model) for model in chart["modelo"]],
        edgecolor="#222222",
        linewidth=0.5,
    )
    ax.set_title(
        f"Comparativo geral dos modelos por {metric_info['title_metric']} - {validation_label}",
        pad=16,
        fontweight="bold",
    )
    ax.set_xlabel(metric_info["label"])
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    if metric_info["is_accuracy"]:
        ax.set_xlim(0, max(100, chart["valor_plot"].max() * 1.12))
        add_value_labels_barh(ax, chart["valor_plot"], fmt="{:.2f}%")
    else:
        ax.set_xlim(min(-0.15, chart["valor_plot"].min() - 0.05), 1.03)
        ax.axvline(0, color="#333333", linewidth=0.8)
        add_value_labels_barh(ax, chart["valor_plot"], fmt="{:.4f}")

    ax.text(
        0.99,
        0.04,
        metric_info["hint"],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD"),
    )
    add_source_note(fig)
    save_fig(fig, filename)


# 1. Correlação - 10-fold CV
cv_corr = cv.sort_values("correlacao", ascending=True)
fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.barh(
    cv_corr["modelo"],
    cv_corr["correlacao"],
    color=[model_color(model) for model in cv_corr["modelo"]],
    edgecolor="#222222",
    linewidth=0.5,
)
ax.set_title("Comparação da Correlação dos Modelos - 10-fold CV", pad=16, fontweight="bold")
ax.set_xlabel("Coeficiente de correlação")
ax.set_xlim(min(-0.15, cv_corr["correlacao"].min() - 0.05), 1.03)
ax.axvline(0, color="#333333", linewidth=0.8)
ax.grid(axis="x", linestyle="--", alpha=0.25)
add_value_labels_barh(ax, cv_corr["correlacao"])
ax.text(
    0.99,
    0.04,
    "Quanto mais próximo de 1, melhor",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD"),
)
add_source_note(fig)
save_fig(fig, "01_comparacao_correlacao_10fold_cv")

# 2. RMSE - 10-fold CV
cv_rmse = cv.sort_values("rmse", ascending=False)
fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.barh(
    cv_rmse["modelo"],
    cv_rmse["rmse"],
    color=[model_color(model) for model in cv_rmse["modelo"]],
    edgecolor="#222222",
    linewidth=0.5,
)
ax.set_title("Comparação do RMSE dos Modelos - 10-fold CV", pad=16, fontweight="bold")
ax.set_xlabel("RMSE")
ax.set_xlim(0, cv_rmse["rmse"].max() * 1.18)
ax.grid(axis="x", linestyle="--", alpha=0.25)
add_value_labels_barh(ax, cv_rmse["rmse"])
ax.text(
    0.99,
    0.04,
    "Quanto menor, melhor",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD"),
)
add_source_note(fig)
save_fig(fig, "02_comparacao_rmse_10fold_cv")

# 3. MAE - 10-fold CV
cv_mae = cv.sort_values("mae", ascending=False)
fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.barh(
    cv_mae["modelo"],
    cv_mae["mae"],
    color=[model_color(model) for model in cv_mae["modelo"]],
    edgecolor="#222222",
    linewidth=0.5,
)
ax.set_title("Comparação do MAE dos Modelos - 10-fold CV", pad=16, fontweight="bold")
ax.set_xlabel("MAE")
ax.set_xlim(0, cv_mae["mae"].max() * 1.18)
ax.grid(axis="x", linestyle="--", alpha=0.25)
add_value_labels_barh(ax, cv_mae["mae"])
ax.text(
    0.99,
    0.04,
    "Quanto menor, melhor",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD"),
)
add_source_note(fig)
save_fig(fig, "03_comparacao_mae_10fold_cv")

# 4. RMSE - 10-fold CV x Split 70/30
pivot_rmse = df.pivot(index="modelo", columns="validacao_curta", values="rmse")
pivot_rmse = pivot_rmse.loc[cv.sort_values("rmse")["modelo"]]

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
label_grouped_bars(ax, bars1, max(pivot_rmse.max()) * 0.018)
label_grouped_bars(ax, bars2, max(pivot_rmse.max()) * 0.018)
ax.text(
    0.99,
    0.93,
    "Menor RMSE indica menor erro quadrático",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=11,
    color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD"),
)
add_source_note(fig)
save_fig(fig, "04_rmse_10fold_vs_split")

# 5. Correlação - 10-fold CV x Split 70/30
pivot_corr = df.pivot(index="modelo", columns="validacao_curta", values="correlacao")
pivot_corr = pivot_corr.loc[cv.sort_values("correlacao", ascending=False)["modelo"]]

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
    for bar in bars:
        height = bar.get_height()
        vertical_align = "bottom" if height >= 0 else "top"
        y_offset = 0.025 if height >= 0 else -0.035
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            f"{height:.3f}",
            ha="center",
            va=vertical_align,
            fontsize=9.5,
            fontweight="bold",
        )
ax.text(
    0.99,
    0.05,
    "Maior correlação indica maior associação entre previsto e real",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD"),
)
add_source_note(fig)
save_fig(fig, "05_correlacao_10fold_vs_split")

# 6. Impacto do k no IBk
ibk = df[df["algoritmo"] == "IBk"].copy()
ibk["k"] = ibk["configuracao"].str.extract(r"k=(\d+)").astype(int)
ibk_cv = ibk[ibk["validacao"] == VALIDATION_CV].sort_values("k")
ibk_split = ibk[ibk["validacao"] == VALIDATION_SPLIT].sort_values("k")

fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.plot(ibk_cv["k"], ibk_cv["rmse"], marker="o", linewidth=3, markersize=9, label="RMSE - 10-fold CV", color="#E76F51")
ax.plot(ibk_split["k"], ibk_split["rmse"], marker="o", linewidth=3, markersize=9, label="RMSE - Split 70/30", color="#F4A261")
ax.plot(ibk_cv["k"], ibk_cv["mae"], marker="s", linewidth=3, markersize=8, label="MAE - 10-fold CV", color="#2A9D8F")
ax.plot(ibk_split["k"], ibk_split["mae"], marker="s", linewidth=3, markersize=8, label="MAE - Split 70/30", color="#457B9D")
ax.set_title("Impacto do valor de k no IBk", pad=16, fontweight="bold")
ax.set_xlabel("Número de vizinhos mais próximos (k)")
ax.set_ylabel("Erro")
ax.set_xticks([1, 3, 5])
ax.set_ylim(0, max(ibk["rmse"].max(), ibk["mae"].max()) * 1.22)
ax.grid(True, linestyle="--", alpha=0.25)
ax.legend(frameon=True, ncol=2, loc="upper right")
for data, metric, color in [
    (ibk_cv, "rmse", "#E76F51"),
    (ibk_split, "rmse", "#F4A261"),
    (ibk_cv, "mae", "#2A9D8F"),
    (ibk_split, "mae", "#457B9D"),
]:
    for _, row in data.iterrows():
        ax.text(
            row["k"],
            row[metric] + 0.015,
            f"{row[metric]:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=color,
        )
ax.text(
    0.03,
    0.05,
    "No 10-fold CV, k=5 apresentou o menor RMSE entre os valores testados.",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=11,
    color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD"),
)
add_source_note(fig)
save_fig(fig, "06_impacto_k_ibk")

# 7. Mapa de desempenho - 10-fold CV
metrics = ["correlacao", "mae", "rmse", "rae", "rrse"]
metric_labels = ["Correlação", "MAE", "RMSE", "RAE (%)", "RRSE (%)"]
heat = cv.set_index("modelo")[metrics].copy()
score = pd.DataFrame(index=heat.index)
score["correlacao"] = (heat["correlacao"] - heat["correlacao"].min()) / (heat["correlacao"].max() - heat["correlacao"].min())
for metric in ["mae", "rmse", "rae", "rrse"]:
    score[metric] = 1 - ((heat[metric] - heat[metric].min()) / (heat[metric].max() - heat[metric].min()))
score = score.loc[cv.sort_values("rmse")["modelo"]]

fig, ax = plt.subplots(figsize=(13.33, 7.5))
image = ax.imshow(score.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_title("Mapa de Desempenho dos Modelos - 10-fold CV", pad=16, fontweight="bold")
ax.set_xticks(np.arange(len(metrics)))
ax.set_xticklabels(metric_labels)
ax.set_yticks(np.arange(len(score.index)))
ax.set_yticklabels(score.index)
for row_index in range(score.shape[0]):
    for column_index in range(score.shape[1]):
        original_value = heat.loc[score.index[row_index], metrics[column_index]]
        ax.text(
            column_index,
            row_index,
            f"{original_value:.3f}",
            ha="center",
            va="center",
            color="white" if score.iloc[row_index, column_index] < 0.55 else "#111111",
            fontsize=9.5,
            fontweight="bold",
        )
colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
colorbar.set_label("Pontuação visual normalizada\n(mais claro = melhor)", fontsize=10)
ax.text(
    0.99,
    -0.12,
    "Valores exibidos são as métricas originais; a cor representa o desempenho normalizado.",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    color="#444444",
)
add_source_note(fig)
save_fig(fig, "07_mapa_desempenho_10fold_cv")

# 8. Ranking visual - 10-fold CV
ranking = cv.copy()
corr_norm = (ranking["correlacao"] - ranking["correlacao"].min()) / (ranking["correlacao"].max() - ranking["correlacao"].min())
rmse_inv = 1 - ((ranking["rmse"] - ranking["rmse"].min()) / (ranking["rmse"].max() - ranking["rmse"].min()))
ranking["score_visual"] = 0.55 * corr_norm + 0.45 * rmse_inv
ranking = ranking.sort_values("score_visual", ascending=True)

fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.barh(
    ranking["modelo"],
    ranking["score_visual"],
    color=[model_color(model) for model in ranking["modelo"]],
    edgecolor="#222222",
    linewidth=0.5,
)
ax.set_title("Ranking Visual dos Modelos - 10-fold CV", pad=16, fontweight="bold")
ax.set_xlabel("Score visual combinado")
ax.set_xlim(0, 1.10)
ax.grid(axis="x", linestyle="--", alpha=0.25)
for index, (_, row) in enumerate(ranking.iterrows()):
    ax.text(
        row["score_visual"] + 0.015,
        index,
        f"{row['score_visual']:.3f}",
        va="center",
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
ax.text(
    0.02,
    0.96,
    "Score = 55% correlação + 45% RMSE invertido",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=11,
    color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD"),
)
add_source_note(fig)
save_fig(fig, "08_ranking_visual_modelos_10fold_cv")

# 9 e 10. Comparativos gerais por acurácia quando a coluna existir.
accuracy_metric = choose_accuracy_metric(df)
plot_general_accuracy(
    cv,
    "10-fold CV",
    "09_comparativo_geral_acuracia_10fold_cv",
    accuracy_metric,
)
plot_general_accuracy(
    split,
    "Split 70/30",
    "10_comparativo_geral_acuracia_split_70_30",
    accuracy_metric,
)

if not accuracy_metric["is_accuracy"]:
    print("Aviso: coluna de acurácia não encontrada; gráficos 09 e 10 foram gerados com correlação.")

print(f"Gráficos gerados com sucesso em: {OUT_DIR.resolve()}")
