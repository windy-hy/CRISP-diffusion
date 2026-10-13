from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

BASE = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
# 优先读取 group-level；若不存在，则回退到 split-seed level
CANDIDATE_INPUTS = [
    BASE / "ablation_metrics_group.csv",
    BASE / "ablation_metrics_split_seed.csv",
]

# 原始模型名（CSV 里）
DET_MODEL = "Deterministic"
DIFF_MODEL = "Diffusion"
MEANSHIFT_MODEL = "MeanShift"

# Gaussian 的候选名字；脚本会自动检测第一个存在的
GAUSSIAN_MODEL_CANDIDATES = [
    "Gaussian_sigma0.1",
    "Gaussian_sigma",
    "Gaussian",
]

# 画图显示名
DISPLAY_LABELS = {
    DET_MODEL: "Det.",
    DIFF_MODEL: "Diffusion",
    MEANSHIFT_MODEL: "MeanShift",
}

GAUSSIAN_DISPLAY = r"Gaussian ($\sigma$=0.1)"

COLORS = {
    DET_MODEL: "#BFBFBF",
    "GAUSSIAN": "#E69F00",
    DIFF_MODEL: "#0072B2",
}

METRICS = [
    ("r2score_de", r"$R^2$"),
    ("pearson_delta_de", r"$Pr_{\Delta}$"),
    ("sinkhorn_de", "Sinkhorn"),
    ("centered_sinkhorn_de", "Centered Sinkhorn"),
    ("variance_log_mae_de", "Variance log-MAE"),
    ("responder_error", "Responder error"),
]

OUT_PREFIX = "Fig8a_core_generator_ablation_pretty"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_input_df() -> pd.DataFrame:
    for p in CANDIDATE_INPUTS:
        if p.exists():
            print(f"[INFO] Using input file: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError(
        "None of the input CSV files were found:\n"
        + "\n".join(str(p) for p in CANDIDATE_INPUTS)
    )


def detect_gaussian_model(df: pd.DataFrame) -> str:
    models = set(df["model"].astype(str).unique())
    for cand in GAUSSIAN_MODEL_CANDIDATES:
        if cand in models:
            print(f"[INFO] Using Gaussian model: {cand}")
            return cand
    raise ValueError(
        f"Cannot detect Gaussian model from candidates {GAUSSIAN_MODEL_CANDIDATES}. "
        f"Available models: {sorted(models)}"
    )


def get_index_cols(df: pd.DataFrame):
    preferred = ["split", "seed", "group"]
    return [c for c in preferred if c in df.columns]


def paired_wilcoxon(df: pd.DataFrame, metric: str, left: str, right: str):
    idx_cols = get_index_cols(df)
    x = df.pivot_table(index=idx_cols, columns="model", values=metric, aggfunc="mean")
    needed = [left, right]
    if any(m not in x.columns for m in needed):
        return np.nan
    y = x[needed].dropna()
    if len(y) < 2:
        return np.nan
    try:
        return wilcoxon(y[left], y[right]).pvalue
    except ValueError:
        return np.nan


def get_meanshift_baseline(df: pd.DataFrame, metric: str):
    x = df.loc[df["model"] == MEANSHIFT_MODEL, metric].dropna()
    if len(x) == 0:
        return np.nan
    return float(x.mean())


def draw_jitter(ax, x_center, values, color, width=0.08, size=16, alpha=0.55):
    if len(values) == 0:
        return
    rng = np.random.default_rng(2026)
    jitter = rng.uniform(-width, width, size=len(values))
    ax.scatter(
        np.full(len(values), x_center) + jitter,
        values,
        s=size,
        color=color,
        alpha=alpha,
        zorder=2,
        linewidths=0,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
df = load_input_df()
GAUSSIAN_MODEL = detect_gaussian_model(df)

main_models = [DET_MODEL, GAUSSIAN_MODEL, DIFF_MODEL]
plot_labels = [
    DISPLAY_LABELS[DET_MODEL],
    GAUSSIAN_DISPLAY,
    DISPLAY_LABELS[DIFF_MODEL],
]
plot_colors = [
    COLORS[DET_MODEL],
    COLORS["GAUSSIAN"],
    COLORS[DIFF_MODEL],
]

# 仅保留主比较模型和 MeanShift 基线
valid_models = set(main_models + [MEANSHIFT_MODEL])
df = df[df["model"].isin(valid_models)].copy()

# 统计 p 值：Diffusion vs Deterministic / Diffusion vs Gaussian
stat = {}
for metric, _ in METRICS:
    stat[metric] = {
        DET_MODEL: paired_wilcoxon(df[df["model"].isin(main_models)], metric, DIFF_MODEL, DET_MODEL),
        GAUSSIAN_MODEL: paired_wilcoxon(df[df["model"].isin(main_models)], metric, DIFF_MODEL, GAUSSIAN_MODEL),
    }

fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.6))
axes = axes.flatten()

for ax, (metric, title) in zip(axes, METRICS):
    data = []
    for model in main_models:
        vals = df.loc[df.model == model, metric].dropna().astype(float).values
        data.append(vals)

    bp = ax.boxplot(
        data,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.1),
        capprops=dict(linewidth=1.1),
        medianprops=dict(linewidth=1.4, color="#D55E00"),
    )

    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor(color)

    for whisker, color in zip(bp["whiskers"], [plot_colors[0], plot_colors[0], plot_colors[1], plot_colors[1], plot_colors[2], plot_colors[2]]):
        whisker.set_color(color)
    for cap, color in zip(bp["caps"], [plot_colors[0], plot_colors[0], plot_colors[1], plot_colors[1], plot_colors[2], plot_colors[2]]):
        cap.set_color(color)

    # jitter points + means
    for i, (vals, color) in enumerate(zip(data, plot_colors), start=1):
        draw_jitter(ax, i, vals, color=color)
        if len(vals) > 0:
            ax.scatter(
                i,
                np.mean(vals),
                marker="D",
                s=34,
                color="black",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )

    # MeanShift baseline as dashed horizontal line
    ms_value = get_meanshift_baseline(df, metric)
    if np.isfinite(ms_value):
        ax.axhline(
            ms_value,
            color="#CC3311",
            linestyle=(0, (5, 3)),
            linewidth=1.4,
            alpha=0.95,
            zorder=1,
        )
        ax.text(
            0.985,
            0.055,
            f"MeanShift={ms_value:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#AA2E25",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(plot_labels, fontsize=10)
    ax.set_ylabel(title, fontsize=11)

    p_det = stat[metric][DET_MODEL]
    p_gau = stat[metric][GAUSSIAN_MODEL]
    p_text = (
        f"Diff vs Det: P={p_det:.3g}" if np.isfinite(p_det) else "Diff vs Det: P=NA"
    ) + "\n" + (
        f"Diff vs Gau: P={p_gau:.3g}" if np.isfinite(p_gau) else "Diff vs Gau: P=NA"
    )

    ax.text(
        0.03,
        0.96,
        p_text,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.tick_params(axis="both", labelsize=10)

# Panel letters
for ax, letter in zip(axes, "ABCDEF"):
    ax.text(
        -0.16,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
    )

# Figure legend
legend_handles = [
    Line2D([0], [0], marker="D", color="black", markersize=6, linestyle="None", label="Mean of box"),
    Line2D([0, 1], [0, 0], color="#CC3311", linestyle=(0, (5, 3)), linewidth=1.4, label="MeanShift baseline"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
    fontsize=10,
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

pdf_path = BASE / f"{OUT_PREFIX}.pdf"
png_path = BASE / f"{OUT_PREFIX}.png"
plt.savefig(pdf_path, dpi=500, bbox_inches="tight")
plt.savefig(png_path, dpi=700, bbox_inches="tight")
plt.close()

print(f"[DONE] Saved figure to:\n  {pdf_path}\n  {png_path}")
