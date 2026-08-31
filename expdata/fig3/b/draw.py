from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "sampling_summary.csv"
OUT_FILE = BASE / "sampling_variation.pdf"

LOW_PERCENTILE = 2.5
HIGH_PERCENTILE = 97.5


def percentile_keep(x):
    low, high = np.percentile(
        x,
        [LOW_PERCENTILE, HIGH_PERCENTILE]
    )
    return (x >= low) & (x <= high)


def paired_p(x, y):
    if np.allclose(x, y):
        return 1.0
    return float(wilcoxon(x, y).pvalue)


def draw_pairs(ax, left, right, ylabel, title, log_scale=False):
    jitter = np.random.default_rng(42).normal(
        0, 0.018, len(left)
    )

    for i in range(len(left)):
        ax.plot(
            [1 + jitter[i], 2 + jitter[i]],
            [left[i], right[i]],
            color="#bbbbbb",
            linewidth=0.4,
            alpha=0.45,
            zorder=1
        )

    ax.scatter(
        1 + jitter,
        left,
        s=6,
        color="#4C78A8",
        alpha=0.8,
        edgecolors="none",
        zorder=2
    )

    ax.scatter(
        2 + jitter,
        right,
        s=6,
        color="#E45756",
        alpha=0.8,
        edgecolors="none",
        zorder=2
    )

    ax.plot(
        [0.82, 1.18],
        [np.median(left)] * 2,
        color="black",
        linewidth=1.1
    )

    ax.plot(
        [1.82, 2.18],
        [np.median(right)] * 2,
        color="black",
        linewidth=1.1
    )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlim(0.65, 2.35)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["CRISP", "DiffCRISP"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =========================
# 读取
# =========================
data = pd.read_csv(DATA_FILE)

required = [
    "group",
    "crisp_between_seed",
    "diffcrisp_between_seed",
    "crisp_ratio",
    "diffcrisp_ratio"
]

if not set(required).issubset(data.columns):
    raise RuntimeError("CSV缺少必要列")

finite = np.isfinite(data[required[1:]]).all(axis=1)

positive = (
    (data["crisp_ratio"] > 0)
    & (data["diffcrisp_ratio"] > 0)
)

data = data.loc[finite & positive].copy()


# =========================
# 统一删除极端组合
# =========================
a_c = data["crisp_between_seed"].to_numpy()
a_d = data["diffcrisp_between_seed"].to_numpy()

b_c = np.log10(
    data["crisp_ratio"].to_numpy()
)

b_d = np.log10(
    data["diffcrisp_ratio"].to_numpy()
)

keep = (
    percentile_keep(a_c)
    & percentile_keep(a_d)
    & percentile_keep(b_c)
    & percentile_keep(b_d)
)

removed = data.loc[~keep, "group"].tolist()
data = data.loc[keep].reset_index(drop=True)

print("原始组合数：", len(keep))
print("删除组合数：", len(removed))
print("最终组合数：", len(data))

print("\n删除的组合：")
for name in removed:
    print(name)


# =========================
# 统计
# =========================
between_c = data["crisp_between_seed"].to_numpy()
between_d = data["diffcrisp_between_seed"].to_numpy()

ratio_c = data["crisp_ratio"].to_numpy()
ratio_d = data["diffcrisp_ratio"].to_numpy()

error_c = np.abs(np.log(ratio_c))
error_d = np.abs(np.log(ratio_d))

p_between = paired_p(between_c, between_d)
p_ratio = paired_p(error_c, error_d)

closer = np.mean(error_d < error_c) * 100

print("\n面板a p值：", p_between)
print("面板b p值：", p_ratio)
print("DiffCRISP更接近1：", closer)
print("CRISP ratio中位数：", np.median(ratio_c))
print("DiffCRISP ratio中位数：", np.median(ratio_d))


# =========================
# 画图
# =========================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig, axes = plt.subplots(
    1,
    2,
    figsize=(6.5, 3.2)
)

draw_pairs(
    axes[0],
    between_c,
    between_d,
    ylabel="Between-seed Sinkhorn",
    title="Between-seed variation"
)

draw_pairs(
    axes[1],
    ratio_c,
    ratio_d,
    ylabel="Sampling variation ratio",
    title="Relative sampling variation",
    log_scale=True
)

axes[1].axhline(
    1,
    linestyle="--",
    color="black",
    linewidth=1
)

# for ax, label in zip(axes, ["a", "b"]):
#     ax.text(
#         -0.15,
#         1.06,
#         label,
#         transform=ax.transAxes,
#         fontsize=13,
#         fontweight="bold",
#         va="top"
#     )

plt.tight_layout()

plt.savefig(
    OUT_FILE,
    bbox_inches="tight"
)

plt.show()

print("\n保存：", OUT_FILE)