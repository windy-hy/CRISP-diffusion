from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon, gaussian_kde


# =========================
# 文件
# =========================
SUMMARY = "./response_analysis/responder_proportion.csv"
CASE_DIR = Path("./response_analysis/known_drug_cases")
OUT = "Fig4_responder_recovery.pdf"

# 1120 × 650 pt
FIG_W = 1120 / 72
FIG_H = 750 / 72


# =========================
# 字体
# =========================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================
# 配色
# =========================
COLORS = {
    "CRISP": "#4F76A8",
    "DiffCRISP": "#C96F5B",
    "Control": "#B8B8B8",
    "Observed": "#444444",
}

SPLIT_COLORS = {
    "split": "#4C78A8",
    "split2": "#F58518",
    "split3": "#54A24B",
}

SPLITS = ["split", "split2", "split3"]
SPLIT_NAMES = ["Split 1", "Split 2", "Split 3"]


# =========================
# 读取数据
# =========================
df = pd.read_csv(SUMMARY)


def load_case(drug):
    for f in CASE_DIR.glob("*.csv"):
        x = pd.read_csv(f)
        if len(x) and (
            x["cell_type"].iloc[0] == "T cells CD4+"
            and x["drug"].iloc[0] == drug
        ):
            return x

    raise FileNotFoundError(drug)


idelalisib = load_case("Idelalisib")
palbociclib = load_case("Palbociclib")


# =========================
# 画布
# =========================
fig = plt.figure(figsize=(FIG_W, FIG_H))

gs = fig.add_gridspec(
    2, 6,
    height_ratios=[1, 1.15],
    left=0.065,
    right=0.985,
    bottom=0.09,
    top=0.95,
    wspace=0.78,
    hspace=0.42,
)

a = fig.add_subplot(gs[0, 0:2])
b = fig.add_subplot(gs[0, 2:4])
c = fig.add_subplot(gs[0, 4:6])

d1 = fig.add_subplot(gs[1, 0:3])
d2 = fig.add_subplot(gs[1, 3:6])


# =========================================================
# a. Responder proportion error
# =========================================================
for i, split in enumerate(SPLITS):

    x = df[df["split"] == split]

    for pos, column, model in [
        (i - 0.14, "crisp_error", "CRISP"),
        (i + 0.14, "diffcrisp_error", "DiffCRISP"),
    ]:
        values = x[column].to_numpy()

        bp = a.boxplot(
            values,
            positions=[pos],
            widths=0.25,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            whis=1.5,
            boxprops={"linewidth": 0.8, "edgecolor": "#555555"},
            medianprops={"linewidth": 0.9, "color": "#333333"},
            whiskerprops={"linewidth": 0.8, "color": "#666666"},
            capprops={"linewidth": 0.8, "color": "#666666"},
        )

        bp["boxes"][0].set_facecolor(COLORS[model])

        rng = np.random.default_rng(100 + i)
        a.scatter(
            pos + rng.normal(0, 0.022, len(values)),
            values,
            s=5,
            alpha=0.20,
            color=COLORS[model],
            edgecolors="none",
        )

    p = wilcoxon(
        x["crisp_error"],
        x["diffcrisp_error"]
    ).pvalue

    text = f"$P$ = {p:.1e}" if p < 0.001 else f"$P$ = {p:.4f}"

    a.text(
        i, 1.035, text,
        ha="center",
        fontsize=8.5,
    )


a.set(
    xlim=(-0.55, 2.55),
    ylim=(0, 1.12),
    ylabel="Absolute error of\nresponder proportion",
    title="Responder proportion error",
)

a.set_xticks(range(3))
a.set_xticklabels(SPLIT_NAMES)

a.legend(
    handles=[
        Line2D(
            [0], [0],
            marker="s",
            linestyle="none",
            markerfacecolor=COLORS["CRISP"],
            markeredgecolor="#666666",
            label="CRISP",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="none",
            markerfacecolor=COLORS["DiffCRISP"],
            markeredgecolor="#666666",
            label="DiffCRISP",
        ),
    ],
    frameon=False,
    fontsize=9,
    loc="upper right",
)


# =========================================================
# b. Perturbation-level comparison
# =========================================================
for split, name in zip(SPLITS, SPLIT_NAMES):

    x = df[df["split"] == split]

    b.scatter(
        x["crisp_error"],
        x["diffcrisp_error"],
        s=13,
        alpha=0.40,
        color=SPLIT_COLORS[split],
        edgecolors="none",
        label=name,
    )


b.plot(
    [0, 1], [0, 1],
    "--",
    lw=0.75,
    color="#666666",
)

b.set(
    xlim=(-0.02, 1.02),
    ylim=(-0.02, 1.02),
    xlabel="CRISP responder proportion error",
    ylabel="DiffCRISP responder proportion error",
    title="Perturbation-level comparison",
)

b.set_aspect("equal")

overall_win = np.mean(
    df["diffcrisp_error"] < df["crisp_error"]
)

b.text(
    0.97, 0.05,
    f"DiffCRISP win rate\n= {overall_win * 100:.1f}%",
    transform=b.transAxes,
    ha="right",
    fontsize=9,
)

b.legend(
    frameon=False,
    fontsize=8.5,
    loc="upper left",
)


# =========================================================
# c. Win rate
# =========================================================
rates, ns = [], []

for split in SPLITS:
    x = df[df["split"] == split]
    rates.append(
        np.mean(
            x["diffcrisp_error"] < x["crisp_error"]
        )
    )
    ns.append(len(x))

rates.append(overall_win)
ns.append(len(df))

c.bar(
    range(4),
    rates,
    width=0.56,
    color=COLORS["DiffCRISP"],
)

c.axhline(
    0.5,
    ls="--",
    lw=0.75,
    color="#888888",
)

for i, v in enumerate(rates):
    c.text(
        i, v + 0.025,
        f"{v * 100:.1f}%",
        ha="center",
        fontsize=9,
    )

c.set(
    ylim=(0, 1.05),
    ylabel="DiffCRISP win rate",
    title="Perturbation-level win rate",
)

c.set_xticks(range(4))
c.set_xticklabels([
    f"Split 1\n$n$ = {ns[0]}",
    f"Split 2\n$n$ = {ns[1]}",
    f"Split 3\n$n$ = {ns[2]}",
    f"Overall\n$n$ = {ns[3]}",
])


# =========================================================
# d. 两个案例
# =========================================================
def draw_case(ax, data, drug):

    threshold = data["threshold"].iloc[0]
    groups = ["Control", "Observed", "CRISP", "DiffCRISP"]

    scores = {
        g: data.loc[data["group"] == g, "score"].to_numpy()
        for g in groups
    }

    props = {
        g: np.mean(scores[g] > threshold)
        for g in groups
    }

    values = np.concatenate(list(scores.values()))

    low, high = np.percentile(values, [0.5, 99.5])
    low = min(low, threshold)
    high = max(high, threshold)

    xx = np.linspace(
        low - 0.07 * (high - low),
        high + 0.07 * (high - low),
        400,
    )

    for g in groups:
        if np.std(scores[g]) > 1e-8:
            kde = gaussian_kde(scores[g])

            ax.plot(
                xx,
                kde(xx),
                lw=1.3,
                color=COLORS[g],
                label=f"{g} ({props[g]:.2f})",
            )

    ax.axvline(
        threshold,
        ls="--",
        lw=0.8,
        color="#444444",
    )

    ax.text(
        threshold,
        0.97,
        "Responder threshold",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="top",
        fontsize=8.5,
    )

    dose = float(data["dose"].iloc[0])

    ax.set(
        xlabel="Single-cell response score",
        ylabel="Density",
        title=f"CD4+ T cells · {drug} · dose {dose:g}",
    )

    ax.legend(
        title="Responder proportion",
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )


draw_case(d1, idelalisib, "Idelalisib")
draw_case(d2, palbociclib, "Palbociclib")


# =========================================================
# 全局样式
# =========================================================
axes = [a, b, c, d1, d2]

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.tick_params(
        width=0.5,
        length=3,
        labelsize=9.5,
    )


# a b c d
for ax, letter in [
    (a, "a"),
    (b, "b"),
    (c, "c"),
    (d1, "d"),
]:
    ax.text(
        -0.13,
        1.07,
        letter,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
    )


plt.savefig(
    OUT,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04,
)

plt.show()

print("已保存：", OUT)
print("尺寸：1120 × 650 pt")