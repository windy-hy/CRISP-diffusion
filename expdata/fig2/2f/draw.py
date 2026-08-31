from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = Path(__file__).resolve().parent

FILES = {
    "ZFAS1": DATA_DIR / "figure2f_ZFAS1.csv",
    "SNHG6": DATA_DIR / "figure2f_SNHG6.csv",
}

ORDER = ["DIFFCRISP", "CRISP", "True", "Ctrl"]
COLOR = "#5B8DBD"

sns.set_theme(style="white")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def draw_violin(ax, gene, csv_path):
    df = pd.read_csv(csv_path)

    df["exp"] = pd.to_numeric(
        df["exp"],
        errors="coerce",
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=["exp", "label"]
    )

    stats = (
        df.groupby("label")["exp"]
        .agg(["mean", "std"])
        .reindex(ORDER)
    )

    sns.violinplot(
        data=df,
        x="label",
        y="exp",
        order=ORDER,
        inner="box",
        cut=0,
        width=0.42,
        linewidth=0.6,
        color=COLOR,
        density_norm="width",
        ax=ax,
    )

    ax.set_title(
        gene,
        fontsize=18,
        pad=14,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(4.4, 9.8)

    ax.set_xticks(
        np.arange(len(ORDER))
    )

    ax.set_xticklabels(
        [""] * len(ORDER)
    )

    # 手动绘制组名和均值±标准差
    for x, label in enumerate(ORDER):
        mean = stats.loc[label, "mean"]
        std = stats.loc[label, "std"]

        # 四个组名
        ax.text(
            x,
            -0.06,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12.5,
            clip_on=False,
        )

        # 均值±标准差：字号加大并向下移动
        ax.text(
            x,
            -0.145,
            f"{mean:.2f}±{std:.2f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10.5,
            clip_on=False,
        )

    ax.tick_params(
        axis="y",
        labelsize=12,
        width=0.6,
        length=3,
    )

    ax.tick_params(
        axis="x",
        length=3,
        width=0.6,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(9.2, 6.6),
)

for ax, (gene, csv_path) in zip(
    axes,
    FILES.items(),
):
    draw_violin(
        ax,
        gene,
        csv_path,
    )

# 左侧纵坐标标题稍微缩小
axes[0].set_ylabel(
    "Gene expression",
    fontsize=16,
    labelpad=8,
)

# 底部总标题
fig.text(
    0.5,
    0.05,
    "Dactolisib and CD4+ T cells",
    ha="center",
    va="center",
    fontsize=18,
)

# 左上角编号
fig.text(
    0.015,
    0.975,
    "f",
    ha="left",
    va="top",
    fontsize=22,
    fontweight="bold",
)

plt.subplots_adjust(
    left=0.105,
    right=0.985,
    top=0.88,
    bottom=0.31,
    wspace=0.35,
)

pdf_path = DATA_DIR / "fig2f.pdf"

plt.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.show()
print("PDF已保存：", pdf_path)
