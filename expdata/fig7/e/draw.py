import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CRISP_FILE = "fig4e_crisp.csv"
DIFFCRISP_FILE = "fig4e_diffcrisp.csv"

OUTPUT_PDF = "fig4e.pdf"


BATCH_ORDER = [
    "Drop-seq",
    "Smart-seq2",
    "inDrops",
    "10x Chromium (v2)",
    "10x Chromium (v3)",
    "CEL-Seq2",
]

SHORT_NAMES = {
    "Drop-seq": "Drop-seq",
    "Smart-seq2": "Smart-seq2",
    "inDrops": "inDrops",
    "10x Chromium (v2)": "10x (v2)",
    "10x Chromium (v3)": "10x (v3)",
    "CEL-Seq2": "CEL-Seq2",
}

COLORS = {
    "DiffCRISP": "#82A8B2",
    "CRISP": "#C8C5A9",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 10,

    # 数学文字也统一为 Arial
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",

    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    # PDF 中保持可编辑字体
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# 读取数据
# =========================

crisp = pd.read_csv(CRISP_FILE, encoding="utf-8-sig")
diffcrisp = pd.read_csv(DIFFCRISP_FILE, encoding="utf-8-sig")

crisp["method"] = "CRISP"
diffcrisp["method"] = "DiffCRISP"

df = pd.concat(
    [diffcrisp, crisp],
    ignore_index=True
)

df["direction_acc"] = pd.to_numeric(
    df["direction_acc"],
    errors="coerce"
)

df["n_cells"] = pd.to_numeric(
    df["n_cells"],
    errors="coerce"
)


# 兼容没有batch_group的旧CSV
if "batch_group" not in df.columns:
    source = (
        df["batch_original"]
        if "batch_original" in df.columns
        else df["batch"]
    )

    df["batch_group"] = source.astype(str)

    df.loc[
        df["batch_group"].str.startswith(
            "10x Chromium (v2)"
        ),
        "batch_group"
    ] = "10x Chromium (v2)"


df = df[
    df["batch_group"].isin(BATCH_ORDER)
    & df["direction_acc"].notna()
].copy()


# =========================
# 计算横轴细胞数
# =========================

raw_batch_col = (
    "batch_original"
    if "batch_original" in diffcrisp.columns
    else "batch_group"
)

n_table = diffcrisp[
    [
        raw_batch_col,
        "batch_group",
        "n_cells",
    ]
].drop_duplicates(
    subset=[raw_batch_col]
)

n_by_batch = (
    n_table.groupby("batch_group")["n_cells"]
    .sum()
    .astype(int)
    .to_dict()
)

print("\n每个平台、每种方法的结果数量：")
print(
    df.groupby(
        ["batch_group", "method"]
    )
    .size()
    .unstack(fill_value=0)
)


# =========================
# 绘图
# =========================

fig, ax = plt.subplots(figsize=(6.2, 4.2))

x = np.arange(len(BATCH_ORDER))
width = 0.30

METHODS = ["DiffCRISP", "CRISP"]
OFFSETS = [-0.18, 0.18]

for method, offset in zip(METHODS, OFFSETS):

    values = [
        df.loc[
            (df["batch_group"] == batch)
            & (df["method"] == method),
            "direction_acc"
        ].to_numpy()
        for batch in BATCH_ORDER
    ]

    box = ax.boxplot(
        values,
        positions=x + offset,
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        showfliers=True,
        whis=1.5,
        medianprops={
            "color": "#666666",
            "linewidth": 0.9,
        },
        boxprops={
            "edgecolor": "#707070",
            "linewidth": 0.8,
        },
        whiskerprops={
            "color": "#707070",
            "linewidth": 0.8,
        },
        capprops={
            "color": "#707070",
            "linewidth": 0.8,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#707070",
            "markeredgewidth": 0.7,
            "markersize": 5,
        },
    )

    for patch in box["boxes"]:
        patch.set_facecolor(COLORS[method])


# trivial baseline
# ax.axhline(
#     BASELINE,
#     color="black",
#     linewidth=0.9,
#     linestyle=(0, (4, 3)),
#     zorder=1,
# )


# =========================
# 坐标轴
# =========================

x_labels = [
    (
        f"{SHORT_NAMES[batch]}\n"
        rf"($m={n_by_batch.get(batch, 0)}$)"
    )
    for batch in BATCH_ORDER
]

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9)

ax.set_xlim(-0.55, len(BATCH_ORDER) - 0.45)
ax.set_ylim(0, 0.90)
ax.set_yticks(np.arange(0, 0.91, 0.2))

ax.set_ylabel(
    "Predicted direction accuracy",
    fontsize=11
)

ax.set_title(
    "NK cells (unseen drugs)",
    fontsize=11,
    pad=8
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="both",
    direction="out",
    length=4
)


# =========================
# 图例
# =========================

handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=COLORS["DiffCRISP"],
        markeredgecolor="none",
        markersize=8,
        label="DiffCRISP",
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=COLORS["CRISP"],
        markeredgecolor="none",
        markersize=8,
        label="CRISP",
    ),
    # Line2D(
    #     [0], [0],
    #     color="black",
    #     linestyle=(0, (4, 3)),
    #     linewidth=0.9,
    #     label="Trivial baseline",
    # ),
]

ax.legend(
    handles=handles,
    loc="upper left",
    bbox_to_anchor=(0.85, 1.02),
    frameon=False,
    handlelength=1.5,
    handletextpad=0.5,
)


# 面板编号
ax.text(
    -0.13,
    1.08,
    "e",
    transform=ax.transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
)


# =========================
# 保存
# =========================

plt.subplots_adjust(
    left=0.15,
    right=0.96,
    top=0.88,
    bottom=0.23,
)

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.show()