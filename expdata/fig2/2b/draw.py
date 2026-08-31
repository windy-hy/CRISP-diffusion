import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 全局字体：Arial
# =========================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,

    "axes.titlesize": 16,
    "axes.labelsize": 11,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,

    # 数学公式也尽量统一为 Arial
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",

    # PDF 中保留 TrueType 字体，方便 Illustrator
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================
# 读取数据
# =========================
df = pd.read_excel("fig2b.xlsx", header=None)

methods = [
    "DIFFCRISP",
    "CRISP",
    "CellOT",
    "scGen",
    "Biolord",
    "ChemCPA",
    "CPA"
]

keys = ["pearson", "r2", "sinkhorn"]

titles = [
    r"$\mathrm{Pr}_{\Delta}$ top 50 DE genes (↑)",
    "R² top 50 DE genes (↑)",
    "Sinkhorn top 50 DE genes (↓)"
]

colors = [
    "#d95f5f",
    "#e39b57",
    "#d1bf77",
    "#c8b27e",
    "#9cc9a9",
    "#87a8cf",
    "#b39ddb"
]


def get_data(key):
    text = df[0].fillna("").astype(str).str.lower()
    row = df.index[text.str.contains(key, regex=False)][0]

    headers = df.iloc[row + 1].astype(str).str.strip()
    data = df.iloc[row + 2:row + 11].copy()
    data.columns = headers

    return data[methods].apply(
        pd.to_numeric,
        errors="coerce"
    )


# =========================
# 绘图
# =========================
fig, axes = plt.subplots(
    1,
    3,
    figsize=(14, 5),
    sharey=True
)

rng = np.random.default_rng(42)


for ax, key, title in zip(axes, keys, titles):

    data = get_data(key)

    values = [
        data[m].dropna().values
        for m in methods
    ]

    boxes = ax.boxplot(
        values,
        vert=False,
        labels=methods,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        whis=(0, 100)
    )

    # =========================
    # 箱体
    # =========================
    for box, color in zip(
        boxes["boxes"],
        colors
    ):
        box.set_facecolor(color)
        box.set_edgecolor("black")

        # 原来 0.6 → 0.45
        box.set_linewidth(0.45)


    # =========================
    # whisker / cap / median
    # =========================
    for name in [
        "whiskers",
        "caps",
        "medians"
    ]:
        for line in boxes[name]:
            line.set_color("black")

            # 原来 0.6 → 0.45
            line.set_linewidth(0.45)


    # =========================
    # 原始数据点
    # =========================
    for position, value in enumerate(
        values,
        start=1
    ):

        y = rng.normal(
            position,
            0.025,
            len(value)
        )

        ax.scatter(
            value,
            y,
            s=4,
            color="black",
            linewidths=0,
            alpha=0.8,
            zorder=10
        )


    # =========================
    # 标题
    # 原来 fontsize=15
    # 略微放大到 16
    # =========================
    ax.set_title(
        title,
        fontsize=16
    )


    # =========================
    # 网格线
    # =========================
    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.45,
        alpha=0.4
    )


    ax.invert_yaxis()


    # =========================
    # 坐标轴边框稍微变细
    # =========================
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


    # =========================
    # tick 线稍微变细
    # 不改变 tick 的位置
    # =========================
    ax.tick_params(
        axis="both",
        width=0.5
    )


# =========================
# 左侧总标签
# 原来 fontsize=16
# 略微放大到 17
# =========================
fig.text(
    0.015,
    0.5,
    "NeurIPS unseen cell type",
    rotation=90,
    va="center",
    fontsize=17
)


# =========================
# 布局
# =========================
plt.tight_layout(
    rect=[0.035, 0, 1, 1]
)


# =========================
# 保存
# =========================
plt.savefig(
    "fig2b.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "fig2b.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()