import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =========================
# 文件路径
# =========================

CRISP_FILE = "fig4d_crisp_summary.csv"
DIFFCRISP_FILE = "fig4d_diffcrisp_summary.csv"

OUTPUT_PDF = "fig4d.pdf"



# =========================
# 顺序与显示名称
# =========================

SETTING_ORDER = [
    "Seen drugs",
    "Unseen drugs",
]

CELL_ORDER = [
    "B cells",
    "NK cells",
    "T cells CD4+",
]

CELL_LABELS = {
    "B cells": "B cells",
    "NK cells": "NK cells",
    "T cells CD4+": "CD4+ T",
}

METHOD_ORDER = [
    "DiffCRISP",
    "CRISP",
]

COLORS = {
    "DiffCRISP": "#4478B8",
    "CRISP": "#C5AF82",
}


# =========================
# 全局样式
# =========================

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

crisp = pd.read_csv(
    CRISP_FILE,
    encoding="utf-8-sig",
)

diffcrisp = pd.read_csv(
    DIFFCRISP_FILE,
    encoding="utf-8-sig",
)

crisp["method"] = "CRISP"
diffcrisp["method"] = "DiffCRISP"

df = pd.concat(
    [diffcrisp, crisp],
    ignore_index=True,
)


# =========================
# 数值列转换
# =========================

for col in [
    "mean_direction_acc",
    "std_direction_acc",
    "sem_direction_acc",
    "n",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )


required_columns = [
    "setting",
    "cell_type",
    "method",
    "mean_direction_acc",
    "sem_direction_acc",
    "n",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise KeyError(
        f"CSV缺少列：{missing_columns}\n"
        f"当前列：{df.columns.tolist()}"
    )


# 只保留需要的组
df = df[
    df["setting"].isin(SETTING_ORDER)
    & df["cell_type"].isin(CELL_ORDER)
    & df["method"].isin(METHOD_ORDER)
].copy()


# =========================
# 检查数据
# =========================

print("\n读取到的数据：")

print(
    df[
        [
            "setting",
            "cell_type",
            "method",
            "mean_direction_acc",
            "sem_direction_acc",
            "n",
        ]
    ].sort_values(
        [
            "setting",
            "cell_type",
            "method",
        ]
    ).to_string(index=False)
)

group_count = (
    df.groupby(
        [
            "setting",
            "cell_type",
            "method",
        ]
    )
    .size()
)

if not (group_count == 1).all():
    raise ValueError(
        "每个setting × cell_type × method "
        "应该只有一行汇总结果"
    )


# =========================
# 绘图
# =========================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.2, 4.0),
    sharey=True,
)

x = np.arange(len(CELL_ORDER))

bar_width = 0.32

OFFSETS = {
    "DiffCRISP": -bar_width / 2,
    "CRISP": bar_width / 2,
}


for panel_index, setting in enumerate(
    SETTING_ORDER
):

    ax = axes[panel_index]

    panel_df = df[
        df["setting"] == setting
    ]

    for method in METHOD_ORDER:

        means = []
        errors = []
        sample_sizes = []

        for cell_type in CELL_ORDER:

            row = panel_df[
                (panel_df["cell_type"] == cell_type)
                & (panel_df["method"] == method)
            ]

            if len(row) != 1:
                raise ValueError(
                    f"找不到唯一结果："
                    f"{setting} / {cell_type} / {method}"
                )

            means.append(
                float(
                    row["mean_direction_acc"].iloc[0]
                )
            )

            errors.append(
                float(
                    row["sem_direction_acc"].iloc[0]
                )
            )

            sample_sizes.append(
                int(
                    row["n"].iloc[0]
                )
            )

        ax.bar(
            x + OFFSETS[method],
            means,
            width=bar_width,
            color=COLORS[method],
            edgecolor="none",
            yerr=errors,
            error_kw={
                "ecolor": "#333333",
                "elinewidth": 0.9,
                "capsize": 4,
                "capthick": 0.9,
            },
            zorder=2,
        )

    # 横轴标签的n取两种方法一致的样本数
    x_labels = []

    for cell_type in CELL_ORDER:

        rows = panel_df[
            panel_df["cell_type"] == cell_type
        ]

        n_values = (
            rows["n"]
            .dropna()
            .astype(int)
            .unique()
        )

        if len(n_values) == 1:
            n_value = int(n_values[0])
        else:
            # 两种方法数量不一致时，显示各自数量
            n_value = "/".join(
                map(str, sorted(n_values))
            )

        x_labels.append(
            f"{CELL_LABELS[cell_type]}\n"
            rf"($n={n_value}$)"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        x_labels,
        fontsize=9,
    )

    ax.set_title(
        setting,
        fontsize=11,
        pad=10,
    )

    ax.set_xlim(
        -0.55,
        len(CELL_ORDER) - 0.45,
    )

    ax.set_ylim(0, 0.70)
    ax.set_yticks(
        np.arange(0, 0.61, 0.2)
    )

    ax.tick_params(
        axis="both",
        direction="out",
        length=4,
        width=0.8,
        labelsize=9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


# 左侧纵轴标题
axes[0].set_ylabel(
    "Predicted direction accuracy",
    fontsize=11,
)

# 右图也显示纵轴刻度数字
axes[1].tick_params(
    labelleft=True
)


# =========================
# 图例
# =========================

legend_handles = [
    Patch(
        facecolor=COLORS["DiffCRISP"],
        edgecolor="none",
        label="DiffCRISP",
    ),
    Patch(
        facecolor=COLORS["CRISP"],
        edgecolor="none",
        label="CRISP",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.54, 0.015),
    ncol=2,
    frameon=False,
    fontsize=10,
    handlelength=1.0,
    handletextpad=0.5,
    columnspacing=1.8,
)


# =========================
# 面板编号
# =========================

fig.text(
    0.015,
    0.975,
    "d",
    fontsize=14,
    fontweight="bold",
    ha="left",
    va="top",
)


# =========================
# 保存
# =========================

plt.subplots_adjust(
    left=0.12,
    right=0.98,
    top=0.88,
    bottom=0.23,
    wspace=0.22,
)

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.show()