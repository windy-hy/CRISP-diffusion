import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


# =========================
# 1. 文件
# =========================

csv_path = "fig2e.csv"
output_pdf = "fig2e.pdf"


# =========================
# 2. Arial字体
# =========================

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

plt.rcParams["font.size"] = 11

plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


# =========================
# 3. 读取数据
# =========================

df = pd.read_csv(csv_path)

df = df.loc[
    :,
    ~df.columns.str.startswith("Unnamed")
]

required_columns = [
    "pos_ratio",
    "model",
    "cov_drug",
    "cell_type",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            "CSV中缺少列：" + column
        )


df["pos_ratio"] = pd.to_numeric(
    df["pos_ratio"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "pos_ratio",
        "model",
        "cell_type",
    ]
).copy()


# =========================
# 4. 顺序
# =========================

cell_order = [
    "Myeloid cells",
    "T regulatory cells",
    "B cells",
    "T cells CD4+",
    "NK cells",
    "T cells CD8+",
]

method_order = [
    "DIFFCRISP",
    "CRISP",
]


print(
    df.groupby(
        ["cell_type", "model"]
    )
    .size()
    .unstack(fill_value=0)
)


# =========================
# 5. 显示名称
# =========================

cell_display = {
    "Myeloid cells": "Myeloid cells",
    "T regulatory cells": "T reg cells",
    "B cells": "B cells",
    "T cells CD4+": "CD4+ T cells",
    "NK cells": "NK cells",
    "T cells CD8+": "CD8+ T cells",
}

method_display = {
    "DIFFCRISP": "DIFFCRISP",
    "CRISP": "CRISP",
}


# =========================
# 6. Control cell数量
# =========================

control_counts = {
    "Myeloid cells": 1078,
    "T regulatory cells": 160,
    "B cells": 1100,
    "T cells CD4+": 2695,
    "NK cells": 1446,
    "T cells CD8+": 324,
}


# =========================
# 7. 每个cell type的扰动组合数量
# 自动由cov_drug计算
# =========================

combination_counts = (
    df.groupby("cell_type")["cov_drug"]
    .nunique()
    .to_dict()
)

print("\n各细胞类型扰动组合数量：")

for cell_type in cell_order:
    print(
        cell_type,
        combination_counts.get(
            cell_type,
            0
        )
    )


# =========================
# 8. 配色
# =========================

colors = {
    "DIFFCRISP": "#C96F5B",
    "CRISP": "#4F76A8",
}


# =========================
# 9. 画布
# =========================

fig, ax = plt.subplots(
    figsize=(8.5, 4.5)
)

group_positions = np.arange(
    len(cell_order)
)


# 两个模型重新居中
offsets = {
    "DIFFCRISP": -0.13,
    "CRISP": 0.13,
}

box_width = 0.24


# =========================
# 10. 绘制箱线图
# =========================

for method in method_order:

    values_list = []

    for cell_type in cell_order:

        values = df.loc[
            (df["model"] == method)
            & (df["cell_type"] == cell_type),
            "pos_ratio",
        ].to_numpy()

        if len(values) == 0:
            raise ValueError(
                "没有找到数据："
                + method
                + "，"
                + cell_type
            )

        values_list.append(values)


    positions = (
        group_positions
        + offsets[method]
    )


    result = ax.boxplot(
        values_list,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        manage_ticks=False,

        whis=1.5,

        showfliers=True,
        showmeans=False,

        boxprops={
            "edgecolor": "#4D4D4D",
            "linewidth": 0.6,
        },

        medianprops={
            "color": "#3F3F3F",
            "linewidth": 0.7,
        },

        whiskerprops={
            "color": "#5A5A5A",
            "linewidth": 0.6,
        },

        capprops={
            "color": "#5A5A5A",
            "linewidth": 0.6,
        },

        flierprops={
            "marker": "o",
            "markersize": 2.8,
            "markerfacecolor": "white",
            "markeredgecolor": "#4D4D4D",
            "markeredgewidth": 0.5,
            "linestyle": "none",
        },
    )


    for box in result["boxes"]:
        box.set_facecolor(
            colors[method]
        )


# =========================
# 11. 横坐标标签
# =========================

x_labels = []

for cell_type in cell_order:

    display_name = cell_display[
        cell_type
    ]

    m_value = control_counts[
        cell_type
    ]

    n_value = combination_counts.get(
        cell_type,
        0
    )

    label = (
        display_name
        + "\n"
        + r"($m = "
        + str(m_value)
        + "$"
        + "\n"
        + r"$n = "
        + str(n_value)
        + "$)"
    )

    x_labels.append(label)


ax.set_xticks(
    group_positions
)

ax.set_xticklabels(
    x_labels,
    fontsize=9.6,
    fontfamily="Arial",
)

ax.tick_params(
    axis="x",
    length=0,
    pad=7,
)


# =========================
# 12. 纵坐标
# =========================

ax.set_ylim(
    -0.03,
    1.05
)

ax.set_yticks(
    np.arange(
        0,
        1.01,
        0.2
    )
)


def format_y_tick(
    value,
    position
):

    if abs(value) < 1e-8:
        return "0"

    return f"{value:.1f}"


ax.yaxis.set_major_formatter(
    FuncFormatter(
        format_y_tick
    )
)


ax.set_ylabel(
    "Predicted direction accuracy\n"
    "on top 50 DE genes",
    fontsize=11.5,
    fontfamily="Arial",
)


ax.tick_params(
    axis="y",
    labelsize=10,
    width=0.6,
    length=3,
)


# =========================
# 13. 坐标轴
# =========================

ax.set_xlim(
    -0.48,
    len(cell_order) - 0.52,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(0.6)
ax.spines["bottom"].set_linewidth(0.6)

ax.spines["left"].set_color(
    "#333333"
)

ax.spines["bottom"].set_color(
    "#333333"
)

ax.grid(False)


# =========================
# 14. 图例
# =========================

legend_handles = []

for method in method_order:

    legend_handles.append(
        Line2D(
            [0],
            [0],

            marker="o",
            linestyle="none",

            markerfacecolor=colors[method],
            markeredgecolor="#888888",
            markeredgewidth=0.5,

            markersize=7.5,

            label=method_display[method],
        )
    )


ax.legend(
    handles=legend_handles,

    loc="upper center",
    bbox_to_anchor=(0.50, 1.16),

    ncol=2,
    frameon=False,

    fontsize=11,

    handlelength=1,
    handletextpad=0.35,
    columnspacing=1.5,
)


# =========================
# 15. 面板字母 e
# =========================

fig.text(
    0.025,
    0.965,

    "e",

    fontsize=17,
    fontweight="bold",
    fontfamily="Arial",

    ha="left",
    va="top",
)


# =========================
# 16. 布局和保存
# =========================

fig.subplots_adjust(
    left=0.13,
    right=0.99,
    bottom=0.24,
    top=0.82,
)


plt.savefig(
    output_pdf,
    format="pdf",
    bbox_inches="tight",
)

plt.show()


print(
    "已保存：",
    output_pdf
)


print(
    df.groupby(
        ["cell_type", "model"]
    )
    .size()
    .unstack(fill_value=0)
)