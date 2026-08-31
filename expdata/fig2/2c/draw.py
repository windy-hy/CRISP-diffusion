import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =========================================================
# 1. 文件设置
# =========================================================

excel_path = "fig2c.xlsx"
sheet_name = 0

output_png = "result_heatmap.png"
output_pdf = "result_heatmap.pdf"
output_mean_excel = "mean_results.xlsx"


methods = [
    "DIFFCRISP",
    "CRISP",
    "CellOT",
    "scGen",
    "Biolord",
    "ChemCPA",
    "CPA",
]

split_names = [
    "split",
    "split2",
    "split3",
]


# =========================================================
# 2. 读取整个Excel
# =========================================================

raw = pd.read_excel(
    excel_path,
    sheet_name=sheet_name,
    header=None,
)


# =========================================================
# 3. 提取表格
# =========================================================

def extract_table(
    dataframe,
    header_row,
    data_start_row,
    data_end_row,
    column_start,
    column_end,
):
    table = dataframe.iloc[
        data_start_row:data_end_row,
        column_start:column_end,
    ].copy()

    headers = dataframe.iloc[
        header_row,
        column_start:column_end,
    ].tolist()

    table.columns = [
        str(column).strip()
        for column in headers
    ]

    table = table.apply(
        pd.to_numeric,
        errors="coerce",
    )

    missing_methods = [
        method
        for method in methods
        if method not in table.columns
    ]

    if missing_methods:
        raise ValueError(
            "表格中缺少这些方法："
            + str(missing_methods)
            + "\n当前读取到的方法："
            + str(table.columns.tolist())
        )

    return table[methods]


# NeurIPS R²
nips_r2_raw = extract_table(
    dataframe=raw,
    header_row=1,
    data_start_row=2,
    data_end_row=11,
    column_start=0,
    column_end=7,
)

# NeurIPS Pr Delta
nips_pr_raw = extract_table(
    dataframe=raw,
    header_row=1,
    data_start_row=2,
    data_end_row=11,
    column_start=8,
    column_end=15,
)

# SciPlex3 R²
sci_r2_raw = extract_table(
    dataframe=raw,
    header_row=13,
    data_start_row=14,
    data_end_row=23,
    column_start=0,
    column_end=7,
)

# SciPlex3 Pr Delta
sci_pr_raw = extract_table(
    dataframe=raw,
    header_row=13,
    data_start_row=14,
    data_end_row=23,
    column_start=8,
    column_end=15,
)


# =========================================================
# 4. 每3行随机种子计算均值
# =========================================================

def mean_by_split(table):
    result = pd.DataFrame(
        index=methods,
        columns=split_names,
        dtype=float,
    )

    result["split"] = table.iloc[0:3].mean(axis=0)
    result["split2"] = table.iloc[3:6].mean(axis=0)
    result["split3"] = table.iloc[6:9].mean(axis=0)

    return result


nips_r2 = mean_by_split(nips_r2_raw)
sci_r2 = mean_by_split(sci_r2_raw)

nips_pr = mean_by_split(nips_pr_raw)
sci_pr = mean_by_split(sci_pr_raw)


# =========================================================
# 5. 打印并保存均值
# =========================================================

print("\nNeurIPS R²：")
print(nips_r2.round(4))

print("\nSciPlex3 R²：")
print(sci_r2.round(4))

print("\nNeurIPS Pr Delta：")
print(nips_pr.round(4))

print("\nSciPlex3 Pr Delta：")
print(sci_pr.round(4))


mean_output = pd.concat(
    {
        "NeurIPS_R2": nips_r2,
        "SciPlex3_R2": sci_r2,
        "NeurIPS_PrDelta": nips_pr,
        "SciPlex3_PrDelta": sci_pr,
    },
    axis=1,
)

mean_output.to_excel(
    output_mean_excel
)


# =========================================================
# 6. 全局字体
#
# 只改字体相关，不改线宽等其他样式
# =========================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# PDF中保留TrueType文字，方便Illustrator编辑
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# 数学文字尽量保持Arial
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


# =========================================================
# 7. 创建画布
#
# 原来：(18.5, 11.5)
# 现在：(18.5, 14.5)
#
# 宽度完全不动，只增加高度
# =========================================================

fig = plt.figure(
    figsize=(18.5, 14.5),
)

outer_grid = GridSpec(
    nrows=1,
    ncols=3,
    figure=fig,

    width_ratios=[1, 0.035, 1],

    wspace=0.0,
)


# 左侧R²区域
left_grid = outer_grid[0, 0].subgridspec(
    nrows=1,
    ncols=2,
    wspace=0.012,
)


# 右侧Pr Delta区域
right_grid = outer_grid[0, 2].subgridspec(
    nrows=1,
    ncols=2,
    wspace=0.012,
)


ax_nips_r2 = fig.add_subplot(
    left_grid[0, 0]
)

ax_sci_r2 = fig.add_subplot(
    left_grid[0, 1]
)

ax_nips_pr = fig.add_subplot(
    right_grid[0, 0]
)

ax_sci_pr = fig.add_subplot(
    right_grid[0, 1]
)


# =========================================================
# 8. 热力图参数
# =========================================================

color_map = plt.cm.RdBu_r.copy()
color_map.set_bad("white")

value_min = -0.15
value_max = 1.0

normalizer = plt.Normalize(
    vmin=value_min,
    vmax=value_max,
)


# =========================================================
# 9. 热力图绘制函数
# =========================================================

def draw_heatmap(
    ax,
    dataframe,
    title,
    show_y_labels=False,
):
    values = dataframe.values.astype(float)

    masked_values = np.ma.masked_invalid(
        values
    )

    ax.imshow(
        masked_values,
        cmap=color_map,
        vmin=value_min,
        vmax=value_max,

        # 保持原来的auto
        aspect="auto",

        interpolation="none",
    )


    # =====================================================
    # 数据集标题
    # 原来 26 → 30
    # =====================================================

    ax.set_title(
        title,
        fontsize=30,
        pad=16,
        fontweight="normal",
        fontfamily="Arial",
    )


    # =====================================================
    # split名称
    # 原来 22 → 25
    # =====================================================

    ax.set_xticks(
        np.arange(len(split_names))
    )

    ax.set_xticklabels(
        split_names,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=25,
        fontfamily="Arial",
    )


    # =====================================================
    # 方法名称
    # 原来 23 → 26
    # =====================================================

    ax.set_yticks(
        np.arange(len(methods))
    )

    if show_y_labels:
        ax.set_yticklabels(
            methods,
            fontsize=26,
            fontfamily="Arial",
        )
    else:
        ax.set_yticklabels([])


    ax.tick_params(
        axis="both",
        which="major",
        length=0,
        pad=7,
    )


    # =====================================================
    # 每个格子的数字
    # 原来 21 → 24
    # =====================================================

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):

            value = values[row, column]

            if np.isnan(value):
                continue

            rgba = color_map(
                normalizer(value)
            )

            luminance = (
                0.299 * rgba[0]
                + 0.587 * rgba[1]
                + 0.114 * rgba[2]
            )

            if luminance < 0.56:
                text_color = "white"
            else:
                text_color = "#282828"

            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",

                fontsize=24,
                fontfamily="Arial",

                color=text_color,
            )


    # =====================================================
    # 白色单元格边框
    # 保持原样
    # =====================================================

    ax.set_xticks(
        np.arange(
            -0.5,
            values.shape[1],
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            values.shape[0],
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.8,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )


    # 去掉坐标轴外框
    for spine in ax.spines.values():
        spine.set_visible(False)


# =========================================================
# 10. 绘制四张热力图
# =========================================================

draw_heatmap(
    ax=ax_nips_r2,
    dataframe=nips_r2,
    title="NeurIPS",
    show_y_labels=True,
)

draw_heatmap(
    ax=ax_sci_r2,
    dataframe=sci_r2,
    title="SciPlex3",
    show_y_labels=False,
)

draw_heatmap(
    ax=ax_nips_pr,
    dataframe=nips_pr,
    title="NeurIPS",
    show_y_labels=False,
)

draw_heatmap(
    ax=ax_sci_pr,
    dataframe=sci_pr,
    title="SciPlex3",
    show_y_labels=False,
)


# =========================================================
# 11. 整体边距
#
# 基本保持原来的比例
# 由于高度增加，适当利用更多纵向空间
# =========================================================

fig.subplots_adjust(
    left=0.115,
    right=0.992,
    top=0.90,
    bottom=0.20,
)


# 获取两组图中心位置
fig.canvas.draw()

left_group_center = (
    ax_nips_r2.get_position().x0
    + ax_sci_r2.get_position().x1
) / 2

right_group_center = (
    ax_nips_pr.get_position().x0
    + ax_sci_pr.get_position().x1
) / 2


# =========================================================
# 12. 底部指标标题
#
# 原来24 → 28
# =========================================================

fig.text(
    left_group_center,
    0.085,
    r"$R^2$ score on top 50 DE genes",
    ha="center",
    va="center",
    fontsize=28,
    fontfamily="Arial",
)

fig.text(
    right_group_center,
    0.085,
    r"$\mathrm{Pr}_{\Delta}$ on top 50 DE genes",
    ha="center",
    va="center",
    fontsize=28,
    fontfamily="Arial",
)


# =========================================================
# 13. 保存
# =========================================================

plt.savefig(
    output_png,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    output_pdf,
    bbox_inches="tight",
    pad_inches=0.12,
    facecolor="white",
)

plt.show()


print("\n绘图完成：")
print("PNG：", output_png)
print("PDF：", output_pdf)
print("均值表：", output_mean_excel)