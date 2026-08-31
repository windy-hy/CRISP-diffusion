import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


FILE = "fig3c.xlsx"
OUTPUT_PDF = "fig3c.pdf"

METHODS = {
    "DiffCRISP": [1, 2, 3],       # B-D
    "Without L1 loss": [4, 5, 6], # E-G
    "Without CFG": [7, 8, 9],     # H-J
}

COLORS = {
    "DiffCRISP": "#4778B7",
    "Without L1 loss": "#C3AE82",
    "Without CFG": "#78B5C4",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 10,

    # 数学公式也统一 Arial
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",

    "axes.linewidth": 0.8,

    # PDF中保持可编辑字体
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================
# 读取Excel
# =========================

raw = pd.read_excel(FILE, header=None)

# 自动找到split、split2、split3所在行
mask = raw.iloc[:, 0].astype(str).str.fullmatch(
    r"split\d*",
    case=False,
    na=False,
)

data = raw.loc[mask].reset_index(drop=True)
splits = data.iloc[:, 0].astype(str).tolist()

if len(data) == 0:
    raise ValueError("没有找到split数据，请检查Excel第一列。")


# =========================
# 绘图
# =========================

fig, ax = plt.subplots(figsize=(5.25, 3.45))

x = np.arange(len(splits))
bar_width = 0.22
offsets = [-bar_width, 0, bar_width]
markers = ["o", "^", "D"]

summary = []

for method_index, (method, columns) in enumerate(METHODS.items()):

    values = (
        data.iloc[:, columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    means = np.nanmean(values, axis=1)
    stds = np.nanstd(values, axis=1, ddof=1)
    positions = x + offsets[method_index]

    ax.bar(
        positions,
        means,
        width=bar_width * 0.82,
        color=COLORS[method],
        edgecolor="#555555",
        linewidth=0.6,
        zorder=2,
    )

    ax.errorbar(
        positions,
        means,
        yerr=stds,
        fmt="none",
        ecolor="#666666",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        zorder=4,
    )

    # 三次重复分别显示
    point_offsets = np.linspace(-0.035, 0.035, values.shape[1])

    for split_index in range(len(splits)):
        for repeat_index, value in enumerate(values[split_index]):

            if np.isnan(value):
                continue

            ax.scatter(
                positions[split_index] + point_offsets[repeat_index],
                value,
                s=6,
                marker=markers[repeat_index],
                facecolor=COLORS[method],
                edgecolor="#555555",
                linewidth=0.5,
                zorder=5,
            )

        summary.append({
            "split": splits[split_index],
            "method": method,
            "mean": means[split_index],
            "std": stds[split_index],
        })


# =========================
# 坐标轴
# =========================

ax.set_xticks(x)
ax.set_xticklabels(
    ["Split1", "Split2", "Split3"]
)

ax.set_ylim(0.50, 0.86)
ax.set_yticks(np.arange(0.5, 0.81, 0.1))

ax.set_title(
    r"$\mathrm{Pr}_{\Delta}$ DE (NeurIPS)",
    fontsize=11,
    pad=8,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="both",
    direction="out",
    length=4,
)


# =========================
# 图例
# =========================

handles = [
    Patch(
        facecolor=COLORS[method],
        edgecolor="#555555",
        linewidth=0.6,
        label=method,
    )
    for method in METHODS
]

fig.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.53, 0.015),
    ncol=3,
    frameon=False,
    handlelength=0.9,
    handleheight=0.9,
    handletextpad=0.4,
    columnspacing=1.2,
)


# 面板编号
ax.text(
    -0.14,
    1.08,
    "c",
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
    right=0.98,
    top=0.87,
    bottom=0.25,
)

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    pad_inches=0.03,
)


plt.show()


# 输出均值和标准差
summary_df = pd.DataFrame(summary)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda v: f"{v:.4f}",
    )
)