import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t


CRISP_FILE = "fig3f_crisp.csv"
DIFFCRISP_FILE = "fig3f_diffcrisp.csv"
MMD_FILE = "fig3f_mmd.csv"

OUTPUT_PDF = "fig3f.pdf"
# OUTPUT_PNG = "fig3f.png"

DOSES = [0.01, 0.1, 1, 10]

COLORS = {
    "DIFFCRISP": "#4778B7",
    "CRISP": "#A99BE3",
    "MMD": "#8A8A8A"
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


def convert_dose(series):
    series = pd.to_numeric(series, errors="coerce")

    # Pearson文件为0.001、0.01、0.1、1，转换成0.01、0.1、1、10
    if series.max() <= 1.01 and series.min() < 0.005:
        series = series * 10

    return series.round(6)


def summarize(df, value_col, groups):
    result = (
        df.groupby(groups)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    result["ci95"] = result.apply(
        lambda row: (
            t.ppf(0.975, row["count"] - 1)
            * row["std"]
            / np.sqrt(row["count"])
            if row["count"] > 1
            else 0
        ),
        axis=1
    )

    return result


# =========================
# 读取两个模型的Pearson数据
# =========================

crisp = pd.read_csv(
    CRISP_FILE,
    encoding="utf-8-sig"
)

diffcrisp = pd.read_csv(
    DIFFCRISP_FILE,
    encoding="utf-8-sig"
)

crisp.columns = crisp.columns.str.strip()
diffcrisp.columns = diffcrisp.columns.str.strip()

crisp["method"] = "CRISP"
diffcrisp["method"] = "DIFFCRISP"

pearson = pd.concat(
    [diffcrisp, crisp],
    ignore_index=True
)

pearson["dose"] = convert_dose(
    pearson["dose"]
)

pearson["pearson_delta_de"] = pd.to_numeric(
    pearson["pearson_delta_de"],
    errors="coerce"
)

pearson = pearson.dropna(
    subset=[
        "method",
        "dose",
        "pearson_delta_de"
    ]
)

pearson = pearson[
    pearson["dose"].isin(DOSES)
].copy()


# =========================
# 读取MMD数据
# =========================

mmd = pd.read_csv(
    MMD_FILE,
    encoding="utf-8-sig"
)

mmd.columns = mmd.columns.str.strip()

mmd["dose"] = convert_dose(
    mmd["dose"]
)

mmd["mmd_de"] = pd.to_numeric(
    mmd["mmd_de"],
    errors="coerce"
)

mmd = mmd.dropna(
    subset=[
        "cell_type",
        "drug",
        "dose",
        "mmd_de"
    ]
)

mmd = mmd[
    mmd["dose"].isin(DOSES)
].copy()

# MMD与模型方法无关，同一个组合只保留一次
mmd = mmd.drop_duplicates(
    subset=[
        "cell_type",
        "drug",
        "dose"
    ]
)


# =========================
# 计算均值和95%置信区间
# =========================

pearson_summary = summarize(
    pearson,
    "pearson_delta_de",
    ["method", "dose"]
)

mmd_summary = summarize(
    mmd,
    "mmd_de",
    ["dose"]
)


# =========================
# 检查每个剂量的数据量
# =========================

print("\nPearson每个剂量的数据数量：")
print(
    pearson.groupby(
        ["method", "dose"]
    ).size()
)

print("\nMMD每个剂量的数据数量：")
print(
    mmd.groupby("dose").size()
)

print("\nPearson统计结果：")
print(
    pearson_summary.to_string(index=False)
)

print("\nMMD统计结果：")
print(
    mmd_summary.to_string(index=False)
)


# =========================
# 绘图
# =========================

fig, ax1 = plt.subplots(
    figsize=(3.55, 3.55)
)

x = np.arange(len(DOSES))

settings = {
    "DIFFCRISP": {
        "marker": "o",
        "label": "DiffCRISP"
    },
    "CRISP": {
        "marker": "s",
        "label": "CRISP"
    }
}

for method in ["DIFFCRISP", "CRISP"]:

    data = (
        pearson_summary[
            pearson_summary["method"] == method
        ]
        .set_index("dose")
        .reindex(DOSES)
    )

    ax1.errorbar(
        x,
        data["mean"],
        yerr=data["ci95"],
        marker=settings[method]["marker"],
        markersize=5.5,
        markerfacecolor=COLORS[method],
        markeredgecolor=COLORS[method],
        color=COLORS[method],
        linewidth=0.9,
        elinewidth=0.8,
        capsize=0,
        label=settings[method]["label"],
        zorder=3
    )


# =========================
# 左侧坐标轴
# =========================

ax1.set_xlim(-0.35, 3.35)
ax1.set_ylim(0, 0.85)

ax1.set_xticks(x)
ax1.set_xticklabels(
    ["0.01", "0.1", "1", "10"]
)

ax1.set_xlabel(
    r"Dosage ($\mathrm{\mu M}$)",
    fontsize=11
)

ax1.set_ylabel(
    r"$\mathrm{Pr}_{\Delta}$ DE",
    fontsize=11
)

ax1.spines["top"].set_visible(False)

ax1.tick_params(
    axis="both",
    direction="out",
    length=4
)


# =========================
# 右侧MMD坐标轴
# =========================

ax2 = ax1.twinx()

mmd_plot = (
    mmd_summary
    .set_index("dose")
    .reindex(DOSES)
)

ax2.errorbar(
    x,
    mmd_plot["mean"],
    yerr=mmd_plot["ci95"],
    marker="x",
    markersize=7,
    markeredgewidth=0.9,
    color=COLORS["MMD"],
    linewidth=0.8,
    linestyle="--",
    elinewidth=0.8,
    capsize=0,
    label=r"MMD DE",
    zorder=2
)

mmd_upper = np.nanmax(
    mmd_plot["mean"]
    + mmd_plot["ci95"]
)

ax2.set_ylim(
    0,
    mmd_upper * 1.25
)

ax2.set_ylabel(
    r"MMD DE",
    fontsize=11,
    rotation=270,
    labelpad=18
)

ax2.spines["top"].set_visible(False)

ax2.tick_params(
    axis="y",
    direction="out",
    length=4
)


# =========================
# 图例
# =========================

legend_handles = [
    Line2D(
        [0],
        [0],
        color=COLORS["DIFFCRISP"],
        marker="o",
        markersize=5.5,
        linewidth=0.9,
        label="DiffCRISP"
    ),
    Line2D(
        [0],
        [0],
        color=COLORS["CRISP"],
        marker="s",
        markersize=5.5,
        linewidth=0.9,
        label="CRISP"
    ),
    Line2D(
        [0],
        [0],
        color=COLORS["MMD"],
        marker="x",
        markersize=7,
        linewidth=0.8,
        linestyle="--",
        label=r"MMD$_{\mathrm{DE}}$"
    )
]

ax1.legend(
    handles=legend_handles,
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(-0.20, 1.02),
    ncol=2,
    handlelength=1.7,
    handletextpad=0.4,
    columnspacing=1.2,
    borderaxespad=0
)


# =========================
# 面板编号
# =========================

ax1.text(
    -0.31,
    1.20,
    "f",
    transform=ax1.transAxes,
    fontsize=14,
    fontweight="bold",
    va="top"
)


# =========================
# 保存
# =========================

fig.subplots_adjust(
    left=0.22,
    right=0.79,
    bottom=0.17,
    top=0.78
)

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    pad_inches=0.03
)
#

plt.show()