import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from matplotlib.lines import Line2D


# =========================
# 文件路径
# =========================

CRISP_FILE = "figure4f_CRISP.csv"
DIFFCRISP_FILE = "figure4f_diffcrisp.csv"

OUTPUT_PDF = "figure4f.pdf"
OUTPUT_PNG = "figure4f.png"


# =========================
# Arial字体
# =========================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]

plt.rcParams["font.size"] = 10

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# 保持你原来的线宽设置
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8


# =========================
# 工具函数
# =========================

def prepare(df):
    df = df.copy()

    # 防止de列被读成字符串
    if df["de"].dtype != bool:
        df["de"] = (
            df["de"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
        )

    df = df[
        np.isfinite(df["gt"])
        & np.isfinite(df["pred"])
        & df["de"].notna()
    ].copy()

    df["de"] = df["de"].astype(bool)

    return df


def get_metrics(df):
    de_df = df[df["de"]].copy()

    if len(de_df) < 2:
        return np.nan, np.nan

    pr = pearsonr(
        de_df["gt"],
        de_df["pred"]
    )[0]

    sp = spearmanr(
        de_df["gt"],
        de_df["pred"]
    )[0]

    return pr, sp


# =========================
# 读取数据
# =========================

crisp = pd.read_csv(CRISP_FILE)
diffcrisp = pd.read_csv(DIFFCRISP_FILE)

crisp = prepare(crisp)
diffcrisp = prepare(diffcrisp)

crisp_pr, crisp_sp = get_metrics(crisp)
diff_pr, diff_sp = get_metrics(diffcrisp)


print("CRISP")
print("总基因数：", len(crisp))
print("DE基因数：", int(crisp["de"].sum()))
print("Pr_DE：", crisp_pr)
print("Sp_DE：", crisp_sp)

print("\nDiffCRISP")
print("总基因数：", len(diffcrisp))
print("DE基因数：", int(diffcrisp["de"].sum()))
print("Pr_DE：", diff_pr)
print("Sp_DE：", diff_sp)


# =========================
# 画图
# =========================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.2, 3.5),
    sharex=True,
    sharey=True,
)


datasets = [
    (crisp, "CRISP", crisp_pr, crisp_sp),
    (diffcrisp, "DiffCRISP", diff_pr, diff_sp),
]


# 颜色
other_color = "#A8CFE5"
de_color = "#1F77B4"
zero_color = "#F28E2B"


# 固定坐标范围
xlim = (-2.5, 4.5)
ylim = (-2.5, 6.5)

xticks = [-2, -1, 0, 1, 2, 3, 4]
yticks = [-2, 0, 2, 4, 6]


# =========================
# 开始绘图
# =========================

for ax, (df, title, pr, sp) in zip(
    axes,
    datasets
):

    others = df[~df["de"]]
    de = df[df["de"]]


    # =====================
    # Others
    # 只把散点栅格化
    # =====================

    ax.scatter(
        others["gt"],
        others["pred"],
        s=16,
        facecolors=other_color,
        edgecolors="white",
        linewidths=0.25,
        alpha=0.95,
        zorder=2,

        # 关键：散点转位图
        rasterized=True,
    )


    # =====================
    # Top 50 DE genes
    # 同样栅格化
    # =====================

    ax.scatter(
        de["gt"],
        de["pred"],
        s=18,
        facecolors=de_color,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.98,
        zorder=3,

        # 关键：散点转位图
        rasterized=True,
    )


    # =====================
    # 0轴虚线
    # 保持矢量
    # =====================

    ax.plot(
        [xlim[0], xlim[1]],
        [0, 0],
        color=zero_color,
        linestyle="--",
        linewidth=1.0,
        dashes=(5, 4),
        alpha=0.95,
        zorder=10,
    )

    ax.plot(
        [0, 0],
        [ylim[0], ylim[1]],
        color=zero_color,
        linestyle="--",
        linewidth=1.0,
        dashes=(5, 4),
        alpha=0.95,
        zorder=10,
    )


    # =====================
    # 标题
    # =====================

    ax.set_title(
        title,
        fontsize=11,
        pad=22,
        fontfamily="Arial",
    )


    # =====================
    # 坐标范围和刻度
    # =====================

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)


    ax.tick_params(
        axis="both",
        labelsize=8,
        direction="out",
        length=3,
        width=0.7,
    )


    # =====================
    # 坐标轴样式
    # =====================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


    # =====================
    # 相关系数
    #
    # 不再使用mathtext
    # 确保文字全部走Arial
    # =====================

    text = (
        f"PrΔ DE: {pr:.3f}"
        "\n"
        f"SpΔ DE: {sp:.3f}"
    )


    ax.text(
        0.58,
        0.08,
        text,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="Arial",
        ha="left",
        va="bottom",
    )


# =========================
# 坐标标题
# =========================

axes[0].set_ylabel(
    "Dactolisib-NK cells-10x (v3)\n"
    "Pred. log(FC) (PBMC-Bench)",
    fontsize=9,
    fontfamily="Arial",
)


for ax in axes:
    ax.set_xlabel(
        "True log(FC) (NeurIPS)",
        fontsize=9,
        fontfamily="Arial",
    )


# 右图也显示纵轴刻度数字
axes[1].tick_params(
    labelleft=True
)


# =========================
# 自定义图例
# =========================

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=de_color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=8,
        label="Top 50 DE genes",
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=other_color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=8,
        label="Others",
    ),
]


axes[1].legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.44, 1.08),
    frameon=False,
    fontsize=8.5,
    handletextpad=0.5,
    borderaxespad=0,
)


# =========================
# 面板字母
# =========================

fig.text(
    0.015,
    0.965,
    "f",
    fontsize=13,
    fontweight="bold",
    fontfamily="Arial",
    ha="left",
    va="top",
)


# =========================
# 布局
# =========================

plt.subplots_adjust(
    left=0.14,
    right=0.98,
    bottom=0.18,
    top=0.82,
    wspace=0.22,
)


# =========================
# 保存
#
# dpi=600只影响rasterized散点
# 文字/坐标轴/虚线仍然是矢量
# =========================

plt.savefig(
    OUTPUT_PDF,
    format="pdf",
    dpi=600,
    bbox_inches="tight",
)


plt.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight",
)


plt.show()