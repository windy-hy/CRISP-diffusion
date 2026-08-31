import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from matplotlib.lines import Line2D


# =========================================================
# 字体：统一 Arial
# =========================================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]

# PDF 中字体保持 TrueType，方便 Illustrator 编辑
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# 数学符号也尽量使用 Arial
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


# =========================
# 1. 你的6个文件所在文件夹
# =========================
data_dir = "./"


# =========================
# 2. 文件名
# =========================
file_map = {
    ("Palbociclib", "DIFFCRISP"): "DIFFCRISP_Palbociclib_Bcells.csv",
    ("Palbociclib", "CRISP"): "CRISP_Palbociclib_Bcells.csv",
    ("Palbociclib", "ChemCPA"): "chemcpa_Palbociclib_Bcells.csv",

    ("Idelalisib", "DIFFCRISP"): "DIFFCRISP_Idelalisib_Bcells.csv",
    ("Idelalisib", "CRISP"): "CRISP_Idelalisib_Bcells.csv",
    ("Idelalisib", "ChemCPA"): "chemcpa_Idelalisib_Bcells.csv",
}

drugs = ["Palbociclib", "Idelalisib"]
methods = ["DIFFCRISP", "CRISP", "ChemCPA"]


# =========================
# 3. 读csv
# =========================
def load_csv(path):
    df = pd.read_csv(path)

    df["gt"] = pd.to_numeric(
        df["gt"],
        errors="coerce"
    )

    df["pred"] = pd.to_numeric(
        df["pred"],
        errors="coerce"
    )

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
                "0": False
            })
        )

    df = df.dropna(
        subset=["gt", "pred", "de"]
    ).copy()

    df["de"] = df["de"].astype(bool)

    return df


# =========================
# 4. 先把所有数据读进来
# =========================
all_data = {}
all_values = []

for drug in drugs:
    for method in methods:

        path = os.path.join(
            data_dir,
            file_map[(drug, method)]
        )

        df = load_csv(path)

        all_data[(drug, method)] = df

        all_values.extend(
            df["gt"].tolist()
        )

        all_values.extend(
            df["pred"].tolist()
        )


# =========================
# 坐标范围统一
# =========================
vmin = min(all_values)
vmax = max(all_values)

limit = max(
    abs(vmin),
    abs(vmax)
)

limit = max(
    limit,
    2.8
)

limit = np.ceil(
    limit * 10
) / 10.0

xlim = (-2.8, 2.8)
ylim = (-2.8, 2.8)


# =========================
# 5. 画图
# =========================
fig, axes = plt.subplots(
    2,
    3,
    figsize=(10.8, 6.2)
)

plt.subplots_adjust(
    wspace=0.28,
    hspace=0.15
)


light_blue = "#8fbcd4"
dark_blue = "#0b6aa9"
line_orange = "#ff8c1a"


for i, drug in enumerate(drugs):

    for j, method in enumerate(methods):

        ax = axes[i, j]

        df = all_data[
            (drug, method)
        ]

        de_df = df[
            df["de"]
        ].copy()

        other_df = df[
            ~df["de"]
        ].copy()


        # =================================================
        # Others
        #
        # rasterized=True：
        # 只把散点转成位图
        # =================================================
        ax.scatter(
            other_df["gt"],
            other_df["pred"],

            s=12,
            c=light_blue,

            edgecolors="white",
            linewidths=0.25,

            alpha=0.9,
            zorder=2,

            # 关键
            rasterized=True,
        )


        # =================================================
        # Top 50 DE
        # 同样只栅格化散点
        # =================================================
        ax.scatter(
            de_df["gt"],
            de_df["pred"],

            s=14,
            c=dark_blue,

            edgecolors="white",
            linewidths=0.3,

            alpha=0.95,
            zorder=3,

            # 关键
            rasterized=True,
        )


        # =================================================
        # 对角虚线
        # 不栅格化，继续保持矢量
        # =================================================
        ax.plot(
            [xlim[0], xlim[1]],
            [ylim[0], ylim[1]],

            color=line_orange,
            linestyle="--",

            linewidth=1.0,
            dashes=(5, 4),

            alpha=0.9,
            zorder=10,
        )


        # =================================================
        # 计算指标
        # =================================================
        if len(de_df) > 1:

            pr_de = pearsonr(
                de_df["gt"],
                de_df["pred"]
            )[0]

            sp_de = spearmanr(
                de_df["gt"],
                de_df["pred"]
            )[0]

        else:

            pr_de = np.nan
            sp_de = np.nan


        # =================================================
        # 指标文字
        # 保持矢量 Arial
        # =================================================
        ax.text(
            0.05,
            0.94,

            r"$\mathrm{Pr}_{\Delta}$ DE: "
            + f"{pr_de:.3f}\n"
            + f"Sp DE: {sp_de:.3f}",

            transform=ax.transAxes,

            ha="left",
            va="top",

            fontsize=10,
            fontfamily="Arial"
        )


        # =================================================
        # 方法标题
        # =================================================
        if i == 0:

            ax.set_title(
                method,
                fontsize=15,
                pad=8,
                fontfamily="Arial"
            )

        else:

            ax.set_title("")


        # =================================================
        # 坐标范围
        # =================================================
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_box_aspect(1)


        ax.set_xticks(
            [-2, 0, 2]
        )

        ax.set_yticks(
            [-2, 0, 2]
        )


        ax.tick_params(
            axis="both",
            labelsize=11,
            length=0
        )


        # =================================================
        # 坐标轴样式
        # 保持你原来的设置
        # =================================================
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["bottom"].set_linewidth(1.1)
        ax.spines["left"].set_linewidth(1.1)

        ax.spines["bottom"].set_color(
            "#555555"
        )

        ax.spines["left"].set_color(
            "#555555"
        )


        # =================================================
        # y 标签
        # =================================================
        if j == 0:

            ax.set_ylabel(
                f"{drug}\npred. log(FC)",
                fontsize=13,
                fontfamily="Arial"
            )

        else:

            ax.set_ylabel("")


        # =================================================
        # x 标签
        # =================================================
        if i == 1:

            ax.set_xlabel(
                "True log(FC)",
                fontsize=13,
                fontfamily="Arial"
            )

        else:

            ax.set_xlabel("")


# =========================================================
# 图例
# =========================================================
legend_handles = [

    Line2D(
        [0],
        [0],

        marker="o",
        color="none",

        markerfacecolor=dark_blue,
        markeredgecolor="white",
        markeredgewidth=0.4,

        markersize=8,

        label="Top 50 DE"
    ),

    Line2D(
        [0],
        [0],

        marker="o",
        color="none",

        markerfacecolor=light_blue,
        markeredgecolor="white",
        markeredgewidth=0.4,

        markersize=8,

        label="Others"
    ),
]


axes[0, 2].legend(
    handles=legend_handles,

    title="Genes",

    frameon=False,

    fontsize=11,
    title_fontsize=12,

    loc="upper left",

    bbox_to_anchor=(
        1.02,
        1.02
    )
)


# =========================================================
# panel label
# =========================================================
fig.text(
    0.015,
    0.98,

    "d",

    fontsize=18,
    fontweight="bold",
    fontfamily="Arial",

    va="top"
)


# =========================================================
# 保存 PDF
#
# dpi=600：
# 只控制 rasterized=True 的散点分辨率
#
# 文字、坐标轴、虚线仍然是矢量
# =========================================================
plt.savefig(
    "fig2d.pdf",

    format="pdf",

    dpi=600,

    bbox_inches="tight"
)


plt.show()