import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# =========================================================
# 只改字体
# 不改 axes linewidth / tick / lines / figsize 等任何样式
# =========================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11

# PDF 中字体保持 TrueType，方便 Illustrator
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# =========================================================
# 文件
# =========================================================

CRISP_FILE = "crisp_distribution_metrics.csv"
DIFF_FILE = "diffcrisp_distribution_metrics.csv"
OUTPUT = "distribution_comparison.pdf"

SPLITS = ["split", "split2", "split3"]

METRICS = {
    "sinkhorn_de": "Sinkhorn distance",
    "centered_sinkhorn_de": "Centered Sinkhorn",
    "variance_log_mae_de": "Variance log-MAE"
}


# =========================================================
# 读取数据
# =========================================================

crisp = pd.read_csv(CRISP_FILE)
diff = pd.read_csv(DIFF_FILE)

crisp = crisp[crisp["row_type"] == "group"]
diff = diff[diff["row_type"] == "group"]


# =========================================================
# 按 split 和扰动组合配对
# =========================================================

data = crisp.merge(
    diff,
    on=["split", "group"],
    suffixes=("_crisp", "_diff")
)

print("成功配对：")
print(data.groupby("split").size())


# =========================================================
# 保持你原来的画布尺寸
# =========================================================

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 9)
)

x = np.arange(len(SPLITS))
width = 0.35


# =========================================================
# 绘图
# =========================================================

for col, (metric, title) in enumerate(METRICS.items()):

    # =====================================================
    # 上排：split均值 ± SEM
    # =====================================================

    crisp_mean = []
    crisp_sem = []

    diff_mean = []
    diff_sem = []

    for split in SPLITS:

        d = data[data["split"] == split]

        crisp_mean.append(
            d[f"{metric}_crisp"].mean()
        )

        crisp_sem.append(
            d[f"{metric}_crisp"].sem()
        )

        diff_mean.append(
            d[f"{metric}_diff"].mean()
        )

        diff_sem.append(
            d[f"{metric}_diff"].sem()
        )


    ax = axes[0, col]


    ax.bar(
        x - width / 2,
        crisp_mean,
        width,
        yerr=crisp_sem,
        capsize=4,
        error_kw={
            "elinewidth": 0.6,  # 误差棒竖线粗细
            "capthick": 0.6  # 上下横线粗细
        },
        label="CRISP"
    )


    ax.bar(
        x + width / 2,
        diff_mean,
        width,
        yerr=diff_sem,
        capsize=4,
        error_kw={
            "elinewidth": 0.6,  # 误差棒竖线粗细
            "capthick": 0.6  # 上下横线粗细
        },
        label="DiffCRISP"
    )


    ax.set_xticks(x)

    ax.set_xticklabels(
        SPLITS
    )

    ax.set_title(
        title
    )

    ax.set_ylabel(
        "Mean ± SEM"
    )


    if col == 0:
        ax.legend(
            frameon=False
        )


    # =====================================================
    # 下排：每个组合配对散点
    # =====================================================

    ax = axes[1, col]


    for split in SPLITS:

        d = data[
            data["split"] == split
        ]


        x_plot = np.log1p(
            d[f"{metric}_crisp"]
        )

        y_plot = np.log1p(
            d[f"{metric}_diff"]
        )


        ax.scatter(
            x_plot,
            y_plot,
            s=18,
            alpha=0.5,
            label=split
        )


    # =====================================================
    # 原始数值用于胜率和统计检验
    # =====================================================

    pair = data[
        [
            f"{metric}_crisp",
            f"{metric}_diff"
        ]
    ].dropna()


    crisp_value = pair[
        f"{metric}_crisp"
    ].to_numpy()


    diff_value = pair[
        f"{metric}_diff"
    ].to_numpy()


    # =====================================================
    # log1p 后的坐标范围
    # =====================================================

    crisp_plot = np.log1p(
        crisp_value
    )

    diff_plot = np.log1p(
        diff_value
    )


    lower = min(
        crisp_plot.min(),
        diff_plot.min()
    )

    upper = max(
        crisp_plot.max(),
        diff_plot.max()
    )


    padding = (
        upper - lower
    ) * 0.05


    # =====================================================
    # y=x 对角线
    # =====================================================

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1
    )


    ax.set_xlim(
        lower - padding,
        upper + padding
    )

    ax.set_ylim(
        lower - padding,
        upper + padding
    )


    # =====================================================
    # 胜率
    # =====================================================

    win_rate = np.mean(
        diff_value < crisp_value
    ) * 100


    try:

        p_value = wilcoxon(
            diff_value,
            crisp_value
        ).pvalue

    except ValueError:

        p_value = np.nan


    ax.set_title(
        f"Win rate = {win_rate:.1f}%   "
        f"p = {p_value:.2e}"
    )


    ax.set_xlabel(
        "log(1 + CRISP)"
    )

    ax.set_ylabel(
        "log(1 + DiffCRISP)"
    )


    if col == 0:

        ax.legend(
            frameon=False
        )


# =========================================================
# 总标题
# 保持你原来的 fontsize=16
# =========================================================

fig.suptitle(
    "CRISP vs DiffCRISP Distribution Recovery",
    fontsize=16
)


# =========================================================
# 原来的布局
# =========================================================

# 只调整6个子图的坐标轴边框粗细
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

for ax in axes.flat:
    ax.tick_params(width=0.5)

plt.tight_layout()


# =========================================================
# 只压矮上面三个柱状图
# 下排三个散点图完全不动
# =========================================================

shrink_top = 0.90

for ax in axes[0, :]:

    pos = ax.get_position()

    new_height = (
        pos.height * shrink_top
    )

    ax.set_position([
        pos.x0,
        pos.y0 + (
            pos.height - new_height
        ),
        pos.width,
        new_height
    ])


# =========================================================
# 保存 PDF
# =========================================================

plt.savefig(
    OUTPUT,
    format="pdf",
    bbox_inches="tight"
)

plt.show()

print(
    "已保存：",
    OUTPUT
)


# =========================================================
# 输出统计
# =========================================================

for metric, title in METRICS.items():

    pair = data[
        [
            f"{metric}_crisp",
            f"{metric}_diff"
        ]
    ].dropna()


    win = (
        pair[f"{metric}_diff"]
        <
        pair[f"{metric}_crisp"]
    ).mean()


    print(
        f"{title}: "
        f"CRISP="
        f"{pair[f'{metric}_crisp'].mean():.4f}, "
        f"DiffCRISP="
        f"{pair[f'{metric}_diff'].mean():.4f}, "
        f"胜率={win:.2%}"
    )