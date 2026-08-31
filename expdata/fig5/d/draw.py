import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


# =====================
# 读取数据
# =====================

df = pd.read_csv("fig2h.csv")
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

methods = ["DIFFCRISP", "CRISP"]

metrics = [
    {
        "suffix": "R2_DE",
        "title": r"$R^2$ score DE ($\uparrow$)",
        "ylim": (0, 0.8),
        "yticks": [0, 0.2, 0.4, 0.6, 0.8],
        "alternative": "greater",
    },
    {
        "suffix": "Pearson_Delta_DE",
        "title": r"$\mathrm{Pr}_{\Delta}$ DE ($\uparrow$)",
        "ylim": (0.1, 0.4),
        "yticks": [0.1, 0.2,0.3,0.4],
        "alternative": "greater",
    },
    {
        "suffix": "Sinkhorn_DE",
        "title": r"Sinkhorn DE ($\downarrow$)",
        "ylim": (4, 12),
        "yticks": [4, 6, 8, 10, 12],
        "alternative": "less",
    },
]

colors = {
    "DIFFCRISP": "#ef6351",
    "CRISP": "#d4ccc3",
}


# =====================
# 统计函数
# =====================

def get_values(method, suffix):
    return pd.to_numeric(
        df[f"{method}_{suffix}"],
        errors="coerce",
    ).to_numpy()


def get_stats(values):
    values = values[~np.isnan(values)]

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    ci95 = 1.96 * std / np.sqrt(len(values))

    return values, mean, ci95


def paired_p(x, y, alternative):
    mask = ~np.isnan(x) & ~np.isnan(y)

    result = ttest_rel(
        x[mask],
        y[mask],
        alternative=alternative,
    )

    return result.pvalue


def format_p(p):
    if p < 0.0001:
        return r"$P < 0.0001$"

    return rf"$P = {p:.4f}$"


def add_p_value(ax, x1, x2, text):
    transform = ax.get_xaxis_transform()

    # 使用坐标轴比例，三个子图位置一致
    line_y = 1.02
    text_y = 1.04

    # 只画水平线，没有两端触须
    ax.plot(
        [x1, x2],
        [line_y, line_y],
        color="red",
        linewidth=0.6,
        transform=transform,
        clip_on=False,
    )

    ax.text(
        (x1 + x2) / 2,
        text_y,
        text,
        color="red",
        fontsize=8,
        ha="center",
        va="bottom",
        transform=transform,
    )


# =====================
# 画图
# =====================

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# 统一使用 Arial
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

# 避免数学文字使用另一套默认字体
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"

fig, axes = plt.subplots(
    1,
    3,
    figsize=(6.2, 3.1),
)

plt.subplots_adjust(
    left=0.09,
    right=0.98,
    top=0.82,
    bottom=0.24,
    wspace=0.52,
)

# 两根柱子的距离
x = np.array([0.0, 0.32])
bar_width = 0.18
rng = np.random.default_rng(42)


for ax, info in zip(axes, metrics):

    suffix = info["suffix"]

    values_list = []
    means = []
    cis = []

    for method in methods:
        values, mean, ci95 = get_stats(
            get_values(method, suffix)
        )

        values_list.append(values)
        means.append(mean)
        cis.append(ci95)

        print(
            suffix,
            method,
            f"mean={mean:.6f}",
            f"ci95={ci95:.6f}",
        )

    # 柱子，不使用Matplotlib自带误差棒
    ax.bar(
        x,
        means,
        width=bar_width,
        color=[colors[m] for m in methods],
        edgecolor="#333333",
        linewidth=0.4,
        zorder=2,
    )

    # 手动画仅向上的误差棒
    cap_width = 0.035

    for xi, mean, ci95 in zip(x, means, cis):

        # 只从均值向上画
        ax.plot(
            [xi, xi],
            [mean, mean + ci95],
            color="#444444",
            linewidth=0.6,
            zorder=4,
        )

        # 只保留顶部横帽
        ax.plot(
            [
                xi - cap_width,
                xi + cap_width,
            ],
            [
                mean + ci95,
                mean + ci95,
            ],
            color="#444444",
            linewidth=0.6,
            zorder=4,
        )

    # 原始点
    for i, values in enumerate(values_list):

        jitter = rng.normal(
            0,
            0.016,
            size=len(values),
        )

        ax.scatter(
            np.full(len(values), x[i]) + jitter,
            values,
            s=2,
            color="black",
            alpha=0.85,
            linewidths=0,
            zorder=3,
        )

    # 配对单尾t检验
    raw_x = get_values(methods[0], suffix)
    raw_y = get_values(methods[1], suffix)

    p_value = paired_p(
        raw_x,
        raw_y,
        info["alternative"],
    )

    add_p_value(
        ax,
        x[0],
        x[1],
        format_p(p_value),
    )

    ax.set_title(
        info["title"],
        fontsize=13,
        pad=20,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        methods,
        rotation=45,
        ha="center",
        fontsize=9,
    )


    ax.set_xlim(-0.16, 0.48)
    ax.set_ylim(*info["ylim"])
    ax.set_yticks(info["yticks"])

    # 删除横坐标下方的小竖线
    ax.tick_params(
        axis="x",
        bottom=False,
        top=False,
        length=0,
    )

    ax.tick_params(
        axis="y",
        labelsize=10,
        width=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


fig.text(
    0.01,
    0.965,
    "h",
    fontsize=18,
    fontweight="bold",
    va="top",
)

plt.savefig(
    "fig2h.pdf",
    bbox_inches="tight",
)


plt.show()