import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


# =====================
# 读取数据
# =====================

df = pd.read_csv("fig2h.csv")
df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()

# ---------------------------------------------------------
# 关键修改：
# fig2h.csv 当前是 3 splits × 36 drug-dose = 108 行。
# 这里按 condition + dose_val 再跨 split 求均值，
# 最终每个方法只保留 36 个 drug-dose 点。
# ---------------------------------------------------------

metric_cols = [
    "DIFFCRISP_R2_DE",
    "CRISP_R2_DE",
    "ChemCPA_R2_DE",

    "DIFFCRISP_Pearson_Delta_DE",
    "CRISP_Pearson_Delta_DE",
    "ChemCPA_Pearson_Delta_DE",

    "DIFFCRISP_Sinkhorn_DE",
    "CRISP_Sinkhorn_DE",
    "ChemCPA_Sinkhorn_DE",
]

for col in metric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

df["condition"] = df["condition"].astype(str).str.strip()
df["dose_val"] = pd.to_numeric(
    df["dose_val"],
    errors="coerce",
).round(6)

df36 = (
    df
    .groupby(
        ["condition", "dose_val"],
        as_index=False,
    )[metric_cols]
    .mean()
    .sort_values(
        ["condition", "dose_val"]
    )
    .reset_index(drop=True)
)

print("原始 fig2h.csv 行数：", len(df))
print("按 drug-dose 跨 split 求均值后行数：", len(df36))

if len(df36) != 36:
    print(
        f"[警告] 预期 36 个 drug-dose 组合，实际得到 {len(df36)} 个。"
    )


# =====================
# 方法和指标
# =====================

methods = [
    "DIFFCRISP",
    "CRISP",
    "ChemCPA",
]

metrics = [
    {
        "suffix": "R2_DE",
        "title": r"$R^2$ score DE ($\uparrow$)",
        "ylim": (0.0, 0.8),
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8],
        "alternative": "greater",
    },
    {
        "suffix": "Pearson_Delta_DE",
        "title": r"$\mathrm{Pr}_{\Delta}$ DE ($\uparrow$)",
        # 按你的要求，从 0 开始，不画负区间
        "ylim": (0.0, 0.8),
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8],
        "alternative": "greater",
    },
    {
        "suffix": "Sinkhorn_DE",
        "title": r"Sinkhorn DE ($\downarrow$)",
        # 同样从 0 开始
        "ylim": (0.0, 8.0),
        "yticks": [0, 2, 4, 6, 8],
        "alternative": "less",
    },
]

colors = {
    "DIFFCRISP": "#ef6351",
    "CRISP": "#d4ccc3",
    "ChemCPA": "#7ea6d8",
}


# =====================
# 统计函数
# =====================

def get_values(method, suffix):
    col = f"{method}_{suffix}"

    if col not in df36.columns:
        raise KeyError(
            f"找不到列: {col}\n"
            f"当前列: {list(df36.columns)}"
        )

    return pd.to_numeric(
        df36[col],
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
    if np.isnan(p):
        return "P = NA"

    if p < 0.0001:
        return "P < 0.0001"

    return f"P = {p:.4f}"


# =====================
# 画图设置
# =====================

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.0, 4.0),
)

plt.subplots_adjust(
    left=0.075,
    right=0.985,
    top=0.82,
    bottom=0.24,
    wspace=0.42,
)

# 三根柱
x = np.array([
    0.00,
    0.34,
    0.68,
])

bar_width = 0.20
rng = np.random.default_rng(42)


for ax, info in zip(
    axes,
    metrics,
):

    suffix = info["suffix"]

    values_list = []
    means = []
    cis = []

    # ---------------------
    # 均值 + 95% CI
    # ---------------------
    for method in methods:

        values, mean, ci95 = get_stats(
            get_values(
                method,
                suffix,
            )
        )

        values_list.append(values)
        means.append(mean)
        cis.append(ci95)

        print(
            suffix,
            method,
            f"mean={mean:.6f}",
            f"ci95={ci95:.6f}",
            f"n={len(values)}",
        )

    # ---------------------
    # 柱子
    # ---------------------
    ax.bar(
        x,
        means,
        width=bar_width,
        color=[
            colors[m]
            for m in methods
        ],
        edgecolor="#333333",
        linewidth=0.4,
        zorder=2,
    )

    # ---------------------
    # 仅向上的 95% CI
    # ---------------------
    cap_width = 0.04

    for xi, mean, ci95 in zip(
        x,
        means,
        cis,
    ):

        ax.plot(
            [xi, xi],
            [mean, mean + ci95],
            color="#444444",
            linewidth=0.5,
            zorder=4,
        )

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
            linewidth=0.5,
            zorder=4,
        )

    # ---------------------
    # 36 个 drug-dose 点
    # 每个点已经是：
    # 3 splits × 3 seeds 的平均结果
    # ---------------------
    for i, values in enumerate(
        values_list
    ):

        jitter = rng.normal(
            0,
            0.012,
            size=len(values),
        )

        ax.scatter(
            np.full(
                len(values),
                x[i],
            )
            + jitter,
            values,
            s=1.5,
            color="black",
            alpha=0.68,
            linewidths=0,
            zorder=3,
        )

    # ---------------------
    # 配对单尾 t 检验
    # 现在配对单位也是 36 个 drug-dose
    # ---------------------
    diff_values = get_values(
        "DIFFCRISP",
        suffix,
    )

    crisp_values = get_values(
        "CRISP",
        suffix,
    )

    chemcpa_values = get_values(
        "ChemCPA",
        suffix,
    )

    p_diff_crisp = paired_p(
        diff_values,
        crisp_values,
        info["alternative"],
    )

    p_diff_chemcpa = paired_p(
        diff_values,
        chemcpa_values,
        info["alternative"],
    )

    # 不再在标题上面画横线，避免和标题重叠。
    # 直接放在子图内部左上角。
    ax.text(
        0.03,
        0.96,
        "Diff vs CRISP: "
        + format_p(p_diff_crisp)
        + "\n"
        + "Diff vs ChemCPA: "
        + format_p(p_diff_chemcpa),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color="#b22222",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.78,
            pad=1.2,
        ),
        zorder=10,
    )

    # ---------------------
    # 标题和坐标轴
    # ---------------------
    ax.set_title(
        info["title"],
        fontsize=12,
        pad=14,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            "DIFFCRISP",
            "CRISP",
            "ChemCPA",
        ],
        rotation=38,
        ha="right",
        rotation_mode="anchor",
        fontsize=8.5,
    )

    ax.set_xlim(
        -0.16,
        0.84,
    )

    ax.set_ylim(
        *info["ylim"]
    )

    ax.set_yticks(
        info["yticks"]
    )

    ax.tick_params(
        axis="x",
        bottom=False,
        top=False,
        length=0,
    )

    ax.tick_params(
        axis="y",
        labelsize=9,
        width=0.8,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.spines[
        "left"
    ].set_linewidth(0.8)

    ax.spines[
        "bottom"
    ].set_linewidth(0.8)


fig.text(
    0.008,
    0.965,
    "h",
    fontsize=18,
    fontweight="bold",
    va="top",
)


plt.savefig(
    "fig2h_3methods_36drugdose.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "fig2h_3methods_36drugdose.png",
    bbox_inches="tight",
    dpi=600,
)

plt.show()
