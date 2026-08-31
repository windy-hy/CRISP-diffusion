import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


PEARSON_FILE = "fig3e.csv"
MMD_FILE = "fig3e_mmd.csv"

OUTPUT_PDF = "fig3e.pdf"
OUTPUT_PNG = "fig3e.png"

CELLS = ["A549", "K562", "MCF7"]

COLORS = {
    "DIFFCRISP": "#4472B2",
    "CRISP": "#D9D9D9"
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


def normalize_method(x):
    x = str(x).upper().replace("-", "").replace("_", "").replace(" ", "")

    if x == "CRISP":
        return "CRISP"

    if x in ["DIFFCRISP", "DCRISP"]:
        return "DIFFCRISP"

    return x


def format_tick(x, pos):
    if abs(x) < 1e-10:
        return "0"

    return f"{x:.3f}".rstrip("0").rstrip(".")


# =========================
# 读取 Pearson 数据
# =========================

raw = pd.read_csv(
    PEARSON_FILE,
    header=None,
    encoding="utf-8-sig"
)

records = []

for _, row in raw.iterrows():
    cell = str(row.iloc[0]).strip().upper()

    if cell not in CELLS:
        continue

    for col in [1, 2, 3]:
        value = pd.to_numeric(row.iloc[col], errors="coerce")

        if pd.notna(value):
            records.append([cell, "CRISP", value])

    for col in [4, 5, 6]:
        value = pd.to_numeric(row.iloc[col], errors="coerce")

        if pd.notna(value):
            records.append([cell, "DIFFCRISP", value])

pearson = pd.DataFrame(
    records,
    columns=["cell_type", "method", "value"]
)

if pearson.empty:
    raise ValueError("fig3e.csv 中没有读取到 Pearson 数据")


# =========================
# 读取 MMD 数据
# =========================

mmd = pd.read_csv(
    MMD_FILE,
    encoding="utf-8-sig"
)

mmd.columns = mmd.columns.str.strip()

mmd["cell_type"] = (
    mmd["cell_type"]
    .astype(str)
    .str.strip()
    .str.upper()
)

mmd["method"] = mmd["method"].map(normalize_method)

mmd["mmd_de"] = pd.to_numeric(
    mmd["mmd_de"],
    errors="coerce"
)

mmd = mmd[
    mmd["cell_type"].isin(CELLS)
    & mmd["mmd_de"].notna()
].copy()

if mmd.empty:
    raise ValueError("fig3e_mmd.csv 中没有读取到有效 MMD 数据")


# 优先使用 DiffCRISP，没有则使用 CRISP
available_methods = set(mmd["method"])

if "DIFFCRISP" in available_methods:
    mmd_method = "DIFFCRISP"
elif "CRISP" in available_methods:
    mmd_method = "CRISP"
else:
    mmd_method = None

if mmd_method is not None:
    mmd_plot = mmd[mmd["method"] == mmd_method].copy()
else:
    mmd_plot = mmd.copy()

print("顶部 MMD 使用的方法：", mmd_method or "全部数据")


# =========================
# 创建画布
# =========================

fig = plt.figure(figsize=(3.2, 4.3))

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[0.43, 1],
    hspace=0.22
)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

x = np.arange(len(CELLS))


# =========================
# 上图：MMD
# =========================

mmd_mean = []
mmd_error = []

for cell in CELLS:
    values = mmd_plot.loc[
        mmd_plot["cell_type"] == cell,
        "mmd_de"
    ].to_numpy()

    mmd_mean.append(np.mean(values))

    if len(values) > 1:
        mmd_error.append(
            np.std(values, ddof=1) / np.sqrt(len(values))
        )
    else:
        mmd_error.append(0)

mmd_mean = np.array(mmd_mean)
mmd_error = np.array(mmd_error)

ax1.errorbar(
    x,
    mmd_mean,
    yerr=mmd_error,
    marker="o",
    markersize=6,
    linewidth=0.9,
    color="#8F87E8",
    capsize=4,
    elinewidth=0.9
)

ax1.set_xlim(-0.65, 2.65)
ax1.set_ylim(
    0,
    max(mmd_mean + mmd_error) * 1.25
)

ax1.set_xticks([])
ax1.set_ylabel(r"MMD DE", fontsize=11)

ax1.yaxis.set_major_formatter(
    FuncFormatter(format_tick)
)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# =========================
# 下图：Pearson Delta
# =========================

bar_width = 0.28

offsets = {
    "DIFFCRISP": -0.18,
    "CRISP": 0.18
}

statistics = {}
rng = np.random.default_rng(42)

for method in ["DIFFCRISP", "CRISP"]:
    means = []
    stds = []
    positions = []

    for i, cell in enumerate(CELLS):
        values = pearson.loc[
            (pearson["cell_type"] == cell)
            & (pearson["method"] == method),
            "value"
        ].to_numpy()

        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0
        position = i + offsets[method]

        means.append(mean)
        stds.append(std)
        positions.append(position)

        statistics[(cell, method)] = {
            "mean": mean,
            "std": std
        }

        jitter = np.linspace(
            -0.075,
            0.075,
            len(values)
        )

        ax2.scatter(
            position + jitter,
            values,
            s=7,
            facecolor=COLORS[method],
            edgecolor="#555555",
            linewidth=0.4,
            zorder=5
        )

    ax2.bar(
        positions,
        means,
        width=bar_width,
        color=COLORS[method],
        edgecolor="none",
        zorder=2
    )

    ax2.errorbar(
        positions,
        means,
        yerr=stds,
        fmt="none",
        ecolor="#666666",
        capsize=2.5,
        elinewidth=0.8,
        zorder=4
    )


# =========================
# 标注提升比例
# =========================

for i, cell in enumerate(CELLS):
    diff_mean = statistics[(cell, "DIFFCRISP")]["mean"]
    crisp_mean = statistics[(cell, "CRISP")]["mean"]

    diff_std = statistics[(cell, "DIFFCRISP")]["std"]
    crisp_std = statistics[(cell, "CRISP")]["std"]

    improvement = (
        (diff_mean - crisp_mean)
        / abs(crisp_mean)
        * 100
    )

    if improvement <= 0:
        continue

    left = i + offsets["DIFFCRISP"]
    right = i + offsets["CRISP"]

    y = max(
        diff_mean + diff_std,
        crisp_mean + crisp_std
    ) + 0.025

    ax2.plot(
        [left, left, right, right],
        [y, y + 0.015, y + 0.015, y],
        color="crimson",
        linewidth=0.9
    )

    ax2.text(
        i,
        y + 0.025,
        f"↑{improvement:.1f}%",
        ha="center",
        color="crimson",
        fontsize=9
    )


# =========================
# 坐标轴和图例
# =========================

ax2.set_xlim(-0.65, 2.65)
ax2.set_ylim(0, 0.85)

ax2.set_xticks(x)
ax2.set_xticklabels(CELLS)

ax2.set_yticks(
    np.arange(0, 0.81, 0.2)
)

ax2.set_ylabel(
    r"$\mathrm{Pr}_{\Delta}$ DE",
    fontsize=11
)

ax2.yaxis.set_major_formatter(
    FuncFormatter(format_tick)
)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

legend = [
    Patch(
        facecolor=COLORS["DIFFCRISP"],
        label="DiffCRISP"
    ),
    Patch(
        facecolor=COLORS["CRISP"],
        label="CRISP"
    )
]

ax2.legend(
    handles=legend,
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1.18, 1.10),
    handlelength=1,
    handletextpad=0.4
)


# =========================
# 面板编号和保存
# =========================

fig.text(
    0.035,
    0.975,
    "e",
    fontsize=14,
    fontweight="bold",
    va="top"
)

fig.subplots_adjust(
    left=0.25,
    right=0.95,
    top=0.94,
    bottom=0.12
)

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    pad_inches=0.03
)

plt.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()