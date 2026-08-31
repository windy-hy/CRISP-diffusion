import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


INPUT_FILE = "fig3b.xlsx"
OUTPUT_FILE = "fig3b.pdf"

ENCODINGS = ["FCFP4", "RDKit2D", "ChemBERTa"]

COLORS = {
    "FCFP4": "#4C78A8",
    "RDKit2D": "#E45756",
    "ChemBERTa": "#72B7B2",
}

MARKERS = {
    "FCFP4": "o",
    "RDKit2D": "^",
    "ChemBERTa": "s",
}

METRICS = [
    ("pearson_delta_de", r"Pr$_{\Delta}$ DE $\uparrow$"),
    ("sinkhorn_de", r"Sinkhorn DE $\downarrow$"),
    ("r2_de", r"$R^2$ score DE $\uparrow$"),
]

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


# 只有点接近重合时才左右错开
def get_x_positions(values):
    values = np.asarray(values, dtype=float)
    x = np.zeros(len(values))

    span = max(
        np.ptp(values),
        abs(values.mean()) * 0.01,
        1e-6
    )

    threshold = span * 0.10
    order = np.argsort(values)

    groups = []
    current = [order[0]]

    for idx in order[1:]:
        if abs(values[idx] - values[current[-1]]) < threshold:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]

    groups.append(current)

    for group in groups:
        if len(group) > 1:
            offsets = np.linspace(
                -0.035,
                0.035,
                len(group)
            )

            for idx, offset in zip(group, offsets):
                x[idx] = offset

    return x


df = pd.read_excel(INPUT_FILE)

summary = df.groupby("encoding", observed=True)[
    ["pearson_delta_de", "sinkhorn_de", "r2_de"]
].mean()


fig, axes = plt.subplots(
    1,
    3,
    figsize=(4.9, 2.65)
)

for ax, (metric, xlabel) in zip(axes, METRICS):

    values = summary.loc[ENCODINGS, metric].to_numpy()
    x_positions = get_x_positions(values)

    for i, encoding in enumerate(ENCODINGS):
        ax.scatter(
            x_positions[i],
            values[i],
            s=54,
            marker=MARKERS[encoding],
            color=COLORS[encoding],
            edgecolors="none",
            zorder=3
        )

    # 横向范围很窄
    ax.set_xlim(-0.12, 0.12)

    # 高度/宽度，数值越大子图越窄
    ax.set_box_aspect(1.55)

    value_range = values.max() - values.min()
    margin = max(
        value_range * 0.30,
        abs(values.mean()) * 0.002,
        0.0005
    )

    ax.set_ylim(
        values.min() - margin,
        values.max() + margin
    )

    ax.set_xticks([0])
    ax.set_xticklabels([xlabel])

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=4)
    )

    if metric == "sinkhorn_de":
        ax.yaxis.set_major_formatter(
            FormatStrFormatter("%.1f")
        )
    else:
        ax.yaxis.set_major_formatter(
            FormatStrFormatter("%.3f")
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3.5,
        width=0.8
    )


axes[0].set_ylabel(
    "NeurIPS",
    fontsize=11,
    labelpad=7
)

fig.text(
    0.025,
    0.96,
    "b",
    fontsize=13,
    fontweight="bold",
    va="top"
)


handles = [
    Line2D(
        [0], [0],
        marker=MARKERS[e],
        linestyle="none",
        markerfacecolor=COLORS[e],
        markeredgecolor="none",
        markersize=7.5,
        label=e
    )
    for e in ENCODINGS
]

fig.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.01),
    handletextpad=0.4,
    columnspacing=1.3
)

fig.subplots_adjust(
    left=0.12,
    right=0.985,
    top=0.91,
    bottom=0.29,
    wspace=0.62
)

fig.savefig(
    OUTPUT_FILE,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()