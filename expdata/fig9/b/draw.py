from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# =========================
# style
# =========================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

BASE = Path(__file__).parent

INPUT = (
    BASE /
    "ddim_sensitivity.csv"
)

OUT_PDF = (
    BASE /
    "Fig9b_ddim_sensitivity.pdf"
)

df = pd.read_csv(INPUT)
df = df.sort_values(
    "ddim_steps"
)

x_label = df["ddim_steps"].values

# categorical x
x = np.arange(
    len(x_label)
)

# =========================
# metrics
# =========================

plots = [

    (
        "R2_DE_mean",
        r"$R^2$"
    ),

    (
        "Pearson_Delta_DE_mean",
        r"$Pr_{\Delta}$"
    ),

    (
        "Sinkhorn_DE_mean",
        "Sinkhorn distance"
    )

]

# =========================
# plot
# =========================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(10,3.2)
)

for ax,(metric,title) in zip(
        axes,
        plots
):

    y = df[metric].values


    ax.plot(
        x,
        y,
        marker="o",
        markersize=6,
        linewidth=2,
        color="#0072B2",
        markerfacecolor="white",
        markeredgewidth=1.5
    )


    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        x_label,
        fontsize=9
    )

    ax.set_xlabel(
        "DDIM steps",
        fontsize=10
    )

    ax.set_ylabel(
        title,
        fontsize=11
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# panel labels

for ax,label in zip(
        axes,
        ["A","B","C"]
):

    ax.text(
        -0.18,
        1.1,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold"
    )



plt.tight_layout()


plt.savefig(
    OUT_PDF,
    dpi=400,
    bbox_inches="tight"
)

plt.close()

print("saved:")
print(OUT_PDF)