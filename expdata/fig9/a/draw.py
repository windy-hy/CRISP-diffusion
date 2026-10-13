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


# =========================
# path
# =========================

BASE = Path(__file__).parent


INPUT = (
    BASE /
    "latent_sensitivity_split_mean.csv"
)


OUT_PDF = (
    BASE /
    "Fig9a_latent_sensitivity.pdf"
)

OUT_PNG = (
    BASE /
    "Fig9a_latent_sensitivity.png"
)



# =========================
# load
# =========================

df = pd.read_csv(INPUT)



# =========================
# calculate mean/std
# =========================

summary = (

    df

    .groupby(
        "latent_dim"
    )

    [
        [
            "R2_DE",
            "Pearson_Delta_DE",
            "Sinkhorn_DE"
        ]
    ]

    .agg(
        [
            "mean",
            "std"
        ]
    )

)


latent = summary.index.values
x = np.arange(len(latent))


# =========================
# plot
# =========================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(10,3.2)
)



plots = [

    (
        "R2_DE",
        r"$R^2$",
        "higher"
    ),

    (
        "Pearson_Delta_DE",
        r"$Pr_{\Delta}$",
        "higher"
    ),

    (
        "Sinkhorn_DE",
        "Sinkhorn distance",
        "lower"
    )

]

for ax,(metric,title,_) in zip(
        axes,
        plots
):


    mean = summary[
        (metric,"mean")
    ].values


    std = summary[
        (metric,"std")
    ].values

    ax.plot(
        x,
        mean,
        marker="o",
        markersize=7,
        linewidth=2,
        color="#0072B2",
        markerfacecolor="white",
        markeredgewidth=1.5
    )

    ax.set_xlabel(
        "Latent dimension",
        fontsize=10
    )

    ax.set_ylabel(
        title,
        fontsize=11
    )
    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        latent
    )
    ax.grid(
        axis="y",
        alpha=0.25
    )


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



# panel label

for ax,label in zip(
        axes,
        ["A","B","C"]
):

    ax.text(
        -0.2,
        1.12,
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


plt.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight"
)


plt.close()



print("saved:")
print(OUT_PDF)
print(OUT_PNG)