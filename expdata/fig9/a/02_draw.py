from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42


BASE = Path(__file__).parent

df = pd.read_csv(
    BASE / "ablation_metrics_group.csv"
)


methods = [
    "Deterministic",
    "Gaussian",
    "Diffusion"
]

labels = [
    "Det.",
    "Gaussian",
    "Diffusion"
]


colors = [
    "#999999",
    "#E69F00",
    "#0072B2"
]


metrics = [

    ("r2score_de", r"$R^2$"),

    ("pearson_delta_de", r"$Pr_{\Delta}$"),

    ("sinkhorn_de", "Sinkhorn"),

    ("centered_sinkhorn_de", "Centered Sinkhorn"),

    ("variance_log_mae_de", "Variance log-MAE"),

    ("responder_error", "Responder error")

]


# 统计
stat = {}

for m, _ in metrics:

    x = df.pivot_table(
        index=["split","group"],
        columns="model",
        values=m,
        aggfunc="mean"
    )

    stat[m] = {}

    for name in ["Deterministic","Gaussian"]:

        y = x[["Diffusion",name]].dropna()

        stat[m][name] = wilcoxon(
            y["Diffusion"],
            y[name]
        ).pvalue



fig, axes = plt.subplots(
    2,
    3,
    figsize=(10,6)
)


axes = axes.flatten()


for ax,(metric,title) in zip(
        axes,
        metrics
):

    data=[]

    for method in methods:

        data.append(
            df.loc[
                df.model==method,
                metric
            ].dropna()
        )


    bp=ax.boxplot(
        data,
        widths=0.55,
        showfliers=False,
        patch_artist=False
    )


    # mean
    for i,v in enumerate(data,1):

        ax.scatter(
            i,
            np.mean(v),
            marker="D",
            s=25,
            color="black",
            zorder=3
        )


    ax.set_xticks([1,2,3])
    ax.set_xticklabels(
        labels,
        fontsize=9
    )


    ax.set_ylabel(
        title,
        fontsize=10
    )


    p1=stat[metric]["Deterministic"]
    p2=stat[metric]["Gaussian"]


    ax.text(
        0.03,
        0.95,
        f"Diff vs Det: P={p1:.3g}\n"
        f"Diff vs Gau: P={p2:.3g}",
        transform=ax.transAxes,
        fontsize=7,
        va="top"
    )


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        alpha=0.25
    )


# A-F
for ax,l in zip(
    axes,
    "ABCDEF"
):
    ax.text(
        -0.18,
        1.05,
        l,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold"
    )


plt.tight_layout()


plt.savefig(
    BASE/"Fig8a_core_generator_ablation.pdf",
    dpi=400,
    bbox_inches="tight"
)


plt.savefig(
    BASE/"Fig8a_core_generator_ablation.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()