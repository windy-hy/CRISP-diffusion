from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# 路径
# =========================================================

ROOT = Path(
    "./mcf7_screening"
)

FIG_DIR = ROOT / "figures"

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 字体
# =========================================================

sns.set_theme(
    style="white"
)

plt.rcParams[
    "font.family"
] = "Arial"

plt.rcParams[
    "font.sans-serif"
] = ["Arial"]

plt.rcParams[
    "pdf.fonttype"
] = 42

plt.rcParams[
    "ps.fonttype"
] = 42


# =========================================================
# 数据
# =========================================================

precision = pd.read_csv(

    ROOT
    / "precision_at_k.csv"
)


ranking = pd.read_csv(

    ROOT
    / "drug_ranking.csv"
)


pathway = pd.read_csv(

    ROOT
    / "pathway_precision.csv"
)


# =========================================================
# 1. Precision@K
# =========================================================

fig, ax = plt.subplots(
    figsize=(4.5, 3.6)
)


ax.plot(

    precision["K"],

    precision["Precision"],

    marker="o",

    linewidth=1.5,

    label="DiffCRISP"
)


baseline = precision[
    "Random_baseline"
].iloc[0]


ax.axhline(

    baseline,

    linestyle="--",

    linewidth=1.0,

    label="Random baseline"
)


ax.set_xlabel(
    "Top K candidate drugs"
)

ax.set_ylabel(
    "Precision"
)


ax.set_ylim(
    0,
    1.05
)


ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)


ax.legend(
    frameon=False
)


plt.tight_layout()


plt.savefig(

    FIG_DIR
    / "precision_at_k.pdf",

    bbox_inches="tight"
)


plt.close()


# =========================================================
# 2. Top 30 candidate drugs
# =========================================================

top30 = (

    ranking

    .head(30)

    .copy()
)


colors = [

    "lightsalmon"
    if x
    else "lightgray"

    for x
    in top30[
        "reference_positive"
    ]
]


fig, ax = plt.subplots(
    figsize=(9.0, 4.2)
)


ax.bar(

    range(
        len(top30)
    ),

    top30[
        "score"
    ],

    color=colors,

    edgecolor="black",

    linewidth=0.4
)


ax.set_xticks(

    range(
        len(top30)
    )
)


ax.set_xticklabels(

    top30[
        "drug"
    ],

    rotation=60,

    ha="right",

    fontsize=8
)


ax.set_ylabel(
    "Signed pathway score"
)


ax.set_xlabel(
    ""
)


ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)


plt.tight_layout()


plt.savefig(

    FIG_DIR
    / "top30_drugs.pdf",

    bbox_inches="tight"
)


plt.close()


# =========================================================
# 3. Pathway precision
# =========================================================

pathway_top = (

    pathway

    .sort_values(
        "pathway_precision",
        ascending=False
    )

    .head(30)

    .copy()
)


fig, ax = plt.subplots(
    figsize=(7.5, 4.0)
)


ax.bar(

    range(
        len(pathway_top)
    ),

    pathway_top[
        "pathway_precision"
    ],

    edgecolor="black",

    linewidth=0.4
)


ax.set_xticks(

    range(
        len(pathway_top)
    )
)


ax.set_xticklabels(

    pathway_top[
        "drug"
    ],

    rotation=60,

    ha="right",

    fontsize=8
)


ax.set_ylabel(
    "Pathway precision"
)


ax.set_ylim(
    0,
    1.05
)


ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)


plt.tight_layout()


plt.savefig(

    FIG_DIR
    / "pathway_precision.pdf",

    bbox_inches="tight"
)


plt.close()


print(
    "绘图完成"
)