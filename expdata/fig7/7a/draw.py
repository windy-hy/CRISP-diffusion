import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm



# =========================
# 数据
# NaN = 该药物下这条通路不显著
# =========================
df = pd.DataFrame(
    {
        "Toremifene":   [np.nan,  2.12, -1.69, -2.04, -2.30, -1.96],
        "Fulvestrant":  [ 2.18,   1.86, -1.99, -2.25, -2.63, -2.68],
        "Lapatinib":    [np.nan,  1.82,  np.nan, -1.74, -1.98, -1.92],
    },
    index=[
        "Breast cancer pathway (1/3)",
        "MAPK signaling (3/3)",
        "ESR-mediated signaling (2/3)",
        "Rho GTPase effectors (3/3)",
        "Cell cycle checkpoints (3/3)",
        "Cell cycle (3/3)",
    ]
)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

fig, ax = plt.subplots(figsize=(5.4, 4.2))

cmap = sns.color_palette(
    "RdBu_r",
    as_cmap=True
)
cmap.set_bad("#F5F5F5")

hm = sns.heatmap(
    df,
    cmap=cmap,
    center=0,
    vmin=-3,
    vmax=3,
    annot=True,
    fmt=".2f",
    linewidths=0.6,
    linecolor="white",
    cbar_kws={
        "label": "NES",
        "shrink": 0.72,
        "aspect": 12
    },
    ax=ax
)

ax.set_xlabel("")
ax.set_ylabel("")

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=30,
    ha="right"
)

ax.set_title(
    "Representative breast cancer-related pathways",
    fontsize=13,
    pad=10
)

# colorbar去黑框
cbar = hm.collections[0].colorbar
cbar.outline.set_visible(False)

plt.tight_layout()

plt.savefig(
    "breast_pathway_heatmap.pdf",
    bbox_inches="tight"
)

plt.show()