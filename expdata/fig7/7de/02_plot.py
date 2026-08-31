import pickle, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

df = pd.read_csv("gsea_plot.csv")
with open("gsea_curve.pkl", "rb") as f:
    curves = pickle.load(f)

df["NES"] = pd.to_numeric(df["NES"])
df["FDR q-val"] = pd.to_numeric(df["FDR q-val"])
df["c"] = (-np.log10(df["FDR q-val"].clip(lower=1e-2))).clip(1, 2)

def tag_pct(x):
    nums = re.findall(r"[\d.]+", str(x))
    return float(nums[0]) / float(nums[1]) * 100 if len(nums) >= 2 and float(nums[1]) > 0 else 30

df["pct"] = df["Tag %"].apply(tag_pct) if "Tag %" in df.columns else 30

def short_name(x):
    x = re.sub(r"^(REACTOME_|WP_|KEGG_|PID_)", "", str(x))
    x = x.replace("_", " ").title()
    x = x.replace("Pi3K", "PI3K").replace("Akt", "AKT").replace("Mtor", "mTOR")
    x = x.replace("Nrf2", "NRF2")
    return x

pos = df[df["NES"] > 0].sort_values("NES", ascending=False).head(5)
neg = df[df["NES"] < 0].sort_values("NES", ascending=False).head(5)   # 绝对值大的在下面
plot_df = pd.concat([pos, neg]).reset_index(drop=True)

top_term = "WP_BREAST_CANCER_PATHWAY" if "WP_BREAST_CANCER_PATHWAY" in df["Term"].values else pos.iloc[0]["Term"]
bottom_term = neg.iloc[-1]["Term"]

cmap = LinearSegmentedColormap.from_list("paper", ["#4F91A4", "#F2F1EE", "#D46749"])
norm = Normalize(1, 2)

fig = plt.figure(figsize=(7.2, 4.2))
gs = fig.add_gridspec(2, 3, width_ratios=[0.75, 0.07, 1.0], wspace=0.35, hspace=0.45)

ax0 = fig.add_subplot(gs[:, 0])
axc = fig.add_subplot(gs[:, 1])

y = np.arange(len(plot_df))
sc = ax0.scatter(
    plot_df["NES"], y,
    s=60 + plot_df["pct"] * 4.5,   # 点放大
    c=plot_df["c"], cmap=cmap, norm=norm,
    edgecolors="none"
)

ax0.set_yticks(y)
ax0.set_yticklabels([short_name(t) for t in plot_df["Term"]], fontsize=8.5)
ax0.invert_yaxis()
ax0.set_xlim(-4, 4)
ax0.set_xticks([ -2, 0, 2])
ax0.set_xlabel("Normalized enrichment score", fontsize=10)
ax0.set_title("Fulvestrant (MCF7)", fontsize=11)
ax0.tick_params(axis="both", labelsize=8.5, width=0.8, length=3)
for s in ax0.spines.values():
    s.set_linewidth(0.8)

axc.axis("off")
cax = axc.inset_axes([0.25, 0.28, 0.35, 0.26])
cb = fig.colorbar(sc, cax=cax)
cb.set_ticks([1.0, 1.5, 2.0])
cb.ax.tick_params(labelsize=8, length=0)
cb.outline.set_visible(False)
cb.set_label(r"$\log_{10}(1/\mathrm{FDR})$", fontsize=9)

def draw_panel(cell, term):
    ax = fig.add_subplot(cell)

    r = curves[term]
    res = np.asarray(r["RES"])
    hits = np.asarray(r["hits"])
    n = len(res)
    x = np.arange(n)

    ymin, ymax = res.min(), res.max()
    span = ymax - ymin

    # rug 放在曲线最低点下面
    split_y = ymin - 0.08 * span
    bottom_y = ymin - 0.22 * span

    # 两端留白，和论文一样
    pad = n * 0.05
    ax.set_xlim(-pad, n - 1 + pad)
    ax.set_ylim(bottom_y, ymax + 0.05 * span)

    # GSEA curve
    ax.plot(x, res, color="#7FBF3F", lw=1.7)

    # curve / rug 分隔线
    ax.axhline(split_y, color="black", lw=0.8)

    # gene hits
    ax.vlines(
        hits,
        bottom_y,
        split_y,
        color="black",
        lw=0.5
    )

    # 整个外框闭合
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)

    ax.set_xticks([])
    ax.set_ylabel("Enrichment score", fontsize=8.5)
    ax.set_title(short_name(term), fontsize=10, pad=4)

    ax.text(
        0.05, 0.16,
        f"NES: {r['nes']:.3f}\n"
        f"Pval: {r['pval']:.3g}\n"
        f"FDR: {r['fdr']:.3g}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom"
    )

    ax.tick_params(axis="y", labelsize=8, width=0.8, length=3)

draw_panel(gs[0, 2], top_term)
draw_panel(gs[1, 2], bottom_term)

fig.subplots_adjust(left=0.24, right=0.97, top=0.92, bottom=0.12)
plt.savefig("Fulvestrant_GSEA_case.pdf", bbox_inches="tight")
plt.show()