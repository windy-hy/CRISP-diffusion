from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("./mcf7_screening")
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="white")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

df = pd.read_csv(ROOT / "drug_ranking.csv").head(30).copy()
colors = ["lightsalmon" if x else "lightgray"
          for x in df["reference_positive"]]

fig, ax = plt.subplots(figsize=(9.0, 3.6))

x = range(len(df))
ax.bar(
    x, df["score"],
    width=0.75,                 # 柱子变窄，间隔稍大
    color=colors,
    edgecolor="black",
    linewidth=0.4
)

ax.set_xticks(list(x))
ax.set_xticklabels(df["drug"], rotation=60, ha="right", fontsize=8)
ax.set_ylabel("Signed pathway score")
ax.set_xlim(-0.35, len(df) - 0.65)   # 减少左右空隙

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.subplots_adjust(left=0.075, right=0.995, bottom=0.32, top=0.98)

plt.savefig(
    OUT / "top30_pathway_score.pdf",
    bbox_inches="tight",
    pad_inches=0.02
)
plt.close()
