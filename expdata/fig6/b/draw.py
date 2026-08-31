import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from DIFFCRISP.trainer import Trainer


PBMC_PATH = "../../../data/pbmc/pbmc_bench_pp_all.h5ad"
MODEL_PATH = "../../../model/nips/split/seed0/model_200_split.pt"
FM_KEY = "X_scGPT_blood"


# 接近原文的细胞类型顺序和颜色
CELL_ORDER = [
    "B cell",
    "CD4+ T cell",
    "CD14+ monocyte",
    "CD16+ monocyte",
    "Cytotoxic T cell",
    "Dendritic cell",
    "Megakaryocyte",
    "Natural killer cell",
    "Plasmacytoid dendritic cell",
    "Unassigned",
]

PALETTE = {
    "B cell": "#0645AD",
    "CD4+ T cell": "#BEC1D4",
    "CD14+ monocyte": "#D6BCC0",
    "CD16+ monocyte": "#BB7784",
    "Cytotoxic T cell": "#8595E1",
    "Dendritic cell": "#E07B91",
    "Megakaryocyte": "#D33F6A",
    "Natural killer cell": "#8DD593",
    "Plasmacytoid dendritic cell": "#C6DEC7",
    "Unassigned": "#F0B98D",
}


# =========================
# Arial + PDF字体
# =========================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================
# 读取数据和模型
# =========================
pbmc = sc.read_h5ad(PBMC_PATH)

exp = Trainer()
exp.load_model(MODEL_PATH)

device = exp.device
exp.autoencoder.to(device)
exp.autoencoder.eval()


# =========================
# 提取 DiffCRISP 潜在表示
# =========================
with torch.inference_mode():
    fm = torch.tensor(
        pbmc.obsm[FM_KEY],
        dtype=torch.float32,
        device=device,
    )

    z = exp.autoencoder.fm_proj(fm)

    if isinstance(z, tuple):
        z = z[0]

    z = z.cpu().numpy()


adata_lat = sc.AnnData(
    X=z,
    obs=pbmc.obs.copy(),
)

print("PBMC shape:", pbmc.shape)
print("Latent shape:", adata_lat.shape)


# =========================
# 统一类别顺序
# =========================
present_cells = [
    cell for cell in CELL_ORDER
    if cell in pbmc.obs["cell_type"].astype(str).unique()
]

for adata in [pbmc, adata_lat]:
    adata.obs["cell_type"] = pd.Categorical(
        adata.obs["cell_type"].astype(str),
        categories=present_cells,
        ordered=True,
    )


# =========================
# 计算 UMAP
# =========================
sc.pp.neighbors(pbmc)
sc.tl.umap(pbmc, min_dist=0.3)

sc.pp.neighbors(adata_lat)
sc.tl.umap(adata_lat, min_dist=0.3)


# =========================
# 工具函数：把散点栅格化
# =========================
def rasterize_scatter_in_axis(ax):
    """
    将当前坐标轴中的散点(collections)栅格化，
    文字、标题、图例、轴线仍保持矢量。
    """
    for coll in ax.collections:
        coll.set_rasterized(True)


# =========================
# 绘图
# 原来 figsize=(7.6, 3.8)
# 现在把高度加大一些
# =========================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.6, 4.8),
)

sc.pl.umap(
    pbmc,
    color="cell_type",
    palette=PALETTE,
    ax=axes[0],
    show=False,
    frameon=False,
    legend_loc=None,
    size=4,
    title="Gene expression matrix",
)

sc.pl.umap(
    adata_lat,
    color="cell_type",
    palette=PALETTE,
    ax=axes[1],
    show=False,
    frameon=False,
    legend_loc=None,
    size=4,
    title="DiffCRISP's latent space",
)


# =========================
# 关键：把UMAP点栅格化
# =========================
for ax in axes:
    rasterize_scatter_in_axis(ax)


# =========================
# 统一细节
# =========================
for ax in axes:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.title.set_fontsize(10)
    ax.title.set_fontweight("normal")


# =========================
# 底部图例
# 图例保持矢量
# =========================
handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=PALETTE[cell],
        markeredgecolor="none",
        markersize=6,
        label=cell,
    )
    for cell in present_cells
]

fig.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.52, 0.01),
    ncol=4,
    frameon=False,
    columnspacing=1.1,
    handletextpad=0.35,
)


# =========================
# 面板标记和左侧文字
# =========================
fig.text(
    0.015,
    0.96,
    "b",
    fontsize=13,
    fontweight="bold",
)

fig.text(
    0.035,
    0.55,
    "PBMC-bench",
    rotation=90,
    ha="center",
    va="center",
    fontsize=10,
)


# =========================
# 布局
# 因为整体变高，底部图例空间也略放宽一点
# =========================
plt.subplots_adjust(
    left=0.08,
    right=0.99,
    top=0.90,
    bottom=0.24,
    wspace=0.08,
)


# =========================
# 保存
# dpi=600 只影响被 rasterized 的点
# =========================
plt.savefig(
    "fig4b.pdf",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.savefig(
    "fig4b.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.show()