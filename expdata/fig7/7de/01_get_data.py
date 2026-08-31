import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import gseapy as gp

DATA = "../../../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"
GMT = "../../../data/gsea/c2.cp.v2024.1.Hs.symbols.gmt"
DRUG = "Fulvestrant"

adata = sc.read_h5ad(DATA)
genes = adata.var_names.astype(str)

def dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)

ctrl = adata[
    (adata.obs["control"] == 1) &
    (adata.obs["split3"] == "ood") &
    (adata.obs["cell_type"] == "MCF7")
].copy()

treated = adata[
    (adata.obs["condition"] == DRUG) &
    (adata.obs["split3"] == "ood") &
    (adata.obs["cell_type"] == "MCF7") &
    (adata.obs["dose_val"] == 1.0)
].copy()

rng = np.random.default_rng(42)
if ctrl.n_obs > 1000:
    ctrl = ctrl[rng.choice(ctrl.n_obs, 1000, replace=False)].copy()

a = pd.DataFrame(dense(treated.X).T, index=genes)
b = pd.DataFrame(dense(ctrl.X).T, index=genes)
a.columns = ["Perturb"] * a.shape[1]
b.columns = ["Ctrl"] * b.shape[1]
df = pd.concat([a, b], axis=1)

gs = gp.gsea(
    data=df,
    gene_sets=GMT,
    cls=list(df.columns),
    min_size=10,
    permutation_type="phenotype",
    permutation_num=1000,
    method="signal_to_noise",
    outdir=None,
    threads=4,
    seed=42
)

res = gs.res2d.copy()
res["NES"] = pd.to_numeric(res["NES"])
res["FDR q-val"] = pd.to_numeric(res["FDR q-val"])

sig = res[(res["NES"].abs() > 1) & (res["FDR q-val"] < 0.25)]
pos = sig[sig["NES"] > 0].nlargest(5, "NES")
neg = sig[sig["NES"] < 0].nsmallest(5, "NES")
selected = pd.concat([pos, neg])

selected.to_csv("gsea_plot.csv", index=False)

# 右上固定 Breast cancer pathway
POS_TERM = "WP_BREAST_CANCER_PATHWAY"

# 右下最强负富集
NEG_TERM = neg.iloc[0]["Term"]

terms = [POS_TERM, NEG_TERM]

curve = {
    t: gs.results[t]
    for t in terms
}

with open("gsea_curve.pkl", "wb") as f:
    pickle.dump(curve, f)

print("正通路：", POS_TERM)
print("负通路：", NEG_TERM)

