import scanpy as sc

nips = sc.read_h5ad(
    "../../../data/nips/nips_pp_scFM_resplit.h5ad",
    backed="r"
)

split_cols = [
    c for c in nips.obs.columns
    if "split" in c.lower()
]

print(split_cols)

for col in split_cols:
    print("\n", col)
    print(nips.obs[col].value_counts())