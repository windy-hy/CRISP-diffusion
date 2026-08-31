import scanpy as sc
import pandas as pd


DATA_PATH = "nips_pp_scFM_resplit.h5ad"

adata = sc.read_h5ad(DATA_PATH, backed="r")

drug_key = "condition"
dose_key = "dose_val"
smiles_key = "SMILES"

obs = adata.obs[[drug_key, dose_key, smiles_key, "neg_control"]].copy()

# 去掉对照组
control = pd.to_numeric(
    obs["neg_control"],
    errors="coerce"
).eq(1)

obs = obs[~control].copy()

# 剂量转为数值
obs[dose_key] = pd.to_numeric(
    obs[dose_key],
    errors="coerce"
)

dose_table = (
    obs.groupby(drug_key, observed=True)
    .agg(
        SMILES=(smiles_key, "first"),
        doses=(dose_key, lambda x: sorted(x.dropna().unique().tolist())),
        n_doses=(dose_key, "nunique"),
        n_cells=(dose_key, "size"),
    )
    .reset_index()
    .sort_values(drug_key)
)

print("药物数：", dose_table[drug_key].nunique())
print(dose_table.to_string(index=False))

dose_table.to_csv(
    "nips_drug_doses.csv",
    index=False
)

print("\n保存：nips_drug_doses.csv")

adata.file.close()