import pandas as pd
import scanpy as sc


adata = sc.read_h5ad(
    "../../../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"
)

obs = adata.obs

UNSEEN_DRUGS = [
    "Dacinostat",
    "Givinostat",
    "Belinostat",
    "Hesperadin",
    "Quisinostat",
    "Alvespimycin",
    "Tanespimycin",
    "TAK-901",
    "Flavopiridol",
]

SETTINGS = [
    ("split_drugs", "MCF7"),
    ("split_drugs2", "A549"),
    ("split_drugs3", "K562"),
]

control = pd.to_numeric(
    obs["control"],
    errors="coerce",
).fillna(0).astype(int)

all_conditions = []

for split_key, target_cell in SETTINGS:
    split = (
        obs[split_key]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mask = (
        (control == 0)
        & (split == "ood")
        & (obs["cell_type"].astype(str) == target_cell)
        & (obs["condition"].astype(str).isin(UNSEEN_DRUGS))
    )

    conditions = (
        obs.loc[
            mask,
            [
                "condition",
                "dose_val",
                "cell_type",
                "cov_drug_dose_name",
            ],
        ]
        .drop_duplicates("cov_drug_dose_name")
        .copy()
    )

    conditions["split_key"] = split_key
    all_conditions.append(conditions)

    print("\n", split_key, "目标细胞：", target_cell)
    print("药物数：", conditions["condition"].nunique())
    print("组合数：", len(conditions))
    print(
        conditions.groupby("condition")
        .size()
        .to_string()
    )

all_conditions = pd.concat(
    all_conditions,
    ignore_index=True,
)

print("\n三个split总组合数：", len(all_conditions))