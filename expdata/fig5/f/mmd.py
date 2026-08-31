import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from DIFFCRISP.losses import gaussian_mmd


adata = sc.read_h5ad(
    "../../../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"
)

obs = adata.obs
de_dict = adata.uns["rank_genes_groups_cov"]

cell_types = ["MCF7", "A549", "K562"]

drugs = [
    "Dacinostat", "Givinostat", "Belinostat",
    "Hesperadin", "Quisinostat", "Alvespimycin",
    "Tanespimycin", "TAK-901", "Flavopiridol",
]

doses = [0.001, 0.01, 0.1, 1.0]

rows = []


for cell_type in cell_types:

    ctrl_mask = (
        (obs["cell_type"].astype(str) == cell_type)
        & (obs["control"].astype(int) == 1)
    )

    for drug in drugs:
        for dose_val in doses:

            treat_mask = (
                (obs["cell_type"].astype(str) == cell_type)
                & (obs["condition"].astype(str) == drug)
                & np.isclose(
                    obs["dose_val"].astype(float),
                    dose_val,
                )
                & (obs["control"].astype(int) != 1)
            )

            if ctrl_mask.sum() < 5 or treat_mask.sum() < 5:
                continue

            comb = obs.loc[
                treat_mask,
                "cov_drug_dose_name",
            ].iloc[0]

            de_key = (
                comb
                if comb in de_dict
                else f"{cell_type}_{drug}"
            )

            genes = [
                str(g.decode() if isinstance(g, bytes) else g)
                for g in de_dict[de_key][:50]
            ]

            idx = adata.var_names.get_indexer(genes)
            idx = idx[idx >= 0]

            ctrl = adata[ctrl_mask, idx].X
            treat = adata[treat_mask, idx].X

            if sparse.issparse(ctrl):
                ctrl = ctrl.toarray()

            if sparse.issparse(treat):
                treat = treat.toarray()

            ctrl = torch.tensor(
                ctrl,
                dtype=torch.float32,
            )

            treat = torch.tensor(
                treat,
                dtype=torch.float32,
            )

            with torch.no_grad():
                mmd = gaussian_mmd(
                    ctrl,
                    treat,
                    blur=1,
                ).item()

            rows.append({
                "mmd_de": mmd,
                "comb": comb,
                "method": "CRISP",
                "cell_type": cell_type,
                "drug": drug,
                "dose": dose_val * 10,
            })


result = pd.DataFrame(rows)

result.to_csv(
    "fig3e_mmd.csv",
    index=False,
)

print(result)
print("总行数：", len(result))