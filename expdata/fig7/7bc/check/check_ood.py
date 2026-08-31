import scanpy as sc
import pandas as pd

DATA = "../../../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"

adata = sc.read_h5ad(DATA)

drugs = [
    "Capecitabine",
    "Fluorouracil",
    "Toremifene",
    "Fulvestrant",
    "Lapatinib",
    "Thiotepa",
]

for split in ["split", "split2", "split3"]:

    print(f"\n========== {split} ==========")

    mcf7 = adata.obs[
        adata.obs["cell_type"].astype(str) == "MCF7"
    ]

    print("\nMCF7 split counts:")
    print(mcf7[split].value_counts(dropna=False))

    print("\n六个药：")

    for drug in drugs:

        x = mcf7[
            mcf7["condition"].astype(str) == drug
        ]

        print(
            drug,
            "total =", len(x),
            "|",
            x[split].value_counts().to_dict()
        )

    ctrl = mcf7[
        mcf7["control"] == 1
    ]

    print(
        "\nControl:",
        len(ctrl),
        ctrl[split].value_counts().to_dict()
    )