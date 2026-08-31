import scanpy as sc

adata = sc.read_h5ad(
    "../../../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"
)

drugs = sorted(
    adata.obs["condition"]
    .astype(str)
    .unique()
)

targets = [
    "Imatinib",
    "Dasatinib",
    "Nilotinib",
    "Bosutinib",
    "Ponatinib",
    "Asciminib",
]

for d in targets:
    hits = [
        x for x in drugs
        if d.lower() in x.lower()
    ]
    print(d, "->", hits)