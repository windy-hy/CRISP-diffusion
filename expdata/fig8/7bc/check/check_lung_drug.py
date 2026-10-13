import scanpy as sc
import re


DATA = "../../../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"

adata = sc.read_h5ad(DATA)


# =========================================================
# NCI NSCLC drugs
# 去掉商品名重复后，统一为主要通用名
# =========================================================

NCI_NSCLC_DRUGS = [
    "Paclitaxel",
    "Adagrasib",
    "Afatinib",
    "Everolimus",
    "Alectinib",
    "Pemetrexed",
    "Brigatinib",
    "Bevacizumab",
    "Amivantamab",
    "Atezolizumab",
    "Repotrectinib",
    "Binimetinib",
    "Zenocutuzumab",
    "Encorafenib",
    "Capmatinib",
    "Cemiplimab",
    "Ceritinib",
    "Crizotinib",
    "Ramucirumab",
    "Dabrafenib",
    "Dacomitinib",
    "Docetaxel",
    "Doxorubicin",
    "Durvalumab",
    "Ensartinib",
    "Entrectinib",
    "Erlotinib",
    "Fam-Trastuzumab Deruxtecan",
    "Pralsetinib",
    "Gefitinib",
    "Gemcitabine",
    "Tremelimumab",
    "Ipilimumab",
    "Pembrolizumab",
    "Lazertinib",
    "Lorlatinib",
    "Sotorasib",
    "Trametinib",
    "Methotrexate",
    "Necitumumab",
    "Nivolumab",
    "Osimertinib",
    "Selpercatinib",
    "Tepotinib",
    "Vinorelbine",
]


# =========================================================
# 显式别名
# 这里只允许已知化学名称/盐形式，不使用模糊substring
# =========================================================

ALIASES = {

    "Paclitaxel": [
        "Paclitaxel",
    ],

    "Afatinib": [
        "Afatinib",
        "Afatinib Dimaleate",
    ],

    "Pemetrexed": [
        "Pemetrexed",
        "Pemetrexed Disodium",
    ],

    "Capmatinib": [
        "Capmatinib",
        "Capmatinib Hydrochloride",
    ],

    "Dabrafenib": [
        "Dabrafenib",
        "Dabrafenib Mesylate",
    ],

    "Doxorubicin": [
        "Doxorubicin",
        "Doxorubicin Hydrochloride",
    ],

    "Erlotinib": [
        "Erlotinib",
        "Erlotinib Hydrochloride",
    ],

    "Gemcitabine": [
        "Gemcitabine",
        "Gemcitabine Hydrochloride",
    ],

    "Lazertinib": [
        "Lazertinib",
        "Lazertinib Mesylate",
        "Lazertinib Mesylate Hydrate",
    ],

    "Osimertinib": [
        "Osimertinib",
        "Osimertinib Mesylate",
    ],

    "Tepotinib": [
        "Tepotinib",
        "Tepotinib Hydrochloride",
    ],

    "Trametinib": [
        "Trametinib",
        "Trametinib Dimethyl Sulfoxide",
    ],

    "Vinorelbine": [
        "Vinorelbine",
        "Vinorelbine Tartrate",
    ],

    "Ensartinib": [
        "Ensartinib",
        "Ensartinib Hydrochloride",
    ],
}


def normalize(x):

    x = str(x).lower()

    x = re.sub(
        r"[^a-z0-9]",
        "",
        x
    )

    return x


# =========================================================
# SciPlex3 drugs
# =========================================================

sciplex_drugs = sorted(
    adata.obs["condition"]
    .dropna()
    .astype(str)
    .unique()
)


# =========================================================
# 严格匹配
# =========================================================

matched = []


for approved in NCI_NSCLC_DRUGS:

    aliases = ALIASES.get(
        approved,
        [approved]
    )

    aliases_norm = {
        normalize(x)
        for x in aliases
    }

    hits = []

    for drug in sciplex_drugs:

        if normalize(drug) in aliases_norm:
            hits.append(drug)

    if hits:

        matched.append(
            (
                approved,
                hits
            )
        )


print("\n====================================")
print("NCI NSCLC drugs ∩ SciPlex3")
print("====================================\n")


for approved, hits in matched:

    print(
        f"{approved:30s} -> {hits}"
    )


print(
    "\nMatched approved drugs:",
    len(matched)
)


# =========================================================
# 再看 A549 中是否实际存在
# =========================================================

print("\n====================================")
print("A549 中实际有数据")
print("====================================\n")


a549_obs = adata.obs[
    adata.obs["cell_type"].astype(str)
    == "A549"
]


final_drugs = []


for approved, hits in matched:

    for drug in hits:

        n = (
            a549_obs["condition"]
            .astype(str)
            .eq(drug)
        ).sum()

        if n > 0:

            final_drugs.append(
                drug
            )

            print(
                f"{drug:30s}",
                "cells =",
                n
            )


print(
    "\nFinal A549 drugs:",
    len(final_drugs)
)

print(final_drugs)