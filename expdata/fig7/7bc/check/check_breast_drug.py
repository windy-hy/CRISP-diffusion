import scanpy as sc
import re


DATA = "../../../data/sciplex3/sciplex3_pp_hvgenes_scFM_resplit.h5ad"

adata = sc.read_h5ad(DATA)


# =========================================================
# NCI: FDA-approved drugs to TREAT breast cancer
# 商品名重复已去除，统一成主要通用名
# =========================================================

NCI_BREAST_DRUGS = [
    "Abemaciclib",
    "Paclitaxel",
    "Ado-Trastuzumab Emtansine",
    "Everolimus",
    "Alpelisib",
    "Anastrozole",
    "Pamidronate",
    "Exemestane",
    "Capecitabine",
    "Capivasertib",
    "Cyclophosphamide",
    "Datopotamab Deruxtecan",
    "Docetaxel",
    "Doxorubicin",
    "Elacestrant",
    "Epirubicin",
    "Fam-Trastuzumab Deruxtecan",
    "Eribulin",
    "Fluorouracil",
    "Toremifene",
    "Fulvestrant",
    "Letrozole",
    "Gemcitabine",
    "Goserelin",
    "Trastuzumab",
    "Palbociclib",
    "Imlunestrant",
    "Inavolisib",
    "Ixabepilone",
    "Pembrolizumab",
    "Ribociclib",
    "Lapatinib",
    "Olaparib",
    "Margetuximab",
    "Megestrol",
    "Methotrexate",
    "Neratinib",
    "Pertuzumab",
    "Sacituzumab Govitecan",
    "Tamoxifen",
    "Talazoparib",
    "Atezolizumab",
    "Thiotepa",
    "Tucatinib",
    "Vinblastine",
]


# =========================================================
# 盐形式 / NCI正式名称别名
# 禁止substring模糊匹配
# =========================================================

ALIASES = {

    "Paclitaxel": [
        "Paclitaxel",
        "Paclitaxel Albumin-stabilized Nanoparticle Formulation",
    ],

    "Pamidronate": [
        "Pamidronate",
        "Pamidronate Disodium",
    ],

    "Doxorubicin": [
        "Doxorubicin",
        "Doxorubicin Hydrochloride",
    ],

    "Elacestrant": [
        "Elacestrant",
        "Elacestrant Dihydrochloride",
    ],

    "Epirubicin": [
        "Epirubicin",
        "Epirubicin Hydrochloride",
    ],

    "Eribulin": [
        "Eribulin",
        "Eribulin Mesylate",
    ],

    "Fluorouracil": [
        "Fluorouracil",
        "Fluorouracil Injection",
        "5-FU",
    ],

    "Gemcitabine": [
        "Gemcitabine",
        "Gemcitabine Hydrochloride",
    ],

    "Goserelin": [
        "Goserelin",
        "Goserelin Acetate",
    ],

    "Lapatinib": [
        "Lapatinib",
        "Lapatinib Ditosylate",
    ],

    "Megestrol": [
        "Megestrol",
        "Megestrol Acetate",
    ],

    "Methotrexate": [
        "Methotrexate",
        "Methotrexate Sodium",
    ],

    "Neratinib": [
        "Neratinib",
        "Neratinib Maleate",
    ],

    "Ribociclib": [
        "Ribociclib",
        "Ribociclib Succinate",
    ],

    "Talazoparib": [
        "Talazoparib",
        "Talazoparib Tosylate",
    ],

    "Tamoxifen": [
        "Tamoxifen",
        "Tamoxifen Citrate",
    ],

    "Vinblastine": [
        "Vinblastine",
        "Vinblastine Sulfate",
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
# SciPlex3 全部药物
# =========================================================

sciplex_drugs = sorted(
    adata.obs["condition"]
    .dropna()
    .astype(str)
    .unique()
)


# =========================================================
# 严格药名/别名匹配
# =========================================================

matched = []

for approved in NCI_BREAST_DRUGS:

    aliases = ALIASES.get(
        approved,
        [approved]
    )

    alias_norm = {
        normalize(x)
        for x in aliases
    }

    hits = []

    for drug in sciplex_drugs:

        if normalize(drug) in alias_norm:
            hits.append(drug)

    if hits:

        matched.append(
            (
                approved,
                hits
            )
        )


print("\n====================================")
print("NCI breast cancer drugs ∩ SciPlex3")
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
# 检查这些药在 MCF7 中是否真的有数据
# =========================================================

print("\n====================================")
print("MCF7 中实际有数据")
print("====================================\n")


mcf7_obs = adata.obs[
    adata.obs["cell_type"].astype(str)
    == "MCF7"
]


final_drugs = []


for approved, hits in matched:

    for drug in hits:

        n = (
            mcf7_obs["condition"]
            .astype(str)
            .eq(drug)
        ).sum()

        if n > 0:

            final_drugs.append(drug)

            print(
                f"{drug:30s}",
                "cells =",
                n
            )


print(
    "\nFinal MCF7 approved breast cancer drugs:",
    len(final_drugs)
)

print(final_drugs)