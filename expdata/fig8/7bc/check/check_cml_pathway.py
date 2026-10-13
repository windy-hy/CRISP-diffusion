from pathlib import Path
import pandas as pd
import re


GSEA_DIR = Path("../7b/extended_fig2/gsea")

DRUGS = [
    "Dasatinib",
    "Nilotinib",
    "Bosutinib",
]

CML_PATHWAYS = [
    "KEGG_LEUKOCYTE_TRANSENDOTHELIAL_MIGRATION",
    "KEGG_PATHWAYS_IN_CANCER",
    "PID_PDGFRB_PATHWAY",
    "REACTOME_RHO_GTPASE_CYCLE",
    "REACTOME_SIGNALING_BY_RHO_GTPASES_MIRO_GTPASES_AND_RHOBTB3",
    "REACTOME_SIGNALING_BY_THE_B_CELL_RECEPTOR_BCR",
    "KEGG_CHRONIC_MYELOID_LEUKEMIA",
]


def safe_name(name):
    return re.sub(
        r'[<>:"/\\|?*]',
        "_",
        str(name)
    ).rstrip(". ")


# =========================================================
# 读取三个药的真实GSEA
# =========================================================

drug_df = {}

for drug in DRUGS:

    file = (
        GSEA_DIR
        / f"{safe_name(drug)}_gsea_truth.csv"
    )

    df = pd.read_csv(file)

    # 只保留负富集
    df = df[
        df["NES"] < 0
    ].copy()

    drug_df[drug] = df.set_index("Term")


# =========================================================
# 1. 不同NES阈值下，3/3共同通路数量
# =========================================================

thresholds = [
    -1.0,
    -1.25,
    -1.5,
    -1.75,
    -2.0,
    -2.25,
]

print("\n==============================")
print("不同 NES 强度下的 3/3 共同通路")
print("==============================")

for threshold in thresholds:

    sets = []

    for drug in DRUGS:

        df = drug_df[drug]

        terms = set(
            df[
                df["NES"] <= threshold
            ].index
        )

        sets.append(terms)

    common = set.intersection(*sets)

    print(
        f"NES <= {threshold:5.2f}: "
        f"{len(common)} 条"
    )


# =========================================================
# 2. 三药共同通路，按照平均NES强度排序
# =========================================================

all_common = set.intersection(
    *[
        set(df.index)
        for df in drug_df.values()
    ]
)

rows = []

for pathway in all_common:

    nes = [
        drug_df[drug].loc[
            pathway,
            "NES"
        ]
        for drug in DRUGS
    ]

    rows.append({
        "Term": pathway,
        "Dasatinib_NES": nes[0],
        "Nilotinib_NES": nes[1],
        "Bosutinib_NES": nes[2],
        "Mean_NES": sum(nes) / 3,
        "Weakest_NES": max(nes),
    })


common_df = pd.DataFrame(rows)

common_df = common_df.sort_values(
    "Mean_NES"
)

common_df.to_csv(
    "common_CML_pathways_by_strength.csv",
    index=False
)


print("\n==============================")
print("最强的30条三药共同负富集通路")
print("==============================")

print(
    common_df[
        [
            "Term",
            "Dasatinib_NES",
            "Nilotinib_NES",
            "Bosutinib_NES",
            "Mean_NES",
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# =========================================================
# 3. 专门检查作者那7条的NES
# =========================================================

print("\n==============================")
print("作者7条 pathway 的 NES")
print("==============================")

rows = []

for pathway in CML_PATHWAYS:

    row = {
        "Term": pathway
    }

    values = []

    for drug in DRUGS:

        if pathway in drug_df[drug].index:

            nes = drug_df[drug].loc[
                pathway,
                "NES"
            ]

            row[f"{drug}_NES"] = nes
            values.append(nes)

        else:

            row[f"{drug}_NES"] = None

    if values:

        row["Mean_NES"] = sum(values) / len(values)
        row["n_drugs"] = len(values)

    rows.append(row)


author_df = pd.DataFrame(rows)

print(
    author_df.to_string(
        index=False
    )
)

author_df.to_csv(
    "author_7_CML_pathways_NES.csv",
    index=False
)