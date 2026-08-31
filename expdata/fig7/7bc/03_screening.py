from pathlib import Path

import re

import numpy as np
import pandas as pd


# =========================================================
# 路径
# =========================================================

ROOT = Path(
    "./mcf7_screening"
)

GSEA_DIR = ROOT / "gsea"


# =========================================================
# 固定 6 条 pathway
#
# -1 = expected negative
# +1 = expected positive
# =========================================================

SIGNATURE = {

    "REACTOME_CELL_CYCLE_CHECKPOINTS":
        -1,

    "REACTOME_CELL_CYCLE":
        -1,

    "REACTOME_RHO_GTPASE_EFFECTORS":
        -1,

    "REACTOME_ESR_MEDIATED_SIGNALING":
        -1,

    "WP_MAPK_SIGNALING":
        1,

    "WP_BREAST_CANCER_PATHWAY":
        1,
}


# =========================================================
# 工具
# =========================================================

def safe_name(name):

    return re.sub(

        r'[<>:"/\\|?*]',

        "_",

        str(name)

    ).rstrip(
        ". "
    )


def get_correct_hits(df):

    hits = {}


    for _, row in df.iterrows():

        term = row[
            "Term"
        ]


        if term not in SIGNATURE:
            continue


        nes = float(
            row["NES"]
        )


        expected = SIGNATURE[
            term
        ]


        # expected positive
        if (
            expected == 1
            and nes > 0
        ):

            hits[
                term
            ] = nes


        # expected negative
        elif (
            expected == -1
            and nes < 0
        ):

            hits[
                term
            ] = nes


    return hits


# =========================================================
# Valid drugs
# =========================================================

manifest = pd.read_csv(

    ROOT
    / "gsea_manifest.csv"
)


drugs = (

    manifest[
        "drug"
    ]

    .astype(str)

    .tolist()
)


print(
    "Valid drugs:",
    len(drugs)
)


# =========================================================
# Screening
# =========================================================

rows = []

positive_drugs = []


for drug in drugs:

    drug_file = safe_name(
        drug
    )


    truth_file = (

        GSEA_DIR
        / f"{drug_file}_gsea_truth.csv"
    )


    pred_file = (

        GSEA_DIR
        / f"{drug_file}_gsea_pred.csv"
    )


    if not truth_file.exists():
        continue


    if not pred_file.exists():
        continue


    truth = pd.read_csv(
        truth_file
    )


    pred = pd.read_csv(
        pred_file
    )


    truth_hits = get_correct_hits(
        truth
    )


    pred_hits = get_correct_hits(
        pred
    )


    # =====================================================
    # Ground-truth positive
    #
    # 与 CRISP 原实验类似：
    # 至少一个 disease-related pathway
    # 按正确方向显著富集
    # =====================================================

    is_positive = (
        len(
            truth_hits
        )
        > 0
    )


    if is_positive:

        positive_drugs.append(
            drug
        )


    # =====================================================
    # DiffCRISP signed pathway score
    #
    # 正确方向的 |NES| 相加
    # score 越高越好
    # =====================================================

    score = sum(

        abs(nes)

        for nes
        in pred_hits.values()
    )


    rows.append({

        "drug":
            drug,

        "score":
            score,

        "reference_positive":
            is_positive,

        "n_truth_hits":
            len(
                truth_hits
            ),

        "n_pred_hits":
            len(
                pred_hits
            ),

        "truth_pathways":
            ";".join(
                truth_hits.keys()
            ),

        "pred_pathways":
            ";".join(
                pred_hits.keys()
            ),
    })


# =========================================================
# Drug ranking
# =========================================================

ranking = pd.DataFrame(
    rows
)


ranking = ranking.sort_values(

    "score",

    ascending=False
)


ranking.to_csv(

    ROOT
    / "drug_ranking.csv",

    index=False
)


# =========================================================
# Random baseline
# =========================================================

n_valid = len(
    ranking
)


n_positive = int(

    ranking[
        "reference_positive"
    ].sum()
)


baseline = (

    n_positive
    / n_valid

    if n_valid > 0

    else np.nan
)


print(
    "\nReference positive drugs:",
    n_positive
)


print(
    "Valid drugs:",
    n_valid
)


print(
    "Random baseline:",
    baseline
)


# =========================================================
# Precision@K
#
# 和 CRISP 原筛药逻辑一样：
# 只考虑 score > 0 的候选药
# =========================================================

predicted_drugs = (

    ranking[

        ranking[
            "score"
        ]
        > 0

    ]["drug"]

    .tolist()
)


positive_set = set(
    positive_drugs
)


precision_rows = []


for k in [

    10,
    20,
    30,
    40,
    50

]:

    topk = predicted_drugs[
        :k
    ]


    hits = len(

        set(topk)

        &

        positive_set
    )


    precision = (
        hits / k
    )


    precision_rows.append({

        "K":
            k,

        "Precision":
            precision,

        "Hits":
            hits,

        "Random_baseline":
            baseline
    })


    print(

        f"Precision@{k}:",
        round(
            precision,
            4
        )
    )


precision_df = pd.DataFrame(
    precision_rows
)


precision_df.to_csv(

    ROOT
    / "precision_at_k.csv",

    index=False
)


# =========================================================
# Pathway precision / recall
# =========================================================

pathway_rows = []


for _, row in ranking.iterrows():

    if not row[
        "reference_positive"
    ]:

        continue


    truth_set = set(

        x
        for x
        in str(
            row["truth_pathways"]
        ).split(";")
        if x
        and x != "nan"
    )


    pred_set = set(

        x
        for x
        in str(
            row["pred_pathways"]
        ).split(";")
        if x
        and x != "nan"
    )


    intersection = (

        truth_set
        & pred_set
    )


    if len(
        pred_set
    ) > 0:

        precision = (

            len(intersection)
            / len(pred_set)
        )

    else:

        precision = 0.0


    if len(
        truth_set
    ) > 0:

        recall = (

            len(intersection)
            / len(truth_set)
        )

    else:

        recall = 0.0


    pathway_rows.append({

        "drug":
            row["drug"],

        "pathway_precision":
            precision,

        "pathway_recall":
            recall,

        "n_truth":
            len(truth_set),

        "n_pred":
            len(pred_set),

        "n_overlap":
            len(intersection),
    })


pathway_df = pd.DataFrame(
    pathway_rows
)


pathway_df = pathway_df.sort_values(

    "pathway_precision",

    ascending=False
)


pathway_df.to_csv(

    ROOT
    / "pathway_precision.csv",

    index=False
)


print(
    "\nScreening 完成"
)