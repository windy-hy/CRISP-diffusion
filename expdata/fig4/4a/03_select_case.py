from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


DATA = "../../../data/nips/nips_pp_scFM_resplit.h5ad"

DIFF_DIR = Path("./nips_predictions_diffcrisp")
CRISP_DIR = Path("./nips_predictions_crisp")

SUMMARY = "./response_analysis/responder_proportion.csv"

OUT = Path("./response_analysis/known_drug_cases")
OUT.mkdir(parents=True, exist_ok=True)

TOP = 50
Q = 0.95


# =========================
# 想看的药物
# =========================

DRUGS = [
    "Dactolisib",
    "Palbociclib",
    "Idelalisib",
    "Cerdulatinib",
]


def dense(x):
    return (
        x.toarray()
        if sp.issparse(x)
        else np.asarray(x)
    )


def load_npz(root, file, split):
    p = Path(file)

    if not p.exists():
        p = root / str(split) / p.name

    return np.load(p)


# =========================
# 找候选case
# =========================

summary = pd.read_csv(SUMMARY)

cand = summary[
    summary["drug"].isin(DRUGS)
].copy()

print("\n找到的case：")

print(
    cand[
        [
            "split",
            "cell_type",
            "drug",
            "dose",
            "true_prop",
            "crisp_prop",
            "diffcrisp_prop",
            "crisp_error",
            "diffcrisp_error",
        ]
    ].to_string(index=False)
)

cand.to_csv(
    OUT / "known_drug_summary.csv",
    index=False
)


# =========================
# manifest
# =========================

dm = pd.read_csv(
    DIFF_DIR / "manifest.csv"
)

cm = pd.read_csv(
    CRISP_DIR / "manifest.csv"
)

keys = [
    "split",
    "cell_type",
    "drug",
    "dose",
]

manifest = dm.merge(
    cm,
    on=keys,
    suffixes=("_diff", "_crisp")
)


adata = sc.read_h5ad(DATA)


# =========================
# 每个case导出response score
# =========================

for _, case in cand.iterrows():

    matched = manifest[
        (manifest["split"] == case["split"])
        & (manifest["cell_type"] == case["cell_type"])
        & (manifest["drug"] == case["drug"])
        & np.isclose(
            manifest["dose"].astype(float),
            float(case["dose"])
        )
    ]

    if len(matched) != 1:
        print(
            "跳过：",
            case["drug"],
            "匹配数量 =",
            len(matched)
        )
        continue

    row = matched.iloc[0]

    diff = load_npz(
        DIFF_DIR,
        row["file_diff"],
        row["split"]
    )

    crisp = load_npz(
        CRISP_DIR,
        row["file_crisp"],
        row["split"]
    )


    ctrl = dense(
        adata[
            diff["ctrl_idx"]
        ].X
    )

    true = dense(
        adata[
            diff["treated_idx"]
        ].X
    )

    pred_d = dense(
        diff["pred"]
    )

    pred_c = dense(
        crisp["pred"]
    )


    # =====================
    # response direction
    # =====================

    ctrl_mean = ctrl.mean(0)

    delta = (
        true.mean(0)
        - ctrl_mean
    )

    idx = np.argsort(
        np.abs(delta)
    )[-TOP:]

    direction = delta[idx]
    direction /= np.linalg.norm(direction)


    def score(x):
        return (
            x[:, idx]
            - ctrl_mean[idx]
        ) @ direction


    scores = {
        "Control": score(ctrl),
        "Observed": score(true),
        "CRISP": score(pred_c),
        "DiffCRISP": score(pred_d),
    }

    threshold = np.quantile(
        scores["Control"],
        Q
    )


    result = pd.concat(
        [
            pd.DataFrame({
                "score": values,
                "group": group,
            })
            for group, values
            in scores.items()
        ],
        ignore_index=True
    )

    result["threshold"] = threshold
    result["split"] = case["split"]
    result["cell_type"] = case["cell_type"]
    result["drug"] = case["drug"]
    result["dose"] = case["dose"]


    name = (
        str(case["split"])
        + "_"
        + str(case["cell_type"])
        + "_"
        + str(case["drug"])
        + "_"
        + str(case["dose"])
    )

    name = (
        name
        .replace(" ", "_")
        .replace("/", "_")
    )


    result.to_csv(
        OUT / f"{name}.csv",
        index=False
    )


    print(
        "\n",
        case["cell_type"],
        "|",
        case["drug"],
        "|",
        case["dose"]
    )

    for group, values in scores.items():

        prop = np.mean(
            values > threshold
        )

        print(
            group,
            "=",
            round(prop, 3)
        )


print(
    "\n全部保存到：",
    OUT
)