from pathlib import Path

import pandas as pd
import scanpy as sc


ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = "sciplex3_pp_hvgenes_scFM_resplit.h5ad"
SPLIT_KEY = "split_drugs3"

adata = sc.read_h5ad(DATA_PATH)
obs = adata.obs.copy()

required = [
    SPLIT_KEY,
    "control",
    "cell_type",
    "condition",
    "dose_val",
]

missing = [x for x in required if x not in obs.columns]

if missing:
    raise KeyError("缺少字段：" + str(missing))


control = pd.to_numeric(
    obs["control"],
    errors="coerce",
)

if control.isna().any():
    text = (
        obs["control"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    control = control.fillna(
        text.map({
            "true": 1,
            "false": 0,
            "control": 1,
            "ctrl": 1,
            "treated": 0,
            "perturbed": 0,
        })
    )

if control.isna().any():
    raise ValueError(
        "control字段包含无法识别的值："
        + str(obs.loc[control.isna(), "control"].unique())
    )


obs["_control"] = control.astype(int)

obs["_split"] = (
    obs[SPLIT_KEY]
    .astype(str)
    .str.strip()
    .str.lower()
)

obs["_cell_type"] = (
    obs["cell_type"]
    .astype(str)
    .str.strip()
)

obs["_drug"] = (
    obs["condition"]
    .astype(str)
    .str.strip()
)

obs["_dose"] = pd.to_numeric(
    obs["dose_val"],
    errors="coerce",
)


print("数据路径：", DATA_PATH)
print("数据形状：", adata.shape)
print("检查划分：", SPLIT_KEY)

print("\nsplit取值：")
print(obs["_split"].value_counts(dropna=False))

print("\n各细胞系在不同split中的细胞数：")
print(
    pd.crosstab(
        obs["_cell_type"],
        obs["_split"],
    )
)


treated = obs[obs["_control"] == 0].copy()
ctrl = obs[obs["_control"] == 1].copy()


print("\n各细胞系对照细胞数量：")
print(
    ctrl.groupby(
        "_cell_type",
        observed=True,
    )
    .size()
    .rename("control_cells")
)


summary = (
    treated.groupby(
        [
            "_cell_type",
            "_split",
            "_drug",
            "_dose",
        ],
        observed=True,
    )
    .size()
    .reset_index(name="cell_count")
    .rename(
        columns={
            "_cell_type": "cell_type",
            "_split": "split",
            "_drug": "condition",
            "_dose": "dose_val",
        }
    )
    .sort_values(
        [
            "cell_type",
            "split",
            "condition",
            "dose_val",
        ]
    )
)

print("\n各细胞系、各split的药物情况：")

for cell_type in sorted(
    treated["_cell_type"].unique()
):
    print("\n" + "=" * 65)
    print("细胞系：", cell_type)
    print("=" * 65)

    cell_df = treated[
        treated["_cell_type"] == cell_type
    ]

    for split_name in sorted(
        cell_df["_split"].unique()
    ):
        part = cell_df[
            cell_df["_split"] == split_name
        ]

        combinations = part[
            [
                "_drug",
                "_dose",
            ]
        ].drop_duplicates()

        print(
            "\n",
            split_name,
            "处理细胞数：",
            len(part),
            "药物数：",
            part["_drug"].nunique(),
            "药物-剂量组合数：",
            len(combinations),
        )

        if split_name == "ood":
            ood_table = (
                part.groupby(
                    "_drug",
                    observed=True,
                )
                .agg(
                    dose_count=(
                        "_dose",
                        "nunique",
                    ),
                    cell_count=(
                        "_drug",
                        "size",
                    ),
                )
                .reset_index()
                .rename(
                    columns={
                        "_drug": "condition",
                    }
                )
                .sort_values("condition")
            )

            print("\nOOD药物：")
            print(
                ood_table.to_string(
                    index=False
                )
            )


print("\nOOD组合明细：")

ood_summary = summary[
    summary["split"] == "ood"
]

print(
    ood_summary.to_string(
        index=False
    )
)
