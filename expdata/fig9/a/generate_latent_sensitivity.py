from pathlib import Path
import pandas as pd


# =====================================================
# path
# =====================================================

ROOT = Path(__file__).resolve().parents[3]

MODEL_ROOT = ROOT / "model" / "nips"

SAVE_DIR = Path(__file__).resolve().parent

RAW_OUT = SAVE_DIR / "latent_sensitivity_raw.csv"
MEAN_SPLIT_OUT = SAVE_DIR / "latent_sensitivity_split_mean.csv"
FINAL_OUT = SAVE_DIR / "latent_sensitivity.csv"


# =====================================================
# experiments
# =====================================================

latent_dims = [
    32,
    64,
    128,
    256
]

splits = [
    "split",
    "split2",
    "split3"
]

seeds = [
    "seed0",
    "seed42",
    "seed123"
]

# =====================================================
# read one txt
# =====================================================

def read_result(txt):

    rows = []
    with open(
        txt,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line=line.strip()

            if not line:
                continue
            if "Epoch" in line:
                continue


            parts=line.split()


            if len(parts)<4:
                continue

            try:

                rows.append(
                    {
                        "epoch":int(parts[0]),
                        "R2_DE":float(parts[1]),
                        "Pearson_Delta_DE":float(parts[2]),
                        "Sinkhorn_DE":float(parts[3])
                    }
                )

            except:
                continue

    if len(rows)==0:
        return None

    df=pd.DataFrame(rows)


    # 优先300
    if 300 in df["epoch"].values:

        return df[
            df["epoch"]==300
        ].iloc[0]


    # 没300取200
    if 200 in df["epoch"].values:

        return df[
            df["epoch"]==200
        ].iloc[0]


    # 都没有取最大epoch
    return df.sort_values(
        "epoch"
    ).iloc[-1]

raw=[]

for dim in latent_dims:

    model_dir = (
        MODEL_ROOT /
        f"fcfp4_mmd0.0_loss0.5l1l2+0.2_{dim}"
    )

    if not model_dir.exists():

        print(
            "missing model:",
            model_dir
        )
        continue

    print(
        "\nprocessing:",
        model_dir.name
    )

    for split in splits:

        for seed in seeds:
            seed_dir = (
                    model_dir /
                    split /
                    seed
            )
            # 优先1.5 guidance
            files = list(
                seed_dir.glob(
                    "*_1.5_results.txt"
                )
            )
            if len(files) == 0:
                print(
                    "missing:",
                    seed_dir
                )
                continue
            txt = files[0]

            if not txt.exists():

                print(
                    "missing:",
                    txt
                )

                continue



            result = read_result(txt)


            if result is None:

                print(
                    "empty:",
                    txt
                )

                continue



            raw.append(
                {
                    "latent_dim":dim,
                    "split":split,
                    "seed":seed,
                    "epoch_used":int(result["epoch"]),
                    "R2_DE":result["R2_DE"],
                    "Pearson_Delta_DE":
                        result["Pearson_Delta_DE"],
                    "Sinkhorn_DE":
                        result["Sinkhorn_DE"],
                    "file":str(txt)
                }
            )

raw_df=pd.DataFrame(raw)


raw_df.to_csv(
    RAW_OUT,
    index=False
)


print(
    "\nRaw saved:",
    RAW_OUT
)

split_mean=(

    raw_df

    .groupby(
        [
            "latent_dim",
            "split"
        ],
        as_index=False
    )

    [
        [
            "R2_DE",
            "Pearson_Delta_DE",
            "Sinkhorn_DE"
        ]
    ]

    .mean()

)



split_mean.to_csv(
    MEAN_SPLIT_OUT,
    index=False
)


print(
    "Split mean saved:",
    MEAN_SPLIT_OUT
)


final=(
    split_mean
    .groupby(
        [
            "latent_dim"
        ],
        as_index=False
    )
    [
        [
            "R2_DE",
            "Pearson_Delta_DE",
            "Sinkhorn_DE"
        ]
    ]
    .mean()
)



final.rename(
    columns={
        "R2_DE":
            "R2_DE_mean",

        "Pearson_Delta_DE":
            "Pearson_Delta_DE_mean",

        "Sinkhorn_DE":
            "Sinkhorn_DE_mean"
    },
    inplace=True
)

final.to_csv(
    FINAL_OUT,
    index=False
)

print(
    "\nFinal saved:",
    FINAL_OUT
)


print("\n========== Result ==========")

print(final)