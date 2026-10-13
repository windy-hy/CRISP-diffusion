from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MODEL = (
    ROOT /
    "model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2_64"
)

OUT = Path(__file__).resolve().parent

SPLITS = [
    "split",
    "split2",
    "split3"
]

SEEDS = [
    "seed0",
    "seed42",
    "seed123"
]

CFGS = [
    1.0,
    1.5,
    2.0,
    3.0
]

DDIM = 20

def read_result(txt):

    rows = []

    with open(
        txt,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            if p[0] == "Epoch":
                continue

            try:
                rows.append(
                    [
                        int(p[0]),
                        float(p[1]),
                        float(p[2]),
                        float(p[3])
                    ]
                )
            except:
                continue

    df = pd.DataFrame(
        rows,
        columns=[
            "epoch",
            "R2_DE",
            "Pearson_Delta_DE",
            "Sinkhorn_DE"
        ]
    )


    if len(df)==0:
        return None

    # 优先300
    for e in [300,200]:

        tmp = df[
            df["epoch"]==e
        ]
        if len(tmp)>0:
            return tmp.iloc[0]

    return df.iloc[-1]

results=[]

for split in SPLITS:

    for seed in SEEDS:
        seed_dir = (
            MODEL /
            split /
            seed
        )

        for cfg in CFGS:
            txt = (
                seed_dir /
                f"{split}_{cfg}_ddim{DDIM}_results.txt"
            )

            if not txt.exists():
                print(
                    "missing:",
                    txt
                )
                continue
            r = read_result(txt)

            if r is None:
                continue

            results.append(
                {

                    "split":split,
                    "seed":seed,
                    "cfg":cfg,
                    "ddim_steps":DDIM,
                    "epoch_used":int(r["epoch"]),
                    "R2_DE":r["R2_DE"],
                    "Pearson_Delta_DE":
                        r["Pearson_Delta_DE"],
                    "Sinkhorn_DE":
                        r["Sinkhorn_DE"],
                    "file":txt.name
                }
            )

raw = pd.DataFrame(results)

raw.to_csv(
    OUT /
    "cfg_sensitivity_raw.csv",
    index=False
)

print(
    "raw:",
    len(raw)
)


split_mean = (
    raw
    .groupby(
        [
            "cfg",
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
    OUT /
    "cfg_sensitivity_split_mean.csv",
    index=False
)


final=(

    split_mean
    .groupby(
        "cfg",
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
    OUT /
    "cfg_sensitivity.csv",
    index=False
)

print("\nFinal:")
print(final)