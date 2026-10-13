from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "model/nips/fcfp4_mmd0.0_loss0.5l1l2+0.2_64"
OUT = Path(__file__).resolve().parent

SPLITS = ["split", "split2", "split3"]
SEEDS = ["seed0", "seed42", "seed123"]
DDIMS = [5, 10, 20, 30, 50, 100, 200]
CFG = 1.5


def read_epoch(txt):
    rows = []
    with open(txt, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if not p or p[0] == "Epoch":
                continue
            try:
                rows.append([int(p[0]), float(p[1]), float(p[2]), float(p[3])])
            except (ValueError, IndexError):
                pass

    df = pd.DataFrame(rows, columns=["epoch", "R2_DE", "Pearson_Delta_DE", "Sinkhorn_DE"])
    if df.empty:
        return None

    for epoch in [300, 200]:
        x = df[df["epoch"] == epoch]
        if not x.empty:
            return x.iloc[0]

    return df.sort_values("epoch").iloc[-1]


rows = []

for split in SPLITS:
    for seed in SEEDS:
        seed_dir = MODEL / split / seed

        for ddim in DDIMS:
            txt = seed_dir / f"{split}_{CFG}_ddim{ddim}_results.txt"

            if not txt.exists():
                print("missing:", txt)
                continue

            r = read_epoch(txt)
            if r is None:
                continue

            rows.append({
                "split": split,
                "seed": seed,
                "cfg": CFG,
                "ddim_steps": ddim,
                "epoch_used": int(r["epoch"]),
                "R2_DE": r["R2_DE"],
                "Pearson_Delta_DE": r["Pearson_Delta_DE"],
                "Sinkhorn_DE": r["Sinkhorn_DE"],
                "file": txt.name
            })

raw = pd.DataFrame(rows)
if raw.empty:
    raise RuntimeError("No result files were found. Check MODEL path and filenames.")

raw.to_csv(OUT / "ddim_sensitivity_raw.csv", index=False)

split_mean = (
    raw.groupby(["ddim_steps", "split"], as_index=False)
       [["R2_DE", "Pearson_Delta_DE", "Sinkhorn_DE"]]
       .mean()
)
split_mean.to_csv(OUT / "ddim_sensitivity_split_mean.csv", index=False)

final = (
    split_mean.groupby("ddim_steps", as_index=False)
              [["R2_DE", "Pearson_Delta_DE", "Sinkhorn_DE"]]
              .mean()
              .rename(columns={
                  "R2_DE": "R2_DE_mean",
                  "Pearson_Delta_DE": "Pearson_Delta_DE_mean",
                  "Sinkhorn_DE": "Sinkhorn_DE_mean"
              })
)
final.to_csv(OUT / "ddim_sensitivity.csv", index=False)

print("\nRaw rows:", len(raw), "(expected 63 = 3 splits × 3 seeds × 7 DDIM)")
print("\nFinal result:")
print(final.to_string(index=False))
print("\nSaved to:", OUT)
