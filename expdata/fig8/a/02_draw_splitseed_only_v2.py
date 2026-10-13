from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

BASE = Path(__file__).resolve().parent

# ============================================================
# IMPORTANT:
# Only use split-seed summary data.
# Do NOT use ablation_metrics_group.csv.
# Each learned model therefore has 3 splits × 3 seeds = 9 values.
# MeanShift has one value per split and is shown only as a dashed baseline.
# ============================================================
INPUT_CSV = BASE / "ablation_metrics_split_seed.csv"

# Gaussian sigma=0.1:
# The script tries these names in order because your merged CSV may use
# "Gaussian_sigma", "Gaussian_sigma0.1", or another explicit sigma-0.1 label.
GAUSSIAN_CANDIDATES = [
    "Gaussian_sigma",
    "Gaussian_sigma0.1",
    "Gaussian_sigma0.10",
    "Gaussian",
]

DET_MODEL = "Deterministic"
DIFF_MODEL = "Diffusion"
MEANSHIFT_MODEL = "MeanShift"

METRICS = [
    ("r2score_de", r"$R^2$"),
    ("pearson_delta_de", r"$Pr_{\Delta}$"),
    ("sinkhorn_de", "Sinkhorn"),
    ("centered_sinkhorn_de", "Centered Sinkhorn"),
    ("variance_log_mae_de", "Variance log-MAE"),
    ("responder_error", "Responder error"),
]

COLORS = {
    DET_MODEL: "#BDBDBD",
    "GAUSSIAN": "#E69F00",
    DIFF_MODEL: "#0072B2",
}

OUTPUT_PREFIX = "Fig8a_core_generator_ablation_splitseed"


def detect_gaussian_model(df):
    available = set(df["model"].astype(str).unique())
    for name in GAUSSIAN_CANDIDATES:
        if name in available:
            print(f"[INFO] Gaussian rows: {name}")
            return name
    raise ValueError(
        "Cannot find Gaussian sigma=0.1 rows.\n"
        f"Available model names: {sorted(available)}"
    )


def paired_wilcoxon(df, metric, left, right):
    """
    Pair by split + seed only.
    This is intentionally different from the old group-level test.
    """
    x = df.pivot_table(
        index=["split", "seed"],
        columns="model",
        values=metric,
        aggfunc="mean",
    )

    if left not in x.columns or right not in x.columns:
        return np.nan

    y = x[[left, right]].dropna()

    if len(y) < 2:
        return np.nan

    try:
        return float(
            wilcoxon(
                y[left],
                y[right],
            ).pvalue
        )
    except ValueError:
        return np.nan


def meanshift_mean(df, metric):
    """
    MeanShift has seed=-1 and one row per split.
    Use the mean of the three split-level values as the dashed reference.
    """
    x = df.loc[
        df["model"] == MEANSHIFT_MODEL,
        metric,
    ].dropna()

    if len(x) == 0:
        return np.nan

    return float(x.mean())


def padded_ylim(values, baseline=np.nan):
    """
    Keep the axis driven by the 9 split-seed values plus the MeanShift baseline,
    not hundreds of perturbation-level outliers.
    """
    vals = []
    for arr in values:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        vals.extend(arr.tolist())

    if np.isfinite(baseline):
        vals.append(float(baseline))

    if not vals:
        return None

    lo = min(vals)
    hi = max(vals)
    span = hi - lo

    if span <= 1e-12:
        span = max(abs(hi), 1.0) * 0.15

    pad = 0.10 * span

    # For naturally non-negative metrics, keep the lower bound near zero.
    lower = lo - pad
    upper = hi + pad

    return lower, upper


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    gaussian_model = detect_gaussian_model(df)

    main_models = [
        DET_MODEL,
        gaussian_model,
        DIFF_MODEL,
    ]

    labels = [
        "Det.",
        r"Gaussian ($\sigma$=0.1)",
        "Diffusion",
    ]

    plot_colors = [
        COLORS[DET_MODEL],
        COLORS["GAUSSIAN"],
        COLORS[DIFF_MODEL],
    ]

    # Only the three learned models + MeanShift baseline.
    df = df[
        df["model"].isin(
            main_models + [MEANSHIFT_MODEL]
        )
    ].copy()

    # Check expected split-seed counts.
    print("\n[INFO] Counts")
    print(
        df[df["model"].isin(main_models)]
        .groupby("model")
        .size()
    )

    # p-values on 9 paired split-seed observations.
    stat = {}
    learned_df = df[df["model"].isin(main_models)]

    for metric, _ in METRICS:
        stat[metric] = {
            DET_MODEL: paired_wilcoxon(
                learned_df,
                metric,
                DIFF_MODEL,
                DET_MODEL,
            ),
            gaussian_model: paired_wilcoxon(
                learned_df,
                metric,
                DIFF_MODEL,
                gaussian_model,
            ),
        }

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.2, 7.2),
    )
    axes = axes.flatten()

    for ax, (metric, ylabel) in zip(
        axes,
        METRICS,
    ):
        data = [
            df.loc[
                df["model"] == model,
                metric,
            ].dropna().astype(float).to_numpy()
            for model in main_models
        ]

        bp = ax.boxplot(
            data,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            # No individual points: only box distribution + mean diamond.
            boxprops=dict(linewidth=1.25),
            whiskerprops=dict(linewidth=1.15),
            capprops=dict(linewidth=1.15),
            medianprops=dict(
                linewidth=1.5,
                color="#D55E00",
            ),
        )

        # Box colors
        for patch, color in zip(
            bp["boxes"],
            plot_colors,
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.20)
            patch.set_edgecolor(color)

        whisker_colors = [
            plot_colors[0], plot_colors[0],
            plot_colors[1], plot_colors[1],
            plot_colors[2], plot_colors[2],
        ]
        for obj, color in zip(
            bp["whiskers"],
            whisker_colors,
        ):
            obj.set_color(color)

        cap_colors = whisker_colors
        for obj, color in zip(
            bp["caps"],
            cap_colors,
        ):
            obj.set_color(color)

        # Mean only; no split/seed point cloud.
        for i, vals in enumerate(data, start=1):
            if len(vals):
                ax.scatter(
                    i,
                    np.mean(vals),
                    marker="D",
                    s=38,
                    color="black",
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=5,
                )

        # MeanShift dashed baseline.
        ms = meanshift_mean(
            df,
            metric,
        )
        if np.isfinite(ms):
            ax.axhline(
                ms,
                color="#CC3311",
                linestyle=(0, (5, 3)),
                linewidth=1.45,
                zorder=1,
            )

            ax.text(
                0.985,
                0.045,
                f"MeanShift={ms:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#B52A1D",
                bbox=dict(
                    boxstyle="round,pad=0.16",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )

        # Axis based ONLY on the 9 split-seed values + MeanShift.
        ylim = padded_ylim(
            data,
            baseline=ms,
        )
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(
            labels,
            fontsize=10,
        )
        ax.set_ylabel(
            ylabel,
            fontsize=11,
        )

        p_det = stat[metric][DET_MODEL]
        p_gau = stat[metric][gaussian_model]

        p_det_text = (
            f"{p_det:.3g}"
            if np.isfinite(p_det)
            else "NA"
        )
        p_gau_text = (
            f"{p_gau:.3g}"
            if np.isfinite(p_gau)
            else "NA"
        )

        ax.text(
            0.03,
            0.96,
            f"Diff vs Det: P={p_det_text}\n"
            f"Diff vs Gau: P={p_gau_text}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="none",
                alpha=0.80,
            ),
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.22,
        )

        ax.tick_params(
            axis="both",
            labelsize=10,
        )

    # A-F panel labels
    for ax, letter in zip(
        axes,
        "ABCDEF",
    ):
        ax.text(
            -0.16,
            1.04,
            letter,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="black",
            markersize=6,
            linestyle="None",
            label="Mean across split-seed runs",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#CC3311",
            linestyle=(0, (5, 3)),
            linewidth=1.45,
            label="MeanShift baseline",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.012),
        fontsize=10,
    )

    plt.tight_layout(
        rect=[0, 0.045, 1, 1]
    )

    pdf_path = (
        BASE
        / f"{OUTPUT_PREFIX}.pdf"
    )
    png_path = (
        BASE
        / f"{OUTPUT_PREFIX}.png"
    )

    plt.savefig(
        pdf_path,
        dpi=500,
        bbox_inches="tight",
    )
    plt.savefig(
        png_path,
        dpi=700,
        bbox_inches="tight",
    )
    plt.close()

    print("\n[DONE]")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
