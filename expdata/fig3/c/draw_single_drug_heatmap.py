import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter


BASE = Path(__file__).resolve().parent
ROOT = (BASE / "../../..").resolve()
sys.path.insert(0, str(ROOT))

CRISP_FILE = BASE / "selected_allgenes_crisp.pt"
DIFF_FILE = BASE / "selected_allgenes_diffcrisp.pt"

TARGET_GROUP = "K562_Cerdulatinib_1.0"
OUT_FILE = BASE / f"heatmap_{TARGET_GROUP}_allgenes.pdf"

# DiffCRISP固定色标
DIFF_VMIN = 1.5
DIFF_VMAX = 2.0


def load_predictions(path):
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    pred = np.asarray(
        data["groups"][TARGET_GROUP]["predictions"],
        dtype=np.float32
    )

    if pred.ndim != 3:
        raise RuntimeError(f"预测形状错误：{pred.shape}")

    return pred


def build_matrix(pred, sinkhorn_dist):
    n_seed = pred.shape[0]
    matrix = np.zeros((n_seed, n_seed), dtype=np.float32)

    tensors = [
        torch.from_numpy(pred[i]).float()
        for i in range(n_seed)
    ]

    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            with torch.inference_mode():
                value = sinkhorn_dist(
                    tensors[i],
                    tensors[j]
                )

            matrix[i, j] = float(value.item())
            matrix[j, i] = matrix[i, j]

    return matrix


def main():
    from CRISP.losses import sinkhorn_dist

    pred_c = load_predictions(CRISP_FILE)
    pred_d = load_predictions(DIFF_FILE)

    print("CRISP：", pred_c.shape)
    print("DiffCRISP：", pred_d.shape)

    mat_c = build_matrix(pred_c, sinkhorn_dist)
    mat_d = build_matrix(pred_d, sinkhorn_dist)

    upper = np.triu_indices(mat_c.shape[0], k=1)

    values_c = mat_c[upper]
    values_d = mat_d[upper]

    mean_c = float(values_c.mean())
    mean_d = float(values_d.mean())

    # CRISP使用自身非对角线范围
    vmin_c = float(values_c.min())
    vmax_c = float(values_c.max())

    print("CRISP范围：", vmin_c, "—", vmax_c)
    print("DiffCRISP实际范围：",
          float(values_d.min()), "—", float(values_d.max()))
    print("DiffCRISP显示范围：", DIFF_VMIN, "—", DIFF_VMAX)

    # 遮掉对角线
    plot_c = mat_c.copy()
    plot_d = mat_d.copy()

    np.fill_diagonal(plot_c, np.nan)
    np.fill_diagonal(plot_d, np.nan)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.15)
    )

    im_c = axes[0].imshow(
        plot_c,
        cmap=cmap,
        vmin=vmin_c,
        vmax=vmax_c,
        interpolation="nearest"
    )

    im_d = axes[1].imshow(
        plot_d,
        cmap=cmap,
        vmin=DIFF_VMIN,
        vmax=DIFF_VMAX,
        interpolation="nearest"
    )

    titles = [
        f"CRISP\nMean distance = {mean_c:.3g}",
        f"DiffCRISP\nMean distance = {mean_d:.3g}"
    ]

    for ax, title, label in zip(
        axes,
        titles,
        ["a", "b"]
    ):
        ax.set_title(title)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Seed")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))

        # ax.text(
        #     -0.17,
        #     1.08,
        #     label,
        #     transform=ax.transAxes,
        #     fontsize=12,
        #     fontweight="bold"
        # )

    # CRISP色条
    cbar_c = fig.colorbar(
        im_c,
        ax=axes[0],
        fraction=0.046,
        pad=0.04
    )
    cbar_c.set_label("Sinkhorn distance")
    cbar_c.ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.3f")
    )

    # 判断DiffCRISP是否有截断
    d_min = float(values_d.min())
    d_max = float(values_d.max())

    if d_min < DIFF_VMIN and d_max > DIFF_VMAX:
        extend = "both"
    elif d_min < DIFF_VMIN:
        extend = "min"
    elif d_max > DIFF_VMAX:
        extend = "max"
    else:
        extend = "neither"

    cbar_d = fig.colorbar(
        im_d,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        extend=extend
    )
    cbar_d.set_label("Sinkhorn distance")
    cbar_d.set_ticks(
        np.arange(1.5, 2.01, 0.1)
    )
    cbar_d.ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    cell, drug, dose = TARGET_GROUP.rsplit("_", 2)

    fig.suptitle(
        f"{cell} · {drug} · Dose {dose}",
        fontsize=10.5,
        y=0.99
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        bottom=0.15,
        top=0.78,
        wspace=0.42
    )

    plt.savefig(
        OUT_FILE,
        bbox_inches="tight",
        pad_inches=0.03
    )

    plt.show()

    print("CRISP平均距离：", mean_c)
    print("DiffCRISP平均距离：", mean_d)
    print("保存：", OUT_FILE)


if __name__ == "__main__":
    main()