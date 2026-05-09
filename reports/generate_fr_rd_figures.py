"""Generate figures for the FR Representation Distance paper."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

COND_LABELS = {
    "clean": "CE (clean)",
    "fr_loss": "FR (clean)",
    "smoothed": "LS (clean)",
    "noisy_30": "CE 30% noise",
    "noisy_60": "CE 60% noise",
}
COND_ORDER = ["clean", "fr_loss", "smoothed", "noisy_30", "noisy_60"]
COND_COLORS = {
    "clean": "tab:blue",
    "fr_loss": "tab:orange",
    "smoothed": "tab:green",
    "noisy_30": "tab:red",
    "noisy_60": "tab:purple",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def save_pairwise_heatmap() -> None:
    rows = read_rows(RESULTS / "fr_rd_pairwise.csv")
    conds = COND_ORDER
    n = len(conds)
    mat_fr = np.zeros((n, n))
    mat_cka = np.zeros((n, n))
    counts = np.zeros((n, n))

    for r in rows:
        i = conds.index(r["cond_a"]) if r["cond_a"] in conds else -1
        j = conds.index(r["cond_b"]) if r["cond_b"] in conds else -1
        if i < 0 or j < 0:
            continue
        mat_fr[i, j] += float(r["fr_rd"])
        mat_fr[j, i] += float(r["fr_rd"])
        mat_cka[i, j] += float(r["cka"])
        mat_cka[j, i] += float(r["cka"])
        counts[i, j] += 1
        counts[j, i] += 1

    np.fill_diagonal(counts, 1)
    mat_fr /= counts
    mat_cka /= counts

    labels = [COND_LABELS[c] for c in conds]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, mat, title, cmap in [
        (axes[0], mat_fr, "FR-RD (lower = more similar)", "YlOrRd"),
        (axes[1], mat_cka, "Linear CKA (higher = more similar)", "YlGnBu"),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        for ii in range(n):
            for jj in range(n):
                ax.text(jj, ii, f"{mat[ii, jj]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if mat[ii, jj] > mat.max() * 0.6 else "black")
        plt.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGURES / "fr_rd_pairwise_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fr_rd_pairwise_heatmap.pdf")


def save_trajectory_curves() -> None:
    rows = read_rows(RESULTS / "fr_rd_trajectory.csv")
    by_cond_step: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        by_cond_step[(r["condition"], int(r["step"]))].append(float(r["fr_rd_to_final"]))

    all_steps = sorted({int(r["step"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond in COND_ORDER:
        means = [np.mean(by_cond_step[(cond, s)]) for s in all_steps]
        stds = [np.std(by_cond_step[(cond, s)]) for s in all_steps]
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(all_steps, means, label=COND_LABELS[cond], color=COND_COLORS[cond])
        ax.fill_between(all_steps, means - stds, means + stds,
                        alpha=0.15, color=COND_COLORS[cond])

    ax.set_xlabel("Training step")
    ax.set_ylabel("FR-RD to final model")
    ax.set_title("Training dynamics: FR-RD to converged model")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "fr_rd_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fr_rd_trajectory.pdf")


def save_scatter_acc_diff() -> None:
    rows = read_rows(RESULTS / "fr_rd_pairwise.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    fr_rd = np.array([float(r["fr_rd"]) for r in rows])
    cka = np.array([float(r["cka"]) for r in rows])
    acc_diff = np.array([float(r["acc_diff"]) for r in rows])

    # Color by condition pair type
    colors = []
    for r in rows:
        same = r["cond_a"] == r["cond_b"]
        colors.append("tab:blue" if same else "tab:red")

    for ax, metric, xlabel, title in [
        (axes[0], fr_rd, "FR-RD", "FR-RD vs |accuracy difference|"),
        (axes[1], 1 - cka, "1 - CKA", "1 - CKA vs |accuracy difference|"),
    ]:
        ax.scatter(metric, acc_diff, c=colors, alpha=0.4, s=15)
        corr = np.corrcoef(metric, acc_diff)[0, 1]
        ax.set_xlabel(xlabel)
        ax.set_ylabel("|accuracy difference|")
        ax.set_title(f"{title}\nr = {corr:.3f}", fontsize=9)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', label='same condition'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', label='different condition'),
    ]
    axes[0].legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fr_rd_acc_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fr_rd_acc_scatter.pdf")


def main() -> None:
    save_pairwise_heatmap()
    save_trajectory_curves()
    save_scatter_acc_diff()


if __name__ == "__main__":
    main()
