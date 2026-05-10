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

DATASET_TITLES = {
    "digits": "UCI Digits",
    "mnist": "MNIST",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def build_matrix(rows: list[dict[str, str]], conds: list[str]) -> tuple[np.ndarray, np.ndarray]:
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
    return mat_fr, mat_cka


def save_pairwise_heatmap() -> None:
    datasets = ["digits", "mnist"]
    pairwise_files = {
        "digits": RESULTS / "fr_rd_digits_pairwise.csv",
        "mnist": RESULTS / "fr_rd_mnist_pairwise.csv",
    }
    available = [d for d in datasets if pairwise_files[d].exists()]
    if not available:
        pairwise_files = {"digits": RESULTS / "fr_rd_pairwise.csv"}
        available = ["digits"]

    conds = COND_ORDER
    labels = [COND_LABELS[c] for c in conds]
    n_datasets = len(available)

    # 2 columns per dataset (FR-RD, CKA); if two datasets → 2×2 grid
    fig, axes = plt.subplots(n_datasets, 2, figsize=(11, 4.5 * n_datasets))
    if n_datasets == 1:
        axes = axes[np.newaxis, :]

    for row_idx, dataset in enumerate(available):
        rows = read_rows(pairwise_files[dataset])
        mat_fr, mat_cka = build_matrix(rows, conds)
        ds_title = DATASET_TITLES.get(dataset, dataset)

        for col_idx, (mat, col_title, cmap) in enumerate([
            (mat_fr, f"FR-RD — {ds_title}", "YlOrRd"),
            (mat_cka, f"Linear CKA — {ds_title}", "YlGnBu"),
        ]):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(mat, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(conds)))
            ax.set_yticks(range(len(conds)))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            for ii in range(len(conds)):
                for jj in range(len(conds)):
                    ax.text(jj, ii, f"{mat[ii, jj]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if mat[ii, jj] > mat.max() * 0.6 else "black")
            plt.colorbar(im, ax=ax)
            ax.set_title(col_title, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "fr_rd_pairwise_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fr_rd_pairwise_heatmap.pdf")


def save_trajectory_curves() -> None:
    traj_file = RESULTS / "fr_rd_digits_trajectory.csv"
    if not traj_file.exists():
        traj_file = RESULTS / "fr_rd_trajectory.csv"

    rows = read_rows(traj_file)
    by_cond_step: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        by_cond_step[(r["condition"], int(r["step"]))].append(float(r["fr_rd_to_final"]))

    all_steps = sorted({int(r["step"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond in COND_ORDER:
        means = np.array([np.mean(by_cond_step[(cond, s)]) for s in all_steps])
        stds = np.array([np.std(by_cond_step[(cond, s)]) for s in all_steps])
        ax.plot(all_steps, means, label=COND_LABELS[cond], color=COND_COLORS[cond])
        ax.fill_between(all_steps, means - stds, means + stds,
                        alpha=0.15, color=COND_COLORS[cond])

    ax.set_xlabel("Training step")
    ax.set_ylabel("FR-RD to final model")
    ax.set_title("Training dynamics: FR-RD to converged model (UCI Digits)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "fr_rd_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fr_rd_trajectory.pdf")


def save_scatter_acc_diff() -> None:
    datasets = {
        "digits": RESULTS / "fr_rd_digits_pairwise.csv",
        "mnist": RESULTS / "fr_rd_mnist_pairwise.csv",
    }
    available = {k: v for k, v in datasets.items() if v.exists()}
    if not available:
        available = {"digits": RESULTS / "fr_rd_pairwise.csv"}

    n_datasets = len(available)
    fig, axes = plt.subplots(n_datasets, 2, figsize=(9, 4 * n_datasets))
    if n_datasets == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (dataset, path) in enumerate(available.items()):
        rows = read_rows(path)
        fr_rd = np.array([float(r["fr_rd"]) for r in rows])
        cka = np.array([float(r["cka"]) for r in rows])
        acc_diff = np.array([float(r["acc_diff"]) for r in rows])
        colors = ["tab:blue" if r["cond_a"] == r["cond_b"] else "tab:red" for r in rows]
        ds_title = DATASET_TITLES.get(dataset, dataset)

        for col_idx, (metric, xlabel, title_prefix) in enumerate([
            (fr_rd, "FR-RD", "FR-RD vs |accuracy diff|"),
            (1 - cka, "1 - CKA", "1-CKA vs |accuracy diff|"),
        ]):
            ax = axes[row_idx, col_idx]
            ax.scatter(metric, acc_diff, c=colors, alpha=0.4, s=15)
            corr = np.corrcoef(metric, acc_diff)[0, 1]
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("|accuracy difference|", fontsize=9)
            ax.set_title(f"{title_prefix} — {ds_title}\nr = {corr:.3f}", fontsize=9)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', label='same condition'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red',
               label='different condition'),
    ]
    axes[0, 0].legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fr_rd_acc_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fr_rd_acc_scatter.pdf")


def save_ood_comparison() -> None:
    ood_file = RESULTS / "fr_rd_digits_ood.csv"
    if not ood_file.exists():
        print("fr_rd_digits_ood.csv not found, skipping OOD figure")
        return
    rows = read_rows(ood_file)
    conds = ["clean", "fr_loss", "smoothed", "noisy_30", "noisy_60"]
    cond_labels_ood = {
        "clean": "CE (clean)", "fr_loss": "FR (clean)", "smoothed": "LS (clean)",
        "noisy_30": "CE 30% noise", "noisy_60": "CE 60% noise",
    }
    global_sep = {c: [] for c in conds}
    cc_sep = {c: [] for c in conds}
    for r in rows:
        c = r.get("condition")
        if c in conds:
            global_sep[c].append(float(r["separation"]))
            if "cc_separation" in r:
                cc_sep[c].append(float(r["cc_separation"]))

    x = np.arange(len(conds))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    g_means = [np.mean(global_sep[c]) for c in conds]
    g_stds = [np.std(global_sep[c]) for c in conds]
    cc_means = [np.mean(cc_sep[c]) if cc_sep[c] else 0 for c in conds]
    cc_stds = [np.std(cc_sep[c]) if cc_sep[c] else 0 for c in conds]

    ax.bar(x - width / 2, g_means, width, yerr=g_stds, capsize=4,
           label="Global centroid", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, cc_means, width, yerr=cc_stds, capsize=4,
           label="Class-conditional centroid", color="tab:orange", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([cond_labels_ood[c] for c in conds], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean separation (OOD score − ID score)")
    ax.set_title(
        "OOD detection: global vs class-conditional centroid\n(UCI Digits ID vs MNIST OOD)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fr_rd_ood_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fr_rd_ood_comparison.pdf")


def main() -> None:
    save_pairwise_heatmap()
    save_trajectory_curves()
    save_scatter_acc_diff()
    save_ood_comparison()


if __name__ == "__main__":
    main()
