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

    # Win rates: proportion of seeds where OOD score > ID score
    def win_rate(col_ood: str, col_id: str, c: str) -> tuple[float, float]:
        vals = [(float(r[col_ood]) > float(r[col_id])) for r in rows if r.get("condition") == c]
        return (float(np.mean(vals)) if vals else 0.0, float(np.std(vals)) if vals else 0.0)

    # For global and cc we already have the separation column
    def win_rate_sep(col: str, c: str) -> float:
        vals = [(float(r[col]) > 0) for r in rows if r.get("condition") == c and col in r]
        return float(np.mean(vals)) if vals else 0.0

    x = np.arange(len(conds))
    n_methods = 5
    width = 0.14
    offsets = np.linspace(-2 * width, 2 * width, n_methods)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    labels = ["Global FR-RD", "FR-RD CC (ours)", "MSP", "Mahalanobis", "Energy"]
    sep_cols = [
        "separation", "cc_separation", "msp_separation",
        "mahal_separation", "energy_separation",
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    for mi, (col, label, color) in enumerate(zip(sep_cols, labels, colors, strict=True)):
        win_vals = [win_rate_sep(col, c) for c in conds]
        ax.bar(x + offsets[mi], win_vals, width, label=label, color=color, alpha=0.8)

    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [cond_labels_ood[c] for c in conds], rotation=20, ha="right", fontsize=8
    )
    ax.set_ylabel("Win rate (fraction of seeds with correct OOD ranking)")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "OOD detection: 5-method comparison\n(UCI Digits ID vs MNIST-8×8 OOD, 10 seeds)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fr_rd_ood_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fr_rd_ood_comparison.pdf")


def save_finetuning_scatter() -> None:
    path = RESULTS / "fr_rd_finetuning.csv"
    if not path.exists():
        return
    rows = read_rows(path)
    fracs = sorted({float(r["fraction"]) for r in rows})
    mean_frd = []
    std_frd = []
    mean_gap = []
    std_gap = []
    for frac in fracs:
        subset = [r for r in rows if float(r["fraction"]) == frac]
        frds = [float(r["fr_rd_to_ref"]) for r in subset]
        gaps = [float(r["acc_gap"]) for r in subset]
        mean_frd.append(np.mean(frds))
        std_frd.append(np.std(frds))
        mean_gap.append(np.mean(gaps))
        std_gap.append(np.std(gaps))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # left: FR-RD vs fraction
    ax = axes[0]
    ax.errorbar(fracs, mean_frd, yerr=std_frd, fmt="o-", color="tab:blue", capsize=4)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("FR-RD to reference model")
    ax.set_title("FR-RD decreases with more training data")
    ax.grid(alpha=0.3)

    # right: scatter FR-RD vs accuracy gap (5 seeds × 5 fractions, excluding ref)
    ax = axes[1]
    frd_all = [float(r["fr_rd_to_ref"]) for r in rows if float(r["fraction"]) < 1.0]
    gap_all = [float(r["acc_gap"]) for r in rows if float(r["fraction"]) < 1.0]
    frac_all = [float(r["fraction"]) for r in rows if float(r["fraction"]) < 1.0]
    sc = ax.scatter(frd_all, gap_all, c=frac_all, cmap="viridis_r", alpha=0.7, s=40)
    plt.colorbar(sc, ax=ax, label="Data fraction")
    # correlation
    r_val = np.corrcoef(frd_all, gap_all)[0, 1]
    ax.set_xlabel("FR-RD to reference model")
    ax.set_ylabel("Accuracy gap to reference")
    ax.set_title(f"Pearson r = {r_val:.3f}")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIGURES / "fr_rd_finetuning.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fr-rd] saved {out}")


def main() -> None:
    save_pairwise_heatmap()
    save_trajectory_curves()
    save_scatter_acc_diff()
    save_ood_comparison()
    save_finetuning_scatter()


if __name__ == "__main__":
    main()
