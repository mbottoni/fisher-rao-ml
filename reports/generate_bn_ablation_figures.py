"""Generate figures for the BN ablation (cifar10_no_bn_ablation.py results).

Creates a heatmap of accuracy drop (with-BN minus without-BN) per
objective × noise regime, and a summary bar chart of mean drop.

Run after cifar10_no_bn_ablation.py and cifar10_noisy_label_benchmark.py complete.
"""

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

NOISE_REGIMES = ["clean", "sym_20", "sym_40", "sym_60", "asym_40"]
NOISE_LABELS = {
    "clean": "Clean", "sym_20": "Sym 20%",
    "sym_40": "Sym 40%", "sym_60": "Sym 60%", "asym_40": "Asym 40%",
}
OBJ_ORDER = ["kl", "fisher_rao", "sce", "hellinger", "gce", "mae"]
OBJ_LABELS = {
    "kl": "KL (CE)", "fisher_rao": "Fisher–Rao", "sce": "SCE",
    "hellinger": "Hellinger", "gce": "GCE", "mae": "MAE",
}


def read_acc(path: Path) -> dict[tuple[str, str, int], float]:
    """Return {(noise_regime, objective, seed): accuracy}."""
    out: dict[tuple[str, str, int], float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[(row["noise_regime"], row["objective"], int(row["seed"]))] = float(
                row["eval_accuracy"]
            )
    return out


def compute_drops(
    with_bn: dict[tuple[str, str, int], float],
    no_bn: dict[tuple[str, str, int], float],
) -> dict[tuple[str, str], list[float]]:
    """Paired seed differences: with_bn - no_bn per (noise_regime, objective)."""
    drops: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (regime, obj, seed), acc in with_bn.items():
        key = (regime, obj)
        if (regime, obj, seed) in no_bn:
            drops[key].append(acc - no_bn[(regime, obj, seed)])
    return drops


def plot_heatmap(drops: dict[tuple[str, str], list[float]]) -> None:
    """Heatmap: objectives × noise regimes, showing mean accuracy drop (pp)."""
    n_obj = len(OBJ_ORDER)
    n_reg = len(NOISE_REGIMES)
    data = np.full((n_obj, n_reg), np.nan)
    for i, obj in enumerate(OBJ_ORDER):
        for j, regime in enumerate(NOISE_REGIMES):
            vals = drops.get((regime, obj), [])
            if vals:
                data[i, j] = np.mean(vals) * 100  # percentage points

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=-50, vmax=0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy drop (pp)\nwith-BN minus without-BN")

    ax.set_xticks(range(n_reg))
    ax.set_xticklabels([NOISE_LABELS[r] for r in NOISE_REGIMES], rotation=20, ha="right")
    ax.set_yticks(range(n_obj))
    ax.set_yticklabels([OBJ_LABELS[o] for o in OBJ_ORDER])

    for i in range(n_obj):
        for j in range(n_reg):
            if not np.isnan(data[i, j]):
                text = f"{data[i, j]:.1f}"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=9, color="white" if data[i, j] < -20 else "black")

    ax.set_title(
        "BN ablation: accuracy drop by removing BatchNorm (pp)\n"
        "5 seeds, CIFAR-10 ConvNet",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES / "bn_ablation_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}")


def plot_mean_drop_bars(drops: dict[tuple[str, str], list[float]]) -> None:
    """Bar chart: mean drop across all regimes per objective, with per-regime points."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(OBJ_ORDER))

    regime_colors = {
        "clean": "tab:blue", "sym_20": "tab:green", "sym_40": "tab:orange",
        "sym_60": "tab:red", "asym_40": "tab:purple",
    }

    mean_drops = []
    for obj in OBJ_ORDER:
        all_vals: list[float] = []
        for regime in NOISE_REGIMES:
            all_vals.extend(drops.get((regime, obj), []))
        mean_drops.append(np.mean(all_vals) * 100 if all_vals else np.nan)

    ax.bar(
        x, mean_drops,
        color=["tab:gray"] * len(OBJ_ORDER), alpha=0.6,
        edgecolor="black", linewidth=0.8,
        label="Mean across regimes",
    )

    for _j, regime in enumerate(NOISE_REGIMES):
        regime_means = []
        for obj in OBJ_ORDER:
            vals = drops.get((regime, obj), [])
            regime_means.append(np.mean(vals) * 100 if vals else np.nan)
        ax.scatter(
            x, regime_means,
            label=NOISE_LABELS[regime],
            color=regime_colors[regime],
            zorder=5, s=40, marker="D",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([OBJ_LABELS[o] for o in OBJ_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Accuracy drop (pp) — with-BN minus without-BN")
    ax.set_title(
        "Differential BN sensitivity: mean accuracy drop on removing BatchNorm\n"
        "(less negative = more robust to BN removal)",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIGURES / "bn_ablation_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}")


def main() -> None:
    bn_path = RESULTS / "cifar10_noisy_label_full.csv"
    nobn_path = RESULTS / "cifar10_no_bn_full.csv"

    for p in (bn_path, nobn_path):
        if not p.exists():
            print(f"missing {p.name}")
            return

    with_bn = read_acc(bn_path)
    no_bn = read_acc(nobn_path)
    drops = compute_drops(with_bn, no_bn)

    if not drops:
        print("No paired (regime, objective, seed) rows found.")
        return

    plot_heatmap(drops)
    plot_mean_drop_bars(drops)
    print("Done.")


if __name__ == "__main__":
    main()
