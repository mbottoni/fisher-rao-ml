"""Generate figures for the wrong-label probability trajectory analysis.

Shows p_k(wrong_label) over training epochs for each objective,
with danger zone boundary at p_k=0.032. Compares sym_40 vs asym_40
to visualize why FR stays in the danger zone longer under asym noise.

Run after: uv run --project . python experiments/wrong_label_prob_analysis.py
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

DANGER_ZONE = 0.032

OBJ_ORDER = ["kl", "fisher_rao", "hellinger", "gce", "mae"]
OBJ_LABELS = {
    "kl": "KL (CE)", "fisher_rao": "Fisher–Rao",
    "hellinger": "Hellinger", "gce": "GCE", "mae": "MAE",
}
OBJ_COLORS = {
    "kl": "tab:blue", "fisher_rao": "tab:red",
    "hellinger": "tab:green", "gce": "tab:orange", "mae": "tab:purple",
}
OBJ_LINES = {
    "kl": "-", "fisher_rao": "--", "hellinger": "-.",
    "gce": ":", "mae": (0, (3, 1, 1, 1)),
}


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_wrong_prob_trajectories(rows: list[dict]) -> None:
    """Line plot: mean p_k(wrong) vs epoch, one panel per noise regime."""
    # Collect mean over seeds per (obj, noise_regime, epoch)
    data: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in rows:
        data[r["noise_regime"]][r["objective"]][int(r["epoch"])].append(
            float(r["mean_wrong_prob"])
        )

    noise_regimes = sorted(data.keys())
    n = len(noise_regimes)
    if n == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, regime in zip(axes, noise_regimes, strict=False):
        for obj in OBJ_ORDER:
            epoch_data = data[regime].get(obj, {})
            if not epoch_data:
                continue
            epochs = sorted(epoch_data.keys())
            means = [np.mean(epoch_data[e]) for e in epochs]
            ax.plot(
                epochs, means,
                label=OBJ_LABELS.get(obj, obj),
                color=OBJ_COLORS.get(obj, "gray"),
                linestyle=OBJ_LINES.get(obj, "-"),
                linewidth=1.8,
            )

        ax.axhline(DANGER_ZONE, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(
            2, DANGER_ZONE + 0.003,
            f"Danger zone boundary ({DANGER_ZONE})",
            fontsize=8, color="black", alpha=0.7,
        )
        label_map = {
            "sym_40": "Sym 40% noise",
            "asym_40": "Asym 40% noise",
            "sym_20": "Sym 20% noise",
            "sym_60": "Sym 60% noise",
        }
        ax.set_title(label_map.get(regime, regime), fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("$p_k$(wrong label)" if ax is axes[0] else "")
        ax.set_ylim(0, 0.25)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Shade danger zone
        ax.axhspan(DANGER_ZONE, 0.25, alpha=0.04, color="red")

    fig.suptitle(
        "Wrong-label probability $p_k$ during training\n"
        "(lower = model pushes away from corrupted label)",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIGURES / "wrong_label_prob_trajectories.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}")


def plot_frac_below_threshold(rows: list[dict]) -> None:
    """Line plot: fraction of noisy samples with p_k < 0.032 (safely below danger zone)."""
    data: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in rows:
        data[r["noise_regime"]][r["objective"]][int(r["epoch"])].append(
            float(r["frac_below_threshold"])
        )

    noise_regimes = sorted(data.keys())
    n = len(noise_regimes)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, regime in zip(axes, noise_regimes, strict=False):
        for obj in OBJ_ORDER:
            epoch_data = data[regime].get(obj, {})
            if not epoch_data:
                continue
            epochs = sorted(epoch_data.keys())
            means = [np.mean(epoch_data[e]) for e in epochs]
            ax.plot(
                epochs, means,
                label=OBJ_LABELS.get(obj, obj),
                color=OBJ_COLORS.get(obj, "gray"),
                linestyle=OBJ_LINES.get(obj, "-"),
                linewidth=1.8,
            )

        label_map = {
            "sym_40": "Sym 40% noise",
            "asym_40": "Asym 40% noise",
        }
        ax.set_title(label_map.get(regime, regime), fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(
            "Fraction of noisy samples\nbelow danger zone" if ax is axes[0] else ""
        )
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Fraction of corrupted-label samples with $p_k < 0.032$ (safe saturation zone)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES / "wrong_label_frac_below.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}")


def main() -> None:
    path = RESULTS / "wrong_label_prob_full.csv"
    if not path.exists():
        print("wrong_label_prob_full.csv not found. Run wrong_label_prob_analysis.py first.")
        return

    rows = read_rows(path)
    if not rows:
        print("Empty CSV.")
        return

    plot_wrong_prob_trajectories(rows)
    plot_frac_below_threshold(rows)
    print("Done.")


if __name__ == "__main__":
    main()
