"""Generate figures for the FR-Contrastive paper."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

OBJ_LABELS = {"nt_xent": "NT-Xent (KL)", "fr_contrastive": "FR-Contrastive"}
OBJ_COLORS = {"nt_xent": "tab:blue", "fr_contrastive": "tab:orange"}
OBJ_MARKERS = {"nt_xent": "o", "fr_contrastive": "s"}


def save_fn_curve() -> None:
    agg = list(csv.DictReader((RESULTS / "fr_contrastive_agg.csv").open()))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for obj in ("nt_xent", "fr_contrastive"):
        rows = sorted([r for r in agg if r["objective"] == obj], key=lambda r: float(r["fn_rate"]))
        xs = [float(r["fn_rate"]) for r in rows]
        ys = [float(r["knn_accuracy_mean"]) for r in rows]
        stds = [float(r["knn_accuracy_std"]) for r in rows]
        xs, ys, stds = np.array(xs), np.array(ys), np.array(stds)
        ax.plot(xs, ys, label=OBJ_LABELS[obj], color=OBJ_COLORS[obj],
                marker=OBJ_MARKERS[obj], markersize=5)
        ax.fill_between(xs, ys - stds, ys + stds, alpha=0.15, color=OBJ_COLORS[obj])

    ax.set_xlabel("False-negative injection rate")
    ax.set_ylabel("5-NN accuracy")
    ax.set_title("Robustness to false negatives (UCI Digits)")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.01, 0.32)
    fig.tight_layout()
    fig.savefig(FIGURES / "fr_contrastive_fn_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fr_contrastive_fn_curve.pdf")


def main() -> None:
    save_fn_curve()


if __name__ == "__main__":
    main()
