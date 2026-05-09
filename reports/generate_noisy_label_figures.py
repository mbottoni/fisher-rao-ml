"""Generate figures for the FR noisy-label paper."""

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

NOISE_ORDER = ["clean", "sym_20", "sym_40", "sym_60", "sym_80", "asym_40"]
NOISE_LABELS = {
    "clean": "0%", "sym_20": "sym 20%", "sym_40": "sym 40%",
    "sym_60": "sym 60%", "sym_80": "sym 80%", "asym_40": "asym 40%",
}
OBJ_ORDER = ["kl", "fisher_rao", "hellinger", "gce", "mae", "sce"]
OBJ_LABELS = {
    "kl": "CE (KL)", "fisher_rao": "Fisher-Rao", "hellinger": "Hellinger",
    "gce": "GCE", "mae": "MAE", "sce": "SCE",
}
OBJ_COLORS = {
    "kl": "tab:blue", "fisher_rao": "tab:orange", "hellinger": "tab:green",
    "gce": "tab:red", "mae": "tab:purple", "sce": "tab:brown",
}
OBJ_MARKERS = {"kl": "o", "fisher_rao": "s", "hellinger": "^", "gce": "D", "mae": "v", "sce": "*"}


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def save_accuracy_curves() -> None:
    agg = read_rows(RESULTS / "noisy_label_aggregated.csv")
    datasets = sorted(set(r["dataset"] for r in agg))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        rows = [r for r in agg if r["dataset"] == dataset]
        for obj in OBJ_ORDER:
            obj_rows = {r["noise_regime"]: r for r in rows if r["objective"] == obj}
            xs, ys, stds = [], [], []
            for i, nr in enumerate(NOISE_ORDER):
                if nr in obj_rows:
                    xs.append(i)
                    ys.append(float(obj_rows[nr]["eval_accuracy_mean"]))
                    stds.append(float(obj_rows[nr]["eval_accuracy_std"]))
            if not xs:
                continue
            xs, ys, stds = np.array(xs), np.array(ys), np.array(stds)
            ax.plot(xs, ys, label=OBJ_LABELS[obj], color=OBJ_COLORS[obj],
                    marker=OBJ_MARKERS[obj], markersize=5)
            ax.fill_between(xs, ys - stds, ys + stds, alpha=0.1, color=OBJ_COLORS[obj])

        ax.set_xticks(range(len(NOISE_ORDER)))
        ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_ORDER], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Test accuracy")
        ax.set_title(dataset)
        ax.legend(fontsize=7)

    fig.suptitle("Test accuracy under symmetric and asymmetric label noise", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "noisy_label_accuracy_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved noisy_label_accuracy_curves.pdf")


def save_gain_heatmap() -> None:
    sig = read_rows(RESULTS / "noisy_label_significance.csv")
    datasets = sorted(set(r["dataset"] for r in sig))
    objs = [o for o in OBJ_ORDER if o != "kl"]
    noise_regimes = NOISE_ORDER

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 3.5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        mat = np.full((len(objs), len(noise_regimes)), np.nan)
        for r in sig:
            if r["dataset"] != dataset:
                continue
            if r["objective"] not in objs:
                continue
            i = objs.index(r["objective"])
            if r["noise_regime"] not in noise_regimes:
                continue
            j = noise_regimes.index(r["noise_regime"])
            mat[i, j] = float(r["eval_accuracy_oriented_gain"])

        vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 0.2
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(noise_regimes)))
        ax.set_xticklabels([NOISE_LABELS[n] for n in noise_regimes], rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(objs)))
        ax.set_yticklabels([OBJ_LABELS[o] for o in objs], fontsize=9)
        for i in range(len(objs)):
            for j in range(len(noise_regimes)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=7,
                            color="black" if abs(mat[i, j]) < vmax * 0.6 else "white")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{dataset}: accuracy gain vs CE (KL)", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "noisy_label_gain_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved noisy_label_gain_heatmap.pdf")


def main() -> None:
    save_accuracy_curves()
    save_gain_heatmap()


if __name__ == "__main__":
    main()
