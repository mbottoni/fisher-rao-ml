"""Generate figures for the FR noisy-label paper (multi-dataset)."""

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

NOISE_ORDER = ["clean", "sym_20", "sym_40", "sym_60", "sym_80", "asym_40"]
NOISE_LABELS = {
    "clean": "0%", "sym_20": "sym 20%", "sym_40": "sym 40%",
    "sym_60": "sym 60%", "sym_80": "sym 80%",
    "asym_20": "asym 20%", "asym_40": "asym 40%", "asym_60": "asym 60%",
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
DATASET_TITLES = {
    "digits": "UCI Digits / MLP", "mnist": "MNIST / MLP",
    "fashion_mnist": "FashionMNIST / MLP", "cifar10": "CIFAR-10 / ConvNet",
}
CIFAR10_NOISE_ORDER = ["clean", "sym_20", "sym_40", "sym_60", "asym_20", "asym_40", "asym_60"]


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def save_accuracy_curves() -> None:
    agg = read_rows(RESULTS / "noisy_label_aggregated.csv")
    datasets = [d for d in ["digits", "mnist", "fashion_mnist", "cifar10"]
                if any(r["dataset"] == d for r in agg)]
    if not datasets:
        print("no data for accuracy curves")
        return
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5.0 * n_ds, 3.8), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets, strict=True):
        noise_order = CIFAR10_NOISE_ORDER if dataset == "cifar10" else NOISE_ORDER
        rows = [r for r in agg if r["dataset"] == dataset]
        for obj in OBJ_ORDER:
            obj_rows = {r["noise_regime"]: r for r in rows if r["objective"] == obj}
            xs, ys, stds = [], [], []
            for i, nr in enumerate(noise_order):
                if nr in obj_rows:
                    xs.append(i)
                    ys.append(float(obj_rows[nr]["eval_accuracy_mean"]))
                    stds.append(float(obj_rows[nr]["eval_accuracy_std"]))
            if not xs:
                continue
            xs_a, ys_a, stds_a = np.array(xs), np.array(ys), np.array(stds)
            ax.plot(xs_a, ys_a, label=OBJ_LABELS[obj], color=OBJ_COLORS[obj],
                    marker=OBJ_MARKERS[obj], markersize=5, linewidth=1.5)
            ax.fill_between(
                xs_a, ys_a - stds_a, ys_a + stds_a,
                alpha=0.08, color=OBJ_COLORS[obj]
            )

        ax.set_xticks(range(len(noise_order)))
        ax.set_xticklabels(
            [NOISE_LABELS[n] for n in noise_order], rotation=25, ha="right", fontsize=8
        )
        ax.set_ylabel("Test accuracy" if dataset == datasets[0] else "")
        ax.set_title(DATASET_TITLES.get(dataset, dataset), fontsize=9)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
        if dataset == datasets[-1]:
            ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Test accuracy under label noise (MLP vs ConvNet)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "noisy_label_accuracy_curves.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved noisy_label_accuracy_curves.pdf")


def save_gain_heatmap() -> None:
    sig = read_rows(RESULTS / "noisy_label_significance.csv")
    datasets = [d for d in ["digits", "mnist", "fashion_mnist", "cifar10"]
                if any(r["dataset"] == d for r in sig)]
    if not datasets:
        print("no data for gain heatmap")
        return
    objs = [o for o in OBJ_ORDER if o != "kl"]

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(6.0 * n_ds, 3.8))
    if n_ds == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets, strict=True):
        noise_regimes = CIFAR10_NOISE_ORDER if dataset == "cifar10" else NOISE_ORDER
        mat = np.full((len(objs), len(noise_regimes)), np.nan)
        for r in sig:
            if r["dataset"] != dataset or r["objective"] not in objs:
                continue
            if r["noise_regime"] not in noise_regimes:
                continue
            i = objs.index(r["objective"])
            j = noise_regimes.index(r["noise_regime"])
            mat[i, j] = float(r["eval_accuracy_oriented_gain"])

        vmax = max(np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 0.2, 0.01)
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(noise_regimes)))
        ax.set_xticklabels(
            [NOISE_LABELS[n] for n in noise_regimes], rotation=25, ha="right", fontsize=8
        )
        ax.set_yticks(range(len(objs)))
        ax.set_yticklabels([OBJ_LABELS[o] for o in objs], fontsize=9)
        for i in range(len(objs)):
            for j in range(len(noise_regimes)):
                if not np.isnan(mat[i, j]):
                    txt_color = "black" if abs(mat[i, j]) < vmax * 0.6 else "white"
                    ax.text(
                        j, i, f"{mat[i, j]:+.3f}",
                        ha="center", va="center", fontsize=7, color=txt_color
                    )
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax.set_title(
            f"{DATASET_TITLES.get(dataset, dataset)}\naccuracy gain vs CE (KL)", fontsize=8.5
        )

    fig.tight_layout()
    fig.savefig(FIGURES / "noisy_label_gain_heatmap.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved noisy_label_gain_heatmap.pdf")


def save_cifar10_ece() -> None:
    """ECE calibration comparison on CIFAR-10 ConvNet."""
    path = RESULTS / "cifar10_noisy_label_aggregated.csv"
    if not path.exists():
        print("cifar10_noisy_label_aggregated.csv not found — skipping")
        return

    rows = read_rows(path)
    data: dict[str, dict[str, float]] = {}
    for r in rows:
        data.setdefault(r["noise_regime"], {})[r["objective"]] = float(r["eval_ece_mean"])

    noise_regimes = [nr for nr in CIFAR10_NOISE_ORDER if nr in data]
    focus = ["kl", "fisher_rao", "hellinger", "gce", "mae"]
    available = [o for o in focus if any(data[nr].get(o) is not None for nr in noise_regimes)]
    if not available:
        print("no ECE data — skipping")
        return

    x = np.arange(len(noise_regimes))
    width = 0.14
    n = len(available)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, obj in enumerate(available):
        vals = [data[nr].get(obj, np.nan) for nr in noise_regimes]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=OBJ_LABELS[obj], color=OBJ_COLORS[obj])

    ax.set_xlabel("Noise Regime", fontsize=12)
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=12)
    ax.set_title("CIFAR-10 ConvNet: Calibration (ECE) — lower is better", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([NOISE_LABELS.get(nr, nr) for nr in noise_regimes], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES / "cifar10_ece.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {out.name}")


def main() -> None:
    save_accuracy_curves()
    save_gain_heatmap()
    save_cifar10_ece()


if __name__ == "__main__":
    main()
