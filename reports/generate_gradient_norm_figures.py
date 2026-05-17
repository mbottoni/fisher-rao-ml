"""Generate gradient norm analysis figures for Direction 1 paper.

Plots gradient norm trajectories for clean vs noisy samples during training,
providing mechanistic evidence for the bounded-gradient hypothesis.
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

OBJ_LABELS = {
    "kl": "CE (KL)", "fisher_rao": "Fisher-Rao", "hellinger": "Hellinger",
    "gce": "GCE", "mae": "MAE", "sce": "SCE",
}
OBJ_COLORS = {
    "kl": "tab:blue", "fisher_rao": "tab:orange", "hellinger": "tab:green",
    "gce": "tab:red", "mae": "tab:purple", "sce": "tab:brown",
}
# Objectives to highlight in the main figure (4 key comparisons)
MAIN_OBJECTIVES = ["kl", "fisher_rao", "gce", "mae"]


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def save_gradient_norm_trajectories() -> None:
    """Figure 1: gradient norm on noisy samples over training epochs."""
    path = RESULTS / "gradient_norm_full.csv"
    if not path.exists():
        print("gradient_norm_full.csv not found — skipping")
        return

    rows = read_rows(path)
    # Group: obj -> sample_type -> epoch -> [grad_norms]
    data: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in rows:
        data[r["objective"]][r["sample_type"]][int(r["epoch"])].append(
            float(r["mean_grad_norm"])
        )

    objectives = [o for o in MAIN_OBJECTIVES if o in data]
    max_epoch = max(
        int(r["epoch"]) for r in rows
    )
    epochs = list(range(max_epoch + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, sample_type, title in [
        (axes[0], "noisy", "Gradient Norm — Noisy Samples"),
        (axes[1], "clean", "Gradient Norm — Clean Samples"),
    ]:
        for obj in objectives:
            if sample_type not in data[obj]:
                continue
            means, stds = [], []
            for ep in epochs:
                vals = data[obj][sample_type][ep]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    means.append(np.nan)
                    stds.append(0.0)
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(epochs, means, color=OBJ_COLORS[obj], label=OBJ_LABELS[obj], linewidth=2)
            ax.fill_between(
                epochs, means - stds, means + stds,
                color=OBJ_COLORS[obj], alpha=0.15
            )
        ax.set_xlabel("Training Epoch", fontsize=12)
        ax.set_ylabel("Gradient Norm (L2)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "CIFAR-10 ConvNet, sym 40% noise: gradient norms on clean vs noisy training samples",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    out = FIGURES / "gradient_norm_trajectories.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {out.name}")


def save_gradient_norm_ratio() -> None:
    """Figure 2: ratio of noisy/clean gradient norm over training."""
    path = RESULTS / "gradient_norm_full.csv"
    if not path.exists():
        return

    rows = read_rows(path)
    data: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in rows:
        data[r["objective"]][r["sample_type"]][int(r["epoch"])].append(
            float(r["mean_grad_norm"])
        )

    objectives = [o for o in ["kl", "fisher_rao", "gce", "mae", "hellinger", "sce"] if o in data]
    max_epoch = max(int(r["epoch"]) for r in rows)
    epochs = list(range(max_epoch + 1))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for obj in objectives:
        noisy_means = []
        clean_means = []
        for ep in epochs:
            n_vals = data[obj]["noisy"][ep]
            c_vals = data[obj]["clean"][ep]
            noisy_means.append(np.mean(n_vals) if n_vals else np.nan)
            clean_means.append(np.mean(c_vals) if c_vals else np.nan)

        noisy_means = np.array(noisy_means)
        clean_means = np.array(clean_means)
        ratio = noisy_means / np.maximum(clean_means, 1e-8)

        ax.plot(epochs, ratio, color=OBJ_COLORS[obj], label=OBJ_LABELS[obj], linewidth=2)

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="ratio=1")
    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("Gradient Norm Ratio (noisy / clean)", fontsize=12)
    ax.set_title(
        "CIFAR-10 ConvNet, sym 40%: noisy/clean gradient norm ratio over training",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES / "gradient_norm_ratio.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {out.name}")


def save_loss_trajectories() -> None:
    """Figure 3: loss values on noisy vs clean samples over training."""
    path = RESULTS / "gradient_norm_full.csv"
    if not path.exists():
        return

    rows = read_rows(path)
    data: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in rows:
        data[r["objective"]][r["sample_type"]][int(r["epoch"])].append(
            float(r["mean_loss"])
        )

    objectives = [o for o in MAIN_OBJECTIVES if o in data]
    max_epoch = max(int(r["epoch"]) for r in rows)
    epochs = list(range(max_epoch + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, sample_type, title in [
        (axes[0], "noisy", "Loss — Noisy Samples"),
        (axes[1], "clean", "Loss — Clean Samples"),
    ]:
        for obj in objectives:
            means = []
            for ep in epochs:
                vals = data[obj][sample_type][ep]
                means.append(np.mean(vals) if vals else np.nan)
            ax.plot(epochs, means, color=OBJ_COLORS[obj], label=OBJ_LABELS[obj], linewidth=2)
        ax.set_xlabel("Training Epoch", fontsize=12)
        ax.set_ylabel("Mean Loss", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "CIFAR-10 ConvNet, sym 40%: loss trajectories on clean vs noisy samples",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    out = FIGURES / "gradient_norm_loss_curves.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {out.name}")


def save_architecture_comparison() -> None:
    """Side-by-side ConvNet vs MLP gradient ratio: shows architecture-specific dynamics.

    Key visual: FR ratio stays near 1 on ConvNet (gradient saturation works)
    but grows to 2.61 on MLP (saturation fails — early-amplification dominates).
    """
    convnet_path = RESULTS / "gradient_norm_full.csv"
    mlp_path = RESULTS / "mlp_gradient_norm_full.csv"
    if not convnet_path.exists() or not mlp_path.exists():
        print("missing gradient norm data — skipping architecture comparison")
        return

    def load_ratio(path: Path) -> dict[str, dict[int, float]]:
        rows = read_rows(path)
        data: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for r in rows:
            data[r["objective"]][r["sample_type"]][int(r["epoch"])].append(
                float(r["mean_grad_norm"])
            )
        out: dict[str, dict[int, float]] = {}
        for obj, stype_data in data.items():
            epochs = sorted(set(stype_data["noisy"].keys()) & set(stype_data["clean"].keys()))
            out[obj] = {}
            for ep in epochs:
                n = np.mean(stype_data["noisy"][ep]) if stype_data["noisy"][ep] else np.nan
                c = np.mean(stype_data["clean"][ep]) if stype_data["clean"][ep] else np.nan
                out[obj][ep] = n / max(c, 1e-8) if not np.isnan(c) else np.nan
        return out

    convnet_ratios = load_ratio(convnet_path)
    mlp_ratios = load_ratio(mlp_path)

    objectives = ["kl", "fisher_rao", "hellinger", "gce", "mae"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, (ratios, title, xlabel) in zip(
        axes,
        [
            (convnet_ratios, "CIFAR-10 ConvNet (sym 40%)", "Training Epoch (0–59)"),
            (mlp_ratios, "UCI Digits MLP (sym 40%)", "Training Epoch (0–99)"),
        ],
        strict=True,
    ):
        for obj in objectives:
            if obj not in ratios:
                continue
            ep_ratio = ratios[obj]
            epochs = sorted(ep_ratio.keys())
            vals = [ep_ratio[e] for e in epochs]
            ax.plot(epochs, vals, color=OBJ_COLORS.get(obj, "gray"),
                    label=OBJ_LABELS.get(obj, obj), linewidth=2)
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Noisy/clean gradient norm ratio" if ax is axes[0] else "")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Architecture-specific gradient dynamics under sym 40% noise\n"
        "FR ratio stays near 1 on ConvNet (gradient saturation); "
        "grows to 2.61 on MLP (early-amplification dominates)",
        fontsize=10,
    )
    fig.tight_layout()
    out = FIGURES / "gradient_norm_architecture_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}")


if __name__ == "__main__":
    save_gradient_norm_trajectories()
    save_gradient_norm_ratio()
    save_loss_trajectories()
    save_architecture_comparison()
    print("Done.")
