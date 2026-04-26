from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURE_DIR = Path(__file__).resolve().parent / "figures"
RESULT_DIR = Path(__file__).resolve().parent / "results"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def save_tsne_loss_curves() -> None:
    steps = np.array([0, 25, 50, 100, 150, 199])
    kl = np.array([1.613072, 0.642285, 0.298945, 0.182499, 0.141878, 0.119724])
    fisher_rao = np.array([4.565233, 1.897354, 0.896415, 0.527659, 0.406908, 0.339899])

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(steps, kl / kl[0], marker="o", label="KL, normalized")
    ax.plot(steps, fisher_rao / fisher_rao[0], marker="s", label="Fisher-Rao squared, normalized")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("objective / initial objective")
    ax.set_title("t-SNE objective decrease")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "tsne_loss_curves.pdf")
    plt.close(fig)


def save_vae_training_components() -> None:
    labels = ["KL", "Fisher-Rao"]
    reconstruction = np.array([216.819153, 219.285187])
    regularization = np.array([7.706926, 9.214247])
    total = np.array([224.526077, 228.499435])

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.bar(x - width, reconstruction, width, label="reconstruction")
    ax.bar(x, regularization, width, label="regularization")
    ax.bar(x + width, total, width, label="total")
    ax.set_xticks(x, labels)
    ax.set_ylabel("loss at step 25")
    ax.set_title("VAE smoke-run training components")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vae_training_components.pdf")
    plt.close(fig)


def save_categorical_objective_shape() -> None:
    p0 = np.linspace(0.01, 0.99, 400)
    p = np.stack([p0, 1.0 - p0], axis=1)
    q = np.array([0.5, 0.5])
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    affinity = np.sum(np.sqrt(p * q), axis=1)
    fisher_rao_squared = (2.0 * np.arccos(np.clip(affinity, -1.0, 1.0))) ** 2

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(p0, kl, label=r"$\mathrm{KL}(p\,||\,q)$")
    ax.plot(p0, fisher_rao_squared, label=r"$d_{\mathrm{FR}}(p,q)^2$")
    ax.set_xlabel(r"$p(y=0)$")
    ax.set_ylabel("objective value")
    ax.set_title("Categorical two-class objective shape")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "categorical_objective_shape.pdf")
    plt.close(fig)


def save_tsne_robustness_metrics() -> None:
    path = RESULT_DIR / "tsne_robustness_metrics.csv"
    if not path.exists():
        print(f"Skipping robustness figure; missing {path}")
        return

    rows = read_csv_rows(path)
    objectives = ["kl", "fisher_rao"]
    labels = {"kl": "KL", "fisher_rao": "Fisher-Rao"}
    metrics = [
        ("eval_trustworthiness", "trustworthiness"),
        ("eval_neighborhood_recall", "neighborhood recall"),
        ("eval_silhouette", "silhouette"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3), sharex=True)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for objective in objectives:
            selected = [row for row in rows if row["objective"] == objective]
            selected = sorted(selected, key=lambda row: float(row["noise_std_fraction"]))
            noise = [float(row["noise_std_fraction"]) for row in selected]
            values = [float(row[metric]) for row in selected]
            ax.plot(noise, values, marker="o", label=labels[objective])
        ax.set_title(title)
        ax.set_xlabel("feature noise / data std")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("metric value")
    axes[-1].legend()
    fig.suptitle("t-SNE final embedding quality under feature noise", y=1.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "tsne_robustness_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def save_vae_final_metrics() -> None:
    path = RESULT_DIR / "vae_final_metrics.csv"
    if not path.exists():
        print(f"Skipping VAE final metrics figure; missing {path}")
        return

    rows = read_csv_rows(path)
    labels = [
        f"{row['regularizer'].replace('_', '-')}\n$\\beta={float(row['beta']):g}$"
        for row in rows
    ]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    clean_bce = [float(row["eval_bce_per_pixel"]) for row in rows]
    noisy_bce = [float(row["eval_noisy_bce_per_pixel"]) for row in rows]
    latent_knn = [float(row["eval_latent_knn_accuracy"]) for row in rows]
    active_units = [float(row["eval_active_units"]) for row in rows]

    width = 0.36
    axes[0].bar(x - width / 2, clean_bce, width, label="clean")
    axes[0].bar(x + width / 2, noisy_bce, width, label="noisy input")
    axes[0].set_title("held-out BCE / pixel")
    axes[0].legend()

    axes[1].bar(x, latent_knn, color="tab:green")
    axes[1].set_title("latent kNN accuracy")
    axes[1].set_ylim(0.0, max(0.6, max(latent_knn) + 0.05))

    axes[2].bar(x, active_units, color="tab:purple")
    axes[2].set_title("active latent units")
    axes[2].set_ylim(0.0, 8.5)

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("VAE final representation and robustness metrics", y=1.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vae_final_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_tsne_loss_curves()
    save_vae_training_components()
    save_categorical_objective_shape()
    save_tsne_robustness_metrics()
    save_vae_final_metrics()
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
