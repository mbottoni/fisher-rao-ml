from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURE_DIR = Path(__file__).resolve().parent / "figures"
RESULT_DIR = Path(__file__).resolve().parent / "results"

DATASET_LABELS = {
    "blobs": "Blobs (8d)",
    "digits": "Digits (64d)",
    "mnist": "MNIST (784d)",
    "mnist_resnet18": "MNIST ResNet18 features",
    "fashion_mnist": "Fashion-MNIST",
    "kmnist": "KMNIST",
}

OBJECTIVE_LABELS = {
    "kl": "KL",
    "kl_smoothed": "Smoothed KL",
    "kl_capped": "Capped KL",
    "jensen_shannon": "Jensen-Shannon",
    "hellinger": "Hellinger",
    "fisher_rao": "Fisher-Rao",
}
OBJECTIVE_COLORS = {
    "kl": "tab:blue",
    "kl_smoothed": "tab:cyan",
    "kl_capped": "tab:purple",
    "jensen_shannon": "tab:orange",
    "hellinger": "tab:green",
    "fisher_rao": "tab:red",
}
OBJECTIVE_MARKERS = {
    "kl": "o",
    "kl_smoothed": "^",
    "kl_capped": "v",
    "jensen_shannon": "D",
    "hellinger": "P",
    "fisher_rao": "s",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def save_categorical_objective_shape() -> None:
    p0 = np.linspace(0.01, 0.99, 400)
    p = np.stack([p0, 1.0 - p0], axis=1)
    q = np.array([0.5, 0.5])
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    affinity = np.sum(np.sqrt(p * q), axis=1)
    fisher_rao_squared = (2.0 * np.arccos(np.clip(affinity, -1.0, 1.0))) ** 2

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(p0, kl, label=r"$\mathrm{KL}(p\,||\,q)$", color="tab:blue")
    ax.plot(
        p0,
        fisher_rao_squared,
        label=r"$d_{\mathrm{FR}}(p,q)^2$",
        color="tab:red",
    )
    ax.set_xlabel(r"$p(y=0)$")
    ax.set_ylabel("objective value")
    ax.set_title("Categorical two-class objective shape")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "categorical_objective_shape.pdf")
    plt.close(fig)


def save_bounded_codomain_figure() -> None:
    """Show that FR squared distance is bounded on the simplex while KL is not.

    For a binary categorical with q fixed at the uniform distribution, plot
    KL(p || q) and d_FR(p, q)^2 as p sweeps the simplex. KL diverges at the
    boundary; d_FR^2 is bounded above by pi^2.
    """
    p0 = np.linspace(1e-4, 1 - 1e-4, 1000)
    p = np.stack([p0, 1.0 - p0], axis=1)
    q = np.array([0.5, 0.5])
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    affinity = np.sum(np.sqrt(p * q), axis=1)
    fr_squared = (2.0 * np.arccos(np.clip(affinity, -1.0, 1.0))) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].plot(p0, kl, color="tab:blue")
    axes[0].set_title(r"$\mathrm{KL}(p\,||\,q)$ (unbounded)")
    axes[0].set_xlabel(r"$p(y=0)$")
    axes[0].set_ylabel("nats")
    axes[0].grid(alpha=0.25)

    axes[1].plot(p0, fr_squared, color="tab:red")
    axes[1].axhline(np.pi**2, color="black", linestyle="--", linewidth=0.8, label=r"$\pi^2$")
    axes[1].set_title(r"$d_{\mathrm{FR}}(p,q)^2$ (bounded by $\pi^2$)")
    axes[1].set_xlabel(r"$p(y=0)$")
    axes[1].set_ylabel("squared geodesic length")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Categorical objective scale (q is uniform)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bounded_codomain.pdf")
    plt.close(fig)


def save_robustness_lines() -> None:
    path = RESULT_DIR / "tsne_robustness_aggregated.csv"
    if not path.exists():
        print(f"Skipping robustness lines; missing {path}")
        return
    rows = read_csv_rows(path)

    metrics = [
        ("eval_trustworthiness", "trustworthiness"),
        ("eval_neighborhood_recall", "neighborhood recall"),
        ("eval_silhouette", "silhouette"),
        ("eval_knn_accuracy", "kNN accuracy"),
    ]
    datasets = sorted({row["dataset"] for row in rows})
    objectives = ["kl", "fisher_rao"]

    fig, axes = plt.subplots(
        len(datasets),
        len(metrics),
        figsize=(3.0 * len(metrics), 2.5 * len(datasets)),
        sharex=True,
    )
    if len(datasets) == 1:
        axes = np.array([axes])

    for row_idx, dataset in enumerate(datasets):
        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for objective in objectives:
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == dataset and row["objective"] == objective
                ]
                selected.sort(key=lambda row: float(row["noise_std_fraction"]))
                noise = np.array([float(row["noise_std_fraction"]) for row in selected])
                mean = np.array([float(row[f"{metric}_mean"]) for row in selected])
                std = np.array([float(row[f"{metric}_std"]) for row in selected])
                ax.plot(
                    noise,
                    mean,
                    label=OBJECTIVE_LABELS[objective],
                    color=OBJECTIVE_COLORS[objective],
                    marker=OBJECTIVE_MARKERS[objective],
                )
                ax.fill_between(
                    noise,
                    mean - std,
                    mean + std,
                    color=OBJECTIVE_COLORS[objective],
                    alpha=0.18,
                )
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(DATASET_LABELS.get(dataset, dataset))
            if row_idx == len(datasets) - 1:
                ax.set_xlabel("feature noise / data std")
            ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Final t-SNE embedding quality vs feature noise (mean $\\pm$ std over seeds)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(FIGURE_DIR / "tsne_robustness_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def save_robustness_deltas() -> None:
    """Bar plot of mean (FR - KL) per dataset per metric, averaged over noise levels."""
    path = RESULT_DIR / "tsne_robustness_significance.csv"
    if not path.exists():
        print(f"Skipping robustness deltas; missing {path}")
        return
    rows = read_csv_rows(path)
    if not rows:
        return

    metrics = [
        "eval_trustworthiness",
        "eval_neighborhood_recall",
        "eval_silhouette",
        "eval_knn_accuracy",
    ]
    metric_labels = ["trust.", "recall", "silhouette", "kNN acc."]
    datasets = sorted({row["dataset"] for row in rows})

    accumulated: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in metrics:
            key = (row["dataset"], metric)
            value = float(row[f"{metric}_mean_diff"])
            if np.isfinite(value):
                accumulated[key].append(value)

    width = 0.18
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for i, dataset in enumerate(datasets):
        means = [
            float(np.mean(accumulated[(dataset, metric)]))
            if accumulated[(dataset, metric)]
            else 0.0
            for metric in metrics
        ]
        ax.bar(
            x + (i - (len(datasets) - 1) / 2) * width,
            means,
            width,
            label=DATASET_LABELS.get(dataset, dataset),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, metric_labels)
    ax.set_ylabel(r"mean Fisher-Rao $-$ KL across noise levels")
    ax.set_title("Average paired improvement of Fisher-Rao over KL")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "tsne_paired_deltas.pdf")
    plt.close(fig)


def save_qualitative_embeddings() -> None:
    path = RESULT_DIR / "tsne_qualitative_embeddings.csv"
    if not path.exists():
        print(f"Skipping qualitative figure; missing {path}")
        return
    rows = read_csv_rows(path)
    if not rows:
        return

    chosen_noise = [0.0, 0.5, 1.0]
    objectives = ["kl", "fisher_rao"]

    grouped: dict[tuple[float, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        noise = float(row["noise_std_fraction"])
        if noise in chosen_noise:
            grouped[(noise, row["objective"])].append(row)

    if not grouped:
        return

    fig, axes = plt.subplots(
        len(objectives),
        len(chosen_noise),
        figsize=(3.0 * len(chosen_noise), 3.0 * len(objectives)),
    )

    for col_idx, noise in enumerate(chosen_noise):
        for row_idx, objective in enumerate(objectives):
            ax = axes[row_idx, col_idx]
            entries = grouped.get((noise, objective), [])
            if not entries:
                ax.set_axis_off()
                continue
            xs = np.array([float(entry["x"]) for entry in entries])
            ys = np.array([float(entry["y"]) for entry in entries])
            labels = np.array([int(entry["label"]) for entry in entries])
            ax.scatter(xs, ys, c=labels, cmap="tab10", s=12, alpha=0.9)
            ax.set_title(f"{OBJECTIVE_LABELS[objective]}, noise={noise}")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("t-SNE embeddings of digits dataset under increasing feature noise (single seed)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / "tsne_qualitative.pdf", bbox_inches="tight")
    plt.close(fig)


def save_loss_curves() -> None:
    path = RESULT_DIR / "tsne_training_dynamics.csv"
    if not path.exists():
        print(f"Skipping loss curves; missing {path}")
        return
    rows = read_csv_rows(path)
    if not rows:
        return

    target_dataset = "digits" if any(row["dataset"] == "digits" for row in rows) else (
        rows[0]["dataset"]
    )
    target_noise = 0.5
    matching = [
        row
        for row in rows
        if row["dataset"] == target_dataset
        and abs(float(row["noise_std_fraction"]) - target_noise) < 1e-6
    ]
    if not matching:
        return

    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in matching:
        objective = row["objective"]
        step = int(row["step"])
        grouped[objective][step].append(float(row["loss"]))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for objective, per_step in grouped.items():
        steps = sorted(per_step.keys())
        means = np.array([np.mean(per_step[s]) for s in steps])
        if len(means) == 0 or means[0] == 0:
            continue
        normalized = means / means[0]
        ax.plot(
            steps,
            normalized,
            label=OBJECTIVE_LABELS[objective],
            color=OBJECTIVE_COLORS[objective],
            marker=OBJECTIVE_MARKERS[objective],
            markevery=max(len(steps) // 8, 1),
        )
    ax.set_xlabel("optimization step")
    ax.set_ylabel("objective / initial objective")
    ax.set_title(
        f"Normalized t-SNE objective curves ({DATASET_LABELS[target_dataset]}, "
        f"noise={target_noise})"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "tsne_loss_curves.pdf")
    plt.close(fig)


def stress_corruption_type(row: dict[str, str]) -> str:
    return row.get("corruption_type") or "none"


def safe_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def save_dimred_false_edge_curves() -> None:
    path = RESULT_DIR / "dimred_stress_significance.csv"
    if not path.exists():
        print(f"Skipping dimensionality-reduction stress curves; missing {path}")
        return
    all_rows = [
        row
        for row in read_csv_rows(path)
        if row["experiment"] == "noisy_affinity"
        and stress_corruption_type(row) != "none"
    ]
    if not all_rows:
        return
    preferred_dataset = "digits" if any(row["dataset"] == "digits" for row in all_rows) else (
        all_rows[0]["dataset"]
    )
    rows = [row for row in all_rows if row["dataset"] == preferred_dataset]
    corruption_types = sorted({stress_corruption_type(row) for row in rows})

    metrics = [
        ("eval_trustworthiness", "trustworthiness", 1.0),
        ("eval_neighborhood_recall", "clean-neighbor recall", 1.0),
        ("eval_local_label_purity", "local label purity", 1.0),
        ("eval_corrupted_edge_preservation", "bad-edge preservation", -1.0),
        ("eval_corrupted_edge_q_mass", "bad-edge $Q$ mass", -1.0),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.1), sharex=True)
    for ax, (metric, label, sign) in zip(axes, metrics, strict=True):
        for corruption_type in corruption_types:
            selected = [row for row in rows if stress_corruption_type(row) == corruption_type]
            selected.sort(key=lambda row: float(row["stress_level"]))
            x = np.array([float(row["stress_level"]) for row in selected])
            y = np.array([safe_float(row, f"{metric}_mean_diff") * sign for row in selected])
            ax.plot(x, y, marker="o", label=corruption_type)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel("false-edge mass")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("oriented Fisher-Rao improvement over KL")
    axes[-1].legend(fontsize=8)
    fig.suptitle(
        "Structured false-affinity robustness on "
        f"{DATASET_LABELS.get(preferred_dataset, preferred_dataset)}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURE_DIR / "dimred_false_edge_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def save_dimred_noisy_affinity_means() -> None:
    path = RESULT_DIR / "dimred_stress_aggregated.csv"
    if not path.exists():
        print(f"Skipping noisy-affinity means; missing {path}")
        return
    all_rows = [
        row
        for row in read_csv_rows(path)
        if row["experiment"] == "noisy_affinity"
        and stress_corruption_type(row) != "none"
    ]
    if not all_rows:
        return
    preferred_dataset = "digits" if any(row["dataset"] == "digits" for row in all_rows) else (
        all_rows[0]["dataset"]
    )
    preferred_corruption = "hub" if any(
        row["dataset"] == preferred_dataset and stress_corruption_type(row) == "hub"
        for row in all_rows
    ) else stress_corruption_type(all_rows[0])
    rows = [
        row
        for row in all_rows
        if row["dataset"] == preferred_dataset
        and stress_corruption_type(row) == preferred_corruption
    ]

    metrics = [
        ("eval_trustworthiness", "trustworthiness $\\uparrow$"),
        ("eval_neighborhood_recall", "clean recall $\\uparrow$"),
        ("eval_corrupted_edge_preservation", "bad-edge preservation $\\downarrow$"),
        ("eval_corrupted_edge_q_mass", "bad-edge $Q$ mass $\\downarrow$"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.2), sharex=True)
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        for objective in ["kl", "fisher_rao"]:
            selected = [row for row in rows if row["objective"] == objective]
            selected.sort(key=lambda row: float(row["stress_level"]))
            x = np.array([float(row["stress_level"]) for row in selected])
            y = np.array([safe_float(row, f"{metric}_mean") for row in selected])
            std = np.array([safe_float(row, f"{metric}_std") for row in selected])
            ax.plot(
                x,
                y,
                label=OBJECTIVE_LABELS[objective],
                color=OBJECTIVE_COLORS[objective],
                marker=OBJECTIVE_MARKERS[objective],
            )
            ax.fill_between(x, y - std, y + std, color=OBJECTIVE_COLORS[objective], alpha=0.16)
        ax.set_title(label)
        ax.set_xlabel("false-edge mass")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle(
        "Noisy-affinity stress test means "
        f"({DATASET_LABELS.get(preferred_dataset, preferred_dataset)}, {preferred_corruption})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURE_DIR / "dimred_noisy_affinity_means.pdf", bbox_inches="tight")
    plt.close(fig)


def save_dimred_stress_overview() -> None:
    path = RESULT_DIR / "dimred_stress_significance.csv"
    if not path.exists():
        print(f"Skipping dimensionality-reduction stress overview; missing {path}")
        return
    rows = read_csv_rows(path)
    if not rows:
        return

    metric_by_experiment = {
        "noisy_affinity": ("eval_corrupted_edge_preservation", -1.0, "bad edges avoided"),
        "outlier_influence": ("eval_outlier_influence", -1.0, "outlier stability"),
        "global_geometry": ("eval_trustworthiness", 1.0, "manifold trust."),
        "symmetric_mismatch": ("eval_between_manifold_leakage", -1.0, "less leakage"),
    }
    selected_rows = []
    for row in rows:
        if row["experiment"] not in metric_by_experiment:
            continue
        if row["experiment"] == "noisy_affinity" and stress_corruption_type(row) not in {
            "none",
            "uniform",
        }:
            continue
        selected_rows.append(row)
    if not selected_rows:
        return

    labels = [
        f"{row['experiment'].replace('_', ' ')}\n{row['dataset']}, {float(row['stress_level']):g}"
        for row in selected_rows
    ]
    values = []
    for row in selected_rows:
        metric, sign, _title = metric_by_experiment[row["experiment"]]
        values.append(safe_float(row, f"{metric}_mean_diff") * sign)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(values)), 3.7))
    colors = ["tab:red" if value >= 0 else "tab:blue" for value in values]
    ax.bar(np.arange(len(values)), values, color=colors, alpha=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(values)), labels, rotation=75, ha="right")
    ax.set_ylabel("oriented Fisher-Rao improvement over KL")
    ax.set_title("Stress-test overview: positive bars favor Fisher-Rao")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dimred_stress_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def save_dimred_bad_edge_qualitative() -> None:
    embedding_path = RESULT_DIR / "dimred_stress_embeddings.csv"
    edge_path = RESULT_DIR / "dimred_stress_edges.csv"
    if not embedding_path.exists() or not edge_path.exists():
        print(f"Skipping bad-edge qualitative figure; missing {embedding_path} or {edge_path}")
        return
    embeddings = read_csv_rows(embedding_path)
    edges = read_csv_rows(edge_path)
    if not embeddings:
        return

    objectives = ["kl", "fisher_rao"]
    states = ["clean", "corrupted"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.8))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in embeddings:
        grouped[(row["target_state"], row["objective"])].append(row)

    for row_idx, state in enumerate(states):
        for col_idx, objective in enumerate(objectives):
            ax = axes[row_idx, col_idx]
            selected = sorted(
                grouped.get((state, objective), []),
                key=lambda row: int(row["index"]),
            )
            if not selected:
                ax.set_axis_off()
                continue
            xs = np.array([float(row["x"]) for row in selected])
            ys = np.array([float(row["y"]) for row in selected])
            labels = np.array([int(row["label"]) for row in selected])
            if state == "corrupted":
                for edge in edges:
                    i = int(edge["source"])
                    j = int(edge["target"])
                    if i < len(xs) and j < len(xs):
                        ax.plot(
                            [xs[i], xs[j]],
                            [ys[i], ys[j]],
                            color="tab:red",
                            alpha=0.08,
                            linewidth=0.45,
                            zorder=1,
                        )
            ax.scatter(xs, ys, c=labels, cmap="tab10", s=12, alpha=0.9, zorder=2)
            ax.set_title(f"{OBJECTIVE_LABELS[objective]}, {state} target")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Representative corrupted-edge embeddings; red lines are injected bad edges")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / "dimred_bad_edge_qualitative.pdf", bbox_inches="tight")
    plt.close(fig)


def save_false_edge_mechanism() -> None:
    p_false = 0.05
    q_false = np.geomspace(1e-5, 0.25, 600)
    p = np.stack([np.full_like(q_false, p_false), np.full_like(q_false, 1.0 - p_false)], axis=1)
    q = np.stack([q_false, 1.0 - q_false], axis=1)
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    affinity = np.sum(np.sqrt(p * q), axis=1)
    fr = (2.0 * np.arccos(np.clip(affinity, -1.0, 1.0))) ** 2
    kl_grad_mag = p_false / q_false
    fr_grad_mag = np.abs(np.gradient(fr, q_false))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    axes[0].plot(q_false, kl, color="tab:blue", label="KL")
    axes[0].plot(q_false, fr, color="tab:red", label="Fisher-Rao")
    axes[0].axvline(p_false, color="black", linestyle="--", linewidth=0.8, label="$q=p$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("embedding mass on false edge $q_{ij}$")
    axes[0].set_ylabel("two-bin objective")
    axes[0].set_title("False-edge objective shape")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(q_false, kl_grad_mag, color="tab:blue", label="KL")
    axes[1].plot(q_false, fr_grad_mag, color="tab:red", label="Fisher-Rao")
    axes[1].axvline(p_false, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("embedding mass on false edge $q_{ij}$")
    axes[1].set_ylabel("gradient magnitude")
    axes[1].set_title("KL pressure grows as $1/q_{ij}$")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Why KL overfits high-confidence false edges")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIGURE_DIR / "dimred_false_edge_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)


def save_dimred_significance_heatmap() -> None:
    path = RESULT_DIR / "dimred_stress_significance.csv"
    if not path.exists():
        print(f"Skipping dimred significance heatmap; missing {path}")
        return
    rows = [
        row
        for row in read_csv_rows(path)
        if row["experiment"] == "noisy_affinity"
        and stress_corruption_type(row) != "none"
        and float(row["stress_level"]) > 0
    ]
    if not rows:
        return
    rows.sort(
        key=lambda row: (
            row["dataset"],
            stress_corruption_type(row),
            float(row["stress_level"]),
        )
    )
    metrics = [
        ("eval_neighborhood_recall", "Recall", 1.0),
        ("eval_corrupted_edge_preservation", "Bad edges", -1.0),
    ]
    labels = [
        f"{row['dataset']}\n{stress_corruption_type(row)}\n{float(row['stress_level']):g}"
        for row in rows
    ]
    matrix = np.array(
        [
            [safe_float(row, f"{metric}_mean_diff") * sign for metric, _label, sign in metrics]
            for row in rows
        ]
    ).T
    fig, ax = plt.subplots(figsize=(max(9.0, 0.34 * len(rows)), 2.5))
    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(metrics)), [label for _metric, label, _sign in metrics])
    ax.set_xticks(np.arange(len(labels)), labels, rotation=75, ha="right", fontsize=7)
    ax.set_title("Oriented Fisher-Rao improvement by dataset, corruption type, and level")
    fig.colorbar(image, ax=ax, shrink=0.75, label="oriented FR improvement")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dimred_significance_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def save_knn_graph_baselines() -> None:
    path = RESULT_DIR / "dimred_stress_aggregated.csv"
    if not path.exists():
        print(f"Skipping kNN graph baseline figure; missing {path}")
        return
    rows = [
        row
        for row in read_csv_rows(path)
        if row["experiment"] == "knn_graph"
        and float(row["stress_level"]) > 0
        and stress_corruption_type(row) != "none"
    ]
    if not rows:
        return
    metrics = [
        ("eval_neighborhood_recall", "clean-neighbor recall $\\uparrow$"),
        ("eval_corrupted_edge_preservation", "bad-edge preservation $\\downarrow$"),
    ]
    datasets = sorted({row["dataset"] for row in rows})
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(9.0, 2.8 * len(datasets)))
    if len(datasets) == 1:
        axes = np.array([axes])
    for row_idx, dataset in enumerate(datasets):
        subset = [
            row
            for row in rows
            if row["dataset"] == dataset and stress_corruption_type(row) == "uniform"
        ]
        if not subset:
            subset = [row for row in rows if row["dataset"] == dataset]
        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for objective in sorted({row["objective"] for row in subset}):
                selected = [row for row in subset if row["objective"] == objective]
                selected.sort(key=lambda row: float(row["stress_level"]))
                x = np.array([float(row["stress_level"]) for row in selected])
                y = np.array([safe_float(row, f"{metric}_mean") for row in selected])
                if not len(x):
                    continue
                ax.plot(
                    x,
                    y,
                    marker=OBJECTIVE_MARKERS.get(objective, "o"),
                    color=OBJECTIVE_COLORS.get(objective),
                    label=OBJECTIVE_LABELS.get(objective, objective),
                )
            ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}: {title}")
            ax.set_xlabel("corrupted kNN edge fraction")
            ax.grid(alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Corrupted neighbor graph robustness across divergences")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / "dimred_knn_graph_baselines.pdf", bbox_inches="tight")
    plt.close(fig)


def save_ml_stress_curves() -> None:
    path = RESULT_DIR / "ml_stress_aggregated.csv"
    if not path.exists():
        print(f"Skipping ML stress curves; missing {path}")
        return
    rows = [
        row
        for row in read_csv_rows(path)
        if row["experiment"] == "soft_label"
        and row["corruption_type"] in {"adversarial", "symmetric_noise"}
    ]
    if not rows:
        return
    metrics = [
        ("eval_accuracy", "accuracy $\\uparrow$"),
        ("eval_ece", "ECE $\\downarrow$"),
        ("eval_brier", "Brier $\\downarrow$"),
    ]
    datasets = sorted({row["dataset"] for row in rows})
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(9.5, 2.9 * len(datasets)))
    if len(datasets) == 1:
        axes = np.array([axes])
    for row_idx, dataset in enumerate(datasets):
        subset = [
            row
            for row in rows
            if row["dataset"] == dataset and row["corruption_type"] == "adversarial"
        ]
        if not subset:
            subset = [row for row in rows if row["dataset"] == dataset]
        for col_idx, (metric, label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for objective in sorted({row["objective"] for row in subset}):
                selected = [row for row in subset if row["objective"] == objective]
                selected.sort(key=lambda row: float(row["stress_level"]))
                x = np.array([float(row["stress_level"]) for row in selected])
                y = np.array([safe_float(row, f"{metric}_mean") for row in selected])
                ax.plot(
                    x,
                    y,
                    marker=OBJECTIVE_MARKERS.get(objective, "o"),
                    color=OBJECTIVE_COLORS.get(objective),
                    label=OBJECTIVE_LABELS.get(objective, objective),
                )
            ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}: {label}")
            ax.set_xlabel("corrupted target fraction")
            ax.grid(alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Noisy soft-label classification under overconfident wrong targets")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / "ml_soft_label_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def save_ml_distillation_curves() -> None:
    path = RESULT_DIR / "ml_stress_aggregated.csv"
    if not path.exists():
        print(f"Skipping ML distillation curves; missing {path}")
        return
    rows = [
        row
        for row in read_csv_rows(path)
        if row["experiment"] == "distillation"
        and row["corruption_type"] in {"random_wrong", "class_confusion"}
    ]
    if not rows:
        return
    metrics = [
        ("eval_accuracy", "accuracy $\\uparrow$"),
        ("eval_teacher_error_imitation", "teacher-error imitation $\\downarrow$"),
        ("eval_ece", "ECE $\\downarrow$"),
    ]
    datasets = sorted({row["dataset"] for row in rows})
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(9.5, 2.9 * len(datasets)))
    if len(datasets) == 1:
        axes = np.array([axes])
    for row_idx, dataset in enumerate(datasets):
        subset = [
            row
            for row in rows
            if row["dataset"] == dataset and row["corruption_type"] == "random_wrong"
        ]
        if not subset:
            subset = [row for row in rows if row["dataset"] == dataset]
        for col_idx, (metric, label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for objective in sorted({row["objective"] for row in subset}):
                selected = [row for row in subset if row["objective"] == objective]
                selected.sort(key=lambda row: float(row["stress_level"]))
                x = np.array([float(row["stress_level"]) for row in selected])
                y = np.array([safe_float(row, f"{metric}_mean") for row in selected])
                ax.plot(
                    x,
                    y,
                    marker=OBJECTIVE_MARKERS.get(objective, "o"),
                    color=OBJECTIVE_COLORS.get(objective),
                    label=OBJECTIVE_LABELS.get(objective, objective),
                )
            ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}: {label}")
            ax.set_xlabel("corrupted teacher fraction")
            ax.grid(alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Student distillation under corrupted teacher probabilities")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / "ml_distillation_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def save_ml_stress_power() -> None:
    path = RESULT_DIR / "ml_stress_power_summary.csv"
    if not path.exists():
        print(f"Skipping ML stress power figure; missing {path}")
        return
    rows = read_csv_rows(path)
    rows = [
        row
        for row in rows
        if row["metric_label"] in {"accuracy", "ece", "brier", "teacher_error_imitation"}
    ]
    if not rows:
        return
    labels = [f"{row['experiment']}\n{row['metric_label']}" for row in rows]
    values = [
        int(row["n_improves_over_kl"]) / max(int(row["n_cells"]), 1)
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(rows)), 3.2))
    ax.bar(np.arange(len(rows)), values, color="tab:red", alpha=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(len(rows)), labels, rotation=45, ha="right")
    ax.set_ylabel("fraction of cells where Fisher-Rao improves over KL")
    ax.set_title("ML probability-target stress summary")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ml_stress_power_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def save_vae_final_metrics() -> None:
    path = RESULT_DIR / "vae_by_beta_aggregated.csv"
    if not path.exists():
        print(f"Skipping VAE beta trade-off figure; missing {path}")
        return

    rows = read_csv_rows(path)
    if not rows:
        return

    datasets = sorted({row["dataset"] for row in rows})
    metrics = [
        ("eval_bce_per_pixel_mean", "BCE / pixel", "lower"),
        ("eval_latent_knn_accuracy_mean", "latent kNN", "higher"),
        ("eval_active_units_mean", "active units", "higher"),
    ]
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(10.5, 2.7 * len(datasets)))
    if len(datasets) == 1:
        axes = np.array([axes])

    for row_idx, dataset in enumerate(datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        for col_idx, (metric, title, direction) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for regularizer in ["kl", "fisher_rao"]:
                selected = [row for row in subset if row["regularizer"] == regularizer]
                selected.sort(key=lambda row: float(row["beta"]))
                if not selected:
                    continue
                betas = np.array([float(row["beta"]) for row in selected])
                means = np.array([float(row[metric]) for row in selected])
                stds = np.array([float(row[metric.replace("_mean", "_std")]) for row in selected])
                ax.errorbar(
                    betas,
                    means,
                    yerr=stds,
                    label=OBJECTIVE_LABELS.get(regularizer, regularizer),
                    color=OBJECTIVE_COLORS.get(regularizer, "black"),
                    marker=OBJECTIVE_MARKERS.get(regularizer, "o"),
                    capsize=2,
                )
            ax.set_xscale("log")
            if row_idx == 0:
                ax.set_title(f"{title} ({direction} is better)")
            if col_idx == 0:
                ax.set_ylabel(DATASET_LABELS.get(dataset, dataset))
            if row_idx == len(datasets) - 1:
                ax.set_xlabel(r"$\beta$")
            ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("VAE beta trade-offs (mean $\\pm$ std over seeds)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(FIGURE_DIR / "vae_final_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def save_vae_best_beta_deltas() -> None:
    path = RESULT_DIR / "vae_best_beta_significance.csv"
    if not path.exists():
        print(f"Skipping VAE paired deltas; missing {path}")
        return
    rows = read_csv_rows(path)
    if not rows:
        return
    metrics = [
        ("eval_bce_per_pixel", "BCE/pix", -1.0),
        ("eval_latent_knn_accuracy", "latent kNN", 1.0),
        ("eval_aggregated_posterior_mmd", "posterior MMD", -1.0),
        ("eval_noise_0.25_dropout_0_bce_per_pixel", "noise BCE", -1.0),
        ("eval_noise_0_dropout_0.25_bce_per_pixel", "dropout BCE", -1.0),
    ]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    for i, row in enumerate(rows):
        values = [
            float(row[f"{metric}_mean_diff_fr_minus_kl"]) * sign
            for metric, _label, sign in metrics
        ]
        ax.bar(x + (i - (len(rows) - 1) / 2) * width, values, width, label=row["dataset"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, [label for _metric, label, _sign in metrics])
    ax.set_ylabel("oriented Fisher-Rao improvement over KL")
    ax.set_title("Best-beta VAE paired deltas")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vae_best_beta_deltas.pdf")
    plt.close(fig)


def save_vae_training_curves() -> None:
    path = RESULT_DIR / "vae_training_dynamics.csv"
    if not path.exists():
        print(f"Skipping VAE training curves; missing {path}")
        return
    rows = read_csv_rows(path)
    if not rows:
        return
    target_dataset = (
        "mnist" if any(row.get("dataset") == "mnist" for row in rows) else rows[0]["dataset"]
    )
    grouped: dict[tuple[str, float, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") == target_dataset:
            grouped[(row["regularizer"], float(row["beta"]), int(row["step"]))].append(row)
    if not grouped:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharex=True)
    for regularizer in ["kl", "fisher_rao"]:
        beta_candidates = sorted({beta for reg, beta, _step in grouped if reg == regularizer})
        if not beta_candidates:
            continue
        beta = 1.0 if 1.0 in beta_candidates else beta_candidates[0]
        steps = sorted({step for reg, b, step in grouped if reg == regularizer and b == beta})
        reconstruction = [
            np.mean([float(row["reconstruction"]) for row in grouped[(regularizer, beta, step)]])
            for step in steps
        ]
        regularization = [
            np.mean([float(row["regularization"]) for row in grouped[(regularizer, beta, step)]])
            for step in steps
        ]
        label = f"{OBJECTIVE_LABELS[regularizer]}, beta={beta:g}"
        axes[0].plot(steps, reconstruction, label=label, color=OBJECTIVE_COLORS[regularizer])
        axes[1].plot(steps, regularization, label=label, color=OBJECTIVE_COLORS[regularizer])
    axes[0].set_title("reconstruction")
    axes[1].set_title("regularization")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"VAE training dynamics ({DATASET_LABELS.get(target_dataset, target_dataset)})")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vae_training_dynamics.pdf")
    plt.close(fig)


def save_vae_latent_embeddings() -> None:
    path = RESULT_DIR / "vae_latent_embeddings.csv"
    if not path.exists():
        print(f"Skipping VAE latent embeddings; missing {path}")
        return
    rows = read_csv_rows(path)
    if not rows:
        return
    configs = []
    for row in rows:
        key = (row["regularizer"], float(row["beta"]))
        if key not in configs:
            configs.append(key)
    configs = configs[:4]
    fig, axes = plt.subplots(1, len(configs), figsize=(3.1 * len(configs), 3.0))
    if len(configs) == 1:
        axes = np.array([axes])
    for ax, (regularizer, beta) in zip(axes, configs, strict=True):
        selected = [
            row
            for row in rows
            if row["regularizer"] == regularizer and abs(float(row["beta"]) - beta) < 1e-12
        ]
        xs = np.array([float(row["x"]) for row in selected])
        ys = np.array([float(row["y"]) for row in selected])
        labels = np.array([int(row["label"]) for row in selected])
        ax.scatter(xs, ys, c=labels, cmap="tab10", s=10, alpha=0.85)
        ax.set_title(f"{OBJECTIVE_LABELS[regularizer]}\n$\\beta={beta:g}$")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Representative VAE latent means projected with PCA")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vae_latent_geometry.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_categorical_objective_shape()
    save_bounded_codomain_figure()
    save_robustness_lines()
    save_robustness_deltas()
    save_qualitative_embeddings()
    save_loss_curves()
    save_dimred_false_edge_curves()
    save_dimred_noisy_affinity_means()
    save_dimred_stress_overview()
    save_dimred_bad_edge_qualitative()
    save_false_edge_mechanism()
    save_dimred_significance_heatmap()
    save_knn_graph_baselines()
    save_ml_stress_curves()
    save_ml_distillation_curves()
    save_ml_stress_power()
    save_vae_final_metrics()
    save_vae_best_beta_deltas()
    save_vae_training_curves()
    save_vae_latent_embeddings()
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
