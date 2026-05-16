"""Generate CIFAR-N figures for Direction 1 paper (§3.3).

Compares 6 objectives under real human-annotated noisy labels across
three annotation conditions (aggre ~9%, random1 ~17%, worse ~40%).
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

NOISE_TYPES = ["aggre", "random1", "worse"]
NOISE_LABELS = {
    "aggre": "Aggre\n(~9%)",
    "random1": "Random1\n(~17%)",
    "worse": "Worse\n(~40%)",
}
NOISE_RATES = {"aggre": 0.09, "random1": 0.172, "worse": 0.402}

OBJ_ORDER = ["kl", "fisher_rao", "hellinger", "gce", "mae", "sce"]
OBJ_LABELS = {
    "kl": "CE (KL)", "fisher_rao": "Fisher-Rao", "hellinger": "Hellinger",
    "gce": "GCE", "mae": "MAE", "sce": "SCE",
}
OBJ_COLORS = {
    "kl": "tab:blue", "fisher_rao": "tab:orange", "hellinger": "tab:green",
    "gce": "tab:red", "mae": "tab:purple", "sce": "tab:brown",
}
OBJ_MARKERS = {
    "kl": "o", "fisher_rao": "s", "hellinger": "^", "gce": "D", "mae": "v", "sce": "*",
}


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def save_cifar_n_accuracy_bars() -> None:
    """Bar chart of mean accuracy per objective per noise type."""
    path = RESULTS / "cifar_n_full.csv"
    if not path.exists():
        print("cifar_n_full.csv not found — skipping")
        return

    rows = read_rows(path)
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["noise_type"]][r["objective"]].append(float(r["eval_accuracy"]))

    noise_types = [nt for nt in NOISE_TYPES if nt in data]
    if not noise_types:
        print("no CIFAR-N data yet")
        return

    n_noise = len(noise_types)
    n_obj = len(OBJ_ORDER)
    width = 0.12
    x = np.arange(n_noise)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, obj in enumerate(OBJ_ORDER):
        means, errs = [], []
        for nt in noise_types:
            vals = data[nt][obj]
            means.append(np.mean(vals) * 100 if vals else np.nan)
            errs.append(np.std(vals) * 100 if len(vals) > 1 else 0.0)
        offset = (i - n_obj / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width,
            label=OBJ_LABELS[obj],
            color=OBJ_COLORS[obj],
            yerr=errs, capsize=3, error_kw={"linewidth": 1},
        )

    ax.set_xlabel("CIFAR-N Noise Condition", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(
        "CIFAR-N: Real Human-Annotated Label Noise — ConvNet Accuracy",
        fontsize=13
    )
    ax.set_xticks(x)
    ax.set_xticklabels([NOISE_LABELS[nt] for nt in noise_types], fontsize=11)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(50, 100)

    plt.tight_layout()
    out = FIGURES / "cifar_n_accuracy_bars.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {out.name}")


def save_cifar_n_vs_synthetic() -> None:
    """Compare CIFAR-N vs synthetic noise objective rankings."""
    cifar_n_path = RESULTS / "cifar_n_full.csv"
    synth_path = RESULTS / "cifar10_noisy_label_full.csv"
    if not cifar_n_path.exists() or not synth_path.exists():
        print("missing data files — skipping cifar_n_vs_synthetic")
        return

    # CIFAR-N data
    cn_rows = read_rows(cifar_n_path)
    cn_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in cn_rows:
        cn_data[r["noise_type"]][r["objective"]].append(float(r["eval_accuracy"]))

    # Synthetic data
    sy_rows = read_rows(synth_path)
    sy_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in sy_rows:
        sy_data[r["noise_regime"]][r["objective"]].append(float(r["eval_accuracy"]))

    # Map CIFAR-N conditions to closest synthetic condition
    pairs = [
        ("aggre", "clean", "CIFAR-N aggre 9% vs synthetic clean"),
        ("random1", "sym_20", "CIFAR-N random1 17% vs synthetic sym 20%"),
        ("worse", "sym_40", "CIFAR-N worse 40% vs synthetic sym 40%"),
    ]

    objectives = [o for o in OBJ_ORDER if cn_data.get("aggre", {}).get(o)]
    if not objectives:
        print("insufficient CIFAR-N data for comparison — skipping")
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(14, 5), sharey=False)
    if len(pairs) == 1:
        axes = [axes]

    for ax, (cn_type, sy_type, title) in zip(axes, pairs, strict=True):
        cn_means = [np.mean(cn_data[cn_type][o]) * 100 if cn_data[cn_type][o] else np.nan
                    for o in objectives]
        sy_means = [np.mean(sy_data[sy_type][o]) * 100 if sy_data[sy_type][o] else np.nan
                    for o in objectives]

        x = np.arange(len(objectives))
        width = 0.35
        ax.bar(x - width / 2, cn_means, width, label="CIFAR-N (real)",
               color="tab:blue", alpha=0.8)
        ax.bar(x + width / 2, sy_means, width, label="Synthetic",
               color="tab:orange", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([OBJ_LABELS[o] for o in objectives], rotation=30, ha="right")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "CIFAR-N real noise vs synthetic noise: objective ranking consistency",
        fontsize=12
    )
    plt.tight_layout()
    out = FIGURES / "cifar_n_vs_synthetic.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {out.name}")


def print_cifar_n_table() -> None:
    """Print a LaTeX table of CIFAR-N results for copy-paste into the paper."""
    path = RESULTS / "cifar_n_full.csv"
    if not path.exists():
        return

    rows = read_rows(path)
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["noise_type"]][r["objective"]].append(float(r["eval_accuracy"]))

    objectives = ["kl", "fisher_rao", "hellinger", "gce", "mae", "sce"]
    obj_headers = " & ".join([OBJ_LABELS.get(o, o) for o in objectives])

    print("\n=== LaTeX Table for CIFAR-N §3.3 ===")
    print(f"{'Condition':<12}  {obj_headers}")
    for nt in NOISE_TYPES:
        if nt not in data:
            continue
        n = max(len(data[nt][o]) for o in objectives if data[nt][o])
        row_vals = []
        for o in objectives:
            vals = data[nt][o]
            if vals:
                row_vals.append(f"{np.mean(vals)*100:.1f}")
            else:
                row_vals.append("---")
        print(f"{nt} (n={n}): " + " & ".join(row_vals))


if __name__ == "__main__":
    save_cifar_n_accuracy_bars()
    save_cifar_n_vs_synthetic()
    print_cifar_n_table()
    print("Done.")
