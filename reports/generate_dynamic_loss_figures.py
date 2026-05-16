"""Generate figures for the dynamic loss-switching experiment.

Shows whether FR->GCE curriculum outperforms single objectives under symmetric noise.
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

NOISE_TYPES = ["clean", "sym_20", "sym_40", "sym_60", "asym_40"]
NOISE_LABELS = {
    "clean": "Clean", "sym_20": "Sym 20%",
    "sym_40": "Sym 40%", "sym_60": "Sym 60%", "asym_40": "Asym 40%",
}

SCHEDULE_ORDER = [
    "kl", "fisher_rao", "gce", "mae",
    "fr_then_gce", "fr_then_kl", "kl_then_gce", "gce_then_fr", "fr_then_mae",
]
SCHEDULE_LABELS = {
    "kl": "KL (CE)", "fisher_rao": "FR", "gce": "GCE",
    "mae": "MAE", "sce": "SCE", "hellinger": "Hellinger",
    "fr_then_gce": "FR→GCE", "fr_then_kl": "FR→KL",
    "kl_then_gce": "KL→GCE", "gce_then_fr": "GCE→FR",
    "fr_then_mae": "FR→MAE",
}
SCHEDULE_COLORS = {
    "kl": "tab:blue", "fisher_rao": "tab:orange", "gce": "tab:red",
    "mae": "tab:purple", "sce": "tab:brown", "hellinger": "tab:green",
    "fr_then_gce": "gold", "fr_then_kl": "darkorange",
    "kl_then_gce": "steelblue", "gce_then_fr": "tomato",
    "fr_then_mae": "mediumpurple",
}
SCHEDULE_MARKERS = {
    "kl": "o", "fisher_rao": "s", "gce": "D", "mae": "v",
    "fr_then_gce": "*", "fr_then_kl": "^", "kl_then_gce": "P",
    "gce_then_fr": "X", "fr_then_mae": "h",
}


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def save_dynamic_vs_static() -> None:
    """Bar chart: dynamic schedules vs static baselines across noise regimes."""
    path = RESULTS / "dynamic_loss_full.csv"
    if not path.exists():
        print("dynamic_loss_full.csv not found — skipping")
        return

    rows = read_rows(path)
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["noise_regime"]][r["schedule"]].append(float(r["eval_accuracy"]))

    noise_types = [nt for nt in NOISE_TYPES if nt in data]
    focus_schedules = ["kl", "fisher_rao", "gce", "fr_then_gce", "kl_then_gce"]
    available = [s for s in focus_schedules if any(data[nt].get(s) for nt in noise_types)]
    if not available:
        print("insufficient data — skipping")
        return

    n_noise = len(noise_types)
    n_sched = len(available)
    width = 0.15
    x = np.arange(n_noise)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, sched in enumerate(available):
        means, errs = [], []
        for nt in noise_types:
            vals = data[nt].get(sched, [])
            means.append(np.mean(vals) * 100 if vals else np.nan)
            errs.append(np.std(vals) * 100 if len(vals) > 1 else 0.0)
        offset = (i - n_sched / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width,
            label=SCHEDULE_LABELS.get(sched, sched),
            color=SCHEDULE_COLORS.get(sched, "gray"),
            yerr=errs, capsize=3, error_kw={"linewidth": 1},
        )

    ax.set_xlabel("Noise Regime", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Dynamic Loss Switching vs Static Baselines — CIFAR-10 ConvNet", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([NOISE_LABELS.get(nt, nt) for nt in noise_types], fontsize=10)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES / "dynamic_loss_vs_static.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {out.name}")


def save_dynamic_gain_over_fr() -> None:
    """Gain of FR->GCE over FR alone across noise regimes."""
    path = RESULTS / "dynamic_loss_full.csv"
    if not path.exists():
        return

    rows = read_rows(path)
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["noise_regime"]][r["schedule"]].append(float(r["eval_accuracy"]))

    noise_types = [nt for nt in NOISE_TYPES if nt in data]
    comparisons = [
        ("fr_then_gce", "fisher_rao", "FR→GCE vs FR"),
        ("fr_then_gce", "gce", "FR→GCE vs GCE"),
        ("fr_then_gce", "kl", "FR→GCE vs KL"),
        ("kl_then_gce", "kl", "KL→GCE vs KL"),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["tab:orange", "tab:red", "tab:blue", "steelblue"]

    for (sched_a, sched_b, label), color in zip(comparisons, colors, strict=True):
        gains = []
        valid_nt = []
        for nt in noise_types:
            a_vals = data[nt].get(sched_a, [])
            b_vals = data[nt].get(sched_b, [])
            if a_vals and b_vals:
                gains.append((np.mean(a_vals) - np.mean(b_vals)) * 100)
                valid_nt.append(nt)
        if gains:
            ax.plot(
                range(len(valid_nt)), gains, "o-", label=label, color=color, linewidth=2
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(noise_types)))
    ax.set_xticklabels([NOISE_LABELS.get(nt, nt) for nt in noise_types], fontsize=10)
    ax.set_ylabel("Accuracy gain (%)", fontsize=12)
    ax.set_title("Dynamic switching gains over static baselines", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES / "dynamic_loss_gain.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {out.name}")


def print_dynamic_table() -> None:
    """Print LaTeX table of dynamic vs static results."""
    path = RESULTS / "dynamic_loss_full.csv"
    if not path.exists():
        return

    rows = read_rows(path)
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["noise_regime"]][r["schedule"]].append(float(r["eval_accuracy"]))

    schedules = ["kl", "fisher_rao", "gce", "mae", "fr_then_gce", "fr_then_kl", "kl_then_gce"]
    print("\n=== LaTeX Table: Dynamic Loss Switching ===")
    headers = " & ".join([SCHEDULE_LABELS.get(s, s) for s in schedules])
    print(f"{'Regime':<12}  {headers}")

    for nt in NOISE_TYPES:
        if nt not in data:
            continue
        n = max((len(data[nt].get(s, [])) for s in schedules if data[nt].get(s)), default=0)
        row_vals = []
        for s in schedules:
            vals = data[nt].get(s, [])
            row_vals.append(f"{np.mean(vals)*100:.1f}" if vals else "---")
        print(f"{nt} (n={n}): " + " & ".join(row_vals))


if __name__ == "__main__":
    save_dynamic_vs_static()
    save_dynamic_gain_over_fr()
    print_dynamic_table()
    print("Done.")
