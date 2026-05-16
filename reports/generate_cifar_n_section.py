"""Generate the LaTeX §3.3 CIFAR-N section from cifar_n_full.csv.

Run after the CIFAR-N experiment completes (90 rows).
Prints the complete LaTeX for the CIFAR-N results table and surrounding text,
ready to paste into fr_noisy_labels.tex §3.3.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

RESULTS = Path("reports/results")

OBJ_ORDER = ["kl", "fisher_rao", "hellinger", "gce", "mae", "sce"]
OBJ_LABELS = {
    "kl": "KL (CE)", "fisher_rao": "Fisher--Rao", "hellinger": "Hellinger",
    "gce": "GCE", "mae": "MAE", "sce": "SCE",
}
NOISE_TYPES = ["aggre", "random1", "worse"]
NOISE_LABELS = {
    "aggre": "Aggre ($\\approx$9\\%)", "random1": "Random1 ($\\approx$17\\%)",
    "worse": "Worse ($\\approx$40\\%)",
}
NOISE_RATES = {"aggre": 0.09, "random1": 0.17, "worse": 0.40}


def generate_section() -> None:
    path = RESULTS / "cifar_n_full.csv"
    if not path.exists():
        print("cifar_n_full.csv not found.")
        return

    with path.open() as f:
        rows = list(csv.DictReader(f))

    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["noise_type"]][r["objective"]].append(float(r["eval_accuracy"]))

    # Only show noise types with complete data (all 6 objectives, at least 3 seeds)
    noise_types = []
    for nt in NOISE_TYPES:
        if nt not in data:
            continue
        min_seeds = min(len(data[nt].get(o, [])) for o in OBJ_ORDER)
        if min_seeds < 3:
            print(f"# {nt}: only {min_seeds} seeds — excluding from table")
            continue
        noise_types.append(nt)

    if not noise_types:
        print("# Insufficient data for any noise type.")
        return

    # Build table
    header_objs = " & ".join([OBJ_LABELS[o] for o in OBJ_ORDER])
    table_rows = []
    result_lines = []

    for nt in noise_types:
        kl_vals = sorted(data[nt]["kl"])
        row_cells = [NOISE_LABELS[nt]]
        for obj in OBJ_ORDER:
            vals = sorted(data[nt][obj])
            mean = np.mean(vals) * 100
            # Bold if best; underline if significantly below KL
            cell = f"{mean:.1f}"
            if obj != "kl" and len(kl_vals) >= 3 and len(vals) >= 3:
                try:
                    _, p = scipy_stats.wilcoxon(vals[:len(kl_vals)], kl_vals[:len(vals)])
                    wins = sum(v > k for v, k in zip(vals, kl_vals, strict=False))
                    is_sig_better = p < 0.05 and np.mean(vals) > np.mean(kl_vals)
                    is_sig_worse = p < 0.05 and np.mean(vals) < np.mean(kl_vals)
                    if is_sig_better:
                        cell = f"\\textbf{{{cell}}}"
                    elif is_sig_worse:
                        cell = f"\\underline{{{cell}}}"
                except Exception:
                    pass
            row_cells.append(cell)
        table_rows.append(" & ".join(row_cells) + " \\\\")

        # Text description line
        for obj in ["fisher_rao", "gce", "mae"]:
            vals = sorted(data[nt][obj])
            kl_mean = np.mean(kl_vals) * 100
            obj_mean = np.mean(vals) * 100
            diff = obj_mean - kl_mean
            if len(vals) >= 3 and len(kl_vals) >= 3:
                try:
                    _, p = scipy_stats.wilcoxon(vals[:len(kl_vals)], kl_vals[:len(vals)])
                    wins = sum(v > k for v, k in zip(vals, kl_vals, strict=False))
                    result_lines.append(
                        f"# {nt}/{obj}: KL={kl_mean:.1f}% {OBJ_LABELS[obj]}={obj_mean:.1f}% "
                        f"({diff:+.1f}%, {wins}/{len(vals)} wins, p={p:.3f})"
                    )
                except Exception:
                    pass

    # Print the complete section
    n_seeds = min(len(data[nt][o]) for nt in noise_types for o in OBJ_ORDER if data[nt].get(o))
    print(f"% === CIFAR-N §3.3 SECTION (n={n_seeds} seeds) ===")
    print()
    print("% Summary stats for text:")
    for line in result_lines:
        print(line)
    print()
    print("% LaTeX table:")
    print("\\begin{table}[t]")
    print("  \\centering")
    print(f"  \\caption{{CIFAR-N real human-annotated noisy labels on CIFAR-10 ConvNet "
          f"({n_seeds} seeds). Bold: significantly better than KL ($p<0.05$); "
          f"underline: significantly worse.}}")
    print("  \\label{tab:cifar_n}")
    print("  \\small")
    print("  \\begin{tabular}{l" + "c" * len(OBJ_ORDER) + "}")
    print("    \\toprule")
    print(f"    Condition & {header_objs} \\\\")
    print("    \\midrule")
    for row in table_rows:
        print(f"    {row}")
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    generate_section()
