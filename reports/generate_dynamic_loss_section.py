"""Generate the LaTeX Appendix A dynamic loss section from dynamic_loss_full.csv.

Run after the dynamic-loss experiment completes.
Prints stats and LaTeX table ready to paste into fr_noisy_labels.tex Appendix A.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

RESULTS = Path("reports/results")

NOISE_TYPES = ["clean", "sym_20", "sym_40", "sym_60", "asym_40"]
NOISE_LABELS = {
    "clean": "Clean", "sym_20": "Sym\\,20\\%",
    "sym_40": "Sym\\,40\\%", "sym_60": "Sym\\,60\\%", "asym_40": "Asym\\,40\\%",
}

BASELINES = ["kl", "fisher_rao", "hellinger", "gce", "mae", "sce"]
SCHEDULES = ["fr_then_gce", "fr_then_kl", "gce_then_fr", "kl_then_gce", "fr_then_mae"]

LABELS = {
    "kl": "KL (CE)", "fisher_rao": "FR", "hellinger": "Hellinger",
    "gce": "GCE", "mae": "MAE", "sce": "SCE",
    "fr_then_gce": "FR\\textrightarrow GCE", "fr_then_kl": "FR\\textrightarrow KL",
    "gce_then_fr": "GCE\\textrightarrow FR", "kl_then_gce": "KL\\textrightarrow GCE",
    "fr_then_mae": "FR\\textrightarrow MAE",
}

ALL_SCHEDULES = BASELINES + SCHEDULES


def generate_section() -> None:
    path = RESULTS / "dynamic_loss_full.csv"
    if not path.exists():
        print("dynamic_loss_full.csv not found.")
        return

    with path.open() as f:
        rows = list(csv.DictReader(f))

    # Store by noise_type × schedule × seed
    data: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in rows:
        data[r["noise_regime"]][r["schedule"]][int(r["seed"])] = float(r["eval_accuracy"])

    # Check completeness
    for nt in NOISE_TYPES:
        min_seeds = min(len(data[nt].get(s, {})) for s in ALL_SCHEDULES)
        print(f"# {nt}: {min_seeds} seeds complete (need ≥3 for p-values)")

    print()

    # Compute stats vs KL and vs best static baseline
    print("% === DYNAMIC LOSS SECTION STATS ===")
    print()

    for nt in NOISE_TYPES:
        kl_by_seed = data[nt].get("kl", {})
        kl_seeds = sorted(kl_by_seed.keys())
        if not kl_seeds:
            continue
        kl_vals = [kl_by_seed[s] for s in kl_seeds]
        kl_mean = np.mean(kl_vals) * 100
        print(f"% --- {nt} (KL={kl_mean:.1f}%) ---")

        # Find best static baseline (by mean)
        best_static_obj = "kl"
        best_static_mean = kl_mean
        for obj in BASELINES:
            if obj == "kl":
                continue
            obj_d = data[nt].get(obj, {})
            paired = [s for s in kl_seeds if s in obj_d]
            if len(paired) >= 3:
                m = np.mean([obj_d[s] for s in paired]) * 100
                if m > best_static_mean:
                    best_static_mean = m
                    best_static_obj = obj

        for sched in SCHEDULES:
            sched_d = data[nt].get(sched, {})
            paired_kl = [s for s in kl_seeds if s in sched_d]
            n = len(paired_kl)
            if n < 2:
                print(f"#  {sched}: only {n} seeds")
                continue
            sched_vals = [sched_d[s] for s in paired_kl]
            kl_paired = [kl_by_seed[s] for s in paired_kl]
            mean = np.mean(sched_vals) * 100
            diff = mean - kl_mean
            wins = sum(v > k for v, k in zip(sched_vals, kl_paired, strict=False))
            p_str = ""
            if n >= 3:
                try:
                    _, p = scipy_stats.wilcoxon(sched_vals, kl_paired)
                    p_str = f"p={p:.4f}"
                except Exception:
                    p_str = "n/a"
            print(f"#  {sched}: {mean:.1f}% ({diff:+.1f}%, {wins}/{n} vs KL, {p_str})")

            # vs best static
            if best_static_obj != "kl":
                best_d = data[nt].get(best_static_obj, {})
                paired_best = [s for s in paired_kl if s in best_d]
                if len(paired_best) >= 3:
                    best_v = [best_d[s] for s in paired_best]
                    sched_v2 = [sched_d[s] for s in paired_best]
                    diff2 = np.mean(sched_v2) * 100 - np.mean(best_v) * 100
                    wins2 = sum(v > b for v, b in zip(sched_v2, best_v, strict=False))
                    try:
                        _, p2 = scipy_stats.wilcoxon(sched_v2, best_v)
                        n_b = len(paired_best)
                        msg = f"#    vs {best_static_obj}: {diff2:+.1f}%, {wins2}/{n_b} wins"
                        print(f"{msg}, p={p2:.4f}")
                    except Exception:
                        pass
        print()

    # Build LaTeX table: all baselines + fr_then_gce column
    print("% === LaTeX table (baselines + FR→GCE) ===")
    print("\\begin{table}[t]")
    print("  \\centering")
    print("  \\caption{Dynamic loss-switching results on CIFAR-10 ConvNet (5 seeds).")
    print("    FR\\textrightarrow GCE: phase\\,1 FR (epochs 0--29), phase\\,2 GCE (30--59).")
    print("    Bold: $p<0.05$ better than KL within this benchmark; underline: $p<0.05$ worse.}")
    print("  \\label{tab:dynamic_loss}")
    print("  \\small")
    ncols = len(BASELINES) + 1  # baselines + fr_then_gce
    print(f"  \\begin{{tabular}}{{l{'c' * ncols}}}")
    print("    \\toprule")
    col_headers = " & ".join([LABELS[o] for o in BASELINES] + [LABELS["fr_then_gce"]])
    print(f"    Noise & {col_headers} \\\\")
    print("    \\midrule")

    for nt in NOISE_TYPES:
        kl_by_seed = data[nt].get("kl", {})
        kl_seeds = sorted(kl_by_seed.keys())
        if not kl_seeds:
            continue
        kl_vals = [kl_by_seed[s] for s in kl_seeds]
        cells = [NOISE_LABELS[nt]]
        for obj in BASELINES + ["fr_then_gce"]:
            obj_d = data[nt].get(obj, {})
            paired = [s for s in kl_seeds if s in obj_d]
            if not paired:
                cells.append("---")
                continue
            vals = [obj_d[s] for s in paired]
            mean = np.mean(vals) * 100
            cell = f"{mean:.1f}"
            if obj != "kl" and len(paired) >= 3:
                kl_p = [kl_by_seed[s] for s in paired]
                try:
                    _, p = scipy_stats.wilcoxon(vals, kl_p)
                    if p < 0.05 and np.mean(vals) > np.mean(kl_p):
                        cell = f"\\textbf{{{cell}}}"
                    elif p < 0.05 and np.mean(vals) < np.mean(kl_p):
                        cell = f"\\underline{{{cell}}}"
                except Exception:
                    pass
            cells.append(cell)
        print(f"    {' & '.join(cells)} \\\\")

    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    generate_section()
