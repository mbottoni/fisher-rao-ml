"""Aggregate per-seed t-SNE robustness results into mean / std and significance tables.

Inputs:
    reports/results/tsne_robustness_full.csv

Outputs:
    reports/results/tsne_robustness_aggregated.csv
    reports/results/tsne_robustness_significance.csv

The aggregated CSV contains mean and standard deviation of each metric for every
(dataset, noise, objective) cell. The significance CSV contains paired tests comparing
Fisher-Rao against KL across seeds for every (dataset, noise) cell. We use the
Wilcoxon signed-rank test (paired, non-parametric) and Cliff's delta as the effect
size, which is robust to the small-sample regime of 5--10 seeds.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

METRIC_COLUMNS = (
    "eval_trustworthiness",
    "eval_neighborhood_recall",
    "eval_silhouette",
    "eval_knn_accuracy",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cliffs_delta(x: list[float], y: list[float]) -> float:
    """Cliff's delta in [-1, 1]; positive means x tends to exceed y."""
    if not x or not y:
        return float("nan")
    greater = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                greater += 1
            elif xi < yj:
                less += 1
    return (greater - less) / float(len(x) * len(y))


def safe_wilcoxon(diffs: list[float]) -> tuple[float, float]:
    diffs = [d for d in diffs if np.isfinite(d)]
    if not diffs:
        return float("nan"), float("nan")
    if all(abs(d) < 1e-12 for d in diffs):
        return 0.0, 1.0
    try:
        result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan")
    return float(result.statistic), float(result.pvalue)


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in METRIC_COLUMNS}
    )

    for row in rows:
        key = (row["dataset"], row["objective"], float(row["noise_std_fraction"]))
        for metric in METRIC_COLUMNS:
            grouped[key][metric].append(float(row[metric]))

    aggregated: list[dict[str, object]] = []
    for (dataset, objective, noise), per_metric in sorted(grouped.items()):
        record: dict[str, object] = {
            "dataset": dataset,
            "objective": objective,
            "noise_std_fraction": noise,
            "n_seeds": len(next(iter(per_metric.values()))),
        }
        for metric, values in per_metric.items():
            arr = np.asarray(values, dtype=np.float64)
            record[f"{metric}_mean"] = float(arr.mean())
            record[f"{metric}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        aggregated.append(record)
    return aggregated


def significance(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    paired: dict[tuple[str, float, int], dict[str, dict[str, float]]] = defaultdict(
        lambda: {"kl": {}, "fisher_rao": {}}
    )
    for row in rows:
        key = (row["dataset"], float(row["noise_std_fraction"]), int(row["seed"]))
        objective = row["objective"]
        if objective not in ("kl", "fisher_rao"):
            continue
        for metric in METRIC_COLUMNS:
            paired[key][objective][metric] = float(row[metric])

    grouped: dict[tuple[str, float], dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: {metric: [] for metric in METRIC_COLUMNS}
    )
    for (dataset, noise, _seed), bundle in paired.items():
        if not bundle["kl"] or not bundle["fisher_rao"]:
            continue
        for metric in METRIC_COLUMNS:
            kl_value = bundle["kl"].get(metric)
            fr_value = bundle["fisher_rao"].get(metric)
            if kl_value is None or fr_value is None:
                continue
            grouped[(dataset, noise)][metric].append((kl_value, fr_value))

    significance_rows: list[dict[str, object]] = []
    for (dataset, noise), per_metric in sorted(grouped.items()):
        record: dict[str, object] = {
            "dataset": dataset,
            "noise_std_fraction": noise,
        }
        for metric, pairs in per_metric.items():
            kl_values = [pair[0] for pair in pairs]
            fr_values = [pair[1] for pair in pairs]
            diffs = [fr - kl for kl, fr in pairs]
            stat, pvalue = safe_wilcoxon(diffs)
            delta = cliffs_delta(fr_values, kl_values)
            record[f"{metric}_n"] = len(pairs)
            record[f"{metric}_mean_diff"] = float(np.mean(diffs)) if diffs else float("nan")
            record[f"{metric}_wilcoxon_stat"] = stat
            record[f"{metric}_wilcoxon_p"] = pvalue
            record[f"{metric}_cliffs_delta"] = delta
        significance_rows.append(record)
    return significance_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate t-SNE robustness results.")
    parser.add_argument("--input", default="reports/results/tsne_robustness_full.csv")
    parser.add_argument("--aggregated", default="reports/results/tsne_robustness_aggregated.csv")
    parser.add_argument(
        "--significance",
        default="reports/results/tsne_robustness_significance.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.input))
    print(f"[aggregate] Read {len(rows)} rows from {args.input}")

    aggregated = aggregate(rows)
    write_rows(Path(args.aggregated), aggregated)
    print(f"[aggregate] Wrote {len(aggregated)} aggregated rows to {args.aggregated}")

    significance_rows = significance(rows)
    write_rows(Path(args.significance), significance_rows)
    print(f"[aggregate] Wrote {len(significance_rows)} significance rows to {args.significance}")

    print("\n[aggregate] Aggregated head:")
    for record in aggregated[:6]:
        keys = ["dataset", "objective", "noise_std_fraction", "n_seeds"]
        keys += [
            f"{m}_mean"
            for m in (
                "eval_trustworthiness",
                "eval_neighborhood_recall",
                "eval_silhouette",
                "eval_knn_accuracy",
            )
        ]
        compact = {k: record[k] for k in keys if k in record}
        print(f"  {compact}")


if __name__ == "__main__":
    main()
