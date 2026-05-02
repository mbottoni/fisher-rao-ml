"""Aggregate dimensionality-reduction stress tests for KL vs Fisher-Rao.

Inputs:
    reports/results/dimred_stress_full.csv

Outputs:
    reports/results/dimred_stress_aggregated.csv
    reports/results/dimred_stress_significance.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


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


def metric_columns(rows: list[dict[str, str]]) -> list[str]:
    metrics: list[str] = []
    for row in rows:
        for key in row:
            if key.startswith("eval_") and key not in metrics and row[key] != "":
                metrics.append(key)
    return metrics


def cliffs_delta(x: list[float], y: list[float]) -> float:
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
    diffs = [diff for diff in diffs if np.isfinite(diff)]
    if not diffs:
        return float("nan"), float("nan")
    if all(abs(diff) < 1e-12 for diff in diffs):
        return 0.0, 1.0
    try:
        result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan")
    return float(result.statistic), float(result.pvalue)


def aggregate(rows: list[dict[str, str]], metrics: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float, str], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in metrics}
    )
    for row in rows:
        key = (
            row["experiment"],
            row["dataset"],
            float(row["stress_level"]),
            row["objective"],
        )
        for metric in metrics:
            value = row.get(metric, "")
            if value == "":
                continue
            grouped[key][metric].append(float(value))

    aggregated: list[dict[str, object]] = []
    for (experiment, dataset, stress_level, objective), per_metric in sorted(grouped.items()):
        record: dict[str, object] = {
            "experiment": experiment,
            "dataset": dataset,
            "stress_level": stress_level,
            "objective": objective,
        }
        for metric, values in per_metric.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            record[f"{metric}_mean"] = float(arr.mean())
            record[f"{metric}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            record[f"{metric}_n"] = int(arr.size)
        aggregated.append(record)
    return aggregated


def significance(rows: list[dict[str, str]], metrics: list[str]) -> list[dict[str, object]]:
    paired: dict[tuple[str, str, float, int], dict[str, dict[str, float]]] = defaultdict(
        lambda: {"kl": {}, "fisher_rao": {}}
    )
    for row in rows:
        objective = row["objective"]
        if objective not in ("kl", "fisher_rao"):
            continue
        key = (
            row["experiment"],
            row["dataset"],
            float(row["stress_level"]),
            int(row["seed"]),
        )
        for metric in metrics:
            value = row.get(metric, "")
            if value == "":
                continue
            paired[key][objective][metric] = float(value)

    grouped: dict[tuple[str, str, float], dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: {metric: [] for metric in metrics}
    )
    for (experiment, dataset, stress_level, _seed), bundle in paired.items():
        if not bundle["kl"] or not bundle["fisher_rao"]:
            continue
        for metric in metrics:
            kl_value = bundle["kl"].get(metric)
            fr_value = bundle["fisher_rao"].get(metric)
            if kl_value is None or fr_value is None:
                continue
            grouped[(experiment, dataset, stress_level)][metric].append((kl_value, fr_value))

    records: list[dict[str, object]] = []
    for (experiment, dataset, stress_level), per_metric in sorted(grouped.items()):
        record: dict[str, object] = {
            "experiment": experiment,
            "dataset": dataset,
            "stress_level": stress_level,
        }
        for metric, pairs in per_metric.items():
            if not pairs:
                continue
            kl_values = [pair[0] for pair in pairs]
            fr_values = [pair[1] for pair in pairs]
            diffs = [fr - kl for kl, fr in pairs]
            stat, pvalue = safe_wilcoxon(diffs)
            record[f"{metric}_n"] = len(pairs)
            record[f"{metric}_mean_diff"] = float(np.mean(diffs)) if diffs else float("nan")
            record[f"{metric}_wilcoxon_stat"] = stat
            record[f"{metric}_wilcoxon_p"] = pvalue
            record[f"{metric}_cliffs_delta"] = cliffs_delta(fr_values, kl_values)
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate dimensionality-reduction stress tests.")
    parser.add_argument("--input", default="reports/results/dimred_stress_full.csv")
    parser.add_argument("--aggregated", default="reports/results/dimred_stress_aggregated.csv")
    parser.add_argument(
        "--significance",
        default="reports/results/dimred_stress_significance.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.input))
    metrics = metric_columns(rows)
    print(f"[stress-aggregate] Read {len(rows)} rows and {len(metrics)} metrics")

    aggregated = aggregate(rows, metrics)
    write_rows(Path(args.aggregated), aggregated)
    print(f"[stress-aggregate] Wrote {len(aggregated)} rows to {args.aggregated}")

    significance_rows = significance(rows, metrics)
    write_rows(Path(args.significance), significance_rows)
    print(f"[stress-aggregate] Wrote {len(significance_rows)} rows to {args.significance}")

    print("\n[stress-aggregate] Significance head:")
    for record in significance_rows[:6]:
        compact = {
            key: value
            for key, value in record.items()
            if key in {"experiment", "dataset", "stress_level"}
            or key.endswith("_mean_diff")
        }
        print(f"  {compact}")


if __name__ == "__main__":
    main()
