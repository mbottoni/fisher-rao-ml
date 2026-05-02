"""Aggregate dimensionality-reduction stress tests for KL vs Fisher-Rao.

Inputs:
    reports/results/dimred_stress_full.csv

Outputs:
    reports/results/dimred_stress_aggregated.csv
    reports/results/dimred_stress_significance.csv
    reports/results/dimred_stress_power_summary.csv
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


def row_corruption_type(row: dict[str, str]) -> str:
    return row.get("corruption_type") or "none"


def aggregate(rows: list[dict[str, str]], metrics: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, float, str], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in metrics}
    )
    for row in rows:
        key = (
            row["experiment"],
            row["dataset"],
            row_corruption_type(row),
            float(row["stress_level"]),
            row["objective"],
        )
        for metric in metrics:
            value = row.get(metric, "")
            if value == "":
                continue
            grouped[key][metric].append(float(value))

    aggregated: list[dict[str, object]] = []
    for (experiment, dataset, corruption_type, stress_level, objective), per_metric in sorted(
        grouped.items()
    ):
        record: dict[str, object] = {
            "experiment": experiment,
            "dataset": dataset,
            "corruption_type": corruption_type,
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
    paired: dict[tuple[str, str, str, float, int], dict[str, dict[str, float]]] = defaultdict(
        lambda: {"kl": {}, "fisher_rao": {}}
    )
    for row in rows:
        objective = row["objective"]
        if objective not in ("kl", "fisher_rao"):
            continue
        key = (
            row["experiment"],
            row["dataset"],
            row_corruption_type(row),
            float(row["stress_level"]),
            int(row["seed"]),
        )
        for metric in metrics:
            value = row.get(metric, "")
            if value == "":
                continue
            paired[key][objective][metric] = float(value)

    grouped: dict[
        tuple[str, str, str, float], dict[str, list[tuple[float, float]]]
    ] = defaultdict(lambda: {metric: [] for metric in metrics})
    for (experiment, dataset, corruption_type, stress_level, _seed), bundle in paired.items():
        if not bundle["kl"] or not bundle["fisher_rao"]:
            continue
        for metric in metrics:
            kl_value = bundle["kl"].get(metric)
            fr_value = bundle["fisher_rao"].get(metric)
            if kl_value is None or fr_value is None:
                continue
            grouped[(experiment, dataset, corruption_type, stress_level)][metric].append(
                (kl_value, fr_value)
            )

    records: list[dict[str, object]] = []
    for (experiment, dataset, corruption_type, stress_level), per_metric in sorted(
        grouped.items()
    ):
        record: dict[str, object] = {
            "experiment": experiment,
            "dataset": dataset,
            "corruption_type": corruption_type,
            "stress_level": stress_level,
        }
        for metric, pairs in per_metric.items():
            if not pairs:
                continue
            kl_values = [pair[0] for pair in pairs]
            fr_values = [pair[1] for pair in pairs]
            diffs = [fr - kl for kl, fr in pairs]
            stat, pvalue = safe_wilcoxon(diffs)
            positive = sum(diff > 0 for diff in diffs)
            negative = sum(diff < 0 for diff in diffs)
            ties = len(diffs) - positive - negative
            record[f"{metric}_n"] = len(pairs)
            record[f"{metric}_mean_diff"] = float(np.mean(diffs)) if diffs else float("nan")
            record[f"{metric}_wilcoxon_stat"] = stat
            record[f"{metric}_wilcoxon_p"] = pvalue
            record[f"{metric}_cliffs_delta"] = cliffs_delta(fr_values, kl_values)
            record[f"{metric}_positive_seed_count"] = positive
            record[f"{metric}_negative_seed_count"] = negative
            record[f"{metric}_tie_seed_count"] = ties
            record[f"{metric}_direction_consistency"] = max(positive, negative, ties) / len(diffs)
        records.append(record)
    return records


def oriented_improvement(row: dict[str, object], metric: str, sign: float) -> float:
    value = row.get(f"{metric}_mean_diff")
    if value is None:
        return float("nan")
    return float(value) * sign


def power_summary(significance_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        row
        for row in significance_rows
        if row["experiment"] == "noisy_affinity" and float(row["stress_level"]) > 0
    ]
    if not rows:
        return []

    summary_specs = [
        ("clean_recall", "eval_neighborhood_recall", 1.0),
        ("bad_edge_preservation", "eval_corrupted_edge_preservation", -1.0),
        ("bad_edge_q_mass", "eval_corrupted_edge_q_mass", -1.0),
        ("trustworthiness", "eval_trustworthiness", 1.0),
        ("local_purity", "eval_local_label_purity", 1.0),
    ]
    records: list[dict[str, object]] = []
    for label, metric, sign in summary_specs:
        available = [row for row in rows if row.get(f"{metric}_mean_diff") is not None]
        improved = [row for row in available if oriented_improvement(row, metric, sign) > 0]
        significant = [
            row
            for row in improved
            if float(row.get(f"{metric}_wilcoxon_p", float("nan"))) < 0.05
        ]
        consistent = [
            row
            for row in improved
            if float(row.get(f"{metric}_direction_consistency", 0.0)) >= 0.8
        ]
        records.append(
            {
                "metric_label": label,
                "metric": metric,
                "n_cells": len(available),
                "n_fr_improves": len(improved),
                "n_fr_improves_p_lt_0_05": len(significant),
                "n_fr_improves_direction_consistency_ge_0_8": len(consistent),
                "mean_oriented_improvement": float(
                    np.mean([oriented_improvement(row, metric, sign) for row in available])
                )
                if available
                else float("nan"),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate dimensionality-reduction stress tests.")
    parser.add_argument("--input", default="reports/results/dimred_stress_full.csv")
    parser.add_argument("--aggregated", default="reports/results/dimred_stress_aggregated.csv")
    parser.add_argument(
        "--significance",
        default="reports/results/dimred_stress_significance.csv",
    )
    parser.add_argument(
        "--power-summary",
        default="reports/results/dimred_stress_power_summary.csv",
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

    power_rows = power_summary(significance_rows)
    write_rows(Path(args.power_summary), power_rows)
    print(f"[stress-aggregate] Wrote {len(power_rows)} rows to {args.power_summary}")

    print("\n[stress-aggregate] Significance head:")
    for record in significance_rows[:6]:
        compact = {
            key: value
            for key, value in record.items()
            if key in {"experiment", "dataset", "corruption_type", "stress_level"}
            or key.endswith("_mean_diff")
        }
        print(f"  {compact}")


if __name__ == "__main__":
    main()
