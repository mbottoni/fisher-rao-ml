"""Aggregate ML stress tests for categorical distribution objectives."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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
        for key, value in row.items():
            if key.startswith("eval_") and value != "" and key not in metrics:
                metrics.append(key)
    return metrics


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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
    grouped: dict[tuple[str, str, str, float, str], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in metrics}
    )
    for row in rows:
        key = (
            row["experiment"],
            row["dataset"],
            row["corruption_type"],
            float(row["stress_level"]),
            row["objective"],
        )
        for metric in metrics:
            value = row.get(metric, "")
            if value != "":
                grouped[key][metric].append(float(value))

    records: list[dict[str, object]] = []
    for (experiment, dataset, corruption_type, level, objective), per_metric in sorted(
        grouped.items()
    ):
        record: dict[str, object] = {
            "experiment": experiment,
            "dataset": dataset,
            "corruption_type": corruption_type,
            "stress_level": level,
            "objective": objective,
        }
        for metric, values in per_metric.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            record[f"{metric}_mean"] = float(arr.mean())
            record[f"{metric}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            record[f"{metric}_n"] = int(arr.size)
        records.append(record)
    return records


def paired_significance(
    rows: list[dict[str, str]],
    metrics: list[str],
    baseline: str = "kl",
) -> list[dict[str, object]]:
    paired: dict[tuple[str, str, str, float, int], dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        key = (
            row["experiment"],
            row["dataset"],
            row["corruption_type"],
            float(row["stress_level"]),
            int(row["seed"]),
        )
        paired[key].setdefault(row["objective"], {})
        for metric in metrics:
            value = row.get(metric, "")
            if value != "":
                paired[key][row["objective"]][metric] = float(value)

    grouped: dict[
        tuple[str, str, str, float, str], dict[str, list[tuple[float, float]]]
    ] = defaultdict(lambda: {metric: [] for metric in metrics})
    for (experiment, dataset, corruption_type, level, _seed), bundle in paired.items():
        if baseline not in bundle:
            continue
        for objective, objective_metrics in bundle.items():
            if objective == baseline:
                continue
            for metric in metrics:
                baseline_value = bundle[baseline].get(metric)
                objective_value = objective_metrics.get(metric)
                if baseline_value is None or objective_value is None:
                    continue
                grouped[(experiment, dataset, corruption_type, level, objective)][metric].append(
                    (baseline_value, objective_value)
                )

    records: list[dict[str, object]] = []
    for (experiment, dataset, corruption_type, level, objective), per_metric in sorted(
        grouped.items()
    ):
        record: dict[str, object] = {
            "experiment": experiment,
            "dataset": dataset,
            "corruption_type": corruption_type,
            "stress_level": level,
            "objective": objective,
            "baseline": baseline,
        }
        for metric, pairs in per_metric.items():
            if not pairs:
                continue
            baseline_values = [pair[0] for pair in pairs]
            objective_values = [pair[1] for pair in pairs]
            diffs = [value - baseline_value for baseline_value, value in pairs]
            stat, pvalue = safe_wilcoxon(diffs)
            positive = sum(diff > 0 for diff in diffs)
            negative = sum(diff < 0 for diff in diffs)
            ties = len(diffs) - positive - negative
            record[f"{metric}_n"] = len(pairs)
            record[f"{metric}_mean_diff"] = float(np.mean(diffs))
            record[f"{metric}_wilcoxon_stat"] = stat
            record[f"{metric}_wilcoxon_p"] = pvalue
            record[f"{metric}_cliffs_delta"] = cliffs_delta(objective_values, baseline_values)
            record[f"{metric}_positive_seed_count"] = positive
            record[f"{metric}_negative_seed_count"] = negative
            record[f"{metric}_tie_seed_count"] = ties
            record[f"{metric}_direction_consistency"] = max(positive, negative, ties) / len(diffs)
        records.append(record)
    return records


def oriented_improvement(row: dict[str, object], metric: str, sign: float) -> float:
    return safe_float(row.get(f"{metric}_mean_diff")) * sign


def power_summary(significance_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    specs = [
        ("accuracy", "eval_accuracy", 1.0),
        ("nll", "eval_nll", -1.0),
        ("ece", "eval_ece", -1.0),
        ("brier", "eval_brier", -1.0),
        ("teacher_error_imitation", "eval_teacher_error_imitation", -1.0),
        ("accuracy_on_teacher_wrong", "eval_accuracy_on_teacher_wrong", 1.0),
    ]
    records: list[dict[str, object]] = []
    groups = sorted(
        {
            (row["experiment"], row["objective"])
            for row in significance_rows
            if row["objective"] == "fisher_rao"
        }
    )
    for experiment, objective in groups:
        rows = [
            row
            for row in significance_rows
            if row["experiment"] == experiment
            and row["objective"] == objective
            and float(row["stress_level"]) > 0
        ]
        for label, metric, sign in specs:
            available = [row for row in rows if row.get(f"{metric}_mean_diff") is not None]
            if not available:
                continue
            improved = [row for row in available if oriented_improvement(row, metric, sign) > 0]
            significant = [
                row
                for row in improved
                if safe_float(row.get(f"{metric}_wilcoxon_p")) < 0.05
            ]
            records.append(
                {
                    "experiment": experiment,
                    "objective": objective,
                    "metric_label": label,
                    "metric": metric,
                    "n_cells": len(available),
                    "n_improves_over_kl": len(improved),
                    "n_improves_p_lt_0_05": len(significant),
                    "mean_oriented_improvement": float(
                        np.mean([oriented_improvement(row, metric, sign) for row in available])
                    ),
                }
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ML stress benchmark CSVs.")
    parser.add_argument("--output-dir", default="reports/results")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "reports/results/ml_stress_soft_label.csv",
            "reports/results/ml_stress_distillation.csv",
        ],
    )
    parser.add_argument("--full", default="reports/results/ml_stress_full.csv")
    parser.add_argument("--aggregated", default="reports/results/ml_stress_aggregated.csv")
    parser.add_argument("--significance", default="reports/results/ml_stress_significance.csv")
    parser.add_argument("--power-summary", default="reports/results/ml_stress_power_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for path in args.inputs:
        rows.extend(read_rows(Path(path)))
    metrics = metric_columns(rows)
    print(f"[ml-aggregate] Read {len(rows)} rows and {len(metrics)} metrics")
    write_rows(Path(args.full), rows)

    aggregated = aggregate(rows, metrics)
    write_rows(Path(args.aggregated), aggregated)
    print(f"[ml-aggregate] Wrote {len(aggregated)} rows to {args.aggregated}")

    significance = paired_significance(rows, metrics)
    write_rows(Path(args.significance), significance)
    print(f"[ml-aggregate] Wrote {len(significance)} rows to {args.significance}")

    power = power_summary(significance)
    write_rows(Path(args.power_summary), power)
    print(f"[ml-aggregate] Wrote {len(power)} rows to {args.power_summary}")


if __name__ == "__main__":
    main()
