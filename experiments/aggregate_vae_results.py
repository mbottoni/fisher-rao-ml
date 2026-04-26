"""Aggregate VAE beta-sweep results and compute paired KL vs Fisher-Rao tests."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

METRIC_COLUMNS = (
    "eval_bce_per_pixel",
    "eval_mse",
    "eval_latent_knn_accuracy",
    "eval_latent_linear_accuracy",
    "eval_latent_silhouette",
    "eval_active_units",
    "eval_aggregated_posterior_mmd",
    "eval_sample_test_mmd",
    "eval_sample_pixel_variance",
    "eval_noise_0.25_dropout_0_bce_per_pixel",
    "eval_noise_0.5_dropout_0_bce_per_pixel",
    "eval_noise_0_dropout_0.25_bce_per_pixel",
    "eval_noise_0_dropout_0.5_bce_per_pixel",
    "eval_stable",
)

LOWER_IS_BETTER = {
    "eval_bce_per_pixel",
    "eval_mse",
    "eval_aggregated_posterior_mmd",
    "eval_sample_test_mmd",
    "eval_noise_0.25_dropout_0_bce_per_pixel",
    "eval_noise_0.5_dropout_0_bce_per_pixel",
    "eval_noise_0_dropout_0.25_bce_per_pixel",
    "eval_noise_0_dropout_0.5_bce_per_pixel",
}


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


def finite_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
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


def aggregate_by_beta(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in METRIC_COLUMNS}
    )
    for row in rows:
        key = (row["dataset"], row["regularizer"], float(row["beta"]))
        for metric in METRIC_COLUMNS:
            value = finite_float(row, metric)
            if np.isfinite(value):
                grouped[key][metric].append(value)

    aggregated: list[dict[str, object]] = []
    for (dataset, regularizer, beta), per_metric in sorted(grouped.items()):
        record: dict[str, object] = {
            "dataset": dataset,
            "regularizer": regularizer,
            "beta": beta,
            "n_seeds": len(per_metric["eval_bce_per_pixel"]),
        }
        for metric, values in per_metric.items():
            arr = np.asarray(values, dtype=np.float64)
            record[f"{metric}_mean"] = float(arr.mean()) if arr.size else float("nan")
            record[f"{metric}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        aggregated.append(record)
    return aggregated


def selection_score(row: dict[str, str]) -> float:
    """Lower is better; balances reconstruction and latent usefulness."""
    bce = finite_float(row, "eval_bce_per_pixel")
    knn = finite_float(row, "eval_latent_knn_accuracy")
    stable = finite_float(row, "eval_stable")
    if not np.isfinite(bce) or not np.isfinite(knn) or stable < 1.0:
        return float("inf")
    return bce - 0.05 * knn


def best_beta_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["regularizer"], int(row["seed"]))].append(row)

    selected: list[dict[str, object]] = []
    for (dataset, regularizer, seed), candidates in sorted(grouped.items()):
        best = min(candidates, key=selection_score)
        selected.append(
            {
                "dataset": dataset,
                "regularizer": regularizer,
                "seed": seed,
                "selected_beta": float(best["beta"]),
                "selection_score": selection_score(best),
                **{metric: finite_float(best, metric) for metric in METRIC_COLUMNS},
            }
        )
    return selected


def best_beta_significance(selected: list[dict[str, object]]) -> list[dict[str, object]]:
    paired: dict[tuple[str, int], dict[str, dict[str, float]]] = defaultdict(dict)
    for row in selected:
        paired[(str(row["dataset"]), int(row["seed"]))][str(row["regularizer"])] = {
            metric: float(row[metric]) for metric in METRIC_COLUMNS
        }

    grouped: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: {metric: [] for metric in METRIC_COLUMNS}
    )
    for (dataset, _seed), bundle in paired.items():
        if "kl" not in bundle or "fisher_rao" not in bundle:
            continue
        for metric in METRIC_COLUMNS:
            kl_value = bundle["kl"].get(metric, float("nan"))
            fr_value = bundle["fisher_rao"].get(metric, float("nan"))
            if np.isfinite(kl_value) and np.isfinite(fr_value):
                grouped[dataset][metric].append((kl_value, fr_value))

    rows: list[dict[str, object]] = []
    for dataset, per_metric in sorted(grouped.items()):
        record: dict[str, object] = {"dataset": dataset}
        for metric, pairs in per_metric.items():
            kl_values = [pair[0] for pair in pairs]
            fr_values = [pair[1] for pair in pairs]
            diffs = [fr - kl for kl, fr in pairs]
            if metric in LOWER_IS_BETTER:
                oriented = [-diff for diff in diffs]
            else:
                oriented = diffs
            stat, pvalue = safe_wilcoxon(oriented)
            record[f"{metric}_n"] = len(pairs)
            record[f"{metric}_mean_diff_fr_minus_kl"] = (
                float(np.mean(diffs)) if diffs else float("nan")
            )
            record[f"{metric}_oriented_mean_diff"] = (
                float(np.mean(oriented)) if oriented else float("nan")
            )
            record[f"{metric}_wilcoxon_stat"] = stat
            record[f"{metric}_wilcoxon_p"] = pvalue
            record[f"{metric}_cliffs_delta"] = cliffs_delta(fr_values, kl_values)
        rows.append(record)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate VAE benchmark results.")
    parser.add_argument("--input", default="reports/results/vae_full_metrics.csv")
    parser.add_argument("--by-beta", default="reports/results/vae_by_beta_aggregated.csv")
    parser.add_argument("--best-beta", default="reports/results/vae_best_beta.csv")
    parser.add_argument("--significance", default="reports/results/vae_best_beta_significance.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.input))
    print(f"[aggregate:VAE] Read {len(rows)} rows from {args.input}")
    by_beta = aggregate_by_beta(rows)
    selected = best_beta_rows(rows)
    significance = best_beta_significance(selected)
    write_rows(Path(args.by_beta), by_beta)
    write_rows(Path(args.best_beta), selected)
    write_rows(Path(args.significance), significance)
    print(f"[aggregate:VAE] Wrote {len(by_beta)} beta rows to {args.by_beta}")
    print(f"[aggregate:VAE] Wrote {len(selected)} selected rows to {args.best_beta}")
    print(f"[aggregate:VAE] Wrote {len(significance)} significance rows to {args.significance}")


if __name__ == "__main__":
    main()
