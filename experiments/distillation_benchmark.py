"""Teacher-student distillation benchmark with corrupted teacher distributions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from soft_label_benchmark import (
    CONFUSERS,
    evaluate_classifier,
    load_dataset,
    one_hot,
    train_classifier,
    wrong_labels,
)

from fisher_rao_ml.device import get_device
from fisher_rao_ml.distribution_losses import OBJECTIVES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrupted-teacher distillation stress tests.")
    parser.add_argument("--output-dir", default="reports/results")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["digits", "mnist"],
        choices=["digits", "mnist"],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[101, 202, 303, 404, 505, 606, 707, 808, 909, 1001],
    )
    parser.add_argument("--objectives", nargs="+", default=list(OBJECTIVES))
    parser.add_argument(
        "--corruption-types",
        nargs="+",
        default=["clean", "low_temperature", "high_temperature", "random_wrong", "class_confusion"],
    )
    parser.add_argument("--corruption-levels", type=float, nargs="+", default=[0.0, 0.1, 0.3])
    parser.add_argument("--teacher-steps", type=int, default=260)
    parser.add_argument("--student-steps", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--mnist-train-samples", type=int, default=1200)
    parser.add_argument("--mnist-eval-samples", type=int, default=500)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predict_probs(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        return torch.softmax(logits, dim=-1).cpu().numpy()


def sharpen_or_soften(probs: np.ndarray, temperature: float) -> np.ndarray:
    adjusted = np.power(np.clip(probs, 1e-8, 1.0), 1.0 / temperature)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def corrupt_teacher_targets(
    teacher_probs: np.ndarray,
    labels: np.ndarray,
    corruption_type: str,
    level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_classes = teacher_probs.shape[1]
    mask = np.zeros(len(labels), dtype=bool)
    if corruption_type == "clean" or level == 0.0:
        return teacher_probs.astype(np.float32), mask
    if corruption_type == "low_temperature":
        temperature = max(0.2, 1.0 - 0.8 * level)
        return (
            sharpen_or_soften(teacher_probs, temperature).astype(np.float32),
            np.ones(len(labels), dtype=bool),
        )
    if corruption_type == "high_temperature":
        temperature = 1.0 + 5.0 * level
        return (
            sharpen_or_soften(teacher_probs, temperature).astype(np.float32),
            np.ones(len(labels), dtype=bool),
        )

    rng = np.random.default_rng(seed)
    mask = rng.random(len(labels)) < level
    targets = teacher_probs.copy()
    if corruption_type == "random_wrong":
        noisy = wrong_labels(labels, rng, n_classes)
    elif corruption_type == "class_confusion":
        noisy = np.array([CONFUSERS[int(label)] for label in labels], dtype=np.int64)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    wrong = one_hot(noisy, n_classes)
    targets[mask] = 0.99 * wrong[mask] + 0.01 / n_classes
    return targets.astype(np.float32), mask


def distillation_metrics(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    corrupted_train_fraction: float,
    device: torch.device,
) -> dict[str, float]:
    metrics = evaluate_classifier(student, x_eval, y_eval, corrupted_train_fraction, device)
    student_probs = predict_probs(student, x_eval, device)
    teacher_probs = predict_probs(teacher, x_eval, device)
    student_pred = student_probs.argmax(axis=1)
    teacher_pred = teacher_probs.argmax(axis=1)
    teacher_wrong = teacher_pred != y_eval
    metrics["eval_teacher_accuracy"] = float((teacher_pred == y_eval).mean())
    metrics["eval_teacher_error_imitation"] = (
        float((student_pred[teacher_wrong] == teacher_pred[teacher_wrong]).mean())
        if np.any(teacher_wrong)
        else 0.0
    )
    metrics["eval_accuracy_on_teacher_wrong"] = (
        float((student_pred[teacher_wrong] == y_eval[teacher_wrong]).mean())
        if np.any(teacher_wrong)
        else 1.0
    )
    metrics["eval_teacher_student_js"] = float(
        0.5
        * (
            teacher_probs
            * (
                np.log(np.clip(teacher_probs, 1e-8, 1.0))
                - np.log(np.clip(0.5 * (teacher_probs + student_probs), 1e-8, 1.0))
            )
            + student_probs
            * (
                np.log(np.clip(student_probs, 1e-8, 1.0))
                - np.log(np.clip(0.5 * (teacher_probs + student_probs), 1e-8, 1.0))
            )
        )
        .sum(axis=1)
        .mean()
    )
    return metrics


def run(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    device = get_device()
    rows: list[dict[str, float | int | str]] = []
    for dataset in args.datasets:
        for seed in args.seeds:
            x_train, x_eval, y_train, y_eval = load_dataset(
                dataset,
                seed,
                args.mnist_train_samples,
                args.mnist_eval_samples,
            )
            teacher_targets = one_hot(y_train, 10)
            teacher = train_classifier(
                x_train,
                teacher_targets,
                "kl",
                args.teacher_steps,
                args.lr,
                seed + 77,
                device,
            )
            teacher_probs = predict_probs(teacher, x_train, device)
            for corruption_type in args.corruption_types:
                levels = [0.0] if corruption_type == "clean" else [
                    level for level in args.corruption_levels if level > 0.0
                ]
                for level in levels:
                    targets, mask = corrupt_teacher_targets(
                        teacher_probs,
                        y_train,
                        corruption_type,
                        level,
                        seed=20_000 + seed,
                    )
                    for objective in args.objectives:
                        student = train_classifier(
                            x_train,
                            targets,
                            objective,
                            args.student_steps,
                            args.lr,
                            seed,
                            device,
                        )
                        metrics = distillation_metrics(
                            student,
                            teacher,
                            x_eval,
                            y_eval,
                            float(mask.mean()),
                            device,
                        )
                        rows.append(
                            {
                                "experiment": "distillation",
                                "dataset": dataset,
                                "corruption_type": corruption_type,
                                "stress_level": level,
                                "seed": seed,
                                "objective": objective,
                                **metrics,
                            }
                        )
                        print(
                            f"[distill] dataset={dataset} type={corruption_type} "
                            f"level={level:g} seed={seed} objective={objective} "
                            f"acc={metrics['eval_accuracy']:.4f} "
                            f"imit={metrics['eval_teacher_error_imitation']:.4f}"
                        )
    return rows


def main() -> None:
    args = parse_args()
    rows = run(args)
    path = Path(args.output_dir) / "ml_stress_distillation.csv"
    write_rows(path, rows)
    print(f"[distill] Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
