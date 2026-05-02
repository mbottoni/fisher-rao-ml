"""Noisy probability-target classification benchmark for categorical divergences."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets

from fisher_rao_ml.device import get_device
from fisher_rao_ml.distribution_losses import OBJECTIVES, distribution_loss_from_logits

CONFUSERS = {0: 6, 1: 7, 2: 7, 3: 8, 4: 9, 5: 6, 6: 0, 7: 1, 8: 3, 9: 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run noisy soft-label classification stress tests."
    )
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
        default=["clean", "smoothing", "symmetric_noise", "adversarial", "ambiguous"],
    )
    parser.add_argument("--corruption-levels", type=float, nargs="+", default=[0.0, 0.1, 0.3])
    parser.add_argument("--steps", type=int, default=160)
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


def standardize_train_eval(
    x_train: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(x_train)
    return (
        scaler.transform(x_train).astype(np.float32),
        scaler.transform(x_eval).astype(np.float32),
    )


def load_dataset(
    name: str,
    seed: int,
    train_samples: int,
    eval_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if name == "digits":
        bundle = load_digits()
        x_train, x_eval, y_train, y_eval = train_test_split(
            bundle.data.astype(np.float32),
            bundle.target.astype(np.int64),
            test_size=0.3,
            random_state=seed,
            stratify=bundle.target,
        )
        return (*standardize_train_eval(x_train, x_eval), y_train, y_eval)

    full_train = datasets.MNIST(root="data", train=True, download=True)
    full_eval = datasets.MNIST(root="data", train=False, download=True)
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(full_train), size=min(train_samples, len(full_train)), replace=False)
    eval_idx = rng.choice(len(full_eval), size=min(eval_samples, len(full_eval)), replace=False)
    x_train = np.stack(
        [np.asarray(full_train[int(i)][0], dtype=np.float32).reshape(-1) for i in train_idx]
    )
    y_train = np.array([int(full_train[int(i)][1]) for i in train_idx], dtype=np.int64)
    x_eval = np.stack(
        [np.asarray(full_eval[int(i)][0], dtype=np.float32).reshape(-1) for i in eval_idx]
    )
    y_eval = np.array([int(full_eval[int(i)][1]) for i in eval_idx], dtype=np.int64)
    return (*standardize_train_eval(x_train, x_eval), y_train, y_eval)


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    targets = np.zeros((len(labels), n_classes), dtype=np.float32)
    targets[np.arange(len(labels)), labels] = 1.0
    return targets


def wrong_labels(labels: np.ndarray, rng: np.random.Generator, n_classes: int) -> np.ndarray:
    offsets = rng.integers(1, n_classes, size=len(labels))
    return (labels + offsets) % n_classes


def make_targets(
    labels: np.ndarray,
    corruption_type: str,
    level: float,
    seed: int,
    n_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    base = one_hot(labels, n_classes)
    mask = np.zeros(len(labels), dtype=bool)
    if corruption_type == "clean" or level == 0.0:
        return base, mask
    if corruption_type == "smoothing":
        uniform = np.full_like(base, 1.0 / n_classes)
        return (1.0 - level) * base + level * uniform, np.ones(len(labels), dtype=bool)

    rng = np.random.default_rng(seed)
    mask = rng.random(len(labels)) < level
    targets = base.copy()
    if corruption_type == "symmetric_noise":
        noisy = wrong_labels(labels, rng, n_classes)
        targets[mask] = one_hot(noisy[mask], n_classes)
    elif corruption_type == "adversarial":
        noisy = wrong_labels(labels, rng, n_classes)
        wrong = one_hot(noisy, n_classes)
        targets[mask] = 0.99 * wrong[mask] + 0.01 / n_classes
    elif corruption_type == "ambiguous":
        confused = np.array([CONFUSERS[int(label)] for label in labels], dtype=np.int64)
        targets = (1.0 - level) * base + level * one_hot(confused, n_classes)
        mask = np.ones(len(labels), dtype=bool)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    return targets.astype(np.float32), mask


def train_classifier(
    x_train: np.ndarray,
    targets: np.ndarray,
    objective: str,
    steps: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = torch.nn.Linear(x_train.shape[1], targets.shape[1]).to(device)
    x = torch.tensor(x_train, dtype=torch.float32, device=device)
    y = torch.tensor(targets, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = distribution_loss_from_logits(y, model(x), objective=objective)
        loss.backward()
        optimizer.step()
    return model


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > left) & (confidence <= right)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def evaluate_classifier(
    model: torch.nn.Module,
    x_eval: np.ndarray,
    labels: np.ndarray,
    corrupted_train_fraction: float,
    device: torch.device,
) -> dict[str, float]:
    with torch.no_grad():
        logits = model(torch.tensor(x_eval, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    onehot = one_hot(labels, probs.shape[1])
    true_probs = probs[np.arange(len(labels)), labels]
    return {
        "eval_accuracy": float((probs.argmax(axis=1) == labels).mean()),
        "eval_nll": float(-np.log(np.clip(true_probs, 1e-8, 1.0)).mean()),
        "eval_ece": expected_calibration_error(probs, labels),
        "eval_brier": float(np.square(probs - onehot).sum(axis=1).mean()),
        "eval_mean_true_probability": float(true_probs.mean()),
        "eval_train_corrupted_fraction": float(corrupted_train_fraction),
    }


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
            for corruption_type in args.corruption_types:
                levels = [0.0] if corruption_type == "clean" else [
                    level for level in args.corruption_levels if level > 0.0
                ]
                for level in levels:
                    targets, mask = make_targets(
                        y_train,
                        corruption_type,
                        level,
                        seed=10_000 + seed,
                    )
                    for objective in args.objectives:
                        model = train_classifier(
                            x_train,
                            targets,
                            objective,
                            args.steps,
                            args.lr,
                            seed,
                            device,
                        )
                        metrics = evaluate_classifier(
                            model,
                            x_eval,
                            y_eval,
                            float(mask.mean()),
                            device,
                        )
                        rows.append(
                            {
                                "experiment": "soft_label",
                                "dataset": dataset,
                                "corruption_type": corruption_type,
                                "stress_level": level,
                                "seed": seed,
                                "objective": objective,
                                **metrics,
                            }
                        )
                        print(
                            f"[soft-label] dataset={dataset} type={corruption_type} "
                            f"level={level:g} seed={seed} objective={objective} "
                            f"acc={metrics['eval_accuracy']:.4f} ece={metrics['eval_ece']:.4f}"
                        )
    return rows


def main() -> None:
    args = parse_args()
    rows = run(args)
    path = Path(args.output_dir) / "ml_stress_soft_label.csv"
    write_rows(path, rows)
    print(f"[soft-label] Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
