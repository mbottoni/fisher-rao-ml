"""FR convergence ablation at high noise (ResNet-18, CIFAR-10, sym_60).

Tests whether fisher_rao's gap behind GCE/SCE at sym_60 on ResNet-18 is partly
a convergence artifact of bounded gradients on deeper networks. Varies training
length and peak learning rate for fisher_rao, anchored against kl, gce, sce.

Fixed setting: sym_60 noise, 10k stratified CIFAR-10 subset.

Grid (per seed):
  fisher_rao × n_epochs {100, 200, 300} × lr_scale {1.0, 2.0, 4.0}  (9 configs)
  kl, gce, sce  at n_epochs=100, lr_scale=1.0                       (3 anchors)
  = 12 configs per seed.

Outputs:
  reports/results/fr_convergence_ablation_full.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from resnet_noisy_label_benchmark import (
    N_CLASSES,
    build_resnet18,
    inject_symmetric_noise,
    load_cifar10,
    make_one_hot,
    random_crop_flip,
)

from fisher_rao_ml.device import get_device
from fisher_rao_ml.distribution_losses import distribution_loss_from_logits

NOISE_RATE = 0.60
BASE_LR = 0.1


def train_and_eval(
    x_tr: np.ndarray,
    y_tr_oh: torch.Tensor,
    y_tr_noisy: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    objective: str,
    seed: int,
    device: torch.device,
    n_epochs: int,
    lr: float,
    batch_size: int = 128,
) -> dict:
    torch.manual_seed(seed)
    model = build_resnet18(n_classes=N_CLASSES).to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )

    n = len(x_tr)
    warmup_epochs = 5
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs - warmup_epochs, eta_min=1e-4
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    x_tr_t = torch.from_numpy(x_tr)
    x_te_t = torch.from_numpy(x_te)
    idx_all = np.arange(n)
    rng = np.random.default_rng(seed + 99999)

    model.train()
    for _epoch in range(n_epochs):
        rng.shuffle(idx_all)
        for start in range(0, n, batch_size):
            batch_idx = idx_all[start:start + batch_size]
            xb = x_tr_t[batch_idx].to(device)
            yb = y_tr_oh[batch_idx].to(device)
            xb = random_crop_flip(xb)
            opt.zero_grad()
            logits = model(xb)
            loss = distribution_loss_from_logits(yb, logits, objective=objective)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(x_te), batch_size):
            xb = x_te_t[start:start + batch_size].to(device)
            logits = model(xb)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs_te = np.concatenate(all_probs, axis=0)

    train_preds = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = x_tr_t[start:start + batch_size].to(device)
            logits = model(xb)
            train_preds.append(logits.argmax(dim=-1).cpu().numpy())
    train_pred = np.concatenate(train_preds, axis=0)
    train_accuracy = float(np.mean(train_pred == y_tr_noisy))

    y_true = np.array(y_te)
    preds = probs_te.argmax(axis=1)
    accuracy = float(np.mean(preds == y_true))

    n_bins = 10
    confidences = probs_te.max(axis=1)
    accuracies_bin = (preds == y_true).astype(float)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            ece += mask.sum() / len(y_true) * abs(
                accuracies_bin[mask].mean() - confidences[mask].mean()
            )

    brier = float(np.mean(
        np.sum((probs_te - np.eye(N_CLASSES)[y_true]) ** 2, axis=1)
    ))
    nll = float(-np.mean(
        np.log(np.clip(probs_te[np.arange(len(y_true)), y_true], 1e-8, 1.0))
    ))

    return {
        "eval_accuracy": accuracy,
        "eval_ece": ece,
        "eval_brier": brier,
        "eval_nll": nll,
        "train_accuracy": train_accuracy,
    }


def build_grid(smoke: bool) -> list[tuple[str, int, float]]:
    if smoke:
        return [
            ("fisher_rao", 1, 1.0),
            ("kl", 1, 1.0),
        ]
    grid: list[tuple[str, int, float]] = []
    for n_epochs in (100, 200, 300):
        for lr_scale in (1.0, 2.0, 4.0):
            grid.append(("fisher_rao", n_epochs, lr_scale))
    for obj in ("kl", "gce", "sce"):
        grid.append((obj, 100, 1.0))
    return grid


def _load_done(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {
            (r["objective"], r["n_epochs"], r["lr_scale"], r["seed"])
            for r in csv.DictReader(f)
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-full", default="reports/results/fr_convergence_ablation_full.csv")
    args = p.parse_args()

    device = get_device()
    out_full = Path(args.out_full)
    done = set() if args.force else _load_done(out_full)
    if args.force and out_full.exists():
        out_full.unlink()

    grid = build_grid(args.smoke)
    print(
        f"[fr-convergence] device={device}, {args.n_train} train, "
        f"{args.n_test} test, sym_60, {len(grid)} configs/seed, smoke={args.smoke}"
    )

    for seed in range(args.seeds):
        x_tr, y_clean, x_te, y_te = load_cifar10(
            n_train=args.n_train, n_test=args.n_test, seed=seed
        )
        rng = np.random.default_rng(seed + 42)
        y_tr = inject_symmetric_noise(y_clean, NOISE_RATE, N_CLASSES, rng)
        y_tr_oh = make_one_hot(y_tr, N_CLASSES)

        for obj, n_epochs, lr_scale in grid:
            key = (obj, str(n_epochs), str(lr_scale), str(seed))
            if key in done:
                print(
                    f"[fr-convergence] skip seed={seed} {obj} "
                    f"ep={n_epochs} lr_scale={lr_scale}"
                )
                continue

            print(
                f"[fr-convergence] seed={seed} {obj} ep={n_epochs} "
                f"lr_scale={lr_scale} ...",
                flush=True,
            )
            metrics = train_and_eval(
                x_tr, y_tr_oh, y_tr, x_te, y_te,
                objective=obj,
                seed=seed,
                device=device,
                n_epochs=n_epochs,
                lr=BASE_LR * lr_scale,
            )
            row = {
                "objective": obj,
                "n_epochs": n_epochs,
                "lr_scale": lr_scale,
                "seed": seed,
                "eval_accuracy": metrics["eval_accuracy"],
                "eval_ece": metrics["eval_ece"],
                "eval_brier": metrics["eval_brier"],
                "eval_nll": metrics["eval_nll"],
                "train_accuracy": metrics["train_accuracy"],
            }
            done.add(key)

            write_header = not out_full.exists()
            out_full.parent.mkdir(parents=True, exist_ok=True)
            with out_full.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)
            print(
                f"    acc={metrics['eval_accuracy']:.4f} "
                f"train_acc={metrics['train_accuracy']:.4f}",
                flush=True,
            )

    print(f"\n[fr-convergence] Done. Results in {args.out_full}")


if __name__ == "__main__":
    main()
