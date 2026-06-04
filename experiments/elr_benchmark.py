"""ELR baseline for the CIFAR-10 noisy-label benchmark (Liu et al., NeurIPS 2020).

Mirrors experiments/cifar10_noisy_label_benchmark.py exactly in data loading, noise
injection, ConvNet architecture, optimizer/scheduler, epochs, and evaluation metrics so
rows are directly comparable to cifar10_noisy_label_full.csv. Trains a single method
(objective column = "elr") that needs per-sample indices for the temporal-ensemble buffer.

loss = CE + lam * log(1 - <p_i, t_i>),  t_i <- beta * t_i + (1 - beta) * p_i.

Outputs:
  reports/results/elr_full.csv
  reports/results/elr_aggregated.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from cifar10_noisy_label_benchmark import (
    NOISE_REGIMES,
    ConvNet,
    _append_rows,
    _compute_ece,
    _overwrite_rows,
    aggregate_rows,
    inject_asymmetric_noise,
    inject_symmetric_noise,
    load_cifar10_subset,
    random_crop_flip,
)

from fisher_rao_ml.device import get_device
from fisher_rao_ml.elr import ELRLoss

N_CLASSES = 10

REGIMES = ("clean", "sym_20", "sym_40", "sym_60", "asym_40")


def train_and_eval_elr(
    x_tr: np.ndarray,
    y_tr_noisy: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
    device: torch.device,
    beta: float,
    lam: float,
    n_epochs: int = 60,
    batch_size: int = 128,
    lr: float = 0.05,
) -> dict:
    torch.manual_seed(seed)
    model = ConvNet(n_classes=N_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    n = len(x_tr)
    warmup_epochs = 5
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    elr_loss = ELRLoss(n_samples=n, n_classes=N_CLASSES, beta=beta, lam=lam).to(device)
    labels_t = torch.from_numpy(y_tr_noisy).long()

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
            yb = labels_t[batch_idx].to(device)
            idx_t = torch.from_numpy(batch_idx).long().to(device)
            xb = random_crop_flip(xb)
            opt.zero_grad()
            logits = model(xb)
            loss = elr_loss(logits, yb, idx_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        scheduler.step()

    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(x_te), batch_size):
            xb = x_te_t[start:start + batch_size].to(device)
            probs = torch.softmax(model(xb), dim=-1).cpu()
            all_probs.append(probs)
    probs_te = torch.cat(all_probs, dim=0).numpy()

    y_true = y_te
    y_pred = probs_te.argmax(axis=1)
    acc = float((y_pred == y_true).mean())

    confidences = probs_te.max(axis=1)
    ece = _compute_ece(confidences, y_pred == y_true, n_bins=10)

    oh_te = np.zeros((len(y_true), N_CLASSES), dtype=np.float32)
    oh_te[np.arange(len(y_true)), y_true] = 1.0
    brier = float(np.mean(np.sum((probs_te - oh_te) ** 2, axis=1)))
    nll = float(-np.mean(np.log(np.clip(probs_te[np.arange(len(y_true)), y_true], 1e-8, 1.0))))

    return {"eval_accuracy": acc, "eval_ece": ece, "eval_brier": brier, "eval_nll": nll}


def _load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r["noise_regime"], r["seed"]) for r in csv.DictReader(f)}


def main() -> None:
    p = argparse.ArgumentParser(description="ELR baseline for CIFAR-10 noisy-label benchmark")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--beta", type=float, default=0.7)
    p.add_argument("--lam", type=float, default=3.0)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--regimes", nargs="+", default=list(REGIMES))
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-full", default="reports/results/elr_full.csv")
    p.add_argument("--out-aggregated", default="reports/results/elr_aggregated.csv")
    args = p.parse_args()

    device = get_device()
    out_full = Path(args.out_full)
    done = set() if args.force else _load_done(out_full)
    if args.force and out_full.exists():
        out_full.unlink()

    print(
        f"[elr] device={device}, {args.n_train} train, {args.n_test} test, "
        f"{args.n_epochs} epochs, beta={args.beta}, lam={args.lam}"
    )

    new_rows: list[dict] = []
    for seed in range(args.seeds):
        print(f"\n[elr] loading CIFAR-10 subset seed={seed}")
        x_tr_raw, y_tr_raw, x_te, y_te = load_cifar10_subset(
            n_train=args.n_train, n_test=args.n_test, seed=seed
        )

        for noise_regime in args.regimes:
            noise_type, noise_rate = NOISE_REGIMES[noise_regime]
            rng = np.random.default_rng(seed * 1000 + int(noise_rate * 100))
            if noise_rate == 0.0:
                y_tr_noisy = y_tr_raw.copy()
            elif noise_type == "sym":
                y_tr_noisy = inject_symmetric_noise(y_tr_raw, noise_rate, N_CLASSES, rng)
            else:
                y_tr_noisy = inject_asymmetric_noise(y_tr_raw, noise_rate, N_CLASSES, rng)

            key = (noise_regime, str(seed))
            if key in done:
                print(f"[elr] skip {noise_regime} seed={seed}")
                continue
            print(f"[elr] {noise_regime} elr seed={seed}")
            metrics = train_and_eval_elr(
                x_tr_raw, y_tr_noisy, x_te, y_te,
                seed=seed, device=device, beta=args.beta, lam=args.lam,
                n_epochs=args.n_epochs,
            )
            row = {
                "noise_regime": noise_regime,
                "objective": "elr",
                "seed": seed,
                **metrics,
            }
            new_rows.append(row)
            done.add(key)
            _append_rows(out_full, [row])

    print(f"\n[elr] wrote {len(new_rows)} new rows → {out_full}")

    all_rows = list(csv.DictReader(out_full.open())) if out_full.exists() else []
    if all_rows:
        agg = aggregate_rows(all_rows)
        _overwrite_rows(Path(args.out_aggregated), agg)
        print(f"[elr] aggregated → {args.out_aggregated}")


if __name__ == "__main__":
    main()
