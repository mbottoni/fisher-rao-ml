"""Adaptive gradient-ratio loss-switching benchmark on CIFAR-10.

The fixed-epoch dynamic schedule (dynamic_loss_benchmark.py, switch at epoch 30)
beats static FR at symmetric noise but never beats static GCE. The gradient-norm
analysis suggests the switch should be data-driven: trigger the FR->GCE transition
exactly when memorization begins, signalled by the noisy/clean gradient-norm ratio.

This script reuses the exact model, data pipeline, and SGD setup of
dynamic_loss_benchmark.py (build_convnet; SGD momentum 0.9, weight decay 1e-3,
cosine annealing, no warmup, 60 epochs) so accuracy numbers are comparable within
that benchmark family. Each epoch it measures the noisy/clean gradient-norm ratio
using fixed held-out clean/noisy probe batches (same measurement as
gradient_norm_analysis.py).

Adaptive trigger rule (schedule adaptive_fr_gce), evaluated once per epoch AFTER
the optimizer step and the per-epoch ratio measurement:

  Phase 1 trains with fisher_rao. No switch is permitted during a burn-in of the
  first --min-epochs epochs. From epoch index >= min_epochs onward, the FR->GCE
  switch fires permanently at the FIRST epoch satisfying EITHER:

    (A) Memorization onset: ratio[e] > --ratio-threshold
        (ratio > 1 means noisy-labelled samples contribute more gradient mass
        than clean ones, i.e. the model has started fitting wrong labels).

    (B) Stagnation fallback: the ratio has not set a new running maximum for
        --patience consecutive epochs (the ratio signal has plateaued, so trigger
        A will not escalate further and the FR phase has settled).

  The switch epoch (the index of the FIRST epoch trained with GCE) is recorded.
  If neither trigger fires before the last epoch, no switch occurs and
  switch_epoch is recorded as -1 (the schedule degenerates to static fisher_rao).

Schedules compared:
  adaptive_fr_gce : FR with the adaptive ratio trigger above.
  fixed30_fr_gce  : FR -> GCE, switch fixed at epoch 30 (dynamic_loss baseline).
  fisher_rao      : static FR baseline.
  gce             : static GCE baseline.

Outputs:
  reports/results/adaptive_switch_full.csv
  (columns: noise_regime, schedule, seed, switch_epoch,
            eval_accuracy, eval_ece, eval_brier, eval_nll)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fisher_rao_ml.device import get_device
from fisher_rao_ml.distribution_losses import distribution_loss_from_logits

SCHEDULES = ("adaptive_fr_gce", "fixed30_fr_gce", "fisher_rao", "gce")

NOISE_REGIMES = {
    "sym_40": ("sym", 0.40),
    "sym_60": ("sym", 0.60),
    "asym_40": ("asym", 0.40),
}

N_CLASSES = 10


def load_cifar10(n_train: int, n_test: int, seed: int) -> tuple:
    try:
        import torchvision.transforms as T
        from torchvision.datasets import CIFAR10
    except ImportError as e:
        raise RuntimeError("torchvision required") from e

    transform = T.Compose([T.ToTensor()])
    train_ds = CIFAR10(root="data", train=True, download=True, transform=transform)
    test_ds = CIFAR10(root="data", train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)

    def _stratified_sample(ds, n: int) -> tuple:
        labels = np.array([ds[i][1] for i in range(len(ds))])
        classes = np.unique(labels)
        n_per_class = max(1, n // len(classes))
        idxs = []
        for c in classes:
            c_idxs = np.where(labels == c)[0]
            chosen = rng.choice(c_idxs, size=min(n_per_class, len(c_idxs)), replace=False)
            idxs.extend(chosen.tolist())
        idxs = np.array(idxs[:n])
        rng.shuffle(idxs)
        x = np.stack([ds[int(i)][0].numpy() for i in idxs])
        y = labels[idxs]
        return x.astype(np.float32), y.astype(np.int64)

    x_tr, y_tr = _stratified_sample(train_ds, n_train)
    x_te, y_te = _stratified_sample(test_ds, n_test)

    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std
    return x_tr, y_tr, x_te, y_te


def inject_noise(
    y: np.ndarray,
    noise_type: str,
    noise_rate: float,
    n_classes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy_labels, is_noisy_mask)."""
    noisy = y.copy()
    is_noisy = np.zeros(len(y), dtype=bool)
    if noise_rate == 0.0:
        return noisy, is_noisy
    if noise_type == "sym":
        n_noisy = int(noise_rate * len(y))
        idx = rng.choice(len(y), size=n_noisy, replace=False)
        for i in idx:
            choices = [c for c in range(n_classes) if c != int(noisy[i])]
            noisy[i] = int(rng.choice(choices))
        is_noisy[idx] = True
    else:
        flip = rng.random(len(y)) < noise_rate
        noisy[flip] = (y[flip] + 1) % n_classes
        is_noisy = flip & (noisy != y)
    return noisy, is_noisy


def make_one_hot(y: np.ndarray, n_classes: int) -> torch.Tensor:
    oh = torch.zeros(len(y), n_classes)
    oh.scatter_(1, torch.from_numpy(y).long().unsqueeze(1), 1.0)
    return oh


def build_convnet(n_classes: int = 10) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, n_classes),
    )


def random_crop_flip(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    b, c, h, w = x.shape
    padded = F.pad(x, [pad] * 4, mode="reflect")
    i = torch.randint(0, 2 * pad, (b,), dtype=torch.long)
    j = torch.randint(0, 2 * pad, (b,), dtype=torch.long)
    rows = torch.arange(h, dtype=torch.long).unsqueeze(0) + i.unsqueeze(1)
    cols = torch.arange(w, dtype=torch.long).unsqueeze(0) + j.unsqueeze(1)
    out = padded[
        torch.arange(b, dtype=torch.long).view(b, 1, 1, 1),
        torch.arange(c, dtype=torch.long).view(1, c, 1, 1),
        rows.view(b, 1, h, 1),
        cols.view(b, 1, 1, w),
    ]
    flip_mask = torch.rand(b) > 0.5
    out[flip_mask] = out[flip_mask].flip(-1)
    return out


def _grad_norm(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_oh_batch: torch.Tensor,
    objective: str,
    device: torch.device,
) -> float:
    """One forward+backward on a probe batch; return grad L2 norm. No optimizer step."""
    model.zero_grad()
    logits = model(x_batch.to(device))
    loss = distribution_loss_from_logits(y_oh_batch.to(device), logits, objective=objective)
    loss.backward()
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return float(total ** 0.5)


def train_and_eval(
    x_tr: np.ndarray,
    y_tr_clean: np.ndarray,
    y_tr_noisy: np.ndarray,
    is_noisy: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    schedule: str,
    seed: int,
    device: torch.device,
    n_epochs: int,
    ratio_threshold: float,
    min_epochs: int,
    patience: int,
    fixed_switch_epoch: int,
    probe_batch_size: int,
    batch_size: int = 128,
    lr: float = 0.05,
) -> dict:
    torch.manual_seed(seed)
    model = build_convnet(n_classes=N_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-4)

    y_tr_oh_noisy = make_one_hot(y_tr_noisy, N_CLASSES)
    x_tr_t = torch.from_numpy(x_tr)
    x_te_t = torch.from_numpy(x_te)
    n = len(x_tr)
    idx_all = np.arange(n)
    rng = np.random.default_rng(seed + 99999)

    clean_idx = np.where(~is_noisy)[0]
    noisy_idx = np.where(is_noisy)[0]
    rng_probe = np.random.default_rng(seed + 12345)
    probe_clean = rng_probe.choice(
        clean_idx, size=min(probe_batch_size, len(clean_idx)), replace=False
    )
    probe_noisy = rng_probe.choice(
        noisy_idx, size=min(probe_batch_size, len(noisy_idx)), replace=False
    )
    x_probe_clean = x_tr_t[probe_clean]
    y_oh_probe_clean = make_one_hot(y_tr_clean[probe_clean], N_CLASSES)
    x_probe_noisy = x_tr_t[probe_noisy]
    y_oh_probe_noisy = make_one_hot(y_tr_noisy[probe_noisy], N_CLASSES)

    switched = schedule in ("gce",)
    switch_epoch = 0 if schedule == "gce" else -1
    best_ratio = -np.inf
    epochs_since_best = 0

    def _current_obj() -> str:
        if schedule == "fisher_rao":
            return "fisher_rao"
        if schedule == "gce":
            return "gce"
        return "gce" if switched else "fisher_rao"

    for epoch in range(n_epochs):
        obj = _current_obj()
        model.train()
        rng.shuffle(idx_all)
        for start in range(0, n, batch_size):
            batch_idx = idx_all[start:start + batch_size]
            xb = x_tr_t[batch_idx].to(device)
            yb = y_tr_oh_noisy[batch_idx].to(device)
            xb = random_crop_flip(xb)
            opt.zero_grad()
            logits = model(xb)
            loss = distribution_loss_from_logits(yb, logits, objective=obj)
            loss.backward()
            opt.step()
        scheduler.step()

        if schedule in ("adaptive_fr_gce", "fixed30_fr_gce") and not switched:
            model.train()
            grad_clean = _grad_norm(model, x_probe_clean, y_oh_probe_clean, "fisher_rao", device)
            grad_noisy = _grad_norm(model, x_probe_noisy, y_oh_probe_noisy, "fisher_rao", device)
            ratio = grad_noisy / max(grad_clean, 1e-8)

            if schedule == "fixed30_fr_gce":
                if epoch + 1 >= fixed_switch_epoch:
                    switched = True
                    switch_epoch = epoch + 1
            else:
                if ratio > best_ratio:
                    best_ratio = ratio
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1
                if epoch + 1 >= min_epochs:
                    trigger_a = ratio > ratio_threshold
                    trigger_b = epochs_since_best >= patience
                    if trigger_a or trigger_b:
                        switched = True
                        switch_epoch = epoch + 1

    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(x_te), batch_size):
            xb = x_te_t[start:start + batch_size].to(device)
            logits = model(xb)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

    probs_te = np.concatenate(all_probs, axis=0)
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

    brier = float(np.mean(np.sum((probs_te - np.eye(N_CLASSES)[y_true]) ** 2, axis=1)))
    nll = float(-np.mean(
        np.log(np.clip(probs_te[np.arange(len(y_true)), y_true], 1e-8, 1.0))
    ))
    return {
        "switch_epoch": switch_epoch,
        "eval_accuracy": accuracy,
        "eval_ece": ece,
        "eval_brier": brier,
        "eval_nll": nll,
    }


def _load_done(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r["noise_regime"], r["schedule"], r["seed"]) for r in csv.DictReader(f)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--ratio-threshold", type=float, default=1.1)
    p.add_argument("--min-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--fixed-switch-epoch", type=int, default=30)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument(
        "--regimes", nargs="+", default=list(NOISE_REGIMES.keys()),
        choices=list(NOISE_REGIMES.keys()),
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default="reports/results/adaptive_switch_full.csv")
    args = p.parse_args()

    device = get_device()
    out = Path(args.out)
    done = set() if args.force else _load_done(out)
    if args.force and out.exists():
        out.unlink()

    print(
        f"[adaptive-switch] device={device}, threshold={args.ratio_threshold}, "
        f"min_epochs={args.min_epochs}, patience={args.patience}, "
        f"{args.n_train} train, {args.n_epochs} epochs"
    )

    for seed in range(args.seeds):
        x_tr, y_clean, x_te, y_te = load_cifar10(
            n_train=args.n_train, n_test=args.n_test, seed=seed
        )
        rng = np.random.default_rng(seed + 42)

        for noise_name in args.regimes:
            noise_type, noise_rate = NOISE_REGIMES[noise_name]
            y_noisy, is_noisy = inject_noise(y_clean, noise_type, noise_rate, N_CLASSES, rng)

            for schedule in SCHEDULES:
                key = (noise_name, schedule, str(seed))
                if key in done:
                    print(f"[adaptive-switch] skip seed={seed} {noise_name}/{schedule}")
                    continue

                print(f"[adaptive-switch] seed={seed} {noise_name}/{schedule} ...", flush=True)
                metrics = train_and_eval(
                    x_tr=x_tr,
                    y_tr_clean=y_clean,
                    y_tr_noisy=y_noisy,
                    is_noisy=is_noisy,
                    x_te=x_te,
                    y_te=y_te,
                    schedule=schedule,
                    seed=seed,
                    device=device,
                    n_epochs=args.n_epochs,
                    ratio_threshold=args.ratio_threshold,
                    min_epochs=args.min_epochs,
                    patience=args.patience,
                    fixed_switch_epoch=args.fixed_switch_epoch,
                    probe_batch_size=args.probe_batch_size,
                )
                row = {
                    "noise_regime": noise_name,
                    "schedule": schedule,
                    "seed": seed,
                    **metrics,
                }
                done.add(key)

                write_header = not out.exists()
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if write_header:
                        w.writeheader()
                    w.writerow(row)
                print(
                    f"    switch_epoch={metrics['switch_epoch']} "
                    f"acc={metrics['eval_accuracy']:.4f}",
                    flush=True,
                )

    print(f"\n[adaptive-switch] Done. Results in {args.out}")


if __name__ == "__main__":
    main()
