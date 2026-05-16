"""Gradient norm analysis: clean vs noisy sample gradient norms during training.

Tests the bounded-gradient hypothesis: FR's loss is bounded (≤π²), so gradient
norms on noisy samples cannot explode the way KL gradients can when the model
grows confident in the wrong class.

Protocol:
- CIFAR-10 10k subset, sym_40 regime (same setup as main benchmark)
- Every epoch, compute gradient norm on a held-out batch of clean samples
  and a held-out batch of noisy samples (separate forward/backward passes)
- Compare KL vs FR vs GCE vs MAE

Outputs:
  reports/results/gradient_norm_full.csv
  (columns: objective, seed, epoch, sample_type, mean_grad_norm, loss)
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

OBJECTIVES = ("kl", "fisher_rao", "gce", "mae", "hellinger", "sce")
N_CLASSES = 10
NOISE_RATE = 0.40  # sym_40


def load_cifar10_subset(
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torchvision.transforms as T
        from torchvision.datasets import CIFAR10
    except ImportError as e:
        raise RuntimeError("torchvision required") from e

    transform = T.Compose([T.ToTensor()])
    train_ds = CIFAR10(root="data", train=True, download=True, transform=transform)
    test_ds = CIFAR10(root="data", train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)

    def _stratified_sample(ds, n: int) -> tuple[np.ndarray, np.ndarray]:
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


def inject_symmetric_noise(
    y: np.ndarray, rate: float, n_classes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy_labels, is_noisy_mask)."""
    noisy = y.copy()
    n_noisy = int(rate * len(y))
    idx = rng.choice(len(y), size=n_noisy, replace=False)
    for i in idx:
        choices = [c for c in range(n_classes) if c != int(noisy[i])]
        noisy[i] = int(rng.choice(choices))
    is_noisy = np.zeros(len(y), dtype=bool)
    is_noisy[idx] = True
    return noisy, is_noisy


def make_one_hot(y: np.ndarray, n_classes: int) -> torch.Tensor:
    oh = torch.zeros(len(y), n_classes)
    oh.scatter_(1, torch.from_numpy(y).long().unsqueeze(1), 1.0)
    return oh


class ConvNet(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


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


def compute_grad_norm_and_loss(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_oh_batch: torch.Tensor,
    objective: str,
    device: torch.device,
) -> tuple[float, float]:
    """One forward+backward on a batch; return (grad_norm, mean_loss). No optimizer step."""
    model.zero_grad()
    xb = x_batch.to(device)
    yb = y_oh_batch.to(device)
    logits = model(xb)
    loss = distribution_loss_from_logits(yb, logits, objective=objective)
    loss.backward()
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().norm(2).item() ** 2
    return float(total_norm ** 0.5), float(loss.item())


def run_objective(
    x_tr: np.ndarray,
    y_tr_clean: np.ndarray,
    y_tr_noisy: np.ndarray,
    is_noisy: np.ndarray,
    objective: str,
    seed: int,
    device: torch.device,
    n_epochs: int,
    batch_size: int,
    probe_batch_size: int,
) -> list[dict]:
    """Train with noisy labels; record gradient norms on clean vs noisy probes each epoch."""
    torch.manual_seed(seed)
    model = ConvNet(n_classes=N_CLASSES).to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True
    )

    n = len(x_tr)
    warmup_epochs = 5
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    y_tr_oh_noisy = make_one_hot(y_tr_noisy, N_CLASSES)
    x_tr_t = torch.from_numpy(x_tr)
    idx_all = np.arange(n)
    rng = np.random.default_rng(seed + 99999)

    # Held-out probing batches (fixed, no augmentation)
    clean_idx = np.where(~is_noisy)[0]
    noisy_idx = np.where(is_noisy)[0]
    rng_probe = np.random.default_rng(seed + 12345)
    n_probe = probe_batch_size
    probe_clean = rng_probe.choice(clean_idx, size=min(n_probe, len(clean_idx)), replace=False)
    probe_noisy = rng_probe.choice(noisy_idx, size=min(n_probe, len(noisy_idx)), replace=False)
    x_probe_clean = x_tr_t[probe_clean]
    y_oh_probe_clean = make_one_hot(y_tr_clean[probe_clean], N_CLASSES)  # true labels for clean
    x_probe_noisy = x_tr_t[probe_noisy]
    y_oh_probe_noisy = make_one_hot(y_tr_noisy[probe_noisy], N_CLASSES)  # corrupted labels

    rows = []

    for epoch in range(n_epochs):
        model.train()
        rng.shuffle(idx_all)
        for start in range(0, n, batch_size):
            batch_idx = idx_all[start:start + batch_size]
            xb = x_tr_t[batch_idx].to(device)
            yb = y_tr_oh_noisy[batch_idx].to(device)
            xb = random_crop_flip(xb)
            opt.zero_grad()
            logits = model(xb)
            loss = distribution_loss_from_logits(yb, logits, objective=objective)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        # Probe gradient norms (no optimizer step, no augmentation)
        model.train()  # keep BN in train mode for consistency
        grad_clean, loss_clean = compute_grad_norm_and_loss(
            model, x_probe_clean, y_oh_probe_clean, objective, device
        )
        grad_noisy, loss_noisy = compute_grad_norm_and_loss(
            model, x_probe_noisy, y_oh_probe_noisy, objective, device
        )

        rows.append({
            "objective": objective,
            "seed": seed,
            "epoch": epoch,
            "sample_type": "clean",
            "mean_grad_norm": grad_clean,
            "mean_loss": loss_clean,
        })
        rows.append({
            "objective": objective,
            "seed": seed,
            "epoch": epoch,
            "sample_type": "noisy",
            "mean_grad_norm": grad_noisy,
            "mean_loss": loss_noisy,
        })

    return rows


def _load_done(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r["objective"], r["seed"]) for r in csv.DictReader(f) if r["epoch"] == "59"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default="reports/results/gradient_norm_full.csv")
    args = p.parse_args()

    device = get_device()
    out = Path(args.out)
    done = set() if args.force else _load_done(out)
    if args.force and out.exists():
        out.unlink()

    print(
        f"[grad-norm] device={device}, noise=sym_{int(NOISE_RATE*100)}, "
        f"{args.n_train} train, {args.n_epochs} epochs"
    )

    objectives = list(OBJECTIVES)

    for seed in range(args.seeds):
        # Load data once per seed
        x_tr, y_clean, x_te, y_te = load_cifar10_subset(
            n_train=args.n_train, n_test=args.n_test, seed=seed
        )
        rng = np.random.default_rng(seed + 42)
        y_noisy, is_noisy = inject_symmetric_noise(y_clean, NOISE_RATE, N_CLASSES, rng)

        noisy_frac = float(np.mean(is_noisy))
        print(f"[grad-norm] seed={seed}, actual noise={noisy_frac:.3f}")

        for obj in objectives:
            key = (obj, str(seed))
            if key in done:
                print(f"  skip {obj} (done)")
                continue

            print(f"  obj={obj} ...", flush=True)
            rows = run_objective(
                x_tr=x_tr,
                y_tr_clean=y_clean,
                y_tr_noisy=y_noisy,
                is_noisy=is_noisy,
                objective=obj,
                seed=seed,
                device=device,
                n_epochs=args.n_epochs,
                batch_size=128,
                probe_batch_size=args.probe_batch_size,
            )

            # Write all epochs for this (obj, seed)
            write_header = not out.exists()
            with out.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if write_header:
                    w.writeheader()
                w.writerows(rows)

            done.add(key)
            final = rows[-1]
            print(
                f"    epoch=59 clean_grad={final['mean_grad_norm']:.4f} "
                f"noisy_grad={rows[-2]['mean_grad_norm']:.4f}",
                flush=True,
            )

    print(f"\n[grad-norm] Done. Results in {args.out}")


if __name__ == "__main__":
    main()
