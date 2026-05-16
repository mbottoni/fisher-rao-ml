"""ResNet-18 noisy-label benchmark on full CIFAR-10 (50k training samples).

Scales the Direction 1 finding from a 4-layer ConvNet (10k samples) to
ResNet-18 (50k samples), testing whether the architecture-dependent reversal
holds at a larger, more practically relevant scale.

Noise regimes: clean, sym_20, sym_40, sym_60, asym_40
Objectives: kl, gce, mae, sce, hellinger, fisher_rao (same 6 as main benchmark)

Requires torchvision. Recommended: GPU (CUDA or MPS). On MPS with 50k samples
each run takes ~5-10 min; full 5-seed experiment ~12-18 hours. On a single GPU
(A100/V100) expect ~3-5 hours for the full 5-seed run.

Outputs:
  reports/results/resnet_noisy_label_full.csv
  reports/results/resnet_noisy_label_aggregated.csv
  reports/results/resnet_noisy_label_significance.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fisher_rao_ml.device import get_device
from fisher_rao_ml.distribution_losses import distribution_loss_from_logits

OBJECTIVES = ("kl", "gce", "mae", "sce", "hellinger", "fisher_rao")

NOISE_REGIMES = {
    "clean": ("sym", 0.0),
    "sym_20": ("sym", 0.20),
    "sym_40": ("sym", 0.40),
    "sym_60": ("sym", 0.60),
    "asym_40": ("asym", 0.40),
}

N_CLASSES = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cifar10(
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
) -> np.ndarray:
    noisy = y.copy()
    n_noisy = int(rate * len(y))
    idx = rng.choice(len(y), size=n_noisy, replace=False)
    for i in idx:
        choices = [c for c in range(n_classes) if c != int(noisy[i])]
        noisy[i] = int(rng.choice(choices))
    return noisy


def inject_asymmetric_noise(
    y: np.ndarray, rate: float, n_classes: int, rng: np.random.Generator
) -> np.ndarray:
    noisy = y.copy()
    flip = rng.random(len(y)) < rate
    noisy[flip] = (y[flip] + 1) % n_classes
    return noisy


def make_one_hot(y: np.ndarray, n_classes: int) -> torch.Tensor:
    oh = torch.zeros(len(y), n_classes)
    oh.scatter_(1, torch.from_numpy(y).long().unsqueeze(1), 1.0)
    return oh


# ---------------------------------------------------------------------------
# Model: ResNet-18 (from torchvision, adapted for CIFAR-10)
# ---------------------------------------------------------------------------

def build_resnet18(n_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for CIFAR-10: smaller initial conv (3×3, stride 1), no maxpool."""
    try:
        from torchvision.models import resnet18
    except ImportError as e:
        raise RuntimeError("torchvision required for ResNet-18") from e

    model = resnet18(weights=None, num_classes=n_classes)
    # CIFAR adaptation: replace 7×7 stride-2 conv with 3×3 stride-1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # type: ignore[assignment]
    return model


# ---------------------------------------------------------------------------
# Augmentation (same vectorized crop+flip as main benchmark)
# ---------------------------------------------------------------------------

def random_crop_flip(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_eval(
    x_tr: np.ndarray,
    y_tr_oh: torch.Tensor,
    x_te: np.ndarray,
    y_te: np.ndarray,
    objective: str,
    seed: int,
    device: torch.device,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.1,
) -> dict:
    torch.manual_seed(seed)
    model = build_resnet18(n_classes=N_CLASSES).to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )

    n = len(x_tr)
    # Warmup 5 epochs then cosine decay to 0
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
    }


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def _load_done(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r["noise_regime"], r["objective"], r["seed"]) for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-train", type=int, default=50000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-full", default="reports/results/resnet_noisy_label_full.csv")
    p.add_argument(
        "--out-aggregated",
        default="reports/results/resnet_noisy_label_aggregated.csv",
    )
    p.add_argument(
        "--out-significance",
        default="reports/results/resnet_noisy_label_significance.csv",
    )
    args = p.parse_args()

    device = get_device()
    out_full = Path(args.out_full)
    done = set() if args.force else _load_done(out_full)
    if args.force and out_full.exists():
        out_full.unlink()

    print(
        f"[resnet-noisy] device={device}, "
        f"{args.n_train} train, {args.n_test} test, {args.n_epochs} epochs"
    )

    new_rows: list[dict] = []

    for seed in range(args.seeds):
        x_tr, y_clean, x_te, y_te = load_cifar10(
            n_train=args.n_train, n_test=args.n_test, seed=seed
        )
        rng = np.random.default_rng(seed + 42)

        for noise_name, (noise_type, noise_rate) in NOISE_REGIMES.items():
            # Apply noise
            if noise_rate == 0.0:
                y_tr = y_clean.copy()
            elif noise_type == "sym":
                y_tr = inject_symmetric_noise(y_clean, noise_rate, N_CLASSES, rng)
            else:
                y_tr = inject_asymmetric_noise(y_clean, noise_rate, N_CLASSES, rng)

            y_tr_oh = make_one_hot(y_tr, N_CLASSES)

            for obj in OBJECTIVES:
                key = (noise_name, obj, str(seed))
                if key in done:
                    print(f"[resnet-noisy] skip seed={seed} {noise_name}/{obj}")
                    continue

                print(f"[resnet-noisy] seed={seed} {noise_name}/{obj} ...", flush=True)
                metrics = train_and_eval(
                    x_tr, y_tr_oh, x_te, y_te,
                    objective=obj,
                    seed=seed,
                    device=device,
                    n_epochs=args.n_epochs,
                )
                row = {
                    "noise_regime": noise_name,
                    "objective": obj,
                    "seed": seed,
                    **metrics,
                }
                new_rows.append(row)
                done.add(key)

                fieldnames = list(row.keys())
                write_header = not out_full.exists()
                with out_full.open("a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        w.writeheader()
                    w.writerow(row)
                print(f"    acc={metrics['eval_accuracy']:.4f}", flush=True)

    # Aggregate
    all_rows = list(csv.DictReader(out_full.open())) if out_full.exists() else []
    if not all_rows:
        return

    from collections import defaultdict  # noqa: PLC0415

    from scipy import stats as scipy_stats  # noqa: PLC0415

    grouped = defaultdict(list)
    for r in all_rows:
        grouped[(r["noise_regime"], r["objective"])].append(float(r["eval_accuracy"]))

    noise_names_all = sorted(set(r["noise_regime"] for r in all_rows))
    objectives_all = sorted(set(r["objective"] for r in all_rows))

    agg_rows, sig_rows = [], []
    for nr in noise_names_all:
        kl_vals = grouped[(nr, "kl")]
        for obj in objectives_all:
            vals = grouped[(nr, obj)]
            if not vals:
                continue
            agg_rows.append({
                "noise_regime": nr,
                "objective": obj,
                "n_seeds": len(vals),
                "mean_accuracy": np.mean(vals),
                "std_accuracy": np.std(vals),
            })
            if obj != "kl" and len(kl_vals) >= 5 and len(vals) == len(kl_vals):
                stat, p = scipy_stats.wilcoxon(vals, kl_vals)
                wins = sum(v > k for v, k in zip(vals, kl_vals, strict=True))
                sig_rows.append({
                    "noise_regime": nr,
                    "objective": obj,
                    "n_seeds": len(vals),
                    "mean_diff": np.mean(vals) - np.mean(kl_vals),
                    "wins": wins,
                    "wilcoxon_p": p,
                    "significant": p < 0.05,
                })

    Path(args.out_aggregated).parent.mkdir(parents=True, exist_ok=True)
    if agg_rows:
        with open(args.out_aggregated, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader()
            w.writerows(agg_rows)
    if sig_rows:
        with open(args.out_significance, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sig_rows[0].keys()))
            w.writeheader()
            w.writerows(sig_rows)

    print(f"\n[resnet-noisy] Done. Results in {args.out_full}")


if __name__ == "__main__":
    main()
