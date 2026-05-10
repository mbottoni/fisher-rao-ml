"""CIFAR-10 noisy-label benchmark — scaled version of noisy_label_benchmark.py.

Compares 6 objectives on CIFAR-10 with synthetic label noise:
  kl (CE), gce, mae, sce, hellinger, fisher_rao

Noise regimes:
  - clean (0% noise)
  - sym_20, sym_40, sym_60: symmetric uniform noise at 20/40/60%
  - asym_40: asymmetric (class-conditional) noise at 40%

Uses a 4-layer ConvNet on a 10k-sample stratified subset of CIFAR-10 for
fast reproducible results on Apple Silicon MPS (approx 30-60s per run).

Outputs:
  reports/results/cifar10_noisy_label_full.csv
  reports/results/cifar10_noisy_label_aggregated.csv
  reports/results/cifar10_noisy_label_significance.csv
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

def load_cifar10_subset(
    n_train: int,
    n_test: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return stratified CIFAR-10 subset as float32 arrays, shape (N, 3, 32, 32)."""
    try:
        import torchvision.transforms as T
        from torchvision.datasets import CIFAR10
    except ImportError as e:
        raise RuntimeError("torchvision required for CIFAR-10 benchmark") from e

    transform = T.Compose([T.ToTensor()])
    train_ds = CIFAR10(root="data", train=True, download=True, transform=transform)
    test_ds = CIFAR10(root="data", train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)

    def _stratified_sample(ds, n: int) -> tuple[np.ndarray, np.ndarray]:
        labels = np.array([ds[i][1] for i in range(len(ds))])
        classes, counts = np.unique(labels, return_counts=True)
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

    # Normalize with CIFAR-10 mean/std
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    return x_tr, y_tr, x_te, y_te


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

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
    """Each class c → (c+1) % n_classes with probability rate (truck→automobile, etc.)."""
    noisy = y.copy()
    flip = rng.random(len(y)) < rate
    noisy[flip] = (y[flip] + 1) % n_classes
    return noisy


def make_one_hot(y: np.ndarray, n_classes: int) -> torch.Tensor:
    oh = torch.zeros(len(y), n_classes)
    oh.scatter_(1, torch.from_numpy(y).long().unsqueeze(1), 1.0)
    return oh


# ---------------------------------------------------------------------------
# Model: 4-layer ConvNet (fast on MPS, ~85% clean acc with 10k samples)
# ---------------------------------------------------------------------------

class ConvNet(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Dropout2d(0.1),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Dropout2d(0.2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
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


# ---------------------------------------------------------------------------
# Training with augmentation
# ---------------------------------------------------------------------------

def random_crop_flip(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random crop with padding and horizontal flip — vectorized CPU augmentation.

    x must be on CPU. Operates on all batch items in parallel using unfold.
    """
    b, c, h, w = x.shape
    padded = F.pad(x, [pad] * 4, mode="reflect")
    i = torch.randint(0, 2 * pad, (b,), dtype=torch.long)
    j = torch.randint(0, 2 * pad, (b,), dtype=torch.long)
    rows = torch.arange(h, dtype=torch.long).unsqueeze(0) + i.unsqueeze(1)
    cols = torch.arange(w, dtype=torch.long).unsqueeze(0) + j.unsqueeze(1)
    # Index all four dims explicitly — mixing a slice with non-contiguous advanced indices
    # reorders dims to (b,h,w,c); avoid that by indexing channels explicitly too.
    out = padded[
        torch.arange(b, dtype=torch.long).view(b, 1, 1, 1),
        torch.arange(c, dtype=torch.long).view(1, c, 1, 1),
        rows.view(b, 1, h, 1),
        cols.view(b, 1, 1, w),
    ]
    flip_mask = torch.rand(b) > 0.5
    out[flip_mask] = out[flip_mask].flip(-1)
    return out


def train_and_eval(
    x_tr: np.ndarray,
    y_tr_oh: torch.Tensor,
    x_te: np.ndarray,
    y_te: np.ndarray,
    objective: str,
    seed: int,
    device: torch.device,
    n_epochs: int = 60,
    batch_size: int = 128,
    lr: float = 0.05,
) -> dict:
    torch.manual_seed(seed)
    model = ConvNet(n_classes=N_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    n = len(x_tr)
    # Warmup 5 epochs then cosine decay
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

    x_tr_t = torch.from_numpy(x_tr)  # (N, 3, 32, 32)
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
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        scheduler.step()

    model.eval()
    # Evaluate in batches to avoid OOM
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


def _compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_rows(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    metrics = ["eval_accuracy", "eval_ece", "eval_brier", "eval_nll"]
    grouped: dict = defaultdict(list)
    for r in rows:
        k = (r["noise_regime"], r["objective"])
        grouped[k].append(r)
    out = []
    for (noise_regime, objective), rs in sorted(grouped.items()):
        rec: dict = {"noise_regime": noise_regime, "objective": objective}
        for m in metrics:
            vals = [float(r[m]) for r in rs if m in r]
            if vals:
                rec[f"{m}_mean"] = float(np.mean(vals))
                rec[f"{m}_std"] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                rec[f"{m}_n"] = len(vals)
        out.append(rec)
    return out


def significance_rows(rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    from scipy.stats import wilcoxon
    metrics = ["eval_accuracy", "eval_ece", "eval_brier", "eval_nll"]
    by_key: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["noise_regime"], int(r["seed"]))
        by_key[k][r["objective"]].append(r)

    paired: dict = defaultdict(lambda: defaultdict(list))
    for (noise_regime, _seed), obj_dict in by_key.items():
        kl_rs = obj_dict.get("kl", [])
        for obj, rs in obj_dict.items():
            if obj == "kl" or not kl_rs or not rs:
                continue
            for m in metrics:
                kl_v = float(kl_rs[0].get(m, float("nan")))
                obj_v = float(rs[0].get(m, float("nan")))
                paired[(noise_regime, obj)][m].append((kl_v, obj_v))

    out = []
    for (noise_regime, objective), metric_pairs in sorted(paired.items()):
        rec: dict = {"noise_regime": noise_regime, "objective": objective}
        for m, pairs in metric_pairs.items():
            diffs = [o - k for k, o in pairs]
            rec[f"{m}_oriented_gain"] = float(np.mean(diffs))
            n_imp = sum(1 for d in diffs if d > 0)
            rec[f"{m}_n_improves"] = n_imp
            rec[f"{m}_n_pairs"] = len(diffs)
            if len(diffs) >= 6 and len(set(diffs)) > 1:
                try:
                    _, p = wilcoxon(diffs)
                    rec[f"{m}_wilcoxon_p"] = float(p)
                except Exception:
                    rec[f"{m}_wilcoxon_p"] = float("nan")
            else:
                rec[f"{m}_wilcoxon_p"] = float("nan")
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r["noise_regime"], r["objective"], r["seed"]) for r in csv.DictReader(f)}


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def _overwrite_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="CIFAR-10 noisy-label benchmark")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--out-full", default="reports/results/cifar10_noisy_label_full.csv"
    )
    p.add_argument(
        "--out-aggregated",
        default="reports/results/cifar10_noisy_label_aggregated.csv",
    )
    p.add_argument(
        "--out-significance",
        default="reports/results/cifar10_noisy_label_significance.csv",
    )
    args = p.parse_args()

    device = get_device()
    out_full = Path(args.out_full)
    done = set() if args.force else _load_done(out_full)
    if args.force and out_full.exists():
        out_full.unlink()

    print(
        f"[cifar10-noisy-label] device={device}, "
        f"{args.n_train} train, {args.n_test} test, {args.n_epochs} epochs"
    )

    # Load dataset once per seed (different stratified sample each time)
    new_rows: list[dict] = []
    for seed in range(args.seeds):
        print(f"\n[cifar10-noisy-label] loading CIFAR-10 subset seed={seed}")
        x_tr_raw, y_tr_raw, x_te, y_te = load_cifar10_subset(
            n_train=args.n_train, n_test=args.n_test, seed=seed
        )

        for noise_regime, (noise_type, noise_rate) in NOISE_REGIMES.items():
            rng = np.random.default_rng(seed * 1000 + int(noise_rate * 100))
            if noise_rate == 0.0:
                y_tr_noisy = y_tr_raw.copy()
            elif noise_type == "sym":
                y_tr_noisy = inject_symmetric_noise(y_tr_raw, noise_rate, N_CLASSES, rng)
            else:
                y_tr_noisy = inject_asymmetric_noise(y_tr_raw, noise_rate, N_CLASSES, rng)
            y_tr_oh = make_one_hot(y_tr_noisy, N_CLASSES)

            for objective in OBJECTIVES:
                key = (noise_regime, objective, str(seed))
                if key in done:
                    print(f"[cifar10-noisy-label] skip {noise_regime} {objective} seed={seed}")
                    continue
                print(f"[cifar10-noisy-label] {noise_regime} {objective} seed={seed}")
                metrics = train_and_eval(
                    x_tr_raw, y_tr_oh, x_te, y_te,
                    objective=objective, seed=seed, device=device,
                    n_epochs=args.n_epochs,
                )
                row = {
                    "noise_regime": noise_regime,
                    "objective": objective,
                    "seed": seed,
                    **metrics,
                }
                new_rows.append(row)
                done.add(key)
                _append_rows(out_full, [row])

    print(f"\n[cifar10-noisy-label] wrote {len(new_rows)} new rows → {out_full}")

    all_rows = list(csv.DictReader(out_full.open())) if out_full.exists() else []
    if all_rows:
        agg = aggregate_rows(all_rows)
        _overwrite_rows(Path(args.out_aggregated), agg)
        print(f"[cifar10-noisy-label] aggregated → {args.out_aggregated}")

        sig = significance_rows(all_rows)
        _overwrite_rows(Path(args.out_significance), sig)
        print(f"[cifar10-noisy-label] significance → {args.out_significance}")

        print("\n[cifar10-noisy-label] Accuracy summary (vs KL baseline):")
        for r in sig:
            if r["objective"] == "fisher_rao":
                gain = float(r.get("eval_accuracy_oriented_gain", 0))
                p_val = float(r.get("eval_accuracy_wilcoxon_p", float("nan")))
                n_imp = r.get("eval_accuracy_n_improves", "?")
                n_tot = r.get("eval_accuracy_n_pairs", "?")
                print(
                    f"  FR {r['noise_regime']:12s}: "
                    f"gain={gain:+.4f} p={p_val:.3f} win={n_imp}/{n_tot}"
                )


if __name__ == "__main__":
    main()
