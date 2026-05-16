"""CIFAR-N benchmark: real human-annotated noisy labels on CIFAR-10.

Downloads CIFAR-N labels from the official repository and trains with the same
6 objectives as cifar10_noisy_label_benchmark.py to evaluate behavior under
instance-dependent (non-synthetic) label noise.

Noise conditions (from CIFAR-N, wei2022learning):
  - aggre:   aggregate (majority vote) ~ 9% noise
  - random1: single annotator run 1   ~ 17% noise
  - worse:   worst of 3 annotators    ~ 40% noise

Uses the same 4-layer ConvNet on 10k stratified subsets for consistency
with the existing CIFAR-10 results.

Outputs:
  reports/results/cifar_n_full.csv
  reports/results/cifar_n_aggregated.csv
  reports/results/cifar_n_significance.csv
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fisher_rao_ml.device import get_device
from fisher_rao_ml.distribution_losses import distribution_loss_from_logits

OBJECTIVES = ("kl", "gce", "mae", "sce", "hellinger", "fisher_rao")
N_CLASSES = 10

CIFAR_N_URL = (
    "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-10_human.pt"
)
CIFAR_N_CACHE = Path("data/CIFAR-10_human.pt")

# Approximate noise rates (empirical from CIFAR-N paper)
NOISE_RATES = {
    "aggre": 0.09,
    "random1": 0.17,
    "worse": 0.40,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def download_cifar_n(cache: Path = CIFAR_N_CACHE) -> dict[str, np.ndarray]:
    """Download CIFAR-N human labels to cache and return as numpy arrays."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"[cifar-n] Downloading CIFAR-N labels to {cache} ...")
        urllib.request.urlretrieve(CIFAR_N_URL, cache)
        print("[cifar-n] Download complete.")
    raw = torch.load(cache, map_location="cpu", weights_only=False)
    return {k: np.array(v).astype(np.int64) for k, v in raw.items()}


def load_cifar10_with_cifar_n(
    n_train: int,
    n_test: int,
    noise_type: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return stratified CIFAR-10 subset with CIFAR-N noisy labels."""
    try:
        import torchvision.transforms as T
        from torchvision.datasets import CIFAR10
    except ImportError as e:
        raise RuntimeError("torchvision required") from e

    transform = T.Compose([T.ToTensor()])
    train_ds = CIFAR10(root="data", train=True, download=True, transform=transform)
    test_ds = CIFAR10(root="data", train=False, download=True, transform=transform)

    # Load CIFAR-N human labels (indexed by full 50k training set)
    cifar_n = download_cifar_n()
    key_map = {
        "aggre": "aggre_label",
        "random1": "random_label1",
        "worse": "worse_label",
    }
    noisy_labels_full = cifar_n[key_map[noise_type]]

    rng = np.random.default_rng(seed)

    clean_labels = np.array([train_ds[i][1] for i in range(len(train_ds))])

    def _stratified_sample_train(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        classes = np.unique(clean_labels)
        n_per_class = max(1, n // len(classes))
        idxs = []
        for c in classes:
            c_idxs = np.where(clean_labels == c)[0]
            chosen = rng.choice(c_idxs, size=min(n_per_class, len(c_idxs)), replace=False)
            idxs.extend(chosen.tolist())
        idxs = np.array(idxs[:n])
        rng.shuffle(idxs)
        x = np.stack([train_ds[int(i)][0].numpy() for i in idxs])
        y_noisy = noisy_labels_full[idxs]
        return x.astype(np.float32), y_noisy

    def _stratified_sample_test(n: int) -> tuple[np.ndarray, np.ndarray]:
        test_labels = np.array([test_ds[i][1] for i in range(len(test_ds))])
        classes = np.unique(test_labels)
        n_per_class = max(1, n // len(classes))
        idxs = []
        for c in classes:
            c_idxs = np.where(test_labels == c)[0]
            chosen = rng.choice(c_idxs, size=min(n_per_class, len(c_idxs)), replace=False)
            idxs.extend(chosen.tolist())
        idxs = np.array(idxs[:n])
        x = np.stack([test_ds[int(i)][0].numpy() for i in idxs])
        y = test_labels[idxs]
        return x.astype(np.float32), y

    x_tr, y_tr = _stratified_sample_train(n_train)
    x_te, y_te = _stratified_sample_test(n_test)

    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    return x_tr, y_tr, x_te, y_te


def make_one_hot(y: np.ndarray, n_classes: int) -> torch.Tensor:
    oh = torch.zeros(len(y), n_classes)
    oh.scatter_(1, torch.from_numpy(y).long().unsqueeze(1), 1.0)
    return oh


# ---------------------------------------------------------------------------
# Model (same as cifar10_noisy_label_benchmark.py)
# ---------------------------------------------------------------------------

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
) -> dict:
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
    all_probs, all_true = [], []
    with torch.no_grad():
        for start in range(0, len(x_te), batch_size):
            xb = x_te_t[start:start + batch_size].to(device)
            logits = model(xb)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_true = y_te

    probs_te = np.concatenate(all_probs, axis=0)
    y_true = np.array(all_true)
    preds = probs_te.argmax(axis=1)
    accuracy = float(np.mean(preds == y_true))

    # ECE
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
    nll = float(-np.mean(np.log(np.clip(probs_te[np.arange(len(y_true)), y_true], 1e-8, 1.0))))

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
        return {(r["noise_type"], r["objective"], r["seed"]) for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-full", default="reports/results/cifar_n_full.csv")
    p.add_argument("--out-aggregated", default="reports/results/cifar_n_aggregated.csv")
    p.add_argument("--out-significance", default="reports/results/cifar_n_significance.csv")
    args = p.parse_args()

    device = get_device()
    out_full = Path(args.out_full)
    done = set() if args.force else _load_done(out_full)
    if args.force and out_full.exists():
        out_full.unlink()

    print(
        f"[cifar-n] device={device}, {args.n_train} train, "
        f"{args.n_test} test, {args.n_epochs} epochs"
    )

    noise_types = list(NOISE_RATES.keys())
    new_rows: list[dict] = []

    for seed in range(args.seeds):
        for noise_type in noise_types:
            # Load data once per (seed, noise_type)
            need_objs = [
                obj for obj in OBJECTIVES
                if (noise_type, obj, str(seed)) not in done
            ]
            if not need_objs:
                print(f"[cifar-n] skip seed={seed} noise={noise_type} (all done)")
                continue

            print(f"[cifar-n] loading seed={seed} noise={noise_type} ...")
            x_tr, y_tr, x_te, y_te = load_cifar10_with_cifar_n(
                n_train=args.n_train,
                n_test=args.n_test,
                noise_type=noise_type,
                seed=seed,
            )
            y_tr_oh = make_one_hot(y_tr, N_CLASSES)

            for obj in need_objs:
                key = (noise_type, obj, str(seed))
                if key in done:
                    continue
                print(f"  obj={obj} ...", flush=True)
                metrics = train_and_eval(
                    x_tr, y_tr_oh, x_te, y_te,
                    objective=obj,
                    seed=seed,
                    device=device,
                    n_epochs=args.n_epochs,
                )
                row = {
                    "noise_type": noise_type,
                    "objective": obj,
                    "seed": seed,
                    **metrics,
                }
                new_rows.append(row)
                done.add(key)
                # Append immediately
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
        grouped[(r["noise_type"], r["objective"])].append(float(r["eval_accuracy"]))

    noise_types_all = sorted(set(r["noise_type"] for r in all_rows))
    objectives_all = sorted(set(r["objective"] for r in all_rows))

    agg_rows = []
    sig_rows = []
    for nt in noise_types_all:
        kl_vals = grouped[(nt, "kl")]
        for obj in objectives_all:
            vals = grouped[(nt, obj)]
            if not vals:
                continue
            row_agg = {
                "noise_type": nt,
                "noise_rate": NOISE_RATES.get(nt, 0.0),
                "objective": obj,
                "n_seeds": len(vals),
                "mean_accuracy": np.mean(vals),
                "std_accuracy": np.std(vals),
            }
            agg_rows.append(row_agg)
            if obj != "kl" and len(kl_vals) >= 5 and len(vals) == len(kl_vals):
                stat, p = scipy_stats.wilcoxon(vals, kl_vals)
                wins = sum(v > k for v, k in zip(vals, kl_vals, strict=True))
                sig_rows.append({
                    "noise_type": nt,
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

    print(f"\n[cifar-n] Done. Results in {args.out_full}")


if __name__ == "__main__":
    main()
