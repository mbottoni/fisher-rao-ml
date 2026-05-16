"""Dynamic loss-switching benchmark on CIFAR-10.

Motivated by the gradient-norm analysis: KL memorizes noisy samples (ratio>1,
increasing), FR bounds gradients (ratio≈1, stable), GCE downweights noisy
samples (ratio<1, decreasing). A two-phase schedule should combine FR's stable
early learning with GCE's active denoising in the later phase.

Phase 1 (epochs 0..switch_epoch-1): loss_a
Phase 2 (epochs switch_epoch..n_epochs-1): loss_b

Schedules tested:
  fr_then_gce   : FR -> GCE  (hypothesis: best at high symmetric noise)
  fr_then_kl    : FR -> KL   (control: does switching help at all?)
  gce_then_fr   : GCE -> FR  (reverse order: does order matter?)
  kl_then_gce   : KL -> GCE  (ablation: does early stability matter?)

Compared against single-objective baselines: kl, fr, gce, mae (from existing CSV
if available, otherwise re-run).

Outputs:
  reports/results/dynamic_loss_full.csv
  reports/results/dynamic_loss_aggregated.csv
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

SCHEDULES = {
    "fr_then_gce": ("fisher_rao", "gce"),
    "fr_then_kl": ("fisher_rao", "kl"),
    "gce_then_fr": ("gce", "fisher_rao"),
    "kl_then_gce": ("kl", "gce"),
    "fr_then_mae": ("fisher_rao", "mae"),
}

BASELINES = ("kl", "fisher_rao", "gce", "mae", "sce", "hellinger")

NOISE_REGIMES = {
    "clean": ("sym", 0.0),
    "sym_20": ("sym", 0.20),
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
) -> np.ndarray:
    if noise_rate == 0.0:
        return y.copy()
    noisy = y.copy()
    if noise_type == "sym":
        n_noisy = int(noise_rate * len(y))
        idx = rng.choice(len(y), size=n_noisy, replace=False)
        for i in idx:
            choices = [c for c in range(n_classes) if c != int(noisy[i])]
            noisy[i] = int(rng.choice(choices))
    else:
        flip = rng.random(len(y)) < noise_rate
        noisy[flip] = (y[flip] + 1) % n_classes
    return noisy


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


def train_and_eval(
    x_tr: np.ndarray,
    y_tr_oh: torch.Tensor,
    x_te: np.ndarray,
    y_te: np.ndarray,
    objective: str | tuple[str, str],
    seed: int,
    device: torch.device,
    n_epochs: int = 60,
    switch_epoch: int | None = None,
    batch_size: int = 128,
    lr: float = 0.05,
) -> dict:
    """Train with optional two-phase objective.

    objective: single str or (phase1_obj, phase2_obj) tuple.
    switch_epoch: epoch at which to switch (must be set when objective is tuple).
    """
    torch.manual_seed(seed)
    model = build_convnet(n_classes=N_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-4)

    x_tr_t = torch.from_numpy(x_tr)
    x_te_t = torch.from_numpy(x_te)
    n = len(x_tr)
    idx_all = np.arange(n)
    rng = np.random.default_rng(seed + 99999)

    def _get_obj(epoch: int) -> str:
        if isinstance(objective, tuple):
            return objective[0] if epoch < switch_epoch else objective[1]
        return objective

    model.train()
    for epoch in range(n_epochs):
        obj = _get_obj(epoch)
        rng.shuffle(idx_all)
        for start in range(0, n, batch_size):
            batch_idx = idx_all[start:start + batch_size]
            xb = x_tr_t[batch_idx].to(device)
            yb = y_tr_oh[batch_idx].to(device)
            xb = random_crop_flip(xb)
            opt.zero_grad()
            logits = model(xb)
            loss = distribution_loss_from_logits(yb, logits, objective=obj)
            loss.backward()
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
    return {"eval_accuracy": accuracy, "eval_ece": ece, "eval_brier": brier, "eval_nll": nll}


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
    p.add_argument("--switch-epoch", type=int, default=30)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-full", default="reports/results/dynamic_loss_full.csv")
    p.add_argument("--out-aggregated", default="reports/results/dynamic_loss_aggregated.csv")
    args = p.parse_args()

    device = get_device()
    out_full = Path(args.out_full)
    done = set() if args.force else _load_done(out_full)
    if args.force and out_full.exists():
        out_full.unlink()

    print(
        f"[dynamic-loss] device={device}, switch_epoch={args.switch_epoch}, "
        f"{args.n_train} train, {args.n_epochs} epochs"
    )

    all_schedules = {
        **{name: obj_pair for name, obj_pair in SCHEDULES.items()},
        **{obj: obj for obj in BASELINES},
    }

    for seed in range(args.seeds):
        x_tr, y_clean, x_te, y_te = load_cifar10(
            n_train=args.n_train, n_test=args.n_test, seed=seed
        )
        rng = np.random.default_rng(seed + 42)

        for noise_name, (noise_type, noise_rate) in NOISE_REGIMES.items():
            y_tr = inject_noise(y_clean, noise_type, noise_rate, N_CLASSES, rng)
            y_tr_oh = make_one_hot(y_tr, N_CLASSES)

            for schedule_name, objective in all_schedules.items():
                key = (noise_name, schedule_name, str(seed))
                if key in done:
                    print(f"[dynamic-loss] skip seed={seed} {noise_name}/{schedule_name}")
                    continue

                switch = args.switch_epoch if isinstance(objective, tuple) else None
                print(
                    f"[dynamic-loss] seed={seed} {noise_name}/{schedule_name} ...",
                    flush=True,
                )
                metrics = train_and_eval(
                    x_tr, y_tr_oh, x_te, y_te,
                    objective=objective,
                    seed=seed,
                    device=device,
                    n_epochs=args.n_epochs,
                    switch_epoch=switch,
                )
                row = {
                    "noise_regime": noise_name,
                    "schedule": schedule_name,
                    "seed": seed,
                    **metrics,
                }
                done.add(key)

                write_header = not out_full.exists()
                with out_full.open("a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if write_header:
                        w.writeheader()
                    w.writerow(row)
                print(f"    acc={metrics['eval_accuracy']:.4f}", flush=True)

    # Aggregate
    if not out_full.exists():
        return
    all_rows = list(csv.DictReader(out_full.open()))
    if not all_rows:
        return

    from collections import defaultdict  # noqa: PLC0415

    grouped: dict = defaultdict(list)
    for r in all_rows:
        grouped[(r["noise_regime"], r["schedule"])].append(float(r["eval_accuracy"]))

    noise_names_all = sorted(set(r["noise_regime"] for r in all_rows))
    schedules_all = sorted(set(r["schedule"] for r in all_rows))

    agg_rows = []
    for nr in noise_names_all:
        kl_vals = grouped[(nr, "kl")]
        for sched in schedules_all:
            vals = grouped[(nr, sched)]
            if not vals:
                continue
            agg_rows.append({
                "noise_regime": nr,
                "schedule": sched,
                "n_seeds": len(vals),
                "mean_accuracy": np.mean(vals),
                "std_accuracy": np.std(vals),
                "mean_diff_vs_kl": np.mean(vals) - np.mean(kl_vals) if kl_vals else float("nan"),
            })

    Path(args.out_aggregated).parent.mkdir(parents=True, exist_ok=True)
    if agg_rows:
        with open(args.out_aggregated, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader()
            w.writerows(agg_rows)

    print(f"\n[dynamic-loss] Done. Results in {args.out_full}")


if __name__ == "__main__":
    main()
