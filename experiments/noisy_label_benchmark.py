"""Noisy-label benchmark comparing Fisher-Rao against standard robust losses.

Compares 6 objectives on digits and MNIST with synthetic label noise:
  kl (CE), gce, mae, sce, hellinger, fisher_rao

Noise regimes:
  - clean (0% noise)
  - sym_20, sym_40, sym_60, sym_80: symmetric uniform noise at 20/40/60/80%
  - asym_40: asymmetric (class-conditional) noise at 40%

Outputs:
  reports/results/noisy_label_full.csv
  reports/results/noisy_label_aggregated.csv
  reports/results/noisy_label_significance.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fisher_rao_ml.device import get_device
from fisher_rao_ml.distribution_losses import distribution_loss_from_logits

OBJECTIVES = ("kl", "gce", "mae", "sce", "hellinger", "fisher_rao")

NOISE_REGIMES = {
    "clean": ("sym", 0.0),
    "sym_20": ("sym", 0.20),
    "sym_40": ("sym", 0.40),
    "sym_60": ("sym", 0.60),
    "sym_80": ("sym", 0.80),
    "asym_40": ("asym", 0.40),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, int]:
    if name == "digits":
        data = load_digits()
        return data.data.astype(np.float32), data.target.astype(np.int64), 10
    if name == "mnist":
        try:
            from torchvision.datasets import MNIST
            import torchvision.transforms as T
            ds = MNIST("data", train=True, download=True,
                       transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))]))
            x = torch.stack([ds[i][0] for i in range(min(3000, len(ds)))]).numpy()
            y = np.array([ds[i][1] for i in range(min(3000, len(ds)))], dtype=np.int64)
            return x.astype(np.float32), y, 10
        except Exception as e:
            print(f"[noisy-label] MNIST unavailable: {e}")
            return None, None, -1
    raise ValueError(f"Unknown dataset: {name}")


def inject_symmetric_noise(y: np.ndarray, rate: float, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    noisy = y.copy()
    n = len(y)
    n_noisy = int(rate * n)
    idx = rng.choice(n, size=n_noisy, replace=False)
    for i in idx:
        choices = [c for c in range(n_classes) if c != int(noisy[i])]
        noisy[i] = int(rng.choice(choices))
    return noisy


def inject_asymmetric_noise(y: np.ndarray, rate: float, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    """Each class c flips to (c+1) % n_classes with probability rate."""
    noisy = y.copy()
    flip = rng.random(len(y)) < rate
    noisy[flip] = (y[flip] + 1) % n_classes
    return noisy


def make_one_hot(y: np.ndarray, n_classes: int) -> torch.Tensor:
    oh = torch.zeros(len(y), n_classes)
    oh.scatter_(1, torch.from_numpy(y).long().unsqueeze(1), 1.0)
    return oh


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_eval(
    x_tr: torch.Tensor,
    y_tr_noisy: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    n_classes: int,
    objective: str,
    seed: int,
    device: torch.device,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> dict:
    torch.manual_seed(seed)
    model = MLP(x_tr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    n = len(x_tr)
    n_batches = math.ceil(n / batch_size)
    idx_all = np.arange(n)

    model.train()
    rng = np.random.default_rng(seed + 10000)
    for epoch in range(n_epochs):
        rng.shuffle(idx_all)
        for b in range(n_batches):
            batch_idx = idx_all[b * batch_size: (b + 1) * batch_size]
            xb = x_tr[batch_idx].to(device)
            yb = y_tr_noisy[batch_idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = distribution_loss_from_logits(yb, logits, objective=objective)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        logits_te = model(x_te.to(device))
        probs_te = torch.softmax(logits_te, dim=-1).cpu()
        y_pred = probs_te.argmax(dim=1).numpy()
        y_true = y_te.numpy()
        acc = float((y_pred == y_true).mean())

        # ECE with 10 bins
        confidences = probs_te.max(dim=1).values.numpy()
        ece = _compute_ece(confidences, y_pred == y_true, n_bins=10)

        # Brier
        oh_te = make_one_hot(y_true.astype(np.int64), n_classes).numpy()
        brier = float(np.mean(np.sum((probs_te.numpy() - oh_te) ** 2, axis=1)))

        # NLL
        nll = float(-np.mean(np.log(np.clip(probs_te.numpy()[np.arange(len(y_true)), y_true], 1e-8, 1.0))))

    return {
        "eval_accuracy": acc,
        "eval_ece": ece,
        "eval_brier": brier,
        "eval_nll": nll,
    }


def _compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_rows(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    grouped: dict[tuple, dict[str, list]] = defaultdict(lambda: {})
    metrics = ["eval_accuracy", "eval_ece", "eval_brier", "eval_nll"]
    for r in rows:
        k = (r["dataset"], r["noise_regime"], r["objective"])
        for m in metrics:
            grouped[k].setdefault(m, []).append(float(r[m]))
    out = []
    for (dataset, noise_regime, objective), mv in sorted(grouped.items()):
        record: dict = {"dataset": dataset, "noise_regime": noise_regime, "objective": objective}
        for m, vals in mv.items():
            arr = np.array(vals)
            record[f"{m}_mean"] = float(arr.mean())
            record[f"{m}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            record[f"{m}_n"] = len(arr)
        out.append(record)
    return out


def significance_rows(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    from scipy.stats import wilcoxon
    metrics = ["eval_accuracy", "eval_ece", "eval_brier", "eval_nll"]
    paired: dict[tuple, dict[str, dict[str, float]]] = defaultdict(lambda: {})
    for r in rows:
        k = (r["dataset"], r["noise_regime"], int(r["seed"]))
        paired[k][r["objective"]] = {m: float(r[m]) for m in metrics}
    grouped: dict[tuple, dict[str, list]] = defaultdict(lambda: {})
    for (dataset, noise_regime, _seed), bundle in paired.items():
        if "kl" not in bundle:
            continue
        for obj in OBJECTIVES:
            if obj == "kl" or obj not in bundle:
                continue
            k2 = (dataset, noise_regime, obj)
            for m in metrics:
                kl_v = bundle["kl"].get(m)
                obj_v = bundle[obj].get(m)
                if kl_v is None or obj_v is None:
                    continue
                grouped[k2].setdefault(m, []).append((kl_v, obj_v))
    out = []
    for (dataset, noise_regime, objective), pm in sorted(grouped.items()):
        record: dict = {"dataset": dataset, "noise_regime": noise_regime, "objective": objective}
        for m, pairs in pm.items():
            diffs = [obj - kl for kl, obj in pairs]
            # acc/brier/ece: oriented so positive means improvement
            sign = -1.0 if m in ("eval_ece", "eval_brier", "eval_nll") else 1.0
            oriented = [d * sign for d in diffs]
            try:
                if all(abs(d) < 1e-12 for d in diffs):
                    stat, pval = 0.0, 1.0
                else:
                    res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
                    stat, pval = float(res.statistic), float(res.pvalue)
            except Exception:
                stat, pval = float("nan"), float("nan")
            record[f"{m}_mean_diff"] = float(np.mean(diffs))
            record[f"{m}_oriented_gain"] = float(np.mean(oriented))
            record[f"{m}_wilcoxon_p"] = pval
            record[f"{m}_n_improves"] = sum(1 for d in oriented if d > 0)
            record[f"{m}_n_pairs"] = len(pairs)
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_existing(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open() as f:
        reader = csv.DictReader(f)
        return {(r["dataset"], r["noise_regime"], r["objective"], r["seed"]) for r in reader}


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def overwrite_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["digits", "mnist"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--out-full", default="reports/results/noisy_label_full.csv")
    p.add_argument("--out-aggregated", default="reports/results/noisy_label_aggregated.csv")
    p.add_argument("--out-significance", default="reports/results/noisy_label_significance.csv")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"[noisy-label] device={device}, epochs={args.n_epochs}, seeds={args.seeds}")

    out_full = Path(args.out_full)
    done = set() if args.force else read_existing(out_full)
    new_rows: list[dict] = []

    for dataset_name in args.datasets:
        x_raw, y_raw, n_classes = load_dataset(dataset_name)
        if x_raw is None:
            continue

        x_tr_raw, x_te_raw, y_tr_raw, y_te_raw = train_test_split(
            x_raw, y_raw, test_size=0.2, random_state=0, stratify=y_raw
        )
        scaler = StandardScaler().fit(x_tr_raw)
        x_tr = torch.from_numpy(scaler.transform(x_tr_raw))
        x_te = torch.from_numpy(scaler.transform(x_te_raw))
        y_te = torch.from_numpy(y_te_raw)

        for noise_regime, (noise_type, noise_rate) in NOISE_REGIMES.items():
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 1000 + int(noise_rate * 100))
                if noise_rate == 0.0:
                    y_tr_noisy = y_tr_raw.copy()
                elif noise_type == "sym":
                    y_tr_noisy = inject_symmetric_noise(y_tr_raw, noise_rate, n_classes, rng)
                else:
                    y_tr_noisy = inject_asymmetric_noise(y_tr_raw, noise_rate, n_classes, rng)
                y_tr_oh = make_one_hot(y_tr_noisy, n_classes)

                for objective in OBJECTIVES:
                    key = (dataset_name, noise_regime, objective, str(seed))
                    if key in done:
                        continue
                    print(f"[noisy-label] {dataset_name} {noise_regime} {objective} seed={seed}")
                    metrics = train_and_eval(
                        x_tr, y_tr_oh, x_te, y_te, n_classes,
                        objective=objective, seed=seed, device=device,
                        n_epochs=args.n_epochs,
                    )
                    row = {
                        "dataset": dataset_name,
                        "noise_regime": noise_regime,
                        "objective": objective,
                        "seed": seed,
                        **metrics,
                    }
                    new_rows.append(row)
                    done.add(key)

    write_rows(out_full, new_rows)
    print(f"[noisy-label] wrote {len(new_rows)} new rows → {out_full}")

    all_rows = list(csv.DictReader(out_full.open())) if out_full.exists() else new_rows
    agg = aggregate_rows(all_rows)
    overwrite_rows(Path(args.out_aggregated), agg)
    print(f"[noisy-label] aggregated {len(agg)} rows → {args.out_aggregated}")

    sig = significance_rows(all_rows)
    overwrite_rows(Path(args.out_significance), sig)
    print(f"[noisy-label] significance {len(sig)} rows → {args.out_significance}")

    # Quick summary
    print("\n[noisy-label] FR vs KL accuracy gain summary:")
    for r in sig:
        if r["objective"] == "fisher_rao" and r["dataset"] == "digits":
            gain = float(r.get("eval_accuracy_oriented_gain", 0))
            p = float(r.get("eval_accuracy_wilcoxon_p", 1))
            n_imp = r.get("eval_accuracy_n_improves", "?")
            n_tot = r.get("eval_accuracy_n_pairs", "?")
            print(f"  {r['noise_regime']:10s}: gain={gain:+.4f} p={p:.3f} win={n_imp}/{n_tot}")


if __name__ == "__main__":
    main()
