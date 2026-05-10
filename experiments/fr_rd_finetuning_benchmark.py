"""FR-RD fine-tuning / data-fraction divergence benchmark.

Trains MLP classifiers on varying fractions of the UCI Digits training set and
computes FR-RD between each model and a fully-trained reference model.

Hypothesis: FR-RD to the reference increases as training fraction decreases,
and correlates with the accuracy gap relative to the reference.

Fractions: 0.1, 0.2, 0.4, 0.6, 0.8, 1.0
Seeds: 5 per fraction (one reference model trained on full data per seed)

Outputs:
  reports/results/fr_rd_finetuning.csv
    -- fraction, seed, n_train, accuracy, fr_rd_to_ref, acc_gap
  reports/results/fr_rd_finetuning_pairwise.csv
    -- fraction_a, seed_a, fraction_b, seed_b, fr_rd
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from fisher_rao_ml.device import get_device
from fisher_rao_ml.distribution_losses import distribution_loss_from_logits
from fisher_rao_ml.representation_distance import fr_representation_distance

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

FRACTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
N_SEEDS = 5
N_EPOCHS = 200
LR = 1e-3
HIDDEN = 128
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.net(x))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_digits_full(
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, StandardScaler, int]:
    data = load_digits()
    x, y = data.data.astype(np.float32), data.target
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.3, random_state=seed
    )
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_te = scaler.transform(x_te)
    n_classes = int(y.max()) + 1
    return (
        torch.from_numpy(x_tr),
        torch.from_numpy(x_te),
        torch.from_numpy(y_tr).long(),
        torch.from_numpy(y_te).long(),
        scaler,
        n_classes,
    )


def subsample(x: Tensor, y: Tensor, fraction: float, seed: int) -> tuple[Tensor, Tensor]:
    n = len(x)
    k = max(1, int(n * fraction))
    rng = torch.Generator()
    rng.manual_seed(seed + 1000)
    idx = torch.randperm(n, generator=rng)[:k]
    return x[idx], y[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    x_tr: Tensor,
    y_tr: Tensor,
    x_te: Tensor,
    y_te: Tensor,
    n_classes: int,
    seed: int,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    hidden: int = HIDDEN,
    batch_size: int = BATCH_SIZE,
    device: torch.device | None = None,
) -> tuple[MLP, float]:
    if device is None:
        device = get_device()
    torch.manual_seed(seed)
    in_dim = x_tr.shape[1]
    model = MLP(in_dim, hidden, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)

    n = len(x_tr)
    for _ in range(n_epochs):
        model.train()
        idx = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            xb, yb = x_tr[batch], y_tr[batch]
            targets = torch.zeros(len(yb), n_classes, device=device)
            targets.scatter_(1, yb.unsqueeze(1), 1.0)
            logits = model(xb)
            loss = distribution_loss_from_logits(targets, logits, "kl")
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_te.to(device))
        acc = (logits.argmax(dim=1) == y_te.to(device)).float().mean().item()
    return model, acc


@torch.no_grad()
def get_probs(model: MLP, x: Tensor, device: torch.device) -> Tensor:
    model.eval()
    logits = model(x.to(device))
    return torch.softmax(logits, dim=-1).cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/results/fr_rd_finetuning.csv"),
    )
    p.add_argument(
        "--out-pairwise",
        type=Path,
        default=Path("reports/results/fr_rd_finetuning_pairwise.csv"),
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    existing: set[tuple] = set()
    if args.out.exists() and not args.force:
        with open(args.out) as f:
            for row in csv.DictReader(f):
                existing.add((float(row["fraction"]), int(row["seed"])))

    ref_writer_open = not args.out.exists() or args.force
    out_f = open(args.out, "w" if args.force else "a", newline="")
    writer = csv.writer(out_f)
    if ref_writer_open:
        writer.writerow(["fraction", "seed", "n_train", "accuracy", "fr_rd_to_ref", "acc_gap"])

    pair_existing: set[tuple] = set()
    if args.out_pairwise.exists() and not args.force:
        with open(args.out_pairwise) as f:
            for row in csv.DictReader(f):
                pair_existing.add(
                    (float(row["fraction_a"]), int(row["seed_a"]),
                     float(row["fraction_b"]), int(row["seed_b"]))
                )

    pair_f = open(args.out_pairwise, "w" if args.force else "a", newline="")
    pair_writer = csv.writer(pair_f)
    if not args.out_pairwise.exists() or args.force:
        pair_writer.writerow(["fraction_a", "seed_a", "fraction_b", "seed_b", "fr_rd"])

    for seed in range(args.seeds):
        x_tr_full, x_te, y_tr_full, y_te, _scaler, n_classes = load_digits_full(seed=seed)

        models: dict[float, MLP] = {}
        accs: dict[float, float] = {}

        for frac in FRACTIONS:
            if (frac, seed) in existing:
                print(f"  skip fraction={frac:.1f} seed={seed}")
                continue
            x_tr_sub, y_tr_sub = subsample(x_tr_full, y_tr_full, frac, seed)
            model, acc = train_model(
                x_tr_sub, y_tr_sub, x_te, y_te, n_classes,
                seed=seed, n_epochs=args.n_epochs, device=device,
            )
            models[frac] = model
            accs[frac] = acc
            print(f"  seed={seed} frac={frac:.1f} n={len(x_tr_sub)} acc={acc:.4f}")

        if not models:
            continue

        ref_model = models[1.0]
        ref_acc = accs[1.0]
        ref_probs = get_probs(ref_model, x_te, device)

        for frac in FRACTIONS:
            if frac not in models:
                continue
            probs = get_probs(models[frac], x_te, device)
            frd = fr_representation_distance(probs, ref_probs)
            acc_gap = ref_acc - accs[frac]
            n_tr = len(x_tr_full[: int(len(x_tr_full) * frac)])
            writer.writerow([frac, seed, n_tr, accs[frac], frd, acc_gap])
            out_f.flush()
            print(f"    FR-RD to ref: {frd:.4f}  acc_gap: {acc_gap:.4f}")

        # pairwise FR-RD
        frac_list = sorted(models.keys())
        all_probs = {frac: get_probs(models[frac], x_te, device) for frac in frac_list}
        for i, fa in enumerate(frac_list):
            for fb in frac_list[i:]:
                key = (fa, seed, fb, seed)
                if key in pair_existing:
                    continue
                frd = fr_representation_distance(all_probs[fa], all_probs[fb])
                pair_writer.writerow([fa, seed, fb, seed, frd])
                pair_f.flush()

    out_f.close()
    pair_f.close()
    print(f"[fr-rd-finetuning] done → {args.out}")


if __name__ == "__main__":
    main()
