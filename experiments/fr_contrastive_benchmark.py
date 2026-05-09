"""FR-Contrastive vs NT-Xent benchmark with false-negative injection.

Uses UCI Digits (10 classes) as a fast proxy for self-supervised metric learning.
Each anchor has exactly one designated positive (a different sample from the same class).
False negatives are additional same-class samples injected into the negative pool
at rate fn_rate: with probability fn_rate, each negative slot is filled with a
same-class sample instead of a different-class sample.

Evaluates: KNN accuracy (k=5) on test set after contrastive training.

Outputs:
  reports/results/fr_contrastive_fn.csv     -- per-run results
  reports/results/fr_contrastive_agg.csv    -- aggregated means/stds
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fisher_rao_ml.device import get_device

OBJECTIVES = ("nt_xent", "fr_contrastive")
FALSE_NEG_RATES = (0.0, 0.1, 0.2, 0.3)
TEMPERATURE = 0.5


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def nt_xent_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    tau: float = TEMPERATURE,
) -> torch.Tensor:
    """NT-Xent with explicit (anchor, positive, negatives) triples.

    anchors:   (B, D)
    positives: (B, D)  — one positive per anchor
    negatives: (B, K, D) — K negatives per anchor (some may be false negatives)
    """
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Positive similarity: (B,)
    pos_sim = (anchors * positives).sum(-1) / tau

    # Negative similarity: (B, K)
    neg_sim = torch.einsum("bd,bkd->bk", anchors, negatives) / tau

    # log softmax: log(exp(pos) / (exp(pos) + sum_k exp(neg_k)))
    all_logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+K)
    log_prob = F.log_softmax(all_logits, dim=-1)[:, 0]  # log p(positive)
    return -log_prob.mean()


def fr_contrastive_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    tau: float = TEMPERATURE,
    eps: float = 1e-6,
) -> torch.Tensor:
    """FR-Contrastive with explicit (anchor, positive, negatives) triples.

    Replaces KL(e_i || p_i) with d_FR(e_i, p_i)^2 = 4*arccos^2(sqrt(p_ij(i))).
    """
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    pos_sim = (anchors * positives).sum(-1) / tau
    neg_sim = torch.einsum("bd,bkd->bk", anchors, negatives) / tau

    all_logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    p = F.softmax(all_logits, dim=-1)
    p_pos = p[:, 0].clamp(eps, 1.0 - eps)
    fr2 = 4.0 * torch.arccos(torch.sqrt(p_pos)).pow(2)
    return fr2.mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Batch builder with false-negative injection
# ---------------------------------------------------------------------------

def build_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    n_neg: int,
    fn_rate: float,
    n_classes: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (anchor, positive, negatives) triples with false-negative injection.

    For each anchor:
      - One positive: a different sample of the same class.
      - n_neg negatives: mix of true negatives (different class) and false negatives
        (same class, injected at rate fn_rate per slot).
    """
    n = len(x)
    y_np = y.numpy()

    # Index samples by class
    class_idx: list[list[int]] = [[] for _ in range(n_classes)]
    for i, c in enumerate(y_np):
        class_idx[c].append(i)

    anchors_list, pos_list, neg_list = [], [], []

    # Pick anchor indices
    anchor_idx = rng.choice(n, size=batch_size, replace=False)

    for ai in anchor_idx:
        c = int(y_np[ai])
        same = [j for j in class_idx[c] if j != ai]
        diff = [j for j in range(n) if y_np[j] != c]

        # Positive: random same-class sample
        pos_i = int(rng.choice(same))

        # Negatives: n_neg slots; each slot is a false negative with prob fn_rate
        neg_indices = []
        for _ in range(n_neg):
            if rng.random() < fn_rate and len(same) > 1:
                # False negative: same-class sample (excluding anchor and designated positive)
                fn_cands = [j for j in same if j != pos_i]
                neg_indices.append(int(rng.choice(fn_cands)))
            else:
                neg_indices.append(int(rng.choice(diff)))

        anchors_list.append(x[ai])
        pos_list.append(x[pos_i])
        neg_list.append(torch.stack([x[j] for j in neg_indices]))

    return (
        torch.stack(anchors_list),
        torch.stack(pos_list),
        torch.stack(neg_list),
    )


# ---------------------------------------------------------------------------
# KNN evaluation
# ---------------------------------------------------------------------------

def knn_accuracy(
    train_feats: torch.Tensor, train_labels: torch.Tensor,
    test_feats: torch.Tensor, test_labels: torch.Tensor,
    k: int = 5,
) -> float:
    train_feats = F.normalize(train_feats, dim=-1)
    test_feats = F.normalize(test_feats, dim=-1)
    sim = test_feats @ train_feats.T
    topk = sim.topk(k, dim=-1).indices
    pred = train_labels[topk].mode(dim=-1).values
    return float((pred == test_labels).float().mean().item())


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------

def train_and_eval(
    x_tr: torch.Tensor, y_tr: torch.Tensor,
    x_te: torch.Tensor, y_te: torch.Tensor,
    objective: str,
    fn_rate: float,
    n_classes: int,
    seed: int,
    device: torch.device,
    n_epochs: int = 200,
    batch_size: int = 128,
    n_neg: int = 31,
    lr: float = 1e-3,
    embed_dim: int = 64,
) -> dict:
    torch.manual_seed(seed)
    model = Encoder(x_tr.shape[1], embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    rng = np.random.default_rng(seed * 31337 + int(fn_rate * 1000))

    n_batches_per_epoch = max(1, len(x_tr) // batch_size)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches_per_epoch):
            anc, pos, neg = build_batch(
                x_tr, y_tr, batch_size, n_neg, fn_rate, n_classes, rng
            )
            anc = anc.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            z_anc = model(anc)
            z_pos = model(pos)
            B, K, D_raw = neg.shape
            z_neg = model(neg.view(B * K, -1)).view(B, K, -1)

            if objective == "nt_xent":
                loss = nt_xent_loss(z_anc, z_pos, z_neg)
            else:
                loss = fr_contrastive_loss(z_anc, z_pos, z_neg)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        tr_feats = model(x_tr.to(device)).cpu()
        te_feats = model(x_te.to(device)).cpu()

    knn_acc = knn_accuracy(tr_feats, y_tr, te_feats, y_te, k=5)
    return {"knn_accuracy": knn_acc}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_existing(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r["objective"], r["fn_rate"], r["seed"]) for r in csv.DictReader(f)}


def write_rows(path: Path, rows: list[dict], append: bool = True) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (append and path.exists()) else "w"
    with path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def overwrite_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        k = (r["objective"], r["fn_rate"])
        grouped[k].append(float(r["knn_accuracy"]))
    out = []
    for (obj, fn_rate), vals in sorted(grouped.items()):
        arr = np.array(vals)
        out.append({
            "objective": obj,
            "fn_rate": fn_rate,
            "knn_accuracy_mean": float(arr.mean()),
            "knn_accuracy_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n_seeds": len(arr),
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--out-full", default="reports/results/fr_contrastive_fn.csv")
    p.add_argument("--out-agg", default="reports/results/fr_contrastive_agg.csv")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"[fr-contrastive] device={device}, epochs={args.n_epochs}, seeds={args.seeds}")

    data = load_digits()
    x_raw = data.data.astype(np.float32)
    y_raw = data.target.astype(np.int64)
    n_classes = 10

    x_tr_raw, x_te_raw, y_tr_raw, y_te_raw = train_test_split(
        x_raw, y_raw, test_size=0.2, random_state=0, stratify=y_raw
    )
    scaler = StandardScaler().fit(x_tr_raw)
    x_tr = torch.from_numpy(scaler.transform(x_tr_raw))
    x_te = torch.from_numpy(scaler.transform(x_te_raw))
    y_tr = torch.from_numpy(y_tr_raw)
    y_te = torch.from_numpy(y_te_raw)

    out_full = Path(args.out_full)
    done = set() if args.force else read_existing(out_full)
    new_rows: list[dict] = []

    for fn_rate in FALSE_NEG_RATES:
        for objective in OBJECTIVES:
            for seed in range(args.seeds):
                key = (objective, str(fn_rate), str(seed))
                if key in done:
                    continue
                print(f"[fr-contrastive] {objective} fn_rate={fn_rate} seed={seed}")
                metrics = train_and_eval(
                    x_tr, y_tr, x_te, y_te,
                    objective=objective, fn_rate=fn_rate,
                    n_classes=n_classes, seed=seed, device=device,
                    n_epochs=args.n_epochs,
                )
                row = {"objective": objective, "fn_rate": fn_rate, "seed": seed, **metrics}
                new_rows.append(row)
                done.add(key)

    write_rows(out_full, new_rows, append=not args.force)
    print(f"[fr-contrastive] wrote {len(new_rows)} new rows → {out_full}")

    all_rows = list(csv.DictReader(out_full.open())) if out_full.exists() else new_rows
    agg = aggregate(all_rows)
    overwrite_rows(Path(args.out_agg), agg)
    print(f"[fr-contrastive] aggregated → {args.out_agg}")

    print("\n[fr-contrastive] KNN accuracy summary:")
    print(f"  {'objective':15} {'fn_rate':8} {'mean_knn':10} {'std':8}")
    for r in agg:
        print(f"  {r['objective']:15} {float(r['fn_rate']):8.2f} "
              f"  {float(r['knn_accuracy_mean']):.4f}   {float(r['knn_accuracy_std']):.4f}")


if __name__ == "__main__":
    main()
