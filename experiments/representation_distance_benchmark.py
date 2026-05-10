"""FR Representation Distance benchmark.

Trains multiple small MLP classifiers under varying conditions and compares
FR-RD against linear CKA and L2 weight distance as model comparison metrics.

Conditions:
  - clean: standard cross-entropy on clean labels
  - noisy_30: 30% uniform random label noise
  - noisy_60: 60% uniform random label noise
  - fr_loss: Fisher-Rao loss (clean labels)
  - smoothed: label smoothing 0.1 (clean labels)

For each condition × seed pair, trains a two-layer MLP on sklearn digits and
MNIST, saves softmax outputs and penultimate features on a held-out test set,
then computes all pairwise FR-RD, CKA, and weight-L2 matrices.

Also measures:
  - FR-RD tracking along a training trajectory (every 10 steps vs final model)
  - FR-RD-based OOD detection (digits models evaluated on MNIST inputs)

Outputs:
  reports/results/fr_rd_pairwise.csv       -- pairwise distance matrix rows
  reports/results/fr_rd_trajectory.csv     -- per-step FR-RD to final model
  reports/results/fr_rd_ood.csv            -- OOD scores
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

from fisher_rao_ml.device import get_device
from fisher_rao_ml.distribution_losses import distribution_loss_from_logits
from fisher_rao_ml.representation_distance import (
    cka_linear,
    fr_ood_score,
    fr_ood_score_class_conditional,
    fr_representation_distance,
)

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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.net(x)
        logits = self.head(features)
        return logits, features


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_digits_split(
    test_size: float = 0.3,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    data = load_digits()
    x, y = data.data.astype(np.float32), data.target
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(x_tr)
    x_tr = torch.from_numpy(scaler.transform(x_tr))
    x_te = torch.from_numpy(scaler.transform(x_te))
    y_tr = torch.from_numpy(y_tr)
    y_te = torch.from_numpy(y_te)
    return x_tr, y_tr, x_te, y_te, int(y.max()) + 1


def load_mnist_subset(
    n: int = 500,
    seed: int = 0,
    scaler: object = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a small MNIST subset, standardized with a provided scaler."""
    try:
        import torchvision.transforms as T
        from torchvision.datasets import MNIST
        mnist = MNIST(root="data", train=False, download=True,
                      transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))]))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(mnist), size=min(n, len(mnist)), replace=False)
        x = torch.stack([mnist[int(i)][0] for i in idx]).numpy().astype(np.float32)
        y = torch.tensor([mnist[int(i)][1] for i in idx])
        if scaler is not None:
            x = scaler.transform(x)
        return torch.from_numpy(x), y
    except Exception:
        return None, None


def load_mnist_split(
    n_train: int = 3000,
    n_test: int = 500,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load MNIST train/test split, standardized, flattened to 784 dims."""
    try:
        import torchvision.transforms as T
        from torchvision.datasets import MNIST
        transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))])
        train_ds = MNIST(root="data", train=True, download=True, transform=transform)
        test_ds = MNIST(root="data", train=False, download=True, transform=transform)
        rng = np.random.default_rng(seed)

        def _sample(ds, n: int) -> tuple[np.ndarray, np.ndarray]:
            labels = np.array([ds[i][1] for i in range(len(ds))])
            classes = np.unique(labels)
            n_per = max(1, n // len(classes))
            idxs = []
            for c in classes:
                c_idxs = np.where(labels == c)[0]
                chosen = rng.choice(c_idxs, size=min(n_per, len(c_idxs)), replace=False)
                idxs.extend(chosen.tolist())
            idxs = np.array(idxs[:n])
            rng.shuffle(idxs)
            x = torch.stack([ds[int(i)][0] for i in idxs]).numpy().astype(np.float32)
            return x, labels[idxs].astype(np.int64)

        x_tr, y_tr = _sample(train_ds, n_train)
        x_te, y_te = _sample(test_ds, n_test)
        scaler = StandardScaler().fit(x_tr)
        x_tr = torch.from_numpy(scaler.transform(x_tr))
        x_te = torch.from_numpy(scaler.transform(x_te))
        return x_tr, torch.from_numpy(y_tr), x_te, torch.from_numpy(y_te), 10
    except Exception as e:
        raise RuntimeError(f"MNIST unavailable: {e}") from e


def inject_noise(
    labels: torch.Tensor, noise_rate: float, n_classes: int, seed: int
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    noisy = labels.clone()
    n = len(labels)
    n_noisy = int(noise_rate * n)
    idx = rng.choice(n, size=n_noisy, replace=False)
    for i in idx:
        choices = [c for c in range(n_classes) if c != int(noisy[i])]
        noisy[i] = int(rng.choice(choices))
    return noisy


def make_one_hot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    oh = torch.zeros(len(labels), n_classes)
    oh.scatter_(1, labels.unsqueeze(1), 1.0)
    return oh


def make_smoothed(labels: torch.Tensor, n_classes: int, smoothing: float = 0.1) -> torch.Tensor:
    oh = make_one_hot(labels, n_classes)
    return (1.0 - smoothing) * oh + smoothing / n_classes


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

CONDITION_NOISE = {
    "clean": (0.0, "kl", 0.0),
    "noisy_30": (0.3, "kl", 0.0),
    "noisy_60": (0.6, "kl", 0.0),
    "fr_loss": (0.0, "fisher_rao", 0.0),
    "smoothed": (0.0, "kl", 0.1),
}


def train_model(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    n_classes: int,
    condition: str,
    seed: int,
    device: torch.device,
    n_steps: int = 300,
    hidden: int = 128,
    lr: float = 1e-3,
    trajectory_steps: list[int] | None = None,
) -> tuple[MLP, list[tuple[int, torch.Tensor]]]:
    """Train an MLP. Returns (final_model, trajectory_probs_list).

    trajectory_probs_list contains (step, probs_on_train) for each trajectory step.
    """
    torch.manual_seed(seed)
    noise_rate, objective, smoothing = CONDITION_NOISE[condition]

    noisy_labels = inject_noise(y_tr, noise_rate, n_classes, seed) if noise_rate > 0 else y_tr
    if smoothing > 0:
        targets = make_smoothed(noisy_labels, n_classes, smoothing).to(device)
    else:
        targets = make_one_hot(noisy_labels, n_classes).to(device)

    x = x_tr.to(device)
    model = MLP(x.shape[1], hidden, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    trajectory: list[tuple[int, torch.Tensor]] = []
    traj_steps_set = set(trajectory_steps or [])

    model.train()
    for step in range(1, n_steps + 1):
        opt.zero_grad()
        logits, _ = model(x)
        loss = distribution_loss_from_logits(targets, logits, objective=objective)
        loss.backward()
        opt.step()

        if step in traj_steps_set:
            with torch.no_grad():
                model.eval()
                logits_tr, _ = model(x)
                probs = torch.softmax(logits_tr, dim=-1).cpu()
                trajectory.append((step, probs))
                model.train()

    return model, trajectory


@torch.no_grad()
def get_probs_and_features(
    model: MLP,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (probs, features, logits), all on CPU."""
    model.eval()
    logits, features = model(x.to(device))
    probs = torch.softmax(logits, dim=-1).cpu()
    return probs, features.cpu(), logits.cpu()


@torch.no_grad()
def accuracy(model: MLP, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    logits, _ = model(x.to(device))
    return float((logits.argmax(dim=1).cpu() == y).float().mean().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="digits", choices=["digits", "mnist"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-steps", type=int, default=300)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--out-pairwise", default=None,
                   help="Output path (auto-set from dataset if not given)")
    p.add_argument("--out-trajectory", default=None)
    p.add_argument("--out-ood", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"[fr-rd] device={device}, dataset={args.dataset}")

    tag = args.dataset
    out_pairwise = Path(args.out_pairwise or f"reports/results/fr_rd_{tag}_pairwise.csv")
    out_trajectory = Path(args.out_trajectory or f"reports/results/fr_rd_{tag}_trajectory.csv")
    out_ood = Path(args.out_ood or f"reports/results/fr_rd_{tag}_ood.csv")

    if args.dataset == "mnist":
        x_tr, y_tr, x_te, y_te, n_classes = load_mnist_split(seed=0)
    else:
        x_tr, y_tr, x_te, y_te, n_classes = load_digits_split(seed=0)
    conditions = list(CONDITION_NOISE.keys())
    seeds = list(range(args.seeds))

    # --- Part 1: pairwise distances ---
    traj_steps = list(range(10, args.n_steps + 1, 10))
    probs_map: dict[tuple[str, int], torch.Tensor] = {}
    features_map: dict[tuple[str, int], torch.Tensor] = {}
    logits_map: dict[tuple[str, int], torch.Tensor] = {}
    weights_map: dict[tuple[str, int], torch.Tensor] = {}
    acc_map: dict[tuple[str, int], float] = {}
    traj_map: dict[tuple[str, int], list[tuple[int, torch.Tensor]]] = {}
    models_map: dict[tuple[str, int], nn.Module] = {}

    for cond in conditions:
        for seed in seeds:
            key = (cond, seed)
            print(f"[fr-rd] training {cond} seed={seed}")
            model, traj = train_model(
                x_tr, y_tr, n_classes, cond, seed, device,
                n_steps=args.n_steps, hidden=args.hidden,
                trajectory_steps=traj_steps,
            )
            probs, feats, logits = get_probs_and_features(model, x_te, device)
            probs_map[key] = probs
            features_map[key] = feats
            logits_map[key] = logits
            acc_map[key] = accuracy(model, x_te, y_te, device)
            traj_map[key] = traj
            weights_map[key] = torch.cat([p.data.cpu().flatten() for p in model.parameters()])
            models_map[key] = model

    # Pairwise distances
    all_keys = [(c, s) for c in conditions for s in seeds]
    pairwise_rows = []
    for i, ki in enumerate(all_keys):
        for j, kj in enumerate(all_keys):
            if j <= i:
                continue
            fr_d = fr_representation_distance(probs_map[ki], probs_map[kj])
            ck = cka_linear(features_map[ki], features_map[kj])
            w_l2 = float((weights_map[ki] - weights_map[kj]).norm().item())
            pairwise_rows.append({
                "cond_a": ki[0], "seed_a": ki[1],
                "cond_b": kj[0], "seed_b": kj[1],
                "acc_a": acc_map[ki], "acc_b": acc_map[kj],
                "acc_diff": abs(acc_map[ki] - acc_map[kj]),
                "fr_rd": fr_d,
                "cka": ck,
                "weight_l2": w_l2,
            })

    write_rows(out_pairwise, pairwise_rows)
    print(f"[fr-rd] wrote {len(pairwise_rows)} pairwise rows → {out_pairwise}")

    # --- Part 2: trajectory (training dynamics) ---
    traj_rows = []
    for cond in conditions:
        for seed in seeds:
            key = (cond, seed)
            final_train_probs = None
            for step, probs in reversed(traj_map[key]):
                if step == traj_steps[-1]:
                    final_train_probs = probs
                    break
            if final_train_probs is None:
                continue
            for step, step_probs in traj_map[key]:
                fr_d = fr_representation_distance(step_probs, final_train_probs)
                traj_rows.append({
                    "condition": cond,
                    "seed": seed,
                    "step": step,
                    "fr_rd_to_final": fr_d,
                    "final_test_acc": acc_map[key],
                })

    write_rows(out_trajectory, traj_rows)
    print(f"[fr-rd] wrote {len(traj_rows)} trajectory rows → {out_trajectory}")

    # --- Part 3: OOD detection (digits only) ---
    # OOD data: MNIST images resized to 8×8 → 64 features (same space as Digits).
    # In-distribution centroid = mean softmax over all ID test samples.
    # We use the already-computed probs_map (no retraining needed).
    ood_rows = []
    if args.dataset == "digits" and probs_map:
        try:
            import torchvision.transforms as T
            from sklearn.preprocessing import StandardScaler as SS
            from torchvision.datasets import MNIST

            data_digits = load_digits()
            scaler_digits = SS().fit(data_digits.data.astype(np.float32))

            mnist_ds = MNIST(
                root="data", train=False, download=True,
                transform=T.Compose([T.Resize(8), T.ToTensor(),
                                     T.Lambda(lambda x: x.view(-1))]),
            )
            rng_ood = np.random.default_rng(0)
            idx_ood = rng_ood.choice(len(mnist_ds), size=300, replace=False)
            x_mnist_raw = torch.stack(
                [mnist_ds[int(i)][0] for i in idx_ood]
            ).numpy().astype(np.float32)
            x_mnist_ood = torch.from_numpy(scaler_digits.transform(x_mnist_raw))

            # y_te is already a Tensor from load_digits_split / load_mnist_split
            labels_id = y_te if isinstance(y_te, torch.Tensor) else torch.from_numpy(y_te)
            for cond in conditions:
                # Global centroid: mean over all in-distribution probs (all seeds)
                all_probs_id = torch.cat([probs_map[(cond, s)] for s in seeds], dim=0)
                all_labels_id = labels_id.repeat(len(seeds))
                # Mahalanobis: class-conditional mean features + shared covariance
                all_feats_id = torch.cat([features_map[(cond, s)] for s in seeds], dim=0)
                n_feat = all_feats_id.shape[1]
                feat_means = []
                for c in range(n_classes):
                    mask = all_labels_id == c
                    feat_means.append(
                        all_feats_id[mask].mean(dim=0) if mask.sum() > 0
                        else all_feats_id.mean(dim=0)
                    )
                feat_means_t = torch.stack(feat_means, dim=0)
                # shared covariance (pooled, regularized)
                centered = []
                for c in range(n_classes):
                    mask = all_labels_id == c
                    if mask.sum() > 1:
                        centered.append(all_feats_id[mask] - feat_means_t[c])
                cov_pool = torch.cat(centered, dim=0)
                cov_mat = (cov_pool.T @ cov_pool) / len(cov_pool)
                cov_mat += 1e-4 * torch.eye(n_feat)
                cov_inv = torch.linalg.inv(cov_mat)

                for seed in seeds:
                    model_key = (cond, seed)
                    probs_ood, feats_ood, logits_ood = get_probs_and_features(
                        models_map[model_key], x_mnist_ood, device
                    )
                    # Global centroid OOD scores
                    ood_scores_ood = fr_ood_score(all_probs_id, probs_ood).numpy()
                    ood_scores_id = fr_ood_score(all_probs_id, probs_map[model_key]).numpy()
                    # Class-conditional centroid OOD scores
                    cc_ood = fr_ood_score_class_conditional(
                        all_probs_id, all_labels_id, probs_ood
                    ).numpy()
                    cc_id = fr_ood_score_class_conditional(
                        all_probs_id, all_labels_id, probs_map[model_key]
                    ).numpy()
                    # MSP: 1 - max(softmax); higher = more OOD
                    msp_ood = (1.0 - probs_ood.max(dim=-1).values).numpy()
                    msp_id = (1.0 - probs_map[model_key].max(dim=-1).values).numpy()
                    # Mahalanobis: min_c (f - μ_c)^T Σ^{-1} (f - μ_c)
                    def _mahal(
                        feats: torch.Tensor,
                        means: torch.Tensor,
                        inv: torch.Tensor,
                        nc: int,
                    ) -> np.ndarray:
                        scores = torch.zeros(len(feats))
                        for c in range(nc):
                            diff = feats - means[c]
                            d_c = (diff @ inv * diff).sum(dim=-1)
                            scores = d_c if c == 0 else torch.minimum(scores, d_c)
                        return scores.numpy()

                    mahal_ood = _mahal(feats_ood.cpu(), feat_means_t, cov_inv, n_classes)
                    mahal_id = _mahal(
                        features_map[model_key].cpu(), feat_means_t, cov_inv, n_classes
                    )
                    # Energy score: -logsumexp(logits); higher = more OOD
                    energy_ood = -torch.logsumexp(logits_ood, dim=-1).numpy()
                    energy_id = -torch.logsumexp(logits_map[model_key], dim=-1).numpy()
                    ood_rows.append({
                        "condition": cond,
                        "seed": seed,
                        "mean_ood_score_ood": float(ood_scores_ood.mean()),
                        "mean_ood_score_id": float(ood_scores_id.mean()),
                        "separation": float(ood_scores_ood.mean() - ood_scores_id.mean()),
                        "cc_mean_ood_score_ood": float(cc_ood.mean()),
                        "cc_mean_ood_score_id": float(cc_id.mean()),
                        "cc_separation": float(cc_ood.mean() - cc_id.mean()),
                        "msp_mean_ood_score_ood": float(msp_ood.mean()),
                        "msp_mean_ood_score_id": float(msp_id.mean()),
                        "msp_separation": float(msp_ood.mean() - msp_id.mean()),
                        "mahal_mean_ood_score_ood": float(mahal_ood.mean()),
                        "mahal_mean_ood_score_id": float(mahal_id.mean()),
                        "mahal_separation": float(mahal_ood.mean() - mahal_id.mean()),
                        "energy_mean_ood_score_ood": float(energy_ood.mean()),
                        "energy_mean_ood_score_id": float(energy_id.mean()),
                        "energy_separation": float(energy_ood.mean() - energy_id.mean()),
                    })
            print("[fr-rd] OOD separation summary (global / cc / MSP / Mahal / Energy):")
            for cond in conditions:
                cond_rows = [r for r in ood_rows if r["condition"] == cond]
                sep = np.mean([r["separation"] for r in cond_rows])
                cc_sep = np.mean([r["cc_separation"] for r in cond_rows])
                msp_sep = np.mean([r["msp_separation"] for r in cond_rows])
                mahal_sep = np.mean([r["mahal_separation"] for r in cond_rows])
                energy_sep = np.mean([r["energy_separation"] for r in cond_rows])
                print(
                    f"  {cond:12s}: global={sep:+.4f}  cc={cc_sep:+.4f}"
                    f"  msp={msp_sep:+.4f}  mahal={mahal_sep:+.4f}"
                    f"  energy={energy_sep:+.4f}"
                )
        except Exception as e:
            print(f"[fr-rd] OOD section skipped: {e}")

    if ood_rows:
        write_rows(out_ood, ood_rows)
        print(f"[fr-rd] wrote {len(ood_rows)} OOD rows → {out_ood}")

    # Print summary
    print("\n[fr-rd] Pairwise FR-RD summary by condition pair:")
    from collections import defaultdict
    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in pairwise_rows:
        pair = tuple(sorted([r["cond_a"], r["cond_b"]]))
        by_pair[pair].append(r["fr_rd"])
    for pair, vals in sorted(by_pair.items()):
        print(f"  {pair[0]:12s} vs {pair[1]:12s}: mean FR-RD = {np.mean(vals):.4f}")


from torch import Tensor  # noqa: E402 (needed for type annotation in MLP)

if __name__ == "__main__":
    main()
