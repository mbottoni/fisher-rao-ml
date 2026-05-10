# CLAUDE.md — fisher-rao-ml

Research project studying Fisher-Rao geodesic distance as a replacement for KL divergence
in ML objectives. The goal is publications at top-tier venues (NeurIPS, ICML, ICLR).

---

## The Unified Finding

**The operative property is not Fisher-Rao geometry specifically — it is bounded codomain +
symmetry.** FR helps when *corruption concentrates probability mass on wrong targets*
(overconfident false neighbors, wrong soft labels). It hurts or is neutral when *corruption
is in which samples are paired* (symmetric label noise, graph corruption), because that
requires a different structural property: the constant-sum condition (Ghosh 2017), which MAE
satisfies and FR does not.

| Setting | FR vs KL | Mechanism |
|---|---|---|
| Affinity-mass corruption (t-SNE) | +++ | Bounded gradient Θ(u^{-1/2}) on false edges |
| Soft-label / distillation overconfidence | ++ | Same — bounded pressure on wrong distributions |
| **Symmetric label noise** | **−− hurts** | Violates Ghosh noise-tolerance condition |
| kNN-graph corruption | neutral | Different corruption type; condition doesn't apply |
| VAE regularization | neutral | No clear advantage after beta tuning |
| Clean training | neutral | KL and FR equivalent |

---

## Repository Layout

```
src/fisher_rao_ml/
  losses.py                   categorical_fisher_rao_distance/squared
                              diagonal_gaussian_fisher_rao_distance/squared
  distribution_losses.py      distribution_loss() — 10 objectives:
                              kl, kl_smoothed, kl_capped, jensen_shannon,
                              hellinger, fisher_rao, fr_kl_hybrid, gce, mae, sce
  representation_distance.py  fr_representation_distance(), pairwise_fr_rd(),
                              cka_linear(), fr_ood_score()
  tsne.py                     pairwise_student_t_affinities,
                              symmetric_gaussian_affinities,
                              perplexity_gaussian_affinities, tsne_distribution_loss
  evaluation.py               trustworthiness, neighborhood_recall, silhouette,
                              knn_accuracy, corrupted_edge_preservation,
                              corrupted_edge_q_mass
  vae.py                      VAE model (diagonal Gaussian encoder/decoder)
  device.py                   get_device() — MPS on Apple Silicon, else CPU

experiments/
  paper_benchmark.py          Main t-SNE benchmark (3 datasets, feature-noise levels)
  dimred_stress_benchmark.py  Noisy-affinity + kNN-graph corruption stress tests
  soft_label_benchmark.py     Soft-label classification under noisy targets
  distillation_benchmark.py   Corrupted teacher-student distillation
  vae_benchmark.py            VAE with KL vs Fisher-Rao regularizer
  noisy_label_benchmark.py    Direction 1: 6-objective noisy-label benchmark
                              (kl, gce, mae, sce, hellinger, fisher_rao;
                               6 noise regimes; 3 datasets: digits, mnist, fashion_mnist;
                               resumable; outputs to reports/results/)
  cifar10_noisy_label_benchmark.py
                              Direction 1 scale-up: CIFAR-10 ConvNet benchmark
                              (same 6 objectives × 5 noise regimes × 5 seeds;
                               10k train subset, 4-layer ConvNet;
                               outputs to reports/results/cifar10_noisy_label_*.csv)
  representation_distance_benchmark.py
                              Direction 2: 25-model FR-RD experiment
                              (5 training conditions × 5 seeds on UCI Digits)
  fr_contrastive_benchmark.py Direction 3: NT-Xent vs FR-Contrastive
                              with false-negative injection
  aggregate_results.py        paper_benchmark → tsne_robustness_*.csv
  aggregate_dimred_stress.py  dimred_stress → dimred_stress_*.csv
  aggregate_ml_stress.py      soft_label + distillation → ml_stress_*.csv
  aggregate_vae_results.py    vae → vae_*.csv

reports/
  fisher_rao_vs_kl_arxiv.tex  Main paper (26 pages, all original experiments)
  fr_noisy_labels.tex         Direction 1 paper (6 pages)
  fr_representation_distance.tex  Direction 2 paper (7 pages)
  fr_contrastive.tex          Direction 3 paper (5 pages, theory + null result)
  references.bib              Shared bibliography (all papers)
  generate_figures.py         Main paper figures
  generate_noisy_label_figures.py  Direction 1 figures
  generate_fr_rd_figures.py   Direction 2 figures
  generate_fr_contrastive_figures.py  Direction 3 figures
  results/                    All experiment output CSVs (committed)
  figures/                    All generated figures (committed)
```

---

## Environment and Commands

**Package manager:** `uv` (not pip, not conda). Always prefix with `uv run --project .`

```bash
# Install
uv sync --project . --extra dev

# Test
uv run --project . pytest tests

# Lint
uv run --project . --extra dev ruff check .

# Run experiments
uv run --project . python experiments/noisy_label_benchmark.py        # Direction 1
uv run --project . python experiments/representation_distance_benchmark.py  # Direction 2
uv run --project . python experiments/fr_contrastive_benchmark.py     # Direction 3

# Regenerate figures
uv run --project . python reports/generate_figures.py
uv run --project . python reports/generate_noisy_label_figures.py
uv run --project . python reports/generate_fr_rd_figures.py
uv run --project . python reports/generate_fr_contrastive_figures.py

# Rebuild any paper PDF (run from reports/)
cd reports && pdflatex <paper>.tex && bibtex <paper> \
  && pdflatex <paper>.tex && pdflatex <paper>.tex
```

**Device:** Apple Silicon MPS. `get_device()` handles selection automatically.

---

## Core Implementations

### Fisher-Rao distance on the categorical simplex

```python
d_FR(p, q) = 2 * arccos(sum_i sqrt(p_i * q_i))   # bounded by π, symmetric, metric
```

- `categorical_fisher_rao_distance(p, q, eps)` — distance
- `categorical_fisher_rao_squared(p, q, eps)` — distance², used as t-SNE objective

### Distribution loss dispatch

`distribution_loss(target, prediction, objective, eps)` in `distribution_losses.py`.
Both `target` and `prediction` must be normalized probability vectors `(..., classes)`.
Returns a scalar averaged over the batch.

All 10 objectives: `kl`, `kl_smoothed`, `kl_capped`, `jensen_shannon`, `hellinger`,
`fisher_rao`, `fr_kl_hybrid`, `gce`, `mae`, `sce`.

New tests must be added to `test_distribution_objectives_have_gradients` — it auto-iterates
over `OBJECTIVES` in `distribution_losses.py`.

### FR Representation Distance

`fr_representation_distance(probs_a, probs_b)` in `representation_distance.py`.
A proper pseudometric on model output distributions:
```python
FR-RD(θ, φ; X) = E_{x~X}[d_FR(P_θ(x), P_φ(x))]
```
Also exports: `pairwise_fr_rd()`, `cka_linear()` (CKA baseline), `fr_ood_score()`.

---

## Detailed Experimental Results

### Original paper (main t-SNE experiments)

**What works:**
- Noisy-affinity stress (10 seeds): bad-edge preservation 48/48 cells, 43/48 p<0.05
- Noisy-affinity stress (10 seeds): bad-edge Q-mass 48/48, 48/48 p<0.05
- Soft-label classification: accuracy 11/16 cells, 9/16 p<0.05, mean +2.3%
- Distillation (corrupted teacher): teacher-error imitation 10/16 cells, 8/16 p<0.05

**What does NOT work:**
- Silhouette under clean targets: KL wins consistently (bounded codomain hurts separation)
- kNN-graph corruption: only 4/27 cells on bad-edge preservation
- VAE regularization: competitive but no reliable improvement
- JS and Hellinger match FR exactly — FR is not uniquely privileged

### Direction 1: Noisy label learning (`fr_noisy_labels.tex`)

4 datasets × 6 objectives × 5-6 noise regimes × 5-10 seeds.

**MLP family (Digits 10 seeds, MNIST 10 seeds, FashionMNIST 5 seeds):**

| Noise | KL | FR | MAE | GCE |
|---|---|---|---|---|
| Sym 40% Digits | 70.0% | 66.8% ↓ (p=0.002) | **89.6%** | 76.3% |
| Sym 60% Digits | 47.1% | 45.1% ↓ (p=0.004) | **68.2%** | 50.7% |
| Sym 40% MNIST  | 61.6% | 60.2% ↓ (p=0.059) | **76.0%** | 62.7% |

**ConvNet family (CIFAR-10, 5 seeds, 10k-sample subset):**

| Noise | KL | FR | MAE | GCE |
|---|---|---|---|---|
| Sym 20% | 78.7% | **81.1%** ↑ (5/5 wins) | 68.2% ↓ | 79.3% |
| Sym 40% | 72.4% | **74.4%** ↑ (5/5 wins) | 62.4% ↓ | **75.7%** |
| Sym 60% | 58.2% | 60.9% ↑ (5/5 wins) | 54.2% ↓ | **67.0%** |

**Key discovery: architecture-dependent reversal.** FR hurts MLP at sym noise (Ghosh
condition not satisfied) but helps ConvNet (+2-3%, consistent wins). MAE reverses
direction: dominant on MLP (+19.6%), severely harmful on ConvNet (−10%).

**Why (theory):** FR fails the Ghosh noise-tolerance condition — explains MLP behavior.
ConvNet reversal is not explained by asymptotic theory; batch-norm interaction +
FR's bounded gradient (max π²) likely prevent confident memorization of corrupted labels.

**Why (MAE failure on ConvNet):** MAE's flat gradient (constant loss magnitude) prevents
ConvNets from learning discriminative filters at moderate dataset scales (10k samples).

**BN ablation result (seed 0, clean regime only):**
| Objective | With BN | Without BN | Drop |
|---|---|---|---|
| FR | 84.2% | 77.2% | **−7.1%** (smallest drop!) |
| KL | 83.8% | 70.9% | −12.9% |
| SCE | 77.4% | 69.0% | −8.4% |
| GCE | 81.9% | 48.8% | −33.1% |
| Hellinger | 82.5% | 46.3% | −36.2% |
| MAE | 72.4% | 20.2% | −52.2% |

**FR is MORE robust to BN removal than KL** — consistent with bounded gradient providing
intrinsic gradient clipping that partially substitutes for BN's adaptive normalization.

**Practical rule:**
- MLP/tabular: use MAE or GCE for symmetric noise
- ConvNet/image: use FR or GCE; avoid MAE at moderate dataset sizes

### Direction 2: FR Representation Distance (`fr_representation_distance.tex`)

Validated on two datasets, both 25 MLP classifiers (5 conditions × 5 seeds):

**UCI Digits** (1,797 samples): between/within ratio=1.50×, r=0.74, CE vs FR RD=0.064
**MNIST** (3,000 train): between/within ratio=1.47×, r=0.74, CE vs FR RD=0.121

Near-identical statistics confirm dataset-agnostic stability of FR-RD properties.
noisy_60 highest within-condition variability (FR-RD ≈ 2.23-2.30) on both datasets.
Complementary to CKA: 1-CKA achieves r=0.87 on MNIST (vs FR-RD r=0.74), but CKA
is not a proper metric and measures representational geometry, not behavioral equivalence.

**OOD detection experiment (fr_rd_digits_ood.csv):**
Centroid-based FR-RD OOD scoring FAILS for well-trained classifiers:
- clean/fr_loss/smoothed: mean separation = −0.38 to −0.94 (inverted — ID > OOD)
- noisy_30: mixed (3 neg, 2 pos)
- noisy_60: slight positive separation (mean +0.21, 4/5 seeds positive)

**Root cause:** centroid of N near-one-hot predictions ≈ uniform distribution.
Confident ID predictions are FAR from uniform → high FR-RD. Uncertain OOD predictions
are closer → lower FR-RD. Fix: use class-conditional centroids instead.
This is documented as §4.4 in fr_representation_distance.tex and as a key limitation.

### Direction 3: FR-Contrastive (`fr_contrastive.tex`)

Theory: NT-Xent = (1/2N) Σ KL(e_i ‖ p_i). Replacing KL with d_FR² gives gradient
Θ(u^{-1/2}) on false negatives vs KL's Θ(u^{-1}).

Experiment: UCI Digits with explicit false-negative injection (0–30% rate, 5 seeds).
Both NT-Xent and FR-Contrastive achieve ~98.5% 5-NN at all rates — **null result**.
Dataset too easy; false-negative confusion doesn't degrade performance at this scale.
CIFAR-100/ImageNet needed to test the theoretical prediction.

---

## Data Integrity Warning

The original 10-seed confirmatory noisy-affinity run backing Table 1 of the main paper was
**overwritten** by the 5-seed 7-objective run (commit `bbe914f`). Table 1 claims 43/48 p<0.05
but current `dimred_stress_full.csv` has only 5 seeds → 0/48 at p<0.05 (min Wilcoxon p ≈ 0.063).

**Do not regenerate Table 1 from current CSV.** Original data is recoverable from commit
`2667d09` (git show 2667d09:reports/results/...).

---

## Research Priorities Going Forward

### Priority 1 — Nail the noisy-label story (most publishable, clearest finding)

The core Direction 1 finding is now a two-part story: FR hurts MLP, helps ConvNet.
The architecture-dependent reversal is the publishable hook. Current paper status:
- Theorem 1 formally proves FR/Hellinger cannot satisfy the Ghosh noise-tolerance condition
- Corollary 1 connects Theorem 1 to Bayes-optimality failure under symmetric noise
- Related work expanded to 5 paragraphs (noise-tolerant losses, sample-selection,
  real-world benchmarks CIFAR-N/Clothing-1M, architecture interactions, info geometry)
- BN ablation section added with preliminary results showing GCE collapses without BN

**Currently running experiments (as of 2026-05-10):**
- `cifar10_noisy_label_benchmark.py --seeds 10`: expanding CIFAR-10 to 10 seeds for p<0.05
  (at 157/300 rows as of session start; seeds 0-5 clean, seeds 0-4 for other regimes)
- `cifar10_no_bn_ablation.py --seeds 5`: batch-norm ablation on ConvNet
  (at 7/150 rows; only clean/seed=0 and sym_20/kl/seed=0 done)

**CRITICAL BUG FIXED:** The `random_crop_flip` function in both CIFAR-10 scripts had a
PyTorch advanced-indexing bug: mixing a bare `:` slice with non-contiguous advanced indices
produces `(b,h,w,c)` instead of `(b,c,h,w)`, causing Conv2d to crash. Fixed by indexing
all four dims explicitly:
```python
out = padded[
    torch.arange(b).view(b,1,1,1), torch.arange(c).view(1,c,1,1),
    rows.view(b,1,h,1), cols.view(b,1,1,w),
]
```

Remaining work:
1. **When 10-seed CIFAR-10 completes:** re-aggregate all datasets, update Table 3 with
   Wilcoxon p-values, update Limitations section to remove "pending" language.
2. **When no-BN ablation completes:** update the BN ablation table in Section 3.2, add
   interpretation of whether BN is the primary mediator of the ConvNet reversal.
3. **Add CIFAR-N (real human noisy labels).** Does the ConvNet advantage persist on
   real-world instance-dependent noise? Controlled experiment with 10k subsample.
4. **Vary dataset size (10k vs 50k).** To isolate the MAE flat-gradient hypothesis:
   does MAE recover on ConvNet when trained on the full 50k CIFAR-10?

### Priority 2 — FR-RD as a model analysis tool (ICLR 2027 target)

Direction 2 is now validated on two datasets (Digits + MNIST): consistent r=0.74 and
ratio≈1.47-1.50×. OOD experiment done — centroid approach fails for clean models
(documented as limitation + future work in paper). Remaining steps for full submission:

1. **Class-conditional centroid OOD experiment.** Score OOD samples as
   `min_c d_FR(centroid_c, P_θ(x))` where centroid_c is per-class. Expected to fix
   the inversion problem for well-trained models.
2. **Scale to fine-tuning divergence.** Take a pretrained ResNet, fine-tune on CIFAR variants
   with different data fractions, and show FR-RD tracks generalization gaps.
3. **Add 10-seed runs.** Currently 5 seeds per condition; 10 seeds would allow confidence
   intervals on the separation ratio and correlation.

### Priority 3 — FR-Contrastive (needs GPU compute, deferred)

Theory is established (`fr_contrastive.tex`). The null result on UCI Digits is expected and
honestly reported. Next step requires ~4 GPU-hours on CIFAR-10 under SimCLR protocol.
Do not invest further until Priority 1 and 2 are in submission shape.

---

## Statistical Framework

All experiments use this protocol:
- **Paired comparison:** same seeds, same initialization, same data split across objectives
- **Test:** Two-sided Wilcoxon signed-rank test on paired differences
- **Effect size:** Cliff's delta ∈ [-1, 1] (positive = FR tends to exceed KL)
- **Significance threshold:** p < 0.05 (all p-values reported, none Bonferroni-corrected)
- **Power note:**
  - n=5 seeds: min achievable p ≈ 0.063 → never claim p<0.05 with 5 seeds
  - n=10 seeds: min achievable p ≈ 0.002 → sufficient for strong claims
  - Always report win counts and mean oriented improvement alongside p-values

---

## Code Conventions

- All source in `src/fisher_rao_ml/` — import as `from fisher_rao_ml.X import Y`
- Experiments write to `reports/results/*.csv`. Never hardcode paths; use argparse defaults.
- Results CSVs and compiled `.bbl` files are committed. PDFs are gitignored.
- Always use `uv run --project .` — never `pip install`.
- Ruff lint must pass before committing: `uv run --project . --extra dev ruff check .`
- New distribution objectives must be added to `OBJECTIVES` in `distribution_losses.py`
  and are automatically covered by `test_distribution_objectives_have_gradients`.
- All experiments are resumable: read existing CSV on startup, skip completed (dataset,
  noise_regime, objective, seed) tuples unless `--force` is passed.
- No explanatory comments in code. Comments only for non-obvious constraints or invariants.

---

## Paper Status

| Paper | File | Status | Blocking issues |
|---|---|---|---|
| Main (t-SNE) | `fisher_rao_vs_kl_arxiv.tex` | Draft, 26 pages | Table 1 data integrity; no related work; small datasets |
| Direction 1 | `fr_noisy_labels.tex` | Near-complete, 11 pages | 10-seed CIFAR-10 for p-values; full BN ablation (5-seed) |
| Direction 2 | `fr_representation_distance.tex` | Near-complete, 11 pages | Class-conditional OOD fix; fine-tuning experiment |
| Direction 3 | `fr_contrastive.tex` | Theory only, 5 pages | Needs CIFAR/ImageNet experiments |

**Direction 1 paper (fr_noisy_labels.tex, 11 pages) now includes:**
- Theorem 1 + Corollary 1 (formal noise-tolerance analysis for FR/Hellinger)
- 5-paragraph related work section with 9 new references
- BN ablation section (§3.2) with seed-0 results including FR (−7.1%, smallest drop)
- Updated Discussion: FR robust to BN removal, suggesting bounded gradient = implicit clipping

**Direction 2 paper (fr_representation_distance.tex, 11 pages) now includes:**
- §4.4: Empirical OOD experiment — centroid failure mode documented
- Updated contributions, abstract, limitations, conclusion
- 5-paragraph related work
- Formal Limitations section

**Branch:** All work is now on `main`. Feature branches have been merged.
