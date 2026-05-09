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
                               6 noise regimes; resumable; outputs to reports/results/)
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

Benchmark: 6 objectives × 6 noise regimes × 10 seeds on UCI Digits.

| Noise | KL | FR | MAE | GCE | SCE |
|---|---|---|---|---|---|
| Clean | 97.9% | 97.9% | 98.0% | 98.0% | 97.9% |
| Sym 20% | 86.2% | 86.4% | **96.6%** | 92.4% | 91.9% |
| Sym 40% | 70.0% | **66.8%** ↓ | **89.6%** | 76.3% | 76.1% |
| Sym 60% | 47.1% | **45.1%** ↓ | **68.2%** | 50.7% | 50.6% |
| Sym 80% | 21.7% | 22.0% | **29.5%** | 23.1% | 23.6% |
| Asym 40% | 61.1% | 60.3% | **67.3%** | 60.9% | 61.2% |

**Key result:** FR is significantly worse than KL at sym_40 (−3.2%, p=0.002, 0/10 wins)
and sym_60 (−2.0%, p=0.004, 0/10 wins). MAE dominates (+19.6% at sym_40, 10/10 wins).

**Why:** FR does not satisfy the Ghosh (2017) noise-tolerance condition (constant per-class
sum). MAE does. This is a definitive negative result, not a gap to be fixed.

**Practical rule:** Use MAE or GCE for symmetric label noise. Use FR only when targets are
soft probability distributions with overconfident wrong mass.

### Direction 2: FR Representation Distance (`fr_representation_distance.tex`)

25 MLP classifiers (5 conditions × 5 seeds, UCI Digits).
- Between/within condition FR-RD ratio: **1.50×**
- Pearson(FR-RD, |accuracy difference|) = **0.74**
- noisy_60 condition has highest within-condition variability (FR-RD = 2.23), as expected
- Complementary to CKA (output distributions vs hidden layer geometry)

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

The Direction 1 result is scientifically complete and honest: FR is not suitable for
symmetric label noise; the right loss is MAE. The remaining work to make this publishable:

1. **Scale to CIFAR-10 with ResNet-18.** The UCI Digits result is convincing theoretically
   but reviewers will ask for a standard vision benchmark. 10 seeds, same 6 objectives.
2. **Add CIFAR-N (real human noisy labels).** Symmetric noise is synthetic; real noise has
   class structure. Does FR perform better or worse on real noise?
3. **Find where FR does beat MAE/GCE.** The hypothesis: when training targets are soft
   distributions (label smoothing, knowledge distillation, mixture targets), FR's advantage
   reappears. Design a mixed-noise experiment to find the crossover point.
4. **Frame as a diagnostic.** "When to use which robust loss" is a more publishable angle
   than "FR is the best." Ghosh condition as a unifying framework.

### Priority 2 — FR-RD as a model analysis tool (low-hanging fruit, ICLR workshop target)

The Direction 2 results are solid for a workshop paper. To make it a full conference paper:

1. **Scale to fine-tuning divergence.** Take a pretrained ResNet, fine-tune on CIFAR variants
   with different data fractions, and show FR-RD tracks generalization gaps.
2. **OOD detection application.** FR-RD from ID centroid as an OOD score; compare to
   Mahalanobis distance and energy score on standard benchmarks (CIFAR-10 vs SVHN).
3. **Training dynamics.** Plot FR-RD between checkpoints every N epochs across different
   architectures — does it reveal phase transitions in learning?

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
| Direction 1 | `fr_noisy_labels.tex` | Complete, 6 pages | Scale to CIFAR-10 for venue submission |
| Direction 2 | `fr_representation_distance.tex` | Complete, 7 pages | Scale to fine-tuning / OOD experiments |
| Direction 3 | `fr_contrastive.tex` | Theory only, 5 pages | Needs CIFAR/ImageNet experiments |

**Branch:** All work is now on `main`. Feature branches have been merged.
