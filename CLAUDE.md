# CLAUDE.md — fisher-rao-ml

Research project studying Fisher-Rao geodesic distance as a replacement for KL divergence
in ML objectives. The goal is publications at top-tier venues (NeurIPS, ICML, ICLR).

---

## Project Goal

The central research question: **When does replacing KL divergence with a bounded symmetric
divergence improve robustness in representation learning?**

Key finding so far: bounded symmetric divergences (Fisher-Rao, Jensen-Shannon, Hellinger) all
resist high-confidence false-neighbor edges equally well. Fisher-Rao is not uniquely privileged
— the operative property is bounded codomain + symmetry, not the geodesic geometry. The most
compelling positive signal is in **soft-label classification and distillation under
confidently-wrong labels** (+5–13% accuracy, p<0.05 in 10/16 cells).

The paper in `reports/fisher_rao_vs_kl_arxiv.tex` documents all experiments. The branch
`feat/improv` contains the full extended results.

---

## Repository Layout

```
src/fisher_rao_ml/          Core library (import as fisher_rao_ml)
  losses.py                 categorical_fisher_rao_distance/squared
                            diagonal_gaussian_fisher_rao_distance/squared
  distribution_losses.py    distribution_loss() — 7 objectives (kl, kl_smoothed,
                            kl_capped, jensen_shannon, hellinger, fisher_rao, fr_kl_hybrid)
  tsne.py                   pairwise_student_t_affinities, symmetric_gaussian_affinities,
                            perplexity_gaussian_affinities, tsne_distribution_loss
  evaluation.py             trustworthiness, neighborhood_recall, silhouette, knn_accuracy,
                            corrupted_edge_preservation, corrupted_edge_q_mass
  vae.py                    VAE model (diagonal Gaussian encoder/decoder)
  device.py                 get_device() — returns MPS on Apple Silicon, else CPU

experiments/                Runnable scripts (all accept --help)
  paper_benchmark.py        Main t-SNE benchmark (3 datasets, feature-noise levels,
                            --bandwidth-mode global_median|perplexity, --perplexity 30)
  dimred_stress_benchmark.py Noisy-affinity + kNN-graph corruption stress tests
  soft_label_benchmark.py   Soft-label classification under noisy targets
  distillation_benchmark.py Corrupted teacher-student distillation
  vae_benchmark.py          VAE with KL vs Fisher-Rao regularizer
  aggregate_results.py      Aggregates paper_benchmark → tsne_robustness_{full,aggregated,significance}.csv
  aggregate_dimred_stress.py Aggregates dimred_stress → dimred_stress_{aggregated,significance,
                            baseline_significance,power_summary,baseline_power_summary}.csv
  aggregate_ml_stress.py    Aggregates soft_label + distillation → ml_stress_*.csv
  aggregate_vae_results.py  Aggregates vae → vae_*.csv

reports/
  fisher_rao_vs_kl_arxiv.tex  Main paper (LaTeX)
  references.bib              Bibliography
  generate_figures.py         Reads all CSVs → writes figures/*.pdf
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

# Run individual experiments
uv run --project . python experiments/paper_benchmark.py
uv run --project . python experiments/dimred_stress_benchmark.py
uv run --project . python experiments/soft_label_benchmark.py
uv run --project . python experiments/distillation_benchmark.py

# Aggregate results
uv run --project . python experiments/aggregate_results.py
uv run --project . python experiments/aggregate_dimred_stress.py
uv run --project . python experiments/aggregate_ml_stress.py

# Regenerate figures
uv run --project . python reports/generate_figures.py

# Rebuild paper PDF (run from reports/)
cd reports && pdflatex fisher_rao_vs_kl_arxiv.tex && bibtex fisher_rao_vs_kl_arxiv \
  && pdflatex fisher_rao_vs_kl_arxiv.tex && pdflatex fisher_rao_vs_kl_arxiv.tex

# Full pipeline
make paper-all
```

**Device:** Apple Silicon MPS. `get_device()` in `device.py` handles selection automatically.

---

## Core Implementations

### Fisher-Rao distance on the categorical simplex

```python
d_FR(p, q) = 2 * arccos(sum_i sqrt(p_i * q_i))     # bounded by π, symmetric, metric
```

Implemented in `losses.py`:
- `categorical_fisher_rao_distance(p, q, eps)` — returns distance
- `categorical_fisher_rao_squared(p, q, eps)` — returns distance², used as t-SNE objective

### Distribution loss objectives

`distribution_loss(target, prediction, objective, eps)` in `distribution_losses.py` dispatches
to one of 7 objectives. `target` and `prediction` must be normalized probability vectors of
shape `(..., classes)`. Returns a scalar (mean over batch).

### t-SNE affinity construction

- `symmetric_gaussian_affinities(x, bandwidth)` — global bandwidth heuristic (default)
- `perplexity_gaussian_affinities(x, perplexity=30)` — per-point bandwidth via binary search
  on Shannon entropy; symmetrizes as P_ij = (P_{j|i} + P_{i|j}) / 2n, then normalizes

---

## Key Experimental Results

### What works (positive signal)
| Experiment | Metric | Result |
|---|---|---|
| Noisy-affinity stress (10 seeds) | Bad-edge preservation | 48/48 cells, 43/48 p<0.05 |
| Noisy-affinity stress (10 seeds) | Bad-edge Q mass | 48/48 cells, 48/48 p<0.05 |
| Soft-label classification | Accuracy | 11/16 cells, 9/16 p<0.05, mean +2.3% |
| Soft-label classification | ECE | 10/16 cells, 9/16 p<0.05 |
| Soft-label classification | Brier score | 13/16 cells, 10/16 p<0.05 |
| Distillation (corrupted teacher) | Teacher-error imitation | 10/16 cells, 8/16 p<0.05 |

### What does NOT work (negative/boundary)
- **Silhouette under clean targets:** KL wins consistently (Fisher-Rao bounded codomain
  hurts cluster separation). Reverses on blobs only under perplexity-adaptive bandwidth.
- **kNN-graph corruption:** Fisher-Rao does not help when the neighbor graph (not affinity
  mass) is corrupted. 4/27 cells on bad-edge preservation.
- **VAE regularization:** Not compelling after beta tuning. Fisher-Rao is competitive but
  no reliable improvement on reconstruction or latent classification.
- **Jensen-Shannon and Hellinger match Fisher-Rao** on the affinity-corruption task exactly
  (both 47/48 and 48/48). FR is not uniquely privileged.

### Data integrity note
The original confirmatory noisy-affinity run (10 seeds, KL+FR only) that produces Table 1's
p<0.05 numbers was **overwritten** by the 5-seed 7-objective run in commit `bbe914f`. Table 1
numbers (48/48, 43/48 p<0.05) come from commit `HEAD~2` (recoverable from git). The current
`dimred_stress_full.csv` has only 5 seeds → 0/48 at p<0.05 (minimum Wilcoxon p ≈ 0.063).
**Do not regenerate Table 1 from current CSV — it will produce wrong p-value counts.**

---

## Next Research Directions (Toward Publication)

The research agenda is moving toward top-tier conference submissions. Current priorities:

### Priority 1: Noisy Label Learning at Scale (NeurIPS 2026 target)
**Why:** The soft-label signal is the strongest in the paper (+5–13% accuracy under
random-wrong and class-confusion corruption at p=0.002). Scale this to CIFAR-10/100 with
ResNet-18, CIFAR-N (real human noisy labels), and Clothing1M.

**Theory needed:** A formal noise-tolerance lemma showing FR's gradient is O(ε · u^{-1/2})
under a corrupted label distribution, vs KL's O(ε · u^{-1}). This is the information-geometric
extension of Ghosh et al. (2017) for MAE. Add to `reports/fisher_rao_vs_kl_arxiv.tex` as a
theorem (not just a proposition sketch).

**Baselines to add:** Symmetric Cross-Entropy (SCE), GCE (q-loss), MAE, NCE+MAE. These are
all in the noisy-label literature and must be beaten to justify a top-venue submission.

**Key claim:** Fisher-Rao is the *natural* bounded robust loss on the categorical simplex —
it is the geodesic distance under the Fisher information metric, not an ad hoc engineering
choice like GCE or SCE.

### Priority 2: FR as a Representation Similarity Metric (ICLR 2027)
**Why:** This requires no expensive training. CKA (Kornblith et al. 2019) is the dominant
metric for comparing neural network representations and has known limitations (not a metric,
not invariant under isometries). FR-distance between model output distributions is a proper
metric with bounded, interpretable values.

**Definition:**
```
d_FR(θ, φ; X) = E_{x~X}[d_FR(P_θ(x), P_φ(x))]
```

**Applications:** training dynamics, fine-tuning divergence, OOD detection, model compression
quality. Implement in `src/fisher_rao_ml/representation_distance.py`.

### Priority 3: Contrastive Learning with False Negatives (NeurIPS 2027)
**Why:** False negatives in contrastive learning are structurally identical to false
high-confidence affinity edges in t-SNE. NT-Xent can be written as a KL divergence;
replacing it with FR gives bounded gradient pressure on false negatives.

**Challenge:** Needs ImageNet-scale experiments (~4 A100-days). Highest impact if it works.

---

## Statistical Framework

All experiments follow this protocol:
- **Paired comparison:** same seeds, same initialization, same data across objectives
- **Test:** Two-sided Wilcoxon signed-rank test on paired differences (does not assume normality)
- **Effect size:** Cliff's delta ∈ [-1, 1] (positive = Fisher-Rao tends to exceed KL)
- **Significance threshold:** p < 0.05 (not Bonferroni-corrected; all p-values reported)
- **Power note:** With n=5 seeds, minimum achievable Wilcoxon p ≈ 0.063. Never claim p<0.05
  significance in 5-seed experiments. Use win counts and mean oriented improvement instead.
  With n=10 seeds, minimum p ≈ 0.021.

---

## Code Conventions

- All source code in `src/fisher_rao_ml/` — import with `from fisher_rao_ml.X import Y`
- Experiments write to `reports/results/*.csv` (raw and aggregated). Never hardcode paths;
  use argparse defaults so paths are overridable.
- Results CSVs are committed to the repo. Figures are committed after `generate_figures.py`.
- Never write `pip install`. Always use `uv run --project .`.
- Ruff lint passes before every commit (`uv run --project . --extra dev ruff check .`).
- Tests live in `tests/`. Run `pytest tests` before committing. New objectives must be covered
  by `test_distribution_objectives_have_gradients` (it auto-covers all items in `OBJECTIVES`).
- No comments explaining what the code does. Comments only for non-obvious WHY (constraint,
  workaround, invariant).
- Experiments are resumable: completed cells are detected by reading existing CSVs and skipped
  unless `--force` is passed. Maintain this pattern for all new experiments.

---

## Paper Status

**Branch:** `feat/improv`
**File:** `reports/fisher_rao_vs_kl_arxiv.tex` (26 pages)

**Current issues to fix before submission:**
1. Table 1 (lines ~533-545) claims 48/48 FR improves and 43/48 p<0.05 for bad-edge
   preservation — but the current data CSV reflects the 5-seed run (47/48, 0 p<0.05).
   Either re-run the original 10-seed experiment or recover from git and store separately.
2. No figure exists for the perplexity-bandwidth results (Section `sec:perplexity-results`).
   Add `save_perplexity_bandwidth_comparison()` to `reports/generate_figures.py`.
3. No related work section covering robust t-SNE/UMAP variants, UMAP, PaCMAP, and the
   noisy-label literature (GCE, SCE, MAE). Required for any venue submission.
4. Scale: all results are on 200-300 point datasets. For NeurIPS, add ResNet-18 on CIFAR-10.
