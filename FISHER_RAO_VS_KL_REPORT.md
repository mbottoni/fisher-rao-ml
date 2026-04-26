# Fisher-Rao vs KL Training Report

Generated on 2026-04-26 from local smoke experiments in this repository.

## How to Run Tests

From the project root:

```bash
cd /Users/maruanottoni/home/fun/fisher-rao-ml
uv sync --extra dev
uv run --project . --extra dev pytest tests
```

Equivalent Make target:

```bash
make test
```

Current result:

```text
2 passed in 1.18s
```

The tests check the core research requirement: Fisher-Rao objectives produce finite PyTorch
autograd gradients for both categorical distributions and diagonal Gaussian latent distributions.

## Experiment Commands

t-SNE-style distribution matching:

```bash
uv run --project . python experiments/tsne_fisher_rao.py \
  --objective kl --steps 200 --n-samples 180 --lr 0.05 --bandwidth 5.0 --seed 11

uv run --project . python experiments/tsne_fisher_rao.py \
  --objective fisher_rao --steps 200 --n-samples 180 --lr 0.05 --bandwidth 5.0 --seed 11
```

VAE latent regularization:

```bash
uv run --project . python experiments/vae_fisher_rao.py \
  --regularizer kl --epochs 1 --max-train-samples 4096 --batch-size 128 \
  --latent-dim 8 --seed 12

uv run --project . python experiments/vae_fisher_rao.py \
  --regularizer fisher_rao --epochs 1 --max-train-samples 4096 --batch-size 128 \
  --latent-dim 8 --seed 12
```

Open tracked runs:

```bash
uv run --project . mlflow ui --backend-store-uri ./mlruns
```

All runs used `mps` locally.

## Observed Metrics

### t-SNE Smoke Run

Both objectives trained stably for 200 steps.

KL loss:

```text
step 0:   1.613072
step 25:  0.642285
step 50:  0.298945
step 100: 0.182499
step 150: 0.141878
step 199: 0.119724
```

Fisher-Rao squared loss:

```text
step 0:   4.565233
step 25:  1.897354
step 50:  0.896415
step 100: 0.527659
step 150: 0.406908
step 199: 0.339899
```

Interpretation:

- The raw loss values are not directly comparable because KL and squared Fisher-Rao have different
  units and scales.
- The important signal is that both losses decrease smoothly under the same optimizer and learning
  rate.
- Fisher-Rao is viable as a differentiable replacement objective for this t-SNE-style probability
  matching prototype.
- The next comparison should use embedding quality metrics, not only final objective value:
  trustworthiness, neighborhood preservation, cluster separation, and visual artifacts.

### VAE Smoke Run

Both regularizers trained stably for one short MNIST epoch over 4096 samples.

KL regularizer:

```text
step 0:
  total loss:       550.442749
  reconstruction:   550.412415
  regularization:     0.030364

step 25:
  total loss:       224.526077
  reconstruction:   216.819153
  regularization:     7.706926
```

Fisher-Rao regularizer:

```text
step 0:
  total loss:       550.472900
  reconstruction:   550.412415
  regularization:     0.060463

step 25:
  total loss:       228.499435
  reconstruction:   219.285187
  regularization:     9.214247
```

Interpretation:

- Both models reduce reconstruction loss quickly.
- Fisher-Rao starts around 2x the KL regularization value near the standard normal prior in this
  setup, and remains somewhat larger at step 25.
- With `beta=1.0`, Fisher-Rao applies a stronger early latent penalty than KL here.
- This may produce a more geometrically meaningful latent constraint, but it could also slow
  reconstruction unless `beta` is tuned separately for each objective.

## Research Takeaways

The initial result is positive but preliminary:

- Fisher-Rao gradients are practical with PyTorch autograd for the implemented categorical and
  diagonal Gaussian cases.
- In t-SNE, Fisher-Rao can replace KL without immediate optimization failure.
- In the VAE, Fisher-Rao behaves like a stronger latent regularizer than KL under the same `beta`.
- Raw losses should not be compared directly across objectives; compare downstream metrics instead.

## Implemented Final-Model Comparison

The experiment scripts now log objective-independent `eval_*` metrics to MLflow.

t-SNE final embedding metrics:

- `eval_trustworthiness`: local-neighborhood preservation from original space to embedding space.
- `eval_neighborhood_recall`: fraction of original nearest neighbors recovered in the embedding.
- `eval_silhouette`: label-aware cluster separation in the final embedding.
- `eval_knn_accuracy`: kNN classification accuracy using only the final embedding.

VAE final model metrics:

- `eval_loss`, `eval_reconstruction`, `eval_regularization`: final held-out objective terms.
- `eval_bce_per_pixel`: reconstruction quality on held-out data.
- `eval_mean_norm`: typical distance of posterior means from the prior center.
- `eval_variance_mean`: average posterior variance.
- `eval_active_units`: number of latent dimensions with nontrivial posterior-mean variance.
- `eval_latent_knn_accuracy`: class information retained in latent means.

These are the metrics to compare across KL and Fisher-Rao after matching architecture, seed,
training budget, and tuned `beta`.

## Recommended Next Experiments

1. Run beta sweeps for the VAE, especially `beta` values lower than `1.0` for Fisher-Rao.
2. Run multiple seeds and report mean/std for each objective.
3. Compare latent geometry in the VAE with interpolation plots and class-conditioned latent
   statistics.
4. Add robustness tests with corrupted inputs or noisy high-dimensional features.

The main hypothesis to test next is not "Fisher-Rao gives a lower loss", because the loss scales
are different. The better hypothesis is:

```text
Fisher-Rao gives equal or better representation quality, stability, or latent geometry
after tuning its scale separately from KL.
```
