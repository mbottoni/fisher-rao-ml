# Loop Agent — fr_noisy_labels paper maintenance

You are a stateless loop agent maintaining the research paper `reports/fr_noisy_labels.tex`.
Each invocation you check the current state of the repo and do exactly one useful unit of work,
then commit. If nothing needs doing, report idle and stop.

**Repo:** `~/home/fun/fisher-rao-ml`
**Always use:** `uv run --project .` (never pip). Ruff must pass before commits.
**Never re-run experiments** — they are long-running (hours to days). Only process completed results.

---

## Step 0 — Discover current state (do this first, every run)

```bash
# What is uncommitted?
git status --short

# What experiments are actively running?
ps aux | grep -E "(benchmark|gradient_norm|dynamic_loss|cifar)" | grep python | grep -v grep

# Which CSVs are newer than their figures?
# (compare mtime of reports/results/*.csv vs reports/figures/*.pdf)

# Any stubs remaining in the paper?
grep -n "TODO\|pending\|FIXME\|TBD\|\[.*running\]\|\[.*fill\]" reports/fr_noisy_labels.tex
```

Based on what you find, pick the **highest-priority task** below and do it.
Do not do multiple tasks in one run — commit and stop after one.

---

## Priority 1 — Regenerate stale figures

**Condition:** Any `reports/results/*.csv` is newer than its corresponding figure in `reports/figures/`.

**Mapping (CSV → figure script):**

| Results file(s) | Generate script |
|---|---|
| `cifar_n_full.csv` | `reports/generate_cifar_n_figures.py` |
| `gradient_norm_full.csv`, `mlp_gradient_norm_full.csv` | `reports/generate_gradient_norm_figures.py` + `reports/generate_gradient_saturation_figure.py` |
| `dynamic_loss_full.csv` | `reports/generate_dynamic_loss_figures.py` |
| `cifar10_noisy_label_full.csv`, `cifar10_no_bn_full.csv` | `reports/generate_noisy_label_figures.py` + `reports/generate_bn_ablation_figures.py` |
| `noisy_label_full.csv` | `reports/generate_noisy_label_figures.py` |

**Action:**
```bash
uv run --project . python reports/<relevant_script>.py
# then force-add any new/changed PDF figures:
git add -f reports/figures/*.pdf
```

---

## Priority 2 — Commit tracked files modified but not staged

**Condition:** `git status --short` shows `M` (modified) files in `reports/results/` or `reports/figures/`.

These are already git-tracked; they changed because an experiment completed or figures were regenerated.

**Action:** Stage and commit them with a descriptive message.
Do not commit: `*.aux`, `*.log`, `*.out`, `*.fls`, `*.fdb_latexmk`, `texput.log`, `.claude/`, or any PDF outside `reports/figures/` (the main paper PDF is gitignored).

```bash
git add reports/results/<changed>.csv reports/figures/<changed>.pdf
git commit -m "fr_noisy_labels: update <what> results and figures"
```

---

## Priority 3 — Fill paper stubs

**Condition:** `grep` from Step 0 finds any stub text (`[Results pending]`, `TODO`, etc.) in `reports/fr_noisy_labels.tex`.

**Action:**
1. Read the relevant CSV to get actual numbers.
2. Compute statistics: mean accuracy per objective per regime, paired Wilcoxon vs KL, win counts.
   ```bash
   uv run --project . python3 -c "
   import pandas as pd; from scipy.stats import wilcoxon
   df = pd.read_csv('reports/results/<relevant>.csv')
   # ... compute what the stub needs
   "
   ```
3. Replace the stub with real text. Follow the paper's statistical voice:
   - 10 seeds: can claim p=0.002 (p_min=0.002)
   - 5 seeds: say "suggestive (p=0.062 or p=0.125)" — never claim p<0.05
   - Always report: mean accuracy, win counts vs KL, p-value, direction
4. Rebuild PDF (see Priority 5).

---

## Priority 4 — Number consistency audit

**Condition:** No stubs found and no stale figures. Run this pass if Priority 1–3 are all clear.

Check that every number claimed in prose matches the actual CSV data.
Key claims to verify (re-compute from CSVs and compare to paper text):

```bash
uv run --project . python3 -c "
import pandas as pd
from scipy.stats import wilcoxon

# CIFAR-10 10-seed benchmark
df = pd.read_csv('reports/results/cifar10_noisy_label_full.csv')
for regime in ['sym_20','sym_40','sym_60','asym_40','clean']:
    sub = df[df.noise_regime==regime]
    for obj in ['fisher_rao','gce','hellinger','mae']:
        kl = sub[sub.objective=='kl']['test_accuracy'].values
        fr = sub[sub.objective==obj]['test_accuracy'].values
        if len(kl) and len(fr):
            diff = fr - kl
            wins = (diff>0).sum()
            _, p = wilcoxon(diff) if len(diff)>1 else (0,1.0)
            print(f'{regime} {obj}: mean={fr.mean()*100:.1f}% diff={diff.mean()*100:+.1f}pp wins={wins}/{len(diff)} p={p:.3f}')
"
```

If any number in the paper is wrong, correct it and note the fix in the commit message.

---

## Priority 5 — Rebuild PDF

**Condition:** `reports/fr_noisy_labels.tex` is newer than the last compile, OR figures changed.

**Always run from the `reports/` directory context** (the .bbl and .aux live there):

```bash
# Run from repo root — pdflatex needs to find figures/ relative to reports/
cd reports
pdflatex -interaction=nonstopmode fr_noisy_labels.tex
bibtex fr_noisy_labels
pdflatex -interaction=nonstopmode fr_noisy_labels.tex
pdflatex -interaction=nonstopmode fr_noisy_labels.tex 2>&1 | grep -E "^(!|Output written|Overfull)"
cd ..
```

If there are `!` errors, fix them. Overfull hboxes above 10pt badness should be fixed.
The PDF itself is gitignored (`*.pdf` in `.gitignore`) but `.bbl` files are committed:
```bash
git add reports/fr_noisy_labels.bbl
```

---

## Priority 6 — Prose quality pass

**Condition:** Priorities 1–5 are all clear.

One focused improvement per run. Pick whichever of these applies most:

- **Hedge removal:** Find sentences with "may", "might", "could potentially", "seems to" and sharpen them where the data supports a direct claim.
- **Number anchoring:** Every performance claim in the Discussion/Conclusion should cite both the absolute accuracy and the delta vs KL.
- **Ordering consistency:** The objective ranking in prose (GCE > Hellinger > FR > KL > MAE) should be consistent across Abstract, §3, Discussion, and Conclusion.
- **Passive voice:** Prefer active where the subject is clear ("FR outperforms KL" not "KL is outperformed by FR").

Make at most 10–15 targeted edits per run. Do not restructure sections.

---

## Commit rules

Every run that changes any file must end with a commit. Format:

```bash
git commit -m "fr_noisy_labels: <one-line description of what changed>"
```

Never commit:
- `*.aux`, `*.log`, `*.out`, `*.fls`, `*.fdb_latexmk`, `texput.log`
- `.claude/` directory
- `reports/fisher_rao_vs_kl_arxiv.pdf` or any top-level PDF (gitignored but tracked by mistake — do not re-add)
- Any experiment script or source code (this agent only touches `reports/`)

PDFs in `reports/figures/` ARE committed (they were force-added previously). Use `git add -f` to add new ones.

---

## Idle condition

If after Step 0 you find:
- No uncommitted tracked changes
- No CSVs newer than their figures
- No stubs in the paper
- No number inconsistencies
- PDF is up to date

Then report: **IDLE — paper is consistent and all results are committed.** Do not make a trivial commit.

---

## Previously fixed (do not re-fix)

These issues have already been corrected. Skip them during prose passes.

- **Conclusion prose** (commit `5f5d6a7`): "FR and Hellinger exhibit gradient saturation (ratio ≈ 1, stable)" → "FR's gradient saturates (ratio 0.98≈1), Hellinger passively reduces (ratio 0.88), and GCE actively deweights (0.69↓)".
- **Table `tab:grad_ratio`** (commit `539d027`): Hellinger Ep59 annotation changed from `$\approx 1$` to `$\downarrow$ soft`; caption updated to reflect three distinct mechanisms.
- **Discussion body + figure caption** (commit `7a1455f`): Bullet header changed from "FR and Hellinger (gradient saturation, ratio ≈ 1, stable)" to "FR (gradient saturation, ratio 0.98≈1) and Hellinger (soft reduction, ratio 0.88)"; line "Both ratios stabilise near 1.0" corrected; figure caption updated; summary ordering line changed from "gradient saturation" to "soft saturation" for Hellinger.

---

## Project context (do not act on this, just for background)

**Core finding:** FR hurts MLP under symmetric noise (fails Ghosh condition), helps ConvNet
(+2.4% sym_40, 10/10 wins p=0.002). The key property is bounded codomain + symmetry, not
FR geometry. Hellinger behaves nearly identically; GCE dominates at high symmetric noise.

**Completed experiments:**
- `cifar10_noisy_label_full.csv` — 300 rows (6 obj × 5 regimes × 10 seeds, CIFAR-10 ConvNet)
- `cifar10_no_bn_full.csv` — 150 rows (BN ablation, 5 seeds)
- `cifar_n_full.csv` — 180 rows (3 noise types × 6 obj × 10 seeds, real human labels)
- `gradient_norm_full.csv` — 2160 rows (6 obj × 3 seeds × 60 epochs)
- `mlp_gradient_norm_full.csv` — MLP gradient norm analysis
- `dynamic_loss_full.csv` — 275 rows (11 schedules × 5 regimes × 5 seeds)
- `noisy_label_full.csv` — MLP results (Digits/MNIST/FashionMNIST)

**Deferred experiments (needs GPU or long MPS time — do not trigger):**
- `resnet_noisy_label_benchmark.py` — ResNet-18 on full CIFAR-10 (~4h GPU)
- `cifar10_noisy_label_benchmark.py --n-train 50000` — 50k size ablation (~8h MPS)

**Statistical power:**
- 10 seeds → p_min=0.002. Can claim p<0.05 and use "significant".
- 5 seeds → p_min=0.063. Say "suggestive (p=X)" — never claim significance.

**Paper:** `reports/fr_noisy_labels.tex`, 26 pages, compiles clean with no errors or overfull warnings.
Target venue: NeurIPS 2026. Submission deadline not yet set.
