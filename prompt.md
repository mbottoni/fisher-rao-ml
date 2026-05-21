You are continuing work on a research paper about Fisher-Rao loss under label noise.
The repo is at: ~/home/fun/fisher-rao-ml
Always use `uv run --project .` (never pip). Ruff lint must pass before commits.

---

## PROJECT CONTEXT

This project studies Fisher-Rao (FR) geodesic distance as an alternative to KL divergence.
The main paper for this session is `reports/fr_noisy_labels.tex` (12 pages,
"Fisher–Rao Loss Under Label Noise: Architecture Mediates Robustness").

**Core finding:** FR hurts MLP under symmetric noise (fails Ghosh noise-tolerance condition),
but helps ConvNet (+2.4% sym_40, 10/10 wins, p=0.002). MAE reverses on ConvNets.
The property driving the ConvNet advantage is *bounded codomain + symmetry*, not FR geometry
specifically. Hellinger behaves nearly identically to FR.

---

## CURRENT STATE — ALL EXPERIMENTS COMPLETE

All three previously-running experiments are now done:

1. **cifar_n_benchmark** — COMPLETE (180 rows = 3 noise_types × 6 obj × 10 seeds)
   Results already in paper (§3.3, Table 3, Figures cifar_n_accuracy_bars.pdf,
   cifar_n_gain_comparison.pdf, cifar_n_vs_synthetic.pdf). Figures exist.
   Key finding: FR wins 10/10 at all 3 CIFAR-N conditions (p=0.002); GCE hurts at
   aggre (low noise), suggesting gradient saturation is more robust than active
   deweighting for real instance-dependent noise.

2. **gradient_norm_analysis** — COMPLETE (2160 rows = 6 obj × 3 seeds × 60 epochs × 2 phases)
   Figures exist: gradient_norm_trajectories.pdf, gradient_norm_ratio.pdf,
   gradient_norm_loss_curves.pdf, gradient_norm_architecture_comparison.pdf.
   Table 6 in paper has mechanistic ratios: KL=1.43↑ (memorises), FR=0.99≈1 (saturates),
   GCE=0.71↓ (deweights). This section appears complete.

3. **dynamic_loss_benchmark** — COMPLETE (275 rows = 11 schedules × 5 regimes × 5 seeds)
   RESULTS (mean accuracy, 5 seeds):
   | Regime  | KL    | FR    | GCE   | Hell  | FR→GCE | GCE→FR |
   |---------|-------|-------|-------|-------|--------|--------|
   | clean   | 0.831 | 0.830 | 0.827 | 0.833 | 0.822  | 0.832  |
   | sym_20  | 0.729 | 0.783 | 0.796 | 0.789 | 0.789  | 0.792  |
   | sym_40  | 0.625 | 0.671 | 0.748 | 0.719 | 0.736  | 0.711  |
   | sym_60  | 0.467 | 0.470 | 0.628 | 0.553 | 0.587  | 0.540  |
   | asym_40 | 0.545 | 0.533 | 0.556 | 0.547 | 0.564  | 0.546  |

   Statistical comparisons (FR→GCE vs static GCE, paired Wilcoxon, 5 seeds):
   - clean:   diff=−0.005, 1/5 wins, p=0.125 (not significant)
   - sym_20:  diff=−0.007, 1/5 wins, p=0.125
   - sym_40:  diff=−0.012, 0/5 wins, p=0.062
   - sym_60:  diff=−0.042, 0/5 wins, p=0.062
   - asym_40: diff=+0.008, 4/5 wins, p=0.125
   **CONCLUSION: FR→GCE does NOT beat static GCE. Null result (insufficient power at 5 seeds).**
   The curriculum switching hypothesis is not confirmed. GCE dominates at symmetric noise;
   FR→GCE is neutral-to-slightly-worse at sym and marginally better only at asym_40.

---

## WHAT NEEDS TO BE DONE

### Task 1 — Generate dynamic loss figures
Run: `uv run --project . python reports/generate_dynamic_loss_figures.py`
This should produce figures in reports/figures/. If the script errors or the figures
look wrong, debug and fix.

### Task 2 — Fill in the dynamic loss appendix (reports/fr_noisy_labels.tex)
Currently lines ~1448–1491 end with:
  \paragraph{Results.}
  [Results pending — experiment currently running.]
  \end{document}

Replace with actual results. Given the null result, the narrative should:
- Report the numbers honestly (FR→GCE does not beat GCE)
- Note that 5 seeds gives p_min≈0.063, so the trend at asym_40 (4/5 wins) is
  suggestive but not confirmable
- Interpret: GCE's dominance at sym_60 (0.628) over FR (0.470) is so large that
  a warm-start from FR doesn't close the gap — the mechanisms are not simply
  compositional in a two-phase schedule with a fixed switch epoch
- Conclude: the gradient norm analysis motivates the hypothesis but the two-phase
  schedule at switch_epoch=30 does not realize the theoretical benefit; future work
  could use adaptive switching (e.g., when gradient ratio crosses a threshold)

### Task 3 — Improve the paper overall
After completing Tasks 1–2, do a full pass on fr_noisy_labels.tex for:
- **Abstract:** Update to mention CIFAR-N results (FR wins 10/10 at all 3 conditions)
  and the dynamic loss null result (briefly, as negative finding)
- **Contributions list:** Add CIFAR-N and dynamic loss as contributions (iii) and (iv)
- **Discussion:** Check that the gradient saturation narrative is internally consistent
  with the dynamic loss null result — if GCE mechanisms don't compose with FR, what
  does that imply about the mechanisms?
- **Conclusion:** Verify it mentions all four main empirical findings:
  (1) arch reversal, (2) BN ablation, (3) CIFAR-N, (4) gradient norms
- Tighten writing; remove hedge words; ensure all numbers match tables

### Task 4 — Rebuild the PDF and commit
```bash
cd reports && pdflatex fr_noisy_labels.tex && bibtex fr_noisy_labels \
  && pdflatex fr_noisy_labels.tex && pdflatex fr_noisy_labels.tex
```
Then commit:
- New/updated figures from Task 1
- Updated fr_noisy_labels.tex from Tasks 2–3
- Updated reports/results/dynamic_loss_aggregated.csv if regenerated
Commit message format: `fr_noisy_labels: <what changed>`

---

## CODE CONVENTIONS
- All source in src/fisher_rao_ml/ — import as `from fisher_rao_ml.X import Y`
- No explanatory comments; comments only for non-obvious invariants
- Ruff lint must pass: `uv run --project . --extra dev ruff check .`
- Results CSVs and .bbl files are committed; PDFs are gitignored (do not commit PDFs)
- Papers are in reports/*.tex; figures in reports/figures/; results in reports/results/

---

## IMPORTANT CONSTRAINTS
- Do NOT run the benchmark experiments again (they are all complete)
- Do NOT modify any results CSVs
- Dynamic loss is 5 seeds — never claim p<0.05 for it; always say "suggestive (p=0.125)"
- CIFAR-N is 10 seeds — p_min=0.002 claims are valid there
- The paper title and core finding (architecture reversal) must not change
