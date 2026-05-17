---
name: fisher-rao-ml project overview
description: Core goals, structure, experiments, and findings for the Fisher-Rao vs KL research project
type: project
---

Research project: Fisher-Rao geodesic distance as ML training objective. Target: NeurIPS/ICML publication.

**Core Finding (Direction 1 - main submission target):**
Architecture-dependent reversal: FR hurts MLP under symmetric noise (fails Ghosh noise-tolerance condition), helps ConvNet (+2.4% at sym_40, p=0.002). MAE reverses: dominant on MLP (+19.6%), severely harmful on ConvNet (-9.6%).

**Theory (fr_noisy_labels.tex — UPDATED session 7):**
- Theorem 1: FR/Hellinger cannot satisfy Ghosh noise-tolerance condition (explains MLP failure)
- Proposition 2: FR's logit gradient ∂ℓ/∂z_k → 0 as p_k → 0 (gradient saturation) — explains ConvNet RECOVERY
  - FR gradient EXCEEDS KL for p_k ∈ (0.032, 1) — the early-training amplification "danger zone"
  - Only when ConvNets push p_k < 0.032 on corrupted labels does saturation dominate
  - On MLP, early amplification (p_k ≈ 0.28, FR grad = 1.82 > KL max = 1.0) dominates → ratio grows to 2.61
- Corollary 3: FR/KL gradient ratio ~ 2π√p_k = O(√p_k) as p_k → 0
- **"Universally soft" loss class**: A loss is universally soft if |∂ℓ/∂z_k| < |∂ℓ_KL/∂z_k| for ALL p_k ∈ (0,1)
  - Hellinger: ∂ℓ_H/∂z_k = -√p_k(1-p_k)/2, max ≈ 0.192 at p_k=1/3. ALWAYS < KL (proof: √p_k/2 < 1)
  - GCE(q>0): ∂ℓ_GCE/∂z_k = -p_k^q(1-p_k), max ≈ 0.316 at p_k=0.41. ALWAYS < KL (proof: p_k^q < 1)
  - FR: NOT universally soft. Exceeds KL for p_k ∈ (0.032, 1)
  - Explains: Hellinger (+7.6%) > FR (+3.4%) at sym_60; GCE (+9.5%) > Hellinger via active deweighting
- **NEW (session 7): Danger zone persistence explains asym failure**:
  - Sym noise: corrupted class p_k stays low (≈ε/(K-1) per wrong class, spread), often below 0.032
  - Asym noise: consistent class pair (e.g., cat→dog) pushes p_k(wrong) into 0.3-0.4 (deep danger zone)
  - FR amplifies wrong-label gradient MORE than KL at asym_40 → explains -2.7% hurt
  - Hellinger neutral at asym_40 (universally soft → always < KL → less wrong-label pressure)
  - Prediction: asym_20/asym_60 should show same pattern (added to cifar10_noisy_label_benchmark.py)
- **NEW (session 7): GCE low-noise suppression effect**:
  - At low real noise (CIFAR-N aggre ~9%): GCE hurts (p=0.008) because active deweighting
    suppresses gradients from clean majority, reducing useful signal
  - FR/Hellinger don't suppress clean-majority signals → help even at low noise
  - Noise-rate sensitivity principle: active deweighting (GCE) optimal at high noise;
    gradient saturation (FR, Hellinger) preferable at low/instance-dependent noise

**Calibration insight:**
- FR/Hellinger calibration advantage is CONDITIONAL ON NOISE: at clean training, KL is better (ECE=0.044 vs FR=0.083, p=0.002)
- Under noise: FR achieves 9× better ECE at sym_40 (0.025 vs 0.233)
- asym_40: Hellinger maintains accuracy AND 3× better ECE (0.034 vs 0.101, p=0.002, 10/10)

**Gradient norm 3-seed results (COMPLETE):**
- ConvNet (sym_40): KL final ratio 1.49↑, FR: 0.98≈1, Hellinger: 0.88, GCE: 0.69↓, MAE: 0.65, SCE: 0.56
- MLP (Digits sym_40): FR ep99 ratio = 2.61 (same memorisation direction as KL!); MAE: 1.05 (lowest)
- Key: FR gradient saturation (ratio≈1) is ARCHITECTURE-SPECIFIC → ConvNet only

**FashionMNIST 10-seed results (COMPLETE):**
- FR sym_40: p=0.004, 1/10 wins, -1.5%; sym_60: p=0.041, 2/10 wins, -1.3%
- Hellinger asym_40: p=0.002, 0/10 wins, -1.3%; sym_80: p=0.023, 2/10 wins, -1.0%
- GCE/MAE: significantly better at sym noise; only MAE at asym_40
- CRITICAL: Always use seed-paired Wilcoxon (never sort independently before comparing)

**CIFAR-N 10-seed results (COMPLETE — updated session 7):**
- aggre (~9%): FR p=0.002, 10/10 wins (+0.7%); Hellinger neutral (5/10, p=0.475); GCE hurts p=0.008 (1/10, -0.6%); MAE collapses p=0.002; SCE hurts p=0.002
- random1 (~17%): FR p=0.002, 10/10 (+1.0%); Hellinger p=0.002, 10/10 (+0.9%); GCE neutral p=0.969 (4/10); MAE p=0.002; SCE p=0.002
- worse (~40%): FR p=0.002, 10/10 (+1.6%); Hellinger p=0.002, 10/10 (+1.4%); GCE p=0.016, 7/10 (+1.0%); MAE p=0.002; SCE neutral (4/10)

**50k benchmark (COMPLETE — 3 seeds, directional only p_min=0.250):**
- sym_20: FR +1.1% (3/3), Hellinger +1.0% (3/3), GCE +0.8% (3/3), MAE -0.8%
- sym_40: FR +2.2% (3/3), GCE +2.2% (3/3), Hellinger +1.7% (3/3)
- sym_60: GCE +3.6% (3/3), SCE +3.1%, FR +1.0% (3/3), Hellinger +0.7%; MAE -9.8% (3/3)
- asym_40: ALL non-KL objectives hurt: FR -2.4%, Hellinger -3.3%, GCE -5.3%, MAE -21.5% (3/3 each)
  → KL advantage under asym noise AMPLIFIED at 50k scale; GCE counterproductive at scale+asym

**Dynamic loss benchmark (RUNNING — PID 59612, started session 7):**
- Setup: SGD no warmup, weight_decay=1e-3 (different from main benchmark); comparisons valid only within
- 5 schedules (fr_then_gce, fr_then_kl, gce_then_fr, kl_then_gce, fr_then_mae) vs 6 static baselines
- 5 seeds × 5 regimes = 275 total runs; ~23/275 rows done (sym_20/seed0 in progress)
- PID 59612, still running (very slowly, SN priority on MPS); est. days more

**Asym_20/asym_60 benchmark (RUNNING — PID 78454, started session 8):**
- Running cifar10_noisy_label_benchmark.py --seeds 10 to get asym_20 and asym_60 results
- First result: asym_20, KL, seed=0, acc=79.0%
- 120 new rows needed (2 regimes × 6 objectives × 10 seeds); existing 300 rows skipped

**Paper status (fr_noisy_labels.tex, commits through 671be3c, 24 pages, ZERO overfull boxes):**
Session 8 improvements:
- §1 Contributions item 7: add real/instance-dependent noise guidance (GCE hurts at low noise)
- Abstract: add asym_40 result (FR -2.7%, p=0.002; Hellinger neutral p=0.625; danger zone)
- §2 Remark (universally soft): formalize danger zone definition with equation for p* ≈ 0.032
- §4 GCE Discussion bullet: fix probability gradient → logit gradient formula
- §3.3 Calibration: add explicit asym_40 calibration result (Hellinger ECE=0.034 vs KL=0.101, 3×)
- §6 Limitations: add compact 50k table (Table 10) with accuracy gain + ECE for all regimes
  - sym_40: Hellinger ECE=0.024 (12× better); sym_60: GCE ECE=0.026 (16× better)
- Figure captions: clarify GCE "universally soft AND doubly saturating" vs Hellinger "universally soft"
- Formatting: fix all 4 overfull hboxes (FR formula→display equation; table→footnotesize; etc.)

**When dynamic_loss_benchmark completes:**
1. Run generate_dynamic_loss_figures.py
2. Compute per-schedule stats vs KL and vs best static baseline
3. Update Appendix A with full results table and figures
4. Commit

**When asym_20/asym_60 results complete:**
1. Update §3.2 Table 3 with new rows for Asym 20% and Asym 60%
2. Update "When does FR help?" Discussion section
3. Update Limitations sentence from "in progress" to actual results
4. If prediction confirmed (FR hurts at asym_20/60), add as danger zone theory validation

**Next experiments (not yet started):**
1. resnet_noisy_label_benchmark.py (needs GPU ~3-5h A100 — critical for NeurIPS)

**Key practical recommendation (from paper):**
- MLP/tabular + symmetric noise: use MAE or GCE
- ConvNet + moderate sym noise (sym_20-40): FR and GCE competitive; avoid MAE
- ConvNet + high sym noise (sym_60+): prefer Hellinger (universally soft, no danger zone); GCE strongest
- ConvNet + asym noise: Hellinger (neutral accuracy, 3× better ECE); FR significantly hurts
- Low/instance-dependent noise (CIFAR-N aggre ~9%): FR+Hellinger preferred; GCE hurts
- When in doubt: Hellinger is safest ConvNet default

**Why:** Publication goal is NeurIPS 2027 (deadline ~Feb 2027, ~9 months away)
**How to apply:** Focus on completing running experiments; paper near-submission quality
CRITICAL: Always use seed-paired Wilcoxon (match by seed number), never sort independently
