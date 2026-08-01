# KALAVAI — Experiment Log

All experiments, results, and file locations. Single source of truth for what was run, what it showed, and where the data lives.

---

## Phase 1 — Main Results (English Domains)

### EXP-01: Pythia-410M Three-Domain Fusion
**Model:** EleutherAI/pythia-410m @ step10000
**Domains:** Code (code_search_net/python), Science (allenai/sciq), Fiction (pg19)
**Protocol:** 2,000 training steps, freeze=0, 3 seeds (42/137/2026), router 500 steps
**Key result:** +7.70% ±0.02pp vs best specialist (3-seed mean). Seed42 = +7.72%.
**vs base:** +16.33%
**Mean divergence:** 15.65%
**Scripts:** `experiments/kalavai_pythia_experiment.py`, `experiments/kalavai_corrected_eval.py`, `experiments/kalavai_410m_corrected_eval.py`
**Results:** `results/pythia/corrected_eval_42.json`, `corrected_eval_137.json`, `corrected_eval_2026.json`, `corrected_eval_410m_summary.json`

---

### EXP-02: Pythia-1B Three-Domain Fusion
**Model:** EleutherAI/pythia-1b @ step10000
**Domains:** Code / Science / Fiction
**Protocol:** 2,000 steps, freeze=0, 3 seeds (42/137/2026), router 500 steps
**Key result:** +7.49% ±0.01pp vs best specialist (3-seed mean)
**vs base:** +15.5%
**Mean divergence:** 15.28%
**Scripts:** `experiments/kalavai_pythia_1b_experiment.py`
**Results:** `results/pythia/pythia_1b/corrected_eval_42.json`, `corrected_eval_137.json`, `corrected_eval_2026.json`, `corrected_eval_1b_summary.json`

---

### EXP-03: Pythia-6.9B Three-Domain Fusion
**Model:** EleutherAI/pythia-6.9b @ step10000
**Domains:** Code / Science / Fiction
**Protocol:** 2,000 steps, freeze=0 (corrected), 3 seeds, router 500 steps
**Key result:** +6.53% ±0.024pp vs best specialist (3-seed mean)
**vs base:** +8.6%
**Mean divergence:** 8.73%
**Scripts:** `experiments/kalavai_pythia_6b_experiment.py`, `experiments/kalavai_6b_freeze0.py`
**Results:** `results/pythia_6b/step6_fusion_seed42.json`, `step6_fusion_seed137.json`, `step6_fusion_seed2026.json`, `corrected_eval_6b_summary.json`

---

### EXP-04: Qwen-2.5-1.5B Three-Domain Fusion
**Model:** Qwen/Qwen2.5-1.5B @ step143000 (full training)
**Domains:** Code / Fiction (2 domains only; below divergence floor)
**Protocol:** 2,000 steps, freeze=4, 3 seeds, router 500 steps
**Key result:** +1.06% ±0.01pp vs best specialist (3-seed mean)
**Mean divergence:** 3.16% (near divergence floor ~3.3%)
**Note:** Near-zero gain, confirms divergence floor prediction.
**Scripts:** `experiments/kalavai_qwen_experiment.py`
**Results:** `results/real/corrected_eval_qwen_42.json`, `corrected_eval_qwen_137.json`, `corrected_eval_qwen_2026.json`, `corrected_eval_qwen_summary.json`

---

## Phase 1 — Controls & Ablations

### EXP-05: Monolithic Baseline (Equal-Compute)
**Model:** Pythia-410M
**Setup:** Single model trained for 6,000 steps (= 3 specialists × 2,000 steps) on mixed data
**Key result:** MoE (2.218 EW loss) beats monolithic (2.229 EW loss) by +0.47%. MoE wins per-domain.
**Interpretation:** Gain is not from extra compute; mechanism is cooperative specialisation.
**Scripts:** `experiments/kalavai_pythia_monolithic_baseline.py`
**Results:** `results/pythia/monolithic_baseline_seed42.json`, `monolithic_baseline_seed137.json`, `monolithic_baseline_seed2026.json`, `monolithic_baseline_summary.json`

---

### EXP-06: Domain Classifier Baseline (Single-Expert Dispatch)
**Model:** Pythia-410M
**Setup:** 99.3%-accurate domain classifier routes each sample to a single specialist; no MoE combination
**Key result:** −21.1% vs best specialist (catastrophic failure)
**Interpretation:** Specialists forget out-of-domain. Running all experts and combining per-token is essential.
**Scripts:** `experiments/kalavai_domain_classifier_baseline.py`
**Results:** `results/pythia/domain_classifier_baseline.json`

---

### EXP-07: Wider Model Baseline (Parameter-Count Control)
**Model:** Pythia-1.4B (3.5× the parameters of 410M specialists)
**Setup:** Single 1.4B model trained for 6,000 steps on mixed data
**Key result:** +5.9% vs 410M base. MoE with 3×410M gets +7.72%. More parameters ≠ better.
**Scripts:** `experiments/kalavai_wider_model_baseline.py`
**Results:** `results/pythia/wider_model_baseline.json`

---

### EXP-08: Multi-Head Baseline
**Model:** Pythia-410M
**Setup:** Single model with N domain-specific output heads, same parameter count as MoE
**Key result:** −21.1% (identical failure mode to single-expert dispatch)
**Scripts:** `experiments/kalavai_multihead_baseline.py`
**Results:** `results/pythia/multihead_baseline.json`

---

### EXP-09: Freeze Depth Ablation (0–12 Layers)
**Model:** Pythia-410M
**Setup:** Sweep freeze layers 0, 2, 4, 6, 8, 12 at 2,000 steps; 3 seeds
**Key result:** freeze=0 peaks at 2,000 steps (+8.12%). Crossover at ~5,000 steps — beyond that freeze=4 overtakes.
**Best at 2k steps:** freeze=0. Best at 10k+ steps: freeze=4.
**Scripts:** `experiments/kalavai_pythia_ablation_freeze.py`
**Results:** `results/pythia/ablation_freeze_summary.json`

---

### EXP-10: Router Architecture Ablation
**Model:** Pythia-410M
**Setup:** Compare linear router (1 layer) vs 2-layer MLP router; 500 steps
**Key result:** Linear = MLP — 2-layer adds no benefit. Linear router is sufficient.
**Scripts:** `experiments/kalavai_pythia_ablation_router.py`
**Results:** `results/pythia/ablation_router_summary.json`

---

### EXP-11: Training Duration Crossover (50–20,000 Steps)
**Model:** Pythia-410M
**Setup:** Train specialists for 50 / 100 / 500 / 1k / 2k / 5k / 10k / 20k steps; evaluate MoE vs best specialist
**Key result (corrected eval):**

| Steps | freeze=0 gain | freeze=4 gain | Winner |
|-------|--------------|--------------|--------|
| 50    | +4.0%        | +3.9%        | freeze=0 |
| 100   | +5.1%        | +4.8%        | freeze=0 |
| 500   | +5.88%       | +5.31%       | freeze=0 |
| 1000  | +5.94%       | +6.48%       | freeze=4 |
| 2000  | +8.12%       | +7.56%       | freeze=0 ← peak |
| 5000  | +7.79%       | +8.07%       | freeze=4 |
| 10000 | +5.83%       | +7.33%       | freeze=4 ← crossover |
| 20000 | +3.38%       | +6.30%       | freeze=4 |

**Crossover point:** ~10,000 steps
**Divergence floor:** ~50 steps → +4.0% (gain persists even at very short training)
**Scripts:** `experiments/kalavai_training_duration_crossover.py`
**Results:** `results/pythia/training_duration_crossover_corrected.json`, `results/pythia/crossover_regression_points.json`

---

### EXP-12: Five-Domain Specialist Scaling (2→5 Specialists)
**Model:** Pythia-410M
**Domains:** Code, Science, Fiction, Math (GSM8K data), Multilingual (Spanish Wikipedia)
**Setup:** Evaluate fusion as specialist count increases from 2 to 5; 3 seeds
**Key result:**

| Specialists | Domains | Gain vs best spec | Std |
|-------------|---------|-------------------|-----|
| 2 | Code, Fiction | +1.76% | ±0.007% |
| 3 | Code, Science, Fiction | +4.39% | ±0.024% |
| 4 | + Math | +11.39% | ±0.022% |
| 5 | + Multilingual | +12.95% | ±0.022% |

**Scripts:** `experiments/kalavai_pythia_5domain_experiment.py`
**Results:** `results/pythia/five_domain/five_domain_seed42.json`, `five_domain_seed137.json`, `five_domain_seed2026.json`, `five_domain/summary.json`

---

### EXP-13: Base Checkpoint Maturity Sweep (410M)
**Model:** Pythia-410M
**Setup:** Train specialists from different base checkpoints (step5k, step10k, step20k, step50k, step100k, step143k); compare fusion gain
**Key result:** step10000 is optimal starting point; very early checkpoints (step5k) have too little pre-training; late checkpoints (step143k) have too much — specialists under-diverge.
**Scripts:** `experiments/kalavai_pythia_maturity_sweep.py`
**Results:** `results/pythia/maturity_sweep_410m/summary.json`, `checkpoint_step{N}_seed42.json` (per step)

---

### EXP-14: Base Checkpoint Maturity Sweep (1B)
**Model:** Pythia-1B
**Setup:** Same maturity sweep as EXP-13 at 1B scale
**Scripts:** `experiments/kalavai_pythia_1b_maturity_sweep.py`
**Results:** `results/pythia/pythia_1b/maturity_sweep/summary.json`, `result_step{N}_seed42.json` (per step)

---

### EXP-15: Shared Init Ablation (Checkpoint Mismatch Effects)
**Model:** Pythia-410M
**Setup:** Three conditions — (1) control: all from same checkpoint, (2) large_gap: specialists from checkpoints 10k steps apart, (3) small_gap: 1k steps apart
**Key result:**
- Control: +10.37%
- Large gap: +9.44% (−0.93pp)
- Small gap: +9.54% (−0.83pp)
- **Conclusion:** Shared init may not be strictly critical, but minor degradation from mismatched checkpoints is real.
**Scripts:** `experiments/kalavai_shared_init_ablation.py`
**Results:** `results/pythia/shared_init_ablation/summary.json`, `result_control_seed{N}.json`, `result_large_gap_seed{N}.json`, `result_small_gap_seed42.json`

---

### EXP-16: Heterogeneous Cooperative (Robustness to Contributor Variation)
**Model:** Pythia-410M
**Setup:** Specialists trained with different hyperparameters (different LR, batch size, step count) to simulate real-world contributor variation
**Key result:** Fusion remains robust. Max performance spread across heterogeneous conditions: 0.41pp. Cooperative does not require homogeneous training.
**Scripts:** `experiments/kalavai_heterogeneous_cooperative.py`
**Results:** `results/pythia/v2/heterogeneous_v2.json`

---

## Phase 2 — High-Divergence Domains

### EXP-17: Cross-Lingual Fusion (Tamil / Yoruba / Welsh / Code)
**Model:** Pythia-410M @ step10000
**Domains:** Tamil (Wikipedia), Yoruba (Wikipedia), Welsh (Wikipedia), Code (code_search_net)
**Protocol:** 2,000 steps, freeze=0, seeds 42/137/2026, router 500 steps
**Key result (original, seeds 137/2026 only):** +21.76% ±0.005pp vs best specialist
**Seed 42 note (original):** Router collapse — Yoruba routes 99.84% to Tamil expert. Fixed by curriculum warm-start (see EXP-17b).

### EXP-17b: Cross-Lingual Fusion — Curriculum Warm-Start (Router Collapse Fix)
**Status:** COMPLETE — all 3 seeds GO
**Fix:** Stage A: 100 domain-pure router steps (round-robin, one language per step) → Stage B: 400 mixed steps
**Key result:** **+21.87% ±0.12pp** (seed42: +22.04%, seed137: +21.79%, seed2026: +21.77%) — 3-seed mean
**Mean divergence:** ~25.66% across seeds
**Perplexity highlights:** Yoruba 41.9 → 7.7 (5.4×), Welsh 102.7 → 22.1 (4.6×)
**Router collapse:** Eliminated — all seeds route correctly
**Scripts:** `experiments/kalavai_phase2_exp1_curriculum.py`
**Results:** `results/phase2/cross_lingual/curriculum/result_seed42.json`, `result_seed137.json`, `result_seed2026.json`
**Paper status:** Paper updated 2026-04-08 — all cross-lingual occurrences now show +21.87% ±0.12pp (3 seeds, curriculum). LOO analysis updated to use 21.87% primary value.

---

### EXP-18: Private-Domain Fusion (Medical / Legal / Patent)
**Model:** Pythia-410M @ step10000
**Domains:** Medical (PubMed abstracts), Legal (case law), Patent (USPTO abstracts)
**Protocol:** 2,000 steps, freeze=0, 3 seeds (42/137/2026), router 500 steps
**Key result:** +10.17% ±0.15pp vs best specialist (3-seed mean)
**Mean divergence:** 18.52% (medical 12.71%, legal 34.16%, patent 8.68%)
**vs monolithic:** +1.78% (seed 42)
**Scripts:** `experiments/kalavai_phase2_exp2.py`
**Results:** `results/phase2/private_domain/result_seed42.json`, `result_seed137.json`, `result_seed2026.json`, `summary.json`

---

### EXP-19: 20-Contributor Federation
**Model:** Pythia-1B @ step10000
**Domains:** 10 languages (Tamil, Yoruba, Welsh, Spanish, Hindi, Swahili, Vietnamese, Arabic, Indonesian, Thai) + 10 domains (code, medical, legal, patent, math, finance, chemistry, fiction, dialogue, instructions)
**Protocol:** 2,000 steps, freeze=0, router 1,000 steps (not 500 — different from Phase 1), seeds 42/137/2026
**Key result (prior run):** +16.71% ±0.07pp vs best specialist (3-seed mean)
**vs base:** +17.09%
**Mean divergence:** 15.68% (excludes dialogue/instructions which have negative divergence)
**Negative-divergence specialists:** dialogue (−24.9%), instructions (−16.3%) — router correctly down-weights both (<2%)
**Per-seed:** seed42 +16.79%, seed137 +16.65%, seed2026 +16.68%
**Scripts:** `experiments/kalavai_20contributor_experiment.py`
**Results (prior):** `results/phase2/twenty_contributor/result_seed42.json`, `result_seed137.json`, `result_seed2026.json`

**Status:** COMPLETE (2026-04-08)
- Final 3-seed mean: +16.71% ±0.07pp (seed42: +16.79%, seed137: +16.65%, seed2026: +16.68%)
- Results: `results/phase2/twenty_contributor/result_seed{42,137,2026}_router_retry.json`
- Note: Two toxic specialists (dialogue, instructions) confirmed — see FE-03 below

---

### FE-03: 18-Contributor Ablation (Specialist Quality Filtering)
**Model:** Pythia-1B @ step10000
**Domains:** Same as EXP-19 minus dialogue and instructions (18 of 20 specialists)
**Protocol:** Router retrained 1,000 steps on 18-domain mixed data. Specialist checkpoints reused from EXP-19 (seed 42 checkpoints for all 3 router seeds via --specialist-seed flag).
**Key result:** **+21.13% ±0.01pp** vs best specialist (3-seed mean)
**vs base:** +21.56%
**Mean divergence:** 19.75% (rises from 15.68% once negative-divergence specialists removed)
**Per-seed:** seed42 +21.14%, seed137 +21.12%, seed2026 +21.13%
**Gain over EXP-19:** +4.42pp (+16.71% → +21.13%)
**Scripts:** `experiments/kalavai_18contributor_experiment.py`
**Results:** `results/phase2/eighteen_contributor/result_seed42_router_retry.json`, `result_seed137_router_retry.json`, `result_seed2026_router_retry.json`
**Paper status:** Paper updated 2026-04-08 — abstract, negative-divergence paragraph, new Table FE-03, OOS row in divergence-gain table updated to FE-03 numbers, summary table row added.

---

## Analysis Experiments

### EXP-20: Divergence–Gain Regression (n=8 in-sample, updated 2026-04-14)
**Purpose:** Fit predictive formula relating mean specialist divergence to fusion gain
**In-sample conditions:** Qwen-1.5B, Pythia-6.9B, P1 2-domain, Pythia-1B, Pythia-410M, Private-domain, P2 4-domain, Cross-lingual (8 points)
**Formula:** `gain = 0.83 × divergence − 2.80` (R² = 0.872, n=8)
**Full precision:** slope=0.8335, intercept=−2.7956, R²=0.8723
**Divergence floor:** ~3.4% (below this → near-zero gain)
**Scripts:** `experiments/analysis/item1_regression_fit.py`
**Results:** `results/analysis/regression_fit.json`
**Previous (n=6):** slope=0.8170, intercept=−2.7237, R²=0.8565

---

### REG-P1: Regression Extension P1 — 2-Domain (Code + Science) [2026-04-14]
**Purpose:** Add (div=10.77%, gain=+6.22%) data point to fill Gap 1 in divergence-gain regression (8.73%–15.28%)
**Setup:** Pythia-410M @ step10000, 2 specialists (code + science), 2000 steps, 3 seeds [42, 137, 2026]
**Result:** mean div=10.77%, gain=+6.22% ±0.01pp — regression prediction was +6.18% (residual +0.04pp, on the line)
**Scripts:** `experiments/kalavai_regression_p1_2domain.py`
**Results:** `results/regression_extension/p1_2domain_summary.json`

---

### REG-P2: Regression Extension P2 — 4-Domain (Code + Science + Fiction + Legal) [2026-04-14]
**Purpose:** Add (div=19.84%, gain=+14.71%) data point to fill Gap 2 in divergence-gain regression (18.52%–25.65%)
**Setup:** Pythia-410M @ step10000, 4 specialists (code + science + fiction + legal/eurlex), 2000 steps, 3 seeds [42, 137, 2026]
**Result:** mean div=19.84%, gain=+14.71% ±0.02pp — regression prediction was +13.74% (residual +0.97pp, within band)
**Scripts:** `experiments/kalavai_regression_p2_4domain.py`
**Results:** `results/regression_extension/p2_4domain_summary.json`

---

### EXP-21: Expanded Regression Scatter (n=17)
**Purpose:** Add crossover (8 points) and 20-contributor (3 seeds → 1 mean point) to regression scatter
**Key finding:** R²=0.05 with n=17. Crossover points show regime-dependent, non-monotonic behaviour — high-step conditions have divergence returning toward 0 while gain stays positive. Primary n=6 regression remains valid within fixed training protocol.
**Scripts:** `experiments/kalavai_regression_scatter_v2.py`
**Results:** `results/analysis/regression_scatter_v2.json`, `paper/figures/fig_divergence_gain_scatter_v2.png`

---

### EXP-22: Crossover Regression Points Extraction
**Purpose:** Extract (divergence, gain) pairs from training-duration crossover checkpoints for use in EXP-21
**Setup:** Load specialist checkpoints from EXP-11 at each step count, evaluate on all 3 domains, compute EW gain
**Scripts:** `experiments/kalavai_crossover_regression_points.py`
**Results:** `results/pythia/crossover_regression_points.json` (8 points, steps 50–20k)

---

### EXP-23: Router Collapse Analysis (Cross-Lingual Seed 42)
**Purpose:** Diagnose why seed 42 failed in EXP-17
**Finding:** Seed 42 — Yoruba routes 99.84% gate weight to Tamil expert (entropy ≈ 0). Tamil routes correctly. Welsh routes correctly. Code routes correctly. Seeds 137/2026 route all 4 languages correctly at >99.98%.
**Root cause:** Multiple stable minima in router gradient landscape at high divergence (25%+). Stochastic init determines which minimum is reached.
**Scripts:** `experiments/kalavai_router_collapse_analysis.py`
**Results:** `results/phase2/cross_lingual/collapse_analysis.json`

---

### EXP-24: LoRA Ablation (r=8, r=16, r=32, r=64)
**Model:** Pythia-410M
**Purpose:** Determine whether LoRA fine-tuning produces sufficient divergence for cooperative fusion
**Key result:**

| Method | Mean Divergence | Gain vs Spec |
|--------|----------------|-------------|
| Full FT | +15.65% | +7.72% |
| LoRA r=8 | −1.48% | +0.32% |
| LoRA r=16 | −5.57% | −2.65% |
| LoRA r=32 | −12.05% | −7.73% |
| LoRA r=64 | −20.31% | −13.85% |

**Conclusion:** All LoRA ranks produce negative or near-zero divergence. Full FT is required for KALAVAI.
**Scripts:** `experiments/kalavai_ablations_v2.py` (r=8, r=64), `experiments/kalavai_lora_figure.py` (figure)
**Results:** `results/analysis/lora_r8/result_seed42.json`, `lora_r16/result_seed42.json`, `lora_r32/result_seed42.json`, `lora_r64/result_seed42.json`, `paper/figures/fig_lora_ablation.png`

---

### EXP-25: Base-PPL Conversion Rate Analysis
**Purpose:** Explain why cross-lingual (EXP-17) exceeds the linear prediction
**Setup:** Correlate base model perplexity on each domain with (divergence → gain) conversion efficiency
**Key result:** r=+0.560 (log base-PPL) / r=+0.614 (divergence-based), n=6 — suggestive that high base-PPL domains (cross-lingual) generate more gain per unit divergence. Integrated into §4.10.
**Scripts:** (inline analysis)
**Results:** `results/analysis/baseppl_conversion.json`

---

### EXP-26: Oracle Routing Analysis
**Purpose:** Measure gap between learned router and oracle (domain-level ground truth)
**Key result (410M):** Gap < 10⁻⁵ nats — routing is saturated. Linear router matches oracle.
**Scripts:** `experiments/kalavai_hard_routing_verification.py`
**Results:** `results/analysis/oracle_routing.json`

---

## Downstream Benchmarks

### EXP-27: Downstream Benchmarks — Pythia-410M
**Benchmarks:** HellaSwag, ARC-Easy, PIQA (broken — excluded), WinoGrande, LAMBADA, SciQ
**Setup:** 2,000 examples per benchmark, seed=42, manual log-likelihood scoring
**Key result:**

| Model | HellaSwag | ARC-Easy | LAMBADA | SciQ | WinoGrande | Avg |
|-------|-----------|----------|---------|------|------------|-----|
| Base | 33.0% | 40.5% | 39.3% | 71.2% | 50.7% | 47.0% |
| MoE fused | 33.1% | 39.8% | 40.1% | 71.3% | 50.7% | 47.0% |
| Monolithic | 32.6% | 40.0% | 39.5% | 71.1% | 50.8% | 46.8% |

**Note:** Near-parity; task accuracy is not the primary metric at 410M scale.
**Scripts:** `experiments/kalavai_pythia_benchmarks.py`
**Results:** `results/pythia/benchmarks_seed42_v2.json`

---

### EXP-28: Downstream Benchmarks — Pythia-1B
**Benchmarks:** HellaSwag, ARC-Easy, LAMBADA, SciQ, WinoGrande
**Setup:** 2,000 examples per benchmark, seed=42
**Key result:**

| Model | HellaSwag | ARC-Easy | LAMBADA | SciQ | WinoGrande | Avg |
|-------|-----------|----------|---------|------|------------|-----|
| Base | 32.45% | 40.88% | 58.75% | 66.7% | 51.22% | 50.0% |
| MoE fused | 32.15% | 40.53% | 57.85% | 64.7% | 51.93% | 49.4% |
| Monolithic | 32.1% | 38.95% | 57.9% | 67.2% | 51.46% | 49.5% |

**Scripts:** `experiments/kalavai_pythia_1b_benchmarks.py`
**Results:** `results/pythia/pythia_1b/benchmarks_seed42_v2.json`

---

### EXP-29: Downstream Benchmarks — Pythia-6.9B
**Benchmarks:** HellaSwag, ARC-Easy, LAMBADA, SciQ, WinoGrande
**Setup:** 500 examples per benchmark, seed=42, base + MoE only (no individual specialists — compute constraint)
**Key result:** MoE avg 52.2% vs base avg 51.6% (+0.56pp). MoE leads on 4 of 5 benchmarks.
**Scripts:** `experiments/kalavai_pythia_6b_experiment.py`
**Results:** `results/pythia_6b/benchmarks_seed42.json`

---

### EXP-30: GSM8K Evaluation — Math Specialist (Pythia-410M)
**Purpose:** Test whether math specialist from EXP-12 gains reasoning capability
**Setup:** 4-shot chain-of-thought, 500 GSM8K test examples, seed=42
**Key result:**

| Model | GSM8K Accuracy | Correct | No Answer |
|-------|---------------|---------|-----------|
| Base | 1.60% | 8/500 | 28 |
| Math specialist | 1.20% | 6/500 | 6 |
| Δ | −0.40pp | | |

**Conclusion:** Near-chance. Expected — Pythia-410M has not developed multi-step arithmetic reasoning and 2,000 fine-tuning steps cannot instil it. Math specialist value is measured by perplexity improvement on math text (−25.9%), not downstream reasoning accuracy.
**Scripts:** `experiments/kalavai_math_benchmark.py`
**Results:** `results/pythia/five_domain/gsm8k_benchmark.json`

---

### EXP-31: Inference Benchmark (Latency & VRAM)
**Device:** NVIDIA GeForce RTX 5090
**Setup:** 10 measurement runs, warmup 3, seq_len=512
**Key result:** Dense MoE latency = 2.86× base (410M), 3.35× base (1B). VRAM scales linearly with N specialists. Frozen layers run once — effective overhead ≈ 2.5× for freeze=4 configuration.
**Scripts:** `experiments/kalavai_inference_benchmark.py`
**Results:** `results/pythia/inference_benchmark.json`

---

### EXP-32: Leave-One-Out (LOO) Regression Validation
**Purpose:** Validate the "predictive heuristic" claim in the paper — does the divergence-gain formula generalise?
**Setup:** 6-point regression (Qwen, 6.9B, 1B, 410M, Private, Cross-lingual). LOO cross-validation: fit on 5, predict left-out.
**Key results (updated 2026-04-08, primary = +21.87% 3-seed curriculum):**
- LOO-MAE (all 6): 3.82pp
- LOO-MAE (excl. cross-lingual, 5 points): 2.89pp
- LOO-MAE (sensitivity: pre-curriculum +16.55%): 1.62pp
- Cross-lingual LOO residual: +8.43pp (largest outlier — base-model incompetence amplifies gain beyond linear prediction)
**Scripts:** `experiments/analysis/loo_analysis.py`
**Results:** `results/analysis/loo_analysis.json`

---

## Planned Experiments (Not Yet Run)

### PLAN-01: Low-Divergence Ablation Paper Integration
**Purpose:** Document the divergence floor formally — at what training step count does gain go to zero?
**Status:** Data already exists in EXP-11 (50-step checkpoint → +4.0% gain). Requires paper write-up only.
**What to do:** Extract the floor point from `crossover_regression_points.json`, add a paragraph to §4 with the floor derivation and the intersection of the regression line with gain=0 (at divergence ~3.3%).
**Can run on 5090:** N/A (writing-only)

---

### PLAN-01: 20-Contributor with Robust Data
**Purpose:** Rerun EXP-19 replacing the two negative-divergence specialists (dialogue, instructions) with specialists that have clear domain data
**Replacement domains:** Suggested alternatives — legal (more data), biomedical (more data), or a second code language (Rust, JavaScript)
**Expected result:** Higher mean divergence → higher gain; eliminates the negative-divergence footnote in the paper
**Can run on 5090:** YES (~2h on Pythia-1B)
**Results would go to:** `results/phase2/twenty_contributor_robust/`

---

### PLAN-02: Multi-Round Contributors (Thicker Specialists)
**Purpose:** Simulate a realistic cooperative where each contributor does multiple training rounds (3 rounds × shorter sessions) rather than one long run
**Setup:** 3 contributors each train for 3 × 700 steps with data refresh between rounds. Compare to single 2,000-step specialist.
**Hypothesis:** Multi-round training produces richer divergence while avoiding catastrophic forgetting
**Can run on 5090:** YES (~2-4h at 410M)
**Results would go to:** `results/analysis/multi_round/`

---

### PLAN-03: Continual Cooperative (Post-Hoc Specialist Addition)
**Purpose:** Test whether a 4th specialist can join an existing 3-specialist cooperative without retraining the original three
**Setup:** Start with EXP-01 (code/science/fiction MoE). Train a new medical specialist from the same base. Train a new router on all 4 specialists. Compare to full 4-specialist retraining.
**Hypothesis:** Router retraining alone (~500 steps) is sufficient to integrate a new specialist
**Can run on 5090:** YES (~1h at 410M)
**Results would go to:** `results/analysis/continual_cooperative/`

---

## Summary Table

| ID | Experiment | Model | Gain | Seeds | Status |
|----|-----------|-------|------|-------|--------|
| EXP-01 | Pythia-410M 3-domain | 410M | +7.70% ±0.02pp | 3 | Done |
| EXP-02 | Pythia-1B 3-domain | 1B | +7.49% ±0.01pp | 3 | Done |
| EXP-03 | Pythia-6.9B 3-domain | 6.9B | +6.53% ±0.024pp | 3 | Done |
| EXP-04 | Qwen-2.5-1.5B | 1.5B | +1.06% ±0.01pp | 3 | Done |
| EXP-05 | Monolithic baseline | 410M | MoE +0.47% vs mono | 3 | Done |
| EXP-06 | Domain classifier (single-expert) | 410M | −21.1% | 1 | Done |
| EXP-07 | Wider model (3.5× params) | 1.4B | +5.9% | 1 | Done |
| EXP-08 | Multi-head baseline | 410M | −21.1% | 1 | Done |
| EXP-09 | Freeze depth sweep (0–12 layers) | 410M | crossover at ~5k steps | 3 | Done |
| EXP-10 | Router architecture (linear vs MLP) | 410M | identical | 1 | Done |
| EXP-11 | Training duration crossover (50–20k steps) | 410M | crossover at ~10k steps | 1 | Done |
| EXP-12 | 5-domain scaling (2→5 specialists) | 410M | +1.76% to +12.95% | 3 | Done |
| EXP-13 | Maturity sweep — 410M | 410M | step10k optimal | 1–3 | Done |
| EXP-14 | Maturity sweep — 1B | 1B | step10k optimal | 1–3 | Done |
| EXP-15 | Shared init ablation | 410M | −0.93pp for large gap | 3 | Done |
| EXP-16 | Heterogeneous cooperative | 410M | robust (±0.41pp spread) | 1 | Done |
| EXP-17 | Cross-lingual (Tamil/Yoruba/Welsh/Code) | 410M | +21.87% ±0.12pp (curriculum) | 3 | Done |
| EXP-18 | Private-domain (Medical/Legal/Patent) | 410M | +10.17% ±0.15pp | 3 | Done |
| EXP-19 | 20-contributor federation | 1B | +16.71% ±0.07pp | 3 | Done |
| EXP-20 | Divergence–gain regression (n=6) | — | R²=0.857 | — | Done |
| EXP-21 | Expanded regression scatter (n=17) | — | R²=0.05 (regime-dependent) | — | Done |
| EXP-22 | Crossover regression points | 410M | 8 points extracted | — | Done |
| EXP-23 | Router collapse analysis (seed 42) | 410M | Yoruba→Tamil collapse | — | Done |
| EXP-24 | LoRA ablation (r=8/16/32/64) | 410M | all ranks fail | 1 | Done |
| EXP-25 | Base-PPL conversion rate | — | r=+0.560 | — | Done |
| EXP-26 | Oracle routing analysis | 410M | gap < 10⁻⁵ nats | 1 | Done |
| EXP-27 | Benchmarks — 410M | 410M | near-parity | 1 | Done |
| EXP-28 | Benchmarks — 1B | 1B | MoE +0.6pp avg | 1 | Done |
| EXP-29 | Benchmarks — 6.9B | 6.9B | MoE +0.56pp avg | 1 | Done |
| EXP-30 | GSM8K — math specialist | 410M | 1.2% (near-chance) | 1 | Done |
| EXP-31 | Inference benchmark | 410M/1B | 2.86×/3.35× latency | 1 | Done |
| EXP-32 | Low-divergence floor write-up | — | floor=3.3%, regime-scoped | — | Done |
| PLAN-01 | 20-contributor robust data | 1B | — | 3 | Planned |
| PLAN-02 | Multi-round contributors | 410M | — | 3 | Planned |
| PLAN-03 | Continual cooperative | 410M | — | 3 | Planned |

---

## Rebuttal — NeurIPS 2026 (Discussion window Jul 27 – Aug 3 2026)

Reviews reject-leaning (3/2/2). AC conditions to reconsider: (1) a technical delta vs Branch-Train-Mix, (2) downstream task accuracy beyond loss. Plan: `docs/planning/2026-07-24-rebuttal-experiment-plan.md`. Workstreams W1–W6.

| ID | Experiment | Model | Result | Status |
|----|-----------|-------|--------|--------|
| REB-W1 | Cross-lingual downstream eval | 410M | MoE beats best single specialist on avg cloze: seed137 +11.1%, seed2026 +11.0%, seed42 +18.4%. Claim robust across 3 seeds | 3 seeds Done |
| REB-W2 | Grow divergence–gain predictor + held-out validation | 410M | **HONEST NEGATIVE**: 6 new cross-lingual points, only 2/6 in 95% band (RMSE 12.3pp); predictor OVER-predicts for asymmetric subsets → confirms it is regime-scoped, not universal | Done |
| REB-W3 | Compute-matched centralized (monolithic) vs MoE, cross-lingual | 410M | **NEGATIVE:** monolithic (2.016) BEATS MoE (2.278) by **−13%** even cross-lingual (equal compute 8000 steps). Cooperative loses to centralized; advantage is only vs best *individual* specialist. Confirms vacq. | Done |
| REB-W4 | Router-privacy (public-proxy calibration data) | 410M | **POSITIVE (clean):** router trained on public FLORES data gives identical gain (+16.30%) to in-domain (+16.35%). Calibration set needs NO private data → privacy model sound. | Done |
| REB-W5 | BTX head-to-head under no-communication constraint | 410M | KALAVAI +22.7% vs BTX +23.7% gain-vs-base (~1pp gap); KALAVAI recovers ~96% of BTX quality with NO expert training / data pooling. Robust across 2 seeds (137: +22.68/+23.72; 2026: +22.70/+23.74) | 2 seeds Done |
| REB-W6 | Cross-lingual seed-42 router-collapse fix | 410M | **FIXED** by router re-init (try 1: all 4 langs route correctly, no Yoruba→Tamil collapse) | Done |
| REB-W7 | 7B cross-lingual real-task-win gamble | Qwen2.5-7B | **MIXED (honest):** fusion gain scales to 7B (**+10.8%** eq-loss; cloze Yoruba 0.454→0.683, ppl 13.3→5.1) BUT the real-task win **FAILED** — Belebele MoE ≤ base (Tamil 0.36→0.32, Yoruba 0.22→0.18). Raw-text FT erodes the base's QA ability. | Done |

### REB-W1: Cross-lingual DOWNSTREAM evaluation
**Purpose:** answer AC condition #2 — downstream task *accuracy* (not loss) in the high-divergence cross-lingual regime, where the base model is least competent.
**Model:** Pythia-410M @ step10000; cross-lingual specialists (Tamil/Yoruba/Welsh/Code) loaded from HF (`mechramc/kalavai-cross-lingual-*-specialist-seed{137,2026}`).
**Metrics:** cloze next-token accuracy (primary, robust), FLORES-200 perplexity (n=100), Belebele multilingual MC-QA (n=100), self-continuation chrF (n=30). *No MT BLEU — specialists are monolingual LMs, not translators.*
**Script:** `experiments/w1_crosslingual_downstream.py` · **Node bootstrap:** scratchpad `run_w1_node.sh` · **Compute:** Lambda A10, ~1h/seed.
**Results:** `results/rebuttal/w1_crosslingual/w1_downstream_seed{137,2026}.json` · **Figure:** `figures/rebuttal/w1_crosslingual_downstream.png`
**Seed 137 (Done):** cloze — Tamil 0.559→0.605, Yoruba 0.410→0.550, Welsh 0.257→0.371, Code 0.579→0.586 (base→MoE); MoE ≈ best specialist on every language. FLORES ppl: Yoruba 153→21 (7.4×), Welsh 469→52 (9.1×), Tamil 4.4→3.2. Belebele near-chance (~0.25–0.30) — 410M too weak for MC-QA; **reported honestly, not spun.**
**Seed 2026 (Done):** cloze MoE — Tamil 0.604, Yoruba 0.550, Welsh 0.372, Code 0.586. **Agrees with seed 137 to ±0.001** (matches the paper's ±0.005pp cross-lingual reproducibility). 2-seed result is rock-solid.
**Seed 42 (Done, via REB-W6):** retrained fresh (not on HF) with router re-init → **collapse FIXED on try 1** (all 4 languages route to own expert; Yoruba downstream +0.268 over base, correctly routed). Claim holds: **MoE beats best single specialist by +18.4%** (avg cloze).
**3-seed claim summary — MoE avg vs best single specialist:** seed137 **+11.1%**, seed2026 **+11.0%**, seed42 **+18.4%**. Robust across seeds.
**⚠️ HONEST FLAG (investigate / disclose):** seed-42's freshly-trained specialists are *notably stronger* than the HF seed-137/2026 checkpoints (Yoruba spec cloze 0.678 vs 0.550; Welsh 0.443 vs 0.371) despite identical protocol (2000 steps). Likely the HF 137/2026 cross-lingual checkpoints were undertrained relative to a clean 2000-step run, OR high seed variance. **Implication:** don't naively average seed-42 (stronger) with 137/2026 in one absolute-number figure — report the *claim* (MoE>best-single-spec) per seed, and keep the primary figure on the two consistent HF seeds. If time permits, retrain 137/2026 fresh for a clean 3-seed. Primary W1 figure currently uses seeds 137+2026 only.

### REB-W6: Seed-42 router-collapse fix
**Purpose:** resolve the paper's documented seed-42 Yoruba→Tamil router collapse; supply the 3rd W1 seed.
**Method:** retrain 4 cross-lingual specialists (seed 42), then train the MoE router with **re-initialization retry** — if any language fails to route to its own expert on held-out, re-init the router (new seed) and retrain, up to 5 tries.
**Result:** collapse resolved on **try 1** (routing_ok=True for all 4 languages). The collapse was router-init sensitivity, not a fundamental limitation.
**Script:** `experiments/w6_seed42_train_and_downstream.py` · **Result:** `results/rebuttal/w1_crosslingual/w1_downstream_seed42.json` · **Compute:** A100, ~32 min.

### REB-W5: BTX head-to-head (technical delta from Branch-Train-Mix)
**Purpose:** answer AC condition #1 — a genuine technical distinction from BTX. Isolates the axis: does fusion require jointly training experts on POOLED data (BTX), or does post-hoc routing over FROZEN experts (KALAVAI) recover it?
**Method (model-level MoE, framed honestly as "BTX-style joint fine-tuning", NOT layer-level BTX):** same cross-lingual specialists (seed 137, HF); three variants — uniform router (no training), KALAVAI (frozen experts + trained router), BTX-style (experts UNFROZEN + jointly fine-tuned 500 steps on pooled mixed data + router). Equal-weight loss; "what it requires" table.
**Result (seed 137):** gain vs base — uniform +10.6%, **KALAVAI +22.7%**, BTX +23.7%. KALAVAI is within **~1pp** of BTX while requiring **no expert training and no data pooling** (only a small router calibration set). BTX requires pooled mixed data + joint gradient training — exactly what the no-communication cooperative constraint forbids.
**Takeaway:** the technical delta is real — post-hoc routing recovers ~96% of BTX's fused quality under a constraint BTX cannot satisfy. (Smoke artifact note: at 30 BTX steps KALAVAI *beat* BTX because joint fine-tuning on pooled data transiently erodes specialization; full 500-step BTX recovers.)
**Script:** `experiments/w5_btx_headtohead.py` · **Result:** `results/rebuttal/w5_btx/w5_btx_seed137.json` · **Compute:** A100 (BTX joint-finetune of 4 experts).

### REB-W2: Divergence–gain predictor held-out validation (HONEST NEGATIVE)
**Purpose:** answer vacq/ssw3 — validate the gain = a + b·divergence law on held-out conditions with prediction intervals.
**Method:** fit on the original 8 conditions (gain = −2.80 + 0.834·div, R²=0.872); run 6 NEW cross-lingual subset conditions (fresh seed-42 training, router re-init retry) as held-out; check band coverage. Scripts `experiments/w2_regression_condition.py` + `w2_refit.py`. Points `results/rebuttal/w2_regression/cond_*.json`, fig `figures/rebuttal/w2_divergence_gain_refit.png`.
**Result — the predictor does NOT generalize:** only **2/6 held-out in the 95% band, RMSE 12.3pp**; refit-on-all R² drops 0.87→0.57. All 4 out-of-band points are **over-predicted** (actual gain ≪ predicted).
**Mechanism (verified, not artifact):** router re-init confirmed routing was correct (ok=True, try 1) — the low gains are REAL. Mean-divergence is inflated by one high-divergence domain, but **fusion gain is limited by complementarity**: if any single specialist covers the subset (e.g. a language specialist generalizes to code, or one domain has low base loss), gain vs best-specialist is ~0 regardless of mean divergence. IN-band points ({yoruba,welsh}, {tamil,yoruba,welsh}) are the mutually-non-generalizing high-base-loss cases.
**Rebuttal implication:** do NOT lean on the predictor as a novelty claim. Present W2 as the honest held-out validation reviewers asked for — it **scopes** the law (holds for balanced mutually-complementary high-divergence conditions; over-predicts for asymmetric subsets) and identifies **complementarity, not mean divergence, as the governing factor** (future work). Consistent with the paper's existing `regime-dependent` caveat (EXP-21). Lean novelty on **W5** (BTX delta) instead.

### REB-W7: 7B cross-lingual real-task-win gamble (RUNNING)
**Purpose:** close W1's "beyond loss" gap with a *capable* base so a real task (Belebele MC-QA) can move above chance.
**Method:** Qwen2.5-7B, 4 cross-lingual specialists (Tamil/Yoruba/Welsh/Code), 8-bit AdamW + gradient checkpointing on 1×H100-80GB, post-hoc KALAVAI fusion, downstream Belebele/FLORES/cloze. Script `experiments/w7_qwen7b_crosslingual.py`.
**Infra:** two spontaneous Lambda node deaths (A100 ~15min, H100 ~171min) → made it **HF-resilient** (each specialist pushed to private repo `mechramc/kalavai-w7-qwen7b-{lang}-seed42`, resumes from HF on relaunch). User's HF write token in `.env`.
**RESULT (Done, seed 42, n=100) — MIXED, and the negative is the important part:**
- **Fusion gain SCALES to 7B: +10.80%** vs best specialist (eq loss base 2.045 → moe 1.882). Cloze improves: Tamil 0.607→0.700, **Yoruba 0.454→0.683**, Welsh 0.446→0.516 (code 0.718→0.682, slightly down). Perplexity: Yoruba 13.3→5.1, Tamil 4.0→2.8. So the fusion *mechanism* works at 7B.
- **Real-task win FAILED — Belebele MoE ≤ base:** Tamil 0.36→0.32, Yoruba 0.22→0.18 (n=100, chance 0.25). The fused model is a *better language model* but a *worse QA model* than the base.
- **Why:** Qwen2.5-7B base is instruction-capable (Tamil Belebele 0.36 > chance); fine-tuning on raw cc100 **text** (LM objective) improves perplexity/cloze but **erodes the instruction/QA ability** (catastrophic forgetting). The MoE inherits this.
- **The definitive answer to AC condition #2:** even at 7B with a capable base, the KALAVAI protocol delivers **LM-loss / perplexity** gains, NOT **downstream task-accuracy** gains — and can degrade task accuracy. Do NOT claim a downstream task win. Scope the contribution to LM/perplexity + the W5 BTX delta.
- **Infra note:** completed in ~42 min because it resumed 3/4 specialists from HF (Tamil/Yoruba/Welsh pushed before an earlier watchdog kill). The "H100 node deaths" root cause = a 3h node watchdog killing long runs (fixed to 12h), NOT Lambda hosts. HF-resume worked flawlessly.
Result → `results/rebuttal/w7_qwen7b/w7_seed42.json`. Specialists durable on HF: `mechramc/kalavai-w7-qwen7b-{tamil,yoruba,welsh,code}-seed42` (private).

---

## Rebuttal bottom line (for writing the response)
Both AC conditions have concrete results: **W5** = technical delta from BTX (KALAVAI within ~1pp of BTX with no expert training/data pooling) — the lead; **W1** = downstream accuracy (MoE +11–18% over best single specialist, FLORES 7–9× ppl drop) with honest scale caveats. **W2** = honest held-out validation (predictor is regime-scoped — present as integrity, don't oversell). **W6** = robustness (collapse fixable). **W7** = 7B real-task attempt (running; payoff uncertain). Recommend softening the paper's divergence–gain claim per W2.
**Headline (reframed after skeptic review):** per-language the MoE only *ties* its own specialist (router recovers the diagonal). The real claim is the **average across languages: MoE 0.528 vs best SINGLE specialist 0.475 = +11% rel** (both seeds) — one model beats any individual specialist on downstream accuracy, because each specialist is good at only its own language.
**Honest limitations (must state in rebuttal, don't hide):** (1) cloze accuracy is loss-adjacent; FLORES ppl *is* loss. (2) The one genuine task, Belebele MC-QA, is near-chance and MoE is *below* base on Yoruba (0.22 vs 0.29) — Pythia-410M is too weak for MC-QA. (3) Cross-lingual routing is the *easy* case (languages trivially separable). ⇒ W1 answers "downstream accuracy" partially; a clean "real task win beyond loss" needs a capable base (see REB-W7 7B plan). W1 is necessary but does NOT address the novelty (BTX) objection — that's W5/W2.
**Compute discipline:** all runs terminated + API-verified; W&B root cause = CRLF in .env (see RUN_STATE).
