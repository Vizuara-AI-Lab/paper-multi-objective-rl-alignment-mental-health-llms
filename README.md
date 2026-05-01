# Multi-Objective RL Alignment for Mental Health LLMs: Dynamic Reward Weighting and Free Safety Constraints

Aligning a 1.5B language model for mental health counselling using multi-objective GRPO with a Lagrangian safety constraint and Dynamic Reward Weighting.

## Paper
- **PDF**: [tex/main.pdf](tex/main.pdf)
- **Source**: [tex/main.tex](tex/main.tex) — compile with `tectonic -X compile tex/main.tex`
- **Sealed peer review**: [review.md](review.md)

## Primary results

- **All RL conditions outperform the base** model on every reward dimension, peaking on Informativeness (M7: Cohen's d = 0.66; 75% per-question win rate over Base on CounselBench-Eval, 89 prompts).
- **Safety is essentially free**: the Lagrangian constraint costs only 0.25% composite reward (M7 vs. M5); tightening the threshold from d=0.20 to d=0.05 reduces violations from 3.4% to 2.2% at 0.3% reward cost.
- **Empathy is budget-bounded**: allocating the full reward weight to empathy (M8) raises empathy from 4.213 (M7) to 4.918 (p < 0.0001); the M8 vs. M9 contrast (with vs. without safety) is 0.017 (p = 0.528), showing safety does not suppress empathy.
- **1.5B specialised model beats much larger general models** (GPT-4, LLaMA3-70B, Gemini, human counsellors) on every reward dimension under the same scoring instrument; we discuss the LLM-tilted-scorer caveat explicitly.

## Method overview

The policy (Qwen2.5-1.5B-Instruct) is fine-tuned with GRPO using a scalarized reward composed of three frozen DistilRoBERTa quality scorers (Guidance, Informativeness, Empathy) and a Lagrangian penalty on a fourth Safety Cost scorer. A Dynamic Reward Weighting (DRW) module adapts the per-step scalarization weights based on per-axis improvement headroom; in practice DRW converges to ~(0.33, 0.33, 0.33), so M7 effectively becomes the equal-weight M3 baseline.

## Reproduction

The training and evaluation pipeline is described in `tex/main.tex` Section 3 (Method) and Section 4 (Experimental Setup). Hyperparameters: GRPO group size K=4, KL coefficient β=0.04, learning rate 1e-5, max 200 steps with reward-plateau early stopping. Conditions M3, M5, M7 are run with three seeds; remaining variants use seed 42. Compute: a single RTX PRO 6000 (~100 GB) per condition.

## Figures

| Figure | Description |
| --- | --- |
| fig-architecture | Multi-objective RL alignment architecture (GRPO + DRW + Lagrangian) |
| fig-training-pipeline | Reward-model training, policy training, dual-benchmark evaluation |
| fig-main-rewards | Guidance / Informativeness / Empathy by condition on CounselBench-Eval |
| fig-effect-sizes | Cohen's d effect sizes vs. Base across reward dimensions |
| fig-empathy-budget | Empathy with full optimization budget (M8/M9) vs. multi-objective dilution (M7) |
| fig-safety-tradeoff | Safety threshold sweep: violations vs. composite reward |
| fig-baselines | Comparison to GPT-4 / LLaMA3 / Gemini / human counsellors under the same RMs |

## Recommended venues

- **NeurIPS Safe & Trustworthy ML / Alignment Workshop** — workshop tier, exact topical fit (~40-50% acceptance).
- **ACL Clinical NLP / NLP for Positive Impact** — domain-fit workshop track at ACL.
- **AAAI 2027** — mid-tier conference fallback if extended with a within-family scaling sweep.
- **TMLR** — archival journal, evaluates on technical correctness.
- **JMIR Mental Health** — domain-specific journal; would require human-rated companion study.

## Provenance

- Session id: `20260501-063243-b7af`
- See [log.md](log.md) for the per-stage execution log and [state.json](state.json) for the full pipeline state.
