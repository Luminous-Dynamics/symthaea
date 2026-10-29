# MEL-MR-EVAL-009 — Attribute-Aware Audio Preservation Benchmark

**Status:** benchmark protocol draft; no benchmark results claimed  
**Tracking:** [Issue #8955](https://github.com/Luminous-Dynamics/symthaea/issues/8955)  
**Implementation dependency:** [MEL-MR-EVAL-008 render-binding envelope](https://github.com/Luminous-Dynamics/symthaea/pull/8944)

## Research thesis

A good audio editor must make the requested musical change and preserve everything the request did *not* authorize changing. A single similarity score or a broad "edit region" mask is insufficient: a system could change the target note and delete it, alter its dynamics, delay its onset, or damage neighboring material while appearing locally consistent.

The MEL-MR-EVAL line therefore treats preservation as **typed, operation-conditioned obligations**:

1. symbolic obligation — requested change and protected score attributes are independently checked;
2. acoustic obligation — non-target content is compared outside the edit support;
3. within-edit obligation — attributes that should remain invariant inside the changed region, such as attack timing and energy envelope during a pitch edit, remain checked;
4. provenance obligation — receipts replay against exact buffers, masks, detector configs, full Score data, and renderer metadata;
5. uncertainty obligation — missing measurements or unsupported edit types fail closed or abstain; no average score can compensate for a critical failure.

The implementation currently covers a narrow full-Score subset: one non-overlapping part/role, pitch-only edits or transposition, preserved attack timing and RMS envelope, and explicitly bound renderer latency. Duration/onset edits, polyphony, insertion/deletion, and causal renderer attestation remain open work.

## Public benchmark position as of 2026-10-29

### MMAE and the Audio Editing Challenge

The public MMAE set contains 2,000 examples and 17,741 atomic rubrics across speech, music, sound, and mixtures, with six complexity levels. Its official evaluation separates Instruction Following Rate (IFR), Consistency Rate (CR), and Exact Match Rate (EMR). EMR is sample-level all-rubrics success and is the challenge's primary ranking metric. See the [MMAE paper](https://arxiv.org/abs/2606.07229), the [official leaderboard protocol](https://audio-editing-challenge.github.io/leaderboard/), and the [baseline repository](https://github.com/Audio-Editing-Challenge/Audio-Editing-Challenge-Baseline).

Published numbers must remain attached to their scopes and evaluators:

| Reference result | Scope | IFR | CR | EMR |
|---|---|---:|---:|---:|
| Audio Editing Challenge Agent baseline | Full MMAE, 2,000 examples | 44.06% | 74.63% | 7.45% |
| Challenge Single Model baseline | Single-operation subset, 1,003 examples | 42.12% | 75.78% | 7.58% |
| Original full-benchmark comparison: Step-Audio-EditX | Full benchmark | 44.86% | 58.88% | Not leader |
| Original full-benchmark comparison: Ming-UniAudio | Full benchmark | Not leader | Not leader | 3.20% |

The first two rows come from the challenge baseline repository and use different task scopes; they are **not** direct competitors on a common denominator. The latter two numbers are the original MMAE comparison as summarized by the [Audio Editing Survey](https://github.com/MM-Speech/AudioEditSurvey). Results produced with different subsets, versions, or judge models must be reported separately.

As of October 29, 2026, the challenge's hidden 500-example sets and submission SDK are scheduled for **November 10, 2026**. The published registration deadline was **October 8, 2026**; this plan assumes public-set research unless the team is already registered or receives explicit organizer approval. Agent-track model-version/weight release requirements also apply. Do not infer eligibility from this repo work.

### Music-specific complementary evaluation

MMAE is valuable for general instruction following and preservation, but it is not a substitute for musical-structure checks. The [Audio Editing Survey benchmark index](https://github.com/MM-Speech/AudioEditSurvey) lists music-focused resources such as MuseCPBench/MuseCPEval, which evaluate melody, harmony, rhythm, structure, and timbral preservation. Use those as additional axes where task compatibility and licensing permit; do not combine their scores into MMAE's IFR/CR/EMR.

## Evaluation protocol

### Phase A — deterministic conformance

Use synthetic score/audio pairs with known ground truth. Every case must carry a mutation-family label, expected verdict, calibration-profile digest, source/target Score hashes, source/target audio hashes, and exact supported edit operation.

Minimum critical negative-control families:

- **edited-target deletion:** remove or heavily attenuate the pitch-edited note while keeping the changed region inside the acoustic mask;
- **within-edit envelope corruption:** preserve a different pitch but alter its RMS envelope beyond the frozen profile;
- **protected leakage:** alter neighboring or non-target notes outside the authorized region;
- **onset corruption:** delay, remove, or add an attack event—even inside the edited spectral region;
- **mask escalation:** broaden the acoustic mask, add an onset mask where attacks must remain protected, or misstate renderer latency;
- **receipt forgery:** alter a metric/verdict/config and recompute its unkeyed hash; buffer-backed replay must still reject it;
- **profile drift:** change detector thresholds while claiming the same frozen profile;
- **render manifest drift:** swap renderer version, bank/effects digest, format, or latency metadata.

Positive controls must include identity, a pitch edit that preserves amplitude/attack, and an edit whose authorized change is the only change. Unsupported operations should be reported as unsupported/abstained—not scored as successes.

### Phase B — paired editor ablation

Run the same editor, prompts, task IDs, source audio, random seeds, and tool budgets across arms:

| Arm | Configuration | Question answered |
|---|---|---|
| A | Baseline editor, no MEL verifier | Reference performance |
| B | A + symbolic edit-locality verifier | Does explicit symbolic grounding help? |
| C | A + audio preservation verifier | Does audio-locality evidence reduce collateral changes? |
| D | A + symbolic and audio gates | Does the combined non-compensatory gate improve exact success? |
| E | D + one bounded repair attempt and independent re-audit | Does repair recover false negatives without introducing regressions? |

For fairness, the base edit backend and inference settings remain fixed. Repair arm E is allowed one explicitly budgeted edit/retry; report its extra tools, time, and cost. Do not give a guarded arm more hidden retries.

First run a fixed, stratified public pilot for fast debugging; then run the complete public MMAE set using the official evaluator. Freeze detector profiles and any decision thresholds before the final evaluation. Report task/seed manifests and per-example results so the paired comparisons can be reproduced.

### Metrics — report separately, never as a compensating average

**Primary public outcome:** paired sample-level EMR delta, with a task-stratified bootstrap 95% confidence interval.

**Secondary public outcomes:** IFR and CR individually, with confidence intervals; report task-complexity and modality breakdowns where the official labels permit.

**Verifier-specific outcomes:** critical false-accept rate, false-reject rate, abstention rate, rate of unsupported operations, mask-policy rejection count, profile mismatch count, report replay failure count, and sensitivity/precision for edited-note attacks.

**Operational outcomes:** p50/p95 added verification latency, total wall-clock time, editor/tool calls, GPU-hours, and cost per accepted edit. Include the quality/latency trade-off curve rather than reporting only the best threshold.

**Human validation:** run a blinded, randomized paired-listening study for representative music cases. Ask separate questions for (a) requested edit correctness, (b) preservation of non-target material, (c) audible damage inside the edited note, and (d) overall preference. Human judgments should audit failure modes, not be silently replaced by one learned judge.

## Pre-registered targets

These are **goals, not achieved results or forecasts**.

1. **Deterministic conformance:** all declared known-truth positive/negative controls reproduce their expected verdict; zero false accepts on the frozen synthetic critical-negative suite.
2. **Critical false accepts:** at least 384 independent critical-negative examples with zero false accepts gives an approximate two-sided 95% Wilson upper bound below 1%. Count independent composition/task examples—not multiple windows from one composition—and report mutation-family results separately. At 0/6, the upper bound is about 39%; at 0/300, it is about 1.26%. This is why a tiny perfect smoke suite cannot justify a reliability claim.
3. **Public-set improvement target:** after the matched ablation, aim for at least **12% full-MMAE EMR** against the 7.45% public agent baseline, with a paired bootstrap lower 95% bound for the EMR improvement above zero. This is an aspirational target, not a claim that the gate alone will reach it.
4. **Non-regression guardrails:** target IFR no worse than 2 percentage points below the matched baseline, CR at least 2 percentage points higher, and clearly disclosed changes in abstention, runtime, and cost. If the gate raises CR but harms edit completion, report the trade-off rather than calling it unqualified superiority.
5. **Calibration discipline:** no final test tuning; profile hashes and detector thresholds must be frozen before held-out runs. Profiles in the current code are content-bound objects, but their approval must come from a trusted/allowlisted registry.
6. **Human agreement:** report inter-rater agreement and confidence intervals. Do not promote the system from public benchmark gains alone to claims of perceptual equivalence.

## What would count as beating the field?

The claim should be narrow and reproducible:

- on the same complete MMAE version and frozen judge, the guarded system raises EMR with a paired confidence interval excluding zero;
- CR and critical false-accept rates improve, not merely average spectral similarity;
- a music-specific held-out suite verifies attack, RMS-envelope, rhythmic, harmonic, and non-target preservation obligations;
- improvements survive ablations, different renderers/instrument banks, unseen compositions, and a blinded listening audit;
- latency, cost, abstention, and unsupported edit coverage are disclosed.

Until those runs exist, MEL-MR-EVAL-008 is verification infrastructure, not an audio editor leaderboard result or a state-of-the-art claim.
