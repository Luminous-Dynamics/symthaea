# Counterfactual Temporal Memory Value Evaluation (v0.1)

**Tracking:** NEURO-SILICON-011 / #9561  
**Status:** evaluator-side research primitives authored; no build, tests, or intervention runs executed  
**Package:** `research/temporal-evaluator`  
**Candidate allocator:** #9559 / `research/temporal-bench/src/temporal_value.rs`

## 1. Purpose

Estimate the causal value of retaining a memory item by comparing paired runs over the same task, future stream, model, initial state, candidate-visible feature snapshot and resource budget.

For intervention set (A):

[
Delta TED_A =
TED(operatorname{evict}(A))
-
TED(operatorname{retain}(A))
]

The evaluator keeps wrong-belief, abstention and missing-output exposure separate. Positive componentwise delta means retention reduced that exposure. There is no weighted scalar score.

This is a counterfactual outcome, not a candidate feature. The candidate-side allocator receives only its own estimated value, uncertainty and observable metadata. The evaluator's realized outcomes, intervention assignments, future trajectory and protected seeds must remain unavailable to candidate code.

## 2. Paired intervention contract

Each run is bound to an identical `EpisodeContext`:
- episode ID;
- task and future-stream SHA-256 digests;
- model and initial-state digests;
- frozen pre-treatment candidate-feature digest;
- total exposure duration;
- byte, compute, refresh, observation and optional calibrated-energy budgets.

The retain and evict arms must differ only in the declared intervention item set. All other selected memory items must be identical. Duplicate, unsorted, unknown or unexpected item differences fail closed.

The evaluator deterministically randomizes whether retain or evict is executed first using an evaluator-held seed and records the first treatment. This uses SplitMix64 for reproducibility only; it is not a cryptographic random generator or secrecy boundary. The seed must not be exposed to candidate processes.

Each run reports actual bytes, retained byte-microseconds, compute, refresh count, observations, read bytes, write bytes and optional calibrated energy. Declared resource caps are enforced against actual use. Energy-budgeted runs require an energy measurement; absence is not interpreted as zero energy.

## 3. Interaction effects and exact small-subset oracle

Item-level effects are context-dependent. The evaluator therefore accepts group interventions and does not assume item values add linearly.

For at most 16 items, `qualify_exact_subsets` requires one evaluator run for every subset under the same episode context. It returns:
- exact best subset for wrong-belief exposure;
- exact best subset for abstention exposure;
- exact best subset for missing-output exposure;
- a nondominated three-component exposure frontier;
- componentwise regret for a candidate-selected subset.

This is a **clairvoyant evaluator upper bound**, not an implementable candidate policy. It is computationally exponential and is limited to small synthetic episodes. Large episodes need randomized subset designs, fractional factorial designs or other estimators with uncertainty intervals.

The exact subset oracle does not combine the three outcomes into one arbitrary objective. It also does not claim that independently optimal subsets for separate dimensions are interchangeable.

## 4. Authored implementation

Separate package files:
- `research/temporal-evaluator/Cargo.toml`
- `research/temporal-evaluator/src/main.rs`
- `research/temporal-evaluator/README.md`

The source contains eleven authored tests covering:
- paired componentwise treatment effects;
- stream-context mismatch;
- unexpected memory differences;
- deterministic treatment order;
- independent componentwise subset minima and regret;
- incomplete subset enumeration;
- Pareto filtering;
- group interaction effects;
- byte/compute/refresh/observation/energy and exposure caps;
- duplicate registry and selected-item rejection.

No Cargo build, rustfmt, Clippy or test execution has been performed in this authoring environment.

## 5. Security boundary

Separate package structure is not a security boundary. Source code is public and a hostile candidate can inspect public interfaces. Before confirmatory use, #9437 must provide:
- distinct generator, candidate and evaluator OS identities;
- protected evaluator seed/configuration and oracle storage;
- candidate sandbox with no inherited oracle file descriptors or writable evaluator paths;
- evaluator-owned signing key;
- signed receipts binding run, model, config, task, candidate feature snapshot, intervention assignment, raw prediction and resource measurements;
- replay/TOCTOU controls and adversarial filesystem/descriptor tests.

Do not run candidate and evaluator code in the same process and then call the oracle hidden. Do not export exact subset outcomes, treatment assignment or the randomization seed to the candidate.

## 6. Qualification targets (not results)

Use development source families for predictor training/calibration and sealed source families for confirmatory evaluation. Proposed targets from #9561 remain:
- calibration slope 0.9–1.1 after development-only calibration;
- nominal prediction-interval coverage within confidence bounds;
- at least 20% less realized TED than the strongest simple policy at equal byte/refresh budgets on a preregistered family, or equal TED with 30% fewer byte-seconds;
- no increase in stale-memory false recall or future-information leakage;
- publish calibration, uncertainty, group interaction and distribution-shift failure results.

Randomized retain/evict effects are causal only for the declared intervention, candidate system, task distribution, resource envelope and evaluation horizon. They do not establish universal memory value.
