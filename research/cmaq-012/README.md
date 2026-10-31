# CMAQ-012: Paired Advantage-of-Prediction Result Auditor

**Status:** executable Rust auditor v0.1; draft  
**Tracking issue:** [#9122](https://github.com/Luminous-Dynamics/symthaea/issues/9122)  
**Claim ceiling:** result-ledger validation and paired estimand calculation only; no real environment result, competitive claim or SOTA claim

## Why this is a paradigm shift

A world model can appear to outperform a direct policy because it received more compute, a better action proposer, a different candidate set, more favorable initial states, or selective episode reporting. A benchmark must isolate the contribution of predictive rollouts themselves.

This auditor requires five paired arms on the same initial state, proposer, action-candidate set, environment configuration, adapter, rollout horizon and declared planner compute budget:

| Arm | Meaning |
|---|---|
| D0 | Direct policy without predictive rollouts |
| D1 | Compute-matched direct/value policy without learned future rollouts |
| W1 | Learned action-conditioned world-model planner |
| W2 | Shuffled-future negative control that breaks candidate-to-prediction correspondence |
| W3 | Oracle-dynamics upper bound |

The key quantity is **advantage of prediction (AoP) = W1 - D1**. D1 controls for extra computation. W1-D0 is deployment gain; W1-W2 tests whether candidate-aligned prediction matters rather than merely adding a model-shaped computation path; W3-W1 measures remaining oracle gap; D1-D0 estimates compute-only gain.

This is stricter than reporting world-model return beside a direct-policy paper result. Those unpaired numbers do not isolate the value of prediction.

## Connection to current literature

- [Do World Models Make Better Robots?](https://arxiv.org/abs/2609.29669) (2026 survey) reports that only a small fraction of its 160-benchmark corpus explicitly compares VLA and world-model approaches, and only four benchmarks in its taxonomy bridge prediction to executed action. This motivates a paired estimator of predictive advantage, rather than another isolated prediction score.
- [How Should World Models Be Evaluated?](https://arxiv.org/abs/2606.15032) argues for counterfactual action fidelity, closed-loop rollout validity, policy ranking, optimization lift, exploitability and uncertainty calibration.
- [MRS.Q](https://proceedings.mlr.press/v306/chang26a.html) shows that search can hurt despite high model accuracy and that overestimation control matters. Its Gymnasium/MuJoCo and DMC tasks are a relevant future source of environment adapters, not a baseline result produced by this auditor.

## Input contract

The dependency-free Rust CLI consumes strict UTF-8 TSV. It rejects:
- a missing, reordered or extra header field;
- malformed SHA-256 digests, identifiers, numbers or statuses;
- duplicate rows, missing arms or unpaired seeds;
- mismatched initial-state, proposer, candidate-set, environment-config or adapter digests;
- unequal candidate counts, rollout horizons or declared planner-compute budgets.

The exact header is:

    task    capability    split    seed    arm    initial_state_sha256    proposer_sha256    candidate_set_sha256    environment_config_sha256    adapter_sha256    source_sha256    checkpoint_sha256    candidate_count    rollout_horizon    planner_compute_budget_units    status    return    task_success    latency_ms

Each task × capability × split × seed must have exactly one D0, D1, W1, W2 and W3 row. Each row binds its own source and checkpoint digests. The shared state/proposer/candidate/environment/adapter digests and budget fields must match within a paired episode.

Statuses are success, failed and timed_out. Every row—including failed and timed-out rows—must carry an explicit finite adapter-assigned return. Failed/timed-out rows must have task_success=0. They remain in the paired denominator; the auditor never drops them or silently replaces their return.

A qualification group requires at least 20 paired episodes. Smaller groups are labelled smoke_only; 20 is a minimum eligibility gate, not a claim that all statistical power concerns have been solved.

## Outputs and uncertainty

For return and binary task success, each task/capability/split gets five paired contrasts with deterministic paired bootstrap 95% percentile intervals:
- AoP: W1 − D1
- deployment gain: W1 − D0
- prediction mediation: W1 − W2
- oracle gap: W3 − W1
- compute-only gain: D1 − D0

Per-arm return, success rate, status counts and latency p50/p95 are emitted. The report includes the SHA-256 digest of the exact input bytes, bootstrap seed and replicate count.

No results are pooled across tasks or capabilities. Do not use a single headline score to hide a failing capability.

## Run

    cargo test --locked
    cargo run --quiet --locked -- --input fixtures/qualification-smoke.tsv

The checked-in fixture contains four paired episodes and is intentionally labelled smoke_only. It validates mechanics; it is not a qualification result.

## Next

1. Add adapters that emit this ledger from pinned Gymnasium/MuJoCo HalfCheetah-v4 and DeepMind Control quadruped-walk runs.
2. Implement D0/D1/W1/W2/W3 using an identical action proposer, frozen seed manifest and compute accounting.
3. Validate the shuffled-future control: W2 must preserve compute and candidate count while breaking candidate-to-prediction alignment.
4. Add held-out parameter families, calibration/development seeds and a sealed qualification manifest.
5. Independently reproduce the baseline and candidate, preserving raw episode ledgers and all failed runs.

A parser pass is not a benchmark pass. A smoke fixture is not an environment result. An AoP claim requires matched real-environment runs and independent artifact verification.
