# CMAQ-012: Paired Advantage-of-Prediction Result Auditor

**Status:** protocol v0.1 + dependency-free Rust result auditor; draft  
**Tracking issue:** [#9122](https://github.com/Luminous-Dynamics/symthaea/issues/9122)  
**Claim ceiling:** scoring-contract tool with a synthetic fixture; no real environment result, comparative claim or SOTA claim

## Purpose

CMAQ-011 defines a decision-centric benchmark protocol. CMAQ-012 implements its measurement boundary: a single Rust tool validates paired results from independent adapters and computes the predictive-advantage contrasts. Adapters produce per-episode records; they do not get to declare their own results correctly paired.

## Arms

- **D0:** direct policy without predictive rollouts.
- **D1:** compute-matched direct/value policy without learned future rollouts.
- **W1:** learned action-conditioned world-model planner.
- **W2:** shuffled-future negative control that breaks candidate-to-prediction correspondence.
- **W3:** oracle-dynamics upper bound.

## Input

The auditor accepts a strict tab-separated file with the exact header order:

```text
task_id capability split seed arm return success status initial_state_sha256 proposer_sha256 candidate_set_sha256 environment_config_sha256 candidate_count rollout_horizon planner_compute_units latency_ms
```

Each task × capability × split × seed must have exactly one row for each arm. All five arms must share the initial-state, proposer, candidate-set and environment-configuration SHA-256 digests; candidate count; rollout horizon; and declared planner compute budget.

Status values are complete, failed or timeout. Failed and timed-out rows remain in all denominators and must carry a finite numeric penalty return plus success=false; the adapter owns and documents that penalty. This prevents survivorship bias from silently dropping the worst rollouts.

The parser rejects malformed digests, unknown statuses, duplicate/missing arms, mismatched seeds, candidate sets, initial states, environment configs, horizons and compute budgets. A deterministic SHA-256 of the exact input bytes is included in the report. The dependency-free SHA-256 implementation has a known-answer test.

## Estimands

For each task/capability/split, the tool computes paired episode differences:

- **AoP = W1 − D1:** predictive advantage over a compute-matched direct evaluator.
- **Deployment gain = W1 − D0:** gain over the direct policy without extra compute.
- **Prediction mediation = W1 − W2:** sensitivity to correct action-conditioned futures.
- **Oracle gap = W3 − W1:** residual gap to exact dynamics.
- **Compute-only gain = D1 − D0:** benefit from extra direct-policy compute.

Each contrast reports a paired mean and deterministic 95% percentile-bootstrap interval for both return and success rate. Each arm reports mean return, success rate, p50/p95 latency and complete/failed/timeout counts.

A qualification slice is eligible for consideration only when it has at least 20 paired episodes and is labelled qualification. This does not prove that the task family was held out or that the experiment was preregistered; those still require external protocol evidence and artifact digests. Development slices are never qualification-eligible.

The auditor does not pool tasks or capabilities into a single headline number.

## Run

```sh
cargo test --locked
cargo run --quiet --locked -- fixtures/qualification.tsv
```

The checked-in fixture demonstrates schema and scoring mechanics only. Its numbers are synthetic and cannot be cited as Symthaea performance.

## Next integration

1. Build one external adapter for pinned Gymnasium/MuJoCo and one for DMC.
2. Emit this TSV from frozen D0/D1/W1/W2/W3 policy arms using identical initial-state/proposer/candidate/environment digests.
3. Retain raw episode artifacts; compare auditor outputs with an independent Python/R analysis in a separate verification implementation.
4. Add a visual/embodied adapter only after its action, reset and reward contracts can meet the same paired-record invariants.
