# EGDDM ShiftBench D6.12 — Benchmark-oracle integrity and two-sided qualification v0.1

**Date:** 2026-11-16
**Status:** evaluator-only prototype; synthetic known-answer fixture; not independently adjudicated
**Related work:** [Issue #12446](https://github.com/Luminous-Dynamics/symthaea/issues/12446)

## Research question

Can a benchmark's grading pipeline be trusted not to mutate the state it measures, reuse stale results, ignore relevant surfaces, or reject valid alternatives? Can an independent auditor detect these defects while preserving positive and negative controls?

## Paradigm shift

Treat the benchmark harness as a system under test, not a trusted oracle. The measurement pipeline must be qualified using frozen pre/post state, replay, repeat execution, cache identity, observer coverage, valid controls and invalid controls. The auditor must separately answer (a) what the candidate semantically achieved and (b) whether the native verifier's result is trustworthy.

## v0.1 model

Each candidate-visible oracle run binds a candidate input digest, post-candidate state digest, pre/post-grading state digests, replay state digest, native and repeated verifier outcomes, cache identity, and observed state surfaces. Evaluator-only truth supplies the expected semantic outcome, expected post-candidate digest, required observer surfaces and known-answer fault classes.

Faults are derived from raw observations: candidate-state drift, verifier state mutation, replay disagreement, stale-cache mismatch, nondeterminism, observer-coverage gap, false positive, and false negative. The gold expected fault set must exactly match this derivation; it cannot silently override the observations.

## Metrics

- False acceptance, false rejection and unnecessary abstention.
- Fault instances detected/missed and false-alarm fault classes.
- State mutation, replay disagreement, cache mismatch and nondeterminism counts.
- Observer coverage gaps and candidate-state drift.
- False integrity-valid / false integrity-invalid assertions.
- Positive/negative/sham/fault-injection control preservation.

No scalar accuracy score may hide an oracle state mutation, false pass, false reject or unobserved state surface.

## Synthetic known-answer controls

Ten cases cover: reference-valid patch, empty patch, state-mutating verifier, stale cache with false pass, nondeterministic repeat, replay-state disagreement, valid alternative rejected by native verifier, missing database observer surface, sham grading, and candidate-state drift.

Three traces are paired: an independent auditor, a native-verifier-only baseline, and an always-reject baseline. The last should not pass merely by being conservative; it must fail positive controls.

## Acceptance gates

- Zero undetected verifier-state mutation in the registered critical controls.
- 100% pass for valid reference/alternative controls and 100% fail for empty/no-op negative controls.
- 100% detection of deterministic state-mutation, stale-cache, replay, nondeterminism and observer-coverage faults in this finite fixture.
- Repeat grading is idempotent on frozen inputs, or the nondeterminism is surfaced.
- Independent replay matches the post-candidate state, or grading remains invalid/unknown.
- Missing observation surfaces never default to unchanged.
- False-positive and false-negative defects are reported separately.

## Baselines and research scale

Baselines: native verifier only; snapshot/diff audit; independent replay; duplicate oracle; mutation-testing harness; always-reject; and independent human adjudication.

Development: this ten-case fixture. Validity pilot: 10 fault classes × 5 task families × 3 independent seeds = 150 audits. Research target: 12 fault classes × 8 benchmark/task families × 5 seeds = 480 audits. These are proposed targets, not an existing corpus or results.

## Claim ceiling and next gates

D6.12 v0.1 checks supplied state digests, replay evidence, cache identity, repeat outcomes and observer-surface declarations. It does not authenticate telemetry, prove OS-level isolation, independently reimplement arbitrary benchmark semantics, or certify a benchmark universally. Next gates: Rust toolchain validation, independent oracle implementation, actual fault-injection harness, exact state snapshot/replay, benchmark portability trials, blinded annotations and external replication.