# DSM paired evaluator core

This is a dependency-free Rust library for scoring candidate memory outputs against evaluator-owned ground truth.

## Build and test

From the repository root:

~~~bash
cargo fmt --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/evaluator/Cargo.toml -- --check
cargo test --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/evaluator/Cargo.toml --locked
cargo clippy --manifest-path research/mace-decision-sufficient-memory-bench-v0-1/evaluator/Cargo.toml --locked --all-targets -- -D warnings
~~~

The crate has no external dependencies. The LAFS helper is in src/longmemeval_v2.rs and uses the official published Small/Medium operating points; re-check the official benchmark page before each evaluation. Run manifests record the cluster-bootstrap seed/replicate count for DSM and the tier for LongMemEval-V2. Its public API currently provides:

- Validation of evaluator cases and candidate outputs.
- Exact integer action-loss and regret scoring.
- Paired memory versus no-memory comparison on the same episode, enforcing and preserving an identical downstream-policy SHA-256 digest and decision seed.
- Memory-induced harm and helpful/harmful action-change indicators.
- Admissible-evidence precision and necessary-evidence recall as integer ratios.
- Leave-one-out evidence interventions. The runner must invoke the same frozen, deterministic downstream policy digest and decision seed for each intervention, withholding exactly one returned evidence ID. Positive item contribution means retaining the item lowered decision loss; negative contribution means it increased loss.
- Cost receipt reconciliation across non-overlapping phases (their sum, including residual, must equal wall time), retained/index/network bytes, and separately marked measured versus estimated cost.
- Deterministic whole-world cluster bootstrap with 95% percentile intervals for mean paired memory gain and memory-induced harm rate; within-world episodes are resampled together. Paired receipts preserve the downstream-policy SHA-256 digest and decision seed.
- Exact LongMemEval-V2 LAFS integration over the official 1–200 second log-latency budget, including Small and Medium reference frontiers, delta-LAFS helpers and known-answer reference-score tests.

## Critical evaluator isolation

The public candidate fixture is in ../fixtures/candidate-episodes-v0.1.jsonl. It contains only candidate-visible fields; generator seeds and task-family labels are withheld.

EvaluatorCaseV1 and the loss/admissibility labels do not appear in the candidate JSONL schema or seed fixture. The known-answer cases in unit tests are public development tests and must not be treated as a secret test set or benchmark result. Confirmatory answer keys, held-out seeds, evaluator labels and scoring configuration must be held by a separate evaluator service or private artifact store that candidate processes cannot read.

Do not pipe this crate's evaluator structures into candidate prompts, HTTP requests, telemetry, log output or public run manifests. Every paired or leave-one-out run must record the same downstream_policy_digest and decision_seed. The crate API is a scoring boundary, not a sandbox. The caller must enforce OS/process/container isolation and only send candidate-visible episode data to the memory system.

## Current limitations

This is a scoring core, not yet a complete benchmark runner:

- It does not parse JSONL or generate random worlds.
- The LAFS helper computes the metric but does not invoke the official LongMemEval-V2 data preparation, reader, scoring package, or submission validator.
- It does not invoke candidate systems or implement AML Add/Search.
- The LAFS module computes the frontier score from operating points but does not invoke the official LongMemEval-V2 data preparation, fixed reader, scoring package or submission validator.
- It does not implement process isolation, sandboxing, cryptographic provenance or a private key service.
- The current leave-one-out method does not estimate higher-order evidence interactions or Shapley values.
- It has not been compiled, formatted or tested on the current authoring environment because no Rust toolchain is available here.

The first exact-head local gate is format, test and strict Clippy on the commands above. A skipped workflow is not a pass.
