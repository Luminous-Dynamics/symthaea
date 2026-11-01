# PITB paired intervention trials

This directory contains the preregistered plan, paired outcome contract and strict measured-evidence receipt schemas for PITB causal proof-intervention experiments.

## First trial: support-antichain context selection

The synthetic plan fixture defines four arms:

- A0: full context plus standard retrieval (control).
- A1: predicted minimal-support antichain (primary intervention).
- A2: oracle minimal support from an exhaustive small-context lattice (diagnostic upper bound only).
- A3: size-matched random support subset (negative control).

Each pair freezes the task, project component, repository commit, Lean environment, exact initial proof-state and target-statement digests, model seed and six-dimensional resource envelope. Arm execution order is deterministically randomized. Every arm uses a fresh process and independent copy of the same initial state; mutable Lean caches are not shared.

## Contracts

| File | Purpose |
|---|---|
| proof-intervention-trial-plan-v0.schema.json | Preregistered task × state × seed × arm assignment |
| fixtures/support-antichain-intervention.synthetic.json | Eight-pair synthetic plan; not theorem outcomes |
| proof-intervention-trial-outcome-v0.schema.json | One pair × arm measured outcome row |
| fixtures/proof-intervention-outcome.synthetic.json | One synthetic success-shaped row; no evidence artifacts |
| proof-intervention-checker-receipt-v0.schema.json | Strict kernel checker and axiom-closure receipt |
| proof-intervention-run-receipt-v0.schema.json | Exact execution identity, budget, resource and artifact-digest receipt |
| proof-intervention-successor-receipt-v0.schema.json | Downstream/successor test receipt |

The shared plan verifier and evidence validators live in:
- crates/bridges/symthaea-proof-insights/src/proof_intervention_trial.rs
- crates/bridges/symthaea-proof-insights/src/proof_intervention_evidence.rs

The CLIs are verify_proof_intervention_trial and analyze_proof_intervention_outcomes.

## Local validation commands

From the repository root, run:

cargo run --locked --bin verify_proof_intervention_trial -- --self-test data/benchmarks/proof-insight-transfer-v0/intervention-trials/fixtures/support-antichain-intervention.synthetic.json

cargo run --locked --bin analyze_proof_intervention_outcomes -- --self-test data/benchmarks/proof-insight-transfer-v0/intervention-trials/fixtures/support-antichain-intervention.synthetic.json

For measured outcomes, the analyzer requires the frozen plan JSON, a JSONL file containing exactly one row for every preregistered pair × arm, an evidence root containing the exact proof/checker/task-integrity/successor/run artifacts, the benchmark-integrity reviewer-key registry, an externally obtained SHA-256 digest of that registry, and an output report path.

The measured CLI shape is:

cargo run --locked --bin analyze_proof_intervention_outcomes -- --analyze PLAN.json OUTCOMES.jsonl EVIDENCE_ROOT REVIEWER_KEY_REGISTRY EXPECTED_REGISTRY_SHA256 REPORT.json

## What the analyzer currently guarantees

- Complete pair × arm outcome coverage; duplicate rows and run IDs are rejected.
- Exact joins for task, component, source commit, Lean toolchain, environment, initial proof state, target statement, model seed, evaluation horizon, continuation policy and randomized execution order.
- Non-attempts, timeouts and unavailable tools remain failures in the intention-to-treat denominator.
- All six resource dimensions are checked against the frozen per-pair caps.
- Measured evidence paths are canonicalized beneath the evidence root; traversal and symlink escape are rejected; artifact bytes must match the recorded SHA-256.
- Every measured row requires an eligible target-integrity receipt signed by an active reviewer key in the externally pinned registry. It binds the task, statement, source commit, Lean toolchain, cohort and snapshot.
- A verified proof requires a strict checker receipt for the exact target/proof, zero sorryAx, an allowed axiom closure, and any applicable successor receipt.
- A strict run receipt binds the actual resources and receipt digests to the same arm and frozen budget.
- Reports contain per-arm success/cost, paired intention-to-treat risk differences, discordant-pair counts, component-cluster bootstrap intervals and paired mean cost deltas.

## Current trust boundary

The report deliberately sets confirmatory_result_allowed=false. Checker, run and successor receipt schemas and identity joins are validated, but the repository does not yet verify a cryptographic signature from a trusted runner/verifier identity for those execution artifacts. Content-addressed bytes prove that the evidence matches the supplied digest; they do not independently prove who executed the checker or whether the runner was compromised. Add a pinned runner-key registry or independently attested execution environment before confirmatory causal claims.

The checked-in fixtures are synthetic contract data. No measured Lean trial has been run, no real outcome JSONL has been collected, no causal effect is measured, and no PR workflow result is claimed in this document.
