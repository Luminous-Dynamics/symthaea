# symthaea-decision-assessment

A small, non-authoritative Rust composition layer for engineering decision assessments.

## Purpose

Consume qualified outputs from existing owners and combine them into five non-compensatory axes:

1. Process admissibility.
2. Evidence-use validity.
3. Engineering applicability.
4. Causal justification.
5. Action-policy correctness.

It also qualifies the candidate's pre-outcome commitment and provides a paired ACT/HOLD evaluator with step-indexed runtime interruption timing.

## Boundary

This crate does **not**:

- authenticate signatures, SCITT statements, transparency proofs, or trust roots;
- establish source truth, physical correctness, or causal truth;
- implement a temporal ledger, evidence-use engine, applicability oracle, probability calibrator, or action/loss oracle;
- create engineering approval, requirement acceptance, release, manufacture, operation, or external-effect authority.

The caller supplies ReceiptQualification values only after the corresponding owner-native validators have run. The caller also supplies CommitmentQualification only after an independent commitment/timestamp owner has qualified the exact packet digest, timestamp integrity, authenticity, and outcome blinding. Missing owner qualification or commitment witness remains indeterminate.

## API outline

- evaluate_json(packet_bytes, owner_qualifications, commitment_qualification) returns an AssessmentReport with per-axis states, exact reason codes, preserved outcome state, and a hard effect_authority = none boundary.
- evaluate_act_hold_pair(should_act, should_hold) scores both sides of a paired scenario.
- ActHoldEpisode records the first observable stop-trigger step, first irreversible-effect step, and terminal safe-disposition step. This measures timely interruption rather than merely checking the final answer.

The candidate commitment uses the domain-separated canonical projection described in docs/research/ENGINEERING_DECISION_RUST_COMPOSITION_V0_1.md. Evaluator-authored axis findings, defeaters, counterfactual labels, outcome records, and assessment results are excluded from the candidate's pre-outcome commitment.

## Offline known-answer corpus

benchmarks/engineering-change-impact/act-hold-pairs-v0.1.json contains seven synthetic known-answer pairs:

- eligible ACT with pre-execution HOLD;
- always-abstain negative control;
- post-hoc abstention after an irreversible effect;
- action after a runtime-visible stop trigger;
- action without qualified authority;
- timely runtime interruption;
- unknown trigger/hold timing.

The unit test consumes this frozen fixture and compares all reported fields and reason-code sets. This is a deterministic synthetic regression suite, not a population reliability estimate or an external benchmark result.

## Build and test

From the workspace root:

    cargo test -p symthaea-decision-assessment
    cargo clippy -p symthaea-decision-assessment --all-targets -- -D warnings
    cargo fmt --all -- --check

These commands are the intended local verification path. Do not describe them as passed until run against the exact branch head and recorded in a reproducible receipt.
