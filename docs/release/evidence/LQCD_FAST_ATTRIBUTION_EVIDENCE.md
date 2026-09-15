# LQCD fast-lane structural failure-attribution oracle

Status: independently executed standard-library evidence.

## Subject

This evidence tests the cheap structural theorem from issue #3404 over exact GitHub compare surfaces:

- `4bad8af72ff775e7c869b6df83faba718a339a36` -> `6a74273f189b876919e5feee3418201e46bd6415`
- `6a74273f189b876919e5feee3418201e46bd6415` -> `5242b44b50df491a71b047f3c144bdfd8af9afc3`
- cumulative `4bad8af72ff775e7c869b6df83faba718a339a36` -> `5242b44b50df491a71b047f3c144bdfd8af9afc3`

The frozen input contains the exact changed-file lists returned by GitHub compare for those transitions.

## Executed result

- oracle SHA-256: `9a520a300bdf9bcc8670960c881bae99a172d7fad43043ff3905a6db6d934a44`
- input SHA-256: `ce1ab1ae181737fd496e64a4d14c2c3a51901dbf62559ff7bd753fdf5f7595cf`
- result SHA-256: `aac844f1dcc78cc7a8e976c018045f37252b1251a9cabf1f326fe1a11d9130ab`
- return code: 0
- synthetic controls: PASS

All three observed transitions contain zero changes to the gate-relevant scientific subject set:

- `crates/domains/symthaea-particle-physics/**`
- workspace `Cargo.toml`
- `Cargo.lock`
- `rust-toolchain.toml`
- `.cargo/**`

The changed files are confined to the focused verifier/diagnostic plane.

## Classification

The executed oracle classifies:

- base -> `6a742...`: `VerifierProfileDeltaWithoutSubjectDelta`
- `6a742...` -> `5242...`: `VerifierOperationalDeltaWithoutSubjectDelta`
- cumulative base -> `5242...`: `VerifierProfileDeltaWithoutSubjectDelta`

Therefore:

- `candidate_caused_rust_lint_established = false`
- `candidate_rust_repair_authorized = false`
- `base_execution_required_for_inherited_failure_claim = true`

This does **not** establish that the lint is inherited. It establishes only that the observed CI/diagnostic candidate lineage did not modify package/Cargo/lock/toolchain inputs capable of directly introducing a Rust-source lint. Exact-base execution under equivalent verifier/profile/toolchain semantics is still required before claiming `InheritedBaselineFailure`.

## Synthetic controls

The oracle self-test verifies that changes to package Rust source, `Cargo.lock`, `rust-toolchain.toml`, and `.cargo/**` are detected as scientific-subject deltas; canonical qualifier changes are separated from subject deltas; workflow-only changes are separated from subject deltas; and unknown paths fail closed as `UnclassifiedDelta`.

## Claim ceiling

This is structural attribution evidence only. It does not identify the actual Clippy lint, does not prove the base passes or fails, does not authorize a Rust repair, does not qualify any LQCD implementation, and establishes no numerical or physics result.
