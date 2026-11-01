# BASALT Admission Reference Evaluator

**Status:** reference implementation draft; compilation and tests not executed in this environment  
**Tracking:** [#9722](https://github.com/Luminous-Dynamics/symthaea/issues/9722)

## What this module does

The xtask module in basalt_admission.rs reduces a versioned profile's independently supplied obligation outcomes into the BASALT three-valued result:

- valid: every mandatory obligation is valid for the exact subject and intended use;
- invalid: a mandatory obligation or binding is affirmatively rejected;
- unknown: required evidence is missing or its context/policy requires re-verification.

The reducer uses BTreeMap/BTreeSet and sorts output, so the report is deterministic under obligation-order permutations. Missing mandatory obligations are materialized as unknown. Duplicate and undeclared obligation IDs are rejected. A valid mandatory result without an evidence reference is downgraded to unknown. Optional outcomes are retained for audit but do not alter the mandatory conjunction.

## Critical trust boundary

This module is reference-only.

It does not:
- authenticate the admission profile or policy source;
- verify DSSE signatures or signer authorization;
- retrieve or verify Rekor inclusion/consistency proofs;
- verify witness signatures or split-view status;
- read a linearizable etcd authority snapshot;
- verify local SQLite durability or external anti-rollback;
- prove observer delivery or capture completeness;
- verify that an evidence reference points to the bytes claimed.

The input profile and the supplied obligation outcomes are untrusted data until an external layer authenticates/verifies them. The report contains reference_only=true. Do not connect this command directly to a production authorization decision.

The reference evaluator establishes only the truth-table behavior and exact binding semantics for an already assembled set of verifier outcomes.

## CLI

Example input fixture:

    benchmarks/basalt/fixtures/admission-reference-valid.v1.json

Run:

    cargo run --package xtask -- basalt-admission-reference-check --input benchmarks/basalt/fixtures/admission-reference-valid.v1.json

The command prints a JSON report. It exits successfully only for an aggregate valid result; invalid and unknown reports are printed and then return a non-zero error. This is an evaluation of supplied data, not proof that those inputs are authentic.

## Current tests

The Rust module contains 14 unit tests covering:
- all-required-valid admission;
- missing-required unknown;
- all nine pairings of the strong-Kleene AND truth table;
- invalid-over-unknown reduction;
- subject binding mismatch;
- context and policy re-verification;
- duplicate and unexpected obligation IDs;
- optional unknown preservation;
- empty profile rejection;
- unsupported capsule schema rejection;
- obligation-order determinism;
- valid outcome without an evidence reference.

These tests are authored, but were not executed here because cargo/rustc are unavailable in the current environment. Rust compilation, rustfmt, Clippy, schema-validator execution, and CI remain outstanding.

## Integration boundary

The next implementation stage must replace raw caller-provided obligation outcomes with typed outputs from the corresponding independently verified adapters. It must also authenticate the selected profile and enforce its signer, trust-root, witness, source/log, environment, and intended-use scope.

The reducer should remain pure and backend-neutral. SQLite and etcd should provide trusted-state records; they should not be coupled into the truth-table implementation.
