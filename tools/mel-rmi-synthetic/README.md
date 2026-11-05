# Melothaea RMI-Bench v1 — synthetic Track A harness

This isolated, dependency-free Rust crate implements the first offline milestone from the preregistered RMI-Bench v1 protocol.

It is a benchmark-integrity oracle, not a Melothaea evaluation. Its positive control is a rule-based policy whose expected transformation is declared by a small synthetic task fixture. It does not establish human musical relevance, collaborative value, consciousness, or performance against a commercial model.

## What it exercises

- Paired partner observations with identical task, session, seed and frozen history identities.
- A targeted treatment, sham/no-op control, and unrelated-feature negative control; only the declared feature may differ in the targeted arm.
- Fixed-response, imitation-only, one-way-prompt and cross-session yoked controls, plus a rule-based positive control.
- Separate targeted, sham and unrelated-feature action contrasts.
- A task-specific synthetic response-relation check. Imitation can be input-sensitive while still failing this complementarity oracle.
- A four-condition controlled-partner response path with separate targeted/sham/unrelated contrasts and a declared reverse-direction response-rule oracle.
- Exact event-ID, seeded condition-order execution, policy-version and Git-revision binding.
- A fresh policy fork restored from one frozen pre-intervention checkpoint for every condition; mutable policy state cannot leak between counterfactual arms.
- Explicit NotApplicable metric state for unsupported modalities.

## Run

From this directory, run:

```sh
cargo fmt --check
cargo test --locked
MEL_RMI_CODE_REVISION="$(git rev-parse HEAD)" cargo run --locked --example canonical_suite
```

The example emits a CSV over the five feature dimensions for fixed, imitation,
one-way, yoked, unrelated-feature-shortcut and rule-based policies, followed by
four-condition reciprocal controlled-partner trials. The repository workflow
stages this crate outside the root Cargo workspace, tests the exact pull-request
head, and gates all expected positive/negative-control counts.

## Interpretation boundary

Passing these tests means only that this finite synthetic harness detects its known positive and negative controls. The manifest stores exact event identities and a caller-supplied code revision; it does not cryptographically hash arbitrary audio or MIDI payloads. Real artifact digests and Studio transaction receipts must be supplied by the MEL-013A adapter before external or human-session claims.

Track A's deterministic fixture set is a census of those fixtures, not a statistical sample of musicians. Human/live tracks still require a pilot, preregistered power analysis, appropriate ethics review, and blinded musician judgments. No human data collection is included here.
