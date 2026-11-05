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
- SHA-256 binding of exact source/output bytes and exact Studio transaction-receipt bytes, with an independent re-read/re-hash verification function.
- Pinned `sha2 = 0.10.9` dependency and minimal committed lockfile; all test/run commands use `--locked`.
- A fresh policy fork restored from one frozen pre-intervention checkpoint for every condition; mutable policy state cannot leak between counterfactual arms. The same explicit common-random-number seed is passed to every arm so stochastic adapters can pair random draws.
- Explicit NotApplicable metric state for unsupported modalities.

## Run

From this directory, run:

```sh
cargo fmt --check
cargo test --locked
MEL_RMI_CODE_REVISION="$(git rev-parse HEAD)" cargo run --locked --example canonical_suite
```

The example emits a CSV over 160 paired fixtures (32 deterministic seeds per
feature dimension) for fixed, imitation, one-way, yoked, unrelated-feature-
shortcut and rule-based policies, followed by 160 four-condition reciprocal
controlled-partner trials. Every one of the 960 policy trials and 160 reciprocal
trials also receives four input and four output synthetic score/MIDI payload
sets plus exact receipt bytes; the example hashes and re-verifies all 1,120
trial records before reporting `artifact_verified,1120`. The repository workflow
stages this crate outside the root Cargo workspace, tests the exact pull-request
head, and gates all expected positive/negative-control counts.

## Interpretation boundary

Passing these tests means only that this finite synthetic harness detects its known positive and negative controls. The manifest stores exact event identities and a caller-supplied code revision; it does not cryptographically hash arbitrary audio or MIDI payloads. Real artifact digests and Studio transaction receipts must be supplied by the MEL-013A adapter before external or human-session claims.

Track A's deterministic fixture set is a census of those fixtures, not a statistical sample of musicians. Human/live tracks still require a pilot, preregistered power analysis, appropriate ethics review, and blinded musician judgments. No human data collection is included here.
