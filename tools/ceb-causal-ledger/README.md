# CEB causal ledger (standalone research crate)

This crate is the first pure-Rust implementation slice for CEB's causal-investigation ledger.

## Scope

It validates inert records for:
- exact task/source/configuration binding;
- unique record identifiers and append-order sequence numbers;
- evidence-parent ordering and duplicate evidence-origin detection;
- prospective forecast → action → outcome ordering;
- typed action categories and explicit authorization for side-effecting/test actions;
- intervention-authority constraints;
- explicit missingness rather than converting unobserved outcomes to numeric zero;
- hypothesis, repair, primary-grade and pristine-regrade identity links.

## Non-goals

This crate does **not**:
- execute experiments, tests, builds, shell commands or candidate code;
- read or write files, use the network, spawn processes or access credentials;
- compute or authenticate digests;
- verify DSSE signatures, trust roots, runner identity, isolation or grader independence;
- prove that an observation is true or that a causal claim is correct;
- emit or replace a CEB run receipt.

The `Digest` type is opaque. The caller supplies it; this crate only checks equality and linkage. A `PristineRegrade` label is an identity field, not proof that the grading environment was actually independent.

## Run the pure unit tests

From the repository root:

```sh
cargo test --manifest-path tools/ceb-causal-ledger/Cargo.toml --locked
```

The crate has its own `[workspace]` boundary and is intentionally not a member or dependency of the production workspace. Integrate it only after exact-head compilation, independent review, and reconciliation with the existing CEB/general receipt and DSSE work.

## Status

Research prototype. The source and unit tests are committed but have **not** been compiled or executed in this environment. The PR remains draft. CI jobs that were skipped for the draft PR are not evidence of passing compilation or tests.
