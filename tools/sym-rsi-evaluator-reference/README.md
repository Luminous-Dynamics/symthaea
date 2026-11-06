# Symthaea evaluator integrity fixture validator

A standalone, offline Rust crate that validates the structure and declared semantics of the synthetic evaluator-integrity fixture pack.

It does **not** run an evaluator against the threat scenarios. It does not inspect real task data, sealed test contents, hidden seeds, grader code, target labels or external custodians. A valid report means only that the fixture-pack contract is internally consistent.

## Run

```sh
cargo fmt --manifest-path tools/sym-rsi-evaluator-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/sym-rsi-evaluator-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/sym-rsi-evaluator-reference/Cargo.toml
cargo run --manifest-path tools/sym-rsi-evaluator-reference/Cargo.toml -- docs/research/fixtures/SYM_RSI_EVALUATOR_RESEARCH_INTEGRITY_FIXTURES_V0_2026-11-06.json
```

## Contract

- Exactly 20 synthetic fixture descriptions: 16 negative controls and four positive/inconclusive controls.
- All 16 negative control IDs, threat categories and expected dispositions are fixed and checked.
- Negative controls must remain hard gates; a positive control must not be marked as a negative hard gate.
- Fixture IDs and negative control IDs must be unique and complete.
- Every fixture must retain its synthetic-only marker, non-empty scenario and required report fields.
- Unknown controls, altered expected dispositions, missing controls, duplicated IDs and altered status fail closed.
- The fixture pack is not allowed to become a claim of evaluation success or benchmark authority.

The source uses a strict serde model with unknown fields rejected, and tests mutate the checked-in fixture pack to exercise invalid states. The crate is intentionally outside the root production Cargo workspace.
