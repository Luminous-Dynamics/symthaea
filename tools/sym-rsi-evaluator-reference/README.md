# Symthaea evaluator integrity fixture validator

A standalone, offline Rust crate that validates the structure and declared semantics of the synthetic evaluator-integrity fixture pack.

It does **not** run an evaluator against the threat scenarios. It does not inspect real task data, sealed test contents, hidden seeds, grader code, target labels or external custodians. A valid report means only that the fixture-pack contract is internally consistent.

## Run

```sh
cargo fmt --manifest-path tools/sym-rsi-evaluator-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/sym-rsi-evaluator-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/sym-rsi-evaluator-reference/Cargo.toml
cargo run --manifest-path tools/sym-rsi-evaluator-reference/Cargo.toml -- docs/research/fixtures/SYM_RSI_EVALUATOR_RESEARCH_INTEGRITY_FIXTURES_V0_2026-11-06.json
cargo run --manifest-path tools/sym-rsi-evaluator-reference/Cargo.toml -- --state-routing docs/research/fixtures/SYM_RSI_STATE_ROUTING_FIXTURES_V0_2026-11-07.json
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


## State-routing fixture pack

The same offline crate also validates the synthetic state-routing pack used by #10844:

```sh
cargo run --manifest-path tools/sym-rsi-evaluator-reference/Cargo.toml -- --state-routing docs/research/fixtures/SYM_RSI_STATE_ROUTING_FIXTURES_V0_2026-11-07.json
```

It checks exactly 16 fixture descriptions: ten hard-gate negative controls (fixed IDs, categories and dispositions), four positive controls, and two abstention controls. It rejects duplicate IDs, altered dispositions, removed hard gates, forced routing where abstention is expected, missing controls, and empty evidence field names.

A valid pack is not evidence that a real router detects these conditions. The validator does not inspect real task-family labels, raw memory, hidden outcomes or custodian evidence.

The library currently contains 20 authored Rust unit tests across the evaluator-integrity and state-routing fixture contracts. They are authored but have not been executed in an environment without the Rust toolchain.
