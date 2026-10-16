# INTX equipment programming benchmark corpus

This is the first executable scaffold for the proposed 50-case synthetic qualification corpus described in [INTX-PROGRAM-001](../../docs/INTX-PROGRAM-001-EQUIPMENT-PROGRAMMING.md).

It is deliberately a standalone, dependency-free Rust package so it does not alter Symthaea's workspace member graph or introduce a protocol client. The current harness verifies the test data's schema, cardinality, uniqueness, dimension coverage, and a small set of mandatory fail-closed relationships.

## Run

```sh
cargo test --manifest-path crates/intx-program-bench/Cargo.toml --locked
```

Run from the Symthaea repository root. This can be run locally without a device, external service, protocol server, or CI runner.

## Corpus shape

`fixtures/cases.tsv` has 50 stable synthetic scenarios: 10 each for identity, artifact integrity, authority, transport/reconciliation, and lifecycle/audit. Each row records a stable ID, expected state, dispatch expectation, and whether reconciliation is required.

Dispatch classes:
- `none`: no remote effect attempt is permitted or expected.
- `occurred`: the synthetic scenario includes an attempted remote dispatch; this is not a claim of physical effect.
- `may_have_occurred`: the transport outcome is ambiguous and must remain unknown pending reconciliation.

## Current limitation

**The harness validates the corpus contract, not Symthaea runtime behavior.** It does not yet invoke the action runtime, an actual connector, a simulator, or physical equipment. The next gate should bind these fixtures to the existing INTX authority/reconciliation APIs and prove the same outcomes at an exact source head. A green result here must never be represented as device interoperability, functional-safety certification, or field qualification.
