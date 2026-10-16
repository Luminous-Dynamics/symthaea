# INTX equipment programming benchmark corpus

This is the first executable scaffold for the proposed 50-case synthetic qualification corpus described in [INTX-PROGRAM-001](../../docs/INTX-PROGRAM-001-EQUIPMENT-PROGRAMMING.md). The broader [paradigm-shift roadmap](../../docs/INTX-PROGRAM-001-PARADIGM-SHIFTS.md) explains the evidence-carrying engineering compiler model and staged benchmarks. The [API ownership/readiness matrix](../../docs/INTX-PROGRAM-BENCH-API-OWNERSHIP.md) distinguishes issue designs from source-level implementation evidence.

It is deliberately a standalone, dependency-free Rust package so it does not alter Symthaea's workspace member graph or introduce a protocol client. The current harness verifies the test data's schema, cardinality, uniqueness, dimension coverage, and a small set of mandatory fail-closed relationships.

## Run

```sh
sha256sum --check crates/intx-program-bench/fixtures/cases.tsv.sha256
cargo test --manifest-path crates/intx-program-bench/Cargo.toml --locked
cargo fmt --manifest-path crates/intx-program-bench/Cargo.toml -- --check
cargo clippy --manifest-path crates/intx-program-bench/Cargo.toml --locked -- -D warnings
```

Run from the Symthaea repository root. These checks require only the repository-pinned Rust toolchain and no device, external service, protocol server, or full workspace build. A focused [GitHub Actions workflow](../../.github/workflows/intx-program-bench.yml) runs them against the exact PR head.

## Corpus shape

`fixtures/cases.tsv` has 50 stable synthetic scenarios: 10 each for identity, artifact integrity, authority, transport/reconciliation, and lifecycle/audit. Each row records a stable ID, expected state, dispatch expectation, and whether reconciliation is required.

Dispatch classes:
- `none`: no remote effect attempt is permitted or expected.
- `occurred`: the synthetic scenario includes an attempted remote dispatch; this is not a claim of physical effect.
- `may_have_occurred`: the transport outcome is ambiguous and must remain unknown pending reconciliation.

## Current limitation and next gate

**The harness validates the corpus contract, not Symthaea runtime behavior.** It does not yet invoke the action runtime, an actual connector, a simulator, or physical equipment. [INTX-PROGRAM-BENCH-001 / #7702](https://github.com/Luminous-Dynamics/symthaea/issues/7702) tracks the next step: bind each frozen fixture to existing INTX authority and reconciliation APIs through synthetic adapters and an independent expected-state oracle.

A green result here means only that this standalone package's fixture/invariant checks passed at the tested source head. It must never be represented as device interoperability, functional-safety certification, or field qualification.
