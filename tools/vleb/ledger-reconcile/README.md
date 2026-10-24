# VLEB Ledger Reconciliation Reference

This is a small, dependency-free Rust reference implementation for the synthetic ledger task in tools/vleb/examples/ledger-reconciliation/.

It is a benchmark contract fixture, not a production accounting system or a claim of banking-system compatibility.

## Run

From the repository root, with Rust 1.96.0 or a compatible Rust toolchain:

```sh
cargo test --locked --manifest-path tools/vleb/ledger-reconcile/Cargo.toml
cargo run --locked --manifest-path tools/vleb/ledger-reconcile/Cargo.toml -- \
  --input tools/vleb/examples/ledger-reconciliation/input.csv
```

Successful execution writes deterministic JSON to stdout. Invalid CSV, a conflicting transaction ID, invalid values, or arithmetic overflow exits nonzero and writes no report to stdout. Diagnostics are written to stderr.

## Contract

- Header and field count are strict.
- Transaction IDs and account IDs must be non-empty, trimmed, and free of control characters.
- Side must be exactly CREDIT or DEBIT.
- Amounts are positive unsigned integers representing unspecified-currency minor units; floating point is not used.
- Exact duplicate transaction rows are ignored once and counted.
- Reuse of a transaction ID with different content rejects the entire input.
- Account output is deterministic and lexicographically ordered.
- All aggregate additions are checked; overflow fails closed.
- The CLI emits no report until the full input has been validated and reconciled.

The implementation is deliberately narrow. It does not perform postings to a database, handle currencies/exchange rates, implement bank controls, or claim equivalence to COBOL, IBM Db2, z/OS, JCL, or a production system.

## Validation status

The reference crate currently contains 16 unit tests and 4 CLI integration tests, including a seeded differential reducer over 64 generated ledgers, permutation invariance, arbitrary duplicate multiplicity, maximum-value arithmetic, strict integer-token syntax and malformed CSV cases. These tests were authored but have not been compiled or executed in this environment because rustc and cargo are unavailable. CI results must be checked at the exact PR head. The fixture is not score-eligible until those tests execute, an independent reviewer approves the oracle, and adversarial task admission is complete.
