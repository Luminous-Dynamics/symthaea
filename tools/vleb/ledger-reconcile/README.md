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

## Fixture integrity gate

Before the reference tests run, CI executes `bash tools/vleb/check-fixture-integrity.sh`. It checks the three fixture file hashes against reviewed constants, checks those constants against `task.json`, and checks that mutation IDs are unique and mutation test references are registered by the task contract. The gate is intended to catch accidental fixture/contract drift early.

This is a consistency gate, not an independent semantic audit: it does not prove that the reference implementation is correct, that a negative control was actually executed, or that any cataloged mutant was compiled and killed. Those remain separate admission requirements.

## Executable semantic mutation probes

The CI workflow runs `tools/vleb/run_mutation_campaign.py` against isolated copies of the reference crate. The first campaign contains four source-level mutants linked to the preregistered catalog:

- `VLEB-LDG-001`: process exact duplicate transactions a second time.
- `VLEB-LDG-002`: silently ignore conflicting transaction-ID reuse.
- `VLEB-LDG-004`: replace checked aggregate additions with wrapping arithmetic.
- `VLEB-LDG-007`: swap CREDIT and DEBIT aggregation paths.

Each variant must compile before the contract suite is run. A compile failure, timeout, or harness error is **unresolved**, never counted as a semantic kill. A mutant is counted as killed only when its preregistered target test itself fails; unrelated suite failures do not count as a kill. CI retains the machine-readable result and raw per-mutant logs as an artifact, with hashes and both the checked-out revision and expected PR-head SHA.

The catalog contains 17 probes; only these four semantic source mutants are executable in the current harness. The remaining semantic and evidence-manipulation probes remain planned and must not be represented as executed. Even a 4/4 kill result would be an initial oracle-smoke test, not a broad mutation score or a production-safety result.

## Independent evidence auditor

`tools/vleb/audit_receipt.py` recomputes artifact hashes and checks the task's logical-test inventory against concrete execution names, baseline/reference controls, environment identity, mutation target tests, differential evidence, timing and costs. The CLI requires caller-supplied pins for the task contract, protocol, candidate patch, auditor source, verifier binary, checkout revision and immutable runtime image; these must come from the trusted evaluator/run manifest, not from receipt fields. Auditor source and verifier binary have separate identities; matching the source digest does not prove which executable produced the receipt.

Run its synthetic adversarial unit tests with:

```sh
python tools/vleb/test_receipt_auditor.py
```

The auditor deliberately rejects public-development fixtures and does not currently verify external host-attestation signatures. Hashes prove that bytes match a digest, not that the logs truthfully came from the claimed runtime. For confirmatory studies, package the complete evidence bundle and verify signed provenance from a dedicated trusted runner before allowing a score.

## Validation status

The reference crate currently contains 20 unit tests and 4 CLI integration tests, including a seeded differential reducer over 64 generated ledgers, permutation invariance, arbitrary duplicate multiplicity, maximum-value arithmetic, strict integer-token syntax and malformed CSV cases. The latest checked completed CI attempt compiled the crate, passed all 20 unit tests, all 4 CLI tests and strict Clippy. The format gate failed, and the mutation campaign found 2 semantic kills with 2 unresolved mutant constructions; follow-up commits address the remaining formatting and format-sensitive mutation anchors. The exact current PR-head workflow must pass before treating those fixes as validated. The fixture is not score-eligible until task admission, independent oracle review and all required controls are complete.
