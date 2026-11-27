# Standalone EQC reference evaluator

This crate is a small, independent Rust workspace for validating the six-episode public synthetic EQC v0.1 task seed. It intentionally does not depend on the Symthaea workspace, model runtime, Mycelix Health, Holochain or an EHR.

## Reproduce

From the repository root:

```bash
cargo test --locked --manifest-path research/eqc-reference/Cargo.toml
cargo run --locked --manifest-path research/eqc-reference/Cargo.toml -- \
  docs/research/eqc-seed-corpus-v0.1.json /tmp/eqc-reference-report.json
```

The nested `[workspace]` declaration is intentional: the evaluator owns a separate Cargo.lock and dependency graph. CI creates and commits that lockfile on the feature branch once, then all qualification steps use `--locked`.

## Scope

The evaluator checks:
- strict typed task/corpus parsing and required provenance declarations;
- repository-local fixture path containment (including symlink resolution);
- SHA-256 input-byte identity;
- consent/challenge fixture contents against actor, subject, scope, status and time window;
- fail-closed reference decisions for subject mismatch, revocation, expiry, scope mismatch and challenge replay;
- zero-mutation requirements and two-sided no-op / allow-all baseline controls.

It reports `reference_contract_only_not_system_under_test`. This is not an EHR/agent integration benchmark, clinical qualification, cryptographic audit, production authorization certification, or standards conformance result. The JSON Schema files are versioned interchange contracts; this Rust crate currently enforces the typed seed contract and does not claim to execute a full JSON Schema Draft 2020-12 engine.

## Canonical seed and receipts

Task schemas, run-receipt schema, public corpus, and synthetic fixtures remain in `docs/research/`. The corpus binds the three input files by their raw-byte SHA-256. Any fixture change therefore requires an explicit corpus digest update.

