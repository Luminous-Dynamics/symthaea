# EQC standalone reference evaluator

This nested Rust workspace is independent of Symthaea's root Cargo workspace and lockfile.

## Validate the frozen synthetic seed

From the repository root:

```bash
cargo fmt --manifest-path tools/eqc-reference/Cargo.toml -- --check
cargo test --locked --manifest-path tools/eqc-reference/Cargo.toml
cargo run --locked --manifest-path tools/eqc-reference/Cargo.toml -- \
  --corpus docs/research/eqc-seed-corpus-v0.1.json \
  --out /tmp/eqc-reference-report.json
```

The four public tasks are synthetic authorization/replay fixtures. They do not contain live health data and do not establish EHR integration, clinical safety, production authorization, cryptographic security, regulatory compliance, or system qualification.

## Evaluate an adapter observation

```bash
cargo run --locked --manifest-path tools/eqc-reference/Cargo.toml -- \
  --corpus docs/research/eqc-seed-corpus-v0.1.json \
  --task-id eqc-health-wrong-subject-deny-001 \
  --observation path/to/observation.json \
  --out /tmp/eqc-qualification-report.json
```

'contract_checks_pass' means the submitted observation is internally consistent with the task's required contract. 'qualified' remains false in v0.1 because no separately trusted observer attestation verifies actual state, resource metering, or independent replay. The adapter cannot qualify itself by setting 'verifier.independent = true'.

See [the benchmark protocol](../../docs/research/EVIDENCE_QUALIFIED_CAPABILITY_BENCHMARK_V0_1.md).


## Independently trusted observer verification

A self-reported adapter observation remains unqualified. Independent qualification requires a signed observer envelope and evaluator-provisioned trust inputs. See [EQC trusted observer protocol v0.1](../../docs/research/EQC_TRUSTED_OBSERVER_PROTOCOL_V0_1.md).

```bash
cargo test --locked --manifest-path tools/eqc-reference/Cargo.toml
cargo run --locked --manifest-path tools/eqc-reference/Cargo.toml -- \
  --corpus docs/research/eqc-seed-corpus-v0.1.json \
  --task-id eqc-health-wrong-subject-deny-001 \
  --observation run/observation.json \
  --observer-envelope run/observer-envelope.json \
  --observer-trust-store operator/trust-store.json \
  --trust-store-sha256 OPERATOR_PINNED_TRUST_STORE_SHA256 \
  --run-id RUN_ID_FROM_HARNESS \
  --challenge-nonce 64_HEX_CHARACTERS_FROM_HARNESS \
  --verification-time 2026-11-28T12:00:00Z \
  --challenge-ledger /var/lib/eqc/consumed-challenges \
  --out run/qualification-report.json
```

The trust-store pin, run challenge and verification time are evaluator/operator inputs, never values trusted from the adapter. The implementation verifies signed evidence; it does not yet deploy a privileged observer daemon or provide hardware-backed attestation. The challenge ledger is currently single-host filesystem storage, not a distributed anti-replay service.
