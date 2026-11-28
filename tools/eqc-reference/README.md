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
