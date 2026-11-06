# Symthaea R4 preregistration reference validator

This standalone Rust crate performs offline, fail-closed semantic readiness checks for the R4 ignition preregistration in `docs/research/templates/`.

It is deliberately outside the production Cargo workspace. It consumes only a JSON document; it never executes candidates, reads confirmation data, signs artifacts, writes receipts, or authorizes promotion.

## Run

```sh
cargo fmt --manifest-path tools/sym-rsi-r4-prereg-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/sym-rsi-r4-prereg-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/sym-rsi-r4-prereg-reference/Cargo.toml
cargo run --manifest-path tools/sym-rsi-r4-prereg-reference/Cargo.toml -- docs/research/templates/SYM_RSI_R4_IGNITION_PREREGISTRATION_V0_2026-11-06.json
```

The checked-in document is intentionally `TEMPLATE_NOT_PREREGISTERED`, so the CLI must reject it with a non-zero readiness result. Unit tests construct a synthetic, fully populated specimen in memory and mutate it to exercise negative controls; no synthetic specimen is represented as an actual experiment.

## Invariants

- Exactly two declared arms bind to the identical shared inner-agent start reference.
- Both arms use the same declared backend version and frozen context identity.
- Both arms bind to the same digest-identified inclusive total-cost envelope, including failures, retries, confirmation/transfer work and human review/rework.
- Discovery, validation, sealed confirmation and fresh-family transfer have distinct manifest digests; a disjointness-audit declaration is required.
- Evaluator implementation/configuration/task-contract/epoch-manifest digests are explicit and shared across arms; candidate write access is prohibited.
- The paired pilot has at least five seeds; at least two complete-loop reruns are required; recursive-progress claims require at least two complete outer cycles.
- Per-round development/audit fields, the full null-control set, statistical analysis fields, reporting fields and authority guardrails are required.
- Claim dependencies prevent recursive-progress and ignition labels from being selected without their required lower-level evidence profiles.

## Important limits

A syntactically valid digest is not an authenticated digest. Distinct split commitments and a declared zero-overlap audit are not proof that the underlying task/family populations are disjoint. The independent custodian must inspect the actual manifests and retain an audit artifact. Likewise, a shared evaluator identity declaration does not prove evaluator independence or prevent hidden channels; the existing evaluator, isolation, receipt, attestation and promotion authorities remain controlling.

A ready report means only that the document is internally consistent enough for independent review. It is not a preregistration approval, evidence of experiment execution, evidence of an R4 result, or authorization to start a run.
