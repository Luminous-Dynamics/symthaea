# Symthaea R4 preregistration reference validator

This standalone Rust crate performs offline, fail-closed semantic readiness checks for the R4 ignition preregistration in `docs/research/templates/`.

It is deliberately outside the production Cargo workspace. It consumes only a JSON document; it never executes candidates, reads confirmation data, signs artifacts, writes receipts, or authorizes promotion.

The `experiment.protocol_contract_commit` field identifies the exact protocol/checker implementation commit. It is deliberately not the commit containing the completed preregistration file: embedding that file's own future commit SHA would be self-referential. The resulting document commit is recorded externally by the existing canonical run receipt/attestation after the file is frozen.

## Run

```sh
cargo fmt --manifest-path tools/sym-rsi-r4-prereg-reference/Cargo.toml -- --check
cargo clippy --locked --manifest-path tools/sym-rsi-r4-prereg-reference/Cargo.toml --all-targets -- -D warnings
cargo test --locked --manifest-path tools/sym-rsi-r4-prereg-reference/Cargo.toml
cargo run --locked --manifest-path tools/sym-rsi-r4-prereg-reference/Cargo.toml -- docs/research/templates/SYM_RSI_R4_IGNITION_PREREGISTRATION_V0_2026-11-06.json
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
- Research-state continuity binds record classes, cold-restart restore, successor migration, independent recovery audit, and full/no-state/untyped/shuffled/accepted-only plus fixed/adaptive/shuffled/oracle scope-routing ablations. It measures recovery quality, false sharing/fragmentation and downstream research utility rather than memory volume.

## Canonical claim selection

The `experiment.claim_profile.selected_claims` field is `none` or a comma-separated list of the exact enabled claim-flag names. The semantic validator checks that it matches all eight boolean claim flags and rejects unknown/duplicate names or missing/non-boolean flags. This prevents a document from presenting conflicting claim intent in its human-readable and machine-readable fields.


## Canonical inclusive-budget commitment (v1)

The semantic checker recomputes `budget.shared_envelope_digest`; it does not merely compare two copied strings. The lowercase SHA-256 is over compact serde JSON for the fixed-order v1 struct:

- domain: `symthaea.r4.inclusive-budget-envelope`
- schema version: `1`
- accounting boundary, total budget and currency/price basis
- all twelve per-arm caps in the schema's fixed field order
- failure-accounting policy
- the four inclusive/same-envelope booleans

The two arm bindings must each equal this recomputed digest. The canonical struct is intentionally fixed-field rather than a free-form JSON map, so field ordering is stable. Changing any budget or cap value without recomputing the commitment is rejected. A valid digest still authenticates neither its source nor the actual realized costs; canonical receipts and independent accounting remain authoritative.

All Rust commands use the committed `Cargo.lock` and `--locked`. The dedicated Rust 1.96 workflow is a focused validator gate, not a full workspace or R4 experiment result.

## Important limits

A syntactically valid digest is not an authenticated digest. Distinct split commitments and a declared zero-overlap audit are not proof that the underlying task/family populations are disjoint. The independent custodian must inspect the actual manifests and retain an audit artifact. Likewise, a shared evaluator identity declaration does not prove evaluator independence or prevent hidden channels; the existing evaluator, isolation, receipt, attestation and promotion authorities remain controlling.

A ready report means only that the document is internally consistent enough for independent review. It is not a preregistration approval, evidence of experiment execution, evidence of an R4 result, or authorization to start a run.
