# Autonomous Science Bench compatibility reference

This standalone offline Rust crate validates the checked-in, pinned AS-Bench compatibility manifest:

`docs/research/adapters/SYM_RSI_ASB_COMPATIBILITY_MANIFEST_V0_2026-11-07.json`

It checks the upstream repository/commit/API/schema pin, ten Git blob identities, exact v0 campaign identities, round × batch budget arithmetic, discovery and held-out-learning gate mappings, upstream reference-result labels, the provenance distinction between replay and digital-twin outputs, typed Lab API v1 campaign/status/catalog/job payloads, and strict campaign request preflight.

It is deliberately **not** a campaign runner, metric evaluator, oracle, receipt/authentication service, scheduler, memory service or promotion authority. A valid manifest proves only that the local compatibility declaration matches this crate's frozen contract.

The `api_v1` module defines transport-independent types for the public campaign card, status, catalogs, experiment requests/responses, job list summaries, deliverables and API errors. It rejects a public campaign card that contains the private `backend` field, checks stage/budget consistency, validates catalog design IDs and duplicate selections, and checks deliverable kind/version/size. The `job_lifecycle` module ensures only a completed job with one result row per requested design can advance a stage. The lab remains authoritative for every server-side rule.

The `job_lifecycle` module models the pinned API v1 statuses `queued`, `running`, `completed` and `failed`. It only exposes usable measurement rows when a job is completed and the result-row count matches the requested design count. Pending and failed jobs never advance a campaign stage. It does not submit requests or decide whether a source measurement is scientifically valid.

The `api_contract` module parses fixture-driven public API payloads and validates campaign identity, schema version, contiguous stages, batch/budget arithmetic, status budget reconciliation, catalog eligibility, duplicate/repeated design controls and result-row identity. Its checks are local preflight; the upstream lab remains authoritative. `--api-fixtures` runs the checked-in public contract examples without network access.

## Run

```bash
cargo fmt --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml
cargo run --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml -- docs/research/adapters/SYM_RSI_ASB_COMPATIBILITY_MANIFEST_V0_2026-11-07.json
cargo run --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml -- --api-fixtures
```

## Claim boundary

- The upstream pin is a reproducibility anchor, not a claim that it is the latest upstream revision.
- The upstream reference scores in the manifest are not Symthaea results.
- Replay results are published measurements; twin results are model-mediated and are not live laboratory results.
- The checker does not fetch upstream content or cryptographically attest the source. File identities are recorded Git blob SHA-1 identifiers from the pinned commit; they do not authenticate a run or establish truth.
- The JSON Schemas are structural. This crate adds semantic checks for cross-field relationships, the two exact campaign contracts, job lifecycle and public API response/request rules. It now has 31 authored Rust tests; they remain unexecuted.
- No Rust compilation, formatting, Clippy, test or CI result is claimed until those commands execute on an available toolchain.
