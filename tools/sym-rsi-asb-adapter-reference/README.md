# Autonomous Science Bench compatibility reference

This standalone offline Rust crate validates the pinned AS-Bench v0.1 compatibility manifest and public Lab API v1 contracts.

## What it checks

- Upstream repository and exact commit pin.
- Ten Git blob identities, including the two private lab campaign-rule files whose public-card projections are checked in as fixtures.
- Exact replay and digital-twin campaign identities, task paths, round/batch arithmetic, discovery and held-out-learning thresholds, reference-result labels, licenses and measurement provenance.
- Public campaign card, campaign status, catalog, submission preflight and experiment-job response contracts.
- Strict separation of agent-visible public campaign metadata from lab-private configuration. In particular, public campaign cards must not contain the private `backend` field.
- Status consistency: completed/active stage progression, active job and budget accounting.
- Catalog design-ID presence, uniqueness, requested stage and pool size; local fail-fast checks for duplicate, ineligible, previously measured and over-budget selections.
- Job lifecycle: `queued` and `running` are pending; `failed` never advances a stage; only a completed job with a complete result batch, matching result-design identities, and every declared measurement present with its declared scalar/file type advances a stage.
- A small offline end-to-end fixture path through campaign, status, catalog, batch preflight and completed-job checks.

The canonical API implementation is `src/api_contract.rs`; job status and result-row lifecycle are shared with `src/job_lifecycle.rs`. There is one API contract owner, not parallel validators. The code is transport-independent and makes no HTTP requests.

## Pinned campaign fixtures

Two exact public-card projections are generated from the pinned upstream campaigns:

- `fixtures/asb_lab_v1/nuclease_campaign_card.json` — replay, 4 × 96 = 384 measurements.
- `fixtures/asb_lab_v1/propylene_campaign_card.json` — twin, 3 × 20 = 60 measurements.

The source task configurations are pinned by their Git blob identities in the compatibility manifest. The twin fixture's provenance remains distinct: the upstream task says its results comprise exact published measurements plus surrogate-model outputs, not live laboratory measurements.

## Run

```bash
cargo fmt --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml
cargo run --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml -- docs/research/adapters/SYM_RSI_ASB_COMPATIBILITY_MANIFEST_V0_2026-11-07.json
cargo run --manifest-path tools/sym-rsi-asb-adapter-reference/Cargo.toml -- --api-fixtures
```

There are 38 authored Rust tests: 24 public API contract tests, nine job-lifecycle tests and five manifest tests. The public API tests include missing-measurement, non-numeric scalar, blank file-reference and duplicate result-ID negative controls. They have not been executed in the current environment.

## Authority and claim boundaries

- The remote AS-Bench lab remains authoritative for stage order, eligibility, duplicate prevention, accepted batches, budgets, asynchronous status and the ledger.
- The upstream verifier remains authoritative for hidden truth and scoring. This crate never receives verifier truth.
- Client-side catalog and submission validation are fail-fast checks, not a replacement for server validation.
- The manifest checker validates local compatibility declarations. It does not fetch upstream content or authenticate Git history, execution, task custody or scientific truth.
- Upstream reference scores are not Symthaea results.
- Replay values are published measurements. Twin values are model-mediated; neither is a live-lab result.
- No campaign is run, no benchmark is submitted, and no scientific-discovery or R4 claim is made by these tools.
- The crate is intentionally outside the production Cargo workspace. It introduces no production network client, evaluator, receipt/signing service, scheduler, memory service or promotion authority.
- No Rust compilation, formatting, Clippy, unit-test or CI pass is claimed until those commands execute on a Rust toolchain.
