# Mycelix-Core – Validated Claims Index

**Status**: Living document (Jan 2026)  
**Purpose**: Map public claims to concrete tests, benchmarks, and source artifacts.

---

## Federated Learning & Byzantine Tolerance

| Claim | Conditions | Evidence | Status |
|-------|------------|----------|--------|
| 100% detection at ≤33% Byzantine | IID data, ≤33% malicious nodes | 0TML tests (e.g. `0TML/tests/test_30_bft_validation.py`), `docs/CRITICAL_REVIEW_AND_ROADMAP.md` | ✅ Validated |
| ~87.5% detection at 40% Byzantine | IID data, 40% malicious nodes | 0TML tests (e.g. `0TML/tests/test_40_bft_validation.py`), `docs/CRITICAL_REVIEW_AND_ROADMAP.md` | ✅ Measured |
| 45% Byzantine tolerance | 5/11 malicious nodes (≈45.5%) | 0TML FL experiments (`0TML/experiments/**`), narrative in `docs/CRITICAL_REVIEW_AND_ROADMAP.md` | ✅ Demonstrated (Python/0TML) |
| 100 rounds stability | 20 nodes, 33% Byzantine | 0TML long‑run tests (`0TML/tests/**`, session logs in `0TML/SESSION_STATUS_2025-10-28.md`) | ✅ Validated |

> **Note**: As of Jan 2026, the strongest 45% BFT evidence lives in the Python/0TML + canonical Holochain hApps (`0TML/holochain`, `0TML/mycelix_fl`). The Holochain/Rust zome interfaces (`aggregate_gradients`, `classify_round_participants`, `evaluate_round_security`) are fully specified in `docs/HOLOCHAIN_FL_AGGREGATION_DESIGN.md` but not yet implemented end‑to‑end.

---

## Rust FL Aggregator (libs/fl-aggregator)

| Claim | Conditions | Evidence | Status |
|-------|------------|----------|--------|
| Krum / MultiKrum / Median / TrimmedMean implemented correctly | Synthetic scenarios, multiple attack patterns | `libs/fl-aggregator/tests/byzantine_tests.rs` | ✅ Unit tested |
| FedAvg & asynchronous aggregation behave correctly | Reasonable node counts, dimension checks | `libs/fl-aggregator/tests/aggregator_tests.rs`, inline tests in `aggregator.rs` | ✅ Unit tested |
| Robustness at 33–40% Byzantine ratio | Synthetic gradients with scaling, sign‑flip, noise attacks | `libs/fl-aggregator/tests/byzantine_tests.rs` (33% & 40% sections) | ✅ Behavior documented in Rust tests |
| NaN / Infinity / dimension mismatch rejected | Any malformed gradient input | Validation logic in `libs/fl-aggregator/src/aggregator.rs`, tests in `aggregator_tests.rs` | ✅ Enforced |

Latency and scalability benchmarks (FedAvg, Median, TrimmedMean, Krum) are run via Criterion and summarized in:

- `libs/fl-aggregator/benches/**`
- `Mycelix-Core/README.md` performance tables
- `docs/CRITICAL_REVIEW_AND_ROADMAP.md` benchmark section

---

## Holochain & hApps

| Claim | Scope | Evidence | Status |
|-------|-------|----------|--------|
| Canonical Byzantine Defense hApp | `0TML/holochain/` | hApp bundles in `0TML/holochain/dist/byzantine_defense.happ`, audit in `docs/ZOME_AND_HAPP_AUDIT.md` | ✅ Canonicalised |
| Canonical Mycelix FL hApp | `0TML/mycelix_fl/holochain/` | hApp bundle `0TML/mycelix_fl/holochain/workdir/mycelix_fl.happ`, audit in `docs/ZOME_AND_HAPP_AUDIT.md` | ✅ Canonicalised |
| Legacy FL hApps archived | Root `dnas/**`, older FL projects | `.archive/legacy-fl-2025-12-31/ARCHIVE_LOG.md`, `docs/ZOME_AND_HAPP_AUDIT.md` | ✅ Archived |
| Agents zome hardened (validation + rate limiting) | `zomes/agents` | Implementation in `zomes/agents/src/lib.rs` (ID validation, reputation bounds, path‑based rate‑limits) | ✅ Implemented, ⚠️ under‑tested |
| FL zome missing `aggregate_gradients` | Canonical FL zomes | Gap and design captured in `docs/CRITICAL_REVIEW_AND_ROADMAP.md` and `docs/HOLOCHAIN_FL_AGGREGATION_DESIGN.md` | ⚠️ Design‑only (implementation pending) |

---

## SDKs & Ecosystem

| Claim | Component | Evidence | Status |
|-------|-----------|----------|--------|
| MATL implemented in Rust | `mycelix-workspace/sdk` | `sdk/src/matl/**`, tests in `sdk/src/matl/*.rs` and `sdk/tests/happ_integration_tests.rs` | ✅ Implemented & tested |
| MATL implemented in TypeScript | `mycelix-workspace/sdk-ts` | `sdk-ts/src/matl/**`, tests in `sdk-ts/tests/matl.test.ts` & mutation-killer suites | ✅ Implemented & heavily tested |
| Epistemic Charter v2.0 types | Rust + TS SDKs | `sdk/src/epistemic/**`, `sdk-ts/src/epistemic/**`, tests in both repos | ✅ Implemented & tested |
| Cross-hApp reputation aggregation | Bridge protocol & SDK | `sdk/src/bridge/**`, `sdk-ts/tests/bridge*.test.ts`, `tests/ecosystem/tests/reputation-aggregation.test.ts` | ✅ Tested at SDK level |

---

## How to Use This Document

- **For reviewers/investors**: this file is the fastest way to see which claims are backed by code and tests, and which are design‑phase only.
- **For contributors**: when adding a new claim (e.g. “secure aggregation”, “zkSTARK proofs”), add a row here and link it to:
  - Specific test files or benchmarks, and
  - Specific docs or design notes.

As implementation progresses (especially the Holochain `aggregate_gradients` path), this document should be updated alongside `docs/CRITICAL_REVIEW_AND_ROADMAP.md` so that narrative and reality remain aligned. 

