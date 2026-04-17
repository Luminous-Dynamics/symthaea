# Mycelix Improvement Plan

*Drafted 2026-04-17 after deep review. Corrects several errors in the initial Explore-agent review.*

## Ground Truth (verified, not inferred)

| Claim from initial review | Reality |
|---|---|
| hearth-bridge-integrity has compile errors | **CLEAN.** 278 warnings from 4D→8D migration, zero errors |
| commons care-circles-integrity has compile errors | **CLEAN.** Initial check ran `cargo` outside a workspace |
| mycelix-finance has "warnings but functional" | **12 real errors** in zomes/payments + zomes/bridge — wire-types stub drift |
| Unified hApp wires 14 of 18 claimed roles | **19 roles wired**; 4 omitted by design (marketplace, space, mail, desci) |
| TS SDK is a skeletal 7-file stub | **Real SDK is in mycelix-workspace/sdk-ts** (217 test files); top-level `mycelix-sdk-ts/` is an orphan 965-LOC stub |
| SDK claims (226K LOC, 6K+ tests) don't match | **Plausible.** `mycelix-workspace/sdk*/` totals ~513K LOC, 1,493 Rust tests, 217 TS test files |

Net: the codebase is in substantially better shape than the initial review suggested.

---

## Priority 0 — Real issues

### P0.1 — Finance wire-types schema drift (BLOCKER, do not touch from concurrent session)

**Symptom**: `cd mycelix-finance && cargo check --workspace` produces 12 errors in:
- `zomes/payments/coordinator/src/lib.rs` — references 5 missing fields on `SapBalanceResponse` (`member_did`, `raw_balance`, `effective_balance`, `pending_demurrage`, `last_demurrage_at`) and 3 missing types (`ApplyDemurrageInput`, `DemurrageResult`, `MintSapFromGovernanceInput`)
- `zomes/bridge/coordinator/src/lib.rs` — references 6 missing types (`BalanceResponse`, `DepositCollateralInput`, `FeeTierResponse`, `FinanceBridgeHealth`, `ProcessPaymentInput`, `UpdateCollateralHealthInput`) and missing fields on `RegisterCollateralInput`

**Cause**: Commit `9effae6f9b fix(finance): restore workspace — remove ghost members, create wire-types stub` created a partial stub. Commit `9145f03f42 feat(finance,frontends): complete wire-types + radar in 5 more apps` started filling it in. The fill-in is incomplete.

**Recommendation**: **Coordinate before editing.** `finance-wire-types` is a shared type crate — exactly what `.claude/rules/CONCURRENT_SESSIONS.md` flags as high collision risk. When owned solo: add missing fields to `SapBalanceResponse`, export missing `*Input`/`*Response` types from `finance-wire-types/src/lib.rs`, re-run `cargo check --workspace` until green. The Str-size errors (`input.local_commons_pool_id`) suggest the field type should be `Option<String>` not `Option<str>` — probably a typo in the stub.

### P0.2 — Clarify orphan top-level `mycelix-sdk-ts/`

**State**: `/srv/luminous-dynamics/mycelix-sdk-ts/` is a separate `@mycelix/sdk v0.1.0` TypeScript package with 4 source files (965 LOC total: index, sovereign-profile, sovereignty, types). It is **not** the SDK referenced in CLAUDE.md claims.

**Options**:
1. If it's a clean-slate rewrite-in-progress: rename to `mycelix-sdk-ts-v2/` and add a README explaining relationship to `mycelix-workspace/sdk-ts/`
2. If it's an abandoned earlier attempt: delete it
3. If it's a published npm facade that re-exports from workspace: wire it properly

**Recommendation**: Low priority but resolve within 2 weeks — currently misleading.

---

## Priority 1 — Documentation drift (affects new contributors)

### P1.1 — CLAUDE.md SDK path is ambiguous

The line "SDKs: Rust (18 modules, ~50K LOC, 1,036+ tests), TypeScript (37 modules, ~226K LOC, 6,316 tests), Python, WASM" doesn't say **where**. New contributors (and AI agents) land on the top-level stub.

**Fix**: Single-line update to CLAUDE.md — add `mycelix-workspace/sdk{,-ts,-python,-wasm}/` path.

### P1.2 — Unified hApp exclusions should be explicit

The manifest header comment lists included clusters but doesn't say why marketplace, space, mail, desci are excluded. Readers have to infer.

**Fix**: Add one paragraph to `mycelix-workspace/happs/mycelix-unified-happ.yaml` header comment explaining the exclusion policy (likely: DeSci is REST-only, Mail/Marketplace/Space are standalone hApps not yet ready for cross-cluster calls).

---

## Priority 2 — Migration debt (clean up after stability)

### P2.1 — 278 deprecation warnings in bridge-common

`mycelix-bridge-common` has 278 warnings (35 duplicates) from the 4D→8D consciousness profile migration. Every use of `ConsciousnessCredential::profile`, `::tier`, `::issued_at` and `GovernanceRequirement::{min_tier,min_identity,min_community}` is deprecated in favor of `SovereignCredential` and `CivicRequirement`.

**Not blocking**, but:
- Hides real warnings under noise
- Signals migration is incomplete
- Three internal sites inside bridge-common itself still use deprecated fields: `sovereign_gate.rs:43-44,82-84`, `offline_credential.rs:111,123,137,144,145,158,161,169,172,175`

**Fix plan**:
1. Migrate the 13 internal bridge-common call sites first (they're in the same crate — same compile unit, low risk)
2. Audit downstream clusters — likely hearth, commons, civic still call deprecated APIs
3. Once ≤10 warnings, delete the `#[deprecated]` shims entirely

Estimated effort: ~1 day of focused work. Must be done from a **single session** (shared crate).

### P2.2 — Snake-case warning for `pub mod ExtensionKey`

`mycelix-bridge-common/src/consciousness_profile.rs:353` uses CamelCase for a module name. Either rename to `extension_key` and expose re-exports, or `#[allow(non_snake_case)]` with a comment.

---

## Priority 3 — Observability & CI

### P3.1 — No workspace-wide compile check in CI

`mycelix-finance` broken state reached `main` unnoticed. No CI catches `cargo check --workspace` per cluster.

**Recommendation**: Add a GitHub Actions workflow in the standalone mycelix repo (not the private monorepo — Rule #7) that runs `cargo check --workspace` for each cluster that publishes. Even a nightly cron job would have caught the finance drift within 24h.

### P3.2 — Test counts are hard to verify

CLAUDE.md reports specific test counts per cluster ("Commons: 5,276 tests"). Verifying these requires running the full workspace. Consider a `just test-counts` recipe that emits a table.

---

## Priority 4 — Architecture (longer horizon)

### P4.1 — Fractal completion: finish the 4 orphan clusters

Marketplace, Space, Mail, DeSci are "Built" but don't participate in the unified hApp. Either:
- **Promote**: integrate into unified hApp with bridge wiring + routing_registry entries
- **Explicitly demote**: move to a separate `mycelix-standalone/` grouping and document them as non-fractal

Decide by June 2026. Current ambiguity creates discoverability noise.

### P4.2 — Routing registry test coverage

`routing_registry.rs` has 13 routes and 35 tests — good ratio, but some edge cases (unknown cluster, malformed role name, cycle detection) aren't obvious. Worth a property-test sweep in the next bridge-common refactor.

---

## Execution Order

1. **Now (this session)**: This plan document ← you are here
2. **Next session owning finance**: Fix P0.1 (wire-types drift)
3. **Anyone, isolated**: P1.1 + P1.2 (doc edits, low risk)
4. **Dedicated bridge-common session**: P2.1 (deprecation cleanup)
5. **Organizational**: P3.1 (CI), P4.1 (fractal decision)

## What I Deliberately Didn't Do

- **Didn't edit finance-wire-types** — 6 concurrent sessions, shared type crate, high collision risk per CONCURRENT_SESSIONS.md
- **Didn't edit bridge-common** — same reason
- **Didn't delete mycelix-sdk-ts orphan** — needs user call on intent

Per CLAUDE.md Rule #8, this plan should be committed so the next session can pick it up.
