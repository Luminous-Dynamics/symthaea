# Mycelix Improvement Plan

*Drafted 2026-04-17. Revised after verified `cargo check --workspace` across all 26 Rust clusters. Updated with fix progress 2026-04-17.*

## Session Progress (2026-04-17)

Committed fixes:
| Commit | Target | What |
|---|---|---|
| `4ca01a5938` | CLAUDE.md + unified hApp manifest | P1.1 + P1.2 doc fixes |
| `f93b19b1bd` | mycelix-energy | Moved `resolver`/`members` from `[workspace.package]` to `[workspace]` |
| `a42521542f` | mycelix-personal | Added missing `license` to `[workspace.package]` |
| `cf209373b7` | mycelix-desci | Removed dangling `pub mod reproducibility_engine` |
| `d22016222d` | mycelix-civic integrity | Added `RoboticCredential` match arm |
| `dc69f6b716` | mycelix-energy projects | Initialized 4 consciousness-scoring fields as `None` |
| `2330e52d89` | mycelix-conductor-bridge | Pass `None` for new `origin` arg in `AppWebsocket::connect` |
| `a6ba298d9d` | mycelix-domain-template | Sync trait/enum/struct to current `sensorium_domain_trait` |
| `d747944f48` | mycelix-civic coordinator | HDK 0.6 migration: `agent_initial_pubkey` + `LinkQuery` |

Net compile-check delta: **12 PASS → 18 PASS** (of 26 Rust clusters).

Still FAIL (real issues, require coordination or investigation):
- **governance** — shared-crate drift (feldman-dkg `SigningKey::verifying_key`, `ConsciousnessAttestation::validate_with_freshness`, voting `CastAttestedVoteInput.consciousness_attestation_json`). Multiple touchpoints in `mycelix-core/libs/feldman-dkg/` and `mycelix-bridge-common`.
- **finance** — 21 errors in payments + finance_bridge from shared `finance-wire-types` stub. P0.3 blocker.
- **multiworld-sim** — `ExecutiveSummary`, `StandardizedReport`, `ReproducibilityInfo` restructured in `crates/luminous-sim-core/`; consumer uses richer schema. Data-model judgment call.
- **supplychain** — nested workspaces: outer `Cargo.toml` lacks `[workspace.dependencies]`; inner `holochain/Cargo.toml` has them but is shadowed. Requires workspace-structure decision.

Still FAIL (environment issues, not code):
- **craft** — `datachannel-sys` needs libdatachannel C dep (run inside `nix develop`)
- **music-desktop** — `wayland-sys` needs libwayland (same)
- **identity, knowledge** — `sccache rustc -vV` fails; likely stale sccache cache


## Compile Health (verified 2026-04-17)

### PASS — 12 clusters
atlas, attribution, climate, commons, core, health, hearth, manufacturing, portal, position, praxis, space

### FAIL — 14 clusters, split by root cause

**A. Real code errors (6 clusters, drift from shared types):**

| Cluster | Error | Fix surface |
|---|---|---|
| **civic** | `E0004: non-exhaustive patterns: EntryTypes::RoboticCredential(_) not covered` | Add match arm in `robotics_dispatch_integrity` |
| **desci** | `E0583: file not found for module 'reproducibility_engine'` | Create file or remove `mod` decl |
| **domain-template** | Trait method `default_activity` missing + `EntryTypeInfo.entry_type` + `DataSensitivity::Internal` | Sync template to current `sensorium_domain_trait` API |
| **finance** | `payments` (12 errors) + `finance_bridge` (9 errors) — wire-types stub drift | Fill `finance-wire-types` stub |
| **governance** | `feldman-dkg`: `validate_with_freshness`, `verifying_key`, `CastAttestedVoteInput.consciousness_attestation_json` missing | API alignment with `ConsciousnessAttestation` + `SigningKey` |
| **multiworld-sim** | `ExecutiveSummary` missing `critical_events`, `final_cvs`, `final_population`, `worlds_surviving` | Consumer out of sync with type def |

**B. Build-environment issues (5 clusters, NixOS/deps):**

| Cluster | Error | Likely fix |
|---|---|---|
| **conductor-bridge** | `failed to select a version for serde` | Version pin / Cargo.lock rebuild |
| **craft** | `datachannel-sys v0.23.0+0.23.2` custom build failed | Needs C deps (libdatachannel); enter `nix develop` |
| **music-desktop** | `wayland-sys v0.31.11` custom build failed | Needs libwayland in PATH |
| **identity** | `sccache rustc -vV exit status: 2` | Stale sccache state; clear `~/.cache/sccache` or rebuild without sccache |
| **knowledge** | same sccache failure | same fix |

**C. Workspace manifest issues (3 clusters, structural):**

| Cluster | Error | Fix |
|---|---|---|
| **energy** | `virtual manifest, workspace has no members` | Root `Cargo.toml` members list is empty |
| **personal** | `failed to load manifest for workspace member personal-types` | Missing/broken `personal-types/Cargo.toml` |
| **supplychain** | `failed to load manifest for procurement/integrity` | Same pattern |

---

## What the initial review got wrong

| Initial claim | Reality |
|---|---|
| hearth-bridge-integrity has compile errors | **CLEAN.** 267 warnings from 4D→8D migration, zero errors |
| commons care-circles-integrity has compile errors | **CLEAN.** 256 warnings, zero errors |
| mycelix-finance has "warnings but functional" | **21 errors.** Wire-types stub drift across `payments` + `finance_bridge` |
| Only finance is blocked | **6 clusters have real code errors**, 5 have env issues, 3 have manifest issues |
| Unified hApp wires 14 of 18 claimed roles | **19 roles wired**; 4 omitted by design (marketplace, space, mail, desci) |
| TS SDK is a skeletal 7-file stub | Real SDK in `mycelix-workspace/sdk-ts/` (217 test files); top-level `mycelix-sdk-ts/` is a 965-LOC orphan |

---

## Priority 0 — Real compile blockers

### P0.1 — Drift cluster: pattern match in civic

**File**: `mycelix-civic/zomes/robotics-dispatch/integrity/src/lib.rs` (or similar)
**Error**: `EntryTypes::RoboticCredential(_) not covered`
**Effort**: ~5 min. Single match arm.
**Risk**: Low. Integrity zome change needs WASM rebuild.

### P0.2 — Missing module file in desci

**File**: Somewhere a `mod reproducibility_engine;` declaration lacks a backing `.rs` file.
**Effort**: ~10 min. Either create stub or remove decl.
**Risk**: Low.

### P0.3 — Finance wire-types stub (biggest blocker, 21 errors)

Same as before: `finance-wire-types/src/lib.rs` is a partial stub. `payments` and `finance_bridge` zomes import missing types and fields. **Active session-coordination required** before editing shared crate.

### P0.4 — Governance DKG API drift

`feldman-dkg` zome expects APIs on `ConsciousnessAttestation` and `SigningKey` that no longer exist. Either:
- Upstream regressed (restore methods)
- Zome is stale (update calls)

Needs investigation, probably 1-2 hours.

### P0.5 — Multiworld-sim `ExecutiveSummary` struct drift

4 renamed/removed fields. Consumer at `mycelix-multiworld-sim` needs update to current struct shape. ~30 min.

### P0.6 — Domain template API sync

Template is reference scaffold — if broken, new domain authors will copy broken code. Fix or label `EXPERIMENTAL`.

---

## Priority 1 — Environment/config fixes (not code bugs)

### P1.1 — sccache stale for identity + knowledge

Both fail identically at `sccache rustc -vV exit status: 2`. Likely `~/.cache/sccache` corruption from a killed compile.

**Fix**: `SCCACHE_STOP_SERVER=1 sccache --stop-server; rm -rf ~/.cache/sccache/*; sccache --start-server` then retry.

### P1.2 — conductor-bridge serde selection

Cargo can't pick a `serde` version. Probably two crates pinning incompatible versions. Fix with `cargo tree -i serde` + alignment pass.

### P1.3 — NixOS deps for craft + music-desktop

`datachannel-sys` and `wayland-sys` need system libraries. Either:
- Add to `nix develop` shell (`flake.nix`)
- Document: "run this cluster inside `nix develop`"

### P1.4 — Workspace manifest fixes (energy, personal, supplychain)

3 clusters have broken `Cargo.toml` members. One-line fixes each:
- `energy`: populate `[workspace]` members
- `personal`: fix `personal-types/Cargo.toml` (likely missing `[package]`)
- `supplychain`: same for `procurement/integrity`

---

## Priority 2 — Migration debt

### P2.1 — 4D→8D deprecation warnings

256 warnings in commons, 267 in hearth, 251 in finance. Noise hiding real warnings.

**Plan**: One dedicated session fixes `mycelix-bridge-common`'s own internal call sites (13 locations), then downstream clusters. ~1 day focused work, must be single-session (shared crate).

### P2.2 — CamelCase module warning

`mycelix-bridge-common/src/consciousness_profile.rs:353` — `pub mod ExtensionKey` should be `extension_key` or `#[allow(non_snake_case)]`.

### P2.3 — Clean up orphan `mycelix-sdk-ts/`

Top-level `mycelix-sdk-ts/` is a 965-LOC stub separate from the real SDK at `mycelix-workspace/sdk-ts/`. Decide: delete, rename, or wire properly.

---

## Priority 3 — Infrastructure

### P3.1 — Add nightly cross-cluster `cargo check`

Finance was in a broken state for multiple commits unnoticed. Add a nightly GitHub Actions job (in the **standalone** mycelix repo, not the monorepo — Rule #7) that runs `cargo check --workspace` across all clusters and posts a status table.

### P3.2 — `just test-counts` recipe

Generate verifiable test-count table from `cargo test -- --list` across workspace. Prevents CLAUDE.md claim drift.

---

## Priority 4 — Architecture

### P4.1 — Complete or explicitly exclude the 4 orphan clusters

Marketplace, Space, Mail, DeSci are "Built" but not in unified hApp. Decision needed by June 2026:
- **Promote**: bridge wiring + routing_registry entries
- **Demote**: move to `mycelix-standalone/` grouping

Already documented in manifest header (done in commit `4ca01a5938`).

### P4.2 — Routing registry edge-case coverage

`routing_registry.rs` has 35 tests for 13 routes — good ratio. Worth a property-test sweep for: unknown cluster role, malformed zome name, cycle detection.

---

## Execution Order

| # | Task | Owner | Blocker? |
|---|---|---|---|
| 1 | P1.1 (sccache reset) | Any session | No, unblocks identity/knowledge builds |
| 2 | P0.1 (civic pattern match) | Solo session | Yes for civic tests |
| 3 | P0.2 (desci missing file) | Solo session | Yes for desci |
| 4 | P0.5 (multiworld-sim struct drift) | Solo session | Yes for sim |
| 5 | P1.4 (manifest fixes) | Solo session × 3 | Yes for energy/personal/supplychain |
| 6 | P0.3 (finance wire-types) | **Coordinate** — shared crate | Yes for finance |
| 7 | P0.4 (governance DKG drift) | Needs investigation first | Yes for governance |
| 8 | P0.6 (domain template) | Any | Low priority |
| 9 | P2.1 (deprecation cleanup) | Single dedicated session | No |
| 10 | P3.1 (CI nightly) | Standalone repo | No |

## What I Deliberately Didn't Do This Session

- **Didn't edit any shared crates** (bridge-common, finance-wire-types) — 6 concurrent sessions active, high collision risk
- **Didn't attempt sccache reset** — affects other sessions
- **Didn't delete mycelix-sdk-ts orphan** — needs user call on intent
- **Didn't touch wip/fleet-fallback-and-geodesic-synthesis branch** — another session's working branch

Plan committed on main (8d64c8eaf5). P1.1/P1.2 doc fixes committed (4ca01a5938).
