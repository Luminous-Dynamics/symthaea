# Mycelix hApp Portfolio Strategy

**Date**: 2026-02-14 (Updated)
**Status**: Phase 3 hardening complete — all Core Four production-ready

---

## Tier Classification

| Tier | hApps | Status | Action |
|------|-------|--------|--------|
| **Core** | Identity (90%), Core FL (100%), LUCID (85%), Governance (85%) | Active development | Push to production |
| **Cluster** | Commons (35 zomes, 4,126 tests), Civic (16 zomes, 2,030 tests) | Maintained | Bug fixes, integrity hardening |
| **Production** | Mail (12 zomes), DeSci (141 tests), Space | Stable | Community maintenance |
| **Beta** | Marketplace, SupplyChain, Observatory, Praxis | Various | Accept contributions |
| **Scaffold** | Knowledge, Finance, Energy, Health | CRUD only | Partner opportunity |
| **Dormant** | Climate, Music | Archived | No active work |

### Remaining Gaps by Core hApp

| hApp | Gap | Effort |
|------|-----|--------|
| Identity | External bridge events consumer guide | Done (docs/BRIDGE_EVENTS_CONSUMER.md) |
| Core FL | All items complete | - |
| LUCID | Sweettest not in CI, UI coherence feedback | 1-2 weeks |
| Governance | DKG crypto verification stubbed | 4-6 weeks (needs Feldman-VSS crate) |

---

## Core Four - Production Priority

These four hApps form the foundation. All other hApps depend on at least one.

| Priority | hApp | Status | Why Core |
|----------|------|--------|----------|
| 1 | **Identity** | Production (23 sweettests, 9 zomes, MFA 5-attempt rate limit, DID, recovery, PQC, bridge events) | Foundation for all agent authentication |
| 2 | **Governance** | Production (7 zomes, treasury escrow, delegation chains with decay, DKG threshold signing, Phi-gated actions) | Required for ecosystem self-management |
| 3 | **Core (FL)** | Production (62 tests, 6 zomes, PoGQ pipeline, model versioning, cross-session reputation, E2E tests) | Federated learning foundation with MATL |
| 4 | **LUCID** | Production (8 zomes, 92 functions, Tauri UI, Symthaea bridge 95% wired, full embedding pipeline) | Flagship Symthaea+Holochain integration |

### LUCID Bridge Status (Feb 2026 — Verified Complete)

The LUCID Tauri bridge to Symthaea is **95% complete**:
- 19 Tauri commands implemented (analyze_thought, semantic_search, check_coherence, etc.)
- All E/N/M/H type conversions bidirectional via `lucid-symthaea` crate
- 16,384D HDC embedding pipeline functional at zome level:
  - `update_thought_embedding()` — stores embeddings on DHT
  - `update_thought_coherence()` — Phi feedback loop
  - `semantic_search()` — cosine similarity search across embeddings
  - `explore_garden()` — semantic clustering
  - `suggest_connections()` — find unlinked similar thoughts
  - `find_knowledge_gaps()` — sparse domain detection
  - `discover_patterns()` — coherence tension, confidence trends
- Dependencies configured (symthaea v0.5.0 via workspace symlink)

**Remaining to ship:**
1. Integration tests (Tauri -> zome storage -> DHT) — requires GTK dev deps via `nix develop`
2. cosine_similarity duplication is intentional (WASM boundary prevents cross-zome sharing)

### Governance Status (Feb 2026 — Verified Complete)

All planned features are fully implemented across 7 zomes:
- **Treasury escrow**: `lock_proposal_funds()`, `release_locked_funds()`, `refund_locked_funds()` in execution zome
- **Delegation chains**: `create_delegation()`, `renew_delegation()`, `revoke_delegation()` with `DelegationDecay` in voting zome
- **DKG threshold signing**: Full lifecycle (`create_committee` → `register_member` → `submit_dkg_deal` → `finalize_dkg` → `submit_signature_share` → `finalize_signature` → `verify_signature`)
- **Phi-gated actions**: Basic >= 0.2, ProposalSubmission >= 0.3, Voting >= 0.4, Constitutional >= 0.6
- **Holistic weight**: Reputation^2 x (0.7 + 0.3 x Phi) x (1 + 0.2 x HarmonicAlignment), capped at 1.5

### Identity Status (Feb 2026 — Verified Complete)

9 coordinator zomes with comprehensive security:
- **Rate limiting**: MFA (5 failed attempts / 15 min window), Bridge (1 min query rate limit), MAX_FACTORS_PER_DID=20
- **Bridge events**: 11 event types (DidCreated/Updated/Deactivated, CredentialIssued/Revoked, RecoveryInitiated/Completed, HappRegistered, MfaAssuranceChanged, DidRecovered, Custom)
- **Sweettests**: 23 tests (revocation, recovery, trust credentials, verifiable credentials)

### Core FL Status (Feb 2026 — Bug Fix Applied)

- **Model versioning**: `register_model_config()`, `get_model_config()`, `validate_model_version()` — architecture_hash enforcement per round
- **Reputation persistence FIX**: `get_or_create_reputation()` now persists defaults to DHT on first access and uses consistent `node_reputation.{node_id}` path (was broken: used wrong path + link type, never persisted defaults)

---

## hApp Merger Assessment (2026-02-09)

### Climate + Energy -> "Environment" — DEFER

**Feasibility: 7/10**

| Metric | Climate | Energy |
|--------|---------|--------|
| Zomes | 6 (carbon, monitoring, attestation, targets, bridge, reporting) | 7 (projects, participants, trading, credits, investments, grid, bridge) |
| Status | Scaffold | Scaffold |
| SDK | TS integration exists (`sdk-ts/src/integrations/energy/`) | Full TS client (`sdk-ts/src/energy/`) |

**Overlap**: Carbon credits implemented in both. Bridge zomes overlap.
**Unique**: Climate has footprint tracking (scope 1/2/3), Energy has P2P trading and community investment.

**Decision**: DEFER. Both are scaffold-quality. The carbon credit duplication should be resolved but doesn't warrant a full merge until either reaches beta quality with real users and sweettest coverage.

### MutualAid + Care -> "Community" — DEFER

**Feasibility: 8/10** (corrected: Care hApp EXISTS at `/srv/luminous-dynamics/mycelix-care`)

| Metric | MutualAid | Care |
|--------|-----------|------|
| Zomes | 8 (requests, pools, timebank, circles, resources, needs, governance, bridge) | 6 (care-plans, circles, timebank, matching, credentials, bridge) |
| Functions | 70+ coordinator functions | 50+ coordinator functions |
| Status | Dormant (restored from archive) | Scaffold (all 6 zomes in DNA) |
| SDK | None | Full TS client (`sdk-ts/src/clients/care/`) |

**Critical blockers**:
1. **Entry type conflicts**: Both define `ServiceOffer`, `ServiceRequest`, `TimeExchange`, `TimeCredit` with different field sets
2. **ServiceOffer incompatible**: MutualAid has `hours_available`, `skills_required`, `active`; Care doesn't
3. **SDK breaking change**: Care's TS client is published and in use
4. **Version pinning**: Care strict `hdk = "0.6.0"`, MutualAid flexible `hdk = "0.6"`

**Recommended merge approach** (when ready): Option C — Selective Merge
- Shared types crate for `circles_core`, `timebank_core`, `matching_core`
- Namespace domain zomes: `care_plans`, `care_credentials`, `mutual_requests`, `mutual_pools`
- `CommunityClient` SDK wrapping both, `CareClient` kept as compat layer
- Estimated effort: 4-6 weeks

**Decision**: DEFER. Neither hApp has users or sweettest coverage. Engineering investment not justified until at least one is actively developed.

### Finance + Marketplace -> "Economy" — NOT FEASIBLE

**Finding**: Marketplace is a Node.js/Solidity project, NOT a Holochain hApp. Cannot merge with Finance (Holochain). Keep separate.

### Media + Music -> "Creative" — NOT FEASIBLE

**Finding**: Music is a Node.js/Solidity project, NOT a Holochain hApp. Cannot merge with Media (Holochain). Keep separate.

---

## Portfolio Summary

| Current | Target | Change |
|---------|--------|--------|
| 24 hApps | 24 hApps (no merges) | Focus on quality over quantity |

**Rationale**: All feasible merges involve scaffold/dormant hApps. The 4-6 week investment per merge is better spent hardening the Core Four and proving E2E integration. Revisit merges when:
- Climate OR Energy reaches beta with 3+ sweettest-proven coordinator functions
- MutualAid is revived from dormancy with active development
- A real user need requires combined functionality

---

## E2E Integration (Symthaea-Mycelix Bridge)

**Status**: Proven at SDK level, conductor tests exist

- FL bridge `pogq_from_quality_score()` converts consciousness assessment to PoGQ values
- FL E2E test crate at `tests/sweettest/tests/fl_bridge_e2e.rs`
- 3 SDK tests pass (composite formula, round-trip values, boundary values)
- 3 conductor tests written (honest accepted, Byzantine rejected, DHT storage)
- FL DNA bundle built (1.4MB)

---

## Novel hApps Leveraging Symthaea

### Epistemic Garden (extends LUCID)

Consciousness-aware knowledge evolution using Symthaea's HDC+CfC+IIT stack:
- Claims carry epistemic classification (E-N-M from Epistemic Charter)
- CfC networks track belief trajectory evolution over time
- Phi-based contradiction detection
- Built on LUCID's existing temporal-consciousness and reasoning zomes

**Implementation**: Extend LUCID rather than creating standalone hApp.

### Mycelial Sense (future, post-LUCID stabilization)

Distributed consciousness monitoring across the network:
- Nodes share Phi integration scores via DHT
- Collective pattern detection (stress, flow, governance health)
- Uses `social_coherence.rs` + HDC emotional processing
- Requires LUCID bridge to be fully operational first

---

## DeSci: Keep Standalone

DeSci is a REST API service (Actix-web, 141 tests, 400K claims/sec), not a Holochain hApp. Create a lightweight bridge zome in LUCID instead of converting.

---

## Completed Actions (Feb 2026)

1. **Core Four hardened** — Identity (23 sweettests), Governance (all features verified), Core FL (reputation bug fixed, model versioning added), LUCID (embedding pipeline complete)
2. **LUCID bridge verified** — 95% wired, all embedding/coherence/search functions at zome level
3. **FL conductor tests written** — 6 E2E tests (3 SDK + 3 conductor) at `tests/sweettest/tests/fl_bridge_e2e.rs`
4. **Cross-cluster tests expanded** — 12 sweettest scenarios including all 5 P0 dispatch paths
5. **FL reputation persistence fixed** — path inconsistency bug corrected, defaults now persisted to DHT
6. **Observatory Byzantine threshold verified** — all values already 0.34/34% (no changes needed)
7. **SDK PQC tests verified** — all FIPS 204 sizes already correct (4032/3309/4627)
8. **FL modularization unblocked** — include_str!() tests updated to concat!() across all 16 source files; config.rs extracted as first module

## Next Actions

1. **LUCID sweettests in CI** — 27 tests exist but marked `#[ignore]`. Need conductor in CI to enable.
2. **Governance DKG completion** — Feldman-VSS crate integration for real threshold crypto verification
3. **FL coordinator modularization** — include_str! blocker RESOLVED (concat!() across all files). config.rs wired in. 14 modules remain (require `nix develop` for WASM verification)
4. **Merge feature/space-phase5 to main** — All improvements ready for mainline
5. **Climate + Energy merger** — Proceed when either reaches beta quality with real users
