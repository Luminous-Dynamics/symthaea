# Mycelix hApp Portfolio Strategy

**Date**: 2026-02-09 (Updated)
**Status**: Assessment complete, merges deferred to post-beta

---

## Core Four - Production Priority

These four hApps form the foundation. All other hApps depend on at least one.

| Priority | hApp | Status | Why Core |
|----------|------|--------|----------|
| 1 | **Identity** | Most mature (15 sweettests, MFA, DID, recovery, Ed25519 hardened) | Foundation for all agent authentication |
| 2 | **Governance** | Beta (3 sweettests, proposals, voting, delegation) | Required for ecosystem self-management |
| 3 | **Core (FL)** | Production (62 tests, 6 zomes, PoGQ pipeline, E2E tests exist) | Federated learning foundation with MATL |
| 4 | **LUCID** | Beta (8 zomes, Tauri UI, Symthaea bridge 85% wired) | Flagship Symthaea+Holochain integration |

### LUCID Bridge Status

The LUCID Tauri bridge to Symthaea is **architecturally complete** (85%):
- 19 Tauri commands implemented (analyze_thought, semantic_search, check_coherence, etc.)
- All E/N/M/H type conversions bidirectional via `lucid-symthaea` crate
- 16,384D HDC embedding pipeline functional
- Zomes define correct data structures (embedding, phi, coherence, epistemic_code)
- Dependencies configured (symthaea v0.5.0 via workspace symlink)

**Remaining to ship:**
1. Verify Symthaea v0.5.0 API matches bridge expectations
2. Fix cosine_similarity duplication (collective/coordinator reimplements it)
3. Add integration test suite (Tauri -> zome storage -> DHT)

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

## Immediate Actions

1. **Harden Core Four** - expand sweettest coverage for Identity, Governance, Core FL
2. **Complete LUCID bridge verification** - test Symthaea v0.5.0 API compatibility
3. **Run FL conductor tests** - prove honest/Byzantine gradient acceptance
4. **Update ECOSYSTEM_STATUS.md** - reflect Core Four priority and corrected assessments
