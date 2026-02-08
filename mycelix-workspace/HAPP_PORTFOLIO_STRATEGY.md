# Mycelix hApp Portfolio Strategy

**Date**: 2026-02-08
**Status**: Approved plan, pending implementation

---

## Core Four - Production Priority

These four hApps form the foundation. All other hApps depend on at least one.

| Priority | hApp | Status | Why Core |
|----------|------|--------|----------|
| 1 | **Identity** | Most mature (15 sweettests, MFA, DID, recovery, Ed25519 hardened) | Foundation for all agent authentication |
| 2 | **Governance** | Beta (3 sweettests, proposals, voting, delegation) | Required for ecosystem self-management |
| 3 | **Core (FL)** | Production (62 tests, 6 zomes, PoGQ pipeline) | Federated learning foundation with MATL |
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

## hApp Merger Decisions

### Merge: Climate + Energy -> "Environment" (PROCEED)

**Feasibility: 8/10**

| Metric | Climate | Energy | Combined |
|--------|---------|--------|----------|
| Zomes | 3 | 6 | 9 |
| LOC | 89K | 133K | 222K |
| Tests | 7,218 | 10,900 | 18,118 |
| Status | Scaffold | Scaffold | Scaffold |

**Rationale**: Complementary domains (carbon credits + renewable energy certificates), both restored from archive, natural market overlap. Strongest merge candidate.

**Implementation**: Create unified `shared` zome with common types (Project, Verification, Certificate), add `project_type` enum (Carbon, Renewable, WaterConservation).

### Defer: Finance + Marketplace -> "Economy"

**Feasibility: 6/10**

Finance (56K LOC, 3,530 tests) handles debt instruments (loans, credit scoring, treasury). Marketplace (94K LOC, 7,170 tests) handles commodity sales (listings, arbitration, reputation). Different domain semantics make premature merging risky. **Defer until Climate+Energy merge stabilizes.**

### Do Not Merge: Media + Music -> "Creative"

**Feasibility: 3/10**

Music has 0 traditional zomes (uses Solidity contracts), 153 tests vs Media's 5,504. Technology mismatch (hybrid Ethereum/Holochain vs native Holochain). Would destabilize Media. **Keep separate; refactor Music to HDK zomes first.**

### Blocked: MutualAid + Care -> "Community"

Care hApp **does not exist** (listed as scaffold in status docs but never created). MutualAid is solid standalone (8 zomes, 49K LOC, 3,489 tests). **Create Care independently if needed; do not force merge.**

---

## hApp Count Projection

| Phase | Count | Change |
|-------|-------|--------|
| Current | 22 hApps | - |
| After Climate+Energy merge | 21 hApps | -1 |
| After Music HDK refactor | 21 hApps | 0 (quality improvement) |

The original target of 18->14 was overly aggressive. **21 hApps** is the realistic near-term target, with potential for further consolidation after the Finance+Marketplace and Music maturation.

---

## Novel hApps Leveraging Symthaea

### Epistemic Garden (replaces Knowledge scaffold)

Consciousness-aware knowledge evolution using Symthaea's HDC+CfC+IIT stack:
- Claims carry epistemic classification (E-N-M from Epistemic Charter)
- CfC networks track belief trajectory evolution over time
- Phi-based contradiction detection
- Built on LUCID's existing temporal-consciousness and reasoning zomes

**Implementation**: Extend LUCID rather than creating standalone hApp. The temporal-consciousness zome already stores BeliefSnapshots with phi, coherence, and epistemic_code.

### Mycelial Sense (future, post-LUCID stabilization)

Distributed consciousness monitoring across the network:
- Nodes share Phi integration scores via DHT
- Collective pattern detection (stress, flow, governance health)
- Uses `social_coherence.rs` + HDC emotional processing
- Requires LUCID bridge to be fully operational first

---

## DeSci: Keep Standalone

DeSci is a REST API service (Actix-web, 141 tests, 400K claims/sec), not a Holochain hApp. Create a lightweight bridge zome in LUCID instead of converting. This preserves DeSci's performance characteristics while enabling DHT-based claim verification.

---

## Immediate Actions

1. **Complete LUCID bridge verification** - test Symthaea v0.5.0 API compatibility
2. **Begin Climate+Energy merge** - create `environment` workspace, unified types
3. **Update ECOSYSTEM_STATUS.md** - reflect Core Four priority and merger decisions
4. **Sweettest expansion** - add LUCID integration tests to sweettest suite

---

## Success Criteria

- [ ] LUCID bridge compiles and runs with Symthaea v0.5.0
- [ ] Climate+Energy merged into "Environment" hApp
- [ ] Core Four hApps all have sweettest coverage
- [ ] ECOSYSTEM_STATUS.md reflects updated portfolio
