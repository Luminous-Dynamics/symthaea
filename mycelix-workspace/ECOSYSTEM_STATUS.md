# Mycelix Ecosystem Status

**Last verified**: 2026-03-18
**Holochain**: 0.6.0 | **HDK**: 0.6.0 | **HDI**: 0.7.0

## hApp Bundle Status

| hApp | Bundle | Zomes | Manifest | Status |
|------|--------|-------|----------|--------|
| lucid | ✅ 5.6M | 6 | v0 | Built |
| mail | ✅ 18M | 12 | v0 | Built (submodule: holochain/) |
| knowledge | ✅ 5.1M | - | v0 | Built (submodule) |
| identity | ✅ 5.5M | - | v0 | Built (submodule) |
| justice | ✅ 4.4M | - | v0 | Built (submodule) |
| governance | ✅ 4.3M | - | v0 | Built (submodule) |
| finance | ✅ 3.4M | - | v0 | Built (submodule) |
| marketplace | ✅ 3.3M | 8 | v0 | Built (submodule: backend/) |
| epistemic-markets | ✅ 3.2M | - | v0 | Built |
| fabrication | ✅ 7.4M | 6 | v0 | Built |
| supplychain | ✅ 1.9M | 8 | v0 | Built (submodule: holochain/) |
| praxis | ✅ 948K | 10 | v0 | Built (submodule: happ/) |
| health | ✅ 8.4M | 7 (MVP) | v0/v1 | ✅ MVP Core (22 zomes archived 2026-02-15) |
| energy | ✅ | 11 | v0 | ✅ Scaffolded |
| climate | ⏳ | 6 | v0 | ✅ Scaffolded (workdir/) |
| mutualaid | ⏳ | 6 | v0 | ✅ Scaffolded (workdir/) |
| property | ✅ | 9 | v0 | ✅ Scaffolded |
| media | ✅ | 8 | v0 | ✅ Scaffolded |
| consensus | ⏳ | 1 | v0 | ✅ Scaffolded (native workspace) |
| music | ⏳ | 8 | v0 | ✅ Scaffolded |
| food | ⏳ | 4 | v0 | ✅ Scaffolded (Commons cluster) |
| transport | ⏳ | 3 | v0 | ✅ Scaffolded (Commons cluster) |
| core | N/A | 6 | - | REST API (not hApp) |
| desci | N/A | - | - | REST API (not hApp) |

**Legend**: ✅ = Bundle exists | ⏳ = Scaffolded, needs WASM build | v0 = manifest_version "0" (current format)

**Note**: Desktop moved to `tools/desktop/` - it's a Tauri framework, not a hApp.

To build hApp bundles:
```bash
cd mycelix-workspace
nix develop
./scripts/build-happs.sh           # Build all 10 ready hApps
./scripts/build-happs.sh health    # Build specific hApp
```

All 10 scaffolded hApps (health, energy, climate, mutualaid, property, media, consensus, music, food, transport) have:
- `happ.yaml` with manifest_version "0" (current Holochain 0.6 format)
- `dna.yaml` with proper zome references
- Build script support

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| PRODUCTION | Tests pass, benchmarks validated, actively used |
| BETA | Compiles, partial tests, not yet production |
| SCAFFOLD | Types/structure exist, core logic incomplete |
| STUB | Minimal files, no functional code |
| DORMANT | Restored from archive or inactive |

---

## hApps

### Production

| hApp | Zomes | Tests | Notes |
|------|-------|-------|-------|
| **Core (0TML)** | 6 (agents, bridge, dkg, epistemic_storage, federated_learning, pogq_validation) | 62 verified | 45% BFT validated. REST API implemented (4 endpoints: /health, /status, /trust/{id}, /pogq/validate). Python coordinator + Rust zomes. |
| **Mail** | 12 | Submodule tests | PQC encryption, decentralized email. Most complete hApp. |
| **DeSci** | N/A (REST API) | 141 verified | Actix-web service, **not a Holochain hApp**. CLI + REST. |

### Beta

| hApp | Zomes | Tests | Notes |
|------|-------|-------|-------|
| **Marketplace** | 8 | Partial | Multiple build scripts need consolidation. Arbitration zome incomplete. |
| **Supply Chain** | 8 | Partial | Provenance tracking. Submodule. |
| **Observatory** | N/A (SvelteKit) | None | Live at [observatory.mycelix.net](https://luminous-dynamics.github.io/mycelix-observatory/). Demo mode + conductor fallback. DNS CNAME pending. |
| **Epistemic Markets** | Native workspace | Compiles | Heavy documentation (manifesto, rituals, personas), light implementation. Core zome logic exists. |
| **Fabrication** | 6 (bridge, designs, materials, printers, prints, verification) | Compiles | Native workspace hApp. |
| **Praxis** | 10 | Restored | Restored from archive. Needs verification. |
| **Consensus (RBBFT)** | 1 | Compiles | Native workspace hApp. Minimal. |

### Core Four (fully implemented, integrated)

| hApp | Zomes | Tests | Notes |
|------|-------|-------|-------|
| **Identity** | 11 (did_registry, trust_credential, mfa, verifiable_credential, credential_schema, education, revocation, recovery, bridge, name-registry, web-of-trust) | 23 unit + 100+ sweettests + 20 consciousness gating sweettests | W3C DID Core, MFA (5 factor types), ZK-based trust attestations, consciousness credential issuance, name registry, web of trust. 36,820 LOC. |
| **Governance** | 7 (proposals, voting, threshold-signing, councils, constitution, execution, bridge) | 44 unit + 156+ sweettests | 5 proposal types, Phi-weighted voting, Feldman VSS DKG (off-chain ceremony, on-chain commitments), constitutional amendments. 28,364 LOC. |

### Additional Clusters

| Cluster | Zomes | LOC | Tests | Notes |
|---------|-------|-----|-------|-------|
| **Hearth** (FAMILY tier) | 11 + bridge | 30,403 | 1,023 (workspace) | Kinship, gratitude, care, autonomy, decisions, stories, milestones, rhythms, emergency, resources. Consciousness-gated. |
| **Finance** (ECONOMY tier) | 7+ zomes (tend, treasury, staking, payments, recognition, bridge, currency-mint) | 11 consciousness gating sweettests + unit tests | 22,273 LOC. Currency-mint modularized (7 modules), 5 cross-cluster bridge handlers. 8th role in unified hApp. |
| **Personal** (Sovereign tier) | 3 + bridge | 4,447 | 30 sweettests + is_finite() hardened | Identity vault, health vault, credential wallet. is_finite() guards on credential-wallet. |
| **Attribution** (OPEN tier) | 3 zomes | 6,849 | 17 unit + 1 sweettest | Dependency registry, usage receipts, reciprocity pledges. |

### Cluster Domains (active in commons/civic clusters)

These domains are fully implemented as zomes within the Commons or Civic cluster DNAs. They are NOT standalone hApps.

| Domain | Cluster | Zomes | Notes |
|--------|---------|-------|-------|
| **Property** | Commons | 4 | Registry, transfers, disputes, commons management |
| **Housing** | Commons | 6 | Units, membership, finances, maintenance, CLT, governance |
| **Care** | Commons | 5 | Timebank, circles, matching, plans, credentials |
| **Mutual Aid** | Commons | 7 | Needs, circles, governance, pools, requests, resources, timebank |
| **Water** | Commons | 5 | Flow, purity, capture, stewardship, traditional knowledge |
| **Food** | Commons | 4 | Production, distribution, preservation, knowledge |
| **Transport** | Commons | 3 | Routes, sharing, carbon impact |
| **Mesh-Time** | Commons | 1 | Mesh time synchronization |
| **Resource-Mesh** | Commons | 1 | Resource mesh coordination |
| **Justice** | Civic | 5 | Cases, evidence, arbitration, restorative circles, enforcement |
| **Emergency** | Civic | 6 | Incidents, triage, resources, coordination, shelters, comms |
| **Media** | Civic | 4 | Publication, attribution, fact-checking, curation |
| **Resonance-Feed** | Civic | 1 | Resonance feed curation |

### Built (feature-complete, UI pending)

| hApp | Zomes | LOC | Notes |
|------|-------|-----|-------|
| **Knowledge** | 8 (claims, graph, query, inference, factcheck, markets_integration, dkg, bridge) | 14,696 | DKG Truth Engine, epistemic provenance. |
| **Energy** | 5 (projects, investments, regenerative, grid, bridge) | 10,118 | Grid management, renewable project tracking. |
| **Health** | 15 (7 MVP + 8 Tier 2: trials, insurance, fhir_mapping, fhir_bridge, cds, provider_directory, telehealth, nutrition) | 81,382 | Tier 2 promoted 2026-03-18. 22 Tier 3 archived to `_archive-2026-02-15/`. |
| **Space** | 5 + orbital-mechanics lib (orbital_objects, observations, conjunctions, debris_bounties, traffic_control) | 22,774 | SGP4 propagation, Alfano collision probability, CDM parsing. |
| **Climate** | 3 (carbon, projects, bridge) | 6,980 | Carbon credit verification, climate marketplace. |
| **Music** | 4 + 14 support crates (balances, catalog, plays, trust) | 32,772 | Audio codecs, ML-based search, streaming protocol. |

### Stub / Early Stage

| Component | Location | Notes |
|-----------|----------|-------|
| **symthaea-core** | `symthaea-core/` | Re-export facade (16 lines). Depends on `symthaea` crate. |
| **symthaea-mycelix-bridge** | `symthaea-mycelix-bridge/` | Substantial bridge (~25KB). Maps Phi/HDC to epistemic types. |
| **Bots** | `mycelix-bots/` | Discord + Telegram bots (Python). Not Holochain. |

### Dormant

| Component | Notes |
|-----------|-------|
| **Mutual Aid** | 8 zomes, restored from archive (also present in mycelix-commons) |

---

## SDKs

| SDK | Version | Claimed Tests | Verified (2026-03-08) | Notes |
|-----|---------|---------------|----------------------|-------|
| **Rust** (`mycelix-sdk`) | 0.6.0 | 866 | **1,036+ pass** (lib verified) | All tests pass. New finance module (finance.rs, FinanceBridgeClient, 12 tests). E0609 bug fixed (.phi → .coherence field rename). |
| **TypeScript** (`@mycelix/sdk`) | 0.6.0 | 5,828 | **6,316 pass / 15 skip** | All tests pass. libsodium ESM compat fixed. |
| **Python** (`mycelix`) | 0.1.0 | 45 | **45 pass**, 87% coverage | MATL, epistemic, FL, bridge modules. Verified 2026-02-04. |

### SDK Rust Modules
agentic, bridge, credentials, crypto, dkg, economics, epistemic, error, finance, fl, hyperfeel, identity, intentions, matl, pagination, pog, storage, temporal, wasm, zkproof

### SDK TypeScript Integration Modules (37 — re-audited 2026-03-06)

All 37 integration modules have real implementations (types + classes + methods). None are stubs.

**Zome-connected (22 modules, 59%)** — call `callZome` to interact with Holochain conductors:

| Module | Methods | Types | LOC | Files |
|--------|---------|-------|-----|-------|
| energy | 11 | 16 | 4,488 | 10 |
| governance | 6 | 22 | 4,142 | 9 |
| finance | 6 | 20 | 3,596 | 9 |
| property | 6 | 22 | 3,057 | 8 |
| knowledge | 6 | 20 | 2,978 | 7 |
| identity | 9 | 24 | 2,457 | 7 |
| health | 85 | 58 | 1,912 | 2 |
| health-fhir | 11 | 20 | 1,443 | 1 |
| health-marketplace | 10 | 13 | 1,187 | 1 |
| health-food | 11 | 11 | 965 | 1 |
| health-governance | 9 | 10 | 831 | 1 |
| media | 6 | 21 | 810 | 1 |
| justice | 6 | 25 | 803 | 1 |
| food | 37 | 37 | 803 | 1 |
| genetics | 8 | 16 | 637 | 1 |
| health-energy | 8 | 9 | 631 | 1 |
| hearth | 5 | 22 | 576 | 1 |
| commons | 7 | 21 | 567 | 1 |
| attribution | 34 | 25 | 552 | 1 |
| personal | 5 | 15 | 530 | 1 |
| civic | 18 | 23 | 494 | 1 |
| support | 50 | 34 | 756 | 1 |
| transport | 37 | 33 | 640 | 1 |

**Local-only (15 modules, 41%)** — in-memory Map storage with LocalBridge (functional but not yet wired to Holochain):

| Module | Methods | Types | LOC |
|--------|---------|-------|-----|
| fabrication | 100 | 56 | 1,269 |
| epistemic-markets | 109 | 31 | 1,120 |
| water-energy | 88 | 29 | 918 |
| food-shelter | 69 | 21 | 837 |
| consensus | 13 | 7 | 677 |
| academic | 15 | 20 | 640 |
| supplychain | 7 | 6 | 623 |
| desci | 14 | 7 | 605 |
| music | 11 | 7 | 545 |
| praxis | 8 | 6 | 471 |
| marketplace | 9 | 5 | 435 |
| mail | 6 | 4 | 368 |
| mutualaid | 11 | 6 | 367 |
| climate | 12 | 8 | 337 |

**Note**: The previous audit (2026-03-06 initial) incorrectly classified 12 local-only modules as "Stub". All have real business logic, type definitions, and 6-109 methods. They use `LocalBridge` + in-memory `Map` storage rather than `callZome`, which is a valid local-first pattern. They will be wired to Holochain conductors when their corresponding hApps reach beta.

**Total**: ~37,500 LOC across 37 integration modules + bridge-routing.ts (254 LOC).

---

## Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| **Justfile** | Exists | Comprehensive task runner at `mycelix-workspace/justfile` |
| **CI** | Exists | `.github/workflows/mycelix-ci.yml` + `mycelix-release.yml` + `finance-ci.yml` + `health-ci.yml` |
| **Observatory** | Live + Demo | `mycelix-workspace/observatory/` - SvelteKit, 3-tier fallback (live→sim→static) |
| **SMS Gateway** | Exists | `mycelix-workspace/services/sms-gateway/` |
| **Civic hApp** | Exists | `mycelix-workspace/services/civic-happ/` |
| **Website** | Live | https://mycelix.net (GitHub Pages) |

---

## Known Gaps

1. ~~**Rust SDK 5 test failures**~~: Fixed 2026-02-04. All 1,036+ lib tests pass. Finance module added 2026-03-08.
2. ~~**TS SDK libsodium errors**~~: Fixed. All 6,316 tests pass.
3. ~~**Core REST API**~~: Implemented 2026-02-04. 4 endpoints: /health, /status, /trust/{id}, /pogq/validate.
4. ~~**Observatory mock-only**~~: Live conductor connection fully implemented, awaiting conductor.
5. ~~**hApp scaffolding incomplete**~~: Fixed 2026-02-08. All 10 ready-to-build hApps have v0 manifest format.
6. **Scope sprawl**: 24 hApps total, 12 with bundles, 10 scaffolded (ready to build), 2 REST APIs.
7. ~~**SDK-TS bundle size**~~: Re-audited 2026-03-06. All 37 integration modules have real implementations (0 stubs). 22 are zome-connected (callZome), 15 are local-only (in-memory Map + LocalBridge). Total ~37,500 LOC.
8. ~~**Cross-hApp bridges**~~: Unified hApp expanded to 17 roles (added energy, knowledge, climate, praxis). `CrossClusterRole` expanded 8→16 variants. 36 cross-cluster routes registered (was 13). 102 route tests pass. Notification fanout, saga orchestrator, geo-spatial queries, reputation aggregator, participatory budgeting all wired.
9. ~~**WASM builds pending**~~: Climate and music have happ.yaml + dna.yaml ready. Others (mutualaid, food, transport) are commons domain zomes — already built as part of commons DNA. Needs `nix develop` for `hc dna pack`.
10. ~~**TS SDK local-only modules**~~: All 15 local-only modules now have BridgeClient classes for callZome access (alongside existing LocalBridge for offline-first mode).
10. ~~**FL consciousness integration**~~: `ConsciousnessAwareByzantinePlugin` added 2026-02-15. Uses Phi scores for weight adjustment (boost/dampen/veto). 110 tests pass.
11. ~~**Emergency domain status**~~: Promoted from "stub" to "complete". 6 zomes, ~12,700 LOC, cross-domain bridges validated.
12. ~~**Health scope**~~: Reduced from 37 to 7 MVP zomes. 22 archived. 9 deferred (commented out).

---

## Priority Actions

1. ~~Verify SDK test suites (run and record actual pass/fail)~~: Done 2026-02-04
2. ~~Remove Python SDK references or create the package~~: SDK exists, 45 tests pass
3. ~~Implement Core REST API (3 endpoints minimum)~~: 4 endpoints implemented
4. ~~Connect Observatory to real conductor (or clearly label demo mode)~~: 3-tier fallback system
5. ~~Promote Identity + Governance from scaffold to beta~~: Corrected 2026-03-06. Both are fully implemented (Identity: 9 zomes/36.8K LOC, Governance: 7 zomes/28.4K LOC). Moved to Core Four.
6. ~~Audit SDK-TS integration modules for empty exports~~: Re-audited 2026-03-06. All 37 modules have real implementations. 0 stubs found.

---

## Build & Test Quick Reference

```bash
cd mycelix-workspace
just status          # Check ecosystem
just test            # Run all tests
just build           # Build everything
just verify-builds   # Check WASM artifacts
just verify-symlinks # Check hApp symlinks
```

---

*This document is the single source of truth for ecosystem status. Update when status changes.*
