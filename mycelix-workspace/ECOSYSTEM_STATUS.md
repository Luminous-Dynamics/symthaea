# Mycelix Ecosystem Status

**Last verified**: 2026-02-16
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
| edunet | ✅ 948K | 10 | v0 | Built (submodule: happ/) |
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
| **EduNet** | 10 | Restored | Restored from archive. Needs verification. |
| **Consensus (RBBFT)** | 1 | Compiles | Native workspace hApp. Minimal. |

### Core Four (fully implemented, integrated)

| hApp | Zomes | Tests | Notes |
|------|-------|-------|-------|
| **Identity** | 9 (did_registry, trust_credential, mfa, verifiable_credential, credential_schema, education, revocation, recovery, bridge) | 23 unit + 100+ sweettests | W3C DID Core, MFA (5 factor types), ZK-based trust attestations, consciousness credential issuance. 36,820 LOC. |
| **Governance** | 7 (proposals, voting, threshold-signing, councils, constitution, execution, bridge) | 44 unit + 130+ sweettests | 5 proposal types, Phi-weighted voting, Feldman VSS DKG (off-chain ceremony, on-chain commitments), constitutional amendments. 28,364 LOC. |

### Additional Clusters

| Cluster | Zomes | LOC | Tests | Notes |
|---------|-------|-----|-------|-------|
| **Hearth** (FAMILY tier) | 11 + bridge | 30,403 | 1,023 (workspace) | Kinship, gratitude, care, autonomy, decisions, stories, milestones, rhythms, emergency, resources. Consciousness-gated. |
| **Personal** (Sovereign tier) | 3 + bridge | 4,447 | 20 | Identity vault, health vault, credential wallet. Lightweight scaffold. |
| **Attribution** (OPEN tier) | 3 zomes | 6,849 | 17 | Dependency registry, usage receipts, reciprocity pledges. |

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
| **Justice** | Civic | 5 | Cases, evidence, arbitration, restorative circles, enforcement |
| **Emergency** | Civic | 6 | Incidents, triage, resources, coordination, shelters, comms |
| **Media** | Civic | 4 | Publication, attribution, fact-checking, curation |

### Scaffold (compiles to WASM, core logic incomplete)

| hApp | Status | Notes |
|------|--------|-------|
| **Knowledge** | Types + structure | 23 subdirectories. Large scope. |
| **Finance** | Types + structure | 22,273 LOC. Needs promotion to beta. |
| **Energy** | Types + structure | 10,118 LOC. |
| **Health** | MVP (7 zomes) | Reduced from 37 to 7 core zomes. 22 archived to `_archive-2026-02-15/`. |
| **Space** | Types + structure | 12 subdirectories. |

### Stub / Early Stage

| Component | Location | Notes |
|-----------|----------|-------|
| **symthaea-core** | `symthaea-core/` | Re-export facade (16 lines). Depends on `symthaea` crate. |
| **symthaea-mycelix-bridge** | `symthaea-mycelix-bridge/` | Substantial bridge (~25KB). Maps Phi/HDC to epistemic types. |
| **Bots** | `mycelix-bots/` | Discord + Telegram bots (Python). Not Holochain. |
| **Music** | `mycelix-music/` | 41 subdirectories but early stage. |

### Dormant

| Component | Notes |
|-----------|-------|
| **Climate** | 5 zomes, restored from archive |
| **Mutual Aid** | 8 zomes, restored from archive |

---

## SDKs

| SDK | Version | Claimed Tests | Verified (2026-02-04) | Notes |
|-----|---------|---------------|----------------------|-------|
| **Rust** (`mycelix-sdk`) | 0.6.0 | 866 | **996 pass** (1002 w/ parallel feature) | All tests pass. Agentic module tests fixed. |
| **TypeScript** (`@mycelix/sdk`) | 0.6.0 | 5,828 | **6,316 pass / 15 skip** | All tests pass. libsodium ESM compat fixed. |
| **Python** (`mycelix`) | 0.1.0 | 45 | **45 pass**, 87% coverage | MATL, epistemic, FL, bridge modules. Verified 2026-02-04. |

### SDK Rust Modules
agentic, bridge, credentials, crypto, dkg, economics, epistemic, error, fl, hyperfeel, identity, intentions, matl, pagination, pog, storage, temporal, wasm, zkproof

### SDK TypeScript Integration Modules (34 — audited 2026-03-06)

**Full (19 modules, 53%)** — 5+ exported functions with business logic:

| Module | Functions | Types | LOC |
|--------|-----------|-------|-----|
| health-fhir | 11 | 20 | 1,443 |
| health-food | 11 | 11 | 965 |
| health-marketplace | 10 | 13 | 1,187 |
| health-governance | 9 | 10 | 831 |
| identity | 9 | 24 | 749 |
| epistemic-markets | 8 | 31 | 1,120 |
| genetics | 8 | 16 | 637 |
| health-energy | 8 | 9 | 631 |
| commons | 7 | 21 | 567 |
| finance | 6 | 20 | 583 |
| governance | 6 | 22 | 751 |
| justice | 6 | 25 | 803 |
| knowledge | 6 | 20 | 684 |
| media | 6 | 21 | 810 |
| property | 6 | 22 | 619 |
| water-energy | 6 | 29 | 918 |
| food-shelter | 5 | 21 | 837 |
| hearth | 5 | 22 | 576 |
| personal | 5 | 15 | 530 |

**Types (5 modules, 12%)** — primarily type/interface definitions:
fabrication (56 types, 1,269 LOC), food (37 types), health (58 types), support (34 types), transport (33 types)

**Stub (12 modules, 35%)** — minimal re-exports (<3 functions):
academic, attribution, civic, climate, consensus, desci, edunet, energy, mail, marketplace, music, mutualaid, supplychain

**Total**: 23,926 LOC across 34 integration modules. Bridge module adds 835 LOC (10 functions + LocalBridge + BridgeRouter classes).

---

## Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| **Justfile** | Exists | Comprehensive task runner at `mycelix-workspace/justfile` |
| **CI** | Exists | `.github/workflows/mycelix-ci.yml` + `mycelix-release.yml` |
| **Observatory** | Live + Demo | `mycelix-workspace/observatory/` - SvelteKit, 3-tier fallback (live→sim→static) |
| **SMS Gateway** | Exists | `mycelix-workspace/services/sms-gateway/` |
| **Civic hApp** | Exists | `mycelix-workspace/services/civic-happ/` |
| **Website** | Live | https://mycelix.net (GitHub Pages) |

---

## Known Gaps

1. ~~**Rust SDK 5 test failures**~~: Fixed 2026-02-04. All 996 tests pass (1002 with parallel feature).
2. ~~**TS SDK libsodium errors**~~: Fixed. All 6,316 tests pass.
3. ~~**Core REST API**~~: Implemented 2026-02-04. 4 endpoints: /health, /status, /trust/{id}, /pogq/validate.
4. ~~**Observatory mock-only**~~: Live conductor connection fully implemented, awaiting conductor.
5. ~~**hApp scaffolding incomplete**~~: Fixed 2026-02-08. All 10 ready-to-build hApps have v0 manifest format.
6. **Scope sprawl**: 24 hApps total, 12 with bundles, 10 scaffolded (ready to build), 2 REST APIs.
7. ~~**SDK-TS bundle size**~~: Audited 2026-03-06. 34 integration modules: 19 Full (53%, 5+ exported functions), 5 Types (12%, primarily type definitions), 12 Stub (35%, minimal re-exports). Total 23,926 LOC. See SDK-TS Audit section below.
8. **Cross-hApp bridges**: Claimed in architecture docs, not tested in integration.
9. **WASM builds pending**: 6 hApps (climate, mutualaid, consensus, music, food, transport) need `nix develop` + WASM compilation.
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
6. ~~Audit SDK-TS integration modules for empty exports~~: Done 2026-03-06. 19 Full / 5 Types / 12 Stub

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
