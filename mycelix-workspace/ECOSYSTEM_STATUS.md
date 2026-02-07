# Mycelix Ecosystem Status

**Last verified**: 2026-02-08
**Holochain**: 0.6.0 | **HDK**: 0.6.0 | **HDI**: 0.7.0

## hApp Bundle Status

| hApp | Bundle | Zomes | Lines | Ready to Build |
|------|--------|-------|-------|----------------|
| lucid | ✅ 5.6M | 6 | - | - |
| mail | ✅ 18M | 12 | - | - |
| knowledge | ✅ 5.1M | - | - | - |
| identity | ✅ 5.5M | - | - | - |
| justice | ✅ 4.4M | - | - | - |
| governance | ✅ 4.3M | - | - | - |
| finance | ✅ 3.4M | - | - | - |
| marketplace | ✅ 3.3M | 8 | - | - |
| epistemic-markets | ✅ 3.2M | - | - | - |
| fabrication | ✅ 7.4M | 6 | - | - |
| supplychain | ✅ 1.9M | 8 | - | - |
| edunet | ✅ 948K | 10 | - | - |
| **health** | ❌ | 36 | 60K+ | ✅ `./scripts/build-happs.sh health` |
| **energy** | ❌ | 6 | 9.2K | ✅ `./scripts/build-happs.sh energy` |
| **climate** | ❌ | 3 | 2.4K | ✅ `./scripts/build-happs.sh climate` |
| **mutualaid** | ❌ | 8 | 7.8K | ✅ `./scripts/build-happs.sh mutualaid` |
| **property** | ❌ | 6 | 3.3K | ✅ `./scripts/build-happs.sh property` |
| **media** | ❌ | 5 | 2.7K | ✅ `./scripts/build-happs.sh media` |
| **consensus** | ❌ | 1 | 1.4K | ✅ `./scripts/build-happs.sh consensus` |
| **music** | ❌ | 4 | 2.7K | ✅ `./scripts/build-happs.sh music` |
| core | N/A | 6 | - | REST API (not hApp) |
| desci | N/A | - | - | REST API (not hApp) |

**Note**: Desktop moved to `tools/desktop/` - it's a Tauri framework, not a hApp.

To build missing bundles:
```bash
nix develop
./scripts/build-happs.sh  # Build all, or specify hApp name
```

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
| **Observatory** | N/A (SvelteKit) | None | Dashboard runs in **demo mode with mock data**. No live conductor connection. |
| **Epistemic Markets** | Native workspace | Compiles | Heavy documentation (manifesto, rituals, personas), light implementation. Core zome logic exists. |
| **Fabrication** | 6 (bridge, designs, materials, printers, prints, verification) | Compiles | Native workspace hApp. |
| **EduNet** | 10 | Restored | Restored from archive. Needs verification. |
| **Consensus (RBBFT)** | 1 | Compiles | Native workspace hApp. Minimal. |

### Scaffold (compiles to WASM, core logic incomplete)

| hApp | Status | Notes |
|------|--------|-------|
| **Identity** | Types + structure | 15 subdirectories. Foundation for other hApps but missing core zome logic. |
| **Knowledge** | Types + structure | 23 subdirectories. Large scope. |
| **Governance** | Types + structure | Needed for ecosystem self-management. |
| **Justice** | Types + structure | |
| **Finance** | Types + structure | |
| **Property** | Types + structure | |
| **Energy** | Types + structure | |
| **Media** | Types + structure | 13 subdirectories. |
| **Health** | Types + structure | 17 subdirectories. Scope is very large (was claimed 40 zomes). |
| **Space** | Types + structure | 12 subdirectories. |
| **Care** | New scaffold | Restored/created recently. |
| **Emergency** | New scaffold | Restored/created recently. |
| **Water** | New scaffold | Restored/created recently. |
| **Housing** | New scaffold | Restored/created recently. |

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

### SDK TypeScript Integration Modules (29)
academic, climate, consensus, desci, edunet, energy, epistemic-markets, fabrication, finance, food-shelter, genetics, governance, health, health-energy, health-fhir, health-food, health-governance, health-marketplace, identity, justice, knowledge, mail, marketplace, media, music, mutualaid, property, supplychain, water-energy

**Warning**: Many of these 29 integration modules may point to empty or minimal implementations. Audit needed.

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
5. **Scope sprawl**: 27+ hApps, most in scaffold state. 14 scaffolds with types but no core logic.
6. **SDK-TS bundle size**: 29 integration modules, unclear how many are functional.
7. **Cross-hApp bridges**: Claimed in architecture docs, not tested in integration.

---

## Priority Actions

1. ~~Verify SDK test suites (run and record actual pass/fail)~~: Done 2026-02-04
2. ~~Remove Python SDK references or create the package~~: SDK exists, 45 tests pass
3. ~~Implement Core REST API (3 endpoints minimum)~~: 4 endpoints implemented
4. ~~Connect Observatory to real conductor (or clearly label demo mode)~~: 3-tier fallback system
5. Promote Identity + Governance from scaffold to beta
6. Audit SDK-TS integration modules for empty exports

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
