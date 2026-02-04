# Mycelix Ecosystem Status

**Last verified**: 2026-02-02
**Holochain**: 0.6.0 | **HDK**: 0.6.0 | **HDI**: 0.7.0

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
| **Core (0TML)** | 6 (agents, bridge, dkg, epistemic_storage, federated_learning, pogq_validation) | 62 verified | 45% BFT validated. REST API **not implemented** (documented as planned). Python coordinator + Rust zomes. |
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

| SDK | Version | Claimed Tests | Verified (2026-02-02) | Notes |
|-----|---------|---------------|----------------------|-------|
| **Rust** (`mycelix-sdk`) | 0.6.0 | 866 | **868 pass, 5 fail, 2 ignored** (875 total) | 5 pre-existing test failures in agentic/ module (gradient estimator, bandit selection, network health, local DP, byzantine invariant). |
| **TypeScript** (`@mycelix/sdk`) | 0.6.0 | 5,828 | **6,314 pass, 2 fail, 15 skip, 23 errors** (libsodium ESM compat issue) | Actual count higher than claimed. 2 test files fail due to libsodium/vitest incompatibility. |
| **Python** | 0.1.0 | 5 test files | **Verified** - `sdk-python/` exists | MATL, epistemic, FL, bridge modules implemented. |

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
| **Observatory** | Demo only | `mycelix-workspace/observatory/` - SvelteKit, mock data |
| **SMS Gateway** | Exists | `mycelix-workspace/services/sms-gateway/` |
| **Civic hApp** | Exists | `mycelix-workspace/services/civic-happ/` |
| **Website** | Live | https://mycelix.net (GitHub Pages) |

---

## Known Gaps

1. ~~**Rust SDK 5 test failures**~~: Fixed 2026-02-04. All 996 tests pass (1002 with parallel feature).
2. ~~**TS SDK libsodium errors**~~: Fixed. All 6,316 tests pass.
3. **Core REST API**: Documented but not implemented. Most visible credibility gap.
4. ~~**Observatory mock-only**~~: Live conductor connection fully implemented, awaiting conductor.
5. **Scope sprawl**: 27+ hApps, most in scaffold state. 14 scaffolds with types but no core logic.
6. **SDK-TS bundle size**: 29 integration modules, unclear how many are functional.
7. **Cross-hApp bridges**: Claimed in architecture docs, not tested in integration.

---

## Priority Actions

1. Verify SDK test suites (run and record actual pass/fail)
2. Remove Python SDK references or create the package
3. Implement Core REST API (3 endpoints minimum)
4. Connect Observatory to real conductor (or clearly label demo mode)
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
