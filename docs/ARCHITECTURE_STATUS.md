# Mycelix Ecosystem — Architecture Status

**Last verified**: 2026-02-17
**Model**: Fractal CivOS (3-tier agent-centric architecture)
**Canonical reference**: `mycelix-core/architecture/clustermap.yaml`

---

## Status Legend

| Status | Meaning |
|--------|---------|
| **IMPLEMENTED** | Working code + tests + compiles/bundles |
| **SCAFFOLDED** | Code structure exists, limited tests or blocked dependencies |
| **DESIGNED** | Architecture docs exist, no substantial code |
| **ASPIRATIONAL** | Mentioned in roadmap but no docs or code |

---

## Fractal CivOS — 3-Tier Architecture

The Mycelix platform is organized into three trust tiers, each mapping to a Holochain DNA cluster:

| Tier | Cluster | Trust Model | Status | Zomes | Tests |
|------|---------|-------------|--------|-------|-------|
| **Sovereign (ME)** | `mycelix-personal` | Source chain only | SCAFFOLDED | 4 | 86 |
| **Civic (WE)** | `mycelix-civic` | Permissioned DHT | IMPLEMENTED | 16 | 5,516 |
| **Commons (ALL)** | `mycelix-commons` | Public DHT | IMPLEMENTED | 35 | 4,126 |

**Cross-cluster communication**: `CallTargetCell::OtherRole` dispatch via unified hApp (`mycelix-unified-happ.yaml`).
All 3 roles defined. Commons + Civic bidirectional dispatch verified. Personal dispatch scaffolded.

---

## Core hApps (The Core Four)

### Identity — `mycelix-identity/`
| Attribute | Value |
|-----------|-------|
| Status | **IMPLEMENTED** |
| Zomes | 9 (did_registry, mfa, recovery, verifiable_credential, trust_credential, credential_schema, revocation, bridge, education) |
| LOC | ~36,609 Rust |
| Tests | 402 unit + 7 sweettest suites (71 functions) |
| Bundle | `mycelix-identity.happ` (5.5M) |
| MFDI | 9/9 identity factors complete |

### Governance — `mycelix-governance/`
| Attribute | Value |
|-----------|-------|
| Status | **SCAFFOLDED** |
| Zomes | 7 (proposals, voting, councils, constitution, execution, threshold-signing, bridge) |
| LOC | ~18,360 Rust |
| Tests | 188 unit + 1 sweettest suite (6 functions) |
| Bundle | `mycelix-governance.happ` (4.3M) |
| Blocker | threshold-signing DKG not wired (WASM incompatibility) |

### Core FL — `mycelix-workspace/crates/mycelix-fl-core/`
| Attribute | Value |
|-----------|-------|
| Status | **IMPLEMENTED** |
| Modules | 12 (aggregation, byzantine, consciousness_plugin, pipeline, privacy, etc.) |
| LOC | ~10,625 Rust |
| Tests | 119 |
| Pipeline | Validate → DP → Gate → Detect → Trim → Aggregate |
| Byzantine | 34% tolerance validated (consciousness-aware gating) |

### LUCID — `mycelix-workspace/happs/lucid/`
| Attribute | Value |
|-----------|-------|
| Status | **IMPLEMENTED** |
| Zomes | 8 (lucid, temporal, reasoning, sources, privacy, collective, temporal-consciousness, bridge) |
| LOC | ~12,431 Rust + Svelte/TS frontend |
| Tests | 43 unit |
| Bundle | `lucid.happ` (5.6M) |
| Symthaea | 95% wired (19+ Tauri commands) |

---

## Cluster DNAs

### Commons — `mycelix-commons/`
| Attribute | Value |
|-----------|-------|
| Status | **IMPLEMENTED** |
| Domains | property, housing, care, mutualaid, water, food, transport |
| Zomes | 35 (34 domain + 1 bridge) |
| LOC | ~81,734 Rust |
| Tests | 4,126 unit + 14/14 sweettest |
| Bundle | `mycelix-commons.dna` (24M) |

### Civic — `mycelix-civic/`
| Attribute | Value |
|-----------|-------|
| Status | **IMPLEMENTED** |
| Domains | justice, emergency, media |
| Zomes | 16 (15 domain + 1 bridge) |
| LOC | ~84,333 Rust |
| Tests | 5,516 unit + 14/14 sweettest |
| Bundle | `mycelix-civic.dna` (12M) |

### Personal — `mycelix-personal/`
| Attribute | Value |
|-----------|-------|
| Status | **SCAFFOLDED** |
| Zomes | 4 (identity_vault, health_vault, credential_wallet, personal_bridge) |
| LOC | ~2,210 Rust |
| Tests | 86 unit |
| Bundle | Not yet built |
| Next | Wire cross-cluster dispatch, build DNA bundle |

---

## SDKs

| SDK | Path | LOC | Tests | Status |
|-----|------|-----|-------|--------|
| **Rust** | `sdk/` | ~100,644 | 1,052 (996 pass) | IMPLEMENTED |
| **TypeScript** | `sdk-ts/` | ~196,640 | 10,888 (6,316 pass) | IMPLEMENTED |
| **Python** | `sdk-python/` | ~261,908 | 45 (87% cov) | IMPLEMENTED |
| **0TML/zerotrustml** | `mycelix-core/0TML/` | ~70,899 | 10 | IMPLEMENTED |
| **WASM** | `sdk-wasm/` | ~1,275 | 0 | SCAFFOLDED |
| **Ethereum** | `sdk-eth/` | 0 | 0 | ASPIRATIONAL |

---

## Crypto Libraries

| Library | Path | LOC | Tests | Status | Notes |
|---------|------|-----|-------|--------|-------|
| **feldman-dkg** | `mycelix-core/libs/feldman-dkg/` | 2,618 | 39 | IMPLEMENTED | WASM-blocked for zome use |
| **rb-bft-consensus** | `mycelix-core/libs/rb-bft-consensus/` | 8,181 | 143 | IMPLEMENTED | BLS + VRF + slashing + PQC |
| **kvector-zkp** | `mycelix-core/libs/kvector-zkp/` | 3,082 | 62 | IMPLEMENTED | AIR-based ZK proof system |

---

## Bridge Infrastructure

| Component | Path | Tests | Status |
|-----------|------|-------|--------|
| **Commons bridge** | `mycelix-commons/zomes/commons-bridge/` | Part of 4,126 | IMPLEMENTED |
| **Civic bridge** | `mycelix-civic/zomes/civic-bridge/` | Part of 5,516 | IMPLEMENTED |
| **Personal bridge** | `mycelix-personal/zomes/personal-bridge/` | 20 | SCAFFOLDED |
| **Governance bridge** | `mycelix-governance/zomes/bridge/` | Part of 188 | IMPLEMENTED |
| **Symthaea-Mycelix bridge** | `symthaea-mycelix-bridge/` | 17 | IMPLEMENTED |
| **Bridge common types** | `mycelix-commons/crates/`, `mycelix-civic/crates/` | 55 | IMPLEMENTED |

Bridge features: allowlist enforcement, rate limiting, audit trail, cross-cluster dispatch via OtherRole.

---

## Additional hApps

| hApp | Path | LOC | Tests | Status |
|------|------|-----|-------|--------|
| **Health** | `mycelix-health/` | ~23,087 | 68 | SCAFFOLDED (flake broken, 7 MVP zomes) |
| **Epistemic Markets** | `happs/epistemic-markets/` | ~18,103 | 59 | SCAFFOLDED |
| **Fabrication** | `happs/fabrication/` | ~6,232 | 19 | SCAFFOLDED |
| **Consensus (RBBFT)** | `happs/consensus/` | ~998 | 11 | SCAFFOLDED |
| **DeSci** | `happs/desci/` | — | 141 | IMPLEMENTED (REST, not hApp) |
| **Space** | `happs/space/` | — | — | SCAFFOLDED |
| **Mail** | `happs/mail/` | — | — | IMPLEMENTED (12 zomes) |
| **Observatory** | `observatory/` | ~4,702 TS | 0 | IMPLEMENTED (live, demo mode) |

---

## Test Infrastructure

| Suite | Count | Status |
|-------|-------|--------|
| Commons unit tests | 4,126 | PASSING |
| Civic unit tests | 5,516 | PASSING |
| Rust SDK tests | 996 | PASSING |
| TypeScript SDK tests | 6,316 | PASSING |
| FL-core tests | 119 | PASSING |
| Identity tests | 402 | PASSING |
| Governance tests | 188 | PASSING |
| Crypto lib tests | 244 | PASSING (native only) |
| Python SDK tests | 45 | PASSING |
| Commons sweettest | 14 | PASSING |
| Civic sweettest | 14 | PASSING |
| Workspace sweettest | ~233 | SCAFFOLDED (all `#[ignore]`, need DNA bundles) |

**CI**: GitHub Actions `mycelix-ci.yml` — SDK tests, WASM builds, LUCID frontend, sweettest job (conditional on conductor). 5 `continue-on-error` flags removed/replaced with conditional guards.

---

## Symthaea Integration

| Component | Status | Notes |
|-----------|--------|-------|
| **Symthaea core** | IMPLEMENTED | ~343K LOC, 5,050 tests, v0.5.0 |
| **LUCID bridge** | IMPLEMENTED | 95% wired, 19+ Tauri commands |
| **FL consciousness plugin** | IMPLEMENTED | Phi-based weight adjustment |
| **Symthaea-Mycelix bridge** | IMPLEMENTED | PoGQ mapping, epistemic classification |
| **Governance Phi gating** | SCAFFOLDED | Voting queries personal_bridge (fallback to proxy) |

---

## What's NOT Built Yet

These items appear in architecture documents but have no implementation:

1. **mycelix-knowledge** (planned cluster) — Knowledge graphs, DeSci, educational resources
2. **mycelix-finance** (planned cluster) — Treasury, staking, mutual credit
3. **Membrane factory** — Private groups within Commons (capability-grant-based)
4. **Full P2P sync** — iroh integration blocked (sync→async boundary)
5. **Cross-cluster credential presentation** — Personal→Civic/Commons ZK proofs
6. **Federation protocol** — Multi-network FL coordination
7. **WASM-compatible DKG** — feldman-dkg for in-zome verification

---

## Aggregate Numbers

| Metric | Count |
|--------|-------|
| Total Rust LOC (all clusters + SDKs) | ~750K+ |
| Total tests (all suites) | ~28K+ |
| Holochain zomes | 90+ |
| DNA bundles | 5+ |
| hApp bundles | 10+ |
| SDKs | 4 (Rust, TS, Python, WASM) |
| Architecture docs | 22+ |
| CI jobs | 12+ |
