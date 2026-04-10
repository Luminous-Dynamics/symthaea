# 8D Sovereign Profile: Architecture Reference

## Why

When you reduce a human being to a single number -- like a credit score or social credit rating -- you create a system that is infinitely gamifiable and inherently oppressive. Single-axis metrics force homogenization and punish diverse survival strategies.

The **8D Sovereign Profile** maps civic identity as a holographic, multi-faceted geometry rather than a flat number. Eight concrete, physically-grounded dimensions replace the abstract 4D `ConsciousnessProfile`. Each dimension is directly measurable from a primary source, and multiple pathways to citizenship are mathematically respected.

## The 8 Dimensions

| Dim | Name | Source Cluster | Measures | Saturation |
|-----|------|---------------|----------|------------|
| D0 | Epistemic Integrity | knowledge/claims | Truth-telling track record | 50 validated claims |
| D1 | Thermodynamic Yield | energy/grid | Physical energy contribution | 30 verified records |
| D2 | Network Resilience | commons/mesh-time | Node uptime, bandwidth | 720 time anchors |
| D3 | Economic Velocity | finance/TEND | Anti-hoarding compliance | 50 TEND exchanges |
| D4 | Civic Participation | governance/voting | Jury duty, voting, proposals | 20 governance actions |
| D5 | Stewardship & Care | attribution/reciprocity | Verified commons labor | 30 pledges |
| D6 | Semantic Resonance | core-FL/hyperfeel (not yet in unified hApp) | Community value alignment | Cosine similarity |
| D7 | Domain Competence | craft/credentials | Peer-verified expertise | 3 living credentials |

## Architecture

```
8 Source Clusters → Per-agent score externs (get_agent_*_score)
       ↓
Identity Bridge → issue_sovereign_credential() → SovereignCredential (8D)
       ↓
Decay: S_decayed = S_raw * e^(-lambda * dt)
       ↓
HDC: encode_profile() → BinaryHV (16,384-bit)
       ↓
Tier: popcount → CivicTier (Observer→Guardian)
       ↓
ZKP: prove_sovereign_tier() → STARK proof
       ↓
Network receives proof only — not your score, not your dimensions
```

## Crate: `crates/sovereign-profile/`

Pure Rust, zero HDK dependency. Importable by any cluster, frontend, or tool.

### Modules

| Module | Purpose | Tests |
|--------|---------|-------|
| `lib.rs` | SovereignProfile, CivicTier, CivicRequirement, SovereignCredential | 29 |
| `decay.rs` | Exponential decay, constitutional bounds, ramp transitions, grace periods | 28 |
| `hdc.rs` | BinaryHV (16,384-bit), encode_profile(), tier_from_popcount() | 17 |
| `collectors.rs` | Normalization math: DimensionInput → [0,1] score | 12 |
| `weights.rs` | DimensionWeights with 5 presets (governance, energy, knowledge, care, equal) | via lib |
| `compat.rs` | LegacyProfile ↔ SovereignProfile bidirectional From conversions | via lib |
| `i18n.rs` | Dimension/tier labels with i18n keys | via lib |

**Total: 87 tests** (with `--features hdc`)

### Key Types

```rust
pub struct SovereignProfile {
    pub epistemic_integrity: f64,   // [0.0, 1.0]
    pub thermodynamic_yield: f64,
    pub network_resilience: f64,
    pub economic_velocity: f64,
    pub civic_participation: f64,
    pub stewardship_care: f64,
    pub semantic_resonance: f64,
    pub domain_competence: f64,
}

pub enum CivicTier {
    Observer,      // < 0.3 — read-only
    Participant,   // >= 0.3 — basic proposals
    Citizen,       // >= 0.4 — voting rights
    Steward,       // >= 0.6 — constitutional
    Guardian,      // >= 0.8 — emergency powers
}
```

## Decay: Constitutional Bounds

Civic standing decays exponentially without engagement:

```
S_decayed = S_raw * e^(-lambda * elapsed_days)
```

| Constant | Value | Meaning |
|----------|-------|---------|
| LAMBDA_MIN | 0.001 | Half-life 693 days (land trusts) |
| LAMBDA_MAX | 0.020 | Half-life 35 days (emergency pods) |
| RAMP_DAYS_MIN | 30 | Minimum transition period for lambda changes |
| GRACE_PERIOD_DAYS | 30 | Notification window before demotion |

**lambda = 0 is constitutionally impossible** — prevents permanent oligarchy.

Communities configure lambda via Steward-tier governance. Changes ramp gradually (linear interpolation over T_ramp days) to prevent sudden disenfranchisement.

## HDC Encoding

Each dimension owns a weighted fraction of the 16,384 bits. Active bits are proportional to the dimension value. This guarantees:

- **Monotonicity**: higher values produce strictly more bits
- **Determinism**: same profile + weights = same HV
- **Weight respect**: community DimensionWeights allocate bit regions

Tier derivation: `popcount(encoded_hv) >= calibrated_threshold`

## ZKP Privacy Shield

STARK proof (Winterfell 0.13.1) demonstrates `decayed_score >= threshold` without revealing the raw profile, individual dimensions, or last interaction timestamp.

| Metric | Value |
|--------|-------|
| Prove time | ~1ms (release, 32-row range proof) |
| Verify time | ~0.3ms |
| Proof size | ~7.5KB |
| Public inputs | threshold, commitment |
| Private inputs | decayed_score, agent_did |

Commitment: `SHA-256(SOVEREIGN:v1:{did}:{score}:{lambda}:{elapsed})` — prevents replay.

## Gating: gate_civic()

Drop-in replacement for `gate_consciousness()`. All ~80 production call sites across 8 clusters migrated.

```rust
// In any cluster's coordinator zome:
mycelix_bridge_common::gate_civic(
    "craft_bridge",
    &mycelix_bridge_common::civic_requirement_voting(),
    "create_guild",
)?;
```

Presets: `civic_requirement_basic()`, `civic_requirement_proposal()`, `civic_requirement_voting()`, `civic_requirement_constitutional()`, `civic_requirement_guardian()`.

## Dimension Collectors

The identity bridge gathers scores from source clusters via cross-cluster `CallTargetCell::OtherRole` calls. Each collector:
- Fails silently to 0.0 if the source cluster is unavailable
- Falls back to a proxy dimension from the identity cluster
- Uses the normalization formula: `score = min(1, count/baseline) * quality * recency_weight`

Default baselines and half-lives are in `collectors.rs`. Communities can override via governance.

## TypeScript SDK

Two TypeScript modules mirror the Rust crate:
- `mycelix-sdk-ts/src/sovereign-profile.ts` — primary SDK
- `mycelix-workspace/sdk-ts/src/core/sovereign-gate.ts` — workspace SDK with gate middleware

Both provide: `SovereignProfile`, `CivicTier`, `combinedScore()`, `meetsRequirement()`, `decayScore()`, `daysUntilThreshold()`, `DIMENSION_LABELS`, `WEIGHTS_*` presets.

## Frontend

`mycelix-leptos-core` provides:
- `SovereignProfile` as the reactive profile type (replacing ConsciousnessProfile)
- `TierGate` component showing 8-dimension progress bars when gated
- `refresh_consciousness_from_conductor()` fetches 8D from identity bridge
- CSS variables: `--civic-warmth`, `--civic-bond-glow`, `--civic-animation-speed`

Craft-specific:
- `CredentialsPage` with SVG Ebbinghaus decay curves, vitality heatmap, review countdown
- `DashboardPage` with TierGate-protected guild creation and federation

## Migration Guide

### Rust (zome coordinators)

Replace:
```rust
// Old
gate_consciousness("bridge", &requirement_for_voting(), "action")?;

// New
gate_civic("bridge", &civic_requirement_voting(), "action")?;
```

### TypeScript

```typescript
// Old
import { combinedScore, ConsciousnessProfile } from '@mycelix/sdk';

// New
import { sovereignCombinedScore, SovereignProfile } from '@mycelix/sdk';
```

## Running Tests

```bash
# Core types + decay + collectors
cargo test -p sovereign-profile

# With HDC encoding
cargo test -p sovereign-profile --features hdc

# ZKP proofs
cargo test -p mycelix-zkp-core --features backend-winterfell -- sovereign

# Bridge-common integration
cargo test -p mycelix-bridge-common --lib sovereign_gate

# Full bridge-common suite (708 tests)
cargo test -p mycelix-bridge-common --lib
```
