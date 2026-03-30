# Mycelix Bridge-Common Integration Plan

## The Question: Import or Copy?

### Option A: Import bridge-common as dependency
```toml
mycelix-bridge-common = { path = "../crates/mycelix-bridge-common", default-features = false }
```

**Pros:**
- Production-tested (450+ tests)
- Stays in sync with Mycelix governance changes automatically
- The SAME types that will run on Holochain — simulation validates production code
- 6,256 LOC of battle-tested types we don't maintain

**Cons:**
- Pulls in blake3 (~200KB compile, but it's fast)
- 19,272 LOC total crate (most behind `#[cfg(feature = "hdk")]`)
- Version coupling — bridge-common changes could break the sim
- Compilation time increase (~5-10s)

### Option B: Copy type definitions only
Copy the struct/enum definitions we need into a `mycelix_types.rs` module.

**Pros:**
- Zero external dependencies
- Only the 500-1000 LOC we actually use
- No version coupling
- Fastest compile

**Cons:**
- Types drift from production (silent bugs when deploying to Holochain)
- We maintain duplicate type definitions
- Miss bug fixes and improvements to bridge-common
- Lose the 450+ tests

### Option C: Thin bridge trait (RECOMMENDED)
Define a trait in the sim that bridge-common types implement. The sim depends
on the trait, not the concrete types. Bridge-common provides the production
implementation. A lightweight mock provides fast testing.

```rust
// In multiworld-sim:
pub trait ConsciousnessGating {
    fn tier(&self) -> u8;           // 0-4
    fn vote_weight(&self) -> f64;   // 0.0-1.0
    fn can_propose(&self) -> bool;
    fn can_vote(&self) -> bool;
}

// In bridge-common (or an adapter):
impl ConsciousnessGating for ConsciousnessProfile { ... }
```

This doesn't work cleanly because Rust's orphan rules prevent implementing
foreign traits on foreign types without a wrapper.

### RECOMMENDATION: Option A (Import)

Import bridge-common with `default-features = false`. The reasons:

1. **The simulation's PURPOSE is to validate Mycelix governance parameters.**
   Using different types than production defeats this purpose.

2. **HDK is cleanly gated.** Without `hdk` feature, bridge-common is pure
   Rust + serde + blake3. No Holochain runtime.

3. **blake3 is a 200KB, zero-dependency hash library.** Negligible cost.

4. **The version coupling is DESIRABLE.** When bridge-common changes
   ConsciousnessTier thresholds, the sim should break — because the sim
   needs to validate the new thresholds.

5. **Serde is already used by both crates.** No serialization conflict.

## Integration Phases

### Phase 1: Add dependency + PlanetaryBody (1 hour)
- Add `mycelix-bridge-common = { path = "...", default-features = false }`
- Replace `world.location: String` with `PlanetaryBody` enum
- Get `gravity_fraction()`, `solar_flux_fraction()`, `light_delay_to_earth_secs()` for free
- Eliminate all `match world.location.as_str()` blocks

### Phase 2: ConsciousnessProfile → governance (2 hours)
- Replace per-agent scalar phi with `ConsciousnessProfile` 4D
- Agent identity = MFA level (birth = 0, adult = 0.5, leader = 0.8)
- Agent reputation = computed from skills + governance participation
- Agent community = peer attestations (social graph connections)
- Agent engagement = hours of productive work
- Governance decisions gated by REAL tier requirements
- Vote weight from sigmoid function, not flat 1.0

### Phase 3: InterplanetaryRoute → trade (1 hour)
- Replace petgraph edges with `InterplanetaryRoute` structs
- Transfer windows from synodic periods (26 months Earth-Mars)
- Delta-v costs affect cargo economics
- InTransitCargo with actual departure/arrival ticks
- Emergency off-window launches at 2.5x delta-v cost

### Phase 4: ResourceManifest → economy (2 hours)
- Replace abstract "materials"/"food"/"water" with 12 specific resources
- ISRUCapability per planet (Titan exports hydrocarbons, Europa exports water)
- ResourceManifest tracking real kg of iron, silicon, aluminum, etc.
- TendEquivalence converting Cobb-Douglas output to labor-hour accounting

### Phase 5: ConsciousnessSynchronizer → inter-world (2 hours)
- Wire the full synchronizer for cross-planetary consciousness state
- Staleness decay based on light delay
- Blackout prediction during solar conjunction
- Reconciliation when communication resumes (70% actual, 30% predicted)
- DivergenceAlert when remote colony's consciousness drifts > 0.3

### Phase 6: CollectivePhiEngine → civilization metric (1 hour)
- Replace our custom `collective_phi()` with the production engine
- AgentConsciousnessVector per agent
- Proper collective integration measurement
- Synergy detection across consciousness dimensions

## Dependencies After Integration

```
mycelix-multiworld-sim
├── serde (1.0)
├── serde_json (1.0)
├── petgraph (0.7)          ← supply chain graph
├── toml (0.8)              ← config (added by linter)
├── tracing (0.1)           ← logging (added by linter)
└── mycelix-bridge-common   ← NEW (default-features = false)
    ├── serde (1.0)         ← shared
    └── blake3 (1)          ← new, lightweight
```

Total new dependency cost: blake3 only (~200KB, fast compile, zero deps).

## What This Changes

The simulation stops being an APPROXIMATION of Mycelix governance and
becomes a VALIDATOR of it. When we extract governance parameters from
100-seed batch analysis, those parameters are in the exact same type system
that Holochain will execute. ConsciousnessTier::Citizen in the sim IS
ConsciousnessTier::Citizen in production.

This is the difference between a spreadsheet model and an integration test.
