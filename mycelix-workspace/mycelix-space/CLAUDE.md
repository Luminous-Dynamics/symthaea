# Mycelix-Space: Decentralized Space Situational Awareness

## Quick Reference

- **Location**: /srv/luminous-dynamics/mycelix-space/
- **Stack**: Rust + Holochain 0.7 (HDK 0.6 / HDI 0.7) + SGP4 orbital mechanics
- **Status**: Active — consciousness gating, multi-party negotiation, automated screening
- **CI**: `.github/workflows/ci.yml` — check + test + integration + docs + WASM build + hApp packaging
- **Tests**: 91 orbital-mechanics + 18 shared + 8 E2E + 5 integrity = 122+ workspace tests

## Architecture

```
mycelix-space/
├── lib/orbital-mechanics/         # Pure Rust orbital mechanics (no Holochain deps)
│   └── src/                       # SGP4, Alfano Pc, CDM, coordinates, covariance, trust-weighted fusion (~3,000 LOC)
├── zomes/shared/                  # mycelix-space-shared: types + trust fabric + gating (~1,400 LOC)
│   └── src/lib.rs                 # SpaceTimestamp, QualityScore, RiskLevel, trust requirements, gate_space_operation
├── dna/zomes/                     # 5 Holochain zomes (integrity + coordinator each)
│   ├── orbital_objects/           # Catalog of tracked objects + TLE management
│   ├── observations/              # Sensor data ingestion + trust-weighted fusion
│   ├── conjunctions/              # Collision prediction, CDM, screening, rescreen
│   ├── debris_bounties/           # Kessler cleanup market with state machine
│   └── traffic_control/           # Bilateral + multi-party negotiation
├── tests/                         # Integration tests (separate Cargo workspace)
├── tools/celestrak-demo/          # CelesTrak data ingestion CLI
└── tools/screening-daemon/        # Automated conjunction screening daemon
```

## Key Crates

| Crate | Path | Purpose |
|-------|------|---------|
| `orbital-mechanics` | lib/orbital-mechanics/ | Pure math: SGP4, Alfano 2D Pc, CDM parsing, trust-weighted fusion |
| `mycelix-space-shared` | zomes/shared/ | Shared types, trust fabric requirements, consciousness gating |
| `mycelix-space-screener` | tools/screening-daemon/ | Standalone screening daemon (CelesTrak + priority) |
| 5x integrity zomes | dna/zomes/*/integrity/ | Entry types, link types, validation rules |
| 5x coordinator zomes | dna/zomes/*/coordinator/ | CRUD, DHT link indexing, cross-zome calls, queries |

## Universal Trust Fabric (Consciousness Gating)

All state-modifying operations are gated by consciousness tier via `gate_space_operation()`:

| Operation | Min Tier | Rationale |
|-----------|----------|-----------|
| Submit TLE / observation | Observer | Open data, more = better |
| Create conjunction event | Participant | Avoid false alarms |
| Update risk / CDM / fusion | Citizen | Affects operational decisions |
| Create bounty / verification | Steward | Financial commitment |
| Cosign agreement | Steward | Binding operational agreement |

Read operations (queries, screening) are **ungated** — open access.

The gate calls `CallTargetCell::OtherRole("identity")` to the identity cluster. If unreachable (standalone deployment), falls back to bootstrap mode (allows operation with warning).

Requirement builders are in `zomes/shared/src/lib.rs`:
- `requirement_for_tle_submission()`, `requirement_for_observation()` — Observer
- `requirement_for_conjunction_creation()` — Participant (identity >= 0.25)
- `requirement_for_risk_update()`, `requirement_for_fusion()` — Citizen
- `requirement_for_bounty_creation()`, `requirement_for_bounty_verification()` — Steward
- `requirement_for_negotiation()` — Citizen (identity >= 0.3, community >= 0.2)
- `requirement_for_agreement_signing()` — Steward (identity >= 0.5, community >= 0.3)

## Trust-Weighted Observation Fusion

The `FusionPipeline` in `orbital-mechanics/src/fusion.rs` supports trust-weighted quality:

```
effective_quality = data_quality × (trust_floor + (1 - trust_floor) × trust_weight)
```

- `TrustWeighting` struct: `trust_floor` (default 0.3), `sensor_trust: HashMap<String, f64>`
- Unknown sensors get floor weight (0.3), fully trusted get full quality
- Wired in observations coordinator via `lookup_agent_trust_level()` → cross-role call to identity cluster's `trust_credentials` zome

## Multi-Party Traffic Coordination

Extends bilateral negotiation to N-operator scenarios:

**Entry types** (traffic_control integrity):
- `ConjunctionProposal` — N affected operators, pre-computed maneuver options, voting deadline, quorum threshold
- `OperatorVote` — weighted vote with justification
- `MultiPartyAgreement` — cosigned by quorum of approving operators

**State machine**: `Voting → Approved → Executing → Completed` (also `Rejected`, `Expired`)

**Extern functions**:
- `create_conjunction_proposal` — create with maneuver options (Citizen gate)
- `vote_on_proposal` — weighted vote with double-vote prevention (Observer gate)
- `tally_proposal` — count votes, check quorum, create agreement if approved
- `get_proposals_for_conjunction` / `get_votes_for_proposal` / `get_operator_proposals` — queries

**Existing bilateral negotiation is untouched** — backward compatible.

## Screening Daemon

Standalone CLI at `tools/screening-daemon/` for automated conjunction screening:

```bash
# Single run against CelesTrak space stations
mycelix-space-screener \
  --celestrak-url "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle" \
  --protected-objects 25544,48274 \
  --hours-ahead 72 --once --format json

# Daemon mode (every 15 min, re-fetch catalog every 4th cycle)
mycelix-space-screener \
  --celestrak-url "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle" \
  --protected-objects 25544 \
  --interval-seconds 900 --refresh-cycles 4

# Priority screening (higher-tier operators' assets get finer time steps)
mycelix-space-screener --catalog-path catalog.tle \
  --protected-objects 25544,48274 \
  --priority-weights priorities.json --once
```

**Install as systemd service**: `./tools/screening-daemon/install.sh [--enable]`

**Validated**: Successfully screened 32 CelesTrak objects, found CSS vs DUPLEX cubesat conjunction at 51.6 km / Pc 4.7e-7.

## Conjunction Re-screening

`rescreen_conjunction(event_hash)` — re-screens an existing conjunction with fresh TLE data:
- Skips terminal states (Passed, Collision, Mitigated)
- Emits `RiskLevelChanged` signal if risk level crosses a threshold boundary
- Called by screening daemon or on-demand

## Development

```bash
cd /srv/luminous-dynamics/mycelix-space
nix develop                                           # Enter dev shell

# Full workspace check (all 14 crates)
CARGO_TARGET_DIR=target-check cargo check --workspace

# Run orbital mechanics + shared tests (109 tests)
CARGO_TARGET_DIR=target-check cargo test -p orbital-mechanics -p mycelix-space-shared

# Run E2E integration tests (8 tests)
cd tests && CARGO_TARGET_DIR=target-e2e cargo test --test space_coordination_e2e

# Run all integration tests
cd tests && cargo test --test orbital_mechanics --test conjunctions --test propagation --test edge_cases --test space_coordination_e2e

# Build screening daemon
CARGO_TARGET_DIR=target-check cargo build --release -p mycelix-space-screener

# WASM build (zomes only)
cargo build --release --target wasm32-unknown-unknown
```

## Holochain Patterns

### DHT Link Indexing
All coordinators use TypedPath anchors for DHT discoverability:
```rust
fn anchor_for_X(key: T) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("namespace.{}", key));
    let typed = path.typed(LinkTypes::SomeType)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}
```

### Structured Errors
All coordinator errors use `SpaceError` (JSON-serialized in WasmError::Guest):
```rust
SpaceError::new(SpaceErrorCode::BountyNotFound, "Bounty not found")
    .with_context(format!("hash: {}", hash))
    .into_wasm_error()
```

### Pagination
All query endpoints have `_paginated` variants using `PaginationParams` / `PaginatedResponse<T>`.

### Signals
Each coordinator emits typed signals:
- `ConjunctionSignal::Alert` — Medium+ risk conjunctions
- `ConjunctionSignal::RiskLevelChanged` — risk threshold crossings during re-screening
- `BountySignal` — bounty state changes
- `TrafficControlSignal` — bilateral negotiation events
- `MultiPartySignal` — proposal creation, votes, tally, agreement

### Cross-Zome Calls
- `conjunctions` calls `orbital_objects::get_latest_tles` for screening
- `debris_bounties` calls `orbital_objects::get_latest_tles` for tracking status
- `traffic_control` calls `conjunctions::get_conjunctions_for_object` for verification
- `observations` calls identity cluster's `trust_credentials::get_agent_trust_level` for fusion weighting

## Debris Bounties State Machine

```
Open -> Claimed | Expired | Cancelled
Claimed -> InProgress | Open (release)
InProgress -> PendingVerification
PendingVerification -> Completed | InProgress (verification failed)
Terminal: Completed, Expired, Cancelled
```
Enforced by `is_valid_transition()` in the coordinator. Terminal states remove from ActiveBounties anchor.

## Rust SDK

`SpaceClient` in `mycelix-workspace/sdk/src/space.rs` provides typed request builders:
- Screening: `screen_conjunction_request`, `rescreen_conjunction_request`, `get_high_risk_conjunctions_request`
- Multi-party: `create_conjunction_proposal_request`, `vote_on_proposal_request`, `tally_proposal_request`
- Orbital: `submit_tle_request`, `get_latest_tles_request`, `fuse_observations_request`
- Bounties: `create_bounty_request`, `get_active_bounties_request`, `get_bounties_for_debris_request`

## Testing

| Suite | Tests | Command |
|-------|-------|---------|
| orbital-mechanics | 91 | `cargo test -p orbital-mechanics` |
| mycelix-space-shared | 18 | `cargo test -p mycelix-space-shared` |
| E2E space coordination | 8 | `cd tests && cargo test --test space_coordination_e2e` |
| traffic_control integrity | 5 | `cargo test -p traffic_control_integrity` |
| Sweettest (conductor) | 16 | See below |
| **Total** | **138+** | |

**Sweettest** (conductor-based, requires Holochain): `tests/sweettest_integration.rs`
- Needs `LIBCLANG_PATH` set for NixOS (bindgen dependency)
- Run with: `cd tests && CARGO_TARGET_DIR=target-sweettest cargo test --test sweettest_integration -- --ignored`

## Symthaea Integration

**SwarmEvent::SpaceAlert** (feature: `space-alerts` in symthaea):
- 5 alert types: ConjunctionWarning, DebrisProximity, CommWindow, OrbitalAnomaly, ManeuverAnnounced
- Neuromod coupling: conjunction→arousal, debris→valence/confidence, comm window→LR boost
- Dream consolidation via `SpaceEvent` in `OfflineExperienceKind`
- 9 named constants in `thresholds/managers.rs`

**SubstrateType::SpacecraftComputer** (in symthaea-core):
- Radiation degradation model (TID + SEU)
- Per-region CorticalRegion mappings for spacecraft subsystems
- Feasibility: 0.379, EvidenceLevel::Theoretical

## Notes

- `tests/` is a separate Cargo workspace (excluded from main workspace) because sweettest pulls in heavy Holochain conductor dependencies
- WASM `default-members` excludes `orbital-mechanics`, `celestrak-demo`, and `screening-daemon` (native-only)
- `getrandom` 0.3: Do NOT use `js` or `wasm_js` features — Holochain provides `__getrandom_v03_custom`
- Profile: `opt-level = "z"` even in dev (WASM size optimization)
- Screening daemon uses `reqwest` with `rustls-tls` (no OpenSSL dependency on NixOS)
