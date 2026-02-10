# Mycelix Space

**Decentralized Space Domain Awareness Network**

A Holochain-based peer-to-peer network for tracking orbital objects, predicting conjunctions, and coordinating space traffic without relying on any single nation's Space Force.

## Vision

Transform space situational awareness from a government monopoly into a global commons, enabling:
- Operators to share and verify orbital data
- Communities to protect orbital lanes
- Markets for debris removal (Kessler bounties)
- Automated traffic negotiation between operators

## Architecture

```
mycelix-space/
├── lib/orbital-mechanics/     Core orbital math (no Holochain deps, ~2,800 LOC)
│   ├── tle.rs                 TLE parsing and checksum validation
│   ├── state.rs               State vectors with 6x6 covariance
│   ├── covariance.rs          Uncertainty matrix operations
│   ├── propagator.rs          SGP4/SDP4 orbital propagation
│   ├── conjunction.rs         Alfano 2D collision probability
│   └── coordinates.rs         TEME ↔ ECEF ↔ geodetic, look angles
│
├── zomes/shared/              Shared types (~1,100 LOC)
│   └── lib.rs                 SpaceTimestamp, SpaceError, RiskLevel, pagination, signals
│
├── dna/zomes/                 5 Holochain zomes (integrity + coordinator each)
│   ├── orbital_objects/       Catalog of tracked objects + TLE management
│   ├── observations/          Sensor data ingestion (optical, radar, laser, RF)
│   ├── conjunctions/          Collision prediction, CDM screening (SGP4 + Alfano)
│   ├── debris_bounties/       Kessler cleanup market with state machine
│   └── traffic_control/       Automated negotiation + bilateral cosigning
│
├── tests/                     Integration tests (separate Cargo workspace)
├── tools/celestrak-demo/      CelesTrak data ingestion CLI
├── ui/                        SvelteKit dashboard (static, demo + live mode)
├── workdir/                   Build output (dna.yaml, happ.yaml, *.happ)
└── flake.nix                  Nix dev environment
```

### Zome Summary

| Zome | Entries | Key Features |
|------|---------|--------------|
| **orbital_objects** | OrbitalObject, TwoLineElement, StateVector | TLE deduplication, latest-TLE queries, multi-source fusion |
| **observations** | Observation, Sensor | Multi-sensor indexing, per-object and per-sensor queries |
| **conjunctions** | ConjunctionEvent, CDM, Maneuver | SGP4 screening, Alfano 2D Pc, CDM versioning, risk escalation |
| **debris_bounties** | DebrisBounty, RemovalClaim, BountyContribution | State machine, verification threshold, crowdfunded bounties |
| **traffic_control** | NegotiationSession, Position, Proposal, Agreement | Bilateral negotiation, cosigning, operator indexing |

## Key Features

### Orbital Object Catalog
Track satellites, debris, and rocket bodies with decentralized consensus.
- TLE submission and validation (NORAD checksum)
- Operator claims and verification
- Object metadata (RCS, mass, hard-body radius)

### Sensor Observations
Ingest data from ground and space-based sensors.
- Optical (angles-only), radar (range/range-rate), laser ranging, RF signal
- Per-object and per-sensor DHT indexing
- Quality scoring and pagination

### Conjunction Analysis
Calculate collision probabilities with proper uncertainty handling.
- SGP4 propagation + Alfano 2D analytical Pc
- Conjunction Data Messages (CDM) with versioning
- Risk escalation with typed signals
- Cross-zome screening against the object catalog

### Debris Bounties (Kessler Cleanup Market)
Crowdfunded incentives for debris removal with a full state machine:
```
Open ──→ Claimed ──→ InProgress ──→ PendingVerification ──→ Completed
  │         │                              │
  ├──→ Expired    ├──→ Open (release)      └──→ InProgress (failed)
  └──→ Cancelled
```
Terminal states are automatically removed from the active bounties index.

### Automated Traffic Control
Bilateral negotiation between operators for maneuver coordination.
- Position and proposal exchange per conjunction event
- Cryptographic cosigning of agreements
- Operator session indexing

## Why Covariance Matters

**This network tracks probability clouds, not points.**

Every orbital state includes a 6x6 covariance matrix representing uncertainty. This enables:
- Meaningful collision probability (miss distance alone is meaningless)
- Proper conjunction screening (filter by statistical significance)
- Trust-weighted data fusion (lower uncertainty = higher weight)

## Getting Started

### Prerequisites

- [Nix](https://nixos.org/download) with flakes enabled
- Or: Rust stable with `wasm32-unknown-unknown` target

### Development

```bash
# Enter dev shell (Rust + WASM target + Holochain tools)
nix develop

# Check all 12 crates compile
cargo check --workspace

# Run orbital mechanics unit tests (25 tests)
cargo test -p orbital-mechanics

# Run integration tests (separate workspace)
cd tests && cargo test

# Build WASM zomes
cargo build --release --target wasm32-unknown-unknown
```

### CelesTrak Demo

```bash
# Export satellite catalog to JSON
cargo run -p celestrak-demo -- export --source iss --output iss.json

# Ingest TLE data
cargo run -p celestrak-demo -- ingest --source stations
```

## Testing

| Suite | Tests | Coverage |
|-------|-------|----------|
| **orbital-mechanics** (unit) | 25 | SGP4, Alfano Pc, TLE, coordinates, covariance |
| orbital_mechanics_tests | 13 | TLE parsing, propagation, ISS orbit validation |
| conjunctions_tests | 18 | Risk assessment, CDM creation, Alfano Pc |
| propagation_tests | 17 | Multi-object propagation, state vector accuracy |
| edge_case_tests | 55 | Invalid inputs, boundary conditions, degenerate orbits |
| zome_tests (integration) | 30 | Cross-zome interaction patterns |
| sweettest (conductor) | 22 | Full Holochain DHT tests (requires conductor) |

## Design Patterns

### DHT Link Indexing
Every coordinator uses TypedPath anchors for DHT discoverability:
```rust
fn anchor_for_object(norad_id: u32) -> ExternResult<AnyLinkableHash> {
    let path = Path::from(format!("objects.{}", norad_id));
    let typed = path.typed(LinkTypes::ObjectIndex)?;
    typed.ensure()?;
    Ok(typed.path_entry_hash()?.into())
}
```

### Structured Errors
All coordinators return `SpaceError` (JSON-serialized in `WasmError::Guest`):
```rust
SpaceError::new(SpaceErrorCode::InvalidTransition, "Cannot transition")
    .with_context("bounty already finalized")
    .into_wasm_error()
```

### Pagination
Every query endpoint has a `_paginated` variant using `PaginationParams` / `PaginatedResponse<T>`.

### Signals
Each coordinator emits typed signals on state changes for UI reactivity.

### Zero-Knowledge Orbit Proofs
Operators can prove orbital properties without revealing proprietary ephemeris:
- **Hash commitments**: SHA3-256 of (ephemeris || nonce)
- **Claims**: altitude range, collision freedom, disposal compliance, slot compliance, zone avoidance
- **Proof systems**: HashReveal, Bulletproofs, Groth16, PLONK
- **Certificates**: Verified collision-freedom receipts that skip full screening

## Zome API Reference (57 extern functions)

### orbital_objects (5 functions)

| Function | Input | Returns | Description |
|----------|-------|---------|-------------|
| `register_object` | `RegisterObjectInput` | `ActionHash` | Register a new tracked object |
| `submit_tle` | `SubmitTleInput` | `ActionHash` | Submit a Two-Line Element set |
| `claim_operator` | `ClaimOperatorInput` | `ActionHash` | Claim operator status for an object |
| `get_latest_tles` | `Vec<u32>` | `Vec<TleLines>` | Get latest TLEs for NORAD IDs |
| `get_tles_with_metadata` | `GetTlesWithMetadataInput` | `Vec<TleWithMetadata>` | Get TLEs with staleness assessment |

### observations (8 functions)

| Function | Input | Returns | Description |
|----------|-------|---------|-------------|
| `submit_observation` | `SubmitObservationInput` | `ActionHash` | Submit sensor observation |
| `register_sensor` | `RegisterSensorInput` | `ActionHash` | Register a ground/space sensor |
| `get_observations_for_object` | `u32` | `Vec<Observation>` | All observations for a NORAD ID |
| `get_sensor_observations` | `String` | `Vec<Observation>` | All observations from a sensor |
| `list_sensors` | `()` | `Vec<Sensor>` | List all registered sensors |
| `get_observations_paginated` | `PaginatedObjectObsQuery` | `PaginatedResponse<Observation>` | Paginated object observations |
| `get_sensor_observations_paginated` | `PaginatedSensorObsQuery` | `PaginatedResponse<Observation>` | Paginated sensor observations |
| `list_sensors_paginated` | `PaginationParams` | `PaginatedResponse<Sensor>` | Paginated sensor list |

### conjunctions (14 functions)

| Function | Input | Returns | Description |
|----------|-------|---------|-------------|
| `create_conjunction_event` | `CreateEventInput` | `ActionHash` | Create a conjunction event |
| `update_conjunction_risk` | `UpdateRiskInput` | `ActionHash` | Update risk level of an event |
| `submit_cdm` | `SubmitCdmInput` | `ActionHash` | Submit a Conjunction Data Message |
| `announce_maneuver` | `AnnounceManeuverInput` | `ActionHash` | Announce a planned maneuver |
| `mark_maneuver_executed` | `ManeuverExecutedInput` | `ActionHash` | Confirm a maneuver was executed |
| `screen_conjunction` | `ScreenInput` | `Vec<ConjunctionAssessment>` | SGP4 + Alfano Pc screening |
| `screen_conjunction_with_staleness` | `ScreenWithStalenessInput` | `Vec<StalenessAwareAssessment>` | Screening with TLE quality metadata |
| `screen_conjunction_from_tles` | `ScreenFromTlesInput` | `Vec<ConjunctionAssessment>` | Screen from raw TLE strings |
| `get_high_risk_conjunctions` | `()` | `Vec<ConjunctionEvent>` | All active high-risk events |
| `get_conjunctions_for_object` | `u32` | `Vec<ConjunctionEvent>` | Events involving a NORAD ID |
| `get_cdms_for_event` | `String` | `Vec<CdmEntry>` | CDM history for an event |
| `get_maneuvers_for_event` | `String` | `Vec<AvoidanceManeuver>` | Maneuvers for an event |
| `get_conjunctions_for_object_paginated` | `PaginatedConjunctionQuery` | `PaginatedResponse<ConjunctionEvent>` | Paginated conjunction query |
| `get_high_risk_conjunctions_paginated` | `PaginationParams` | `PaginatedResponse<ConjunctionEvent>` | Paginated high-risk events |

### debris_bounties (11 functions)

| Function | Input | Returns | Description |
|----------|-------|---------|-------------|
| `create_bounty` | `CreateBountyInput` | `ActionHash` | Create a debris removal bounty |
| `contribute_to_bounty` | `ContributeInput` | `ActionHash` | Fund an existing bounty |
| `claim_bounty` | `ClaimBountyInput` | `ActionHash` | Claim a bounty for execution |
| `submit_verification` | `SubmitVerificationInput` | `ActionHash` | Submit removal verification |
| `update_bounty_status` | `UpdateBountyStatusInput` | `ActionHash` | Advance the state machine |
| `get_bounties_for_debris` | `u32` | `Vec<DebrisBounty>` | Bounties for a NORAD ID |
| `get_active_bounties` | `()` | `Vec<DebrisBounty>` | All open/active bounties |
| `get_contributions` | `ActionHash` | `Vec<BountyContribution>` | Contributions to a bounty |
| `get_claims` | `ActionHash` | `Vec<RemovalClaim>` | Claims against a bounty |
| `get_bounties_for_debris_paginated` | `PaginatedDebrisQuery` | `PaginatedResponse<DebrisBounty>` | Paginated debris bounties |
| `get_active_bounties_paginated` | `PaginationParams` | `PaginatedResponse<DebrisBounty>` | Paginated active bounties |

### traffic_control (11 functions)

| Function | Input | Returns | Description |
|----------|-------|---------|-------------|
| `initiate_negotiation` | `InitiateNegotiationInput` | `ActionHash` | Start a negotiation session |
| `submit_position` | `SubmitPositionInput` | `ActionHash` | Submit operator's position |
| `submit_proposal` | `SubmitProposalInput` | `ActionHash` | Propose a maneuver plan |
| `accept_proposal` | `AcceptProposalInput` | `ActionHash` | Accept a proposed plan |
| `cosign_agreement` | `CosignAgreementInput` | `ActionHash` | Cosign a bilateral agreement |
| `get_sessions_for_conjunction` | `String` | `Vec<NegotiationSession>` | Sessions for a conjunction |
| `get_session_positions` | `String` | `Vec<NegotiationPosition>` | Positions in a session |
| `get_session_proposals` | `String` | `Vec<ManeuverProposal>` | Proposals in a session |
| `get_operator_sessions` | `AgentPubKey` | `Vec<NegotiationSession>` | Sessions for an operator |
| `get_sessions_paginated` | `PaginatedSessionQuery` | `PaginatedResponse<NegotiationSession>` | Paginated session query |
| `get_operator_sessions_paginated` | `PaginatedOperatorQuery` | `PaginatedResponse<NegotiationSession>` | Paginated operator sessions |

## CI Pipeline

GitHub Actions (`.github/workflows/ci.yml`):
1. **Check & Lint** — `cargo fmt`, `clippy -D warnings`, `cargo-machete`
2. **Test** — `cargo test --workspace` with optional coverage
3. **Integration Tests** — orbital mechanics, conjunctions, propagation, edge cases
4. **Documentation** — `cargo doc --workspace`, dead link check
5. **Build WASM** — all 10 zome WASMs
6. **Package hApp** — DNA + hApp packaging via `hc`

## Stack

- **Runtime**: [Holochain](https://www.holochain.org/) 0.4 (HDK 0.6 / HDI 0.7)
- **Language**: Rust 2021 edition
- **Orbital Math**: [sgp4](https://crates.io/crates/sgp4), [nalgebra](https://nalgebra.org/)
- **Build**: Nix flakes, WASM (`wasm32-unknown-unknown`)

## License

MIT OR Apache-2.0

## Related Projects

- [Mycelix](https://mycelix.net) — The broader decentralized network ecosystem
- [Symthaea](../symthaea) — Holographic Liquid Brain (HDC + IIT + LTC)

---

*"The stars belong to no nation, and neither should the knowledge of what moves among them."*
