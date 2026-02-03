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
├── lib/orbital-mechanics/     # Core orbital mechanics (no Holochain deps)
│   ├── tle.rs                 # TLE parsing and validation
│   ├── state.rs               # State vectors with covariance
│   ├── covariance.rs          # 6x6 uncertainty matrices
│   ├── propagator.rs          # SGP4/SDP4 orbital propagation
│   ├── conjunction.rs         # Collision probability analysis
│   └── coordinates.rs         # Frame transformations (TEME, ECI, ECEF, geodetic)
│
├── zomes/shared/              # Shared types across all DNAs
│   └── lib.rs                 # NoradId, SpaceTimestamp, TrustLevel, CDM types
│
└── dna/zomes/                 # Holochain DNA Zomes
    ├── orbital_objects/       # Catalog of tracked objects
    ├── observations/          # Sensor data ingestion
    ├── conjunctions/          # Collision prediction & CDMs
    ├── debris_bounties/       # Kessler cleanup market
    └── traffic_control/       # Automated negotiation
```

## Key Features

### 1. Orbital Object Catalog
Track satellites, debris, and rocket bodies with decentralized consensus.
- TLE submission and validation
- Operator claims and verification
- Object metadata (RCS, mass, HBR)

### 2. Sensor Observations
Ingest data from ground and space-based sensors.
- Angles-only (optical)
- Radar range/range-rate
- Full state vectors with covariance

### 3. Conjunction Analysis
Calculate collision probabilities with proper uncertainty handling.
- Covariance propagation
- 2D Alfano Pc calculation
- Conjunction Data Messages (CDMs)

### 4. Debris Bounties (Kessler Cleanup Market)
Crowdfunded incentives for debris removal.
- Post bounties on threatening debris
- Aggregate funding from multiple parties
- Verified removal and payout

### 5. Automated Traffic Control
AI-mediated negotiation between operators.
- Capability and preference exchange
- Maneuver proposal generation
- Cryptographic agreement signing

## Why Covariance Matters

**This network tracks "probability clouds", not points.**

Every orbital state includes a 6x6 covariance matrix representing uncertainty. This enables:
- Meaningful collision probability (miss distance alone is meaningless)
- Proper conjunction screening (filter by statistical significance)
- Trust-weighted data fusion (lower uncertainty = higher weight)
- Zero-knowledge proofs (prove properties without revealing orbits)

## Building

```bash
# Check the library (no Holochain)
cargo check -p orbital-mechanics

# Check all zomes
cargo check --workspace

# Build WASM zomes
cargo build --release --target wasm32-unknown-unknown
```

## Development Status

- [x] Orbital mechanics library (complete)
- [x] TLE parsing with checksum validation
- [x] State vectors with covariance
- [x] SGP4 propagation wrapper
- [x] Conjunction probability calculation
- [x] Coordinate transformations
- [x] Shared zome types
- [x] Orbital objects integrity zome
- [x] Orbital objects coordinator zome
- [x] Observations zomes
- [x] Conjunctions zomes
- [x] Debris bounties zomes
- [x] Traffic control zomes
- [ ] DNA packaging
- [ ] Integration tests
- [ ] Web UI

## License

MIT OR Apache-2.0

## Related Projects

- [mycelix-health](../mycelix-health) - Decentralized health records with PQE
- [mycelix-pulse](../01-resonant-coherence/core/the-pulse) - Decentralized messaging

---

*"The stars belong to no nation, and neither should the knowledge of what moves among them."*
