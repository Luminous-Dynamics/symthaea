# Mycelix-Space: Decentralized Space Situational Awareness

## Quick Reference

- **Location**: /srv/luminous-dynamics/mycelix-space/
- **Stack**: Rust + Holochain 0.7 (HDK 0.6 / HDI 0.7) + SGP4 orbital mechanics
- **Status**: Active development — Phase 3 hardening complete
- **CI**: `.github/workflows/ci.yml` — check + test + integration + docs + WASM build + hApp packaging

## Architecture

```
mycelix-space/
├── lib/orbital-mechanics/     # Pure Rust orbital mechanics (no Holochain deps)
│   └── src/                   # SGP4, Alfano Pc, CDM, coordinates, covariance (~2,800 LOC)
├── zomes/shared/              # mycelix-space-shared: types shared across all zomes
│   └── src/lib.rs             # SpaceTimestamp, QualityScore, RiskLevel, SpaceError, pagination (~1,100 LOC)
├── dna/zomes/                 # 5 Holochain zomes (integrity + coordinator each)
│   ├── orbital_objects/       # Catalog of tracked objects + TLE management
│   ├── observations/          # Sensor data ingestion (optical, radar, laser, RF)
│   ├── conjunctions/          # Collision prediction, CDM, screening (SGP4 + Alfano)
│   ├── debris_bounties/       # Kessler cleanup market with state machine
│   └── traffic_control/       # Automated negotiation + bilateral cosigning
├── tests/                     # Integration tests (~96 tests, separate Cargo workspace)
├── tools/celestrak-demo/      # CelesTrak data ingestion CLI
└── flake.nix                  # Nix dev environment
```

## Key Crates

| Crate | Path | Purpose |
|-------|------|---------|
| `orbital-mechanics` | lib/orbital-mechanics/ | Pure math: SGP4 propagation, Alfano 2D Pc, CDM parsing |
| `mycelix-space-shared` | zomes/shared/ | Shared types, SpaceError, pagination, signals, alerts |
| 5x integrity zomes | dna/zomes/*/integrity/ | Entry types, link types, validation rules |
| 5x coordinator zomes | dna/zomes/*/coordinator/ | CRUD, DHT link indexing, cross-zome calls, queries |

## Development

```bash
cd /srv/luminous-dynamics/mycelix-space
nix develop                                           # Enter dev shell

# Full workspace check (all 12 crates)
CARGO_TARGET_DIR=target-check cargo check --workspace

# Run orbital mechanics unit tests (25/25)
cargo test -p orbital-mechanics

# Run integration tests (separate workspace)
cd tests && cargo test --test orbital_mechanics --test conjunctions --test propagation --test edge_cases

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
Clients can `serde_json::from_str::<SpaceError>` the error message for machine-readable codes.

### Pagination
All query endpoints have `_paginated` variants using `PaginationParams` / `PaginatedResponse<T>`.

### Signals
Each coordinator emits typed signals (e.g., `BountySignal`, `ConjunctionSignal`, `TrafficControlSignal`).

### Cross-Zome Calls
- `conjunctions` calls `orbital_objects::get_latest_tles` for screening
- `debris_bounties` calls `orbital_objects::get_latest_tles` for tracking status
- `traffic_control` calls `conjunctions::get_conjunctions_for_object` for verification

## Debris Bounties State Machine

```
Open -> Claimed | Expired | Cancelled
Claimed -> InProgress | Open (release)
InProgress -> PendingVerification
PendingVerification -> Completed | InProgress (verification failed)
Terminal: Completed, Expired, Cancelled
```
Enforced by `is_valid_transition()` in the coordinator. Terminal states remove from ActiveBounties anchor.

## Testing

- **Unit tests**: `cargo test -p orbital-mechanics` — 25 tests covering SGP4, Alfano Pc, CDM parsing
- **Integration tests** (in `tests/` workspace):
  - `orbital_mechanics_tests.rs` — TLE parsing, propagation, ISS orbit validation
  - `conjunctions_tests.rs` — Conjunction assessment, risk levels, CDM creation
  - `propagation_tests.rs` — Multi-object propagation, state vector accuracy
  - `edge_case_tests.rs` — Invalid inputs, boundary conditions, degenerate orbits
- **Sweettest** (conductor-based, requires Holochain): `tests/sweettest_integration.rs` — 16 tests
  - Needs `LIBCLANG_PATH` set for NixOS (bindgen dependency)
  - Run with: `cd tests && CARGO_TARGET_DIR=target-sweettest cargo test --test sweettest_integration -- --ignored`

## Notes

- `tests/` is a separate Cargo workspace (excluded from main workspace) because sweettest pulls in heavy Holochain conductor dependencies
- WASM `default-members` excludes `orbital-mechanics` and `celestrak-demo` (native-only)
- `getrandom` 0.3: Do NOT use `js` or `wasm_js` features — Holochain provides `__getrandom_v03_custom`
- Profile: `opt-level = "z"` even in dev (WASM size optimization)
