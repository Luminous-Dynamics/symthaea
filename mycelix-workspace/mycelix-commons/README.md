# Mycelix Commons

**Universal Resource Coordination** -- property, housing, care, mutual aid, water, food, and transport consolidated into a single Holochain DNA.

## Architecture

Seven resource domains share one DNA, enabling direct cross-domain calls via `call(CallTargetCell::Local, ...)` without network overhead. A unified bridge zome handles inter-cluster communication with mycelix-civic.

```
mycelix-commons/
├── crates/
│   └── commons-types/       # Shared entry types, batch ops, anchors
├── dna/
│   └── dna.yaml             # DNA manifest (35 integrity + 35 coordinator zomes)
├── happ.yaml                # hApp bundle manifest
├── zomes/
│   ├── property-registry/   # Land & asset registries
│   ├── property-transfer/   # Ownership transfers
│   ├── property-disputes/   # Dispute resolution
│   ├── property-commons/    # Commons management
│   ├── housing-units/       # Unit management
│   ├── housing-membership/  # Cooperative membership
│   ├── housing-finances/    # Financial tracking
│   ├── housing-maintenance/ # Maintenance requests
│   ├── housing-clt/         # Community Land Trust
│   ├── housing-governance/  # Governance & voting
│   ├── care-timebank/       # Time banking
│   ├── care-circles/        # Care circles
│   ├── care-matching/       # Provider-need matching
│   ├── care-plans/          # Care plan management
│   ├── care-credentials/    # Caregiver credentials
│   ├── mutualaid-needs/     # Need posting
│   ├── mutualaid-circles/   # Aid circles
│   ├── mutualaid-governance/# Circle governance
│   ├── mutualaid-pools/     # Resource pooling
│   ├── mutualaid-requests/  # Request management
│   ├── mutualaid-resources/ # Resource tracking
│   ├── mutualaid-timebank/  # Time-based exchange
│   ├── water-flow/          # Flow monitoring
│   ├── water-purity/        # Quality testing
│   ├── water-capture/       # Rainwater harvesting
│   ├── water-steward/       # Stewardship roles
│   ├── water-wisdom/        # Traditional knowledge
│   ├── food-production/     # Farm & garden management
│   ├── food-distribution/   # Distribution networks
│   ├── food-preservation/   # Food preservation
│   ├── food-knowledge/      # Agricultural knowledge
│   ├── transport-routes/    # Route management
│   ├── transport-sharing/   # Vehicle sharing
│   ├── transport-impact/    # Carbon tracking
│   └── commons-bridge/      # Unified cross-cluster bridge
└── tests/
    ├── sweettest_integration.rs  # 14 conductor-based tests
    └── cross_domain_sweettest.rs # Cross-domain integration tests
```

## Zome Inventory

| Domain | Zomes | Purpose |
|--------|-------|---------|
| Property | 4 | Land registries, transfers, disputes, commons |
| Housing | 6 | Units, membership, finances, maintenance, CLT, governance |
| Care | 5 | Timebank, circles, matching, plans, credentials |
| Mutual Aid | 7 | Needs, circles, governance, pools, requests, resources, timebank |
| Water | 5 | Flow, purity, capture, steward, wisdom |
| Food | 4 | Production, distribution, preservation, knowledge |
| Transport | 3 | Routes, sharing, impact |
| Bridge | 1 | Cross-cluster dispatch to mycelix-civic |
| **Total** | **35** | **35 integrity + 35 coordinator = 70 WASM crates** |

## Build

```bash
nix develop                                            # Enter environment
cargo build                                            # Build library crates
cargo build --release --target wasm32-unknown-unknown   # Build WASM zomes
hc dna pack dna/ -o dna/mycelix_commons.dna            # Pack DNA bundle
hc app pack . -o mycelix-commons.happ                  # Pack hApp bundle
```

## Test

```bash
cargo test                    # Unit tests (396 tests)
cargo test --test sweettest_integration -- --ignored   # Sweettest (requires conductor)
```

## Cross-Domain Integration

Zomes within the same DNA call each other directly:

```rust
// housing-clt verifying property ownership
let result = call(
    CallTargetCell::Local,
    ZomeName::from("property_registry"),
    FunctionName::from("get_property"),
    None,
    property_hash,
)?;
```

The commons bridge dispatches cross-cluster queries to mycelix-civic:

```rust
// Query civic cluster for active justice cases
let query = CommonsQuery {
    domain: "justice".into(),
    query_type: "active_cases".into(),
    params: serde_json::to_string(&area_id)?,
    ..Default::default()
};
call(CallTargetCell::Local, "commons_bridge", "broadcast_query", None, query)?;
```

## Version Compatibility

| Component | Version |
|-----------|---------|
| Holochain | 0.6.0 |
| HDK | 0.6.0 |
| HDI | 0.7.0 |
| getrandom | 0.3 (`getrandom_backend="custom"`) |
