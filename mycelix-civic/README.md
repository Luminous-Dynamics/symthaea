# Mycelix Civic

**Civic Operations Platform** -- justice, emergency response, and media attribution consolidated into a single Holochain DNA.

## Architecture

Three civic domains share one DNA, enabling direct cross-domain calls via `call(CallTargetCell::Local, ...)`. A unified bridge zome handles inter-cluster communication with mycelix-commons.

```
mycelix-civic/
├── crates/
│   └── civic-types/              # Shared evidence, status, role types
├── dna/
│   └── dna.yaml                  # DNA manifest (16 integrity + 16 coordinator zomes)
├── happ.yaml                     # hApp bundle manifest
├── zomes/
│   ├── justice-cases/            # Case filing & management
│   ├── justice-evidence/         # Evidence chain of custody
│   ├── justice-arbitration/      # Arbitration panels
│   ├── justice-restorative/      # Restorative justice circles
│   ├── justice-enforcement/      # Enforcement actions
│   ├── emergency-incidents/      # Disaster declarations
│   ├── emergency-triage/         # Triage & severity assessment
│   ├── emergency-resources/      # Resource allocation
│   ├── emergency-coordination/   # Multi-agency coordination
│   ├── emergency-shelters/       # Shelter management
│   ├── emergency-comms/          # Emergency communications
│   ├── media-publication/        # Content publishing
│   ├── media-attribution/        # Authorship & licensing
│   ├── media-factcheck/          # Fact-checking workflows
│   ├── media-curation/           # Community curation
│   └── civic-bridge/             # Unified cross-cluster bridge
└── tests/
    └── sweettest_integration.rs  # 14 conductor-based tests
```

## Zome Inventory

| Domain | Zomes | Purpose |
|--------|-------|---------|
| Justice | 5 | Cases, evidence, arbitration, restorative circles, enforcement |
| Emergency | 6 | Incidents, triage, resources, coordination, shelters, comms |
| Media | 4 | Publication, attribution, fact-checking, curation |
| Bridge | 1 | Cross-cluster dispatch to mycelix-commons |
| **Total** | **16** | **16 integrity + 16 coordinator = 32 WASM crates** |

## Build

```bash
nix develop                                            # Enter environment
cargo build                                            # Build library crates
cargo build --release --target wasm32-unknown-unknown   # Build WASM zomes
hc dna pack dna/ -o dna/mycelix_civic.dna              # Pack DNA bundle
hc app pack . -o mycelix-civic.happ                    # Pack hApp bundle
```

## Test

```bash
cargo test                    # Unit tests (640 tests)
cargo test --test sweettest_integration -- --ignored   # Sweettest (requires conductor)
```

## Cross-Domain Integration

Zomes within the same DNA call each other directly:

```rust
// media-factcheck referencing emergency incidents for disaster claims
let result = call(
    CallTargetCell::Local,
    ZomeName::from("emergency_incidents"),
    FunctionName::from("get_disaster"),
    None,
    disaster_hash,
)?;
```

The civic bridge dispatches cross-cluster queries to mycelix-commons:

```rust
// Query commons cluster for property ownership
let query = CivicQuery {
    domain: "property".into(),
    query_type: "verify_ownership".into(),
    params: serde_json::to_string(&property_id)?,
    ..Default::default()
};
call(CallTargetCell::Local, "civic_bridge", "broadcast_query", None, query)?;
```

## Version Compatibility

| Component | Version |
|-----------|---------|
| Holochain | 0.6.0 |
| HDK | 0.6.0 |
| HDI | 0.7.0 |
| getrandom | 0.3 (`getrandom_backend="custom"`) |
