# Mycelix Workspace

Central orchestration layer for the Mycelix ecosystem — a fractal CivOS built on Holochain.

## Purpose

This workspace bundles individual Mycelix domain clusters into unified hApps, provides shared infrastructure crates, and hosts development tooling for the entire ecosystem.

## Structure

```
mycelix-workspace/
├── happs/           # hApp manifests bundling multiple DNAs
│   └── *.happ       # Compiled hApp bundles (mycelix-unified.happ, etc.)
├── crates/          # Shared Rust infrastructure
│   ├── mycelix-fl/          # Federated Learning orchestration
│   ├── mycelix-fl-core/     # FL core algorithms (Byzantine-resilient)
│   ├── mycelix-fl-proofs/   # Zero-knowledge FL proofs
│   ├── mycelix-zome-helpers/ # Shared Holochain zome utilities
│   └── sweettest-harness/   # Integration test harness for sweettests
├── sdk/             # Rust SDK for Mycelix integration
├── sdk-ts/          # TypeScript SDK (37 modules, ~226K LOC, 6,316 tests)
├── sdk-python/      # Python SDK
├── sdk-wasm/        # WebAssembly SDK
├── dashboard/       # React/Vite web dashboard
├── gateway/         # Gateway/bridge infrastructure
├── deploy/          # Deployment configuration
└── ci-templates/    # CI/CD templates for standalone repos
```

## Ecosystem

This workspace orchestrates 25 domain-specific clusters:

| Tier | Clusters |
|------|----------|
| **Core** | commons, civic, hearth, identity, governance, personal |
| **Economy** | finance (MYCEL/SAP/TEND), music, marketplace, attribution |
| **Services** | health (FHIR), praxis, energy, climate, supplychain, manufacturing |
| **Knowledge** | knowledge, desci, core (FL research) |
| **Communication** | mail (PQC-encrypted) |
| **Frontier** | space (orbital coordination), lunar |

## Unified hApp

`happs/mycelix-unified-happ.yaml` bundles all clusters into a single Holochain application where cross-cluster calls use `CallTargetCell::OtherRole`. Routing is centralized in `crates/mycelix-bridge-common/src/routing_registry.rs` (13 routes).

## Federated Learning

The `mycelix-fl-core` crate implements Byzantine-resilient federated learning that breaks the traditional 33% Byzantine fault tolerance limit. FL proofs provide zero-knowledge verification of model updates.

## SDKs

- **Rust** (`sdk/`): 18 modules, ~50K LOC, 1,036+ tests
- **TypeScript** (`sdk-ts/`): 37 modules, ~226K LOC, 6,316 tests
- **Python** (`sdk-python/`): Bindings for data science workflows
- **WASM** (`sdk-wasm/`): Browser-compatible Mycelix client

## Development

```bash
# Build a specific cluster
just build-commons
just build-civic

# Run sweettests
cd crates/sweettest-harness && cargo test

# Pack a hApp bundle
hc app pack happs/
```

## License

AGPL-3.0-or-later. See `COMMERCIAL_LICENSE.md` at repository root for commercial terms.
