# Mycelix Attribution Registry

A Holochain hApp for transparent, privacy-preserving tracking of open-source dependency usage and reciprocal stewardship.

## Overview

The Attribution Registry enables organizations to voluntarily declare their dependency on open-source software, generate zero-knowledge proofs of usage scale, and record reciprocal contributions — all on a peer-to-peer DHT with no central authority.

**Philosophy**: Every dependency is an act of trust. Attribution makes that trust visible and reciprocity measurable.

### Key Capabilities

- **Dependency Registry**: Track open-source packages across 8 ecosystems
- **Usage Attestation**: Voluntary usage receipts with optional ZK-STARK privacy proofs
- **Reciprocity Scoring**: Weighted stewardship metrics (amount, recency, type diversity)
- **Zero-Knowledge Proofs**: Prove usage scale without revealing it (Winterfell ZK-STARK)
- **CLI Scanner**: Auto-detect dependencies from 8 lockfile formats
- **Rate Limiting**: Per-agent sliding-window throttling prevents DHT spam

## Architecture

```
mycelix-attribution/
├── zomes/
│   ├── registry/          # DependencyIdentity CRUD
│   │   ├── integrity/     # Entry types, link types, validation
│   │   └── coordinator/   # Register, query, verify, bulk, paginate
│   ├── usage/             # UsageReceipt + UsageAttestation
│   │   ├── integrity/     # Immutable receipts, ZK-STARK attestations
│   │   └── coordinator/   # Record, attest, verify, revoke, renew
│   └── reciprocity/       # ReciprocityPledge + StewardshipScore
│       ├── integrity/     # Pledge types, currency, validation
│       └── coordinator/   # Pledge, score, leaderboard, under-supported
├── cli/                   # Lockfile scanner (8 formats)
├── verifier/              # Off-chain ZK-STARK proof verification + Ed25519 signing
├── prover/                # ZK-STARK proof generation
├── stark-common/          # Shared AIR definition (prover + verifier)
├── tests/                 # Sweettest integration tests (25 scenarios, requires conductor)
├── dna/                   # DNA manifest
└── happ.yaml              # hApp bundle manifest
```

## Entry Types

### DependencyIdentity (Registry)

| Field | Type | Description |
|-------|------|-------------|
| `id` | `String` | Unique ID (e.g., `crate:serde:1.0`) |
| `name` | `String` | Human-readable name |
| `ecosystem` | `DependencyEcosystem` | RustCrate, NpmPackage, PythonPackage, NixFlake, GoModule, RubyGem, MavenPackage, Other |
| `maintainer_did` | `String` | DID of the maintainer |
| `repository_url` | `Option<String>` | Source repository URL |
| `version` | `Option<String>` | Version string |
| `verified` | `bool` | Whether verified by maintainer |

### UsageReceipt (Usage)

| Field | Type | Description |
|-------|------|-------------|
| `dependency_id` | `String` | References DependencyIdentity.id |
| `user_did` | `String` | DID of the user/organization |
| `usage_type` | `UsageType` | DirectDependency, Transitive, InternalTooling, Production, Research |
| `scale` | `Option<UsageScale>` | Small, Medium, Large, Enterprise |

### UsageAttestation (Usage)

| Field | Type | Description |
|-------|------|-------------|
| `witness_commitment` | `Vec<u8>` | 32-byte Blake3 commitment |
| `proof_bytes` | `Vec<u8>` | Winterfell ZK-STARK proof (up to 500KB) |
| `verified` | `bool` | Whether proof has been verified |
| `verifier_pubkey` | `Option<Vec<u8>>` | Ed25519 verifier public key |

### ReciprocityPledge (Reciprocity)

| Field | Type | Description |
|-------|------|-------------|
| `dependency_id` | `String` | References DependencyIdentity.id |
| `pledge_type` | `PledgeType` | Financial, Compute, Bandwidth, DeveloperTime, QA, Documentation, Other |
| `amount` | `Option<f64>` | Monetary amount (if Financial) |
| `currency` | `Option<Currency>` | USD, EUR, GBP, BTC, ETH, Other |

## Coordinator Functions

### Registry (10 externs)

| Function | Description |
|----------|-------------|
| `register_dependency` | Create + link with O(1) ID lookup |
| `update_dependency` | Author-only update |
| `get_dependency` | O(1) lookup by ID |
| `get_all_dependencies` | List all registered |
| `get_all_dependencies_paginated` | Paginated listing |
| `get_dependencies_by_ecosystem` | Filter by ecosystem tag |
| `get_maintainer_dependencies` | Filter by maintainer DID |
| `verify_dependency` | Mark as verified |
| `bulk_register_dependencies` | Batch creation (skips duplicates) |
| `get_ecosystem_statistics` | Count by ecosystem + verified |

### Usage (12 externs)

| Function | Description |
|----------|-------------|
| `record_usage` | Record receipt (validates dependency exists, rate-limited 50/min) |
| `bulk_record_usage` | Batch recording (rate-limited 5/min) |
| `get_dependency_usage` | All receipts for a dependency |
| `get_user_usage` | All receipts by user |
| `get_usage_count` | Count of usage links |
| `submit_usage_attestation` | Submit ZK-STARK attestation (validates dep, rate-limited 10/min) |
| `verify_usage_attestation` | Update with verifier pubkey + signature |
| `revoke_attestation` | Author-only delete |
| `renew_attestation` | Create new attestation linked to predecessor |
| `get_dependency_attestations` | List active (filters expired) |
| `get_dependency_usage_paginated` | Paginated receipts |
| `get_top_dependencies` | Top-N by usage count |

### Reciprocity (8 externs)

| Function | Description |
|----------|-------------|
| `record_pledge` | Record pledge (validates dependency, rate-limited 20/min) |
| `acknowledge_pledge` | Mark as acknowledged |
| `get_dependency_pledges` | Pledges for a dependency |
| `get_contributor_pledges` | Pledges by contributor |
| `get_dependency_pledges_paginated` | Paginated pledges |
| `compute_stewardship_score` | Weighted score (cross-zome) |
| `get_stewardship_leaderboard` | Top-N by weighted score |
| `get_under_supported_dependencies` | Lowest stewardship first |

## CLI Usage

### Scanning Lockfiles

```bash
# Rust
mycelix-attribution-scan --lockfile Cargo.lock --did did:mycelix:me

# JavaScript
mycelix-attribution-scan --lockfile package-lock.json --did did:mycelix:me

# Python
mycelix-attribution-scan --lockfile requirements.txt --did did:mycelix:me
mycelix-attribution-scan --lockfile pyproject.toml --did did:mycelix:me

# Go
mycelix-attribution-scan --lockfile go.sum --did did:mycelix:me

# Ruby
mycelix-attribution-scan --lockfile Gemfile.lock --did did:mycelix:me

# Java/Maven
mycelix-attribution-scan --lockfile pom.xml --did did:mycelix:me

# Nix
mycelix-attribution-scan --lockfile flake.lock --did did:mycelix:me

# Batch format for bulk_register_dependencies
mycelix-attribution-scan --lockfile Cargo.lock --did did:mycelix:me --format batch

# Submit directly to conductor
mycelix-attribution-scan --lockfile Cargo.lock --did did:mycelix:me --submit ws://localhost:8888
```

## ZK-STARK Workflow

The proof pipeline: **Prover** generates -> **Verifier** validates -> **DHT** stores.

### 1. Generate Proof

```bash
mycelix-attribution-prover \
  --dependency-id "crate:serde:1.0" \
  --user-did "did:mycelix:abc123" \
  --usage-scale medium \
  --organization "Acme Corp" > attestation.json
```

### 2. Verify Proof

```bash
mycelix-attribution-verifier \
  --attestation attestation.json \
  --signing-key key.bin \
  --format submit-payload > verified.json
```

### 3. Submit to DHT

The `verified.json` contains `original_action_hash`, `verifier_pubkey`, and `verifier_signature` for the `verify_usage_attestation` zome call.

### AIR Design

- **Trace**: 9 columns x 16 rows
- **Columns**: scale, dep_hash_lo, dep_hash_hi, did_hash_lo, did_hash_hi, commitment[0..3]
- **Constraints**: Scale range check (degree 4), 8 column constancy constraints (degree 1)
- **Boundary assertions**: 8 (dep hash, did hash, witness commitment — all bound at row 0)
- **Security**: S128 (128 queries, blowup 16, grinding 20)
- **Proof size**: ~80-100 KB

## SDK-TS Examples

```typescript
import { createAttributionClient } from '@mycelix/sdk/integrations/attribution';

const client = createAttributionClient(appWebsocket);

// Register a dependency
await client.registerDependency({
  id: 'crate:serde:1.0',
  name: 'serde',
  ecosystem: 'RustCrate',
  maintainer_did: 'did:mycelix:maintainer',
  description: 'Serialization framework',
  // ...
});

// Record usage
await client.recordUsage({
  id: 'usage-001',
  dependency_id: 'crate:serde:1.0',
  user_did: 'did:mycelix:user',
  usage_type: 'Production',
  // ...
});

// Compute stewardship
const score = await client.computeStewardshipScore('crate:serde:1.0');
console.log(`Ratio: ${score.ratio}, Weighted: ${score.weighted_score}`);

// Leaderboard
const leaders = await client.getStewardshipLeaderboard(10);
```

## Build & Test

### Prerequisites

```bash
nix develop  # Enter Holochain dev environment
```

### Build WASM Zomes

```bash
cargo build --release --target wasm32-unknown-unknown
hc dna pack dna/
```

### Run Tests

```bash
# Zome unit tests (50 tests: 12 registry + 21 reciprocity + 17 usage)
cargo test --lib -p registry -p registry_integrity -p usage -p usage_integrity \
  -p reciprocity -p reciprocity_integrity

# CLI tests (9 tests: 8 parsers + 1 empty lockfile)
cd cli && cargo test

# Verifier tests (13 tests: commitment, proof bounds, e2e STARK, signing, cross-validation, replay)
cd verifier && cargo test

# Prover tests (3 tests: proof gen, roundtrip, scale validation)
cd prover && cargo test

# Stark-common tests (5 tests: hash, commitment, scale, limbs, public inputs)
cd stark-common && cargo test

# Sweettest integration (31 tests, requires conductor)
cd tests && cargo test -- --ignored

# Total: 111 tests (80 unit + 31 sweettest)
```

### Clippy

```bash
cargo clippy --all-targets -- -D warnings
```

## Integration

### Unified hApp

The attribution DNA is wired into `mycelix-workspace/happs/mycelix-unified-happ.yaml` alongside the identity DNA, enabling:

- DID-to-agent resolution for maintainer notifications
- Cross-role calls via `CallTargetCell::OtherRole("identity")`

### Observatory Dashboard

The attribution dashboard is available at `/attribution` in the observatory SvelteKit app, showing:

- Stats grid (6 KPIs), stewardship leaderboard, top dependencies, under-supported highlights, ecosystem breakdown

---

*Building transparent reciprocity in the open-source commons.*
