# Mycelix Identity

**Universal Self-Sovereign Identity for the Mycelix Civilizational OS**

## Overview

Mycelix Identity provides decentralized identity management for the entire Mycelix ecosystem. It enables:

- **DID:mycelix** - Decentralized identifiers for all agents
- **Verifiable Credentials** - W3C-compliant credentials with Mycelix extensions
- **Social Recovery** - No seed phrase loss via trusted contacts
- **Privacy** - Selective disclosure and zero-knowledge proofs

## Architecture

```
mycelix-identity/
├── dna/
│   └── dna.yaml              # DNA manifest
├── zomes/
│   ├── did_registry/         # DID:mycelix format, key management
│   │   ├── integrity/        # Entry validation
│   │   └── coordinator/      # Business logic
│   ├── credential_schema/    # Schema definitions
│   │   ├── integrity/
│   │   └── coordinator/
│   ├── revocation/           # Credential revocation registry
│   │   ├── integrity/
│   │   └── coordinator/
│   └── recovery/             # Social recovery
│       ├── integrity/
│       └── coordinator/
├── client/                   # TypeScript client
└── tests/                    # Integration tests
```

## Zomes

### did_registry

Manages DID:mycelix identifiers:

- Create/update/deactivate DIDs
- Key rotation with continuity
- Service endpoint management
- Cross-hApp identity queries

### credential_schema

Defines credential types for the ecosystem:

- Standard schemas (education, employment, identity)
- Custom schema registration
- Schema versioning
- Validation rules

### revocation

Handles credential lifecycle:

- Revocation list management
- Suspension (temporary revocation)
- Status queries for verifiers
- Batch operations

### recovery

Enables social recovery:

- Trustee designation (3-7 contacts)
- Threshold-based recovery (60% default)
- Time-locked recovery process
- Emergency recovery protocols

## DID Format

```
did:mycelix:<agent_pub_key>

Example:
did:mycelix:uhCAkX1234567890abcdef...
```

## Integration

All Mycelix hApps use Identity for:

```rust
// Query identity from any hApp
use mycelix_identity::did_registry::get_did_document;

let did_doc = get_did_document(agent_pub_key)?;
```

## Development

```bash
# Enter Nix environment
cd /srv/luminous-dynamics/mycelix-workspace
nix develop

# Build zomes
cd ../mycelix-identity
cargo build --release --target wasm32-unknown-unknown

# Package DNA
hc dna pack dna/

# Package hApp
hc app pack .

# Run tests
cd tests && npm test
```

## Part of the Civilizational OS

Identity is the foundation of the Three-Pillar architecture:

- **Governance Pillar**: Voting eligibility, proposal authorship
- **Economic Pillar**: Credit scoring, property ownership
- **Knowledge Pillar**: Author attribution, fact-checker credentials

## License

Apache 2.0
