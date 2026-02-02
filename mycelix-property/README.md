# Mycelix Property

**Land, housing, and asset registries for the Mycelix Civilizational OS**

## Overview

Mycelix Property provides decentralized property registry infrastructure including immutable ownership records, title transfers with escrow, dispute resolution, and commons management for shared resources.

## Zomes

### registry
Immutable ownership records:
- Property registration with geospatial anchoring
- Title deed management
- Ownership history chain
- Fractional ownership support
- Property metadata and attachments

### transfer
Title transfer with escrow:
- Transfer initiation and acceptance
- Multi-party escrow
- Condition verification
- Automatic completion
- Rollback on failure

### disputes
Competing claim resolution:
- Boundary disputes
- Ownership challenges
- Lien management
- Integration with mycelix-justice

### commons
Shared resource management:
- Common pool resources
- Usage rights and quotas
- Maintenance responsibilities
- Benefit distribution
- Governance rules

## Architecture

```
mycelix-property/
├── dna/
│   └── dna.yaml              # DNA manifest
├── zomes/
│   ├── registry/
│   │   ├── integrity/        # Title validation
│   │   └── coordinator/      # Registry management
│   ├── transfer/
│   │   ├── integrity/        # Transfer validation
│   │   └── coordinator/      # Escrow and completion
│   ├── disputes/
│   │   ├── integrity/        # Dispute validation
│   │   └── coordinator/      # Claim resolution
│   └── commons/
│       ├── integrity/        # Commons validation
│       └── coordinator/      # Resource management
├── client/                   # TypeScript client
└── tests/                    # Integration tests
```

## Property Types

- **Land**: Parcels with geospatial boundaries
- **Buildings**: Structures with addresses
- **Units**: Apartments, condos, offices
- **Equipment**: Vehicles, machinery
- **Intellectual**: Patents, copyrights, trademarks
- **Digital**: NFTs, domains, credentials

## Integration Points

- **mycelix-identity**: Owner DID verification
- **mycelix-finance**: Mortgages and liens
- **mycelix-justice**: Dispute escalation
- **mycelix-governance**: Zoning and land use
- **mycelix-energy**: Energy asset registration

## Building

```bash
# Build all zomes
cargo build --release --target wasm32-unknown-unknown

# Package the hApp
hc app pack .
```

## License

Apache-2.0
