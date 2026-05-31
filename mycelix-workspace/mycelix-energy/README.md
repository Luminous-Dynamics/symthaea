# Mycelix Energy

**Energy coordination and community ownership - Terra Atlas Bridge for the Mycelix Civilizational OS**

## Overview

Mycelix Energy provides decentralized energy infrastructure including project registry, investment tracking, regenerative exit mechanisms, and peer-to-peer energy trading. This hApp serves as the bridge between the Mycelix ecosystem and Terra Atlas, enabling community ownership of energy assets.

## Zomes

### projects
Energy project registry:
- Solar, wind, hydro, nuclear, storage projects
- Project metadata and specifications
- Operational status tracking
- Terra Atlas data synchronization
- Community readiness metrics

### investments
Pledge and investment tracking:
- Investment pledges and commitments
- Share ownership records
- Dividend distribution
- Investment portfolio management
- Tax-optimized returns tracking

### regenerative
Community ownership evolution:
- Regenerative Exit smart contracts
- Ownership transition automation
- Community readiness assessment
- Reserve account management
- Transition milestone tracking

### grid
Peer-to-peer energy trading:
- Energy production records
- Consumption tracking
- P2P trading matches
- Price discovery
- Settlement and billing

## Architecture

```
mycelix-energy/
├── dna/
│   └── dna.yaml              # DNA manifest
├── zomes/
│   ├── projects/
│   │   ├── integrity/        # Project validation
│   │   └── coordinator/      # Project management
│   ├── investments/
│   │   ├── integrity/        # Investment validation
│   │   └── coordinator/      # Investment tracking
│   ├── regenerative/
│   │   ├── integrity/        # Transition validation
│   │   └── coordinator/      # Exit management
│   └── grid/
│       ├── integrity/        # Trading validation
│       └── coordinator/      # P2P trading
├── client/                   # TypeScript client
└── tests/                    # Integration tests
```

## Terra Atlas Integration

The bridge to Terra Atlas provides:
- **Project Discovery**: Import project data from Terra Atlas API
- **Investment Flows**: Track investments through Mycelix Finance
- **Community Metrics**: Use EduNet credentials for readiness assessment
- **Governance Integration**: Transition decisions via Mycelix Governance

## Regenerative Exit Model

```
Conditions-Based Ownership Transition:
├── Community Readiness (EduNet credentials)
├── Financial Sustainability (reserve accounts)
├── Operational Competence (certified operators)
├── Governance Maturity (voting participation)
└── Investor Returns (minimum IRR met)
```

## Integration Points

- **mycelix-identity**: Investor and community member verification
- **mycelix-finance**: Investment flows and dividend distribution
- **mycelix-governance**: Transition decisions and policy
- **mycelix-edunet**: Operator certification and readiness
- **mycelix-property**: Energy asset registration
- **Terra Atlas**: External project discovery and data

## Building

```bash
# Build all zomes
cargo build --release --target wasm32-unknown-unknown

# Package the hApp
hc app pack .
```

## License

Apache-2.0
