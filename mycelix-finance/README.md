# Mycelix Finance

**Peer-to-peer lending, credit, and value transfer for the Mycelix Civilizational OS**

## Overview

Mycelix Finance provides decentralized financial infrastructure including MATL-based credit scoring, P2P lending with collateral, multi-currency payments, and community treasury management.

## Zomes

### credit_scoring
MATL-based creditworthiness assessment:
- Credit profiles derived from MATL trust scores
- Payment history tracking
- Collateral ratio calculation
- Cross-hApp reputation aggregation
- Privacy-preserving score computation

### lending
Peer-to-peer loans with collateral:
- Loan requests and offers
- Interest rate calculation (reputation-adjusted)
- Collateral management
- Payment schedules
- Default handling and resolution

### payments
Multi-currency value transfers:
- Direct peer-to-peer payments
- Multi-signature transactions
- Payment channels for frequent transactions
- Currency conversion
- Payment receipts and history

### treasury
Community fund management:
- Community savings pools
- Proposal-based fund allocation
- Yield distribution
- Reserve management
- Emergency fund access

## Architecture

```
mycelix-finance/
├── dna/
│   └── dna.yaml              # DNA manifest
├── zomes/
│   ├── credit_scoring/
│   │   ├── integrity/        # Credit profile validation
│   │   └── coordinator/      # Score calculation
│   ├── lending/
│   │   ├── integrity/        # Loan validation
│   │   └── coordinator/      # Loan management
│   ├── payments/
│   │   ├── integrity/        # Payment validation
│   │   └── coordinator/      # Payment execution
│   └── treasury/
│       ├── integrity/        # Treasury validation
│       └── coordinator/      # Fund management
├── client/                   # TypeScript client
└── tests/                    # Integration tests
```

## Credit Score Formula

```
credit_score = f(
    matl_score * 0.40,           # Trust reputation (40%)
    payment_history * 0.30,       # On-time payments (30%)
    collateral_ratio * 0.15,      # Available collateral (15%)
    account_age * 0.10,           # Time in ecosystem (10%)
    activity_score * 0.05         # Active participation (5%)
)
```

## Integration Points

- **mycelix-identity**: DID for participant verification
- **mycelix-property**: Collateral registration and liens
- **mycelix-governance**: Monetary policy decisions
- **mycelix-justice**: Debt dispute resolution
- **mycelix-energy**: Energy investment flows

## Building

```bash
# Build all zomes
cargo build --release --target wasm32-unknown-unknown

# Package the hApp
hc app pack .
```

## License

Apache-2.0
