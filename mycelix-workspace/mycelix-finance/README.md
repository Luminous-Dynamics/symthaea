# Mycelix Finance

**Three-Currency Commons Economic Infrastructure on Holochain**

## Overview

Mycelix Finance implements a three-currency economic system designed for long-term commons sustainability:

| Currency | Nature | Role |
|----------|--------|------|
| **MYCEL** | Soulbound (non-transferable) | Reputation substrate (0.0 - 1.0) |
| **SAP** | Transferable, subject to demurrage | Circulation medium |
| **TEND** | Mutual credit, zero-sum | Time-based exchange (1 TEND = 1 hour) |

Key design principles:
- **Continuous demurrage** on SAP (2% annual) redistributes idle wealth to commons pools
- **Inalienable reserve** (25% minimum) in every commons pool can never be withdrawn
- **Counter-cyclical TEND limits** expand during economic stress (WIR Bank pattern)
- **Progressive fees** based on MYCEL reputation tier
- **Anti-reflexivity guards** prevent circular dependencies between currencies

## Zomes

### recognition
MYCEL reputation through weighted recognition events:
- 4-component score: Participation (40%), Recognition (20%), Validation (20%), Longevity (20%)
- Weighted recognition: a Steward's (0.7) recognition is worth 7x a new Apprentice's (0.1)
- Apprentice lifecycle: onboarding, mentorship, graduation at MYCEL 0.3
- Jubilee normalization (every 4 years) and passive decay (5% annual)

### tend
TEND mutual credit time exchange:
- 1 TEND = 1 hour of service, balances sum to zero
- Dynamic limits: Normal (+-40), Elevated (+-60), High (+-80), Emergency (+-120)
- Quality ratings feed into MYCEL Validation component
- Dispute resolution: direct negotiation -> mediation panel -> governance vote
- Cross-community bilateral clearing with quarterly SAP settlement

### payments
SAP/TEND payment processing:
- Progressive fees: Newcomer (0.10%), Member (0.03%), Steward (0.01%)
- Lazy demurrage computed on every SAP balance read/mutation
- Payment channels for frequent transactions
- Exit protocol: MYCEL dissolves, SAP follows succession preference, TEND forgiven

### treasury
Commons pool management with inalienable reserves:
- 25% of every contribution locked as inalienable reserve (exempt from demurrage)
- Compost distribution: 70% local, 20% regional, 10% global
- Democratic allocation from circulating zone (75%)

### bridge
Cross-hApp communication and collateral bridge:
- Collateral bridge: ETH/USDC -> SAP at oracle rate
- Rate-limited: max 5% of vault per day per member
- Finance event broadcasting across hApps

### staking
SAP collateral staking:
- MYCEL-weighted collateral stakes
- Slashing for bad-faith behavior

## Architecture

```
mycelix-finance/
├── dna/
│   └── dna.yaml              # DNA manifest
├── types/                    # Shared types crate (Currency, FeeTier, etc.)
├── zomes/
│   ├── shared/               # Common utilities (anchor_hash, verify_caller)
│   ├── recognition/
│   │   ├── integrity/        # MYCEL state, recognition events
│   │   └── coordinator/      # Score computation, apprentice lifecycle
│   ├── tend/
│   │   ├── integrity/        # Exchange records, disputes, bilateral balances
│   │   └── coordinator/      # Time exchange, quality ratings, clearing
│   ├── payments/
│   │   ├── integrity/        # Payment/channel/receipt validation
│   │   └── coordinator/      # SAP demurrage, progressive fees, exit protocol
│   ├── treasury/
│   │   ├── integrity/        # Commons pool, inalienable reserve validation
│   │   └── coordinator/      # Pool management, compost distribution
│   ├── bridge/
│   │   ├── integrity/        # Cross-hApp payment, collateral deposit validation
│   │   └── coordinator/      # Bridge deposit/redemption, rate limiting
│   └── staking/
│       ├── integrity/        # Collateral stake validation
│       └── coordinator/      # Stake management
└── tests/                    # Integration tests (sweettest)
```

## Fee Tiers

| Tier | MYCEL Range | Base Fee |
|------|-------------|----------|
| Newcomer | < 0.3 | 0.10% |
| Member | 0.3 - 0.7 | 0.03% |
| Steward | > 0.7 | 0.01% |

## TEND Dynamic Limits

| Oracle State | TEND Limit | Trigger |
|-------------|-----------|---------|
| Healthy | +-40 | Vitality >= 40 |
| Stressed | +-60 | Vitality 20-40 |
| Critical | +-80 | Vitality 10-20 |
| Failing | +-120 | Vitality < 10 |

## Building

```bash
# Enter development shell
nix develop

# Build all zomes (library crates)
cargo build --release

# Build WASM for Holochain
cargo build --release --target wasm32-unknown-unknown

# Run unit tests
cargo test --workspace

# Run integration tests
cd tests && cargo test
```

## Integration Points

- **mycelix-identity**: DID verification for all operations
- **mycelix-property**: Collateral registration
- **mycelix-governance**: Monetary policy (demurrage rate, fee bounds, TEND limits)
- **mycelix-workspace SDK**: Shared economics module (`sdk/src/economics/`)

## License

Apache-2.0
