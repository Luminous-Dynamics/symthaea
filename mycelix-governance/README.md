# Mycelix Governance

**Decentralized Governance for the Mycelix Civilizational OS**

## Overview

Mycelix Governance enables collective decision-making for the ecosystem through:

- **Proposals**: Submit Mycelix Improvement Proposals (MIPs)
- **Voting**: MATL-weighted voting with delegation support
- **Execution**: Timelock-protected automatic execution
- **Constitution**: Charter amendments and rule changes

## Architecture

```
mycelix-governance/
├── dna/
│   └── dna.yaml              # DNA manifest
├── zomes/
│   ├── proposals/            # MIP submission and management
│   │   ├── integrity/        # Entry validation
│   │   └── coordinator/      # Business logic
│   ├── voting/               # Vote casting and tallying
│   │   ├── integrity/
│   │   └── coordinator/
│   ├── execution/            # Timelock and execution
│   │   ├── integrity/
│   │   └── coordinator/
│   └── constitution/         # Charter and amendments
│       ├── integrity/
│       └── coordinator/
├── client/                   # TypeScript client
└── tests/                    # Integration tests
```

## Proposal Types

| Type | Duration | Quorum | Approval | Use Case |
|------|----------|--------|----------|----------|
| Standard | 7 days | 25% | 60% | Regular changes |
| Emergency | 24 hours | 10% | 75% | Critical fixes |
| Constitutional | 30 days | 50% | 80% | Charter amendments |

## Vote Weight Calculation

```
vote_weight = MATL_score × stake_amount × participation_bonus

Where:
- MATL_score: Trust level (0.0 to 1.0)
- stake_amount: Tokens committed to vote
- participation_bonus: 1.0 + (0.1 × recent_participation_rate)
```

## Features

### Delegation

Inactive members can delegate voting power:

- Single delegation to trusted representative
- Partial delegation (e.g., delegate 50% of votes)
- Topic-specific delegation (delegate only on finance topics)
- Revocable at any time

### Quadratic Voting

Optional for high-stakes decisions:

- Prevents plutocracy
- Vote cost = votes²
- Encourages moderate positions

### Timelock

All approved proposals enter timelock:

- Standard: 48 hours
- Emergency: 6 hours
- Allows cancellation if issues discovered

## Integration Points

- **Identity**: Voter eligibility verification
- **Finance**: Stake locking and rewards
- **Justice**: Disputed vote resolution
- **Knowledge**: Decision rationale as claims

## Development

```bash
# Enter Nix environment
cd /srv/luminous-dynamics/mycelix-workspace
nix develop

# Build zomes
cd ../mycelix-governance
cargo build --release --target wasm32-unknown-unknown

# Package DNA
hc dna pack dna/

# Package hApp
hc app pack .

# Run tests
cd tests && npm test
```

## License

Apache 2.0
