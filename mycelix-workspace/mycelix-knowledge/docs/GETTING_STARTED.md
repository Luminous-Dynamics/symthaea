# Getting Started with Mycelix Knowledge

This guide walks you through setting up the Knowledge hApp and making your first knowledge contributions.

## Prerequisites

- [Holochain](https://developer.holochain.org/install/) v0.4+
- [Nix](https://nixos.org/download.html) (recommended for reproducible builds)
- Node.js 18+ (for the TypeScript SDK)

## Installation

### Option 1: Nix (Recommended)

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/mycelix-knowledge
cd mycelix-knowledge

# Enter the Nix development environment
nix develop

# Build the zomes
cargo build --release --target wasm32-unknown-unknown

# Package the DNA
hc dna pack dna/

# Package the hApp
hc app pack .
```

### Option 2: Manual Setup

```bash
# Clone and enter directory
git clone https://github.com/Luminous-Dynamics/mycelix-knowledge
cd mycelix-knowledge

# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add wasm32-unknown-unknown

# Build
cargo build --release --target wasm32-unknown-unknown

# Package
hc dna pack dna/
hc app pack .
```

## Running the hApp

### Development Mode

```bash
# Start a sandbox conductor
hc sandbox generate workdir/

# Run with the hApp
hc sandbox run --app-id knowledge workdir/knowledge.happ
```

### Using the SDK

```bash
# Install the SDK
cd client
npm install

# Run tests
npm test

# Build for distribution
npm run build
```

## Your First Claim

### Using the TypeScript SDK

```typescript
import { AppWebsocket } from '@holochain/client';
import { KnowledgeClient } from '@mycelix/knowledge-sdk';

async function createFirstClaim() {
  // Connect to your local Holochain conductor
  const client = await AppWebsocket.connect('ws://localhost:8888');
  const knowledge = new KnowledgeClient(client);

  // Create your first claim
  const claimHash = await knowledge.claims.createClaim({
    content: "The Earth orbits the Sun once every 365.25 days",
    classification: {
      empirical: 0.95,   // Highly verifiable through observation
      normative: 0.05,   // Not really a value judgment
      mythic: 0.2        // Some cultural significance
    },
    domain: "astronomy",
    topics: ["earth", "sun", "orbital-period"],
    evidence: [{
      id: "nasa-earth-orbit",
      evidenceType: "Empirical",
      source: "https://nasa.gov/earth-orbit",
      content: "NASA orbital mechanics data",
      strength: 0.95
    }]
  });

  console.log("Created claim:", claimHash);
  return claimHash;
}
```

## Understanding Epistemic Classification

Every claim in Mycelix Knowledge is positioned on a 3D cube:

```
         M (Mythic)
          │
          │    ┌─────────────┐
          │   /│            /│
          │  / │           / │
          │ /  │          /  │
          │┌───┼─────────┐   │
          │|   │         │   │
          │|   └─────────┼───┘
          │|  /          │  /
          │| /           │ /
          │|/            │/
          └┴─────────────┴────── E (Empirical)
         /
        /
       N (Normative)
```

### Empirical (E) - Verifiability
- **0.0-0.2**: Subjective experience (feelings, preferences)
- **0.2-0.4**: Testimonial (witness accounts)
- **0.4-0.6**: Private verification (credentials, ZK proofs)
- **0.6-0.8**: Cryptographic (on-chain verification)
- **0.8-1.0**: Measurable (scientific reproducibility)

### Normative (N) - Value Alignment
- **0.0-0.25**: Personal perspective
- **0.25-0.5**: Community consensus
- **0.5-0.75**: Network-wide agreement
- **0.75-1.0**: Universal principles

### Mythic (M) - Narrative Significance
- **0.0-0.25**: Transient (doesn't persist)
- **0.25-0.5**: Temporal (time-limited relevance)
- **0.5-0.75**: Persistent (permanent record)
- **0.75-1.0**: Foundational (core understanding)

## Examples by Claim Type

| Claim Type | E | N | M | Example |
|------------|---|---|---|---------|
| Scientific Fact | 0.9 | 0.1 | 0.3 | "Water boils at 100°C at sea level" |
| Moral Principle | 0.2 | 0.9 | 0.7 | "All humans have inherent dignity" |
| Origin Story | 0.1 | 0.4 | 0.95 | "In the beginning..." |
| Historical Event | 0.75 | 0.3 | 0.8 | "The Apollo 11 landed on July 20, 1969" |
| Personal Opinion | 0.1 | 0.1 | 0.1 | "I prefer tea over coffee" |
| Legal Precedent | 0.6 | 0.8 | 0.85 | "Brown v. Board established..." |

## Next Steps

1. **[Submitting Claims](./tutorials/01-SUBMITTING_CLAIMS.md)** - Learn claim creation in depth
2. **[Querying Knowledge](./tutorials/02-QUERYING_KNOWLEDGE.md)** - Find existing claims
3. **[Building Relationships](./tutorials/03-BUILDING_RELATIONSHIPS.md)** - Connect claims
4. **[Epistemic Markets](./integration/EPISTEMIC_MARKETS.md)** - Verify claims through markets

## Getting Help

- [GitHub Issues](https://github.com/Luminous-Dynamics/mycelix-knowledge/issues)
- [Discord Community](https://discord.gg/mycelix)
- [API Reference](./reference/API_REFERENCE.md)

---

*Welcome to the decentralized knowledge commons.*
