# Mycelix Knowledge Documentation

**Decentralized Knowledge Graph for the Mycelix Civilizational OS**

Welcome to the Mycelix Knowledge documentation. This guide covers all aspects of the Knowledge hApp, from basic concepts to advanced integration patterns.

## Quick Navigation

### Getting Started
- [Getting Started Guide](./GETTING_STARTED.md) - Set up and run your first query
- [Epistemic Classification](./concepts/EPISTEMIC_CLASSIFICATION.md) - Understand E-N-M positioning
- [Submitting Claims](./tutorials/01-SUBMITTING_CLAIMS.md) - Your first knowledge contribution

### Core Concepts
- [Epistemic Classification](./concepts/EPISTEMIC_CLASSIFICATION.md) - The 3D E-N-M cube
- [Belief Graphs](./concepts/BELIEF_GRAPHS.md) - How beliefs propagate
- [Credibility Engine](./concepts/CREDIBILITY_ENGINE.md) - MATL-enhanced trust
- [Information Value](./concepts/INFORMATION_VALUE.md) - Prioritizing verification

### Integration Guides
- [Epistemic Markets](./integration/EPISTEMIC_MARKETS.md) - Bidirectional market integration
- [Cross-hApp Bridge](./integration/CROSS_HAPP_BRIDGE.md) - Connect any Mycelix hApp
- [MATL Integration](./integration/MATL_INTEGRATION.md) - Trust layer mechanics
- [Fact-Check API](./integration/FACT_CHECK_API.md) - External verification service

### Tutorials
1. [Submitting Claims](./tutorials/01-SUBMITTING_CLAIMS.md)
2. [Querying Knowledge](./tutorials/02-QUERYING_KNOWLEDGE.md)
3. [Building Relationships](./tutorials/03-BUILDING_RELATIONSHIPS.md)
4. [Verification Markets](./tutorials/04-VERIFICATION_MARKETS.md)
5. [Building Integrations](./tutorials/05-BUILDING_INTEGRATIONS.md)

### Reference
- [API Reference](./reference/API_REFERENCE.md) - Complete SDK documentation
- [Zome Functions](./reference/ZOME_FUNCTIONS.md) - All coordinator functions
- [Entry Types](./reference/ENTRY_TYPES.md) - Data structures
- [Error Codes](./reference/ERROR_CODES.md) - Troubleshooting guide

### Governance
- [Epistemic Charter](./governance/EPISTEMIC_CHARTER.md) - Foundational principles
- [Claim Moderation](./governance/CLAIM_MODERATION.md) - Community standards
- [Design Principles](./governance/DESIGN_PRINCIPLES.md) - Architecture decisions

### Operations
- [Security](./operations/SECURITY.md) - Security considerations
- [Performance](./operations/PERFORMANCE.md) - Optimization guide
- [Accessibility](./operations/ACCESSIBILITY.md) - Inclusive design
- [Metrics](./operations/METRICS.md) - Monitoring and observability

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE hApp                              │
├────────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌──────────────────────┐    ┌───────────────┐ │
│  │ claims/ │───▶│ markets_integration/ │───▶│  BRIDGE to    │ │
│  │         │    │                      │    │  Epistemic    │ │
│  │  E-N-M  │◀───│  Bidirectional       │◀───│  Markets      │ │
│  └────┬────┘    └──────────────────────┘    └───────────────┘ │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────┐    ┌──────────────────────┐                      │
│  │ graph/  │───▶│    inference/        │                      │
│  │         │    │                      │                      │
│  │ Belief  │    │  MATL Credibility    │                      │
│  │ Propagn │◀───│  Engine              │                      │
│  └────┬────┘    └──────────────────────┘                      │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────┐    ┌──────────────────────┐                      │
│  │ query/  │    │    factcheck/        │◀── External hApps    │
│  │         │    │                      │    (Media, Gov, etc) │
│  │ Search  │    │  Fact-Check API      │                      │
│  └─────────┘    └──────────────────────┘                      │
└────────────────────────────────────────────────────────────────┘
```

## Key Features

### Epistemic Classification
Every claim is positioned on a 3D cube:
- **E (Empirical)**: How verifiable through observation?
- **N (Normative)**: How aligned with ethical frameworks?
- **M (Mythic)**: What narrative significance?

### Belief Graphs
Claims form a network where:
- Beliefs propagate through relationships
- Contradictions are detected automatically
- Information value guides verification priorities

### MATL Integration
Multi-dimensional Adaptive Trust Layer provides:
- Byzantine fault tolerance (45%)
- Quadratic weighting (matl²)
- Cross-domain reputation

### Market Integration
Bidirectional flow with Epistemic Markets:
- Claims spawn verification markets
- Market resolution updates epistemic levels
- Claims used as evidence in predictions

## Installation

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/mycelix-knowledge

# Enter the workspace
cd mycelix-knowledge
nix develop

# Build the DNA
cargo build --release --target wasm32-unknown-unknown
hc dna pack dna/

# Install the SDK
cd client
npm install
npm run build
```

## Quick Example

```typescript
import { AppWebsocket } from '@holochain/client';
import { KnowledgeClient } from '@mycelix/knowledge-sdk';

// Connect to Holochain
const client = await AppWebsocket.connect('ws://localhost:8888');
const knowledge = new KnowledgeClient(client);

// Create a claim
const claimHash = await knowledge.claims.createClaim({
  content: "Global renewable energy capacity increased 9.6% in 2023",
  classification: { empirical: 0.85, normative: 0.15, mythic: 0.1 },
  domain: "energy",
  topics: ["renewable", "statistics", "2023"]
});

// Fact-check a statement
const result = await knowledge.factcheck.factCheck({
  statement: "Renewable energy is growing rapidly",
  minE: 0.6
});

console.log(`Verdict: ${result.verdict} (${result.confidence * 100}% confidence)`);
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on:
- Submitting issues
- Code contributions
- Documentation improvements
- Community engagement

## License

MIT License - See [LICENSE](../LICENSE)

## Part of Mycelix

Knowledge is a foundational pillar of the Mycelix Civilizational OS:

| Pillar | hApp | Knowledge Integration |
|--------|------|----------------------|
| Identity | mycelix-identity | Author reputation, DID linking |
| Governance | mycelix-governance | Policy claims, decision evidence |
| Justice | mycelix-justice | Case evidence, precedent claims |
| Finance | mycelix-finance | Market predictions, economic claims |
| Media | mycelix-media | Fact-checking, source credibility |
| Education | mycelix-edunet | Learning content, knowledge tests |

---

*Building a shared foundation of verifiable, trustworthy knowledge.*
