# What is Mycelix Knowledge?

Mycelix Knowledge is a **decentralized knowledge graph** built on [Holochain](https://holochain.org) that enables communities to collectively build, verify, and reason about knowledge.

## Core Features

### Epistemic Classification (E-N-M)

Every claim in the knowledge graph is classified along three orthogonal dimensions:

| Dimension | Description | Example |
|-----------|-------------|---------|
| **Empirical** | Verifiable through observation | "Water boils at 100°C" |
| **Normative** | Value-based, ethical | "Privacy is important" |
| **Mythic** | Meaning-making, narrative | "Progress is inevitable" |

This classification enables:
- **Appropriate verification** - Empirical claims get fact-checked differently than normative ones
- **Nuanced understanding** - Most claims are mixtures, not pure types
- **Intellectual honesty** - Distinguishing facts from values from narratives

### Credibility Scoring

Claims receive credibility scores based on multiple factors:

- **Source Diversity** - Independent verification from multiple sources
- **Author Reputation** - Track record of accurate claims
- **Temporal Consistency** - Stability of claim over time
- **Cross-Validation** - Agreement with related claims

### Knowledge Graph

Claims form an interconnected graph through typed relationships:

```
[Claim A] --SUPPORTS--> [Claim B]
[Claim C] --CONTRADICTS--> [Claim B]
[Claim D] --DEPENDS_ON--> [Claim A]
[Claim E] --REFINES--> [Claim A]
```

This enables:
- **Belief propagation** - Updates cascade through dependencies
- **Contradiction detection** - Identify inconsistencies
- **Information value ranking** - Prioritize verification efforts

### Fact-Checking API

Query the knowledge graph to verify statements:

```typescript
const result = await client.factcheck.factCheck({
  statement: "The Earth is round",
  minConfidence: 0.8,
});

// Result:
// {
//   verdict: "TRUE",
//   confidence: 0.98,
//   explanation: "Overwhelming empirical evidence...",
//   supportingClaims: [...],
//   contradictingClaims: [],
//   sources: ["NASA", "ESA", ...]
// }
```

### Verification Markets

Integration with [Epistemic Markets](/guide/markets/verification) enables:

- **Spawn markets** for claims needing verification
- **Incentivize** truth-seeking through prediction markets
- **Update credibility** based on market resolution

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Applications                             │
│  (Demo App, VS Code Extension, Browser Extension, Widgets)   │
├─────────────────────────────────────────────────────────────┤
│                     SDK Layer                                │
│  (TypeScript Client, React Hooks, Svelte Stores)            │
├─────────────────────────────────────────────────────────────┤
│                     API Layer                                │
│  (GraphQL Server, REST API, WebSocket Subscriptions)        │
├─────────────────────────────────────────────────────────────┤
│                     Holochain DNA                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ Claims  │ │  Graph  │ │  Query  │ │Inference│           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │ Bridge  │ │ Markets │ │FactCheck│                       │
│  └─────────┘ └─────────┘ └─────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Principles

### Decentralization
- No central authority controls truth
- Each agent maintains their own DHT shard
- Consensus emerges from collective verification

### Epistemic Humility
- All claims have confidence intervals
- Uncertainty is explicit, not hidden
- Classification acknowledges claim types

### Composability
- Builds on Holochain's agent-centric model
- Integrates with Epistemic Markets
- Extensible through zome bridging

### Transparency
- All credibility factors are visible
- Reasoning chains are traceable
- No black-box verdicts

## Next Steps

- [Quick Start](/guide/quickstart) - Get running in 5 minutes
- [Core Concepts](/guide/concepts) - Deep dive into the framework
- [SDK Documentation](/sdk/) - Build with the TypeScript SDK
