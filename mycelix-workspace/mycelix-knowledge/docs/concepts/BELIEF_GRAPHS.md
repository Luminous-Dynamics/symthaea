# Belief Graphs

Belief Graphs are the connective tissue of Mycelix Knowledge, modeling how beliefs about claims propagate through a network of dependencies.

## Overview

In Mycelix Knowledge, claims don't exist in isolation. They support, contradict, and depend on each other. The Belief Graph captures these relationships and propagates belief strength changes automatically.

```
┌─────────────────────────────────────────────────────────────┐
│                    BELIEF GRAPH                              │
│                                                              │
│    [Claim A] ──supports──▶ [Claim B] ──supports──▶ [Claim C]│
│       │                       │                              │
│       │                       │                              │
│   contradicts             derived-from                       │
│       │                       │                              │
│       ▼                       ▼                              │
│    [Claim D]              [Claim E]                         │
│                                                              │
│    When A's belief changes, B, C, D, E are all updated      │
└─────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Belief Node

Each claim has an associated `BeliefNode` that tracks:

```typescript
interface BeliefNode {
  claimId: string;           // The claim this node represents
  beliefStrength: number;    // Current belief (0.0 to 1.0)
  priorBelief: number;       // Belief before propagation
  confidence: number;        // How certain we are
  supportCount: number;      // Supporting connections
  contradictionCount: number; // Contradicting connections
  converged: boolean;        // Has propagation stabilized?
  influences: BeliefInfluence[];
}
```

### Belief Influences

Influences capture how one claim affects belief in another:

```typescript
interface BeliefInfluence {
  sourceClaimId: string;    // Influencing claim
  influenceType: InfluenceType;
  weight: number;           // Strength of influence (0-1)
  sourceBelief: number;     // Source's belief at influence time
}

type InfluenceType =
  | "Support"           // Increases target belief
  | "Contradiction"     // Decreases target belief
  | "Evidential"        // Evidence relationship
  | "Entailment"        // Logical implication
  | "MarketVerification"; // From resolved market
```

## Dependency Types

Claims can be connected through various dependency types:

| Type | Effect | Example |
|------|--------|---------|
| **Supports** | Positive influence | "Study confirms hypothesis" |
| **Contradicts** | Negative influence | "Counter-evidence found" |
| **DerivedFrom** | Inherited belief | "Conclusion from premises" |
| **Generalizes** | Abstraction | "All mammals breathe" from "Dogs breathe" |
| **Specializes** | Specification | "Border Collies are smart" from "Dogs are smart" |
| **PartOf** | Compositional | "Engine is part of car" |
| **Causes** | Causal relationship | "Smoking causes cancer" |
| **Equivalent** | Bidirectional | "H₂O" = "Water" |

## Belief Propagation Algorithm

When a claim's belief changes, the graph propagates updates:

### Step 1: Initialize
```typescript
// Start with the changed claim
const queue = [changedClaimId];
const visited = new Set<string>();
const maxIterations = 100;
const convergenceThreshold = 0.001;
```

### Step 2: Iterative Relaxation
```typescript
while (queue.length > 0 && iterations < maxIterations) {
  const current = queue.shift();

  // Calculate new belief from influences
  let newBelief = beliefNode.priorBelief;

  for (const influence of beliefNode.influences) {
    if (influence.influenceType === "Support") {
      newBelief += influence.weight * influence.sourceBelief * 0.3;
    } else if (influence.influenceType === "Contradiction") {
      newBelief -= influence.weight * influence.sourceBelief * 0.3;
    }
  }

  // Clamp to [0, 1]
  newBelief = Math.max(0, Math.min(1, newBelief));

  // Check for convergence
  if (Math.abs(newBelief - beliefNode.beliefStrength) > convergenceThreshold) {
    beliefNode.beliefStrength = newBelief;
    // Add dependents to queue
    queue.push(...getDependents(current));
  }
}
```

### Step 3: Cascade Effects
```typescript
// Notify verification markets
for (const marketLink of getMarketLinks(claimId)) {
  await notifyMarket(marketLink.marketId, {
    type: "BeliefChanged",
    newBelief: beliefNode.beliefStrength
  });
}
```

## Using Belief Graphs

### Creating Dependencies

```typescript
import { KnowledgeClient } from '@mycelix/knowledge-sdk';

const knowledge = new KnowledgeClient(client);

// Register that Claim B depends on Claim A
await knowledge.claims.registerDependency(
  "claim-b-id",  // Dependent claim
  "claim-a-id",  // Dependency claim
  "Supports",    // Dependency type
  0.8,           // Weight (0-1)
  "Claim A provides direct evidence for Claim B"
);
```

### Propagating Belief

```typescript
// After updating a claim's evidence or verification status
const result = await knowledge.graph.propagateBelief("claim-a-id");

console.log(`Propagation affected ${result.nodesAffected} claims`);
console.log(`Converged in ${result.iterations} iterations`);
console.log(`Max belief change: ${result.maxDelta}`);
```

### Analyzing Dependencies

```typescript
// Get dependency tree
const tree = await knowledge.graph.getDependencyTree("claim-id", 5);

console.log(`Tree depth: ${tree.depth}`);
console.log(`Total dependencies: ${tree.totalDependencies}`);

// Find circular dependencies (problematic!)
const cycles = await knowledge.graph.detectCircularDependencies("claim-id");

if (cycles.length > 0) {
  console.warn("Circular dependencies detected:", cycles);
}
```

### Calculating Impact

```typescript
// Before making a change, see what would be affected
const impact = await knowledge.graph.calculateCascadeImpact("claim-id");

console.log(`Would affect ${impact.totalAffected} claims`);
console.log(`Max depth: ${impact.maxDepth}`);
console.log(`High-impact claims: ${impact.highImpactClaims}`);

if (impact.totalAffected > 100) {
  console.warn("This change has significant cascade effects");
}
```

## Information Value

Information Value helps prioritize which claims should be verified:

```typescript
interface InformationValue {
  claimId: string;
  expectedValue: number;     // Expected value of resolving
  dependentCount: number;    // Claims that depend on this
  uncertainty: number;       // Current uncertainty
  impactScore: number;       // Potential impact on graph
  recommendedForVerification: boolean;
}
```

### Ranking Claims for Verification

```typescript
// Get claims that would be most valuable to verify
const rankings = await knowledge.graph.rankByInformationValue(10);

for (const iv of rankings) {
  console.log(`Claim ${iv.claimId}:`);
  console.log(`  Expected value: ${iv.expectedValue}`);
  console.log(`  Dependents: ${iv.dependentCount}`);
  console.log(`  Uncertainty: ${iv.uncertainty}`);
  console.log(`  Recommended: ${iv.recommendedForVerification}`);
}
```

## Visualization

The `BeliefGraphService` provides visualization-ready data:

```typescript
import { BeliefGraphService } from '@mycelix/knowledge-sdk';

const graphService = new BeliefGraphService(client);

// Get visualization graph
const viz = await graphService.getVisualizationGraph("claim-id", 3, 50);

// Use with your visualization library
renderGraph({
  nodes: viz.nodes.map(n => ({
    id: n.id,
    size: n.size,
    color: n.color,  // Based on belief strength
    label: n.label
  })),
  edges: viz.edges.map(e => ({
    source: e.source,
    target: e.target,
    width: e.weight * 3,
    color: e.color  // Based on influence type
  }))
});
```

## Consistency Checking

Detect inconsistencies in the belief graph:

```typescript
const analysis = await graphService.analyzeConsistency("claim-id");

if (!analysis.isConsistent) {
  console.log("Inconsistencies found:");
  for (const inc of analysis.inconsistencies) {
    console.log(`  ${inc.claimId}: expected ${inc.expectedBelief}, got ${inc.actualBelief}`);
    console.log(`  Deviation: ${inc.deviation}`);
    console.log(`  Reason: ${inc.reason}`);
  }
}

console.log(`Overall consistency: ${analysis.overallConsistencyScore}`);
console.log("Recommendations:", analysis.recommendations);
```

## Best Practices

### Do
- ✅ Create dependencies when claims logically relate
- ✅ Use appropriate weights (not everything is 1.0)
- ✅ Check for circular dependencies before creating links
- ✅ Propagate beliefs after significant updates
- ✅ Monitor high-impact claims

### Don't
- ❌ Create dependencies between unrelated claims
- ❌ Use weights of 1.0 unless truly absolute
- ❌ Create circular dependency chains
- ❌ Ignore propagation failures
- ❌ Skip impact analysis for important claims

## Integration with Markets

When verification markets resolve:

1. Market resolution triggers `on_market_resolved()`
2. Claim's E level is updated based on outcome
3. Belief propagation cascades through dependents
4. Dependent markets are notified

```typescript
// This happens automatically when markets resolve
// But can be triggered manually for testing
await knowledge.claims.cascadeUpdate("claim-id");
```

## Related Documentation

- [Epistemic Classification](./EPISTEMIC_CLASSIFICATION.md) - Claim positioning
- [Credibility Engine](./CREDIBILITY_ENGINE.md) - Trust assessment
- [Information Value](./INFORMATION_VALUE.md) - Verification prioritization
- [Epistemic Markets](../integration/EPISTEMIC_MARKETS.md) - Market integration

---

*"Knowledge is a web, not a pile."*
