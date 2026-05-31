# Tutorial 3: Building Relationships

Learn how to connect claims in the knowledge graph.

## Why Relationships Matter

Claims don't exist in isolation. Building relationships:
- Enables belief propagation
- Reveals contradictions
- Strengthens credibility through corroboration
- Powers the fact-checking engine

## Relationship Types

| Type | Description | Example |
|------|-------------|---------|
| Supports | Evidence for | "Study A supports Claim B" |
| Contradicts | Evidence against | "Study A contradicts Claim B" |
| DerivedFrom | Conclusion from | "B is derived from A" |
| Generalizes | Abstraction of | "All X" from "This X" |
| Specializes | Instance of | "This X" from "All X" |
| Causes | Causal link | "A causes B" |
| PartOf | Component | "A is part of B" |
| Equivalent | Same meaning | "A = B" |

## Creating Relationships

### Basic Relationship

```typescript
import { KnowledgeClient } from '@mycelix/knowledge-sdk';

const knowledge = new KnowledgeClient(client);

// Create a "supports" relationship
await knowledge.graph.createRelationship(
  sourceClaimId,    // The supporting claim
  targetClaimId,    // The claim being supported
  "Supports",       // Relationship type
  0.8,              // Weight (0-1)
  { notes: "Direct evidence" }  // Optional properties
);
```

### Dependency Registration

For belief graph propagation:

```typescript
// Register a dependency
await knowledge.claims.registerDependency(
  dependentClaimId,    // The claim that depends
  dependencyClaimId,   // The claim depended upon
  "DerivedFrom",       // Dependency type
  0.7,                 // Weight
  "Conclusion based on the source claim's data"  // Justification
);
```

## Example: Building a Claim Network

```typescript
// Create primary claim
const primaryClaim = await knowledge.claims.createClaim({
  content: "Global average temperature increased 1.1°C since pre-industrial times",
  classification: { empirical: 0.9, normative: 0.1, mythic: 0.5 },
  domain: "climate"
});

// Create supporting claims
const supportingClaim1 = await knowledge.claims.createClaim({
  content: "NASA GISS data shows 1.1°C warming",
  classification: { empirical: 0.95, normative: 0.05, mythic: 0.3 },
  domain: "climate"
});

const supportingClaim2 = await knowledge.claims.createClaim({
  content: "NOAA analysis confirms 1.0-1.2°C warming range",
  classification: { empirical: 0.95, normative: 0.05, mythic: 0.3 },
  domain: "climate"
});

// Link them
await knowledge.graph.createRelationship(
  supportingClaim1.toString(),
  primaryClaim.toString(),
  "Supports",
  0.9
);

await knowledge.graph.createRelationship(
  supportingClaim2.toString(),
  primaryClaim.toString(),
  "Supports",
  0.85
);

// Register dependencies for belief propagation
await knowledge.claims.registerDependency(
  primaryClaim.toString(),
  supportingClaim1.toString(),
  "DerivedFrom",
  0.9
);

await knowledge.claims.registerDependency(
  primaryClaim.toString(),
  supportingClaim2.toString(),
  "DerivedFrom",
  0.85
);
```

## Checking for Contradictions

```typescript
// Before creating a claim, check for contradictions
const potentialContradictions = await knowledge.query.search(
  "temperature decrease cooling",
  { domain: "climate", minE: 0.6 }
);

if (potentialContradictions.length > 0) {
  console.log("Found potential contradicting claims:");
  for (const c of potentialContradictions) {
    console.log(`  - ${c.content}`);
    // Link as contradiction
    await knowledge.graph.createRelationship(
      myClaimId,
      c.id,
      "Contradicts",
      0.7
    );
  }
}
```

## Visualizing Relationships

```typescript
import { BeliefGraphService } from '@mycelix/knowledge-sdk';

const graphService = new BeliefGraphService(client);

// Get visualization data
const viz = await graphService.getVisualizationGraph(claimId, 3, 50);

// Render (using your visualization library)
renderGraph({
  nodes: viz.nodes,
  edges: viz.edges
});
```

## Best Practices

1. **Be Accurate** - Only create relationships that are true
2. **Use Appropriate Weights** - Not everything is 1.0
3. **Provide Justification** - Explain why claims are related
4. **Check for Cycles** - Avoid circular dependencies
5. **Update When Wrong** - Relationships can be removed

## Next Steps

- [Tutorial 4: Verification Markets](./04-VERIFICATION_MARKETS.md)
- [Belief Graphs](../concepts/BELIEF_GRAPHS.md)

---

*Connect knowledge to build understanding.*
