# Tutorial 2: Querying Knowledge

Learn how to search and query the knowledge graph effectively.

## Prerequisites

- Connected to Knowledge hApp
- SDK installed

## Basic Search

```typescript
import { KnowledgeClient } from '@mycelix/knowledge-sdk';

const knowledge = new KnowledgeClient(client);

// Simple text search
const results = await knowledge.query.search("climate change temperature");

for (const claim of results) {
  console.log(`Content: ${claim.content}`);
  console.log(`E: ${claim.classification.empirical}, N: ${claim.classification.normative}`);
  console.log(`Domain: ${claim.domain}`);
  console.log('---');
}
```

## Query by Epistemic Position

Find claims within specific epistemic ranges:

```typescript
// Find highly verifiable claims (E > 0.7)
const verifiableClaims = await knowledge.query.queryByEpistemic(
  0.7,    // minE
  1.0,    // maxE
  null,   // minN (any)
  null,   // maxN (any)
  null,   // minM (any)
  null,   // maxM (any)
  50      // limit
);

// Find foundational claims (M > 0.7)
const foundationalClaims = await knowledge.query.queryByEpistemic(
  null, null,  // E: any
  null, null,  // N: any
  0.7, 1.0,    // M: 0.7 to 1.0
  20
);
```

## Find Related Claims

```typescript
// Get claims related to a specific claim
const claimId = "uhCkk...";
const related = await knowledge.query.findRelated(
  claimId,
  2,    // maxDepth: how many hops
  20    // limit
);

console.log(`Found ${related.length} related claims`);
```

## Find Contradictions

```typescript
// Find claims that contradict a given claim
const contradictions = await knowledge.query.findContradictions(claimId);

if (contradictions.length > 0) {
  console.log("Contradicting claims found:");
  for (const c of contradictions) {
    console.log(`  - ${c.content}`);
  }
}
```

## Query by Domain

```typescript
// Get all claims in a domain
const climateClaims = await knowledge.claims.listClaimsByDomain("climate");

// Search within a domain
const energyClaims = await knowledge.query.search("renewable solar", {
  domain: "energy",
  minE: 0.5,
  limit: 30
});
```

## Query by Topic

```typescript
// Search by topic
const results = await knowledge.claims.searchClaimsByTopic("solar-energy");

// Fact-check API can also query by topics
const claims = await knowledge.factcheck.queryClaimsByTopic(
  ["renewable", "2023", "capacity"],
  { minE: 0.6, limit: 20 }
);
```

## Advanced Query

For complex queries:

```typescript
const result = await knowledge.query.advancedQuery({
  filters: [
    { field: "domain", operator: { Equals: "climate" } },
    { field: "classification.empirical", operator: { GreaterThan: 0.6 } },
    { field: "status", operator: { In: ["Published", "Verified"] } },
    { field: "timestamp", operator: { Between: [startTime, endTime] } }
  ],
  sort: [
    { field: "classification.empirical", direction: "Desc" }
  ],
  limit: 50,
  offset: 0
});

console.log(`Total matches: ${result.total}`);
console.log(`Returned: ${result.items.length}`);
console.log(`Has more: ${result.hasMore}`);
```

## Query Options

```typescript
interface ClaimQueryOptions {
  minE?: number;      // Minimum empirical level
  maxE?: number;      // Maximum empirical level
  minN?: number;      // Minimum normative level
  maxN?: number;      // Maximum normative level
  minM?: number;      // Minimum mythic level
  maxM?: number;      // Maximum mythic level
  status?: ClaimStatus;  // Filter by status
  domain?: string;    // Filter by domain
  limit?: number;     // Max results
  offset?: number;    // For pagination
}
```

## Pagination

```typescript
const pageSize = 20;
let offset = 0;
let hasMore = true;

while (hasMore) {
  const result = await knowledge.query.advancedQuery({
    filters: [...],
    limit: pageSize,
    offset
  });

  processClaims(result.items);

  hasMore = result.hasMore;
  offset += pageSize;
}
```

## Tips

1. **Use specific terms** - "global temperature 2023" > "climate"
2. **Set appropriate E levels** - Higher E = more rigorous
3. **Check domains** - Narrow to relevant domain
4. **Use pagination** - For large result sets
5. **Cache results** - Queries can be expensive

## Next Steps

- [Tutorial 3: Building Relationships](./03-BUILDING_RELATIONSHIPS.md)
- [API Reference](../reference/API_REFERENCE.md)

---

*Find the knowledge you need.*
