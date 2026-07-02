# TypeScript SDK

The Mycelix Knowledge SDK provides a complete TypeScript interface for interacting with the decentralized knowledge graph.

## Installation

::: code-group

```bash [npm]
npm install @mycelix/knowledge-client
```

```bash [yarn]
yarn add @mycelix/knowledge-client
```

```bash [pnpm]
pnpm add @mycelix/knowledge-client
```

:::

## Quick Start

```typescript
import { KnowledgeClient } from '@mycelix/knowledge-client';

// Initialize the client
const client = new KnowledgeClient({
  url: 'ws://localhost:8888',
  appId: 'mycelix-knowledge',
});

// Connect to Holochain
await client.connect();

// Create a claim
const claimId = await client.claims.createClaim({
  content: 'The speed of light is approximately 299,792 km/s.',
  classification: {
    empirical: 0.99,
    normative: 0.005,
    mythic: 0.005,
  },
  tags: ['physics', 'constants'],
  sources: ['NIST', 'BIPM'],
});

// Retrieve the claim
const claim = await client.claims.getClaim(claimId);
console.log(claim.content);
console.log(claim.credibility); // Computed credibility score
```

## Client API

### `KnowledgeClient`

The main entry point for all SDK operations.

```typescript
interface KnowledgeClientConfig {
  url: string;           // Holochain conductor URL
  appId: string;         // hApp ID
  roleName?: string;     // Role name (default: 'knowledge')
  timeout?: number;      // Request timeout in ms
}

const client = new KnowledgeClient(config);
```

### Available Clients

| Client | Purpose |
|--------|---------|
| `client.claims` | Create, read, update, delete claims |
| `client.graph` | Manage relationships and propagation |
| `client.query` | Search and list claims |
| `client.inference` | Credibility scoring and evidence |
| `client.factcheck` | Fact-checking API |
| `client.marketsIntegration` | Epistemic Markets bridge |

## Claims Client

### Create a Claim

```typescript
const claimId = await client.claims.createClaim({
  content: 'Your claim statement',
  classification: {
    empirical: 0.7,   // 0-1, verifiable through observation
    normative: 0.2,   // 0-1, value-based
    mythic: 0.1,      // 0-1, meaning-making
  },
  tags: ['science', 'technology'],
  sources: ['Source 1', 'Source 2'],
});
```

### Get a Claim

```typescript
const claim = await client.claims.getClaim(claimId);

// Returns:
// {
//   id: string,
//   content: string,
//   classification: { empirical, normative, mythic },
//   author: string,
//   createdAt: string,
//   updatedAt: string,
//   tags: string[],
//   sourceCount: number,
//   credibility?: EnhancedCredibilityScore,
// }
```

### Update a Claim

```typescript
await client.claims.updateClaim({
  originalHash: claimId,
  content: 'Updated claim statement',
  classification: { empirical: 0.8, normative: 0.15, mythic: 0.05 },
});
```

### Delete a Claim

```typescript
await client.claims.deleteClaim(claimId);
```

## Graph Client

### Create Relationships

```typescript
const relationshipId = await client.graph.createRelationship(
  sourceClaimId,    // The supporting/contradicting claim
  targetClaimId,    // The claim being supported/contradicted
  'SUPPORTS',       // SUPPORTS | CONTRADICTS | REFINES | DEPENDS_ON
  0.85              // Relationship strength (0-1)
);
```

### Get Relationships

```typescript
const relationships = await client.graph.getRelationships(claimId);

// Returns array of:
// {
//   id: string,
//   sourceId: string,
//   targetId: string,
//   type: RelationshipType,
//   weight: number,
//   createdAt: string,
// }
```

### Propagate Belief

```typescript
const result = await client.graph.propagateBelief(claimId);

// Returns:
// {
//   claimsUpdated: number,
//   iterations: number,
//   converged: boolean,
//   affectedClaims: string[],
// }
```

### Get Dependency Tree

```typescript
const tree = await client.graph.getDependencyTree(claimId, maxDepth);

// Returns:
// {
//   root: TreeNode,
//   depth: number,
//   totalNodes: number,
// }
```

## Query Client

### Search Claims

```typescript
const results = await client.query.search('climate change', {
  epistemicType: 'EMPIRICAL',  // EMPIRICAL | NORMATIVE | MYTHIC | MIXED
  minCredibility: 0.5,
  tags: ['science'],
  limit: 20,
  offset: 0,
  sortBy: 'RELEVANCE',         // RELEVANCE | CREDIBILITY | DATE | INFORMATION_VALUE
});
```

### List Claims

```typescript
const claims = await client.query.listClaims({
  author: 'did:key:...',
  minCredibility: 0.6,
  tags: ['physics'],
  limit: 20,
  cursor: 'abc123',  // For pagination
});

// Returns:
// {
//   items: Claim[],
//   total: number,
//   cursor?: string,
// }
```

## Inference Client

### Calculate Credibility

```typescript
const credibility = await client.inference.calculateEnhancedCredibility(
  claimId,
  'Claim'  // Entity type
);

// Returns:
// {
//   score: number,           // 0-1
//   factors: {
//     sourceDiversity: number,
//     authorReputation: number,
//     temporalConsistency: number,
//     crossValidation: number,
//     matlComposite?: number,
//   },
//   confidence: number,      // 0-1
//   calculatedAt: string,
// }
```

### Get Author Reputation

```typescript
const reputation = await client.inference.getAuthorReputation(authorDid);

// Returns:
// {
//   did: string,
//   overallScore: number,
//   domainScores: { [domain: string]: number },
//   claimCount: number,
//   accuracyHistory: number,
// }
```

## Fact-Check Client

### Fact-Check a Statement

```typescript
const result = await client.factcheck.factCheck({
  statement: 'The Earth is approximately 4.5 billion years old.',
  context: 'geology discussion',  // Optional context
  minEpistemicLevel: 2,           // Minimum E-level for supporting claims
  minConfidence: 0.6,             // Minimum confidence threshold
});

// Returns:
// {
//   verdict: 'TRUE' | 'MOSTLY_TRUE' | 'MIXED' | 'MOSTLY_FALSE' | 'FALSE' | 'UNVERIFIABLE' | 'INSUFFICIENT_EVIDENCE',
//   confidence: number,
//   explanation: string,
//   supportingClaims: ClaimSummary[],
//   contradictingClaims: ClaimSummary[],
//   sources: string[],
//   checkedAt: string,
// }
```

### Batch Fact-Check

```typescript
const results = await client.factcheck.batchFactCheck([
  { statement: 'Statement 1' },
  { statement: 'Statement 2' },
  { statement: 'Statement 3' },
]);
```

## Markets Integration Client

### Spawn Verification Market

```typescript
const marketId = await client.marketsIntegration.spawnVerificationMarket({
  claimId: claimId,
  targetE: 4,              // Target epistemic level
  minConfidence: 0.7,
  closesAt: Date.now() + 14 * 24 * 60 * 60 * 1000,  // 14 days
  initialSubsidy: 100,
  tags: ['verification'],
});
```

### Get Claim Markets

```typescript
const markets = await client.marketsIntegration.getClaimMarkets(claimId);

// Returns array of:
// {
//   marketId: string,
//   claimId: string,
//   targetE: number,
//   currentProbability: number,
//   volume: number,
//   closesAt: string,
//   status: 'OPEN' | 'CLOSED' | 'RESOLVED',
// }
```

## Error Handling

```typescript
import { KnowledgeError, NetworkError, ValidationError } from '@mycelix/knowledge-client';

try {
  await client.claims.createClaim({ ... });
} catch (error) {
  if (error instanceof ValidationError) {
    console.error('Invalid input:', error.fields);
  } else if (error instanceof NetworkError) {
    console.error('Network issue:', error.message);
  } else if (error instanceof KnowledgeError) {
    console.error('Knowledge error:', error.code, error.message);
  }
}
```

## TypeScript Types

All types are exported from the package:

```typescript
import type {
  Claim,
  ClaimSummary,
  Classification,
  Relationship,
  RelationshipType,
  EnhancedCredibilityScore,
  FactCheckResult,
  Verdict,
  AuthorReputation,
  MarketSummary,
  PropagationResult,
  DependencyTree,
} from '@mycelix/knowledge-client';
```

## Next Steps

- [React Hooks](/sdk/react/hooks) - Use with React
- [Svelte Stores](/sdk/svelte/stores) - Use with Svelte
- [API Reference](/api/) - Full API documentation
