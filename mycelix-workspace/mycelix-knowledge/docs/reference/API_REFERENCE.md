# API Reference

Complete reference for the Mycelix Knowledge TypeScript SDK.

## Installation

```bash
npm install @mycelix/knowledge-sdk
```

## KnowledgeClient

The main entry point for all knowledge graph operations.

```typescript
import { AppWebsocket } from '@holochain/client';
import { KnowledgeClient } from '@mycelix/knowledge-sdk';

const client = await AppWebsocket.connect('ws://localhost:8888');
const knowledge = new KnowledgeClient(client, 'knowledge');
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `claims` | `ClaimsClient` | Claims management |
| `graph` | `GraphClient` | Knowledge graph operations |
| `query` | `QueryClient` | Search and queries |
| `inference` | `InferenceClient` | Inference and credibility |
| `marketsIntegration` | `MarketsIntegrationClient` | Epistemic Markets integration |
| `factcheck` | `FactCheckClient` | Fact-checking API |

---

## ClaimsClient

### createClaim

Create a new knowledge claim.

```typescript
async createClaim(input: CreateClaimInput): Promise<ActionHash>
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | `string` | ✅ | Claim statement |
| `classification` | `EpistemicPosition` | ✅ | E-N-M position |
| `domain` | `string` | ✅ | Knowledge domain |
| `topics` | `string[]` | ❌ | Related topics |
| `evidence` | `Evidence[]` | ❌ | Supporting evidence |
| `sources` | `Source[]` | ❌ | Reference sources |

**Returns:** `ActionHash` of created claim

---

### getClaim

Retrieve a claim by hash.

```typescript
async getClaim(claimHash: ActionHash): Promise<Claim | null>
```

---

### updateClaim

Update an existing claim.

```typescript
async updateClaim(input: UpdateClaimInput): Promise<ActionHash>
```

---

### addEvidence

Add evidence to an existing claim.

```typescript
async addEvidence(claimId: string, evidence: Evidence): Promise<ActionHash>
```

---

### spawnVerificationMarket

Create a verification market for a claim.

```typescript
async spawnVerificationMarket(input: SpawnVerificationMarketInput): Promise<ActionHash>
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `claimId` | `string` | ✅ | Claim to verify |
| `targetE` | `number` | ✅ | Target epistemic level |
| `minConfidence` | `number` | ✅ | Minimum confidence |
| `closesAt` | `number` | ✅ | Market close timestamp |
| `initialSubsidy` | `number` | ❌ | Liquidity subsidy |
| `tags` | `string[]` | ❌ | Market tags |

---

### registerDependency

Register a dependency between claims.

```typescript
async registerDependency(
  dependentClaimId: string,
  dependencyClaimId: string,
  dependencyType: DependencyType,
  weight: number,
  justification?: string
): Promise<HolochainRecord>
```

---

### cascadeUpdate

Trigger cascade update through belief graph.

```typescript
async cascadeUpdate(claimId: string): Promise<CascadeResult>
```

---

## GraphClient

### createRelationship

Create a relationship between claims.

```typescript
async createRelationship(
  source: string,
  target: string,
  relationshipType: RelationshipType,
  weight: number,
  properties?: Record<string, unknown>
): Promise<ActionHash>
```

---

### propagateBelief

Propagate belief changes through the graph.

```typescript
async propagateBelief(claimId: string): Promise<PropagationResult>
```

**Returns:**
```typescript
interface PropagationResult {
  sourceClaimId: string;
  nodesAffected: number;
  iterations: number;
  converged: boolean;
  maxDelta: number;
  processingTimeMs: number;
  updatedNodes: string[];
}
```

---

### getDependencyTree

Get dependency tree for a claim.

```typescript
async getDependencyTree(claimId: string, maxDepth?: number): Promise<DependencyTree>
```

---

### rankByInformationValue

Get claims ranked by information value.

```typescript
async rankByInformationValue(limit?: number): Promise<InformationValue[]>
```

---

### detectCircularDependencies

Detect circular dependencies in the graph.

```typescript
async detectCircularDependencies(claimId: string): Promise<string[][]>
```

---

### calculateCascadeImpact

Calculate the impact of changes to a claim.

```typescript
async calculateCascadeImpact(claimId: string): Promise<CascadeImpact>
```

---

## QueryClient

### search

Full-text search for claims.

```typescript
async search(queryText: string, options?: ClaimQueryOptions): Promise<Claim[]>
```

---

### queryByEpistemic

Query claims by epistemic position.

```typescript
async queryByEpistemic(
  minE?: number,
  maxE?: number,
  minN?: number,
  maxN?: number,
  minM?: number,
  maxM?: number,
  limit?: number
): Promise<Claim[]>
```

---

### findRelated

Find claims related to a given claim.

```typescript
async findRelated(claimId: string, maxDepth?: number, limit?: number): Promise<Claim[]>
```

---

### findContradictions

Find claims that contradict a given claim.

```typescript
async findContradictions(claimId: string): Promise<Claim[]>
```

---

## InferenceClient

### calculateEnhancedCredibility

Calculate MATL-enhanced credibility score.

```typescript
async calculateEnhancedCredibility(
  subject: string,
  subjectType: CredibilitySubjectType,
  includeMatl?: boolean
): Promise<EnhancedCredibilityScore>
```

---

### assessEvidenceStrength

Assess evidence strength for a claim.

```typescript
async assessEvidenceStrength(claimId: string): Promise<EvidenceStrengthComponents>
```

---

### getAuthorReputation

Get reputation profile for an author.

```typescript
async getAuthorReputation(authorDid: string): Promise<AuthorReputation>
```

---

### batchCredibilityAssessment

Batch assess credibility for multiple claims.

```typescript
async batchCredibilityAssessment(claimIds: string[]): Promise<BatchCredibilityResult>
```

---

## FactCheckClient

### factCheck

Fact-check a statement against the knowledge graph.

```typescript
async factCheck(input: FactCheckInput): Promise<FactCheckResult>
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `statement` | `string` | ✅ | Statement to check |
| `context` | `string` | ❌ | Additional context |
| `minE` | `number` | ❌ | Minimum epistemic level |
| `minN` | `number` | ❌ | Minimum normative level |
| `sourceHapp` | `string` | ❌ | Requesting hApp |

---

### batchFactCheck

Batch fact-check multiple statements.

```typescript
async batchFactCheck(inputs: FactCheckInput[]): Promise<FactCheckResult[]>
```

---

## Utility Functions

### toDiscreteEpistemic

Convert numeric position to discrete levels.

```typescript
function toDiscreteEpistemic(position: EpistemicPosition): {
  empirical: EmpiricalLevel;
  normative: NormativeLevel;
  mythic: MythicLevel;
}
```

---

### calculateInformationValue

Calculate information value for a claim.

```typescript
function calculateInformationValue(
  uncertainty: number,
  dependentCount: number,
  averageWeight: number
): number
```

---

### recommendVerification

Recommend whether to verify a claim.

```typescript
function recommendVerification(
  claim: Claim,
  informationValue: InformationValue
): {
  recommend: boolean;
  reason: string;
  suggestedTargetE: number;
}
```

---

### calculateCompositeCredibility

Calculate composite credibility score.

```typescript
function calculateCompositeCredibility(
  components: CredibilityComponents,
  matl?: MatlCredibilityComponents
): number
```

---

### determineVerdict

Determine fact-check verdict from claims.

```typescript
function determineVerdict(
  supportingClaims: SupportingClaim[],
  contradictingClaims: SupportingClaim[]
): { verdict: FactCheckVerdict; confidence: number }
```

---

## Types Reference

See [Entry Types](./ENTRY_TYPES.md) for complete type definitions.

---

## Error Codes

See [Error Codes](./ERROR_CODES.md) for error handling.
