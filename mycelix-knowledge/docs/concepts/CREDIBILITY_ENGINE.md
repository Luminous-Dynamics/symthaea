# Credibility Engine

The Credibility Engine provides MATL-enhanced trust assessment for claims, sources, and authors in Mycelix Knowledge.

## Overview

Credibility in Mycelix Knowledge is not a simple score - it's a multi-dimensional assessment that considers:

- **Basic Components**: Accuracy, consistency, transparency, track record, corroboration
- **MATL Integration**: Quality, consistency, reputation, stake-weighted dimensions
- **Evidence Strength**: Empirical, testimonial, cryptographic evidence analysis
- **Author Reputation**: Domain-specific historical performance

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     CREDIBILITY ENGINE                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Subject (Claim, Source, or Author)                     │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│
│  │   Basic     │  │   MATL      │  │    Evidence Strength    ││
│  │ Components  │  │ Components  │  │      Components         ││
│  ├─────────────┤  ├─────────────┤  ├─────────────────────────┤│
│  │ Accuracy    │  │ Quality     │  │ Empirical count/avg     ││
│  │ Consistency │  │ Consistency │  │ Testimonial count/avg   ││
│  │ Transparency│  │ Reputation  │  │ Cryptographic count     ││
│  │ Track Record│  │ Stake-weight│  │ Cross-references        ││
│  │ Corroborate │  │ Oracle agree│  │ Market resolutions      ││
│  └─────────────┘  └─────────────┘  └─────────────────────────┘│
│         │                │                      │              │
│         ▼                ▼                      ▼              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              COMPOSITE CREDIBILITY SCORE                 │  │
│  │                                                          │  │
│  │    overall = f(basic, matl, evidence, author)           │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  OUTPUT: EnhancedCredibilityScore with all dimensions          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Credibility Components

### Basic Components

```typescript
interface CredibilityComponents {
  accuracy: number;      // 0-1: Historical accuracy of claims
  consistency: number;   // 0-1: Consistency with related claims
  transparency: number;  // 0-1: Availability of sources/evidence
  trackRecord: number;   // 0-1: Historical performance
  corroboration: number; // 0-1: Independent verification
}
```

**Calculation:**

| Component | Factors |
|-----------|---------|
| Accuracy | Verified true / Total verified |
| Consistency | Alignment with supporting claims - contradictions |
| Transparency | Evidence count × source quality |
| Track Record | Author's historical accuracy weighted by time |
| Corroboration | Independent verifiers / Required threshold |

### MATL Components

```typescript
interface MatlCredibilityComponents {
  matlComposite: number;     // Overall MATL score (0-1)
  matlQuality: number;       // Quality dimension
  matlConsistency: number;   // Consistency dimension
  matlReputation: number;    // Reputation dimension
  matlStakeWeighted: number; // Stake-weighted score
  oracleAgreement?: number;  // Agreement among oracles
  oracleCount: number;       // Number of oracle assessments
}
```

**MATL Quadratic Weighting:**

```typescript
// MATL uses quadratic weighting for Byzantine resilience
// Higher MATL scores contribute proportionally more
const matlWeight = matl.matlComposite ** 2;

// This achieves 45% Byzantine fault tolerance:
// - Attackers with < 45% of stake cannot manipulate outcomes
// - Honest majority always prevails
```

### Evidence Strength

```typescript
interface EvidenceStrengthComponents {
  empiricalEvidenceCount: number;
  testimonialEvidenceCount: number;
  cryptoEvidenceCount: number;
  crossReferenceCount: number;
  marketResolutionCount: number;
  empiricalAverageStrength: number;
  testimonialAverageStrength: number;
  totalEvidenceCount: number;
  evidenceDiversity: number;  // Variety of evidence types (0-1)
}
```

**Evidence Weighting:**

| Type | Weight | Description |
|------|--------|-------------|
| Cryptographic | 1.0 | On-chain, verifiable proofs |
| Empirical | 0.85 | Scientific measurements |
| Market Resolution | 0.75 | Resolved verification markets |
| Cross-Reference | 0.6 | References to other claims |
| Testimonial | 0.4 | Witness accounts |

### Author Reputation

```typescript
interface AuthorReputation {
  authorDid: string;
  overallScore: number;
  domainScores: DomainReputation[];
  historicalAccuracy: number;
  claimsAuthored: number;
  claimsVerifiedTrue: number;
  claimsVerifiedFalse: number;
  averageEpistemicE: number;
  matlTrust: number;
  reputationAgeDays: number;
}

interface DomainReputation {
  domain: string;      // e.g., "climate", "finance"
  score: number;       // 0-1
  claimCount: number;
  accuracyRate: number;
}
```

## Using the Credibility Engine

### Calculate Credibility for a Claim

```typescript
import { KnowledgeClient } from '@mycelix/knowledge-sdk';

const knowledge = new KnowledgeClient(client);

// Get enhanced credibility score
const credibility = await knowledge.inference.calculateEnhancedCredibility(
  claimId,      // Subject ID
  "Claim",      // Subject type: "Claim", "Source", or "Author"
  true          // Include MATL integration
);

console.log(`Overall score: ${credibility.overallScore}`);
console.log(`Assessment confidence: ${credibility.assessmentConfidence}`);

// Basic components
console.log("Basic components:");
console.log(`  Accuracy: ${credibility.components.accuracy}`);
console.log(`  Consistency: ${credibility.components.consistency}`);
console.log(`  Transparency: ${credibility.components.transparency}`);

// MATL components
console.log("MATL components:");
console.log(`  Composite: ${credibility.matl.matlComposite}`);
console.log(`  Quality: ${credibility.matl.matlQuality}`);
console.log(`  Reputation: ${credibility.matl.matlReputation}`);

// Evidence strength
console.log("Evidence strength:");
console.log(`  Empirical count: ${credibility.evidenceStrength.empiricalEvidenceCount}`);
console.log(`  Diversity: ${credibility.evidenceStrength.evidenceDiversity}`);
```

### Assess Evidence Strength

```typescript
// Get detailed evidence strength analysis
const evidence = await knowledge.inference.assessEvidenceStrength(claimId);

console.log(`Total evidence pieces: ${evidence.totalEvidenceCount}`);
console.log(`Evidence types:`);
console.log(`  Empirical: ${evidence.empiricalEvidenceCount} (avg strength: ${evidence.empiricalAverageStrength})`);
console.log(`  Testimonial: ${evidence.testimonialEvidenceCount} (avg strength: ${evidence.testimonialAverageStrength})`);
console.log(`  Cryptographic: ${evidence.cryptoEvidenceCount}`);
console.log(`Diversity score: ${evidence.evidenceDiversity}`);
```

### Get Author Reputation

```typescript
// Get author's reputation profile
const reputation = await knowledge.inference.getAuthorReputation("did:key:z6Mk...");

console.log(`Overall score: ${reputation.overallScore}`);
console.log(`Historical accuracy: ${reputation.historicalAccuracy}`);
console.log(`Claims authored: ${reputation.claimsAuthored}`);
console.log(`  Verified true: ${reputation.claimsVerifiedTrue}`);
console.log(`  Verified false: ${reputation.claimsVerifiedFalse}`);
console.log(`Average epistemic E: ${reputation.averageEpistemicE}`);
console.log(`MATL trust: ${reputation.matlTrust}`);

// Domain-specific scores
console.log("Domain expertise:");
for (const domain of reputation.domainScores) {
  console.log(`  ${domain.domain}: ${domain.score} (${domain.claimCount} claims, ${domain.accuracyRate} accuracy)`);
}
```

### Batch Assessment

```typescript
// Efficiently assess multiple claims
const claimIds = ["claim-1", "claim-2", "claim-3", "claim-4", "claim-5"];
const results = await knowledge.inference.batchCredibilityAssessment(claimIds);

console.log(`Assessed ${results.totalAssessed} claims`);
console.log(`Average score: ${results.averageScore}`);
console.log(`High credibility (>0.7): ${results.highCredibilityCount}`);
console.log(`Low credibility (<0.3): ${results.lowCredibilityCount}`);
console.log(`Processing time: ${results.processingTimeMs}ms`);

for (const summary of results.results) {
  console.log(`${summary.subject}: ${summary.overallScore} (MATL: ${summary.matlComposite})`);
}
```

## Composite Score Calculation

The overall credibility score combines all dimensions:

```typescript
function calculateCompositeCredibility(
  components: CredibilityComponents,
  matl: MatlCredibilityComponents,
  evidence: EvidenceStrengthComponents,
  author?: AuthorReputation
): number {
  // Base score from basic components (equal weights)
  const baseScore = (
    components.accuracy +
    components.consistency +
    components.transparency +
    components.trackRecord +
    components.corroboration
  ) / 5;

  // MATL-weighted score uses quadratic weighting
  const matlWeight = matl.matlComposite ** 2;

  // Evidence bonus (more diverse evidence = higher trust)
  const evidenceBonus = Math.min(0.1, evidence.evidenceDiversity * 0.15);

  // Author reputation factor
  const authorFactor = author
    ? 0.1 * author.overallScore
    : 0;

  // Blend all factors
  const composite =
    baseScore * (0.5 - matlWeight * 0.15) +
    matl.matlComposite * matlWeight * 0.3 +
    evidenceBonus +
    authorFactor;

  return Math.max(0, Math.min(1, composite));
}
```

## Credibility Factors

The engine tracks factors that contributed to the score:

```typescript
interface CredibilityFactor {
  name: string;
  value: number;       // Positive or negative
  explanation: string;
}

// Example factors for a claim
const factors = credibility.factors;
// [
//   { name: "High evidence diversity", value: 0.08, explanation: "4 types of evidence" },
//   { name: "Author track record", value: 0.05, explanation: "Author has 95% accuracy" },
//   { name: "Recent contradiction", value: -0.1, explanation: "Contradicts verified claim" },
//   { name: "Market verification", value: 0.12, explanation: "Verified in prediction market" }
// ]
```

## Score Expiration

Credibility scores have an expiration time:

```typescript
const credibility = await knowledge.inference.calculateEnhancedCredibility(
  claimId,
  "Claim"
);

if (credibility.expiresAt && Date.now() > credibility.expiresAt) {
  // Score has expired, recalculate
  const fresh = await knowledge.inference.calculateEnhancedCredibility(
    claimId,
    "Claim"
  );
}

// Expiration depends on:
// - Claim domain volatility
// - Evidence freshness
// - Market activity
```

## Integration with Belief Graphs

Credibility affects belief propagation:

```typescript
// High-credibility claims have stronger influence
// in belief propagation

// When propagating belief through the graph:
// influenceWeight = dependencyWeight * credibility.overallScore

// This means:
// - High-credibility claims propagate beliefs more strongly
// - Low-credibility claims have dampened influence
// - Contradictions from high-credibility claims are more impactful
```

## Best Practices

### Do
- ✅ Calculate credibility before major decisions
- ✅ Consider all dimensions, not just overall score
- ✅ Check author reputation for new claims
- ✅ Monitor credibility changes over time
- ✅ Provide evidence to improve credibility

### Don't
- ❌ Trust overall score blindly without examining factors
- ❌ Ignore low credibility warnings
- ❌ Assume credibility is permanent
- ❌ Skip evidence strength analysis
- ❌ Ignore domain-specific reputation

## Related Documentation

- [MATL Integration](../integration/MATL_INTEGRATION.md) - Trust layer mechanics
- [Belief Graphs](./BELIEF_GRAPHS.md) - How credibility affects propagation
- [Epistemic Classification](./EPISTEMIC_CLASSIFICATION.md) - E-N-M positioning
- [Information Value](./INFORMATION_VALUE.md) - Verification prioritization

---

*"Trust, but verify. Then trust more."*
