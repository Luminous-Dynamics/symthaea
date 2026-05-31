# Information Value

Information Value quantifies how valuable it would be to verify a given claim, helping prioritize verification efforts and market creation.

## Overview

Not all claims are equally worth verifying. Information Value helps answer:
- Which claims should we verify first?
- How much subsidy should a verification market receive?
- What's the expected impact of resolving uncertainty?

## The Formula

```
InformationValue = Uncertainty × Impact × Urgency

Where:
- Uncertainty = 1 - EpistemicLevel (E)
- Impact = DependentCount × AverageWeight
- Urgency = TimeDecay factor
```

## Components

### Uncertainty

Claims with moderate epistemic levels have the most potential for improvement:

```typescript
function calculateUncertainty(claim: Claim): number {
  const e = claim.classification.empirical;

  // High E claims have low uncertainty
  // Low E claims might not be verifiable
  // Sweet spot is 0.3-0.7
  if (e > 0.8) return 0.1;  // Already verified
  if (e < 0.2) return 0.2;  // Hard to verify
  return 1 - e;  // Moderate = highest value
}
```

### Impact

Impact measures how many other claims depend on this one:

```typescript
interface Impact {
  dependentCount: number;      // Claims that depend on this
  averageWeight: number;       // Average dependency weight
  maxCascadeDepth: number;     // How far changes propagate
  highImpactClaims: string[];  // Most affected claims
}

function calculateImpact(tree: DependencyTree): number {
  return tree.totalDependencies * tree.aggregateWeight / tree.depth;
}
```

### Urgency

Some verifications are time-sensitive:

```typescript
function calculateUrgency(claim: Claim, context: Context): number {
  let urgency = 1.0;

  // Active markets increase urgency
  if (context.activeMarkets > 0) {
    urgency *= 1.5;
  }

  // Pending governance decisions increase urgency
  if (context.pendingGovernanceRefs > 0) {
    urgency *= 2.0;
  }

  // Recent contradictions increase urgency
  if (context.recentContradictions > 0) {
    urgency *= 1.3;
  }

  return Math.min(urgency, 3.0);  // Cap at 3x
}
```

## Using Information Value

### Rank Claims for Verification

```typescript
import { KnowledgeClient } from '@mycelix/knowledge-sdk';

const knowledge = new KnowledgeClient(client);

// Get top 10 claims by information value
const rankings = await knowledge.graph.rankByInformationValue(10);

for (const iv of rankings) {
  console.log(`Claim: ${iv.claimId}`);
  console.log(`  Expected Value: ${iv.expectedValue.toFixed(3)}`);
  console.log(`  Dependents: ${iv.dependentCount}`);
  console.log(`  Uncertainty: ${iv.uncertainty.toFixed(2)}`);
  console.log(`  Impact Score: ${iv.impactScore.toFixed(2)}`);
  console.log(`  Recommended: ${iv.recommendedForVerification ? 'YES' : 'NO'}`);
  console.log(`  Reasoning: ${iv.reasoning}`);
  console.log('---');
}
```

### Make Verification Decisions

```typescript
import { recommendVerification } from '@mycelix/knowledge-sdk';

// For a specific claim
const claim = await knowledge.claims.getClaim(claimHash);
const iv = await getInformationValue(claimHash);  // Helper

const recommendation = recommendVerification(claim, iv);

if (recommendation.recommend) {
  console.log(`Recommend verification:`);
  console.log(`  Reason: ${recommendation.reason}`);
  console.log(`  Target E: ${recommendation.suggestedTargetE}`);

  // Create verification market
  await knowledge.claims.spawnVerificationMarket({
    claimId: claimHash.toString(),
    targetE: recommendation.suggestedTargetE,
    minConfidence: 0.7,
    closesAt: Date.now() + 14 * 24 * 60 * 60 * 1000
  });
} else {
  console.log(`Verification not recommended: ${recommendation.reason}`);
}
```

### Calculate Market Subsidy

```typescript
// Use information value to determine market subsidy
const assessment = await knowledge.marketsIntegration.calculateMarketValue(claimId);

// Higher information value = higher recommended subsidy
// This ensures high-value verifications get adequate liquidity

console.log(`Expected value: ${assessment.expectedValue}`);
console.log(`Recommended subsidy: ${assessment.recommendedSubsidy}`);
// Example: IV of 0.8 might recommend $500 subsidy
// Example: IV of 0.2 might recommend $50 subsidy

// Create market with calculated subsidy
await knowledge.claims.spawnVerificationMarket({
  claimId,
  targetE: 0.8,
  minConfidence: 0.7,
  closesAt: Date.now() + 7 * 24 * 60 * 60 * 1000,
  initialSubsidy: assessment.recommendedSubsidy
});
```

## Decision Matrix

| Uncertainty | Impact | Recommendation |
|-------------|--------|----------------|
| High (>0.5) | High (>10 deps) | **Strong** - Create market immediately |
| High (>0.5) | Low (<5 deps) | Moderate - Consider market |
| Low (<0.3) | High (>10 deps) | Low - Already sufficiently verified |
| Low (<0.3) | Low (<5 deps) | **Skip** - Not worth verifying |

## Information Value Over Time

Information Value changes as the knowledge graph evolves:

```typescript
// Track IV changes over time
interface InformationValueHistory {
  claimId: string;
  history: Array<{
    timestamp: number;
    value: number;
    reason: string;
  }>;
}

// IV increases when:
// - New claims depend on this one
// - Contradictions are found
// - Markets reference it as evidence

// IV decreases when:
// - Claim is verified (E increases)
// - Dependents are verified independently
// - Contradictions are resolved
```

## Integration with Services

### KnowledgeService Auto-Verification

```typescript
import { KnowledgeService } from '@mycelix/knowledge-sdk';

const service = new KnowledgeService(client);

// Submit claim and get automatic IV assessment
const result = await service.submitAndAnalyzeClaim({
  content: "Global renewable capacity reached 3,372 GW in 2022",
  classification: { empirical: 0.6, normative: 0.1, mythic: 0.3 },
  domain: "energy"
});

console.log(`Claim created: ${result.claimHash}`);
console.log(`Credibility: ${result.credibility.overallScore}`);
console.log(`Information Value: ${result.informationValue.expectedValue}`);

if (result.verificationRecommendation.recommend) {
  console.log(`Should verify: ${result.verificationRecommendation.reason}`);
}
```

### Automatic Verification Trigger

```typescript
// Automatically verify high-value claims
const verification = await service.autoVerifyIfRecommended(claimHash);

if (verification.requested) {
  console.log(`Market created: ${verification.marketHash}`);
  console.log(`Reason: ${verification.reason}`);
} else {
  console.log(`Not verified: ${verification.reason}`);
}
```

## Best Practices

### Do
- ✅ Check IV before creating markets
- ✅ Consider all factors (uncertainty, impact, urgency)
- ✅ Re-evaluate IV as graph changes
- ✅ Allocate subsidy proportional to IV
- ✅ Track IV trends over time

### Don't
- ❌ Create markets for low-IV claims
- ❌ Ignore high-IV unverified claims
- ❌ Treat IV as static
- ❌ Overcomplicate subsidy calculations
- ❌ Verify claims in isolation (consider graph)

## Examples

### High Information Value Claim

```typescript
const claim = {
  content: "Fusion reactor achieved net energy gain",
  classification: { empirical: 0.4, normative: 0.2, mythic: 0.7 }
};

// High IV because:
// - Moderate E (room to improve)
// - Many energy policy claims depend on it
// - High urgency (affects investment decisions)
// - Foundational (high M)

// IV calculation:
// Uncertainty: 1 - 0.4 = 0.6
// Impact: 50 dependents × 0.7 avg weight = 35
// Urgency: 1.5 (active markets)
// IV = 0.6 × 35 × 1.5 = 31.5 (very high)
```

### Low Information Value Claim

```typescript
const claim = {
  content: "The meeting room has 10 chairs",
  classification: { empirical: 0.8, normative: 0.05, mythic: 0.05 }
};

// Low IV because:
// - High E (already verified)
// - No dependents
// - No urgency
// - Transient (low M)

// IV calculation:
// Uncertainty: 1 - 0.8 = 0.2
// Impact: 0 dependents × 0 = 0
// Urgency: 1.0 (baseline)
// IV = 0.2 × 0 × 1.0 = 0 (don't verify)
```

## Related Documentation

- [Belief Graphs](./BELIEF_GRAPHS.md) - Dependency tracking
- [Epistemic Markets](../integration/EPISTEMIC_MARKETS.md) - Verification through markets
- [Credibility Engine](./CREDIBILITY_ENGINE.md) - Trust assessment

---

*"Verify what matters most."*
