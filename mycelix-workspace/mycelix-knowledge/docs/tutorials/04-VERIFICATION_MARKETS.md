# Tutorial 4: Verification Markets

Learn how to verify claims through prediction markets.

## Overview

Verification markets leverage collective intelligence to verify claims. When you create a verification market for a claim, oracles stake reputation to vote on whether the claim achieves a target epistemic level.

## When to Use Verification Markets

Use markets when:
- Claim has moderate E (0.3-0.7) that could be improved
- Many other claims depend on this one
- Traditional verification is difficult/expensive
- You have subsidy budget for liquidity

## Creating a Verification Market

### Step 1: Assess Information Value

```typescript
import { KnowledgeClient, recommendVerification } from '@mycelix/knowledge-sdk';

const knowledge = new KnowledgeClient(client);

// Get claim and its information value
const claim = await knowledge.claims.getClaim(claimHash);
const ivRecords = await knowledge.graph.rankByInformationValue(100);
const iv = ivRecords.find(r => r.claimId === claimHash.toString());

// Get recommendation
const recommendation = recommendVerification(claim, iv);

console.log(`Recommend: ${recommendation.recommend}`);
console.log(`Reason: ${recommendation.reason}`);
console.log(`Target E: ${recommendation.suggestedTargetE}`);
```

### Step 2: Calculate Market Value

```typescript
const assessment = await knowledge.marketsIntegration.calculateMarketValue(
  claimHash.toString()
);

console.log(`Expected value: ${assessment.expectedValue}`);
console.log(`Recommended subsidy: ${assessment.recommendedSubsidy}`);
console.log(`Reasoning: ${assessment.reasoning}`);
```

### Step 3: Create the Market

```typescript
if (recommendation.recommend) {
  const marketHash = await knowledge.claims.spawnVerificationMarket({
    claimId: claimHash.toString(),
    targetE: recommendation.suggestedTargetE,
    minConfidence: 0.7,   // Required oracle confidence
    closesAt: Date.now() + 14 * 24 * 60 * 60 * 1000, // 2 weeks
    initialSubsidy: assessment.recommendedSubsidy,
    tags: claim.topics
  });

  console.log("Created verification market:", marketHash);
}
```

## Monitoring Your Market

```typescript
// Get markets for your claim
const markets = await knowledge.claims.getClaimMarkets(claimHash.toString());

for (const market of markets) {
  console.log(`Market: ${market.marketId}`);
  console.log(`  Status: ${market.status}`);
  console.log(`  Current price: ${market.currentPrice}`);
  console.log(`  Participants: ${market.participantCount}`);
  console.log(`  Total stake: ${market.totalStake}`);
}
```

## Understanding Resolution

When the market closes:

1. **Oracle Voting Period** - Oracles submit votes with evidence
2. **MATL Weighting** - Votes weighted by MATL²
3. **Outcome Determination** - YES/NO/INCONCLUSIVE
4. **Claim Update** - E level updated based on outcome
5. **Cascade** - Dependent claims notified

```typescript
// After resolution, check updated claim
const updatedClaim = await knowledge.claims.getClaim(claimHash);
console.log(`New E level: ${updatedClaim.classification.empirical}`);

// Cascade update propagates to dependents
const cascade = await knowledge.claims.cascadeUpdate(claimHash.toString());
console.log(`Claims updated: ${cascade.claimsUpdated}`);
console.log(`Markets notified: ${cascade.marketNotifications.length}`);
```

## Handling Outcomes

### Verified (YES)

```typescript
if (market.status === "Resolved" && market.resolution.outcome === "Verified") {
  // Claim E increased to targetE
  console.log(`Claim verified! New E: ${claim.classification.empirical}`);

  // Belief propagates to dependents
  await knowledge.graph.propagateBelief(claimHash.toString());
}
```

### Refuted (NO)

```typescript
if (market.status === "Resolved" && market.resolution.outcome === "Refuted") {
  // Claim E stays same or decreases
  console.log("Claim not verified. Consider updating or retracting.");

  // Option: Retract or update the claim
  await knowledge.claims.updateClaim({
    claimHash,
    status: "Disputed"
  });
}
```

### Inconclusive

```typescript
if (market.status === "Resolved" && market.resolution.outcome === "Inconclusive") {
  // Not enough confidence either way
  console.log("Verification inconclusive. Consider:");
  console.log("  - Adding more evidence");
  console.log("  - Creating new market with longer duration");
  console.log("  - Increasing subsidy for more participation");
}
```

## Best Practices

1. **Check IV First** - Don't create markets for low-value claims
2. **Set Realistic Targets** - Don't jump from E=0.3 to E=0.9
3. **Provide Subsidy** - Markets need liquidity to function
4. **Allow Sufficient Time** - Complex claims need more time
5. **Monitor Progress** - Address issues early

## Cost Considerations

| Claim Type | Suggested Duration | Suggested Subsidy |
|------------|-------------------|-------------------|
| Simple fact | 1 week | $50-100 |
| Complex claim | 2 weeks | $100-500 |
| High-impact | 4 weeks | $500+ |

## Next Steps

- [Tutorial 5: Building Integrations](./05-BUILDING_INTEGRATIONS.md)
- [Epistemic Markets Integration](../integration/EPISTEMIC_MARKETS.md)

---

*Let the market reveal the truth.*
