# Epistemic Markets Integration

Mycelix Knowledge integrates bidirectionally with Epistemic Markets, enabling claims to be verified through prediction markets and markets to reference claims as evidence.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  BIDIRECTIONAL INTEGRATION                       │
│                                                                  │
│   KNOWLEDGE                              EPISTEMIC MARKETS       │
│   ┌──────────┐                          ┌──────────────────┐    │
│   │  Claim   │──spawn_verification────▶│ Verification     │    │
│   │  E=0.4   │                          │ Market           │    │
│   │          │◀──on_market_resolved────│ "Verify claim X" │    │
│   │  E=0.8   │                          │ Resolved: YES    │    │
│   └──────────┘                          └──────────────────┘    │
│                                                                  │
│   ┌──────────┐                          ┌──────────────────┐    │
│   │  Claim   │◀──register_as_evidence──│ Prediction       │    │
│   │  "GDP+3%"│                          │ Market           │    │
│   │          │                          │ "Will Q4 GDP...?"│    │
│   └──────────┘                          └──────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Integration Flows

### 1. Claim → Verification Market

When a claim needs stronger verification, spawn a market:

```typescript
import { KnowledgeClient } from '@mycelix/knowledge-sdk';

const knowledge = new KnowledgeClient(client);

// A claim with moderate empirical level
const claimId = "uhCkk..."; // Existing claim hash

// Request verification through a market
const marketHash = await knowledge.claims.spawnVerificationMarket({
  claimId: claimId,
  targetE: 0.8,           // Target epistemic level
  minConfidence: 0.7,     // Minimum oracle confidence
  closesAt: Date.now() + 7 * 24 * 60 * 60 * 1000, // 1 week
  initialSubsidy: 100,    // Optional liquidity subsidy
  tags: ["science", "climate"]
});

console.log("Verification market created:", marketHash);
```

### 2. Market Resolution → Claim Update

When the market resolves, the claim is automatically updated:

```typescript
// This is called automatically by the bridge zome
// when Epistemic Markets signals resolution

// Internal flow:
// 1. Epistemic Markets resolves "Will claim X achieve E4?"
// 2. Bridge zome receives on_market_resolved()
// 3. Claim's epistemic position is updated
// 4. Belief propagation cascades through dependents

// You can also manually trigger (for testing):
await knowledge.claims.cascadeUpdate(claimId);
```

### 3. Claim → Evidence in Prediction

Claims can be referenced as evidence in prediction markets:

```typescript
// Register a claim as evidence in a market prediction
await knowledge.marketsIntegration.registerClaimAsEvidence(
  claimId,       // The claim being referenced
  marketId,      // The prediction market
  "supporting"   // How the claim is used: "supporting", "opposing", "contextual"
);

// Get all claims referenced in a market
const claims = await knowledge.marketsIntegration.getClaimsUsedInMarket(marketId);
```

## Market Link Types

| Type | Direction | Description |
|------|-----------|-------------|
| `VerificationMarket` | Knowledge → Markets | Market created to verify a claim |
| `ClaimAsEvidence` | Knowledge → Markets | Claim referenced in a prediction |
| `PredictionAboutClaim` | Markets → Knowledge | Market predicting claim outcome |
| `ResolutionSource` | Markets → Knowledge | Market resolved based on claim |

## Verification Market Workflow

### Step 1: Identify Candidate Claims

```typescript
import { recommendVerification } from '@mycelix/knowledge-sdk';

// Get claims that would benefit from verification
const candidates = await knowledge.graph.rankByInformationValue(20);

for (const iv of candidates) {
  const claim = await knowledge.claims.getClaim(iv.claimId);
  const recommendation = recommendVerification(claim, iv);

  if (recommendation.recommend) {
    console.log(`Claim ${iv.claimId}:`);
    console.log(`  Reason: ${recommendation.reason}`);
    console.log(`  Suggested target E: ${recommendation.suggestedTargetE}`);
  }
}
```

### Step 2: Spawn Verification Market

```typescript
// Create market for the highest-value verification
const marketHash = await knowledge.claims.spawnVerificationMarket({
  claimId: topCandidate.claimId,
  targetE: recommendation.suggestedTargetE,
  minConfidence: 0.7,
  closesAt: Date.now() + 14 * 24 * 60 * 60 * 1000, // 2 weeks
  tags: claim.topics
});
```

### Step 3: Monitor Market Progress

```typescript
// Get market status
const markets = await knowledge.claims.getClaimMarkets(claimId);

for (const market of markets) {
  console.log(`Market ${market.marketId}:`);
  console.log(`  Status: ${market.status}`);
  console.log(`  Current price: ${market.currentPrice}`);
  console.log(`  Participants: ${market.participantCount}`);
  console.log(`  Total stake: ${market.totalStake}`);
}
```

### Step 4: Handle Resolution

When the market resolves, the claim is updated automatically:

```typescript
// The resolution triggers this flow:
// 1. Oracle votes are tallied (MATL-weighted)
// 2. Outcome is determined with confidence
// 3. on_market_resolved() is called
// 4. Claim's E level is updated
// 5. Cascade propagates to dependents

// Check the updated claim
const updatedClaim = await knowledge.claims.getClaim(claimId);
console.log(`New empirical level: ${updatedClaim.classification.empirical}`);
```

## Resolution Mechanisms

### Standard Oracle Resolution

For claims requiring human judgment:

```
Market Question: "Will claim 'X increases Y by 15%' achieve E4 verification?"

Oracle Voting:
- Oracles submit votes with confidence and evidence
- Votes are MATL-weighted (matl²)
- Byzantine tolerance: 45%
- Requires quorum and confidence threshold

Resolution:
- YES (confidence > minConfidence) → E level increases to targetE
- NO → E level stays same or decreases
- INCONCLUSIVE → Market extends or cancels
```

### Automated Resolution

For claims with on-chain evidence:

```typescript
// Example: A claim about an on-chain event
const claim = {
  content: "ETH price exceeded $5000 on Dec 1, 2024",
  classification: { empirical: 0.7, normative: 0.1, mythic: 0.3 }
};

// Verification market with automated resolution
const market = {
  question: "Did ETH exceed $5000 on Dec 1, 2024?",
  resolutionType: "Automated",
  oracleSource: "chainlink-eth-usd",
  resolutionCondition: "price > 5000 at timestamp"
};

// Resolution is automatic based on oracle feed
```

## MATL-Weighted Voting

Oracle votes are weighted by their MATL (Multi-dimensional Adaptive Trust Layer) scores:

```typescript
interface OracleVote {
  oracle: AgentPubKey;
  outcome: "YES" | "NO";
  confidence: number;
  evidence: EvidenceLink[];
  matlWeight: number;     // Oracle's MATL score
  reputationStake: number; // Reputation staked
}

// MATL weighting uses quadratic formula
// This provides 45% Byzantine fault tolerance
function calculateFinalOutcome(votes: OracleVote[]): {
  outcome: string;
  confidence: number;
} {
  let yesWeight = 0;
  let noWeight = 0;

  for (const vote of votes) {
    const weight = vote.matlWeight ** 2 * vote.confidence;
    if (vote.outcome === "YES") {
      yesWeight += weight;
    } else {
      noWeight += weight;
    }
  }

  const total = yesWeight + noWeight;
  return {
    outcome: yesWeight > noWeight ? "YES" : "NO",
    confidence: Math.abs(yesWeight - noWeight) / total
  };
}
```

## Market Value Assessment

Calculate the expected value of creating a verification market:

```typescript
const assessment = await knowledge.marketsIntegration.calculateMarketValue(claimId);

console.log(`Expected value: ${assessment.expectedValue}`);
console.log(`Recommended subsidy: ${assessment.recommendedSubsidy}`);
console.log(`Reasoning: ${assessment.reasoning}`);

// Only create market if value exceeds cost
if (assessment.expectedValue > assessment.recommendedSubsidy) {
  await knowledge.claims.spawnVerificationMarket({
    claimId,
    targetE: 0.8,
    initialSubsidy: assessment.recommendedSubsidy,
    // ...
  });
}
```

## Cross-hApp Usage

Other Mycelix hApps can use knowledge claims:

### Media hApp

```typescript
// Fact-check an article claim
const result = await knowledge.factcheck.factCheck({
  statement: "Unemployment fell to 3.5% in November",
  context: "Economic news article",
  minE: 0.7,
  sourceHapp: "mycelix-media"
});

// Create market if fact-check is inconclusive
if (result.verdict === "InsufficientEvidence") {
  // Spawn verification market
}
```

### Governance hApp

```typescript
// Verify a claim used in a governance proposal
const claims = await knowledge.query.search("carbon credits verified");

// Reference high-E claims in the proposal
const proposal = {
  content: "Implement carbon credit system",
  evidenceClaims: claims
    .filter(c => c.classification.empirical > 0.7)
    .map(c => c.id)
};
```

## Error Handling

```typescript
try {
  await knowledge.claims.spawnVerificationMarket({
    claimId,
    targetE: 0.9,
    minConfidence: 0.8,
    closesAt: Date.now() + 7 * 24 * 60 * 60 * 1000
  });
} catch (error) {
  if (error.code === "MARKET_ALREADY_EXISTS") {
    // A verification market already exists for this claim
    const markets = await knowledge.claims.getClaimMarkets(claimId);
    console.log("Existing market:", markets[0]);
  } else if (error.code === "INSUFFICIENT_STAKE") {
    // Not enough stake to create market
    console.log("Need more stake to create market");
  } else if (error.code === "CLAIM_TOO_RECENT") {
    // Claim must exist for minimum time before verification
    console.log("Claim is too new for verification");
  }
}
```

## Best Practices

### Do
- ✅ Check information value before creating markets
- ✅ Set realistic target E levels (incremental improvements)
- ✅ Provide initial subsidy for liquidity
- ✅ Use appropriate timeframes for market duration
- ✅ Monitor market progress and handle disputes

### Don't
- ❌ Create markets for low-value claims
- ❌ Set unrealistic target E levels
- ❌ Ignore market outcomes
- ❌ Create duplicate verification markets
- ❌ Skip cascade updates after resolution

## Related Documentation

- [Belief Graphs](../concepts/BELIEF_GRAPHS.md) - How resolutions propagate
- [MATL Integration](./MATL_INTEGRATION.md) - Trust layer mechanics
- [Credibility Engine](../concepts/CREDIBILITY_ENGINE.md) - How MATL affects credibility
- [Fact-Check API](./FACT_CHECK_API.md) - External verification service

---

*"Markets aggregate knowledge. Knowledge informs markets."*
