# MATL Integration

How the Multi-dimensional Adaptive Trust Layer (MATL) integrates with Mycelix Knowledge.

## Overview

MATL is the trust backbone of the Mycelix ecosystem. In Knowledge, MATL provides:
- Oracle weighting for verification markets
- Author reputation tracking
- Credibility score enhancement
- Byzantine fault tolerance

## MATL Components

### Quality Dimension

Measures the quality of an agent's contributions:

```typescript
interface QualityScore {
  accuracy: number;           // Historical accuracy of claims
  evidenceStrength: number;   // Quality of evidence provided
  sourceReliability: number;  // Trustworthiness of sources cited
  verificationRate: number;   // Claims that got verified
}

// Calculated as weighted average
matlQuality = 0.4 * accuracy + 0.3 * evidenceStrength +
              0.2 * sourceReliability + 0.1 * verificationRate;
```

### Consistency Dimension

Measures behavioral consistency:

```typescript
interface ConsistencyScore {
  claimConsistency: number;     // Claims don't contradict self
  voteConsistency: number;      // Voting pattern coherence
  activityPattern: number;      // Regular, non-suspicious activity
  domainFocus: number;          // Expertise in specific domains
}

matlConsistency = average([claimConsistency, voteConsistency,
                           activityPattern, domainFocus]);
```

### Reputation Dimension

Accumulated trust over time:

```typescript
interface ReputationScore {
  age: number;                  // Days of reputation history
  totalClaims: number;          // Claims authored
  verifiedCorrect: number;      // Claims verified true
  verifiedIncorrect: number;    // Claims verified false
  oracleAccuracy: number;       // Accuracy as oracle
  disputes: number;             // Disputes lost
}

// Reputation builds slowly, degrades on failures
matlReputation = calculateReputationScore(history);
```

### Composite MATL

The overall trust score:

```typescript
matlComposite = w1 * matlQuality +
                w2 * matlConsistency +
                w3 * matlReputation;

// Default weights
const weights = {
  quality: 0.35,
  consistency: 0.25,
  reputation: 0.40
};
```

## Quadratic Weighting

MATL uses quadratic weighting for Byzantine resilience:

```typescript
// Linear weighting: vulnerable to 51% attack
linearInfluence = matlScore;

// Quadratic weighting: requires 67%+ to attack
quadraticInfluence = matlScore ** 2;

// In practice, achieves 45% Byzantine tolerance
// because low-MATL attackers have minimal influence
```

### Example

```
Agent A: MATL = 0.9, Linear = 0.9, Quadratic = 0.81
Agent B: MATL = 0.8, Linear = 0.8, Quadratic = 0.64
Agent C: MATL = 0.3, Linear = 0.3, Quadratic = 0.09 (attacker)

Even with 33% of agents attacking:
- Linear: Attacker has 0.3 / (0.9+0.8+0.3) = 15% influence
- Quadratic: Attacker has 0.09 / (0.81+0.64+0.09) = 5.8% influence
```

## Integration Points

### 1. Oracle Voting

```typescript
// Oracle votes are MATL-weighted
function calculateResolution(votes: OracleVote[]): ResolutionOutcome {
  const weighted = votes.map(v => ({
    outcome: v.outcome,
    weight: v.matlWeight ** 2 * v.confidence * v.reputationStake
  }));

  // Aggregate by outcome
  const totals = weighted.reduce((acc, v) => {
    acc[v.outcome] = (acc[v.outcome] || 0) + v.weight;
    return acc;
  }, {});

  // Winner takes all
  return Object.entries(totals)
    .sort(([,a], [,b]) => b - a)[0][0];
}
```

### 2. Credibility Scores

```typescript
// MATL enhances credibility calculations
async function calculateCredibility(claimId: string): Promise<EnhancedCredibilityScore> {
  const basicScore = await calculateBasicCredibility(claimId);
  const author = await getAuthor(claimId);
  const authorMatl = await getMatlScore(author);

  return {
    ...basicScore,
    matl: {
      matlComposite: authorMatl.composite,
      matlQuality: authorMatl.quality,
      matlConsistency: authorMatl.consistency,
      matlReputation: authorMatl.reputation,
      matlStakeWeighted: authorMatl.composite ** 2
    }
  };
}
```

### 3. Claim Weighting

```typescript
// Higher-MATL authors' claims have more weight in belief propagation
function calculateInfluenceWeight(claim: Claim): number {
  const authorMatl = getMatlScore(claim.author);
  const baseWeight = claim.evidence.length * 0.1 + 0.5;

  // MATL multiplier
  return baseWeight * (1 + authorMatl.composite * 0.5);
}
```

### 4. Market Participation

```typescript
// MATL determines market participation eligibility
const marketRequirements = {
  participant: {
    minMatl: 0.1,
    maxStake: 0.05  // 5% of total
  },
  oracle: {
    minMatl: 0.3,
    minAge: 30,
    minAccuracy: 0.6
  },
  marketCreator: {
    minMatl: 0.5,
    minClaims: 10
  }
};
```

## Domain-Specific MATL

Authors can have different MATL scores in different domains:

```typescript
interface DomainMATL {
  domain: string;
  score: number;
  claimCount: number;
  accuracy: number;
}

// Example: High trust in climate, low in finance
const authorDomainScores: DomainMATL[] = [
  { domain: "climate", score: 0.85, claimCount: 47, accuracy: 0.91 },
  { domain: "finance", score: 0.35, claimCount: 3, accuracy: 0.67 },
  { domain: "technology", score: 0.62, claimCount: 18, accuracy: 0.78 }
];

// Domain MATL affects claims in that domain
function getEffectiveMatl(author: string, domain: string): number {
  const domainScore = getDomainMatl(author, domain);
  const globalScore = getGlobalMatl(author);

  // Blend domain and global (domain has more weight)
  return domainScore * 0.7 + globalScore * 0.3;
}
```

## MATL Updates

### Positive Events

| Event | MATL Impact |
|-------|-------------|
| Claim verified true | +0.01 to +0.05 |
| Correct oracle vote | +0.01 to +0.03 |
| Evidence verified | +0.005 to +0.02 |
| High-credibility claim | +0.01 to +0.02 |
| Market profit (honest) | +0.005 to +0.01 |

### Negative Events

| Event | MATL Impact |
|-------|-------------|
| Claim verified false | -0.02 to -0.10 |
| Incorrect oracle vote | -0.02 to -0.05 |
| Lost dispute | -0.03 to -0.10 |
| Detected coordination | -0.20 to -0.50 |
| Spam/fraud | -0.50 to -1.00 |

## Using MATL in SDK

```typescript
import { KnowledgeClient } from '@mycelix/knowledge-sdk';

const knowledge = new KnowledgeClient(client);

// Get MATL-enhanced credibility
const credibility = await knowledge.inference.calculateEnhancedCredibility(
  claimId,
  "Claim",
  true  // Include MATL
);

console.log(`MATL Composite: ${credibility.matl.matlComposite}`);
console.log(`MATL Quality: ${credibility.matl.matlQuality}`);
console.log(`MATL Consistency: ${credibility.matl.matlConsistency}`);
console.log(`MATL Reputation: ${credibility.matl.matlReputation}`);
console.log(`MATL Stake Weight: ${credibility.matl.matlStakeWeighted}`);

// Get author reputation with MATL
const reputation = await knowledge.inference.getAuthorReputation("did:key:...");
console.log(`Overall MATL: ${reputation.matlTrust}`);
console.log(`Domain scores:`);
for (const domain of reputation.domainScores) {
  console.log(`  ${domain.domain}: ${domain.score}`);
}
```

## Best Practices

### Building MATL

1. **Start Small** - Submit well-sourced claims in your expertise area
2. **Be Consistent** - Regular, quality contributions
3. **Participate Honestly** - Accurate oracle votes
4. **Focus on Domains** - Build expertise in specific areas
5. **Provide Evidence** - Strong evidence builds trust faster

### Maintaining MATL

1. **Stay Active** - Inactive accounts slowly decay
2. **Correct Mistakes** - Update claims when wrong
3. **Avoid Disputes** - Losing disputes hurts MATL
4. **No Coordination** - Detected collusion is severely penalized

## Related Documentation

- [Credibility Engine](../concepts/CREDIBILITY_ENGINE.md) - How MATL affects credibility
- [Epistemic Markets](./EPISTEMIC_MARKETS.md) - MATL in market resolution
- [Security](../operations/SECURITY.md) - Byzantine resistance

---

*Trust is earned, not claimed.*
