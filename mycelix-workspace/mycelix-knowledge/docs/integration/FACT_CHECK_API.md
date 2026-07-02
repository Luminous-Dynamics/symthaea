# Fact-Check API

The Fact-Check API provides external hApps with structured knowledge verification, enabling Media, Governance, Justice, and other Mycelix pillars to verify statements against the knowledge graph.

## Overview

```
┌───────────────────────────────────────────────────────────────┐
│                     FACT-CHECK API                             │
│                                                                │
│   External hApp                    Knowledge hApp              │
│   ┌──────────┐                    ┌─────────────────┐         │
│   │  Media   │──fact_check()────▶│    factcheck/   │         │
│   │  hApp    │                    │      zome       │         │
│   │          │◀──FactCheckResult──│                 │         │
│   └──────────┘                    │  ┌───────────┐  │         │
│                                   │  │  claims/  │  │         │
│   ┌──────────┐                    │  │  graph/   │  │         │
│   │  Gov     │──query_claims()───▶│  │  query/   │  │         │
│   │  hApp    │                    │  │inference/ │  │         │
│   │          │◀──ClaimSummary[]───│  └───────────┘  │         │
│   └──────────┘                    └─────────────────┘         │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

## Fact-Check Verdicts

| Verdict | Description | Confidence Range |
|---------|-------------|------------------|
| `True` | Strongly supported by evidence | >90% support |
| `MostlyTrue` | Supported with minor caveats | 70-90% support |
| `Mixed` | Conflicting evidence | 45-55% support |
| `MostlyFalse` | More contradicting than supporting | 10-30% support |
| `False` | Strongly contradicted | <10% support |
| `Unverifiable` | Cannot be verified empirically | E < 0.3 |
| `InsufficientEvidence` | Not enough claims to assess | <3 relevant claims |

## Using the API

### Basic Fact Check

```typescript
import { KnowledgeClient } from '@mycelix/knowledge-sdk';

const knowledge = new KnowledgeClient(client);

// Fact-check a statement
const result = await knowledge.factcheck.factCheck({
  statement: "Global sea levels rose 3.4mm per year from 1993-2022",
  context: "Climate change article fact-checking",
  minE: 0.7,  // Minimum epistemic level for supporting claims
  sourceHapp: "mycelix-media"  // Requesting hApp
});

console.log(`Verdict: ${result.verdict}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
console.log(`Reasoning: ${result.reasoning}`);
console.log(`Epistemic assessment: E=${result.epistemicAssessment.empirical}`);
```

### Detailed Result

```typescript
interface FactCheckResult {
  id: string;
  statement: string;
  verdict: FactCheckVerdict;
  confidence: number;
  supportingClaims: SupportingClaim[];
  contradictingClaims: SupportingClaim[];
  reasoning: string;
  epistemicAssessment: EpistemicPosition;
  checkedAt: number;
  expiresAt?: number;
  sourceHapp?: string;
}

// Supporting claim details
interface SupportingClaim {
  claimId: string;
  relevance: number;        // How relevant to the statement (0-1)
  relationship: "Supports" | "Contradicts" | "Contextual";
  epistemicPosition: EpistemicPosition;
  credibilityScore: number;
}
```

### Example Response

```typescript
{
  id: "fc-abc123",
  statement: "Global sea levels rose 3.4mm per year from 1993-2022",
  verdict: "True",
  confidence: 0.92,
  supportingClaims: [
    {
      claimId: "uhCkk-noaa-sea-level",
      relevance: 0.95,
      relationship: "Supports",
      epistemicPosition: { empirical: 0.9, normative: 0.1, mythic: 0.4 },
      credibilityScore: 0.88
    },
    {
      claimId: "uhCkk-ipcc-ar6",
      relevance: 0.85,
      relationship: "Supports",
      epistemicPosition: { empirical: 0.85, normative: 0.2, mythic: 0.5 },
      credibilityScore: 0.92
    }
  ],
  contradictingClaims: [],
  reasoning: "Statement is strongly supported by 5 claims from credible sources including NOAA, IPCC AR6, and peer-reviewed studies. No significant contradicting evidence found.",
  epistemicAssessment: { empirical: 0.88, normative: 0.12, mythic: 0.42 },
  checkedAt: 1704067200000,
  expiresAt: 1706745600000,  // 30 days
  sourceHapp: "mycelix-media"
}
```

## Query Claims

### By Subject

```typescript
// Find claims about a specific subject
const claims = await knowledge.factcheck.queryClaimsBySubject(
  "sea level rise",
  {
    minE: 0.6,       // Minimum empirical level
    domain: "climate",
    limit: 20
  }
);

for (const claim of claims) {
  console.log(`${claim.content}`);
  console.log(`  E: ${claim.classification.empirical}`);
  console.log(`  Status: ${claim.status}`);
}
```

### By Topic

```typescript
// Find claims matching multiple topics
const claims = await knowledge.factcheck.queryClaimsByTopic(
  ["renewable-energy", "cost", "2023"],
  {
    minE: 0.5,
    limit: 50
  }
);
```

### Batch Fact-Check

```typescript
// Fact-check multiple statements efficiently
const inputs = [
  { statement: "Statement 1...", minE: 0.6 },
  { statement: "Statement 2...", minE: 0.7 },
  { statement: "Statement 3...", minE: 0.5 }
];

const results = await knowledge.factcheck.batchFactCheck(inputs);

for (const result of results) {
  console.log(`"${result.statement.substring(0, 50)}..."`);
  console.log(`  Verdict: ${result.verdict}`);
}
```

## Integration Patterns

### Media hApp Integration

```typescript
// In a media/journalism hApp
async function factCheckArticle(article: Article): Promise<FactCheckReport> {
  const knowledge = new KnowledgeClient(client, "knowledge");
  const factualClaims = extractFactualClaims(article);  // NLP extraction

  const results: FactCheckResult[] = [];

  for (const claim of factualClaims) {
    const result = await knowledge.factcheck.factCheck({
      statement: claim.text,
      context: `Article: ${article.title}`,
      minE: 0.6,
      sourceHapp: "mycelix-media"
    });
    results.push(result);
  }

  return {
    articleId: article.id,
    overallAssessment: calculateOverallAssessment(results),
    claims: results,
    checkedAt: Date.now()
  };
}
```

### Governance hApp Integration

```typescript
// In a governance hApp
async function validateProposalEvidence(proposal: Proposal): Promise<ValidationReport> {
  const knowledge = new KnowledgeClient(client, "knowledge");

  // Verify each claim referenced in the proposal
  const validations = await Promise.all(
    proposal.evidenceClaims.map(async (claimRef) => {
      // Get the claim directly
      const claims = await knowledge.query.search(claimRef, { limit: 1 });

      if (claims.length === 0) {
        return { claimRef, status: "NotFound" };
      }

      const claim = claims[0];

      // Check credibility
      const credibility = await knowledge.inference.calculateEnhancedCredibility(
        claim.id,
        "Claim"
      );

      return {
        claimRef,
        status: claim.classification.empirical > 0.7 ? "Verified" : "LowConfidence",
        credibility: credibility.overallScore
      };
    })
  );

  return {
    proposalId: proposal.id,
    validations,
    overallCredibility: average(validations.map(v => v.credibility || 0))
  };
}
```

### Justice hApp Integration

```typescript
// In a justice hApp
async function verifyEvidenceClaim(evidenceId: string): Promise<VerificationResult> {
  const knowledge = new KnowledgeClient(client, "knowledge");

  // Get the evidence claim
  const claim = await knowledge.claims.getClaim(evidenceId);

  if (!claim) {
    return { status: "NotFound", message: "Evidence claim not found" };
  }

  // Check epistemic level
  if (claim.classification.empirical < 0.6) {
    return {
      status: "InsufficientVerification",
      message: `Epistemic level ${claim.classification.empirical} below threshold`,
      recommendation: "Request verification market"
    };
  }

  // Get full credibility assessment
  const credibility = await knowledge.inference.calculateEnhancedCredibility(
    claim.id,
    "Claim"
  );

  return {
    status: credibility.overallScore > 0.7 ? "Verified" : "Questionable",
    credibility: credibility.overallScore,
    matlScore: credibility.matl.matlComposite,
    evidenceStrength: credibility.evidenceStrength
  };
}
```

## Caching and Performance

### Result Caching

```typescript
// Fact-check results include expiration
const result = await knowledge.factcheck.factCheck({
  statement: "...",
  minE: 0.6
});

// Check if result is still valid
if (result.expiresAt && Date.now() < result.expiresAt) {
  // Use cached result
} else {
  // Re-check
}

// Get cached history for a statement
const history = await knowledge.factcheck.getFactCheckHistory(statement);
```

### Batch Processing

```typescript
// For checking many statements, use batch API
// This is more efficient than individual calls

const statements = [
  "Statement 1",
  "Statement 2",
  // ... up to 100 statements
];

const results = await knowledge.factcheck.batchFactCheck(
  statements.map(s => ({ statement: s, minE: 0.6 }))
);
```

## Error Handling

```typescript
try {
  const result = await knowledge.factcheck.factCheck({
    statement,
    minE: 0.8
  });
} catch (error) {
  if (error.code === "NO_RELEVANT_CLAIMS") {
    // No claims found in the knowledge graph
    console.log("Cannot fact-check: no relevant knowledge");
  } else if (error.code === "STATEMENT_TOO_VAGUE") {
    // Statement is too vague to fact-check
    console.log("Please provide a more specific statement");
  } else if (error.code === "RATE_LIMITED") {
    // Too many requests
    console.log("Please wait before making more requests");
  }
}
```

## Best Practices

### Do
- ✅ Provide context for better matching
- ✅ Use appropriate minE for your use case
- ✅ Handle all verdict types
- ✅ Check expiration before using cached results
- ✅ Use batch API for multiple statements

### Don't
- ❌ Treat verdicts as absolute truth
- ❌ Ignore confidence levels
- ❌ Use very high minE (>0.9) - may exclude valid claims
- ❌ Fact-check subjective statements
- ❌ Ignore contradicting claims in the result

## Related Documentation

- [Epistemic Classification](../concepts/EPISTEMIC_CLASSIFICATION.md) - Understanding E-N-M
- [Credibility Engine](../concepts/CREDIBILITY_ENGINE.md) - Trust assessment
- [Cross-hApp Bridge](./CROSS_HAPP_BRIDGE.md) - General integration patterns

---

*"Verify before you amplify."*
