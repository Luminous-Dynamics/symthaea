# Tutorial 5: Building Integrations

Learn how to integrate Mycelix Knowledge with your own hApp.

## Overview

Any Mycelix hApp can integrate with Knowledge to:
- Submit domain-specific claims
- Query the knowledge graph
- Request fact-checks
- Use credibility scores

## Setup

### 1. Add SDK Dependency

```json
{
  "dependencies": {
    "@mycelix/knowledge-sdk": "^0.1.0"
  }
}
```

### 2. Connect to Knowledge

```typescript
import { AppWebsocket } from '@holochain/client';
import { KnowledgeClient, KnowledgeService } from '@mycelix/knowledge-sdk';

// Option 1: Direct client
const appClient = await AppWebsocket.connect('ws://localhost:8888');
const knowledge = new KnowledgeClient(appClient, 'knowledge');

// Option 2: High-level service
const knowledgeService = new KnowledgeService(appClient);
```

## Common Integration Patterns

### Pattern 1: Fact-Checking User Content

```typescript
async function factCheckPost(post: UserPost): Promise<FactCheckBadge> {
  // Extract factual claims (using NLP or rules)
  const claims = extractFactualClaims(post.content);

  if (claims.length === 0) {
    return { status: "no-claims", badge: null };
  }

  // Fact-check each claim
  const results = await Promise.all(
    claims.map(claim =>
      knowledge.factcheck.factCheck({
        statement: claim.text,
        context: `Post by ${post.authorId}`,
        minE: 0.6,
        sourceHapp: "my-social-happ"
      })
    )
  );

  // Generate badge
  const trueCount = results.filter(r => r.verdict === "True" || r.verdict === "MostlyTrue").length;
  const falseCount = results.filter(r => r.verdict === "False" || r.verdict === "MostlyFalse").length;

  return {
    status: "checked",
    badge: {
      total: claims.length,
      verified: trueCount,
      disputed: falseCount,
      confidence: average(results.map(r => r.confidence))
    }
  };
}
```

### Pattern 2: Evidence-Based Features

```typescript
async function createProposalWithEvidence(
  proposal: Proposal,
  evidenceClaimIds: string[]
): Promise<ProposalHash> {
  // Verify all evidence claims meet threshold
  const credibilities = await knowledge.inference.batchCredibilityAssessment(evidenceClaimIds);

  const lowCredibility = credibilities.results.filter(c => c.overallScore < 0.6);
  if (lowCredibility.length > 0) {
    throw new Error(`${lowCredibility.length} claims have insufficient credibility`);
  }

  // Create proposal with verified evidence
  return createProposal({
    ...proposal,
    evidenceClaims: evidenceClaimIds,
    evidenceCredibility: credibilities.averageScore,
    verifiedAt: Date.now()
  });
}
```

### Pattern 3: Domain-Specific Claim Submission

```typescript
// Energy hApp submitting project data as claims
async function recordProjectMetrics(project: EnergyProject): Promise<void> {
  // Submit production claim
  await knowledge.claims.createClaim({
    content: `Project ${project.id} produced ${project.kwh} kWh in ${project.period}`,
    classification: {
      empirical: 0.9,    // Measured data
      normative: 0.1,
      mythic: 0.3
    },
    domain: "energy",
    topics: ["production", project.type, project.region],
    evidence: [{
      id: `smart-meter-${project.id}`,
      evidenceType: "Cryptographic",
      source: `metering://project/${project.id}`,
      content: "Smart meter data",
      strength: 0.95
    }]
  });
}
```

### Pattern 4: Credibility-Weighted Features

```typescript
async function rankContributors(contributors: Contributor[]): Promise<RankedList> {
  // Get author reputation for each
  const withReputation = await Promise.all(
    contributors.map(async (c) => {
      const rep = await knowledge.inference.getAuthorReputation(c.did);
      return {
        ...c,
        credibilityScore: rep.overallScore,
        domainExpertise: rep.domainScores
      };
    })
  );

  // Rank by domain-specific credibility
  return withReputation.sort((a, b) => {
    const aDomain = a.domainExpertise.find(d => d.domain === "my-domain")?.score || 0;
    const bDomain = b.domainExpertise.find(d => d.domain === "my-domain")?.score || 0;
    return bDomain - aDomain;
  });
}
```

## Error Handling

```typescript
async function safeFactCheck(statement: string): Promise<FactCheckResult | null> {
  try {
    return await knowledge.factcheck.factCheck({
      statement,
      minE: 0.6
    });
  } catch (error) {
    if (error.code === "NO_RELEVANT_CLAIMS") {
      // No knowledge available
      return null;
    }
    if (error.code === "RATE_LIMITED") {
      // Wait and retry
      await sleep(1000);
      return safeFactCheck(statement);
    }
    throw error;
  }
}
```

## Caching Strategies

```typescript
// Cache credibility scores (they change slowly)
const credibilityCache = new Map<string, { score: EnhancedCredibilityScore; expires: number }>();

async function getCachedCredibility(claimId: string): Promise<EnhancedCredibilityScore> {
  const cached = credibilityCache.get(claimId);
  if (cached && cached.expires > Date.now()) {
    return cached.score;
  }

  const score = await knowledge.inference.calculateEnhancedCredibility(claimId, "Claim");
  credibilityCache.set(claimId, {
    score,
    expires: Date.now() + 5 * 60 * 1000  // 5 minutes
  });

  return score;
}
```

## Testing Your Integration

```typescript
// Mock client for testing
const mockClient = {
  factcheck: {
    factCheck: jest.fn().mockResolvedValue({
      verdict: "True",
      confidence: 0.85
    })
  }
};

test("fact-check integration", async () => {
  const result = await factCheckPost(
    { content: "The earth is round", authorId: "user123" },
    mockClient
  );

  expect(result.status).toBe("checked");
  expect(mockClient.factcheck.factCheck).toHaveBeenCalled();
});
```

## Best Practices

1. **Handle Failures Gracefully** - Knowledge may be unavailable
2. **Cache Appropriately** - Credibility scores, fact-checks
3. **Use Batch APIs** - For multiple items
4. **Set Reasonable Thresholds** - Don't require E=1.0
5. **Attribute Your hApp** - Include sourceHapp in requests

## Related Documentation

- [Cross-hApp Bridge](../integration/CROSS_HAPP_BRIDGE.md)
- [Fact-Check API](../integration/FACT_CHECK_API.md)
- [API Reference](../reference/API_REFERENCE.md)

---

*Build on the knowledge commons.*
