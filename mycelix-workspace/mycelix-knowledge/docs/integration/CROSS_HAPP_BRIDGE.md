# Cross-hApp Bridge

How Mycelix Knowledge connects with other hApps in the Mycelix ecosystem.

## Overview

The Bridge zome enables cross-hApp communication, allowing any Mycelix hApp to submit claims, query knowledge, and request fact-checks.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MYCELIX ECOSYSTEM                               │
│                                                                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│   │ Identity │  │Governance│  │  Justice │  │  Media   │           │
│   │   hApp   │  │   hApp   │  │   hApp   │  │   hApp   │           │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│        │             │             │             │                  │
│        └─────────────┼─────────────┼─────────────┘                  │
│                      │             │                                │
│                      ▼             ▼                                │
│             ┌────────────────────────────┐                         │
│             │        Bridge Zome         │                         │
│             │    (Cross-hApp Gateway)    │                         │
│             └────────────┬───────────────┘                         │
│                          │                                          │
│                          ▼                                          │
│             ┌────────────────────────────┐                         │
│             │    KNOWLEDGE hApp          │                         │
│             │  claims / graph / query    │                         │
│             │  inference / factcheck     │                         │
│             └────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Bridge Functions

### submit_external_claim

Submit a claim from another hApp:

```rust
#[hdk_extern]
pub fn submit_external_claim(input: ExternalClaimInput) -> ExternResult<ActionHash> {
    // Verify calling hApp identity
    let caller_happ = verify_caller_happ()?;

    // Create claim with source attribution
    let claim = Claim {
        content: input.content,
        classification: input.classification,
        source_happ: Some(caller_happ),
        ..Default::default()
    };

    create_entry(&EntryTypes::Claim(claim))
}
```

### query_for_happ

Query claims relevant to a specific hApp:

```rust
#[hdk_extern]
pub fn query_for_happ(input: HappQueryInput) -> ExternResult<Vec<ClaimSummary>> {
    let claims = query_claims_by_domain(input.domain)?;

    // Filter by hApp-specific criteria
    let filtered = claims
        .into_iter()
        .filter(|c| c.classification.e >= input.min_e)
        .filter(|c| matches_topics(&c.topics, &input.topics))
        .collect();

    Ok(filtered)
}
```

## Integration Patterns

### Identity hApp Integration

Author reputation linked to decentralized identity:

```typescript
// In Identity hApp
async function getAuthorCredibility(did: string): Promise<CredibilityReport> {
  const knowledge = await connectToKnowledge();

  // Get author reputation from Knowledge
  const reputation = await knowledge.inference.getAuthorReputation(did);

  // Get claims authored
  const claims = await knowledge.claims.listClaimsByAuthor(did);

  return {
    did,
    matlTrust: reputation.matlTrust,
    claimsAuthored: reputation.claimsAuthored,
    verifiedTrue: reputation.claimsVerifiedTrue,
    verifiedFalse: reputation.claimsVerifiedFalse,
    domainExpertise: reputation.domainScores
  };
}
```

### Governance hApp Integration

Evidence-based governance proposals:

```typescript
// In Governance hApp
async function createProposalWithEvidence(
  proposal: ProposalContent,
  evidenceClaims: string[]
): Promise<ProposalHash> {
  const knowledge = await connectToKnowledge();

  // Verify all evidence claims exist and meet threshold
  for (const claimId of evidenceClaims) {
    const claim = await knowledge.claims.getClaim(claimId);
    if (!claim || claim.classification.empirical < 0.6) {
      throw new Error(`Claim ${claimId} does not meet evidence threshold`);
    }
  }

  // Create proposal with verified evidence links
  return createProposal({
    ...proposal,
    evidenceClaims,
    evidenceVerifiedAt: Date.now()
  });
}
```

### Justice hApp Integration

Case evidence verification:

```typescript
// In Justice hApp
async function submitEvidence(
  caseId: string,
  evidence: CaseEvidence
): Promise<EvidenceRecord> {
  const knowledge = await connectToKnowledge();

  // Create claim in Knowledge graph
  const claimHash = await knowledge.claims.createClaim({
    content: evidence.description,
    classification: {
      empirical: evidence.verifiability,
      normative: 0.7,  // Legal standard
      mythic: 0.6      // Case precedent
    },
    domain: "justice",
    topics: [caseId, evidence.type],
    evidence: evidence.supportingDocs.map(toKnowledgeEvidence)
  });

  // Get credibility assessment
  const credibility = await knowledge.inference.calculateEnhancedCredibility(
    claimHash.toString(),
    "Claim"
  );

  // Record in case
  return recordEvidence(caseId, {
    claimHash,
    credibility: credibility.overallScore,
    admissible: credibility.overallScore > 0.6
  });
}
```

### Media hApp Integration

Article fact-checking:

```typescript
// In Media hApp
async function factCheckArticle(article: Article): Promise<FactCheckReport> {
  const knowledge = await connectToKnowledge();

  // Extract factual claims from article
  const claims = extractFactualClaims(article.content);

  // Fact-check each claim
  const results = await Promise.all(
    claims.map(claim =>
      knowledge.factcheck.factCheck({
        statement: claim.text,
        context: `Article: ${article.title}`,
        sourceHapp: "mycelix-media"
      })
    )
  );

  // Aggregate results
  return {
    articleId: article.id,
    claims: results,
    overallRating: calculateOverallRating(results),
    timestamp: Date.now()
  };
}
```

### Finance hApp Integration

Market data claims:

```typescript
// In Finance hApp
async function recordMarketData(data: MarketDataPoint): Promise<void> {
  const knowledge = await connectToKnowledge();

  // Submit as high-empirical claim
  await knowledge.claims.createClaim({
    content: `${data.asset} price: $${data.price} at ${new Date(data.timestamp).toISOString()}`,
    classification: {
      empirical: 0.95,   // On-chain/exchange data
      normative: 0.05,
      mythic: 0.2
    },
    domain: "finance",
    topics: [data.asset, "price", "market-data"],
    evidence: [{
      id: `${data.exchange}-${data.timestamp}`,
      evidenceType: "Cryptographic",
      source: data.exchangeUrl,
      content: `Exchange: ${data.exchange}`,
      strength: 0.95
    }]
  });
}
```

## Authentication

### hApp Identity Verification

```rust
fn verify_caller_happ() -> ExternResult<String> {
    // Get caller's cell ID
    let caller = call_info()?.provenance;

    // Verify against registered hApps
    let registered = get_registered_happs()?;

    if !registered.contains(&caller) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Unregistered hApp".into()
        )));
    }

    // Return hApp identifier
    Ok(get_happ_name(&caller)?)
}
```

### Permission Levels

| hApp Type | Permissions |
|-----------|-------------|
| Core (Identity, Governance) | Full read/write |
| Standard (Media, Justice) | Read all, write own domain |
| External | Read public, limited write |

## Best Practices

### For Integrating hApps

1. **Register First** - Register your hApp with the bridge
2. **Use Correct Domain** - Submit claims to appropriate domains
3. **Handle Errors** - Knowledge queries may fail
4. **Cache Results** - Fact-check results are cacheable
5. **Attribute Sources** - Always include sourceHapp

### For Knowledge hApp

1. **Validate Callers** - Verify hApp identity
2. **Rate Limit** - Prevent abuse
3. **Audit Trail** - Log cross-hApp calls
4. **Domain Isolation** - Respect domain boundaries

## Related Documentation

- [Fact-Check API](./FACT_CHECK_API.md) - External verification
- [Epistemic Markets](./EPISTEMIC_MARKETS.md) - Market integration

---

*Knowledge flows across boundaries.*
