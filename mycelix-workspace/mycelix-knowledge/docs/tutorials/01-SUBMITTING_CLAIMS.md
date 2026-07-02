# Tutorial 1: Submitting Claims

This tutorial walks you through creating your first knowledge claim with proper epistemic classification.

## Prerequisites

- Mycelix Knowledge hApp installed and running
- TypeScript SDK installed: `npm install @mycelix/knowledge-sdk`
- Connected to a Holochain conductor

## Step 1: Connect to the hApp

```typescript
import { AppWebsocket } from '@holochain/client';
import { KnowledgeClient, EpistemicPosition } from '@mycelix/knowledge-sdk';

// Connect to your local conductor
const client = await AppWebsocket.connect('ws://localhost:8888');
const knowledge = new KnowledgeClient(client);

console.log("Connected to Knowledge hApp");
```

## Step 2: Understand Your Claim

Before submitting, analyze your claim:

### Questions to Ask

1. **What is the claim?**
   - Clear, specific statement
   - Not a question or opinion

2. **How can it be verified?** (Empirical)
   - Subjective experience? → E: 0.1
   - Witness testimony? → E: 0.3
   - Private verification (credentials)? → E: 0.5
   - Cryptographic proof? → E: 0.7
   - Scientific measurement? → E: 0.9

3. **Who must agree?** (Normative)
   - Just you? → N: 0.1
   - Your community? → N: 0.4
   - The whole network? → N: 0.6
   - Universal truth? → N: 0.9

4. **How long will it matter?** (Mythic)
   - Just today? → M: 0.1
   - This quarter/year? → M: 0.4
   - Permanently? → M: 0.7
   - Foundational? → M: 0.9

## Step 3: Create the Claim

### Example: Scientific Fact

```typescript
// A scientific measurement claim
const claimHash = await knowledge.claims.createClaim({
  content: "Global average temperature in 2023 was 1.45°C above pre-industrial levels",
  classification: {
    empirical: 0.85,   // Measurable by multiple agencies
    normative: 0.15,   // Methodology requires consensus
    mythic: 0.5        // Important for climate understanding
  },
  domain: "climate",
  topics: ["temperature", "global-warming", "2023", "measurement"],
  evidence: [{
    id: "wmo-2023-report",
    evidenceType: "Empirical",
    source: "https://wmo.int/2023-climate-report",
    content: "World Meteorological Organization 2023 State of Climate",
    strength: 0.9
  }],
  sources: [{
    uri: "https://wmo.int/2023-climate-report",
    title: "WMO State of the Global Climate 2023",
    author: "World Meteorological Organization",
    publishedAt: Date.now() - 30 * 24 * 60 * 60 * 1000, // 30 days ago
    reliability: 0.95
  }]
});

console.log("Created claim:", claimHash);
```

### Example: Historical Event

```typescript
// A historical event claim
const historicalClaim = await knowledge.claims.createClaim({
  content: "The Apollo 11 mission landed on the Moon on July 20, 1969",
  classification: {
    empirical: 0.9,    // Documented, witnessed, verified
    normative: 0.3,    // Some interpretation involved
    mythic: 0.85       // Culturally foundational
  },
  domain: "history",
  topics: ["apollo-11", "moon-landing", "1969", "nasa", "space-exploration"],
  evidence: [{
    id: "nasa-apollo-11",
    evidenceType: "Cryptographic",  // NASA official records
    source: "https://nasa.gov/apollo-11",
    content: "Official NASA mission archives",
    strength: 0.95
  }]
});
```

### Example: Community Standard

```typescript
// A community-agreed standard
const standardClaim = await knowledge.claims.createClaim({
  content: "The Mycelix network uses Apache 2.0 license for core components",
  classification: {
    empirical: 0.95,   // Documented in repositories
    normative: 0.7,    // Community decision
    mythic: 0.6        // Governance standard
  },
  domain: "governance",
  topics: ["mycelix", "licensing", "apache-2.0", "open-source"],
  evidence: [{
    id: "mycelix-license",
    evidenceType: "Cryptographic",
    source: "https://github.com/Luminous-Dynamics/mycelix-knowledge/blob/main/LICENSE",
    content: "Repository LICENSE file",
    strength: 0.99
  }]
});
```

## Step 4: Add Evidence

You can add evidence after creating the claim:

```typescript
// Add additional evidence
await knowledge.claims.addEvidence(claimHash.toString(), {
  id: "nasa-apollo-photos",
  evidenceType: "Empirical",
  source: "https://nasa.gov/apollo-11-photos",
  content: "Original mission photography archives",
  strength: 0.85
});

await knowledge.claims.addEvidence(claimHash.toString(), {
  id: "moon-rock-samples",
  evidenceType: "Empirical",
  source: "https://curation.jsc.nasa.gov",
  content: "Moon rock samples in JSC curation facility",
  strength: 0.95
});
```

## Step 5: Verify Your Claim

After creating, retrieve and verify:

```typescript
const claim = await knowledge.claims.getClaim(claimHash);

console.log("Claim content:", claim.content);
console.log("Classification:", claim.classification);
console.log("Evidence count:", claim.evidence.length);
console.log("Status:", claim.status);

// Check credibility
const credibility = await knowledge.inference.calculateEnhancedCredibility(
  claimHash.toString(),
  "Claim"
);

console.log("Credibility score:", credibility.overallScore);
```

## Classification Examples

| Claim Type | Example | E | N | M |
|------------|---------|---|---|---|
| Scientific fact | "Water boils at 100°C at sea level" | 0.95 | 0.1 | 0.4 |
| Survey result | "60% of respondents prefer X" | 0.7 | 0.3 | 0.3 |
| Legal ruling | "Court ruled in favor of..." | 0.8 | 0.8 | 0.75 |
| Personal experience | "I saw a bird outside" | 0.15 | 0.05 | 0.1 |
| Religious teaching | "The universe was created by..." | 0.1 | 0.5 | 0.95 |
| Economic prediction | "GDP will grow 3% next year" | 0.4 | 0.4 | 0.5 |

## Common Mistakes

### ❌ Inflating Empirical Level

```typescript
// WRONG: Opinion presented as fact
const badClaim = {
  content: "Electric cars are better than gas cars",
  classification: { empirical: 0.8, ... }  // Too high for a value judgment
};

// CORRECT: Specify the measurable aspect
const goodClaim = {
  content: "Electric cars produce 50% less lifetime CO2 emissions than equivalent gas cars",
  classification: { empirical: 0.75, ... }  // Measurable with caveats
};
```

### ❌ Missing Evidence

```typescript
// WRONG: High E claim without evidence
const badClaim = {
  content: "The new drug reduces symptoms by 40%",
  classification: { empirical: 0.85 },
  evidence: []  // No evidence!
};

// CORRECT: Provide supporting evidence
const goodClaim = {
  content: "The new drug reduces symptoms by 40%",
  classification: { empirical: 0.85 },
  evidence: [{
    id: "clinical-trial-2023",
    evidenceType: "Empirical",
    source: "https://clinicaltrials.gov/NCT...",
    content: "Phase 3 clinical trial results",
    strength: 0.88
  }]
};
```

### ❌ Vague Claims

```typescript
// WRONG: Too vague
const badClaim = {
  content: "Things are getting better"  // What things? Better how?
};

// CORRECT: Specific and measurable
const goodClaim = {
  content: "Global extreme poverty rate fell from 36% in 1990 to 9.2% in 2019"
};
```

## Next Steps

1. **[Tutorial 2: Querying Knowledge](./02-QUERYING_KNOWLEDGE.md)** - Find existing claims
2. **[Tutorial 3: Building Relationships](./03-BUILDING_RELATIONSHIPS.md)** - Connect claims
3. **[Epistemic Classification](../concepts/EPISTEMIC_CLASSIFICATION.md)** - Deep dive into E-N-M

---

*Your first contribution to the knowledge commons.*
