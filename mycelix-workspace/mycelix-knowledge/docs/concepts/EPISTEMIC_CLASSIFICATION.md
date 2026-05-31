# Epistemic Classification

The heart of Mycelix Knowledge is the **Epistemic Classification System** - a three-dimensional framework for categorizing all knowledge claims.

## The E-N-M Cube

Every claim occupies a position in a 3D cube defined by three axes:

```
         M (Mythic/Materiality)
          │
          │    High Mythic
          │    (Foundational narratives)
          │   ╱│
          │  ╱ │
          │ ╱  │
          │╱───┼─────────────────────────── E (Empirical)
          │    │                    High Empirical
          │    │                    (Scientific facts)
          │    │
          │───┘
         ╱
        ╱
       N (Normative)
       High Normative
       (Universal values)
```

## Axis Definitions

### E - Empirical (Verifiability)

The Empirical axis measures how a claim can be verified through observation and evidence.

| Level | Name | Score | Description | Example |
|-------|------|-------|-------------|---------|
| E0 | Subjective | 0.0-0.2 | Personal experience, feelings | "I feel cold" |
| E1 | Testimonial | 0.2-0.4 | Witness accounts, reports | "I saw a UFO" |
| E2 | Private Verify | 0.4-0.6 | Credentials, ZK proofs | "I have a PhD" |
| E3 | Cryptographic | 0.6-0.8 | On-chain, digital proofs | "This transaction occurred" |
| E4 | Measurable | 0.8-1.0 | Scientific reproducibility | "Water boils at 100°C" |

**Verification Methods by Level:**

```typescript
// E0: No external verification possible
{ empirical: 0.1, verificationMethod: "self-report" }

// E1: Requires corroborating witnesses
{ empirical: 0.3, verificationMethod: "multiple-testimony" }

// E2: Credential verification
{ empirical: 0.5, verificationMethod: "zk-credential-proof" }

// E3: Blockchain verification
{ empirical: 0.7, verificationMethod: "on-chain-event" }

// E4: Scientific measurement
{ empirical: 0.95, verificationMethod: "reproducible-experiment" }
```

### N - Normative (Value Alignment)

The Normative axis measures the scope of agreement required for the claim to be considered valid.

| Level | Name | Score | Description | Example |
|-------|------|-------|-------------|---------|
| N0 | Personal | 0.0-0.25 | Individual perspective | "Chocolate is the best" |
| N1 | Communal | 0.25-0.5 | Group/community consensus | "Our community values X" |
| N2 | Network | 0.5-0.75 | Network-wide agreement | "Mycelix protocol rules" |
| N3 | Universal | 0.75-1.0 | Objective/universal truth | "Math truths, logic" |

**Consensus Requirements:**

```typescript
// N0: No consensus needed - personal truth
{ normative: 0.1, consensusRequired: "none" }

// N1: Community must agree
{ normative: 0.4, consensusRequired: "community-majority" }

// N2: Network validation needed
{ normative: 0.6, consensusRequired: "network-consensus" }

// N3: Must be universally true
{ normative: 0.9, consensusRequired: "logical-necessity" }
```

### M - Mythic/Materiality (Narrative Significance)

The Mythic axis measures the lasting significance and narrative weight of a claim.

| Level | Name | Score | Description | Example |
|-------|------|-------|-------------|---------|
| M0 | Transient | 0.0-0.25 | Temporary, won't persist | "The meeting is at 3pm" |
| M1 | Temporal | 0.25-0.5 | Time-limited relevance | "Q3 earnings report" |
| M2 | Persistent | 0.5-0.75 | Should be permanently recorded | "Historical treaty" |
| M3 | Foundational | 0.75-1.0 | Core to understanding | "Origin story, constitution" |

**Persistence Requirements:**

```typescript
// M0: Can be forgotten
{ mythic: 0.1, persistence: "ephemeral" }

// M1: Archive for reference
{ mythic: 0.4, persistence: "archived" }

// M2: Permanent record
{ mythic: 0.65, persistence: "permanent" }

// M3: Foundational document
{ mythic: 0.9, persistence: "immutable-core" }
```

## Classifying Claims

### Decision Framework

When classifying a claim, ask:

1. **Empirical**: "How can this be verified?"
   - Can anyone measure/observe it? → High E
   - Requires trust in testimony? → Medium E
   - Only I can know? → Low E

2. **Normative**: "Who must agree for this to be true?"
   - Just me? → Low N
   - My community? → Medium N
   - Everyone, universally? → High N

3. **Mythic**: "How long will this matter?"
   - Just today? → Low M
   - For this era? → Medium M
   - Forever? → High M

### Code Example

```typescript
import { KnowledgeClient, EpistemicPosition } from '@mycelix/knowledge-sdk';

// Helper function for classification
function classifyClaim(description: string): EpistemicPosition {
  // Scientific fact: high empirical, low normative, medium mythic
  if (description.includes("measured") || description.includes("observed")) {
    return { empirical: 0.85, normative: 0.15, mythic: 0.4 };
  }

  // Value statement: low empirical, high normative, variable mythic
  if (description.includes("should") || description.includes("ought")) {
    return { empirical: 0.2, normative: 0.8, mythic: 0.5 };
  }

  // Historical event: high empirical, medium normative, high mythic
  if (description.includes("happened") || description.includes("occurred")) {
    return { empirical: 0.7, normative: 0.3, mythic: 0.8 };
  }

  // Default: moderate on all axes
  return { empirical: 0.5, normative: 0.5, mythic: 0.5 };
}

// Creating a well-classified claim
const claimHash = await knowledge.claims.createClaim({
  content: "The 2024 global temperature anomaly was +1.29°C",
  classification: {
    empirical: 0.9,   // Measured by multiple agencies
    normative: 0.2,   // Some interpretation in methodology
    mythic: 0.6       // Significant for climate understanding
  },
  domain: "climate",
  evidence: [{
    id: "noaa-2024-temp",
    evidenceType: "Empirical",
    source: "https://noaa.gov/climate/2024",
    content: "NOAA Global Climate Report 2024",
    strength: 0.92
  }]
});
```

## Classification Archetypes

### Scientific Fact
```
E: 0.85-0.95 (measurable, reproducible)
N: 0.1-0.3 (methodology consensus needed)
M: 0.3-0.6 (contributes to knowledge corpus)
```

### Moral Principle
```
E: 0.1-0.3 (not empirically verifiable)
N: 0.7-0.95 (requires broad agreement)
M: 0.6-0.9 (foundational to society)
```

### Personal Experience
```
E: 0.0-0.2 (only accessible to experiencer)
N: 0.0-0.2 (personal perspective)
M: 0.1-0.4 (limited lasting significance)
```

### Legal Precedent
```
E: 0.5-0.7 (documented, verifiable)
N: 0.7-0.9 (jurisdictional consensus)
M: 0.8-0.95 (foundational to legal system)
```

### Breaking News
```
E: 0.3-0.6 (developing, not fully verified)
N: 0.2-0.5 (reporting standards vary)
M: 0.1-0.3 (may be superseded quickly)
```

## Best Practices

### Do
- ✅ Be honest about uncertainty (lower E if unsure)
- ✅ Consider your claim's scope (N reflects this)
- ✅ Think about lasting relevance (M should be realistic)
- ✅ Provide evidence matching your E claim
- ✅ Update classification as verification improves

### Don't
- ❌ Inflate E without evidence
- ❌ Claim universal truth (high N) for opinions
- ❌ Overclaim mythic significance
- ❌ Ignore contradicting evidence
- ❌ Set and forget - classifications should evolve

## Verification Markets

Claims can request verification through Epistemic Markets:

```typescript
// Request verification market to raise E level
await knowledge.claims.spawnVerificationMarket({
  claimId: claimHash.toString(),
  targetE: 0.8,         // Target epistemic level
  minConfidence: 0.7,   // Minimum confidence required
  closesAt: Date.now() + 7 * 24 * 60 * 60 * 1000, // 1 week
  tags: ["climate", "temperature"]
});
```

When the market resolves, the claim's E level is updated based on oracle consensus.

## Related Documentation

- [Belief Graphs](./BELIEF_GRAPHS.md) - How classifications propagate
- [Credibility Engine](./CREDIBILITY_ENGINE.md) - Trust assessment
- [Epistemic Markets Integration](../integration/EPISTEMIC_MARKETS.md) - Verification through markets

---

*"Know what you know, and know that you know it."*
