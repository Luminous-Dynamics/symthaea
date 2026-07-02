# Entry Types Reference

Complete reference for all entry types in the Knowledge hApp.

## Overview

The Knowledge hApp uses Holochain's entry system to store structured data in the DHT. Each entry type has an integrity zome definition and associated validation rules.

## claims/ Zome Entries

### Claim

The primary entry type representing a knowledge claim.

```rust
#[hdk_entry_helper]
pub struct Claim {
    /// Unique identifier
    pub id: String,

    /// The claim content/statement
    pub content: String,

    /// Epistemic classification (E-N-M position)
    pub classification: EpistemicPosition,

    /// Knowledge domain
    pub domain: String,

    /// Topic tags for discovery
    pub topics: Vec<String>,

    /// Supporting evidence
    pub evidence: Vec<Evidence>,

    /// Current status
    pub status: ClaimStatus,

    /// Source hApp if submitted externally
    pub source_happ: Option<String>,

    /// Creation timestamp
    pub created_at: u64,

    /// Last update timestamp
    pub updated_at: u64,
}
```

**Validation Rules**:
- `content` must be non-empty and < 10,000 characters
- `classification` values must be in [0.0, 1.0]
- `domain` must be from allowed domains list
- `topics` limited to 20 tags
- `evidence` limited to 50 items per claim

---

### ClaimMarketLink

Links a claim to its verification markets.

```rust
#[hdk_entry_helper]
pub struct ClaimMarketLink {
    /// The claim being verified
    pub claim_id: String,

    /// The verification market ID
    pub market_id: String,

    /// Target epistemic level for verification
    pub target_e: f64,

    /// Market creation timestamp
    pub created_at: u64,

    /// Current market status
    pub status: MarketLinkStatus,
}

pub enum MarketLinkStatus {
    Active,
    Resolved,
    Cancelled,
}
```

---

### ClaimDependency

Represents a belief dependency between claims.

```rust
#[hdk_entry_helper]
pub struct ClaimDependency {
    /// The claim that depends on another
    pub dependent_claim_id: String,

    /// The claim being depended upon
    pub dependency_claim_id: String,

    /// Type of dependency relationship
    pub dependency_type: DependencyType,

    /// Weight/strength of dependency (0.0-1.0)
    pub weight: f64,

    /// Justification for the dependency
    pub justification: String,

    /// Creation timestamp
    pub created_at: u64,
}

pub enum DependencyType {
    /// Derived from logical reasoning
    DerivedFrom,
    /// Empirically depends on
    EmpiricallyDependsOn,
    /// Shares evidence with
    SharedEvidence,
    /// Conditional dependency
    Conditional,
}
```

---

## graph/ Zome Entries

### Relationship

Connects two claims with a typed relationship.

```rust
#[hdk_entry_helper]
pub struct Relationship {
    /// Source claim ID
    pub source_id: String,

    /// Target claim ID
    pub target_id: String,

    /// Type of relationship
    pub relationship_type: RelationshipType,

    /// Relationship strength (0.0-1.0)
    pub weight: f64,

    /// Optional metadata
    pub properties: HashMap<String, String>,

    /// Creation timestamp
    pub created_at: u64,

    /// Creating agent
    pub created_by: AgentPubKey,
}

pub enum RelationshipType {
    /// A supports B (evidence for)
    Supports,
    /// A contradicts B (evidence against)
    Contradicts,
    /// A is derived from B
    DerivedFrom,
    /// A generalizes B (abstraction)
    Generalizes,
    /// A specializes B (instance)
    Specializes,
    /// A causes B (causal link)
    Causes,
    /// A is part of B (component)
    PartOf,
    /// A is equivalent to B
    Equivalent,
}
```

**Validation Rules**:
- `source_id` and `target_id` must exist
- `weight` must be in [0.0, 1.0]
- No self-referential relationships (`source_id` != `target_id`)
- Contradicts relationships require minimum weight of 0.5

---

### BeliefNode

Represents a node in the belief propagation graph.

```rust
#[hdk_entry_helper]
pub struct BeliefNode {
    /// Associated claim ID
    pub claim_id: String,

    /// Current belief strength (0.0-1.0)
    pub belief_strength: f64,

    /// Last propagation timestamp
    pub last_updated: u64,

    /// Number of propagation iterations
    pub iteration_count: u32,

    /// Whether belief has converged
    pub converged: bool,

    /// Incoming influence scores
    pub influences: Vec<Influence>,
}

pub struct Influence {
    pub source_claim_id: String,
    pub influence_weight: f64,
    pub relationship_type: RelationshipType,
}
```

---

### InformationValue

Tracks the verification priority of claims.

```rust
#[hdk_entry_helper]
pub struct InformationValue {
    /// Associated claim ID
    pub claim_id: String,

    /// Overall information value score
    pub value: f64,

    /// Component breakdown
    pub components: InformationValueComponents,

    /// Calculation timestamp
    pub calculated_at: u64,
}

pub struct InformationValueComponents {
    /// Uncertainty (higher = more valuable to verify)
    pub uncertainty: f64,

    /// Number of connections
    pub connectivity: f64,

    /// Potential cascade impact
    pub cascade_potential: f64,

    /// Age factor (older unverified = higher priority)
    pub age_factor: f64,
}
```

---

## inference/ Zome Entries

### CredibilityAssessment

Cached credibility assessment for an entity.

```rust
#[hdk_entry_helper]
pub struct CredibilityAssessment {
    /// Entity being assessed
    pub entity_id: String,

    /// Type of entity
    pub entity_type: EntityType,

    /// Overall credibility score
    pub overall_score: f64,

    /// Detailed score breakdown
    pub score_details: EnhancedCredibilityScore,

    /// Assessment timestamp
    pub assessed_at: u64,

    /// Time-to-live (validity period)
    pub ttl_seconds: u64,
}

pub enum EntityType {
    Claim,
    Author,
    Source,
    Evidence,
}
```

---

### AuthorProfile

Aggregated reputation data for an author.

```rust
#[hdk_entry_helper]
pub struct AuthorProfile {
    /// Author's DID
    pub did: String,

    /// Agent public key
    pub agent_pubkey: AgentPubKey,

    /// MATL trust score
    pub matl_trust: f64,

    /// Total claims authored
    pub claims_authored: u32,

    /// Claims verified as true
    pub claims_verified_true: u32,

    /// Claims verified as false
    pub claims_verified_false: u32,

    /// Domain-specific scores
    pub domain_scores: Vec<DomainScore>,

    /// Historical accuracy rate
    pub historical_accuracy: f64,

    /// Total citations received
    pub citation_count: u32,

    /// Profile last updated
    pub updated_at: u64,
}

pub struct DomainScore {
    pub domain: String,
    pub score: f64,
    pub claims_in_domain: u32,
}
```

---

## factcheck/ Zome Entries

### FactCheckRequest

An external fact-check request.

```rust
#[hdk_entry_helper]
pub struct FactCheckRequest {
    /// Unique request ID
    pub request_id: String,

    /// Statement to fact-check
    pub statement: String,

    /// Optional context
    pub context: Option<String>,

    /// Minimum E level required
    pub min_e: Option<f64>,

    /// Minimum N level required
    pub min_n: Option<f64>,

    /// Requesting hApp
    pub source_happ: Option<String>,

    /// Request timestamp
    pub requested_at: u64,

    /// Request status
    pub status: RequestStatus,
}

pub enum RequestStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}
```

---

### FactCheckResult

The result of a fact-check operation.

```rust
#[hdk_entry_helper]
pub struct FactCheckResult {
    /// Associated request ID
    pub request_id: String,

    /// The verdict
    pub verdict: Verdict,

    /// Confidence in the verdict (0.0-1.0)
    pub confidence: f64,

    /// Claims supporting the statement
    pub supporting_claims: Vec<ClaimEvidence>,

    /// Claims contradicting the statement
    pub contradicting_claims: Vec<ClaimEvidence>,

    /// Human-readable explanation
    pub explanation: String,

    /// Completion timestamp
    pub checked_at: u64,
}

pub enum Verdict {
    /// Statement is true
    True,
    /// Statement is mostly true with minor issues
    MostlyTrue,
    /// Mixed evidence, partially true/false
    Mixed,
    /// Statement is mostly false
    MostlyFalse,
    /// Statement is false
    False,
    /// Cannot be verified (unfalsifiable)
    Unverifiable,
    /// Not enough evidence either way
    InsufficientEvidence,
}

pub struct ClaimEvidence {
    pub claim_id: String,
    pub relevance: f64,
    pub supports: bool,
    pub excerpt: String,
}
```

---

## markets_integration/ Zome Entries

### VerificationMarketRequest

Request to create a verification market.

```rust
#[hdk_entry_helper]
pub struct VerificationMarketRequest {
    /// Associated claim ID
    pub claim_id: String,

    /// Target epistemic level to verify
    pub target_e: f64,

    /// Required confidence for resolution
    pub min_confidence: f64,

    /// Market close timestamp
    pub closes_at: u64,

    /// Initial subsidy amount
    pub initial_subsidy: u64,

    /// Topic tags for the market
    pub tags: Vec<String>,

    /// Request status
    pub status: MarketRequestStatus,

    /// Created market ID (if successful)
    pub market_id: Option<String>,

    /// Request timestamp
    pub requested_at: u64,
}

pub enum MarketRequestStatus {
    Pending,
    Created,
    Rejected,
    Cancelled,
}
```

---

### MarketEvidence

Evidence from resolved verification markets.

```rust
#[hdk_entry_helper]
pub struct MarketEvidence {
    /// The resolved market ID
    pub market_id: String,

    /// Associated claim ID
    pub claim_id: String,

    /// Market outcome
    pub outcome: MarketOutcome,

    /// Final price (probability)
    pub final_price: f64,

    /// Resolution confidence
    pub confidence: f64,

    /// Number of participants
    pub participant_count: u32,

    /// Total stake in market
    pub total_stake: u64,

    /// Resolution timestamp
    pub resolved_at: u64,
}

pub enum MarketOutcome {
    Verified,
    Refuted,
    Inconclusive,
}
```

---

### ClaimAsMarketEvidence

Tracks claim usage in prediction markets.

```rust
#[hdk_entry_helper]
pub struct ClaimAsMarketEvidence {
    /// The claim being referenced
    pub claim_id: String,

    /// The market using this claim
    pub market_id: String,

    /// How the claim is being used
    pub usage_type: UsageType,

    /// When the reference was created
    pub referenced_at: u64,

    /// Claim's E level at time of reference
    pub e_at_reference: f64,
}

pub enum UsageType {
    /// Used as evidence for a prediction
    Evidence,
    /// Part of market resolution condition
    Condition,
    /// Linked to market outcome
    Outcome,
}
```

---

## bridge/ Zome Entries

### HappRegistration

Registration for cross-hApp access.

```rust
#[hdk_entry_helper]
pub struct HappRegistration {
    /// hApp identifier
    pub happ_id: String,

    /// Human-readable name
    pub name: String,

    /// hApp description
    pub description: String,

    /// Allowed domains for this hApp
    pub allowed_domains: Vec<String>,

    /// Permission level
    pub permission_level: PermissionLevel,

    /// Registration timestamp
    pub registered_at: u64,

    /// Registration status
    pub status: RegistrationStatus,
}

pub enum PermissionLevel {
    /// Full read/write access
    Full,
    /// Standard access (read all, write own domain)
    Standard,
    /// Limited access (read public, limited write)
    Limited,
}

pub enum RegistrationStatus {
    Active,
    Suspended,
    Revoked,
}
```

---

## Shared Types

### EpistemicPosition

The core 3D classification system.

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpistemicPosition {
    /// Empirical verifiability (0.0-1.0)
    /// 0.0 = purely theoretical
    /// 1.0 = directly measurable
    pub empirical: f64,

    /// Normative content (0.0-1.0)
    /// 0.0 = purely descriptive
    /// 1.0 = purely prescriptive
    pub normative: f64,

    /// Mythic/foundational significance (0.0-1.0)
    /// 0.0 = mundane fact
    /// 1.0 = foundational worldview element
    pub mythic: f64,
}
```

---

### Evidence

Supporting evidence for claims.

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Evidence {
    /// Unique evidence identifier
    pub id: String,

    /// Type of evidence
    pub evidence_type: EvidenceType,

    /// Source URL or reference
    pub source: String,

    /// Evidence content/description
    pub content: String,

    /// Evidence strength (0.0-1.0)
    pub strength: f64,

    /// When evidence was added
    pub added_at: Option<u64>,
}

pub enum EvidenceType {
    /// Observable, measurable data
    Empirical,
    /// Witness testimony
    Testimonial,
    /// Documents, records
    Documentary,
    /// Cryptographic proofs
    Cryptographic,
    /// Statistical analysis
    Statistical,
}
```

---

### ClaimStatus

Lifecycle status of a claim.

```rust
pub enum ClaimStatus {
    /// Work in progress, not published
    Draft,
    /// Published and visible
    Published,
    /// Currently being reviewed/verified
    UnderReview,
    /// Verified through market or other means
    Verified,
    /// Disputed by contradicting evidence
    Disputed,
    /// Retracted by author
    Retracted,
}
```

---

## Link Types

The Knowledge hApp uses the following link types:

| Link Type | From | To | Purpose |
|-----------|------|----|---------|
| `ClaimToAuthor` | Claim | AgentPubKey | Track authorship |
| `AuthorToClaims` | AgentPubKey | Claim | Find author's claims |
| `ClaimToEvidence` | Claim | Evidence | Link evidence |
| `ClaimToDomain` | Claim | Domain | Domain indexing |
| `ClaimToTopic` | Claim | Topic | Topic indexing |
| `ClaimToRelationship` | Claim | Relationship | Graph edges |
| `ClaimToMarket` | Claim | MarketLink | Market tracking |
| `ClaimToDependency` | Claim | Dependency | Belief graph |
| `ClaimToFactCheck` | Claim | FactCheckResult | Fact-check results |

---

## Validation

All entry types are validated in the integrity zomes. Key validation patterns:

1. **Range validation**: Scores must be in [0.0, 1.0]
2. **Reference validation**: Referenced entries must exist
3. **Author validation**: Only authors can modify their entries
4. **Domain validation**: Domains must be from allowed list
5. **Size limits**: Content and collections have maximum sizes

See individual zome documentation for specific validation rules.

---

*Complete entry type reference for the Knowledge hApp.*
