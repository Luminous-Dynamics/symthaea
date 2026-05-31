# Zome Functions Reference

Complete reference for all zome functions in the Knowledge hApp.

## claims/ Zome

### Entry Management

#### `create_claim(input: CreateClaimInput) → ActionHash`

Create a new claim in the knowledge graph.

```rust
pub struct CreateClaimInput {
    pub content: String,
    pub classification: EpistemicPosition,
    pub domain: String,
    pub topics: Vec<String>,
    pub evidence: Vec<Evidence>,
}
```

**Returns**: ActionHash of the created claim entry.

**Errors**:
- `InvalidClassification` - E, N, or M outside [0.0, 1.0]
- `EmptyContent` - Content string is empty
- `InvalidDomain` - Domain not in allowed list

---

#### `get_claim(hash: ActionHash) → Option<Record>`

Retrieve a claim by its hash.

**Returns**: The claim record if found, None otherwise.

---

#### `update_claim(input: UpdateClaimInput) → ActionHash`

Update an existing claim. Only the original author can update.

```rust
pub struct UpdateClaimInput {
    pub original_hash: ActionHash,
    pub content: Option<String>,
    pub classification: Option<EpistemicPosition>,
    pub status: Option<ClaimStatus>,
}
```

**Returns**: ActionHash of the updated entry.

**Errors**:
- `NotAuthor` - Caller is not the original author
- `ClaimNotFound` - Original claim doesn't exist

---

#### `delete_claim(hash: ActionHash) → ActionHash`

Mark a claim as deleted (soft delete).

**Returns**: ActionHash of the delete action.

---

### Claim Queries

#### `list_claims_by_author(author: AgentPubKey) → Vec<Record>`

Get all claims by a specific author.

---

#### `list_claims_by_domain(domain: String) → Vec<Record>`

Get all claims in a domain.

---

#### `search_claims_by_topic(topic: String) → Vec<Record>`

Search claims by topic tag.

---

#### `get_claims_pending_verification(min_e: f64) → Vec<Record>`

Get claims that need verification (E below threshold).

---

### Market Integration

#### `spawn_verification_market(input: SpawnMarketInput) → EntryHash`

Create a verification market for a claim.

```rust
pub struct SpawnMarketInput {
    pub claim_id: String,
    pub target_e: f64,
    pub min_confidence: f64,
    pub closes_at: u64,
    pub initial_subsidy: u64,
    pub tags: Vec<String>,
}
```

**Returns**: EntryHash of the market request.

---

#### `on_market_resolved(input: MarketResolutionInput) → ()`

Callback when a verification market resolves.

```rust
pub struct MarketResolutionInput {
    pub market_id: String,
    pub claim_id: String,
    pub outcome: MarketOutcome,  // Verified | Refuted | Inconclusive
    pub final_price: f64,
    pub confidence: f64,
}
```

---

#### `get_claim_markets(claim_id: String) → Vec<MarketSummary>`

Get all markets associated with a claim.

---

### Dependency Management

#### `register_dependency(input: DependencyInput) → Record`

Register a belief dependency between claims.

```rust
pub struct DependencyInput {
    pub dependent_claim_id: String,
    pub dependency_claim_id: String,
    pub dependency_type: DependencyType,
    pub weight: f64,
    pub justification: String,
}
```

---

#### `cascade_update(claim_id: String) → CascadeResult`

Propagate updates to dependent claims.

```rust
pub struct CascadeResult {
    pub claims_updated: u32,
    pub markets_notified: Vec<String>,
    pub errors: Vec<String>,
}
```

---

## graph/ Zome

### Relationship Management

#### `create_relationship(input: RelationshipInput) → ActionHash`

Create a relationship between claims.

```rust
pub struct RelationshipInput {
    pub source_id: String,
    pub target_id: String,
    pub relationship_type: RelationshipType,
    pub weight: f64,
    pub properties: Option<HashMap<String, String>>,
}

pub enum RelationshipType {
    Supports,
    Contradicts,
    DerivedFrom,
    Generalizes,
    Specializes,
    Causes,
    PartOf,
    Equivalent,
}
```

---

#### `get_relationships(claim_id: String) → Vec<Relationship>`

Get all relationships for a claim.

---

#### `delete_relationship(hash: ActionHash) → ActionHash`

Remove a relationship.

---

### Belief Propagation

#### `propagate_belief(claim_id: String) → PropagationResult`

Propagate belief updates through the graph.

```rust
pub struct PropagationResult {
    pub nodes_updated: u32,
    pub iterations: u32,
    pub converged: bool,
    pub max_change: f64,
}
```

---

#### `get_dependency_tree(claim_id: String, max_depth: u32) → DependencyTree`

Get the dependency tree for a claim.

```rust
pub struct DependencyTree {
    pub root: String,
    pub dependencies: Vec<DependencyNode>,
    pub depth: u32,
    pub total_nodes: u32,
}
```

---

#### `detect_circular_dependencies(claim_id: String) → Vec<Vec<String>>`

Detect circular dependencies starting from a claim.

**Returns**: List of cycles (each cycle is a list of claim IDs).

---

### Information Value

#### `rank_by_information_value(limit: u32) → Vec<InformationValueRecord>`

Rank claims by their information value for verification.

```rust
pub struct InformationValueRecord {
    pub claim_id: String,
    pub value: f64,
    pub components: InformationValueComponents,
}

pub struct InformationValueComponents {
    pub uncertainty: f64,
    pub connectivity: f64,
    pub cascade_potential: f64,
    pub age_factor: f64,
}
```

---

#### `calculate_cascade_impact(claim_id: String) → CascadeImpact`

Calculate the potential impact of updating a claim.

```rust
pub struct CascadeImpact {
    pub direct_dependents: u32,
    pub total_reachable: u32,
    pub weighted_impact: f64,
    pub high_value_affected: Vec<String>,
}
```

---

## query/ Zome

### Search Functions

#### `search(query: String, options: Option<QueryOptions>) → Vec<ClaimSummary>`

Full-text search for claims.

```rust
pub struct QueryOptions {
    pub min_e: Option<f64>,
    pub max_e: Option<f64>,
    pub min_n: Option<f64>,
    pub max_n: Option<f64>,
    pub min_m: Option<f64>,
    pub max_m: Option<f64>,
    pub domain: Option<String>,
    pub status: Option<ClaimStatus>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}
```

---

#### `query_by_epistemic(input: EpistemicQuery) → Vec<ClaimSummary>`

Query by epistemic position range.

```rust
pub struct EpistemicQuery {
    pub min_e: Option<f64>,
    pub max_e: Option<f64>,
    pub min_n: Option<f64>,
    pub max_n: Option<f64>,
    pub min_m: Option<f64>,
    pub max_m: Option<f64>,
    pub limit: u32,
}
```

---

#### `advanced_query(input: AdvancedQueryInput) → QueryResult`

Complex queries with filters and sorting.

```rust
pub struct AdvancedQueryInput {
    pub filters: Vec<QueryFilter>,
    pub sort: Vec<SortSpec>,
    pub limit: u32,
    pub offset: u32,
}

pub struct QueryFilter {
    pub field: String,
    pub operator: FilterOperator,
}

pub enum FilterOperator {
    Equals(String),
    NotEquals(String),
    GreaterThan(f64),
    LessThan(f64),
    Between(f64, f64),
    In(Vec<String>),
    Contains(String),
}
```

---

### Relationship Queries

#### `find_related(claim_id: String, max_depth: u32, limit: u32) → Vec<RelatedClaim>`

Find claims related to a given claim.

---

#### `find_contradictions(claim_id: String) → Vec<ClaimSummary>`

Find claims that contradict the given claim.

---

## inference/ Zome

### Credibility Assessment

#### `calculate_enhanced_credibility(input: CredibilityInput) → EnhancedCredibilityScore`

Calculate comprehensive credibility score.

```rust
pub struct CredibilityInput {
    pub entity_id: String,
    pub entity_type: EntityType,  // Claim | Author | Source
}

pub struct EnhancedCredibilityScore {
    pub overall_score: f64,
    pub epistemic_scores: EpistemicScores,
    pub matl_components: MatlComponents,
    pub evidence_strength: EvidenceStrengthComponents,
    pub temporal_factors: TemporalFactors,
    pub network_position: NetworkPosition,
    pub confidence_interval: ConfidenceInterval,
}
```

---

#### `batch_credibility_assessment(claim_ids: Vec<String>) → BatchCredibilityResult`

Assess multiple claims efficiently.

```rust
pub struct BatchCredibilityResult {
    pub results: Vec<EnhancedCredibilityScore>,
    pub average_score: f64,
    pub processing_time_ms: u64,
}
```

---

### Author Reputation

#### `get_author_reputation(author_did: String) → AuthorReputation`

Get reputation metrics for an author.

```rust
pub struct AuthorReputation {
    pub did: String,
    pub overall_score: f64,
    pub matl_trust: f64,
    pub claims_authored: u32,
    pub claims_verified_true: u32,
    pub claims_verified_false: u32,
    pub domain_scores: Vec<DomainScore>,
    pub historical_accuracy: f64,
    pub citation_count: u32,
}
```

---

### Evidence Analysis

#### `assess_evidence_strength(claim_id: String) → EvidenceStrengthComponents`

Analyze the evidence supporting a claim.

```rust
pub struct EvidenceStrengthComponents {
    pub empirical_evidence_count: u32,
    pub testimonial_evidence_count: u32,
    pub cryptographic_evidence_count: u32,
    pub corroboration_score: f64,
    pub source_diversity: f64,
    pub recency_weighted_strength: f64,
}
```

---

## factcheck/ Zome

### Fact-Checking

#### `fact_check(input: FactCheckInput) → FactCheckResult`

Perform a fact-check on a statement.

```rust
pub struct FactCheckInput {
    pub statement: String,
    pub context: Option<String>,
    pub min_e: Option<f64>,
    pub min_n: Option<f64>,
    pub source_happ: Option<String>,
}

pub struct FactCheckResult {
    pub verdict: Verdict,
    pub confidence: f64,
    pub supporting_claims: Vec<ClaimEvidence>,
    pub contradicting_claims: Vec<ClaimEvidence>,
    pub explanation: String,
    pub checked_at: u64,
}

pub enum Verdict {
    True,
    MostlyTrue,
    Mixed,
    MostlyFalse,
    False,
    Unverifiable,
    InsufficientEvidence,
}
```

---

#### `batch_fact_check(inputs: Vec<FactCheckInput>) → Vec<FactCheckResult>`

Fact-check multiple statements.

---

#### `query_claims_by_topic(topics: Vec<String>, options: QueryOptions) → Vec<ClaimSummary>`

Query claims by topic for fact-checking.

---

## markets_integration/ Zome

### Market Operations

#### `request_verification_market(input: VerificationMarketInput) → EntryHash`

Request creation of a verification market.

```rust
pub struct VerificationMarketInput {
    pub claim_id: String,
    pub target_epistemic: f64,
    pub subsidy_amount: u64,
    pub duration_days: u32,
}
```

---

#### `calculate_market_value(claim_id: String) → MarketValueAssessment`

Calculate the value of creating a market for a claim.

```rust
pub struct MarketValueAssessment {
    pub expected_value: f64,
    pub recommended_subsidy: u64,
    pub reasoning: String,
    pub risk_factors: Vec<String>,
}
```

---

#### `on_market_created(market_id: String, claim_id: String) → ()`

Callback when Epistemic Markets creates a verification market.

---

#### `on_verification_market_resolved(input: ResolutionInput) → ()`

Callback when a verification market resolves.

---

#### `register_claim_in_prediction(input: ClaimPredictionInput) → Record`

Register that a claim is being used as evidence in a prediction market.

```rust
pub struct ClaimPredictionInput {
    pub claim_id: String,
    pub market_id: String,
    pub usage_type: UsageType,  // Evidence | Condition | Outcome
}
```

---

## bridge/ Zome

### Cross-hApp Communication

#### `submit_external_claim(input: ExternalClaimInput) → ActionHash`

Submit a claim from another hApp.

```rust
pub struct ExternalClaimInput {
    pub content: String,
    pub classification: EpistemicPosition,
    pub source_happ: String,
    pub domain: String,
    pub evidence: Vec<Evidence>,
}
```

---

#### `query_for_happ(input: HappQueryInput) → Vec<ClaimSummary>`

Query claims for a specific hApp context.

```rust
pub struct HappQueryInput {
    pub domain: String,
    pub min_e: f64,
    pub topics: Vec<String>,
    pub limit: u32,
}
```

---

#### `register_happ(input: HappRegistration) → EntryHash`

Register an external hApp for bridge access.

---

#### `get_registered_happs() → Vec<HappInfo>`

List all registered hApps.

---

## Common Types

### EpistemicPosition

```rust
pub struct EpistemicPosition {
    pub empirical: f64,   // 0.0 - 1.0
    pub normative: f64,   // 0.0 - 1.0
    pub mythic: f64,      // 0.0 - 1.0
}
```

### ClaimStatus

```rust
pub enum ClaimStatus {
    Draft,
    Published,
    UnderReview,
    Verified,
    Disputed,
    Retracted,
}
```

### Evidence

```rust
pub struct Evidence {
    pub id: String,
    pub evidence_type: EvidenceType,
    pub source: String,
    pub content: String,
    pub strength: f64,
}

pub enum EvidenceType {
    Empirical,
    Testimonial,
    Documentary,
    Cryptographic,
    Statistical,
}
```

---

## Error Handling

All functions return `ExternResult<T>` which may contain:
- Success value
- `WasmError` with error details

See [Error Codes](./ERROR_CODES.md) for complete error reference.

---

*Complete function reference for all Knowledge zomes.*
