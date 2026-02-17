# Mycelix DeSci - Charter-Aligned Epistemic Integration

**Version**: 0.3.0
**Date**: January 3, 2026
**Status**: Complete Epistemic Framework with Advanced Features

---

## Overview

Mycelix DeSci implements the **complete Epistemic Charter v2.0** framework through a multi-layer Epistemic Tensor architecture, plus advanced epistemic systems:

### Core Epistemic Layers

| Layer | Name | Purpose | Source |
|-------|------|---------|--------|
| **1** | LEM Cube | Governance classification | Epistemic Charter v2.0 |
| **2** | Type Position | Claim nature (E-N-M) | DeSci Extension |
| **3** | Quality Metrics | Scientific rigor | Research Standards |
| **4** | Network Position | Claim relationships | Constitution Schema |
| **+** | MATL Trust | Reputation-weighted verification | Economic Charter |

### Advanced Epistemic Systems (v0.3.0)

| System | Purpose | Module |
|--------|---------|--------|
| **Evolution** | Claim versioning & ancestry | `evolution.rs` |
| **Dispute Resolution** | Challenge & arbitration | `dispute.rs` |
| **Cartel Detection** | Anti-collusion for MATL | `cartel.rs` |
| **Reproducibility** | Replication tracking | `reproducibility.rs` |
| **Prediction Markets** | Epistemic forecasting | `prediction.rs` |
| **Semantic Similarity** | Duplicate detection | `semantic.rs` |
| **Decay Mechanics** | Time-based trust decay | `decay.rs` |

---

## Layer 1: LEM Cube (Official Charter v2.0)

The Layered Epistemic Model from the official Mycelix Epistemic Charter:

### E-Axis: Empirical Verifiability (E0-E4)
*How do we verify this claim?*

| Level | Name | Description |
|-------|------|-------------|
| E0 | Null | Unverifiable belief, subjective opinion |
| E1 | Testimonial | Personal attestation, witness account |
| E2 | Privately Verifiable | Expert verification (audit guild) |
| E3 | Cryptographically Proven | ZKP, merkle proofs, on-chain |
| E4 | Publicly Reproducible | Open data/code, anyone can verify |

### N-Axis: Normative Authority (N0-N3)
*Who agrees this is binding?*

| Level | Name | Description |
|-------|------|-------------|
| N0 | Personal | Self only, individual preference |
| N1 | Communal | Local DAO or working group consensus |
| N2 | Network | Global network consensus |
| N3 | Axiomatic | Constitutional or mathematical axiom |

### M-Axis: Materiality (M0-M3)
*How long does this matter?*

| Level | Name | Description |
|-------|------|-------------|
| M0 | Ephemeral | Discard immediately, real-time only |
| M1 | Temporal | Prune after state change, session-bound |
| M2 | Persistent | Archive after time, historical record |
| M3 | Foundational | Preserve forever, constitutional |

### Common LEM Patterns

| Claim Type | E | N | M | Example |
|------------|---|---|---|---------|
| Scientific Publication | E4 | N2 | M2 | Peer-reviewed paper |
| Governance Decision | E0 | N2 | M3 | Passed MIP |
| Personal Opinion | E0 | N0 | M0 | Chat message |
| Cryptographic Proof | E3 | N3 | M3 | ZK verification |
| Expert Review | E2 | N1 | M2 | Audit guild assessment |

---

## Layer 2: Claim Type Position (DeSci Extension)

Three-dimensional classification of claim **nature**:

- **Empirical (E)**: How observation-based? (0.0-1.0)
- **Normative (N)**: How value-laden? (0.0-1.0)
- **Mythic (M)**: How meaning-laden? (0.0-1.0)

| Claim Type | Empirical | Normative | Mythic | Example |
|------------|-----------|-----------|--------|---------|
| Scientific fact | 0.9 | 0.1 | 0.1 | "Water boils at 100°C" |
| Moral principle | 0.2 | 0.9 | 0.5 | "All humans have dignity" |
| Origin story | 0.1 | 0.4 | 0.9 | "The Big Bang" |
| Historical event | 0.8 | 0.5 | 0.8 | "Moon landing 1969" |

---

## Layer 3: Quality Metrics

Scientific rigor assessment:

```rust
pub struct QualityMetrics {
    pub methodology: f64,      // Study design quality
    pub data_quality: f64,     // Completeness, accuracy
    pub statistical_rigor: f64,// Appropriate methods
    pub preregistration: f64,  // Hypothesis preregistered?
    pub open_science: f64,     // Data/code availability
}
```

---

## Layer 4: Network Position (Claim Relationships)

From Constitution Schema v2.0:

```rust
pub enum ClaimRelationType {
    Supports,     // Evidence supporting another claim
    Refutes,      // Evidence against another claim
    Supercedes,   // Replaces/updates a previous claim
    Clarifies,    // Explains another claim
    Restricts,    // Limits scope of another claim
    Extends,      // Expands scope of another claim
    Cites,        // References as a source
    DependsOn,    // Requires another claim to be true
    Predicts,     // Makes prediction about another
    Conflicts,    // Direct conflict with another
}
```

---

## MATL Integration (Economic Charter)

Reputation-weighted verification:

```rust
pub struct MATLTrust {
    pub pogq_score: f64,           // Proof-of-Genuine-Query
    pub tcdm_score: f64,           // Trust Composition & Decay
    pub entropy_score: f64,        // Diversity of verifiers
    pub weighted_verifications: f64,// Reputation-weighted count
    pub raw_verifications: usize,   // Raw verification count
}
```

**Composite Trust Formula**:
```
Trust = 0.40 * PoGQ + 0.35 * TCDM + 0.25 * Entropy
```

---

## Epistemic Fingerprint (Unified View)

The `EpistemicFingerprint` combines all layers:

```rust
pub struct EpistemicFingerprint {
    pub lem_cube: LEMCube,           // Layer 1
    pub type_position: EpistemicPosition, // Layer 2
    pub quality: QualityMetrics,     // Layer 3
    pub network: NetworkPosition,     // Layer 4
    pub matl_trust: MATLTrust,       // MATL integration
    pub legacy_tier: EpistemicTier,  // E0-E4 compatibility
}
```

**Confidence Score**:
```
Confidence = 0.25 * LEM_weight + 0.10 * Type_weight
           + 0.25 * Quality_score + 0.20 * Network_support
           + 0.20 * MATL_trust
```

---

## API Usage

### Creating Claims

```rust
use mycelix_desci_core::*;

// Scientific claim with full fingerprint
let claim = DesciClaim::scientific(content, "researcher@uni.edu".to_string());
// Automatically: E4 LEM, high empirical position, high quality

// Cryptographic proof claim
let claim = DesciClaim::cryptographic_proof(content, "prover@chain.eth".to_string());
// Automatically: E3 LEM (crypto), axiomatic normative, perfect quality

// Custom fingerprint
let fingerprint = EpistemicFingerprint {
    lem_cube: LEMCube::new(
        EmpiricalAxis::E2PrivatelyVerifiable,
        NormativeAxis::N1Communal,
        MaterialityAxis::M2Persistent,
    ),
    type_position: EpistemicPosition::new(0.7, 0.4, 0.3),
    quality: QualityMetrics::high_quality(),
    network: NetworkPosition::default(),
    matl_trust: MATLTrust::default(),
    legacy_tier: EpistemicTier::E2,
};
let claim = DesciClaim::with_fingerprint(fingerprint, content, creator);
```

### Adding Relationships

```rust
// Support relationship
claim.add_relation(other_claim_id, ClaimRelationType::Supports, 0.9);

// Refutation
claim.add_relation(disputed_claim_id, ClaimRelationType::Refutes, 0.8);

// Citation
claim.add_relation(source_claim_id, ClaimRelationType::Cites, 1.0);
```

### MATL-Weighted Verification

```rust
// Standard verification (weight = 1.0)
claim.add_verification(verification);

// High-reputation verifier (weight = 2.5)
claim.add_verification_with_reputation(verification, 2.5);

// This affects tier progression via MATL
assert_eq!(claim.fingerprint.matl_trust.weighted_verifications, 2.5);
```

### Querying Confidence

```rust
// Get unified confidence score (0.0-1.0)
let confidence = claim.confidence();

// Get human-readable summary
let summary = claim.epistemic_summary();
// "LEM(E4, N2, M2) | Type:empirical | Quality:88% | Support:50% | MATL:0% | Confidence:45%"
```

---

## Integration with Mycelix Ecosystem

### With mycelix-knowledge
```rust
// DeSci claims sync to knowledge graph with full LEM classification
```

### With mycelix-media
```rust
// Media fact-checking uses LEM E-axis for verification type
```

### With mycelix-governance
```rust
// Governance MIPs use LEM (E0, N2, M3) pattern
```

### With mycelix-edunet
```rust
// Research credentials reference verified E4 claims
```

---

## Advanced Epistemic Systems

### Claim Evolution & Versioning

From Constitution Schema v2.0 - tracks claim ancestry and modifications:

```rust
pub enum EvolutionType {
    Genesis,      // Original claim
    Amendment,    // Minor update preserving core findings
    Correction,   // Error correction
    Retraction,   // Full withdrawal
    Supersession, // Replaced by newer research
    Extension,    // Extended with new findings
    Consolidation,// Merged from multiple claims
}

pub struct ClaimEvolution {
    pub version: u32,
    pub parent_claim_id: Option<Uuid>,
    pub evolution_type: EvolutionType,
    pub changelog: String,
    pub evolution_chain: Vec<Uuid>, // Full ancestry
}
```

### Dispute Resolution System

From Epistemic Charter §5 - challenge claims and reach resolution:

```rust
pub enum ChallengeType {
    Factual,           // Data/conclusions are wrong
    Methodological,    // Flawed study design
    Ethical,           // Research ethics violations
    Reproducibility,   // Cannot replicate results
    Attribution,       // Plagiarism issues
    ConflictOfInterest,// Undisclosed conflicts
    DataIntegrity,     // Fabrication/falsification
    OverClaim,         // Conclusions exceed evidence
}

pub enum ResolutionOutcome {
    ChallengeUpheld,    // Claim is invalid
    PartiallyUpheld,    // Minor issues found
    ChallengeRejected,  // Claim stands
    Inconclusive,       // Insufficient evidence
    MutualResolution,   // Parties reached agreement
}
```

### Cartel Detection for MATL

Anti-collusion algorithms to prevent gaming of reputation:

```rust
pub enum CartelPattern {
    MutualVerification,   // A verifies B, B verifies A
    SynchronizedTiming,   // Coordinated timing
    ExclusiveClique,      // Closed group only verifies each other
    CoordinatedVoting,    // Identical vote patterns
    VelocityAnomaly,      // Abnormal verification frequency
}

// Automatic trust penalty based on detection
pub fn calculate_trust_penalty(result: &CartelDetectionResult) -> f64;
```

### Reproducibility Tracking

Track replication attempts and outcomes:

```rust
pub enum ReplicationOutcome {
    FullReplication,     // Same results within error margin
    PartialReplication,  // Some findings confirmed
    FailureToReplicate,  // Contradictory results
    Inconclusive,        // Unable to determine
}

pub struct ReproducibilityStats {
    pub total_attempts: usize,
    pub full_replications: usize,
    pub reproducibility_score: f64,  // 0.0-1.0
    pub score_confidence: f64,        // Based on sample size
}
```

### Prediction Markets

Futarchy-style markets for epistemic forecasting:

```rust
pub struct PredictionMarket {
    pub claim_id: Uuid,
    pub current_probability: f64,    // Market price
    pub total_stake: f64,            // Total reputation staked
    pub state: MarketState,          // Open/Locked/Resolved
}

// Brier scoring rule for settlements
// accuracy = 1 - (prediction - outcome)²
```

### Semantic Similarity Engine

Detect duplicate and related claims:

```rust
pub enum SimilarityRelationship {
    Duplicate,       // >= 0.95 similarity
    NearDuplicate,   // >= 0.85
    HighlyRelated,   // >= 0.70
    Related,         // >= 0.50
    WeaklyRelated,   // >= 0.30
    Unrelated,       // < 0.30
}

pub struct DuplicateCheckResult {
    pub has_duplicate: bool,
    pub recommendation: DuplicateRecommendation,
}
```

### Decay Mechanics

Time-based decay for trust and verification weights:

```rust
pub enum DecayFunction {
    Linear,       // weight = 1 - (age / max_age)
    Exponential,  // weight = exp(-λ * age)
    Logarithmic,  // weight = 1 / (1 + ln(1 + age/scale))
    Step,         // Full weight until threshold
    None,         // No decay (foundational claims)
}

// Presets based on LEM M-axis (Materiality)
pub mod presets {
    fn ephemeral() -> DecayConfig;    // M0: 7-day half-life
    fn temporal() -> DecayConfig;     // M1: 90-day half-life
    fn persistent() -> DecayConfig;   // M2: 365-day half-life
    fn foundational() -> DecayConfig; // M3: No decay
}
```

---

## Test Coverage

**217 tests passing** covering:

### Core Epistemic (36 tests)
- LEM Cube construction and trust weights
- Quality metrics computation
- Network position tracking
- MATL trust scoring and tier derivation
- Epistemic fingerprint confidence
- Full claim integration
- Relationship management
- Cryptographic proof claims

### Advanced Systems (42 tests)
- Claim evolution and versioning (5 tests)
- Dispute resolution lifecycle (5 tests)
- Cartel pattern detection (5 tests)
- Reproducibility tracking (5 tests)
- Prediction market mechanics (5 tests)
- Semantic similarity algorithms (7 tests)
- Decay function calculations (10 tests)

### Supporting Infrastructure (139 tests)
- Query engine and filters
- Storage operations
- Validation utilities
- Error handling

---

## Building & Testing

```bash
cd mycelix-desci

# Using Nix (recommended)
nix develop
cargo test --lib

# Run specific module tests
cargo test --lib evolution::tests
cargo test --lib dispute::tests
cargo test --lib cartel::tests
cargo test --lib reproducibility::tests
cargo test --lib prediction::tests
cargo test --lib semantic::tests
cargo test --lib decay::tests

# Run API server
cargo run --bin mycelix-api
```

---

## Version History

- **v0.3.0** (2026-01-03): Advanced Epistemic Systems
  - Added Claim Evolution & Versioning (Constitution Schema v2.0)
  - Added Dispute Resolution (Epistemic Charter §5)
  - Added Cartel Detection for MATL anti-collusion
  - Added Reproducibility Tracking for scientific claims
  - Added Prediction Markets (futarchy-style)
  - Added Semantic Similarity Engine (duplicate detection)
  - Added Decay Mechanics (time-based trust decay)
  - 217 tests passing

- **v0.2.0** (2026-01-03): Full Charter v2.0 alignment
  - Added LEM Cube (E/N/M axes)
  - Added Quality Metrics
  - Added Network Position & relationships
  - Added MATL Trust integration
  - Added Epistemic Fingerprint
  - 36 tests passing

- **v0.1.0** (2026-01-03): Initial MVP
  - Basic E0-E4 tiers
  - Simple E-N-M type position
  - 12 tests passing

---

*Part of the Mycelix Civilizational OS - Knowledge Pillar*
