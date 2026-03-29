// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic Claims - Charter-Aligned Implementation
//!
//! Comprehensive epistemic classification system for DeSci claims,
//! implementing the Mycelix Protocol's Epistemic Charter v2.0.
//!
//! ## Epistemic Tensor Architecture
//!
//! This module implements a multi-dimensional classification:
//!
//! ### Layer 1: LEM Cube (Official Charter v2.0)
//! Three-dimensional governance properties:
//! - **E-Axis (Empirical Verifiability)**: E0-E4 - How is this verified?
//! - **N-Axis (Normative Authority)**: N0-N3 - Who agrees it's binding?
//! - **M-Axis (Materiality)**: M0-M3 - How long does this matter?
//!
//! ### Layer 2: Claim Type Position (DeSci Extension)
//! Three-dimensional claim nature:
//! - **Empirical**: How observation-based? (0.0-1.0)
//! - **Normative**: How value-laden? (0.0-1.0)
//! - **Mythic**: How meaning-laden? (0.0-1.0)
//!
//! ### Layer 3: Quality Metrics (Scientific Rigor)
//! - Methodology, data quality, statistical rigor, preregistration, open science
//!
//! ### Layer 4: Network Position (Claim Relationships)
//! - SUPPORTS, REFUTES, SUPERCEDES, CLARIFIES, RESTRICTS, EXTENDS
//! - Citation graph, dependencies, conflicts, predictions
//!
//! ## MATL Integration
//! Trust scores are reputation-weighted through the Mycelix Adaptive Trust Layer.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// LAYER 1: LEM CUBE (Official Epistemic Charter v2.0)
// ============================================================================

/// E-Axis: Empirical Verifiability (from Epistemic Charter v2.0)
///
/// How do we verify this claim?
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EmpiricalAxis {
    /// E0: Null - Unverifiable belief, subjective opinion
    E0Null = 0,
    /// E1: Testimonial - Personal attestation, witness account
    E1Testimonial = 1,
    /// E2: Privately Verifiable - Audit guild, expert verification
    E2PrivatelyVerifiable = 2,
    /// E3: Cryptographically Proven - ZKP, merkle proofs, on-chain
    E3CryptographicallyProven = 3,
    /// E4: Publicly Reproducible - Open data/code, anyone can verify
    E4PubliclyReproducible = 4,
}

impl EmpiricalAxis {
    pub fn description(&self) -> &'static str {
        match self {
            Self::E0Null => "Unverifiable belief or subjective opinion",
            Self::E1Testimonial => "Personal attestation or witness account",
            Self::E2PrivatelyVerifiable => "Expert verification (audit guild)",
            Self::E3CryptographicallyProven => "Zero-knowledge or cryptographic proof",
            Self::E4PubliclyReproducible => "Open data/code, publicly reproducible",
        }
    }
}

/// N-Axis: Normative Authority (from Epistemic Charter v2.0)
///
/// Who agrees this is binding?
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NormativeAxis {
    /// N0: Personal - Self only, individual preference
    N0Personal = 0,
    /// N1: Communal - Local DAO, working group consensus
    N1Communal = 1,
    /// N2: Network - Global consensus, cross-DAO agreement
    N2Network = 2,
    /// N3: Axiomatic - Constitutional/mathematical, immutable principles
    N3Axiomatic = 3,
}

impl NormativeAxis {
    pub fn description(&self) -> &'static str {
        match self {
            Self::N0Personal => "Self only, individual preference",
            Self::N1Communal => "Local DAO or working group consensus",
            Self::N2Network => "Global network consensus",
            Self::N3Axiomatic => "Constitutional or mathematical axiom",
        }
    }
}

/// M-Axis: Materiality (from Epistemic Charter v2.0)
///
/// How long does this matter? State management and permanence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MaterialityAxis {
    /// M0: Ephemeral - Discard immediately, real-time only
    M0Ephemeral = 0,
    /// M1: Temporal - Prune after state change, session-bound
    M1Temporal = 1,
    /// M2: Persistent - Archive after time, historical record
    M2Persistent = 2,
    /// M3: Foundational - Preserve forever, constitutional
    M3Foundational = 3,
}

impl MaterialityAxis {
    pub fn description(&self) -> &'static str {
        match self {
            Self::M0Ephemeral => "Ephemeral, discard immediately",
            Self::M1Temporal => "Temporal, prune after state change",
            Self::M2Persistent => "Persistent, archive after time",
            Self::M3Foundational => "Foundational, preserve forever",
        }
    }
}

/// LEM Cube: Official 3D epistemic classification from Charter v2.0
///
/// Every claim is positioned in the Layered Epistemic Model:
/// - E-Axis: How is this verified? (E0-E4)
/// - N-Axis: Who agrees it's binding? (N0-N3)
/// - M-Axis: How long does it matter? (M0-M3)
///
/// # Examples
/// - Passed MIP: (E0, N2, M3) - Unverifiable belief, network consensus, permanent
/// - Scientific paper: (E4, N2, M2) - Publicly reproducible, network consensus, persistent
/// - Personal note: (E0, N0, M0) - Null verification, personal, ephemeral
/// - Constitutional amendment: (E0, N3, M3) - Axiomatic, foundational
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LEMCube {
    /// E-Axis: Empirical Verifiability (E0-E4)
    pub empirical: EmpiricalAxis,
    /// N-Axis: Normative Authority (N0-N3)
    pub normative: NormativeAxis,
    /// M-Axis: Materiality (M0-M3)
    pub materiality: MaterialityAxis,
}

impl LEMCube {
    pub fn new(empirical: EmpiricalAxis, normative: NormativeAxis, materiality: MaterialityAxis) -> Self {
        Self { empirical, normative, materiality }
    }

    /// Scientific publication: E4, N2, M2
    pub fn scientific_publication() -> Self {
        Self::new(EmpiricalAxis::E4PubliclyReproducible, NormativeAxis::N2Network, MaterialityAxis::M2Persistent)
    }

    /// Governance proposal (passed MIP): E0, N2, M3
    pub fn governance_decision() -> Self {
        Self::new(EmpiricalAxis::E0Null, NormativeAxis::N2Network, MaterialityAxis::M3Foundational)
    }

    /// Personal opinion: E0, N0, M0
    pub fn personal_opinion() -> Self {
        Self::new(EmpiricalAxis::E0Null, NormativeAxis::N0Personal, MaterialityAxis::M0Ephemeral)
    }

    /// Cryptographic proof: E3, N3, M3
    pub fn cryptographic_proof() -> Self {
        Self::new(EmpiricalAxis::E3CryptographicallyProven, NormativeAxis::N3Axiomatic, MaterialityAxis::M3Foundational)
    }

    /// Expert review: E2, N1, M2
    pub fn expert_review() -> Self {
        Self::new(EmpiricalAxis::E2PrivatelyVerifiable, NormativeAxis::N1Communal, MaterialityAxis::M2Persistent)
    }

    /// Compute trust weight based on LEM position
    /// Higher E + higher N + higher M = more trustworthy for permanent decisions
    pub fn trust_weight(&self) -> f64 {
        let e_weight = (self.empirical as u8) as f64 / 4.0;
        let n_weight = (self.normative as u8) as f64 / 3.0;
        let m_weight = (self.materiality as u8) as f64 / 3.0;
        (e_weight + n_weight + m_weight) / 3.0
    }

    /// Human-readable summary
    pub fn summary(&self) -> String {
        format!("(E{}, N{}, M{})",
            self.empirical as u8,
            self.normative as u8,
            self.materiality as u8)
    }
}

impl Default for LEMCube {
    fn default() -> Self {
        Self::new(EmpiricalAxis::E0Null, NormativeAxis::N0Personal, MaterialityAxis::M0Ephemeral)
    }
}

// ============================================================================
// LAYER 3: QUALITY METRICS (Scientific Rigor)
// ============================================================================

/// Quality metrics for scientific rigor assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Methodology rigor (0.0-1.0): How well-designed is the study?
    pub methodology: f64,
    /// Data quality (0.0-1.0): Completeness, accuracy, provenance
    pub data_quality: f64,
    /// Statistical rigor (0.0-1.0): Appropriate methods, power, effect sizes
    pub statistical_rigor: f64,
    /// Preregistration (0.0-1.0): Was hypothesis/analysis preregistered?
    pub preregistration: f64,
    /// Open science (0.0-1.0): Data/code/materials availability
    pub open_science: f64,
}

impl QualityMetrics {
    pub fn new(methodology: f64, data_quality: f64, statistical_rigor: f64,
               preregistration: f64, open_science: f64) -> Self {
        Self {
            methodology: methodology.clamp(0.0, 1.0),
            data_quality: data_quality.clamp(0.0, 1.0),
            statistical_rigor: statistical_rigor.clamp(0.0, 1.0),
            preregistration: preregistration.clamp(0.0, 1.0),
            open_science: open_science.clamp(0.0, 1.0),
        }
    }

    /// Compute composite quality score
    pub fn composite_score(&self) -> f64 {
        (self.methodology + self.data_quality + self.statistical_rigor
         + self.preregistration + self.open_science) / 5.0
    }

    /// High-quality research template
    pub fn high_quality() -> Self {
        Self::new(0.9, 0.9, 0.9, 0.8, 0.9)
    }

    /// Unknown/unassessed quality
    pub fn unknown() -> Self {
        Self::new(0.5, 0.5, 0.5, 0.0, 0.5)
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self::unknown()
    }
}

// ============================================================================
// LAYER 4: NETWORK POSITION (Claim Relationships)
// ============================================================================

/// Relationship types between claims (from Constitution Claim Schema v2.0)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClaimRelationType {
    /// This claim provides evidence supporting another
    Supports,
    /// This claim provides evidence against another
    Refutes,
    /// This claim replaces/updates a previous claim
    Supercedes,
    /// This claim clarifies or explains another
    Clarifies,
    /// This claim restricts the scope of another
    Restricts,
    /// This claim extends the scope of another
    Extends,
    /// This claim cites another as a source
    Cites,
    /// This claim depends on another being true
    DependsOn,
    /// This claim makes a prediction about another
    Predicts,
    /// This claim and another are in direct conflict
    Conflicts,
}

/// A relationship between two claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimRelation {
    /// ID of the related claim
    pub target_claim_id: Uuid,
    /// Type of relationship
    pub relation_type: ClaimRelationType,
    /// Strength of relationship (0.0-1.0)
    pub strength: f64,
    /// Optional explanation
    pub explanation: Option<String>,
    /// When this relationship was established
    pub established_at: DateTime<Utc>,
}

impl ClaimRelation {
    pub fn new(target_claim_id: Uuid, relation_type: ClaimRelationType, strength: f64) -> Self {
        Self {
            target_claim_id,
            relation_type,
            strength: strength.clamp(0.0, 1.0),
            explanation: None,
            established_at: Utc::now(),
        }
    }

    pub fn with_explanation(mut self, explanation: String) -> Self {
        self.explanation = Some(explanation);
        self
    }
}

/// Network position metrics for a claim
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkPosition {
    /// Claims that this claim relates to
    pub relations: Vec<ClaimRelation>,
    /// Number of supporting claims
    pub support_count: usize,
    /// Number of refuting claims
    pub refutation_count: usize,
    /// Number of citations
    pub citation_count: usize,
    /// Network centrality score (computed from graph analysis)
    pub centrality: Option<f64>,
}

impl NetworkPosition {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_relation(&mut self, relation: ClaimRelation) {
        match relation.relation_type {
            ClaimRelationType::Supports => self.support_count += 1,
            ClaimRelationType::Refutes => self.refutation_count += 1,
            ClaimRelationType::Cites => self.citation_count += 1,
            _ => {}
        }
        self.relations.push(relation);
    }

    /// Net support score (supports - refutations, normalized)
    pub fn net_support(&self) -> f64 {
        let total = (self.support_count + self.refutation_count) as f64;
        if total == 0.0 {
            return 0.5; // Neutral when no evidence either way
        }
        self.support_count as f64 / total
    }
}

// ============================================================================
// MATL INTEGRATION (Mycelix Adaptive Trust Layer)
// ============================================================================

/// MATL Trust Score - Reputation-weighted verification
///
/// From Economic Charter: Trust = PoGQ + TCDM + Entropy components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MATLTrust {
    /// Proof-of-Genuine-Query score (0.0-1.0)
    pub pogq_score: f64,
    /// Trust Composition and Decay Metric (0.0-1.0)
    pub tcdm_score: f64,
    /// Entropy component (diversity of verifiers) (0.0-1.0)
    pub entropy_score: f64,
    /// Reputation-weighted verification count
    pub weighted_verifications: f64,
    /// Raw verification count (unweighted)
    pub raw_verifications: usize,
    /// Last MATL update timestamp
    pub last_updated: DateTime<Utc>,
}

impl MATLTrust {
    pub fn new() -> Self {
        Self {
            pogq_score: 0.0,
            tcdm_score: 0.0,
            entropy_score: 0.0,
            weighted_verifications: 0.0,
            raw_verifications: 0,
            last_updated: Utc::now(),
        }
    }

    /// Compute composite trust score (MATL formula)
    pub fn composite_trust(&self) -> f64 {
        // Weighted combination: PoGQ (40%) + TCDM (35%) + Entropy (25%)
        0.40 * self.pogq_score + 0.35 * self.tcdm_score + 0.25 * self.entropy_score
    }

    /// Add a verification with reputation weight
    pub fn add_verification(&mut self, verifier_reputation: f64) {
        self.raw_verifications += 1;
        self.weighted_verifications += verifier_reputation;
        self.last_updated = Utc::now();
    }

    /// Determine E-Tier from weighted verifications (reputation-adjusted)
    pub fn derived_tier(&self) -> EpistemicTier {
        match self.weighted_verifications as usize {
            0 => EpistemicTier::E0,
            1 => EpistemicTier::E1,
            2..=4 => EpistemicTier::E2,
            5..=9 => EpistemicTier::E3,
            _ => EpistemicTier::E4,
        }
    }
}

impl Default for MATLTrust {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LAYER 2: CLAIM TYPE POSITION (DeSci Extension)
// ============================================================================

/// Three-dimensional epistemic position in the E-N-M cube
///
/// Every claim is positioned in a 3D space measuring:
/// - **Empirical (E)**: How verifiable through observation/experiment (0.0-1.0)
/// - **Normative (N)**: How aligned with ethical/value frameworks (0.0-1.0)
/// - **Mythic (M)**: What narrative/meaning significance it holds (0.0-1.0)
///
/// # Examples
/// - Scientific fact: High E (0.9), Low N (0.1), Low M (0.1)
/// - Moral principle: Low E (0.2), High N (0.9), Variable M (0.5)
/// - Origin story: Low E (0.1), Variable N (0.4), High M (0.9)
/// - Historical event: High E (0.8), Variable N (0.5), High M (0.8)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EpistemicPosition {
    /// Empirical dimension: How verifiable through observation? (0.0-1.0)
    pub empirical: f64,
    /// Normative dimension: How aligned with ethical frameworks? (0.0-1.0)
    pub normative: f64,
    /// Mythic dimension: What narrative/meaning significance? (0.0-1.0)
    pub mythic: f64,
}

impl EpistemicPosition {
    /// Create a new epistemic position
    pub fn new(empirical: f64, normative: f64, mythic: f64) -> Self {
        Self {
            empirical: empirical.clamp(0.0, 1.0),
            normative: normative.clamp(0.0, 1.0),
            mythic: mythic.clamp(0.0, 1.0),
        }
    }

    /// Create position for a scientific/empirical claim
    pub fn scientific(empirical: f64) -> Self {
        Self::new(empirical, 0.1, 0.1)
    }

    /// Create position for an ethical/normative claim
    pub fn ethical(normative: f64) -> Self {
        Self::new(0.2, normative, 0.3)
    }

    /// Create position for a narrative/mythic claim
    pub fn narrative(mythic: f64) -> Self {
        Self::new(0.1, 0.3, mythic)
    }

    /// Compute Euclidean distance to another position
    pub fn distance(&self, other: &EpistemicPosition) -> f64 {
        let de = self.empirical - other.empirical;
        let dn = self.normative - other.normative;
        let dm = self.mythic - other.mythic;
        (de * de + dn * dn + dm * dm).sqrt()
    }

    /// Get the dominant dimension
    pub fn dominant_dimension(&self) -> &'static str {
        if self.empirical >= self.normative && self.empirical >= self.mythic {
            "empirical"
        } else if self.normative >= self.empirical && self.normative >= self.mythic {
            "normative"
        } else {
            "mythic"
        }
    }

    /// Check if this is a balanced claim (all dimensions within 0.3 of each other)
    pub fn is_balanced(&self) -> bool {
        let max = self.empirical.max(self.normative).max(self.mythic);
        let min = self.empirical.min(self.normative).min(self.mythic);
        max - min <= 0.3
    }
}

impl Default for EpistemicPosition {
    fn default() -> Self {
        Self::new(0.5, 0.5, 0.5)
    }
}

/// Epistemic tier classification (E0-E4)
///
/// Based on Mycelix epistemic framework:
/// - E0: Unverified claim
/// - E1: Single-source verification
/// - E2: Multi-source verification
/// - E3: Reproducible with documented methodology
/// - E4: Peer-reviewed and independently reproduced
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum EpistemicTier {
    E0 = 0,
    E1 = 1,
    E2 = 2,
    E3 = 3,
    E4 = 4,
}

impl EpistemicTier {
    /// Returns a human-readable description of the tier
    pub fn description(&self) -> &'static str {
        match self {
            EpistemicTier::E0 => "Unverified claim",
            EpistemicTier::E1 => "Single-source verification",
            EpistemicTier::E2 => "Multi-source verification",
            EpistemicTier::E3 => "Reproducible with documented methodology",
            EpistemicTier::E4 => "Peer-reviewed and independently reproduced",
        }
    }

    /// Minimum number of verifications required for this tier
    pub fn min_verifications(&self) -> usize {
        match self {
            EpistemicTier::E0 => 0,
            EpistemicTier::E1 => 1,
            EpistemicTier::E2 => 2,
            EpistemicTier::E3 => 3,
            EpistemicTier::E4 => 5,
        }
    }
}

/// Provenance information for a claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Source identifier (e.g., "PubChem ID:123", "DOI:10.1234/example")
    pub source: String,

    /// Type of source (e.g., "database", "publication", "repository")
    pub source_type: String,

    /// URL or URI for accessing the source
    pub url: Option<String>,

    /// Timestamp when the claim was created from this source
    pub timestamp: DateTime<Utc>,

    /// Additional metadata
    pub metadata: serde_json::Value,
}

impl Provenance {
    /// Create a new provenance entry
    pub fn new(source: String, source_type: String) -> Self {
        Self {
            source,
            source_type,
            url: None,
            timestamp: Utc::now(),
            metadata: serde_json::json!({}),
        }
    }

    /// Set the URL for this provenance
    pub fn with_url(mut self, url: String) -> Self {
        self.url = Some(url);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        if let Some(obj) = self.metadata.as_object_mut() {
            obj.insert(key.to_string(), value);
        }
        self
    }
}

/// Content of a DeSci claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimContent {
    /// Hash of the dataset or content (e.g., SHA-256)
    pub dataset_hash: String,

    /// Description of the claim
    pub description: String,

    /// Category (e.g., "genomics", "longevity", "climate")
    pub category: String,

    /// Keywords for searchability
    pub keywords: Vec<String>,

    /// IPFS CID or other storage reference
    pub storage_ref: Option<String>,

    /// Reproducibility score (0.0 - 1.0)
    pub reproducibility_score: Option<f64>,

    /// License (e.g., "CC-BY-4.0", "MIT")
    pub license: Option<String>,
}

// ============================================================================
// EPISTEMIC FINGERPRINT (Unified Multi-Layer Assessment)
// ============================================================================

/// Complete epistemic assessment combining all layers
///
/// The Epistemic Fingerprint provides a unified view of a claim's
/// epistemic status across all four layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicFingerprint {
    /// Layer 1: Official LEM Cube (Charter v2.0)
    pub lem_cube: LEMCube,
    /// Layer 2: Claim type position (DeSci E-N-M)
    pub type_position: EpistemicPosition,
    /// Layer 3: Quality metrics
    pub quality: QualityMetrics,
    /// Layer 4: Network position
    pub network: NetworkPosition,
    /// MATL trust score
    pub matl_trust: MATLTrust,
    /// Legacy E0-E4 tier (for backwards compatibility)
    pub legacy_tier: EpistemicTier,
}

impl EpistemicFingerprint {
    pub fn new(lem_cube: LEMCube, type_position: EpistemicPosition) -> Self {
        Self {
            lem_cube,
            type_position,
            quality: QualityMetrics::default(),
            network: NetworkPosition::default(),
            matl_trust: MATLTrust::default(),
            legacy_tier: EpistemicTier::E0,
        }
    }

    /// Compute unified confidence score (0.0-1.0)
    ///
    /// Combines all layers with appropriate weights:
    /// - LEM Cube trust weight: 25%
    /// - Type position (dominant dimension strength): 10%
    /// - Quality composite score: 25%
    /// - Network net support: 20%
    /// - MATL composite trust: 20%
    pub fn confidence_score(&self) -> f64 {
        let lem_weight = self.lem_cube.trust_weight();
        let type_weight = self.type_position.empirical.max(self.type_position.normative).max(self.type_position.mythic);
        let quality_weight = self.quality.composite_score();
        let network_weight = self.network.net_support();
        let matl_weight = self.matl_trust.composite_trust();

        0.25 * lem_weight + 0.10 * type_weight + 0.25 * quality_weight
            + 0.20 * network_weight + 0.20 * matl_weight
    }

    /// Human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "LEM{} | Type:{} | Quality:{:.0}% | Support:{:.0}% | MATL:{:.0}% | Confidence:{:.0}%",
            self.lem_cube.summary(),
            self.type_position.dominant_dimension(),
            self.quality.composite_score() * 100.0,
            self.network.net_support() * 100.0,
            self.matl_trust.composite_trust() * 100.0,
            self.confidence_score() * 100.0,
        )
    }
}

impl Default for EpistemicFingerprint {
    fn default() -> Self {
        Self::new(LEMCube::default(), EpistemicPosition::default())
    }
}

/// A DeSci claim with full epistemic tensor classification
///
/// Implements the Charter-aligned multi-layer epistemic model:
/// - **Layer 1 (LEM Cube)**: Official governance classification (E/N/M axes)
/// - **Layer 2 (Type Position)**: Claim nature (empirical/normative/mythic)
/// - **Layer 3 (Quality)**: Scientific rigor metrics
/// - **Layer 4 (Network)**: Claim relationships and citations
/// - **MATL Integration**: Reputation-weighted trust scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesciClaim {
    /// Unique identifier for the claim
    pub id: Uuid,

    /// Complete epistemic fingerprint (all layers)
    pub fingerprint: EpistemicFingerprint,

    /// Legacy: Epistemic tier (E0-E4) - verification-based trust level
    /// Kept for backwards compatibility, derived from MATL trust
    pub epistemic_tier: EpistemicTier,

    /// Legacy: Epistemic position in E-N-M cube - type classification
    /// Aliased to fingerprint.type_position for backwards compatibility
    pub epistemic_position: EpistemicPosition,

    /// Content of the claim
    pub content: ClaimContent,

    /// Provenance chain (multiple sources)
    pub provenance: Vec<Provenance>,

    /// Creator's public key or DID
    pub creator: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,

    /// Verification signatures
    pub verifications: Vec<Verification>,
}

/// A verification signature for a claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verification {
    /// Verifier's public key or DID
    pub verifier: String,

    /// Timestamp of verification
    pub timestamp: DateTime<Utc>,

    /// Signature (cryptographic proof)
    pub signature: Vec<u8>,

    /// Notes from the verifier
    pub notes: Option<String>,
}

impl DesciClaim {
    /// Create a new claim with default epistemic classification
    pub fn new(
        epistemic_tier: EpistemicTier,
        content: ClaimContent,
        creator: String,
    ) -> Self {
        Self::with_position(epistemic_tier, EpistemicPosition::default(), content, creator)
    }

    /// Create a new claim with specific E-N-M type position
    pub fn with_position(
        epistemic_tier: EpistemicTier,
        epistemic_position: EpistemicPosition,
        content: ClaimContent,
        creator: String,
    ) -> Self {
        let now = Utc::now();
        let fingerprint = EpistemicFingerprint {
            lem_cube: LEMCube::default(),
            type_position: epistemic_position,
            quality: QualityMetrics::default(),
            network: NetworkPosition::default(),
            matl_trust: MATLTrust::default(),
            legacy_tier: epistemic_tier,
        };
        Self {
            id: Uuid::new_v4(),
            fingerprint,
            epistemic_tier,
            epistemic_position,
            content,
            provenance: Vec::new(),
            creator,
            created_at: now,
            updated_at: now,
            verifications: Vec::new(),
        }
    }

    /// Create a claim with full fingerprint (all layers specified)
    pub fn with_fingerprint(
        fingerprint: EpistemicFingerprint,
        content: ClaimContent,
        creator: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            epistemic_tier: fingerprint.legacy_tier,
            epistemic_position: fingerprint.type_position,
            fingerprint,
            content,
            provenance: Vec::new(),
            creator,
            created_at: now,
            updated_at: now,
            verifications: Vec::new(),
        }
    }

    /// Create a scientific claim (high empirical, E4 LEM, high quality)
    pub fn scientific(content: ClaimContent, creator: String) -> Self {
        let fingerprint = EpistemicFingerprint {
            lem_cube: LEMCube::scientific_publication(),
            type_position: EpistemicPosition::scientific(0.9),
            quality: QualityMetrics::high_quality(),
            network: NetworkPosition::default(),
            matl_trust: MATLTrust::default(),
            legacy_tier: EpistemicTier::E0,
        };
        Self::with_fingerprint(fingerprint, content, creator)
    }

    /// Create an ethical claim (high normative, governance LEM)
    pub fn ethical(content: ClaimContent, creator: String) -> Self {
        let fingerprint = EpistemicFingerprint {
            lem_cube: LEMCube::governance_decision(),
            type_position: EpistemicPosition::ethical(0.9),
            quality: QualityMetrics::default(),
            network: NetworkPosition::default(),
            matl_trust: MATLTrust::default(),
            legacy_tier: EpistemicTier::E0,
        };
        Self::with_fingerprint(fingerprint, content, creator)
    }

    /// Create a narrative claim (high mythic, persistent LEM)
    pub fn narrative(content: ClaimContent, creator: String) -> Self {
        let fingerprint = EpistemicFingerprint {
            lem_cube: LEMCube::new(
                EmpiricalAxis::E1Testimonial,
                NormativeAxis::N1Communal,
                MaterialityAxis::M2Persistent,
            ),
            type_position: EpistemicPosition::narrative(0.9),
            quality: QualityMetrics::default(),
            network: NetworkPosition::default(),
            matl_trust: MATLTrust::default(),
            legacy_tier: EpistemicTier::E0,
        };
        Self::with_fingerprint(fingerprint, content, creator)
    }

    /// Create a cryptographic proof claim (E3 LEM, axiomatic)
    pub fn cryptographic_proof(content: ClaimContent, creator: String) -> Self {
        let fingerprint = EpistemicFingerprint {
            lem_cube: LEMCube::cryptographic_proof(),
            type_position: EpistemicPosition::scientific(0.95),
            quality: QualityMetrics::new(1.0, 1.0, 1.0, 1.0, 1.0), // Perfect for crypto
            network: NetworkPosition::default(),
            matl_trust: MATLTrust::default(),
            legacy_tier: EpistemicTier::E3,
        };
        Self::with_fingerprint(fingerprint, content, creator)
    }

    /// Get the unified confidence score
    pub fn confidence(&self) -> f64 {
        self.fingerprint.confidence_score()
    }

    /// Get human-readable epistemic summary
    pub fn epistemic_summary(&self) -> String {
        self.fingerprint.summary()
    }

    /// Add a relationship to another claim
    pub fn add_relation(&mut self, target_id: Uuid, relation_type: ClaimRelationType, strength: f64) {
        let relation = ClaimRelation::new(target_id, relation_type, strength);
        self.fingerprint.network.add_relation(relation);
        self.updated_at = Utc::now();
    }

    /// Update quality metrics
    pub fn set_quality(&mut self, quality: QualityMetrics) {
        self.fingerprint.quality = quality;
        self.updated_at = Utc::now();
    }

    /// Update LEM cube classification
    pub fn set_lem(&mut self, lem: LEMCube) {
        self.fingerprint.lem_cube = lem;
        self.updated_at = Utc::now();
    }

    /// Add provenance to the claim
    pub fn add_provenance(&mut self, prov: Provenance) {
        self.provenance.push(prov);
        self.updated_at = Utc::now();
    }

    /// Add a verification with optional verifier reputation
    pub fn add_verification(&mut self, verification: Verification) {
        self.add_verification_with_reputation(verification, 1.0);
    }

    /// Add a verification with explicit verifier reputation weight
    ///
    /// This enables MATL-weighted verification where high-reputation
    /// verifiers contribute more to the trust score.
    pub fn add_verification_with_reputation(&mut self, verification: Verification, reputation: f64) {
        self.verifications.push(verification);
        self.fingerprint.matl_trust.add_verification(reputation);
        self.updated_at = Utc::now();

        // Update tier based on MATL-weighted verifications
        self.update_tier_from_matl();
    }

    /// Update epistemic tier based on MATL-weighted verifications
    fn update_tier_from_matl(&mut self) {
        let potential_tier = self.fingerprint.matl_trust.derived_tier();

        // Only upgrade, never downgrade automatically
        if potential_tier > self.epistemic_tier {
            self.epistemic_tier = potential_tier;
            self.fingerprint.legacy_tier = potential_tier;
        }
    }

    /// Update epistemic tier based on number of verifications (legacy method)
    fn update_tier_from_verifications(&mut self) {
        let verification_count = self.verifications.len();

        // Determine the highest tier we can achieve
        let potential_tier = match verification_count {
            0 => EpistemicTier::E0,
            1 => EpistemicTier::E1,
            2..=4 => EpistemicTier::E2,
            5..=9 => EpistemicTier::E3,
            _ => EpistemicTier::E4,
        };

        // Only upgrade, never downgrade automatically
        if potential_tier > self.epistemic_tier {
            self.epistemic_tier = potential_tier;
        }
    }

    /// Check if the claim meets the minimum requirements for its tier
    pub fn is_valid_for_tier(&self) -> bool {
        let min_verifications = self.epistemic_tier.min_verifications();
        self.verifications.len() >= min_verifications
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_tier_ordering() {
        assert!(EpistemicTier::E4 > EpistemicTier::E0);
        assert!(EpistemicTier::E3 > EpistemicTier::E2);
    }

    #[test]
    fn test_create_claim() {
        let content = ClaimContent {
            dataset_hash: "abc123".to_string(),
            description: "Test dataset".to_string(),
            category: "genomics".to_string(),
            keywords: vec!["test".to_string()],
            storage_ref: None,
            reproducibility_score: Some(0.95),
            license: Some("MIT".to_string()),
        };

        let claim = DesciClaim::new(
            EpistemicTier::E0,
            content,
            "creator_pubkey".to_string(),
        );

        assert_eq!(claim.epistemic_tier, EpistemicTier::E0);
        assert_eq!(claim.verifications.len(), 0);
    }

    #[test]
    fn test_provenance() {
        let prov = Provenance::new(
            "PubChem ID:123".to_string(),
            "database".to_string(),
        )
        .with_url("https://pubchem.ncbi.nlm.nih.gov/compound/123".to_string());

        assert_eq!(prov.source, "PubChem ID:123");
        assert!(prov.url.is_some());
    }

    #[test]
    fn test_epistemic_position_new() {
        let pos = EpistemicPosition::new(0.9, 0.5, 0.2);
        assert_eq!(pos.empirical, 0.9);
        assert_eq!(pos.normative, 0.5);
        assert_eq!(pos.mythic, 0.2);
    }

    #[test]
    fn test_epistemic_position_clamping() {
        let pos = EpistemicPosition::new(1.5, -0.3, 0.5);
        assert_eq!(pos.empirical, 1.0);
        assert_eq!(pos.normative, 0.0);
        assert_eq!(pos.mythic, 0.5);
    }

    #[test]
    fn test_epistemic_position_scientific() {
        let pos = EpistemicPosition::scientific(0.95);
        assert_eq!(pos.empirical, 0.95);
        assert!(pos.normative < 0.2);
        assert!(pos.mythic < 0.2);
        assert_eq!(pos.dominant_dimension(), "empirical");
    }

    #[test]
    fn test_epistemic_position_ethical() {
        let pos = EpistemicPosition::ethical(0.9);
        assert_eq!(pos.normative, 0.9);
        assert_eq!(pos.dominant_dimension(), "normative");
    }

    #[test]
    fn test_epistemic_position_narrative() {
        let pos = EpistemicPosition::narrative(0.85);
        assert_eq!(pos.mythic, 0.85);
        assert_eq!(pos.dominant_dimension(), "mythic");
    }

    #[test]
    fn test_epistemic_position_distance() {
        let pos1 = EpistemicPosition::new(0.0, 0.0, 0.0);
        let pos2 = EpistemicPosition::new(1.0, 0.0, 0.0);
        assert!((pos1.distance(&pos2) - 1.0).abs() < 0.001);

        let pos3 = EpistemicPosition::new(1.0, 1.0, 1.0);
        let expected_distance = (3.0_f64).sqrt();
        assert!((pos1.distance(&pos3) - expected_distance).abs() < 0.001);
    }

    #[test]
    fn test_epistemic_position_balanced() {
        let balanced = EpistemicPosition::new(0.5, 0.6, 0.7);
        assert!(balanced.is_balanced());

        let unbalanced = EpistemicPosition::new(0.1, 0.5, 0.9);
        assert!(!unbalanced.is_balanced());
    }

    #[test]
    fn test_create_scientific_claim() {
        let content = ClaimContent {
            dataset_hash: "hash123".to_string(),
            description: "Research finding".to_string(),
            category: "biology".to_string(),
            keywords: vec!["research".to_string()],
            storage_ref: None,
            reproducibility_score: Some(0.9),
            license: Some("CC-BY-4.0".to_string()),
        };

        let claim = DesciClaim::scientific(content, "researcher@uni.edu".to_string());

        assert_eq!(claim.epistemic_tier, EpistemicTier::E0);
        assert_eq!(claim.epistemic_position.dominant_dimension(), "empirical");
        assert!(claim.epistemic_position.empirical > 0.8);
    }

    #[test]
    fn test_create_claim_with_position() {
        let content = ClaimContent {
            dataset_hash: "hash456".to_string(),
            description: "Mixed claim".to_string(),
            category: "philosophy".to_string(),
            keywords: vec!["ethics".to_string()],
            storage_ref: None,
            reproducibility_score: None,
            license: None,
        };

        let position = EpistemicPosition::new(0.3, 0.8, 0.4);
        let claim = DesciClaim::with_position(
            EpistemicTier::E1,
            position,
            content,
            "philosopher@uni.edu".to_string(),
        );

        assert_eq!(claim.epistemic_tier, EpistemicTier::E1);
        assert_eq!(claim.epistemic_position.normative, 0.8);
        assert_eq!(claim.epistemic_position.dominant_dimension(), "normative");
    }

    // =========================================================================
    // Layer 1: LEM Cube Tests (Charter v2.0)
    // =========================================================================

    #[test]
    fn test_lem_cube_scientific_publication() {
        let lem = LEMCube::scientific_publication();
        assert_eq!(lem.empirical, EmpiricalAxis::E4PubliclyReproducible);
        assert_eq!(lem.normative, NormativeAxis::N2Network);
        assert_eq!(lem.materiality, MaterialityAxis::M2Persistent);
        assert_eq!(lem.summary(), "(E4, N2, M2)");
    }

    #[test]
    fn test_lem_cube_governance_decision() {
        let lem = LEMCube::governance_decision();
        assert_eq!(lem.empirical, EmpiricalAxis::E0Null);
        assert_eq!(lem.normative, NormativeAxis::N2Network);
        assert_eq!(lem.materiality, MaterialityAxis::M3Foundational);
    }

    #[test]
    fn test_lem_cube_cryptographic_proof() {
        let lem = LEMCube::cryptographic_proof();
        assert_eq!(lem.empirical, EmpiricalAxis::E3CryptographicallyProven);
        assert_eq!(lem.normative, NormativeAxis::N3Axiomatic);
        assert_eq!(lem.materiality, MaterialityAxis::M3Foundational);
    }

    #[test]
    fn test_lem_cube_trust_weight() {
        let low_trust = LEMCube::personal_opinion();
        let high_trust = LEMCube::cryptographic_proof();

        assert!(high_trust.trust_weight() > low_trust.trust_weight());
        assert!(low_trust.trust_weight() < 0.5);
        assert!(high_trust.trust_weight() > 0.7);
    }

    #[test]
    fn test_empirical_axis_ordering() {
        assert!(EmpiricalAxis::E4PubliclyReproducible > EmpiricalAxis::E0Null);
        assert!(EmpiricalAxis::E3CryptographicallyProven > EmpiricalAxis::E2PrivatelyVerifiable);
    }

    #[test]
    fn test_normative_axis_descriptions() {
        assert!(NormativeAxis::N0Personal.description().contains("Self"));
        assert!(NormativeAxis::N3Axiomatic.description().contains("axiom"));
    }

    #[test]
    fn test_materiality_axis_ordering() {
        assert!(MaterialityAxis::M3Foundational > MaterialityAxis::M0Ephemeral);
    }

    // =========================================================================
    // Layer 3: Quality Metrics Tests
    // =========================================================================

    #[test]
    fn test_quality_metrics_composite() {
        let quality = QualityMetrics::new(0.8, 0.8, 0.8, 0.8, 0.8);
        assert!((quality.composite_score() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_quality_metrics_high_quality() {
        let quality = QualityMetrics::high_quality();
        assert!(quality.composite_score() > 0.8);
    }

    #[test]
    fn test_quality_metrics_clamping() {
        let quality = QualityMetrics::new(1.5, -0.5, 0.5, 0.5, 0.5);
        assert_eq!(quality.methodology, 1.0);
        assert_eq!(quality.data_quality, 0.0);
    }

    // =========================================================================
    // Layer 4: Network Position Tests
    // =========================================================================

    #[test]
    fn test_claim_relation_types() {
        let target_id = Uuid::new_v4();
        let relation = ClaimRelation::new(target_id, ClaimRelationType::Supports, 0.9);

        assert_eq!(relation.target_claim_id, target_id);
        assert_eq!(relation.relation_type, ClaimRelationType::Supports);
        assert_eq!(relation.strength, 0.9);
    }

    #[test]
    fn test_network_position_tracking() {
        let mut network = NetworkPosition::new();
        let target_id = Uuid::new_v4();

        network.add_relation(ClaimRelation::new(target_id, ClaimRelationType::Supports, 0.8));
        network.add_relation(ClaimRelation::new(Uuid::new_v4(), ClaimRelationType::Cites, 1.0));

        assert_eq!(network.support_count, 1);
        assert_eq!(network.citation_count, 1);
        assert_eq!(network.relations.len(), 2);
    }

    #[test]
    fn test_network_position_net_support() {
        let mut network = NetworkPosition::new();

        // Equal support and refutation = 50%
        network.add_relation(ClaimRelation::new(Uuid::new_v4(), ClaimRelationType::Supports, 1.0));
        network.add_relation(ClaimRelation::new(Uuid::new_v4(), ClaimRelationType::Refutes, 1.0));

        assert!((network.net_support() - 0.5).abs() < 0.001);

        // 2 supports, 1 refute = 66%
        network.add_relation(ClaimRelation::new(Uuid::new_v4(), ClaimRelationType::Supports, 1.0));
        assert!((network.net_support() - 0.666).abs() < 0.01);
    }

    // =========================================================================
    // MATL Trust Tests
    // =========================================================================

    #[test]
    fn test_matl_trust_default() {
        let trust = MATLTrust::new();
        assert_eq!(trust.raw_verifications, 0);
        assert_eq!(trust.weighted_verifications, 0.0);
    }

    #[test]
    fn test_matl_trust_add_verification() {
        let mut trust = MATLTrust::new();

        // Add verification from high-reputation verifier
        trust.add_verification(2.0);
        assert_eq!(trust.raw_verifications, 1);
        assert_eq!(trust.weighted_verifications, 2.0);

        // Add verification from low-reputation verifier
        trust.add_verification(0.5);
        assert_eq!(trust.raw_verifications, 2);
        assert_eq!(trust.weighted_verifications, 2.5);
    }

    #[test]
    fn test_matl_trust_derived_tier() {
        let mut trust = MATLTrust::new();
        assert_eq!(trust.derived_tier(), EpistemicTier::E0);

        trust.add_verification(1.0);
        assert_eq!(trust.derived_tier(), EpistemicTier::E1);

        trust.add_verification(1.0);
        assert_eq!(trust.derived_tier(), EpistemicTier::E2);
    }

    #[test]
    fn test_matl_composite_trust() {
        let mut trust = MATLTrust::new();
        trust.pogq_score = 0.8;
        trust.tcdm_score = 0.6;
        trust.entropy_score = 0.4;

        // 0.4*0.8 + 0.35*0.6 + 0.25*0.4 = 0.32 + 0.21 + 0.10 = 0.63
        assert!((trust.composite_trust() - 0.63).abs() < 0.001);
    }

    // =========================================================================
    // Epistemic Fingerprint Tests
    // =========================================================================

    #[test]
    fn test_epistemic_fingerprint_default() {
        let fingerprint = EpistemicFingerprint::default();
        assert_eq!(fingerprint.lem_cube.empirical, EmpiricalAxis::E0Null);
        assert_eq!(fingerprint.legacy_tier, EpistemicTier::E0);
    }

    #[test]
    fn test_epistemic_fingerprint_confidence() {
        let fingerprint = EpistemicFingerprint {
            lem_cube: LEMCube::scientific_publication(),
            type_position: EpistemicPosition::scientific(0.9),
            quality: QualityMetrics::high_quality(),
            network: NetworkPosition::default(),
            matl_trust: MATLTrust::default(),
            legacy_tier: EpistemicTier::E0,
        };

        // Should have high confidence due to good LEM and quality
        assert!(fingerprint.confidence_score() > 0.4);
    }

    #[test]
    fn test_epistemic_fingerprint_summary() {
        let fingerprint = EpistemicFingerprint::new(
            LEMCube::scientific_publication(),
            EpistemicPosition::scientific(0.9),
        );

        let summary = fingerprint.summary();
        assert!(summary.contains("LEM(E4, N2, M2)"));
        assert!(summary.contains("Type:empirical"));
    }

    // =========================================================================
    // Full Claim Integration Tests
    // =========================================================================

    #[test]
    fn test_claim_with_fingerprint() {
        let fingerprint = EpistemicFingerprint {
            lem_cube: LEMCube::scientific_publication(),
            type_position: EpistemicPosition::scientific(0.9),
            quality: QualityMetrics::high_quality(),
            network: NetworkPosition::default(),
            matl_trust: MATLTrust::default(),
            legacy_tier: EpistemicTier::E0,
        };

        let content = ClaimContent {
            dataset_hash: "hash789".to_string(),
            description: "Full fingerprint claim".to_string(),
            category: "test".to_string(),
            keywords: vec![],
            storage_ref: None,
            reproducibility_score: None,
            license: None,
        };

        let claim = DesciClaim::with_fingerprint(fingerprint, content, "test@test.com".to_string());

        assert_eq!(claim.fingerprint.lem_cube.empirical, EmpiricalAxis::E4PubliclyReproducible);
        assert!(claim.confidence() > 0.4);
    }

    #[test]
    fn test_claim_add_relation() {
        let content = ClaimContent {
            dataset_hash: "hash".to_string(),
            description: "Test".to_string(),
            category: "test".to_string(),
            keywords: vec![],
            storage_ref: None,
            reproducibility_score: None,
            license: None,
        };

        let mut claim = DesciClaim::new(EpistemicTier::E0, content, "test".to_string());
        let target_id = Uuid::new_v4();

        claim.add_relation(target_id, ClaimRelationType::Supports, 0.9);

        assert_eq!(claim.fingerprint.network.support_count, 1);
        assert_eq!(claim.fingerprint.network.relations.len(), 1);
    }

    #[test]
    fn test_claim_matl_verification() {
        let content = ClaimContent {
            dataset_hash: "hash".to_string(),
            description: "Test".to_string(),
            category: "test".to_string(),
            keywords: vec![],
            storage_ref: None,
            reproducibility_score: None,
            license: None,
        };

        let mut claim = DesciClaim::new(EpistemicTier::E0, content, "test".to_string());

        // Add verification from high-reputation verifier
        let verification = Verification {
            verifier: "expert@uni.edu".to_string(),
            timestamp: Utc::now(),
            signature: vec![],
            notes: Some("Verified".to_string()),
        };
        claim.add_verification_with_reputation(verification, 2.0);

        assert_eq!(claim.fingerprint.matl_trust.raw_verifications, 1);
        assert_eq!(claim.fingerprint.matl_trust.weighted_verifications, 2.0);
        assert_eq!(claim.epistemic_tier, EpistemicTier::E2); // 2.0 weighted = E2
    }

    #[test]
    fn test_cryptographic_proof_claim() {
        let content = ClaimContent {
            dataset_hash: "merkle_root_hash".to_string(),
            description: "ZK proof of computation".to_string(),
            category: "cryptography".to_string(),
            keywords: vec!["zkp".to_string()],
            storage_ref: None,
            reproducibility_score: Some(1.0),
            license: Some("MIT".to_string()),
        };

        let claim = DesciClaim::cryptographic_proof(content, "prover@chain.eth".to_string());

        assert_eq!(claim.fingerprint.lem_cube.empirical, EmpiricalAxis::E3CryptographicallyProven);
        assert_eq!(claim.fingerprint.lem_cube.normative, NormativeAxis::N3Axiomatic);
        assert_eq!(claim.fingerprint.quality.methodology, 1.0);
        assert_eq!(claim.epistemic_tier, EpistemicTier::E3);
    }
}
