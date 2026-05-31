// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Claims Integrity Zome
//! Defines entry types and validation for epistemic claims
//!
//! Enhanced with E/N/M/H discrete classification per GIS v4.0
//! - E: Empirical levels (E0-E4)
//! - N: Normative levels (N0-N3)
//! - M: Materiality levels (M0-M3)
//! - H: Harmonic impact dimension

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// E/N/M/H DISCRETE CLASSIFICATION SYSTEM
// ============================================================================

/// Empirical Level (E0-E4)
/// Measures the degree of empirical validation a claim has received
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EmpiricalLevel {
    /// E0: Unverified - No empirical testing
    E0,
    /// E1: Preliminary - Initial observations or anecdotal evidence
    E1,
    /// E2: Tested - Systematic testing with limited sample
    E2,
    /// E3: Replicated - Multiple independent replications
    E3,
    /// E4: Established - Robust empirical consensus
    E4,
}

impl EmpiricalLevel {
    /// Convert to continuous value (0.0-1.0)
    pub fn to_continuous(&self) -> f64 {
        match self {
            EmpiricalLevel::E0 => 0.0,
            EmpiricalLevel::E1 => 0.25,
            EmpiricalLevel::E2 => 0.5,
            EmpiricalLevel::E3 => 0.75,
            EmpiricalLevel::E4 => 1.0,
        }
    }

    /// Convert from continuous value
    pub fn from_continuous(value: f64) -> Self {
        match value {
            v if v < 0.125 => EmpiricalLevel::E0,
            v if v < 0.375 => EmpiricalLevel::E1,
            v if v < 0.625 => EmpiricalLevel::E2,
            v if v < 0.875 => EmpiricalLevel::E3,
            _ => EmpiricalLevel::E4,
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            EmpiricalLevel::E0 => "Unverified - No empirical testing",
            EmpiricalLevel::E1 => "Preliminary - Initial observations",
            EmpiricalLevel::E2 => "Tested - Systematic testing",
            EmpiricalLevel::E3 => "Replicated - Multiple replications",
            EmpiricalLevel::E4 => "Established - Robust consensus",
        }
    }
}

impl Default for EmpiricalLevel {
    fn default() -> Self {
        EmpiricalLevel::E0
    }
}

/// Normative Level (N0-N3)
/// Measures the normative/ethical coherence and endorsement
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NormativeLevel {
    /// N0: Raw - No normative evaluation
    N0,
    /// N1: Contested - Significant disagreement
    N1,
    /// N2: Emerging - Growing consensus
    N2,
    /// N3: Endorsed - Broad normative endorsement
    N3,
}

impl NormativeLevel {
    /// Convert to continuous value (0.0-1.0)
    pub fn to_continuous(&self) -> f64 {
        match self {
            NormativeLevel::N0 => 0.0,
            NormativeLevel::N1 => 0.33,
            NormativeLevel::N2 => 0.67,
            NormativeLevel::N3 => 1.0,
        }
    }

    /// Convert from continuous value
    pub fn from_continuous(value: f64) -> Self {
        match value {
            v if v < 0.17 => NormativeLevel::N0,
            v if v < 0.5 => NormativeLevel::N1,
            v if v < 0.83 => NormativeLevel::N2,
            _ => NormativeLevel::N3,
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            NormativeLevel::N0 => "Raw - No normative evaluation",
            NormativeLevel::N1 => "Contested - Significant disagreement",
            NormativeLevel::N2 => "Emerging - Growing consensus",
            NormativeLevel::N3 => "Endorsed - Broad normative endorsement",
        }
    }
}

impl Default for NormativeLevel {
    fn default() -> Self {
        NormativeLevel::N0
    }
}

/// Materiality Level (M0-M3)
/// Measures the practical/material significance and applicability
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MaterialityLevel {
    /// M0: Abstract - Purely theoretical, no material implications
    M0,
    /// M1: Potential - Theoretical material implications
    M1,
    /// M2: Applicable - Demonstrated practical applications
    M2,
    /// M3: Transformative - Significant material impact
    M3,
}

impl MaterialityLevel {
    /// Convert to continuous value (0.0-1.0)
    pub fn to_continuous(&self) -> f64 {
        match self {
            MaterialityLevel::M0 => 0.0,
            MaterialityLevel::M1 => 0.33,
            MaterialityLevel::M2 => 0.67,
            MaterialityLevel::M3 => 1.0,
        }
    }

    /// Convert from continuous value
    pub fn from_continuous(value: f64) -> Self {
        match value {
            v if v < 0.17 => MaterialityLevel::M0,
            v if v < 0.5 => MaterialityLevel::M1,
            v if v < 0.83 => MaterialityLevel::M2,
            _ => MaterialityLevel::M3,
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            MaterialityLevel::M0 => "Abstract - Purely theoretical",
            MaterialityLevel::M1 => "Potential - Theoretical implications",
            MaterialityLevel::M2 => "Applicable - Practical applications",
            MaterialityLevel::M3 => "Transformative - Significant impact",
        }
    }
}

impl Default for MaterialityLevel {
    fn default() -> Self {
        MaterialityLevel::M0
    }
}

/// Harmonic Dimension - GIS v4.0 Integration
/// Represents alignment with the 12 Harmonies of the Kosmic Song
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HarmonicImpact {
    /// Primary harmony affected
    pub primary_harmony: Harmony,
    /// Secondary harmonies affected (if any)
    pub secondary_harmonies: Vec<Harmony>,
    /// Resonance strength (0.0-1.0)
    pub resonance: f64,
    /// Whether this claim contributes to harmonic coherence
    pub coherence_contribution: bool,
    /// Alignment with Infinite Love principle
    pub love_alignment: f64,
}

impl Default for HarmonicImpact {
    fn default() -> Self {
        Self {
            primary_harmony: Harmony::IntegralWisdom,
            secondary_harmonies: vec![],
            resonance: 0.5,
            coherence_contribution: true,
            love_alignment: 0.5,
        }
    }
}

/// The 12 Harmonies of the Kosmic Song (GIS v4.0)
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Harmony {
    /// H1: Pan-Sentient Flourishing
    PanSentientFlourishing,
    /// H2: Integral Wisdom
    IntegralWisdom,
    /// H3: Resonant Coherence
    ResonantCoherence,
    /// H4: Emergent Justice
    EmergentJustice,
    /// H5: Creative Expression
    CreativeExpression,
    /// H6: Embodied Presence
    EmbodiedPresence,
    /// H7: Relational Depth
    RelationalDepth,
    /// H8: Ecological Symbiosis
    EcologicalSymbiosis,
    /// H9: Temporal Wisdom
    TemporalWisdom,
    /// H10: Mystery Embracing
    MysteryEmbracing,
    /// H11: Playful Becoming
    PlayfulBecoming,
    /// H12: Unified Diversity
    UnifiedDiversity,
}

impl Harmony {
    /// Get the harmony's full name
    pub fn name(&self) -> &'static str {
        match self {
            Harmony::PanSentientFlourishing => "Pan-Sentient Flourishing",
            Harmony::IntegralWisdom => "Integral Wisdom",
            Harmony::ResonantCoherence => "Resonant Coherence",
            Harmony::EmergentJustice => "Emergent Justice",
            Harmony::CreativeExpression => "Creative Expression",
            Harmony::EmbodiedPresence => "Embodied Presence",
            Harmony::RelationalDepth => "Relational Depth",
            Harmony::EcologicalSymbiosis => "Ecological Symbiosis",
            Harmony::TemporalWisdom => "Temporal Wisdom",
            Harmony::MysteryEmbracing => "Mystery Embracing",
            Harmony::PlayfulBecoming => "Playful Becoming",
            Harmony::UnifiedDiversity => "Unified Diversity",
        }
    }
}

/// Full E/N/M/H Epistemic Classification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpistemicClassification {
    /// Empirical level (E0-E4)
    pub empirical: EmpiricalLevel,
    /// Normative level (N0-N3)
    pub normative: NormativeLevel,
    /// Materiality level (M0-M3)
    pub materiality: MaterialityLevel,
    /// Harmonic impact (H-dimension for GIS v4)
    pub harmonic: Option<HarmonicImpact>,
    /// Classification confidence (0.0-1.0)
    pub confidence: f64,
    /// Who classified this claim
    pub classified_by: Option<String>,
    /// When it was classified
    pub classified_at: Option<Timestamp>,
}

impl Default for EpistemicClassification {
    fn default() -> Self {
        Self {
            empirical: EmpiricalLevel::E0,
            normative: NormativeLevel::N0,
            materiality: MaterialityLevel::M0,
            harmonic: None,
            confidence: 0.5,
            classified_by: None,
            classified_at: None,
        }
    }
}

impl EpistemicClassification {
    /// Create from discrete levels
    pub fn new(e: EmpiricalLevel, n: NormativeLevel, m: MaterialityLevel) -> Self {
        Self {
            empirical: e,
            normative: n,
            materiality: m,
            harmonic: None,
            confidence: 0.5,
            classified_by: None,
            classified_at: None,
        }
    }

    /// Create with harmonic impact
    pub fn with_harmonic(
        e: EmpiricalLevel,
        n: NormativeLevel,
        m: MaterialityLevel,
        h: HarmonicImpact,
    ) -> Self {
        Self {
            empirical: e,
            normative: n,
            materiality: m,
            harmonic: Some(h),
            confidence: 0.5,
            classified_by: None,
            classified_at: None,
        }
    }

    /// Get a short code representation (e.g., "E2N1M2")
    pub fn code(&self) -> String {
        format!(
            "E{}N{}M{}",
            self.empirical as u8, self.normative as u8, self.materiality as u8
        )
    }

    /// Calculate overall epistemic strength (composite score)
    pub fn overall_strength(&self) -> f64 {
        let e = self.empirical.to_continuous();
        let n = self.normative.to_continuous();
        let m = self.materiality.to_continuous();

        // Weighted average: E=50%, N=30%, M=20%
        0.5 * e + 0.3 * n + 0.2 * m
    }

    /// Check if claim meets minimum verification threshold
    pub fn is_verified(&self) -> bool {
        self.empirical >= EmpiricalLevel::E2
    }

    /// Check if claim has normative consensus
    pub fn has_consensus(&self) -> bool {
        self.normative >= NormativeLevel::N2
    }

    /// Check if claim has practical relevance
    pub fn is_practical(&self) -> bool {
        self.materiality >= MaterialityLevel::M2
    }

    /// Convert to legacy EpistemicPosition (for backwards compatibility)
    pub fn to_legacy(&self) -> EpistemicPosition {
        EpistemicPosition {
            empirical: self.empirical.to_continuous(),
            normative: self.normative.to_continuous(),
            mythic: self.materiality.to_continuous(), // Note: M maps to legacy "mythic"
        }
    }

    /// Create from legacy EpistemicPosition
    pub fn from_legacy(pos: &EpistemicPosition) -> Self {
        Self {
            empirical: EmpiricalLevel::from_continuous(pos.empirical),
            normative: NormativeLevel::from_continuous(pos.normative),
            materiality: MaterialityLevel::from_continuous(pos.mythic),
            harmonic: None,
            confidence: 0.5,
            classified_by: None,
            classified_at: None,
        }
    }
}

/// Legacy: Epistemic position on the 3D cube (E/N/M axes)
/// Kept for backwards compatibility
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpistemicPosition {
    /// Empirical validity (0.0 to 1.0)
    pub empirical: f64,
    /// Normative coherence (0.0 to 1.0)
    pub normative: f64,
    /// Mythic resonance (0.0 to 1.0) - maps to Materiality in new system
    pub mythic: f64,
}

impl Default for EpistemicPosition {
    fn default() -> Self {
        EpistemicPosition {
            empirical: 0.5,
            normative: 0.5,
            mythic: 0.5,
        }
    }
}

impl EpistemicPosition {
    /// Convert to new EpistemicClassification
    pub fn to_classification(&self) -> EpistemicClassification {
        EpistemicClassification::from_legacy(self)
    }
}

/// A claim in the knowledge graph (legacy - uses continuous values)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Claim {
    pub id: String,
    pub content: String,
    pub classification: EpistemicPosition,
    pub author: String,
    pub sources: Vec<String>,
    pub tags: Vec<String>,
    pub claim_type: ClaimType,
    pub confidence: f64,
    pub expires: Option<Timestamp>,
    pub created: Timestamp,
    pub updated: Timestamp,
    pub version: u32,
}

/// A claim with full E/N/M/H classification (GIS v4.0)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClassifiedClaim {
    /// Unique claim identifier
    pub id: String,
    /// The claim content/statement
    pub content: String,
    /// Full E/N/M/H classification
    pub classification: EpistemicClassification,
    /// Author's DID
    pub author: String,
    /// Source references (URIs, DOIs, etc.)
    pub sources: Vec<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Type of claim
    pub claim_type: ClaimType,
    /// Domain/field this claim relates to
    pub domain: Option<String>,
    /// Optional expiration
    pub expires: Option<Timestamp>,
    /// Creation timestamp
    pub created: Timestamp,
    /// Last update timestamp
    pub updated: Timestamp,
    /// Version number
    pub version: u32,
}

impl ClassifiedClaim {
    /// Convert to legacy Claim format
    pub fn to_legacy(&self) -> Claim {
        Claim {
            id: self.id.clone(),
            content: self.content.clone(),
            classification: self.classification.to_legacy(),
            author: self.author.clone(),
            sources: self.sources.clone(),
            tags: self.tags.clone(),
            claim_type: self.claim_type.clone(),
            confidence: self.classification.confidence,
            expires: self.expires,
            created: self.created,
            updated: self.updated,
            version: self.version,
        }
    }

    /// Create from legacy Claim format
    pub fn from_legacy(claim: &Claim) -> Self {
        Self {
            id: claim.id.clone(),
            content: claim.content.clone(),
            classification: claim.classification.to_classification(),
            author: claim.author.clone(),
            sources: claim.sources.clone(),
            tags: claim.tags.clone(),
            claim_type: claim.claim_type.clone(),
            domain: None,
            expires: claim.expires,
            created: claim.created,
            updated: claim.updated,
            version: claim.version,
        }
    }
}

/// Classification vote/assessment from a community member
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClassificationVote {
    /// Vote identifier
    pub id: String,
    /// Claim being classified
    pub claim_id: String,
    /// Voter's DID
    pub voter: String,
    /// Proposed E level
    pub proposed_e: EmpiricalLevel,
    /// Proposed N level
    pub proposed_n: NormativeLevel,
    /// Proposed M level
    pub proposed_m: MaterialityLevel,
    /// Proposed harmonic impact
    pub proposed_h: Option<HarmonicImpact>,
    /// Confidence in this assessment (0.0-1.0)
    pub confidence: f64,
    /// Justification for the classification
    pub justification: String,
    /// Voter's K-vector trust score (from MATL)
    pub voter_k_trust: f64,
    /// Vote timestamp
    pub voted_at: Timestamp,
}

/// Aggregated classification result from multiple votes
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClassificationConsensus {
    /// Consensus identifier
    pub id: String,
    /// Claim being classified
    pub claim_id: String,
    /// Final E level (consensus)
    pub final_e: EmpiricalLevel,
    /// Final N level (consensus)
    pub final_n: NormativeLevel,
    /// Final M level (consensus)
    pub final_m: MaterialityLevel,
    /// Final harmonic impact (if any)
    pub final_h: Option<HarmonicImpact>,
    /// K-weighted confidence
    pub weighted_confidence: f64,
    /// Number of votes
    pub vote_count: u32,
    /// Total K-trust of voters
    pub total_k_trust: f64,
    /// Agreement level (0.0-1.0)
    pub agreement: f64,
    /// Consensus timestamp
    pub created: Timestamp,
    /// Whether this is the final consensus
    pub is_final: bool,
}

/// Types of claims
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ClaimType {
    Fact,
    Opinion,
    Prediction,
    Hypothesis,
    Definition,
    Historical,
    Normative,
    Narrative,
}

/// Evidence supporting a claim
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Evidence {
    pub id: String,
    pub claim_id: String,
    pub evidence_type: EvidenceType,
    pub source_uri: String,
    pub content: String,
    pub strength: f64,
    pub submitted_by: String,
    pub submitted_at: Timestamp,
}

/// Types of evidence
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EvidenceType {
    Observation,
    Experiment,
    Statistical,
    Expert,
    Document,
    CrossReference,
    Logical,
}

/// Claim challenge/dispute
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClaimChallenge {
    pub id: String,
    pub claim_id: String,
    pub challenger: String,
    pub reason: String,
    pub counter_evidence: Vec<String>,
    pub status: ChallengeStatus,
    pub created: Timestamp,
}

/// Status of a challenge
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ChallengeStatus {
    Pending,
    UnderReview,
    Accepted,
    Rejected,
    Withdrawn,
}

/// Link between a claim and a verification market
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClaimMarketLink {
    pub id: String,
    pub claim_id: String,
    pub market_id: String,
    pub link_type: MarketLinkType,
    pub status: MarketVerificationStatus,
    pub target_epistemic: EpistemicTarget,
    pub requested_by: AgentPubKey,
    pub created_at: Timestamp,
    pub resolved_at: Option<Timestamp>,
    pub resolution: Option<MarketResolution>,
}

/// Type of link between claim and market
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MarketLinkType {
    VerificationMarket,
    EvidenceSource,
    DependsOnMarket,
}

/// Status of market verification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MarketVerificationStatus {
    Pending,
    Active,
    Resolved,
    Cancelled,
}

/// Target epistemic position for verification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpistemicTarget {
    pub target_e: Option<f64>,
    pub target_n: Option<f64>,
    pub verification_criteria: String,
}

impl Default for EpistemicTarget {
    fn default() -> Self {
        EpistemicTarget {
            target_e: Some(0.8),
            target_n: None,
            verification_criteria: "Achieve high empirical validity".to_string(),
        }
    }
}

/// Resolution of a verification market
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MarketResolution {
    pub target_achieved: bool,
    pub confidence: f64,
    pub matl_weighted_confidence: f64,
    pub oracle_count: u32,
    pub new_epistemic: EpistemicPosition,
}

/// Dependency between claims in the belief graph
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClaimDependency {
    pub id: String,
    pub dependent_claim_id: String,
    pub dependency_claim_id: String,
    pub dependency_type: DependencyType,
    pub weight: f64,
    pub influence: InfluenceDirection,
    pub established_by: AgentPubKey,
    pub created_at: Timestamp,
    pub active: bool,
    pub justification: Option<String>,
}

/// Type of dependency between claims
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DependencyType {
    LogicalEntailment,
    EvidentialSupport,
    Premise,
    Refinement,
    Contradiction,
    Correlation,
}

/// Direction of influence between claims
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum InfluenceDirection {
    Positive,
    Negative,
    Bidirectional,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Claim(Claim),
    ClassifiedClaim(ClassifiedClaim),
    ClassificationVote(ClassificationVote),
    ClassificationConsensus(ClassificationConsensus),
    Evidence(Evidence),
    ClaimChallenge(ClaimChallenge),
    ClaimMarketLink(ClaimMarketLink),
    ClaimDependency(ClaimDependency),
}

#[hdk_link_types]
pub enum LinkTypes {
    AuthorToClaim,
    AuthorToClassifiedClaim,
    ClaimToEvidence,
    ClaimToClassificationVote,
    ClaimToClassificationConsensus,
    VoterToClassificationVote,
    EmpiricalLevelToClaim,
    NormativeLevelToClaim,
    MaterialityLevelToClaim,
    HarmonyToClaim,
    ClaimToChallenge,
    TagToClaim,
    TypeToClaim,
    ClaimHistory,
    ClaimToMarketLink,
    MarketToClaim,
    ClaimToDependency,
    DependencyToClaim,
    VerificationStatusIndex,
    EpistemicLevelIndex,
    ClaimIdToClaim,
}

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Claim(claim) => validate_create_claim(action, claim),
                EntryTypes::ClassifiedClaim(claim) => {
                    validate_create_classified_claim(action, claim)
                }
                EntryTypes::ClassificationVote(vote) => {
                    validate_create_classification_vote(action, vote)
                }
                EntryTypes::ClassificationConsensus(consensus) => {
                    validate_create_classification_consensus(action, consensus)
                }
                EntryTypes::Evidence(evidence) => validate_create_evidence(action, evidence),
                EntryTypes::ClaimChallenge(challenge) => {
                    validate_create_challenge(action, challenge)
                }
                EntryTypes::ClaimMarketLink(link) => validate_create_market_link(action, link),
                EntryTypes::ClaimDependency(dep) => validate_create_dependency(action, dep),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Claim(claim) => {
                    validate_update_claim(action, claim, original_action_hash)
                }
                EntryTypes::ClassifiedClaim(claim) => {
                    validate_update_classified_claim(action, claim, original_action_hash)
                }
                EntryTypes::ClassificationVote(_) => Ok(ValidateCallbackResult::Invalid(
                    "Classification votes cannot be updated".into(),
                )),
                EntryTypes::ClassificationConsensus(consensus) => {
                    validate_update_classification_consensus(action, consensus)
                }
                EntryTypes::Evidence(_) => Ok(ValidateCallbackResult::Invalid(
                    "Evidence cannot be updated".into(),
                )),
                EntryTypes::ClaimChallenge(challenge) => {
                    validate_update_challenge(action, challenge)
                }
                EntryTypes::ClaimMarketLink(link) => validate_update_market_link(action, link),
                EntryTypes::ClaimDependency(dep) => validate_update_dependency(action, dep),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AuthorToClaim => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AuthorToClassifiedClaim => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimToEvidence => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimToClassificationVote => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimToClassificationConsensus => Ok(ValidateCallbackResult::Valid),
            LinkTypes::VoterToClassificationVote => Ok(ValidateCallbackResult::Valid),
            LinkTypes::EmpiricalLevelToClaim => Ok(ValidateCallbackResult::Valid),
            LinkTypes::NormativeLevelToClaim => Ok(ValidateCallbackResult::Valid),
            LinkTypes::MaterialityLevelToClaim => Ok(ValidateCallbackResult::Valid),
            LinkTypes::HarmonyToClaim => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimToChallenge => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TagToClaim => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TypeToClaim => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimHistory => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimToMarketLink => Ok(ValidateCallbackResult::Valid),
            LinkTypes::MarketToClaim => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimToDependency => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DependencyToClaim => Ok(ValidateCallbackResult::Valid),
            LinkTypes::VerificationStatusIndex => Ok(ValidateCallbackResult::Valid),
            LinkTypes::EpistemicLevelIndex => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimIdToClaim => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            // Only the original link creator can delete a link
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original link creator can delete a link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(op_update) => {
            // Only the original author can update an entry
            let update_action = match op_update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(update_action.original_action_address.clone())?;
            if update_action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can update an entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(op_delete) => {
            // Only the original author can delete an entry
            let original = must_get_action(op_delete.action.deletes_address.clone())?;
            if op_delete.action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete an entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

/// Validate claim creation
fn validate_create_claim(_action: Create, claim: Claim) -> ExternResult<ValidateCallbackResult> {
    // Validate epistemic position ranges
    if claim.classification.empirical < 0.0 || claim.classification.empirical > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Empirical value must be between 0.0 and 1.0".into(),
        ));
    }
    if claim.classification.normative < 0.0 || claim.classification.normative > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Normative value must be between 0.0 and 1.0".into(),
        ));
    }
    if claim.classification.mythic < 0.0 || claim.classification.mythic > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Mythic value must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate confidence range
    if claim.confidence < 0.0 || claim.confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Confidence must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate author is a DID
    if !claim.author.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Author must be a valid DID".into(),
        ));
    }

    // Validate content is not empty
    if claim.content.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim content cannot be empty".into(),
        ));
    }

    // Validate version starts at 1
    if claim.version != 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial claim version must be 1".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate claim update
fn validate_update_claim(
    _action: Update,
    claim: Claim,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_claim: Claim = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original claim not found".into()
        )))?;

    // Cannot change claim ID
    if claim.id != original_claim.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change claim ID".into(),
        ));
    }

    // Cannot change author
    if claim.author != original_claim.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change claim author".into(),
        ));
    }

    // Version must increment
    if claim.version != original_claim.version + 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Version must be incremented by 1".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate evidence creation
fn validate_create_evidence(
    _action: Create,
    evidence: Evidence,
) -> ExternResult<ValidateCallbackResult> {
    // Validate strength range
    if evidence.strength < 0.0 || evidence.strength > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence strength must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate submitter is a DID
    if !evidence.submitted_by.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Submitter must be a valid DID".into(),
        ));
    }

    // Validate source URI not empty
    if evidence.source_uri.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence source URI is required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate challenge creation
fn validate_create_challenge(
    _action: Create,
    challenge: ClaimChallenge,
) -> ExternResult<ValidateCallbackResult> {
    // Validate challenger is a DID
    if !challenge.challenger.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Challenger must be a valid DID".into(),
        ));
    }

    // Validate reason is not empty
    if challenge.reason.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Challenge reason is required".into(),
        ));
    }

    // Initial status must be Pending
    if challenge.status != ChallengeStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial challenge status must be Pending".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate challenge update
fn validate_update_challenge(
    _action: Update,
    _challenge: ClaimChallenge,
) -> ExternResult<ValidateCallbackResult> {
    // Status updates are allowed
    Ok(ValidateCallbackResult::Valid)
}

/// Validate market link creation
fn validate_create_market_link(
    _action: Create,
    link: ClaimMarketLink,
) -> ExternResult<ValidateCallbackResult> {
    // Validate claim_id not empty
    if link.claim_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim ID is required".into(),
        ));
    }

    // Validate market_id not empty
    if link.market_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Market ID is required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate market link update
fn validate_update_market_link(
    _action: Update,
    _link: ClaimMarketLink,
) -> ExternResult<ValidateCallbackResult> {
    // Status updates are allowed
    Ok(ValidateCallbackResult::Valid)
}

/// Validate dependency creation
fn validate_create_dependency(
    _action: Create,
    dep: ClaimDependency,
) -> ExternResult<ValidateCallbackResult> {
    // Validate weight range
    if dep.weight < 0.0 || dep.weight > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dependency weight must be between 0.0 and 1.0".into(),
        ));
    }

    // Cannot depend on self
    if dep.dependent_claim_id == dep.dependency_claim_id {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim cannot depend on itself".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate dependency update
fn validate_update_dependency(
    _action: Update,
    dep: ClaimDependency,
) -> ExternResult<ValidateCallbackResult> {
    // Validate weight range
    if dep.weight < 0.0 || dep.weight > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dependency weight must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// E/N/M/H CLASSIFICATION VALIDATION
// ============================================================================

/// Validate classified claim creation
fn validate_create_classified_claim(
    _action: Create,
    claim: ClassifiedClaim,
) -> ExternResult<ValidateCallbackResult> {
    // Validate author is a DID
    if !claim.author.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Author must be a valid DID".into(),
        ));
    }

    // Validate content is not empty
    if claim.content.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim content cannot be empty".into(),
        ));
    }

    // Validate classification confidence
    if claim.classification.confidence < 0.0 || claim.classification.confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Classification confidence must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate harmonic impact if present
    if let Some(ref h) = claim.classification.harmonic {
        if h.resonance < 0.0 || h.resonance > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Harmonic resonance must be between 0.0 and 1.0".into(),
            ));
        }
        if h.love_alignment < 0.0 || h.love_alignment > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Love alignment must be between 0.0 and 1.0".into(),
            ));
        }
    }

    // Validate version starts at 1
    if claim.version != 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial claim version must be 1".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate classified claim update
fn validate_update_classified_claim(
    _action: Update,
    claim: ClassifiedClaim,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_claim: ClassifiedClaim = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original classified claim not found".into()
        )))?;

    // Cannot change claim ID
    if claim.id != original_claim.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change claim ID".into(),
        ));
    }

    // Cannot change author
    if claim.author != original_claim.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change claim author".into(),
        ));
    }

    // Version must increment
    if claim.version != original_claim.version + 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Version must be incremented by 1".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate classification vote creation
fn validate_create_classification_vote(
    _action: Create,
    vote: ClassificationVote,
) -> ExternResult<ValidateCallbackResult> {
    // Validate voter is a DID
    if !vote.voter.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter must be a valid DID".into(),
        ));
    }

    // Validate claim_id not empty
    if vote.claim_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim ID is required".into(),
        ));
    }

    // Validate confidence range
    if vote.confidence < 0.0 || vote.confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote confidence must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate K-trust range
    if vote.voter_k_trust < 0.0 || vote.voter_k_trust > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "K-trust must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate justification not empty
    if vote.justification.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Classification justification is required".into(),
        ));
    }

    // Validate harmonic impact if present
    if let Some(ref h) = vote.proposed_h {
        if h.resonance < 0.0 || h.resonance > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Harmonic resonance must be between 0.0 and 1.0".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate classification consensus creation
fn validate_create_classification_consensus(
    _action: Create,
    consensus: ClassificationConsensus,
) -> ExternResult<ValidateCallbackResult> {
    // Validate claim_id not empty
    if consensus.claim_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Claim ID is required".into(),
        ));
    }

    // Validate weighted confidence range
    if consensus.weighted_confidence < 0.0 || consensus.weighted_confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Weighted confidence must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate agreement range
    if consensus.agreement < 0.0 || consensus.agreement > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Agreement must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate vote count > 0
    if consensus.vote_count == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Consensus requires at least one vote".into(),
        ));
    }

    // Validate K-trust non-negative
    if consensus.total_k_trust < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total K-trust cannot be negative".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate classification consensus update
fn validate_update_classification_consensus(
    _action: Update,
    consensus: ClassificationConsensus,
) -> ExternResult<ValidateCallbackResult> {
    // Validate weighted confidence range
    if consensus.weighted_confidence < 0.0 || consensus.weighted_confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Weighted confidence must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate agreement range
    if consensus.agreement < 0.0 || consensus.agreement > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Agreement must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
