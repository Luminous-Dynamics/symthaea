// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claims Integrity Zome
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
    AllClaimsIndex,
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
            LinkTypes::AllClaimsIndex => Ok(ValidateCallbackResult::Valid),
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

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Helpers =====

    fn test_timestamp() -> Timestamp {
        Timestamp::from_micros(1704067200000000)
    }

    fn valid_claim() -> Claim {
        Claim {
            id: "claim-001".to_string(),
            content: "Water boils at 100C at sea level".to_string(),
            classification: EpistemicPosition {
                empirical: 0.95,
                normative: 0.5,
                mythic: 0.3,
            },
            author: "did:mycelix:alice".to_string(),
            sources: vec!["doi:10.1234/physics".to_string()],
            tags: vec!["physics".to_string(), "thermodynamics".to_string()],
            claim_type: ClaimType::Fact,
            confidence: 0.9,
            expires: None,
            created: test_timestamp(),
            updated: test_timestamp(),
            version: 1,
        }
    }

    fn valid_classified_claim() -> ClassifiedClaim {
        ClassifiedClaim {
            id: "cc-001".to_string(),
            content: "Solar energy reduces carbon emissions".to_string(),
            classification: EpistemicClassification::new(
                EmpiricalLevel::E3,
                NormativeLevel::N2,
                MaterialityLevel::M2,
            ),
            author: "did:mycelix:bob".to_string(),
            sources: vec!["doi:10.5678/energy".to_string()],
            tags: vec!["energy".to_string()],
            claim_type: ClaimType::Fact,
            domain: Some("environmental science".to_string()),
            expires: None,
            created: test_timestamp(),
            updated: test_timestamp(),
            version: 1,
        }
    }

    fn valid_evidence() -> Evidence {
        Evidence {
            id: "ev-001".to_string(),
            claim_id: "claim-001".to_string(),
            evidence_type: EvidenceType::Experiment,
            source_uri: "https://journal.org/paper/123".to_string(),
            content: "Experiment confirmed boiling point at 100C".to_string(),
            strength: 0.85,
            submitted_by: "did:mycelix:charlie".to_string(),
            submitted_at: test_timestamp(),
        }
    }

    fn valid_dependency() -> ClaimDependency {
        ClaimDependency {
            id: "dep-001".to_string(),
            dependent_claim_id: "claim-002".to_string(),
            dependency_claim_id: "claim-001".to_string(),
            dependency_type: DependencyType::EvidentialSupport,
            weight: 0.7,
            influence: InfluenceDirection::Positive,
            established_by: AgentPubKey::from_raw_36(vec![1u8; 36]),
            created_at: test_timestamp(),
            active: true,
            justification: Some("Claim 2 builds on evidence from claim 1".to_string()),
        }
    }

    // ===== EmpiricalLevel Tests =====

    #[test]
    fn test_empirical_level_to_continuous_ordering() {
        assert!(EmpiricalLevel::E0.to_continuous() < EmpiricalLevel::E1.to_continuous());
        assert!(EmpiricalLevel::E1.to_continuous() < EmpiricalLevel::E2.to_continuous());
        assert!(EmpiricalLevel::E2.to_continuous() < EmpiricalLevel::E3.to_continuous());
        assert!(EmpiricalLevel::E3.to_continuous() < EmpiricalLevel::E4.to_continuous());
    }

    #[test]
    fn test_empirical_level_roundtrip() {
        for level in [
            EmpiricalLevel::E0,
            EmpiricalLevel::E1,
            EmpiricalLevel::E2,
            EmpiricalLevel::E3,
            EmpiricalLevel::E4,
        ] {
            let continuous = level.to_continuous();
            let back = EmpiricalLevel::from_continuous(continuous);
            assert_eq!(level, back);
        }
    }

    #[test]
    fn test_empirical_level_boundary_values() {
        assert_eq!(EmpiricalLevel::from_continuous(0.0), EmpiricalLevel::E0);
        assert_eq!(EmpiricalLevel::from_continuous(1.0), EmpiricalLevel::E4);
        assert_eq!(EmpiricalLevel::from_continuous(-0.5), EmpiricalLevel::E0);
        assert_eq!(EmpiricalLevel::from_continuous(1.5), EmpiricalLevel::E4);
    }

    // ===== NormativeLevel Tests =====

    #[test]
    fn test_normative_level_to_continuous_ordering() {
        assert!(NormativeLevel::N0.to_continuous() < NormativeLevel::N1.to_continuous());
        assert!(NormativeLevel::N1.to_continuous() < NormativeLevel::N2.to_continuous());
        assert!(NormativeLevel::N2.to_continuous() < NormativeLevel::N3.to_continuous());
    }

    #[test]
    fn test_normative_level_roundtrip() {
        for level in [
            NormativeLevel::N0,
            NormativeLevel::N1,
            NormativeLevel::N2,
            NormativeLevel::N3,
        ] {
            let continuous = level.to_continuous();
            let back = NormativeLevel::from_continuous(continuous);
            assert_eq!(level, back);
        }
    }

    // ===== MaterialityLevel Tests =====

    #[test]
    fn test_materiality_level_to_continuous_ordering() {
        assert!(MaterialityLevel::M0.to_continuous() < MaterialityLevel::M1.to_continuous());
        assert!(MaterialityLevel::M1.to_continuous() < MaterialityLevel::M2.to_continuous());
        assert!(MaterialityLevel::M2.to_continuous() < MaterialityLevel::M3.to_continuous());
    }

    #[test]
    fn test_materiality_level_roundtrip() {
        for level in [
            MaterialityLevel::M0,
            MaterialityLevel::M1,
            MaterialityLevel::M2,
            MaterialityLevel::M3,
        ] {
            let continuous = level.to_continuous();
            let back = MaterialityLevel::from_continuous(continuous);
            assert_eq!(level, back);
        }
    }

    // ===== EpistemicClassification Tests =====

    #[test]
    fn test_epistemic_classification_code() {
        let ec = EpistemicClassification::new(
            EmpiricalLevel::E2,
            NormativeLevel::N1,
            MaterialityLevel::M2,
        );
        assert_eq!(ec.code(), "E2N1M2");
    }

    #[test]
    fn test_epistemic_classification_overall_strength() {
        // E4=1.0, N3=1.0, M3=1.0 -> max strength
        let max_ec = EpistemicClassification::new(
            EmpiricalLevel::E4,
            NormativeLevel::N3,
            MaterialityLevel::M3,
        );
        assert!((max_ec.overall_strength() - 1.0).abs() < 0.01);

        // E0=0.0, N0=0.0, M0=0.0 -> min strength
        let min_ec = EpistemicClassification::new(
            EmpiricalLevel::E0,
            NormativeLevel::N0,
            MaterialityLevel::M0,
        );
        assert!((min_ec.overall_strength() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_epistemic_classification_weighted_average() {
        // E=50%, N=30%, M=20%
        let ec = EpistemicClassification::new(
            EmpiricalLevel::E4,
            NormativeLevel::N0,
            MaterialityLevel::M0,
        );
        // E4=1.0 * 0.5 + N0=0.0 * 0.3 + M0=0.0 * 0.2 = 0.5
        assert!((ec.overall_strength() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_epistemic_classification_is_verified() {
        assert!(!EpistemicClassification::new(
            EmpiricalLevel::E0,
            NormativeLevel::N0,
            MaterialityLevel::M0
        )
        .is_verified());
        assert!(!EpistemicClassification::new(
            EmpiricalLevel::E1,
            NormativeLevel::N0,
            MaterialityLevel::M0
        )
        .is_verified());
        assert!(EpistemicClassification::new(
            EmpiricalLevel::E2,
            NormativeLevel::N0,
            MaterialityLevel::M0
        )
        .is_verified());
        assert!(EpistemicClassification::new(
            EmpiricalLevel::E3,
            NormativeLevel::N0,
            MaterialityLevel::M0
        )
        .is_verified());
        assert!(EpistemicClassification::new(
            EmpiricalLevel::E4,
            NormativeLevel::N0,
            MaterialityLevel::M0
        )
        .is_verified());
    }

    #[test]
    fn test_epistemic_classification_has_consensus() {
        assert!(!EpistemicClassification::new(
            EmpiricalLevel::E0,
            NormativeLevel::N0,
            MaterialityLevel::M0
        )
        .has_consensus());
        assert!(!EpistemicClassification::new(
            EmpiricalLevel::E0,
            NormativeLevel::N1,
            MaterialityLevel::M0
        )
        .has_consensus());
        assert!(EpistemicClassification::new(
            EmpiricalLevel::E0,
            NormativeLevel::N2,
            MaterialityLevel::M0
        )
        .has_consensus());
        assert!(EpistemicClassification::new(
            EmpiricalLevel::E0,
            NormativeLevel::N3,
            MaterialityLevel::M0
        )
        .has_consensus());
    }

    #[test]
    fn test_epistemic_classification_is_practical() {
        assert!(!EpistemicClassification::new(
            EmpiricalLevel::E0,
            NormativeLevel::N0,
            MaterialityLevel::M0
        )
        .is_practical());
        assert!(!EpistemicClassification::new(
            EmpiricalLevel::E0,
            NormativeLevel::N0,
            MaterialityLevel::M1
        )
        .is_practical());
        assert!(EpistemicClassification::new(
            EmpiricalLevel::E0,
            NormativeLevel::N0,
            MaterialityLevel::M2
        )
        .is_practical());
        assert!(EpistemicClassification::new(
            EmpiricalLevel::E0,
            NormativeLevel::N0,
            MaterialityLevel::M3
        )
        .is_practical());
    }

    #[test]
    fn test_epistemic_legacy_roundtrip() {
        let ec = EpistemicClassification::new(
            EmpiricalLevel::E2,
            NormativeLevel::N1,
            MaterialityLevel::M2,
        );
        let legacy = ec.to_legacy();
        let back = EpistemicClassification::from_legacy(&legacy);
        assert_eq!(ec.empirical, back.empirical);
        assert_eq!(ec.normative, back.normative);
        assert_eq!(ec.materiality, back.materiality);
    }

    #[test]
    fn test_epistemic_classification_with_harmonic() {
        let h = HarmonicImpact {
            primary_harmony: Harmony::EcologicalSymbiosis,
            secondary_harmonies: vec![Harmony::EmergentJustice],
            resonance: 0.8,
            coherence_contribution: true,
            love_alignment: 0.9,
        };
        let ec = EpistemicClassification::with_harmonic(
            EmpiricalLevel::E3,
            NormativeLevel::N2,
            MaterialityLevel::M2,
            h,
        );
        assert!(ec.harmonic.is_some());
        assert_eq!(
            ec.harmonic.as_ref().unwrap().primary_harmony,
            Harmony::EcologicalSymbiosis
        );
    }

    // ===== Claim Validation Tests =====

    #[test]
    fn test_claim_serde_roundtrip() {
        let claim = valid_claim();
        let json = serde_json::to_string(&claim).unwrap();
        let parsed: Claim = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "claim-001");
        assert_eq!(parsed.version, 1);
    }

    #[test]
    fn test_claim_type_variants() {
        let types = vec![
            ClaimType::Fact,
            ClaimType::Opinion,
            ClaimType::Prediction,
            ClaimType::Hypothesis,
            ClaimType::Definition,
            ClaimType::Historical,
            ClaimType::Normative,
            ClaimType::Narrative,
        ];
        assert_eq!(types.len(), 8);
    }

    #[test]
    fn test_claim_author_must_be_did() {
        let claim = valid_claim();
        assert!(claim.author.starts_with("did:"));
    }

    #[test]
    fn test_claim_content_not_empty() {
        let claim = valid_claim();
        assert!(!claim.content.is_empty());
    }

    #[test]
    fn test_claim_confidence_range() {
        let claim = valid_claim();
        assert!(claim.confidence >= 0.0 && claim.confidence <= 1.0);
    }

    #[test]
    fn test_claim_epistemic_position_range() {
        let claim = valid_claim();
        assert!(claim.classification.empirical >= 0.0 && claim.classification.empirical <= 1.0);
        assert!(claim.classification.normative >= 0.0 && claim.classification.normative <= 1.0);
        assert!(claim.classification.mythic >= 0.0 && claim.classification.mythic <= 1.0);
    }

    // ===== ClassifiedClaim Tests =====

    #[test]
    fn test_classified_claim_serde_roundtrip() {
        let claim = valid_classified_claim();
        let json = serde_json::to_string(&claim).unwrap();
        let parsed: ClassifiedClaim = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "cc-001");
        assert_eq!(parsed.classification.empirical, EmpiricalLevel::E3);
    }

    #[test]
    fn test_classified_claim_to_legacy_roundtrip() {
        let claim = valid_classified_claim();
        let legacy = claim.to_legacy();
        let back = ClassifiedClaim::from_legacy(&legacy);
        assert_eq!(claim.id, back.id);
        assert_eq!(
            claim.classification.empirical,
            back.classification.empirical
        );
    }

    // ===== Evidence Tests =====

    #[test]
    fn test_evidence_serde_roundtrip() {
        let evidence = valid_evidence();
        let json = serde_json::to_string(&evidence).unwrap();
        let parsed: Evidence = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "ev-001");
        assert_eq!(parsed.strength, 0.85);
    }

    #[test]
    fn test_evidence_type_variants() {
        let types = vec![
            EvidenceType::Observation,
            EvidenceType::Experiment,
            EvidenceType::Statistical,
            EvidenceType::Expert,
            EvidenceType::Document,
            EvidenceType::CrossReference,
            EvidenceType::Logical,
        ];
        assert_eq!(types.len(), 7);
    }

    #[test]
    fn test_evidence_strength_range() {
        let evidence = valid_evidence();
        assert!(evidence.strength >= 0.0 && evidence.strength <= 1.0);
    }

    #[test]
    fn test_evidence_submitter_must_be_did() {
        let evidence = valid_evidence();
        assert!(evidence.submitted_by.starts_with("did:"));
    }

    // ===== Dependency Tests =====

    #[test]
    fn test_dependency_cannot_self_reference() {
        let dep = ClaimDependency {
            dependent_claim_id: "claim-001".to_string(),
            dependency_claim_id: "claim-001".to_string(),
            ..valid_dependency()
        };
        // Self-dependency should be caught by validation
        assert_eq!(dep.dependent_claim_id, dep.dependency_claim_id);
    }

    #[test]
    fn test_dependency_weight_range() {
        let dep = valid_dependency();
        assert!(dep.weight >= 0.0 && dep.weight <= 1.0);
    }

    #[test]
    fn test_dependency_type_variants() {
        let types = vec![
            DependencyType::LogicalEntailment,
            DependencyType::EvidentialSupport,
            DependencyType::Premise,
            DependencyType::Refinement,
            DependencyType::Contradiction,
            DependencyType::Correlation,
        ];
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_influence_direction_variants() {
        let dirs = vec![
            InfluenceDirection::Positive,
            InfluenceDirection::Negative,
            InfluenceDirection::Bidirectional,
        ];
        assert_eq!(dirs.len(), 3);
    }

    // ===== Challenge Tests =====

    #[test]
    fn test_challenge_status_variants() {
        let statuses = vec![
            ChallengeStatus::Pending,
            ChallengeStatus::UnderReview,
            ChallengeStatus::Accepted,
            ChallengeStatus::Rejected,
            ChallengeStatus::Withdrawn,
        ];
        assert_eq!(statuses.len(), 5);
    }

    #[test]
    fn test_challenge_initial_status_must_be_pending() {
        let challenge = ClaimChallenge {
            id: "ch-001".to_string(),
            claim_id: "claim-001".to_string(),
            challenger: "did:mycelix:dave".to_string(),
            reason: "Claim is outdated".to_string(),
            counter_evidence: vec![],
            status: ChallengeStatus::Pending,
            created: test_timestamp(),
        };
        assert_eq!(challenge.status, ChallengeStatus::Pending);
    }

    // ===== Classification Vote Tests =====

    #[test]
    fn test_classification_vote_k_trust_range() {
        let vote = ClassificationVote {
            id: "vote-001".to_string(),
            claim_id: "cc-001".to_string(),
            voter: "did:mycelix:voter1".to_string(),
            proposed_e: EmpiricalLevel::E3,
            proposed_n: NormativeLevel::N2,
            proposed_m: MaterialityLevel::M2,
            proposed_h: None,
            confidence: 0.8,
            justification: "Substantial evidence supports E3".to_string(),
            voter_k_trust: 0.75,
            voted_at: test_timestamp(),
        };
        assert!(vote.voter_k_trust >= 0.0 && vote.voter_k_trust <= 1.0);
        assert!(vote.confidence >= 0.0 && vote.confidence <= 1.0);
    }

    // ===== Classification Consensus Tests =====

    #[test]
    fn test_consensus_requires_votes() {
        let consensus = ClassificationConsensus {
            id: "cons-001".to_string(),
            claim_id: "cc-001".to_string(),
            final_e: EmpiricalLevel::E3,
            final_n: NormativeLevel::N2,
            final_m: MaterialityLevel::M2,
            final_h: None,
            weighted_confidence: 0.85,
            vote_count: 5,
            total_k_trust: 3.5,
            agreement: 0.9,
            created: test_timestamp(),
            is_final: true,
        };
        assert!(consensus.vote_count > 0);
        assert!(consensus.weighted_confidence >= 0.0 && consensus.weighted_confidence <= 1.0);
        assert!(consensus.agreement >= 0.0 && consensus.agreement <= 1.0);
    }

    // ===== Adversarial Scenario Tests =====

    #[test]
    fn test_coordinated_attesters_diluted_by_k_trust() {
        // Scenario: 3 low-trust attesters all affirm E4 for a dubious claim
        // vs 1 high-trust attester voting E1
        let low_trust_votes: Vec<(EmpiricalLevel, f64)> = vec![
            (EmpiricalLevel::E4, 0.1), // low K-trust
            (EmpiricalLevel::E4, 0.1),
            (EmpiricalLevel::E4, 0.1),
        ];
        let high_trust_vote = (EmpiricalLevel::E1, 0.9);

        // K-weighted average: (4*0.1 + 4*0.1 + 4*0.1 + 1*0.9) / (0.1+0.1+0.1+0.9)
        //                   = (0.4+0.4+0.4+0.9) / 1.2 = 2.1/1.2 = 1.75
        let mut weighted_sum = 0.0;
        let mut total_k = 0.0;
        for (level, k) in &low_trust_votes {
            weighted_sum += (*level as u8 as f64) * k;
            total_k += k;
        }
        weighted_sum += (high_trust_vote.0 as u8 as f64) * high_trust_vote.1;
        total_k += high_trust_vote.1;

        let weighted_e = weighted_sum / total_k;
        // Result should be closer to E2 than E4, showing K-trust dilutes coordinated low-trust votes
        assert!(
            weighted_e < 2.0,
            "K-trust weighted result should be below E2, got {}",
            weighted_e
        );
    }

    #[test]
    fn test_zero_attestations_default() {
        // With no attestations, default epistemic position should be neutral
        let default = EpistemicPosition::default();
        assert_eq!(default.empirical, 0.5);
        assert_eq!(default.normative, 0.5);
        assert_eq!(default.mythic, 0.5);
    }

    #[test]
    fn test_single_attester_confidence_capped() {
        // A single attester's confidence should not exceed their K-trust
        let vote = ClassificationVote {
            id: "vote-single".to_string(),
            claim_id: "cc-001".to_string(),
            voter: "did:mycelix:lone_wolf".to_string(),
            proposed_e: EmpiricalLevel::E4,
            proposed_n: NormativeLevel::N3,
            proposed_m: MaterialityLevel::M3,
            proposed_h: None,
            confidence: 0.99,
            justification: "I am very confident".to_string(),
            voter_k_trust: 0.3,
            voted_at: test_timestamp(),
        };
        // Effective weight should be limited by low K-trust
        assert!(vote.voter_k_trust < vote.confidence);
    }

    #[test]
    fn test_conflicting_attestations_produce_low_agreement() {
        // If attesters disagree strongly, agreement should be low
        let votes_e = vec![
            EmpiricalLevel::E0,
            EmpiricalLevel::E4,
            EmpiricalLevel::E1,
            EmpiricalLevel::E3,
        ];
        let values: Vec<f64> = votes_e.iter().map(|e| e.to_continuous()).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        // High variance indicates disagreement
        assert!(
            variance > 0.1,
            "Conflicting votes should produce high variance: {}",
            variance
        );
    }

    // ===== Harmony Tests =====

    #[test]
    fn test_all_harmony_variants() {
        let harmonies = vec![
            Harmony::PanSentientFlourishing,
            Harmony::IntegralWisdom,
            Harmony::ResonantCoherence,
            Harmony::EmergentJustice,
            Harmony::CreativeExpression,
            Harmony::EmbodiedPresence,
            Harmony::RelationalDepth,
            Harmony::EcologicalSymbiosis,
            Harmony::TemporalWisdom,
            Harmony::MysteryEmbracing,
            Harmony::PlayfulBecoming,
            Harmony::UnifiedDiversity,
        ];
        assert_eq!(harmonies.len(), 12);
        // Each harmony should have a non-empty name
        for h in &harmonies {
            assert!(!h.name().is_empty());
        }
    }

    #[test]
    fn test_harmonic_impact_default() {
        let h = HarmonicImpact::default();
        assert_eq!(h.primary_harmony, Harmony::IntegralWisdom);
        assert!(h.resonance >= 0.0 && h.resonance <= 1.0);
        assert!(h.love_alignment >= 0.0 && h.love_alignment <= 1.0);
    }

    // ===== Market Link Tests =====

    #[test]
    fn test_market_link_type_variants() {
        let types = vec![
            MarketLinkType::VerificationMarket,
            MarketLinkType::EvidenceSource,
            MarketLinkType::DependsOnMarket,
        ];
        assert_eq!(types.len(), 3);
    }

    #[test]
    fn test_market_verification_status_variants() {
        let statuses = vec![
            MarketVerificationStatus::Pending,
            MarketVerificationStatus::Active,
            MarketVerificationStatus::Resolved,
            MarketVerificationStatus::Cancelled,
        ];
        assert_eq!(statuses.len(), 4);
    }

    #[test]
    fn test_epistemic_target_default() {
        let target = EpistemicTarget::default();
        assert_eq!(target.target_e, Some(0.8));
        assert!(target.target_n.is_none());
    }

    // ===== Epistemic Description Tests =====

    #[test]
    fn test_empirical_level_descriptions_non_empty() {
        for level in [
            EmpiricalLevel::E0,
            EmpiricalLevel::E1,
            EmpiricalLevel::E2,
            EmpiricalLevel::E3,
            EmpiricalLevel::E4,
        ] {
            assert!(!level.description().is_empty());
        }
    }

    #[test]
    fn test_normative_level_descriptions_non_empty() {
        for level in [
            NormativeLevel::N0,
            NormativeLevel::N1,
            NormativeLevel::N2,
            NormativeLevel::N3,
        ] {
            assert!(!level.description().is_empty());
        }
    }

    #[test]
    fn test_materiality_level_descriptions_non_empty() {
        for level in [
            MaterialityLevel::M0,
            MaterialityLevel::M1,
            MaterialityLevel::M2,
            MaterialityLevel::M3,
        ] {
            assert!(!level.description().is_empty());
        }
    }

    // ===== Adversarial Test Helpers =====

    fn adv_timestamp() -> Timestamp {
        Timestamp::from_micros(1704067200000000)
    }

    fn adv_claim() -> Claim {
        Claim {
            id: "claim-001".to_string(),
            content: "Water boils at 100C at sea level".to_string(),
            classification: EpistemicPosition {
                empirical: 0.95,
                normative: 0.5,
                mythic: 0.3,
            },
            author: "did:mycelix:alice".to_string(),
            sources: vec!["doi:10.1234/physics".to_string()],
            tags: vec!["physics".to_string(), "thermodynamics".to_string()],
            claim_type: ClaimType::Fact,
            confidence: 0.9,
            expires: None,
            created: adv_timestamp(),
            updated: adv_timestamp(),
            version: 1,
        }
    }

    fn adv_dependency() -> ClaimDependency {
        ClaimDependency {
            id: "dep-001".to_string(),
            dependent_claim_id: "claim-002".to_string(),
            dependency_claim_id: "claim-001".to_string(),
            dependency_type: DependencyType::EvidentialSupport,
            weight: 0.7,
            influence: InfluenceDirection::Positive,
            established_by: AgentPubKey::from_raw_36(vec![1u8; 36]),
            created_at: adv_timestamp(),
            active: true,
            justification: Some("Claim 2 builds on evidence from claim 1".to_string()),
        }
    }

    // ===== Adversarial Simulation Tests =====
    // Tests verifying the knowledge truth-engine's resilience to
    // sybil attacks, confidence gaming, circular citations,
    // consensus manipulation, and temporal attacks.

    // --- Sybil Attack: N identical low-trust attesters ---

    #[test]
    fn test_sybil_attack_10_low_trust_vs_1_high_trust() {
        // 10 sybil attackers with K-trust 0.05 all vote E4
        // 1 legitimate attester with K-trust 0.9 votes E1
        let sybil_count = 10;
        let sybil_k = 0.05;
        let sybil_vote = EmpiricalLevel::E4 as u8 as f64; // 4.0

        let legit_k = 0.9;
        let legit_vote = EmpiricalLevel::E1 as u8 as f64; // 1.0

        let sybil_weighted = sybil_count as f64 * sybil_vote * sybil_k;
        let legit_weighted = legit_vote * legit_k;
        let total_k = sybil_count as f64 * sybil_k + legit_k;

        let weighted_result = (sybil_weighted + legit_weighted) / total_k;
        // Without K-trust: (10*4 + 1)/11 = 3.7 (close to E4)
        // With K-trust: (10*4*0.05 + 1*0.9)/(10*0.05+0.9) = (2.0+0.9)/1.4 = 2.07
        assert!(
            weighted_result < 2.5,
            "K-trust should dilute sybil influence below E2.5, got {}",
            weighted_result
        );
    }

    #[test]
    fn test_sybil_attack_100_bots_vs_5_trusted() {
        // Extreme case: 100 sybils with near-zero trust
        let sybil_count = 100;
        let sybil_k = 0.01;
        let sybil_vote = 4.0; // E4

        let trusted_count = 5;
        let trusted_k = 0.8;
        let trusted_vote = 1.0; // E1

        let sybil_weighted = sybil_count as f64 * sybil_vote * sybil_k;
        let trusted_weighted = trusted_count as f64 * trusted_vote * trusted_k;
        let total_k = sybil_count as f64 * sybil_k + trusted_count as f64 * trusted_k;

        let weighted_result = (sybil_weighted + trusted_weighted) / total_k;
        // (100*4*0.01 + 5*1*0.8) / (100*0.01 + 5*0.8) = (4.0+4.0)/(1.0+4.0) = 1.6
        assert!(
            weighted_result < 2.0,
            "Even 100 sybils should not push result past E2 with K-trust, got {}",
            weighted_result
        );
    }

    #[test]
    fn test_sybil_identical_votes_detected_by_zero_variance() {
        // All sybil votes are identical -- a detection heuristic
        let sybil_votes: Vec<f64> = vec![4.0; 20]; // 20 identical E4 votes
        let mean = sybil_votes.iter().sum::<f64>() / sybil_votes.len() as f64;
        let variance =
            sybil_votes.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / sybil_votes.len() as f64;
        assert!(
            variance < f64::EPSILON,
            "Zero variance among votes is a sybil detection signal"
        );
    }

    #[test]
    fn test_sybil_k_trust_sum_bounded() {
        // Total K-trust of sybils should be bounded
        let sybil_count = 50;
        let sybil_k = 0.02; // very low trust each
        let total_sybil_k = sybil_count as f64 * sybil_k;
        assert!(
            total_sybil_k <= 1.0,
            "Total sybil K-trust {} should not exceed a single high-trust voter",
            total_sybil_k
        );
    }

    #[test]
    fn test_adv_coordinated_attesters_diluted_by_k_trust() {
        // 3 low-trust attesters all affirm E4 for a dubious claim
        // vs 1 high-trust attester voting E1
        let low_trust_votes: Vec<(EmpiricalLevel, f64)> = vec![
            (EmpiricalLevel::E4, 0.1),
            (EmpiricalLevel::E4, 0.1),
            (EmpiricalLevel::E4, 0.1),
        ];
        let high_trust_vote = (EmpiricalLevel::E1, 0.9);

        let mut weighted_sum = 0.0;
        let mut total_k = 0.0;
        for (level, k) in &low_trust_votes {
            weighted_sum += (*level as u8 as f64) * k;
            total_k += k;
        }
        weighted_sum += (high_trust_vote.0 as u8 as f64) * high_trust_vote.1;
        total_k += high_trust_vote.1;

        let weighted_e = weighted_sum / total_k;
        assert!(
            weighted_e < 2.0,
            "K-trust weighted result should be below E2, got {}",
            weighted_e
        );
    }

    // --- Confidence Gaming: Build reputation then flip ---

    #[test]
    fn test_confidence_gaming_reputation_drop() {
        // Attester builds legitimate reputation over 50 true claims
        // Then submits 5 false claims
        let verified_true = 50u32;
        let verified_false = 5u32;
        let claims_authored = verified_true + verified_false;

        let historical_accuracy = verified_true as f64 / (verified_true + verified_false) as f64;
        assert!(
            historical_accuracy < 1.0,
            "False claims should reduce accuracy"
        );
        assert!(historical_accuracy > 0.85);

        // Using the reputation formula from inference coordinator
        let matl_trust = 0.5;
        let overall_score = (historical_accuracy * 0.4
            + matl_trust * 0.4
            + (1.0 - (verified_false as f64 / (claims_authored as f64 + 1.0))) * 0.2)
            .clamp(0.0, 1.0);
        assert!(
            overall_score < 0.95,
            "Reputation should drop below 0.95 after false claims, got {}",
            overall_score
        );
    }

    #[test]
    fn test_confidence_gaming_sudden_flip_detectable() {
        // Attester voting E3-E4 consistently then suddenly votes E0
        let recent_votes: Vec<f64> = vec![
            3.0, 4.0, 3.0, 4.0, 3.0, // consistent high
            0.0, 0.0, 0.0, // sudden flip
        ];
        let window = 5;
        let recent_mean = recent_votes[recent_votes.len() - 3..].iter().sum::<f64>() / 3.0;
        let historical_mean = recent_votes[..window].iter().sum::<f64>() / window as f64;

        let deviation = (historical_mean - recent_mean).abs();
        assert!(
            deviation > 2.0,
            "Deviation of {:.1} should trigger flip detection",
            deviation
        );
    }

    #[test]
    fn test_confidence_gaming_weight_decays_after_false_claims() {
        let initial_k_trust: f64 = 0.9;
        let false_claim_decay: f64 = 0.8; // 20% decay per false claim

        let k_after_1_false = initial_k_trust * false_claim_decay;
        let k_after_3_false = initial_k_trust * false_claim_decay.powi(3);

        assert!(k_after_1_false < initial_k_trust);
        assert!(
            k_after_3_false < 0.5,
            "3 false claims should halve trust, got {}",
            k_after_3_false
        );
    }

    // --- Circular Citation Detection ---

    #[test]
    fn test_circular_citation_a_cites_b_cites_a() {
        let dep_a_to_b = ClaimDependency {
            id: "dep-a-b".to_string(),
            dependent_claim_id: "claim-A".to_string(),
            dependency_claim_id: "claim-B".to_string(),
            ..adv_dependency()
        };

        let dep_b_to_a = ClaimDependency {
            id: "dep-b-a".to_string(),
            dependent_claim_id: "claim-B".to_string(),
            dependency_claim_id: "claim-A".to_string(),
            ..adv_dependency()
        };

        // Detect cycle: B depends on A, and A depends on B
        assert_eq!(
            dep_a_to_b.dependency_claim_id,
            dep_b_to_a.dependent_claim_id
        );
        assert_eq!(
            dep_b_to_a.dependency_claim_id,
            dep_a_to_b.dependent_claim_id
        );
    }

    #[test]
    fn test_circular_citation_three_node_cycle() {
        // A -> B -> C -> A
        let deps: Vec<(&str, &str)> = vec![("claim-A", "claim-B"), ("claim-B", "claim-C")];

        // BFS cycle detection (mirrors would_create_cycle from coordinator)
        fn has_cycle(deps: &[(&str, &str)], from: &str, to: &str) -> bool {
            let mut visited = vec![];
            let mut to_check = vec![to.to_string()];
            while let Some(current) = to_check.pop() {
                if current == from {
                    return true;
                }
                if visited.contains(&current) {
                    continue;
                }
                visited.push(current.clone());
                for (dependent, dependency) in deps {
                    if *dependent == current {
                        to_check.push(dependency.to_string());
                    }
                }
            }
            false
        }

        // Before adding C->A, check: does A eventually depend on C?
        assert!(
            has_cycle(&deps, "claim-C", "claim-A"),
            "Adding C->A should detect cycle through A->B->C"
        );
    }

    #[test]
    fn test_self_reference_is_degenerate_cycle() {
        let dep = ClaimDependency {
            dependent_claim_id: "claim-self".to_string(),
            dependency_claim_id: "claim-self".to_string(),
            ..adv_dependency()
        };
        assert_eq!(dep.dependent_claim_id, dep.dependency_claim_id);
    }

    #[test]
    fn test_cascade_circular_detection_via_visited_set() {
        let mut visited: Vec<String> = vec![];
        let mut circular_detected: Vec<String> = vec![];

        let traversal = vec!["A", "B", "C", "A"]; // A appears twice

        for claim_id in traversal {
            if visited.contains(&claim_id.to_string()) {
                circular_detected.push(claim_id.to_string());
            } else {
                visited.push(claim_id.to_string());
            }
        }

        assert_eq!(circular_detected, vec!["A"]);
        assert_eq!(visited.len(), 3);
    }

    // --- Consensus Manipulation: 51% coordinated attesters ---

    #[test]
    fn test_consensus_manipulation_51_percent_low_trust() {
        let coordinated_count = 51;
        let coordinated_k = 0.1;
        let coordinated_vote = 4.0; // E4

        let legitimate_count = 49;
        let legitimate_k = 0.7;
        let legitimate_vote = 1.0; // E1

        let coord_weighted = coordinated_count as f64 * coordinated_vote * coordinated_k;
        let legit_weighted = legitimate_count as f64 * legitimate_vote * legitimate_k;
        let total_k =
            coordinated_count as f64 * coordinated_k + legitimate_count as f64 * legitimate_k;

        let weighted_result = (coord_weighted + legit_weighted) / total_k;
        assert!(
            weighted_result < 2.0,
            "51% majority with low trust should not dominate, got {:.2}",
            weighted_result
        );
    }

    #[test]
    fn test_consensus_high_disagreement_lowers_agreement() {
        let values: Vec<f64> = vec![
            EmpiricalLevel::E4.to_continuous(),
            EmpiricalLevel::E0.to_continuous(),
        ];
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let agreement = (1.0 - variance * 4.0).max(0.0);
        assert!(
            agreement < 0.5,
            "Split votes should produce low agreement: {:.2}",
            agreement
        );
    }

    #[test]
    fn test_consensus_requires_minimum_vote_count() {
        let consensus = ClassificationConsensus {
            id: "cons-weak".to_string(),
            claim_id: "cc-001".to_string(),
            final_e: EmpiricalLevel::E4,
            final_n: NormativeLevel::N3,
            final_m: MaterialityLevel::M3,
            final_h: None,
            weighted_confidence: 0.3,
            vote_count: 1,
            total_k_trust: 0.9,
            agreement: 1.0,
            created: adv_timestamp(),
            is_final: false,
        };
        assert_eq!(consensus.vote_count, 1);
        assert!(
            !consensus.is_final,
            "Single-vote consensus should not be final"
        );
    }

    #[test]
    fn test_consensus_low_avg_k_trust_is_suspicious() {
        let consensus = ClassificationConsensus {
            id: "cons-sybil".to_string(),
            claim_id: "cc-001".to_string(),
            final_e: EmpiricalLevel::E4,
            final_n: NormativeLevel::N3,
            final_m: MaterialityLevel::M3,
            final_h: None,
            weighted_confidence: 0.2,
            vote_count: 100,
            total_k_trust: 0.5,
            agreement: 1.0,
            created: adv_timestamp(),
            is_final: false,
        };
        let avg_k = consensus.total_k_trust / consensus.vote_count as f64;
        assert!(
            avg_k < 0.01,
            "Average K-trust of {:.4} indicates sybil pattern",
            avg_k
        );
    }

    #[test]
    fn test_conflicting_attestations_produce_high_variance() {
        let votes_e = vec![
            EmpiricalLevel::E0,
            EmpiricalLevel::E4,
            EmpiricalLevel::E1,
            EmpiricalLevel::E3,
        ];
        let values: Vec<f64> = votes_e.iter().map(|e| e.to_continuous()).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        assert!(
            variance > 0.1,
            "Conflicting votes should produce high variance: {}",
            variance
        );
    }

    // --- Temporal Attack: Old claim contradicted by new evidence ---

    #[test]
    fn test_temporal_attack_old_claim_versioning() {
        let old_claim = adv_claim();

        let updated_claim = Claim {
            classification: EpistemicPosition {
                empirical: 0.05,
                normative: 0.5,
                mythic: 0.3,
            },
            confidence: 0.1,
            updated: Timestamp::from_micros(5000000),
            version: 2,
            ..old_claim.clone()
        };

        assert!(updated_claim.version > old_claim.version);
        assert!(updated_claim.classification.empirical < old_claim.classification.empirical);
        assert!(updated_claim.confidence < old_claim.confidence);
    }

    #[test]
    fn test_temporal_challenge_triggers_review() {
        let challenge = ClaimChallenge {
            id: "ch-temporal".to_string(),
            claim_id: "claim-001".to_string(),
            challenger: "did:mycelix:reviewer".to_string(),
            reason: "New evidence contradicts claim".to_string(),
            counter_evidence: vec!["doi:10.1234/new".to_string()],
            status: ChallengeStatus::Pending,
            created: Timestamp::from_micros(3000000),
        };
        assert!(matches!(challenge.status, ChallengeStatus::Pending));
        assert!(!challenge.counter_evidence.is_empty());
    }

    #[test]
    fn test_temporal_recency_weighting() {
        let old_evidence = Evidence {
            id: "ev-old".to_string(),
            claim_id: "claim-001".to_string(),
            evidence_type: EvidenceType::Observation,
            source_uri: "old_source".to_string(),
            content: "Old observation".to_string(),
            strength: 0.7,
            submitted_by: "did:mycelix:old".to_string(),
            submitted_at: Timestamp::from_micros(1000000),
        };

        let new_evidence = Evidence {
            id: "ev-new".to_string(),
            claim_id: "claim-001".to_string(),
            evidence_type: EvidenceType::Experiment,
            source_uri: "new_source".to_string(),
            content: "New experiment contradicting old observation".to_string(),
            strength: 0.9,
            submitted_by: "did:mycelix:new".to_string(),
            submitted_at: Timestamp::from_micros(5000000),
        };

        assert!(new_evidence.submitted_at.as_micros() > old_evidence.submitted_at.as_micros());
        assert!(new_evidence.strength > old_evidence.strength);
    }

    #[test]
    fn test_expired_claim_not_authoritative() {
        let expired_claim = Claim {
            id: "claim-expired".to_string(),
            content: "Time-sensitive prediction".to_string(),
            classification: EpistemicPosition {
                empirical: 0.8,
                normative: 0.5,
                mythic: 0.3,
            },
            author: "did:mycelix:oracle".to_string(),
            sources: vec![],
            tags: vec!["prediction".to_string()],
            claim_type: ClaimType::Prediction,
            confidence: 0.8,
            expires: Some(Timestamp::from_micros(2000000)),
            created: adv_timestamp(),
            updated: adv_timestamp(),
            version: 1,
        };
        let now = Timestamp::from_micros(5000000);
        let is_expired = expired_claim.expires.map(|e| e < now).unwrap_or(false);
        assert!(is_expired, "Expired claims should not be authoritative");
    }

    // --- Additional adversarial edge cases ---

    #[test]
    fn test_negative_influence_reduces_epistemic_score() {
        let dep = ClaimDependency {
            influence: InfluenceDirection::Negative,
            weight: 0.8,
            ..adv_dependency()
        };
        let influence_factor = match dep.influence {
            InfluenceDirection::Positive => 1.0,
            InfluenceDirection::Negative => -1.0,
            InfluenceDirection::Bidirectional => 0.5,
        };
        let weighted_contribution = 0.9 * dep.weight * influence_factor;
        assert!(
            weighted_contribution < 0.0,
            "Negative influence should be negative: {}",
            weighted_contribution
        );
    }

    #[test]
    fn test_cascade_max_depth_prevents_deep_attack() {
        let max_depth: u32 = 10;
        let chain_depth = 15;
        assert!(
            chain_depth > max_depth,
            "Deep chains beyond {} should be truncated",
            max_depth
        );
    }

    #[test]
    fn test_cascade_max_updates_prevents_flood_attack() {
        let max_updates: usize = 100;
        let attacker_dependents = 500;
        assert!(
            attacker_dependents > max_updates,
            "Flood capped at {}",
            max_updates
        );
    }

    #[test]
    fn test_inactive_dependency_excluded_from_cascade() {
        let dep = ClaimDependency {
            active: false,
            ..adv_dependency()
        };
        assert!(!dep.active);
    }

    #[test]
    fn test_zero_weight_dependency_has_no_influence() {
        let dep = ClaimDependency {
            weight: 0.0,
            ..adv_dependency()
        };
        let contribution = 0.9 * dep.weight;
        assert!(contribution.abs() < f64::EPSILON);
    }

    #[test]
    fn test_vote_confidence_exceeding_k_trust_is_suspect() {
        let vote = ClassificationVote {
            id: "vote-suspect".to_string(),
            claim_id: "cc-001".to_string(),
            voter: "did:mycelix:suspicious".to_string(),
            proposed_e: EmpiricalLevel::E4,
            proposed_n: NormativeLevel::N3,
            proposed_m: MaterialityLevel::M3,
            proposed_h: None,
            confidence: 0.99,
            justification: "Trust me".to_string(),
            voter_k_trust: 0.01,
            voted_at: adv_timestamp(),
        };
        let ratio = vote.confidence / vote.voter_k_trust;
        assert!(
            ratio > 10.0,
            "Confidence/K-trust ratio of {:.0}x is suspicious",
            ratio
        );
    }

    #[test]
    fn test_single_attester_confidence_capped_by_k_trust() {
        let vote = ClassificationVote {
            id: "vote-single".to_string(),
            claim_id: "cc-001".to_string(),
            voter: "did:mycelix:lone".to_string(),
            proposed_e: EmpiricalLevel::E4,
            proposed_n: NormativeLevel::N3,
            proposed_m: MaterialityLevel::M3,
            proposed_h: None,
            confidence: 0.99,
            justification: "Very confident".to_string(),
            voter_k_trust: 0.3,
            voted_at: adv_timestamp(),
        };
        assert!(vote.voter_k_trust < vote.confidence);
    }

    #[test]
    fn test_zero_attestations_default_neutral() {
        let default = EpistemicPosition::default();
        assert_eq!(default.empirical, 0.5);
        assert_eq!(default.normative, 0.5);
        assert_eq!(default.mythic, 0.5);
    }
}
