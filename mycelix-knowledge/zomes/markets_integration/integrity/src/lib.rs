// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Markets Integration Integrity Zome
//!
//! Defines entry types for bidirectional integration between Knowledge and Epistemic Markets:
//! - Verification market requests (Knowledge → Markets)
//! - Market evidence (Markets → Knowledge)
//! - Claim usage in predictions (Markets → Knowledge)
//!
//! This enables claims to be verified through prediction markets, and predictions
//! to reference knowledge claims as evidence.
//!
//! Updated to use HDI 0.7 patterns

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// EPISTEMIC TYPES (shared with claims)
// ============================================================================

/// Epistemic position in the 3D E-N-M space
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpistemicPosition {
    /// Empirical validity (0.0-1.0): How verifiable through observation?
    pub empirical: f64,
    /// Normative coherence (0.0-1.0): How ethically coherent?
    pub normative: f64,
    /// Mythic resonance (0.0-1.0): What narrative significance?
    pub mythic: f64,
}

/// Target epistemic position for verification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpistemicTarget {
    /// Target empirical level (None = don't change)
    pub empirical: Option<f64>,
    /// Target normative level (None = don't change)
    pub normative: Option<f64>,
    /// Target mythic level (None = don't change)
    pub mythic: Option<f64>,
}

// ============================================================================
// VERIFICATION MARKET REQUEST
// ============================================================================

/// Request to spawn a verification market for a claim
///
/// When a claim needs verification, this entry tracks the request
/// and its lifecycle through market creation and resolution.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerificationMarketRequest {
    /// Unique identifier
    pub id: String,

    /// The claim being verified
    pub claim_id: String,

    /// Market question (e.g., "Will claim X achieve E4 verification?")
    pub question: String,

    /// Current epistemic position of the claim
    pub current_epistemic: EpistemicPosition,

    /// Target epistemic position after verification
    pub target_epistemic: EpistemicTarget,

    /// Bounty in tokens for market participation
    pub bounty: u64,

    /// Deadline for market resolution
    pub deadline: Timestamp,

    /// Agent who requested the verification
    pub requester: AgentPubKey,

    /// Current status of the request
    pub status: MarketRequestStatus,

    /// Supporting evidence submitted with request
    pub supporting_evidence: Vec<String>,

    /// Counter-evidence to be challenged
    pub counter_evidence: Vec<String>,

    /// When the request was created
    pub created_at: Timestamp,

    /// When the request was last updated
    pub updated_at: Timestamp,
}

/// Status of a verification market request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MarketRequestStatus {
    /// Request submitted, awaiting market creation
    Pending,

    /// Market has been created in Epistemic Markets
    MarketCreated {
        market_id: String,
        created_at: Timestamp,
    },

    /// Market is active and accepting predictions
    MarketActive {
        market_id: String,
        current_probability: f64,
        participant_count: u32,
    },

    /// Market has resolved, awaiting claim update
    MarketResolved {
        market_id: String,
        outcome: String,
        confidence: f64,
        resolved_at: Timestamp,
    },

    /// Claim has been updated based on resolution
    ClaimUpdated {
        market_id: String,
        new_epistemic: EpistemicPosition,
        updated_at: Timestamp,
    },

    /// Request expired without market creation
    Expired {
        reason: String,
    },

    /// Request was cancelled
    Cancelled {
        reason: String,
        cancelled_by: AgentPubKey,
    },
}

// ============================================================================
// MARKET EVIDENCE
// ============================================================================

/// Evidence from a resolved market that updates a claim's credibility
///
/// When an Epistemic Market resolves, this records the outcome
/// as evidence affecting the linked claim.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MarketEvidence {
    /// Unique identifier
    pub id: String,

    /// The claim this evidence applies to
    pub claim_id: String,

    /// The market that generated this evidence
    pub market_id: String,

    /// Type of evidence from the market
    pub evidence_type: MarketEvidenceType,

    /// Raw confidence from market resolution
    pub confidence: f64,

    /// Number of oracles that participated
    pub oracle_count: u32,

    /// MATL-weighted confidence (accounts for oracle trust)
    pub matl_weighted_confidence: f64,

    /// Total stake in the market
    pub total_stake: u64,

    /// When this evidence was recorded
    pub timestamp: Timestamp,
}

/// Types of evidence that can come from markets
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MarketEvidenceType {
    /// A prediction market resolved with this outcome
    PredictionResolved {
        outcome: String,
        winning_probability: f64,
    },

    /// Expert consensus reached through oracle voting
    ConsensusReached {
        consensus_level: f64,
        agreement_rate: f64,
    },

    /// Specific expert votes on the claim
    ExpertVotes {
        expert_count: u32,
        average_confidence: f64,
        domain_relevance: f64,
    },

    /// Verification through cryptographic proof
    CryptographicVerification {
        proof_type: String,
        verification_status: bool,
    },

    /// Verification through external data source
    DataSourceVerification {
        source: String,
        match_confidence: f64,
    },
}

// ============================================================================
// CLAIM AS MARKET EVIDENCE
// ============================================================================

/// Tracks when a knowledge claim is used as evidence in a prediction
///
/// When predictors reference knowledge claims to support their predictions,
/// this creates a bidirectional link that can affect claim credibility
/// based on prediction outcomes.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClaimAsMarketEvidence {
    /// Unique identifier
    pub id: String,

    /// The claim being referenced
    pub claim_id: String,

    /// The market containing the prediction
    pub market_id: String,

    /// The specific prediction referencing this claim
    pub prediction_id: String,

    /// How the claim is being used
    pub usage: ClaimUsageType,

    /// Relevance score assigned by the predictor
    pub relevance_score: f64,

    /// Agent who made the reference
    pub referenced_by: AgentPubKey,

    /// When this reference was created
    pub timestamp: Timestamp,
}

/// How a claim is used in a prediction
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ClaimUsageType {
    /// Claim supports the predicted outcome
    SupportingEvidence,

    /// Claim contradicts the predicted outcome
    ContradictingEvidence,

    /// Claim provides context but doesn't directly support/contradict
    ContextualReference,

    /// Claim is part of the resolution criteria
    ResolutionCriteria,

    /// Claim establishes a prerequisite condition
    PrerequisiteCondition,
}

// ============================================================================
// MARKET VALUE ASSESSMENT
// ============================================================================

/// Assessment of whether creating a verification market is worthwhile
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MarketValueAssessment {
    /// Unique identifier
    pub id: String,

    /// The claim being assessed
    pub claim_id: String,

    /// Information value score (how much would verification teach us?)
    pub information_value: f64,

    /// Decision impact score (how many decisions depend on this?)
    pub decision_impact: f64,

    /// Current uncertainty level
    pub uncertainty: f64,

    /// Number of claims that depend on this one
    pub dependency_count: u32,

    /// Estimated market participation
    pub estimated_participation: u32,

    /// Estimated verification cost
    pub verification_cost: f64,

    /// Recommendation based on analysis
    pub recommendation: MarketRecommendation,

    /// When this assessment was made
    pub assessed_at: Timestamp,

    /// Assessment expires after this time
    pub expires_at: Timestamp,
}

/// Recommendation for market creation
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MarketRecommendation {
    /// Create the market now
    CreateMarket {
        reason: String,
        suggested_bounty: u64,
        suggested_deadline_days: u32,
    },

    /// Wait for more evidence before creating market
    WaitForEvidence {
        needed_evidence: Vec<String>,
        estimated_wait_days: u32,
    },

    /// Not recommended to create market
    NotRecommended {
        reason: String,
        alternative_action: Option<String>,
    },

    /// Escalate to governance for decision
    EscalateToGovernance {
        reason: String,
    },
}

// ============================================================================
// BRIDGE EVENTS
// ============================================================================

/// Events exchanged between Knowledge and Epistemic Markets
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MarketsIntegrationEvent {
    /// Request to create verification market (Knowledge → Markets)
    VerificationRequested {
        request_id: String,
        claim_id: String,
        question: String,
        target_epistemic: EpistemicTarget,
        bounty: u64,
        deadline: Timestamp,
    },

    /// Market was created (Markets → Knowledge)
    MarketCreated {
        request_id: String,
        market_id: String,
        question: String,
    },

    /// Market status update (Markets → Knowledge)
    MarketStatusUpdate {
        market_id: String,
        current_probability: f64,
        participant_count: u32,
        total_stake: u64,
    },

    /// Market resolved (Markets → Knowledge)
    MarketResolved {
        market_id: String,
        claim_id: String,
        outcome: String,
        confidence: f64,
        oracle_count: u32,
        matl_weighted_confidence: f64,
    },

    /// Claim referenced in prediction (Markets → Knowledge)
    ClaimReferenced {
        claim_id: String,
        market_id: String,
        prediction_id: String,
        usage: ClaimUsageType,
    },

    /// Claim update completed (Knowledge → Markets)
    ClaimUpdated {
        claim_id: String,
        market_id: String,
        new_epistemic: EpistemicPosition,
    },
}

// ============================================================================
// ENTRY AND LINK TYPES
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    VerificationMarketRequest(VerificationMarketRequest),
    MarketEvidence(MarketEvidence),
    ClaimAsMarketEvidence(ClaimAsMarketEvidence),
    MarketValueAssessment(MarketValueAssessment),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Claim → VerificationMarketRequest
    ClaimToMarketRequest,

    /// VerificationMarketRequest → Market (external reference)
    RequestToMarket,

    /// Claim → MarketEvidence
    ClaimToMarketEvidence,

    /// Market → Claim (external to internal)
    MarketToClaim,

    /// Claim → ClaimAsMarketEvidence (claim used in prediction)
    ClaimUsedInPrediction,

    /// Prediction → Claim (external to internal)
    PredictionToClaim,

    /// Claim → MarketValueAssessment
    ClaimToValueAssessment,

    /// Agent → VerificationMarketRequest (requests by agent)
    AgentToRequest,

    /// Tag-based indexing for requests
    RequestByStatus,
    RequestByClaimId,
}

// ============================================================================
// VALIDATION
// ============================================================================

/// Validate VerificationMarketRequest
pub fn validate_verification_market_request(
    request: VerificationMarketRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Validate claim_id is not empty
    if request.claim_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "claim_id cannot be empty".to_string(),
        ));
    }

    // Validate question is not empty
    if request.question.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "question cannot be empty".to_string(),
        ));
    }

    // Validate epistemic values are in range
    if request.current_epistemic.empirical < 0.0
        || request.current_epistemic.empirical > 1.0
        || request.current_epistemic.normative < 0.0
        || request.current_epistemic.normative > 1.0
        || request.current_epistemic.mythic < 0.0
        || request.current_epistemic.mythic > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Epistemic values must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Validate target epistemic values if provided
    if let Some(e) = request.target_epistemic.empirical {
        if e < 0.0 || e > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Target empirical must be between 0.0 and 1.0".to_string(),
            ));
        }
    }

    // Validate deadline is in the future
    // (Skip for now as we don't have access to sys_time in integrity)

    Ok(ValidateCallbackResult::Valid)
}

/// Validate MarketEvidence
pub fn validate_market_evidence(evidence: MarketEvidence) -> ExternResult<ValidateCallbackResult> {
    // Validate claim_id is not empty
    if evidence.claim_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "claim_id cannot be empty".to_string(),
        ));
    }

    // Validate market_id is not empty
    if evidence.market_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "market_id cannot be empty".to_string(),
        ));
    }

    // Validate confidence is in range
    if evidence.confidence < 0.0 || evidence.confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "confidence must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Validate matl_weighted_confidence is in range
    if evidence.matl_weighted_confidence < 0.0 || evidence.matl_weighted_confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "matl_weighted_confidence must be between 0.0 and 1.0".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate ClaimAsMarketEvidence
pub fn validate_claim_as_market_evidence(
    reference: ClaimAsMarketEvidence,
) -> ExternResult<ValidateCallbackResult> {
    // Validate claim_id is not empty
    if reference.claim_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "claim_id cannot be empty".to_string(),
        ));
    }

    // Validate market_id is not empty
    if reference.market_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "market_id cannot be empty".to_string(),
        ));
    }

    // Validate prediction_id is not empty
    if reference.prediction_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "prediction_id cannot be empty".to_string(),
        ));
    }

    // Validate relevance_score is in range
    if reference.relevance_score < 0.0 || reference.relevance_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "relevance_score must be between 0.0 and 1.0".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::VerificationMarketRequest(request) => {
                        validate_verification_market_request(request)
                    }
                    EntryTypes::MarketEvidence(evidence) => validate_market_evidence(evidence),
                    EntryTypes::ClaimAsMarketEvidence(reference) => {
                        validate_claim_as_market_evidence(reference)
                    }
                    EntryTypes::MarketValueAssessment(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
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
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
