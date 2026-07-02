// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fact-Check Integrity Zome
//!
//! Defines entry types for the fact-checking API that external hApps
//! (Media, Governance, Justice) use to verify statements against the
//! knowledge graph.
//!
//! ## Verdicts
//! - True: Statement fully supported by high-E claims
//! - MostlyTrue: Statement supported with minor qualifications
//! - Mixed: Conflicting evidence in knowledge graph
//! - MostlyFalse: Statement contradicted with exceptions
//! - False: Statement directly contradicted by high-E claims
//! - Unverifiable: Statement cannot be verified (low E-level domain)
//! - InsufficientEvidence: Not enough claims to make determination
//!
//! Updated to use HDI 0.7 patterns

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// EPISTEMIC TYPES
// ============================================================================

/// Epistemic position in the 3D E-N-M space
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpistemicPosition {
    pub empirical: f64,
    pub normative: f64,
    pub mythic: f64,
}

// ============================================================================
// FACT CHECK REQUEST
// ============================================================================

/// Request from an external hApp to fact-check a statement
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FactCheckRequest {
    /// Unique identifier
    pub id: String,

    /// Source hApp making the request
    pub source_happ: String,

    /// Statement to fact-check
    pub statement: String,

    /// Additional context for the statement
    pub context: Option<String>,

    /// Subject of the statement (person, organization, topic)
    pub subject: Option<String>,

    /// Topics/tags related to the statement
    pub topics: Vec<String>,

    /// Minimum empirical level required for evidence
    pub min_epistemic_e: f64,

    /// Minimum normative level required for evidence
    pub min_epistemic_n: f64,

    /// Agent who made the request
    pub requester: AgentPubKey,

    /// When the request was made
    pub requested_at: Timestamp,

    /// Current processing status
    pub status: FactCheckStatus,
}

/// Status of a fact check request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FactCheckStatus {
    /// Request received, queued for processing
    Pending,

    /// Currently being processed
    Processing,

    /// Fact check completed
    Completed,

    /// Failed to complete
    Failed { reason: String },
}

// ============================================================================
// FACT CHECK RESULT
// ============================================================================

/// Structured fact check result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FactCheckResult {
    /// Unique identifier
    pub id: String,

    /// Reference to the original request
    pub request_id: String,

    /// The statement that was checked
    pub statement: String,

    /// Overall verdict
    pub verdict: FactCheckVerdict,

    /// Confidence in the verdict (0.0-1.0)
    pub verdict_confidence: f64,

    /// Claims that support the statement
    pub supporting_claims: Vec<ClaimSummary>,

    /// Claims that contradict the statement
    pub contradicting_claims: Vec<ClaimSummary>,

    /// Claims that provide context
    pub related_claims: Vec<ClaimSummary>,

    /// Aggregate epistemic position of evidence
    pub aggregate_epistemic: EpistemicPosition,

    /// Overall credibility score of evidence
    pub credibility_score: f64,

    /// How diverse are the sources?
    pub source_diversity: f64,

    /// Total evidence considered
    pub evidence_count: u32,

    /// When the fact check was completed
    pub processed_at: Timestamp,

    /// How long the check took in milliseconds
    pub processing_time_ms: u64,

    /// Suggested follow-up actions
    pub suggested_actions: Vec<SuggestedAction>,
}

/// Fact check verdict
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FactCheckVerdict {
    /// Statement is fully supported by verified claims
    True,

    /// Statement is mostly supported with minor qualifications
    MostlyTrue,

    /// Evidence is mixed - some support, some contradiction
    Mixed,

    /// Statement is mostly contradicted with exceptions
    MostlyFalse,

    /// Statement is directly contradicted by verified claims
    False,

    /// Statement cannot be verified (subjective domain, low E-level)
    Unverifiable,

    /// Not enough evidence to make a determination
    InsufficientEvidence,

    /// Statement is outdated - evidence has changed
    Outdated {
        /// When the statement was accurate
        was_accurate_until: Timestamp,
    },

    /// Statement requires context to evaluate
    NeedsContext {
        /// What context is needed
        context_needed: String,
    },
}

/// Summary of a claim used in fact-checking
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ClaimSummary {
    /// Claim identifier
    pub claim_id: String,

    /// Snippet of claim content
    pub content_snippet: String,

    /// Epistemic position
    pub epistemic: EpistemicPosition,

    /// Credibility score
    pub credibility: f64,

    /// How relevant to the statement (0.0-1.0)
    pub relevance_score: f64,

    /// Relationship to the statement
    pub relationship: ClaimRelationship,

    /// Source of the claim
    pub source: Option<String>,

    /// When the claim was created
    pub created_at: Timestamp,
}

/// Relationship between claim and statement being checked
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ClaimRelationship {
    /// Claim directly supports the statement
    DirectSupport,

    /// Claim implies the statement is true
    IndirectSupport,

    /// Claim directly contradicts the statement
    DirectContradiction,

    /// Claim implies the statement is false
    IndirectContradiction,

    /// Claim provides context without supporting/contradicting
    Context,

    /// Claim is tangentially related
    Related,
}

/// Suggested follow-up action
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SuggestedAction {
    /// Type of action
    pub action_type: SuggestedActionType,

    /// Description of the action
    pub description: String,

    /// Priority (1-5, 1 being highest)
    pub priority: u8,
}

/// Types of suggested actions
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SuggestedActionType {
    /// Request verification market for key claim
    RequestVerificationMarket { claim_id: String },

    /// Submit new claim with evidence
    SubmitClaim,

    /// Challenge an existing claim
    ChallengeClaim { claim_id: String },

    /// Add more context
    AddContext,

    /// Wait for pending verification
    WaitForVerification { claim_id: String },

    /// Escalate to governance
    EscalateToGovernance,
}

// ============================================================================
// QUERY TEMPLATES
// ============================================================================

/// Saved query template for common fact-checks
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FactCheckTemplate {
    /// Unique identifier
    pub id: String,

    /// Template name
    pub name: String,

    /// Description of what this template checks
    pub description: String,

    /// Topics this template applies to
    pub topics: Vec<String>,

    /// Required subjects to match
    pub subject_patterns: Vec<String>,

    /// Minimum epistemic requirements
    pub min_epistemic_e: f64,
    pub min_epistemic_n: f64,

    /// Custom scoring weights
    pub scoring_weights: ScoringWeights,

    /// Who created this template
    pub created_by: AgentPubKey,

    /// When created
    pub created_at: Timestamp,

    /// Usage count
    pub usage_count: u32,
}

/// Custom scoring weights for fact-check evaluation
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ScoringWeights {
    /// Weight for empirical evidence (default: 0.4)
    pub empirical_weight: f64,

    /// Weight for normative coherence (default: 0.3)
    pub normative_weight: f64,

    /// Weight for source diversity (default: 0.2)
    pub diversity_weight: f64,

    /// Weight for recency (default: 0.1)
    pub recency_weight: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            empirical_weight: 0.4,
            normative_weight: 0.3,
            diversity_weight: 0.2,
            recency_weight: 0.1,
        }
    }
}

// ============================================================================
// BATCH OPERATIONS
// ============================================================================

/// Batch fact check result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BatchFactCheckResult {
    /// Unique identifier
    pub id: String,

    /// Source hApp
    pub source_happ: String,

    /// Individual results
    pub results: Vec<String>, // References to FactCheckResult IDs

    /// Summary statistics
    pub summary: BatchSummary,

    /// When completed
    pub completed_at: Timestamp,
}

/// Summary of batch fact check
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BatchSummary {
    /// Total statements checked
    pub total_checked: u32,

    /// Count by verdict
    pub true_count: u32,
    pub mostly_true_count: u32,
    pub mixed_count: u32,
    pub mostly_false_count: u32,
    pub false_count: u32,
    pub unverifiable_count: u32,
    pub insufficient_count: u32,

    /// Average credibility
    pub average_credibility: f64,

    /// Average processing time
    pub average_processing_time_ms: u64,
}

// ============================================================================
// ENTRY AND LINK TYPES
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    FactCheckRequest(FactCheckRequest),
    FactCheckResult(FactCheckResult),
    FactCheckTemplate(FactCheckTemplate),
    BatchFactCheckResult(BatchFactCheckResult),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// hApp → FactCheckRequest
    HappToFactCheck,

    /// Subject → FactCheckRequest
    SubjectToFactCheck,

    /// Topic → FactCheckRequest
    TopicToFactCheck,

    /// Request → Result
    RequestToResult,

    /// Template → Request (template used)
    TemplateToRequest,

    /// Index by verdict
    VerdictIndex,

    /// Batch → Results
    BatchToResults,
}

// ============================================================================
// VALIDATION
// ============================================================================

/// Validate FactCheckRequest
pub fn validate_fact_check_request(
    request: FactCheckRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Statement cannot be empty
    if request.statement.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "statement cannot be empty".to_string(),
        ));
    }

    // Statement should have reasonable length
    if request.statement.len() > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "statement too long (max 10000 chars)".to_string(),
        ));
    }

    // source_happ cannot be empty
    if request.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "source_happ cannot be empty".to_string(),
        ));
    }

    // Epistemic requirements must be valid
    if request.min_epistemic_e < 0.0 || request.min_epistemic_e > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "min_epistemic_e must be between 0.0 and 1.0".to_string(),
        ));
    }

    if request.min_epistemic_n < 0.0 || request.min_epistemic_n > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "min_epistemic_n must be between 0.0 and 1.0".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate FactCheckResult
pub fn validate_fact_check_result(result: FactCheckResult) -> ExternResult<ValidateCallbackResult> {
    // Must reference a request
    if result.request_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "request_id cannot be empty".to_string(),
        ));
    }

    // Confidence must be valid
    if result.verdict_confidence < 0.0 || result.verdict_confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "verdict_confidence must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Credibility must be valid
    if result.credibility_score < 0.0 || result.credibility_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "credibility_score must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Source diversity must be valid
    if result.source_diversity < 0.0 || result.source_diversity > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "source_diversity must be between 0.0 and 1.0".to_string(),
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
                    EntryTypes::FactCheckRequest(request) => validate_fact_check_request(request),
                    EntryTypes::FactCheckResult(result) => validate_fact_check_result(result),
                    EntryTypes::FactCheckTemplate(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::BatchFactCheckResult(_) => Ok(ValidateCallbackResult::Valid),
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
