// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM-safe types for the Knowledge cluster frontend.
//!
//! E/N/M classification re-exported from the shared `mycelix-claim-types` crate
//! for ecosystem-wide consistency (Prism browser uses the same types).

use serde::{Deserialize, Serialize};

// Re-export unified epistemic types — single source of truth
pub use mycelix_claim_types::{
    EmpiricalLevel, NormativeLevel, MaterialityLevel,
    ClaimType, EpistemicClassification,
};

/// Knowledge-specific view of an epistemic classification.
/// Wraps the unified type for backwards compatibility.
pub type EpistemicClassificationView = EpistemicClassification;

// ---------------------------------------------------------------------------
// Claims (ClaimType re-exported from mycelix-claim-types with label() + all())
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClaimView {
    pub id: String,
    pub content: String,
    pub classification: EpistemicClassificationView,
    pub author: String,
    pub sources: Vec<String>,
    pub tags: Vec<String>,
    pub claim_type: ClaimType,
    pub created: i64,
    pub version: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceView {
    pub id: String,
    pub claim_id: String,
    pub evidence_type: String,
    pub source_uri: String,
    pub content: String,
    pub strength: f64,
    pub submitted_by: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChallengeStatus { Pending, UnderReview, Accepted, Rejected, Withdrawn }
impl ChallengeStatus {
    pub fn label(&self) -> &'static str {
        match self { Self::Pending => "Pending", Self::UnderReview => "Under Review", Self::Accepted => "Accepted", Self::Rejected => "Rejected", Self::Withdrawn => "Withdrawn" }
    }
    pub fn css_class(&self) -> &'static str {
        match self { Self::Pending => "warning", Self::UnderReview => "info", Self::Accepted => "success", Self::Rejected => "error", Self::Withdrawn => "muted" }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChallengeView {
    pub id: String,
    pub claim_id: String,
    pub challenger: String,
    pub reason: String,
    pub status: ChallengeStatus,
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationshipType { Supports, Contradicts, DerivedFrom, ExampleOf, Generalizes, PartOf, Causes, RelatedTo, Equivalent }
impl RelationshipType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Supports => "Supports", Self::Contradicts => "Contradicts", Self::DerivedFrom => "Derived From",
            Self::ExampleOf => "Example Of", Self::Generalizes => "Generalizes", Self::PartOf => "Part Of",
            Self::Causes => "Causes", Self::RelatedTo => "Related To", Self::Equivalent => "Equivalent",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelationshipView {
    pub id: String,
    pub source: String,
    pub target: String,
    pub relationship_type: RelationshipType,
    pub weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphStatsView {
    pub relationship_count: u64,
}

// ---------------------------------------------------------------------------
// Fact-checking
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FactCheckVerdict { True, MostlyTrue, Mixed, MostlyFalse, False, Unverifiable, InsufficientEvidence }
impl FactCheckVerdict {
    pub fn label(&self) -> &'static str {
        match self {
            Self::True => "True", Self::MostlyTrue => "Mostly True", Self::Mixed => "Mixed",
            Self::MostlyFalse => "Mostly False", Self::False => "False",
            Self::Unverifiable => "Unverifiable", Self::InsufficientEvidence => "Insufficient Evidence",
        }
    }
    pub fn css_class(&self) -> &'static str {
        match self {
            Self::True => "success", Self::MostlyTrue => "success", Self::Mixed => "warning",
            Self::MostlyFalse => "error", Self::False => "error",
            Self::Unverifiable => "muted", Self::InsufficientEvidence => "muted",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactCheckResultView {
    pub id: String,
    pub statement: String,
    pub verdict: FactCheckVerdict,
    pub verdict_confidence: f64,
    pub credibility_score: f64,
    pub evidence_count: u32,
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum InferenceType { ImpliedRelation, Contradiction, Pattern, Prediction, Synthesis, Similarity, Anomaly, CredibilityAssessment }
impl InferenceType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::ImpliedRelation => "Implied Relation", Self::Contradiction => "Contradiction",
            Self::Pattern => "Pattern", Self::Prediction => "Prediction",
            Self::Synthesis => "Synthesis", Self::Similarity => "Similarity",
            Self::Anomaly => "Anomaly", Self::CredibilityAssessment => "Credibility",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceView {
    pub id: String,
    pub inference_type: InferenceType,
    pub source_claims: Vec<String>,
    pub conclusion: String,
    pub confidence: f64,
    pub reasoning: String,
    pub verified: bool,
}

// ---------------------------------------------------------------------------
// Author Reputation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthorReputationView {
    pub author_did: String,
    pub overall_score: f64,
    pub historical_accuracy: f64,
    pub claims_authored: u32,
    pub claims_verified_true: u32,
    pub claims_verified_false: u32,
}
