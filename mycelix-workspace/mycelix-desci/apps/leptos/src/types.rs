// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Data types matching the DeSci REST API models.

use serde::{Deserialize, Serialize};

/// Epistemic tier (E0-E4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicTier {
    E0,
    E1,
    E2,
    E3,
    E4,
}

impl EpistemicTier {
    pub fn label(&self) -> &'static str {
        match self {
            Self::E0 => "E0: Unverified",
            Self::E1 => "E1: Testimonial",
            Self::E2 => "E2: Verifiable",
            Self::E3 => "E3: Reproducible",
            Self::E4 => "E4: Peer Reviewed",
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::E0 => "e0",
            Self::E1 => "e1",
            Self::E2 => "e2",
            Self::E3 => "e3",
            Self::E4 => "e4",
        }
    }

    pub fn all() -> [Self; 5] {
        [Self::E0, Self::E1, Self::E2, Self::E3, Self::E4]
    }
}

impl std::fmt::Display for EpistemicTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Claim content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimContent {
    pub dataset_hash: String,
    pub description: String,
    pub category: String,
    pub keywords: Vec<String>,
    pub storage_ref: Option<String>,
    pub reproducibility_score: Option<f64>,
    pub license: Option<String>,
}

/// API claim response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimResponse {
    pub id: String,
    pub tier: EpistemicTier,
    pub content: ClaimContent,
    pub creator: String,
    pub created_at: String,
    pub verifications_count: usize,
    pub provenance_count: usize,
}

/// Query request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryRequest {
    pub category: Option<String>,
    pub keywords: Option<Vec<String>>,
    pub tier: Option<EpistemicTier>,
    pub creator: Option<String>,
    pub page: Option<usize>,
    pub page_size: Option<usize>,
}

/// Query response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub results: Vec<ClaimResponse>,
    pub total_count: usize,
    pub page: usize,
    pub page_size: usize,
    pub total_pages: usize,
}

/// Stats response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsResponse {
    pub total_claims: usize,
    pub claims_by_tier: std::collections::HashMap<String, usize>,
    pub total_categories: usize,
    pub total_keywords: usize,
}

/// Create claim request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateClaimRequest {
    pub tier: EpistemicTier,
    pub content: ClaimContent,
    pub creator: String,
}

/// Discovery result from physics catalog search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResult {
    pub name: String,
    pub domain: String,
    pub score: f32,
    pub latex: String,
}

/// LEM classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LEMClassification {
    pub empirical: u8,
    pub normative: u8,
    pub materiality: u8,
}

// ── Verification ──

/// Verification record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verification {
    pub verifier: String,
    pub timestamp: String,
    pub evidence: String,
    pub methodology_match: String,
    pub outcome: String,
}

// ── Trust ──

/// Trust score for a participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScore {
    pub participant: String,
    pub score: f64,
    pub confidence: f64,
    pub interaction_count: usize,
    pub expertise_domains: Vec<String>,
}

// ── Reproducibility ──

/// Replication attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationAttempt {
    pub claim_id: String,
    pub replicator: String,
    pub status: String,
    pub outcome: String,
    pub methodology_match: String,
    pub sample_size: Option<usize>,
    pub timestamp: String,
}

/// Reproducibility stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityStats {
    pub claim_id: String,
    pub score: f64,
    pub confidence: f64,
    pub total_attempts: usize,
    pub successful: usize,
    pub failed: usize,
}

// ── Prediction Markets ──

/// Prediction market for a claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMarket {
    pub claim_id: String,
    pub question: String,
    pub current_probability: f64,
    pub total_staked: f64,
    pub positions: usize,
    pub status: String,
    pub resolution_method: String,
}

// ── Citations ──

/// Citation link between claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub from_claim: String,
    pub to_claim: String,
    pub citation_type: String,
    pub strength: f64,
}

/// Citation metrics for a claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationMetrics {
    pub claim_id: String,
    pub pagerank: f64,
    pub h_index: usize,
    pub citation_count: usize,
    pub is_foundational: bool,
}

// ── Consensus ──

/// Consensus state for a claim over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusSnapshot {
    pub timestamp: String,
    pub support_ratio: f64,
    pub verification_count: usize,
    pub state: String,
    pub trend: String,
}
