// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! API request and response models

use chrono::{DateTime, Utc};
use mycelix_desci_core::claims::EpistemicTier;
use mycelix_desci_core::query::SortOrder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

// =============================================================================
// Claims API Models
// =============================================================================

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CreateClaimRequest {
    pub tier: EpistemicTier,
    pub content: ClaimContentRequest,
    pub creator: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ClaimContentRequest {
    pub dataset_hash: String,
    pub description: String,
    pub category: String,
    pub keywords: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reproducibility_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ClaimResponse {
    pub id: Uuid,
    pub tier: EpistemicTier,
    pub content: ClaimContentRequest,
    pub creator: String,
    pub created_at: DateTime<Utc>,
    pub verifications_count: usize,
    pub provenance_count: usize,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AddVerificationRequest {
    pub verifier: String,
    pub signature: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AddProvenanceRequest {
    pub source: String,
    pub source_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

// =============================================================================
// Query API Models
// =============================================================================

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct QueryRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keywords: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<EpistemicTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub creator: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<QuerySort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_size: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct QuerySort {
    pub field: String,
    pub order: SortOrder,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct QueryResponse {
    pub results: Vec<ClaimResponse>,
    pub total_count: usize,
    pub page: usize,
    pub page_size: usize,
    pub total_pages: usize,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CategoriesResponse {
    pub categories: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct QueryStatsResponse {
    pub total_claims: usize,
    pub claims_by_tier: std::collections::HashMap<String, usize>,
    pub total_categories: usize,
    pub total_keywords: usize,
}

// =============================================================================
// Trust API Models
// =============================================================================

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct UpdateTrustScoreRequest {
    pub delta: f64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct TrustScoreResponse {
    pub participant: String,
    pub score: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct TrustStatsResponse {
    pub total_participants: usize,
    pub average_score: f64,
    pub median_score: f64,
    pub highest_score: f64,
    pub lowest_score: f64,
}

// =============================================================================
// System API Models
// =============================================================================

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub checks: Vec<HealthCheck>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct HealthCheck {
    pub name: String,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MetricsResponse {
    pub uptime_seconds: u64,
    pub total_claims: usize,
    pub total_participants: usize,
    pub queries_executed: u64,
    pub claims_created: u64,
    pub verifications_added: u64,
    pub average_response_time_ms: f64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct VersionResponse {
    pub version: String,
    pub build_date: String,
    pub git_commit: String,
    pub rust_version: String,
}

// =============================================================================
// Conversion implementations
// =============================================================================

impl From<&mycelix_desci_core::claims::DesciClaim> for ClaimResponse {
    fn from(claim: &mycelix_desci_core::claims::DesciClaim) -> Self {
        Self {
            id: claim.id,
            tier: claim.epistemic_tier,
            content: ClaimContentRequest {
                dataset_hash: claim.content.dataset_hash.clone(),
                description: claim.content.description.clone(),
                category: claim.content.category.clone(),
                keywords: claim.content.keywords.clone(),
                storage_ref: claim.content.storage_ref.clone(),
                reproducibility_score: claim.content.reproducibility_score,
                license: claim.content.license.clone(),
            },
            creator: claim.creator.clone(),
            created_at: claim.created_at,
            verifications_count: claim.verifications.len(),
            provenance_count: claim.provenance.len(),
        }
    }
}
