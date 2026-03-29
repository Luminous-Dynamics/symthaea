// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CLI commands

pub mod claims;
pub mod query;
pub mod system;
pub mod trust;

// Shared types used across commands
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct ClaimResponse {
    pub id: Uuid,
    pub tier: String,
    pub content: ClaimContent,
    pub creator: String,
    pub created_at: String,
    pub verifications_count: usize,
    pub provenance_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClaimContent {
    pub dataset_hash: String,
    pub description: String,
    pub category: String,
    pub keywords: Vec<String>,
    pub storage_ref: Option<String>,
    pub reproducibility_score: Option<f64>,
    pub license: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub checks: Vec<HealthCheck>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsResponse {
    pub uptime_seconds: u64,
    pub total_claims: usize,
    pub total_participants: usize,
    pub queries_executed: u64,
    pub claims_created: u64,
    pub verifications_added: u64,
    pub average_response_time_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VersionResponse {
    pub version: String,
    pub build_date: String,
    pub git_commit: String,
    pub rust_version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrustScoreResponse {
    pub participant: String,
    pub score: f64,
    pub last_updated: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResponse {
    pub results: Vec<ClaimResponse>,
    pub total_count: usize,
    pub page: usize,
    pub page_size: usize,
    pub total_pages: usize,
}
