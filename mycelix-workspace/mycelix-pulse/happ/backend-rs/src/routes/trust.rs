// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust score routes
//!
//! Provides access to MATL-based trust scoring with caching.

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};

use crate::error::AppResult;
use crate::middleware::AuthenticatedUser;
use crate::routes::AppState;
use crate::types::{ApiError, TrustScoreInfo};

/// Create trust routes
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/score/:did", get(get_trust_score))
        .route("/scores", post(get_trust_scores_batch))
        .route("/byzantine/:did", get(check_byzantine))
        .route("/cross-happ/:did", get(get_cross_happ_reputation))
        .route("/cache/stats", get(get_cache_stats))
        .route("/cache/invalidate/:did", post(invalidate_cache))
}

/// Get trust score for a DID
#[utoipa::path(
    get,
    path = "/trust/score/{did}",
    tag = "trust",
    params(
        ("did" = String, Path, description = "DID to get trust score for")
    ),
    responses(
        (status = 200, description = "Trust score info", body = TrustScoreInfo),
        (status = 401, description = "Not authenticated", body = ApiError),
        (status = 404, description = "DID not found", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_trust_score(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
    Path(did): Path<String>,
) -> AppResult<Json<TrustScoreInfo>> {
    tracing::debug!("Fetching trust score for {}", did);

    let cached = state.trust_cache.get_trust(&did).await?;

    Ok(Json(TrustScoreInfo {
        did: cached.did,
        score: cached.score,
        last_updated: chrono::DateTime::from_timestamp(
            cached.cached_at.elapsed().as_secs() as i64,
            0,
        )
        .unwrap_or_else(chrono::Utc::now),
        source: "matl".to_string(),
        is_byzantine: cached.is_byzantine,
        interaction_count: cached.interaction_count,
    }))
}

/// Get trust scores for multiple DIDs
#[utoipa::path(
    post,
    path = "/trust/scores",
    tag = "trust",
    request_body = Vec<String>,
    responses(
        (status = 200, description = "Trust scores for requested DIDs", body = Vec<TrustScoreInfo>),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_trust_scores_batch(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
    Json(dids): Json<Vec<String>>,
) -> AppResult<Json<Vec<TrustScoreInfo>>> {
    tracing::debug!("Fetching batch trust scores for {} DIDs", dids.len());

    let results = state.trust_cache.get_trust_batch(&dids).await;

    let scores: Vec<TrustScoreInfo> = results
        .into_iter()
        .filter_map(|(did, result)| {
            result.ok().map(|cached| TrustScoreInfo {
                did: cached.did,
                score: cached.score,
                last_updated: chrono::Utc::now(),
                source: "matl".to_string(),
                is_byzantine: cached.is_byzantine,
                interaction_count: cached.interaction_count,
            })
        })
        .collect();

    Ok(Json(scores))
}

/// Check if a DID is flagged as Byzantine
#[utoipa::path(
    get,
    path = "/trust/byzantine/{did}",
    tag = "trust",
    params(
        ("did" = String, Path, description = "DID to check for Byzantine behavior")
    ),
    responses(
        (status = 200, description = "Byzantine check result", body = ByzantineCheckResult),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn check_byzantine(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
    Path(did): Path<String>,
) -> AppResult<Json<ByzantineCheckResult>> {
    let is_byzantine = state.trust_cache.is_byzantine(&did).await?;
    let trust = state.trust_cache.get_trust(&did).await?;

    Ok(Json(ByzantineCheckResult {
        did,
        is_byzantine,
        trust_score: trust.score,
        threshold: state.config.byzantine_threshold,
    }))
}

/// Get cross-hApp reputation for a DID
#[utoipa::path(
    get,
    path = "/trust/cross-happ/{did}",
    tag = "trust",
    params(
        ("did" = String, Path, description = "DID to get cross-hApp reputation for")
    ),
    responses(
        (status = 200, description = "Cross-hApp reputation", body = CrossHappReputationResponse),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_cross_happ_reputation(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
    Path(did): Path<String>,
) -> AppResult<Json<CrossHappReputationResponse>> {
    tracing::debug!("Fetching cross-hApp reputation for {}", did);

    let reputation = state.holochain.get_cross_happ_reputation(&did).await?;

    Ok(Json(CrossHappReputationResponse {
        did: reputation.did,
        local_score: reputation.local_score,
        aggregate_score: reputation.aggregate,
        happ_scores: reputation
            .cross_happ_scores
            .into_iter()
            .map(|s| HappScoreInfo {
                happ_id: s.happ_id,
                score: s.score,
            })
            .collect(),
    }))
}

/// Get cache statistics
#[utoipa::path(
    get,
    path = "/trust/cache/stats",
    tag = "trust",
    responses(
        (status = 200, description = "Cache statistics", body = CacheStatsResponse),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_cache_stats(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
) -> AppResult<Json<CacheStatsResponse>> {
    let stats = state.trust_cache.stats();

    Ok(Json(CacheStatsResponse {
        entry_count: stats.entry_count,
        weighted_size: stats.weighted_size,
        ttl_seconds: state.config.trust_cache_ttl_secs,
        max_entries: state.config.trust_cache_max_entries,
    }))
}

/// Invalidate cache for a specific DID
#[utoipa::path(
    post,
    path = "/trust/cache/invalidate/{did}",
    tag = "trust",
    params(
        ("did" = String, Path, description = "DID to invalidate cache for")
    ),
    responses(
        (status = 200, description = "Cache invalidated"),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn invalidate_cache(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
    Path(did): Path<String>,
) -> AppResult<Json<serde_json::Value>> {
    state.trust_cache.invalidate(&did).await;

    Ok(Json(serde_json::json!({
        "invalidated": true,
        "did": did
    })))
}

// Response types

/// Byzantine check result
#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct ByzantineCheckResult {
    /// The DID checked
    pub did: String,
    /// Whether flagged as Byzantine
    pub is_byzantine: bool,
    /// Current trust score
    pub trust_score: f64,
    /// Byzantine threshold used
    pub threshold: f64,
}

/// Cross-hApp reputation response
#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct CrossHappReputationResponse {
    /// The DID
    pub did: String,
    /// Local trust score in this hApp
    pub local_score: f64,
    /// Aggregate score across all hApps
    pub aggregate_score: f64,
    /// Per-hApp scores
    pub happ_scores: Vec<HappScoreInfo>,
}

/// Per-hApp score info
#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct HappScoreInfo {
    /// hApp identifier
    pub happ_id: String,
    /// Trust score in this hApp
    pub score: f64,
}

/// Cache statistics response
#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct CacheStatsResponse {
    /// Number of cached entries
    pub entry_count: u64,
    /// Weighted size of cache
    pub weighted_size: u64,
    /// TTL in seconds
    pub ttl_seconds: u64,
    /// Maximum entries allowed
    pub max_entries: u64,
}
