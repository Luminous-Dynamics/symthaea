// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge routes for cross-hApp communication
//!
//! Provides API endpoints for:
//! - Querying cross-hApp identity
//! - Getting aggregated reputation scores
//! - Reporting reputation interactions
//! - Bridge status and statistics

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::error::AppResult;
use crate::middleware::AuthenticatedUser;
use crate::routes::AppState;
use crate::services::bridge::{
    BridgeCacheStats, CrossHappIdentity, CrossHappReputation,
    HappId, HappReputationScore, has_sufficient_reputation, spam_likelihood,
};
use crate::types::ApiError;

/// Create bridge routes
pub fn router() -> Router<AppState> {
    Router::new()
        // Status and diagnostics
        .route("/status", get(get_bridge_status))
        .route("/stats", get(get_bridge_stats))
        // Identity queries
        .route("/identity/:did", get(query_identity))
        // Reputation queries
        .route("/reputation/:did", get(get_reputation))
        .route("/reputation/:did/full", get(get_full_reputation))
        .route("/reputation/:did/check", get(check_reputation))
        // Reputation reporting
        .route("/report/positive", post(report_positive))
        .route("/report/negative", post(report_negative))
        .route("/report/spam", post(report_spam))
        .route("/report/vouch", post(report_vouch))
        // Cache management
        .route("/cache/invalidate/:did", post(invalidate_cache))
        .route("/cache/clear", post(clear_cache))
}

// ============================================================================
// Status Endpoints
// ============================================================================

/// Bridge status response
#[derive(Serialize, ToSchema)]
pub struct BridgeStatusResponse {
    /// Connection status
    pub status: String,
    /// Whether Bridge is operational
    pub connected: bool,
    /// Whether in fallback mode
    pub fallback_mode: bool,
    /// Bridge URL if configured
    pub bridge_url: Option<String>,
}

/// Get Bridge connection status
#[utoipa::path(
    get,
    path = "/bridge/status",
    tag = "bridge",
    responses(
        (status = 200, description = "Bridge status", body = BridgeStatusResponse),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_bridge_status(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
) -> AppResult<Json<BridgeStatusResponse>> {
    let connected = state.bridge.is_connected().await;
    let fallback = state.bridge.is_fallback().await;
    let connection_state = state.bridge.connection_state().await;

    Ok(Json(BridgeStatusResponse {
        status: format!("{:?}", connection_state),
        connected,
        fallback_mode: fallback,
        bridge_url: state.config.bridge_url.clone(),
    }))
}

/// Get Bridge cache statistics
#[utoipa::path(
    get,
    path = "/bridge/stats",
    tag = "bridge",
    responses(
        (status = 200, description = "Bridge statistics", body = BridgeCacheStats),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_bridge_stats(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
) -> AppResult<Json<BridgeCacheStats>> {
    let stats = state.bridge.cache_stats().await;
    Ok(Json(stats))
}

// ============================================================================
// Identity Endpoints
// ============================================================================

/// Cross-hApp identity response
#[derive(Serialize, ToSchema)]
pub struct IdentityResponse {
    /// The identity information
    #[serde(flatten)]
    pub identity: CrossHappIdentity,
    /// Source of the data (bridge, cache, fallback)
    pub source: String,
}

/// Query cross-hApp identity for a DID
#[utoipa::path(
    get,
    path = "/bridge/identity/{did}",
    tag = "bridge",
    params(
        ("did" = String, Path, description = "DID to query identity for")
    ),
    responses(
        (status = 200, description = "Identity information", body = IdentityResponse),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn query_identity(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
    Path(did): Path<String>,
) -> AppResult<Json<IdentityResponse>> {
    tracing::debug!("Querying cross-hApp identity for {}", did);

    let identity = state.bridge.query_identity(&did).await?;
    let source = if state.bridge.is_fallback().await {
        "fallback"
    } else {
        "bridge"
    };

    Ok(Json(IdentityResponse {
        identity,
        source: source.to_string(),
    }))
}

// ============================================================================
// Reputation Endpoints
// ============================================================================

/// Reputation response with analysis
#[derive(Serialize, ToSchema)]
pub struct ReputationResponse {
    /// The full reputation data
    #[serde(flatten)]
    pub reputation: CrossHappReputation,
    /// Spam likelihood (0.0 = unlikely, 1.0 = very likely)
    pub spam_likelihood: f64,
    /// Whether this DID meets minimum trust threshold
    pub meets_threshold: bool,
    /// Source of the data
    pub source: String,
}

/// Get cross-hApp reputation for a DID
#[utoipa::path(
    get,
    path = "/bridge/reputation/{did}",
    tag = "bridge",
    params(
        ("did" = String, Path, description = "DID to get reputation for")
    ),
    responses(
        (status = 200, description = "Reputation information", body = ReputationResponse),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_reputation(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
    Path(did): Path<String>,
) -> AppResult<Json<ReputationResponse>> {
    tracing::debug!("Querying cross-hApp reputation for {}", did);

    let reputation = state.bridge.get_reputation(&did).await?;
    let source = if state.bridge.is_fallback().await {
        "fallback"
    } else {
        "bridge"
    };

    let spam_like = spam_likelihood(&reputation);
    let meets = has_sufficient_reputation(&reputation, state.config.default_min_trust);

    Ok(Json(ReputationResponse {
        reputation,
        spam_likelihood: spam_like,
        meets_threshold: meets,
        source: source.to_string(),
    }))
}

/// Context hApps for reputation query
#[derive(Deserialize, ToSchema)]
pub struct ReputationContextQuery {
    /// hApps to query reputation from
    pub happs: Option<Vec<String>>,
}

/// Get full cross-hApp reputation with all sources
#[utoipa::path(
    get,
    path = "/bridge/reputation/{did}/full",
    tag = "bridge",
    params(
        ("did" = String, Path, description = "DID to get full reputation for")
    ),
    responses(
        (status = 200, description = "Full reputation with all hApp scores", body = CrossHappReputation),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_full_reputation(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
    Path(did): Path<String>,
) -> AppResult<Json<CrossHappReputation>> {
    // Query from all known hApps
    let context_happs = vec![
        HappId::Identity,
        HappId::Mail,
        HappId::Finance,
        HappId::Justice,
        HappId::Governance,
        HappId::Knowledge,
        HappId::Marketplace,
    ];

    let reputation = state.bridge.get_reputation_with_context(&did, &context_happs).await?;
    Ok(Json(reputation))
}

/// Reputation check response
#[derive(Serialize, ToSchema)]
pub struct ReputationCheckResponse {
    /// DID that was checked
    pub did: String,
    /// Whether the DID passes trust checks
    pub trusted: bool,
    /// Aggregate reputation score
    pub score: f64,
    /// Whether flagged as Byzantine
    pub is_byzantine: bool,
    /// Spam likelihood
    pub spam_likelihood: f64,
    /// Recommendation for handling
    pub recommendation: String,
}

/// Quick reputation check for a DID
#[utoipa::path(
    get,
    path = "/bridge/reputation/{did}/check",
    tag = "bridge",
    params(
        ("did" = String, Path, description = "DID to check")
    ),
    responses(
        (status = 200, description = "Reputation check result", body = ReputationCheckResponse),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn check_reputation(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
    Path(did): Path<String>,
) -> AppResult<Json<ReputationCheckResponse>> {
    let reputation = state.bridge.get_reputation(&did).await?;
    let spam_like = spam_likelihood(&reputation);
    let trusted = has_sufficient_reputation(&reputation, state.config.default_min_trust);

    let recommendation = if reputation.is_byzantine {
        "Block - Byzantine behavior detected"
    } else if spam_like > 0.8 {
        "Quarantine - High spam likelihood"
    } else if spam_like > 0.5 {
        "Review - Moderate spam likelihood"
    } else if trusted {
        "Allow - Trusted sender"
    } else {
        "Review - Unknown sender"
    };

    Ok(Json(ReputationCheckResponse {
        did,
        trusted,
        score: reputation.aggregate_score,
        is_byzantine: reputation.is_byzantine,
        spam_likelihood: spam_like,
        recommendation: recommendation.to_string(),
    }))
}

// ============================================================================
// Reputation Reporting Endpoints
// ============================================================================

/// Reputation report input
#[derive(Deserialize, ToSchema)]
pub struct ReputationReportInput {
    /// DID to report about
    pub did: String,
    /// Optional context/reason
    pub context: Option<String>,
}

/// Report positive interaction
#[utoipa::path(
    post,
    path = "/bridge/report/positive",
    tag = "bridge",
    request_body = ReputationReportInput,
    responses(
        (status = 200, description = "Positive interaction reported"),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn report_positive(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Json(input): Json<ReputationReportInput>,
) -> AppResult<Json<serde_json::Value>> {
    tracing::info!("User {} reporting positive interaction with {}", user.did, input.did);

    let context = input.context.unwrap_or_else(|| "positive_interaction".to_string());
    state.bridge.report_positive_interaction(&input.did, &context).await?;

    Ok(Json(serde_json::json!({
        "reported": true,
        "type": "positive",
        "subject": input.did
    })))
}

/// Report negative interaction
#[utoipa::path(
    post,
    path = "/bridge/report/negative",
    tag = "bridge",
    request_body = ReputationReportInput,
    responses(
        (status = 200, description = "Negative interaction reported"),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn report_negative(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Json(input): Json<ReputationReportInput>,
) -> AppResult<Json<serde_json::Value>> {
    tracing::info!("User {} reporting negative interaction with {}", user.did, input.did);

    let context = input.context.unwrap_or_else(|| "negative_interaction".to_string());
    state.bridge.report_negative_interaction(&input.did, &context).await?;

    Ok(Json(serde_json::json!({
        "reported": true,
        "type": "negative",
        "subject": input.did
    })))
}

/// Report spam
#[utoipa::path(
    post,
    path = "/bridge/report/spam",
    tag = "bridge",
    request_body = ReputationReportInput,
    responses(
        (status = 200, description = "Spam reported"),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn report_spam(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Json(input): Json<ReputationReportInput>,
) -> AppResult<Json<serde_json::Value>> {
    tracing::info!("User {} reporting spam from {}", user.did, input.did);

    let context = input.context.unwrap_or_else(|| "spam_report".to_string());
    state.bridge.report_spam(&input.did, &context).await?;

    Ok(Json(serde_json::json!({
        "reported": true,
        "type": "spam",
        "subject": input.did
    })))
}

/// Vouch for a DID
#[utoipa::path(
    post,
    path = "/bridge/report/vouch",
    tag = "bridge",
    request_body = ReputationReportInput,
    responses(
        (status = 200, description = "Vouch recorded"),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn report_vouch(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Json(input): Json<ReputationReportInput>,
) -> AppResult<Json<serde_json::Value>> {
    tracing::info!("User {} vouching for {}", user.did, input.did);

    let context = input.context.unwrap_or_else(|| "trust_vouch".to_string());
    state.bridge.vouch_for(&input.did, &context).await?;

    Ok(Json(serde_json::json!({
        "reported": true,
        "type": "vouch",
        "subject": input.did
    })))
}

// ============================================================================
// Cache Management Endpoints
// ============================================================================

/// Invalidate cache for a specific DID
#[utoipa::path(
    post,
    path = "/bridge/cache/invalidate/{did}",
    tag = "bridge",
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
    state.bridge.invalidate_cache(&did).await;

    Ok(Json(serde_json::json!({
        "invalidated": true,
        "did": did
    })))
}

/// Clear all Bridge caches
#[utoipa::path(
    post,
    path = "/bridge/cache/clear",
    tag = "bridge",
    responses(
        (status = 200, description = "All caches cleared"),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn clear_cache(
    State(state): State<AppState>,
    _user: AuthenticatedUser,
) -> AppResult<Json<serde_json::Value>> {
    state.bridge.clear_cache().await;

    Ok(Json(serde_json::json!({
        "cleared": true
    })))
}
