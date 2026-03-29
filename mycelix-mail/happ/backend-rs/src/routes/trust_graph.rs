// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust Graph API routes
//!
//! Endpoints for visualizing and managing the trust graph.

use axum::{
    extract::{Extension, Path, Query},
    routing::{get, post, delete},
    Json, Router,
};

use crate::error::{AppError, AppResult};
use crate::middleware::Claims as JwtClaims;
use crate::routes::AppState;
use crate::services::trust_graph::{
    RelationType, TrustAttestation, TrustEdge, TrustGraph,
    TrustGraphService, TrustPath,
};

/// Create the trust graph router
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/graph/:did", get(get_trust_graph))
        .route("/path/:from/:to", get(find_trust_path))
        .route("/attest", post(create_attestation))
        .route("/attest/:id", delete(revoke_attestation))
        .route("/trusted-by/:did", get(get_trusted_by))
        .route("/trusters/:did", get(get_trusters))
}

/// Query params for graph depth
#[derive(Debug, serde::Deserialize, utoipa::IntoParams, utoipa::ToSchema)]
pub struct GraphQuery {
    #[serde(default = "default_depth")]
    pub depth: usize,
}

fn default_depth() -> usize {
    2
}

/// Get the trust graph centered on a DID
#[utoipa::path(
    get,
    path = "/trust-graph/graph/{did}",
    tag = "trust-graph",
    params(
        ("did" = String, Path, description = "Center DID"),
        GraphQuery,
    ),
    responses(
        (status = 200, description = "Trust graph", body = TrustGraph),
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_trust_graph(
    Extension(_claims): Extension<JwtClaims>,
    Extension(_state): Extension<AppState>,
    Path(did): Path<String>,
    Query(query): Query<GraphQuery>,
) -> AppResult<Json<TrustGraph>> {
    let graph_service = TrustGraphService::new();
    let graph = graph_service.get_subgraph(&did, query.depth).await;
    Ok(Json(graph))
}

/// Find trust path between two DIDs
#[utoipa::path(
    get,
    path = "/trust-graph/path/{from}/{to}",
    tag = "trust-graph",
    params(
        ("from" = String, Path, description = "Starting DID"),
        ("to" = String, Path, description = "Target DID"),
    ),
    responses(
        (status = 200, description = "Trust path", body = TrustPath),
        (status = 404, description = "No path found", body = crate::types::ApiError),
    ),
    security(("bearer_auth" = []))
)]
pub async fn find_trust_path(
    Extension(_claims): Extension<JwtClaims>,
    Path((from, to)): Path<(String, String)>,
) -> AppResult<Json<TrustPath>> {
    let graph_service = TrustGraphService::new();

    match graph_service.find_path(&from, &to).await {
        Ok(path) => Ok(Json(path)),
        Err(_) => Err(AppError::NotFound("No trust path found".to_string())),
    }
}

/// Request to create a trust attestation
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct CreateAttestationRequest {
    pub subject_did: String,
    pub message: String,
    pub relationship: RelationType,
    pub trust_score: f64,
    pub expires_days: Option<u32>,
}

/// Response with created attestation
#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub struct CreateAttestationResponse {
    pub attestation: TrustAttestation,
}

/// Create a trust attestation
#[utoipa::path(
    post,
    path = "/trust-graph/attest",
    tag = "trust-graph",
    request_body = CreateAttestationRequest,
    responses(
        (status = 200, description = "Attestation created", body = CreateAttestationResponse),
        (status = 400, description = "Invalid request", body = crate::types::ApiError),
    ),
    security(("bearer_auth" = []))
)]
pub async fn create_attestation(
    Extension(claims): Extension<JwtClaims>,
    Json(request): Json<CreateAttestationRequest>,
) -> AppResult<Json<CreateAttestationResponse>> {
    // Validate trust score
    if request.trust_score < 0.0 || request.trust_score > 1.0 {
        return Err(AppError::ValidationError(
            "Trust score must be between 0 and 1".to_string(),
        ));
    }

    let graph_service = TrustGraphService::new();

    let now = chrono::Utc::now();
    let expires_at = request.expires_days.map(|days| {
        now + chrono::Duration::days(days as i64)
    });

    let attestation = TrustAttestation {
        id: uuid::Uuid::new_v4().to_string(),
        attestor_did: claims.sub.clone(),
        subject_did: request.subject_did,
        message: request.message,
        relationship: request.relationship,
        trust_score: request.trust_score,
        created_at: now,
        expires_at,
    };

    graph_service.create_attestation(attestation.clone()).await?;

    Ok(Json(CreateAttestationResponse { attestation }))
}

/// Revoke a trust attestation
#[utoipa::path(
    delete,
    path = "/trust-graph/attest/{id}",
    tag = "trust-graph",
    params(
        ("id" = String, Path, description = "Attestation ID"),
    ),
    responses(
        (status = 200, description = "Attestation revoked"),
        (status = 404, description = "Attestation not found", body = crate::types::ApiError),
    ),
    security(("bearer_auth" = []))
)]
pub async fn revoke_attestation(
    Extension(_claims): Extension<JwtClaims>,
    Path(id): Path<String>,
) -> AppResult<Json<serde_json::Value>> {
    // In production: actually revoke the attestation
    Ok(Json(serde_json::json!({
        "revoked": true,
        "id": id
    })))
}

/// Response with trust edges
#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub struct TrustEdgesResponse {
    pub edges: Vec<TrustEdge>,
}

/// Get who a DID trusts
#[utoipa::path(
    get,
    path = "/trust-graph/trusted-by/{did}",
    tag = "trust-graph",
    params(
        ("did" = String, Path, description = "DID to query"),
    ),
    responses(
        (status = 200, description = "Trust edges", body = TrustEdgesResponse),
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_trusted_by(
    Extension(_claims): Extension<JwtClaims>,
    Path(did): Path<String>,
) -> AppResult<Json<TrustEdgesResponse>> {
    let graph_service = TrustGraphService::new();
    let edges = graph_service.get_trusted_by(&did).await;
    Ok(Json(TrustEdgesResponse { edges }))
}

/// Get who trusts a DID
#[utoipa::path(
    get,
    path = "/trust-graph/trusters/{did}",
    tag = "trust-graph",
    params(
        ("did" = String, Path, description = "DID to query"),
    ),
    responses(
        (status = 200, description = "Trust edges", body = TrustEdgesResponse),
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_trusters(
    Extension(_claims): Extension<JwtClaims>,
    Path(did): Path<String>,
) -> AppResult<Json<TrustEdgesResponse>> {
    let graph_service = TrustGraphService::new();
    let edges = graph_service.get_trusters(&did).await;
    Ok(Json(TrustEdgesResponse { edges }))
}
