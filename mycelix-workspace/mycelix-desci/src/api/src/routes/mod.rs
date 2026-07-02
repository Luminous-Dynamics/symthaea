// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! API route definitions

use axum::{
    Router,
    middleware::from_fn,
    routing::{get, post, put},
};
use std::sync::Arc;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::error::ApiError;
use crate::middleware::require_auth;
use crate::{handlers, models::*, state::AppState};

/// API routes
///
/// Split into a public (read-only GET) router and a protected (mutating
/// POST/PUT/DELETE) router gated by [`require_auth`]. `/query` stays public
/// even though it is a POST, since it is a read-only search endpoint (no
/// state is mutated) — only routes that create or modify stored data require
/// a bearer token.
pub fn api_routes() -> Router<Arc<AppState>> {
    public_routes().merge(protected_routes())
}

/// Read-only routes. No authentication required.
fn public_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/claims/:id", get(handlers::get_claim))
        // Query endpoints (POST but read-only: no mutation of stored data)
        .route("/query", post(handlers::execute_query))
        .route("/query/categories", get(handlers::get_categories))
        .route("/query/stats", get(handlers::get_stats))
        // Trust endpoints (read)
        .route("/trust/:participant", get(handlers::get_trust_score))
        .route("/trust/stats", get(handlers::get_trust_stats))
        // System endpoints
        .route("/system/health", get(handlers::health_check))
        .route("/system/metrics", get(handlers::get_metrics))
        .route("/system/version", get(handlers::get_version))
}

/// Mutating routes (POST/PUT/DELETE) that create or modify stored data.
/// Gated by [`require_auth`] — callers must present a valid bearer JWT.
fn protected_routes() -> Router<Arc<AppState>> {
    Router::new()
        // Claims endpoints
        .route("/claims", post(handlers::create_claim))
        .route("/claims/:id/verify", put(handlers::add_verification))
        .route("/claims/:id/provenance", put(handlers::add_provenance))
        // Trust endpoints (write)
        .route("/trust/:participant", put(handlers::update_trust_score))
        .route_layer(from_fn(require_auth))
}

/// OpenAPI documentation
#[derive(OpenApi)]
#[openapi(
    paths(
        // Claims
        handlers::create_claim,
        handlers::get_claim,
        handlers::add_verification,
        handlers::add_provenance,
        // Query
        handlers::execute_query,
        handlers::get_categories,
        handlers::get_stats,
        // Trust
        handlers::get_trust_score,
        handlers::update_trust_score,
        handlers::get_trust_stats,
        // System
        handlers::health_check,
        handlers::get_metrics,
        handlers::get_version,
    ),
    components(
        schemas(
            // Request models
            CreateClaimRequest,
            ClaimContentRequest,
            AddVerificationRequest,
            AddProvenanceRequest,
            QueryRequest,
            QuerySort,
            UpdateTrustScoreRequest,
            // Response models
            ClaimResponse,
            QueryResponse,
            CategoriesResponse,
            QueryStatsResponse,
            TrustScoreResponse,
            TrustStatsResponse,
            HealthResponse,
            HealthCheck,
            MetricsResponse,
            VersionResponse,
            ApiError,
        )
    ),
    tags(
        (name = "claims", description = "Claims management endpoints"),
        (name = "query", description = "Query and search endpoints"),
        (name = "trust", description = "Trust network endpoints"),
        (name = "system", description = "System health and metrics endpoints")
    ),
    info(
        title = "Mycelix-DeSci API",
        version = "1.0.0",
        description = "REST API for the Mycelix-DeSci decentralized science platform",
        contact(
            name = "Mycelix-DeSci Team",
            url = "https://github.com/Luminous-Dynamics/mycelix-desci"
        ),
        license(
            name = "MIT",
            url = "https://opensource.org/licenses/MIT"
        )
    )
)]
pub struct ApiDoc;

/// Create Swagger UI routes
pub fn swagger_ui() -> SwaggerUi {
    SwaggerUi::new("/docs").url("/api-docs/openapi.json", ApiDoc::openapi())
}
