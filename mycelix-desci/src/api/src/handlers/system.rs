// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! System API handlers

use axum::{Json, extract::State, http::StatusCode};
use std::sync::Arc;
use tracing::{info, instrument};

use crate::{error::Result, models::*, state::AppState};

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/api/v1/system/health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse),
        (status = 503, description = "Service is unhealthy", body = HealthResponse),
    ),
    tag = "system"
)]
#[instrument(skip(state))]
pub async fn health_check(
    State(state): State<Arc<AppState>>,
) -> Result<(StatusCode, Json<HealthResponse>)> {
    info!("Health check");

    // TODO: Implement actual health checks
    // For now, assume all components are healthy if they exist
    let overall_healthy = true;

    let response = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.uptime().as_secs(),
        checks: vec![
            HealthCheck {
                name: "storage".to_string(),
                status: "passing".to_string(),
            },
            HealthCheck {
                name: "query_engine".to_string(),
                status: "passing".to_string(),
            },
            HealthCheck {
                name: "trust_manager".to_string(),
                status: "passing".to_string(),
            },
        ],
    };

    let status_code = StatusCode::OK;

    Ok((status_code, Json(response)))
}

/// Get system metrics
#[utoipa::path(
    get,
    path = "/api/v1/system/metrics",
    responses(
        (status = 200, description = "Metrics retrieved", body = MetricsResponse),
    ),
    tag = "system"
)]
#[instrument(skip(state))]
pub async fn get_metrics(State(state): State<Arc<AppState>>) -> Result<Json<MetricsResponse>> {
    info!("Retrieving metrics");

    // TODO: Implement proper metrics aggregation
    let response = MetricsResponse {
        uptime_seconds: state.uptime().as_secs(),
        total_claims: 0,
        total_participants: 0,
        queries_executed: state
            .metrics
            .queries_executed
            .load(std::sync::atomic::Ordering::Relaxed),
        claims_created: state
            .metrics
            .claims_created
            .load(std::sync::atomic::Ordering::Relaxed),
        verifications_added: state
            .metrics
            .verifications_added
            .load(std::sync::atomic::Ordering::Relaxed),
        average_response_time_ms: state.metrics.average_response_time_ms(),
    };

    Ok(Json(response))
}

/// Get version information
#[utoipa::path(
    get,
    path = "/api/v1/system/version",
    responses(
        (status = 200, description = "Version information", body = VersionResponse),
    ),
    tag = "system"
)]
#[instrument]
pub async fn get_version() -> Result<Json<VersionResponse>> {
    info!("Retrieving version");

    let response = VersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        build_date: option_env!("BUILD_DATE").unwrap_or("unknown").to_string(),
        git_commit: option_env!("GIT_COMMIT").unwrap_or("unknown").to_string(),
        rust_version: option_env!("RUSTC_VERSION")
            .unwrap_or("unknown")
            .to_string(),
    };

    Ok(Json(response))
}
