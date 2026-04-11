// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix-Mail Backend Library
//!
//! This library exposes the core components of the Mycelix-Mail backend
//! for testing and integration purposes.

pub mod config;
pub mod error;
pub mod middleware;
pub mod openapi;
pub mod routes;
pub mod services;
pub mod types;
pub mod validation;

use axum::{
    http::{header, Method},
    Extension, Router,
};
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    trace::TraceLayer,
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::config::Config;
use crate::middleware::JwtSecret;
use crate::openapi::ApiDoc;
use crate::routes::{create_router, AppState};

/// Create a test router for integration testing
///
/// This creates the full application router without starting a server.
pub async fn create_test_router() -> Router {
    let config = Config::from_env().expect("Failed to load test config");

    let state = AppState::new(config.clone());

    // Build CORS layer
    let cors = CorsLayer::new()
        .allow_origin(
            config
                .cors_origins
                .iter()
                .filter_map(|o| o.parse().ok())
                .collect::<Vec<_>>(),
        )
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE, Method::OPTIONS])
        .allow_headers([
            header::CONTENT_TYPE,
            header::AUTHORIZATION,
            header::ACCEPT,
        ])
        .allow_credentials(true);

    Router::new()
        .merge(create_router(state.clone()))
        .merge(SwaggerUi::new("/api-docs").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/health", axum::routing::get(health_check))
        .layer(Extension(state))
        .layer(Extension(JwtSecret(config.jwt_secret.clone())))
        .layer(CompressionLayer::new())
        .layer(cors)
        .layer(TraceLayer::new_for_http())
}

/// Create a test JWT token for authentication testing
pub fn create_test_token(did: &str, expiration_hours: i64) -> String {
    let claims = middleware::Claims::new(
        did.to_string(),
        "test_agent_pubkey_base64".to_string(),
        expiration_hours as u64,
    );

    middleware::create_token(
        &claims,
        &std::env::var("JWT_SECRET").unwrap_or_else(|_| "test_jwt_secret_for_testing_only_32chars!".to_string()),
    )
    .expect("Failed to create test token")
}

/// Health check handler (for lib usage)
async fn health_check(
    Extension(state): Extension<AppState>,
) -> axum::Json<types::HealthResponse> {
    use std::time::Instant;
    static START_TIME: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START_TIME.get_or_init(Instant::now);

    let holochain_connected = state.holochain.is_connected().await;
    let bridge_connected = state.bridge.is_connected().await;
    let bridge_state = state.bridge.connection_state().await;

    let status = if holochain_connected && bridge_connected {
        "healthy"
    } else if holochain_connected {
        "degraded"
    } else {
        "unhealthy"
    };

    axum::Json(types::HealthResponse {
        status: status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        holochain_connected,
        bridge_connected,
        bridge_mode: Some(format!("{:?}", bridge_state)),
        uptime_seconds: start.elapsed().as_secs(),
    })
}
