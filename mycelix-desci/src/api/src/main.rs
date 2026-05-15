// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix-DeSci REST API Server
//!
//! Production-grade REST API for the Mycelix-DeSci platform.
//! Provides HTTP interface for claims, queries, trust management, and more.

use axum::{
    extract::DefaultBodyLimit,
    http::{header, HeaderValue, Method},
    response::IntoResponse,
    Router,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::info;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

mod error;
mod handlers;
mod metrics;
mod middleware;
mod models;
mod routes;
mod state;

use error::Result;
use middleware::MetricsMiddleware;
use state::AppState;

/// API version
const API_VERSION: &str = "1.0.0";

/// Main entry point
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    init_logging();

    info!("🚀 Starting Mycelix-DeSci API v{}", API_VERSION);

    // Load configuration
    let config = load_config()?;
    info!("✅ Configuration loaded");

    // Initialize application state
    let state = Arc::new(AppState::new().await?);
    info!("✅ Application state initialized");

    // Start background workers
    start_quantum_anchor(Arc::clone(&state));

    // Build application
    let app = create_app(Arc::clone(&state));

    // Create server address
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    info!("🌐 Server listening on http://{}", addr);
    info!("📖 API docs available at http://{}/docs", addr);

    // Start server
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Initialize logging with tracing-subscriber
fn init_logging() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "mycelix_api=debug,tower_http=debug,axum=trace".into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}

/// Load configuration from environment
fn load_config() -> Result<Config> {
    Ok(Config {
        port: std::env::var("PORT")
            .unwrap_or_else(|_| "8080".to_string())
            .parse()
            .unwrap_or(8080),
        cors_allowed_origins: std::env::var("CORS_ORIGINS")
            .unwrap_or_else(|_| "*".to_string())
            .split(',')
            .map(String::from)
            .collect(),
    })
}

#[derive(Debug, Clone)]
struct Config {
    port: u16,
    cors_allowed_origins: Vec<String>,
}

/// Create the Axum application with all routes and middleware
fn create_app(state: AppState) -> Router {
    // Create OpenAPI documentation
    let openapi = routes::ApiDoc::openapi();

    Router::new()
        // Mount API routes at /api/v1
        .nest("/api/v1", routes::api_routes())
        // Swagger UI at /docs
        .merge(SwaggerUi::new("/docs").url("/api-docs/openapi.json", openapi))
        // Health check at root
        .route("/health", axum::routing::get(handlers::system::health_check))
        // Metrics endpoint for Prometheus
        .route("/metrics", axum::routing::get(metrics_handler))
        // Application state
        .with_state(Arc::new(state))
        // Middleware stack (applied in reverse order)
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CompressionLayer::new())
                .layer(configure_cors())
                .layer(TimeoutLayer::new(std::time::Duration::from_secs(30)))
                .layer(DefaultBodyLimit::max(10 * 1024 * 1024)), // 10MB max
        )
        // Add metrics middleware
        .layer(axum::middleware::from_fn(MetricsMiddleware::track))
}

/// Configure CORS middleware
fn configure_cors() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(
            "*".parse::<HeaderValue>()
                .expect("Static CORS origin '*' must be valid")
        )
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
}

/// Metrics handler for Prometheus
async fn metrics_handler() -> impl axum::response::IntoResponse {
    match metrics::encode_metrics() {
        Ok(metrics) => (
            axum::http::StatusCode::OK,
            [("content-type", "text/plain; version=0.0.4")],
            metrics,
        )
            .into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to encode metrics: {}", e),
        )
            .into_response(),
    }
}
