// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix-Mail Backend
//!
//! Axum-based REST API server that connects the React frontend to Holochain DNA.
//! All DID resolution happens via DHT - no external database needed.
//!
//! ## Architecture
//!
//! ```text
//! React Frontend
//!      │
//!      │ HTTP/WebSocket
//!      ▼
//! ┌─────────────────┐
//! │  Axum Backend   │  ← This server
//! │  (Rust)         │
//! └────────┬────────┘
//!          │ AppAgentWebsocket
//!          ▼
//! ┌─────────────────┐     ┌─────────────────┐
//! │   Holochain     │     │  Bridge Zome    │
//! │   Conductor     │────▶│  (Cross-hApp)   │
//! └────────┬────────┘     └────────┬────────┘
//!          │                       │
//!          ▼                       ▼
//! ┌─────────────────┐     ┌─────────────────┐
//! │ mycelix_mail    │     │ Other Mycelix   │
//! │ DNA             │     │ hApps           │
//! └─────────────────┘     └─────────────────┘
//! ```
//!
//! ## Bridge Integration
//!
//! The backend integrates with the Bridge zome for cross-hApp reputation:
//! - Query identity from Identity hApp
//! - Get aggregated reputation across all Mycelix hApps
//! - Report interactions to the ecosystem-wide trust system
//! - Graceful fallback when Bridge is unavailable

use std::net::SocketAddr;
use std::time::Instant;

use axum::{
    http::{header, Method},
    Extension, Router,
};
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use mycelix_mail_backend::{
    config::Config,
    middleware::JwtSecret,
    openapi::ApiDoc,
    routes::{create_router, AppState},
    types,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load configuration
    let config = Config::from_env()?;

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| config.log_level.clone().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("🍄 Mycelix-Mail Backend starting...");
    tracing::info!("Version: {}", env!("CARGO_PKG_VERSION"));

    // Create application state
    let state = AppState::new(config.clone());

    // Connect to Holochain conductor
    if config.is_stub_mode() {
        tracing::info!("Holochain STUB MODE enabled - no real conductor connection");
    } else {
        tracing::info!("Connecting to Holochain at {}", config.holochain_conductor_url);
    }
    if let Err(e) = state.holochain.connect().await {
        tracing::warn!("Failed to connect to Holochain: {}. Will retry on first request.", e);
    } else {
        let mode = if config.is_stub_mode() { "stub" } else { "production" };
        tracing::info!("Holochain service initialized in {} mode", mode);
    }

    // Wire up HolochainService to BridgeClient for Bridge zome calls
    // This must happen after Holochain connection is established
    state.initialize().await;
    tracing::debug!("Application state initialized with Bridge zome integration");

    // Connect to Bridge for cross-hApp communication
    if config.bridge_stub_mode {
        tracing::info!("Bridge STUB MODE enabled - using mock cross-hApp data");
    } else if config.bridge_url.is_some() {
        tracing::info!("Connecting to Bridge at {:?}", config.bridge_url);
    } else {
        tracing::info!("Bridge URL not configured - will use Holochain Bridge zome if available");
    }
    if let Err(e) = state.bridge.connect().await {
        tracing::warn!("Failed to connect to Bridge: {}. Cross-hApp features will use fallback.", e);
    } else {
        let bridge_state = state.bridge.connection_state().await;
        tracing::info!("Bridge service initialized in {:?} mode", bridge_state);
    }

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

    // Build the router
    let app = Router::new()
        .merge(create_router(state.clone()))
        // OpenAPI documentation
        .merge(SwaggerUi::new("/api-docs").url("/api-docs/openapi.json", ApiDoc::openapi()))
        // Health check endpoint
        .route("/health", axum::routing::get(health_check))
        // Add middleware
        .layer(Extension(state))
        .layer(Extension(JwtSecret(config.jwt_secret.clone())))
        .layer(CompressionLayer::new())
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    // Parse server address
    let addr: SocketAddr = config.server_addr().parse()?;

    tracing::info!("Server listening on http://{}", addr);
    tracing::info!("API docs available at http://{}/api-docs/", addr);
    if config.is_stub_mode() {
        tracing::info!("   Holochain: STUB MODE (no conductor)");
    } else {
        tracing::info!("   Holochain Conductor: {}", config.holochain_conductor_url);
        tracing::info!("   Holochain Admin: {}", config.holochain_admin_url);
    }
    tracing::info!("   App ID: {}", config.holochain_app_id);
    tracing::info!("   CORS origins: {:?}", config.cors_origins);

    // Start server
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Health status", body = types::HealthResponse)
    )
)]
pub async fn health_check(
    Extension(state): Extension<AppState>,
) -> axum::Json<types::HealthResponse> {
    static START_TIME: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START_TIME.get_or_init(Instant::now);

    let holochain_connected = state.holochain.is_connected().await;
    let bridge_connected = state.bridge.is_connected().await;
    let bridge_state = state.bridge.connection_state().await;

    // Determine overall health
    let status = if holochain_connected && bridge_connected {
        "healthy"
    } else if holochain_connected {
        "degraded" // Holochain works but no cross-hApp reputation
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
