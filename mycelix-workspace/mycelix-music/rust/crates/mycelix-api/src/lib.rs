// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix API - REST and GraphQL service layer
//!
//! Provides HTTP endpoints for:
//! - Track upload and management
//! - Audio analysis
//! - Similarity search
//! - Real-time streaming via WebSocket
//! - Session management

use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, Multipart, Path, Query, State, WebSocketUpgrade},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use mycelix_core::{CoreConfig, MycelixEngine};

pub mod graphql;
pub mod handlers;
pub mod websocket;

/// API error type
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Core error: {0}")]
    Core(#[from] mycelix_core::CoreError),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            ApiError::Core(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        };

        let body = Json(ErrorResponse { error: message });
        (status, body).into_response()
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

/// Error response body
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<MycelixEngine>,
}

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub max_upload_size: usize,
    pub enable_graphql: bool,
    pub enable_websocket: bool,
    /// Allowed CORS origins. Empty = localhost only (secure-by-default).
    #[serde(default)]
    pub allowed_origins: Vec<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            // Secure-by-default: bind to localhost unless explicitly configured otherwise.
            host: "127.0.0.1".to_string(),
            port: 3000,
            max_upload_size: 100 * 1024 * 1024, // 100MB
            enable_graphql: true,
            enable_websocket: true,
            allowed_origins: Vec::new(),
        }
    }
}

/// Create the API router
pub fn create_router(state: AppState, config: &ApiConfig) -> Router {
    let mut router = Router::new()
        // Health check
        .route("/health", get(health_check))
        // Track endpoints
        .route("/api/tracks", post(handlers::upload_track))
        .route("/api/tracks/:id", get(handlers::get_track))
        .route("/api/tracks/:id", delete(handlers::delete_track))
        .route("/api/tracks/:id/analyze", post(handlers::analyze_track))
        .route("/api/tracks/:id/stream", get(handlers::stream_track))
        // Search endpoints
        .route("/api/search/similar/:id", get(handlers::find_similar))
        .route("/api/search/query", post(handlers::search_query))
        // Session endpoints
        .route("/api/sessions", post(handlers::create_session))
        .route("/api/sessions/:id", get(handlers::get_session))
        .route("/api/sessions/:id", delete(handlers::close_session))
        .route("/api/sessions/:id/play", post(handlers::session_play))
        .route("/api/sessions/:id/pause", post(handlers::session_pause))
        .route("/api/sessions/:id/seek", post(handlers::session_seek))
        // Analysis endpoints
        .route("/api/analyze/audio", post(handlers::analyze_audio))
        .route("/api/analyze/batch", post(handlers::batch_analyze))
        // Codec endpoints
        .route("/api/codec/transcode", post(handlers::transcode))
        .route("/api/codec/formats", get(handlers::list_formats));

    // Add GraphQL if enabled
    if config.enable_graphql {
        let schema = graphql::create_schema(state.engine.clone());
        router = router
            .route("/graphql", get(graphql::graphql_playground).post(graphql::graphql_handler))
            .layer(axum::Extension(schema));
    }

    // Add WebSocket if enabled
    if config.enable_websocket {
        router = router.route("/ws", get(websocket::ws_handler));
    }

    let cors = if config.allowed_origins.is_empty() {
        CorsLayer::new()
            .allow_origin(AllowOrigin::predicate(|origin, _| {
                let o = origin.as_bytes();
                o.starts_with(b"http://localhost")
                    || o.starts_with(b"https://localhost")
                    || o.starts_with(b"http://127.0.0.1")
                    || o.starts_with(b"https://127.0.0.1")
                    || o.starts_with(b"http://[::1]")
                    || o.starts_with(b"https://[::1]")
                    || o == b"null"
            }))
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        let origins: Vec<_> = config
            .allowed_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods(Any)
            .allow_headers(Any)
    };

    router
        .layer(DefaultBodyLimit::max(config.max_upload_size))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Health check endpoint
async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

/// Start the API server
pub async fn start_server(config: ApiConfig, core_config: CoreConfig) -> Result<(), ApiError> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Create engine
    let engine = MycelixEngine::new(core_config)
        .await
        .map_err(ApiError::Core)?;

    let state = AppState {
        engine: Arc::new(engine),
    };

    let router = create_router(state, &config);
    let addr = format!("{}:{}", config.host, config.port);

    tracing::info!("Starting Mycelix API server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    axum::serve(listener, router)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    Ok(())
}

/// Query parameters for pagination
#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    #[serde(default = "default_page")]
    pub page: u32,
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_page() -> u32 {
    1
}

fn default_limit() -> u32 {
    20
}

/// Standard API response wrapper
#[derive(Serialize)]
pub struct ApiResponse<T: Serialize> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(message: impl Into<String>) -> ApiResponse<()> {
        ApiResponse {
            success: false,
            data: None,
            error: Some(message.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ApiConfig::default();
        assert_eq!(config.port, 3000);
        assert!(config.enable_graphql);
    }
}
