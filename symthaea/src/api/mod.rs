//! # Symthaea Benchmark API
//!
//! RESTful API for consciousness measurement proxies (spectral connectivity) and benchmark submission.
//!
//! ## Endpoints
//! - POST /v1/submit - Submit model for spectral connectivity (lambda2) evaluation
//! - GET /v1/results/{id} - Get evaluation results
//! - GET /v1/leaderboard - Public leaderboard
//! - GET /v1/datasets - List available datasets
//! - POST /v1/compare - Compare two models
//!
//! ## Usage
//! ```rust
//! use symthaea::api::create_router;
//!
//! #[tokio::main]
//! async fn main() {
//!     let app = create_router();
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
//!     axum::serve(listener, app).await.unwrap();
//! }
//! ```

pub mod handlers;
pub mod metrics;
pub mod models;
pub mod state;

use axum::{
    routing::{get, post},
    Router,
    middleware,
    extract::Request,
    http::{StatusCode, HeaderMap, Method},
    response::Response,
};
use tower_http::cors::CorsLayer;
use std::sync::Arc;
use crate::api::state::AppState;

/// API configuration for security settings
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Allowed CORS origins (empty = localhost only)
    pub allowed_origins: Vec<String>,
    /// Optional Bearer token for authenticated endpoints.
    /// If None, all endpoints are public (development mode).
    pub bearer_token: Option<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec![
                "http://localhost:3000".to_string(),
                "http://127.0.0.1:3000".to_string(),
            ],
            bearer_token: None,
        }
    }
}

/// Authentication middleware that checks for Bearer token on mutating endpoints.
///
/// Health and GET leaderboard/dataset endpoints are public.
/// POST endpoints require authentication when a bearer token is configured.
async fn auth_middleware(
    headers: HeaderMap,
    request: Request,
    next: middleware::Next,
) -> Result<Response, StatusCode> {
    let state = request.extensions()
        .get::<Arc<AppState>>();

    // Extract configured token from app state
    let required_token = state.and_then(|s| s.bearer_token());

    // If no token is configured, allow everything (development mode)
    let Some(expected_token) = required_token else {
        return Ok(next.run(request).await);
    };

    // Allow health checks without auth
    let path = request.uri().path();
    if path == "/health" || path == "/v1/health" {
        return Ok(next.run(request).await);
    }

    // Allow public GET endpoints without auth
    let method = request.method().clone();
    if method == Method::GET && (
        path.starts_with("/v1/leaderboard")
        || path.starts_with("/v1/datasets")
        || path.starts_with("/v1/results")
    ) {
        return Ok(next.run(request).await);
    }

    // For all other endpoints, require Bearer token
    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(value) if value.starts_with("Bearer ") => {
            let token = &value[7..];
            if token == expected_token {
                Ok(next.run(request).await)
            } else {
                Err(StatusCode::UNAUTHORIZED)
            }
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

/// Build a CORS layer with restricted origins.
fn build_cors_layer(config: &ApiConfig) -> CorsLayer {
    use tower_http::cors::AllowOrigin;

    if config.allowed_origins.is_empty() {
        // Default: localhost only
        CorsLayer::new()
            .allow_origin(AllowOrigin::predicate(|origin, _| {
                let origin_str = origin.as_bytes();
                origin_str.starts_with(b"http://localhost")
                    || origin_str.starts_with(b"http://127.0.0.1")
                    || origin_str.starts_with(b"http://[::1]")
            }))
            .allow_methods([Method::GET, Method::POST])
            .allow_headers(tower_http::cors::Any)
    } else {
        let origins: Vec<_> = config.allowed_origins.iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([Method::GET, Method::POST])
            .allow_headers(tower_http::cors::Any)
    }
}

/// Create the API router with all endpoints (default config: localhost CORS, no auth)
pub fn create_router() -> Router {
    create_router_with_config(ApiConfig::default())
}

/// Create the API router with custom security configuration
pub fn create_router_with_config(config: ApiConfig) -> Router {
    let state = Arc::new(AppState::new_with_config(&config));
    let cors = build_cors_layer(&config);

    Router::new()
        // Core endpoints
        .route("/v1/submit", post(handlers::submit_model))
        .route("/v1/results/:submission_id", get(handlers::get_results))
        .route("/v1/leaderboard", get(handlers::get_leaderboard))
        .route("/v1/leaderboard/topologies", get(handlers::get_topology_rankings))
        .route("/v1/datasets", get(handlers::list_datasets))
        .route("/v1/datasets/:dataset_id", get(handlers::get_dataset))
        .route("/v1/compare", post(handlers::compare_models))
        .route("/v1/dimensional-sweep", post(handlers::dimensional_sweep))
        // Metrics endpoints
        .route("/metrics", get(handlers::metrics_prometheus))       // Prometheus scrape endpoint
        .route("/v1/metrics", get(handlers::metrics_json))          // JSON metrics
        // Health check
        .route("/health", get(handlers::health_check))
        .route("/v1/health", get(handlers::health_check))
        // Security layers
        .layer(middleware::from_fn(auth_middleware))
        .layer(cors)
        // Add shared state
        .with_state(state)
}

/// Start the API server
pub async fn serve(addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_router();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Symthaea API listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

/// Start the API server with custom config
pub async fn serve_with_config(addr: &str, config: ApiConfig) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_router_with_config(config);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Symthaea API listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}
