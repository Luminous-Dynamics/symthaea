// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

pub mod demo_runner;
pub mod handlers;
pub mod holon;
pub mod metrics;
pub mod models;
pub mod state;
pub mod ws;

use crate::api::state::AppState;
use crate::control_plane::parse_bearer_token;
use axum::{
    Router,
    extract::DefaultBodyLimit,
    extract::{Request, State, ws::WebSocketUpgrade},
    http::{HeaderMap, Method, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

/// API configuration for security settings
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Allowed CORS origins (empty = localhost only)
    pub allowed_origins: Vec<String>,
    /// Optional Bearer token for authenticated endpoints.
    /// If None, all endpoints are public (development mode).
    pub bearer_token: Option<String>,
    /// Max request body size in bytes.
    pub max_body_bytes: usize,
    /// Max number of in-flight requests before backpressure.
    pub max_in_flight_requests: usize,
    /// Optional JSONL audit log path.
    pub audit_log_path: Option<PathBuf>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec![
                "http://localhost:3000".to_string(),
                "http://127.0.0.1:3000".to_string(),
            ],
            bearer_token: None,
            max_body_bytes: 16 * 1024 * 1024,
            max_in_flight_requests: 32,
            audit_log_path: None,
        }
    }
}

/// Authentication middleware that checks for Bearer token on protected endpoints.
///
/// Health and read-only public catalog endpoints remain public.
/// Audit access and mutating endpoints require authentication when a bearer token is configured.
async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: Request,
    next: middleware::Next,
) -> Result<Response, StatusCode> {
    let required_token = state.bearer_token();

    // If no token is configured, allow everything (development mode)
    let Some(expected_token) = required_token else {
        return Ok(next.run(request).await);
    };

    // Allow health checks without auth
    let path = request.uri().path();
    if path == "/health" || path == "/v1/health" {
        return Ok(next.run(request).await);
    }

    // Allow public catalog endpoints without auth. Results remain on the GET allowlist
    // because the handler performs per-submission privacy checks.
    let method = request.method().clone();
    if route_is_public_without_auth(&method, path) {
        return Ok(next.run(request).await);
    }

    // For all other endpoints, require Bearer token
    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_bearer_token);

    match auth_header {
        Some(token) if token == expected_token => Ok(next.run(request).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

fn route_is_public_without_auth(method: &Method, path: &str) -> bool {
    *method == Method::GET
        && (path.starts_with("/v1/leaderboard")
            || path.starts_with("/v1/datasets")
            || path.starts_with("/v1/results"))
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
        let origins: Vec<_> = config
            .allowed_origins
            .iter()
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
        .route(
            "/v1/leaderboard/topologies",
            get(handlers::get_topology_rankings),
        )
        .route("/v1/datasets", get(handlers::list_datasets))
        .route("/v1/datasets/:dataset_id", get(handlers::get_dataset))
        .route("/v1/compare", post(handlers::compare_models))
        .route("/v1/dimensional-sweep", post(handlers::dimensional_sweep))
        .route("/v1/audit/events", get(handlers::get_audit_events))
        // Metrics endpoints
        .route("/metrics", get(handlers::metrics_prometheus)) // Prometheus scrape endpoint
        .route("/v1/metrics", get(handlers::metrics_json)) // JSON metrics
        // Health check
        .route("/health", get(handlers::health_check))
        .route("/v1/health", get(handlers::health_check))
        // Security layers
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(cors)
        .layer(DefaultBodyLimit::max(config.max_body_bytes))
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
pub async fn serve_with_config(
    addr: &str,
    config: ApiConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_router_with_config(config);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Symthaea API listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

/// Create a demo router with WebSocket endpoint and static file serving.
///
/// This router includes all benchmark endpoints plus:
/// - `/v1/ws/live` — WebSocket stream of live consciousness telemetry
/// - `/` — Static file serving from `static/` directory
pub fn create_demo_router() -> Result<Router, Box<dyn std::error::Error>> {
    create_demo_router_with_config(ApiConfig::default())
}

/// Internal router builder — accepts a pre-built, already-Arc-wrapped runner.
/// Separated from `create_demo_router_with_config` so `serve_demo_with_config`
/// can do async P2P init on the runner *before* it is wrapped in Mutex.
fn build_demo_router(
    runner: Arc<Mutex<demo_runner::DemoRunner>>,
    config: ApiConfig,
) -> Result<Router, Box<dyn std::error::Error>> {
    let state = Arc::new(AppState::new_with_config(&config));
    let cors = build_cors_layer(&config);

    // Determine static dir relative to the manifest
    let static_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");

    let app = Router::new()
        // Core benchmark endpoints
        .route("/v1/submit", post(handlers::submit_model))
        .route("/v1/results/:submission_id", get(handlers::get_results))
        .route("/v1/leaderboard", get(handlers::get_leaderboard))
        .route(
            "/v1/leaderboard/topologies",
            get(handlers::get_topology_rankings),
        )
        .route("/v1/datasets", get(handlers::list_datasets))
        .route("/v1/datasets/:dataset_id", get(handlers::get_dataset))
        .route("/v1/compare", post(handlers::compare_models))
        .route("/v1/dimensional-sweep", post(handlers::dimensional_sweep))
        .route("/v1/audit/events", get(handlers::get_audit_events))
        // Metrics
        .route("/metrics", get(handlers::metrics_prometheus))
        .route("/v1/metrics", get(handlers::metrics_json))
        // Health
        .route("/health", get(handlers::health_check))
        .route("/v1/health", get(handlers::health_check));

    // Iroh P2P ticket endpoint — returns JSON EndpointAddr for peer bootstrap.
    // GET /v1/iroh/node-addr → {"node_id": "<hex>", "ticket": "<json-endpoint-addr>"}
    let ticket_runner = runner.clone();
    let app = app.route(
        "/v1/iroh/node-addr",
        get(move || {
            let runner = ticket_runner.clone();
            async move {
                let guard = runner.lock().await;
                let node_id = guard.iroh_node_id();
                #[cfg(all(feature = "identity", feature = "swarm"))]
                {
                    match guard.iroh_ticket() {
                        Some(Ok(ticket)) => (
                            StatusCode::OK,
                            axum::Json(serde_json::json!({
                                "node_id": node_id,
                                "ticket": ticket,
                            })),
                        )
                            .into_response(),
                        Some(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
                        None => (
                            StatusCode::SERVICE_UNAVAILABLE,
                            "P2P not initialised — start with swarm+identity features",
                        )
                            .into_response(),
                    }
                }
                #[cfg(not(all(feature = "identity", feature = "swarm")))]
                (
                    StatusCode::NOT_IMPLEMENTED,
                    axum::Json(serde_json::json!({
                        "node_id": node_id,
                        "ticket": null,
                        "note": "swarm+identity features not enabled",
                    })),
                )
                    .into_response()
            }
        }),
    );

    // WebSocket route — captures the runner Arc (clone per-request for Fn trait)
    let ws_runner = runner.clone();
    let app = app.route(
        "/v1/ws/live",
        get(move |ws_upgrade: WebSocketUpgrade| {
            let runner = ws_runner.clone();
            async move { ws::ws_handler(ws_upgrade, runner).await }
        }),
    );

    // Static file serving (if the directory exists)
    let app = if static_dir.exists() {
        app.fallback_service(tower_http::services::ServeDir::new(static_dir))
    } else {
        app.fallback(|| async { (StatusCode::NOT_FOUND, "No static directory found") })
    };

    let app = app
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(cors)
        .layer(DefaultBodyLimit::max(config.max_body_bytes))
        .with_state(state);

    Ok(app)
}

/// Create a demo router with custom security configuration.
pub fn create_demo_router_with_config(
    config: ApiConfig,
) -> Result<Router, Box<dyn std::error::Error>> {
    let runner = Arc::new(Mutex::new(demo_runner::DemoRunner::new()?));
    build_demo_router(runner, config)
}

/// Start the demo server with WebSocket and static file serving.
pub async fn serve_demo(addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    serve_demo_with_config(addr, ApiConfig::default()).await
}

/// Start the demo server with custom security configuration.
pub async fn serve_demo_with_config(
    addr: &str,
    config: ApiConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build runner before wrapping so we can do async P2P/mesh init.
    let mut runner = demo_runner::DemoRunner::new()?;
    #[cfg(all(feature = "identity", feature = "swarm"))]
    runner.enable_p2p().await;
    #[cfg(feature = "mesh")]
    runner.enable_mesh_daemon().await;
    let app = build_demo_router(Arc::new(Mutex::new(runner)), config)?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Symthaea Demo running at http://{}", addr);
    println!("  Live visualization: http://{}", addr);
    println!("  WebSocket endpoint: ws://{}/v1/ws/live", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::util::ServiceExt;

    fn authed_router() -> Router {
        create_router_with_config(ApiConfig {
            bearer_token: Some("test-token".to_string()),
            ..ApiConfig::default()
        })
    }

    #[test]
    fn route_policy_marks_public_get_routes() {
        assert!(route_is_public_without_auth(
            &Method::GET,
            "/v1/leaderboard"
        ));
        assert!(route_is_public_without_auth(&Method::GET, "/v1/datasets"));
        assert!(route_is_public_without_auth(
            &Method::GET,
            "/v1/results/submission-1"
        ));
        assert!(!route_is_public_without_auth(
            &Method::GET,
            "/v1/audit/events"
        ));
        assert!(!route_is_public_without_auth(&Method::POST, "/v1/submit"));
    }

    #[tokio::test]
    async fn public_routes_remain_accessible_without_auth() {
        let app = authed_router();

        for request in [
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("health request"),
            Request::builder()
                .uri("/v1/leaderboard")
                .body(Body::empty())
                .expect("leaderboard request"),
            Request::builder()
                .uri("/v1/datasets")
                .body(Body::empty())
                .expect("datasets request"),
            Request::builder()
                .uri("/v1/results/non-existent")
                .body(Body::empty())
                .expect("results request"),
        ] {
            let response = app.clone().oneshot(request).await.expect("route response");
            assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
        }
    }

    #[tokio::test]
    async fn protected_routes_require_auth_when_configured() {
        let app = authed_router();

        for request in [
            Request::builder()
                .uri("/v1/audit/events")
                .body(Body::empty())
                .expect("audit request"),
            Request::builder()
                .method(Method::POST)
                .uri("/v1/submit")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model_name":"test-model","topology_type":"ring","n_nodes":8}"#,
                ))
                .expect("submit request"),
            Request::builder()
                .method(Method::POST)
                .uri("/v1/compare")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model_a":"a","model_b":"b"}"#))
                .expect("compare request"),
        ] {
            let response = app.clone().oneshot(request).await.expect("route response");
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        }
    }

    #[tokio::test]
    async fn authenticated_submit_is_not_rejected_by_middleware() {
        let app = authed_router();
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/submit")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(
                        r#"{"model_name":"auth-model","topology_type":"ring","n_nodes":8}"#,
                    ))
                    .expect("submit request"),
            )
            .await
            .expect("route response");

        assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
