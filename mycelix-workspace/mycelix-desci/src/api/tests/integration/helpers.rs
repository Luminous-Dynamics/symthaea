// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test helpers
//!
//! Provides utilities for integration testing the Mycelix-DeSci API.

use axum::{
    extract::DefaultBodyLimit,
    http::{header, HeaderValue, Method},
    Router,
};
use mycelix_desci_api::state::AppState;
use mycelix_desci_api::routes;
use mycelix_desci_api::handlers;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

/// Create a test application instance
async fn create_test_app() -> Result<Router, Box<dyn std::error::Error>> {
    // Create application state
    let state = AppState::new().await?;

    // Create OpenAPI documentation
    let openapi = routes::ApiDoc::openapi();

    Ok(Router::new()
        // Mount API routes at /api/v1
        .nest("/api/v1", routes::api_routes())
        // Swagger UI at /docs
        .merge(SwaggerUi::new("/docs").url("/api-docs/openapi.json", openapi))
        // Health check at root
        .route("/health", axum::routing::get(handlers::system::health_check))
        // Application state
        .with_state(Arc::new(state))
        // Middleware stack
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CompressionLayer::new())
                .layer(configure_cors())
                .layer(TimeoutLayer::new(std::time::Duration::from_secs(30)))
                .layer(DefaultBodyLimit::max(10 * 1024 * 1024)),
        ))
}

/// Configure CORS middleware for tests
fn configure_cors() -> CorsLayer {
    CorsLayer::new()
        .allow_origin("*".parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
}

/// Test server instance
pub struct TestServer {
    pub addr: SocketAddr,
    pub base_url: String,
    pub client: reqwest::Client,
    _shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl TestServer {
    /// Start a new test server on a random port
    pub async fn start() -> Result<Self, Box<dyn std::error::Error>> {
        // Create application (using the test_utils module we'll add to the API crate)
        let app = create_test_app().await?;

        // Bind to random port
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let base_url = format!("http://{}", addr);

        // Create shutdown channel
        let (_shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        // Spawn the server
        tokio::spawn(async move {
            let server = axum::serve(listener, app);
            let graceful = server.with_graceful_shutdown(async {
                shutdown_rx.await.ok();
            });
            graceful.await.ok();
        });

        // Wait a bit for server to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(Self {
            addr,
            base_url,
            client: reqwest::Client::new(),
            _shutdown_tx,
        })
    }

    /// Get the API base URL
    pub fn api_url(&self, path: &str) -> String {
        format!("{}/api/v1{}", self.base_url, path)
    }

    /// GET request helper
    pub async fn get(&self, path: &str) -> reqwest::Response {
        self.client
            .get(self.api_url(path))
            .send()
            .await
            .expect("Failed to send GET request")
    }

    /// POST request helper
    pub async fn post<T: serde::Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> reqwest::Response {
        self.client
            .post(self.api_url(path))
            .json(body)
            .send()
            .await
            .expect("Failed to send POST request")
    }

    /// PUT request helper
    pub async fn put<T: serde::Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> reqwest::Response {
        self.client
            .put(self.api_url(path))
            .json(body)
            .send()
            .await
            .expect("Failed to send PUT request")
    }
}

/// Create a sample claim request
pub fn sample_claim_request(tier: &str, category: &str, creator: &str) -> serde_json::Value {
    serde_json::json!({
        "tier": tier,
        "content": {
            "dataset_hash": format!("blake3:test{}", uuid::Uuid::new_v4()),
            "description": format!("Test claim for {}", category),
            "category": category,
            "keywords": vec!["test", category]
        },
        "creator": creator
    })
}

/// Create a sample verification request
pub fn sample_verification_request(verifier: &str) -> serde_json::Value {
    serde_json::json!({
        "verifier": verifier,
        "signature": format!("{:064x}", rand::random::<u64>()),
        "notes": format!("Verification by {}", verifier)
    })
}

/// Create a sample provenance request
pub fn sample_provenance_request(source_id: Option<&str>) -> serde_json::Value {
    serde_json::json!({
        "source_id": source_id,
        "transformation": "test_transformation",
        "notes": "Test provenance entry"
    })
}
