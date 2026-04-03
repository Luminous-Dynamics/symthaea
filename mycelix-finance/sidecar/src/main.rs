// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Fiat Bridge Sidecar
//!
//! Axum REST service that bridges Stripe payment webhooks to
//! mycelix-finance Holochain zome calls via the MultiConductor.
//!
//! # Flow
//!
//! ```text
//! Stripe payment_intent.succeeded
//!   → POST /fiat/webhook
//!   → Validate HMAC-SHA256 signature
//!   → Extract recipient_did from metadata
//!   → Calculate SAP amount from fiat
//!   → Record FiatBridgeDeposit on DHT
//!   → Return 200 to Stripe
//! ```
//!
//! # Usage
//!
//! ```bash
//! STRIPE_WEBHOOK_SECRET=whsec_... cargo run
//! ```

mod config;
mod stripe;

use axum::routing::{get, post};
use axum::Router;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::trace::TraceLayer;

use config::Config;

/// Shared application state.
pub struct AppState {
    pub config: Config,
    /// Set of processed Stripe event IDs for idempotency.
    pub processed_events: Mutex<HashSet<String>>,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mycelix_fiat_bridge=info,tower_http=info".into()),
        )
        .init();

    // Load configuration
    let config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Configuration error: {e}");
            std::process::exit(1);
        }
    };

    let bind_addr = config.bind_addr();

    tracing::info!(
        host = %config.host,
        port = config.port,
        conductors = ?config.conductor_urls,
        "Starting Mycelix Fiat Bridge"
    );

    let state = Arc::new(AppState {
        config,
        processed_events: Mutex::new(HashSet::new()),
    });

    // Build router
    // Webhooks don't need CORS, but if this service is used from a browser in dev, keep it locked
    // to localhost instead of allowing arbitrary sites to drive localhost endpoints.
    let cors = CorsLayer::new()
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
        .allow_headers(Any);

    let app = Router::new()
        .route("/fiat/webhook", post(stripe::webhook_handler))
        .route("/health", get(health))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .expect("Failed to bind to address");

    tracing::info!("Fiat bridge listening on http://{bind_addr}");

    axum::serve(listener, app)
        .await
        .expect("Server error");
}

/// GET /health — Health check endpoint.
async fn health() -> &'static str {
    "ok"
}
