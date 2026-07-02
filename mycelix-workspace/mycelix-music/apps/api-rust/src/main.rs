// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Music API - Rust/Axum Implementation
//!
//! This is the high-performance Rust backend for Mycelix Music,
//! designed for ecosystem consistency with Mycelix-Core and
//! future Holochain integration.

use axum::{
    routing::{get, post},
    Router,
    http::StatusCode,
    Json,
    extract::State,
};
use ethers::types::Address;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::{
    cors::{AllowOrigin, Any, CorsLayer},
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod routes;
mod services;
mod models;

use services::indexer::{IndexerConfig, spawn_indexer};
use services::blockchain::BlockchainService;

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub db_pool: sqlx::PgPool,
    pub redis: redis::Client,
    pub ipfs_client: ipfs_api_backend_hyper::IpfsClient,
    pub blockchain: Option<Arc<BlockchainService>>,
}

/// Health check response
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    services: ServiceStatus,
}

#[derive(Serialize)]
struct ServiceStatus {
    database: bool,
    redis: bool,
    ipfs: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "mycelix_music_api=debug,tower_http=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load environment variables
    dotenvy::dotenv().ok();

    // Database connection
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://mycelix:mycelix_dev_pass@localhost:5432/mycelix_music".into());

    let db_pool = sqlx::PgPool::connect(&database_url).await?;
    tracing::info!("Connected to PostgreSQL");

    // Redis connection
    let redis_url = std::env::var("REDIS_URL")
        .unwrap_or_else(|_| "redis://localhost:6379".into());
    let redis = redis::Client::open(redis_url)?;
    tracing::info!("Connected to Redis");

    // IPFS client
    let ipfs_url = std::env::var("IPFS_API_URL")
        .unwrap_or_else(|_| "http://localhost:5001".into());
    let ipfs_client = ipfs_api_backend_hyper::IpfsClient::from_str(&ipfs_url)?;
    tracing::info!("Connected to IPFS");

    // Initialize blockchain service (if configured)
    let blockchain = if let Ok(router_address) = std::env::var("ROUTER_ADDRESS") {
        let rpc_url = std::env::var("RPC_URL")
            .unwrap_or_else(|_| "http://localhost:8545".into());

        // Check if we have a private key for signing transactions
        let blockchain_service = if let Ok(private_key) = std::env::var("PAYMENT_PRIVATE_KEY") {
            match BlockchainService::with_signer(&rpc_url, &router_address, &private_key) {
                Ok(service) => {
                    tracing::info!("Blockchain service initialized with signer for payments");
                    Some(Arc::new(service))
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize blockchain service with signer: {}", e);
                    // Fall back to read-only mode
                    BlockchainService::new(&rpc_url, &router_address)
                        .ok()
                        .map(Arc::new)
                }
            }
        } else {
            // Read-only mode (no signing capability)
            match BlockchainService::new(&rpc_url, &router_address) {
                Ok(service) => {
                    tracing::info!("Blockchain service initialized (read-only, no signer)");
                    Some(Arc::new(service))
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize blockchain service: {}", e);
                    None
                }
            }
        };

        // Start event indexer if blockchain is configured
        if let Ok(router_addr) = router_address.parse::<Address>() {
            let start_block = std::env::var("INDEXER_START_BLOCK")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            let indexer_config = IndexerConfig {
                rpc_url: rpc_url.clone(),
                router_address: router_addr,
                start_block,
                poll_interval_secs: 12, // ~1 block on Gnosis
                confirmations: 3,
            };

            tracing::info!(
                "Starting event indexer for router {:?} from block {}",
                router_addr,
                start_block
            );

            spawn_indexer(indexer_config, db_pool.clone());
        }

        blockchain_service
    } else {
        tracing::info!("Blockchain service disabled (ROUTER_ADDRESS not set)");
        None
    };

    // Create app state
    let state = Arc::new(AppState {
        db_pool,
        redis,
        ipfs_client,
        blockchain,
    });

    // Build router
    let permissive_cors = std::env::var("PERMISSIVE_CORS")
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);

    let allowed_origins = std::env::var("ALLOWED_ORIGINS").unwrap_or_default();
    let origins: Vec<_> = allowed_origins
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let cors = if permissive_cors {
        tracing::warn!("PERMISSIVE_CORS enabled: allowing any origin (insecure)");
        CorsLayer::permissive()
    } else if origins.is_empty() {
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
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods(Any)
            .allow_headers(Any)
    };

    let app = Router::new()
        // Health & Status
        .route("/health", get(health_check))
        .route("/", get(root))

        // Songs
        .route("/api/songs", get(routes::songs::list_songs))
        .route("/api/songs", post(routes::songs::create_song))
        .route("/api/songs/:id", get(routes::songs::get_song))
        .route("/api/songs/:id/play", post(routes::songs::record_play))

        // Artists
        .route("/api/artists/:address", get(routes::artists::get_artist))
        .route("/api/artists/:address/songs", get(routes::artists::get_artist_songs))

        // Analytics
        .route("/api/analytics/artist/:address", get(routes::analytics::artist_analytics))
        .route("/api/analytics/song/:id", get(routes::analytics::song_analytics))
        .route("/api/analytics/top-songs", get(routes::analytics::top_songs))

        // Uploads
        .route("/api/upload", post(routes::uploads::upload_file))

        // Economic Strategies
        .route("/api/strategies", get(routes::strategies::list_strategies))
        .route("/api/strategies/:id/preview", post(routes::strategies::preview_splits))

        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state);

    // Start server
    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port = std::env::var("PORT").unwrap_or_else(|_| "3100".into());
    let addr = format!("{}:{}", host, port);

    tracing::info!("🎵 Mycelix Music API starting on {}", addr);
    tracing::info!("   Vision: Default choice for the entire music industry");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Root endpoint
async fn root() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "name": "Mycelix Music API",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "Decentralized music platform with modular economics",
        "vision": "The default choice for the entire music industry",
        "features": [
            "10-50x artist earnings",
            "Instant settlements",
            "Zero-cost streaming (via Holochain)",
            "Community-owned infrastructure"
        ],
        "docs": "/docs",
        "health": "/health"
    }))
}

/// Health check endpoint
async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let db_ok = sqlx::query("SELECT 1")
        .fetch_one(&state.db_pool)
        .await
        .is_ok();

    let redis_ok = state.redis
        .get_connection()
        .map(|_| true)
        .unwrap_or(false);

    // IPFS check (simple version query)
    let ipfs_ok = state.ipfs_client
        .version()
        .await
        .is_ok();

    Json(HealthResponse {
        status: if db_ok && redis_ok { "healthy".into() } else { "degraded".into() },
        version: env!("CARGO_PKG_VERSION").into(),
        services: ServiceStatus {
            database: db_ok,
            redis: redis_ok,
            ipfs: ipfs_ok,
        },
    })
}
