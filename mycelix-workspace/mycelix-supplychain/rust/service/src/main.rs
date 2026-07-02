// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix ERP Service
//!
//! REST API for blockchain-auditable enterprise resource planning:
//! - Supply Chain Management (SCM) - Event ingestion and verifiable claims
//! - Finance (FIN) - Double-entry bookkeeping with cryptographic signatures

use anyhow::Result;
use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use mycelix_erp_service::AppState;
use std::sync::Arc;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tracing::info;

// ERP modules
use mycelix_erp_service::fin;
use mycelix_erp_service::auth::{self, AuthState, auth_middleware};
use mycelix_erp_service::crm;
use mycelix_erp_service::hr;
use mycelix_erp_service::inventory;
use sqlx::postgres::PgPoolOptions;

#[tokio::main]
async fn main() -> Result<()> {
    // Load config (for now, use defaults)
    dotenvy::dotenv().ok();

    // Initialize structured logging
    mycelix_erp_service::observability::init_tracing();

    // Initialize health tracking
    mycelix_erp_service::health::init();

    // Generate keypair (in production, load from secure storage)
    let keypair = crypto::KeyPair::generate();
    info!("Service DID: {}", keypair.did());

    // Initialize database if DATABASE_URL is set
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite://data/claims.db".to_string());

    let db = match mycelix_erp_service::db::Database::new(&database_url).await {
        Ok(database) => {
            info!("Using SQLite database for storage");
            Some(database)
        }
        Err(e) => {
            tracing::warn!(
                "Failed to initialize database: {}. Falling back to in-memory storage",
                e
            );
            None
        }
    };

    // Create app state
    let state = Arc::new(AppState {
        keypair,
        db,
        claims: tokio::sync::RwLock::new(std::collections::HashMap::new()),
        pool: None,  // Will be set when FIN_DATABASE_URL is available
    });

    // Initialize Finance module (PostgreSQL)
    let fin_state = match std::env::var("FIN_DATABASE_URL") {
        Ok(fin_db_url) => {
            info!("Initializing Finance module with PostgreSQL");
            match PgPoolOptions::new()
                .max_connections(10)
                .connect(&fin_db_url)
                .await
            {
                Ok(pool) => {
                    info!("✅ Finance database connected");
                    let ledger = fin::ledger::LedgerService::new(pool.clone());
                    let invoicing = fin::invoicing::InvoicingService::new(pool.clone());
                    let payments = fin::payments::PaymentService::new(pool.clone());
                    Some(fin::api::FinState {
                        ledger,
                        invoicing,
                        payments,
                    })
                }
                Err(e) => {
                    tracing::warn!("Failed to connect to Finance database: {}. FIN endpoints will be unavailable", e);
                    None
                }
            }
        }
        Err(_) => {
            info!("FIN_DATABASE_URL not set. Finance module disabled.");
            None
        }
    };

    // Configure CORS with environment-based origins
    use axum::http::Method;
    use std::time::Duration;

    let allowed_origins = std::env::var("ALLOWED_ORIGINS")
        .unwrap_or_else(|_| "http://localhost:3000,http://localhost:8080".to_string());

    // Parse allowed origins
    let origins: Vec<_> = allowed_origins
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let permissive_cors = std::env::var("PERMISSIVE_CORS")
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);

    let cors = if origins.is_empty() {
        if permissive_cors {
            tracing::warn!(
                "PERMISSIVE_CORS enabled and no valid CORS origins configured; allowing any origin (insecure)"
            );
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        } else {
            tracing::warn!(
                "No valid CORS origins configured; defaulting to localhost-only CORS. \
                 Set ALLOWED_ORIGINS or (insecure) PERMISSIVE_CORS=1"
            );
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
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers([
                    axum::http::header::CONTENT_TYPE,
                    axum::http::header::AUTHORIZATION,
                    axum::http::HeaderName::from_static("x-request-id"),
                ])
                .max_age(Duration::from_secs(3600))
        }
    } else {
        // Strict CORS with specific origins
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
                axum::http::HeaderName::from_static("x-request-id"),
            ])
            .max_age(Duration::from_secs(3600))  // Cache preflight for 1 hour
    };

    // Create rate limiter from environment
    let rate_limit_config = mycelix_erp_service::middleware::RateLimitConfig::from_env();
    let rate_limiter = mycelix_erp_service::middleware::create_rate_limiter(rate_limit_config.clone());

    info!(
        "Rate limiting: {} req/s (burst: {})",
        rate_limit_config.requests_per_second,
        rate_limit_config.burst_size
    );

    // Build router with security-first middleware ordering
    use tower_http::limit::RequestBodyLimitLayer;

    // Initialize JWT service for supply chain authentication
    let jwt_service = auth::JwtService::from_env();
    let scm_auth_state = AuthState::new(jwt_service.clone());

    // Check if authentication is required for supply chain endpoints
    let require_scm_auth = std::env::var("SCM_REQUIRE_AUTH")
        .map(|v| v.to_lowercase() != "false")
        .unwrap_or(true); // Default: require authentication in production

    // Public routes (no authentication required)
    // - Health checks for Kubernetes probes
    // - Metrics for monitoring
    // - VC verification (public verification is a feature)
    let public_routes: Router = Router::new()
        .route("/health", get(mycelix_erp_service::health::health))
        .route("/health/live", get(mycelix_erp_service::health::liveness))
        .route("/health/ready", get(mycelix_erp_service::health::readiness))
        .route("/metrics", get(mycelix_erp_service::api::metrics_endpoint))
        .route("/v1/verify", post(mycelix_erp_service::api::verify_vc))
        .with_state(state.clone());

    // Protected supply chain routes (JWT authentication required)
    // These endpoints handle sensitive supply chain operations:
    // - Event ingestion (creates verifiable credentials)
    // - Claim retrieval and search
    // - Lineage queries
    let protected_scm_routes: Router = Router::new()
        .route("/v1/events", post(mycelix_erp_service::api::ingest_event))
        .route("/v1/events/batch", post(mycelix_erp_service::batch::ingest_batch))
        .route("/v1/claims/:id", get(mycelix_erp_service::api::get_claim))
        .route("/v1/claims", get(mycelix_erp_service::lineage_api::search_claims))
        .route("/v1/batches/:batch_id/claims", get(mycelix_erp_service::lineage_api::get_batch_claims))
        .route("/v1/lineage/:batch_id", get(mycelix_erp_service::lineage_api::get_lineage))
        .with_state(state.clone());

    // Apply authentication middleware conditionally
    let mut app: Router = if require_scm_auth {
        info!("Supply chain endpoints require JWT authentication (SCM_REQUIRE_AUTH=true)");
        public_routes.merge(
            protected_scm_routes.layer(middleware::from_fn_with_state(
                scm_auth_state.clone(),
                auth_middleware,
            ))
        )
    } else {
        tracing::warn!(
            "Supply chain endpoints accessible WITHOUT authentication (SCM_REQUIRE_AUTH=false). \
            This should only be used for development/testing!"
        );
        public_routes.merge(protected_scm_routes)
    };

    // Add Finance module routes if enabled (already has its own state)
    if let Some(fin) = fin_state.clone() {
        info!("✅ Finance module endpoints enabled at /v1/fin/*");
        app = app.merge(fin::api::router(fin));
    } else {
        info!("⚠️  Finance module disabled (FIN_DATABASE_URL not set)");
    }

    // Initialize Auth, CRM, and HR modules (requires FIN database)
    if let Some(_fin) = fin_state {
        // Reuse FIN database pool for all modules
        let fin_db_url = std::env::var("FIN_DATABASE_URL").unwrap();
        let shared_pool = PgPoolOptions::new()
            .max_connections(20)
            .connect(&fin_db_url)
            .await
            .expect("Failed to create shared database pool");

        // Auth module - reuse the jwt_service created earlier for consistency
        let auth_state = auth::AuthApiState::new(jwt_service.clone(), shared_pool.clone());
        info!("✅ Auth module endpoints enabled at /v1/auth/*");
        app = app.merge(auth::router(auth_state));

        // CRM module
        let crm_state = crm::api::CrmState::new(shared_pool.clone());
        info!("✅ CRM module endpoints enabled at /v1/crm/*");
        app = app.merge(crm::api::crm_router(crm_state));

        // HR module
        info!("✅ HR module endpoints enabled at /v1/hr/*");
        app = app.nest("/v1/hr", hr::api::hr_router(shared_pool.clone()));

        // Inventory module
        info!("✅ Inventory module endpoints enabled at /v1/inventory/*");
        app = app.nest("/v1/inventory", inventory::api::routes().with_state(
            mycelix_erp_service::PoolState { pool: shared_pool.clone() }
        ));
    } else {
        info!("⚠️  Auth/CRM/HR/Inventory modules disabled (requires FIN_DATABASE_URL)");
    }

    let app = app
        .layer(RequestBodyLimitLayer::new(2 * 1024 * 1024))  // 2MB max request size
        .layer(cors)
        .layer(middleware::from_fn(mycelix_erp_service::middleware::security_headers))
        .layer(middleware::from_fn(mycelix_erp_service::observability::request_logging_middleware))
        .layer(middleware::from_fn_with_state(
            rate_limiter,
            mycelix_erp_service::middleware::rate_limit_middleware
        ));

    // Start server (secure-by-default: localhost only)
    let addr = std::env::var("BIND_ADDRESS").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Enable graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown complete");
    Ok(())
}

/// Graceful shutdown handler
///
/// Listens for SIGTERM (Kubernetes) and SIGINT (Ctrl+C) signals.
/// Allows in-flight requests to complete before shutdown.
async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received SIGINT (Ctrl+C), starting graceful shutdown");
        },
        _ = terminate => {
            tracing::info!("Received SIGTERM, starting graceful shutdown");
        },
    }

    tracing::info!("Graceful shutdown initiated - finishing in-flight requests");
}
