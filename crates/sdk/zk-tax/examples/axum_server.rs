// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Production-ready Axum HTTP server for ZK tax proofs.
//!
//! Run with: `cargo run --example axum_server --features server`
//!
//! Endpoints:
//! - POST /api/v1/prove/bracket - Generate a tax bracket proof
//! - POST /api/v1/prove/range - Generate a range proof
//! - POST /api/v1/prove/batch - Generate a batch (multi-year) proof
//! - POST /api/v1/verify - Verify any proof
//! - GET /api/v1/jurisdictions - List supported jurisdictions
//! - GET /api/v1/brackets - Get brackets for a jurisdiction
//! - GET /api/v1/health - Health check
//! - GET /openapi.json - OpenAPI specification

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use mycelix_zk_tax::{
    api::*,
    proof::{BatchProofBuilder, RangeProofBuilder},
    TaxBracketProver, Jurisdiction, FilingStatus,
};
use serde::Deserialize;
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::Instant,
};
use tokio::sync::RwLock;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

// =============================================================================
// Application State
// =============================================================================

/// Shared application state
#[derive(Clone)]
struct AppState {
    /// The prover instance (thread-safe)
    prover: Arc<TaxBracketProver>,
    /// Simple in-memory cache for proof results
    cache: Arc<RwLock<HashMap<String, String>>>,
    /// Request counter for metrics
    request_count: Arc<RwLock<u64>>,
}

impl AppState {
    fn new(dev_mode: bool) -> Self {
        let prover = if dev_mode {
            TaxBracketProver::dev_mode()
        } else {
            TaxBracketProver::new()
        };

        Self {
            prover: Arc::new(prover),
            cache: Arc::new(RwLock::new(HashMap::new())),
            request_count: Arc::new(RwLock::new(0)),
        }
    }

    async fn increment_requests(&self) -> u64 {
        let mut count = self.request_count.write().await;
        *count += 1;
        *count
    }
}

// =============================================================================
// Route Handlers
// =============================================================================

/// Health check endpoint
async fn health_check(State(state): State<AppState>) -> Json<ApiResponse<HealthResponse>> {
    let request_count = *state.request_count.read().await;

    let mut health = HealthResponse::default();
    health.status = format!("healthy (requests: {})", request_count);

    Json(ApiResponse::success(health))
}

/// Generate a tax bracket proof
async fn prove_bracket(
    State(state): State<AppState>,
    Json(req): Json<BracketProofRequest>,
) -> Result<Json<ApiResponse<BracketProofResponse>>, AppError> {
    let start = Instant::now();
    state.increment_requests().await;

    // Parse inputs
    let jurisdiction = req.parse_jurisdiction()?;
    let filing_status = req.parse_filing_status()?;
    let income = req.income();

    info!(
        "Generating proof: income=${}, jurisdiction={:?}, status={:?}, year={}",
        income, jurisdiction, filing_status, req.tax_year
    );

    // Generate proof
    let proof = state.prover.prove(income, jurisdiction, filing_status, req.tax_year)?;

    let response = BracketProofResponse::from_proof(proof);
    let elapsed = start.elapsed().as_millis() as u64;

    Ok(Json(
        ApiResponse::success(response)
            .with_processing_time(elapsed)
    ))
}

/// Generate a range proof
async fn prove_range(
    State(state): State<AppState>,
    Json(req): Json<RangeProofRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, AppError> {
    let start = Instant::now();
    state.increment_requests().await;

    let income = req.income_cents / 100;
    let min = req.min_income_cents.map(|c| c / 100);
    let max = req.max_income_cents.map(|c| c / 100);

    let builder = RangeProofBuilder::new(income, req.tax_year);

    let proof = match (min, max) {
        (Some(min_val), Some(max_val)) => builder.prove_between(min_val, max_val)?,
        (Some(min_val), None) => builder.prove_above(min_val)?,
        (None, Some(max_val)) => builder.prove_below(max_val)?,
        (None, None) => builder.prove_above(0)?,  // Default: prove non-negative
    };
    let elapsed = start.elapsed().as_millis() as u64;

    Ok(Json(
        ApiResponse::success(serde_json::to_value(&proof)?)
            .with_processing_time(elapsed)
    ))
}

/// Generate a batch (multi-year) proof
async fn prove_batch(
    State(state): State<AppState>,
    Json(req): Json<BatchProofRequest>,
) -> Result<Json<ApiResponse<serde_json::Value>>, AppError> {
    let start = Instant::now();
    state.increment_requests().await;

    let jurisdiction = Jurisdiction::from_code(&req.jurisdiction)
        .ok_or_else(|| mycelix_zk_tax::Error::unsupported_jurisdiction(&req.jurisdiction))?;

    let filing_status = match req.filing_status.to_lowercase().as_str() {
        "single" => FilingStatus::Single,
        "mfj" | "married_filing_jointly" => FilingStatus::MarriedFilingJointly,
        "mfs" | "married_filing_separately" => FilingStatus::MarriedFilingSeparately,
        "hoh" | "head_of_household" => FilingStatus::HeadOfHousehold,
        _ => return Err(AppError::BadRequest(format!("Invalid filing status: {}", req.filing_status))),
    };

    let mut builder = BatchProofBuilder::new(jurisdiction, filing_status);

    for year_income in &req.years {
        builder = builder.add_year(year_income.year, year_income.income_cents / 100);
    }

    let proof = builder.build_dev()?;
    let elapsed = start.elapsed().as_millis() as u64;

    Ok(Json(
        ApiResponse::success(serde_json::to_value(&proof)?)
            .with_processing_time(elapsed)
    ))
}

/// Verify a proof
async fn verify_proof(
    State(state): State<AppState>,
    Json(req): Json<VerifyRequest>,
) -> Result<Json<ApiResponse<VerifyResponse>>, AppError> {
    let start = Instant::now();
    state.increment_requests().await;

    // Try to deserialize as TaxBracketProof
    let result = serde_json::from_value::<mycelix_zk_tax::TaxBracketProof>(req.proof.clone());

    let (valid, details) = match result {
        Ok(proof) => {
            match proof.verify() {
                Ok(_) => (true, Some("Proof verified successfully".to_string())),
                Err(e) => (false, Some(format!("Verification failed: {}", e))),
            }
        }
        Err(e) => (false, Some(format!("Invalid proof format: {}", e))),
    };

    let elapsed = start.elapsed().as_millis() as u64;

    Ok(Json(
        ApiResponse::success(VerifyResponse { valid, details })
            .with_processing_time(elapsed)
    ))
}

/// List supported jurisdictions
async fn list_jurisdictions(
    State(state): State<AppState>,
) -> Json<ApiResponse<JurisdictionListResponse>> {
    state.increment_requests().await;

    let jurisdictions = vec![
        JurisdictionInfo {
            code: "US".to_string(),
            name: "United States".to_string(),
            filing_statuses: vec!["single".to_string(), "mfj".to_string(), "mfs".to_string(), "hoh".to_string()],
            tax_years: vec![2020, 2021, 2022, 2023, 2024, 2025],
        },
        JurisdictionInfo {
            code: "UK".to_string(),
            name: "United Kingdom".to_string(),
            filing_statuses: vec!["single".to_string()],
            tax_years: vec![2020, 2021, 2022, 2023, 2024, 2025],
        },
        JurisdictionInfo {
            code: "DE".to_string(),
            name: "Germany".to_string(),
            filing_statuses: vec!["single".to_string(), "mfj".to_string()],
            tax_years: vec![2020, 2021, 2022, 2023, 2024, 2025],
        },
        JurisdictionInfo {
            code: "CA".to_string(),
            name: "Canada".to_string(),
            filing_statuses: vec!["single".to_string()],
            tax_years: vec![2020, 2021, 2022, 2023, 2024, 2025],
        },
        JurisdictionInfo {
            code: "AU".to_string(),
            name: "Australia".to_string(),
            filing_statuses: vec!["single".to_string()],
            tax_years: vec![2020, 2021, 2022, 2023, 2024, 2025],
        },
        // Add more as needed
    ];

    Json(ApiResponse::success(JurisdictionListResponse { jurisdictions }))
}

/// Query parameters for brackets endpoint
#[derive(Deserialize)]
struct BracketsQuery {
    jurisdiction: String,
    tax_year: u32,
    filing_status: Option<String>,
}

/// Get brackets for a jurisdiction
async fn get_brackets(
    State(state): State<AppState>,
    Query(query): Query<BracketsQuery>,
) -> Result<Json<ApiResponse<BracketsResponse>>, AppError> {
    state.increment_requests().await;

    let jurisdiction = Jurisdiction::from_code(&query.jurisdiction)
        .ok_or_else(|| AppError::BadRequest(format!("Unknown jurisdiction: {}", query.jurisdiction)))?;

    let filing_status = query.filing_status.as_deref().unwrap_or("single");

    // Get brackets from the SDK
    let brackets = mycelix_zk_tax::brackets::get_brackets(
        jurisdiction,
        query.tax_year,
        FilingStatus::Single, // Default
    )?;

    let bracket_infos: Vec<BracketInfo> = brackets
        .iter()
        .enumerate()
        .map(|(i, b)| BracketInfo {
            index: i as u8,
            rate_percent: b.rate_bps as f64 / 100.0,
            lower_bound: b.lower,
            upper_bound: b.upper,
        })
        .collect();

    Ok(Json(ApiResponse::success(BracketsResponse {
        jurisdiction: query.jurisdiction,
        tax_year: query.tax_year,
        filing_status: filing_status.to_string(),
        brackets: bracket_infos,
    })))
}

/// OpenAPI specification
async fn openapi_spec() -> impl IntoResponse {
    let spec = mycelix_zk_tax::api::generate_openapi_spec();
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        spec,
    )
}

// =============================================================================
// Error Handling
// =============================================================================

/// Application error type
#[derive(Debug)]
enum AppError {
    Sdk(mycelix_zk_tax::Error),
    BadRequest(String),
    Serialization(serde_json::Error),
}

impl From<mycelix_zk_tax::Error> for AppError {
    fn from(e: mycelix_zk_tax::Error) -> Self {
        AppError::Sdk(e)
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError::Serialization(e)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self {
            AppError::Sdk(e) => (StatusCode::BAD_REQUEST, e.to_string()),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AppError::Serialization(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };

        let body = Json(ApiResponse::<()>::error(ApiError {
            code: match &self {
                AppError::Sdk(e) => e.code().to_string(),
                AppError::BadRequest(_) => "BAD_REQUEST".to_string(),
                AppError::Serialization(_) => "SERIALIZATION_ERROR".to_string(),
            },
            message: message.lines().next().unwrap_or("Unknown error").to_string(),
            details: Some(message),
        }));

        (status, body).into_response()
    }
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() {
    // Initialize tracing
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    // Check for dev mode
    let dev_mode = std::env::var("DEV_MODE").unwrap_or_default() == "true";
    if dev_mode {
        warn!("Running in DEV MODE - proofs are not cryptographically secure!");
    }

    // Create app state
    let state = AppState::new(dev_mode);

    // Build router
    let app = Router::new()
        // Proof generation
        .route("/api/v1/prove/bracket", post(prove_bracket))
        .route("/api/v1/prove/range", post(prove_range))
        .route("/api/v1/prove/batch", post(prove_batch))
        // Verification
        .route("/api/v1/verify", post(verify_proof))
        // Discovery
        .route("/api/v1/jurisdictions", get(list_jurisdictions))
        .route("/api/v1/brackets", get(get_brackets))
        // System
        .route("/api/v1/health", get(health_check))
        .route("/openapi.json", get(openapi_spec))
        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);

    // Start server
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Starting ZK Tax Proof API server on http://{}", addr);
    info!("OpenAPI spec available at http://{}/openapi.json", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
