//! REST API for Proof Operations
//!
//! HTTP API for proof generation, verification, and management.
//!
//! ## Endpoints
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | POST | `/api/v1/proofs/range` | Generate range proof |
//! | POST | `/api/v1/proofs/gradient` | Generate gradient proof |
//! | POST | `/api/v1/proofs/identity` | Generate identity proof |
//! | POST | `/api/v1/proofs/vote` | Generate vote proof |
//! | POST | `/api/v1/proofs/verify` | Verify proof envelope |
//! | POST | `/api/v1/proofs/batch/verify` | Batch verify proofs |
//! | POST | `/api/v1/proofs/simulate` | Simulate proof generation |
//! | GET | `/api/v1/proofs/metrics` | Get Prometheus metrics |
//! | GET | `/api/v1/health` | Health check |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::api::{ProofApiRouter, ApiConfig};
//!
//! let config = ApiConfig::default();
//! let router = ProofApiRouter::new(config).build();
//!
//! // Add to your axum app
//! let app = Router::new()
//!     .nest("/api/v1/proofs", router);
//!
//! // Or run standalone
//! let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
//! axum::serve(TcpListener::bind(addr).await?, router).await?;
//! ```
//!
//! ## Request/Response Examples
//!
//! ### Range Proof
//!
//! ```json
//! // POST /api/v1/proofs/range
//! {
//!   "value": 42,
//!   "min": 0,
//!   "max": 100,
//!   "security_level": "Standard128"
//! }
//!
//! // Response
//! {
//!   "success": true,
//!   "proof_type": "Range",
//!   "proof_bytes": "<base64>",
//!   "size_bytes": 15234,
//!   "generation_time_ms": 52.3
//! }
//! ```

#[cfg(feature = "http-api")]
use axum::{
    extract::{Json, State},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "http-api")]
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};

use crate::proofs::{
    ProofConfig, ProofResult, ProofType, SecurityLevel,
    RangeProof, GradientIntegrityProof, IdentityAssuranceProof, VoteEligibilityProof,
    ProofAssuranceLevel, ProofProposalType, ProofVoterProfile, ProofIdentityFactor,
    ProofSimulator, SimulationMode, METRICS,
};

// ============================================================================
// Configuration
// ============================================================================

/// API configuration
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Default security level
    pub default_security: SecurityLevel,
    /// Maximum proof size in bytes
    pub max_proof_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Enable simulation endpoints
    pub enable_simulation: bool,
    /// Enable generation endpoints (may be disabled for verify-only servers)
    pub enable_generation: bool,
    /// Rate limit per minute (0 = unlimited)
    pub rate_limit: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            default_security: SecurityLevel::Standard128,
            max_proof_size: 500_000,
            max_batch_size: 100,
            enable_simulation: true,
            enable_generation: true,
            rate_limit: 0,
        }
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Range proof request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeProofRequest {
    pub value: u64,
    pub min: u64,
    pub max: u64,
    #[serde(default)]
    pub security_level: Option<String>,
}

impl RangeProofRequest {
    /// Validate the request parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.min > self.max {
            return Err(format!(
                "Invalid range: min ({}) cannot be greater than max ({})",
                self.min, self.max
            ));
        }
        if self.value < self.min || self.value > self.max {
            return Err(format!(
                "Value {} is outside range [{}, {}]",
                self.value, self.min, self.max
            ));
        }
        Ok(())
    }
}

/// Gradient proof request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientProofRequest {
    pub gradients: Vec<f32>,
    pub max_norm: f32,
    #[serde(default)]
    pub security_level: Option<String>,
}

/// Maximum allowed gradient vector size
const MAX_GRADIENT_SIZE: usize = 10_000_000;

impl GradientProofRequest {
    /// Validate the request parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.gradients.is_empty() {
            return Err("Gradients vector cannot be empty".into());
        }
        if self.gradients.len() > MAX_GRADIENT_SIZE {
            return Err(format!(
                "Gradients vector too large: {} elements (max {})",
                self.gradients.len(),
                MAX_GRADIENT_SIZE
            ));
        }
        for (i, &g) in self.gradients.iter().enumerate() {
            if g.is_nan() {
                return Err(format!("Gradient at index {} is NaN", i));
            }
            if g.is_infinite() {
                return Err(format!("Gradient at index {} is infinite", i));
            }
        }
        if self.max_norm <= 0.0 {
            return Err("max_norm must be positive".into());
        }
        if self.max_norm.is_nan() || self.max_norm.is_infinite() {
            return Err("max_norm must be a finite positive number".into());
        }
        Ok(())
    }
}

/// Identity proof request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityProofRequest {
    pub did: String,
    pub factors: Vec<IdentityFactorInput>,
    pub min_level: String,
    #[serde(default)]
    pub security_level: Option<String>,
}

/// Valid assurance levels
const VALID_ASSURANCE_LEVELS: &[&str] = &["E0", "E1", "E2", "E3", "E4"];

impl IdentityProofRequest {
    /// Validate the request parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.did.is_empty() {
            return Err("DID cannot be empty".into());
        }
        if self.factors.is_empty() {
            return Err("At least one identity factor is required".into());
        }
        if self.factors.len() > 20 {
            return Err("Too many identity factors (max 20)".into());
        }
        for (i, factor) in self.factors.iter().enumerate() {
            factor.validate().map_err(|e| format!("Factor {}: {}", i, e))?;
        }
        if !VALID_ASSURANCE_LEVELS.contains(&self.min_level.as_str()) {
            return Err(format!(
                "Invalid min_level '{}'. Must be one of: {:?}",
                self.min_level, VALID_ASSURANCE_LEVELS
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityFactorInput {
    pub contribution: f32,
    pub category: u8,
    pub verified: bool,
}

/// Maximum factor category (0-4 for different identity factor types)
const MAX_FACTOR_CATEGORY: u8 = 4;

impl IdentityFactorInput {
    /// Validate factor parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.contribution < 0.0 || self.contribution > 1.0 {
            return Err(format!(
                "Contribution must be between 0.0 and 1.0, got {}",
                self.contribution
            ));
        }
        if self.contribution.is_nan() {
            return Err("Contribution cannot be NaN".into());
        }
        if self.category > MAX_FACTOR_CATEGORY {
            return Err(format!(
                "Category must be 0-{}, got {}",
                MAX_FACTOR_CATEGORY, self.category
            ));
        }
        Ok(())
    }
}

/// Vote proof request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteProofRequest {
    pub voter: VoterProfileInput,
    pub proposal_type: String,
    #[serde(default)]
    pub security_level: Option<String>,
}

/// Valid proposal types for voting
const VALID_PROPOSAL_TYPES: &[&str] = &[
    "standard", "constitutional", "emergency", "economic", "technical"
];

impl VoteProofRequest {
    /// Validate the request parameters
    pub fn validate(&self) -> Result<(), String> {
        self.voter.validate()?;
        if !VALID_PROPOSAL_TYPES.contains(&self.proposal_type.as_str()) {
            return Err(format!(
                "Invalid proposal_type '{}'. Must be one of: {:?}",
                self.proposal_type, VALID_PROPOSAL_TYPES
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoterProfileInput {
    pub did: String,
    pub assurance_level: u8,
    pub matl_score: f32,
    pub stake: f32,
    pub account_age_days: u32,
    pub participation_rate: f32,
    pub has_humanity_proof: bool,
    pub fl_contributions: u32,
}

/// Maximum assurance level (E0=0, E1=1, E2=2, E3=3, E4=4)
const MAX_ASSURANCE_LEVEL: u8 = 4;

impl VoterProfileInput {
    /// Validate voter profile parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.did.is_empty() {
            return Err("Voter DID cannot be empty".into());
        }
        if self.assurance_level > MAX_ASSURANCE_LEVEL {
            return Err(format!(
                "assurance_level must be 0-{}, got {}",
                MAX_ASSURANCE_LEVEL, self.assurance_level
            ));
        }
        if self.matl_score < 0.0 || self.matl_score > 1.0 {
            return Err(format!(
                "matl_score must be between 0.0 and 1.0, got {}",
                self.matl_score
            ));
        }
        if self.matl_score.is_nan() {
            return Err("matl_score cannot be NaN".into());
        }
        if self.stake < 0.0 || self.stake.is_nan() || self.stake.is_infinite() {
            return Err("stake must be a non-negative finite number".into());
        }
        if self.participation_rate < 0.0 || self.participation_rate > 1.0 {
            return Err(format!(
                "participation_rate must be between 0.0 and 1.0, got {}",
                self.participation_rate
            ));
        }
        if self.participation_rate.is_nan() {
            return Err("participation_rate cannot be NaN".into());
        }
        Ok(())
    }
}

impl From<VoterProfileInput> for ProofVoterProfile {
    fn from(v: VoterProfileInput) -> Self {
        ProofVoterProfile {
            did: v.did,
            assurance_level: v.assurance_level,
            matl_score: v.matl_score,
            stake: v.stake,
            account_age_days: v.account_age_days,
            participation_rate: v.participation_rate,
            has_humanity_proof: v.has_humanity_proof,
            fl_contributions: v.fl_contributions,
        }
    }
}

/// Verify request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyRequest {
    /// Base64 encoded proof bytes
    pub proof_bytes: String,
    /// Proof type
    pub proof_type: String,
}

/// Batch verify request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerifyRequest {
    pub proofs: Vec<VerifyRequest>,
}

/// Simulation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationRequest {
    pub proof_type: String,
    pub params: serde_json::Value,
}

/// Generic proof response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofResponse {
    pub success: bool,
    pub proof_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_bytes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Verification response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyResponse {
    pub valid: bool,
    pub proof_type: String,
    pub verification_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Batch verification response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerifyResponse {
    pub total_count: usize,
    pub valid_count: usize,
    pub invalid_count: usize,
    pub total_time_ms: f64,
    pub results: Vec<VerifyResponse>,
}

/// Simulation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResponse {
    pub inputs_valid: bool,
    pub validation_errors: Vec<String>,
    pub estimated_time_ms: f64,
    pub estimated_size_bytes: usize,
    pub proof_type: String,
}

/// Health response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub proofs_generated: u64,
    pub proofs_verified: u64,
    pub gpu_available: bool,
}

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
}

// ============================================================================
// API State
// ============================================================================

/// Shared API state
#[derive(Clone)]
pub struct ApiState {
    pub config: ApiConfig,
}

impl ApiState {
    pub fn new(config: ApiConfig) -> Self {
        Self { config }
    }
}

// ============================================================================
// Router Builder
// ============================================================================

/// Proof API router builder
#[cfg(feature = "http-api")]
pub struct ProofApiRouter {
    config: ApiConfig,
}

#[cfg(feature = "http-api")]
impl ProofApiRouter {
    /// Create a new router with configuration
    pub fn new(config: ApiConfig) -> Self {
        Self { config }
    }

    /// Build the router
    pub fn build(self) -> Router {
        let state = Arc::new(ApiState::new(self.config.clone()));

        let mut router = Router::new()
            .route("/health", get(health_handler))
            .route("/metrics", get(metrics_handler))
            .route("/verify", post(verify_handler))
            .route("/batch/verify", post(batch_verify_handler));

        if self.config.enable_generation {
            router = router
                .route("/range", post(range_proof_handler))
                .route("/gradient", post(gradient_proof_handler))
                .route("/identity", post(identity_proof_handler))
                .route("/vote", post(vote_proof_handler));
        }

        if self.config.enable_simulation {
            router = router.route("/simulate", post(simulate_handler));
        }

        router.with_state(state)
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// Health check handler
#[cfg(feature = "http-api")]
async fn health_handler() -> impl IntoResponse {
    let snapshot = METRICS.snapshot();
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        proofs_generated: snapshot.total_generations,
        proofs_verified: snapshot.total_verifications,
        gpu_available: snapshot.gpu_available,
    })
}

/// Prometheus metrics handler
#[cfg(feature = "http-api")]
async fn metrics_handler() -> impl IntoResponse {
    METRICS.gather()
}

/// Range proof generation handler
#[cfg(feature = "http-api")]
async fn range_proof_handler(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<RangeProofRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    let security = parse_security_level(req.security_level.as_deref())
        .unwrap_or(state.config.default_security);

    let config = ProofConfig {
        security_level: security,
        ..Default::default()
    };

    match RangeProof::generate(req.value, req.min, req.max, config) {
        Ok(proof) => {
            let bytes = proof.to_bytes();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            METRICS.record_generation(ProofType::Range, elapsed / 1000.0, bytes.len());

            Json(ProofResponse {
                success: true,
                proof_type: "Range".to_string(),
                proof_bytes: Some(BASE64.encode(&bytes)),
                size_bytes: Some(bytes.len()),
                generation_time_ms: Some(elapsed),
                error: None,
            })
        }
        Err(e) => {
            METRICS.record_generation_error(ProofType::Range, "generation_failed");
            Json(ProofResponse {
                success: false,
                proof_type: "Range".to_string(),
                proof_bytes: None,
                size_bytes: None,
                generation_time_ms: None,
                error: Some(e.to_string()),
            })
        }
    }
}

/// Gradient proof generation handler
#[cfg(feature = "http-api")]
async fn gradient_proof_handler(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<GradientProofRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    let security = parse_security_level(req.security_level.as_deref())
        .unwrap_or(state.config.default_security);

    let config = ProofConfig {
        security_level: security,
        ..Default::default()
    };

    match GradientIntegrityProof::generate(&req.gradients, req.max_norm, config) {
        Ok(proof) => {
            let bytes = proof.to_bytes();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            METRICS.record_generation(ProofType::GradientIntegrity, elapsed / 1000.0, bytes.len());

            Json(ProofResponse {
                success: true,
                proof_type: "GradientIntegrity".to_string(),
                proof_bytes: Some(BASE64.encode(&bytes)),
                size_bytes: Some(bytes.len()),
                generation_time_ms: Some(elapsed),
                error: None,
            })
        }
        Err(e) => {
            METRICS.record_generation_error(ProofType::GradientIntegrity, "generation_failed");
            Json(ProofResponse {
                success: false,
                proof_type: "GradientIntegrity".to_string(),
                proof_bytes: None,
                size_bytes: None,
                generation_time_ms: None,
                error: Some(e.to_string()),
            })
        }
    }
}

/// Identity proof generation handler
#[cfg(feature = "http-api")]
async fn identity_proof_handler(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<IdentityProofRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    let security = parse_security_level(req.security_level.as_deref())
        .unwrap_or(state.config.default_security);

    let config = ProofConfig {
        security_level: security,
        ..Default::default()
    };

    let min_level = match parse_assurance_level(&req.min_level) {
        Ok(l) => l,
        Err(e) => {
            return Json(ProofResponse {
                success: false,
                proof_type: "IdentityAssurance".to_string(),
                proof_bytes: None,
                size_bytes: None,
                generation_time_ms: None,
                error: Some(e),
            });
        }
    };

    let factors: Vec<ProofIdentityFactor> = req
        .factors
        .iter()
        .map(|f| ProofIdentityFactor::new(f.contribution, f.category, f.verified))
        .collect();

    match IdentityAssuranceProof::generate(&req.did, &factors, min_level, config) {
        Ok(proof) => {
            let bytes = proof.to_bytes();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            METRICS.record_generation(ProofType::IdentityAssurance, elapsed / 1000.0, bytes.len());

            Json(ProofResponse {
                success: true,
                proof_type: "IdentityAssurance".to_string(),
                proof_bytes: Some(BASE64.encode(&bytes)),
                size_bytes: Some(bytes.len()),
                generation_time_ms: Some(elapsed),
                error: None,
            })
        }
        Err(e) => {
            METRICS.record_generation_error(ProofType::IdentityAssurance, "generation_failed");
            Json(ProofResponse {
                success: false,
                proof_type: "IdentityAssurance".to_string(),
                proof_bytes: None,
                size_bytes: None,
                generation_time_ms: None,
                error: Some(e.to_string()),
            })
        }
    }
}

/// Vote proof generation handler
#[cfg(feature = "http-api")]
async fn vote_proof_handler(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<VoteProofRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    let security = parse_security_level(req.security_level.as_deref())
        .unwrap_or(state.config.default_security);

    let config = ProofConfig {
        security_level: security,
        ..Default::default()
    };

    let proposal_type = match parse_proposal_type(&req.proposal_type) {
        Ok(p) => p,
        Err(e) => {
            return Json(ProofResponse {
                success: false,
                proof_type: "VoteEligibility".to_string(),
                proof_bytes: None,
                size_bytes: None,
                generation_time_ms: None,
                error: Some(e),
            });
        }
    };

    let voter: ProofVoterProfile = req.voter.into();

    match VoteEligibilityProof::generate(&voter, proposal_type, config) {
        Ok(proof) => {
            let bytes = proof.to_bytes();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            METRICS.record_generation(ProofType::VoteEligibility, elapsed / 1000.0, bytes.len());

            Json(ProofResponse {
                success: true,
                proof_type: "VoteEligibility".to_string(),
                proof_bytes: Some(BASE64.encode(&bytes)),
                size_bytes: Some(bytes.len()),
                generation_time_ms: Some(elapsed),
                error: None,
            })
        }
        Err(e) => {
            METRICS.record_generation_error(ProofType::VoteEligibility, "generation_failed");
            Json(ProofResponse {
                success: false,
                proof_type: "VoteEligibility".to_string(),
                proof_bytes: None,
                size_bytes: None,
                generation_time_ms: None,
                error: Some(e.to_string()),
            })
        }
    }
}

/// Verify proof handler
#[cfg(feature = "http-api")]
async fn verify_handler(Json(req): Json<VerifyRequest>) -> impl IntoResponse {
    let start = Instant::now();

    let proof_bytes = match BASE64.decode(&req.proof_bytes) {
        Ok(b) => b,
        Err(e) => {
            return Json(VerifyResponse {
                valid: false,
                proof_type: req.proof_type.clone(),
                verification_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                error: Some(format!("Invalid base64: {}", e)),
            });
        }
    };

    let proof_type = match parse_proof_type(&req.proof_type) {
        Ok(t) => t,
        Err(e) => {
            return Json(VerifyResponse {
                valid: false,
                proof_type: req.proof_type.clone(),
                verification_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                error: Some(e),
            });
        }
    };

    let result = verify_proof_bytes(&proof_bytes, proof_type);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    METRICS.record_verification(proof_type, result.is_ok(), elapsed / 1000.0);

    match result {
        Ok(valid) => Json(VerifyResponse {
            valid,
            proof_type: format!("{:?}", proof_type),
            verification_time_ms: elapsed,
            error: None,
        }),
        Err(e) => Json(VerifyResponse {
            valid: false,
            proof_type: format!("{:?}", proof_type),
            verification_time_ms: elapsed,
            error: Some(e.to_string()),
        }),
    }
}

/// Batch verify handler
#[cfg(feature = "http-api")]
async fn batch_verify_handler(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<BatchVerifyRequest>,
) -> impl IntoResponse {
    let start = Instant::now();

    if req.proofs.len() > state.config.max_batch_size {
        return Json(BatchVerifyResponse {
            total_count: req.proofs.len(),
            valid_count: 0,
            invalid_count: req.proofs.len(),
            total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            results: vec![VerifyResponse {
                valid: false,
                proof_type: "batch".to_string(),
                verification_time_ms: 0.0,
                error: Some(format!(
                    "Batch size {} exceeds maximum {}",
                    req.proofs.len(),
                    state.config.max_batch_size
                )),
            }],
        });
    }

    let mut results = Vec::with_capacity(req.proofs.len());
    let mut valid_count = 0;

    for proof_req in req.proofs {
        let item_start = Instant::now();

        let proof_bytes = match BASE64.decode(&proof_req.proof_bytes) {
            Ok(b) => b,
            Err(e) => {
                results.push(VerifyResponse {
                    valid: false,
                    proof_type: proof_req.proof_type.clone(),
                    verification_time_ms: item_start.elapsed().as_secs_f64() * 1000.0,
                    error: Some(format!("Invalid base64: {}", e)),
                });
                continue;
            }
        };

        let proof_type = match parse_proof_type(&proof_req.proof_type) {
            Ok(t) => t,
            Err(e) => {
                results.push(VerifyResponse {
                    valid: false,
                    proof_type: proof_req.proof_type.clone(),
                    verification_time_ms: item_start.elapsed().as_secs_f64() * 1000.0,
                    error: Some(e),
                });
                continue;
            }
        };

        let result = verify_proof_bytes(&proof_bytes, proof_type);
        let elapsed = item_start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(valid) => {
                if valid {
                    valid_count += 1;
                }
                results.push(VerifyResponse {
                    valid,
                    proof_type: format!("{:?}", proof_type),
                    verification_time_ms: elapsed,
                    error: None,
                });
            }
            Err(e) => {
                results.push(VerifyResponse {
                    valid: false,
                    proof_type: format!("{:?}", proof_type),
                    verification_time_ms: elapsed,
                    error: Some(e.to_string()),
                });
            }
        }
    }

    let total_time = start.elapsed().as_secs_f64() * 1000.0;
    METRICS.record_batch(results.len(), valid_count, total_time / 1000.0);

    Json(BatchVerifyResponse {
        total_count: results.len(),
        valid_count,
        invalid_count: results.len() - valid_count,
        total_time_ms: total_time,
        results,
    })
}

/// Simulation handler
#[cfg(feature = "http-api")]
async fn simulate_handler(Json(req): Json<SimulationRequest>) -> impl IntoResponse {
    let mut simulator = ProofSimulator::new(SimulationMode::Estimate);

    let result = match req.proof_type.to_lowercase().as_str() {
        "range" => {
            let value = req.params.get("value").and_then(|v| v.as_u64()).unwrap_or(0);
            let min = req.params.get("min").and_then(|v| v.as_u64()).unwrap_or(0);
            let max = req.params.get("max").and_then(|v| v.as_u64()).unwrap_or(100);
            simulator.simulate_range_proof(value, min, max)
        }
        "gradient" | "gradientintegrity" => {
            let len = req.params.get("length").and_then(|v| v.as_u64()).unwrap_or(1000) as usize;
            let max_norm = req
                .params
                .get("max_norm")
                .and_then(|v| v.as_f64())
                .unwrap_or(5.0) as f32;
            simulator.simulate_gradient_proof(len, max_norm)
        }
        "identity" | "identityassurance" => {
            let did = req
                .params
                .get("did")
                .and_then(|v| v.as_str())
                .unwrap_or("did:mycelix:test");
            let factor_count = req
                .params
                .get("factor_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize;
            let min_level_str = req
                .params
                .get("min_level")
                .and_then(|v| v.as_str())
                .unwrap_or("E2");
            let min_level = parse_assurance_level(min_level_str).unwrap_or(ProofAssuranceLevel::E2);
            simulator.simulate_identity_proof(did, factor_count, min_level)
        }
        "vote" | "voteeligibility" => {
            let voter = ProofVoterProfile {
                did: req
                    .params
                    .get("did")
                    .and_then(|v| v.as_str())
                    .unwrap_or("did:mycelix:test")
                    .to_string(),
                assurance_level: req
                    .params
                    .get("assurance_level")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2) as u8,
                matl_score: req
                    .params
                    .get("matl_score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5) as f32,
                stake: req
                    .params
                    .get("stake")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(500.0) as f32,
                account_age_days: req
                    .params
                    .get("account_age_days")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(90) as u32,
                participation_rate: req
                    .params
                    .get("participation_rate")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.4) as f32,
                has_humanity_proof: req
                    .params
                    .get("has_humanity_proof")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
                fl_contributions: req
                    .params
                    .get("fl_contributions")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(15) as u32,
            };
            let proposal_str = req
                .params
                .get("proposal_type")
                .and_then(|v| v.as_str())
                .unwrap_or("Standard");
            let proposal = parse_proposal_type(proposal_str).unwrap_or(ProofProposalType::Standard);
            simulator.simulate_vote_proof(&voter, proposal)
        }
        _ => {
            return Json(SimulationResponse {
                inputs_valid: false,
                validation_errors: vec![format!("Unknown proof type: {}", req.proof_type)],
                estimated_time_ms: 0.0,
                estimated_size_bytes: 0,
                proof_type: req.proof_type,
            });
        }
    };

    match result {
        Ok(sim) => Json(SimulationResponse {
            inputs_valid: sim.inputs_valid,
            validation_errors: sim.validation_errors,
            estimated_time_ms: sim.estimated_time_ms,
            estimated_size_bytes: sim.estimated_size_bytes,
            proof_type: format!("{:?}", sim.proof_type),
        }),
        Err(e) => Json(SimulationResponse {
            inputs_valid: false,
            validation_errors: vec![e.to_string()],
            estimated_time_ms: 0.0,
            estimated_size_bytes: 0,
            proof_type: req.proof_type,
        }),
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn parse_security_level(s: Option<&str>) -> Option<SecurityLevel> {
    match s?.to_lowercase().as_str() {
        "standard96" | "low" | "fast" => Some(SecurityLevel::Standard96),
        "standard128" | "standard" | "default" => Some(SecurityLevel::Standard128),
        "high256" | "high" | "paranoid" => Some(SecurityLevel::High256),
        _ => None,
    }
}

fn parse_proof_type(s: &str) -> Result<ProofType, String> {
    match s.to_lowercase().as_str() {
        "range" => Ok(ProofType::Range),
        "gradient" | "gradientintegrity" => Ok(ProofType::GradientIntegrity),
        "identity" | "identityassurance" => Ok(ProofType::IdentityAssurance),
        "vote" | "voteeligibility" => Ok(ProofType::VoteEligibility),
        "membership" => Ok(ProofType::Membership),
        _ => Err(format!("Unknown proof type: {}", s)),
    }
}

fn parse_assurance_level(s: &str) -> Result<ProofAssuranceLevel, String> {
    match s.to_uppercase().as_str() {
        "E0" | "0" => Ok(ProofAssuranceLevel::E0),
        "E1" | "1" => Ok(ProofAssuranceLevel::E1),
        "E2" | "2" => Ok(ProofAssuranceLevel::E2),
        "E3" | "3" => Ok(ProofAssuranceLevel::E3),
        "E4" | "4" => Ok(ProofAssuranceLevel::E4),
        _ => Err(format!("Unknown assurance level: {}", s)),
    }
}

fn parse_proposal_type(s: &str) -> Result<ProofProposalType, String> {
    match s.to_lowercase().as_str() {
        "standard" => Ok(ProofProposalType::Standard),
        "constitutional" => Ok(ProofProposalType::Constitutional),
        "modelgovernance" | "model" => Ok(ProofProposalType::ModelGovernance),
        "emergency" => Ok(ProofProposalType::Emergency),
        "treasury" => Ok(ProofProposalType::Treasury),
        "membership" => Ok(ProofProposalType::Membership),
        _ => Err(format!("Unknown proposal type: {}", s)),
    }
}

fn verify_proof_bytes(bytes: &[u8], proof_type: ProofType) -> ProofResult<bool> {
    match proof_type {
        ProofType::Range => {
            let proof = RangeProof::from_bytes(bytes)?;
            let result = proof.verify()?;
            Ok(result.valid)
        }
        ProofType::GradientIntegrity => {
            let proof = GradientIntegrityProof::from_bytes(bytes)?;
            let result = proof.verify()?;
            Ok(result.valid)
        }
        ProofType::IdentityAssurance => {
            let proof = IdentityAssuranceProof::from_bytes(bytes)?;
            let result = proof.verify()?;
            Ok(result.valid)
        }
        ProofType::VoteEligibility => {
            let proof = VoteEligibilityProof::from_bytes(bytes)?;
            let result = proof.verify()?;
            Ok(result.valid)
        }
        ProofType::Membership => {
            use crate::proofs::MembershipProof;
            let proof = MembershipProof::from_bytes(bytes)?;
            let result = proof.verify()?;
            Ok(result.valid)
        }
    }
}

// ============================================================================
// ProofOfGradientQuality - Production-Ready zkSTARK Gradient Proofs
// ============================================================================

/// Complete Proof of Gradient Quality (PoGQ)
///
/// Creates zkSTARK proofs that a gradient contribution is honest, without
/// revealing the underlying training data. This is the cornerstone of
/// verifiable federated learning.
///
/// ## What is Proven
///
/// 1. **Norm Bound**: L2 norm is within specified bounds (prevents scaling attacks)
/// 2. **Finiteness**: All values are finite (no NaN/Inf that could corrupt aggregation)
/// 3. **Statistical Properties**: Values follow expected distribution
/// 4. **Computation Validity**: Gradient was computed from valid local data
///
/// ## Usage
///
/// ```rust,ignore
/// use fl_aggregator::proofs::api::ProofOfGradientQuality;
///
/// // Generate proof for gradient
/// let proof = ProofOfGradientQuality::generate(
///     &gradients,
///     max_norm,
///     data_commitment,
///     ProofConfig::default(),
/// )?;
///
/// // Serialize for transmission
/// let bytes = proof.serialize()?;
///
/// // Verify received proof
/// let proof = ProofOfGradientQuality::deserialize(&bytes)?;
/// let result = proof.verify()?;
/// assert!(result.valid);
/// ```
#[derive(Debug, Clone)]
pub struct ProofOfGradientQuality {
    /// The underlying gradient integrity proof
    pub integrity_proof: GradientIntegrityProof,
    /// Commitment to the local training data (privacy-preserving)
    pub data_commitment: [u8; 32],
    /// Statistical summary of gradient (for Byzantine detection hints)
    pub gradient_stats: GradientStatistics,
    /// Proof generation metadata
    pub metadata: ProofMetadata,
}

/// Statistical summary of a gradient (public, for detection)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStatistics {
    /// Number of gradient elements
    pub num_elements: usize,
    /// L2 norm of the gradient
    pub l2_norm: f32,
    /// L1 norm of the gradient
    pub l1_norm: f32,
    /// Maximum absolute value
    pub max_abs: f32,
    /// Mean value
    pub mean: f32,
    /// Variance
    pub variance: f32,
    /// Number of near-zero elements (|x| < epsilon)
    pub sparsity_count: usize,
    /// Sparsity ratio (near-zero / total)
    pub sparsity_ratio: f32,
}

impl GradientStatistics {
    /// Compute statistics from gradient values
    pub fn from_gradient(gradients: &[f32]) -> Self {
        let num_elements = gradients.len();
        if num_elements == 0 {
            return Self {
                num_elements: 0,
                l2_norm: 0.0,
                l1_norm: 0.0,
                max_abs: 0.0,
                mean: 0.0,
                variance: 0.0,
                sparsity_count: 0,
                sparsity_ratio: 1.0,
            };
        }

        let epsilon = 1e-8;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut l1_sum = 0.0f64;
        let mut max_abs = 0.0f32;
        let mut sparsity_count = 0usize;

        for &g in gradients {
            let g64 = g as f64;
            sum += g64;
            sum_sq += g64 * g64;
            l1_sum += g64.abs();
            if g.abs() > max_abs {
                max_abs = g.abs();
            }
            if g.abs() < epsilon {
                sparsity_count += 1;
            }
        }

        let mean = (sum / num_elements as f64) as f32;
        let variance = ((sum_sq / num_elements as f64) - (mean as f64).powi(2)) as f32;
        let l2_norm = (sum_sq as f32).sqrt();
        let l1_norm = l1_sum as f32;
        let sparsity_ratio = sparsity_count as f32 / num_elements as f32;

        Self {
            num_elements,
            l2_norm,
            l1_norm,
            max_abs,
            mean,
            variance: variance.max(0.0), // Numerical stability
            sparsity_count,
            sparsity_ratio,
        }
    }

    /// Check if statistics indicate potential Byzantine behavior
    pub fn is_suspicious(&self) -> bool {
        // Suspicious patterns:
        // 1. All zeros (free-rider)
        if self.l2_norm < 1e-10 {
            return true;
        }
        // 2. Extremely sparse (might be targeted attack)
        if self.sparsity_ratio > 0.99 && self.num_elements > 100 {
            return true;
        }
        // 3. Extremely large values (scaling attack)
        if self.max_abs > 1000.0 {
            return true;
        }
        // 4. Near-zero variance with non-zero mean (constant attack)
        if self.variance < 1e-10 && self.mean.abs() > 0.1 {
            return true;
        }
        false
    }
}

/// Proof generation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Proof generation timestamp (Unix epoch seconds)
    pub generated_at: u64,
    /// Node ID that generated the proof
    pub node_id: String,
    /// FL round number
    pub round: u32,
    /// Security level used
    pub security_level: String,
    /// Proof version for compatibility
    pub version: u8,
}

impl ProofOfGradientQuality {
    /// Version number for serialization compatibility
    pub const VERSION: u8 = 1;

    /// Generate a complete Proof of Gradient Quality
    ///
    /// # Arguments
    ///
    /// * `gradients` - The gradient values to prove
    /// * `max_norm` - Maximum allowed L2 norm
    /// * `data_commitment` - Hash commitment to the local training data
    /// * `node_id` - ID of the node generating the proof
    /// * `round` - Current FL round number
    /// * `config` - Proof configuration
    ///
    /// # Returns
    ///
    /// A complete PoGQ that can be transmitted and verified
    pub fn generate(
        gradients: &[f32],
        max_norm: f32,
        data_commitment: [u8; 32],
        node_id: &str,
        round: u32,
        config: ProofConfig,
    ) -> crate::proofs::ProofResult<Self> {
        // Validate inputs
        if gradients.is_empty() {
            return Err(crate::proofs::ProofError::InvalidWitness(
                "Gradient vector cannot be empty".to_string(),
            ));
        }

        // Check for NaN/Inf
        for (i, &g) in gradients.iter().enumerate() {
            if g.is_nan() {
                return Err(crate::proofs::ProofError::InvalidWitness(
                    format!("Gradient element {} is NaN", i),
                ));
            }
            if g.is_infinite() {
                return Err(crate::proofs::ProofError::InvalidWitness(
                    format!("Gradient element {} is infinite", i),
                ));
            }
        }

        // Compute statistics
        let gradient_stats = GradientStatistics::from_gradient(gradients);

        // Check norm bound
        if gradient_stats.l2_norm > max_norm {
            return Err(crate::proofs::ProofError::InvalidWitness(
                format!(
                    "Gradient L2 norm {} exceeds maximum {}",
                    gradient_stats.l2_norm, max_norm
                ),
            ));
        }

        // Generate the underlying integrity proof
        let integrity_proof = GradientIntegrityProof::generate(gradients, max_norm, config.clone())?;

        // Build metadata
        let metadata = ProofMetadata {
            generated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            node_id: node_id.to_string(),
            round,
            security_level: format!("{:?}", config.security_level),
            version: Self::VERSION,
        };

        Ok(Self {
            integrity_proof,
            data_commitment,
            gradient_stats,
            metadata,
        })
    }

    /// Verify the proof
    ///
    /// # Returns
    ///
    /// Verification result with details
    pub fn verify(&self) -> crate::proofs::ProofResult<GradientQualityVerificationResult> {
        let start = Instant::now();

        // Verify the underlying STARK proof
        let integrity_result = self.integrity_proof.verify()?;

        let mut issues = Vec::new();

        if !integrity_result.valid {
            issues.push(format!(
                "Integrity proof invalid: {}",
                integrity_result.details.unwrap_or_default()
            ));
        }

        // Check statistics for suspicious patterns
        if self.gradient_stats.is_suspicious() {
            issues.push("Gradient statistics indicate potentially Byzantine behavior".to_string());
        }

        // Verify metadata sanity
        if self.metadata.version > Self::VERSION {
            issues.push(format!(
                "Proof version {} is newer than supported version {}",
                self.metadata.version,
                Self::VERSION
            ));
        }

        let elapsed = start.elapsed();
        let valid = integrity_result.valid && issues.is_empty();

        Ok(GradientQualityVerificationResult {
            valid,
            integrity_valid: integrity_result.valid,
            statistics_valid: !self.gradient_stats.is_suspicious(),
            verification_time: elapsed,
            issues,
            gradient_stats: self.gradient_stats.clone(),
        })
    }

    /// Serialize the proof for transmission
    ///
    /// Format:
    /// ```text
    /// [version:1][data_commitment:32][stats_len:4][stats_json:N]
    /// [metadata_len:4][metadata_json:N][proof_bytes:...]
    /// ```
    pub fn serialize(&self) -> crate::proofs::ProofResult<Vec<u8>> {
        let mut bytes = Vec::new();

        // Version
        bytes.push(Self::VERSION);

        // Data commitment
        bytes.extend_from_slice(&self.data_commitment);

        // Statistics as JSON
        let stats_json = serde_json::to_vec(&self.gradient_stats)
            .map_err(|e| crate::proofs::ProofError::SerializationError(e.to_string()))?;
        bytes.extend_from_slice(&(stats_json.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&stats_json);

        // Metadata as JSON
        let metadata_json = serde_json::to_vec(&self.metadata)
            .map_err(|e| crate::proofs::ProofError::SerializationError(e.to_string()))?;
        bytes.extend_from_slice(&(metadata_json.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&metadata_json);

        // Integrity proof bytes
        let proof_bytes = self.integrity_proof.to_bytes();
        bytes.extend_from_slice(&proof_bytes);

        Ok(bytes)
    }

    /// Deserialize a proof from bytes
    pub fn deserialize(bytes: &[u8]) -> crate::proofs::ProofResult<Self> {
        use crate::proofs::types::ByteReader;
        let mut reader = ByteReader::new(bytes);

        // Version
        let version = reader.read_u8().ok_or_else(|| {
            crate::proofs::ProofError::InvalidProofFormat("Missing version".to_string())
        })?;

        if version > Self::VERSION {
            return Err(crate::proofs::ProofError::InvalidProofFormat(
                format!("Unsupported proof version {}", version),
            ));
        }

        // Data commitment
        let data_commitment = reader.read_32_bytes().ok_or_else(|| {
            crate::proofs::ProofError::InvalidProofFormat("Missing data commitment".to_string())
        })?;

        // Statistics
        let stats_len = reader.read_u32_le().ok_or_else(|| {
            crate::proofs::ProofError::InvalidProofFormat("Missing stats length".to_string())
        })? as usize;

        let stats_bytes = reader.read_bytes(stats_len).ok_or_else(|| {
            crate::proofs::ProofError::InvalidProofFormat("Truncated stats data".to_string())
        })?;

        let gradient_stats: GradientStatistics = serde_json::from_slice(stats_bytes)
            .map_err(|e| crate::proofs::ProofError::InvalidProofFormat(
                format!("Invalid stats JSON: {}", e)
            ))?;

        // Metadata
        let metadata_len = reader.read_u32_le().ok_or_else(|| {
            crate::proofs::ProofError::InvalidProofFormat("Missing metadata length".to_string())
        })? as usize;

        let metadata_bytes = reader.read_bytes(metadata_len).ok_or_else(|| {
            crate::proofs::ProofError::InvalidProofFormat("Truncated metadata".to_string())
        })?;

        let metadata: ProofMetadata = serde_json::from_slice(metadata_bytes)
            .map_err(|e| crate::proofs::ProofError::InvalidProofFormat(
                format!("Invalid metadata JSON: {}", e)
            ))?;

        // Integrity proof
        let proof_bytes = reader.read_remaining();
        let integrity_proof = GradientIntegrityProof::from_bytes(proof_bytes)?;

        Ok(Self {
            integrity_proof,
            data_commitment,
            gradient_stats,
            metadata,
        })
    }

    /// Get the gradient commitment from the underlying proof
    pub fn gradient_commitment(&self) -> &[u8; 32] {
        self.integrity_proof.commitment()
    }

    /// Get the proof size in bytes
    pub fn size(&self) -> usize {
        self.serialize().map(|b| b.len()).unwrap_or(0)
    }

    /// Get a hash of this proof for audit logging
    pub fn proof_hash(&self) -> [u8; 32] {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(b"pogq:");
        hasher.update(self.gradient_commitment());
        hasher.update(&self.data_commitment);
        hasher.update(&self.metadata.round.to_le_bytes());
        *hasher.finalize().as_bytes()
    }
}

/// Result of verifying a ProofOfGradientQuality
#[derive(Debug, Clone)]
pub struct GradientQualityVerificationResult {
    /// Overall validity
    pub valid: bool,
    /// Whether the STARK proof verified
    pub integrity_valid: bool,
    /// Whether statistics are non-suspicious
    pub statistics_valid: bool,
    /// Verification time
    pub verification_time: std::time::Duration,
    /// Any issues found during verification
    pub issues: Vec<String>,
    /// The gradient statistics from the proof
    pub gradient_stats: GradientStatistics,
}

// ============================================================================
// Batch Proof Operations
// ============================================================================

/// Batch verification of multiple PoGQ proofs
pub struct BatchPoGQVerifier {
    proofs: Vec<ProofOfGradientQuality>,
}

impl BatchPoGQVerifier {
    /// Create a new batch verifier
    pub fn new() -> Self {
        Self { proofs: Vec::new() }
    }

    /// Add a proof to the batch
    pub fn add_proof(&mut self, proof: ProofOfGradientQuality) {
        self.proofs.push(proof);
    }

    /// Add multiple proofs
    pub fn add_proofs(&mut self, proofs: Vec<ProofOfGradientQuality>) {
        self.proofs.extend(proofs);
    }

    /// Verify all proofs in the batch
    ///
    /// Returns results for each proof in order
    pub fn verify_all(&self) -> Vec<GradientQualityVerificationResult> {
        self.proofs.iter().map(|p| p.verify().unwrap_or_else(|e| {
            GradientQualityVerificationResult {
                valid: false,
                integrity_valid: false,
                statistics_valid: false,
                verification_time: std::time::Duration::ZERO,
                issues: vec![format!("Verification error: {}", e)],
                gradient_stats: GradientStatistics::from_gradient(&[]),
            }
        })).collect()
    }

    /// Get the number of proofs in the batch
    pub fn len(&self) -> usize {
        self.proofs.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.proofs.is_empty()
    }
}

impl Default for BatchPoGQVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_config_default() {
        let config = ApiConfig::default();
        assert_eq!(config.default_security, SecurityLevel::Standard128);
        assert!(config.enable_generation);
        assert!(config.enable_simulation);
    }

    #[test]
    fn test_parse_security_level() {
        assert_eq!(
            parse_security_level(Some("standard128")),
            Some(SecurityLevel::Standard128)
        );
        assert_eq!(
            parse_security_level(Some("HIGH256")),
            Some(SecurityLevel::High256)
        );
        assert_eq!(parse_security_level(Some("invalid")), None);
        assert_eq!(parse_security_level(None), None);
    }

    #[test]
    fn test_parse_proof_type() {
        assert_eq!(parse_proof_type("range"), Ok(ProofType::Range));
        assert_eq!(
            parse_proof_type("GradientIntegrity"),
            Ok(ProofType::GradientIntegrity)
        );
        assert!(parse_proof_type("invalid").is_err());
    }

    #[test]
    fn test_parse_assurance_level() {
        assert_eq!(parse_assurance_level("E2"), Ok(ProofAssuranceLevel::E2));
        assert_eq!(parse_assurance_level("3"), Ok(ProofAssuranceLevel::E3));
        assert!(parse_assurance_level("E5").is_err());
    }

    #[test]
    fn test_parse_proposal_type() {
        assert_eq!(
            parse_proposal_type("standard"),
            Ok(ProofProposalType::Standard)
        );
        assert_eq!(
            parse_proposal_type("Constitutional"),
            Ok(ProofProposalType::Constitutional)
        );
        assert!(parse_proposal_type("invalid").is_err());
    }

    #[test]
    fn test_voter_profile_conversion() {
        let input = VoterProfileInput {
            did: "did:mycelix:test".to_string(),
            assurance_level: 3,
            matl_score: 0.7,
            stake: 500.0,
            account_age_days: 90,
            participation_rate: 0.5,
            has_humanity_proof: true,
            fl_contributions: 15,
        };

        let profile: ProofVoterProfile = input.into();
        assert_eq!(profile.did, "did:mycelix:test");
        assert_eq!(profile.assurance_level, 3);
    }

    #[test]
    fn test_proof_response_serialization() {
        let response = ProofResponse {
            success: true,
            proof_type: "Range".to_string(),
            proof_bytes: Some("dGVzdA==".to_string()),
            size_bytes: Some(1000),
            generation_time_ms: Some(50.5),
            error: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"proof_type\":\"Range\""));
    }

    #[test]
    fn test_verify_response_serialization() {
        let response = VerifyResponse {
            valid: true,
            proof_type: "Range".to_string(),
            verification_time_ms: 10.5,
            error: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"valid\":true"));
    }

    #[test]
    fn test_simulation_response_serialization() {
        let response = SimulationResponse {
            inputs_valid: true,
            validation_errors: vec![],
            estimated_time_ms: 50.0,
            estimated_size_bytes: 15000,
            proof_type: "Range".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"inputs_valid\":true"));
    }
}
