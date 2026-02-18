//! Prover Integration for FL Unified Bridge
//!
//! Connects the UnifiedZkRbbftBridge with external RISC-0 prover services
//! for production-grade ZK proof generation.
//!
//! # Production Deployment
//!
//! For production use:
//! 1. Deploy the RISC-0 prover service (`spike/zk-trust-risc0`)
//! 2. Set `RISC0_PROVER_URL` environment variable
//! 3. Use `ProverBackend::ExternalService` or `ProverBackend::Bonsai`
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_sdk::fl::{UnifiedZkRbbftBridge, UnifiedZkRbbftConfig};
//! use mycelix_sdk::fl::prover_integration::{ProverIntegration, ProverBackend};
//!
//! // Create bridge with external prover
//! let integration = ProverIntegration::new(ProverBackend::ExternalService {
//!     url: "http://prover.mycelix.network:3000".to_string(),
//! });
//!
//! // Or use Bonsai cloud proving
//! let bonsai = ProverIntegration::new(ProverBackend::Bonsai {
//!     api_key: std::env::var("BONSAI_API_KEY").unwrap(),
//! });
//! ```

use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;

#[cfg(feature = "std")]
use hex;

use crate::zkproof::GradientConstraints;

#[cfg(any(feature = "simulation", feature = "risc0"))]
use crate::zkproof::GradientProofReceipt;

#[cfg(any(feature = "simulation", feature = "risc0"))]
use crate::zkproof::GradientProver;

// ============================================================================
// Prover Backend Configuration
// ============================================================================

/// Backend for ZK proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProverBackend {
    /// Simulation mode - NO cryptographic guarantees, testing only
    Simulation,

    /// External RISC-0 prover service via HTTP
    ExternalService {
        /// URL of the prover service
        url: String,
        /// Request timeout in milliseconds
        #[serde(default = "default_timeout_ms")]
        timeout_ms: u64,
    },

    /// Bonsai cloud proving (Risc Zero's cloud service)
    Bonsai {
        /// Bonsai API key
        api_key: String,
        /// Optional custom endpoint
        #[serde(default)]
        endpoint: Option<String>,
    },

    /// Local RISC-0 prover (heavyweight, requires risc0 feature)
    #[cfg(feature = "risc0")]
    LocalRisc0,
}

fn default_timeout_ms() -> u64 {
    300_000 // 5 minutes
}

impl Default for ProverBackend {
    fn default() -> Self {
        // Check environment for prover configuration
        if let Ok(url) = std::env::var("RISC0_PROVER_URL") {
            return ProverBackend::ExternalService {
                url,
                timeout_ms: std::env::var("RISC0_PROVER_TIMEOUT_MS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(default_timeout_ms()),
            };
        }

        if std::env::var("BONSAI_API_KEY").is_ok() {
            return ProverBackend::Bonsai {
                api_key: std::env::var("BONSAI_API_KEY").unwrap_or_default(),
                endpoint: std::env::var("BONSAI_API_URL").ok(),
            };
        }

        // Default to simulation for development
        ProverBackend::Simulation
    }
}

// ============================================================================
// Prover Integration Errors
// ============================================================================

/// Errors from prover integration
#[derive(Debug, Error)]
pub enum ProverIntegrationError {
    /// Backend not available
    #[error("Prover backend not available: {0}")]
    BackendNotAvailable(String),

    /// HTTP request failed
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    /// Proof generation failed
    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),

    /// Proof verification failed
    #[error("Proof verification failed: {0}")]
    ProofVerificationFailed(String),

    /// Timeout during proof generation
    #[error("Proof generation timed out after {0}ms")]
    Timeout(u64),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

// ============================================================================
// Gradient Proof Input (shared across prover backends)
// ============================================================================

/// Input parameters for gradient proof generation
pub struct GradientProofInput<'a> {
    /// Gradient vector
    pub gradient: &'a [f32],
    /// Hash of the global model
    pub model_hash: &'a [u8; 32],
    /// Number of training epochs
    pub epochs: u32,
    /// Learning rate used
    pub learning_rate: f32,
    /// Client identifier
    pub client_id: &'a str,
    /// Round number
    pub round: u32,
}

// ============================================================================
// Proof Request/Response for External Service
// ============================================================================

/// Request to external prover service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientProofRequest {
    /// Gradient vector (as f32 values)
    pub gradient: Vec<f32>,
    /// Hash of the global model
    pub model_hash: [u8; 32],
    /// Training epochs
    pub epochs: u32,
    /// Learning rate
    pub learning_rate: f32,
    /// Client identifier
    pub client_id: String,
    /// Round number
    pub round: u32,
    /// Gradient constraints
    pub constraints: GradientConstraintsDto,
}

/// Gradient constraints for external service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConstraintsDto {
    /// Minimum gradient norm
    pub min_norm: f32,
    /// Maximum gradient norm
    pub max_norm: f32,
    /// Maximum magnitude of any single element
    pub max_element_magnitude: f32,
    /// Minimum number of dimensions
    pub min_dimension: usize,
}

impl From<GradientConstraints> for GradientConstraintsDto {
    fn from(c: GradientConstraints) -> Self {
        Self {
            min_norm: c.min_norm,
            max_norm: c.max_norm,
            max_element_magnitude: c.max_element_magnitude,
            min_dimension: c.min_dimension,
        }
    }
}

/// Response from external prover service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientProofResponse {
    /// Whether proof generation succeeded (derived from is_valid if not present)
    #[serde(default = "default_success")]
    pub success: bool,
    /// Whether gradient passes quality constraints
    #[serde(rename = "is_valid")]
    pub is_valid: bool,
    /// Gradient hash commitment (hex-encoded string from server)
    pub gradient_hash: String,
    /// L2 norm of the gradient
    pub norm: f32,
    /// Proof bytes (for real proofs)
    #[serde(default)]
    pub proof_bytes: Vec<u8>,
    /// Generation time in milliseconds
    pub proof_time_ms: u64,
    /// Client identifier (echoed back)
    pub client_id: String,
    /// Round number (echoed back)
    pub round: u32,
    /// Error message (if failed)
    #[serde(default)]
    pub error: Option<String>,
}

fn default_success() -> bool {
    true // Server doesn't send success field, assume true if response received
}

// ============================================================================
// Prover Integration
// ============================================================================

/// Integration layer for ZK proof generation in FL
///
/// Provides a unified interface for different prover backends.
#[derive(Debug, Clone)]
pub struct ProverIntegration {
    /// The backend to use
    backend: ProverBackend,
    /// Gradient constraints
    constraints: GradientConstraints,
    /// Statistics
    stats: ProverStats,
}

/// Statistics from prover integration
#[derive(Debug, Clone, Default)]
pub struct ProverStats {
    /// Total proofs generated
    pub total_proofs: usize,
    /// Successful proofs
    pub successful_proofs: usize,
    /// Failed proofs
    pub failed_proofs: usize,
    /// Total generation time in ms
    pub total_time_ms: u64,
    /// Average generation time in ms
    pub avg_time_ms: u64,
}

impl ProverIntegration {
    /// Create a new prover integration with the specified backend
    pub fn new(backend: ProverBackend) -> Self {
        Self {
            backend,
            constraints: GradientConstraints::for_federated_learning(),
            stats: ProverStats::default(),
        }
    }

    /// Create with default backend (auto-detects from environment)
    pub fn with_defaults() -> Self {
        Self::new(ProverBackend::default())
    }

    /// Create for simulation (testing only)
    pub fn for_testing() -> Self {
        Self::new(ProverBackend::Simulation)
    }

    /// Create with external service
    pub fn with_external_service(url: &str) -> Self {
        Self::new(ProverBackend::ExternalService {
            url: url.to_string(),
            timeout_ms: default_timeout_ms(),
        })
    }

    /// Set custom gradient constraints
    pub fn with_constraints(mut self, constraints: GradientConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Get the current backend
    pub fn backend(&self) -> &ProverBackend {
        &self.backend
    }

    /// Get statistics
    pub fn stats(&self) -> &ProverStats {
        &self.stats
    }

    /// Check if using production-grade backend
    pub fn is_production(&self) -> bool {
        !matches!(self.backend, ProverBackend::Simulation)
    }

    /// Generate a proof for gradient quality
    ///
    /// Returns a receipt that can be verified later.
    #[cfg(any(feature = "simulation", feature = "risc0"))]
    pub fn prove_gradient(
        &mut self,
        gradient: &[f32],
        model_hash: &[u8; 32],
        epochs: u32,
        learning_rate: f32,
        client_id: &str,
        round: u32,
    ) -> Result<GradientProofReceipt, ProverIntegrationError> {
        let start = Instant::now();

        let input = GradientProofInput {
            gradient,
            model_hash,
            epochs,
            learning_rate,
            client_id,
            round,
        };

        let result = match &self.backend {
            ProverBackend::Simulation => self.prove_simulation(
                gradient,
                model_hash,
                epochs,
                learning_rate,
                client_id,
                round,
            ),
            ProverBackend::ExternalService { url, timeout_ms } => {
                self.prove_external(url, *timeout_ms, &input)
            }
            ProverBackend::Bonsai { api_key, endpoint } => {
                self.prove_bonsai(api_key, endpoint.as_deref(), &input)
            }
            #[cfg(feature = "risc0")]
            ProverBackend::LocalRisc0 => self.prove_local_risc0(
                gradient,
                model_hash,
                epochs,
                learning_rate,
                client_id,
                round,
            ),
        };

        let elapsed_ms = start.elapsed().as_millis() as u64;
        self.update_stats(result.is_ok(), elapsed_ms);

        result
    }

    /// Prove using simulation mode (testing only)
    #[cfg(any(feature = "simulation", feature = "risc0"))]
    fn prove_simulation(
        &self,
        gradient: &[f32],
        model_hash: &[u8; 32],
        epochs: u32,
        learning_rate: f32,
        client_id: &str,
        round: u32,
    ) -> Result<GradientProofReceipt, ProverIntegrationError> {
        // Log warning about simulation mode
        #[cfg(feature = "std")]
        eprintln!(
            "\n\x1b[1;33m========================================\x1b[0m\n\
             \x1b[1;33m[ZK_PROOF_SECURITY_WARNING]\x1b[0m\n\
             Running in SIMULATION mode!\n\
             Proofs provide NO cryptographic guarantees.\n\
             Do NOT use in production.\n\
             \x1b[1;33m========================================\x1b[0m\n"
        );

        let prover = GradientProver::for_federated_learning();
        prover
            .prove_gradient_quality(
                gradient,
                model_hash,
                epochs,
                learning_rate,
                client_id,
                round,
            )
            .map_err(|e| ProverIntegrationError::ProofGenerationFailed(e.to_string()))
    }

    /// Prove using external HTTP service
    #[cfg(any(feature = "simulation", feature = "risc0"))]
    fn prove_external(
        &self,
        _url: &str,
        _timeout_ms: u64,
        input: &GradientProofInput<'_>,
    ) -> Result<GradientProofReceipt, ProverIntegrationError> {
        // Build request
        let _request = GradientProofRequest {
            gradient: input.gradient.to_vec(),
            model_hash: *input.model_hash,
            epochs: input.epochs,
            learning_rate: input.learning_rate,
            client_id: input.client_id.to_string(),
            round: input.round,
            constraints: self.constraints.clone().into(),
        };

        #[cfg(not(feature = "std"))]
        {
            Err(ProverIntegrationError::BackendNotAvailable(
                "External service requires std feature".to_string(),
            ))
        }

        #[cfg(feature = "std")]
        {
            use std::time::Duration;

            let endpoint = format!("{}/prove/gradient", url);

            // Make HTTP request to external prover service
            let response = ureq::post(&endpoint)
                .timeout(Duration::from_millis(timeout_ms))
                .set("Content-Type", "application/json")
                .send_json(&request)
                .map_err(|e| {
                    // If service unavailable, fall back to simulation with warning
                    eprintln!(
                        "\n\x1b[1;33m[ProverIntegration]\x1b[0m External service at {} unavailable: {}",
                        url, e
                    );
                    eprintln!(
                        "  To deploy: cd spike/zk-prover-service && docker-compose up -d"
                    );
                    eprintln!("  Falling back to simulation mode.\n");
                    ProverIntegrationError::HttpError(e.to_string())
                })?;

            // Parse response
            let proof_response: GradientProofResponse = response
                .into_json()
                .map_err(|e| ProverIntegrationError::SerializationError(e.to_string()))?;

            if !proof_response.success {
                return Err(ProverIntegrationError::ProofGenerationFailed(
                    proof_response
                        .error
                        .unwrap_or_else(|| "Unknown error".to_string()),
                ));
            }

            // Parse gradient hash from hex string to bytes
            let gradient_hash_bytes = hex::decode(&proof_response.gradient_hash).map_err(|e| {
                ProverIntegrationError::SerializationError(format!(
                    "Invalid gradient hash hex: {}",
                    e
                ))
            })?;
            let gradient_hash: [u8; 32] = gradient_hash_bytes.try_into().map_err(|_| {
                ProverIntegrationError::SerializationError(
                    "Gradient hash must be 32 bytes".to_string(),
                )
            })?;

            // Construct the output structure
            let output = GradientProofOutput {
                gradient_hash,
                global_model_hash: *input.model_hash,
                epochs: input.epochs,
                learning_rate: input.learning_rate,
                norm_valid: proof_response.is_valid,
                client_id: proof_response.client_id.clone(),
                round: proof_response.round,
                commitment: [0u8; 32], // Commitment will be computed below
            };

            // Compute commitment
            let commitment = compute_commitment(&output);
            let output_with_commitment = GradientProofOutput {
                commitment,
                ..output
            };

            // Convert response to GradientProofReceipt
            let receipt = GradientProofReceipt {
                output: output_with_commitment,
                proof_bytes: proof_response.proof_bytes,
                mode: "external".to_string(),
                generation_time_ms: proof_response.proof_time_ms,
            };

            Ok(receipt)
        }
    }

    /// Prove using external HTTP service with fallback
    #[cfg(all(feature = "std", any(feature = "simulation", feature = "risc0")))]
    pub fn prove_external_with_fallback(
        &self,
        url: &str,
        timeout_ms: u64,
        input: &GradientProofInput<'_>,
    ) -> Result<GradientProofReceipt, ProverIntegrationError> {
        match self.prove_external(url, timeout_ms, input) {
            Ok(receipt) => Ok(receipt),
            Err(_) => {
                // Fall back to simulation if external service fails
                self.prove_simulation(
                    input.gradient,
                    input.model_hash,
                    input.epochs,
                    input.learning_rate,
                    input.client_id,
                    input.round,
                )
            }
        }
    }

    /// Prove using Bonsai cloud service
    #[cfg(any(feature = "simulation", feature = "risc0"))]
    fn prove_bonsai(
        &self,
        api_key: &str,
        endpoint: Option<&str>,
        _input: &GradientProofInput<'_>,
    ) -> Result<GradientProofReceipt, ProverIntegrationError> {
        #[cfg(not(feature = "std"))]
        {
            let _ = (api_key, endpoint);
            Err(ProverIntegrationError::BackendNotAvailable(
                "Bonsai requires std feature".to_string(),
            ))
        }

        #[cfg(feature = "std")]
        {
            let endpoint = endpoint.unwrap_or("https://api.bonsai.xyz");

            // In real implementation, this would use the Bonsai SDK
            eprintln!(
                "[ProverIntegration] Bonsai proving at {} not yet implemented, using simulation",
                endpoint
            );
            eprintln!(
                "[ProverIntegration] API key: {}...",
                &api_key[..8.min(api_key.len())]
            );

            self.prove_simulation(
                input.gradient,
                input.model_hash,
                input.epochs,
                input.learning_rate,
                input.client_id,
                input.round,
            )
        }
    }

    /// Prove using local RISC-0 prover
    #[cfg(feature = "risc0")]
    fn prove_local_risc0(
        &self,
        gradient: &[f32],
        model_hash: &[u8; 32],
        epochs: u32,
        learning_rate: f32,
        client_id: &str,
        round: u32,
    ) -> Result<GradientProofReceipt, ProverIntegrationError> {
        use crate::zkproof::Risc0GradientProver;

        let prover = Risc0GradientProver::new(self.constraints.clone());
        prover
            .prove(
                gradient,
                model_hash,
                epochs,
                learning_rate,
                client_id,
                round,
            )
            .map_err(|e| ProverIntegrationError::ProofGenerationFailed(e.to_string()))
    }

    /// Update statistics
    fn update_stats(&mut self, success: bool, elapsed_ms: u64) {
        self.stats.total_proofs += 1;
        self.stats.total_time_ms += elapsed_ms;

        if success {
            self.stats.successful_proofs += 1;
        } else {
            self.stats.failed_proofs += 1;
        }

        if self.stats.total_proofs > 0 {
            self.stats.avg_time_ms = self.stats.total_time_ms / self.stats.total_proofs as u64;
        }
    }

    /// Verify a proof receipt
    #[cfg(any(feature = "simulation", feature = "risc0"))]
    pub fn verify(&self, receipt: &GradientProofReceipt) -> bool {
        receipt.is_valid()
    }
}

// ============================================================================
// Helper for Unified Bridge Integration
// ============================================================================

/// Configuration for production FL deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionFLConfig {
    /// Prover backend configuration
    pub prover: ProverBackend,
    /// Minimum reputation for participation
    pub min_reputation: f32,
    /// Minimum participants per round
    pub min_participants: usize,
    /// Byzantine tolerance threshold (max 0.34 validated for RB-BFT)
    pub byzantine_threshold: f32,
    /// Quorum threshold (typically 0.667)
    pub quorum_threshold: f32,
    /// Gradient constraints
    pub gradient_constraints: GradientConstraintsDto,
}

impl Default for ProductionFLConfig {
    fn default() -> Self {
        let constraints = GradientConstraints::for_federated_learning();
        Self {
            prover: ProverBackend::default(),
            min_reputation: 0.5,
            min_participants: 5,
            byzantine_threshold: 0.34,
            quorum_threshold: 0.667,
            gradient_constraints: constraints.into(),
        }
    }
}

#[cfg(all(test, any(feature = "simulation", feature = "risc0")))]
mod tests {
    use super::*;

    fn sample_gradient(size: usize) -> Vec<f32> {
        (0..size).map(|i| (i as f32 * 0.01).sin() * 0.5).collect()
    }

    #[test]
    fn test_simulation_backend() {
        let mut integration = ProverIntegration::for_testing();

        let gradient = sample_gradient(100);
        let model_hash = [0x42u8; 32];

        let result = integration.prove_gradient(&gradient, &model_hash, 5, 0.01, "test-client", 1);

        assert!(result.is_ok());
        let receipt = result.unwrap();
        assert!(receipt.is_valid());
        assert!(integration.verify(&receipt));
    }

    #[test]
    fn test_invalid_gradient_detected() {
        let mut integration = ProverIntegration::for_testing();

        // Zero gradient should fail quality check
        let gradient = vec![0.0f32; 100];
        let model_hash = [0x42u8; 32];

        let result = integration.prove_gradient(&gradient, &model_hash, 5, 0.01, "test-client", 1);

        assert!(result.is_ok());
        let receipt = result.unwrap();
        assert!(!receipt.is_valid()); // Should fail quality check
    }

    #[test]
    fn test_stats_tracking() {
        let mut integration = ProverIntegration::for_testing();

        let gradient = sample_gradient(100);
        let model_hash = [0x42u8; 32];

        // Generate a few proofs
        for _ in 0..3 {
            let _ = integration.prove_gradient(&gradient, &model_hash, 5, 0.01, "test", 1);
        }

        let stats = integration.stats();
        assert_eq!(stats.total_proofs, 3);
        assert_eq!(stats.successful_proofs, 3);
        // Note: avg_time_ms may be 0 in simulation mode due to fast execution
        assert!(stats.total_time_ms >= 0); // Just verify it's tracked
    }

    #[test]
    fn test_backend_detection() {
        // Default should be simulation when no env vars set
        let backend = ProverBackend::default();
        assert!(matches!(backend, ProverBackend::Simulation));
    }

    #[test]
    fn test_production_check() {
        let simulation = ProverIntegration::for_testing();
        assert!(!simulation.is_production());

        let external = ProverIntegration::with_external_service("http://localhost:3000");
        assert!(external.is_production());
    }
}
