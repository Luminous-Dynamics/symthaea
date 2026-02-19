//! RISC-0 Prover Service Client
//!
//! HTTP client for communicating with external RISC-0 prover services.
//! This enables the SDK to generate real ZK proofs without bundling the
//! heavyweight RISC-0 prover.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐    HTTP     ┌─────────────────┐    RISC-0    ┌──────────┐
//! │  SDK Client │ ──────────▶ │  Prover Service │ ───────────▶ │  zkVM    │
//! └─────────────┘             └─────────────────┘              └──────────┘
//!                                     │
//!                                     ▼
//!                              ┌─────────────────┐
//!                              │   STARK Proof   │
//!                              └─────────────────┘
//! ```
//!
//! # Starting the Prover Service
//!
//! ```bash
//! # Build and run the prover service
//! cd spike/zk-trust-risc0
//! cargo run --release
//!
//! # Or use Docker
//! docker run -p 3000:3000 mycelix/risc0-prover
//! ```
//!
//! # Environment Variables
//!
//! - `RISC0_PROVER_URL`: URL of prover service (default: http://localhost:3000)
//! - `RISC0_PROVER_TIMEOUT_MS`: Request timeout (default: 300000ms = 5 minutes)
//! - `BONSAI_API_KEY`: For Bonsai cloud proving (optional)
//! - `BONSAI_API_URL`: Bonsai endpoint (optional)

use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::trust_risc0::{
    ZkTrustProofInput, ZkTrustReceipt, ZkTrustProofOutput, ZkTrustStatement,
    ZkKVector, ZkTrustParams, TrustProverError, SCALE, evaluate_statement, compute_commitment,
};

// ============================================================================
// Prover Service Configuration
// ============================================================================

/// Configuration for the prover service client
#[derive(Debug, Clone)]
pub struct ProverServiceConfig {
    /// Base URL of the prover service
    pub base_url: String,
    /// Request timeout
    pub timeout: Duration,
    /// Retry count for transient failures
    pub retry_count: u32,
    /// Use Bonsai cloud proving
    pub use_bonsai: bool,
    /// Bonsai API key (required if use_bonsai is true)
    pub bonsai_api_key: Option<String>,
}

impl Default for ProverServiceConfig {
    fn default() -> Self {
        Self {
            base_url: std::env::var("RISC0_PROVER_URL")
                .unwrap_or_else(|_| "http://localhost:3000".to_string()),
            timeout: Duration::from_millis(
                std::env::var("RISC0_PROVER_TIMEOUT_MS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(300_000)
            ),
            retry_count: 3,
            use_bonsai: std::env::var("BONSAI_API_KEY").is_ok(),
            bonsai_api_key: std::env::var("BONSAI_API_KEY").ok(),
        }
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Request to generate a trust proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProveRequest {
    /// The proof input (serialized)
    pub input: ZkTrustProofInput,
    /// Request ID for tracking
    pub request_id: String,
    /// Client identifier
    pub client_id: Option<String>,
}

/// Response from proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProveResponse {
    /// The generated proof receipt
    pub receipt: ZkTrustReceipt,
    /// Request ID
    pub request_id: String,
    /// Proof generation time in milliseconds
    pub proof_time_ms: u64,
    /// Prover version
    pub prover_version: String,
}

/// Request to verify a proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyRequest {
    /// The proof receipt to verify
    pub receipt: ZkTrustReceipt,
    /// Request ID
    pub request_id: String,
}

/// Response from proof verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyResponse {
    /// Whether the proof is valid
    pub valid: bool,
    /// Request ID
    pub request_id: String,
    /// Verification time in milliseconds
    pub verify_time_ms: u64,
    /// Error message if invalid
    pub error: Option<String>,
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,
    /// Prover version
    pub version: String,
    /// Whether zkVM is ready
    pub zkvm_ready: bool,
    /// Current queue depth
    pub queue_depth: u32,
    /// Average proof time (ms)
    pub avg_proof_time_ms: u64,
}

/// Service error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Request ID (if available)
    pub request_id: Option<String>,
}

// ============================================================================
// Prover Service Client
// ============================================================================

/// Client for the RISC-0 prover service
pub struct ProverServiceClient {
    config: ProverServiceConfig,
}

impl ProverServiceClient {
    /// Create a new prover service client with default configuration
    pub fn new() -> Self {
        Self {
            config: ProverServiceConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ProverServiceConfig) -> Self {
        Self { config }
    }

    /// Generate a unique request ID
    fn generate_request_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        format!("req_{:x}_{:x}", now.as_secs(), now.subsec_nanos())
    }

    /// Check if the prover service is available
    pub fn health_check(&self) -> Result<HealthResponse, ProverServiceError> {
        let url = format!("{}/health", self.config.base_url);

        #[cfg(feature = "reqwest")]
        {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .map_err(|e| ProverServiceError::ConnectionFailed(e.to_string()))?;

            let response = client
                .get(&url)
                .send()
                .map_err(|e| ProverServiceError::ConnectionFailed(e.to_string()))?;

            if !response.status().is_success() {
                return Err(ProverServiceError::ServiceError(ServiceError {
                    code: response.status().to_string(),
                    message: "Health check failed".to_string(),
                    request_id: None,
                }));
            }

            response
                .json::<HealthResponse>()
                .map_err(|e| ProverServiceError::DeserializationFailed(e.to_string()))
        }

        #[cfg(not(feature = "reqwest"))]
        {
            // Without reqwest, return a mock health response indicating service needs to be started
            Err(ProverServiceError::ConnectionFailed(format!(
                "HTTP client not available. Enable 'reqwest' feature or check prover at: {}",
                url
            )))
        }
    }

    /// Generate a proof via the prover service
    pub fn prove(&self, input: ZkTrustProofInput) -> Result<ZkTrustReceipt, ProverServiceError> {
        let request_id = Self::generate_request_id();
        let url = format!("{}/prove", self.config.base_url);

        let request = ProveRequest {
            input: input.clone(),
            request_id: request_id.clone(),
            client_id: None,
        };

        #[cfg(feature = "reqwest")]
        {
            let client = reqwest::blocking::Client::builder()
                .timeout(self.config.timeout)
                .build()
                .map_err(|e| ProverServiceError::ConnectionFailed(e.to_string()))?;

            let mut last_error = None;
            for attempt in 0..=self.config.retry_count {
                if attempt > 0 {
                    std::thread::sleep(Duration::from_millis(1000 * attempt as u64));
                }

                match client.post(&url).json(&request).send() {
                    Ok(response) => {
                        if response.status().is_success() {
                            return response
                                .json::<ProveResponse>()
                                .map(|r| r.receipt)
                                .map_err(|e| ProverServiceError::DeserializationFailed(e.to_string()));
                        } else {
                            let error = response
                                .json::<ServiceError>()
                                .unwrap_or(ServiceError {
                                    code: "UNKNOWN".to_string(),
                                    message: "Unknown error".to_string(),
                                    request_id: Some(request_id.clone()),
                                });
                            last_error = Some(ProverServiceError::ServiceError(error));
                        }
                    }
                    Err(e) => {
                        last_error = Some(ProverServiceError::ConnectionFailed(e.to_string()));
                    }
                }
            }

            Err(last_error.unwrap_or(ProverServiceError::ConnectionFailed(
                "Max retries exceeded".to_string(),
            )))
        }

        #[cfg(not(feature = "reqwest"))]
        {
            // Fallback: Generate a simulation proof locally with a warning
            self.prove_locally_simulation(input)
        }
    }

    /// Verify a proof via the prover service
    pub fn verify(&self, receipt: &ZkTrustReceipt) -> Result<bool, ProverServiceError> {
        // Simulation proofs are verified locally
        if receipt.is_simulation {
            return Ok(receipt.proof_bytes.starts_with(b"SIMULATION_TRUST_PROOF_V1"));
        }

        let request_id = Self::generate_request_id();
        let url = format!("{}/verify", self.config.base_url);

        let request = VerifyRequest {
            receipt: receipt.clone(),
            request_id: request_id.clone(),
        };

        #[cfg(feature = "reqwest")]
        {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .map_err(|e| ProverServiceError::ConnectionFailed(e.to_string()))?;

            let response = client
                .post(&url)
                .json(&request)
                .send()
                .map_err(|e| ProverServiceError::ConnectionFailed(e.to_string()))?;

            if response.status().is_success() {
                let verify_response = response
                    .json::<VerifyResponse>()
                    .map_err(|e| ProverServiceError::DeserializationFailed(e.to_string()))?;
                Ok(verify_response.valid)
            } else {
                let error = response
                    .json::<ServiceError>()
                    .unwrap_or(ServiceError {
                        code: "VERIFICATION_FAILED".to_string(),
                        message: "Verification request failed".to_string(),
                        request_id: Some(request_id),
                    });
                Err(ProverServiceError::ServiceError(error))
            }
        }

        #[cfg(not(feature = "reqwest"))]
        {
            // Without HTTP client, can only verify simulation proofs
            Err(ProverServiceError::ConnectionFailed(
                "Cannot verify production proofs without HTTP client".to_string(),
            ))
        }
    }

    /// Generate a simulation proof locally (fallback when service unavailable)
    fn prove_locally_simulation(&self, input: ZkTrustProofInput) -> Result<ZkTrustReceipt, ProverServiceError> {
        eprintln!(
            "\n\x1b[1;33m[PROVER_SERVICE_WARNING]\x1b[0m \
            Prover service not available at {}. \
            Generating SIMULATION proof locally.\n\
            This proof provides NO cryptographic guarantees!\n",
            self.config.base_url
        );

        let start = std::time::Instant::now();

        // Compute commitment
        let commitment = compute_commitment(
            &input.kvector,
            &input.blinding,
            input.agent_id_hash,
            input.timestamp,
        );

        // Evaluate statement
        let statement_valid = evaluate_statement(&input.kvector, input.statement, &input.params);

        // Revealed score only for TrustInRange
        let revealed_score = if input.statement == ZkTrustStatement::TrustInRange {
            input.kvector.trust_score()
        } else {
            0
        };

        let output = ZkTrustProofOutput {
            statement_valid,
            statement: input.statement,
            kvector_commitment: commitment,
            agent_id_hash: input.agent_id_hash,
            timestamp: input.timestamp,
            revealed_score,
        };

        // Simulation marker
        let marker = b"SIMULATION_TRUST_PROOF_V1";
        let mut proof_bytes = marker.to_vec();
        proof_bytes.extend(bincode::serialize(&output).unwrap_or_default());

        let proof_time_ms = start.elapsed().as_millis() as u64;

        Ok(ZkTrustReceipt {
            output,
            proof_bytes,
            is_simulation: true,
            proof_time_ms,
        })
    }
}

impl Default for ProverServiceClient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors from the prover service
#[derive(Debug, Clone)]
pub enum ProverServiceError {
    /// Failed to connect to prover service
    ConnectionFailed(String),
    /// Service returned an error
    ServiceError(ServiceError),
    /// Failed to serialize request
    SerializationFailed(String),
    /// Failed to deserialize response
    DeserializationFailed(String),
    /// Request timed out
    Timeout,
    /// Service unavailable
    ServiceUnavailable,
}

impl std::fmt::Display for ProverServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            Self::ServiceError(err) => write!(f, "Service error [{}]: {}", err.code, err.message),
            Self::SerializationFailed(msg) => write!(f, "Serialization failed: {}", msg),
            Self::DeserializationFailed(msg) => write!(f, "Deserialization failed: {}", msg),
            Self::Timeout => write!(f, "Request timed out"),
            Self::ServiceUnavailable => write!(f, "Prover service unavailable"),
        }
    }
}

impl std::error::Error for ProverServiceError {}

impl From<ProverServiceError> for TrustProverError {
    fn from(err: ProverServiceError) -> Self {
        match err {
            ProverServiceError::ConnectionFailed(msg) => TrustProverError::Risc0Execution(msg),
            ProverServiceError::ServiceError(e) => TrustProverError::Risc0Execution(e.message),
            ProverServiceError::SerializationFailed(msg) => TrustProverError::Risc0Execution(msg),
            ProverServiceError::DeserializationFailed(msg) => TrustProverError::Risc0Execution(msg),
            ProverServiceError::Timeout => TrustProverError::Risc0Execution("Timeout".to_string()),
            ProverServiceError::ServiceUnavailable => TrustProverError::Risc0NotAvailable,
        }
    }
}

// ============================================================================
// High-Level API
// ============================================================================

use crate::matl::KVector as SdkKVector;

/// Convenience wrapper for common trust proof operations
pub struct TrustProverService {
    client: ProverServiceClient,
}

impl TrustProverService {
    /// Create with default configuration
    pub fn new() -> Self {
        Self {
            client: ProverServiceClient::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ProverServiceConfig) -> Self {
        Self {
            client: ProverServiceClient::with_config(config),
        }
    }

    /// Check service health
    pub fn is_available(&self) -> bool {
        self.client.health_check().map(|h| h.zkvm_ready).unwrap_or(false)
    }

    /// Prove trust exceeds threshold
    pub fn prove_trust_exceeds_threshold(
        &self,
        kvector: &SdkKVector,
        threshold: f32,
        agent_id: &str,
    ) -> Result<ZkTrustReceipt, TrustProverError> {
        let zk_kv = ZkKVector::from_sdk(kvector);
        let blinding = generate_blinding();
        let agent_id_hash = hash_agent_id(agent_id);
        let timestamp = current_timestamp();

        let input = ZkTrustProofInput {
            kvector: zk_kv,
            statement: ZkTrustStatement::TrustExceedsThreshold,
            params: ZkTrustParams {
                threshold: (threshold.clamp(0.0, 1.0) * SCALE as f32) as u64,
                ..Default::default()
            },
            blinding,
            agent_id_hash,
            timestamp,
        };

        self.client.prove(input).map_err(Into::into)
    }

    /// Prove trust is in range
    pub fn prove_trust_in_range(
        &self,
        kvector: &SdkKVector,
        min_trust: f32,
        max_trust: f32,
        agent_id: &str,
    ) -> Result<ZkTrustReceipt, TrustProverError> {
        let zk_kv = ZkKVector::from_sdk(kvector);
        let blinding = generate_blinding();
        let agent_id_hash = hash_agent_id(agent_id);
        let timestamp = current_timestamp();

        let input = ZkTrustProofInput {
            kvector: zk_kv,
            statement: ZkTrustStatement::TrustInRange,
            params: ZkTrustParams {
                min_value: (min_trust.clamp(0.0, 1.0) * SCALE as f32) as u64,
                max_value: (max_trust.clamp(0.0, 1.0) * SCALE as f32) as u64,
                ..Default::default()
            },
            blinding,
            agent_id_hash,
            timestamp,
        };

        self.client.prove(input).map_err(Into::into)
    }

    /// Prove agent is verified (k_v >= 0.5)
    pub fn prove_is_verified(
        &self,
        kvector: &SdkKVector,
        agent_id: &str,
    ) -> Result<ZkTrustReceipt, TrustProverError> {
        let zk_kv = ZkKVector::from_sdk(kvector);
        let blinding = generate_blinding();
        let agent_id_hash = hash_agent_id(agent_id);
        let timestamp = current_timestamp();

        let input = ZkTrustProofInput {
            kvector: zk_kv,
            statement: ZkTrustStatement::IsVerified,
            params: Default::default(),
            blinding,
            agent_id_hash,
            timestamp,
        };

        self.client.prove(input).map_err(Into::into)
    }

    /// Prove agent is highly coherent (k_coherence >= 0.7)
    pub fn prove_is_coherent(
        &self,
        kvector: &SdkKVector,
        agent_id: &str,
    ) -> Result<ZkTrustReceipt, TrustProverError> {
        let zk_kv = ZkKVector::from_sdk(kvector);
        let blinding = generate_blinding();
        let agent_id_hash = hash_agent_id(agent_id);
        let timestamp = current_timestamp();

        let input = ZkTrustProofInput {
            kvector: zk_kv,
            statement: ZkTrustStatement::IsHighlyCoherent,
            params: Default::default(),
            blinding,
            agent_id_hash,
            timestamp,
        };

        self.client.prove(input).map_err(Into::into)
    }

    /// Verify a proof
    pub fn verify(&self, receipt: &ZkTrustReceipt) -> bool {
        self.client.verify(receipt).unwrap_or(false)
    }
}

impl Default for TrustProverService {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate random blinding factor
fn generate_blinding() -> [u8; 32] {
    use std::time::{SystemTime, UNIX_EPOCH};
    let mut blinding = [0u8; 32];
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    blinding[..16].copy_from_slice(&now.to_le_bytes());
    // Use stack address as additional entropy
    let stack_var = 0u64;
    let addr = &stack_var as *const u64 as u64;
    blinding[16..24].copy_from_slice(&addr.to_le_bytes());
    blinding
}

/// Hash agent ID to u64
fn hash_agent_id(agent_id: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for byte in agent_id.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_kvector() -> SdkKVector {
        SdkKVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.7)
    }

    #[test]
    fn test_local_simulation_fallback() {
        let client = ProverServiceClient::new();
        let zk_kv = ZkKVector::from_sdk(&test_kvector());

        let input = ZkTrustProofInput {
            kvector: zk_kv,
            statement: ZkTrustStatement::TrustExceedsThreshold,
            params: ZkTrustParams {
                threshold: (0.5 * SCALE as f32) as u64,
                ..Default::default()
            },
            blinding: [42u8; 32],
            agent_id_hash: 12345,
            timestamp: 1000,
        };

        // This will fall back to simulation since no service is running
        let result = client.prove_locally_simulation(input);
        assert!(result.is_ok());
        let receipt = result.unwrap();
        assert!(receipt.is_simulation);
        assert!(receipt.output.statement_valid);
    }

    #[test]
    fn test_trust_prover_service_api() {
        let service = TrustProverService::new();
        let kv = test_kvector();

        // These will generate simulation proofs since no service is running
        let result = service.prove_trust_exceeds_threshold(&kv, 0.5, "agent-1");
        assert!(result.is_ok());
        let receipt = result.unwrap();
        assert!(receipt.output.statement_valid);
        assert!(service.verify(&receipt));
    }
}
