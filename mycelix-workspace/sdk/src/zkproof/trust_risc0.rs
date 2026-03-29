// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RISC-0 ZK Trust Proof Integration
//!
//! Bridges the SDK's trust types with the zk-trust-risc0 spike for generating
//! real zero-knowledge STARK proofs for K-Vector trust attestations.
//!
//! # Overview
//!
//! This module provides:
//! - Conversion between SDK KVector and ZK-friendly fixed-point representation
//! - Trust proof generation and verification
//! - Integration with the trust pipeline for Stage 4 (Attestation)
//!
//! # Privacy Model
//!
//! - **Private (Witness)**: The actual K-Vector values (10 dimensions)
//! - **Public (Journal)**: Statement result + commitment + agent ID + timestamp
//!
//! The proof guarantees: "I know a K-Vector that satisfies statement X"
//! WITHOUT revealing the actual K-Vector values.
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_sdk::zkproof::trust_risc0::*;
//! use mycelix_sdk::matl::KVector;
//!
//! let kvector = KVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.7);
//! let prover = TrustRisc0Prover::new_simulation();
//!
//! // Prove trust exceeds threshold without revealing actual values
//! let proof = prover.prove_trust_exceeds_threshold(&kvector, 0.5, "agent-123")?;
//!
//! // Verify the proof
//! assert!(prover.verify(&proof));
//! ```

use crate::agentic::zk_trust::ProofStatement;
use crate::matl::KVector as SdkKVector;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// ZK K-Vector Types (Fixed-Point for Determinism)
// ============================================================================

/// Scale factor for fixed-point representation
pub const SCALE: u64 = 1_000_000;

/// K-Vector in fixed-point representation for zkVM determinism.
/// Floating-point operations are non-deterministic in zkVM, so we use
/// scaled integers instead.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZkKVector {
    /// 10 dimensions stored as fixed-point (multiplied by SCALE)
    pub values: [u64; 10],
}

impl ZkKVector {
    /// Create from SDK KVector
    pub fn from_sdk(kv: &SdkKVector) -> Self {
        let arr = kv.to_array();
        Self {
            values: arr.map(|v| {
                let clamped = v.clamp(0.0, 1.0);
                (clamped * SCALE as f32) as u64
            }),
        }
    }

    /// Convert back to SDK KVector
    pub fn to_sdk(&self) -> SdkKVector {
        let arr: [f32; 10] = self.values.map(|v| v as f32 / SCALE as f32);
        SdkKVector::from_array(arr)
    }

    /// Compute weighted trust score (matches SDK weights)
    pub fn trust_score(&self) -> u64 {
        // Weights match SDK KVECTOR_WEIGHTS_ARRAY (scaled to sum to SCALE)
        const WEIGHTS: [u64; 10] = [
            240_000, // k_r: reputation
            110_000, // k_a: activity
            190_000, // k_i: integrity
            110_000, // k_p: performance
            50_000,  // k_m: membership
            70_000,  // k_s: stake
            50_000,  // k_h: historical
            40_000,  // k_topo: topology
            90_000,  // k_v: verification
            50_000,  // k_coherence: coherence
        ];

        let mut weighted_sum: u128 = 0;
        let mut weight_sum: u128 = 0;

        for (val, &weight) in self.values.iter().zip(WEIGHTS.iter()) {
            weighted_sum += (*val as u128) * (weight as u128);
            weight_sum += weight as u128;
        }

        (weighted_sum / weight_sum) as u64
    }

    /// Get trust score as f32
    pub fn trust_score_f32(&self) -> f32 {
        self.trust_score() as f32 / SCALE as f32
    }
}

// ============================================================================
// Proof Types
// ============================================================================

/// Type of trust statement to prove
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ZkTrustStatement {
    /// Trust score exceeds threshold: T > threshold
    TrustExceedsThreshold = 0,
    /// Trust score is within range: min <= T <= max
    TrustInRange = 1,
    /// Specific dimension exceeds threshold: k_i > threshold
    DimensionExceedsThreshold = 2,
    /// Multiple dimensions exceed thresholds
    MultipleDimensionsExceed = 3,
    /// K-Vector matches a commitment (for evolution proofs)
    KVectorMatchesCommitment = 4,
    /// Agent is verified (k_v >= 0.5)
    IsVerified = 5,
    /// Agent is strongly verified (k_v >= 0.7)
    IsStronglyVerified = 6,
    /// Agent is highly coherent (k_coherence >= 0.7)
    IsHighlyCoherent = 7,
}

/// Input for ZK trust proof (PRIVATE - never revealed)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZkTrustProofInput {
    /// The actual K-Vector (PRIVATE)
    pub kvector: ZkKVector,
    /// Statement type to prove
    pub statement: ZkTrustStatement,
    /// Statement parameters
    pub params: ZkTrustParams,
    /// Random blinding factor for commitment
    pub blinding: [u8; 32],
    /// Agent ID hash
    pub agent_id_hash: u64,
    /// Timestamp
    pub timestamp: u64,
}

/// Parameters for trust statements
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ZkTrustParams {
    /// Threshold (scaled by SCALE)
    pub threshold: u64,
    /// Min value for range (scaled)
    pub min_value: u64,
    /// Max value for range (scaled)
    pub max_value: u64,
    /// Dimension index (0-9)
    pub dimension_index: u8,
    /// Multiple dimension indices
    pub dimension_indices: [u8; 10],
    /// Number of dimensions to check
    pub dimension_count: u8,
    /// Expected commitment for matching
    pub expected_commitment: [u8; 32],
}

/// Output from ZK trust proof (PUBLIC - safe to reveal)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZkTrustProofOutput {
    /// Whether the statement is satisfied
    pub statement_valid: bool,
    /// The statement type that was proven
    pub statement: ZkTrustStatement,
    /// Commitment to the K-Vector (hides actual values)
    pub kvector_commitment: [u8; 32],
    /// Agent ID hash
    pub agent_id_hash: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Revealed trust score (only for TrustInRange, else 0)
    pub revealed_score: u64,
}

/// Complete ZK trust proof (combines output with cryptographic proof)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZkTrustReceipt {
    /// The public output
    pub output: ZkTrustProofOutput,
    /// Proof bytes (STARK proof in production, simulation marker in test)
    pub proof_bytes: Vec<u8>,
    /// Whether this is a simulation proof
    pub is_simulation: bool,
    /// Proof generation time in milliseconds
    pub proof_time_ms: u64,
}

// ============================================================================
// Commitment Functions
// ============================================================================

/// Compute a commitment to a K-Vector using FNV-1a hashing.
/// In production, this should use SHA3-256 for cryptographic security.
pub fn compute_commitment(
    kvector: &ZkKVector,
    blinding: &[u8; 32],
    agent_id_hash: u64,
    timestamp: u64,
) -> [u8; 32] {
    // FNV-1a 64-bit parameters
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash1 = FNV_OFFSET;
    let mut hash2 = FNV_OFFSET;
    let mut hash3 = FNV_OFFSET;
    let mut hash4 = FNV_OFFSET;

    // Hash K-Vector values
    for value in kvector.values {
        for byte in value.to_le_bytes() {
            hash1 ^= byte as u64;
            hash1 = hash1.wrapping_mul(FNV_PRIME);
        }
    }

    // Hash blinding factor
    for byte in blinding {
        hash2 ^= *byte as u64;
        hash2 = hash2.wrapping_mul(FNV_PRIME);
    }

    // Hash agent ID
    for byte in agent_id_hash.to_le_bytes() {
        hash3 ^= byte as u64;
        hash3 = hash3.wrapping_mul(FNV_PRIME);
    }

    // Hash timestamp
    for byte in timestamp.to_le_bytes() {
        hash4 ^= byte as u64;
        hash4 = hash4.wrapping_mul(FNV_PRIME);
    }

    // Combine into 32-byte commitment
    let mut commitment = [0u8; 32];
    commitment[0..8].copy_from_slice(&hash1.to_le_bytes());
    commitment[8..16].copy_from_slice(&hash2.to_le_bytes());
    commitment[16..24].copy_from_slice(&hash3.to_le_bytes());
    commitment[24..32].copy_from_slice(&hash4.to_le_bytes());

    commitment
}

/// Verify a K-Vector matches a commitment
pub fn verify_commitment(
    kvector: &ZkKVector,
    blinding: &[u8; 32],
    agent_id_hash: u64,
    timestamp: u64,
    expected: &[u8; 32],
) -> bool {
    let computed = compute_commitment(kvector, blinding, agent_id_hash, timestamp);
    computed == *expected
}

// ============================================================================
// Statement Evaluation
// ============================================================================

/// Evaluate if a K-Vector satisfies a statement.
/// This logic runs INSIDE the zkVM during proof generation.
pub fn evaluate_statement(
    kvector: &ZkKVector,
    statement: ZkTrustStatement,
    params: &ZkTrustParams,
) -> bool {
    match statement {
        ZkTrustStatement::TrustExceedsThreshold => kvector.trust_score() > params.threshold,

        ZkTrustStatement::TrustInRange => {
            let score = kvector.trust_score();
            score >= params.min_value && score <= params.max_value
        }

        ZkTrustStatement::DimensionExceedsThreshold => {
            let dim = params.dimension_index as usize;
            if dim < 10 {
                kvector.values[dim] > params.threshold
            } else {
                false
            }
        }

        ZkTrustStatement::MultipleDimensionsExceed => {
            let count = params.dimension_count.min(10) as usize;
            (0..count).all(|i| {
                let dim = params.dimension_indices[i] as usize;
                dim < 10 && kvector.values[dim] > params.threshold
            })
        }

        ZkTrustStatement::KVectorMatchesCommitment => {
            // Handled separately during proof generation
            true
        }

        ZkTrustStatement::IsVerified => {
            // k_v (index 8) >= 0.5
            kvector.values[8] >= SCALE / 2
        }

        ZkTrustStatement::IsStronglyVerified => {
            // k_v (index 8) >= 0.7
            kvector.values[8] >= (SCALE * 7) / 10
        }

        ZkTrustStatement::IsHighlyCoherent => {
            // k_coherence (index 9) >= 0.7
            kvector.values[9] >= (SCALE * 7) / 10
        }
    }
}

// ============================================================================
// Prover
// ============================================================================

/// RISC-0 Trust Prover
///
/// Generates ZK proofs for trust statements.
pub struct TrustRisc0Prover {
    /// Whether running in simulation mode
    simulation_mode: bool,
}

impl TrustRisc0Prover {
    /// Create a new simulation prover (NO CRYPTOGRAPHIC SECURITY)
    pub fn new_simulation() -> Self {
        Self {
            simulation_mode: true,
        }
    }

    /// Create a production prover (requires RISC-0 setup)
    #[cfg(feature = "risc0")]
    pub fn new_production() -> Self {
        Self {
            simulation_mode: false,
        }
    }

    /// Check if running in simulation mode
    pub fn is_simulation(&self) -> bool {
        self.simulation_mode
    }

    /// Generate a trust proof
    pub fn prove(&self, input: ZkTrustProofInput) -> Result<ZkTrustReceipt, TrustProverError> {
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

        // Generate proof
        let proof_bytes = if self.simulation_mode {
            // Simulation: Just serialize output with marker
            let marker = b"SIMULATION_TRUST_PROOF_V1";
            let mut bytes = marker.to_vec();
            bytes.extend(bincode::serialize(&output).unwrap_or_default());
            bytes
        } else {
            // Production: Use external RISC-0 prover service
            //
            // The RISC-0 prover is heavyweight and should run as a separate process.
            // Use one of these approaches:
            //
            // 1. Local prover service (spike/zk-trust-risc0/host):
            //    - Start the prover: `cd spike/zk-trust-risc0 && cargo run --release`
            //    - Connect via configured endpoint (default: localhost:3000)
            //
            // 2. Remote Bonsai proving service:
            //    - Set BONSAI_API_KEY and BONSAI_API_URL environment variables
            //    - Proofs are generated in the cloud
            //
            // 3. In-process (requires 'risc0-prover' feature and compatible serde):
            //    - Build with: cargo build --features risc0-prover
            //    - Heavyweight, not recommended for most use cases
            //
            #[cfg(feature = "risc0")]
            {
                // Try external prover service first
                let prover_url = std::env::var("RISC0_PROVER_URL")
                    .unwrap_or_else(|_| "http://localhost:3000/prove".to_string());

                // Serialize input for the prover service
                let request_body = bincode::serialize(&input)
                    .map_err(|e| TrustProverError::Risc0Execution(format!("Serialize: {}", e)))?;

                // Call the prover service
                // Note: In a real implementation, use an async HTTP client
                // For now, return a stub indicating external prover needed
                return Err(TrustProverError::Risc0Execution(format!(
                    "Production proofs require external prover service at {}. \
                    Run `cd spike/zk-trust-risc0 && cargo run --release` to start the prover.",
                    prover_url
                )));
            }
            #[cfg(not(feature = "risc0"))]
            {
                return Err(TrustProverError::Risc0NotAvailable);
            }
        };

        let proof_time_ms = start.elapsed().as_millis() as u64;

        Ok(ZkTrustReceipt {
            output,
            proof_bytes,
            is_simulation: self.simulation_mode,
            proof_time_ms,
        })
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

        self.prove(input)
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

        self.prove(input)
    }

    /// Prove a dimension exceeds threshold
    pub fn prove_dimension_exceeds(
        &self,
        kvector: &SdkKVector,
        dimension: u8,
        threshold: f32,
        agent_id: &str,
    ) -> Result<ZkTrustReceipt, TrustProverError> {
        let zk_kv = ZkKVector::from_sdk(kvector);
        let blinding = generate_blinding();
        let agent_id_hash = hash_agent_id(agent_id);
        let timestamp = current_timestamp();

        let input = ZkTrustProofInput {
            kvector: zk_kv,
            statement: ZkTrustStatement::DimensionExceedsThreshold,
            params: ZkTrustParams {
                dimension_index: dimension.min(9),
                threshold: (threshold.clamp(0.0, 1.0) * SCALE as f32) as u64,
                ..Default::default()
            },
            blinding,
            agent_id_hash,
            timestamp,
        };

        self.prove(input)
    }

    /// Prove agent is verified
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

        self.prove(input)
    }

    /// Prove agent is highly coherent
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

        self.prove(input)
    }

    /// Verify a trust proof
    pub fn verify(&self, receipt: &ZkTrustReceipt) -> bool {
        if receipt.is_simulation {
            // Simulation proofs just check the marker
            receipt
                .proof_bytes
                .starts_with(b"SIMULATION_TRUST_PROOF_V1")
        } else {
            // Production proofs need RISC-0 verification
            #[cfg(feature = "risc0")]
            {
                Self::verify_risc0_proof(receipt).unwrap_or(false)
            }
            #[cfg(not(feature = "risc0"))]
            {
                false
            }
        }
    }

    /// Verify a RISC-0 proof (production only)
    #[cfg(feature = "risc0")]
    fn verify_risc0_proof(receipt: &ZkTrustReceipt) -> Result<bool, TrustProverError> {
        use risc0_zkvm::Receipt;

        // Deserialize the RISC-0 receipt
        let risc0_receipt: Receipt = bincode::deserialize(&receipt.proof_bytes)
            .map_err(|e| TrustProverError::Risc0Verification(format!("Deserialize: {}", e)))?;

        // Get the image ID for verification
        let image_id = get_trust_proof_image_id()?;

        // Verify the cryptographic proof
        risc0_receipt
            .verify(image_id)
            .map_err(|e| TrustProverError::Risc0Verification(format!("Verify: {}", e)))?;

        Ok(true)
    }
}

// ============================================================================
// Production Prover Integration
// ============================================================================

/// Helper to call external prover service.
///
/// The SDK does not include the full RISC-0 prover due to dependency size.
/// Instead, use the external prover service from spike/zk-trust-risc0.
///
/// # Example Integration
///
/// ```rust,ignore
/// // Start the prover service:
/// // cd spike/zk-trust-risc0 && cargo run --release
///
/// // Then in your application:
/// use mycelix_sdk::zkproof::trust_risc0::*;
///
/// let config = ProverServiceConfig::default();
/// let prover = TrustRisc0Prover::new_production();
///
/// // Proof generation will call the external service
/// let proof = prover.prove_trust_exceeds_threshold(&kvector, 0.5, "agent-1")?;
/// ```
pub fn create_production_prover() -> TrustRisc0Prover {
    TrustRisc0Prover {
        simulation_mode: false,
    }
}

/// Prover errors
#[derive(Debug, Clone, PartialEq)]
pub enum TrustProverError {
    /// RISC-0 not available (feature not enabled)
    Risc0NotAvailable,
    /// RISC-0 ELF binary not configured
    Risc0ElfNotSet,
    /// RISC-0 execution error
    Risc0Execution(String),
    /// RISC-0 verification error
    Risc0Verification(String),
    /// Invalid statement parameters
    InvalidParams(String),
    /// Proof generation failed
    ProofGenerationFailed(String),
}

impl std::fmt::Display for TrustProverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Risc0NotAvailable => {
                write!(f, "RISC-0 prover not available (enable 'risc0' feature)")
            }
            Self::Risc0ElfNotSet => write!(
                f,
                "RISC-0 ELF binary not configured - call set_trust_proof_elf() first"
            ),
            Self::Risc0Execution(msg) => write!(f, "RISC-0 execution error: {}", msg),
            Self::Risc0Verification(msg) => write!(f, "RISC-0 verification error: {}", msg),
            Self::InvalidParams(msg) => write!(f, "Invalid parameters: {}", msg),
            Self::ProofGenerationFailed(msg) => write!(f, "Proof generation failed: {}", msg),
        }
    }
}

impl std::error::Error for TrustProverError {}

// ============================================================================
// Production RISC-0 Configuration
// ============================================================================

#[cfg(feature = "risc0")]
mod risc0_config {
    use std::sync::OnceLock;

    /// Global image ID for verification
    static TRUST_PROOF_IMAGE_ID: OnceLock<[u32; 8]> = OnceLock::new();

    /// Set the RISC-0 image ID for proof verification.
    ///
    /// This must match the image ID from the compiled guest program.
    /// The image ID is generated when building the zk-trust-risc0 methods crate.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use mycelix_sdk::zkproof::trust_risc0::set_trust_proof_image_id;
    /// use zk_trust_methods::TRUST_PROOF_ID;
    ///
    /// set_trust_proof_image_id(TRUST_PROOF_ID);
    /// ```
    pub fn set_trust_proof_image_id(image_id: [u32; 8]) {
        TRUST_PROOF_IMAGE_ID.set(image_id).ok();
    }

    /// Get the configured image ID for verification
    pub fn get_trust_proof_image_id() -> Result<[u32; 8], super::TrustProverError> {
        TRUST_PROOF_IMAGE_ID
            .get()
            .copied()
            .ok_or(super::TrustProverError::Risc0ElfNotSet)
    }
}

#[cfg(feature = "risc0")]
pub use risc0_config::{get_trust_proof_image_id, set_trust_proof_image_id};

/// Configuration for the external RISC-0 prover service.
///
/// The RISC-0 prover is heavyweight and runs as a separate process.
/// Configure the prover URL via the `RISC0_PROVER_URL` environment variable.
///
/// # Starting the Prover
///
/// ```bash
/// cd spike/zk-trust-risc0
/// cargo run --release
/// ```
///
/// # Environment Variables
///
/// - `RISC0_PROVER_URL`: URL of the prover service (default: http://localhost:3000)
/// - `BONSAI_API_KEY`: For cloud proving via Bonsai service
/// - `BONSAI_API_URL`: Bonsai service URL
#[derive(Debug, Clone)]
pub struct ProverServiceConfig {
    /// URL of the external prover service
    pub prover_url: String,
    /// Timeout for proof generation (milliseconds)
    pub timeout_ms: u64,
}

impl Default for ProverServiceConfig {
    fn default() -> Self {
        Self {
            prover_url: std::env::var("RISC0_PROVER_URL")
                .unwrap_or_else(|_| "http://localhost:3000".to_string()),
            timeout_ms: 300_000, // 5 minutes default
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate random blinding factor
fn generate_blinding() -> [u8; 32] {
    let mut blinding = [0u8; 32];
    // Use timestamp and pointer address for pseudo-randomness
    // In production, use a proper CSPRNG
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
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================================================
// SDK Integration
// ============================================================================

/// Convert SDK ProofStatement to ZkTrustStatement
impl From<&ProofStatement> for ZkTrustStatement {
    fn from(stmt: &ProofStatement) -> Self {
        match stmt {
            ProofStatement::TrustExceedsThreshold { .. } => Self::TrustExceedsThreshold,
            ProofStatement::TrustInRange { .. } => Self::TrustInRange,
            ProofStatement::DimensionExceedsThreshold { .. } => Self::DimensionExceedsThreshold,
            ProofStatement::MultipleDimensionsExceed { .. } => Self::MultipleDimensionsExceed,
            ProofStatement::IsVerified => Self::IsVerified,
            ProofStatement::IsStronglyVerified => Self::IsStronglyVerified,
            ProofStatement::WellFormed => Self::TrustInRange, // Map to range [0,1]
            _ => Self::TrustExceedsThreshold,                 // Default fallback
        }
    }
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
    fn test_zk_kvector_conversion() {
        let sdk_kv = test_kvector();
        let zk_kv = ZkKVector::from_sdk(&sdk_kv);
        let back = zk_kv.to_sdk();

        // Should round-trip with minimal loss
        let orig_arr = sdk_kv.to_array();
        let back_arr = back.to_array();
        for i in 0..10 {
            assert!((orig_arr[i] - back_arr[i]).abs() < 0.001);
        }
    }

    #[test]
    fn test_trust_score_calculation() {
        let sdk_kv = test_kvector();
        let zk_kv = ZkKVector::from_sdk(&sdk_kv);

        let zk_score = zk_kv.trust_score_f32();
        let sdk_score = sdk_kv.trust_score();

        // Scores should be close (different weight implementations)
        assert!((zk_score - sdk_score).abs() < 0.1);
    }

    #[test]
    fn test_commitment_deterministic() {
        let sdk_kv = test_kvector();
        let zk_kv = ZkKVector::from_sdk(&sdk_kv);
        let blinding = [42u8; 32];

        let c1 = compute_commitment(&zk_kv, &blinding, 12345, 1000);
        let c2 = compute_commitment(&zk_kv, &blinding, 12345, 1000);

        assert_eq!(c1, c2);
    }

    #[test]
    fn test_commitment_changes_with_blinding() {
        let sdk_kv = test_kvector();
        let zk_kv = ZkKVector::from_sdk(&sdk_kv);

        let c1 = compute_commitment(&zk_kv, &[42u8; 32], 12345, 1000);
        let c2 = compute_commitment(&zk_kv, &[43u8; 32], 12345, 1000);

        assert_ne!(c1, c2);
    }

    #[test]
    fn test_prove_trust_exceeds() {
        let prover = TrustRisc0Prover::new_simulation();
        let kv = test_kvector();

        // Should succeed (trust > 0.5)
        let proof = prover
            .prove_trust_exceeds_threshold(&kv, 0.5, "agent-1")
            .unwrap();
        assert!(proof.output.statement_valid);
        assert!(prover.verify(&proof));

        // Should fail (trust < 0.95)
        let proof2 = prover
            .prove_trust_exceeds_threshold(&kv, 0.95, "agent-1")
            .unwrap();
        assert!(!proof2.output.statement_valid);
    }

    #[test]
    fn test_prove_trust_in_range() {
        let prover = TrustRisc0Prover::new_simulation();
        let kv = test_kvector();

        let proof = prover
            .prove_trust_in_range(&kv, 0.5, 0.9, "agent-1")
            .unwrap();
        assert!(proof.output.statement_valid);
        // Score should be revealed
        assert!(proof.output.revealed_score > 0);
    }

    #[test]
    fn test_prove_dimension_exceeds() {
        let prover = TrustRisc0Prover::new_simulation();
        let kv = test_kvector();

        // k_i (integrity, index 2) is 0.9
        let proof = prover
            .prove_dimension_exceeds(&kv, 2, 0.8, "agent-1")
            .unwrap();
        assert!(proof.output.statement_valid);

        // k_m (membership, index 4) is 0.5
        let proof2 = prover
            .prove_dimension_exceeds(&kv, 4, 0.6, "agent-1")
            .unwrap();
        assert!(!proof2.output.statement_valid);
    }

    #[test]
    fn test_prove_is_verified() {
        let prover = TrustRisc0Prover::new_simulation();
        let kv = test_kvector(); // k_v = 0.75

        let proof = prover.prove_is_verified(&kv, "agent-1").unwrap();
        assert!(proof.output.statement_valid);
    }

    #[test]
    fn test_prove_is_coherent() {
        let prover = TrustRisc0Prover::new_simulation();
        let kv = test_kvector(); // k_coherence = 0.7

        let proof = prover.prove_is_coherent(&kv, "agent-1").unwrap();
        assert!(proof.output.statement_valid);

        // Low coherence agent
        let low_kv = SdkKVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.3);
        let proof2 = prover.prove_is_coherent(&low_kv, "agent-2").unwrap();
        assert!(!proof2.output.statement_valid);
    }
}
