// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RISC-0 zkVM Integration
//!
//! Provides real ZK proof generation and verification using RISC-0 zkVM.
//!
//! # Architecture
//!
//! This module bridges the SDK's proof interfaces with RISC-0:
//!
//! 1. **GradientQualityInput** - Private witness (gradient data, constraints)
//! 2. **GradientQualityOutput** - Public output (validity, commitments)
//! 3. **Risc0GradientProver** - Host-side prover that runs the zkVM
//!
//! # Guest Program
//!
//! The guest program (runs inside zkVM) is located at:
//! `/methods/gradient-quality/guest/src/main.rs`
//!
//! It verifies:
//! - Gradient norm is within bounds (not too small, not too large)
//! - Gradient dimensions match expected size
//! - Gradient was computed from claimed model (via commitment)
//!
//! # Feature Flag
//!
//! Requires the `risc0` feature flag:
//! ```toml
//! mycelix-sdk = { version = "0.1", features = ["risc0"] }
//! ```

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

/// Input to the gradient quality zkVM guest (private witness)
///
/// This data is NEVER revealed - only the proof that it satisfies constraints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientQualityInput {
    /// The gradient values (PRIVATE - never revealed)
    pub gradients: Vec<f32>,
    /// Hash of the global model this gradient was computed from
    pub global_model_hash: [u8; 32],
    /// Number of local training epochs
    pub epochs: u32,
    /// Learning rate used
    pub learning_rate: f32,
    /// Client identifier
    pub client_id: String,
    /// Training round number
    pub round: u32,
    /// Constraint: minimum L2 norm
    pub min_norm: f32,
    /// Constraint: maximum L2 norm
    pub max_norm: f32,
}

/// Output from the gradient quality zkVM guest (public)
///
/// This is what the verifier sees - proof that SOME valid gradient exists
/// without knowing the actual values.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GradientQualityOutput {
    /// SHA3-256 hash of the gradient (commitment)
    pub gradient_hash: [u8; 32],
    /// Hash of the global model
    pub global_model_hash: [u8; 32],
    /// Number of epochs
    pub epochs: u32,
    /// Learning rate
    pub learning_rate: f32,
    /// Whether gradient norm is valid (within bounds)
    pub norm_valid: bool,
    /// Client identifier
    pub client_id: String,
    /// Training round
    pub round: u32,
    /// Overall commitment (hash of all public outputs)
    pub commitment: [u8; 32],
}

impl GradientQualityOutput {
    /// Compute commitment from output fields
    pub fn compute_commitment(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(self.gradient_hash);
        hasher.update(self.global_model_hash);
        hasher.update(self.epochs.to_le_bytes());
        hasher.update(self.learning_rate.to_le_bytes());
        hasher.update([self.norm_valid as u8]);
        hasher.update(self.client_id.as_bytes());
        hasher.update(self.round.to_le_bytes());

        let result = hasher.finalize();
        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(&result);
        commitment
    }
}

/// Hash gradient values using SHA3-256
pub fn hash_gradient(gradients: &[f32]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    for g in gradients {
        hasher.update(g.to_le_bytes());
    }
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Verify gradient quality constraints (same logic as zkVM guest)
///
/// This is used in simulation mode and can be used to pre-check inputs
/// before generating real proofs.
pub fn verify_gradient_constraints(gradients: &[f32], min_norm: f32, max_norm: f32) -> bool {
    if gradients.is_empty() {
        return false;
    }

    // Compute L2 norm
    let norm_squared: f32 = gradients.iter().map(|g| g * g).sum();
    let norm = norm_squared.sqrt();

    // Check bounds
    norm >= min_norm && norm <= max_norm && norm.is_finite()
}

/// Receipt from a real RISC-0 proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Risc0ProofReceipt {
    /// Public output from the proof
    pub output: GradientQualityOutput,
    /// Serialized RISC-0 receipt (contains STARK proof)
    pub receipt_bytes: Vec<u8>,
    /// Image ID of the guest program
    pub image_id: [u32; 8],
    /// Proof generation time in milliseconds
    pub generation_time_ms: u64,
}

impl Risc0ProofReceipt {
    /// Check if the proof indicates valid gradient quality
    pub fn is_valid(&self) -> bool {
        self.output.norm_valid
    }

    /// Get the gradient hash commitment
    pub fn gradient_hash(&self) -> &[u8; 32] {
        &self.output.gradient_hash
    }
}

/// RISC-0 Gradient Prover configuration
#[derive(Clone, Debug)]
pub struct Risc0ProverConfig {
    /// Path to the compiled ELF (gradient-quality guest)
    pub elf_path: Option<String>,
    /// ELF bytes (alternative to path)
    pub elf_bytes: Option<Vec<u8>>,
    /// Image ID for verification
    pub image_id: [u32; 8],
    /// Use development/dev mode (faster, less secure)
    pub dev_mode: bool,
}

impl Default for Risc0ProverConfig {
    fn default() -> Self {
        Self {
            elf_path: None,
            elf_bytes: None,
            image_id: [0u32; 8],
            dev_mode: true, // Default to dev mode for testing
        }
    }
}

// ============================================================================
// RISC-0 Feature-Gated Implementation
// ============================================================================

#[cfg(feature = "risc0")]
mod risc0_impl {
    use super::*;
    use risc0_zkvm::{default_prover, ExecutorEnv, Receipt};
    use std::time::Instant;

    /// RISC-0 Gradient Quality Prover
    ///
    /// Generates real STARK proofs using RISC-0 zkVM.
    pub struct Risc0GradientProver {
        config: Risc0ProverConfig,
    }

    impl Risc0GradientProver {
        /// Create a new prover with configuration
        pub fn new(config: Risc0ProverConfig) -> Self {
            Self { config }
        }

        /// Generate a proof of gradient quality
        pub fn prove(
            &self,
            input: &GradientQualityInput,
        ) -> Result<Risc0ProofReceipt, Risc0ProverError> {
            let start = Instant::now();

            // Get ELF bytes
            let elf = self.get_elf()?;

            // Build executor environment with input
            let env = ExecutorEnv::builder()
                .write(input)
                .map_err(|e| Risc0ProverError::EnvBuildFailed(e.to_string()))?
                .build()
                .map_err(|e| Risc0ProverError::EnvBuildFailed(e.to_string()))?;

            // Generate proof
            let prover = default_prover();
            let prove_info = prover
                .prove(env, &elf)
                .map_err(|e| Risc0ProverError::ProofFailed(e.to_string()))?;

            // Extract receipt and decode output
            let receipt = prove_info.receipt;
            let output: GradientQualityOutput = receipt
                .journal
                .decode()
                .map_err(|e| Risc0ProverError::DecodeFailed(e.to_string()))?;

            // Serialize receipt
            let receipt_bytes = bincode::serialize(&receipt)
                .map_err(|e| Risc0ProverError::SerializationFailed(e.to_string()))?;

            let generation_time_ms = start.elapsed().as_millis() as u64;

            Ok(Risc0ProofReceipt {
                output,
                receipt_bytes,
                image_id: self.config.image_id,
                generation_time_ms,
            })
        }

        /// Verify a proof receipt
        pub fn verify(&self, proof: &Risc0ProofReceipt) -> Result<bool, Risc0ProverError> {
            // Deserialize receipt
            let receipt: Receipt = bincode::deserialize(&proof.receipt_bytes)
                .map_err(|e| Risc0ProverError::DeserializationFailed(e.to_string()))?;

            // Verify with image ID
            receipt
                .verify(proof.image_id)
                .map_err(|e| Risc0ProverError::VerificationFailed(e.to_string()))?;

            // SECURITY: Verify commitment using constant-time comparison (FIND-004)
            let expected_commitment = proof.output.compute_commitment();
            if !secure_compare_hash(&proof.output.commitment, &expected_commitment) {
                return Ok(false);
            }

            Ok(proof.output.norm_valid)
        }

        /// Get ELF bytes from config
        fn get_elf(&self) -> Result<Vec<u8>, Risc0ProverError> {
            if let Some(ref bytes) = self.config.elf_bytes {
                return Ok(bytes.clone());
            }

            if let Some(ref path) = self.config.elf_path {
                return std::fs::read(path)
                    .map_err(|e| Risc0ProverError::ElfLoadFailed(e.to_string()));
            }

            Err(Risc0ProverError::NoElfConfigured)
        }
    }
}

#[cfg(feature = "risc0")]
pub use risc0_impl::Risc0GradientProver;

// ============================================================================
// Errors
// ============================================================================

/// Errors from RISC-0 prover
#[derive(Debug, Clone)]
pub enum Risc0ProverError {
    /// No ELF configured
    NoElfConfigured,
    /// Failed to load ELF
    ElfLoadFailed(String),
    /// Failed to build executor environment
    EnvBuildFailed(String),
    /// Proof generation failed
    ProofFailed(String),
    /// Failed to decode output
    DecodeFailed(String),
    /// Serialization failed
    SerializationFailed(String),
    /// Deserialization failed
    DeserializationFailed(String),
    /// Verification failed
    VerificationFailed(String),
}

impl std::fmt::Display for Risc0ProverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Risc0ProverError::NoElfConfigured => write!(f, "No ELF configured"),
            Risc0ProverError::ElfLoadFailed(e) => write!(f, "ELF load failed: {}", e),
            Risc0ProverError::EnvBuildFailed(e) => write!(f, "Env build failed: {}", e),
            Risc0ProverError::ProofFailed(e) => write!(f, "Proof failed: {}", e),
            Risc0ProverError::DecodeFailed(e) => write!(f, "Decode failed: {}", e),
            Risc0ProverError::SerializationFailed(e) => write!(f, "Serialization failed: {}", e),
            Risc0ProverError::DeserializationFailed(e) => {
                write!(f, "Deserialization failed: {}", e)
            }
            Risc0ProverError::VerificationFailed(e) => write!(f, "Verification failed: {}", e),
        }
    }
}

impl std::error::Error for Risc0ProverError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_hash_deterministic() {
        let gradients = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let hash1 = hash_gradient(&gradients);
        let hash2 = hash_gradient(&gradients);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_gradient_hash_different_for_different_input() {
        let g1 = vec![0.1f32, 0.2, 0.3];
        let g2 = vec![0.1f32, 0.2, 0.4];
        assert_ne!(hash_gradient(&g1), hash_gradient(&g2));
    }

    #[test]
    fn test_verify_constraints_valid() {
        let gradients: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();

        // Compute expected norm
        let norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();

        assert!(verify_gradient_constraints(&gradients, 0.0, norm + 10.0));
    }

    #[test]
    fn test_verify_constraints_too_small() {
        let gradients = vec![0.0f32; 100]; // Zero gradient
        assert!(!verify_gradient_constraints(&gradients, 0.1, 100.0));
    }

    #[test]
    fn test_verify_constraints_too_large() {
        let gradients = vec![1000.0f32; 100]; // Exploding gradient
        let norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!(!verify_gradient_constraints(&gradients, 0.0, norm - 1.0));
    }

    #[test]
    fn test_verify_constraints_empty() {
        assert!(!verify_gradient_constraints(&[], 0.0, 100.0));
    }

    #[test]
    fn test_output_commitment() {
        let output = GradientQualityOutput {
            gradient_hash: [0x42u8; 32],
            global_model_hash: [0x43u8; 32],
            epochs: 5,
            learning_rate: 0.01,
            norm_valid: true,
            client_id: "client-1".to_string(),
            round: 1,
            commitment: [0u8; 32],
        };

        let commitment = output.compute_commitment();
        assert_ne!(commitment, [0u8; 32]);

        // Deterministic
        assert_eq!(commitment, output.compute_commitment());
    }

    #[test]
    fn test_input_serialization() {
        let input = GradientQualityInput {
            gradients: vec![0.1, 0.2, 0.3],
            global_model_hash: [0x42u8; 32],
            epochs: 5,
            learning_rate: 0.01,
            client_id: "client-1".to_string(),
            round: 1,
            min_norm: 0.0,
            max_norm: 100.0,
        };

        let serialized = serde_json::to_string(&input).unwrap();
        let deserialized: GradientQualityInput = serde_json::from_str(&serialized).unwrap();

        assert_eq!(input.gradients, deserialized.gradients);
        assert_eq!(input.epochs, deserialized.epochs);
    }
}
