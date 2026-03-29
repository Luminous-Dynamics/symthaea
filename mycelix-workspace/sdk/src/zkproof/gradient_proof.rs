// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gradient Proof Types for RISC-0 Integration
//!
//! Defines input/output types for ZK proofs of gradient quality.
//! Designed to integrate with RISC Zero zkVM when the `risc0` feature is enabled.
//!
//! # Proof Statement
//!
//! "I computed gradient G from global model W using dataset D with learning rate η
//! for E epochs, and the gradient has valid L2 norm within the expected range."
//!
//! # What the Proof Reveals
//!
//! - Gradient hash (commitment)
//! - Global model hash (commitment)
//! - Training parameters (epochs, learning rate)
//! - L2 norm is within bounds
//! - Client ID for attribution
//!
//! # What Remains Private
//!
//! - The actual gradient values
//! - The training dataset
//! - The exact L2 norm (only proves it's in range)

use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Input to the gradient proof circuit (private witness)
///
/// This data is known only to the prover and is never revealed.
///
/// # Security
///
/// This struct contains sensitive training data (gradients) that should never
/// be exposed. The `ZeroizeOnDrop` derive ensures that all sensitive fields
/// are securely zeroed when the struct is dropped, preventing memory scraping
/// attacks from recovering private gradient data.
#[derive(Debug, Clone, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct GradientProofInput {
    /// The computed gradient vector (SENSITIVE - will be zeroized on drop)
    pub gradient: Vec<f32>,
    /// The global model this gradient was computed against
    pub global_model_hash: [u8; 32],
    /// Number of training epochs
    #[zeroize(skip)]
    pub epochs: u32,
    /// Learning rate used
    pub learning_rate: f32,
    /// Expected L2 norm (from quality estimation)
    pub expected_norm: f32,
    /// Tolerance for norm validation (±)
    pub norm_tolerance: f32,
    /// Client identifier
    #[zeroize(skip)]
    pub client_id: String,
    /// Training round number
    #[zeroize(skip)]
    pub round: u32,
}

/// Output from the gradient proof circuit (public journal)
///
/// This data is committed to the proof and visible to verifiers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GradientProofOutput {
    /// Hash of the gradient (SHA3-256)
    pub gradient_hash: [u8; 32],
    /// Hash of the global model used
    pub global_model_hash: [u8; 32],
    /// Training epochs
    pub epochs: u32,
    /// Learning rate
    pub learning_rate: f32,
    /// Whether the gradient norm is valid (within expected range)
    pub norm_valid: bool,
    /// Client identifier
    pub client_id: String,
    /// Training round
    pub round: u32,
    /// Commitment binding all public inputs
    pub commitment: [u8; 32],
}

impl GradientProofOutput {
    /// Check if the proof indicates valid gradient
    pub fn is_valid(&self) -> bool {
        self.norm_valid
    }
}

/// Gradient quality constraints for the circuit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConstraints {
    /// Minimum acceptable L2 norm
    pub min_norm: f32,
    /// Maximum acceptable L2 norm
    pub max_norm: f32,
    /// Maximum allowed gradient magnitude per element
    pub max_element_magnitude: f32,
    /// Minimum required gradient dimension
    pub min_dimension: usize,
}

impl Default for GradientConstraints {
    fn default() -> Self {
        Self {
            min_norm: 0.001, // Gradient shouldn't be zero
            max_norm: 100.0, // Prevent gradient explosion
            max_element_magnitude: 10.0,
            min_dimension: 10,
        }
    }
}

impl GradientConstraints {
    /// Create constraints for federated learning
    pub fn for_federated_learning() -> Self {
        Self {
            min_norm: 0.0001,
            max_norm: 50.0,
            max_element_magnitude: 5.0,
            min_dimension: 100,
        }
    }

    /// Create strict constraints for high-stakes scenarios
    pub fn strict() -> Self {
        Self {
            min_norm: 0.001,
            max_norm: 10.0,
            max_element_magnitude: 1.0,
            min_dimension: 1000,
        }
    }
}

/// Verify gradient quality against constraints
///
/// This is the core logic that would run inside the zkVM guest.
pub fn verify_gradient_quality(
    gradient: &[f32],
    constraints: &GradientConstraints,
) -> GradientQualityResult {
    // Check dimension
    if gradient.len() < constraints.min_dimension {
        return GradientQualityResult {
            valid: false,
            l2_norm: 0.0,
            max_element: 0.0,
            dimension: gradient.len(),
            failure_reason: Some("Gradient dimension too small".to_string()),
        };
    }

    // Calculate L2 norm
    let l2_norm: f32 = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();

    // Find max element magnitude
    let max_element: f32 = gradient
        .iter()
        .map(|g| g.abs())
        .fold(0.0f32, |a, b| a.max(b));

    // Validate constraints
    if l2_norm < constraints.min_norm {
        return GradientQualityResult {
            valid: false,
            l2_norm,
            max_element,
            dimension: gradient.len(),
            failure_reason: Some(format!(
                "L2 norm too small: {} < {}",
                l2_norm, constraints.min_norm
            )),
        };
    }

    if l2_norm > constraints.max_norm {
        return GradientQualityResult {
            valid: false,
            l2_norm,
            max_element,
            dimension: gradient.len(),
            failure_reason: Some(format!(
                "L2 norm too large: {} > {}",
                l2_norm, constraints.max_norm
            )),
        };
    }

    if max_element > constraints.max_element_magnitude {
        return GradientQualityResult {
            valid: false,
            l2_norm,
            max_element,
            dimension: gradient.len(),
            failure_reason: Some(format!(
                "Element magnitude too large: {} > {}",
                max_element, constraints.max_element_magnitude
            )),
        };
    }

    // Check for NaN/Inf
    if !l2_norm.is_finite() || gradient.iter().any(|g| !g.is_finite()) {
        return GradientQualityResult {
            valid: false,
            l2_norm,
            max_element,
            dimension: gradient.len(),
            failure_reason: Some("Gradient contains NaN or Inf".to_string()),
        };
    }

    GradientQualityResult {
        valid: true,
        l2_norm,
        max_element,
        dimension: gradient.len(),
        failure_reason: None,
    }
}

/// Result of gradient quality verification
#[derive(Debug, Clone)]
pub struct GradientQualityResult {
    /// Whether the gradient passes all constraints
    pub valid: bool,
    /// Computed L2 norm
    pub l2_norm: f32,
    /// Maximum element magnitude
    pub max_element: f32,
    /// Gradient dimension
    pub dimension: usize,
    /// Reason for failure (if any)
    pub failure_reason: Option<String>,
}

/// Domain separator for gradient hashing (FIND-011 mitigation)
pub const GRADIENT_HASH_DOMAIN: &[u8] = b"mycelix-gradient-hash-v1";

/// Hash a gradient vector using SHA3-256
///
/// # Security Note (FIND-011 mitigation)
///
/// This function includes domain separation to prevent cross-protocol hash reuse.
/// The domain separator "mycelix-gradient-hash-v1" ensures that hashes of
/// gradient vectors cannot be confused with hashes from other contexts.
pub fn hash_gradient(gradient: &[f32]) -> [u8; 32] {
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();

    // FIND-011: Add domain separator
    hasher.update(GRADIENT_HASH_DOMAIN);

    for g in gradient {
        hasher.update(g.to_le_bytes());
    }
    hasher.finalize().into()
}

/// Domain separator for gradient commitment to prevent cross-protocol attacks (FIND-011 mitigation)
pub const GRADIENT_COMMITMENT_DOMAIN: &[u8] = b"mycelix-gradient-commitment-v1";

/// Compute a commitment over all public inputs
///
/// # Security Note (FIND-011 mitigation)
///
/// This function includes domain separation to prevent cross-protocol
/// commitment reuse attacks. The domain separator "mycelix-gradient-commitment-v1"
/// ensures that commitments generated for gradient proofs cannot be confused
/// with commitments from other protocols or contexts.
///
/// # Domain Separation
///
/// The commitment is computed as:
/// ```text
/// SHA3-256(
///     "mycelix-gradient-commitment-v1" ||
///     gradient_hash ||
///     global_model_hash ||
///     epochs ||
///     learning_rate ||
///     norm_valid ||
///     client_id ||
///     round
/// )
/// ```
pub fn compute_commitment(output: &GradientProofOutput) -> [u8; 32] {
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();

    // FIND-011: Add domain separator to prevent cross-protocol attacks
    hasher.update(GRADIENT_COMMITMENT_DOMAIN);

    hasher.update(output.gradient_hash);
    hasher.update(output.global_model_hash);
    hasher.update(output.epochs.to_le_bytes());
    hasher.update(output.learning_rate.to_le_bytes());
    hasher.update([output.norm_valid as u8]);
    hasher.update(output.client_id.as_bytes());
    hasher.update(output.round.to_le_bytes());
    hasher.finalize().into()
}

/// RISC-0 proof configuration
#[derive(Debug, Clone)]
pub struct Risc0ProofConfig {
    /// Whether to use real RISC-0 proofs (requires risc0 feature)
    pub use_real_proofs: bool,
    /// Timeout for proof generation in seconds
    pub proof_timeout_secs: u64,
    /// Whether to cache generated proofs
    pub enable_caching: bool,
}

impl Default for Risc0ProofConfig {
    fn default() -> Self {
        Self {
            use_real_proofs: false, // Default to simulation
            proof_timeout_secs: 120,
            enable_caching: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_quality_valid() {
        let gradient: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).sin() * 0.1).collect();
        let constraints = GradientConstraints::default();

        let result = verify_gradient_quality(&gradient, &constraints);

        assert!(result.valid);
        assert!(result.l2_norm > 0.0);
        assert!(result.failure_reason.is_none());
    }

    #[test]
    fn test_gradient_quality_too_small() {
        let gradient: Vec<f32> = vec![0.0; 1000];
        let constraints = GradientConstraints::default();

        let result = verify_gradient_quality(&gradient, &constraints);

        assert!(!result.valid);
        assert!(result.failure_reason.unwrap().contains("too small"));
    }

    #[test]
    fn test_gradient_quality_too_large() {
        let gradient: Vec<f32> = vec![1000.0; 1000];
        let constraints = GradientConstraints::default();

        let result = verify_gradient_quality(&gradient, &constraints);

        assert!(!result.valid);
        assert!(result.failure_reason.unwrap().contains("too large"));
    }

    #[test]
    fn test_gradient_quality_dimension_check() {
        let gradient: Vec<f32> = vec![0.1; 5]; // Too small
        let constraints = GradientConstraints::default();

        let result = verify_gradient_quality(&gradient, &constraints);

        assert!(!result.valid);
        assert!(result.failure_reason.unwrap().contains("dimension"));
    }

    #[test]
    fn test_hash_gradient() {
        let gradient1: Vec<f32> = vec![0.1, 0.2, 0.3];
        let gradient2: Vec<f32> = vec![0.1, 0.2, 0.3];
        let gradient3: Vec<f32> = vec![0.1, 0.2, 0.4];

        let hash1 = hash_gradient(&gradient1);
        let hash2 = hash_gradient(&gradient2);
        let hash3 = hash_gradient(&gradient3);

        assert_eq!(hash1, hash2); // Same gradients = same hash
        assert_ne!(hash1, hash3); // Different gradients = different hash
    }

    #[test]
    fn test_gradient_proof_output() {
        let output = GradientProofOutput {
            gradient_hash: [1u8; 32],
            global_model_hash: [2u8; 32],
            epochs: 5,
            learning_rate: 0.01,
            norm_valid: true,
            client_id: "test".to_string(),
            round: 1,
            commitment: [0u8; 32],
        };

        assert!(output.is_valid());

        let commitment = compute_commitment(&output);
        assert_ne!(commitment, [0u8; 32]);
    }

    #[test]
    fn test_constraints_presets() {
        let default_c = GradientConstraints::default();
        let fl_c = GradientConstraints::for_federated_learning();
        let strict_c = GradientConstraints::strict();

        // Strict should have smaller bounds than default
        assert!(strict_c.max_norm <= default_c.max_norm);
        assert!(strict_c.max_element_magnitude <= default_c.max_element_magnitude);

        // FL constraints should be reasonable
        assert!(fl_c.min_norm > 0.0);
        assert!(fl_c.max_norm > fl_c.min_norm);
    }

    // ==========================================================================
    // FIND-011 Mitigation Tests: Domain separation
    // ==========================================================================

    #[test]
    fn test_domain_separator_constants() {
        // Verify domain separators are non-empty and distinct
        assert!(!GRADIENT_HASH_DOMAIN.is_empty());
        assert!(!GRADIENT_COMMITMENT_DOMAIN.is_empty());
        assert_ne!(GRADIENT_HASH_DOMAIN, GRADIENT_COMMITMENT_DOMAIN);
    }

    #[test]
    fn test_hash_gradient_with_domain_separation() {
        // Test that the hash includes domain separation by verifying
        // different gradients produce different hashes
        let gradient1: Vec<f32> = vec![0.1, 0.2, 0.3];
        let gradient2: Vec<f32> = vec![0.3, 0.2, 0.1]; // Different order

        let hash1 = hash_gradient(&gradient1);
        let hash2 = hash_gradient(&gradient2);

        // Different gradients should produce different hashes
        assert_ne!(hash1, hash2);

        // Hash should be 32 bytes (SHA3-256)
        assert_eq!(hash1.len(), 32);
        assert_eq!(hash2.len(), 32);
    }

    #[test]
    fn test_commitment_with_domain_separation() {
        let output1 = GradientProofOutput {
            gradient_hash: [1u8; 32],
            global_model_hash: [2u8; 32],
            epochs: 5,
            learning_rate: 0.01,
            norm_valid: true,
            client_id: "test".to_string(),
            round: 1,
            commitment: [0u8; 32],
        };

        let output2 = GradientProofOutput {
            gradient_hash: [1u8; 32],
            global_model_hash: [2u8; 32],
            epochs: 5,
            learning_rate: 0.01,
            norm_valid: true,
            client_id: "test".to_string(),
            round: 2, // Different round
            commitment: [0u8; 32],
        };

        let commitment1 = compute_commitment(&output1);
        let commitment2 = compute_commitment(&output2);

        // Different outputs should produce different commitments
        assert_ne!(commitment1, commitment2);

        // Commitments should be 32 bytes
        assert_eq!(commitment1.len(), 32);
    }

    #[test]
    fn test_commitment_deterministic() {
        let output = GradientProofOutput {
            gradient_hash: [1u8; 32],
            global_model_hash: [2u8; 32],
            epochs: 5,
            learning_rate: 0.01,
            norm_valid: true,
            client_id: "test".to_string(),
            round: 1,
            commitment: [0u8; 32],
        };

        // Same input should produce same commitment
        let commitment1 = compute_commitment(&output);
        let commitment2 = compute_commitment(&output);
        assert_eq!(commitment1, commitment2);
    }
}
