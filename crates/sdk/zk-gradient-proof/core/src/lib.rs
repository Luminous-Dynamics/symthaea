// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ZK Gradient Proof Core Types
//!
//! Shared types for gradient quality proofs in federated learning.
//! Used by both the guest (zkVM) and host (prover/verifier).

use serde::{Deserialize, Serialize};

/// Input to the gradient proof circuit (private witness)
///
/// This data is known only to the prover and is never revealed.
/// The gradient values remain private while proving their quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientProofInput {
    /// The computed gradient vector (PRIVATE - never leaves zkVM)
    pub gradient: Vec<f32>,
    /// Hash of the global model this gradient was computed against
    pub global_model_hash: [u8; 32],
    /// Number of local training epochs
    pub epochs: u32,
    /// Learning rate used for training
    pub learning_rate: f32,
    /// Client identifier for attribution
    pub client_id: String,
    /// Training round number
    pub round: u32,
    /// Constraint parameters
    pub constraints: GradientConstraints,
}

/// Output from the gradient proof circuit (public journal)
///
/// This data is committed to the proof and visible to verifiers.
/// It proves gradient quality without revealing gradient values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GradientProofOutput {
    /// Hash commitment of the gradient (SHA3-256)
    pub gradient_hash: [u8; 32],
    /// Hash of the global model used
    pub global_model_hash: [u8; 32],
    /// Training epochs
    pub epochs: u32,
    /// Learning rate
    pub learning_rate: f32,
    /// Whether the gradient passes all quality constraints
    pub quality_valid: bool,
    /// Client identifier
    pub client_id: String,
    /// Training round
    pub round: u32,
    /// Gradient dimension (public metadata)
    pub dimension: usize,
    /// Commitment binding all public inputs
    pub commitment: u64,
}

impl GradientProofOutput {
    /// Check if the proof indicates valid gradient quality
    pub fn is_valid(&self) -> bool {
        self.quality_valid
    }
}

/// Gradient quality constraints verified inside the zkVM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConstraints {
    /// Minimum acceptable L2 norm (prevents zero gradients)
    pub min_norm: f32,
    /// Maximum acceptable L2 norm (prevents gradient explosion)
    pub max_norm: f32,
    /// Maximum allowed magnitude per element (clipping bound)
    pub max_element_magnitude: f32,
    /// Minimum required gradient dimension
    pub min_dimension: usize,
}

impl Default for GradientConstraints {
    fn default() -> Self {
        Self {
            min_norm: 0.001,
            max_norm: 100.0,
            max_element_magnitude: 10.0,
            min_dimension: 10,
        }
    }
}

impl GradientConstraints {
    /// Constraints for federated learning scenarios
    pub fn for_federated_learning() -> Self {
        Self {
            min_norm: 0.0001,
            max_norm: 50.0,
            max_element_magnitude: 5.0,
            min_dimension: 100,
        }
    }

    /// Strict constraints for high-stakes scenarios
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
/// This is the core verification logic that runs inside the zkVM guest.
/// Returns (valid, l2_norm, max_element).
pub fn verify_gradient_quality(
    gradient: &[f32],
    constraints: &GradientConstraints,
) -> (bool, f32, f32) {
    // Check dimension
    if gradient.len() < constraints.min_dimension {
        return (false, 0.0, 0.0);
    }

    // Calculate L2 norm
    let l2_norm: f32 = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();

    // Find max element magnitude
    let max_element: f32 = gradient.iter()
        .map(|g| g.abs())
        .fold(0.0f32, |a, b| if a > b { a } else { b });

    // Validate constraints
    let norm_valid = l2_norm >= constraints.min_norm && l2_norm <= constraints.max_norm;
    let element_valid = max_element <= constraints.max_element_magnitude;
    let finite_valid = l2_norm.is_finite() && gradient.iter().all(|g| g.is_finite());

    (norm_valid && element_valid && finite_valid, l2_norm, max_element)
}

/// Compute FNV-1a hash for gradient commitment
///
/// Creates a deterministic commitment from gradient data.
/// Uses FNV-1a for zkVM compatibility (no std crypto needed).
pub fn compute_gradient_hash(gradient: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    let prime: u64 = 0x100000001b3;         // FNV-1a prime

    for g in gradient {
        for byte in g.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }
    }
    hash
}

/// Compute commitment over proof output
///
/// Binds the proof to specific parameters, preventing proof reuse.
pub fn compute_commitment(
    gradient_hash: u64,
    global_model_hash: &[u8; 32],
    epochs: u32,
    round: u32,
    client_id: &str,
) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    let prime: u64 = 0x100000001b3;

    // Hash gradient commitment
    for byte in gradient_hash.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    // Hash model hash
    for byte in global_model_hash {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    // Hash epochs
    for byte in epochs.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    // Hash round
    for byte in round.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    // Hash client ID
    for byte in client_id.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    hash
}

/// Convert u64 hash to [u8; 32] for compatibility
pub fn hash_to_bytes32(hash: u64) -> [u8; 32] {
    let mut result = [0u8; 32];
    result[..8].copy_from_slice(&hash.to_le_bytes());
    // Fill remaining bytes with hash rotations for better distribution
    let h2 = hash.rotate_left(13);
    let h3 = hash.rotate_left(29);
    let h4 = hash.rotate_left(41);
    result[8..16].copy_from_slice(&h2.to_le_bytes());
    result[16..24].copy_from_slice(&h3.to_le_bytes());
    result[24..32].copy_from_slice(&h4.to_le_bytes());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_gradient_quality_valid() {
        let gradient: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let constraints = GradientConstraints::default();

        let (valid, norm, _max) = verify_gradient_quality(&gradient, &constraints);

        assert!(valid);
        assert!(norm > 0.0);
    }

    #[test]
    fn test_verify_gradient_quality_too_small() {
        let gradient: Vec<f32> = vec![0.0; 100];
        let constraints = GradientConstraints::default();

        let (valid, _, _) = verify_gradient_quality(&gradient, &constraints);

        assert!(!valid);
    }

    #[test]
    fn test_gradient_hash_deterministic() {
        let gradient: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let hash1 = compute_gradient_hash(&gradient);
        let hash2 = compute_gradient_hash(&gradient);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_gradient_hash_different() {
        let gradient1: Vec<f32> = vec![0.1, 0.2, 0.3];
        let gradient2: Vec<f32> = vec![0.1, 0.2, 0.4];

        let hash1 = compute_gradient_hash(&gradient1);
        let hash2 = compute_gradient_hash(&gradient2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_commitment_binds_parameters() {
        let gradient_hash = compute_gradient_hash(&vec![0.1, 0.2, 0.3]);
        let model_hash = [1u8; 32];

        let c1 = compute_commitment(gradient_hash, &model_hash, 5, 1, "client1");
        let c2 = compute_commitment(gradient_hash, &model_hash, 5, 1, "client2");
        let c3 = compute_commitment(gradient_hash, &model_hash, 5, 2, "client1");

        assert_ne!(c1, c2); // Different client
        assert_ne!(c1, c3); // Different round
    }
}
