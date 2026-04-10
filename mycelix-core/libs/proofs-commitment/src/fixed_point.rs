// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fixed-point commitment scheme for floating-point values
//!
//! This module provides commitments for floating-point values using fixed-point
//! scaling, which is essential for deterministic ZK proofs where floating-point
//! precision issues could cause verification failures.

use sha3::{Digest, Sha3_256};

use crate::scheme::constant_time_eq;
use crate::{CommitmentError, CommitmentHash, CommitmentResult};

/// Standard K-Vector scale factor (4 decimal places)
pub const KVECTOR_SCALE_FACTOR: u64 = 10_000;

/// Configuration for fixed-point scaling
#[derive(Clone, Debug)]
pub struct ScalingConfig {
    /// Scale factor for converting float to integer
    pub scale_factor: u64,
    /// Minimum valid value (before scaling)
    pub min_value: f32,
    /// Maximum valid value (before scaling)
    pub max_value: f32,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            scale_factor: KVECTOR_SCALE_FACTOR,
            min_value: 0.0,
            max_value: 1.0,
        }
    }
}

impl ScalingConfig {
    /// Create a new scaling configuration
    pub fn new(scale_factor: u64, min_value: f32, max_value: f32) -> Self {
        Self {
            scale_factor,
            min_value,
            max_value,
        }
    }

    /// Scale a float value to fixed-point integer
    ///
    /// Clamps to valid range and rounds to nearest integer.
    pub fn scale(&self, value: f32) -> u64 {
        let clamped = value.clamp(self.min_value, self.max_value);
        let scaled = (clamped * self.scale_factor as f32).round() as u64;
        scaled.min(self.scale_factor)
    }

    /// Scale with strict validation (no clamping)
    pub fn scale_strict(&self, value: f32) -> CommitmentResult<u64> {
        if !value.is_finite() {
            return Err(CommitmentError::NonFiniteValue(format!("{}", value)));
        }

        if value < self.min_value || value > self.max_value {
            return Err(CommitmentError::ValueOutOfRange {
                value: value as f64,
                min: self.min_value as f64,
                max: self.max_value as f64,
            });
        }

        let scaled = (value * self.scale_factor as f32).round() as u64;
        if scaled > self.scale_factor {
            return Err(CommitmentError::ScalingOverflow {
                value: value as f64,
                scale: self.scale_factor,
            });
        }

        Ok(scaled)
    }

    /// Unscale a fixed-point integer back to float
    pub fn unscale(&self, scaled: u64) -> f32 {
        scaled as f32 / self.scale_factor as f32
    }
}

/// Fixed-point commitment scheme for floating-point values
///
/// Converts floats to fixed-point integers before hashing to ensure
/// deterministic commitments across platforms.
///
/// # Example
///
/// ```
/// use proofs_commitment::FixedPointCommitment;
///
/// let commitment = FixedPointCommitment::default();
///
/// // Commit to K-Vector values
/// let values = [0.8f32, 0.7, 0.9, 0.6, 0.5, 0.4, 0.85, 0.75];
/// let hash = commitment.commit_f32_slice(&values);
///
/// // Verify
/// assert!(commitment.verify_f32_slice(&values, &hash));
/// ```
#[derive(Clone, Debug, Default)]
pub struct FixedPointCommitment {
    config: ScalingConfig,
}

impl FixedPointCommitment {
    /// Create with default K-Vector scaling (0-1, 4 decimals)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom scaling configuration
    pub fn with_config(config: ScalingConfig) -> Self {
        Self { config }
    }

    /// Create for K-Vector commitments (8 components, 0-1 range)
    pub fn kvector() -> Self {
        Self::default()
    }

    /// Create for gradient commitments (arbitrary range)
    pub fn gradient(max_gradient: f32) -> Self {
        Self::with_config(ScalingConfig::new(
            1000, // 3 decimal places for gradients
            -max_gradient,
            max_gradient,
        ))
    }

    /// Get the scaling configuration
    pub fn config(&self) -> &ScalingConfig {
        &self.config
    }

    /// Commit to a slice of f32 values
    pub fn commit_f32_slice(&self, values: &[f32]) -> CommitmentHash {
        let mut hasher = Sha3_256::new();

        for &value in values {
            let scaled = self.config.scale(value);
            hasher.update(scaled.to_le_bytes());
        }

        hasher.finalize().into()
    }

    /// Commit to a slice of f32 values with strict validation
    pub fn commit_f32_slice_strict(&self, values: &[f32]) -> CommitmentResult<CommitmentHash> {
        let mut hasher = Sha3_256::new();

        for &value in values {
            let scaled = self.config.scale_strict(value)?;
            hasher.update(scaled.to_le_bytes());
        }

        Ok(hasher.finalize().into())
    }

    /// Commit to pre-scaled u64 values
    pub fn commit_scaled_slice(&self, values: &[u64]) -> CommitmentHash {
        let mut hasher = Sha3_256::new();

        for &value in values {
            hasher.update(value.to_le_bytes());
        }

        hasher.finalize().into()
    }

    /// Verify a commitment against f32 values
    pub fn verify_f32_slice(&self, values: &[f32], commitment: &CommitmentHash) -> bool {
        let computed = self.commit_f32_slice(values);
        constant_time_eq(&computed, commitment)
    }

    /// Verify a commitment against pre-scaled values
    pub fn verify_scaled_slice(&self, values: &[u64], commitment: &CommitmentHash) -> bool {
        let computed = self.commit_scaled_slice(values);
        constant_time_eq(&computed, commitment)
    }

    /// Scale values and return both scaled values and commitment
    ///
    /// Useful when you need both for ZK proof generation.
    pub fn prepare_for_proof(&self, values: &[f32]) -> (Vec<u64>, CommitmentHash) {
        let scaled: Vec<u64> = values.iter().map(|&v| self.config.scale(v)).collect();
        let commitment = self.commit_scaled_slice(&scaled);
        (scaled, commitment)
    }

    /// Scale values with strict validation
    pub fn prepare_for_proof_strict(
        &self,
        values: &[f32],
    ) -> CommitmentResult<(Vec<u64>, CommitmentHash)> {
        let scaled: Result<Vec<u64>, _> = values
            .iter()
            .map(|&v| self.config.scale_strict(v))
            .collect();
        let scaled = scaled?;
        let commitment = self.commit_scaled_slice(&scaled);
        Ok((scaled, commitment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_config_default() {
        let config = ScalingConfig::default();
        assert_eq!(config.scale_factor, KVECTOR_SCALE_FACTOR);
        assert_eq!(config.min_value, 0.0);
        assert_eq!(config.max_value, 1.0);
    }

    #[test]
    fn test_scale_values() {
        let config = ScalingConfig::default();

        assert_eq!(config.scale(0.0), 0);
        assert_eq!(config.scale(0.5), 5000);
        assert_eq!(config.scale(1.0), 10000);
        assert_eq!(config.scale(0.1234), 1234);
    }

    #[test]
    fn test_scale_clamps() {
        let config = ScalingConfig::default();

        // Below min
        assert_eq!(config.scale(-0.1), 0);
        // Above max
        assert_eq!(config.scale(1.1), 10000);
        // NaN clamps to 0
        assert_eq!(config.scale(f32::NAN), 0);
    }

    #[test]
    fn test_scale_strict_rejects_invalid() {
        let config = ScalingConfig::default();

        // Out of range
        assert!(config.scale_strict(-0.1).is_err());
        assert!(config.scale_strict(1.1).is_err());

        // Non-finite
        assert!(config.scale_strict(f32::NAN).is_err());
        assert!(config.scale_strict(f32::INFINITY).is_err());
    }

    #[test]
    fn test_round_trip() {
        let config = ScalingConfig::default();

        let original = 0.7531f32;
        let scaled = config.scale(original);
        let recovered = config.unscale(scaled);

        assert!((original - recovered).abs() < 0.0001);
    }

    #[test]
    fn test_commitment_deterministic() {
        let commitment = FixedPointCommitment::default();
        let values = [0.8f32, 0.7, 0.9, 0.6, 0.5, 0.4, 0.85, 0.75];

        let hash1 = commitment.commit_f32_slice(&values);
        let hash2 = commitment.commit_f32_slice(&values);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_commitment_sensitive_to_changes() {
        let commitment = FixedPointCommitment::default();

        let values1 = [0.8f32, 0.7, 0.9, 0.6, 0.5, 0.4, 0.85, 0.75];
        let values2 = [0.81f32, 0.7, 0.9, 0.6, 0.5, 0.4, 0.85, 0.75];

        let hash1 = commitment.commit_f32_slice(&values1);
        let hash2 = commitment.commit_f32_slice(&values2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_prepare_for_proof() {
        let commitment = FixedPointCommitment::default();
        let values = [0.8f32, 0.7, 0.9, 0.6, 0.5, 0.4, 0.85, 0.75];

        let (scaled, hash) = commitment.prepare_for_proof(&values);

        assert_eq!(scaled.len(), 8);
        assert_eq!(scaled[0], 8000);
        assert_eq!(scaled[1], 7000);

        // Verify commitment matches
        assert!(commitment.verify_scaled_slice(&scaled, &hash));
    }

    #[test]
    fn test_gradient_commitment() {
        let commitment = FixedPointCommitment::gradient(1000.0);

        // Should handle negative values
        let gradients = [123.456f32, -789.012, 0.0, 500.5];
        let hash = commitment.commit_f32_slice(&gradients);

        assert!(commitment.verify_f32_slice(&gradients, &hash));
    }

    #[test]
    fn test_cross_platform_consistency() {
        // These values should produce the same commitment on any platform
        let commitment = FixedPointCommitment::default();
        let values = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let hash = commitment.commit_f32_slice(&values);

        // The scaled values are: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000
        // This should produce a deterministic hash
        let expected_scaled = [1000u64, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let expected_hash = commitment.commit_scaled_slice(&expected_scaled);

        assert_eq!(hash, expected_hash);
    }
}
