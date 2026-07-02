// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gradient Proof Core Types
//!
//! This crate provides shared type definitions for gradient validity proofs
//! that are used by both the host (prover) and guest (zkVM) programs.
//!
//! All types are designed to be serializable and compatible with RISC Zero's
//! zkVM environment.

#![cfg_attr(not(feature = "std"), no_std)]

use serde::{Deserialize, Serialize};

// =============================================================================
// Constants
// =============================================================================

/// Q16.16 fixed-point scale factor
pub const FIXED_SCALE: i64 = 65536;

/// Maximum allowed L2 norm squared (default)
/// Approximately 1000.0 in float terms
pub const DEFAULT_MAX_NORM_SQUARED: i64 = 1_000_000 * FIXED_SCALE;

/// Minimum allowed L2 norm squared (to prevent zero gradients)
pub const MIN_NORM_SQUARED: i64 = 1;

/// Reserved sentinel value for invalid (max)
pub const FIXED_INVALID_MAX: i32 = i32::MAX;

/// Reserved sentinel value for invalid (min)
pub const FIXED_INVALID_MIN: i32 = i32::MIN;

// =============================================================================
// Public Input Types
// =============================================================================

/// Public inputs for the gradient validity proof
///
/// These values are known to both the prover and verifier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GradientProofPublicInputs {
    /// Node identifier (32-byte hash of public key)
    pub node_id: [u8; 32],

    /// Training round number (for replay protection)
    pub round_number: u64,

    /// Commitment to the gradient (SHA256 hash)
    pub gradient_commitment: [u8; 32],

    /// Hash of the global model being updated
    pub model_hash: [u8; 32],

    /// Maximum allowed L2 norm squared (0 = use default)
    pub max_norm_squared: i64,

    /// Number of gradient elements
    pub gradient_len: u32,
}

// =============================================================================
// Journal Output Types
// =============================================================================

/// Public outputs committed to the proof journal
///
/// These values are cryptographically bound to the proof and
/// can be verified by anyone.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GradientProofJournal {
    /// Node that generated this gradient
    pub node_id: [u8; 32],

    /// Round number (replay protection)
    pub round_number: u64,

    /// Verified gradient hash (matches commitment)
    pub gradient_hash: [u8; 32],

    /// Model hash
    pub model_hash: [u8; 32],

    /// Number of gradient elements
    pub gradient_len: u32,

    /// L2 norm squared (in Q16.16 * 2^16 = Q32.32 for squared)
    pub norm_squared: i64,

    /// Mean gradient value (Q16.16)
    pub mean: i32,

    /// Gradient variance (Q16.16)
    pub variance: i32,

    /// Validity flag (always 1 if proof verified)
    pub is_valid: u8,
}

impl GradientProofJournal {
    /// Size of the serialized journal in bytes
    pub const SERIALIZED_SIZE: usize = 32 + 8 + 32 + 32 + 4 + 8 + 4 + 4 + 1; // 125 bytes

    /// Check if the gradient proof indicates validity
    pub fn is_valid(&self) -> bool {
        self.is_valid == 1
    }

    /// Get the L2 norm as a float (approximate)
    pub fn norm_f32(&self) -> f32 {
        ((self.norm_squared as f64) / (FIXED_SCALE as f64)).sqrt() as f32
    }

    /// Get the mean as a float
    pub fn mean_f32(&self) -> f32 {
        self.mean as f32 / FIXED_SCALE as f32
    }

    /// Get the variance as a float
    pub fn variance_f32(&self) -> f32 {
        self.variance as f32 / FIXED_SCALE as f32
    }
}

// =============================================================================
// Gradient Statistics
// =============================================================================

/// Statistical summary of a gradient vector
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GradientStats {
    /// Number of elements
    pub len: u32,

    /// L2 norm squared (Q32.32 for squared values)
    pub norm_squared: i64,

    /// Mean value (Q16.16)
    pub mean: i32,

    /// Variance (Q16.16)
    pub variance: i32,

    /// Minimum element value (Q16.16)
    pub min: i32,

    /// Maximum element value (Q16.16)
    pub max: i32,
}

impl GradientStats {
    /// Compute statistics for a gradient vector
    pub fn compute(gradient: &[i32]) -> Self {
        if gradient.is_empty() {
            return Self {
                len: 0,
                norm_squared: 0,
                mean: 0,
                variance: 0,
                min: 0,
                max: 0,
            };
        }

        let len = gradient.len() as u32;

        // Compute norm squared
        let mut norm_squared: i64 = 0;
        for &g in gradient {
            let g_squared = (g as i64 * g as i64) >> 16;
            norm_squared = norm_squared.saturating_add(g_squared);
        }

        // Compute mean
        let sum: i64 = gradient.iter().map(|&g| g as i64).sum();
        let mean = (sum / len as i64) as i32;

        // Compute variance
        let mut variance_sum: i64 = 0;
        for &g in gradient {
            let diff = (g as i64) - (mean as i64);
            variance_sum += (diff * diff) >> 16;
        }
        let variance = (variance_sum / len as i64) as i32;

        // Find min/max
        let min = *gradient.iter().min().unwrap_or(&0);
        let max = *gradient.iter().max().unwrap_or(&0);

        Self {
            len,
            norm_squared,
            mean,
            variance,
            min,
            max,
        }
    }
}

// =============================================================================
// Validation Results
// =============================================================================

/// Result of gradient validation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValidationResult {
    /// Gradient is valid
    Valid,

    /// Gradient has invalid values (NaN/Inf equivalents)
    InvalidValues { index: u32, value: i32 },

    /// Gradient norm exceeds maximum
    NormTooLarge { actual: i64, max: i64 },

    /// Gradient norm is too small (zero or near-zero)
    NormTooSmall { actual: i64, min: i64 },

    /// Gradient commitment mismatch
    CommitmentMismatch,
}

impl ValidationResult {
    /// Check if the result indicates validity
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidationResult::Valid)
    }
}

// =============================================================================
// Fixed-Point Utilities
// =============================================================================

/// Convert f32 to Q16.16 fixed-point
#[inline]
pub fn f32_to_fixed(f: f32) -> i32 {
    (f * FIXED_SCALE as f32) as i32
}

/// Convert Q16.16 fixed-point to f32
#[inline]
pub fn fixed_to_f32(fixed: i32) -> f32 {
    fixed as f32 / FIXED_SCALE as f32
}

/// Convert f64 to Q16.16 fixed-point
#[inline]
pub fn f64_to_fixed(f: f64) -> i32 {
    (f * FIXED_SCALE as f64) as i32
}

/// Convert Q16.16 fixed-point to f64
#[inline]
pub fn fixed_to_f64(fixed: i32) -> f64 {
    fixed as f64 / FIXED_SCALE as f64
}

/// Multiply two Q16.16 values
#[inline]
pub fn fixed_mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 16) as i32
}

/// Divide two Q16.16 values
#[inline]
pub fn fixed_div(a: i32, b: i32) -> i32 {
    (((a as i64) << 16) / (b as i64)) as i32
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_conversion() {
        let values = [0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 0.001];
        for &v in &values {
            let fixed = f32_to_fixed(v);
            let back = fixed_to_f32(fixed);
            assert!((v - back).abs() < 0.001, "Failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_fixed_mul() {
        let a = f32_to_fixed(2.0);
        let b = f32_to_fixed(3.0);
        let result = fixed_mul(a, b);
        assert!((fixed_to_f32(result) - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_fixed_div() {
        let a = f32_to_fixed(6.0);
        let b = f32_to_fixed(2.0);
        let result = fixed_div(a, b);
        assert!((fixed_to_f32(result) - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_gradient_stats() {
        let gradient = vec![
            f32_to_fixed(1.0),
            f32_to_fixed(-1.0),
            f32_to_fixed(0.5),
            f32_to_fixed(-0.5),
        ];

        let stats = GradientStats::compute(&gradient);

        assert_eq!(stats.len, 4);
        assert!((fixed_to_f32(stats.mean) - 0.0).abs() < 0.01);
        assert!(stats.norm_squared > 0);
    }

    #[test]
    fn test_gradient_stats_empty() {
        let gradient: Vec<i32> = vec![];
        let stats = GradientStats::compute(&gradient);

        assert_eq!(stats.len, 0);
        assert_eq!(stats.norm_squared, 0);
        assert_eq!(stats.mean, 0);
    }

    #[test]
    fn test_journal_conversions() {
        let journal = GradientProofJournal {
            node_id: [0u8; 32],
            round_number: 1,
            gradient_hash: [0u8; 32],
            model_hash: [0u8; 32],
            gradient_len: 100,
            norm_squared: FIXED_SCALE * 100, // ~10.0 norm
            mean: f32_to_fixed(0.5),
            variance: f32_to_fixed(0.1),
            is_valid: 1,
        };

        assert!(journal.is_valid());
        assert!((journal.mean_f32() - 0.5).abs() < 0.001);
        assert!((journal.variance_f32() - 0.1).abs() < 0.001);
    }
}
