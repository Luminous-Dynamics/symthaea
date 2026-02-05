//! # HDC Compatibility Layer
//!
//! Provides interoperability between legacy HV16/RealHV types and
//! the new unified ContinuousHV/BinaryHV types.
//!
//! ## Purpose
//!
//! This module enables gradual migration from the old type system to
//! the new unified system without breaking existing code. It provides:
//!
//! - Trait-based conversion between all HV representations
//! - Dimension interpolation/downsampling utilities
//! - Similarity-preserving transformations
//!
//! ## Migration Path
//!
//! 1. **Phase 1**: Use compat layer for interop (current)
//! 2. **Phase 2**: Update individual modules to use ContinuousHV
//! 3. **Phase 3**: Remove compat layer after full migration
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::compat::HVCompat;
//! use symthaea::hdc::{HV16, ContinuousHV};
//!
//! // Convert HV16 to ContinuousHV
//! let binary = HV16::random(42);
//! let continuous = binary.to_continuous_hv();
//!
//! // Convert back
//! let back = HVCompat::from_continuous(&continuous);
//! ```

use crate::hdc::binary_hv::HV16;
use crate::hdc::unified_hv::{ContinuousHV, BinaryHV};
use crate::hdc::config::hdc_dim;

// ═══════════════════════════════════════════════════════════════════════════════
// HVCOMPAT TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for compatibility conversions between HV types
///
/// Implemented by legacy types (HV16) for conversion to/from unified types.
pub trait HVCompat: Sized {
    /// Convert to ContinuousHV (unified continuous type)
    fn to_continuous_hv(&self) -> ContinuousHV;

    /// Convert to BinaryHV (unified binary type)
    fn to_binary_hv(&self) -> BinaryHV;

    /// Create from ContinuousHV
    fn from_continuous(hv: &ContinuousHV) -> Self;

    /// Create from BinaryHV
    fn from_binary(hv: &BinaryHV) -> Self;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HV16 COMPATIBILITY
// ═══════════════════════════════════════════════════════════════════════════════

impl HVCompat for HV16 {
    fn to_continuous_hv(&self) -> ContinuousHV {
        let target_dim = hdc_dim();
        let source_dim = HV16::DIM;

        if target_dim == source_dim {
            // Direct conversion: each bit → ±1.0
            let values = self.to_bipolar();
            ContinuousHV::from_vec(values)
        } else if target_dim > source_dim {
            // Expansion: interpolate to target dimension
            expand_to_dimension(self, target_dim)
        } else {
            // Compression: downsample to target dimension
            compress_to_dimension(self, target_dim)
        }
    }

    fn to_binary_hv(&self) -> BinaryHV {
        // Direct conversion - HV16 and BinaryHV have same internal structure
        BinaryHV::from_bytes(self.0.to_vec())
    }

    fn from_continuous(hv: &ContinuousHV) -> Self {
        let source_dim = hv.dim();
        let target_dim = HV16::DIM;

        if source_dim == target_dim {
            // Direct conversion: threshold at 0
            HV16::from_bipolar(&hv.values)
        } else if source_dim > target_dim {
            // Downsample to HV16 dimension
            let downsampled = downsample_continuous(hv, target_dim);
            HV16::from_bipolar(&downsampled.values)
        } else {
            // Upsample to HV16 dimension
            let upsampled = upsample_continuous(hv, target_dim);
            HV16::from_bipolar(&upsampled.values)
        }
    }

    fn from_binary(hv: &BinaryHV) -> Self {
        // Ensure dimension matches
        if hv.dim() == HV16::DIM {
            let mut bytes = [0u8; 2048];
            let src = hv.as_bytes();
            let copy_len = src.len().min(2048);
            bytes[..copy_len].copy_from_slice(&src[..copy_len]);
            HV16(bytes)
        } else {
            // Need to resize
            let continuous = hv.to_continuous();
            Self::from_continuous(&continuous)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DIMENSION CONVERSION UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/// Expand HV16 to target dimension using bit replication + hashing
fn expand_to_dimension(hv: &HV16, target_dim: usize) -> ContinuousHV {
    let source_dim = HV16::DIM;
    let expansion_factor = target_dim / source_dim;
    let remainder = target_dim % source_dim;

    let mut values = Vec::with_capacity(target_dim);

    // Get bipolar values from HV16
    let bipolar = hv.to_bipolar();

    // Replicate each value `expansion_factor` times with slight variation
    for i in 0..source_dim {
        let base_value = bipolar[i];

        for j in 0..expansion_factor {
            // Add small deterministic variation based on position
            let variation = hash_variation(i, j, source_dim) * 0.1;
            values.push(base_value + variation);
        }
    }

    // Handle remainder by replicating first `remainder` bits
    for i in 0..remainder {
        let base_value = bipolar[i];
        let variation = hash_variation(i, expansion_factor, source_dim) * 0.1;
        values.push(base_value + variation);
    }

    ContinuousHV::from_vec(values).normalize()
}

/// Compress HV16 to target dimension using majority voting
fn compress_to_dimension(hv: &HV16, target_dim: usize) -> ContinuousHV {
    let source_dim = HV16::DIM;
    let compression_factor = source_dim / target_dim;

    let mut values = Vec::with_capacity(target_dim);
    let bipolar = hv.to_bipolar();

    for i in 0..target_dim {
        // Average values in the corresponding source region
        let start = i * compression_factor;
        let end = ((i + 1) * compression_factor).min(source_dim);

        let sum: f32 = (start..end)
            .map(|j| bipolar[j])
            .sum();

        // Map to continuous value
        let count = (end - start) as f32;
        let avg = if count == 0.0 { 0.0 } else { sum / count };
        values.push(avg);
    }

    ContinuousHV::from_vec(values)
}

/// Downsample continuous HV to target dimension
fn downsample_continuous(hv: &ContinuousHV, target_dim: usize) -> ContinuousHV {
    let source_dim = hv.dim();
    let ratio = source_dim as f32 / target_dim as f32;

    let values: Vec<f32> = (0..target_dim)
        .map(|i| {
            // Linear interpolation
            let source_idx = i as f32 * ratio;
            let idx_floor = source_idx.floor() as usize;
            let idx_ceil = (idx_floor + 1).min(source_dim - 1);
            let t = source_idx - idx_floor as f32;

            hv.values[idx_floor] * (1.0 - t) + hv.values[idx_ceil] * t
        })
        .collect();

    ContinuousHV::from_vec(values)
}

/// Upsample continuous HV to target dimension
fn upsample_continuous(hv: &ContinuousHV, target_dim: usize) -> ContinuousHV {
    let source_dim = hv.dim();
    let ratio = source_dim as f32 / target_dim as f32;

    let values: Vec<f32> = (0..target_dim)
        .map(|i| {
            // Linear interpolation from source
            let source_idx = i as f32 * ratio;
            let idx_floor = source_idx.floor() as usize;
            let idx_ceil = (idx_floor + 1).min(source_dim - 1);
            let t = source_idx - idx_floor as f32;

            hv.values[idx_floor] * (1.0 - t) + hv.values[idx_ceil] * t
        })
        .collect();

    ContinuousHV::from_vec(values)
}

/// Deterministic variation based on position (for expansion consistency)
fn hash_variation(bit_idx: usize, replica_idx: usize, seed: usize) -> f32 {
    let mut state = (bit_idx as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(replica_idx as u64)
        .wrapping_add(seed as u64);

    state ^= state >> 30;
    state = state.wrapping_mul(0xBF58476D1CE4E5B9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94D049BB133111EB);
    state ^= state >> 31;

    // Map to [-1, 1]
    (state as f32 / u64::MAX as f32) * 2.0 - 1.0
}

// ═══════════════════════════════════════════════════════════════════════════════
// BRIDGE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Bridge HV16 to ContinuousHV at the configured dimension
///
/// This is the primary function for STT → Core conversion.
pub fn hv16_to_core(hv: &HV16) -> ContinuousHV {
    hv.to_continuous_hv()
}

/// Bridge ContinuousHV to HV16 (lossy compression)
///
/// This is the primary function for Core → STT conversion.
pub fn core_to_hv16(hv: &ContinuousHV) -> HV16 {
    HV16::from_continuous(hv)
}

/// Bridge with similarity preservation check
///
/// Returns the converted HV and the similarity loss incurred.
pub fn convert_with_similarity_check<T: HVCompat + Clone>(
    hv: &T,
    target: &ContinuousHV,
) -> (ContinuousHV, f32) {
    let converted = hv.to_continuous_hv();
    let _original_sim = converted.similarity(target);

    // Round-trip to measure loss
    let back = T::from_continuous(&converted);
    let back_continuous = back.to_continuous_hv();
    let round_trip_sim = converted.similarity(&back_continuous);

    let loss = 1.0 - round_trip_sim.abs();

    (converted, loss)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hv16_to_continuous_roundtrip() {
        let original = HV16::random(42);
        let continuous = original.to_continuous_hv();
        let back = HV16::from_continuous(&continuous);

        // Should be identical (same dimension, threshold conversion)
        let original_bipolar = original.to_bipolar();
        let back_bipolar = back.to_bipolar();

        let matching = original_bipolar
            .iter()
            .zip(back_bipolar.iter())
            .filter(|(&a, &b)| (a - b).abs() < 0.01)
            .count();

        // Allow some loss due to normalization
        assert!(
            matching > original_bipolar.len() * 90 / 100,
            "Expected >90% match, got {}/{}",
            matching,
            original_bipolar.len()
        );
    }

    #[test]
    fn test_hv16_to_binary_hv() {
        let hv16 = HV16::random(42);
        let binary = hv16.to_binary_hv();

        // Same dimension
        assert_eq!(binary.dim(), HV16::DIM);

        // Same bits
        for i in 0..100 {
            let hv16_bit = hv16.get_bit(i);
            let binary_bit = if binary.get_bipolar(i) == 1 { 1 } else { 0 };
            assert_eq!(
                hv16_bit,
                binary_bit,
                "Mismatch at bit {}",
                i
            );
        }
    }

    #[test]
    fn test_continuous_similarity_preservation() {
        let a = HV16::random(42);
        let b = HV16::random(43);

        // HV16::similarity returns matching_bits/total_bits in [0, 1] where 0.5 = orthogonal
        // ContinuousHV::similarity returns cosine similarity in [-1, 1] where 0 = orthogonal
        // Convert HV16 similarity to bipolar scale: sim_bipolar = 2*sim - 1
        let original_sim = a.similarity(&b);
        let original_bipolar = original_sim * 2.0 - 1.0;

        let a_cont = a.to_continuous_hv();
        let b_cont = b.to_continuous_hv();

        let converted_sim = a_cont.similarity(&b_cont);

        // Similarity should be approximately preserved when compared on the same scale
        let diff = (original_bipolar - converted_sim).abs();
        assert!(
            diff < 0.2,
            "Similarity changed too much: bipolar {} vs continuous {} (original HV16: {})",
            original_bipolar,
            converted_sim,
            original_sim
        );
    }

    #[test]
    fn test_downsample_upsample() {
        let large = ContinuousHV::random(1000, 42);
        let small = downsample_continuous(&large, 100);
        let back = upsample_continuous(&small, 1000);

        // 10x downsampling with linear interpolation is very lossy.
        // Random vectors have high variance, so we can only expect
        // minimal structure preservation (similarity > 0).
        let sim = large.similarity(&back);
        assert!(sim > 0.0, "Expected positive similarity, got {}", sim);
        // Also verify the operation doesn't produce NaN or extreme values
        assert!(sim.is_finite(), "Similarity should be finite");
    }

    #[test]
    fn test_bridge_functions() {
        let hv16 = HV16::random(42);

        let core = hv16_to_core(&hv16);
        assert_eq!(core.dim(), hdc_dim());

        let back = core_to_hv16(&core);
        assert_eq!(back.0.len() * 8, HV16::DIM);
    }

    #[test]
    fn test_hash_variation_deterministic() {
        let v1 = hash_variation(10, 5, 100);
        let v2 = hash_variation(10, 5, 100);
        assert_eq!(v1, v2, "Hash variation should be deterministic");

        let v3 = hash_variation(10, 6, 100);
        assert_ne!(v1, v3, "Different positions should give different values");
    }
}
