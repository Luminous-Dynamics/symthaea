// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HV Bridge — BinaryHV ↔ ContinuousHV Conversion
//!
//! Bridges the two hypervector types used across Symthaea's code generation subsystems:
//!
//! - **BinaryHV** (16,384 bits, 2KB): Used by Geodesic synthesis, topology encoding,
//!   program memory, and binding-reversibility operations. Fast XOR binding, Copy-able.
//!
//! - **ContinuousHV** (16,384 × f32, 64KB): Used by CodeGenerator, CodeAlgebra,
//!   Broca language generation, and CfC neural dynamics. Supports gradients and learning.
//!
//! This bridge enables the unified Code Orchestrator to route requests across backends
//! that use different HV representations.
//!
//! # Conversions
//!
//! ```text
//! ContinuousHV → sign(values) → BinaryHV     (threshold at 0.0: ≥0 → 1, <0 → 0)
//! BinaryHV → bipolar(bits) → ContinuousHV     (0 → -1.0, 1 → +1.0)
//! ```
//!
//! Cross-similarity computes similarity without materializing the full conversion.

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::ContinuousHV;

/// Bridge between BinaryHV and ContinuousHV representations.
///
/// Provides efficient bidirectional conversion and cross-type similarity
/// for the unified code synthesis pipeline.
pub struct HvBridge;

impl HvBridge {
    /// Convert ContinuousHV to BinaryHV using sign thresholding.
    ///
    /// Each dimension: `value >= 0.0 → bit 1 (+1)`, `value < 0.0 → bit 0 (-1)`.
    /// This is the standard HDC quantization (Kanerva 2009).
    #[inline]
    pub fn continuous_to_binary(chv: &ContinuousHV) -> BinaryHV {
        chv.to_binary(0.0)
    }

    /// Convert BinaryHV to ContinuousHV using bipolar expansion.
    ///
    /// Each bit: `1 → +1.0`, `0 → -1.0`.
    /// The result is a unit-norm vector in bipolar space.
    #[inline]
    pub fn binary_to_continuous(bhv: &BinaryHV) -> ContinuousHV {
        bhv.to_continuous()
    }

    /// Compute similarity between a ContinuousHV and a BinaryHV without
    /// allocating a full intermediate vector.
    ///
    /// Converts BinaryHV to bipolar on-the-fly and computes cosine similarity
    /// against the ContinuousHV. This is more efficient than materializing
    /// the full conversion when you only need the similarity score.
    pub fn cross_similarity(chv: &ContinuousHV, bhv: &BinaryHV) -> f32 {
        let dim = chv.values.len().min(BinaryHV::DIM);

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        // BinaryHV bipolar values are ±1.0 so norm_b = sqrt(dim)
        let norm_b = (dim as f32).sqrt();

        for i in 0..dim {
            let a = chv.values[i];
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bipolar = if (bhv.0[byte_idx] >> bit_idx) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            };
            dot += a * bipolar;
            norm_a += a * a;
        }

        let denominator = norm_a.sqrt() * norm_b;
        if denominator > 0.0 {
            dot / denominator
        } else {
            0.0
        }
    }

    /// Compute similarity between two HVs of the same type.
    ///
    /// Convenience method that dispatches to the native similarity for each type.
    pub fn similarity_continuous(a: &ContinuousHV, b: &ContinuousHV) -> f32 {
        a.similarity(b)
    }

    /// Compute Hamming-based similarity between two BinaryHVs.
    pub fn similarity_binary(a: &BinaryHV, b: &BinaryHV) -> f32 {
        a.similarity(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_to_continuous_roundtrip() {
        let bhv = BinaryHV::random(42);
        let chv = HvBridge::binary_to_continuous(&bhv);

        // Should be 16,384 dimensions
        assert_eq!(chv.values.len(), BinaryHV::DIM);

        // All values should be ±1.0
        for &v in &chv.values {
            assert!(v == 1.0 || v == -1.0, "Expected ±1.0, got {v}");
        }

        // Round-trip: continuous → binary should recover original
        let bhv_recovered = HvBridge::continuous_to_binary(&chv);
        assert_eq!(
            bhv, bhv_recovered,
            "Round-trip should be lossless for bipolar vectors"
        );
    }

    #[test]
    fn test_continuous_to_binary_thresholding() {
        // Create a ContinuousHV with known values
        let mut values = vec![-1.0f32; BinaryHV::DIM];
        // Set first 100 dimensions to positive
        for v in values.iter_mut().take(100) {
            *v = 1.0;
        }
        let chv = ContinuousHV::from_values(values);
        let bhv = HvBridge::continuous_to_binary(&chv);

        // First 100 bits should be 1, rest 0
        for i in 0..100 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            assert_eq!(
                (bhv.0[byte_idx] >> bit_idx) & 1,
                1,
                "Bit {i} should be 1 (positive value)"
            );
        }
        // Check a sample bit beyond 100
        let byte_idx = 200 / 8;
        let bit_idx = 200 % 8;
        assert_eq!(
            (bhv.0[byte_idx] >> bit_idx) & 1,
            0,
            "Bit 200 should be 0 (negative value)"
        );
    }

    #[test]
    fn test_cross_similarity_identical() {
        let bhv = BinaryHV::random(123);
        let chv = HvBridge::binary_to_continuous(&bhv);

        let sim = HvBridge::cross_similarity(&chv, &bhv);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical vectors should have similarity ~1.0, got {sim}"
        );
    }

    #[test]
    fn test_cross_similarity_orthogonal() {
        let bhv1 = BinaryHV::random(42);
        let bhv2 = BinaryHV::random(43);

        // Different random vectors should have ~0 similarity
        let chv1 = HvBridge::binary_to_continuous(&bhv1);
        let sim = HvBridge::cross_similarity(&chv1, &bhv2);
        assert!(
            sim.abs() < 0.1,
            "Unrelated vectors should have near-zero similarity, got {sim}"
        );
    }

    #[test]
    fn test_cross_similarity_consistency() {
        // Cross-similarity should match within-type similarity after conversion
        let bhv = BinaryHV::random(99);
        let chv_a = ContinuousHV::random(BinaryHV::DIM, 77);
        let chv_b = HvBridge::binary_to_continuous(&bhv);

        let cross_sim = HvBridge::cross_similarity(&chv_a, &bhv);
        let native_sim = chv_a.similarity(&chv_b);

        assert!(
            (cross_sim - native_sim).abs() < 1e-4,
            "Cross sim ({cross_sim}) should match native sim ({native_sim})"
        );
    }
}
