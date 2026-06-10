// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC encoder for isotopic signatures into 16,384-dimensional continuous hypervectors.

use crate::isotope::IsotopicSignature;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

const SEEDS: [u64; 9] = [
    0xACE_0001, 0xACE_0002, 0xACE_0003, 0xACE_0004, 0xACE_0005, 0xACE_0006, 0xACE_0007, 0xACE_0008,
    0xACE_0009,
];

/// Encodes isotopic signatures into 16,384D continuous hypervectors.
///
/// Uses 9 orthogonal basis vectors (one per isotope ratio) with
/// weighted superposition. Signatures from similar nuclear sources
/// produce similar hypervectors, enabling HDC attribution.
pub struct IsotopicHdcEncoder {
    bases: [ContinuousHV; 9],
}

impl IsotopicHdcEncoder {
    /// Create a new encoder with deterministic basis vectors.
    pub fn new() -> Self {
        Self {
            bases: std::array::from_fn(|i| ContinuousHV::random(HDC_DIMENSION, SEEDS[i])),
        }
    }

    /// Encode a signature's 9 normalized isotope ratios into a 16,384D hypervector.
    pub fn encode(&self, sig: &IsotopicSignature) -> ContinuousHV {
        let weights = sig.normalized_values();
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }

    /// Cosine similarity between two isotopic signatures in HDC space.
    pub fn similarity(&self, a: &IsotopicSignature, b: &IsotopicSignature) -> f32 {
        self.encode(a).similarity(&self.encode(b))
    }

    /// Compute pairwise similarity matrix for a set of signatures.
    pub fn similarity_matrix(&self, signatures: &[IsotopicSignature]) -> Vec<Vec<f32>> {
        let hvs: Vec<ContinuousHV> = signatures.iter().map(|s| self.encode(s)).collect();
        hvs.iter()
            .map(|a| hvs.iter().map(|b| a.similarity(b)).collect())
            .collect()
    }
}

impl Default for IsotopicHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_dimension() {
        assert_eq!(
            IsotopicHdcEncoder::new()
                .encode(&IsotopicSignature::highly_enriched_uranium())
                .dim(),
            HDC_DIMENSION
        );
    }
    #[test]
    fn test_encode_nonzero_norm() {
        assert!(
            IsotopicHdcEncoder::new()
                .encode(&IsotopicSignature::highly_enriched_uranium())
                .norm()
                > 0.0
        );
    }

    #[test]
    fn test_self_similarity() {
        let enc = IsotopicHdcEncoder::new();
        let s = enc.similarity(
            &IsotopicSignature::highly_enriched_uranium(),
            &IsotopicSignature::highly_enriched_uranium(),
        );
        assert!(s > 0.99, "Self-similarity should be ~1.0, got {}", s);
    }

    #[test]
    fn test_heu_vs_natural_different() {
        let enc = IsotopicHdcEncoder::new();
        let s = enc.similarity(
            &IsotopicSignature::highly_enriched_uranium(),
            &IsotopicSignature::natural_uranium(),
        );
        assert!(
            s < 0.999,
            "HEU vs Natural should be distinguishable, got {}",
            s
        );
    }

    #[test]
    fn test_perturbed_high_similarity() {
        let enc = IsotopicHdcEncoder::new();
        let heu = IsotopicSignature::highly_enriched_uranium();
        let p = heu.perturbed(0.05, 42);
        let s = enc.similarity(&heu, &p);
        assert!(s > 0.9, "Perturbed similarity should be high, got {}", s);
    }

    #[test]
    fn test_similarity_matrix_size() {
        let m = IsotopicHdcEncoder::new().similarity_matrix(&IsotopicSignature::references());
        assert_eq!(m.len(), 5);
        assert_eq!(m[0].len(), 5);
    }

    #[test]
    fn test_similarity_matrix_symmetric() {
        let m = IsotopicHdcEncoder::new().similarity_matrix(&IsotopicSignature::references());
        for i in 0..5 {
            for j in 0..5 {
                assert!((m[i][j] - m[j][i]).abs() < 1e-6);
            }
        }
    }
}
