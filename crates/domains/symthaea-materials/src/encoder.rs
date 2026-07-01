// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC encoder for material properties into 16,384-dimensional continuous hypervectors.

use crate::properties::MaterialProperty;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

const SEEDS: [u64; 8] = [
    0xCA7_A001, 0xCA7_A002, 0xCA7_A003, 0xCA7_A004, 0xCA7_A005, 0xCA7_A006, 0xCA7_A007, 0xCA7_A008,
];

/// Encodes material properties into 16,384D continuous hypervectors.
///
/// Uses 8 orthogonal basis vectors (one per property dimension) with
/// weighted superposition. Materials with similar properties produce
/// similar hypervectors, enabling HDC similarity search.
pub struct MaterialHdcEncoder {
    bases: [ContinuousHV; 8],
}

impl MaterialHdcEncoder {
    /// Create a new encoder with deterministic basis vectors.
    pub fn new() -> Self {
        Self {
            bases: std::array::from_fn(|i| ContinuousHV::random(HDC_DIMENSION, SEEDS[i])),
        }
    }

    /// Encode a material's 8 normalized properties into a 16,384D hypervector.
    pub fn encode(&self, material: &MaterialProperty) -> ContinuousHV {
        let weights = material.normalized_values();
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }

    /// Cosine similarity between two materials in HDC space.
    pub fn similarity(&self, a: &MaterialProperty, b: &MaterialProperty) -> f32 {
        self.encode(a).similarity(&self.encode(b))
    }

    /// Compute pairwise similarity matrix for a set of materials.
    pub fn similarity_matrix(&self, materials: &[MaterialProperty]) -> Vec<Vec<f32>> {
        let hvs: Vec<ContinuousHV> = materials.iter().map(|m| self.encode(m)).collect();
        hvs.iter()
            .map(|a| hvs.iter().map(|b| a.similarity(b)).collect())
            .collect()
    }
}

impl Default for MaterialHdcEncoder {
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
            MaterialHdcEncoder::new()
                .encode(&MaterialProperty::steel_a36())
                .dim(),
            HDC_DIMENSION
        );
    }
    #[test]
    fn test_encode_nonzero() {
        assert!(
            MaterialHdcEncoder::new()
                .encode(&MaterialProperty::steel_a36())
                .norm()
                > 0.0
        );
    }

    #[test]
    fn test_self_similarity() {
        let enc = MaterialHdcEncoder::new();
        let s = enc.similarity(
            &MaterialProperty::steel_a36(),
            &MaterialProperty::steel_a36(),
        );
        assert!(s > 0.99, "Self-similarity should be ~1.0, got {}", s);
    }

    #[test]
    fn test_metals_cluster() {
        let enc = MaterialHdcEncoder::new();
        let steel_titanium = enc.similarity(
            &MaterialProperty::steel_a36(),
            &MaterialProperty::titanium_ti6al4v(),
        );
        let steel_carbon = enc.similarity(
            &MaterialProperty::steel_a36(),
            &MaterialProperty::carbon_fiber_t300(),
        );
        assert!(
            steel_titanium > steel_carbon,
            "Steel-Titanium ({}) should be more similar than Steel-CarbonFiber ({})",
            steel_titanium,
            steel_carbon
        );
    }

    #[test]
    fn test_similarity_matrix_size() {
        let m = MaterialHdcEncoder::new().similarity_matrix(&MaterialProperty::presets());
        assert_eq!(m.len(), 6);
        assert_eq!(m[0].len(), 6);
    }

    #[test]
    fn test_similarity_matrix_symmetric() {
        let m = MaterialHdcEncoder::new().similarity_matrix(&MaterialProperty::presets());
        for i in 0..6 {
            for j in 0..6 {
                assert!((m[i][j] - m[j][i]).abs() < 1e-6);
            }
        }
    }
}
