//! HDC encoder for material properties into 16,384-dimensional continuous hypervectors.

use crate::properties::MaterialProperty;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

const SEEDS: [u64; 8] = [
    0xCA7_A001, 0xCA7_A002, 0xCA7_A003, 0xCA7_A004,
    0xCA7_A005, 0xCA7_A006, 0xCA7_A007, 0xCA7_A008,
];

pub struct MaterialHdcEncoder {
    bases: [ContinuousHV; 8],
}

impl MaterialHdcEncoder {
    pub fn new() -> Self {
        Self {
            bases: std::array::from_fn(|i| ContinuousHV::random(HDC_DIMENSION, SEEDS[i])),
        }
    }

    pub fn encode(&self, material: &MaterialProperty) -> ContinuousHV {
        let vals = material.normalized_values();
        let scaled: Vec<ContinuousHV> = self.bases.iter().zip(vals.iter())
            .map(|(base, &v)| {
                let mut hv = base.clone();
                for x in hv.values.as_mut_slice() { *x *= v; }
                hv
            })
            .collect();
        ContinuousHV::bundle(&scaled.iter().collect::<Vec<_>>())
    }

    pub fn similarity(&self, a: &MaterialProperty, b: &MaterialProperty) -> f32 {
        self.encode(a).similarity(&self.encode(b))
    }

    pub fn similarity_matrix(&self, materials: &[MaterialProperty]) -> Vec<Vec<f32>> {
        let hvs: Vec<ContinuousHV> = materials.iter().map(|m| self.encode(m)).collect();
        hvs.iter().map(|a| hvs.iter().map(|b| a.similarity(b)).collect()).collect()
    }
}

impl Default for MaterialHdcEncoder { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_encode_dimension() { assert_eq!(MaterialHdcEncoder::new().encode(&MaterialProperty::steel_a36()).dim(), HDC_DIMENSION); }
    #[test] fn test_encode_nonzero() { assert!(MaterialHdcEncoder::new().encode(&MaterialProperty::steel_a36()).norm() > 0.0); }

    #[test]
    fn test_self_similarity() {
        let enc = MaterialHdcEncoder::new();
        let s = enc.similarity(&MaterialProperty::steel_a36(), &MaterialProperty::steel_a36());
        assert!(s > 0.99, "Self-similarity should be ~1.0, got {}", s);
    }

    #[test]
    fn test_metals_cluster() {
        let enc = MaterialHdcEncoder::new();
        let steel_titanium = enc.similarity(&MaterialProperty::steel_a36(), &MaterialProperty::titanium_ti6al4v());
        let steel_carbon = enc.similarity(&MaterialProperty::steel_a36(), &MaterialProperty::carbon_fiber_t300());
        assert!(steel_titanium > steel_carbon,
            "Steel-Titanium ({}) should be more similar than Steel-CarbonFiber ({})",
            steel_titanium, steel_carbon);
    }

    #[test]
    fn test_similarity_matrix_size() {
        let m = MaterialHdcEncoder::new().similarity_matrix(&MaterialProperty::presets());
        assert_eq!(m.len(), 5);
        assert_eq!(m[0].len(), 5);
    }

    #[test]
    fn test_similarity_matrix_symmetric() {
        let m = MaterialHdcEncoder::new().similarity_matrix(&MaterialProperty::presets());
        for i in 0..5 { for j in 0..5 { assert!((m[i][j] - m[j][i]).abs() < 1e-6); } }
    }
}
