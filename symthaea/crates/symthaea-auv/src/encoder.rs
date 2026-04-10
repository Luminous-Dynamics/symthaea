// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AUV HDC encoder: 32D state (24 kinematic + 8 chemical) → 16,384D ContinuousHV.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::types::{AuvState, NUM_STATE_CHANNELS};

/// Channel ranges for normalization.
const CHANNEL_RANGES: [[f32; 2]; NUM_STATE_CHANNELS] = [
    [-1000.0, 1000.0], // pos_x
    [-1000.0, 1000.0], // pos_y
    [0.0, 500.0],      // pos_z (depth)
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0], // quaternion
    [-3.0, 3.0],
    [-3.0, 3.0],
    [-3.0, 3.0], // linear vel (AUVs are slow)
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0],     // angular vel
    [0.0, 500.0],    // depth
    [0.0, 5000.0],   // pressure kPa
    [0.0, 40.0],     // water temp °C
    [-500.0, 500.0], // buoyancy force N
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0], // thruster feedback (first 4)
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0], // thruster feedback (last 3)
    // Chemical channels
    [0.0, 14.0],   // pH
    [0.0, 1000.0], // turbidity NTU
    [0.0, 5000.0], // TDS ppm
    [0.0, 100.0],  // nitrates mg/L
    [0.0, 100.0],  // arsenic µg/L
    [0.0, 100.0],  // lead µg/L
    [0.0, 5.0],    // chlorine mg/L
    [0.0, 20.0],   // dissolved oxygen mg/L
];

/// Semantic weights: chemical sensors and depth boosted for commons stewardship.
const CHANNEL_WEIGHTS: [f32; NUM_STATE_CHANNELS] = [
    1.0, 1.0, 2.0, // position (depth critical)
    1.5, 1.5, 1.5, 1.5, // quaternion
    1.0, 1.0, 1.0, // velocity
    1.5, 1.5, 1.5, // angular velocity
    2.0, // depth (critical for AUV)
    1.0, // pressure
    1.0, // water temp
    1.0, // buoyancy
    0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, // thruster feedback
    // Chemical channels — boosted for water stewardship mission
    2.0, // pH (critical)
    2.0, // turbidity (critical)
    1.5, // TDS
    2.5, // nitrates (SUPREME — agricultural runoff)
    3.0, // arsenic (SUPREME — health critical)
    3.0, // lead (SUPREME — health critical)
    1.5, // chlorine
    2.0, // dissolved oxygen (critical for ecosystem)
];

/// HDC encoder for AUV state.
pub struct AuvHdcEncoder {
    base_vectors: Vec<ContinuousHV>,
    level_vectors: Vec<ContinuousHV>,
    prev_channels: Option<[f32; NUM_STATE_CHANNELS]>,
    derivative_weight: f32,
    num_levels: usize,
}

impl AuvHdcEncoder {
    pub fn new(genesis: &GenesisSeed, num_levels: usize) -> Self {
        let dim = symthaea_core::hdc::HDC_DIMENSION;
        let base_vectors: Vec<ContinuousHV> = (0..NUM_STATE_CHANNELS)
            .map(|i| ContinuousHV::from_genesis(genesis, &format!("auv::ch::{i}"), dim))
            .collect();
        let level_vectors: Vec<ContinuousHV> = (0..num_levels)
            .map(|i| ContinuousHV::from_genesis(genesis, &format!("auv::lv::{i}"), dim))
            .collect();

        Self {
            base_vectors,
            level_vectors,
            prev_channels: None,
            derivative_weight: 0.3,
            num_levels,
        }
    }

    pub fn encode(&mut self, state: &AuvState) -> ContinuousHV {
        let channels = state.to_channels();
        let dim = symthaea_core::hdc::HDC_DIMENSION;
        let mut result = ContinuousHV::zero(dim);

        for ch in 0..NUM_STATE_CHANNELS {
            let [lo, hi] = CHANNEL_RANGES[ch];
            let range = hi - lo;
            let norm = if range.abs() < 1e-10 {
                0.5
            } else {
                ((channels[ch] - lo) / range).clamp(0.0, 1.0)
            };
            let k = (norm * (self.num_levels - 1) as f32).round() as usize;
            let k = k.min(self.num_levels - 1);

            let mut level_hv = ContinuousHV::zero(dim);
            for l in 0..=k {
                level_hv.add_in_place(&self.level_vectors[l]);
            }
            level_hv = level_hv.normalize();

            let bound = level_hv.bind(&self.base_vectors[ch]);
            let mut scaled = bound;
            scaled.scale_in_place(CHANNEL_WEIGHTS[ch]);
            result.add_in_place(&scaled);
        }

        self.prev_channels = Some(channels);
        result = result.normalize();
        result
    }

    pub fn reset(&mut self) {
        self.prev_channels = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder() -> AuvHdcEncoder {
        AuvHdcEncoder::new(&GenesisSeed::from_phrase("test-auv-encoder"), 32)
    }

    #[test]
    fn test_encode_dimension() {
        let mut enc = make_encoder();
        let hv = enc.encode(&AuvState::neutral_buoyancy(10.0));
        assert_eq!(hv.dim(), symthaea_core::hdc::HDC_DIMENSION);
    }

    #[test]
    fn test_encode_deterministic() {
        let mut e1 = make_encoder();
        let mut e2 = make_encoder();
        let state = AuvState::neutral_buoyancy(10.0);
        let h1 = e1.encode(&state);
        let h2 = e2.encode(&state);
        assert!(h1.similarity(&h2) > 0.999);
    }

    #[test]
    fn test_different_depths_differ() {
        let mut enc = make_encoder();
        let h1 = enc.encode(&AuvState::neutral_buoyancy(10.0));
        enc.reset();
        let h2 = enc.encode(&AuvState::neutral_buoyancy(100.0));
        assert!(h1.similarity(&h2) < 0.99, "Different depths should differ");
    }

    #[test]
    fn test_contaminated_water_differs() {
        let mut enc = make_encoder();
        let mut clean = AuvState::neutral_buoyancy(10.0);
        let h_clean = enc.encode(&clean);
        enc.reset();
        clean.chemical.arsenic_ug_l = 50.0; // Contaminated
        let h_dirty = enc.encode(&clean);
        let sim = h_clean.similarity(&h_dirty);
        assert!(
            sim < 0.99,
            "Contamination should change encoding: sim={sim}"
        );
    }
}
