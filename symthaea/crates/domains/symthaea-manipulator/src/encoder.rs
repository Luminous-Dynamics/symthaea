// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Manipulator HDC encoder: 21D state → 16,384D ContinuousHV.

use crate::types::{ManipulatorState, NUM_STATE_CHANNELS};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

const CHANNEL_RANGES: [[f32; 2]; NUM_STATE_CHANNELS] = [
    [-3.14, 3.14],
    [-3.14, 3.14],
    [-3.14, 3.14],
    [-3.14, 3.14], // J1-4 angles
    [-3.14, 3.14],
    [-3.14, 3.14],
    [-3.14, 3.14], // J5-7 angles
    [-5.0, 5.0],
    [-5.0, 5.0],
    [-5.0, 5.0],
    [-5.0, 5.0], // J1-4 velocities
    [-5.0, 5.0],
    [-5.0, 5.0],
    [-5.0, 5.0], // J5-7 velocities
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0], // end-effector pos
    [-50.0, 50.0],
    [-50.0, 50.0],
    [-50.0, 50.0], // end-effector force
    [0.0, 1.0],    // gripper
];

const CHANNEL_WEIGHTS: [f32; NUM_STATE_CHANNELS] = [
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, // Joint angles (precision-critical)
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // Velocities
    2.0, 2.0, 2.0, // End-effector position (CRITICAL)
    2.5, 2.5, 2.5, // Force feedback (SUPREME — safety)
    1.5, // Gripper state
];

pub struct ManipulatorHdcEncoder {
    base_vectors: Vec<ContinuousHV>,
    level_vectors: Vec<ContinuousHV>,
    num_levels: usize,
}

impl ManipulatorHdcEncoder {
    pub fn new(genesis: &GenesisSeed, num_levels: usize) -> Self {
        let dim = symthaea_core::hdc::HDC_DIMENSION;
        let base_vectors: Vec<ContinuousHV> = (0..NUM_STATE_CHANNELS)
            .map(|i| ContinuousHV::from_genesis(genesis, &format!("manip::ch::{i}"), dim))
            .collect();
        let level_vectors: Vec<ContinuousHV> = (0..num_levels)
            .map(|i| ContinuousHV::from_genesis(genesis, &format!("manip::lv::{i}"), dim))
            .collect();
        Self {
            base_vectors,
            level_vectors,
            num_levels,
        }
    }

    pub fn encode(&mut self, state: &ManipulatorState) -> ContinuousHV {
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
        result = result.normalize();
        result
    }

    pub fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ManipulatorState;

    #[test]
    fn test_encode_dimension() {
        let mut enc = ManipulatorHdcEncoder::new(&GenesisSeed::from_phrase("test"), 32);
        let hv = enc.encode(&ManipulatorState::home());
        assert_eq!(hv.dim(), symthaea_core::hdc::HDC_DIMENSION);
    }

    #[test]
    fn test_encode_deterministic() {
        let genesis = GenesisSeed::from_phrase("test-det");
        let mut e1 = ManipulatorHdcEncoder::new(&genesis, 32);
        let mut e2 = ManipulatorHdcEncoder::new(&genesis, 32);
        let state = ManipulatorState::home();
        assert!(e1.encode(&state).similarity(&e2.encode(&state)) > 0.999);
    }

    #[test]
    fn test_different_states_differ() {
        let mut enc = ManipulatorHdcEncoder::new(&GenesisSeed::from_phrase("test"), 32);
        let h1 = enc.encode(&ManipulatorState::home());
        let mut bent = ManipulatorState::home();
        bent.joint_angles[3] = -2.0; // Bend elbow significantly
        bent.end_effector_position = [0.5, 0.2, 0.3];
        let h2 = enc.encode(&bent);
        assert!(h1.similarity(&h2) < 0.99, "Different states should differ");
    }
}
