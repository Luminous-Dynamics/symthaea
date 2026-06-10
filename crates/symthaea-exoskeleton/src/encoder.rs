// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{ExoskeletonState, NUM_STATE_CHANNELS};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
const DIM: usize = symthaea_core::hdc::HDC_DIMENSION;

const RANGES: [[f32; 2]; NUM_STATE_CHANNELS] = [
    [-2.0, 2.0],
    [0.0, 2.5],
    [-0.5, 0.5],
    [-2.0, 2.0],
    [0.0, 2.5],
    [-0.5, 0.5],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-100.0, 100.0],
    [-100.0, 100.0],
    [-100.0, 100.0],
    [-100.0, 100.0],
    [-100.0, 100.0],
    [-100.0, 100.0],
    [-50.0, 50.0],
    [-50.0, 50.0],
    [-50.0, 50.0],
    [-50.0, 50.0],
    [-50.0, 50.0],
    [-50.0, 50.0],
    [-0.15, 0.15],
    [-0.15, 0.15],
    [0.0, 2000.0],
    [0.0, 1.0],
];

pub struct ExoskeletonHdcEncoder {
    base: Vec<ContinuousHV>,
    levels: Vec<ContinuousHV>,
    prev: Option<[f32; NUM_STATE_CHANNELS]>,
    num_levels: usize,
}

impl ExoskeletonHdcEncoder {
    pub fn new(genesis: &GenesisSeed, num_levels: usize) -> Self {
        Self {
            base: (0..NUM_STATE_CHANNELS)
                .map(|i| ContinuousHV::from_genesis(genesis, &format!("exo::ch::{i}"), DIM))
                .collect(),
            levels: (0..num_levels)
                .map(|l| ContinuousHV::from_genesis(genesis, &format!("exo::lv::{l}"), DIM))
                .collect(),
            prev: None,
            num_levels,
        }
    }
    pub fn encode(&mut self, state: &ExoskeletonState) -> ContinuousHV {
        let ch = state.to_channels();
        let mut result = ContinuousHV::zero(DIM);
        for i in 0..NUM_STATE_CHANNELS {
            let [lo, hi] = RANGES[i];
            let r = hi - lo;
            let n = if r.abs() < 1e-10 {
                0.5
            } else {
                ((ch[i] - lo) / r).clamp(0.0, 1.0)
            };
            let k = (n * (self.num_levels - 1) as f32).round() as usize;
            let k = k.min(self.num_levels - 1);
            let mut lhv = ContinuousHV::zero(DIM);
            for l in 0..=k {
                lhv.add_in_place(&self.levels[l]);
            }
            lhv = lhv.normalize();
            result.add_in_place(&lhv.bind(&self.base[i]));
        }
        self.prev = Some(ch);
        result.normalize()
    }
    pub fn reset(&mut self) {
        self.prev = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dim() {
        let mut e = ExoskeletonHdcEncoder::new(&GenesisSeed::from_phrase("t"), 32);
        assert_eq!(e.encode(&ExoskeletonState::standing()).dim(), DIM);
    }
}
