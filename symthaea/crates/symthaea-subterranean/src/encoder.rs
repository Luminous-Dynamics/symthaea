// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{SubterraneanState, NUM_STATE_CHANNELS};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

const DIM: usize = symthaea_core::hdc::HDC_DIMENSION;
const CHANNEL_RANGES: [(f32, f32); NUM_STATE_CHANNELS] = [
    (0.0, 200.0),
    (-2.0, 2.0),
    (-0.6, 0.6),
    (-0.5, 0.5),
    (0.0, 180.0),
    (0.0, 160.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 2.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
];

pub struct SubterraneanHdcEncoder {
    base: Vec<ContinuousHV>,
    levels: Vec<ContinuousHV>,
    nl: usize,
}

impl SubterraneanHdcEncoder {
    pub fn new(g: &GenesisSeed, nl: usize) -> Self {
        Self {
            base: (0..NUM_STATE_CHANNELS)
                .map(|i| ContinuousHV::from_genesis(g, &format!("subterranean::ch::{i}"), DIM))
                .collect(),
            levels: (0..nl)
                .map(|l| ContinuousHV::from_genesis(g, &format!("subterranean::lv::{l}"), DIM))
                .collect(),
            nl,
        }
    }

    pub fn encode(&mut self, state: &SubterraneanState) -> ContinuousHV {
        let ch = state.to_channels();
        let mut res = ContinuousHV::zero(DIM);
        for i in 0..NUM_STATE_CHANNELS {
            let (min_v, max_v) = CHANNEL_RANGES[i];
            let span = (max_v - min_v).max(1e-6);
            let n = ((ch[i] - min_v) / span).clamp(0.0, 1.0);
            let k = (n * (self.nl - 1) as f32).round() as usize;
            let k = k.min(self.nl - 1);
            let mut lhv = ContinuousHV::zero(DIM);
            for l in 0..=k {
                lhv.add_in_place(&self.levels[l]);
            }
            lhv = lhv.normalize();
            res.add_in_place(&lhv.bind(&self.base[i]));
        }
        res.normalize()
    }

    pub fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_encode_dim() {
        let mut enc = SubterraneanHdcEncoder::new(&GenesisSeed::from_phrase("test"), 32);
        assert_eq!(enc.encode(&SubterraneanState::home()).dim(), DIM);
    }

    #[test]
    fn test_encoder_distinguishes_surface_and_deep_states() {
        let mut enc = SubterraneanHdcEncoder::new(&GenesisSeed::from_phrase("test"), 32);
        let surface = SubterraneanState::home();
        let mut deep = SubterraneanState::home();
        deep.channels[0] = 120.0;
        deep.channels[4] = 140.0;
        deep.channels[23] = 0.7;
        deep.channels[29] = 0.85;
        let a = enc.encode(&surface);
        let b = enc.encode(&deep);
        assert!(a.similarity(&b) < 0.999);
    }
}
