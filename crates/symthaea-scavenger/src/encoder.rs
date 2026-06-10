// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{NUM_STATE_CHANNELS, ScavengerState};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

const DIM: usize = symthaea_core::hdc::HDC_DIMENSION;
const CHANNEL_RANGES: [(f32, f32); NUM_STATE_CHANNELS] = [
    (-1.0, 1.0),
    (-1.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1500.0),
    (-1.0, 1.0),
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
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
];

pub struct ScavengerHdcEncoder {
    base: Vec<ContinuousHV>,
    levels: Vec<ContinuousHV>,
    nl: usize,
}

impl ScavengerHdcEncoder {
    pub fn new(g: &GenesisSeed, nl: usize) -> Self {
        Self {
            base: (0..NUM_STATE_CHANNELS)
                .map(|i| ContinuousHV::from_genesis(g, &format!("scavenger::ch::{i}"), DIM))
                .collect(),
            levels: (0..nl)
                .map(|l| ContinuousHV::from_genesis(g, &format!("scavenger::lv::{l}"), DIM))
                .collect(),
            nl,
        }
    }

    pub fn encode(&mut self, state: &ScavengerState) -> ContinuousHV {
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
        let mut enc = ScavengerHdcEncoder::new(&GenesisSeed::from_phrase("test"), 32);
        assert_eq!(enc.encode(&ScavengerState::home()).dim(), DIM);
    }

    #[test]
    fn test_encoder_distinguishes_safe_and_hazardous_salvage_state() {
        let mut enc = ScavengerHdcEncoder::new(&GenesisSeed::from_phrase("test"), 32);
        let safe = ScavengerState::home();
        let mut hazardous = ScavengerState::home();
        hazardous.channels[13] = 0.9;
        hazardous.channels[19] = 0.9;
        hazardous.channels[23] = 0.8;
        hazardous.channels[27] = 0.8;
        hazardous.channels[28] = 0.75;
        let a = enc.encode(&safe);
        let b = enc.encode(&hazardous);
        assert!(a.similarity(&b) < 0.999);
    }
}
