// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{NUM_STATE_CHANNELS, STATE_CHANNEL_RANGES, SubterraneanState};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

const DIM: usize = symthaea_core::hdc::HDC_DIMENSION;
pub const CHANNEL_RANGES: [(f64, f64); NUM_STATE_CHANNELS] = STATE_CHANNEL_RANGES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderError {
    TooFewLevels,
}

impl std::fmt::Display for EncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewLevels => f.write_str("subterranean encoder requires at least two levels"),
        }
    }
}

impl std::error::Error for EncoderError {}

pub fn normalized_channels(state: &SubterraneanState) -> [f64; NUM_STATE_CHANNELS] {
    let mut normalized = [0.0; NUM_STATE_CHANNELS];
    for (index, output) in normalized.iter_mut().enumerate() {
        let (min_value, max_value) = CHANNEL_RANGES[index];
        let span = (max_value - min_value).max(1e-9);
        let value = state.channels[index];
        *output = if value.is_finite() {
            ((value - min_value) / span).clamp(0.0, 1.0)
        } else {
            // The hazard supervisor will fail closed. Keep the representation
            // finite so FEP/HDC cannot be poisoned by NaN propagation.
            match index {
                crate::types::CUTTER_TEMP_C
                | crate::types::MOTOR_TEMP_C
                | crate::types::SPOIL_BUFFER_FILL
                | crate::types::WATER_INGRESS_RATIO
                | crate::types::AQUIFER_RISK
                | crate::types::GAS_RISK
                | crate::types::HULL_STRESS
                | crate::types::SLURRY_LOAD
                | crate::types::ABORT_RECOMMENDATION => 1.0,
                crate::types::BATTERY_RATIO
                | crate::types::COMM_SIGNAL
                | crate::types::ROOF_STABILITY
                | crate::types::ESCAPE_CONFIDENCE
                | crate::types::LOCALIZATION_CONFIDENCE
                | crate::types::RELAY_LINK_QUALITY
                | crate::types::SEAL_INTEGRITY => 0.0,
                _ => 0.5,
            }
        };
    }
    normalized
}

pub struct SubterraneanHdcEncoder {
    base: Vec<ContinuousHV>,
    /// Cumulative normalized level hypervectors. Entry `k` is the normalized
    /// sum of raw level vectors `0..=k`, so encoding is O(channels) rather than
    /// rebuilding each cumulative level on every control tick.
    cumulative_levels: Vec<ContinuousHV>,
}

impl SubterraneanHdcEncoder {
    pub fn new(genesis: &GenesisSeed, levels: usize) -> Self {
        let levels = levels.max(2);
        Self::build(genesis, levels)
    }

    pub fn try_new(genesis: &GenesisSeed, levels: usize) -> Result<Self, EncoderError> {
        if levels < 2 {
            return Err(EncoderError::TooFewLevels);
        }
        Ok(Self::build(genesis, levels))
    }

    fn build(genesis: &GenesisSeed, levels: usize) -> Self {
        let base = (0..NUM_STATE_CHANNELS)
            .map(|index| {
                ContinuousHV::from_genesis(genesis, &format!("subterranean::ch::{index}"), DIM)
            })
            .collect();

        let mut cumulative = ContinuousHV::zero(DIM);
        let mut cumulative_levels = Vec::with_capacity(levels);
        for level in 0..levels {
            let raw =
                ContinuousHV::from_genesis(genesis, &format!("subterranean::lv::{level}"), DIM);
            cumulative.add_in_place(&raw);
            cumulative_levels.push(cumulative.clone().normalize());
        }

        Self {
            base,
            cumulative_levels,
        }
    }

    pub fn encode(&self, state: &SubterraneanState) -> ContinuousHV {
        let channels = normalized_channels(state);
        let mut result = ContinuousHV::zero(DIM);
        let last_level = self.cumulative_levels.len() - 1;
        for (index, normalized) in channels.iter().enumerate() {
            let level = (*normalized * last_level as f64).round() as usize;
            result.add_in_place(&self.cumulative_levels[level].bind(&self.base[index]));
        }
        result.normalize()
    }

    pub fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_dim() {
        let encoder = SubterraneanHdcEncoder::new(&GenesisSeed::from_phrase("test"), 32);
        assert_eq!(encoder.encode(&SubterraneanState::home()).dim(), DIM);
    }

    #[test]
    fn test_encoder_distinguishes_surface_and_deep_states() {
        let encoder = SubterraneanHdcEncoder::new(&GenesisSeed::from_phrase("test"), 32);
        let surface = SubterraneanState::home();
        let mut deep = SubterraneanState::home();
        deep.channels[0] = 120.0;
        deep.channels[4] = 140.0;
        deep.channels[23] = 0.7;
        deep.channels[29] = 0.85;
        let a = encoder.encode(&surface);
        let b = encoder.encode(&deep);
        assert!(a.similarity(&b) < 0.999);
    }

    #[test]
    fn invalid_level_count_is_rejected_by_checked_constructor() {
        assert!(matches!(
            SubterraneanHdcEncoder::try_new(&GenesisSeed::from_phrase("test"), 1),
            Err(EncoderError::TooFewLevels)
        ));
    }

    #[test]
    fn normalization_respects_physical_ranges() {
        let mut state = SubterraneanState::home();
        state.channels[crate::types::DEPTH_M] = 100.0;
        state.channels[crate::types::CUTTER_TEMP_C] = 90.0;
        let values = normalized_channels(&state);
        assert!((values[crate::types::DEPTH_M] - 0.5).abs() < 1e-6);
        assert!((values[crate::types::CUTTER_TEMP_C] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn malformed_channels_still_encode_to_finite_hypervector() {
        let encoder = SubterraneanHdcEncoder::new(&GenesisSeed::from_phrase("fault"), 32);
        let mut state = SubterraneanState::home();
        state.channels[crate::types::GAS_RISK] = f64::NAN;
        let normalized = normalized_channels(&state);
        assert!(normalized.iter().all(|value| value.is_finite()));
        assert!(
            encoder
                .encode(&state)
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        );
    }
}
