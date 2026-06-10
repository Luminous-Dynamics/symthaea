// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quadrotor HDC encoder: sensor state → ContinuousHV (16,384D).
//!
//! Follows the `PlasmaHdcEncoder` pattern from `symthaea/src/physics/plasma_hdc_encoder.rs`.
//! Each of 13 sensor channels gets a genesis-seeded base vector. Values are level-encoded
//! (thermometer coding) then bound with the base vector. The result is bundled into a single
//! 16,384D ContinuousHV. Derivative encoding provides velocity context.
//!
//! # Shared HDC Sensor Encoding Pattern
//!
//! Both this encoder and `PlasmaHdcEncoder` (in `symthaea-physics`) follow the same
//! 5-step pipeline for encoding continuous sensor data into hyperdimensional vectors:
//!
//! 1. **Normalize** each channel to \[0, 1\] using known physical ranges
//! 2. **Level-encode** via thermometer coding (bundle levels 0..=k)
//! 3. **Bind** the level HV with a genesis-seeded channel base vector
//! 4. **Bundle** all bound channel HVs into a single representation
//! 5. **Derivative encoding** (optional): encode rate-of-change channels
//!
//! If a third domain encoder appears, consider extracting a shared `SensorHdcEncoder`
//! trait to formalize this pattern.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

use crate::types::FlightState;

/// Channel names for the 13 sensor dimensions.
const CHANNEL_NAMES: [&str; 13] = [
    "pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z", "vel_x", "vel_y", "vel_z",
    "angvel_x", "angvel_y", "angvel_z",
];

/// Typical sensor ranges for normalization to [0, 1].
/// [min, max] per channel.
const CHANNEL_RANGES: [[f32; 2]; 13] = [
    [-2.0, 2.0],   // pos_x (meters)
    [-2.0, 2.0],   // pos_y
    [-0.5, 2.0],   // pos_z (altitude: ground to 2m)
    [-1.0, 1.0],   // quat_w
    [-1.0, 1.0],   // quat_x
    [-1.0, 1.0],   // quat_y
    [-1.0, 1.0],   // quat_z
    [-5.0, 5.0],   // vel_x (m/s)
    [-5.0, 5.0],   // vel_y
    [-5.0, 5.0],   // vel_z
    [-20.0, 20.0], // angvel_x (rad/s)
    [-20.0, 20.0], // angvel_y
    [-20.0, 20.0], // angvel_z
];

/// HDC encoder for quadrotor sensor state.
///
/// Encodes a 13D `FlightState` into a full 16,384D `ContinuousHV` via:
/// 1. Per-channel normalization to [0, 1]
/// 2. Level encoding (thermometer coding via bundled levels 0..k)
/// 3. Binding with genesis-seeded channel base vectors
/// 4. Bundling all 13 bound channel HVs
/// 5. Optional derivative encoding (rate-of-change channels)
pub struct QuadrotorHdcEncoder {
    /// Base vectors for each of 13 channels.
    base_vectors: Vec<ContinuousHV>,
    /// Level codebook (num_levels entries).
    level_vectors: Vec<ContinuousHV>,
    /// Base vectors for derivative channels (13 more).
    deriv_base_vectors: Vec<ContinuousHV>,
    /// Previous state for derivative computation.
    prev_channels: Option<[f32; 13]>,
    /// Weight for derivative encoding vs position encoding.
    derivative_weight: f32,
    /// Number of levels in the codebook.
    num_levels: usize,
}

impl QuadrotorHdcEncoder {
    /// Create a new encoder from a genesis seed.
    ///
    /// Each channel and level vector is deterministically derived from the seed.
    pub fn new(genesis: &GenesisSeed, num_levels: usize) -> Self {
        // Generate base vectors for each channel
        let base_vectors: Vec<ContinuousHV> = CHANNEL_NAMES
            .iter()
            .map(|name| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("flight::channel::{name}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        // Generate level codebook
        let level_vectors: Vec<ContinuousHV> = (0..num_levels)
            .map(|i| {
                ContinuousHV::from_genesis(genesis, &format!("flight::level::{i}"), HDC_DIMENSION)
            })
            .collect();

        // Generate derivative base vectors
        let deriv_base_vectors: Vec<ContinuousHV> = CHANNEL_NAMES
            .iter()
            .map(|name| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("flight::deriv::{name}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        Self {
            base_vectors,
            level_vectors,
            deriv_base_vectors,
            prev_channels: None,
            derivative_weight: 0.3,
            num_levels,
        }
    }

    /// Normalize a raw channel value to [0, 1] using known ranges.
    fn normalize_channel(channel: usize, value: f32) -> f32 {
        let [min, max] = CHANNEL_RANGES[channel];
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Level-encode a normalized [0,1] value using thermometer coding.
    ///
    /// Bundles levels 0..=k where k = floor(value * num_levels).
    /// This preserves magnitude ordering: nearby values have high similarity.
    fn encode_level(&self, normalized: f32) -> ContinuousHV {
        let k = ((normalized * self.num_levels as f32) as usize).min(self.num_levels - 1);
        if k == 0 {
            self.level_vectors[0].clone()
        } else {
            let refs: Vec<&ContinuousHV> = self.level_vectors[..=k].iter().collect();
            ContinuousHV::bundle(&refs)
        }
    }

    /// Encode a flight state into a full 16,384D ContinuousHV.
    pub fn encode(&mut self, state: &FlightState) -> ContinuousHV {
        let channels = state.to_channels();

        // Encode position channels: base_i ⊗ level(value_i)
        let mut bound_hvs: Vec<ContinuousHV> = Vec::with_capacity(26);
        for i in 0..13 {
            let normalized = Self::normalize_channel(i, channels[i]);
            let level_hv = self.encode_level(normalized);
            bound_hvs.push(self.base_vectors[i].bind(&level_hv));
        }

        // Encode derivative channels if we have a previous state
        if let Some(prev) = &self.prev_channels {
            for i in 0..13 {
                let delta = channels[i] - prev[i];
                // Normalize derivative: use half the channel range as scale
                let range = CHANNEL_RANGES[i][1] - CHANNEL_RANGES[i][0];
                let normalized_delta = ((delta / range) + 0.5).clamp(0.0, 1.0);
                let level_hv = self.encode_level(normalized_delta);
                bound_hvs.push(self.deriv_base_vectors[i].bind(&level_hv));
            }
        }

        self.prev_channels = Some(channels);

        // Bundle all with appropriate weights
        if bound_hvs.len() > 13 {
            // Weight position channels more than derivative channels
            let pos_weight = 1.0 - self.derivative_weight;
            let deriv_weight = self.derivative_weight;
            let mut weights = vec![pos_weight; 13];
            weights.extend(vec![deriv_weight; 13]);
            let refs: Vec<&ContinuousHV> = bound_hvs.iter().collect();
            ContinuousHV::weighted_bundle(&refs, &weights)
        } else {
            let refs: Vec<&ContinuousHV> = bound_hvs.iter().collect();
            ContinuousHV::bundle(&refs)
        }
    }

    /// Encode a flight state WITHOUT updating derivative state.
    ///
    /// Used for experience replay where temporal continuity is broken.
    /// Takes `&self` (not `&mut self`) — safe to call during replay without
    /// polluting encoder state. Produces position-only encoding (no derivatives).
    pub fn encode_stateless(&self, state: &FlightState) -> ContinuousHV {
        let channels = state.to_channels();
        let mut bound_hvs: Vec<ContinuousHV> = Vec::with_capacity(13);
        for i in 0..13 {
            let normalized = Self::normalize_channel(i, channels[i]);
            let level_hv = self.encode_level(normalized);
            bound_hvs.push(self.base_vectors[i].bind(&level_hv));
        }
        let refs: Vec<&ContinuousHV> = bound_hvs.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Reset the encoder state (clear derivative memory).
    pub fn reset(&mut self) {
        self.prev_channels = None;
    }

    /// Number of levels in the codebook.
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-flight-encoder")
    }

    #[test]
    fn test_encoder_determinism() {
        let genesis = test_genesis();
        let mut enc1 = QuadrotorHdcEncoder::new(&genesis, 32);
        let mut enc2 = QuadrotorHdcEncoder::new(&genesis, 32);

        let state = FlightState::hover(0.1);
        let hv1 = enc1.encode(&state);
        let hv2 = enc2.encode(&state);

        assert_eq!(hv1.dim(), HDC_DIMENSION);
        assert!(
            (hv1.similarity(&hv2) - 1.0).abs() < 1e-5,
            "Same genesis → identical encoding"
        );
    }

    #[test]
    fn test_encoder_output_dimension() {
        let genesis = test_genesis();
        let mut enc = QuadrotorHdcEncoder::new(&genesis, 32);
        let hv = enc.encode(&FlightState::hover(0.1));
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_encoder_differentiation() {
        let genesis = test_genesis();
        let mut enc = QuadrotorHdcEncoder::new(&genesis, 32);

        let hover = FlightState::hover(0.1);
        // Dramatically different state: displaced, tilted, moving fast
        let flying = FlightState {
            position: [1.5, -1.0, 1.5],
            quaternion: [0.5, 0.5, 0.5, 0.5], // ~120° rotation
            linear_velocity: [3.0, -2.0, 1.0],
            angular_velocity: [10.0, -5.0, 3.0],
            timestamp: 1.0,
        };

        let hv_hover = enc.encode(&hover);
        enc.reset();
        let hv_flying = enc.encode(&flying);

        // Dramatically different states should produce noticeably different HVs
        let sim = hv_hover.similarity(&hv_flying);
        assert!(sim < 0.95, "Very different states should differ: sim={sim}");
        // But should still not be orthogonal (they share structure)
        assert!(sim > 0.0, "Should share some structure: sim={sim}");
    }

    #[test]
    fn test_encoder_similar_states_high_similarity() {
        let genesis = test_genesis();
        let mut enc = QuadrotorHdcEncoder::new(&genesis, 32);

        let state1 = FlightState::hover(0.10);
        let state2 = FlightState::hover(0.11); // 1cm difference

        let hv1 = enc.encode(&state1);
        enc.reset();
        let hv2 = enc.encode(&state2);

        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.8,
            "Similar states should have high similarity: sim={sim}"
        );
    }

    #[test]
    fn test_encoder_derivative_encoding() {
        let genesis = test_genesis();
        let mut enc = QuadrotorHdcEncoder::new(&genesis, 32);

        // First encode (no derivative)
        let state1 = FlightState::hover(0.1);
        let hv1 = enc.encode(&state1);

        // Second encode (with derivative from state1→state2)
        let state2 = FlightState {
            position: [0.0, 0.0, 0.15],
            ..FlightState::hover(0.15)
        };
        let hv2 = enc.encode(&state2);

        // Second call includes derivative info, so dimensions are weighted differently
        assert_eq!(hv1.dim(), HDC_DIMENSION);
        assert_eq!(hv2.dim(), HDC_DIMENSION);
        // They should differ since hv2 includes derivative information
        let sim = hv1.similarity(&hv2);
        assert!(sim < 0.99, "Derivative should change encoding: sim={sim}");
    }

    #[test]
    fn test_encoder_reset() {
        let genesis = test_genesis();
        let mut enc = QuadrotorHdcEncoder::new(&genesis, 32);

        let state = FlightState::hover(0.1);
        enc.encode(&state);
        assert!(enc.prev_channels.is_some());

        enc.reset();
        assert!(enc.prev_channels.is_none());
    }

    #[test]
    fn test_encode_stateless_deterministic() {
        let genesis = test_genesis();
        let enc = QuadrotorHdcEncoder::new(&genesis, 32);
        let state = FlightState::hover(0.1);
        let hv1 = enc.encode_stateless(&state);
        let hv2 = enc.encode_stateless(&state);
        assert!(
            (hv1.similarity(&hv2) - 1.0).abs() < 1e-5,
            "encode_stateless should be deterministic"
        );
    }

    #[test]
    fn test_encode_stateless_no_side_effects() {
        let genesis = test_genesis();
        let mut enc = QuadrotorHdcEncoder::new(&genesis, 32);
        assert!(enc.prev_channels.is_none());

        let state = FlightState::hover(0.1);
        let _ = enc.encode_stateless(&state);

        // Should NOT have updated prev_channels
        assert!(
            enc.prev_channels.is_none(),
            "encode_stateless must not modify encoder state"
        );

        // Now call encode() — first call, no derivatives
        let hv_encode = enc.encode(&state);
        let hv_stateless = enc.encode_stateless(&state);

        // First encode() call (no prev) should match encode_stateless (both lack derivatives)
        assert!(
            hv_encode.similarity(&hv_stateless) > 0.99,
            "First encode() should match encode_stateless: sim={}",
            hv_encode.similarity(&hv_stateless)
        );
    }

    #[test]
    fn test_normalize_channel() {
        // pos_z range: [-0.5, 2.0]
        assert!((QuadrotorHdcEncoder::normalize_channel(2, -0.5) - 0.0).abs() < 1e-6);
        assert!((QuadrotorHdcEncoder::normalize_channel(2, 2.0) - 1.0).abs() < 1e-6);
        assert!((QuadrotorHdcEncoder::normalize_channel(2, 0.75) - 0.5).abs() < 1e-6);
        // Clamped
        assert!((QuadrotorHdcEncoder::normalize_channel(2, 10.0) - 1.0).abs() < 1e-6);
    }
}
