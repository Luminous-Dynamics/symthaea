// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Helicopter HDC encoder: 18D sensor state → 16,384D ContinuousHV.
//!
//! Follows the same bind-and-bundle pattern as symthaea-multirotor encoder:
//! 1. Normalize each channel to [0, 1] using known ranges
//! 2. Level-encode via thermometer coding (bundled levels 0..k)
//! 3. Bind with genesis-seeded per-channel base vectors
//! 4. Apply semantic weights (altitude-critical channels boosted)
//! 5. Bundle all bound HVs into a single 16,384D representation
//!
//! Derivative encoding (30% weight) tracks rate-of-change for anticipatory control.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::types::{HelicopterState, NUM_STATE_CHANNELS};

/// Sensor ranges for normalization to [0, 1].
const CHANNEL_RANGES: [[f32; 2]; NUM_STATE_CHANNELS] = [
    [-500.0, 500.0], // pos_x (meters, SAR search area)
    [-500.0, 500.0], // pos_y
    [0.0, 200.0],    // pos_z (altitude: ground to 200m)
    [-1.0, 1.0],     // quat_w
    [-1.0, 1.0],     // quat_x
    [-1.0, 1.0],     // quat_y
    [-1.0, 1.0],     // quat_z
    [-50.0, 50.0],   // vel_x (m/s, max ~180 km/h)
    [-50.0, 50.0],   // vel_y
    [-30.0, 30.0],   // vel_z (climb/descent rate)
    [-5.0, 5.0],     // angvel_x (rad/s)
    [-5.0, 5.0],     // angvel_y
    [-5.0, 5.0],     // angvel_z
    [0.0, 6000.0],   // main_rotor_rpm
    [0.0, 4000.0],   // tail_rotor_rpm
    [-0.3, 0.3],     // collective_pitch (rad)
    [-0.2, 0.2],     // cyclic_lon_feedback (rad)
    [-0.2, 0.2],     // cyclic_lat_feedback (rad)
];

/// Semantic channel weights for encoding importance.
/// Balance-critical and altitude channels are boosted.
const CHANNEL_WEIGHTS: [f32; NUM_STATE_CHANNELS] = [
    1.0, // pos_x
    1.0, // pos_y
    2.0, // pos_z (altitude — critical for SAR)
    1.5, // quat_w (orientation — balance)
    1.5, // quat_x
    1.5, // quat_y
    1.5, // quat_z
    1.0, // vel_x
    1.0, // vel_y
    1.5, // vel_z (climb rate — critical)
    1.5, // angvel_x (angular velocity — stability)
    1.5, // angvel_y
    1.5, // angvel_z
    2.0, // main_rotor_rpm (rotor state — critical)
    1.5, // tail_rotor_rpm (yaw stability)
    1.0, // collective_pitch
    0.8, // cyclic_lon_feedback
    0.8, // cyclic_lat_feedback
];

/// HDC encoder for helicopter sensor state.
pub struct HelicopterHdcEncoder {
    /// Base vectors for each of 18 channels (genesis-seeded).
    base_vectors: Vec<ContinuousHV>,
    /// Level codebook (num_levels entries).
    level_vectors: Vec<ContinuousHV>,
    /// Base vectors for derivative channels.
    deriv_base_vectors: Vec<ContinuousHV>,
    /// Previous channels for derivative computation.
    prev_channels: Option<[f32; NUM_STATE_CHANNELS]>,
    /// Weight for derivative encoding.
    derivative_weight: f32,
    /// Number of levels in the codebook.
    num_levels: usize,
}

impl HelicopterHdcEncoder {
    /// Create a new encoder from a genesis seed.
    pub fn new(genesis: &GenesisSeed, num_levels: usize) -> Self {
        assert!(num_levels >= 2, "HDC encoder requires at least two levels");
        let dim = symthaea_core::hdc::HDC_DIMENSION;

        let base_vectors: Vec<ContinuousHV> = (0..NUM_STATE_CHANNELS)
            .map(|i| ContinuousHV::from_genesis(genesis, &format!("helicopter::channel::{i}"), dim))
            .collect();

        // Precompute cumulative thermometer levels once. The previous hot
        // path rebuilt and normalized levels 0..=k for every channel at every
        // physics tick.
        let raw_levels: Vec<ContinuousHV> = (0..num_levels)
            .map(|i| ContinuousHV::from_genesis(genesis, &format!("helicopter::level::{i}"), dim))
            .collect();
        let mut cumulative = ContinuousHV::zero(dim);
        let mut level_vectors = Vec::with_capacity(num_levels);
        for level in &raw_levels {
            cumulative.add_in_place(level);
            level_vectors.push(cumulative.clone().normalize());
        }

        let deriv_base_vectors: Vec<ContinuousHV> = (0..NUM_STATE_CHANNELS)
            .map(|i| ContinuousHV::from_genesis(genesis, &format!("helicopter::deriv::{i}"), dim))
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

    /// Encode a helicopter state into a 16,384D ContinuousHV.
    ///
    /// This compatibility entry point treats the interval as one second.
    /// Control loops should call [`Self::encode_with_dt`] so derivative
    /// channels represent rates rather than scheduler-dependent deltas.
    pub fn encode(&mut self, state: &HelicopterState) -> ContinuousHV {
        self.encode_with_dt(state, 1.0)
    }

    /// Encode a state with an explicit sample interval in seconds.
    pub fn encode_with_dt(&mut self, state: &HelicopterState, dt: f64) -> ContinuousHV {
        let channels = state.to_channels();
        let dim = symthaea_core::hdc::HDC_DIMENSION;
        let mut result = ContinuousHV::zero(dim);

        for ch in 0..NUM_STATE_CHANNELS {
            // Normalize to [0, 1]
            let [lo, hi] = CHANNEL_RANGES[ch];
            let range = hi - lo;
            let normalized = if range.abs() < 1e-10 {
                0.5
            } else {
                ((channels[ch] - lo) / range).clamp(0.0, 1.0)
            };

            // Level encode: bundle levels 0..k where k = normalized * (num_levels - 1)
            let k = (normalized * (self.num_levels - 1) as f32).round() as usize;
            let k = k.min(self.num_levels - 1);
            let level_hv = &self.level_vectors[k];

            // Bind with channel base vector
            let bound = level_hv.bind(&self.base_vectors[ch]);

            // Weighted accumulate
            let weight = CHANNEL_WEIGHTS[ch];
            let mut scaled = bound;
            scaled.scale_in_place(weight);
            result.add_in_place(&scaled);
        }

        // Derivative encoding (if we have previous channels)
        if let Some(ref prev) = self.prev_channels {
            for ch in 0..NUM_STATE_CHANNELS {
                let delta = channels[ch] - prev[ch];
                let [lo, hi] = CHANNEL_RANGES[ch];
                let range = hi - lo;
                let normalized_delta = normalize_rate(delta, range, dt);

                let k = (normalized_delta * (self.num_levels - 1) as f32).round() as usize;
                let k = k.min(self.num_levels - 1);
                let level_hv = &self.level_vectors[k];

                let bound = level_hv.bind(&self.deriv_base_vectors[ch]);
                let mut scaled = bound;
                scaled.scale_in_place(CHANNEL_WEIGHTS[ch] * self.derivative_weight);
                result.add_in_place(&scaled);
            }
        }

        self.prev_channels = Some(channels);
        result = result.normalize();
        result
    }

    /// Reset derivative state (for new episodes).
    pub fn reset(&mut self) {
        self.prev_channels = None;
    }
}

fn normalize_rate(delta: f32, range: f32, dt: f64) -> f32 {
    if range.abs() < 1e-10 || !dt.is_finite() || dt <= 0.0 {
        return 0.5;
    }
    let rate_fraction = delta / dt as f32 / range;
    rate_fraction.clamp(-1.0, 1.0) * 0.5 + 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder() -> HelicopterHdcEncoder {
        let genesis = GenesisSeed::from_phrase("test-helicopter-encoder");
        HelicopterHdcEncoder::new(&genesis, 32)
    }

    #[test]
    #[should_panic(expected = "at least two levels")]
    fn test_encoder_rejects_degenerate_codebook() {
        let genesis = GenesisSeed::from_phrase("test-helicopter-encoder");
        let _ = HelicopterHdcEncoder::new(&genesis, 1);
    }

    #[test]
    fn test_derivative_normalization_is_rate_based() {
        let slow = normalize_rate(1.0, 100.0, 0.1);
        let fast = normalize_rate(0.1, 100.0, 0.01);
        assert!((slow - fast).abs() < 1e-6);
        assert_eq!(normalize_rate(1.0, 100.0, 0.0), 0.5);
    }

    #[test]
    fn test_encode_produces_correct_dimension() {
        let mut enc = make_encoder();
        let state = HelicopterState::hover(20.0);
        let hv = enc.encode(&state);
        assert_eq!(hv.dim(), symthaea_core::hdc::HDC_DIMENSION);
    }

    #[test]
    fn test_encode_deterministic() {
        let mut enc1 = make_encoder();
        let mut enc2 = make_encoder();
        let state = HelicopterState::hover(20.0);
        let hv1 = enc1.encode(&state);
        let hv2 = enc2.encode(&state);
        assert!(
            hv1.similarity(&hv2) > 0.999,
            "Same genesis + same state should produce same encoding"
        );
    }

    #[test]
    fn test_different_states_differ() {
        let mut enc = make_encoder();
        let hover = HelicopterState::hover(20.0);
        let high = HelicopterState::hover(100.0);
        let hv_hover = enc.encode(&hover);
        enc.reset();
        let hv_high = enc.encode(&high);
        let sim = hv_hover.similarity(&hv_high);
        assert!(
            sim < 0.99,
            "Different altitudes should produce different encodings: sim={sim}"
        );
    }

    #[test]
    fn test_derivative_changes_encoding() {
        let mut enc = make_encoder();
        let state = HelicopterState::hover(20.0);

        // First encoding (no derivative)
        let hv1 = enc.encode(&state);

        // Second encoding (with derivative — should differ slightly)
        let hv2 = enc.encode(&state);

        // Same state twice → derivative is zero → should be very similar
        let sim = hv1.similarity(&hv2);
        assert!(sim > 0.95, "Same state twice should be similar: sim={sim}");
    }

    #[test]
    fn test_reset_clears_derivative() {
        let mut enc = make_encoder();
        let state = HelicopterState::hover(20.0);
        enc.encode(&state);
        assert!(enc.prev_channels.is_some());
        enc.reset();
        assert!(enc.prev_channels.is_none());
    }
}
