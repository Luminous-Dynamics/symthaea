// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Humanoid HDC encoder: sensor state → ContinuousHV (16,384D).
//!
//! Follows the `QuadrotorHdcEncoder` pattern from `symthaea-flight/src/encoder.rs`.
//! Each of 72 sensor channels gets a genesis-seeded base vector. Values are level-encoded
//! (thermometer coding) then bound with the base vector. Channels are organized in semantic
//! groups with balance-critical channels weighted higher.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

use crate::types::HumanoidState;

/// Semantic channel groups for weighted encoding.
///
/// Balance-critical channels (root height, head height, uprightness) are weighted
/// higher to bias the HDC representation toward stability-relevant features.
#[derive(Debug, Clone, Copy)]
struct ChannelGroup {
    start: usize,
    count: usize,
    weight: f32,
}

/// Group definitions matching the `to_channels()` layout:
///
/// | Group           | Channels    | Count | Weight | Rationale                  |
/// |-----------------|-------------|-------|--------|----------------------------|
/// | Root height     | [0]         | 1     | 2.0    | Critical for standing      |
/// | Root quaternion | [1..5]      | 4     | 1.5    | Balance orientation        |
/// | Joint angles    | [5..26]     | 21    | 1.0    | Posture                    |
/// | Root lin vel    | [26..29]    | 3     | 1.5    | Fall detection             |
/// | Root ang vel    | [29..32]    | 3     | 1.5    | Fall detection             |
/// | Joint vel       | [32..53]    | 21    | 0.8    | Dynamic state              |
/// | Head height     | [53]        | 1     | 2.0    | Standing reward signal     |
/// | Torso vertical  | [54..57]    | 3     | 1.5    | Uprightness                |
/// | Extremities     | [57..69]    | 12    | 0.5    | Less critical initially    |
/// | COM velocity    | [69..72]    | 3     | 1.0    | Locomotion target          |
const CHANNEL_GROUPS: [ChannelGroup; 10] = [
    ChannelGroup {
        start: 0,
        count: 1,
        weight: 2.0,
    }, // root height
    ChannelGroup {
        start: 1,
        count: 4,
        weight: 1.5,
    }, // root quaternion
    ChannelGroup {
        start: 5,
        count: 21,
        weight: 1.0,
    }, // joint angles
    ChannelGroup {
        start: 26,
        count: 3,
        weight: 1.5,
    }, // root linear velocity
    ChannelGroup {
        start: 29,
        count: 3,
        weight: 1.5,
    }, // root angular velocity
    ChannelGroup {
        start: 32,
        count: 21,
        weight: 0.8,
    }, // joint velocities
    ChannelGroup {
        start: 53,
        count: 1,
        weight: 2.0,
    }, // head height
    ChannelGroup {
        start: 54,
        count: 3,
        weight: 1.5,
    }, // torso vertical
    ChannelGroup {
        start: 57,
        count: 12,
        weight: 0.5,
    }, // extremities
    ChannelGroup {
        start: 69,
        count: 3,
        weight: 1.0,
    }, // COM velocity
];

/// Total number of channels (must match HumanoidState::num_channels()).
const NUM_CHANNELS: usize = 72;

/// Channel normalization ranges [min, max].
/// Chosen to cover typical bipedal humanoid operation ranges.
const CHANNEL_RANGES: [[f32; 2]; NUM_CHANNELS] = [
    // Root height (1)
    [0.0, 2.0],
    // Root quaternion (4) — unit quaternion components
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    // Joint angles (21) — radians, typical range ~[-2, 2]
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0], // abdomen y/z/x
    [-1.5, 1.5],
    [-1.5, 1.5],
    [-2.0, 0.5],
    [-2.5, 0.0], // right hip x/z/y, knee
    [-1.0, 1.0],
    [-1.0, 1.0], // right ankle x/y
    [-1.5, 1.5],
    [-1.5, 1.5],
    [-2.0, 0.5],
    [-2.5, 0.0], // left hip x/z/y, knee
    [-1.0, 1.0],
    [-1.0, 1.0], // left ankle x/y
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.5, 0.0], // right shoulder1/2, elbow
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.5, 0.0], // left shoulder1/2, elbow
    // Root linear velocity (3) — m/s
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
    // Root angular velocity (3) — rad/s
    [-20.0, 20.0],
    [-20.0, 20.0],
    [-20.0, 20.0],
    // Joint velocities (21) — rad/s
    [-20.0, 20.0],
    [-20.0, 20.0],
    [-20.0, 20.0], // abdomen
    [-20.0, 20.0],
    [-20.0, 20.0],
    [-20.0, 20.0],
    [-20.0, 20.0], // right hip/knee
    [-20.0, 20.0],
    [-20.0, 20.0], // right ankle
    [-20.0, 20.0],
    [-20.0, 20.0],
    [-20.0, 20.0],
    [-20.0, 20.0], // left hip/knee
    [-20.0, 20.0],
    [-20.0, 20.0], // left ankle
    [-20.0, 20.0],
    [-20.0, 20.0],
    [-20.0, 20.0], // right arm
    [-20.0, 20.0],
    [-20.0, 20.0],
    [-20.0, 20.0], // left arm
    // Head height (1)
    [0.0, 2.0],
    // Torso vertical (3) — direction vector components
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    // Extremities (12) — world positions (meters)
    [-2.0, 2.0],
    [-2.0, 2.0],
    [0.0, 2.0], // right hand
    [-2.0, 2.0],
    [-2.0, 2.0],
    [0.0, 2.0], // left hand
    [-2.0, 2.0],
    [-2.0, 2.0],
    [0.0, 0.5], // right foot
    [-2.0, 2.0],
    [-2.0, 2.0],
    [0.0, 0.5], // left foot
    // COM velocity (3) — m/s
    [-10.0, 10.0],
    [-10.0, 10.0],
    [-10.0, 10.0],
];

/// HDC encoder for humanoid sensor state.
///
/// Encodes 72D `HumanoidState` into a full 16,384D `ContinuousHV` via:
/// 1. Per-channel normalization to [0, 1]
/// 2. Level encoding (thermometer coding via bundled levels 0..k)
/// 3. Binding with genesis-seeded channel base vectors
/// 4. Semantic group weighting (balance-critical channels weighted higher)
/// 5. Bundling all bound channel HVs
/// 6. Optional derivative encoding (rate-of-change channels)
pub struct HumanoidHdcEncoder {
    /// Base vectors for each of 72 channels.
    base_vectors: Vec<ContinuousHV>,
    /// Level codebook (num_levels entries).
    level_vectors: Vec<ContinuousHV>,
    /// Base vectors for derivative channels (72 more).
    deriv_base_vectors: Vec<ContinuousHV>,
    /// Previous state channels for derivative computation.
    prev_channels: Option<Vec<f32>>,
    /// Weight for derivative encoding vs position encoding.
    derivative_weight: f32,
    /// Number of levels in the codebook.
    num_levels: usize,
}

impl HumanoidHdcEncoder {
    /// Create a new encoder from a genesis seed.
    pub fn new(genesis: &GenesisSeed, num_levels: usize) -> Self {
        let base_vectors: Vec<ContinuousHV> = (0..NUM_CHANNELS)
            .map(|i| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("humanoid::channel::{i}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        let level_vectors: Vec<ContinuousHV> = (0..num_levels)
            .map(|i| {
                ContinuousHV::from_genesis(genesis, &format!("humanoid::level::{i}"), HDC_DIMENSION)
            })
            .collect();

        let deriv_base_vectors: Vec<ContinuousHV> = (0..NUM_CHANNELS)
            .map(|i| {
                ContinuousHV::from_genesis(genesis, &format!("humanoid::deriv::{i}"), HDC_DIMENSION)
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
    fn encode_level(&self, normalized: f32) -> ContinuousHV {
        let k = ((normalized * self.num_levels as f32) as usize).min(self.num_levels - 1);
        if k == 0 {
            self.level_vectors[0].clone()
        } else {
            let refs: Vec<&ContinuousHV> = self.level_vectors[..=k].iter().collect();
            ContinuousHV::bundle(&refs)
        }
    }

    /// Get the semantic group weight for a given channel index.
    fn channel_weight(channel: usize) -> f32 {
        for group in &CHANNEL_GROUPS {
            if channel >= group.start && channel < group.start + group.count {
                return group.weight;
            }
        }
        1.0
    }

    /// Encode a humanoid state into a full 16,384D ContinuousHV.
    pub fn encode(&mut self, state: &HumanoidState) -> ContinuousHV {
        let channels = state.to_channels();
        let n = channels.len();

        // Encode position channels: base_i bind level(value_i)
        let mut bound_hvs: Vec<ContinuousHV> = Vec::with_capacity(n * 2);
        let mut weights: Vec<f32> = Vec::with_capacity(n * 2);

        for i in 0..n {
            let normalized = Self::normalize_channel(i, channels[i]);
            let level_hv = self.encode_level(normalized);
            bound_hvs.push(self.base_vectors[i].bind(&level_hv));
            weights.push(Self::channel_weight(i) * (1.0 - self.derivative_weight));
        }

        // Encode derivative channels if we have a previous state
        if let Some(ref prev) = self.prev_channels {
            for i in 0..n {
                let delta = channels[i] - prev[i];
                let range = CHANNEL_RANGES[i][1] - CHANNEL_RANGES[i][0];
                let normalized_delta = ((delta / range) + 0.5).clamp(0.0, 1.0);
                let level_hv = self.encode_level(normalized_delta);
                bound_hvs.push(self.deriv_base_vectors[i].bind(&level_hv));
                weights.push(Self::channel_weight(i) * self.derivative_weight);
            }
        }

        self.prev_channels = Some(channels);

        let refs: Vec<&ContinuousHV> = bound_hvs.iter().collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
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
        GenesisSeed::from_phrase("test-humanoid-encoder")
    }

    #[test]
    fn test_encoder_determinism() {
        let genesis = test_genesis();
        let mut enc1 = HumanoidHdcEncoder::new(&genesis, 32);
        let mut enc2 = HumanoidHdcEncoder::new(&genesis, 32);

        let state = HumanoidState::standing();
        let hv1 = enc1.encode(&state);
        let hv2 = enc2.encode(&state);

        assert_eq!(hv1.dim(), HDC_DIMENSION);
        assert!(
            (hv1.similarity(&hv2) - 1.0).abs() < 1e-5,
            "Same genesis -> identical encoding"
        );
    }

    #[test]
    fn test_encoder_output_dimension() {
        let genesis = test_genesis();
        let mut enc = HumanoidHdcEncoder::new(&genesis, 32);
        let hv = enc.encode(&HumanoidState::standing());
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_encoder_differentiation() {
        let genesis = test_genesis();
        let mut enc = HumanoidHdcEncoder::new(&genesis, 32);

        let standing = HumanoidState::standing();
        let mut fallen = HumanoidState::standing();
        fallen.root_height = 0.3;
        fallen.head_height = 0.4;
        fallen.torso_vertical = [0.7, 0.0, 0.7];
        fallen.root_angular_velocity = [5.0, -3.0, 2.0];

        let hv_standing = enc.encode(&standing);
        enc.reset();
        let hv_fallen = enc.encode(&fallen);

        let sim = hv_standing.similarity(&hv_fallen);
        assert!(sim < 0.95, "Different states should differ: sim={sim}");
        assert!(sim > 0.0, "Should share some structure: sim={sim}");
    }

    #[test]
    fn test_encoder_similar_states_high_similarity() {
        let genesis = test_genesis();
        let mut enc = HumanoidHdcEncoder::new(&genesis, 32);

        let state1 = HumanoidState::standing();
        let mut state2 = HumanoidState::standing();
        state2.root_height = 1.31; // 1cm difference

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
        let mut enc = HumanoidHdcEncoder::new(&genesis, 32);

        let state1 = HumanoidState::standing();
        let hv1 = enc.encode(&state1);

        let mut state2 = HumanoidState::standing();
        state2.root_height = 1.35;
        let hv2 = enc.encode(&state2);

        assert_eq!(hv1.dim(), HDC_DIMENSION);
        assert_eq!(hv2.dim(), HDC_DIMENSION);
        let sim = hv1.similarity(&hv2);
        assert!(sim < 0.99, "Derivative should change encoding: sim={sim}");
    }

    #[test]
    fn test_encoder_reset() {
        let genesis = test_genesis();
        let mut enc = HumanoidHdcEncoder::new(&genesis, 32);

        enc.encode(&HumanoidState::standing());
        assert!(enc.prev_channels.is_some());

        enc.reset();
        assert!(enc.prev_channels.is_none());
    }

    #[test]
    fn test_normalize_channel() {
        // root_height range: [0.0, 2.0]
        assert!((HumanoidHdcEncoder::normalize_channel(0, 0.0) - 0.0).abs() < 1e-6);
        assert!((HumanoidHdcEncoder::normalize_channel(0, 2.0) - 1.0).abs() < 1e-6);
        assert!((HumanoidHdcEncoder::normalize_channel(0, 1.0) - 0.5).abs() < 1e-6);
        // Clamped
        assert!((HumanoidHdcEncoder::normalize_channel(0, 5.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_channel_weight_groups() {
        // Root height should be 2.0
        assert!((HumanoidHdcEncoder::channel_weight(0) - 2.0).abs() < 1e-6);
        // Head height should be 2.0
        assert!((HumanoidHdcEncoder::channel_weight(53) - 2.0).abs() < 1e-6);
        // Joint angles should be 1.0
        assert!((HumanoidHdcEncoder::channel_weight(10) - 1.0).abs() < 1e-6);
        // Extremities should be 0.5
        assert!((HumanoidHdcEncoder::channel_weight(60) - 0.5).abs() < 1e-6);
    }
}
