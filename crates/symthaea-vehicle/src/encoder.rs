// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vehicle HDC encoder: sensor state -> ContinuousHV (16,384D).
//!
//! Follows the `HumanoidHdcEncoder` pattern from `symthaea-humanoid/src/encoder.rs`.
//! Each of 20 sensor channels gets a genesis-seeded base vector. Values are level-encoded
//! (thermometer coding) then bound with the base vector. Channels are organized in semantic
//! groups with safety-critical channels weighted higher.
//!
//! The 20-channel "Sensory Schema":
//! - Proprioceptive Core (14): physics state from the bicycle model
//! - Road Vector (2): lateral_offset, curvature_ahead
//! - Swarm Vector (4): nearest_peer_distance, peer_relative_speed, mesh_brake_density, mesh_confidence
//!
//! `mesh_brake_density` gets the highest weight (3.0x) — if the swarm slams on the brakes
//! five cars ahead, that signal physically overpowers the accelerator in hyperdimensional space.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

use crate::types::{NUM_STATE_CHANNELS, VehicleState};

/// Semantic channel groups for weighted encoding.
#[derive(Debug, Clone, Copy)]
struct ChannelGroup {
    start: usize,
    count: usize,
    weight: f32,
}

/// Group definitions matching the `to_channels()` layout (20 channels):
///
/// | Group              | Channels  | Count | Weight | Rationale                          |
/// |--------------------|-----------|-------|--------|------------------------------------|
/// | Speed              | [0]       | 1     | 2.0    | Critical for safety/EFE            |
/// | Lateral velocity   | [1]       | 1     | 2.0    | Stability indicator                |
/// | Yaw rate           | [2]       | 1     | 2.0    | Spin detection                     |
/// | Heading            | [3]       | 1     | 1.5    | Lane tracking                      |
/// | Position           | [4..6]    | 2     | 1.0    | World reference                    |
/// | Steering feedback  | [6]       | 1     | 1.5    | Control state                      |
/// | Throttle/brake     | [7..9]    | 2     | 1.0    | Command feedback                   |
/// | Acceleration       | [9..11]   | 2     | 1.5    | Dynamics feel                      |
/// | Tire slip          | [11..13]  | 2     | 2.5    | CRITICAL: grip limit warning       |
/// | Pitch              | [13]      | 1     | 0.8    | Weight transfer indicator          |
/// | Road: lateral off  | [14]      | 1     | 2.0    | Lane position reference            |
/// | Road: curvature    | [15]      | 1     | 1.5    | Anticipatory steering              |
/// | Mesh: peer dist    | [16]      | 1     | 1.5    | Proximity awareness                |
/// | Mesh: peer speed   | [17]      | 1     | 1.5    | Closing rate                       |
/// | Mesh: brake dens.  | [18]      | 1     | 3.0    | SUPREME: swarm emergency broadcast |
/// | Mesh: confidence   | [19]      | 1     | 1.0    | Signal quality (meta)              |
const CHANNEL_GROUPS: [ChannelGroup; 16] = [
    ChannelGroup {
        start: 0,
        count: 1,
        weight: 2.0,
    }, // speed
    ChannelGroup {
        start: 1,
        count: 1,
        weight: 2.0,
    }, // lateral velocity
    ChannelGroup {
        start: 2,
        count: 1,
        weight: 2.0,
    }, // yaw rate
    ChannelGroup {
        start: 3,
        count: 1,
        weight: 1.5,
    }, // heading
    ChannelGroup {
        start: 4,
        count: 2,
        weight: 1.0,
    }, // position x/y
    ChannelGroup {
        start: 6,
        count: 1,
        weight: 1.5,
    }, // steering angle
    ChannelGroup {
        start: 7,
        count: 2,
        weight: 1.0,
    }, // throttle/brake
    ChannelGroup {
        start: 9,
        count: 2,
        weight: 1.5,
    }, // accel lon/lat
    ChannelGroup {
        start: 11,
        count: 2,
        weight: 2.5,
    }, // tire slip (CRITICAL)
    ChannelGroup {
        start: 13,
        count: 1,
        weight: 0.8,
    }, // pitch
    ChannelGroup {
        start: 14,
        count: 1,
        weight: 2.0,
    }, // road: lateral offset
    ChannelGroup {
        start: 15,
        count: 1,
        weight: 1.5,
    }, // road: curvature ahead
    ChannelGroup {
        start: 16,
        count: 1,
        weight: 1.5,
    }, // mesh: peer distance
    ChannelGroup {
        start: 17,
        count: 1,
        weight: 1.5,
    }, // mesh: peer relative speed
    ChannelGroup {
        start: 18,
        count: 1,
        weight: 3.0,
    }, // mesh: brake density (SUPREME)
    ChannelGroup {
        start: 19,
        count: 1,
        weight: 1.0,
    }, // mesh: confidence
];

/// Channel normalization ranges [min, max].
const CHANNEL_RANGES: [[f32; 2]; NUM_STATE_CHANNELS] = [
    // Proprioceptive (14)
    [0.0, 50.0],     // speed (m/s): 0 to ~180 km/h
    [-5.0, 5.0],     // lateral_velocity (m/s)
    [-2.0, 2.0],     // yaw_rate (rad/s)
    [-3.15, 3.15],   // heading (rad): ~[-pi, pi]
    [-500.0, 500.0], // position_x (m)
    [-50.0, 50.0],   // position_y (m): lane width scale
    [-0.6, 0.6],     // steering_angle (rad): ~±35 deg
    [0.0, 1.0],      // throttle_position
    [0.0, 1.0],      // brake_pressure
    [-15.0, 10.0],   // longitudinal_accel (m/s²): ~[-1.5g, 1g]
    [-15.0, 15.0],   // lateral_accel (m/s²): ~±1.5g
    [0.0, 0.5],      // tire_slip_front (rad)
    [0.0, 0.5],      // tire_slip_rear (rad)
    [-0.15, 0.15],   // pitch (rad): ~±8.5 deg
    // Road (2)
    [-5.0, 5.0],   // lateral_offset (m): ~±1.5 lanes
    [-0.05, 0.05], // curvature_ahead (1/m): R=20m min
    // Mesh (4)
    [0.0, 200.0],  // nearest_peer_distance (m)
    [-30.0, 30.0], // peer_relative_speed (m/s)
    [0.0, 1.0],    // mesh_brake_density (fraction)
    [0.0, 1.0],    // mesh_confidence (fraction)
];

/// HDC encoder for vehicle sensor state.
///
/// Encodes 20D `VehicleState` into a full 16,384D `ContinuousHV` via:
/// 1. Per-channel normalization to [0, 1]
/// 2. Level encoding (thermometer coding via bundled levels 0..k)
/// 3. Binding with genesis-seeded channel base vectors
/// 4. Semantic group weighting (safety-critical channels weighted higher)
/// 5. Bundling all bound channel HVs
/// 6. Optional derivative encoding (rate-of-change channels)
pub struct VehicleHdcEncoder {
    /// Base vectors for each of 20 channels.
    base_vectors: Vec<ContinuousHV>,
    /// Level codebook (num_levels entries).
    level_vectors: Vec<ContinuousHV>,
    /// Base vectors for derivative channels (20 more).
    deriv_base_vectors: Vec<ContinuousHV>,
    /// Previous state channels for derivative computation.
    prev_channels: Option<Vec<f32>>,
    /// Weight for derivative encoding vs position encoding.
    derivative_weight: f32,
    /// Number of levels in the codebook.
    num_levels: usize,
}

impl VehicleHdcEncoder {
    /// Create a new encoder from a genesis seed.
    pub fn new(genesis: &GenesisSeed, num_levels: usize) -> Self {
        let base_vectors: Vec<ContinuousHV> = (0..NUM_STATE_CHANNELS)
            .map(|i| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("vehicle::channel::{i}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        let level_vectors: Vec<ContinuousHV> = (0..num_levels)
            .map(|i| {
                ContinuousHV::from_genesis(genesis, &format!("vehicle::level::{i}"), HDC_DIMENSION)
            })
            .collect();

        let deriv_base_vectors: Vec<ContinuousHV> = (0..NUM_STATE_CHANNELS)
            .map(|i| {
                ContinuousHV::from_genesis(genesis, &format!("vehicle::deriv::{i}"), HDC_DIMENSION)
            })
            .collect();

        Self {
            base_vectors,
            level_vectors,
            deriv_base_vectors,
            prev_channels: None,
            derivative_weight: 0.35,
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

    /// Encode a vehicle state into a full 16,384D ContinuousHV.
    pub fn encode(&mut self, state: &VehicleState) -> ContinuousHV {
        let channels = state.to_channels();
        let n = channels.len();

        let mut bound_hvs: Vec<ContinuousHV> = Vec::with_capacity(n * 2);
        let mut weights: Vec<f32> = Vec::with_capacity(n * 2);

        for i in 0..n {
            let normalized = Self::normalize_channel(i, channels[i]);
            let level_hv = self.encode_level(normalized);
            bound_hvs.push(self.base_vectors[i].bind(&level_hv));
            weights.push(Self::channel_weight(i) * (1.0 - self.derivative_weight));
        }

        // Derivative channels from previous state
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
        GenesisSeed::from_phrase("test-vehicle-encoder")
    }

    #[test]
    fn test_encoder_determinism() {
        let genesis = test_genesis();
        let mut enc1 = VehicleHdcEncoder::new(&genesis, 32);
        let mut enc2 = VehicleHdcEncoder::new(&genesis, 32);

        let state = VehicleState::cruising(13.4);
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
        let mut enc = VehicleHdcEncoder::new(&genesis, 32);
        let hv = enc.encode(&VehicleState::cruising(10.0));
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_encoder_20_channels() {
        let state = VehicleState::cruising(13.4);
        let channels = state.to_channels();
        assert_eq!(channels.len(), 20);
    }

    #[test]
    fn test_encoder_differentiation() {
        let genesis = test_genesis();
        let mut enc = VehicleHdcEncoder::new(&genesis, 32);

        let cruising = VehicleState::cruising(13.4);
        let mut sliding = VehicleState::cruising(13.4);
        sliding.lateral_velocity = 3.0;
        sliding.yaw_rate = 0.8;
        sliding.tire_slip_front = 0.2;
        sliding.tire_slip_rear = 0.3;

        let hv_cruising = enc.encode(&cruising);
        enc.reset();
        let hv_sliding = enc.encode(&sliding);

        let sim = hv_cruising.similarity(&hv_sliding);
        assert!(sim < 0.95, "Different states should differ: sim={sim}");
        assert!(sim > 0.0, "Should share some structure: sim={sim}");
    }

    #[test]
    fn test_encoder_similar_states_high_similarity() {
        let genesis = test_genesis();
        let mut enc = VehicleHdcEncoder::new(&genesis, 32);

        let state1 = VehicleState::cruising(13.4);
        let state2 = VehicleState::cruising(13.5);

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
        let mut enc = VehicleHdcEncoder::new(&genesis, 32);

        let state1 = VehicleState::cruising(13.4);
        let hv1 = enc.encode(&state1);

        let state2 = VehicleState::cruising(14.0);
        let hv2 = enc.encode(&state2);

        assert_eq!(hv1.dim(), HDC_DIMENSION);
        assert_eq!(hv2.dim(), HDC_DIMENSION);
        let sim = hv1.similarity(&hv2);
        assert!(sim < 0.99, "Derivative should change encoding: sim={sim}");
    }

    #[test]
    fn test_encoder_reset() {
        let genesis = test_genesis();
        let mut enc = VehicleHdcEncoder::new(&genesis, 32);

        enc.encode(&VehicleState::cruising(10.0));
        assert!(enc.prev_channels.is_some());

        enc.reset();
        assert!(enc.prev_channels.is_none());
    }

    #[test]
    fn test_normalize_channel() {
        // speed range: [0.0, 50.0]
        assert!((VehicleHdcEncoder::normalize_channel(0, 0.0) - 0.0).abs() < 1e-6);
        assert!((VehicleHdcEncoder::normalize_channel(0, 50.0) - 1.0).abs() < 1e-6);
        assert!((VehicleHdcEncoder::normalize_channel(0, 25.0) - 0.5).abs() < 1e-6);
        // mesh_brake_density range: [0.0, 1.0]
        assert!((VehicleHdcEncoder::normalize_channel(18, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_channel_weight_groups() {
        // Speed should be 2.0
        assert!((VehicleHdcEncoder::channel_weight(0) - 2.0).abs() < 1e-6);
        // Tire slip should be 2.5 (CRITICAL)
        assert!((VehicleHdcEncoder::channel_weight(11) - 2.5).abs() < 1e-6);
        assert!((VehicleHdcEncoder::channel_weight(12) - 2.5).abs() < 1e-6);
        // Road: lateral offset should be 2.0
        assert!((VehicleHdcEncoder::channel_weight(14) - 2.0).abs() < 1e-6);
        // Mesh: brake density should be 3.0 (SUPREME)
        assert!((VehicleHdcEncoder::channel_weight(18) - 3.0).abs() < 1e-6);
        // Mesh: confidence should be 1.0
        assert!((VehicleHdcEncoder::channel_weight(19) - 1.0).abs() < 1e-6);
        // Pitch should be 0.8
        assert!((VehicleHdcEncoder::channel_weight(13) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_mesh_brake_dominates_encoding() {
        let genesis = test_genesis();

        let baseline = VehicleState::cruising(13.4);

        // Swarm emergency: brake density spikes
        let mut brake_state = VehicleState::cruising(13.4);
        brake_state.mesh_brake_density = 0.8;
        brake_state.mesh_confidence = 0.9;

        // Only position differs
        let mut pos_state = VehicleState::cruising(13.4);
        pos_state.position_x = 100.0;

        let mut enc = VehicleHdcEncoder::new(&genesis, 32);
        let hv_base = enc.encode(&baseline);

        enc.reset();
        let hv_brake = enc.encode(&brake_state);

        enc.reset();
        let hv_pos = enc.encode(&pos_state);

        let sim_brake = hv_base.similarity(&hv_brake);
        let sim_pos = hv_base.similarity(&hv_pos);

        // Mesh brake density (weight 3.0) should cause more change than position (weight 1.0)
        assert!(
            sim_brake < sim_pos,
            "Mesh brake density (w=3.0) should affect encoding more than position (w=1.0): brake_sim={sim_brake}, pos_sim={sim_pos}"
        );
    }
}
