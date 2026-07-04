// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Humanoid HDC encoder: sensor state → ContinuousHV (16,384D).
//!
//! Follows the `QuadrotorHdcEncoder` pattern from `symthaea-multirotor/src/encoder.rs`.
//! Each of 72 sensor channels gets a genesis-seeded base vector. Values are level-encoded
//! (thermometer coding) then bound with the base vector. Channels are organized in semantic
//! groups with balance-critical channels weighted higher.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HDC_DIMENSION, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};

use crate::morphology::HumanoidMorphology;
use crate::types::HumanoidState;

// ─── Predictive Encoder Layer ───

/// An HDC-LTC prediction layer that generates emergent prediction error.
///
/// Instead of hacking derivatives as `channels[i] - prev_channels[i]`, this layer:
/// 1. Maintains an internal temporal state via CfC closed-form dynamics
/// 2. PREDICTS what the next encoding should look like (from its state)
/// 3. Receives the actual encoding and computes PE as 1 - cosine_similarity
/// 4. Updates its state via σ-interpolation toward the new equilibrium
/// 5. Adapts tau based on PE magnitude (high surprise → faster response)
///
/// The confidence signal (1 - PE, smoothed) IS the safety input.
pub struct PredictiveLayer {
    /// Small HDC-LTC network (1 layer × 4 neurons) for temporal prediction.
    network: HdcLtcUnifiedNetwork,
    /// The network's output after previous evolve (its prediction for next input).
    last_prediction: Option<ContinuousHV>,
    /// Number of inputs processed (PE valid after >= 2).
    tick_count: u32,
    /// Current prediction error [0, 1].
    pub prediction_error: f32,
    /// Smoothed confidence (1 - PE, exponentially averaged).
    pub confidence: f32,
    /// EMA smoothing factor for confidence.
    confidence_alpha: f32,
}

impl PredictiveLayer {
    /// Create a new predictive layer from a genesis seed.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base: 0.025,   // Match 40Hz physics
            backbone_tau: 0.4, // Moderate state dependency
            dimension: HDC_DIMENSION,
            learning_rate: 0.0, // No learning — pure prediction from dynamics
            ..UnifiedConfig::default()
        };

        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![4], // 1 layer, 4 neurons
            neuron_config,
            use_layer_binding: false,
            skip_connections: false,
        };

        Self {
            network: HdcLtcUnifiedNetwork::from_genesis(net_config, genesis),
            last_prediction: None,
            tick_count: 0,
            prediction_error: 0.0,
            confidence: 1.0,
            confidence_alpha: 0.1,
        }
    }

    /// Process a new encoding through the predictive layer.
    ///
    /// Returns a temporally-smoothed output HV. Updates prediction_error and confidence.
    pub fn process(&mut self, raw_hv: &ContinuousHV, dt: f32) -> ContinuousHV {
        self.tick_count += 1;

        // Step 1: Compute PE from consecutive RAW encodings.
        // PE = 1 - similarity(this_raw, last_raw). When the world changes,
        // the raw encoding changes, and PE spikes. This is the true
        // sensorimotor prediction error — "how different is this observation
        // from what I just saw?"
        if self.tick_count >= 2
            && let Some(ref prev_raw) = self.last_prediction
        {
            let sim = prev_raw.similarity(raw_hv).clamp(0.0, 1.0);
            self.prediction_error = 1.0 - sim;
        }

        // Step 2: Store this raw encoding for next tick's PE comparison
        self.last_prediction = Some(raw_hv.clone());

        // Step 3: Evolve the LTC network with the raw input.
        // The network provides temporal smoothing — its output is a
        // filtered version of the input that resists noise and captures
        // temporal dynamics through state-dependent tau.
        self.network.evolve_closed_form(dt, raw_hv);

        // Step 4: Update confidence (EMA of 1-PE)
        let target_confidence = 1.0 - self.prediction_error;
        self.confidence = self.confidence * (1.0 - self.confidence_alpha)
            + target_confidence * self.confidence_alpha;

        // Step 5: Return the temporally-smoothed output from the LTC network
        self.network.output()
    }

    /// Reset the predictive layer state.
    pub fn reset(&mut self) {
        self.network.reset();
        self.last_prediction = None;
        self.tick_count = 0;
        self.prediction_error = 0.0;
        self.confidence = 1.0;
    }
}

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

/// Dynamic channel layout for a specific morphology.
///
/// Encapsulates ranges, groups, and total channel count. Built by
/// `ChannelLayout::for_morphology()`.
struct ChannelLayout {
    num_channels: usize,
    ranges: Vec<[f32; 2]>,
    groups: Vec<ChannelGroup>,
}

impl ChannelLayout {
    /// Build layout for a specific morphology.
    ///
    /// The layout matches `HumanoidState::to_channels()` output order:
    /// [0] root_height, [1..5] quat, [5..5+n] angles, [5+n..8+n] lin_vel,
    /// [8+n..11+n] ang_vel, [11+n..11+2n] velocities, [11+2n] head_height,
    /// [12+2n..15+2n] torso_vert, [15+2n..15+2n+ext] extremities, [..] com_vel
    fn for_morphology(morph: HumanoidMorphology) -> Self {
        let n_joints = morph.num_actuators();
        let n_extremities = if n_joints > 21 { 18 } else { 12 };

        // Build ranges following to_channels() order
        let mut ranges: Vec<[f32; 2]> = Vec::new();
        let mut groups: Vec<ChannelGroup> = Vec::new();

        // [0] root_height
        let root_h_start = 0;
        ranges.push([0.0, 2.0]);

        // [1..5] root quaternion
        let quat_start = 1;
        for _ in 0..4 {
            ranges.push([-1.0, 1.0]);
        }

        // [5..5+n] joint angles
        let angles_start = 5;
        // Base 21 DMC21 angle ranges
        let dmc21_angle_ranges: [[f32; 2]; 21] = [
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-2.0, 2.0], // abdomen
            [-1.5, 1.5],
            [-1.5, 1.5],
            [-2.0, 0.5],
            [-2.5, 0.0], // right leg
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.5, 1.5],
            [-1.5, 1.5],
            [-2.0, 0.5],
            [-2.5, 0.0], // left leg
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-2.5, 0.0], // right arm
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-2.5, 0.0], // left arm
        ];
        for r in &dmc21_angle_ranges {
            ranges.push(*r);
        }
        // Extra hand/neck/wrist/spine joint angles
        for _ in 21..n_joints {
            ranges.push([-0.35, 1.75]);
        }

        // [5+n..8+n] root linear velocity
        let lin_vel_start = 5 + n_joints;
        for _ in 0..3 {
            ranges.push([-10.0, 10.0]);
        }

        // [8+n..11+n] root angular velocity
        let ang_vel_start = 8 + n_joints;
        for _ in 0..3 {
            ranges.push([-20.0, 20.0]);
        }

        // [11+n..11+2n] joint velocities
        let vel_start = 11 + n_joints;
        for _ in 0..n_joints {
            ranges.push([-20.0, 20.0]);
        }

        // [11+2n] head height
        let head_h_start = 11 + 2 * n_joints;
        ranges.push([0.0, 2.0]);

        // [12+2n..15+2n] torso vertical
        let torso_start = 12 + 2 * n_joints;
        for _ in 0..3 {
            ranges.push([-1.0, 1.0]);
        }

        // [15+2n..15+2n+ext] extremities
        let ext_start = 15 + 2 * n_joints;
        for i in 0..n_extremities {
            // z-component of feet has tighter range
            let is_foot_z = (i == 8 || i == 11) && n_extremities >= 12;
            if is_foot_z {
                ranges.push([0.0, 0.5]);
            } else if i % 3 == 2 {
                // z components of hands
                ranges.push([0.0, 2.0]);
            } else {
                ranges.push([-2.0, 2.0]);
            }
        }

        // [15+2n+ext..18+2n+ext] COM velocity
        let com_start = 15 + 2 * n_joints + n_extremities;
        for _ in 0..3 {
            ranges.push([-10.0, 10.0]);
        }

        let num_channels = ranges.len();

        // Semantic groups
        groups.push(ChannelGroup {
            start: root_h_start,
            count: 1,
            weight: 2.0,
        });
        groups.push(ChannelGroup {
            start: quat_start,
            count: 4,
            weight: 1.5,
        });

        if n_joints <= 21 {
            groups.push(ChannelGroup {
                start: angles_start,
                count: n_joints,
                weight: 1.0,
            });
        } else {
            // Split: base body angles vs hand/extra angles
            groups.push(ChannelGroup {
                start: angles_start,
                count: 21,
                weight: 1.0,
            });
            groups.push(ChannelGroup {
                start: angles_start + 21,
                count: n_joints - 21,
                weight: 1.5,
            }); // Hand angles weighted higher
        }

        groups.push(ChannelGroup {
            start: lin_vel_start,
            count: 3,
            weight: 1.5,
        });
        groups.push(ChannelGroup {
            start: ang_vel_start,
            count: 3,
            weight: 1.5,
        });

        if n_joints <= 21 {
            groups.push(ChannelGroup {
                start: vel_start,
                count: n_joints,
                weight: 0.8,
            });
        } else {
            groups.push(ChannelGroup {
                start: vel_start,
                count: 21,
                weight: 0.8,
            });
            groups.push(ChannelGroup {
                start: vel_start + 21,
                count: n_joints - 21,
                weight: 0.8,
            });
        }

        groups.push(ChannelGroup {
            start: head_h_start,
            count: 1,
            weight: 2.0,
        });
        groups.push(ChannelGroup {
            start: torso_start,
            count: 3,
            weight: 1.5,
        });
        groups.push(ChannelGroup {
            start: ext_start,
            count: n_extremities,
            weight: 0.5,
        });
        groups.push(ChannelGroup {
            start: com_start,
            count: 3,
            weight: 1.0,
        });

        // Extra hand centroid features get moderate weight
        if n_extremities > 12 {
            // Override the extremities group to split base + hand centroids
            let last = groups.len() - 2; // extremities group
            groups[last] = ChannelGroup {
                start: ext_start,
                count: 12,
                weight: 0.5,
            };
            groups.push(ChannelGroup {
                start: ext_start + 12,
                count: n_extremities - 12,
                weight: 1.0,
            });
        }

        Self {
            num_channels,
            ranges,
            groups,
        }
    }
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
#[allow(dead_code)]
const DMC21_CHANNEL_GROUPS: [ChannelGroup; 10] = [
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

/// Total number of channels for DMC21 (backward-compat constant).
#[allow(dead_code)]
const DMC21_NUM_CHANNELS: usize = 72;

/// Channel normalization ranges for DMC21 [min, max].
#[allow(dead_code)]
const DMC21_CHANNEL_RANGES: [[f32; 2]; DMC21_NUM_CHANNELS] = [
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
    /// Channel layout (ranges, groups, count) — morphology-specific.
    layout: ChannelLayout,
    /// Base vectors for each channel.
    base_vectors: Vec<ContinuousHV>,
    /// Level codebook (num_levels entries).
    #[allow(dead_code)]
    level_vectors: Vec<ContinuousHV>,
    /// Base vectors for derivative channels.
    deriv_base_vectors: Vec<ContinuousHV>,
    /// Previous state channels for derivative computation (legacy, used when no predictive layer).
    prev_channels: Option<Vec<f32>>,
    /// Weight for derivative encoding vs position encoding.
    derivative_weight: f32,
    /// Number of levels in the codebook.
    num_levels: usize,
    /// Optional predictive layer for emergent prediction error.
    predictive: Option<PredictiveLayer>,
}

impl HumanoidHdcEncoder {
    /// Create a new encoder for DMC21 morphology (backward compatible).
    pub fn new(genesis: &GenesisSeed, num_levels: usize) -> Self {
        Self::new_for(genesis, num_levels, HumanoidMorphology::Dmc21)
    }

    /// Create an encoder for a specific morphology.
    ///
    /// Genesis seed format `"humanoid::channel::{i}"` is identical for channels 0-71
    /// across all morphologies. Channels 72+ get new seeds that don't collide.
    /// This means: for Dmc21, encoding output is bit-identical to `new()`.
    pub fn new_for(
        genesis: &GenesisSeed,
        num_levels: usize,
        morphology: HumanoidMorphology,
    ) -> Self {
        let layout = ChannelLayout::for_morphology(morphology);
        let n = layout.num_channels;

        let base_vectors: Vec<ContinuousHV> = (0..n)
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

        let deriv_base_vectors: Vec<ContinuousHV> = (0..n)
            .map(|i| {
                ContinuousHV::from_genesis(genesis, &format!("humanoid::deriv::{i}"), HDC_DIMENSION)
            })
            .collect();

        Self {
            layout,
            base_vectors,
            level_vectors,
            deriv_base_vectors,
            prev_channels: None,
            derivative_weight: 0.3,
            num_levels,
            predictive: None,
        }
    }

    /// Create an encoder WITH a predictive HDC-LTC layer for emergent prediction error.
    ///
    /// The predictive layer replaces the derivative hack with a temporal CfC network
    /// that generates PE from cosine dissimilarity between its prediction and reality.
    /// Use `prediction_error()` and `confidence()` to read the signals.
    pub fn new_predictive(
        genesis: &GenesisSeed,
        num_levels: usize,
        morphology: HumanoidMorphology,
    ) -> Self {
        let mut enc = Self::new_for(genesis, num_levels, morphology);
        let pred_phrase = format!("{}::predictive_encoder", genesis.phrase());
        let pred_genesis = GenesisSeed::from_phrase(&pred_phrase);
        enc.predictive = Some(PredictiveLayer::new(&pred_genesis));
        enc
    }

    /// Normalize a raw channel value to [0, 1] using morphology-specific ranges.
    fn normalize_channel(&self, channel: usize, value: f32) -> f32 {
        let [min, max] = if channel < self.layout.ranges.len() {
            self.layout.ranges[channel]
        } else {
            [-2.0, 2.0] // Fallback for unknown channels
        };
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Level-encode a normalized [0,1] value using thermometer coding.
    /// Returns a new HV (allocates). Use `encode_level_into` for zero-alloc path.
    #[allow(dead_code)]
    fn encode_level(&self, normalized: f32) -> ContinuousHV {
        let k = ((normalized * self.num_levels as f32) as usize).min(self.num_levels - 1);
        if k == 0 {
            self.level_vectors[0].clone()
        } else {
            let refs: Vec<&ContinuousHV> = self.level_vectors[..=k].iter().collect();
            ContinuousHV::bundle(&refs)
        }
    }

    /// Level-encode into a pre-allocated buffer (zero allocation).
    /// Accumulates level vectors 0..=k directly into `buf`.
    #[allow(dead_code)]
    fn encode_level_into(&self, normalized: f32, buf: &mut [f32]) {
        let k = ((normalized * self.num_levels as f32) as usize).min(self.num_levels - 1);
        let dim = buf.len();
        // Zero the buffer
        for v in buf.iter_mut() {
            *v = 0.0;
        }
        // Accumulate level vectors 0..=k
        for l in 0..=k {
            let lv = self.level_vectors[l].as_slice();
            for j in 0..dim.min(lv.len()) {
                buf[j] += lv[j];
            }
        }
        // Normalize
        let norm: f32 = buf.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 {
            let inv = 1.0 / norm;
            for v in buf.iter_mut() {
                *v *= inv;
            }
        }
    }

    /// Get the semantic group weight for a given channel index.
    fn channel_weight(&self, channel: usize) -> f32 {
        for group in &self.layout.groups {
            if channel >= group.start && channel < group.start + group.count {
                return group.weight;
            }
        }
        1.0
    }

    /// Encode a humanoid state into a full 16,384D ContinuousHV.
    ///
    /// Fast path: uses scalar-weighted binding instead of full thermometer coding.
    /// Each channel contributes: `weight × normalized × base_vector[i]`.
    /// This is O(n_channels × D) instead of O(n_channels × n_levels × D).
    ///
    /// For 72 channels at D=16384: ~1.2M ops vs ~19M ops (16x faster).
    pub fn encode(&mut self, state: &HumanoidState) -> ContinuousHV {
        let channels = state.to_channels();
        let n = channels.len().min(self.base_vectors.len());
        let dim = HDC_DIMENSION;

        let mut result = vec![0.0f32; dim];

        // Fast encoding: weight × normalized_value × base_vector
        // This preserves the semantic weighting and channel binding properties
        // while avoiding the O(levels) thermometer accumulation per channel.
        for (i, &c) in channels.iter().enumerate().take(n) {
            let normalized = self.normalize_channel(i, c);
            let weight = self.channel_weight(i) * (1.0 - self.derivative_weight);
            // Scalar-bind: the normalized value [0,1] scales the base vector contribution.
            // Higher values produce stronger contribution (same effect as thermometer).
            let scale = weight * (normalized * 2.0 - 1.0); // Map [0,1] to [-1,1] for bipolar
            let base = self.base_vectors[i].as_slice();
            for j in 0..dim {
                result[j] += scale * base[j];
            }
        }

        // Derivative channels (skip if predictive layer handles temporal dynamics)
        if self.predictive.is_none()
            && let Some(ref prev) = self.prev_channels
        {
            let prev_n = n.min(prev.len());
            for i in 0..prev_n {
                let delta = channels[i] - prev[i];
                let [min, max] = if i < self.layout.ranges.len() {
                    self.layout.ranges[i]
                } else {
                    [-2.0, 2.0]
                };
                let range = max - min;
                let normalized_delta = ((delta / range) + 0.5).clamp(0.0, 1.0);
                let weight = self.channel_weight(i) * self.derivative_weight;
                let scale = weight * (normalized_delta * 2.0 - 1.0);
                let deriv_base = self.deriv_base_vectors[i].as_slice();
                for j in 0..dim {
                    result[j] += scale * deriv_base[j];
                }
            }
        }

        self.prev_channels = Some(channels);

        // Normalize the result
        let norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 {
            let inv = 1.0 / norm;
            for v in result.iter_mut() {
                *v *= inv;
            }
        }

        let raw_hv = ContinuousHV::from_vec(result);

        // If predictive layer exists, route through it for temporal smoothing + PE
        if let Some(ref mut pred) = self.predictive {
            pred.process(&raw_hv, 0.025) // dt=0.025 (40Hz default)
        } else {
            raw_hv
        }
    }

    /// Encode with explicit dt for the predictive layer.
    pub fn encode_with_dt(&mut self, state: &HumanoidState, dt: f32) -> ContinuousHV {
        let channels = state.to_channels();
        let n = channels.len().min(self.base_vectors.len());
        let dim = HDC_DIMENSION;

        let mut result = vec![0.0f32; dim];

        for (i, &c) in channels.iter().enumerate().take(n) {
            let normalized = self.normalize_channel(i, c);
            let weight = self.channel_weight(i) * (1.0 - self.derivative_weight);
            let scale = weight * (normalized * 2.0 - 1.0);
            let base = self.base_vectors[i].as_slice();
            for j in 0..dim {
                result[j] += scale * base[j];
            }
        }

        if self.predictive.is_none()
            && let Some(ref prev) = self.prev_channels
        {
            let prev_n = n.min(prev.len());
            for i in 0..prev_n {
                let delta = channels[i] - prev[i];
                let [min, max] = if i < self.layout.ranges.len() {
                    self.layout.ranges[i]
                } else {
                    [-2.0, 2.0]
                };
                let range = max - min;
                let normalized_delta = ((delta / range) + 0.5).clamp(0.0, 1.0);
                let weight = self.channel_weight(i) * self.derivative_weight;
                let scale = weight * (normalized_delta * 2.0 - 1.0);
                let deriv_base = self.deriv_base_vectors[i].as_slice();
                for j in 0..dim {
                    result[j] += scale * deriv_base[j];
                }
            }
        }

        self.prev_channels = Some(channels);
        let norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 {
            let inv = 1.0 / norm;
            for v in result.iter_mut() {
                *v *= inv;
            }
        }

        let raw_hv = ContinuousHV::from_vec(result);
        if let Some(ref mut pred) = self.predictive {
            pred.process(&raw_hv, dt)
        } else {
            raw_hv
        }
    }

    /// Reset the encoder state (clear derivative memory + predictive layer).
    pub fn reset(&mut self) {
        self.prev_channels = None;
        if let Some(ref mut pred) = self.predictive {
            pred.reset();
        }
    }

    /// Number of levels in the codebook.
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    /// Current prediction error from the predictive layer [0, 1].
    /// Returns 0.0 if no predictive layer is active.
    pub fn prediction_error(&self) -> f32 {
        self.predictive
            .as_ref()
            .map(|p| p.prediction_error)
            .unwrap_or(0.0)
    }

    /// Current confidence (1 - smoothed PE) from the predictive layer [0, 1].
    /// Returns 1.0 if no predictive layer is active.
    pub fn confidence(&self) -> f32 {
        self.predictive
            .as_ref()
            .map(|p| p.confidence)
            .unwrap_or(1.0)
    }

    /// Whether this encoder has a predictive layer.
    pub fn has_predictive_layer(&self) -> bool {
        self.predictive.is_some()
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
        assert!(sim < 0.9999, "Derivative should change encoding: sim={sim}");
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
        let genesis = test_genesis();
        let enc = HumanoidHdcEncoder::new(&genesis, 32);
        // root_height range: [0.0, 2.0]
        assert!((enc.normalize_channel(0, 0.0) - 0.0).abs() < 1e-6);
        assert!((enc.normalize_channel(0, 2.0) - 1.0).abs() < 1e-6);
        assert!((enc.normalize_channel(0, 1.0) - 0.5).abs() < 1e-6);
        // Clamped
        assert!((enc.normalize_channel(0, 5.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_channel_weight_groups() {
        let genesis = test_genesis();
        let enc = HumanoidHdcEncoder::new(&genesis, 32);
        // Root height should be 2.0
        assert!((enc.channel_weight(0) - 2.0).abs() < 1e-6);
        // Head height should be 2.0 (index 53 in DMC21)
        assert!((enc.channel_weight(53) - 2.0).abs() < 1e-6);
        // Joint angles should be 1.0
        assert!((enc.channel_weight(10) - 1.0).abs() < 1e-6);
        // Extremities should be 0.5
        assert!((enc.channel_weight(60) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_encoder_dexterous53_dimension() {
        use crate::morphology::HumanoidMorphology;
        let genesis = test_genesis();
        let mut enc = HumanoidHdcEncoder::new_for(&genesis, 32, HumanoidMorphology::Dexterous53);
        let state = HumanoidState::standing_for(HumanoidMorphology::Dexterous53);
        let hv = enc.encode(&state);
        assert_eq!(hv.dim(), HDC_DIMENSION, "Should produce 16384D HV");
    }

    #[test]
    fn test_encoder_dexterous53_determinism() {
        use crate::morphology::HumanoidMorphology;
        let genesis = test_genesis();
        let mut enc1 = HumanoidHdcEncoder::new_for(&genesis, 32, HumanoidMorphology::Dexterous53);
        let mut enc2 = HumanoidHdcEncoder::new_for(&genesis, 32, HumanoidMorphology::Dexterous53);
        let state = HumanoidState::standing_for(HumanoidMorphology::Dexterous53);
        let hv1 = enc1.encode(&state);
        let hv2 = enc2.encode(&state);
        assert!(
            (hv1.similarity(&hv2) - 1.0).abs() < 1e-5,
            "Same genesis+morph = identical"
        );
    }

    #[test]
    fn test_encoder_hand_channels_weighted() {
        use crate::morphology::HumanoidMorphology;
        let genesis = test_genesis();
        let enc = HumanoidHdcEncoder::new_for(&genesis, 32, HumanoidMorphology::Dexterous53);
        // In Dexterous53, hand angles start at channel index 5+21 = 26
        // (they're interleaved in the joint_angles section)
        let hand_angle_ch = 5 + 21; // first hand angle channel
        let weight = enc.channel_weight(hand_angle_ch);
        assert!(
            (weight - 1.5).abs() < 1e-6,
            "Hand angle weight should be 1.5, got {weight}"
        );
    }

    // ── Predictive Layer Tests ──

    #[test]
    fn test_predictive_encoder_creates() {
        let genesis = test_genesis();
        let enc = HumanoidHdcEncoder::new_predictive(&genesis, 32, HumanoidMorphology::Dmc21);
        assert!(enc.has_predictive_layer());
        assert_eq!(enc.prediction_error(), 0.0);
        assert!((enc.confidence() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_predictive_encoder_pe_rises_on_change() {
        let genesis = test_genesis();
        let mut enc = HumanoidHdcEncoder::new_predictive(&genesis, 32, HumanoidMorphology::Dmc21);

        // Warm up: encode same state several times to build stable internal state
        let state1 = HumanoidState::standing();
        for _ in 0..10 {
            enc.encode(&state1);
        }
        let pe_stable = enc.prediction_error();

        // Now introduce a dramatic change → PE should spike
        let mut state2 = HumanoidState::standing();
        state2.root_height = 0.5; // Fallen
        state2.head_height = 0.6;
        state2.torso_vertical = [0.7, 0.0, 0.7];
        state2.root_angular_velocity = [5.0, -3.0, 2.0];
        enc.encode(&state2);
        let pe_changed = enc.prediction_error();

        assert!(
            pe_changed > pe_stable,
            "PE should rise on state change: stable={pe_stable}, changed={pe_changed}"
        );
    }

    #[test]
    fn test_predictive_encoder_confidence_drops_on_surprise() {
        let genesis = test_genesis();
        let mut enc = HumanoidHdcEncoder::new_predictive(&genesis, 32, HumanoidMorphology::Dmc21);

        // Several encodes of the same state → confidence should be high
        let state = HumanoidState::standing();
        for _ in 0..10 {
            enc.encode(&state);
        }
        let conf_stable = enc.confidence();

        // Sudden change → confidence should drop
        let mut fallen = HumanoidState::standing();
        fallen.root_height = 0.3;
        fallen.head_height = 0.4;
        fallen.torso_vertical = [0.7, 0.0, 0.7];
        enc.encode(&fallen);
        let conf_surprised = enc.confidence();

        assert!(
            conf_surprised < conf_stable,
            "Confidence should drop on surprise: stable={conf_stable}, surprised={conf_surprised}"
        );
    }

    #[test]
    fn test_predictive_encoder_backward_compat() {
        let genesis = test_genesis();
        let mut enc_legacy = HumanoidHdcEncoder::new(&genesis, 32);
        let enc_pred = HumanoidHdcEncoder::new_predictive(&genesis, 32, HumanoidMorphology::Dmc21);

        // Legacy encoder should NOT have predictive layer
        assert!(!enc_legacy.has_predictive_layer());
        assert_eq!(enc_legacy.prediction_error(), 0.0);
        assert!((enc_legacy.confidence() - 1.0).abs() < 0.01);

        // Legacy encode should still work
        let state = HumanoidState::standing();
        let hv = enc_legacy.encode(&state);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_predictive_encoder_reset() {
        let genesis = test_genesis();
        let mut enc = HumanoidHdcEncoder::new_predictive(&genesis, 32, HumanoidMorphology::Dmc21);

        // Encode a few states to build up PE
        let state = HumanoidState::standing();
        enc.encode(&state);
        let mut fallen = HumanoidState::standing();
        fallen.root_height = 0.3;
        enc.encode(&fallen);
        assert!(enc.prediction_error() > 0.0);

        // Reset should clear everything
        enc.reset();
        assert_eq!(enc.prediction_error(), 0.0);
        assert!((enc.confidence() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_channel_layout_all_morphologies() {
        use crate::morphology::HumanoidMorphology;
        for morph in [
            HumanoidMorphology::Dmc21,
            HumanoidMorphology::Dexterous53,
            HumanoidMorphology::WithNeckWrist,
            HumanoidMorphology::FullSpine,
        ] {
            let layout = ChannelLayout::for_morphology(morph);
            let state = HumanoidState::standing_for(morph);
            let channels = state.to_channels();
            assert_eq!(
                layout.num_channels,
                channels.len(),
                "{morph:?}: layout says {} channels but to_channels() produces {}",
                layout.num_channels,
                channels.len()
            );
        }
    }
}
