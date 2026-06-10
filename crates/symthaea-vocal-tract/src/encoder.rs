// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vocal tract HDC encoder: cognitive voice state → ContinuousHV (16,384D).
//!
//! Follows the `QuadrotorHdcEncoder` pattern from `crates/symthaea-multirotor/src/encoder.rs`.
//! Each of 12 cognitive voice channels gets a genesis-seeded base vector. Values are
//! level-encoded (thermometer coding) then bound with the base vector. The result is
//! bundled into a single 16,384D ContinuousHV. Derivative encoding provides temporal context.
//!
//! # 12-Channel Cognitive Voice State
//!
//! | Channel              | Range    | Description                              |
//! |----------------------|----------|------------------------------------------|
//! | prediction_error     | [0, 2]   | Cognitive loop prediction error          |
//! | emotional_valence    | [-1, 1]  | Emotional valence (negative to positive) |
//! | emotional_arousal    | [0, 1]   | Emotional arousal level                  |
//! | unified_quality      | [0, 1]   | Phase 16 unified quality score           |
//! | epistemic_confidence | [0, 1]   | Phase 16 epistemic gate confidence       |
//! | coherence_velocity   | [-1, 1]  | Rate of coherence change                 |
//! | cross_agreement      | [0, 1]   | Cross-module agreement                   |
//! | consciousness_level  | [0, 1]   | Master consciousness level               |
//! | articulation_quality | [0, 1]   | Voice feedback articulation score        |
//! | rate_stability       | [0, 1]   | Speech rate stability                    |
//! | integrated_phi       | [0, 2]   | IIT integrated information (Phi)         |
//! | expected_free_energy | [0, 5]   | Active inference expected free energy     |

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

/// Number of cognitive voice channels.
pub const NUM_CHANNELS: usize = 12;

/// Channel names for genesis seeding.
const CHANNEL_NAMES: [&str; NUM_CHANNELS] = [
    "prediction_error",
    "emotional_valence",
    "emotional_arousal",
    "unified_quality",
    "epistemic_confidence",
    "coherence_velocity",
    "cross_agreement",
    "consciousness_level",
    "articulation_quality",
    "rate_stability",
    "integrated_phi",
    "expected_free_energy",
];

/// Channel ranges [min, max] for normalization to [0, 1].
const CHANNEL_RANGES: [[f32; 2]; NUM_CHANNELS] = [
    [0.0, 2.0],  // prediction_error
    [-1.0, 1.0], // emotional_valence
    [0.0, 1.0],  // emotional_arousal
    [0.0, 1.0],  // unified_quality
    [0.0, 1.0],  // epistemic_confidence
    [-1.0, 1.0], // coherence_velocity
    [0.0, 1.0],  // cross_agreement
    [0.0, 1.0],  // consciousness_level
    [0.0, 1.0],  // articulation_quality
    [0.0, 1.0],  // rate_stability
    [0.0, 2.0],  // integrated_phi
    [0.0, 5.0],  // expected_free_energy
];

/// Cognitive voice state: 12 channels driving the vocal tract controller.
#[derive(Debug, Clone, Copy)]
pub struct VoiceCognitiveState {
    pub prediction_error: f32,
    pub emotional_valence: f32,
    pub emotional_arousal: f32,
    pub unified_quality: f32,
    pub epistemic_confidence: f32,
    pub coherence_velocity: f32,
    pub cross_agreement: f32,
    pub consciousness_level: f32,
    pub articulation_quality: f32,
    pub rate_stability: f32,
    /// IIT integrated information (Phi). Higher = more integrated consciousness.
    /// Maps to richer prosody: higher Phi → more expressive F0.
    pub integrated_phi: f32,
    /// Active inference expected free energy. Higher = more surprise/uncertainty.
    /// Maps to deliberate speech: higher EFE → reduced energy, slight F0 drop.
    pub expected_free_energy: f32,
}

impl Default for VoiceCognitiveState {
    fn default() -> Self {
        Self {
            prediction_error: 0.0,
            emotional_valence: 0.0,
            emotional_arousal: 0.5,
            unified_quality: 1.0,
            epistemic_confidence: 0.5,
            coherence_velocity: 0.0,
            cross_agreement: 1.0,
            consciousness_level: 0.5,
            articulation_quality: 0.8,
            rate_stability: 0.8,
            integrated_phi: 0.5,
            expected_free_energy: 1.0,
        }
    }
}

impl VoiceCognitiveState {
    /// Convert to a flat array of channel values.
    pub fn to_channels(&self) -> [f32; NUM_CHANNELS] {
        [
            self.prediction_error,
            self.emotional_valence,
            self.emotional_arousal,
            self.unified_quality,
            self.epistemic_confidence,
            self.coherence_velocity,
            self.cross_agreement,
            self.consciousness_level,
            self.articulation_quality,
            self.rate_stability,
            self.integrated_phi,
            self.expected_free_energy,
        ]
    }
}

/// Per-second rate of change of key cognitive voice state channels.
///
/// Used by the extended voice quality profile to detect rapid emotional shifts
/// and modulate vocal quality accordingly (e.g., rising arousal → voice strain).
#[derive(Debug, Clone, Copy, Default)]
pub struct VoiceCognitiveStateDerivatives {
    /// Rate of change of emotional arousal (per second).
    pub delta_arousal: f32,
    /// Rate of change of emotional valence (per second).
    pub delta_valence: f32,
    /// Rate of change of consciousness level (per second).
    pub delta_consciousness: f32,
}

/// HDC encoder for cognitive voice state.
///
/// Encodes a 12D `VoiceCognitiveState` into a full 16,384D `ContinuousHV` via:
/// 1. Per-channel normalization to [0, 1] using known ranges
/// 2. Level encoding (thermometer coding via bundled levels 0..k)
/// 3. Binding with genesis-seeded channel base vectors
/// 4. Bundling all 10 bound channel HVs
/// 5. Optional derivative encoding (rate-of-change channels)
pub struct VocalTractHdcEncoder {
    /// Base vectors for each of 10 channels.
    base_vectors: Vec<ContinuousHV>,
    /// Level codebook (num_levels entries).
    level_vectors: Vec<ContinuousHV>,
    /// Base vectors for derivative channels (10 more).
    deriv_base_vectors: Vec<ContinuousHV>,
    /// Previous state for derivative computation.
    prev_channels: Option<[f32; NUM_CHANNELS]>,
    /// Weight for derivative encoding vs position encoding.
    derivative_weight: f32,
    /// Number of levels in the codebook.
    num_levels: usize,
}

impl VocalTractHdcEncoder {
    /// Create a new encoder from a genesis seed.
    pub fn new(genesis: &GenesisSeed, num_levels: usize) -> Self {
        let base_vectors: Vec<ContinuousHV> = CHANNEL_NAMES
            .iter()
            .map(|name| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("vocal_tract::channel::{name}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        let level_vectors: Vec<ContinuousHV> = (0..num_levels)
            .map(|i| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("vocal_tract::level::{i}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        let deriv_base_vectors: Vec<ContinuousHV> = CHANNEL_NAMES
            .iter()
            .map(|name| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("vocal_tract::deriv::{name}"),
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
    pub fn normalize_channel(channel: usize, value: f32) -> f32 {
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

    /// Encode a cognitive voice state into a full 16,384D ContinuousHV.
    pub fn encode(&mut self, state: &VoiceCognitiveState) -> ContinuousHV {
        let channels = state.to_channels();

        // Encode position channels: base_i ⊗ level(value_i)
        let mut bound_hvs: Vec<ContinuousHV> = Vec::with_capacity(NUM_CHANNELS * 2);
        for i in 0..NUM_CHANNELS {
            let normalized = Self::normalize_channel(i, channels[i]);
            let level_hv = self.encode_level(normalized);
            bound_hvs.push(self.base_vectors[i].bind(&level_hv));
        }

        // Encode derivative channels if we have a previous state
        if let Some(prev) = &self.prev_channels {
            for i in 0..NUM_CHANNELS {
                let delta = channels[i] - prev[i];
                let range = CHANNEL_RANGES[i][1] - CHANNEL_RANGES[i][0];
                let normalized_delta = ((delta / range) + 0.5).clamp(0.0, 1.0);
                let level_hv = self.encode_level(normalized_delta);
                bound_hvs.push(self.deriv_base_vectors[i].bind(&level_hv));
            }
        }

        self.prev_channels = Some(channels);

        // Bundle all with appropriate weights
        if bound_hvs.len() > NUM_CHANNELS {
            let pos_weight = 1.0 - self.derivative_weight;
            let deriv_weight = self.derivative_weight;
            let mut weights = vec![pos_weight; NUM_CHANNELS];
            weights.extend(vec![deriv_weight; NUM_CHANNELS]);
            let refs: Vec<&ContinuousHV> = bound_hvs.iter().collect();
            ContinuousHV::weighted_bundle(&refs, &weights)
        } else {
            let refs: Vec<&ContinuousHV> = bound_hvs.iter().collect();
            ContinuousHV::bundle(&refs)
        }
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
        GenesisSeed::from_phrase("test-vocal-tract-encoder")
    }

    #[test]
    fn test_encoder_determinism() {
        let genesis = test_genesis();
        let mut enc1 = VocalTractHdcEncoder::new(&genesis, 32);
        let mut enc2 = VocalTractHdcEncoder::new(&genesis, 32);

        let state = VoiceCognitiveState::default();
        let hv1 = enc1.encode(&state);
        let hv2 = enc2.encode(&state);

        assert_eq!(hv1.dim(), HDC_DIMENSION);
        assert!(
            (hv1.similarity(&hv2) - 1.0).abs() < 1e-5,
            "Same genesis → identical encoding"
        );
    }

    #[test]
    fn test_encoder_sensitivity() {
        let genesis = test_genesis();
        let mut enc = VocalTractHdcEncoder::new(&genesis, 32);

        let calm = VoiceCognitiveState {
            emotional_arousal: 0.1,
            emotional_valence: 0.2,
            ..Default::default()
        };
        let excited = VoiceCognitiveState {
            emotional_arousal: 0.9,
            emotional_valence: 0.8,
            ..Default::default()
        };

        let hv_calm = enc.encode(&calm);
        enc.reset();
        let hv_excited = enc.encode(&excited);

        let sim = hv_calm.similarity(&hv_excited);
        assert!(
            sim < 0.95,
            "Different emotional states should produce different HVs: sim={sim}"
        );
        assert!(sim > 0.0, "Should share structure: sim={sim}");
    }

    #[test]
    fn test_encoder_derivative() {
        let genesis = test_genesis();
        let mut enc = VocalTractHdcEncoder::new(&genesis, 32);

        let state1 = VoiceCognitiveState::default();
        let hv1 = enc.encode(&state1);

        let state2 = VoiceCognitiveState {
            prediction_error: 0.5,
            emotional_arousal: 0.7,
            ..Default::default()
        };
        let hv2 = enc.encode(&state2);

        assert_eq!(hv1.dim(), HDC_DIMENSION);
        assert_eq!(hv2.dim(), HDC_DIMENSION);
        let sim = hv1.similarity(&hv2);
        assert!(sim < 0.99, "Derivative should change encoding: sim={sim}");
    }

    #[test]
    fn test_encoder_normalization() {
        assert!((VocalTractHdcEncoder::normalize_channel(0, 0.0) - 0.0).abs() < 1e-6);
        assert!((VocalTractHdcEncoder::normalize_channel(0, 2.0) - 1.0).abs() < 1e-6);
        assert!((VocalTractHdcEncoder::normalize_channel(0, 1.0) - 0.5).abs() < 1e-6);
        assert!((VocalTractHdcEncoder::normalize_channel(1, -1.0) - 0.0).abs() < 1e-6);
        assert!((VocalTractHdcEncoder::normalize_channel(1, 1.0) - 1.0).abs() < 1e-6);
        assert!((VocalTractHdcEncoder::normalize_channel(0, 5.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_encoder_extreme_values() {
        let genesis = test_genesis();
        let mut enc = VocalTractHdcEncoder::new(&genesis, 32);

        let extreme = VoiceCognitiveState {
            prediction_error: 10.0,
            emotional_valence: -5.0,
            emotional_arousal: 100.0,
            unified_quality: -1.0,
            epistemic_confidence: 2.0,
            coherence_velocity: -10.0,
            cross_agreement: 50.0,
            consciousness_level: -0.5,
            articulation_quality: 1.5,
            rate_stability: -0.1,
            ..Default::default()
        };

        let hv = enc.encode(&extreme);
        assert_eq!(hv.dim(), HDC_DIMENSION, "Should handle extreme values");
        assert!(hv.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_voice_cognitive_state_accessor_values() {
        let state = VoiceCognitiveState {
            prediction_error: 0.3,
            emotional_valence: 0.5,
            emotional_arousal: 0.7,
            unified_quality: 0.85,
            epistemic_confidence: 0.6,
            coherence_velocity: -0.1,
            cross_agreement: 0.9,
            consciousness_level: 0.75,
            articulation_quality: 0.8,
            rate_stability: 0.9,
            ..Default::default()
        };

        let channels = state.to_channels();
        assert_eq!(channels.len(), NUM_CHANNELS);
        assert!(channels[0] >= 0.0 && channels[0] <= 2.0);
        assert!(channels[1] >= -1.0 && channels[1] <= 1.0);
        assert!(channels[2] >= 0.0 && channels[2] <= 1.0);
        assert!(channels[7] >= 0.0 && channels[7] <= 1.0);
    }

    #[test]
    fn test_consciousness_level_maps_to_encoding() {
        let genesis = test_genesis();

        let mut enc_low = VocalTractHdcEncoder::new(&genesis, 32);
        let low = VoiceCognitiveState {
            consciousness_level: 0.1,
            ..Default::default()
        };
        let hv_low = enc_low.encode(&low);

        let mut enc_high = VocalTractHdcEncoder::new(&genesis, 32);
        let high = VoiceCognitiveState {
            consciousness_level: 0.9,
            ..Default::default()
        };
        let hv_high = enc_high.encode(&high);

        let sim = hv_low.similarity(&hv_high);
        assert!(
            sim < 0.99,
            "Different consciousness levels should produce different HVs: sim={sim}"
        );
        assert!(
            sim > 0.0,
            "Should share structure from other channels: sim={sim}"
        );
    }
}
