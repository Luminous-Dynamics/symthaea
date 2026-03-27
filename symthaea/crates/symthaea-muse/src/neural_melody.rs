// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC neural melody generation using HdcLtcUnifiedNetwork.
//!
//! Encodes cognitive state into a hypervector, evolves through a CfC network,
//! then decodes the output into musical frames. This produces temporally
//! coherent melodies with phrases, motifs, and contours that emerge from
//! the liquid-time dynamics.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::ContinuousHV;

use crate::{MuseConfig, MusicalState, Note};

/// Neural melody generator backed by a CfC HdcLtcUnifiedNetwork.
///
/// Each call to `generate()` evolves the network step-by-step, producing
/// one note per frame. The CfC temporal dynamics create coherent phrases
/// and natural melodic contour.
pub struct NeuralMelody {
    /// The CfC network that generates musical state.
    network: HdcLtcUnifiedNetwork,
    /// HDC dimension used for encoding/decoding.
    dim: usize,
}

impl NeuralMelody {
    /// Create a new neural melody generator from a genesis seed and config.
    pub fn new(genesis: &GenesisSeed, config: &MuseConfig) -> Self {
        let dim = 64; // compact dimension for music (not full 16384)
        let layer_sizes = if config.cfc_layer_sizes.is_empty() {
            vec![16, 16, 8]
        } else {
            config.cfc_layer_sizes.clone()
        };

        let neuron_config = UnifiedConfig {
            dimension: dim,
            tau_base: 0.05, // fast response for musical timing
            backbone_tau: 0.3,
            ..UnifiedConfig::default()
        };

        let net_config = UnifiedNetworkConfig {
            layer_sizes,
            neuron_config,
            use_layer_binding: true,
            skip_connections: true,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, genesis);

        Self { network, dim }
    }

    /// Encode cognitive state into a hypervector input for the network.
    fn encode_state(&self, state: &MusicalState, pitch_selector: f32) -> ContinuousHV {
        let mut values = vec![0.0f32; self.dim];

        // Pack harmony activations into first 8 dimensions
        for (i, &h) in state.harmony_activations.iter().enumerate() {
            if i < self.dim {
                values[i] = h;
            }
        }

        // Neuromodulators in dimensions 8-11
        if self.dim > 11 {
            values[8] = state.dopamine;
            values[9] = state.serotonin;
            values[10] = state.noradrenaline;
            values[11] = state.arousal;
        }

        // Valence, consciousness, prediction error, pitch state
        if self.dim > 15 {
            values[12] = state.valence;
            values[13] = state.consciousness_level;
            values[14] = state.prediction_error;
            values[15] = pitch_selector;
        }

        ContinuousHV::from_values(values)
    }

    /// Decode network output into a musical frame: (frequency_selector, velocity, onset_prob).
    fn decode_frame(&self, output: &ContinuousHV) -> (f32, f32, f32) {
        let vals = &output.values;

        // First dimension: pitch selector [0, 1]
        let pitch_sel = if !vals.is_empty() {
            sigmoid(vals[0] * 3.0)
        } else {
            0.5
        };

        // Second dimension: velocity
        let velocity = if vals.len() > 1 {
            sigmoid(vals[1] * 2.0).clamp(0.1, 1.0)
        } else {
            0.5
        };

        // Third dimension: onset probability
        let onset = if vals.len() > 2 {
            sigmoid(vals[2] * 2.0)
        } else {
            0.7
        };

        (pitch_sel, velocity, onset)
    }

    /// Generate a melody by evolving the CfC network step-by-step.
    ///
    /// Each step produces one potential note. The network's temporal dynamics
    /// create phrase boundaries (via onset probability) and melodic contour.
    pub fn generate(
        &mut self,
        config: &MuseConfig,
        state: &MusicalState,
        scale: &[f32],
        beat_duration: f32,
    ) -> Vec<Note> {
        let mut notes = Vec::new();
        let mut time = 0.0f32;
        let max_time = config.duration_secs;
        let dt = beat_duration * 0.25; // evolve at 16th-note resolution
        let mut pitch_selector = 0.5f32;

        for _ in 0..config.max_notes * 2 {
            if time >= max_time || notes.len() >= config.max_notes {
                break;
            }

            // Encode current state + pitch context
            let input = self.encode_state(state, pitch_selector);

            // Evolve network one step
            self.network.evolve(dt, &input);

            // Decode output
            let output = self.network.output();
            let (pitch_sel, velocity, onset) = self.decode_frame(&output);

            // onset > 0.4 means we produce a note (network decides phrase boundaries)
            if onset > 0.4 {
                let freq = snap_to_scale(pitch_sel, scale);

                let dur = beat_duration * (0.5 + onset * 0.5);
                notes.push(Note {
                    frequency: freq,
                    start_time: time,
                    duration: dur.min(max_time - time),
                    velocity,
                    ..Default::default()
                });

                pitch_selector = pitch_sel;
                time += dur;
            } else {
                // Rest / phrase boundary
                time += dt;
            }
        }

        notes
    }
}

/// Sigmoid activation.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Snap a [0, 1] pitch selector to the nearest frequency in the scale.
fn snap_to_scale(selector: f32, scale: &[f32]) -> f32 {
    if scale.is_empty() {
        return 261.63; // C4 fallback
    }
    let idx = ((selector * (scale.len() - 1) as f32).round() as usize).min(scale.len() - 1);
    scale[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::build_scale;

    #[test]
    fn neural_generates_notes() {
        let genesis = GenesisSeed::from_phrase("test-neural-melody");
        let config = MuseConfig {
            max_notes: 8,
            duration_secs: 4.0,
            ..Default::default()
        };
        let state = MusicalState::default();
        let scale = build_scale(&state);
        let mut neural = NeuralMelody::new(&genesis, &config);
        let notes = neural.generate(&config, &state, &scale, 0.5);
        assert!(!notes.is_empty(), "should produce at least one note");
    }

    #[test]
    fn neural_respects_max_notes() {
        let genesis = GenesisSeed::from_phrase("test-max-notes");
        let config = MuseConfig {
            max_notes: 4,
            duration_secs: 10.0,
            ..Default::default()
        };
        let state = MusicalState::default();
        let scale = build_scale(&state);
        let mut neural = NeuralMelody::new(&genesis, &config);
        let notes = neural.generate(&config, &state, &scale, 0.5);
        assert!(notes.len() <= 4, "should not exceed max_notes");
    }

    #[test]
    fn neural_respects_duration() {
        let genesis = GenesisSeed::from_phrase("test-duration");
        let config = MuseConfig {
            max_notes: 16,
            duration_secs: 2.0,
            ..Default::default()
        };
        let state = MusicalState::default();
        let scale = build_scale(&state);
        let mut neural = NeuralMelody::new(&genesis, &config);
        let notes = neural.generate(&config, &state, &scale, 0.5);
        for note in &notes {
            assert!(
                note.start_time + note.duration <= config.duration_secs + 0.01,
                "note exceeds duration: {} + {} > {}",
                note.start_time,
                note.duration,
                config.duration_secs
            );
        }
    }

    #[test]
    fn snap_to_scale_edges() {
        let scale = vec![100.0, 200.0, 300.0, 400.0];
        assert!((snap_to_scale(0.0, &scale) - 100.0).abs() < 1e-3);
        assert!((snap_to_scale(1.0, &scale) - 400.0).abs() < 1e-3);
    }

    #[test]
    fn sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn encode_state_dimension() {
        let genesis = GenesisSeed::from_phrase("test-encode");
        let config = MuseConfig::default();
        let neural = NeuralMelody::new(&genesis, &config);
        let state = MusicalState::default();
        let hv = neural.encode_state(&state, 0.5);
        assert_eq!(hv.values.len(), 64);
    }
}
