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
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};

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
    /// Trainable decode weights: applied to output before sigmoid.
    /// Default: all 1.0 (no modification). Training adjusts these
    /// to shape the pitch/velocity/onset mappings.
    decode_weights: Vec<f32>,
    /// Trainable decode biases.
    decode_biases: Vec<f32>,
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

        Self {
            network,
            dim,
            decode_weights: vec![3.0, 2.0, 2.0], // default scaling for pitch, vel, onset
            decode_biases: vec![0.0; 3],
        }
    }

    /// Set trainable decode parameters (weights and biases).
    ///
    /// Called by the training loop to inject perturbed parameters.
    /// `projections` should have 2 vectors: [weights(3), biases(3)].
    pub fn set_projections(&mut self, projections: Vec<ContinuousHV>) {
        if projections.len() >= 2 {
            let w = &projections[0].values;
            let b = &projections[1].values;
            for i in 0..3.min(w.len()) {
                self.decode_weights[i] = w[i];
            }
            for i in 0..3.min(b.len()) {
                self.decode_biases[i] = b[i];
            }
        }
    }

    /// Load trained decode parameters from a JSON file.
    ///
    /// Falls back silently to default weights if file doesn't exist or fails.
    /// This enables compose() to use trained projections when available.
    pub fn load_trained_projections(&mut self, path: &std::path::Path) {
        if let Ok(projections) = crate::training::load_projections(path) {
            if projections.len() >= 2 {
                for i in 0..3.min(projections[0].len()) {
                    self.decode_weights[i] = projections[0][i];
                }
                for i in 0..3.min(projections[1].len()) {
                    self.decode_biases[i] = projections[1][i];
                }
            }
        }
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

        // Trainable decode: weight * value + bias → sigmoid
        let pitch_sel = if !vals.is_empty() {
            sigmoid(vals[0] * self.decode_weights[0] + self.decode_biases[0])
        } else {
            0.5
        };

        let velocity = if vals.len() > 1 {
            sigmoid(vals[1] * self.decode_weights[1] + self.decode_biases[1]).clamp(0.1, 1.0)
        } else {
            0.5
        };

        let onset = if vals.len() > 2 {
            sigmoid(vals[2] * self.decode_weights[2] + self.decode_biases[2])
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

            // Onset threshold: network decides phrase boundaries.
            // Lower threshold for low-energy states so contemplative music still
            // has notes. Guarantee at least 6 notes by progressively lowering
            // the threshold as we run out of time without enough notes.
            let time_remaining_frac = 1.0 - (time / max_time).min(1.0);
            let note_deficit = if notes.len() < 6 && time_remaining_frac < 0.5 {
                (6 - notes.len()) as f32 / 6.0 // 0→1 as deficit grows
            } else {
                0.0
            };
            let onset_threshold = (0.4 - note_deficit * 0.3).max(0.1);

            if onset > onset_threshold {
                let freq = snap_to_scale(pitch_sel, scale);

                let dur = beat_duration * (0.5 + onset * 0.5);
                // Quantize onset AND advance to 8th-note grid
                let grid = beat_duration * 0.5;
                let grid_time = if grid > 0.01 {
                    (time / grid).round() * grid
                } else {
                    time
                };
                let grid_steps = if grid > 0.01 {
                    (dur / grid).round().max(1.0)
                } else {
                    1.0
                };
                notes.push(Note {
                    frequency: freq,
                    start_time: grid_time,
                    duration: dur.min(max_time - grid_time),
                    velocity,
                });

                pitch_selector = pitch_sel;
                time = grid_time + grid_steps * grid;
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
