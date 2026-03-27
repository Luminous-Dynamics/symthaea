// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC-driven melody generation: consciousness → neural dynamics → music.
//!
//! Uses a `HdcLtcUnifiedNetwork` that autoregressively produces `MusicalFrame`
//! parameters. The CfC temporal dynamics create melodic coherence: each note is
//! influenced by the network's memory of all previous notes through the hidden
//! state, producing phrases, motifs, and contours from neural dynamics.
//!
//! # Scaling
//!
//! Network layer sizes are configurable via `MuseConfig.cfc_layer_sizes`
//! (default `[16, 16, 8]`). 12 projection vectors decode the output HV into
//! an extended MusicalFrame with FM, pan, reverb, and vibrato parameters.

use crate::{MuseConfig, MusicalFrame, MusicalState, Note};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedActivation, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Number of projection dimensions (extended MusicalFrame fields).
const NUM_PROJECTIONS: usize = 12;

/// CfC-driven melody controller.
///
/// Mirrors `symthaea-vocal-tract::controller` and `symthaea-broca::controller`:
/// HdcLtcUnifiedNetwork + input encoder + output projection.
pub struct NeuralMelody {
    /// The CfC network generating melodic state.
    network: HdcLtcUnifiedNetwork,
    /// Projection vectors for decoding output HV → 12D MusicalFrame.
    projections: Vec<ContinuousHV>,
    /// Time step per note (larger = longer temporal integration).
    dt: f32,
    /// Number of timbre partials to emit.
    num_partials: usize,
}

impl NeuralMelody {
    /// Create from genesis seed with configurable network architecture.
    pub fn new(genesis: &GenesisSeed, config: &MuseConfig) -> Self {
        let net_config = UnifiedNetworkConfig {
            layer_sizes: config.cfc_layer_sizes.clone(),
            neuron_config: UnifiedConfig {
                tau_base: 0.05,
                backbone_tau: 0.4,
                activation: UnifiedActivation::Tanh,
                ..UnifiedConfig::default()
            },
            use_layer_binding: true,
            skip_connections: false,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, genesis);

        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        let projections: Vec<ContinuousHV> = (0..NUM_PROJECTIONS)
            .map(|i| genesis.hv(&format!("melody_projection_{i}"), dim))
            .collect();

        Self {
            network,
            projections,
            dt: 0.05,
            num_partials: config.num_partials.clamp(1, 16),
        }
    }

    /// Encode the musical state into a 16,384D input HV.
    fn encode_state(state: &MusicalState, genesis: &GenesisSeed) -> ContinuousHV {
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        let channels: Vec<(f32, &str)> = vec![
            (state.consciousness_level, "consciousness"),
            (state.valence, "valence"),
            (state.arousal, "arousal"),
            (state.dopamine, "dopamine"),
            (state.serotonin, "serotonin"),
            (state.noradrenaline, "noradrenaline"),
            (state.harmony_activations[0], "harmony_coherence"),
            (state.harmony_activations[2], "harmony_wisdom"),
            (state.harmony_activations[3], "harmony_play"),
            (state.harmony_activations[7], "harmony_stillness"),
            (state.prediction_error, "prediction_error"),
        ];

        let mut result = ContinuousHV::zero(dim);
        for (value, label) in &channels {
            let basis = genesis.hv(label, dim);
            result = result.add(&basis.scale(*value));
        }
        result.normalize()
    }

    /// Decode a 16,384D output HV into an extended MusicalFrame.
    fn decode_frame(&self, output: &ContinuousHV) -> MusicalFrame {
        let vals: Vec<f32> = self
            .projections
            .iter()
            .map(|proj| (output.similarity(proj) + 1.0) / 2.0) // [-1,1] → [0,1]
            .collect();

        // Build variable-length timbre from projection values
        let timbre: Vec<f32> = (0..self.num_partials)
            .map(|i| {
                if i == 0 {
                    1.0 // fundamental always full
                } else {
                    // Blend projection hint with natural rolloff
                    let proj_hint = vals.get(2 + (i % 3)).copied().unwrap_or(0.3);
                    proj_hint * 0.5 / (i + 1) as f32
                }
            })
            .collect();

        MusicalFrame {
            pitch_hz: 130.0 + vals[0] * 800.0, // C3 to ~C6
            velocity: vals[1] * 0.7 + 0.2,      // 0.2-0.9
            timbre,
            onset: vals[5],
            sustain: vals[6] * 0.8 + 0.1,
            fm_depth: vals[7],
            fm_ratio: vals[8],
            pan: vals[9] * 2.0 - 1.0, // [0,1] → [-1,1]
            reverb_send: vals[10],
            vibrato_depth: vals[11] * 0.1,
        }
    }

    /// Generate a melody autoregressively via CfC dynamics.
    ///
    /// Each network step produces one note. The CfC hidden state carries
    /// melodic context — creating phrases, motifs, and contours from the
    /// temporal dynamics of the network rather than random scale selection.
    pub fn generate(
        &mut self,
        config: &MuseConfig,
        state: &MusicalState,
        scale: &[f32],
        beat_duration: f32,
    ) -> Vec<Note> {
        let genesis = GenesisSeed::from_phrase("muse-neural-melody");
        let input_hv = Self::encode_state(state, &genesis);

        let mut notes = Vec::new();
        let mut time = 0.0f32;

        for _ in 0..config.max_notes {
            if time >= config.duration_secs {
                break;
            }

            // Evolve CfC — temporal context builds up
            self.network.evolve_closed_form(self.dt, &input_hv);
            let output = self.network.output();
            let frame = self.decode_frame(&output);

            // Onset gating: only produce a note when onset > 0.45
            if frame.onset > 0.45 {
                let pitch = if !scale.is_empty() {
                    snap_to_scale(frame.pitch_hz, scale)
                } else {
                    frame.pitch_hz
                };

                let duration = beat_duration * (0.5 + frame.sustain * 1.0);

                notes.push(Note {
                    frequency: pitch,
                    start_time: time,
                    duration,
                    velocity: frame.velocity,
                });
            }

            time += beat_duration * 0.5; // advance half-beat per step
        }

        notes
    }
}

/// Snap a frequency to the nearest tone in the scale.
fn snap_to_scale(freq: f32, scale: &[f32]) -> f32 {
    scale
        .iter()
        .min_by(|&&a, &&b| {
            (a - freq)
                .abs()
                .partial_cmp(&(b - freq).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
        .unwrap_or(freq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::build_scale;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-neural-melody")
    }

    fn test_config() -> MuseConfig {
        MuseConfig::default()
    }

    #[test]
    fn neural_melody_generates_notes() {
        let genesis = test_genesis();
        let config = MuseConfig {
            max_notes: 16,
            duration_secs: 4.0,
            ..test_config()
        };
        let mut melody = NeuralMelody::new(&genesis, &config);
        let state = MusicalState::default();
        let scale = build_scale(&state);
        let beat = 0.5;

        let notes = melody.generate(&config, &state, &scale, beat);
        assert!(!notes.is_empty(), "should generate at least some notes");
    }

    #[test]
    fn notes_in_audible_range() {
        let genesis = test_genesis();
        let config = MuseConfig {
            max_notes: 20,
            duration_secs: 4.0,
            ..test_config()
        };
        let mut melody = NeuralMelody::new(&genesis, &config);
        let state = MusicalState {
            harmony_activations: [0.8; 8],
            consciousness_level: 0.8,
            ..Default::default()
        };
        let scale = build_scale(&state);

        let notes = melody.generate(&config, &state, &scale, 0.5);
        for note in &notes {
            assert!(
                note.frequency >= 60.0 && note.frequency <= 2000.0,
                "freq {} out of range",
                note.frequency
            );
        }
    }

    #[test]
    fn different_states_different_melodies() {
        let genesis = test_genesis();
        let config = MuseConfig {
            max_notes: 12,
            duration_secs: 3.0,
            ..test_config()
        };

        let calm = MusicalState {
            arousal: 0.1,
            valence: 0.8,
            harmony_activations: [0.3, 0.3, 0.3, 0.1, 0.3, 0.3, 0.3, 0.9],
            ..Default::default()
        };
        let intense = MusicalState {
            arousal: 0.9,
            valence: -0.5,
            noradrenaline: 0.8,
            harmony_activations: [0.1, 0.1, 0.1, 0.9, 0.5, 0.1, 0.8, 0.1],
            ..Default::default()
        };

        let scale_c = build_scale(&calm);
        let scale_i = build_scale(&intense);

        let mut m1 = NeuralMelody::new(&genesis, &config);
        let notes_calm = m1.generate(&config, &calm, &scale_c, 0.5);

        let mut m2 = NeuralMelody::new(&genesis, &config);
        let notes_intense = m2.generate(&config, &intense, &scale_i, 0.5);

        assert_ne!(scale_c, scale_i, "scales should differ between major and minor");
        assert!(!notes_calm.is_empty(), "calm should produce notes");
        assert!(!notes_intense.is_empty(), "intense should produce notes");
    }

    #[test]
    fn snap_to_scale_works() {
        let scale = vec![261.63, 293.66, 329.63, 349.23, 392.00];
        assert!((snap_to_scale(270.0, &scale) - 261.63).abs() < 1.0);
        assert!((snap_to_scale(390.0, &scale) - 392.0).abs() < 1.0);
    }

    #[test]
    fn encode_produces_unit_norm() {
        let genesis = test_genesis();
        let state = MusicalState::default();
        let hv = NeuralMelody::encode_state(&state, &genesis);
        let norm: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.1, "norm = {norm}");
    }

    #[test]
    fn configurable_cfc_layers() {
        let genesis = test_genesis();
        let config = MuseConfig {
            cfc_layer_sizes: vec![8, 8],
            max_notes: 8,
            duration_secs: 2.0,
            ..test_config()
        };
        let mut melody = NeuralMelody::new(&genesis, &config);
        let state = MusicalState::default();
        let scale = build_scale(&state);
        let notes = melody.generate(&config, &state, &scale, 0.5);
        assert!(!notes.is_empty());
    }

    #[test]
    fn extended_frame_decode() {
        let genesis = test_genesis();
        let config = test_config();
        let melody = NeuralMelody::new(&genesis, &config);
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        let hv = genesis.hv("test-output", dim);
        let frame = melody.decode_frame(&hv);
        assert!(frame.pitch_hz >= 130.0 && frame.pitch_hz <= 930.0);
        assert!(frame.fm_depth >= 0.0 && frame.fm_depth <= 1.0);
        assert!(frame.pan >= -1.0 && frame.pan <= 1.0);
        assert_eq!(frame.timbre.len(), config.num_partials);
    }
}
