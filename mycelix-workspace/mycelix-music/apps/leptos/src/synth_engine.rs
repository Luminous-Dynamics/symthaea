// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Wrapper around symthaea-muse's StreamingSynth for the browser.
//!
//! With ClickHouse stopped and stale processes killed, we can run
//! a richer config: 8 partials, sidechain ducking, consciousness reverb,
//! ambient drone, percussion, motif memory, and dramatic arc.

use symthaea_muse::streaming::StreamingSynth;
use symthaea_muse::{MuseConfig, MusicalState};

pub const SAMPLE_RATE: f32 = 44100.0;

pub struct SynthEngine {
    synth: StreamingSynth,
    state: MusicalState,
    pub running: bool,
}

impl SynthEngine {
    pub fn new() -> Self {
        let config = MuseConfig {
            sample_rate: SAMPLE_RATE as u32,
            duration_secs: 3600.0,
            max_notes: 12,
            num_partials: 8,
            enable_antialiasing: true,
            enable_sub_bass: true,
            ..Default::default()
        };

        let mut synth = StreamingSynth::new(config, SAMPLE_RATE as u32);
        synth.enable_sidechain = true;   // Lead ducks bass — cleaner mix
        synth.enable_binaural = false;   // Still too heavy for WASM
        synth.enable_fep = false;        // Still too heavy for WASM
        synth.feedback_strength = 0.2;

        Self {
            synth,
            state: MusicalState::default(),
            running: false,
        }
    }

    pub fn set_state(&mut self, arousal: f32, valence: f32, phi: f32) {
        self.state.arousal = arousal.clamp(0.0, 1.0);
        self.state.valence = valence.clamp(-1.0, 1.0);
        self.state.consciousness_level = phi.clamp(0.0, 1.0);

        self.state.dopamine = 0.3 + arousal * 0.5 + valence.max(0.0) * 0.2;
        self.state.serotonin = 0.4 + valence.max(0.0) * 0.4;
        self.state.noradrenaline = arousal * 0.5;
        self.state.prediction_error = 0.1 + arousal * 0.2;

        self.state.harmony_activations[0] = 0.5 + phi * 0.5;
        self.state.harmony_activations[1] = 0.3 + valence.max(0.0) * 0.7;
        self.state.harmony_activations[2] = 0.4 + phi * 0.3;
        self.state.harmony_activations[3] = 0.2 + arousal * 0.5 + (-valence).max(0.0) * 0.3;
        self.state.harmony_activations[4] = 0.3 + (1.0 - arousal) * 0.4;
        self.state.harmony_activations[5] = 0.3 + valence.abs() * 0.3 + phi * 0.2;
        self.state.harmony_activations[6] = arousal * 0.8;
        self.state.harmony_activations[7] = (1.0 - arousal) * 0.6 + phi * 0.2;

        self.synth.update_state(&self.state);
    }

    pub fn apply_feedback(&mut self, centroid: f32, rms: f32, flux: f32) {
        let alpha = 0.08_f32;

        let target_h3 = self.state.harmony_activations[3] + (centroid - 0.5) * 0.12;
        self.state.harmony_activations[3] =
            self.state.harmony_activations[3] * (1.0 - alpha) + target_h3.clamp(0.0, 1.0) * alpha;

        let rms_adjust = (rms - 0.4) * 0.06;
        let target_arousal = (self.state.arousal - rms_adjust).clamp(0.0, 1.0);
        self.state.arousal = self.state.arousal * (1.0 - alpha) + target_arousal * alpha;

        let target_h7 = self.state.harmony_activations[7] - flux * 0.08;
        self.state.harmony_activations[7] =
            self.state.harmony_activations[7] * (1.0 - alpha) + target_h7.clamp(0.0, 1.0) * alpha;

        self.synth.update_state(&self.state);
    }

    pub fn render(&mut self) -> Vec<f32> {
        if !self.running {
            let size = self.synth.chunk_samples();
            return vec![0.0; size * 2];
        }

        let chunk = self.synth.render_chunk();
        let mut output = Vec::with_capacity(chunk.len() * 2);
        for pair in &chunk {
            output.push(pair[0]);
            output.push(pair[1]);
        }
        output
    }

    pub fn chunk_samples(&self) -> usize {
        self.synth.chunk_samples()
    }
}
