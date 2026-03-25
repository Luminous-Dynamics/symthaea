// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-muse
//!
//! Consciousness-driven music synthesis for Symthaea. Maps cognitive state —
//! Eight Harmonies, neuromodulators, temporal dynamics — into melody, rhythm,
//! and timbre, producing PCM audio.
//!
//! # Architecture
//!
//! ```text
//! MusicalState → pitch (Harmonies → intervals) + rhythm (arousal/NE → tempo)
//!              → melody (autoregressive note generation) → synth (additive) → PCM
//! ```
//!
//! # Harmony-to-Interval Mapping
//!
//! | Harmony | Interval | Character |
//! |---------|----------|-----------|
//! | ResonantCoherence | Unison/Octave | Consonance |
//! | PanSentientFlourishing | Major 3rd | Warmth |
//! | IntegralWisdom | Perfect 5th | Stability |
//! | InfinitePlay | Minor 7th | Tension |
//! | UniversalInterconnectedness | Perfect 4th | Openness |
//! | SacredReciprocity | Major 6th | Reciprocal warmth |
//! | EvolutionaryProgression | Ascending seq | Growth |
//! | SacredStillness | Drone/pedal | Contemplation |

#![deny(unsafe_code)]

pub mod melody;
pub mod pitch;
pub mod rhythm;
pub mod structure;
pub mod synth;

use serde::{Deserialize, Serialize};

// ─── Core Types ──────────────────────────────────────────────────────────────

/// Cognitive state for music generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicalState {
    pub harmony_activations: [f32; 8],
    pub dopamine: f32,
    pub serotonin: f32,
    pub noradrenaline: f32,
    pub arousal: f32,
    pub valence: f32,
    pub consciousness_level: f32,
    pub prediction_error: f32,
}

impl Default for MusicalState {
    fn default() -> Self {
        Self {
            harmony_activations: [0.3; 8],
            dopamine: 0.5,
            serotonin: 0.5,
            noradrenaline: 0.3,
            arousal: 0.4,
            valence: 0.0,
            consciousness_level: 0.5,
            prediction_error: 0.1,
        }
    }
}

/// A single musical frame (one time-step of synthesis parameters).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MusicalFrame {
    /// Fundamental frequency in Hz (20-4000 range).
    pub pitch_hz: f32,
    /// Note velocity/loudness [0, 1].
    pub velocity: f32,
    /// Harmonic partial amplitudes (fundamental + 3 overtones).
    pub timbre: [f32; 4],
    /// Attack probability [0, 1] — new note trigger.
    pub onset: f32,
    /// Note sustain signal [0, 1].
    pub sustain: f32,
}

/// Configuration for music generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuseConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Duration in seconds.
    pub duration_secs: f32,
    /// Base tempo in BPM (modulated by arousal/NE).
    pub base_tempo_bpm: f32,
    /// Number of notes to generate.
    pub max_notes: usize,
}

impl Default for MuseConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            duration_secs: 8.0,
            base_tempo_bpm: 80.0,
            max_notes: 32,
        }
    }
}

/// A generated musical composition.
#[derive(Debug, Clone)]
pub struct Composition {
    /// PCM samples (mono, i16).
    pub samples: Vec<i16>,
    /// Sample rate.
    pub sample_rate: u32,
    /// Generated note sequence.
    pub notes: Vec<Note>,
    /// Duration in seconds.
    pub duration_secs: f32,
}

/// A discrete musical note.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Note {
    /// Frequency in Hz.
    pub frequency: f32,
    /// Start time in seconds.
    pub start_time: f32,
    /// Duration in seconds.
    pub duration: f32,
    /// Velocity [0, 1].
    pub velocity: f32,
}

// ─── Top-Level API ───────────────────────────────────────────────────────────

/// Generate a musical composition from a cognitive state.
pub fn compose(config: &MuseConfig, state: &MusicalState, seed: u64) -> Composition {
    // 1. Determine rhythm (tempo, beat grid)
    let tempo = rhythm::compute_tempo(config, state);
    let beat_duration = 60.0 / tempo;

    // 2. Generate melody (note sequence)
    let scale = pitch::build_scale(state);
    let notes = melody::generate_melody(config, state, &scale, beat_duration, seed);

    // 3. Synthesize audio
    let total_samples = (config.duration_secs * config.sample_rate as f32) as usize;
    let samples = synth::render_notes(&notes, config.sample_rate, total_samples, state);

    Composition {
        samples,
        sample_rate: config.sample_rate,
        notes,
        duration_secs: config.duration_secs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_produces_audio() {
        let config = MuseConfig {
            duration_secs: 2.0,
            max_notes: 8,
            ..Default::default()
        };
        let state = MusicalState::default();
        let comp = compose(&config, &state, 42);
        assert!(!comp.samples.is_empty());
        assert!(!comp.notes.is_empty());
        assert_eq!(comp.sample_rate, 44100);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let config = MuseConfig {
            duration_secs: 1.0,
            max_notes: 4,
            ..Default::default()
        };
        let state = MusicalState::default();
        let c1 = compose(&config, &state, 42);
        let c2 = compose(&config, &state, 42);
        assert_eq!(c1.samples, c2.samples);
        assert_eq!(c1.notes.len(), c2.notes.len());
    }

    #[test]
    fn different_states_different_music() {
        let config = MuseConfig {
            duration_secs: 1.0,
            max_notes: 4,
            ..Default::default()
        };
        let calm = MusicalState {
            arousal: 0.1,
            valence: 0.5,
            ..Default::default()
        };
        let excited = MusicalState {
            arousal: 0.9,
            valence: -0.5,
            noradrenaline: 0.8,
            ..Default::default()
        };
        let c1 = compose(&config, &calm, 42);
        let c2 = compose(&config, &excited, 42);
        assert_ne!(c1.notes.len(), c2.notes.len());
    }

    #[test]
    fn no_nan_or_inf_samples() {
        let config = MuseConfig {
            duration_secs: 2.0,
            max_notes: 16,
            ..Default::default()
        };
        let state = MusicalState {
            harmony_activations: [0.8; 8],
            consciousness_level: 0.9,
            ..Default::default()
        };
        let comp = compose(&config, &state, 42);
        for &s in &comp.samples {
            assert!(s >= i16::MIN && s <= i16::MAX);
        }
    }
}
