// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rhythm generation from temporal dynamics.
//!
//! CfC tau → tempo, arousal → speed, prediction error periodicity → syncopation.

use crate::{MuseConfig, MusicalState};

/// Compute effective tempo in BPM from cognitive state.
///
/// Base tempo modulated by:
/// - Arousal: higher = faster
/// - Noradrenaline: higher = more energy
/// - SacredStillness (harmony 7): higher = slower (contemplative)
pub fn compute_tempo(config: &MuseConfig, state: &MusicalState) -> f32 {
    let base = config.base_tempo_bpm;

    // Arousal speeds up
    let arousal_factor = 1.0 + state.arousal * 0.5;

    // NE adds energy
    let ne_factor = 1.0 + state.noradrenaline * 0.3;

    // SacredStillness slows down
    let stillness = state.harmony_activations[7];
    let stillness_factor = 1.0 - stillness * 0.4;

    let tempo = base * arousal_factor * ne_factor * stillness_factor;
    tempo.clamp(30.0, 300.0)
}

/// Compute note duration in seconds for a given beat subdivision.
pub fn note_duration(tempo_bpm: f32, subdivision: u8) -> f32 {
    let beat_sec = 60.0 / tempo_bpm;
    beat_sec / subdivision as f32
}

/// Compute syncopation level from prediction error.
///
/// Regular prediction error → straight rhythm.
/// Irregular/high prediction error → syncopated rhythm.
/// Returns a value [0, 1] where 0 = straight, 1 = heavily syncopated.
pub fn syncopation_level(state: &MusicalState) -> f32 {
    // Prediction error drives rhythmic surprise
    let pe = state.prediction_error;
    // InfinitePlay (harmony 3) encourages playful rhythms
    let play = state.harmony_activations[3];

    (pe * 0.6 + play * 0.4).clamp(0.0, 1.0)
}

/// Select beat subdivision based on cognitive state.
///
/// Returns 1, 2, 3, or 4 (whole, half, triplet, quarter).
pub fn select_subdivision(state: &MusicalState) -> u8 {
    let complexity = state.consciousness_level;
    if complexity > 0.7 {
        4 // complex subdivisions
    } else if complexity > 0.4 {
        2 // moderate
    } else {
        1 // simple whole beats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tempo_in_range() {
        let config = MuseConfig::default();
        let state = MusicalState::default();
        let tempo = compute_tempo(&config, &state);
        assert!(tempo >= 30.0 && tempo <= 300.0, "tempo = {tempo}");
    }

    #[test]
    fn high_arousal_faster() {
        let config = MuseConfig::default();
        let calm = MusicalState {
            arousal: 0.1,
            ..Default::default()
        };
        let excited = MusicalState {
            arousal: 0.9,
            ..Default::default()
        };
        assert!(compute_tempo(&config, &excited) > compute_tempo(&config, &calm));
    }

    #[test]
    fn stillness_slows_tempo() {
        let config = MuseConfig::default();
        let active = MusicalState {
            harmony_activations: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0],
            ..Default::default()
        };
        let still = MusicalState {
            harmony_activations: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
            ..Default::default()
        };
        assert!(compute_tempo(&config, &still) < compute_tempo(&config, &active));
    }

    #[test]
    fn note_duration_correct() {
        // 120 BPM, quarter notes = 0.5s per beat, subdivision 2 = 0.25s
        let d = note_duration(120.0, 2);
        assert!((d - 0.25).abs() < 0.01);
    }

    #[test]
    fn syncopation_bounded() {
        let state = MusicalState {
            prediction_error: 0.8,
            harmony_activations: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        let s = syncopation_level(&state);
        assert!(s >= 0.0 && s <= 1.0, "syncopation = {s}");
    }
}
