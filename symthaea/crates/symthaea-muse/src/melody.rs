// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Melody generation via autoregressive note selection.
//!
//! Generates a sequence of notes from the harmony-derived scale,
//! using consciousness state to control contour, rhythm, and dynamics.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::pitch::select_pitch;
use crate::rhythm::{select_subdivision, syncopation_level};
use crate::{MuseConfig, MusicalState, Note};

/// Generate a melody (sequence of notes) from cognitive state.
pub fn generate_melody(
    config: &MuseConfig,
    state: &MusicalState,
    scale: &[f32],
    beat_duration: f32,
    seed: u64,
) -> Vec<Note> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut notes = Vec::new();
    let mut time = 0.0f32;
    let max_time = config.duration_secs;
    let subdivision = select_subdivision(state);
    let base_note_dur = beat_duration / subdivision as f32;
    let syncopation = syncopation_level(state);

    // Contour direction: valence influences ascending/descending tendency
    let ascend_bias = 0.5 + state.valence * 0.3; // positive valence → upward motion
    let mut pitch_selector = 0.5f32; // start in middle of scale

    for i in 0..config.max_notes {
        if time >= max_time {
            break;
        }

        // Select pitch from scale
        let freq = select_pitch(scale, pitch_selector.clamp(0.0, 1.0));

        // Duration with syncopation (random lengthening/shortening)
        let dur_factor = if rng.gen::<f32>() < syncopation {
            // Syncopated: random duration variation
            0.5 + rng.gen::<f32>() * 1.5
        } else {
            1.0 // straight
        };
        let duration = (base_note_dur * dur_factor).max(0.05);

        // Velocity from consciousness + dopamine
        let velocity = (0.3 + state.consciousness_level * 0.4 + state.dopamine * 0.2)
            .clamp(0.1, 1.0);

        // Rest probability: SacredStillness increases rests
        let rest_prob = state.harmony_activations[7] * 0.4;
        if rng.gen::<f32>() < rest_prob {
            time += duration;
            continue; // rest
        }

        notes.push(Note {
            frequency: freq,
            start_time: time,
            duration,
            velocity,
            ..Default::default()
        });

        time += duration;

        // Update pitch selector for next note (contour)
        let step = (rng.gen::<f32>() - (1.0 - ascend_bias)) * 0.2;
        pitch_selector += step;

        // EvolutionaryProgression (harmony 6): tendency to ascend
        if state.harmony_activations[6] > 0.5 {
            pitch_selector += 0.02;
        }

        // Gravity: slowly return to center
        pitch_selector += (0.5 - pitch_selector) * 0.05;

        let _ = i;
    }

    notes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::build_scale;

    #[test]
    fn generates_notes() {
        let config = MuseConfig {
            max_notes: 8,
            duration_secs: 4.0,
            ..Default::default()
        };
        let state = MusicalState::default();
        let scale = build_scale(&state);
        let notes = generate_melody(&config, &state, &scale, 0.5, 42);
        assert!(!notes.is_empty());
        assert!(notes.len() <= 8);
    }

    #[test]
    fn notes_in_time_range() {
        let config = MuseConfig {
            max_notes: 16,
            duration_secs: 2.0,
            ..Default::default()
        };
        let state = MusicalState::default();
        let scale = build_scale(&state);
        let notes = generate_melody(&config, &state, &scale, 0.5, 42);
        for note in &notes {
            assert!(note.start_time >= 0.0);
            assert!(note.start_time < config.duration_secs + 1.0);
        }
    }

    #[test]
    fn frequencies_valid() {
        let config = MuseConfig {
            max_notes: 16,
            duration_secs: 4.0,
            ..Default::default()
        };
        let state = MusicalState {
            harmony_activations: [0.8; 8],
            ..Default::default()
        };
        let scale = build_scale(&state);
        let notes = generate_melody(&config, &state, &scale, 0.5, 42);
        for note in &notes {
            assert!(
                note.frequency >= 20.0 && note.frequency <= 4000.0,
                "freq = {}",
                note.frequency
            );
        }
    }

    #[test]
    fn stillness_creates_rests() {
        let config = MuseConfig {
            max_notes: 20,
            duration_secs: 4.0,
            ..Default::default()
        };
        let active = MusicalState {
            harmony_activations: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0],
            ..Default::default()
        };
        let still = MusicalState {
            harmony_activations: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
            ..Default::default()
        };
        let scale_a = build_scale(&active);
        let scale_s = build_scale(&still);
        let notes_a = generate_melody(&config, &active, &scale_a, 0.5, 42);
        let notes_s = generate_melody(&config, &still, &scale_s, 0.5, 42);
        // Stillness should produce fewer notes (more rests)
        assert!(
            notes_s.len() <= notes_a.len(),
            "still {} should <= active {}",
            notes_s.len(),
            notes_a.len()
        );
    }
}
