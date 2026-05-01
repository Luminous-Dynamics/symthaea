// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Melody generation via autoregressive note selection.
//!
//! Generates a sequence of notes from the harmony-derived scale,
//! using consciousness state to control contour, rhythm, and dynamics.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::emotional_gestures::{detect_emotion, gesture_for_emotion};
use crate::pitch::select_pitch;
use crate::rhythm::{select_subdivision, syncopation_level};
use crate::{MuseConfig, MusicalState, Note};

/// Generate a melody (sequence of notes) from cognitive state.
///
/// Uses the emotional gesture system to shape pitch direction, note duration,
/// velocity, and articulation based on the detected emotional quality.
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

    // Detect emotion and get the corresponding musical gesture
    let emotion = detect_emotion(state);
    let gesture = gesture_for_emotion(emotion);

    // Direction bias: gesture overrides valence-only bias
    let ascend_bias = 0.5 + gesture.direction_bias * 0.5;
    let mut pitch_selector: f32 = if gesture.direction_bias > 0.2 {
        0.30 // start low for ascending gestures (joy, wonder, triumph)
    } else if gesture.direction_bias < -0.2 {
        0.65 // start high for descending gestures (sadness)
    } else {
        0.45 // neutral start
    };

    // Phrase contour: arc shape scaled by arousal + gesture energy
    let phrase_len = config.max_notes.max(4);
    let peak_position = 0.35 + (1.0 - state.arousal) * 0.25;
    let sparsity_boost = if config.max_notes <= 8 { 0.15 } else { 0.0 };
    let contour_strength = (0.35 + state.arousal * 0.45 + sparsity_boost).min(0.9);

    // Velocity: gesture shapes the base level and dynamic range
    let base_velocity = (0.3 + state.consciousness_level * 0.2 + state.dopamine * 0.1)
        .clamp(0.15, 0.8)
        * gesture.velocity_scale;
    let dynamic_range = (0.10 + state.arousal * 0.20) * gesture.velocity_scale;

    for i in 0..config.max_notes {
        if time >= max_time {
            break;
        }

        // Phrase position [0, 1] — where are we in the melodic phrase?
        let phrase_t = i as f32 / (phrase_len - 1).max(1) as f32;

        // Phrase contour: parabolic arc peaking at peak_position.
        // Strength scales with arousal: calm music stays flatter.
        let contour = {
            let dist = (phrase_t - peak_position).abs();
            let width = peak_position.max(1.0 - peak_position);
            (1.0 - (dist / width).powi(2)) * contour_strength
        };

        // Pitch influenced by contour arc — higher in the middle of the phrase
        let contour_pitch = 0.45 + contour * 0.3; // 0.45-0.75, scaled by contour_strength

        // Select pitch from scale
        let freq = select_pitch(scale, pitch_selector.clamp(0.0, 1.0));

        // Duration: gesture shapes note length, but high arousal FORCES longer
        // sustained notes to avoid dense polyphony sounding harsh.
        // Energy should come from harmony and register, not from rapid-fire notes.
        let dur_factor = if rng.gen::<f32>() < syncopation {
            0.5 + rng.gen::<f32>() * 1.5
        } else {
            1.0
        };
        let gesture_dur = gesture.duration_scale;
        // At arousal > 0.5, override gesture's short durations with sustained ones
        let arousal_sustain = if state.arousal > 0.5 {
            let t = (state.arousal - 0.5) * 2.0; // 0→1 as arousal goes 0.5→1.0
            gesture_dur * (1.0 - t) + 2.0 * t // blend toward 2x duration
        } else {
            gesture_dur
        };
        let duration = (base_note_dur * dur_factor * arousal_sustain).max(0.05);

        // Expressive velocity: gesture + contour + jitter
        let vel_jitter = (rng.gen::<f32>() - 0.5) * 0.10; // ±5%
        let velocity = (base_velocity + contour * dynamic_range + vel_jitter).clamp(0.10, 1.0);

        // Rest probability: SacredStillness increases rests.
        let rest_prob = (state.harmony_activations[7] * 0.4).min(0.25);
        if notes.len() >= 4 && rng.gen::<f32>() < rest_prob {
            let rest_steps = (duration / base_note_dur).round().max(1.0);
            time += rest_steps * base_note_dur;
            continue;
        }

        // Quantize onset to subdivision grid
        let grid_time = (time / base_note_dur).round() * base_note_dur;
        let grid_steps = (duration / base_note_dur).round().max(1.0);

        notes.push(Note {
            frequency: freq,
            start_time: grid_time,
            duration,
            velocity,
        });

        time = grid_time + grid_steps * base_note_dur;

        // Pitch contour: arc pull dominates, random walk adds flavor
        let random_step = (rng.gen::<f32>() - (1.0 - ascend_bias)) * 0.12;
        let arc_pull = (contour_pitch - pitch_selector) * 0.30 * contour_strength;
        pitch_selector += random_step + arc_pull;

        // EvolutionaryProgression (harmony 6): tendency to ascend
        if state.harmony_activations[6] > 0.5 {
            pitch_selector += 0.02;
        }

        // Gravity: return toward contour target (stronger pull = clearer arc)
        pitch_selector += (contour_pitch - pitch_selector) * 0.12;

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
