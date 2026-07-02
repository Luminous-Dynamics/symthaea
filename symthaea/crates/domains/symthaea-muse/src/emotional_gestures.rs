// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Emotional melody mapping: consciousness states → recognizable musical gestures.
//!
//! Translates the valence-arousal circumplex into specific musical devices
//! that human listeners associate with emotions. This bridges the gap between
//! consciousness parameters (physics-based) and felt emotion (culture-based).

use crate::MusicalState;

/// Detected emotional quality of the current consciousness state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmotionalQuality {
    Joy,     // positive valence, high arousal
    Peace,   // positive valence, low arousal
    Sadness, // negative valence, low arousal
    Tension, // negative valence, high arousal
    Wonder,  // rising consciousness
    Triumph, // high dopamine + positive valence
    Mystery, // low consciousness, moderate arousal
    Urgency, // high noradrenaline + high arousal
}

/// Musical gesture prescribed for an emotion.
#[derive(Debug, Clone)]
pub struct MusicalGesture {
    pub emotion: EmotionalQuality,
    /// Preferred scale mode: true = major/lydian, false = minor/aeolian.
    pub prefer_major: bool,
    /// Melodic direction bias [-1, 1]: -1 = descending, 0 = neutral, 1 = ascending.
    pub direction_bias: f32,
    /// Preferred note duration multiplier (>1 = longer notes, <1 = shorter).
    pub duration_scale: f32,
    /// Preferred interval size in semitones (larger = wider leaps).
    pub interval_preference: f32,
    /// Dynamics: velocity multiplier.
    pub velocity_scale: f32,
    /// Chord preference: index into progression (0 = tonic, 4 = dominant).
    pub harmonic_tension: f32,
    /// Staccato factor [0, 1]: 0 = legato, 1 = very short.
    pub staccato: f32,
}

/// Detect the emotional quality of a consciousness state.
pub fn detect_emotion(state: &MusicalState) -> EmotionalQuality {
    let v = state.valence;
    let a = state.arousal;
    let psi = state.consciousness_level;
    let da = state.dopamine;
    let ne = state.noradrenaline;
    let pe = state.prediction_error;
    let still = state.harmony_activations[7];

    // Priority-ordered detection
    if ne > 0.7 && a > 0.7 {
        EmotionalQuality::Urgency
    } else if da > 0.7 && v > 0.3 {
        EmotionalQuality::Triumph
    } else if psi < 0.3 && a > 0.3 && a < 0.7 {
        EmotionalQuality::Mystery
    } else if pe > 0.5 || (v < -0.2 && a > 0.5) {
        EmotionalQuality::Tension
    } else if v > 0.3 && a > 0.5 {
        EmotionalQuality::Joy
    } else if (v > 0.0 && a < 0.3) || (still > 0.5 && v > -0.2) {
        EmotionalQuality::Peace
    } else if v < -0.2 && a < 0.4 {
        EmotionalQuality::Sadness
    } else {
        // Default: rising consciousness = wonder
        EmotionalQuality::Wonder
    }
}

/// Get the musical gesture for an emotional quality.
pub fn gesture_for_emotion(emotion: EmotionalQuality) -> MusicalGesture {
    match emotion {
        EmotionalQuality::Joy => MusicalGesture {
            emotion,
            prefer_major: true,
            direction_bias: 0.4,      // ascending
            duration_scale: 0.8,      // slightly shorter (energetic)
            interval_preference: 4.0, // thirds and fourths
            velocity_scale: 1.1,
            harmonic_tension: 0.2,
            staccato: 0.1,
        },
        EmotionalQuality::Peace => MusicalGesture {
            emotion,
            prefer_major: true,
            direction_bias: 0.0,      // neutral
            duration_scale: 2.0,      // long sustained notes
            interval_preference: 2.0, // stepwise motion
            velocity_scale: 0.6,
            harmonic_tension: 0.0, // consonant
            staccato: 0.0,         // full legato
        },
        EmotionalQuality::Sadness => MusicalGesture {
            emotion,
            prefer_major: false,
            direction_bias: -0.4,     // descending
            duration_scale: 1.5,      // longer notes
            interval_preference: 1.5, // small intervals (seconds)
            velocity_scale: 0.5,      // soft
            harmonic_tension: 0.3,
            staccato: 0.0,
        },
        EmotionalQuality::Tension => MusicalGesture {
            emotion,
            prefer_major: false,
            direction_bias: 0.1,      // slightly ascending (building)
            duration_scale: 0.6,      // shorter notes
            interval_preference: 6.0, // tritones, wide leaps
            velocity_scale: 1.2,
            harmonic_tension: 0.8, // very dissonant
            staccato: 0.3,
        },
        EmotionalQuality::Wonder => MusicalGesture {
            emotion,
            prefer_major: true,
            direction_bias: 0.6, // strongly ascending
            duration_scale: 1.2,
            interval_preference: 7.0, // octave leaps
            velocity_scale: 0.7,      // piano to forte
            harmonic_tension: 0.1,
            staccato: 0.0,
        },
        EmotionalQuality::Triumph => MusicalGesture {
            emotion,
            prefer_major: true,
            direction_bias: 0.5,
            duration_scale: 0.7,      // energetic
            interval_preference: 5.0, // fourths and fifths (heroic)
            velocity_scale: 1.3,      // fortissimo
            harmonic_tension: 0.1,    // strong resolution (V→I)
            staccato: 0.2,
        },
        EmotionalQuality::Mystery => MusicalGesture {
            emotion,
            prefer_major: false, // whole-tone / ambiguous
            direction_bias: 0.0,
            duration_scale: 1.8,
            interval_preference: 3.0, // minor thirds (ambiguous)
            velocity_scale: 0.4,      // very quiet
            harmonic_tension: 0.5,    // ambiguous
            staccato: 0.0,
        },
        EmotionalQuality::Urgency => MusicalGesture {
            emotion,
            prefer_major: false,
            direction_bias: 0.3,
            duration_scale: 0.4,      // very short notes
            interval_preference: 1.0, // chromatic motion
            velocity_scale: 1.4,      // loud
            harmonic_tension: 0.7,
            staccato: 0.6, // very staccato
        },
    }
}

/// Apply a musical gesture to modify a note's frequency within the current scale.
///
/// `base_freq`: the note frequency from the generator
/// `scale`: available scale tones
/// `gesture`: the emotional gesture to apply
/// Returns: modified frequency respecting the gesture's direction and interval preferences
pub fn apply_gesture_to_pitch(
    base_freq: f32,
    prev_freq: Option<f32>,
    scale: &[f32],
    gesture: &MusicalGesture,
) -> f32 {
    if scale.is_empty() {
        return base_freq;
    }

    // Find nearest scale tone to base_freq
    let base_idx = scale
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| ((**a - base_freq).abs()).total_cmp(&((**b - base_freq).abs())))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Apply direction bias
    let direction = if let Some(prev) = prev_freq {
        if gesture.direction_bias > 0.2 && base_freq <= prev {
            // Want ascending but note is same or lower — shift up
            1i32
        } else if gesture.direction_bias < -0.2 && base_freq >= prev {
            // Want descending but note is same or higher — shift down
            -1
        } else {
            0
        }
    } else {
        0
    };

    // Apply shift
    let target_idx = (base_idx as i32 + direction).clamp(0, scale.len() as i32 - 1) as usize;
    scale[target_idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joy_is_major_ascending() {
        let gesture = gesture_for_emotion(EmotionalQuality::Joy);
        assert!(gesture.prefer_major);
        assert!(gesture.direction_bias > 0.0);
    }

    #[test]
    fn sadness_is_minor_descending() {
        let gesture = gesture_for_emotion(EmotionalQuality::Sadness);
        assert!(!gesture.prefer_major);
        assert!(gesture.direction_bias < 0.0);
    }

    #[test]
    fn peace_has_long_notes() {
        let gesture = gesture_for_emotion(EmotionalQuality::Peace);
        assert!(gesture.duration_scale > 1.5);
        assert!(gesture.staccato < 0.1);
    }

    #[test]
    fn urgency_is_fast_loud() {
        let gesture = gesture_for_emotion(EmotionalQuality::Urgency);
        assert!(gesture.duration_scale < 0.5);
        assert!(gesture.velocity_scale > 1.3);
        assert!(gesture.staccato > 0.5);
    }

    #[test]
    fn detect_joyful_state() {
        let state = MusicalState {
            valence: 0.6,
            arousal: 0.7,
            ..Default::default()
        };
        assert_eq!(detect_emotion(&state), EmotionalQuality::Joy);
    }

    #[test]
    fn detect_peaceful_state() {
        let state = MusicalState {
            valence: 0.3,
            arousal: 0.1,
            harmony_activations: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7],
            ..Default::default()
        };
        assert_eq!(detect_emotion(&state), EmotionalQuality::Peace);
    }

    #[test]
    fn detect_tense_state() {
        let state = MusicalState {
            valence: -0.5,
            arousal: 0.8,
            prediction_error: 0.6,
            ..Default::default()
        };
        assert_eq!(detect_emotion(&state), EmotionalQuality::Tension);
    }

    #[test]
    fn high_stillness_negative_valence_is_not_peace() {
        // Regression: precedence bug made stillness>0.5 override negative valence
        let state = MusicalState {
            valence: -0.8,
            arousal: 0.1,
            harmony_activations: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7],
            ..Default::default()
        };
        let emotion = detect_emotion(&state);
        assert_ne!(
            emotion,
            EmotionalQuality::Peace,
            "negative valence with high stillness should be Sadness, not Peace"
        );
        assert_eq!(emotion, EmotionalQuality::Sadness);
    }

    #[test]
    fn gesture_applies_direction() {
        let scale = vec![261.63, 293.66, 329.63, 349.23, 392.00, 440.00];
        let gesture = gesture_for_emotion(EmotionalQuality::Joy); // ascending bias

        // If prev note was 329.63 and base is 293.66 (descending), should push up
        let result = apply_gesture_to_pitch(293.66, Some(329.63), &scale, &gesture);
        assert!(result >= 293.66, "joy should push upward: {result}");
    }
}
