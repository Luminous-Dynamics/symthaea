// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pitch system: maps Eight Harmonies to musical intervals and scales.

use crate::MusicalState;

/// Semitone intervals from root for each harmony.
/// These map the Eight Harmonies to characteristic musical intervals.
const HARMONY_INTERVALS: [i32; 8] = [
    0,  // ResonantCoherence → Unison (perfect consonance)
    4,  // PanSentientFlourishing → Major 3rd (warmth)
    7,  // IntegralWisdom → Perfect 5th (stability)
    10, // InfinitePlay → Minor 7th (tension/play)
    5,  // UniversalInterconnectedness → Perfect 4th (openness)
    9,  // SacredReciprocity → Major 6th (warmth)
    2,  // EvolutionaryProgression → Major 2nd (ascending motion)
    0,  // SacredStillness → Unison (pedal/drone)
];

/// Base frequency for middle C (C4).
const C4_HZ: f32 = 261.63;

/// Convert semitones from C4 to frequency in Hz.
pub fn semitones_to_hz(semitones: i32) -> f32 {
    C4_HZ * 2.0_f32.powf(semitones as f32 / 12.0)
}

/// Build a scale from the current harmony activations.
///
/// Active harmonies (> 0.3) contribute their characteristic intervals.
/// Valence shifts between major (positive) and minor (negative) feel
/// by flatting the 3rd and 6th in minor mode.
pub fn build_scale(state: &MusicalState) -> Vec<f32> {
    let mut intervals: Vec<i32> = Vec::new();

    for (i, &activation) in state.harmony_activations.iter().enumerate() {
        if activation > 0.2 {
            let mut interval = HARMONY_INTERVALS[i];

            // Minor mode: flat the 3rd (4→3) and 6th (9→8)
            if state.valence < -0.3 {
                if interval == 4 {
                    interval = 3; // minor 3rd
                }
                if interval == 9 {
                    interval = 8; // minor 6th
                }
            }

            if !intervals.contains(&interval) {
                intervals.push(interval);
            }
        }
    }

    // Always include root
    if !intervals.contains(&0) {
        intervals.push(0);
    }

    intervals.sort();

    // Extend across 2 octaves
    let mut frequencies = Vec::new();
    for octave_offset in [-12, 0, 12] {
        for &interval in &intervals {
            let semitones = interval + octave_offset;
            let hz = semitones_to_hz(semitones);
            if hz >= 65.0 && hz <= 2000.0 {
                // Musical range
                frequencies.push(hz);
            }
        }
    }

    frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    frequencies.dedup_by(|a, b| (*a - *b).abs() < 0.1);
    frequencies
}

/// Select a scale degree based on a selector value [0, 1].
pub fn select_pitch(scale: &[f32], selector: f32) -> f32 {
    if scale.is_empty() {
        return C4_HZ;
    }
    let idx = (selector * (scale.len() - 1) as f32).round() as usize;
    scale[idx.min(scale.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c4_frequency() {
        let hz = semitones_to_hz(0);
        assert!((hz - 261.63).abs() < 0.1);
    }

    #[test]
    fn perfect_fifth() {
        let hz = semitones_to_hz(7);
        assert!((hz - 392.0).abs() < 1.0); // G4 ≈ 392 Hz
    }

    #[test]
    fn octave() {
        let hz = semitones_to_hz(12);
        assert!((hz - 523.25).abs() < 1.0); // C5
    }

    #[test]
    fn build_scale_includes_root() {
        let state = MusicalState::default();
        let scale = build_scale(&state);
        assert!(!scale.is_empty());
        // Should include C4 (261.63)
        assert!(scale.iter().any(|&f| (f - C4_HZ).abs() < 1.0));
    }

    #[test]
    fn minor_mode_flats_third() {
        let major_state = MusicalState {
            valence: 0.5,
            harmony_activations: [0.5; 8],
            ..Default::default()
        };
        let minor_state = MusicalState {
            valence: -0.5,
            harmony_activations: [0.5; 8],
            ..Default::default()
        };
        let major_scale = build_scale(&major_state);
        let minor_scale = build_scale(&minor_state);
        // Scales should differ (minor has flatted 3rd and 6th)
        assert_ne!(major_scale.len(), 0);
        assert_ne!(minor_scale.len(), 0);
        assert_ne!(major_scale, minor_scale);
    }

    #[test]
    fn all_frequencies_in_range() {
        let state = MusicalState {
            harmony_activations: [1.0; 8],
            ..Default::default()
        };
        let scale = build_scale(&state);
        for &f in &scale {
            assert!(f >= 60.0 && f <= 2100.0, "frequency {f} out of range");
        }
    }

    #[test]
    fn select_pitch_bounds() {
        let scale = vec![100.0, 200.0, 300.0, 400.0];
        assert!((select_pitch(&scale, 0.0) - 100.0).abs() < 0.1);
        assert!((select_pitch(&scale, 1.0) - 400.0).abs() < 0.1);
    }
}
