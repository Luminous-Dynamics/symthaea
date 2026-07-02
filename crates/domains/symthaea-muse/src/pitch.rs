// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pitch system: maps Eight Harmonies to musical intervals and scales.

use serde::{Deserialize, Serialize};

use crate::MusicalState;

// ─── Tuning Systems ──────────────────────────────────────────────────────────

/// Tuning system for scale generation.
///
/// Extends beyond 12-TET to support non-Western musical traditions.
/// Each variant produces its characteristic pitch set in Hz, allowing
/// Symthaea to compose in any of the world's musical traditions.
///
/// # Cultural Note
/// These tuning systems represent living musical traditions. Implementation
/// uses pitch ratios from ethnomusicological sources (Touma 1996, Powers 2001,
/// Tenzer 2000) to preserve their characteristic intervals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TuningSystem {
    /// Standard Western 12-tone equal temperament (default).
    TwelveTET,

    /// Arabic/Turkish maqamat (modal scales with microtones).
    ///
    /// Uses 24-TET quarter-tones as the basis. The maqam name selects
    /// the characteristic interval pattern and jins (trichords/tetrachords).
    /// Reference: Touma, H. H. (1996). *The Music of the Arabs*. Amadeus Press.
    Maqamat { maqam: MaqamMode },

    /// Indian classical raga (North Indian / Hindustani system).
    ///
    /// Each raga has ascending (aroh) and descending (avroh) forms,
    /// characteristic ornaments (gamaka), and time-of-day associations.
    /// Reference: Powers, H. S. (2001). "Raga." Grove Music Online.
    Raga { raga: RagaMode },

    /// Javanese/Balinese gamelan scales (pelog and slendro).
    ///
    /// Gamelan tuning varies between ensembles — these are canonical
    /// theoretical tunings. Reference: Tenzer, M. (2000). *Gamelan Gong Kebyar*.
    Gamelan { tuning: GamelanTuning },

    /// Just intonation using pure harmonic ratios.
    ///
    /// All intervals are rational multiples of the fundamental frequency,
    /// producing beatless consonances unavailable in equal temperament.
    JustIntonation,

    /// User-defined microtonal scale specified as cents offsets from the root.
    ///
    /// 100 cents = 1 semitone. Values can be any real number.
    /// Example: `[0.0, 150.0, 350.0, 500.0, 700.0]` = pentatonic with microtones.
    Microtonal { cents: Vec<f32> },
}

impl Default for TuningSystem {
    fn default() -> Self {
        Self::TwelveTET
    }
}

/// Arabic/Turkish maqam modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaqamMode {
    /// Maqam Rast — the most common maqam, with a half-flat 3rd and 7th.
    /// Character: bright, confident. Used in folk and classical Arabic music.
    Rast,
    /// Maqam Bayati — minor feel with a distinctive half-flat 2nd.
    /// Character: expressive, emotional. Common in Egyptian classical music.
    Bayati,
    /// Maqam Hijaz — distinctive augmented 2nd, sounds "exotic" to Western ears.
    /// Character: yearning, dramatic. Named after the Hejaz region of Arabia.
    Hijaz,
    /// Maqam Saba — sorrowful, with multiple half-flat intervals.
    /// Character: melancholy, introspective.
    Saba,
    /// Maqam Nahawand — close to Western natural minor.
    /// Character: warm, emotional. The most accessible maqam for Western listeners.
    Nahawand,
    /// Maqam Kurd — Phrygian-like with a very low 2nd degree.
    /// Character: austere, ancient-sounding.
    Kurd,
    /// Maqam Ushshaq — joyful, with a characteristic ascending pattern.
    Ushshaq,
}

/// Hindustani raga modes (selection of common ragas).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RagaMode {
    /// Raga Bhairav — morning raga, serious, devotional character.
    /// Uses both flat 2nd and flat 6th.
    Bhairav,
    /// Raga Yaman — evening raga, romantic. Uses sharp 4th (tritone from root).
    /// The most commonly taught raga.
    Yaman,
    /// Raga Bhairavi — versatile, often ends concerts. Uses many flats.
    Bhairavi,
    /// Raga Kafi — light, folksy. Based on natural minor with a flat 3rd and 7th.
    Kafi,
    /// Raga Bilawal — equivalent to major scale. Pure, morning raga.
    Bilawal,
    /// Raga Todi — complex, distinctive double-flat 2nd with sharp 4th.
    /// Character: intense, searching.
    Todi,
    /// Raga Desh — rain raga, joyful. Pentatonic ascending, heptatonic descending.
    Desh,
}

/// Gamelan tuning systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GamelanTuning {
    /// Pelog — 7-note scale with large and small intervals. Ceremonial character.
    Pelog,
    /// Slendro — 5-note near-equidistant scale. More joyful, earthy character.
    Slendro,
}

/// Select a tuning system from the current emotional and consciousness state.
///
/// Maps the valence-arousal quadrant to culturally meaningful scale families:
///
/// | Quadrant            | Tuning                | Character                    |
/// |---------------------|-----------------------|------------------------------|
/// | +V +A (joy)         | Raga Desh             | Joyful, rain imagery         |
/// | +V −A (serenity)    | Gamelan Slendro       | Contemplative, meditative    |
/// | −V +A (tension)     | Maqam Hijaz           | Dramatic, augmented 2nd      |
/// | −V −A (sorrow)      | Maqam Saba            | Sorrowful, half-flat cascade |
/// | High consciousness  | Just Intonation       | Pure harmonic ratios         |
/// | Default / moderate  | 12TET                 | Standard Western tuning      |
///
/// The consciousness_level acts as a gate: when Ψ > 0.7, Just Intonation is
/// chosen regardless of emotion — the system "hears" pure intervals. At very
/// low consciousness (< 0.2), 12TET is always used (safe fallback).
///
/// References:
/// - Balkwill & Thompson (1999): cross-cultural emotional cues in music
/// - Castellano, Bharucha & Krumhansl (1984): tonal hierarchies across cultures
/// - Fritz et al. (2009): universal recognition of basic emotions in music
pub fn select_tuning_system(state: &MusicalState) -> TuningSystem {
    // High consciousness → pure harmonics (Just Intonation)
    if state.consciousness_level > 0.7 {
        return TuningSystem::JustIntonation;
    }
    // Very low consciousness → safe 12TET fallback
    if state.consciousness_level < 0.2 {
        return TuningSystem::TwelveTET;
    }
    // Use valence-arousal quadrant for cultural selection
    let v = state.valence;
    let a = state.arousal;

    if v > 0.2 && a > 0.5 {
        // Joyful, high-energy: Raga Desh (rain raga, celebratory pentatonic)
        TuningSystem::Raga {
            raga: RagaMode::Desh,
        }
    } else if v > 0.2 && a <= 0.5 {
        // Serene, peaceful: Gamelan Slendro (5-note equidistant, contemplative)
        TuningSystem::Gamelan {
            tuning: GamelanTuning::Slendro,
        }
    } else if v < -0.2 && a > 0.5 {
        // Tense, dramatic: Maqam Hijaz (augmented 2nd, intense)
        TuningSystem::Maqamat {
            maqam: MaqamMode::Hijaz,
        }
    } else if v < -0.2 && a <= 0.5 {
        // Sorrowful, subdued: Maqam Saba (half-flat intervals, grief)
        TuningSystem::Maqamat {
            maqam: MaqamMode::Saba,
        }
    } else {
        // Neutral zone: standard Western tuning
        TuningSystem::TwelveTET
    }
}

/// Build a scale from the current harmony activations using a specific tuning system.
///
/// This is the culturally-extended version of `build_scale()`. The root frequency
/// is C4 (261.63 Hz) by default, modulated by the valence-driven key selection.
pub fn build_scale_with_tuning(state: &MusicalState, tuning: &TuningSystem) -> Vec<f32> {
    match tuning {
        TuningSystem::TwelveTET => build_scale(state),
        TuningSystem::Maqamat { maqam } => build_maqam_scale(state, *maqam),
        TuningSystem::Raga { raga } => build_raga_scale(state, *raga),
        TuningSystem::Gamelan { tuning } => build_gamelan_scale(state, *tuning),
        TuningSystem::JustIntonation => build_just_scale(state),
        TuningSystem::Microtonal { cents } => build_microtonal_scale(state, cents),
    }
}

/// Convert cents offset from root to frequency in Hz.
///
/// 100 cents = 1 semitone. cents = 0 → root = C4.
pub fn cents_to_hz(cents: f32) -> f32 {
    C4_HZ * 2.0_f32.powf(cents / 1200.0)
}

/// Build a maqam scale (Arabic/Turkish modal system with quarter-tones).
///
/// Uses 24-TET (50 cents per step) for quarter-tone representation.
/// The H (half-flat, 50 cents lower) symbol is standard in maqam notation.
fn build_maqam_scale(state: &MusicalState, maqam: MaqamMode) -> Vec<f32> {
    // Cent offsets from root. "H" = half-flat = -50 cents from the equal-tempered pitch.
    // Reference: Touma (1996), Arel-Ezgi-Uzdilek (AEU) system.
    let cents: &[f32] = match maqam {
        // Rast: T T H T T T H (where H = 150c, T = 200c, but with 3rd and 7th half-flattened)
        // Degrees: 0, 200, 350, 500, 700, 900, 1050, 1200
        MaqamMode::Rast => &[0.0, 200.0, 350.0, 500.0, 700.0, 900.0, 1050.0],

        // Bayati: H T T T H T T — characteristic half-flat 2nd
        // Degrees: 0, 150, 300, 500, 700, 800, 1000
        MaqamMode::Bayati => &[0.0, 150.0, 300.0, 500.0, 700.0, 800.0, 1000.0],

        // Hijaz: S A S T S A S — augmented 2nd between 2nd and 3rd
        // (S=semitone=100c, A=augmented=300c)
        // Degrees: 0, 100, 400, 500, 700, 800, 1100
        MaqamMode::Hijaz => &[0.0, 100.0, 400.0, 500.0, 700.0, 800.0, 1100.0],

        // Saba: T H H A H T T — very sorrowful with multiple microtones
        // Degrees: 0, 200, 350, 450, 700, 800, 1000
        MaqamMode::Saba => &[0.0, 200.0, 350.0, 450.0, 700.0, 800.0, 1000.0],

        // Nahawand: T S T T S T T — natural minor equivalent
        // Degrees: 0, 200, 300, 500, 700, 800, 1000
        MaqamMode::Nahawand => &[0.0, 200.0, 300.0, 500.0, 700.0, 800.0, 1000.0],

        // Kurd: S T T T S T T — Phrygian-like, very low 2nd
        // Degrees: 0, 100, 300, 500, 700, 800, 1000
        MaqamMode::Kurd => &[0.0, 100.0, 300.0, 500.0, 700.0, 800.0, 1000.0],

        // Ushshaq: T H T T T H T — joyful ascending character
        // Degrees: 0, 200, 350, 500, 700, 900, 1050
        MaqamMode::Ushshaq => &[0.0, 200.0, 350.0, 500.0, 700.0, 900.0, 1050.0],
    };

    expand_cents_to_range(cents, state)
}

/// Build a raga scale (Hindustani classical system).
///
/// Many ragas have different ascending (aroh) and descending (avroh) patterns.
/// We use the ascending form for simplicity; a full implementation would
/// return both and the melody generator would select based on contour direction.
fn build_raga_scale(state: &MusicalState, raga: RagaMode) -> Vec<f32> {
    let cents: &[f32] = match raga {
        // Bhairav: S R♭ G♭ M P D♭ N S (flat 2nd and 6th, Phrygian-adjacent)
        // Degrees: 0, 100, 400, 500, 700, 800, 1100
        RagaMode::Bhairav => &[0.0, 100.0, 400.0, 500.0, 700.0, 800.0, 1100.0],

        // Yaman: S R G M# P D N S (sharp 4th = Lydian)
        // Degrees: 0, 200, 400, 600, 700, 900, 1100
        RagaMode::Yaman => &[0.0, 200.0, 400.0, 600.0, 700.0, 900.0, 1100.0],

        // Bhairavi: S R♭ G♭ M P D♭ N♭ S (almost all flats)
        // Degrees: 0, 100, 300, 500, 700, 800, 1000
        RagaMode::Bhairavi => &[0.0, 100.0, 300.0, 500.0, 700.0, 800.0, 1000.0],

        // Kafi: S R G♭ M P D N♭ S (natural minor)
        // Degrees: 0, 200, 300, 500, 700, 900, 1000
        RagaMode::Kafi => &[0.0, 200.0, 300.0, 500.0, 700.0, 900.0, 1000.0],

        // Bilawal: S R G M P D N S (major scale)
        // Degrees: 0, 200, 400, 500, 700, 900, 1100
        RagaMode::Bilawal => &[0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],

        // Todi: S R♭ G♭ M# P D♭ N S (very complex, flat 2nd, double-flat 3rd, sharp 4th)
        // Degrees: 0, 100, 300, 600, 700, 800, 1100
        RagaMode::Todi => &[0.0, 100.0, 300.0, 600.0, 700.0, 800.0, 1100.0],

        // Desh: ascending S R M P N S (pentatonic), descending adds G and D
        // Using ascending form: 0, 200, 500, 700, 1100
        RagaMode::Desh => &[0.0, 200.0, 500.0, 700.0, 1100.0],
    };

    expand_cents_to_range(cents, state)
}

/// Build a gamelan scale (Javanese/Balinese tuning system).
///
/// Gamelan tuning varies between ensembles. These are the theoretical
/// canonical tunings used for composition purposes.
fn build_gamelan_scale(state: &MusicalState, tuning: GamelanTuning) -> Vec<f32> {
    let cents: &[f32] = match tuning {
        // Pelog: 7 tones with large (large ~240c) and small (~120c) intervals.
        // Theoretical canonical pelog from Tenzer (2000).
        // Degrees: 0, 120, 240, 480, 600, 720, 960
        GamelanTuning::Pelog => &[0.0, 120.0, 240.0, 480.0, 600.0, 720.0, 960.0],

        // Slendro: 5 near-equidistant tones (~240c each).
        // Theoretical canonical slendro.
        // Degrees: 0, 240, 480, 720, 960
        GamelanTuning::Slendro => &[0.0, 240.0, 480.0, 720.0, 960.0],
    };

    // Gamelan doesn't naturally have a Western root — use C4 as anchor.
    // Consciousness level gates register (high consciousness = broader range).
    let range_octaves = if state.consciousness_level > 0.5 {
        2
    } else {
        1
    };
    let mut freqs = Vec::new();
    for oct in 0..range_octaves {
        for &c in cents {
            let hz = cents_to_hz(c + oct as f32 * 1200.0);
            if hz >= 65.0 && hz <= 2000.0 {
                freqs.push(hz);
            }
        }
    }
    freqs.sort_by(|a, b| a.total_cmp(b));
    freqs
}

/// Build a just intonation scale using pure harmonic ratios.
///
/// Uses the 5-limit just intonation system (ratios involving primes 2, 3, 5).
/// Produces beatless consonances impossible in equal temperament.
fn build_just_scale(state: &MusicalState) -> Vec<f32> {
    // 5-limit just major scale ratios: 1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2
    // Semitone equivalents in cents: 0, 204, 386, 498, 702, 884, 1088, 1200
    let just_major_cents = [0.0f32, 203.9, 386.3, 498.0, 702.0, 884.4, 1088.3];

    // Minor variant: use 6/5 minor third (315.6c), 5/4 → 6/5, etc.
    let just_minor_cents = [0.0f32, 203.9, 315.6, 498.0, 702.0, 813.7, 1017.6];

    let minor_weight = (0.5 - state.valence * 0.5).clamp(0.0, 1.0);
    let cents: &[f32] = if minor_weight > 0.5 {
        &just_minor_cents
    } else {
        &just_major_cents
    };

    expand_cents_to_range(cents, state)
}

/// Build a scale from user-specified cent offsets.
fn build_microtonal_scale(state: &MusicalState, cents: &[f32]) -> Vec<f32> {
    expand_cents_to_range(cents, state)
}

/// Expand a set of cent offsets into a multi-octave frequency array.
///
/// Applies the same range filtering and octave extension as `build_scale()`.
fn expand_cents_to_range(cents: &[f32], state: &MusicalState) -> Vec<f32> {
    let _ = state; // used for future key-root modulation
    let mut freqs = Vec::new();
    for oct_offset in [-1200.0f32, 0.0, 1200.0] {
        for &c in cents {
            let hz = cents_to_hz(c + oct_offset);
            if hz >= 65.0 && hz <= 2000.0 {
                freqs.push(hz);
            }
        }
    }
    // Always include root
    let root = C4_HZ;
    if !freqs.iter().any(|&f| (f - root).abs() < 0.5) {
        freqs.push(root);
    }
    freqs.sort_by(|a, b| a.total_cmp(b));
    freqs.dedup_by(|a, b| (*a - *b).abs() < 5.0); // deduplicate within 5 Hz
    freqs
}

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

            // Continuous minor mode: valence controls how "minor" the scale sounds.
            // v=+1.0 → always major, v=0.0 → 50% minor, v=-1.0 → always minor.
            // This eliminates the -0.3 dead zone (Eerola et al. 2013).
            let minor_weight = (0.5 - state.valence * 0.5).clamp(0.0, 1.0);
            if interval == 4 && minor_weight > 0.5 {
                interval = 3; // minor 3rd
            }
            if interval == 9 && minor_weight > 0.4 {
                interval = 8; // minor 6th
            }
            // At very negative valence, also flatten the 7th (11→10) for dorian feel
            if interval == 11 && minor_weight > 0.7 {
                interval = 10; // minor 7th
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

    frequencies.sort_by(|a, b| a.total_cmp(b));
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

    // ─── Tuning system tests ──────────────────────────────────────────────────

    #[test]
    fn cents_to_hz_octave() {
        let hz = cents_to_hz(1200.0);
        assert!(
            (hz - C4_HZ * 2.0).abs() < 0.5,
            "1200c should be one octave: {hz}"
        );
    }

    #[test]
    fn cents_to_hz_semitone() {
        let hz = cents_to_hz(100.0);
        let expected = C4_HZ * 2.0_f32.powf(1.0 / 12.0);
        assert!(
            (hz - expected).abs() < 0.5,
            "100c = 1 semitone: {hz} vs {expected}"
        );
    }

    #[test]
    fn tuning_twelve_tet_matches_build_scale() {
        let state = MusicalState::default();
        let tet = build_scale_with_tuning(&state, &TuningSystem::TwelveTET);
        let classic = build_scale(&state);
        assert_eq!(tet, classic, "TwelveTET should match build_scale");
    }

    #[test]
    fn maqam_rast_has_microtones() {
        let state = MusicalState::default();
        let scale = build_scale_with_tuning(
            &state,
            &TuningSystem::Maqamat {
                maqam: MaqamMode::Rast,
            },
        );
        assert!(!scale.is_empty());
        // Rast has a half-flat 3rd at ~350 cents — should produce non-12-TET frequency
        // Check that some frequency is NOT on a 12-TET grid (i.e., not n*semitone from C4)
        let has_microtone = scale.iter().any(|&f| {
            let ratio = f / C4_HZ;
            let semitones = ratio.log2() * 12.0;
            let deviation = (semitones - semitones.round()).abs();
            deviation > 0.05 // more than 5 cents off the grid
        });
        assert!(
            has_microtone,
            "Maqam Rast should contain microtonal pitches"
        );
    }

    #[test]
    fn raga_yaman_has_sharp_four() {
        let state = MusicalState::default();
        let scale = build_scale_with_tuning(
            &state,
            &TuningSystem::Raga {
                raga: RagaMode::Yaman,
            },
        );
        assert!(!scale.is_empty());
        // Yaman has a sharp 4th (tritone = 600 cents from root = ~370 Hz from C4)
        let has_tritone = scale.iter().any(|&f| (f - cents_to_hz(600.0)).abs() < 5.0);
        assert!(
            has_tritone,
            "Yaman should contain the sharp 4th (tritone): {scale:?}"
        );
    }

    #[test]
    fn gamelan_slendro_pentatonic() {
        let state = MusicalState::default();
        let scale = build_scale_with_tuning(
            &state,
            &TuningSystem::Gamelan {
                tuning: GamelanTuning::Slendro,
            },
        );
        // Slendro has 5 degrees per octave
        // With range filtering, expect at least 5 pitches
        assert!(
            scale.len() >= 5,
            "Slendro should have >= 5 pitches, got {}",
            scale.len()
        );
    }

    #[test]
    fn just_intonation_has_pure_fifth() {
        let state = MusicalState::default();
        let scale = build_scale_with_tuning(&state, &TuningSystem::JustIntonation);
        // Just perfect fifth = 3/2 ratio = C4 * 1.5 ≈ 392.44 Hz
        let expected_fifth = C4_HZ * 1.5;
        let has_just_fifth = scale.iter().any(|&f| (f - expected_fifth).abs() < 1.0);
        assert!(
            has_just_fifth,
            "Just intonation should have pure 3/2 fifth: {scale:?}"
        );
    }

    #[test]
    fn microtonal_custom_scale() {
        let state = MusicalState::default();
        let cents = vec![0.0, 150.0, 350.0, 550.0, 750.0]; // arbitrary microtonal scale
        let scale = build_scale_with_tuning(&state, &TuningSystem::Microtonal { cents });
        assert!(!scale.is_empty());
        // Check a custom pitch is present: 150 cents ≈ C4 * 2^(150/1200) ≈ 290.3 Hz
        let expected = cents_to_hz(150.0);
        let has_custom = scale.iter().any(|&f| (f - expected).abs() < 2.0);
        assert!(
            has_custom,
            "Custom microtonal scale should contain 150c pitch: {scale:?}"
        );
    }

    #[test]
    fn all_tuning_systems_non_empty() {
        let state = MusicalState::default();
        let systems = vec![
            TuningSystem::TwelveTET,
            TuningSystem::Maqamat {
                maqam: MaqamMode::Bayati,
            },
            TuningSystem::Maqamat {
                maqam: MaqamMode::Hijaz,
            },
            TuningSystem::Raga {
                raga: RagaMode::Bhairav,
            },
            TuningSystem::Raga {
                raga: RagaMode::Todi,
            },
            TuningSystem::Gamelan {
                tuning: GamelanTuning::Pelog,
            },
            TuningSystem::Gamelan {
                tuning: GamelanTuning::Slendro,
            },
            TuningSystem::JustIntonation,
            TuningSystem::Microtonal {
                cents: vec![0.0, 240.0, 480.0, 720.0, 960.0],
            },
        ];
        for system in &systems {
            let scale = build_scale_with_tuning(&state, system);
            assert!(!scale.is_empty(), "{system:?} produced empty scale");
        }
    }

    #[test]
    fn select_tuning_high_consciousness_is_just() {
        let state = MusicalState {
            consciousness_level: 0.8,
            ..Default::default()
        };
        assert_eq!(select_tuning_system(&state), TuningSystem::JustIntonation);
    }

    #[test]
    fn select_tuning_low_consciousness_is_tet() {
        let state = MusicalState {
            consciousness_level: 0.1,
            ..Default::default()
        };
        assert_eq!(select_tuning_system(&state), TuningSystem::TwelveTET);
    }

    #[test]
    fn select_tuning_joyful_is_raga_desh() {
        let state = MusicalState {
            valence: 0.6,
            arousal: 0.7,
            consciousness_level: 0.5,
            ..Default::default()
        };
        assert!(matches!(
            select_tuning_system(&state),
            TuningSystem::Raga {
                raga: RagaMode::Desh
            }
        ));
    }

    #[test]
    fn select_tuning_tense_is_maqam_hijaz() {
        let state = MusicalState {
            valence: -0.5,
            arousal: 0.8,
            consciousness_level: 0.5,
            ..Default::default()
        };
        assert!(matches!(
            select_tuning_system(&state),
            TuningSystem::Maqamat {
                maqam: MaqamMode::Hijaz
            }
        ));
    }

    #[test]
    fn select_tuning_serene_is_gamelan() {
        let state = MusicalState {
            valence: 0.4,
            arousal: 0.2,
            consciousness_level: 0.5,
            ..Default::default()
        };
        assert!(matches!(
            select_tuning_system(&state),
            TuningSystem::Gamelan {
                tuning: GamelanTuning::Slendro
            }
        ));
    }

    #[test]
    fn select_tuning_sorrowful_is_maqam_saba() {
        let state = MusicalState {
            valence: -0.6,
            arousal: 0.2,
            consciousness_level: 0.5,
            ..Default::default()
        };
        assert!(matches!(
            select_tuning_system(&state),
            TuningSystem::Maqamat {
                maqam: MaqamMode::Saba
            }
        ));
    }

    #[test]
    fn all_pitches_in_musical_range() {
        let state = MusicalState::default();
        for system in [
            TuningSystem::Maqamat {
                maqam: MaqamMode::Rast,
            },
            TuningSystem::Raga {
                raga: RagaMode::Yaman,
            },
            TuningSystem::Gamelan {
                tuning: GamelanTuning::Pelog,
            },
            TuningSystem::JustIntonation,
        ] {
            let scale = build_scale_with_tuning(&state, &system);
            for &f in &scale {
                assert!(
                    f >= 60.0 && f <= 2100.0,
                    "{system:?}: pitch {f} out of musical range"
                );
            }
        }
    }
}
