// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rhythm engine: odd time signatures, syncopation, and groove pocket.
//!
//! Built for ears that listen to Dream Theater, Animals As Leaders,
//! Vulfpeck, and Khruangbin. This module generates rhythmic patterns
//! that aren't just 4/4 — it understands 5/4, 7/8, mixed meter,
//! and the micro-timing that creates groove.
//!
//! The pocket: notes aren't quantized to a grid. They're placed with
//! intentional micro-timing deviations that create the feeling of
//! "in the groove" — slightly behind the beat for laid-back feel,
//! slightly ahead for urgency.

use crate::MusicalState;

/// Time signature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeSignature {
    pub numerator: u8,   // beats per bar
    pub denominator: u8, // beat unit (4 = quarter, 8 = eighth)
}

impl TimeSignature {
    pub const FOUR_FOUR: Self = Self {
        numerator: 4,
        denominator: 4,
    };
    pub const THREE_FOUR: Self = Self {
        numerator: 3,
        denominator: 4,
    };
    pub const FIVE_FOUR: Self = Self {
        numerator: 5,
        denominator: 4,
    };
    pub const SEVEN_EIGHT: Self = Self {
        numerator: 7,
        denominator: 8,
    };
    pub const SIX_EIGHT: Self = Self {
        numerator: 6,
        denominator: 8,
    };
    pub const NINE_EIGHT: Self = Self {
        numerator: 9,
        denominator: 8,
    };
    pub const FIVE_EIGHT: Self = Self {
        numerator: 5,
        denominator: 8,
    };

    /// Beats per bar as float (for timing calculations).
    pub fn beats_per_bar(&self) -> f32 {
        self.numerator as f32 * (4.0 / self.denominator as f32)
    }

    /// Beat groupings (how odd meters subdivide).
    /// 7/8 = 2+2+3, 5/4 = 3+2, etc.
    pub fn groupings(&self) -> Vec<u8> {
        match (self.numerator, self.denominator) {
            (7, 8) => vec![2, 2, 3], // AAL, Dream Theater style
            (5, 4) => vec![3, 2],    // Take Five, Sting
            (5, 8) => vec![3, 2],
            (9, 8) => vec![3, 3, 3], // compound triple
            (6, 8) => vec![3, 3],    // compound duple
            (3, 4) => vec![3],       // waltz
            (4, 4) => vec![4],       // standard
            (n, _) => {
                // Generic: split into 2s and 3s
                let mut groups = Vec::new();
                let mut remaining = n;
                while remaining > 0 {
                    if remaining >= 3 && remaining != 4 {
                        groups.push(3);
                        remaining -= 3;
                    } else {
                        groups.push(2.min(remaining));
                        remaining = remaining.saturating_sub(2);
                    }
                }
                groups
            }
        }
    }

    /// Strong beat positions (for accent patterns).
    pub fn strong_beats(&self) -> Vec<f32> {
        let groups = self.groupings();
        let mut positions = vec![0.0f32];
        let mut pos = 0.0;
        let beat_unit = 4.0 / self.denominator as f32;
        for &g in &groups[..groups.len().saturating_sub(1)] {
            pos += g as f32 * beat_unit;
            positions.push(pos);
        }
        positions
    }
}

/// Select time signature based on consciousness state.
///
/// Higher consciousness + higher prediction error = more complex meters.
/// Low consciousness = simple 4/4 or 3/4.
/// High consciousness + high PE = 7/8, 5/4.
pub fn select_time_signature(state: &MusicalState) -> TimeSignature {
    let psi = state.consciousness_level;
    let pe = state.prediction_error;
    let complexity = psi * 0.6 + pe * 0.4;

    if complexity > 0.7 {
        // Complex: odd meters
        let progress = state.harmony_activations[6]; // EvolutionaryProgression
        if progress > 0.5 {
            TimeSignature::SEVEN_EIGHT // Dream Theater territory
        } else {
            TimeSignature::FIVE_FOUR // Take Five
        }
    } else if complexity > 0.5 {
        // Moderate: compound meters
        if state.arousal > 0.5 {
            TimeSignature::SIX_EIGHT // driving compound
        } else {
            TimeSignature::THREE_FOUR // waltz
        }
    } else {
        TimeSignature::FOUR_FOUR // ground state
    }
}

/// Groove pocket: micro-timing deviations that create feel.
#[derive(Debug, Clone)]
pub struct GroovePocket {
    /// Swing amount [0, 1]: 0 = straight, 0.5 = hard swing.
    pub swing: f32,
    /// Laid-back amount [0, 1]: 0 = on grid, 1 = maximally behind beat.
    pub laid_back: f32,
    /// Ghost note probability [0, 1]: quiet notes between main beats.
    pub ghost_notes: f32,
}

impl GroovePocket {
    /// Select groove feel based on consciousness state.
    pub fn from_state(state: &MusicalState) -> Self {
        let serotonin = state.serotonin;
        let dopamine = state.dopamine;
        let arousal = state.arousal;

        // High serotonin = laid-back (Khruangbin, reggae)
        // High dopamine = energetic swing (funk, Vulfpeck)
        // High arousal = tight, on-grid (metal, intense)
        Self {
            swing: dopamine * 0.3 * (1.0 - arousal * 0.5), // swing reduces at high arousal
            laid_back: serotonin * 0.2 * (1.0 - arousal * 0.7), // tight at high arousal
            ghost_notes: dopamine * 0.4,                   // funk = ghost notes
        }
    }

    /// Apply groove to a beat position. Returns the adjusted position.
    pub fn apply(&self, beat_pos: f32, is_offbeat: bool) -> f32 {
        let mut pos = beat_pos;

        // Swing: delay offbeats
        if is_offbeat {
            pos += self.swing * 0.15; // up to 15% of beat duration
        }

        // Laid-back: slight delay on all notes
        pos += self.laid_back * 0.03; // up to 3% behind

        pos
    }

    /// Should a ghost note be inserted at this position?
    pub fn insert_ghost(&self, seed: u32) -> bool {
        let r = (seed.wrapping_mul(2654435761) >> 16) as f32 / 65536.0;
        r < self.ghost_notes * 0.3 // up to 12% chance at max ghost
    }

    /// Ghost note velocity (much quieter than regular notes).
    pub fn ghost_velocity(&self) -> f32 {
        0.15 + self.ghost_notes * 0.1 // 15-25% of normal
    }
}

/// Syncopation pattern generator.
///
/// Syncopation = emphasis on weak beats or off-beats.
/// Creates rhythmic tension that resolves on strong beats.
pub struct SyncopationPattern {
    /// Beat positions where notes should land (relative to bar).
    pub onset_positions: Vec<f32>,
    /// Accent strengths for each position [0, 1].
    pub accents: Vec<f32>,
}

impl SyncopationPattern {
    /// Generate a syncopation pattern for the given time signature.
    pub fn generate(time_sig: &TimeSignature, syncopation_level: f32, seed: u32) -> Self {
        let beats = time_sig.beats_per_bar();
        let subdivision = 4; // 16th note resolution
        let total_slots = (beats * subdivision as f32) as usize;
        let strong = time_sig.strong_beats();

        let mut onsets = Vec::new();
        let mut accents = Vec::new();
        let mut s = seed;

        for slot in 0..total_slots {
            let pos = slot as f32 / subdivision as f32;
            let is_strong = strong.iter().any(|&sb| (pos - sb).abs() < 0.1);
            let is_beat = slot % subdivision == 0;

            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let r = (s >> 16) as f32 / 65536.0;

            // High syncopation = more offbeat notes, fewer downbeat notes
            let threshold = if is_strong {
                1.0 - syncopation_level * 0.5 // still likely on strong beats
            } else if is_beat {
                0.6 - syncopation_level * 0.2 // moderate on beats
            } else {
                0.3 + syncopation_level * 0.3 // more likely on offbeats when syncopated
            };

            if r < threshold {
                onsets.push(pos);

                // Accent: syncopated notes get emphasis
                let accent = if is_strong && syncopation_level < 0.5 {
                    1.0 // normal strong beat emphasis
                } else if !is_beat && syncopation_level > 0.3 {
                    0.7 + syncopation_level * 0.3 // syncopated accent
                } else {
                    0.5
                };
                accents.push(accent);
            }
        }

        Self {
            onset_positions: onsets,
            accents,
        }
    }

    /// Get the next onset position after the given beat position.
    pub fn next_onset(&self, current_beat: f32) -> Option<(f32, f32)> {
        self.onset_positions
            .iter()
            .zip(self.accents.iter())
            .find(|&(&pos, _)| pos > current_beat + 0.01)
            .map(|(&pos, &accent)| (pos, accent))
    }
}

/// Extended chord types beyond basic triads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtendedChordType {
    Major,
    Minor,
    Major7,      // Cmaj7 (jazzy, Steven Wilson)
    Minor7,      // Cm7 (jazz standard)
    Dominant7,   // C7 (blues, funk — RHCP)
    Minor9,      // Cm9 (neo-soul, Spalding)
    Major9,      // Cmaj9 (dreamy, Pink Floyd)
    Sus2,        // Csus2 (open, atmospheric)
    Sus4,        // Csus4 (tension)
    Diminished7, // Cdim7 (tension, Dream Theater)
    Augmented,   // Caug (mysterious)
    Add9,        // Cadd9 (bright, Yvette Young)
    Minor11,     // Cm11 (deep jazz)
    Dominant9,   // C9 (funk, Vulfpeck)
}

impl ExtendedChordType {
    /// Frequency ratios from root.
    pub fn ratios(&self) -> Vec<f32> {
        match self {
            Self::Major => vec![1.0, 1.2599, 1.4983], // 1, M3, P5
            Self::Minor => vec![1.0, 1.1892, 1.4983], // 1, m3, P5
            Self::Major7 => vec![1.0, 1.2599, 1.4983, 1.8877], // 1, M3, P5, M7
            Self::Minor7 => vec![1.0, 1.1892, 1.4983, 1.7818], // 1, m3, P5, m7
            Self::Dominant7 => vec![1.0, 1.2599, 1.4983, 1.7818], // 1, M3, P5, m7
            Self::Minor9 => vec![1.0, 1.1892, 1.4983, 1.7818, 2.2449], // 1, m3, P5, m7, M9
            Self::Major9 => vec![1.0, 1.2599, 1.4983, 1.8877, 2.2449], // 1, M3, P5, M7, M9
            Self::Sus2 => vec![1.0, 1.1225, 1.4983],  // 1, M2, P5
            Self::Sus4 => vec![1.0, 1.3348, 1.4983],  // 1, P4, P5
            Self::Diminished7 => vec![1.0, 1.1892, std::f32::consts::SQRT_2, 1.6818], // 1, m3, dim5 (2^(6/12) = √2), dim7
            Self::Augmented => vec![1.0, 1.2599, 1.5874],                             // 1, M3, aug5
            Self::Add9 => vec![1.0, 1.2599, 1.4983, 2.2449], // 1, M3, P5, M9
            Self::Minor11 => vec![1.0, 1.1892, 1.4983, 1.7818, 2.2449, 2.6697], // + P11
            Self::Dominant9 => vec![1.0, 1.2599, 1.4983, 1.7818, 2.2449], // 1, M3, P5, m7, M9
        }
    }

    /// Select chord type based on consciousness state.
    pub fn from_state(state: &MusicalState, position_in_bar: f32) -> Self {
        let psi = state.consciousness_level;
        let dopamine = state.dopamine;
        let serotonin = state.serotonin;
        let pe = state.prediction_error;
        let valence = state.valence;

        // High consciousness + jazz influence = extended chords
        if psi > 0.7 && dopamine > 0.5 {
            if valence > 0.3 {
                Self::Major9
            }
            // bright, dreamy
            else {
                Self::Minor9
            } // deep, Spalding
        } else if psi > 0.6 && serotonin > 0.5 {
            if valence > 0.0 {
                Self::Major7
            }
            // Steven Wilson
            else {
                Self::Minor7
            } // jazz standard
        } else if pe > 0.6 {
            // High tension = dissonant chords
            if pe > 0.8 {
                Self::Diminished7
            }
            // Dream Theater tension
            else {
                Self::Augmented
            } // mysterious
        } else if dopamine > 0.6 {
            Self::Dominant9 // funk, Vulfpeck
        } else if serotonin > 0.6 && valence > 0.0 {
            Self::Sus2 // atmospheric, open
        } else if position_in_bar > 3.0 {
            Self::Sus4 // tension before resolution on beat 1
        } else if valence > 0.2 {
            Self::Major
        } else {
            Self::Minor
        }
    }
}

/// Wide tempo range (60-180 BPM, not stuck at 120).
pub fn select_tempo(state: &MusicalState) -> f32 {
    let arousal = state.arousal;
    let psi = state.consciousness_level;
    let stillness = state.harmony_activations[7];

    // Base: arousal maps to tempo
    let base = 60.0 + arousal * 120.0; // 60-180 BPM

    // Stillness pulls tempo down
    let stillness_pull = stillness * 30.0;

    // High consciousness can handle fast tempo
    let psi_boost = if psi > 0.7 { (psi - 0.7) * 40.0 } else { 0.0 };

    (base - stillness_pull + psi_boost).clamp(55.0, 185.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seven_eight_groupings() {
        let ts = TimeSignature::SEVEN_EIGHT;
        assert_eq!(ts.groupings(), vec![2, 2, 3]);
        assert!((ts.beats_per_bar() - 3.5).abs() < 0.01);
    }

    #[test]
    fn five_four_groupings() {
        let ts = TimeSignature::FIVE_FOUR;
        assert_eq!(ts.groupings(), vec![3, 2]);
        assert_eq!(ts.beats_per_bar(), 5.0);
    }

    #[test]
    fn strong_beats_seven_eight() {
        let ts = TimeSignature::SEVEN_EIGHT;
        let strong = ts.strong_beats();
        assert_eq!(strong.len(), 3); // beat 1, beat 3 (2+2), beat 5 (2+2+3)
        assert!((strong[0] - 0.0).abs() < 0.01);
        assert!((strong[1] - 1.0).abs() < 0.01); // after group of 2 eighth notes
    }

    #[test]
    fn high_consciousness_selects_odd_meter() {
        let state = MusicalState {
            consciousness_level: 0.8,
            prediction_error: 0.6,
            harmony_activations: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.2],
            ..Default::default()
        };
        let ts = select_time_signature(&state);
        assert!(
            ts.numerator != 4 || ts.denominator != 4,
            "high complexity should give odd meter"
        );
    }

    #[test]
    fn low_consciousness_stays_four_four() {
        let state = MusicalState {
            consciousness_level: 0.2,
            ..Default::default()
        };
        let ts = select_time_signature(&state);
        assert_eq!(ts, TimeSignature::FOUR_FOUR);
    }

    #[test]
    fn groove_pocket_laid_back_with_serotonin() {
        let state = MusicalState {
            serotonin: 0.8,
            arousal: 0.3,
            ..Default::default()
        };
        let groove = GroovePocket::from_state(&state);
        assert!(
            groove.laid_back > 0.05,
            "high serotonin should be laid-back: {}",
            groove.laid_back
        );
    }

    #[test]
    fn groove_tighter_at_high_arousal() {
        let relaxed = MusicalState {
            serotonin: 0.8,
            arousal: 0.2,
            ..Default::default()
        };
        let intense = MusicalState {
            serotonin: 0.8,
            arousal: 0.9,
            ..Default::default()
        };
        let g_relaxed = GroovePocket::from_state(&relaxed);
        let g_intense = GroovePocket::from_state(&intense);
        assert!(
            g_intense.laid_back < g_relaxed.laid_back,
            "high arousal should be tighter: intense={} relaxed={}",
            g_intense.laid_back,
            g_relaxed.laid_back
        );
    }

    #[test]
    fn syncopation_generates_onsets() {
        let ts = TimeSignature::SEVEN_EIGHT;
        let pattern = SyncopationPattern::generate(&ts, 0.5, 42);
        assert!(
            !pattern.onset_positions.is_empty(),
            "should generate onsets"
        );
        assert_eq!(pattern.onset_positions.len(), pattern.accents.len());
    }

    #[test]
    fn extended_chords_have_correct_notes() {
        let ratios = ExtendedChordType::Minor9.ratios();
        assert_eq!(ratios.len(), 5); // root, m3, P5, m7, M9
        assert!((ratios[0] - 1.0).abs() < 0.001); // root
    }

    #[test]
    fn tempo_range_wide() {
        let slow = MusicalState {
            arousal: 0.0,
            harmony_activations: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
            ..Default::default()
        };
        let fast = MusicalState {
            arousal: 0.9,
            consciousness_level: 0.9,
            ..Default::default()
        };
        let t_slow = select_tempo(&slow);
        let t_fast = select_tempo(&fast);
        assert!(t_slow < 70.0, "should be slow: {t_slow}");
        assert!(t_fast > 140.0, "should be fast: {t_fast}");
        assert!(
            t_fast - t_slow > 80.0,
            "range should be wide: {t_slow}-{t_fast}"
        );
    }

    #[test]
    fn high_dopamine_gives_funk_chord() {
        let state = MusicalState {
            dopamine: 0.8,
            consciousness_level: 0.5,
            ..Default::default()
        };
        let chord = ExtendedChordType::from_state(&state, 1.0);
        assert_eq!(chord, ExtendedChordType::Dominant9); // funk!
    }
}
