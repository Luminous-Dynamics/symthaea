// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A minimal, post-composition melodic-contour probe.
//!
//! Follow-up to [`crate::rhythmic_identity`]: that module's beat-placement
//! hypothesis for a real March/Tango listening-test confusion was
//! FALSIFIED (the two misheard March clips were, if anything, more
//! rhythmically march-like than the correctly-identified ones). The
//! listener's own stated reason for both misses was "angular leaps" —
//! this measures interval size, direction, and leap-recovery behavior
//! instead, on the theory that angularity, not beat placement, may be
//! the actual confusion mechanism.
//!
//! Mode/valence/arousal are deliberately NOT fields on
//! [`MelodicContourIdentityReport`] — they're metadata a caller prints
//! alongside the report (see `examples/march_tango_contour_probe.rs`),
//! not inputs to the contour calculation itself, so a mode effect and a
//! contour effect can't be conflated inside one number.
//!
//! Deliberately narrow, same discipline as `rhythmic_identity`: the
//! "most diagnostic subset" of concepts, not the full melodic-analysis
//! vocabulary a general style classifier would eventually want.

use crate::pitch::Pitch;
use crate::score::{Score, VoiceRole};

/// A leap of this many semitones or more is "large" (at least a perfect
/// fourth) — the threshold the user's own probe request specified.
const LARGE_LEAP_SEMITONES: i32 = 5;
/// A leap of this many semitones or more is an octave leap.
const OCTAVE_SEMITONES: i32 = 12;
/// An interval this small or smaller counts as "stepwise" for
/// leap-recovery detection (whole step or less).
const STEP_SEMITONES: i32 = 2;

/// Interval-contour measurements of a realized [`Score`]'s melody voice.
/// Every field is computed from consecutive-note SEMITONE INTERVALS or
/// registral span, so the whole report is transposition-invariant by
/// construction (shifting every pitch by a constant changes no
/// difference and no span), and untouched by tempo/velocity/rendering
/// (it never reads them).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MelodicContourIdentityReport {
    pub note_count: usize,
    /// Mean absolute interval between consecutive melody notes, in
    /// semitones.
    pub mean_abs_interval_semitones: f32,
    /// Fraction of consecutive-note intervals that are large leaps
    /// (>= a perfect fourth).
    pub large_leap_ratio: f32,
    /// Count of consecutive-note intervals that are an octave or larger.
    pub octave_leap_count: u32,
    /// Fraction of interior notes where the melody changes direction
    /// (an ascending interval followed by a descending one, or vice
    /// versa) relative to the total number of direction-defined interval
    /// pairs.
    pub direction_change_ratio: f32,
    /// Of all large leaps, the fraction immediately followed by ANOTHER
    /// large leap in the opposite direction (an abrupt reversal, not a
    /// resolution).
    pub leap_reversal_ratio: f32,
    /// Of all large leaps, the fraction immediately followed by stepwise
    /// motion (<= a whole step) back toward the leap's origin — the
    /// classical "leap then step back" principle.
    pub leap_recovery_ratio: f32,
    /// Fraction of consecutive-note intervals that are exactly a minor
    /// second (1 semitone) — the chromatic-step/tension proxy.
    pub minor_second_ratio: f32,
    /// The full pitch range the melody covers, in semitones (highest
    /// minus lowest note).
    pub registral_span_semitones: u16,
}

/// Both a piece's OPENING (the first `opening_bars` bars — what a
/// listener actually judged in a stripped-melody test) and its full
/// realized melody, since a short signature hook can dominate perception
/// while disappearing into whole-piece averages.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SectionedContourReport {
    pub opening: MelodicContourIdentityReport,
    pub full_piece: MelodicContourIdentityReport,
}

/// Compute a [`SectionedContourReport`] from `score`. `opening_bars` bars
/// (at `score.meter` beats/bar) define the opening window; pass e.g. `8`
/// to match the "first eight bars" the listening test's judgment window
/// roughly covers.
pub fn melodic_contour_report(score: &Score, opening_bars: u32) -> SectionedContourReport {
    let mut melody: Vec<(f64, Pitch)> = score
        .notes
        .iter()
        .filter(|n| n.role == VoiceRole::Melody)
        .map(|n| (n.onset.beats(), n.pitch))
        .collect();
    melody.sort_by(|a, b| a.0.total_cmp(&b.0));

    let cutoff = score.meter as f64 * opening_bars as f64;
    let opening_pitches: Vec<Pitch> = melody
        .iter()
        .filter(|(onset, _)| *onset < cutoff)
        .map(|(_, p)| *p)
        .collect();
    let full_pitches: Vec<Pitch> = melody.iter().map(|(_, p)| *p).collect();

    SectionedContourReport {
        opening: report_from_pitches(&opening_pitches),
        full_piece: report_from_pitches(&full_pitches),
    }
}

fn report_from_pitches(pitches: &[Pitch]) -> MelodicContourIdentityReport {
    if pitches.len() < 2 {
        return MelodicContourIdentityReport {
            note_count: pitches.len(),
            ..Default::default()
        };
    }
    let midis: Vec<i32> = pitches.iter().map(|p| p.midi() as i32).collect();
    let intervals: Vec<i32> = midis.windows(2).map(|w| w[1] - w[0]).collect();

    let mut abs_sum = 0i64;
    let mut large_leaps = 0u32;
    let mut octave_leaps = 0u32;
    let mut minor_seconds = 0u32;
    for &iv in &intervals {
        let abs_iv = iv.abs();
        abs_sum += abs_iv as i64;
        if abs_iv >= LARGE_LEAP_SEMITONES {
            large_leaps += 1;
        }
        if abs_iv >= OCTAVE_SEMITONES {
            octave_leaps += 1;
        }
        if abs_iv == 1 {
            minor_seconds += 1;
        }
    }

    let mut direction_changes = 0u32;
    let mut direction_pairs = 0u32;
    for w in intervals.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a == 0 || b == 0 {
            continue; // a repeated note has no direction to compare
        }
        direction_pairs += 1;
        if a.signum() != b.signum() {
            direction_changes += 1;
        }
    }

    let mut leap_total = 0u32;
    let mut reversals = 0u32;
    let mut recoveries = 0u32;
    for w in intervals.windows(2) {
        let (leap, next) = (w[0], w[1]);
        if leap.abs() < LARGE_LEAP_SEMITONES {
            continue;
        }
        leap_total += 1;
        let opposite_direction = next.signum() != 0 && next.signum() != leap.signum();
        if opposite_direction && next.abs() >= LARGE_LEAP_SEMITONES {
            reversals += 1;
        } else if opposite_direction && next.abs() <= STEP_SEMITONES {
            recoveries += 1;
        }
    }

    let min = *midis.iter().min().unwrap();
    let max = *midis.iter().max().unwrap();

    MelodicContourIdentityReport {
        note_count: pitches.len(),
        mean_abs_interval_semitones: abs_sum as f32 / intervals.len() as f32,
        large_leap_ratio: large_leaps as f32 / intervals.len() as f32,
        octave_leap_count: octave_leaps,
        direction_change_ratio: if direction_pairs > 0 {
            direction_changes as f32 / direction_pairs as f32
        } else {
            0.0
        },
        leap_reversal_ratio: if leap_total > 0 {
            reversals as f32 / leap_total as f32
        } else {
            0.0
        },
        leap_recovery_ratio: if leap_total > 0 {
            recoveries as f32 / leap_total as f32
        } else {
            0.0
        },
        minor_second_ratio: minor_seconds as f32 / intervals.len() as f32,
        registral_span_semitones: (max - min) as u16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::pitch::PitchClass;
    use crate::rhythm::Duration;
    use crate::score::PartId;
    use crate::score::{Emphasis, ScoreNote, VoiceRole};

    fn note_at(onset_beats: f64, pitch: Pitch) -> ScoreNote {
        ScoreNote {
            part: PartId::UNASSIGNED,
            pitch,
            onset: Duration::new((onset_beats * 480.0).round() as i64, 480),
            duration: Duration::new(480, 480),
            velocity: 0.5,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn score_from_midis(midis: &[u8]) -> Score {
        let mut s = Score::new(Key::major(PitchClass::C), 100.0, 4);
        for (i, &m) in midis.iter().enumerate() {
            s.push(note_at(i as f64, Pitch::from_midi(m)));
        }
        s
    }

    #[test]
    fn fewer_than_two_notes_reports_zeroed_defaults() {
        let s = score_from_midis(&[60]);
        let r = melodic_contour_report(&s, 8);
        assert_eq!(r.full_piece.note_count, 1);
        assert_eq!(r.full_piece.mean_abs_interval_semitones, 0.0);
    }

    #[test]
    fn a_stepwise_scale_run_has_a_small_mean_interval_and_no_large_leaps() {
        // C D E F G -- every interval 1 or 2 semitones.
        let s = score_from_midis(&[60, 62, 64, 65, 67]);
        let r = melodic_contour_report(&s, 8).full_piece;
        assert!(r.mean_abs_interval_semitones <= 2.0);
        assert_eq!(r.large_leap_ratio, 0.0);
        assert_eq!(r.octave_leap_count, 0);
    }

    #[test]
    fn an_octave_leap_is_counted_as_both_large_and_octave() {
        let s = score_from_midis(&[60, 72]); // C4 -> C5
        let r = melodic_contour_report(&s, 8).full_piece;
        assert_eq!(r.large_leap_ratio, 1.0);
        assert_eq!(r.octave_leap_count, 1);
    }

    #[test]
    fn a_leap_up_then_a_step_down_counts_as_recovery_not_reversal() {
        // C4 -> C5 (leap up 12) -> B4 (step down 1, opposite direction).
        let s = score_from_midis(&[60, 72, 71]);
        let r = melodic_contour_report(&s, 8).full_piece;
        assert_eq!(r.leap_recovery_ratio, 1.0);
        assert_eq!(r.leap_reversal_ratio, 0.0);
    }

    #[test]
    fn two_opposite_direction_leaps_in_a_row_count_as_reversal_not_recovery() {
        // C4 -> C5 (leap up 12) -> E3 (leap down 20, opposite direction).
        let s = score_from_midis(&[60, 72, 52]);
        let r = melodic_contour_report(&s, 8).full_piece;
        assert_eq!(r.leap_reversal_ratio, 1.0);
        assert_eq!(r.leap_recovery_ratio, 0.0);
    }

    #[test]
    fn a_zigzag_melody_has_a_high_direction_change_ratio() {
        let s = score_from_midis(&[60, 65, 60, 65, 60]); // up, down, up, down
        let r = melodic_contour_report(&s, 8).full_piece;
        assert_eq!(r.direction_change_ratio, 1.0);
    }

    #[test]
    fn a_monotonic_ascending_run_has_zero_direction_changes() {
        let s = score_from_midis(&[60, 62, 64, 67, 71]);
        let r = melodic_contour_report(&s, 8).full_piece;
        assert_eq!(r.direction_change_ratio, 0.0);
    }

    #[test]
    fn registral_span_is_the_full_pitch_range() {
        let s = score_from_midis(&[60, 72, 55]);
        let r = melodic_contour_report(&s, 8).full_piece;
        assert_eq!(r.registral_span_semitones, 72 - 55);
    }

    #[test]
    fn minor_second_ratio_counts_exact_semitone_steps() {
        let s = score_from_midis(&[60, 61, 65, 66]); // 1, 4, 1 semitones
        let r = melodic_contour_report(&s, 8).full_piece;
        assert!((r.minor_second_ratio - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn the_report_is_transposition_invariant() {
        let low = score_from_midis(&[60, 72, 71, 55, 67]);
        let high = score_from_midis(&[65, 77, 76, 60, 72]); // same intervals, +5
        assert_eq!(
            melodic_contour_report(&low, 8).full_piece,
            melodic_contour_report(&high, 8).full_piece
        );
    }

    #[test]
    fn the_report_is_unaffected_by_tempo_or_velocity() {
        let mut fast = score_from_midis(&[60, 72, 67, 60]);
        fast.tempo_bpm = 220.0;
        for n in fast.notes.iter_mut() {
            n.velocity = 0.9;
        }
        let slow = score_from_midis(&[60, 72, 67, 60]);
        assert_eq!(
            melodic_contour_report(&fast, 8).full_piece,
            melodic_contour_report(&slow, 8).full_piece
        );
    }

    #[test]
    fn the_opening_window_excludes_notes_past_its_bar_cutoff() {
        let mut s = Score::new(Key::major(PitchClass::C), 100.0, 4);
        s.push(note_at(0.0, Pitch::from_midi(60)));
        s.push(note_at(1.0, Pitch::from_midi(65))); // inside 2 bars = 8 beats
        s.push(note_at(20.0, Pitch::from_midi(80))); // well past 2 bars
        let r = melodic_contour_report(&s, 2);
        assert_eq!(r.opening.note_count, 2);
        assert_eq!(r.full_piece.note_count, 3);
    }
}
