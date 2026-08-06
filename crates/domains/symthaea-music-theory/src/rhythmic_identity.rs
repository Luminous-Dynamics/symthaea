// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A minimal, post-composition rhythmic-identity probe.
//!
//! Built to investigate a real finding from a 2026-07-24 blind listening
//! test: two March clips were misheard as Tango. Both styles' motif banks
//! use the identical duration ratio `(3/2, 1/2, 1)` as a signature cell —
//! but a duration RATIO alone can't say whether that cell reinforces the
//! meter (a march's dotted step) or displaces it (a tango's habanera
//! anticipation). That distinction only exists in the REALIZED timeline —
//! onset positions against bar boundaries — so this measures the
//! [`Score`] after motif expansion, phrase placement, and meter
//! assignment, not the motif template.
//!
//! Deliberately narrow: five concepts, no classifier, no learned model.
//! This is meant to test one hypothesis (are the two misheard March clips
//! measurably more Tango-like than clean March controls?), not to become
//! the crate's general style classifier.

use crate::score::{Emphasis, Score, VoiceRole};

/// A graded metrical weight for a beat-in-bar position: 3.0 = downbeat,
/// 2.0 = the mid-measure secondary accent (beat 3 in 4/4), 1.0 = any other
/// on-the-beat-grid position (e.g. beats 2/4 in 4/4), 0.0 = off the beat
/// grid entirely (syncopated placement). Same accent theory
/// [`crate::phrase::is_strong_beat`] already uses (downbeat / mid-measure
/// accent), just graded instead of boolean, and extended to flag
/// off-grid onsets rather than only classing them as "not strong."
fn metrical_weight(onset_in_bar: f64, meter_beats: f64) -> f64 {
    let frac = onset_in_bar.rem_euclid(1.0);
    if frac > 1e-6 && frac < 1.0 - 1e-6 {
        return 0.0; // off the beat grid: a genuinely syncopated attack
    }
    let beat = onset_in_bar.rem_euclid(meter_beats).round();
    if beat.abs() < 1e-6 {
        3.0 // downbeat
    } else if (beat - (meter_beats / 2.0).floor()).abs() < 1e-6 && meter_beats >= 4.0 {
        2.0 // mid-measure secondary accent (beat 3 in 4/4)
    } else {
        1.0 // on-grid but metrically weak (e.g. beats 2/4 in 4/4)
    }
}

/// Rhythmic-placement measurements of a realized [`Score`]'s melody voice,
/// relative to its own bar grid. See the module doc for why this measures
/// the timeline, not the motif template.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RhythmicIdentityReport {
    /// Fraction of melody notes landing on a strong beat (downbeat or the
    /// mid-measure accent) — reinforcing the bar hierarchy.
    pub strong_beat_onset_ratio: f32,
    /// Fraction of melody notes landing on a weak but still on-grid beat
    /// (e.g. beats 2/4 in 4/4).
    pub weak_beat_onset_ratio: f32,
    /// Fraction of melody notes that attack OFF the beat grid AND sustain
    /// across a subsequent strong-beat position without a new attack
    /// there — a genuine anticipation displacing that strong beat.
    pub anticipation_ratio: f32,
    /// Fraction of melody notes that attack off the beat grid and sustain
    /// across ANY subsequent beat boundary (strong or weak) — the general
    /// syncopation measure; `anticipation_ratio` is its strong-beat-only
    /// subset.
    pub syncopation_score: f32,
    /// Of the melody's cadential (phrase-final) notes, the fraction
    /// landing on a strong beat.
    pub phrase_final_downbeat_ratio: f32,
    /// Count of adjacent melody-note pairs with a ~3:1 duration ratio (the
    /// "long-short" dotted cell both March and Tango use) where the LONG
    /// note lands on a strong beat — reinforcing the grid.
    pub long_short_on_strong_beat: u32,
    /// The same 3:1 cell count, but where the long note attacks off the
    /// beat grid — displacing it, a tango-style anticipation.
    pub long_short_anticipations: u32,
}

/// Compute a [`RhythmicIdentityReport`] from `score`'s melody voice. Notes
/// are read in onset order; `score.meter` supplies the bar length.
pub fn rhythmic_identity_report(score: &Score) -> RhythmicIdentityReport {
    let meter_beats = score.meter as f64;
    let mut melody: Vec<&crate::score::ScoreNote> = score
        .notes
        .iter()
        .filter(|n| n.role == VoiceRole::Melody)
        .collect();
    melody.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));

    if melody.is_empty() {
        return RhythmicIdentityReport::default();
    }

    let mut strong = 0u32;
    let mut weak = 0u32;
    let mut anticipations = 0u32;
    let mut syncopations = 0u32;
    let mut long_short_strong = 0u32;
    let mut long_short_anti = 0u32;
    let mut cadential_total = 0u32;
    let mut cadential_strong = 0u32;

    for (i, n) in melody.iter().enumerate() {
        let onset = n.onset.beats();
        let onset_in_bar = onset.rem_euclid(meter_beats);
        let w = metrical_weight(onset_in_bar, meter_beats);
        if w >= 2.0 {
            strong += 1;
        } else if w >= 1.0 {
            weak += 1;
        }

        if w == 0.0 {
            // Off-grid attack: does its sustain cross the next beat
            // boundary without a fresh attack there?
            let end = (n.onset + n.duration).beats();
            let next_beat_in_bar = onset_in_bar.floor() + 1.0;
            let next_beat_abs = onset - onset_in_bar + next_beat_in_bar;
            if end > next_beat_abs + 1e-6 {
                syncopations += 1;
                let next_w = metrical_weight(next_beat_in_bar.rem_euclid(meter_beats), meter_beats);
                if next_w >= 2.0 {
                    anticipations += 1;
                }
            }
        }

        if n.emphasis == Emphasis::Cadential {
            cadential_total += 1;
            if w >= 2.0 {
                cadential_strong += 1;
            }
        }

        // The "long-short" 3:1 cell: this note and the next differ by a
        // duration ratio near 3.0. Classify by where the LONG note (the
        // earlier, longer one) attacks.
        if let Some(next) = melody.get(i + 1) {
            let d0 = n.duration.beats();
            let d1 = next.duration.beats();
            if d0 > 1e-6 && d1 > 1e-6 && (d0 / d1 - 3.0).abs() < 0.15 {
                if w >= 2.0 {
                    long_short_strong += 1;
                } else if w == 0.0 {
                    long_short_anti += 1;
                }
            }
        }
    }

    let total = melody.len() as f32;
    RhythmicIdentityReport {
        strong_beat_onset_ratio: strong as f32 / total,
        weak_beat_onset_ratio: weak as f32 / total,
        anticipation_ratio: anticipations as f32 / total,
        syncopation_score: syncopations as f32 / total,
        phrase_final_downbeat_ratio: if cadential_total > 0 {
            cadential_strong as f32 / cadential_total as f32
        } else {
            0.0
        },
        long_short_on_strong_beat: long_short_strong,
        long_short_anticipations: long_short_anti,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::pitch::{Pitch, PitchClass};
    use crate::rhythm::Duration;
    use crate::score::PartId;
    use crate::score::ScoreNote;

    fn note(onset_beats: f64, dur_beats: f64, emphasis: Emphasis) -> ScoreNote {
        ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: Pitch::new(PitchClass::C, 4),
            onset: Duration::new((onset_beats * 480.0).round() as i64, 480),
            duration: Duration::new((dur_beats * 480.0).round() as i64, 480),
            velocity: 0.5,
            role: VoiceRole::Melody,
            emphasis,
            section_intensity: 1.0,
        }
    }

    fn score_with(notes: Vec<ScoreNote>) -> Score {
        let mut s = Score::new(Key::major(PitchClass::C), 100.0, 4);
        for n in notes {
            s.push(n);
        }
        s
    }

    #[test]
    fn empty_score_reports_all_zero() {
        let s = Score::new(Key::major(PitchClass::C), 100.0, 4);
        assert_eq!(
            rhythmic_identity_report(&s),
            RhythmicIdentityReport::default()
        );
    }

    #[test]
    fn a_note_on_the_downbeat_counts_as_strong() {
        let s = score_with(vec![note(0.0, 1.0, Emphasis::Normal)]);
        let r = rhythmic_identity_report(&s);
        assert_eq!(r.strong_beat_onset_ratio, 1.0);
        assert_eq!(r.weak_beat_onset_ratio, 0.0);
    }

    #[test]
    fn a_note_on_beat_two_counts_as_weak_not_strong() {
        let s = score_with(vec![note(1.0, 1.0, Emphasis::Normal)]);
        let r = rhythmic_identity_report(&s);
        assert_eq!(r.weak_beat_onset_ratio, 1.0);
        assert_eq!(r.strong_beat_onset_ratio, 0.0);
    }

    #[test]
    fn a_march_style_long_short_reinforces_the_downbeat() {
        // Dotted-quarter (1.5 beats) starting ON beat 1, then a short
        // note, then landing squarely on beat 3 (the next strong beat) --
        // the long note attacks on-grid and STRONG.
        let s = score_with(vec![
            note(0.0, 1.5, Emphasis::Normal),
            note(1.5, 0.5, Emphasis::Normal),
            note(2.0, 1.0, Emphasis::Normal),
        ]);
        let r = rhythmic_identity_report(&s);
        assert_eq!(r.long_short_on_strong_beat, 1);
        assert_eq!(r.long_short_anticipations, 0);
        assert_eq!(r.anticipation_ratio, 0.0);
    }

    #[test]
    fn a_tango_style_long_short_anticipates_and_displaces_the_downbeat() {
        // The long note attacks a half-beat EARLY (off-grid) and sustains
        // across the next bar's downbeat without a fresh attack there --
        // a genuine anticipation.
        let s = score_with(vec![
            note(3.5, 1.5, Emphasis::Normal), // off-grid, crosses bar 2's downbeat (4.0)
            note(5.0, 0.5, Emphasis::Normal),
        ]);
        let r = rhythmic_identity_report(&s);
        assert_eq!(r.long_short_on_strong_beat, 0);
        assert_eq!(r.long_short_anticipations, 1);
        assert!(r.anticipation_ratio > 0.0);
        assert!(r.syncopation_score > 0.0);
    }

    #[test]
    fn an_off_grid_note_that_resolves_before_the_next_beat_is_not_syncopation() {
        // Off-grid attack, but it ends BEFORE the next beat boundary --
        // a passing ornament, not a displacement of the beat.
        let s = score_with(vec![note(0.25, 0.5, Emphasis::Normal)]);
        let r = rhythmic_identity_report(&s);
        assert_eq!(r.syncopation_score, 0.0);
        assert_eq!(r.anticipation_ratio, 0.0);
    }

    #[test]
    fn phrase_final_downbeat_ratio_only_considers_cadential_notes() {
        let s = score_with(vec![
            note(0.0, 1.0, Emphasis::Normal),
            note(1.0, 1.0, Emphasis::Cadential), // weak beat -- NOT a downbeat arrival
            note(4.0, 1.0, Emphasis::Cadential), // downbeat -- a real arrival
        ]);
        let r = rhythmic_identity_report(&s);
        assert_eq!(r.phrase_final_downbeat_ratio, 0.5);
    }
}
