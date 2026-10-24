// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned, intent-conditioned feasibility checks for generated musical candidates.
//!
//! A generic scalar cannot tell us whether a repeated-pitch drone is a failure:
//! that depends on the task. This module makes the brief's mechanical constraints
//! explicit and reports measured features separately. It does not claim to measure
//! beauty, originality, emotional impact, or human preference.
//!
//! This is an additive evaluation seam. It intentionally does not alter the legacy
//! CreativeQualityScore or feed hard-constraint results into the live reward loop.
//! Integrate it into candidate selection only after task contracts are calibrated.

use crate::Note;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const MAX_WINDOW_COUNT: usize = 4096;

/// Hard, task-specific requirements for one musical output family.
///
/// Thresholds are supplied by the caller instead of treating one musical style as
/// universal. For example, a melody brief can require pitch motion while a drone
/// brief can allow one dominant pitch. The window ratio measures onset coverage,
/// not sustained-note coverage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreativeIntentContractV1 {
    pub min_note_count: usize,
    pub min_distinct_pitches: usize,
    pub min_pitch_range_semitones: f32,
    pub min_active_window_ratio: f32,
    pub max_dominant_pitch_share: f32,
    pub window_count: usize,
}

impl CreativeIntentContractV1 {
    /// Whether the contract itself is well-formed.
    pub fn is_valid(&self) -> bool {
        self.min_note_count > 0
            && self.min_distinct_pitches > 0
            && self.window_count > 0
            && self.window_count <= MAX_WINDOW_COUNT
            && self.min_pitch_range_semitones.is_finite()
            && self.min_pitch_range_semitones >= 0.0
            && self.min_active_window_ratio.is_finite()
            && (0.0..=1.0).contains(&self.min_active_window_ratio)
            && self.max_dominant_pitch_share.is_finite()
            && (0.0..=1.0).contains(&self.max_dominant_pitch_share)
    }
}

/// Why a candidate was rejected by the intent contract.
///
/// These are mechanical failures, not artistic judgments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CreativeContractViolation {
    InvalidContract,
    InvalidDuration,
    InvalidNote { index: usize },
    TooFewNotes { required: usize, actual: usize },
    TooFewDistinctPitches { required: usize, actual: usize },
    InsufficientPitchRange { required: f32, actual: f32 },
    InsufficientTemporalCoverage { required: f32, actual: f32 },
    DominantPitchShareExceeded { maximum: f32, actual: f32 },
}

/// Objective observations and contract result for one candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreativeCandidateAuditV1 {
    /// True only when the contract and duration are valid, every note is valid,
    /// and all declared constraints pass.
    pub accepted: bool,
    /// Number of individually valid notes used to calculate the observations.
    pub valid_note_count: usize,
    /// Number of distinct rounded MIDI pitches.
    pub distinct_pitch_count: usize,
    /// Difference between highest and lowest MIDI pitch, in semitones.
    pub pitch_range_semitones: f32,
    /// Fraction of equal-duration time windows containing at least one note onset.
    pub active_window_ratio: f32,
    /// Largest share of valid notes belonging to any one rounded MIDI pitch.
    pub dominant_pitch_share: f32,
    /// Number of distinct onset events after coalescing starts within 1 ms.
    pub distinct_onset_count: usize,
    /// Coefficient of variation of consecutive distinct-onset intervals.
    /// Descriptive only: it is not a universal rhythm-quality or syncopation score.
    pub onset_interval_cv: f64,
    /// Complete set of contract and input violations found.
    pub violations: Vec<CreativeContractViolation>,
}

impl CreativeCandidateAuditV1 {
    fn rejected(violations: Vec<CreativeContractViolation>) -> Self {
        Self {
            accepted: false,
            valid_note_count: 0,
            distinct_pitch_count: 0,
            pitch_range_semitones: 0.0,
            active_window_ratio: 0.0,
            dominant_pitch_share: 0.0,
            distinct_onset_count: 0,
            onset_interval_cv: 0.0,
            violations,
        }
    }
}

/// Count distinct onset events and compute inter-onset-interval coefficient of variation.
///
/// Starts within 1 ms are treated as one event so chord tones do not masquerade as
/// separate rhythmic attacks. This is descriptive only; it does not know meter, beat
/// phase, instrument roles, or whether syncopation was intended.
fn onset_interval_stats(valid_notes: &[(Note, f32)]) -> (usize, f64) {
    let mut starts: Vec<f64> = valid_notes
        .iter()
        .map(|(note, _)| f64::from(note.start_time))
        .collect();
    starts.sort_by(f64::total_cmp);

    let mut distinct_starts: Vec<f64> = Vec::with_capacity(starts.len());
    for start in starts {
        let add_start = match distinct_starts.last() {
            Some(previous) => start - *previous > 0.001,
            None => true,
        };
        if add_start {
            distinct_starts.push(start);
        }
    }

    let intervals: Vec<f64> = distinct_starts
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    if intervals.is_empty() {
        return (distinct_starts.len(), 0.0);
    }

    let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
    if mean <= 0.0 || !mean.is_finite() {
        return (distinct_starts.len(), 0.0);
    }
    let variance = intervals
        .iter()
        .map(|interval| (interval - mean).powi(2))
        .sum::<f64>()
        / intervals.len() as f64;
    let cv = variance.sqrt() / mean;
    (distinct_starts.len(), if cv.is_finite() { cv } else { 0.0 })
}

/// Audit notes against an explicit intent contract.
///
/// Non-finite values, non-positive frequency/duration, negative onset, onset at or
/// beyond the declared duration, and velocity outside (0, 1] are rejected rather
/// than silently discarded. Measured observations are still returned for valid
/// notes, but any invalid note makes the candidate inadmissible.
///
/// An accepted candidate has met only the declared mechanical contract. It is not
/// thereby certified as beautiful, playable, culturally appropriate, or original.
pub fn audit_candidate(
    notes: &[Note],
    duration_secs: f32,
    contract: &CreativeIntentContractV1,
) -> CreativeCandidateAuditV1 {
    if !contract.is_valid() {
        return CreativeCandidateAuditV1::rejected(vec![
            CreativeContractViolation::InvalidContract,
        ]);
    }
    if !duration_secs.is_finite() || duration_secs <= 0.0 {
        return CreativeCandidateAuditV1::rejected(vec![
            CreativeContractViolation::InvalidDuration,
        ]);
    }

    let mut violations = Vec::new();
    let mut valid_notes: Vec<(Note, f32)> = Vec::with_capacity(notes.len());

    for (index, note) in notes.iter().copied().enumerate() {
        let fields_valid = note.frequency.is_finite()
            && note.frequency > 0.0
            && note.start_time.is_finite()
            && note.start_time >= 0.0
            && note.start_time < duration_secs
            && note.duration.is_finite()
            && note.duration > 0.0
            && (note.start_time + note.duration).is_finite()
            && note.start_time + note.duration > note.start_time
            && note.velocity.is_finite()
            && note.velocity > 0.0
            && note.velocity <= 1.0;
        let midi_pitch = if fields_valid {
            69.0 + 12.0 * (note.frequency / 440.0).log2()
        } else {
            f32::NAN
        };

        if !fields_valid || !midi_pitch.is_finite() {
            violations.push(CreativeContractViolation::InvalidNote { index });
            continue;
        }

        valid_notes.push((note, midi_pitch));
    }

    let valid_note_count = valid_notes.len();
    let (distinct_onset_count, onset_interval_cv) = onset_interval_stats(&valid_notes);
    let mut pitch_counts: BTreeMap<i32, usize> = BTreeMap::new();
    let mut active_windows = vec![false; contract.window_count];
    let mut lowest_pitch = f32::INFINITY;
    let mut highest_pitch = f32::NEG_INFINITY;

    for (note, midi_pitch) in &valid_notes {
        *pitch_counts.entry(midi_pitch.round() as i32).or_insert(0) += 1;
        lowest_pitch = lowest_pitch.min(*midi_pitch);
        highest_pitch = highest_pitch.max(*midi_pitch);

        let window = ((note.start_time / duration_secs) * contract.window_count as f32)
            .floor() as usize;
        active_windows[window.min(contract.window_count - 1)] = true;
    }

    let distinct_pitch_count = pitch_counts.len();
    let pitch_range_semitones = if valid_notes.is_empty() {
        0.0
    } else {
        (highest_pitch - lowest_pitch).max(0.0)
    };
    let active_window_ratio = active_windows.iter().filter(|active| **active).count() as f32
        / contract.window_count as f32;
    let dominant_pitch_share = pitch_counts
        .values()
        .copied()
        .max()
        .map(|count| count as f32 / valid_note_count as f32)
        .unwrap_or(0.0);

    if valid_note_count < contract.min_note_count {
        violations.push(CreativeContractViolation::TooFewNotes {
            required: contract.min_note_count,
            actual: valid_note_count,
        });
    }
    if distinct_pitch_count < contract.min_distinct_pitches {
        violations.push(CreativeContractViolation::TooFewDistinctPitches {
            required: contract.min_distinct_pitches,
            actual: distinct_pitch_count,
        });
    }
    if pitch_range_semitones < contract.min_pitch_range_semitones {
        violations.push(CreativeContractViolation::InsufficientPitchRange {
            required: contract.min_pitch_range_semitones,
            actual: pitch_range_semitones,
        });
    }
    if active_window_ratio < contract.min_active_window_ratio {
        violations.push(CreativeContractViolation::InsufficientTemporalCoverage {
            required: contract.min_active_window_ratio,
            actual: active_window_ratio,
        });
    }
    if dominant_pitch_share > contract.max_dominant_pitch_share {
        violations.push(CreativeContractViolation::DominantPitchShareExceeded {
            maximum: contract.max_dominant_pitch_share,
            actual: dominant_pitch_share,
        });
    }

    CreativeCandidateAuditV1 {
        accepted: violations.is_empty(),
        valid_note_count,
        distinct_pitch_count,
        pitch_range_semitones,
        active_window_ratio,
        dominant_pitch_share,
        distinct_onset_count,
        onset_interval_cv,
        violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(frequency: f32, start_time: f32) -> Note {
        Note {
            frequency,
            start_time,
            duration: 0.2,
            velocity: 0.8,
        }
    }

    fn melodic_contract() -> CreativeIntentContractV1 {
        CreativeIntentContractV1 {
            min_note_count: 8,
            min_distinct_pitches: 4,
            min_pitch_range_semitones: 5.0,
            min_active_window_ratio: 0.5,
            max_dominant_pitch_share: 0.5,
            window_count: 8,
        }
    }

    #[test]
    fn varied_melody_satisfies_its_declared_contract() {
        let frequencies = [
            261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25,
        ];
        let notes: Vec<Note> = frequencies
            .iter()
            .enumerate()
            .map(|(i, &frequency)| note(frequency, i as f32 * 0.25))
            .collect();

        let audit = audit_candidate(&notes, 2.0, &melodic_contract());
        assert!(audit.accepted, "violations: {:?}", audit.violations);
        assert_eq!(audit.valid_note_count, 8);
        assert_eq!(audit.distinct_pitch_count, 8);
        assert!(audit.pitch_range_semitones > 11.9);
        assert_eq!(audit.active_window_ratio, 1.0);
        assert_eq!(audit.dominant_pitch_share, 0.125);
    }

    #[test]
    fn repeated_pitch_fails_a_melodic_contract() {
        let notes: Vec<Note> = (0..16)
            .map(|i| note(440.0, i as f32 * 0.25))
            .collect();

        let audit = audit_candidate(&notes, 4.0, &melodic_contract());
        assert!(!audit.accepted);
        assert_eq!(audit.distinct_pitch_count, 1);
        assert_eq!(audit.pitch_range_semitones, 0.0);
        assert_eq!(audit.dominant_pitch_share, 1.0);
        assert!(audit.violations.iter().any(|v| matches!(
            v,
            CreativeContractViolation::TooFewDistinctPitches { .. }
        )));
        assert!(audit.violations.iter().any(|v| matches!(
            v,
            CreativeContractViolation::InsufficientPitchRange { .. }
        )));
        assert!(audit.violations.iter().any(|v| matches!(
            v,
            CreativeContractViolation::DominantPitchShareExceeded { .. }
        )));
    }

    #[test]
    fn same_repeated_pitch_is_allowed_when_the_contract_allows_it() {
        // A repeated-pitch texture is not globally "bad"; admissibility depends
        // on the declared creative intention, not on one universal taste score.
        let notes: Vec<Note> = (0..16)
            .map(|i| note(440.0, i as f32 * 0.25))
            .collect();
        let drone_contract = CreativeIntentContractV1 {
            min_note_count: 1,
            min_distinct_pitches: 1,
            min_pitch_range_semitones: 0.0,
            min_active_window_ratio: 0.0,
            max_dominant_pitch_share: 1.0,
            window_count: 8,
        };

        let audit = audit_candidate(&notes, 4.0, &drone_contract);
        assert!(audit.accepted, "violations: {:?}", audit.violations);
    }

    #[test]
    fn malformed_note_fails_closed_and_reports_its_index() {
        let notes = [note(f32::NAN, 0.0), note(440.0, 0.5)];
        let audit = audit_candidate(&notes, 1.0, &melodic_contract());

        assert!(!audit.accepted);
        assert!(audit.violations.contains(&CreativeContractViolation::InvalidNote {
            index: 0
        }));
        assert_eq!(audit.valid_note_count, 1);
    }

    #[test]
    fn onset_at_duration_boundary_is_invalid() {
        let audit = audit_candidate(
            &[note(440.0, 1.0)],
            1.0,
            &CreativeIntentContractV1 {
                min_note_count: 1,
                min_distinct_pitches: 1,
                min_pitch_range_semitones: 0.0,
                min_active_window_ratio: 0.0,
                max_dominant_pitch_share: 1.0,
                window_count: 4,
            },
        );
        assert!(!audit.accepted);
        assert!(audit.violations.contains(&CreativeContractViolation::InvalidNote {
            index: 0
        }));
    }

    #[test]
    fn zero_velocity_silent_events_cannot_satisfy_a_creative_contract() {
        let mut silent = note(440.0, 0.0);
        silent.velocity = 0.0;
        let contract = CreativeIntentContractV1 {
            min_note_count: 1,
            min_distinct_pitches: 1,
            min_pitch_range_semitones: 0.0,
            min_active_window_ratio: 0.0,
            max_dominant_pitch_share: 1.0,
            window_count: 4,
        };

        let audit = audit_candidate(&[silent], 1.0, &contract);
        assert!(!audit.accepted);
        assert_eq!(audit.valid_note_count, 0);
        assert!(audit.violations.contains(&CreativeContractViolation::InvalidNote { index: 0 }));
    }

    #[test]
    fn excessive_window_count_is_rejected_before_allocation() {
        let contract = CreativeIntentContractV1 {
            min_note_count: 1,
            min_distinct_pitches: 1,
            min_pitch_range_semitones: 0.0,
            min_active_window_ratio: 0.0,
            max_dominant_pitch_share: 1.0,
            window_count: usize::MAX,
        };

        let audit = audit_candidate(&[note(440.0, 0.0)], 1.0, &contract);
        assert_eq!(
            audit.violations,
            vec![CreativeContractViolation::InvalidContract]
        );
    }

    #[test]
    fn duration_overflow_is_rejected_as_invalid_note() {
        let notes = [Note {
            frequency: 440.0,
            start_time: f32::MAX / 2.0,
            duration: f32::MAX,
            velocity: 0.8,
        }];

        let audit = audit_candidate(&notes, f32::MAX, &CreativeIntentContractV1 {
            min_note_count: 1,
            min_distinct_pitches: 1,
            min_pitch_range_semitones: 0.0,
            min_active_window_ratio: 0.0,
            max_dominant_pitch_share: 1.0,
            window_count: 4,
        });
        assert!(!audit.accepted);
        assert!(audit.violations.contains(&CreativeContractViolation::InvalidNote { index: 0 }));
    }


    #[test]
    fn chord_tones_within_one_millisecond_count_as_one_onset() {
        let notes = vec![
            note(440.0, 0.0),
            note(523.25, 0.0005),
            note(392.0, 0.5),
        ];
        let contract = CreativeIntentContractV1 {
            min_note_count: 1,
            min_distinct_pitches: 1,
            min_pitch_range_semitones: 0.0,
            min_active_window_ratio: 0.0,
            max_dominant_pitch_share: 1.0,
            window_count: 4,
        };
        let audit = audit_candidate(&notes, 1.0, &contract);
        assert_eq!(audit.distinct_onset_count, 2);
        assert_eq!(audit.onset_interval_cv, 0.0);
    }

    #[test]
    fn interval_variability_is_descriptive_and_deterministic() {
        let contract = CreativeIntentContractV1 {
            min_note_count: 1,
            min_distinct_pitches: 1,
            min_pitch_range_semitones: 0.0,
            min_active_window_ratio: 0.0,
            max_dominant_pitch_share: 1.0,
            window_count: 4,
        };
        let regular: Vec<Note> = [0.0, 0.25, 0.5, 0.75]
            .iter()
            .map(|&start| note(440.0, start))
            .collect();
        let irregular: Vec<Note> = [0.0, 0.1, 0.4, 1.0]
            .iter()
            .map(|&start| note(440.0, start))
            .collect();
        let regular_audit = audit_candidate(&regular, 1.0, &contract);
        let irregular_audit = audit_candidate(&irregular, 1.25, &contract);
        assert_eq!(regular_audit.onset_interval_cv, 0.0);
        assert!(irregular_audit.onset_interval_cv > regular_audit.onset_interval_cv);
    }

    #[test]
    fn invalid_contract_is_rejected_before_scoring() {
        let bad_contract = CreativeIntentContractV1 {
            min_note_count: 1,
            min_distinct_pitches: 1,
            min_pitch_range_semitones: 0.0,
            min_active_window_ratio: 0.0,
            max_dominant_pitch_share: f32::NAN,
            window_count: 4,
        };
        let audit = audit_candidate(&[note(440.0, 0.0)], 1.0, &bad_contract);
        assert_eq!(
            audit.violations,
            vec![CreativeContractViolation::InvalidContract]
        );
    }
}
