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
use std::collections::BTreeMap;

/// Hard, task-specific requirements for one musical output family.
///
/// Thresholds are supplied by the caller instead of treating one musical style as
/// universal. For example, a melody brief can require pitch motion while a drone
/// brief can allow one dominant pitch. The window ratio measures onset coverage,
/// not sustained-note coverage.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
            violations,
        }
    }
}

/// Audit notes against an explicit intent contract.
///
/// Non-finite values, non-positive frequency/duration, negative onset, onset at or
/// beyond the declared duration, and velocity outside [0, 1] are rejected rather
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
            && note.velocity.is_finite()
            && (0.0..=1.0).contains(&note.velocity);
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
