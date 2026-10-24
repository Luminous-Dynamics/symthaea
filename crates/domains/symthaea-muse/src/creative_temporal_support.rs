// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Temporal support measurements for intent-conditioned musical candidates.
//!
//! Attack coverage, sustained support, occupied time, and silence are different
//! properties. This module reports them separately; it does not rank aesthetics
//! and does not alter the V1 creative-intent acceptance contract.

use crate::Note;
use serde::{Deserialize, Serialize};

const MAX_WINDOW_COUNT: usize = 4096;

/// Why temporal support could not be measured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CreativeTemporalSupportError {
    InvalidDuration,
    InvalidWindowCount,
    InvalidNote { index: usize },
}

/// Descriptive temporal measurements, not an artistic-quality score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreativeTemporalSupportV1 {
    /// Fraction of equal-duration windows containing at least one note onset.
    pub onset_window_coverage_ratio: f32,
    /// Fraction of windows overlapped by at least one non-zero-velocity note interval.
    pub sustain_window_coverage_ratio: f32,
    /// Fraction of clip duration covered by the union of non-zero-velocity note intervals.
    pub occupied_time_ratio: f32,
    /// Longest silent interval, including silence at the beginning or end.
    pub maximum_silent_gap_secs: f32,
    /// Number of windows used by both coverage measurements.
    pub window_count: usize,
    /// Number of notes measured.
    pub note_count: usize,
}

/// Measure temporal support while keeping event frequency separate from duration.
///
/// Notes that start within the clip but end beyond it are clipped to the declared
/// duration for these measurements. Any malformed note rejects the measurement
/// rather than being silently discarded. An empty note set is valid descriptive
/// input and reports zero coverage with the full clip as its maximum silent gap.
pub fn measure_temporal_support(
    notes: &[Note],
    duration_secs: f32,
    window_count: usize,
) -> Result<CreativeTemporalSupportV1, CreativeTemporalSupportError> {
    if !duration_secs.is_finite() || duration_secs <= 0.0 {
        return Err(CreativeTemporalSupportError::InvalidDuration);
    }
    if window_count == 0 || window_count > MAX_WINDOW_COUNT {
        return Err(CreativeTemporalSupportError::InvalidWindowCount);
    }

    for (index, note) in notes.iter().enumerate() {
        let end = note.start_time + note.duration;
        let valid = note.frequency.is_finite()
            && note.frequency > 0.0
            && note.start_time.is_finite()
            && note.start_time >= 0.0
            && note.start_time < duration_secs
            && note.duration.is_finite()
            && note.duration > 0.0
            && end.is_finite()
            && end > note.start_time
            && note.velocity.is_finite()
            && note.velocity > 0.0
            && note.velocity <= 1.0;
        if !valid {
            return Err(CreativeTemporalSupportError::InvalidNote { index });
        }
    }

    let total = f64::from(duration_secs);
    let window_count_f64 = window_count as f64;
    let mut onset_windows = vec![false; window_count];
    let mut sustain_windows = vec![false; window_count];
    let mut intervals = Vec::with_capacity(notes.len());

    for note in notes {
        let start = f64::from(note.start_time);
        let end = (start + f64::from(note.duration)).min(total);
        intervals.push((start, end));

        let onset_window = ((start / total) * window_count_f64).floor() as usize;
        onset_windows[onset_window.min(window_count - 1)] = true;

        let first_window = ((start / total) * window_count_f64).floor() as usize;
        let first_window = first_window.min(window_count - 1);
        let end_scaled = (end / total) * window_count_f64;
        let end_exclusive = if end >= total {
            window_count
        } else {
            (end_scaled.ceil() as usize).clamp(first_window + 1, window_count)
        };
        for index in first_window..end_exclusive {
            let window_start = total * index as f64 / window_count_f64;
            let window_end = total * (index + 1) as f64 / window_count_f64;
            if start < window_end && end > window_start {
                sustain_windows[index] = true;
            }
        }
    }

    let onset_window_coverage_ratio =
        onset_windows.iter().filter(|active| **active).count() as f64 / window_count_f64;
    let sustain_window_coverage_ratio =
        sustain_windows.iter().filter(|active| **active).count() as f64 / window_count_f64;

    intervals.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(intervals.len());
    for (start, end) in intervals {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }

    let mut occupied_secs = 0.0_f64;
    let mut maximum_silent_gap_secs = 0.0_f64;
    let mut cursor = 0.0_f64;
    for (start, end) in &merged {
        maximum_silent_gap_secs = maximum_silent_gap_secs.max((start - cursor).max(0.0));
        occupied_secs += (end - start).max(0.0);
        cursor = cursor.max(*end);
    }
    maximum_silent_gap_secs = maximum_silent_gap_secs.max((total - cursor).max(0.0));

    Ok(CreativeTemporalSupportV1 {
        onset_window_coverage_ratio: onset_window_coverage_ratio as f32,
        sustain_window_coverage_ratio: sustain_window_coverage_ratio as f32,
        occupied_time_ratio: (occupied_secs / total).clamp(0.0, 1.0) as f32,
        maximum_silent_gap_secs: maximum_silent_gap_secs as f32,
        window_count,
        note_count: notes.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(start_time: f32, duration: f32) -> Note {
        Note {
            frequency: 440.0,
            start_time,
            duration,
            velocity: 0.8,
        }
    }

    #[test]
    fn sustained_drone_is_not_mistaken_for_low_temporal_support() {
        let result = measure_temporal_support(&[note(0.0, 4.0)], 4.0, 8).unwrap();

        assert_eq!(result.onset_window_coverage_ratio, 0.125);
        assert_eq!(result.sustain_window_coverage_ratio, 1.0);
        assert_eq!(result.occupied_time_ratio, 1.0);
        assert_eq!(result.maximum_silent_gap_secs, 0.0);
    }

    #[test]
    fn staccato_onsets_can_cover_every_window_without_occupying_most_time() {
        let notes: Vec<Note> = (0..8).map(|index| note(index as f32 * 0.5, 0.1)).collect();
        let result = measure_temporal_support(&notes, 4.0, 8).unwrap();

        assert_eq!(result.onset_window_coverage_ratio, 1.0);
        assert_eq!(result.sustain_window_coverage_ratio, 1.0);
        assert!((result.occupied_time_ratio - 0.2).abs() < 1.0e-6);
        assert!((result.maximum_silent_gap_secs - 0.4).abs() < 1.0e-6);
    }

    #[test]
    fn overlapping_notes_count_union_coverage_only_once() {
        let result =
            measure_temporal_support(&[note(0.0, 0.75), note(0.25, 0.75)], 1.0, 4).unwrap();

        assert_eq!(result.occupied_time_ratio, 1.0);
        assert_eq!(result.maximum_silent_gap_secs, 0.0);
    }

    #[test]
    fn empty_score_reports_full_duration_as_silence() {
        let result = measure_temporal_support(&[], 3.0, 6).unwrap();

        assert_eq!(result.onset_window_coverage_ratio, 0.0);
        assert_eq!(result.sustain_window_coverage_ratio, 0.0);
        assert_eq!(result.occupied_time_ratio, 0.0);
        assert_eq!(result.maximum_silent_gap_secs, 3.0);
    }

    #[test]
    fn note_tail_past_clip_is_clipped_for_measurement() {
        let result = measure_temporal_support(&[note(0.75, 1.0)], 1.0, 4).unwrap();

        assert_eq!(result.occupied_time_ratio, 0.25);
        assert_eq!(result.maximum_silent_gap_secs, 0.75);
    }

    #[test]
    fn malformed_notes_fail_closed_with_their_index() {
        let error =
            measure_temporal_support(&[note(0.0, 0.2), note(0.5, f32::NAN)], 1.0, 4).unwrap_err();

        assert_eq!(
            error,
            CreativeTemporalSupportError::InvalidNote { index: 1 }
        );
    }

    #[test]
    fn zero_velocity_silent_notes_cannot_claim_temporal_support() {
        let mut silent = note(0.0, 1.0);
        silent.velocity = 0.0;

        assert_eq!(
            measure_temporal_support(&[silent], 1.0, 4),
            Err(CreativeTemporalSupportError::InvalidNote { index: 0 })
        );
    }

    #[test]
    fn zero_or_excessive_window_counts_are_rejected() {
        assert_eq!(
            measure_temporal_support(&[note(0.0, 0.2)], 1.0, 0),
            Err(CreativeTemporalSupportError::InvalidWindowCount)
        );
        assert_eq!(
            measure_temporal_support(&[note(0.0, 0.2)], 1.0, usize::MAX),
            Err(CreativeTemporalSupportError::InvalidWindowCount)
        );
    }

    #[test]
    fn reversing_note_order_preserves_temporal_measurements() {
        let mut notes = vec![note(0.0, 0.25), note(0.2, 0.5), note(0.8, 0.1)];
        let forward = measure_temporal_support(&notes, 1.0, 8).unwrap();
        notes.reverse();
        let reversed = measure_temporal_support(&notes, 1.0, 8).unwrap();

        assert_eq!(forward, reversed);
    }

    #[test]
    fn duplicate_note_does_not_inflate_union_occupancy() {
        let single = measure_temporal_support(&[note(0.25, 0.5)], 1.0, 4).unwrap();
        let duplicate =
            measure_temporal_support(&[note(0.25, 0.5), note(0.25, 0.5)], 1.0, 4).unwrap();

        assert_eq!(
            single.onset_window_coverage_ratio,
            duplicate.onset_window_coverage_ratio
        );
        assert_eq!(
            single.sustain_window_coverage_ratio,
            duplicate.sustain_window_coverage_ratio
        );
        assert_eq!(single.occupied_time_ratio, duplicate.occupied_time_ratio);
        assert_eq!(
            single.maximum_silent_gap_secs,
            duplicate.maximum_silent_gap_secs
        );
        assert_eq!(duplicate.note_count, single.note_count + 1);
    }
}
