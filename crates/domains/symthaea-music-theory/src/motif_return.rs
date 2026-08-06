// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transformation-aware thematic return analysis.
//!
//! Exact pitch equality is too narrow for long-range musical memory: a theme
//! can return transposed, inverted, augmented, diminished, or fragmented and
//! still be recognizably the same identity. This module compares two melodic
//! regions with explicit, deterministic channels instead of collapsing every
//! relationship into one opaque similarity score.
//!
//! The resulting values are symbolic proxies. They are suitable for score-side
//! obligation verification and candidate ranking, not claims about listener
//! recognition.

use crate::obligation::ReturnTransformation;
use crate::rhythm::Duration;
use crate::score::{Score, ScoreNote, VoiceRole};
use serde::{Deserialize, Serialize};

/// Version of the transformation-aware motif-return measurement contract.
pub const MOTIF_RETURN_MEASUREMENT_VERSION: &str = "motif-return-v1";

/// Auditable evidence that one melodic region recalls another.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotifReturnEvidence {
    pub measurement_version: String,
    pub expected_transformation: ReturnTransformation,
    pub detected_transformation: ReturnTransformation,
    pub source_note_count: usize,
    pub target_note_count: usize,
    pub aligned_note_count: usize,
    /// Exact MIDI-pitch matches at aligned positions.
    pub literal_pitch_similarity: f32,
    /// Pitch matches after one constant semitone translation.
    pub transposed_pitch_similarity: f32,
    /// Directed-interval matches between adjacent notes.
    pub interval_similarity: f32,
    /// Directed-interval matches after sign inversion.
    pub inverted_interval_similarity: f32,
    /// Up/down/same contour matches.
    pub contour_similarity: f32,
    /// Similarity of durations after normalizing each phrase by total length.
    pub rhythmic_shape_similarity: f32,
    /// Mean target/source duration scale when both phrases contain notes.
    pub duration_scale: f32,
    /// Best contiguous interval-window match for fragment recognition.
    pub fragment_similarity: f32,
    /// Fraction of the longer phrase represented by the alignment.
    pub coverage: f32,
    /// Transformation-specific composite in [0, 1].
    pub overall_similarity: f32,
}

impl MotifReturnEvidence {
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.aligned_note_count >= 2 && self.overall_similarity >= threshold.clamp(0.0, 1.0)
    }
}

/// Compare the melody voice in two score regions.
pub fn compare_melodic_regions(
    score: &Score,
    source_start: Duration,
    source_end: Duration,
    target_start: Duration,
    target_end: Duration,
    expected: ReturnTransformation,
) -> MotifReturnEvidence {
    let source = melodic_notes_in_region(score, source_start, source_end);
    let target = melodic_notes_in_region(score, target_start, target_end);
    compare_melodic_sequences(&source, &target, expected)
}

/// Compare two already-extracted melodic sequences.
pub fn compare_melodic_sequences(
    source: &[ScoreNote],
    target: &[ScoreNote],
    expected: ReturnTransformation,
) -> MotifReturnEvidence {
    let mut source = source.to_vec();
    let mut target = target.to_vec();
    source.sort_by(|left, right| left.onset.beats().total_cmp(&right.onset.beats()));
    target.sort_by(|left, right| left.onset.beats().total_cmp(&right.onset.beats()));

    let aligned = source.len().min(target.len());
    let longer = source.len().max(target.len());
    let coverage = if longer == 0 {
        0.0
    } else {
        aligned as f32 / longer as f32
    };

    let literal_pitch_similarity =
        positional_similarity(aligned, |index| source[index].pitch == target[index].pitch);
    let transposition = source
        .first()
        .zip(target.first())
        .map(|(left, right)| left.pitch.semitones_to(right.pitch))
        .unwrap_or(0);
    let transposed_pitch_similarity = positional_similarity(aligned, |index| {
        source[index].pitch.transpose(transposition) == target[index].pitch
    });

    let source_intervals = directed_intervals(&source);
    let target_intervals = directed_intervals(&target);
    let interval_similarity = sequence_similarity(&source_intervals, &target_intervals);
    let inverted_source: Vec<i32> = source_intervals.iter().map(|value| -*value).collect();
    let inverted_interval_similarity = sequence_similarity(&inverted_source, &target_intervals);
    let contour_similarity = sequence_similarity(
        &source_intervals
            .iter()
            .map(|value| value.signum())
            .collect::<Vec<_>>(),
        &target_intervals
            .iter()
            .map(|value| value.signum())
            .collect::<Vec<_>>(),
    );
    let fragment_similarity = best_fragment_similarity(&source_intervals, &target_intervals);
    let rhythmic_shape_similarity = normalized_rhythm_similarity(&source, &target);
    let duration_scale = mean_duration_scale(&source, &target);

    let score_for = |kind| {
        transformation_score(
            kind,
            literal_pitch_similarity,
            transposed_pitch_similarity,
            interval_similarity,
            inverted_interval_similarity,
            contour_similarity,
            rhythmic_shape_similarity,
            duration_scale,
            fragment_similarity,
            coverage,
        )
    };
    // Retain the first transformation on exact ties so a literal return is
    // not relabelled as a more permissive transformation with the same score.
    let transformations = [
        ReturnTransformation::Literal,
        ReturnTransformation::Transposed,
        ReturnTransformation::Inverted,
        ReturnTransformation::Augmented,
        ReturnTransformation::Diminished,
        ReturnTransformation::Fragmented,
        ReturnTransformation::Restored,
    ];
    let mut detected_transformation = transformations[0];
    let mut detected_score = score_for(detected_transformation);
    for transformation in transformations.into_iter().skip(1) {
        let score = score_for(transformation);
        if score > detected_score {
            detected_transformation = transformation;
            detected_score = score;
        }
    }
    let overall_similarity = score_for(expected);

    MotifReturnEvidence {
        measurement_version: MOTIF_RETURN_MEASUREMENT_VERSION.into(),
        expected_transformation: expected,
        detected_transformation,
        source_note_count: source.len(),
        target_note_count: target.len(),
        aligned_note_count: aligned,
        literal_pitch_similarity,
        transposed_pitch_similarity,
        interval_similarity,
        inverted_interval_similarity,
        contour_similarity,
        rhythmic_shape_similarity,
        duration_scale,
        fragment_similarity,
        coverage,
        overall_similarity,
    }
}

/// Extract melody notes whose onset lies in `[start, end)`.
pub fn melodic_notes_in_region(score: &Score, start: Duration, end: Duration) -> Vec<ScoreNote> {
    let mut notes: Vec<_> = score
        .notes
        .iter()
        .copied()
        .filter(|note| {
            note.role == VoiceRole::Melody
                && note.onset.beats() >= start.beats()
                && note.onset.beats() < end.beats()
        })
        .collect();
    notes.sort_by(|left, right| left.onset.beats().total_cmp(&right.onset.beats()));
    notes
}

#[allow(clippy::too_many_arguments)]
fn transformation_score(
    kind: ReturnTransformation,
    literal: f32,
    transposed: f32,
    interval: f32,
    inverted: f32,
    contour: f32,
    rhythm: f32,
    duration_scale: f32,
    fragment: f32,
    coverage: f32,
) -> f32 {
    let ratio_similarity = |expected: f32| {
        if duration_scale <= f32::EPSILON {
            0.0
        } else {
            (1.0 - ((duration_scale / expected).ln().abs() / 2.0_f32.ln())).clamp(0.0, 1.0)
        }
    };
    let value = match kind {
        ReturnTransformation::Literal => 0.55 * literal + 0.25 * rhythm + 0.20 * coverage,
        ReturnTransformation::Transposed => {
            0.45 * transposed + 0.25 * interval + 0.20 * rhythm + 0.10 * coverage
        }
        ReturnTransformation::Inverted => {
            0.65 * inverted + 0.10 * transposed + 0.15 * rhythm + 0.10 * coverage
        }
        ReturnTransformation::Augmented => {
            0.35 * transposed
                + 0.20 * interval
                + 0.20 * rhythm
                + 0.15 * ratio_similarity(2.0)
                + 0.10 * coverage
        }
        ReturnTransformation::Diminished => {
            0.35 * transposed
                + 0.20 * interval
                + 0.20 * rhythm
                + 0.15 * ratio_similarity(0.5)
                + 0.10 * coverage
        }
        ReturnTransformation::Fragmented => {
            0.55 * fragment + 0.20 * contour + 0.15 * rhythm + 0.10 * coverage
        }
        ReturnTransformation::Restored => {
            0.50 * literal.max(transposed) + 0.20 * interval + 0.20 * rhythm + 0.10 * coverage
        }
    };
    value.clamp(0.0, 1.0)
}

fn positional_similarity(length: usize, predicate: impl Fn(usize) -> bool) -> f32 {
    if length == 0 {
        return 0.0;
    }
    (0..length).filter(|index| predicate(*index)).count() as f32 / length as f32
}

fn directed_intervals(notes: &[ScoreNote]) -> Vec<i32> {
    notes
        .windows(2)
        .map(|pair| pair[0].pitch.semitones_to(pair[1].pitch))
        .collect()
}

fn sequence_similarity<T: PartialEq>(source: &[T], target: &[T]) -> f32 {
    let denominator = source.len().max(target.len());
    if denominator == 0 {
        return 0.0;
    }
    source
        .iter()
        .zip(target)
        .filter(|(left, right)| left == right)
        .count() as f32
        / denominator as f32
}

fn normalized_rhythm_similarity(source: &[ScoreNote], target: &[ScoreNote]) -> f32 {
    let count = source.len().min(target.len());
    if count == 0 {
        return 0.0;
    }
    let source_total: f64 = source
        .iter()
        .take(count)
        .map(|note| note.duration.beats())
        .sum();
    let target_total: f64 = target
        .iter()
        .take(count)
        .map(|note| note.duration.beats())
        .sum();
    if source_total <= f64::EPSILON || target_total <= f64::EPSILON {
        return 0.0;
    }
    let mean_error = source
        .iter()
        .zip(target)
        .take(count)
        .map(|(left, right)| {
            let left = left.duration.beats() / source_total;
            let right = right.duration.beats() / target_total;
            (left - right).abs()
        })
        .sum::<f64>()
        / count as f64;
    (1.0 - mean_error as f32 * count as f32).clamp(0.0, 1.0)
}

fn mean_duration_scale(source: &[ScoreNote], target: &[ScoreNote]) -> f32 {
    let ratios: Vec<f64> = source
        .iter()
        .zip(target)
        .filter_map(|(left, right)| {
            let left = left.duration.beats();
            (left > f64::EPSILON).then_some(right.duration.beats() / left)
        })
        .collect();
    if ratios.is_empty() {
        0.0
    } else {
        (ratios.iter().sum::<f64>() / ratios.len() as f64) as f32
    }
}

fn best_fragment_similarity(source: &[i32], target: &[i32]) -> f32 {
    if source.is_empty() || target.is_empty() {
        return 0.0;
    }
    let (longer, shorter) = if source.len() >= target.len() {
        (source, target)
    } else {
        (target, source)
    };
    if shorter.is_empty() {
        return 0.0;
    }
    (0..=longer.len() - shorter.len())
        .map(|offset| {
            shorter
                .iter()
                .zip(&longer[offset..offset + shorter.len()])
                .filter(|(left, right)| left == right)
                .count() as f32
                / shorter.len() as f32
        })
        .fold(0.0_f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::pitch::{Pitch, PitchClass};
    use crate::score::Emphasis;
    use crate::score::PartId;

    fn note(midi: u8, onset: i64, duration: Duration) -> ScoreNote {
        ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: Pitch::from_midi(midi),
            onset: Duration::new(onset, 1),
            duration,
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    #[test]
    fn transposed_theme_is_recognized_without_literal_pitch_equality() {
        let source = vec![
            note(60, 0, Duration::quarter()),
            note(62, 1, Duration::quarter()),
            note(65, 2, Duration::quarter()),
            note(64, 3, Duration::quarter()),
        ];
        let target: Vec<_> = source
            .iter()
            .copied()
            .map(|mut note| {
                note.pitch = note.pitch.transpose(7);
                note
            })
            .collect();
        let evidence =
            compare_melodic_sequences(&source, &target, ReturnTransformation::Transposed);
        assert_eq!(evidence.literal_pitch_similarity, 0.0);
        assert_eq!(evidence.transposed_pitch_similarity, 1.0);
        assert!(evidence.overall_similarity > 0.95);
    }

    #[test]
    fn inversion_and_augmentation_are_distinguished() {
        let source = vec![
            note(60, 0, Duration::quarter()),
            note(62, 1, Duration::quarter()),
            note(65, 2, Duration::quarter()),
        ];
        let inverted = vec![
            note(67, 0, Duration::quarter()),
            note(65, 1, Duration::quarter()),
            note(62, 2, Duration::quarter()),
        ];
        let inversion =
            compare_melodic_sequences(&source, &inverted, ReturnTransformation::Inverted);
        assert_eq!(inversion.inverted_interval_similarity, 1.0);
        assert!(inversion.overall_similarity > 0.85);

        let augmented: Vec<_> = source
            .iter()
            .copied()
            .map(|mut note| {
                note.duration = note.duration.scale(2, 1);
                note
            })
            .collect();
        let augmentation =
            compare_melodic_sequences(&source, &augmented, ReturnTransformation::Augmented);
        assert!((augmentation.duration_scale - 2.0).abs() < 1e-6);
        assert!(augmentation.overall_similarity > 0.9);
    }

    #[test]
    fn region_extraction_ignores_non_melodic_voices() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(note(60, 0, Duration::quarter()));
        let mut harmony = note(67, 0, Duration::quarter());
        harmony.role = VoiceRole::Harmony;
        score.push(harmony);
        assert_eq!(
            melodic_notes_in_region(&score, Duration::zero(), Duration::half()).len(),
            1
        );
    }
}
