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
pub const MOTIF_RETURN_MEASUREMENT_VERSION: &str = "motif-return-v3";

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
    /// Geometric-mean target/source duration scale for aligned notes.
    pub duration_scale: f32,
    /// Multiplicative consistency of the per-event duration ratios in [0, 1].
    /// A value of 1 means every aligned note uses the same duration scale.
    /// Defaults to zero when reading legacy evidence that did not record it.
    #[serde(default)]
    pub duration_scale_consistency: f32,
    /// Best contiguous interval-window match for fragment recognition.
    pub fragment_similarity: f32,
    /// Rhythm-shape similarity measured against the same aligned source window.
    /// Defaults to zero when reading legacy evidence without this channel.
    #[serde(default)]
    pub fragment_rhythmic_shape_similarity: f32,
    /// Zero-based start-note index of the selected fragment in the sorted source.
    /// None means no eligible fragment alignment was found.
    #[serde(default)]
    pub fragment_source_start_note: Option<usize>,
    /// Fraction of the longer phrase represented by the alignment.
    pub coverage: f32,
    /// Transformation-specific composite in [0, 1].
    pub overall_similarity: f32,
}

impl MotifReturnEvidence {
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        // Restoration is a temporal lineage claim: pairwise similarity cannot
        // prove that a motif was eroded and later recovered. Fail closed until
        // the evidence contract carries an independently verifiable history.
        if self.expected_transformation == ReturnTransformation::Restored {
            return false;
        }

        let expected_scale = match self.expected_transformation {
            ReturnTransformation::Augmented => Some(2.0_f32),
            ReturnTransformation::Diminished => Some(0.5_f32),
            _ => None,
        };
        // Duration-shape normalization alone cannot distinguish augmentation
        // from an unchanged phrase. Require the absolute duration scale to be
        // close to the declared transformation as well as internally uniform.
        // A quarter-octave log2 tolerance permits modest quantization noise
        // while rejecting identity, diminution-as-augmentation, and extreme
        // scale changes. This is a provisional engineering bound to calibrate
        // on held-out annotated returns, not a learned perceptual threshold.
        let expected_scale_compatible = expected_scale.is_none_or(|expected| {
            self.duration_scale.is_finite()
                && self.duration_scale > f32::EPSILON
                && (self.duration_scale / expected).log2().abs() <= 0.25
        });
        let uniform_scaling_required = expected_scale.is_some();
        let length_compatible = match self.expected_transformation {
            ReturnTransformation::Literal
            | ReturnTransformation::Transposed
            | ReturnTransformation::Inverted
            | ReturnTransformation::Augmented
            | ReturnTransformation::Diminished => {
                self.source_note_count == self.target_note_count
            }
            ReturnTransformation::Fragmented => {
                self.target_note_count >= 3 && self.target_note_count < self.source_note_count
            }
            // Retained as an explicit fail-closed arm if this contract is
            // later refactored to avoid the early temporal-lineage guard.
            ReturnTransformation::Restored => false,
        };
        self.aligned_note_count >= 2
            && length_compatible
            && self.overall_similarity >= threshold.clamp(0.0, 1.0)
            && expected_scale_compatible
            && (!uniform_scaling_required || self.duration_scale_consistency >= 0.85)
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
    let (fragment_similarity, fragment_rhythmic_shape_similarity, fragment_source_start_note) =
        best_fragment_alignment(&source, &target, &source_intervals, &target_intervals);
    let rhythmic_shape_similarity = normalized_rhythm_similarity(&source, &target);
    let (duration_scale, duration_scale_consistency) =
        duration_scale_metrics(&source, &target);

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
            duration_scale_consistency,
            fragment_similarity,
            fragment_rhythmic_shape_similarity,
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
        duration_scale_consistency,
        fragment_similarity,
        fragment_rhythmic_shape_similarity,
        fragment_source_start_note,
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
    duration_scale_consistency: f32,
    fragment: f32,
    fragment_rhythm: f32,
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
        ReturnTransformation::Literal => {
            // A literal return must preserve absolute note durations, not
            // merely normalized rhythm shape. Otherwise uniform augmentation
            // and diminution tie with literal identity and are misclassified
            // because pitch and normalized duration proportions are unchanged.
            0.45 * literal
                + 0.20 * rhythm
                + 0.20 * coverage
                + 0.15 * ratio_similarity(1.0) * duration_scale_consistency
        },
        ReturnTransformation::Transposed => {
            0.45 * transposed + 0.25 * interval + 0.20 * rhythm + 0.10 * coverage
        }
        ReturnTransformation::Inverted => {
            0.65 * inverted + 0.10 * transposed + 0.15 * rhythm + 0.10 * coverage
        }
        ReturnTransformation::Augmented => {
            (0.35 * transposed
                + 0.20 * interval
                + 0.20 * rhythm
                + 0.15 * ratio_similarity(2.0)
                + 0.10 * coverage)
                * duration_scale_consistency
        }
        ReturnTransformation::Diminished => {
            (0.35 * transposed
                + 0.20 * interval
                + 0.20 * rhythm
                + 0.15 * ratio_similarity(0.5)
                + 0.10 * coverage)
                * duration_scale_consistency
        }
        ReturnTransformation::Fragmented => {
            // Use the rhythm of the same source window that produced the
            // fragment match; prefix-relative rhythm can score the wrong notes.
            0.65 * fragment + 0.25 * fragment_rhythm + 0.10 * coverage
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

fn duration_scale_metrics(source: &[ScoreNote], target: &[ScoreNote]) -> (f32, f32) {
    let mut log_ratios = Vec::with_capacity(source.len().min(target.len()));
    for (left, right) in source.iter().zip(target) {
        let source_duration = left.duration.beats();
        let target_duration = right.duration.beats();
        if source_duration <= f64::EPSILON || target_duration <= f64::EPSILON {
            // A zero/non-positive duration cannot participate in a positive
            // multiplicative scaling claim. Fail closed rather than dropping it.
            return (0.0, 0.0);
        }
        log_ratios.push((target_duration / source_duration).ln());
    }
    if log_ratios.is_empty() {
        return (0.0, 0.0);
    }

    // Rhythmic scaling is multiplicative: geometric means make x2 and x0.5
    // reciprocal, unlike an arithmetic mean of their ratios.
    let mean_log_ratio = log_ratios.iter().sum::<f64>() / log_ratios.len() as f64;
    let scale = mean_log_ratio.exp();
    let mean_absolute_log_deviation = log_ratios
        .iter()
        .map(|ratio| (ratio - mean_log_ratio).abs())
        .sum::<f64>()
        / log_ratios.len() as f64;
    // 1.0 is exact uniform scaling; larger per-note scale disagreement
    // monotonically lowers this confidence channel toward zero.
    let consistency = (-mean_absolute_log_deviation).exp();
    (scale as f32, consistency as f32)
}

fn best_fragment_alignment(
    source: &[ScoreNote],
    target: &[ScoreNote],
    source_intervals: &[i32],
    target_intervals: &[i32],
) -> (f32, f32, Option<usize>) {
    // Fragmentation is directional: target must be a proper fragment of the
    // earlier/source material. Require at least two intervals (three notes).
    if source.len() < 4
        || target.len() < 3
        || target.len() >= source.len()
        || target_intervals.len() >= source_intervals.len()
    {
        return (0.0, 0.0, None);
    }

    let mut best: Option<(f32, f32, usize)> = None;
    for start in 0..=source.len() - target.len() {
        let interval_score = sequence_similarity(
            target_intervals,
            &source_intervals[start..start + target_intervals.len()],
        );
        let rhythm_score = normalized_rhythm_similarity(
            &source[start..start + target.len()],
            target,
        );
        let is_better = best.as_ref().is_none_or(|(best_intervals, best_rhythm, _)| {
            interval_score > *best_intervals
                || (interval_score == *best_intervals && rhythm_score > *best_rhythm)
        });
        if is_better {
            best = Some((interval_score, rhythm_score, start));
        }
    }

    best.map_or((0.0, 0.0, None), |(intervals, rhythm, start)| {
        (intervals, rhythm, Some(start))
    })
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
    fn inconsistent_per_note_scaling_cannot_pass_as_augmentation() {
        let source = vec![
            note(60, 0, Duration::quarter()),
            note(62, 1, Duration::quarter()),
            note(65, 2, Duration::quarter()),
            note(64, 3, Duration::new(16, 1)),
        ];
        // The first three notes are doubled; the long final note is not.
        // Pitch/interval agreement is perfect, so a naive average-scale
        // heuristic can overstate this as a coherent augmentation.
        let target = vec![
            note(60, 0, Duration::new(2, 1)),
            note(62, 1, Duration::new(2, 1)),
            note(65, 2, Duration::new(2, 1)),
            note(64, 3, Duration::new(16, 1)),
        ];
        let evidence =
            compare_melodic_sequences(&source, &target, ReturnTransformation::Augmented);

        assert!(evidence.duration_scale_consistency < 0.85);
        assert!(
            evidence.overall_similarity < 0.85,
            "the transformation score itself must reflect non-uniform scaling"
        );
        assert!(
            !evidence.meets_threshold(0.85),
            "a high aggregate score must not hide non-uniform per-note scaling"
        );

        let reverse =
            compare_melodic_sequences(&target, &source, ReturnTransformation::Diminished);
        assert!(
            (reverse.duration_scale_consistency - evidence.duration_scale_consistency).abs()
                < 1e-6,
            "scale consistency should be invariant when reversing source and target"
        );
        assert!(reverse.overall_similarity < 0.85);
        assert!(!reverse.meets_threshold(0.85));
    }

    #[test]
    fn uniform_augmentation_has_perfect_scale_consistency() {
        let source = vec![
            note(60, 0, Duration::quarter()),
            note(62, 1, Duration::eighth()),
            note(65, 2, Duration::half()),
        ];
        let target: Vec<_> = source
            .iter()
            .copied()
            .map(|mut note| {
                note.duration = note.duration.scale(2, 1);
                note
            })
            .collect();
        let evidence =
            compare_melodic_sequences(&source, &target, ReturnTransformation::Augmented);

        assert!((evidence.duration_scale - 2.0).abs() < 1e-6);
        assert!((evidence.duration_scale_consistency - 1.0).abs() < 1e-6);
        assert!(evidence.meets_threshold(0.85));

        let diminution =
            compare_melodic_sequences(&target, &source, ReturnTransformation::Diminished);
        assert!((diminution.duration_scale - 0.5).abs() < 1e-6);
        assert!((diminution.duration_scale_consistency - 1.0).abs() < 1e-6);
        assert!(diminution.meets_threshold(0.85));
    }

    #[test]
    fn fragment_return_is_directional_and_requires_a_meaningful_fragment() {
        let source = vec![
            note(60, 0, Duration::quarter()),
            note(62, 1, Duration::quarter()),
            note(65, 2, Duration::quarter()),
            note(64, 3, Duration::quarter()),
        ];
        let real_fragment = vec![
            note(62, 0, Duration::quarter()),
            note(65, 1, Duration::quarter()),
            note(64, 2, Duration::quarter()),
        ];
        let evidence = compare_melodic_sequences(
            &source,
            &real_fragment,
            ReturnTransformation::Fragmented,
        );
        assert_eq!(evidence.fragment_similarity, 1.0);
        assert!(evidence.meets_threshold(0.75));

        let longer_target = vec![
            note(55, 0, Duration::quarter()),
            note(60, 1, Duration::quarter()),
            note(62, 2, Duration::quarter()),
            note(65, 3, Duration::quarter()),
            note(64, 4, Duration::quarter()),
        ];
        let false_fragment = compare_melodic_sequences(
            &source,
            &longer_target,
            ReturnTransformation::Fragmented,
        );
        assert_eq!(false_fragment.fragment_similarity, 0.0);
        assert!(!false_fragment.meets_threshold(0.75));

        let two_note_snippet = vec![
            note(62, 0, Duration::quarter()),
            note(65, 1, Duration::quarter()),
        ];
        let too_short = compare_melodic_sequences(
            &source,
            &two_note_snippet,
            ReturnTransformation::Fragmented,
        );
        assert_eq!(too_short.fragment_similarity, 0.0);
        assert!(!too_short.meets_threshold(0.75));
    }

    #[test]
    fn fragment_rhythm_uses_the_same_interior_window_as_interval_match() {
        let source = vec![
            note(60, 0, Duration::quarter()),
            note(62, 1, Duration::eighth()),
            note(65, 2, Duration::half()),
            note(64, 3, Duration::quarter()),
            note(67, 4, Duration::eighth()),
            note(65, 5, Duration::half()),
        ];
        // The target is an exact fragment from source note index 2. Its
        // rhythm differs from the source prefix, but matches its own window.
        let target = vec![
            note(65, 0, Duration::half()),
            note(64, 1, Duration::quarter()),
            note(67, 2, Duration::eighth()),
            note(65, 3, Duration::half()),
        ];
        let evidence = compare_melodic_sequences(
            &source,
            &target,
            ReturnTransformation::Fragmented,
        );

        assert_eq!(evidence.fragment_similarity, 1.0);
        assert_eq!(evidence.fragment_source_start_note, Some(2));
        assert_eq!(evidence.fragment_rhythmic_shape_similarity, 1.0);
        assert!(evidence.rhythmic_shape_similarity < 1.0);
        assert!(evidence.meets_threshold(0.75));
    }

    #[test]
    fn full_motif_transformations_require_equal_event_counts() {
        let source = vec![
            note(60, 0, Duration::quarter()),
            note(62, 1, Duration::quarter()),
            note(65, 2, Duration::quarter()),
            note(64, 3, Duration::quarter()),
        ];
        let longer_target: Vec<_> = source
            .iter()
            .copied()
            .map(|mut note| {
                note.pitch = note.pitch.transpose(7);
                note
            })
            .chain(std::iter::once(note(72, 4, Duration::quarter())))
            .collect();
        let evidence =
            compare_melodic_sequences(&source, &longer_target, ReturnTransformation::Transposed);
        assert!(evidence.overall_similarity > 0.75);
        assert!(!evidence.meets_threshold(0.75));
    }

    #[test]
    fn detected_transformation_distinguishes_identity_augmentation_and_diminution() {
        let source = vec![
            note(60, 0, Duration::quarter()),
            note(62, 1, Duration::eighth()),
            note(65, 2, Duration::half()),
            note(64, 3, Duration::quarter()),
        ];

        let literal = compare_melodic_sequences(&source, &source, ReturnTransformation::Literal);
        assert_eq!(literal.detected_transformation, ReturnTransformation::Literal);

        let augmented: Vec<_> = source
            .iter()
            .copied()
            .map(|mut event| {
                event.duration = event.duration.scale(2, 1);
                event
            })
            .collect();
        let augmentation = compare_melodic_sequences(
            &source,
            &augmented,
            ReturnTransformation::Augmented,
        );
        assert_eq!(augmentation.detected_transformation, ReturnTransformation::Augmented);
        assert!(augmentation.meets_threshold(0.99));

        let diminished: Vec<_> = source
            .iter()
            .copied()
            .map(|mut event| {
                event.duration = event.duration.scale(1, 2);
                event
            })
            .collect();
        let diminution = compare_melodic_sequences(
            &source,
            &diminished,
            ReturnTransformation::Diminished,
        );
        assert_eq!(diminution.detected_transformation, ReturnTransformation::Diminished);
        assert!(diminution.meets_threshold(0.99));
    }

    #[test]
    fn augmentation_and_diminution_require_the_declared_absolute_scale() {
        let source = vec![
            note(60, 0, Duration::quarter()),
            note(62, 1, Duration::quarter()),
            note(65, 2, Duration::quarter()),
            note(64, 3, Duration::quarter()),
        ];

        // Identity has perfect pitch and normalized rhythm-shape scores, but
        // must not satisfy an augmentation/diminution claim at the 0.85 floor.
        let unchanged_augmentation =
            compare_melodic_sequences(&source, &source, ReturnTransformation::Augmented);
        assert!((unchanged_augmentation.overall_similarity - 0.85).abs() < 1e-6);
        assert!(!unchanged_augmentation.meets_threshold(0.85));

        let unchanged_diminution =
            compare_melodic_sequences(&source, &source, ReturnTransformation::Diminished);
        assert!((unchanged_diminution.overall_similarity - 0.85).abs() < 1e-6);
        assert!(!unchanged_diminution.meets_threshold(0.85));

        // A uniform but wrong factor is also not the transformation requested.
        let quadrupled: Vec<_> = source
            .iter()
            .copied()
            .map(|mut event| {
                event.duration = event.duration.scale(4, 1);
                event
            })
            .collect();
        let wrong_augmentation =
            compare_melodic_sequences(&source, &quadrupled, ReturnTransformation::Augmented);
        assert!(!wrong_augmentation.meets_threshold(0.85));

        let quartered: Vec<_> = source
            .iter()
            .copied()
            .map(|mut event| {
                event.duration = event.duration.scale(1, 4);
                event
            })
            .collect();
        let wrong_diminution =
            compare_melodic_sequences(&source, &quartered, ReturnTransformation::Diminished);
        assert!(!wrong_diminution.meets_threshold(0.85));
    }

    #[test]
    fn restored_pairwise_similarity_never_satisfies_threshold_without_lineage() {
        let source = vec![
            note(60, 0, Duration::quarter()),
            note(62, 1, Duration::quarter()),
            note(65, 2, Duration::quarter()),
            note(64, 3, Duration::quarter()),
        ];
        let evidence =
            compare_melodic_sequences(&source, &source, ReturnTransformation::Restored);

        assert!(
            evidence.overall_similarity >= 0.99,
            "fixture should isolate the lineage rule from pairwise similarity"
        );
        assert!(!evidence.meets_threshold(0.0));
        assert!(!evidence.meets_threshold(0.99));
    }

    #[test]
    fn deterministic_metamorphic_motif_return_suite() {
        // 2,048 frozen seeds × 9 comparisons = 18,432 deterministic
        // transformation and adversarial cases. This qualifies symbolic
        // evidence only; it is not a human-musicality benchmark.
        let mut cases_checked = 0usize;

        for seed in 0u64..2_048 {
            let length = 5 + (seed as usize % 6);
            let mut state = seed.wrapping_add(1);
            let mut midi = 60 + (seed % 9) as i32;
            let mut source = Vec::with_capacity(length);

            for index in 0..length {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let step = ((state >> 32) % 9) as i32 - 4;
                midi = (midi + step).clamp(48, 84);
                let duration = match (state >> 16) % 5 {
                    0 => Duration::quarter(),
                    1 => Duration::eighth(),
                    2 => Duration::half(),
                    3 => Duration::new(3, 2),
                    _ => Duration::triplet_eighth(),
                };
                source.push(note(midi as u8, index as i64, duration));
            }

            // Positive 1: exact literal return.
            let literal = compare_melodic_sequences(
                &source,
                &source,
                ReturnTransformation::Literal,
            );
            assert!(
                literal.meets_threshold(0.99),
                "literal positive failed for seed {seed}: {literal:?}"
            );
            cases_checked += 1;

            // Positive 2: exact constant-semitone transposition.
            let transposed: Vec<_> = source
                .iter()
                .copied()
                .map(|mut event| {
                    event.pitch = event.pitch.transpose(5);
                    event
                })
                .collect();
            let transposed_evidence = compare_melodic_sequences(
                &source,
                &transposed,
                ReturnTransformation::Transposed,
            );
            assert!(
                transposed_evidence.meets_threshold(0.99),
                "transposition positive failed for seed {seed}: {transposed_evidence:?}"
            );
            cases_checked += 1;

            // Positive 3: exact pitch inversion around a fixed MIDI pivot.
            let inverted: Vec<_> = source
                .iter()
                .copied()
                .map(|mut event| {
                    event.pitch = Pitch::from_midi((132 - event.pitch.midi() as i32) as u8);
                    event
                })
                .collect();
            let inverted_evidence = compare_melodic_sequences(
                &source,
                &inverted,
                ReturnTransformation::Inverted,
            );
            assert!(
                inverted_evidence.meets_threshold(0.85),
                "inversion positive failed for seed {seed}: {inverted_evidence:?}"
            );
            cases_checked += 1;

            // Positive 4/5: exact uniform augmentation and diminution.
            let augmented: Vec<_> = source
                .iter()
                .copied()
                .map(|mut event| {
                    event.duration = event.duration.scale(2, 1);
                    event
                })
                .collect();
            let augmented_evidence = compare_melodic_sequences(
                &source,
                &augmented,
                ReturnTransformation::Augmented,
            );
            assert!(
                augmented_evidence.meets_threshold(0.99),
                "augmentation positive failed for seed {seed}: {augmented_evidence:?}"
            );
            cases_checked += 1;

            let diminished: Vec<_> = source
                .iter()
                .copied()
                .map(|mut event| {
                    event.duration = event.duration.scale(1, 2);
                    event
                })
                .collect();
            let diminished_evidence = compare_melodic_sequences(
                &source,
                &diminished,
                ReturnTransformation::Diminished,
            );
            assert!(
                diminished_evidence.meets_threshold(0.99),
                "diminution positive failed for seed {seed}: {diminished_evidence:?}"
            );
            cases_checked += 1;

            // Positive 6: exact interior fragment with matching rhythm.
            let fragment = source[1..length - 1].to_vec();
            let fragment_evidence = compare_melodic_sequences(
                &source,
                &fragment,
                ReturnTransformation::Fragmented,
            );
            assert!(
                fragment_evidence.meets_threshold(0.75),
                "fragment positive failed for seed {seed}: {fragment_evidence:?}"
            );
            cases_checked += 1;

            // Negative 7: one corrupted note must not pass a full transposition.
            let mut corrupted = transposed.clone();
            let corrupted_index = length / 2;
            corrupted[corrupted_index].pitch =
                corrupted[corrupted_index].pitch.transpose(1);
            let corrupted_evidence = compare_melodic_sequences(
                &source,
                &corrupted,
                ReturnTransformation::Transposed,
            );
            assert!(
                !corrupted_evidence.meets_threshold(0.95),
                "corrupted transposition was accepted for seed {seed}: {corrupted_evidence:?}"
            );
            cases_checked += 1;

            // Negative 8: one duration receives an eightfold rather than
            // twofold scale; other notes remain doubled.
            let mut irregular_scale = augmented.clone();
            irregular_scale[corrupted_index].duration =
                source[corrupted_index].duration.scale(8, 1);
            let irregular_evidence = compare_melodic_sequences(
                &source,
                &irregular_scale,
                ReturnTransformation::Augmented,
            );
            assert!(
                !irregular_evidence.meets_threshold(0.85),
                "non-uniform augmentation was accepted for seed {seed}: {irregular_evidence:?}"
            );
            cases_checked += 1;

            // Negative 9: a longer target that embeds the source is not a
            // fragment of the shorter source (wrong direction).
            let mut longer_target = Vec::with_capacity(length + 1);
            longer_target.push(note(50, 0, Duration::quarter()));
            longer_target.extend(source.iter().enumerate().map(|(index, mut event)| {
                event.onset = Duration::new(index as i64 + 1, 1);
                event
            }));
            let wrong_direction = compare_melodic_sequences(
                &source,
                &longer_target,
                ReturnTransformation::Fragmented,
            );
            assert!(
                !wrong_direction.meets_threshold(0.75),
                "wrong-direction fragment was accepted for seed {seed}: {wrong_direction:?}"
            );
            cases_checked += 1;
        }

        assert_eq!(cases_checked, 18_432);
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
