// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Score-space evaluation for automatic music transcription.
//!
//! This module deliberately evaluates a normalized notation representation,
//! not MIDI audio-time notes. It separates "the notes were heard" from
//! "the notes were placed in the right measure, beat and meter".
//!
//! The score-grid alignment search follows the documented public benchmark
//! transform family: global scale factors, whole-bar shifts and quarter-beat
//! pickup offsets. Results are tied to the exact normalized inputs.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const SCORE_GRID_EVALUATION_SCHEMA_V1: &str = "muse-score-grid-evaluation-v1";
pub const SCORE_GRID_EVALUATION_ALGORITHM_V1: &str =
    "score-grid-32nd-note-multiset-f1-alignment-search-v1";

pub const SCORE_GRID_UNITS_PER_QUARTER_V1: u16 = 8;
pub const SCORE_GRID_MAX_NOTES_V1: usize = 20_000;
pub const SCORE_GRID_MAX_BARS_V1: u32 = 10_000;
pub const SCORE_GRID_MAX_BEAT_MARKERS_V1: usize = 50_000;
pub const SCORE_GRID_MAX_ALIGNMENT_NOTE_SCANS_V1: usize = 40_000_000;
pub const SCORE_GRID_MAX_EDIT_DISTANCE_CELLS_V1: usize = 10_000_000;

const ALIGNMENT_SCALES_V1: [(u8, u8); 9] = [
    (1, 1),
    (2, 1),
    (1, 2),
    (3, 2),
    (2, 3),
    (3, 1),
    (4, 3),
    (3, 4),
    (1, 3),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ScoreGridNoteV1 {
    /// Zero-based measure index in the exported score.
    pub bar_index: i32,
    /// Onset position inside the measure, in 1/32-note units.
    pub onset_32nd: u16,
    /// Duration in 1/32-note units.
    pub duration_32nd: u16,
    pub pitch_midi: u8,
    /// Logical score PartId when the notation adapter can preserve it.
    pub part_id: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreGridMeterMarkV1 {
    /// Measure at which this meter becomes effective.
    pub bar_index: u32,
    pub numerator: u8,
    pub denominator: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreGridObservationV1 {
    pub bar_count: u32,
    pub notes: Vec<ScoreGridNoteV1>,
    /// Meter changes; the first mark must apply at bar zero.
    pub meter_marks: Vec<ScoreGridMeterMarkV1>,
    /// Audio-time downbeat/barline positions, in seconds.
    pub barline_times_seconds: Vec<f64>,
    /// Audio-time beat positions, used to set a tempo-relative barline tolerance.
    pub beat_times_seconds: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreGridEvaluationMetricV1 {
    pub true_positives: u32,
    pub false_positives: u32,
    pub false_negatives: u32,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreGridAlignmentV1 {
    pub scale_numerator: u8,
    pub scale_denominator: u8,
    pub delta_bars: i8,
    /// Pickup shift in 1/32-note units. One quarter-beat step is two units.
    pub delta_onset_32nd: i8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreGridMeterAccuracyV1 {
    pub correct_bars: u32,
    pub incorrect_bars: u32,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreGridEditDistanceStatusV1 {
    Available,
    WorkBudgetExceeded,
    EmptyReference,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreGridEvaluationReportV1 {
    pub schema: String,
    pub algorithm: String,
    pub reference_sha256: String,
    pub prediction_sha256: String,
    pub alignment: ScoreGridAlignmentV1,
    /// Exact (bar, 1/32-note position, MIDI pitch) F1 after global alignment.
    pub score_f1: ScoreGridEvaluationMetricV1,
    /// Exact (bar, 1/32-note position, duration) F1; pitch is ignored.
    pub rhythm_f1: ScoreGridEvaluationMetricV1,
    /// Exact (bar, position, MIDI pitch, PartId) F1; absent if either side
    /// lacks complete logical-part identities.
    pub part_aware_score_f1: Option<ScoreGridEvaluationMetricV1>,
    /// Levenshtein distance / reference note count over (bar, position, pitch, duration).
    pub score_error_rate: Option<f64>,
    pub score_error_distance: Option<u32>,
    pub score_error_distance_cells: u64,
    pub score_error_distance_status: ScoreGridEditDistanceStatusV1,
    /// Downbeat/barline F1 in audio time, with tolerance 0.175 × median beat interval.
    pub barline_f1: Option<ScoreGridEvaluationMetricV1>,
    pub barline_tolerance_seconds: Option<f64>,
    /// Exact per-measure time-signature accuracy.
    pub meter_accuracy: Option<ScoreGridMeterAccuracyV1>,
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScoreGridEvaluationIssueV1 {
    InvalidBarCount,
    TooManyBars { count: u32, maximum: u32 },
    TooManyNotes { count: usize, maximum: usize },
    TooManyBeatMarkers { count: usize, maximum: usize },
    InvalidMeterMap,
    InvalidMeter { bar_index: u32, numerator: u8, denominator: u8 },
    InvalidNote { ordinal: usize },
    InvalidTimeline,
    AlignmentWorkBudgetExceeded { scans: usize, maximum: usize },
    SerializationFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ScoreKeyV1 {
    bar_index: i32,
    onset_32nd: u16,
    pitch_midi: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RhythmKeyV1 {
    bar_index: i32,
    onset_32nd: u16,
    duration_32nd: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PartScoreKeyV1 {
    bar_index: i32,
    onset_32nd: u16,
    pitch_midi: u8,
    part_id: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TransformedNoteV1 {
    bar_index: i32,
    onset_32nd: u16,
    duration_32nd: u32,
    pitch_midi: u8,
    part_id: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EditTokenV1 {
    bar_index: i32,
    onset_32nd: u16,
    pitch_midi: u8,
    duration_32nd: u32,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn domain_separated_sha256(domain: &str, bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain.as_bytes());
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn meter_for_bar_v1(
    observation: &ScoreGridObservationV1,
    bar_index: i32,
) -> Option<ScoreGridMeterMarkV1> {
    if bar_index < 0 || bar_index as u32 >= observation.bar_count {
        return None;
    }
    let bar_index = bar_index as u32;
    let index = observation
        .meter_marks
        .partition_point(|mark| mark.bar_index <= bar_index);
    index.checked_sub(1).map(|index| observation.meter_marks[index])
}

fn measure_length_32nds_v1(mark: ScoreGridMeterMarkV1) -> u32 {
    u32::from(mark.numerator) * 32 / u32::from(mark.denominator)
}

fn validate_meter_mark_v1(mark: ScoreGridMeterMarkV1) -> bool {
    mark.numerator > 0
        && mark.denominator > 0
        && mark.denominator <= 32
        && mark.denominator.is_power_of_two()
}

fn validate_times_v1(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite() && *value >= 0.0)
        && values.windows(2).all(|window| window[0] < window[1])
}

pub fn validate_score_grid_observation_v1(
    observation: &ScoreGridObservationV1,
) -> Result<(), ScoreGridEvaluationIssueV1> {
    if observation.bar_count == 0 {
        return Err(ScoreGridEvaluationIssueV1::InvalidBarCount);
    }
    if observation.bar_count > SCORE_GRID_MAX_BARS_V1 {
        return Err(ScoreGridEvaluationIssueV1::TooManyBars {
            count: observation.bar_count,
            maximum: SCORE_GRID_MAX_BARS_V1,
        });
    }
    if observation.notes.len() > SCORE_GRID_MAX_NOTES_V1 {
        return Err(ScoreGridEvaluationIssueV1::TooManyNotes {
            count: observation.notes.len(),
            maximum: SCORE_GRID_MAX_NOTES_V1,
        });
    }
    if observation.beat_times_seconds.len() > SCORE_GRID_MAX_BEAT_MARKERS_V1
        || observation.barline_times_seconds.len() > SCORE_GRID_MAX_BARS_V1 as usize + 1
    {
        return Err(ScoreGridEvaluationIssueV1::TooManyBeatMarkers {
            count: observation.beat_times_seconds.len(),
            maximum: SCORE_GRID_MAX_BEAT_MARKERS_V1,
        });
    }
    if !validate_times_v1(&observation.beat_times_seconds)
        || !validate_times_v1(&observation.barline_times_seconds)
    {
        return Err(ScoreGridEvaluationIssueV1::InvalidTimeline);
    }
    if !observation.meter_marks.is_empty() {
        if observation.meter_marks[0].bar_index != 0
            || !observation
                .meter_marks
                .windows(2)
                .all(|window| window[0].bar_index < window[1].bar_index)
            || observation
                .meter_marks
                .iter()
                .any(|mark| mark.bar_index >= observation.bar_count || !validate_meter_mark_v1(*mark))
        {
            return Err(ScoreGridEvaluationIssueV1::InvalidMeterMap);
        }
    }

    for (ordinal, note) in observation.notes.iter().enumerate() {
        if note.bar_index < 0
            || note.bar_index as u32 >= observation.bar_count
            || note.duration_32nd == 0
            || note.pitch_midi > 127
            || note.part_id == Some(u16::MAX)
        {
            return Err(ScoreGridEvaluationIssueV1::InvalidNote { ordinal });
        }
        if let Some(mark) = meter_for_bar_v1(observation, note.bar_index) {
            if u32::from(note.onset_32nd) >= measure_length_32nds_v1(mark) {
                return Err(ScoreGridEvaluationIssueV1::InvalidNote { ordinal });
            }
        }
    }
    Ok(())
}

fn metric_from_counts_v1(
    reference_count: usize,
    prediction_count: usize,
    true_positives: usize,
) -> ScoreGridEvaluationMetricV1 {
    let precision = if prediction_count == 0 {
        0.0
    } else {
        true_positives as f64 / prediction_count as f64
    };
    let recall = if reference_count == 0 {
        0.0
    } else {
        true_positives as f64 / reference_count as f64
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    ScoreGridEvaluationMetricV1 {
        true_positives: u32::try_from(true_positives).unwrap_or(u32::MAX),
        false_positives: u32::try_from(prediction_count.saturating_sub(true_positives))
            .unwrap_or(u32::MAX),
        false_negatives: u32::try_from(reference_count.saturating_sub(true_positives))
            .unwrap_or(u32::MAX),
        precision,
        recall,
        f1,
    }
}

fn multiset_true_positives_v1<K: Ord>(
    reference_keys: impl Iterator<Item = K>,
    prediction_keys: impl Iterator<Item = Option<K>>,
) -> usize {
    let mut counts = BTreeMap::<K, usize>::new();
    for key in reference_keys {
        *counts.entry(key).or_default() += 1;
    }
    let mut true_positives = 0usize;
    for key in prediction_keys.flatten() {
        if let Some(count) = counts.get_mut(&key) {
            if *count > 0 {
                *count -= 1;
                true_positives += 1;
            }
        }
    }
    true_positives
}

fn round_scaled_grid_value_v1(value: u16, numerator: u8, denominator: u8) -> Option<u32> {
    let product = u32::from(value).checked_mul(u32::from(numerator))?;
    let denominator = u32::from(denominator);
    let quotient = product / denominator;
    let remainder = product % denominator;
    let rounded = if remainder * 2 > denominator
        || (remainder * 2 == denominator && quotient % 2 == 1)
    {
        quotient.checked_add(1)?
    } else {
        quotient
    };
    Some(rounded)
}

fn transform_prediction_v1(
    prediction: &ScoreGridObservationV1,
    reference: &ScoreGridObservationV1,
    alignment: ScoreGridAlignmentV1,
) -> Vec<Option<TransformedNoteV1>> {
    prediction
        .notes
        .iter()
        .map(|note| {
            let bar_index = note.bar_index.checked_add(i32::from(alignment.delta_bars))?;
            let scaled_onset = round_scaled_grid_value_v1(
                note.onset_32nd,
                alignment.scale_numerator,
                alignment.scale_denominator,
            )?;
            let onset = i64::from(scaled_onset) + i64::from(alignment.delta_onset_32nd);
            if onset < 0 || onset > i64::from(u16::MAX) {
                return None;
            }
            if let Some(mark) = meter_for_bar_v1(reference, bar_index) {
                if onset as u32 >= measure_length_32nds_v1(mark) {
                    return None;
                }
            }
            let duration = round_scaled_grid_value_v1(
                note.duration_32nd,
                alignment.scale_numerator,
                alignment.scale_denominator,
            )?;
            Some(TransformedNoteV1 {
                bar_index,
                onset_32nd: onset as u16,
                duration_32nd: duration,
                pitch_midi: note.pitch_midi,
                part_id: note.part_id,
            })
        })
        .collect()
}

fn score_metric_for_alignment_v1(
    reference: &ScoreGridObservationV1,
    prediction: &ScoreGridObservationV1,
    alignment: ScoreGridAlignmentV1,
) -> (
    ScoreGridEvaluationMetricV1,
    ScoreGridEvaluationMetricV1,
    Option<ScoreGridEvaluationMetricV1>,
    Vec<Option<TransformedNoteV1>>,
) {
    let transformed = transform_prediction_v1(prediction, reference, alignment);
    let score_tp = multiset_true_positives_v1(
        reference.notes.iter().map(|note| ScoreKeyV1 {
            bar_index: note.bar_index,
            onset_32nd: note.onset_32nd,
            pitch_midi: note.pitch_midi,
        }),
        transformed.iter().map(|note| {
            note.map(|note| ScoreKeyV1 {
                bar_index: note.bar_index,
                onset_32nd: note.onset_32nd,
                pitch_midi: note.pitch_midi,
            })
        }),
    );
    let rhythm_tp = multiset_true_positives_v1(
        reference.notes.iter().map(|note| RhythmKeyV1 {
            bar_index: note.bar_index,
            onset_32nd: note.onset_32nd,
            duration_32nd: note.duration_32nd,
        }),
        transformed.iter().map(|note| {
            note.map(|note| RhythmKeyV1 {
                bar_index: note.bar_index,
                onset_32nd: note.onset_32nd,
                duration_32nd: note.duration_32nd,
            })
        }),
    );
    let part_aware = if reference
        .notes
        .iter()
        .all(|note| note.part_id.is_some())
        && prediction
            .notes
            .iter()
            .all(|note| note.part_id.is_some())
    {
        let true_positives = multiset_true_positives_v1(
            reference.notes.iter().map(|note| PartScoreKeyV1 {
                bar_index: note.bar_index,
                onset_32nd: note.onset_32nd,
                pitch_midi: note.pitch_midi,
                part_id: note.part_id.unwrap_or(u16::MAX),
            }),
            transformed.iter().map(|note| {
                note.and_then(|note| {
                    note.part_id.map(|part_id| PartScoreKeyV1 {
                        bar_index: note.bar_index,
                        onset_32nd: note.onset_32nd,
                        pitch_midi: note.pitch_midi,
                        part_id,
                    })
                })
            }),
        );
        Some(metric_from_counts_v1(
            reference.notes.len(),
            prediction.notes.len(),
            true_positives,
        ))
    } else {
        None
    };
    (
        metric_from_counts_v1(reference.notes.len(), prediction.notes.len(), score_tp),
        metric_from_counts_v1(reference.notes.len(), prediction.notes.len(), rhythm_tp),
        part_aware,
        transformed,
    )
}

fn alignment_candidates_v1() -> impl Iterator<Item = ScoreGridAlignmentV1> {
    ALIGNMENT_SCALES_V1.into_iter().flat_map(|(scale_numerator, scale_denominator)| {
        (-3i8..=3).flat_map(move |delta_bars| {
            (0i8..=12).flat_map(move |step| {
                let offsets = if step == 0 {
                    vec![0]
                } else {
                    vec![-2 * step, 2 * step]
                };
                offsets.into_iter().map(move |delta_onset_32nd| ScoreGridAlignmentV1 {
                    scale_numerator,
                    scale_denominator,
                    delta_bars,
                    delta_onset_32nd,
                })
            })
        })
    })
}

fn median_v1(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    })
}

fn barline_metrics_v1(
    reference: &ScoreGridObservationV1,
    prediction: &ScoreGridObservationV1,
) -> Option<(ScoreGridEvaluationMetricV1, f64)> {
    if reference.beat_times_seconds.len() < 2 {
        return None;
    }
    let beat_intervals: Vec<f64> = reference
        .beat_times_seconds
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect();
    let median_beat_interval = median_v1(&beat_intervals)?;
    if !median_beat_interval.is_finite() || median_beat_interval <= 0.0 {
        return None;
    }
    let tolerance = median_beat_interval * 0.175;
    let mut reference_index = 0usize;
    let mut prediction_index = 0usize;
    let mut true_positives = 0usize;
    while reference_index < reference.barline_times_seconds.len()
        && prediction_index < prediction.barline_times_seconds.len()
    {
        let reference_time = reference.barline_times_seconds[reference_index];
        let prediction_time = prediction.barline_times_seconds[prediction_index];
        let difference = prediction_time - reference_time;
        if difference.abs() <= tolerance {
            true_positives += 1;
            reference_index += 1;
            prediction_index += 1;
        } else if difference < -tolerance {
            prediction_index += 1;
        } else {
            reference_index += 1;
        }
    }
    Some((
        metric_from_counts_v1(
            reference.barline_times_seconds.len(),
            prediction.barline_times_seconds.len(),
            true_positives,
        ),
        tolerance,
    ))
}

fn meter_accuracy_v1(
    reference: &ScoreGridObservationV1,
    prediction: &ScoreGridObservationV1,
) -> Option<ScoreGridMeterAccuracyV1> {
    if reference.meter_marks.is_empty() {
        return None;
    }
    let mut correct = 0usize;
    for bar_index in 0..reference.bar_count {
        let Some(reference_meter) = meter_for_bar_v1(reference, bar_index as i32) else {
            continue;
        };
        if meter_for_bar_v1(prediction, bar_index as i32).is_some_and(|prediction_meter| {
            prediction_meter.numerator == reference_meter.numerator
                && prediction_meter.denominator == reference_meter.denominator
        }) {
            correct += 1;
        }
    }
    let total = reference.bar_count as usize;
    Some(ScoreGridMeterAccuracyV1 {
        correct_bars: u32::try_from(correct).unwrap_or(u32::MAX),
        incorrect_bars: u32::try_from(total.saturating_sub(correct)).unwrap_or(u32::MAX),
        accuracy: correct as f64 / total as f64,
    })
}

fn edit_tokens_v1(
    observation: &ScoreGridObservationV1,
    transformed: Option<&[Option<TransformedNoteV1>]>,
) -> Vec<EditTokenV1> {
    if let Some(transformed) = transformed {
        let mut tokens: Vec<_> = transformed
            .iter()
            .enumerate()
            .map(|(index, note)| match note {
                Some(note) => EditTokenV1 {
                    bar_index: note.bar_index,
                    onset_32nd: note.onset_32nd,
                    pitch_midi: note.pitch_midi,
                    duration_32nd: u32::from(note.duration_32nd),
                },
                None => EditTokenV1 {
                    bar_index: i32::MAX,
                    onset_32nd: u16::MAX,
                    pitch_midi: u8::MAX,
                    duration_32nd: u32::MAX.saturating_sub(index.min(u32::MAX as usize) as u32),
                },
            })
            .collect();
        tokens.sort_unstable();
        tokens
    } else {
        let mut tokens: Vec<_> = observation
            .notes
            .iter()
            .map(|note| EditTokenV1 {
                bar_index: note.bar_index,
                onset_32nd: note.onset_32nd,
                pitch_midi: note.pitch_midi,
                duration_32nd: note.duration_32nd,
            })
            .collect();
        tokens.sort_unstable();
        tokens
    }
}

fn levenshtein_v1(reference: &[EditTokenV1], prediction: &[EditTokenV1]) -> u32 {
    let mut previous: Vec<usize> = (0..=reference.len()).collect();
    let mut current = vec![0usize; reference.len() + 1];
    for (prediction_index, prediction_token) in prediction.iter().enumerate() {
        current[0] = prediction_index + 1;
        for (reference_index, reference_token) in reference.iter().enumerate() {
            let substitution = previous[reference_index]
                + if reference_token != prediction_token { 1 } else { 0 };
            let insertion = previous[reference_index + 1] + 1;
            let deletion = current[reference_index] + 1;
            current[reference_index + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    u32::try_from(previous[reference.len()]).unwrap_or(u32::MAX)
}

fn validate_report_binding_v1(
    reference: &ScoreGridObservationV1,
    prediction: &ScoreGridObservationV1,
    report: &ScoreGridEvaluationReportV1,
) -> bool {
    report.schema == SCORE_GRID_EVALUATION_SCHEMA_V1
        && report.algorithm == SCORE_GRID_EVALUATION_ALGORITHM_V1
        && report.reference_sha256 == sha256_hex(&serde_json::to_vec(reference).unwrap_or_default())
        && report.prediction_sha256 == sha256_hex(&serde_json::to_vec(prediction).unwrap_or_default())
        && score_grid_evaluation_commitment_v1(report)
            .is_ok_and(|digest| digest == report.report_sha256)
}

pub fn score_grid_evaluation_commitment_v1(
    report: &ScoreGridEvaluationReportV1,
) -> Result<String, serde_json::Error> {
    let mut commitment = report.clone();
    commitment.report_sha256.clear();
    Ok(domain_separated_sha256(
        SCORE_GRID_EVALUATION_SCHEMA_V1,
        &serde_json::to_vec(&commitment)?,
    ))
}

/// Evaluate normalized notation observations. The score-space alignment
/// searches the documented scale/bar/pickup transform family and chooses the
/// deterministic first transform with maximal score-note F1.
pub fn evaluate_score_grid_v1(
    reference: &ScoreGridObservationV1,
    prediction: &ScoreGridObservationV1,
) -> Result<ScoreGridEvaluationReportV1, ScoreGridEvaluationIssueV1> {
    validate_score_grid_observation_v1(reference)?;
    validate_score_grid_observation_v1(prediction)?;

    let scans = prediction
        .notes
        .len()
        .saturating_mul(ALIGNMENT_SCALES_V1.len())
        .saturating_mul(7)
        .saturating_mul(25);
    if scans > SCORE_GRID_MAX_ALIGNMENT_NOTE_SCANS_V1 {
        return Err(ScoreGridEvaluationIssueV1::AlignmentWorkBudgetExceeded {
            scans,
            maximum: SCORE_GRID_MAX_ALIGNMENT_NOTE_SCANS_V1,
        });
    }

    let mut best_alignment = ScoreGridAlignmentV1 {
        scale_numerator: 1,
        scale_denominator: 1,
        delta_bars: 0,
        delta_onset_32nd: 0,
    };
    let mut best_score = metric_from_counts_v1(reference.notes.len(), prediction.notes.len(), 0);
    let mut best_rhythm = best_score.clone();
    let mut best_part = None;
    let mut best_transformed = transform_prediction_v1(prediction, reference, best_alignment);
    let mut best_true_positives = 0usize;

    for alignment in alignment_candidates_v1() {
        let (score, rhythm, part, transformed) =
            score_metric_for_alignment_v1(reference, prediction, alignment);
        if score.true_positives as usize > best_true_positives {
            best_true_positives = score.true_positives as usize;
            best_alignment = alignment;
            best_score = score;
            best_rhythm = rhythm;
            best_part = part;
            best_transformed = transformed;
        }
    }

    let reference_tokens = edit_tokens_v1(reference, None);
    let prediction_tokens = edit_tokens_v1(prediction, Some(&best_transformed));
    let edit_cells = reference_tokens.len().saturating_mul(prediction_tokens.len());
    let edit_distance_status = if reference_tokens.is_empty() {
        ScoreGridEditDistanceStatusV1::EmptyReference
    } else if edit_cells > SCORE_GRID_MAX_EDIT_DISTANCE_CELLS_V1 {
        ScoreGridEditDistanceStatusV1::WorkBudgetExceeded
    } else {
        ScoreGridEditDistanceStatusV1::Available
    };
    let (score_error_distance, score_error_rate) =
        if edit_distance_status == ScoreGridEditDistanceStatusV1::Available {
            let distance = levenshtein_v1(&reference_tokens, &prediction_tokens);
            (
                Some(distance),
                Some(distance as f64 / reference_tokens.len() as f64),
            )
        } else {
            (None, None)
        };

    let (barline_f1, barline_tolerance_seconds) =
        barline_metrics_v1(reference, prediction)
            .map(|(metric, tolerance)| (Some(metric), Some(tolerance)))
            .unwrap_or((None, None));

    let mut report = ScoreGridEvaluationReportV1 {
        schema: SCORE_GRID_EVALUATION_SCHEMA_V1.into(),
        algorithm: SCORE_GRID_EVALUATION_ALGORITHM_V1.into(),
        reference_sha256: sha256_hex(
            &serde_json::to_vec(reference)
                .map_err(|_| ScoreGridEvaluationIssueV1::SerializationFailed)?,
        ),
        prediction_sha256: sha256_hex(
            &serde_json::to_vec(prediction)
                .map_err(|_| ScoreGridEvaluationIssueV1::SerializationFailed)?,
        ),
        alignment: best_alignment,
        score_f1: best_score,
        rhythm_f1: best_rhythm,
        part_aware_score_f1: best_part,
        score_error_rate,
        score_error_distance,
        score_error_distance_cells: u64::try_from(edit_cells).unwrap_or(u64::MAX),
        score_error_distance_status: edit_distance_status,
        barline_f1,
        barline_tolerance_seconds,
        meter_accuracy: meter_accuracy_v1(reference, prediction),
        report_sha256: String::new(),
    };
    report.report_sha256 = score_grid_evaluation_commitment_v1(&report)
        .map_err(|_| ScoreGridEvaluationIssueV1::SerializationFailed)?;
    Ok(report)
}

pub fn validate_score_grid_report_v1(
    reference: &ScoreGridObservationV1,
    prediction: &ScoreGridObservationV1,
    report: &ScoreGridEvaluationReportV1,
) -> bool {
    validate_report_binding_v1(reference, prediction, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(
        bar_index: i32,
        onset_32nd: u16,
        duration_32nd: u16,
        pitch_midi: u8,
        part_id: Option<u16>,
    ) -> ScoreGridNoteV1 {
        ScoreGridNoteV1 {
            bar_index,
            onset_32nd,
            duration_32nd,
            pitch_midi,
            part_id,
        }
    }

    fn meter(bar_index: u32, numerator: u8, denominator: u8) -> ScoreGridMeterMarkV1 {
        ScoreGridMeterMarkV1 {
            bar_index,
            numerator,
            denominator,
        }
    }

    fn observation(notes: Vec<ScoreGridNoteV1>) -> ScoreGridObservationV1 {
        ScoreGridObservationV1 {
            bar_count: 4,
            notes,
            meter_marks: vec![meter(0, 4, 4)],
            barline_times_seconds: vec![0.0, 2.0, 4.0, 6.0],
            beat_times_seconds: (0..=16).map(|index| f64::from(index) * 0.5).collect(),
        }
    }

    #[test]
    fn exact_notation_has_perfect_score_rhythm_and_part_metrics() {
        let notes = vec![
            note(0, 0, 8, 60, Some(1)),
            note(0, 16, 8, 64, Some(1)),
            note(1, 0, 16, 67, Some(2)),
        ];
        let reference = observation(notes.clone());
        let prediction = observation(notes);
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(report.score_f1.f1, 1.0);
        assert_eq!(report.rhythm_f1.f1, 1.0);
        assert_eq!(report.part_aware_score_f1.as_ref().unwrap().f1, 1.0);
        assert_eq!(report.score_error_rate, Some(0.0));
        assert_eq!(report.alignment.scale_numerator, 1);
        assert_eq!(report.alignment.scale_denominator, 1);
    }

    #[test]
    fn wrong_measure_is_not_rescued_by_audio_time_note_f1() {
        let reference = observation(vec![
            note(0, 0, 8, 60, None),
            note(1, 0, 8, 62, None),
        ]);
        let prediction = observation(vec![
            note(0, 0, 8, 60, None),
            note(2, 0, 8, 62, None),
        ]);
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(report.score_f1.f1, 0.5);
    }

    #[test]
    fn global_measure_shift_is_searched_and_reported() {
        let reference = observation(vec![note(0, 0, 8, 60, None), note(1, 0, 8, 62, None)]);
        let prediction = observation(vec![note(2, 0, 8, 60, None), note(3, 0, 8, 62, None)]);
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(report.score_f1.f1, 1.0);
        assert_eq!(report.alignment.delta_bars, -2);
    }

    #[test]
    fn half_tempo_score_reading_is_searchable() {
        let reference = observation(vec![note(0, 8, 8, 60, None), note(0, 16, 8, 62, None)]);
        let prediction = observation(vec![note(0, 16, 16, 60, None), note(0, 32, 16, 62, None)]);
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(report.score_f1.f1, 1.0);
        assert_eq!(report.alignment.scale_numerator, 1);
        assert_eq!(report.alignment.scale_denominator, 2);
    }

    #[test]
    fn pickup_offset_on_quarter_note_fraction_is_searchable() {
        let reference = observation(vec![note(0, 8, 8, 60, None)]);
        let prediction = observation(vec![note(0, 10, 8, 60, None)]);
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(report.score_f1.f1, 1.0);
        assert_eq!(report.alignment.delta_onset_32nd, -2);
    }

    #[test]
    fn rhythm_f1_ignores_pitch_but_checks_measure_position_and_duration() {
        let reference = observation(vec![note(0, 0, 8, 60, None)]);
        let prediction = observation(vec![note(0, 0, 8, 67, None)]);
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(report.score_f1.f1, 0.0);
        assert_eq!(report.rhythm_f1.f1, 1.0);
    }

    #[test]
    fn part_aware_metric_is_withheld_when_part_identity_is_missing() {
        let reference = observation(vec![note(0, 0, 8, 60, None)]);
        let prediction = observation(vec![note(0, 0, 8, 60, Some(1))]);
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert!(report.part_aware_score_f1.is_none());
    }

    #[test]
    fn part_aware_score_detects_part_swaps() {
        let reference = observation(vec![
            note(0, 0, 8, 60, Some(1)),
            note(0, 8, 8, 62, Some(2)),
        ]);
        let prediction = observation(vec![
            note(0, 0, 8, 60, Some(2)),
            note(0, 8, 8, 62, Some(1)),
        ]);
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(report.score_f1.f1, 1.0);
        assert_eq!(report.part_aware_score_f1.as_ref().unwrap().f1, 0.0);
    }

    #[test]
    fn barline_f1_uses_seventeen_point_five_percent_of_median_beat() {
        let mut reference = observation(Vec::new());
        let mut prediction = observation(Vec::new());
        reference.barline_times_seconds = vec![0.0, 2.0, 4.0];
        prediction.barline_times_seconds = vec![0.05, 1.90, 4.2];
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(report.barline_tolerance_seconds, Some(0.0875));
        assert_eq!(report.barline_f1.as_ref().unwrap().true_positives, 2);
    }

    #[test]
    fn barline_matching_is_one_to_one() {
        let mut reference = observation(Vec::new());
        let mut prediction = observation(Vec::new());
        reference.barline_times_seconds = vec![0.0];
        prediction.barline_times_seconds = vec![0.01, 0.02];
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(report.barline_f1.as_ref().unwrap().true_positives, 1);
        assert_eq!(report.barline_f1.as_ref().unwrap().false_positives, 1);
    }

    #[test]
    fn meter_accuracy_is_exact_and_measured_per_bar() {
        let reference = observation(Vec::new());
        let mut prediction = observation(Vec::new());
        prediction.meter_marks = vec![meter(0, 4, 4), meter(2, 3, 4)];
        let mut reference = reference;
        reference.meter_marks = vec![meter(0, 4, 4), meter(2, 3, 4)];
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(report.meter_accuracy.as_ref().unwrap().accuracy, 1.0);

        prediction.meter_marks = vec![meter(0, 4, 4), meter(2, 6, 8)];
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(report.meter_accuracy.as_ref().unwrap().accuracy, 0.5);
    }

    #[test]
    fn changing_meter_changes_the_valid_measure_grid() {
        let mut reference = observation(vec![note(2, 22, 2, 60, None)]);
        reference.meter_marks = vec![meter(0, 4, 4), meter(2, 6, 8)];
        assert!(validate_score_grid_observation_v1(&reference).is_ok());
        reference.notes[0].onset_32nd = 24;
        assert_eq!(
            validate_score_grid_observation_v1(&reference),
            Err(ScoreGridEvaluationIssueV1::InvalidNote { ordinal: 0 })
        );
    }

    #[test]
    fn malformed_meter_map_is_rejected() {
        let mut reference = observation(Vec::new());
        reference.meter_marks = vec![meter(1, 4, 4)];
        assert_eq!(
            validate_score_grid_observation_v1(&reference),
            Err(ScoreGridEvaluationIssueV1::InvalidMeterMap)
        );
    }

    #[test]
    fn edit_distance_reports_insertions_and_deletions() {
        let reference = observation(vec![
            note(0, 0, 8, 60, None),
            note(0, 8, 8, 62, None),
        ]);
        let prediction = observation(vec![note(0, 0, 8, 60, None)]);
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(report.score_error_distance, Some(1));
        assert_eq!(report.score_error_rate, Some(0.5));
    }

    #[test]
    fn too_many_score_notes_fail_closed_before_alignment() {
        let notes = vec![note(0, 0, 8, 60, None); SCORE_GRID_MAX_NOTES_V1 + 1];
        let observation = observation(notes);
        assert!(matches!(
            validate_score_grid_observation_v1(&observation),
            Err(ScoreGridEvaluationIssueV1::TooManyNotes { .. })
        ));
    }

    #[test]
    fn empty_reference_marks_edit_distance_undefined() {
        let reference = observation(Vec::new());
        let prediction = observation(vec![note(0, 0, 8, 60, None)]);
        let report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        assert_eq!(
            report.score_error_distance_status,
            ScoreGridEditDistanceStatusV1::EmptyReference
        );
        assert!(report.score_error_rate.is_none());
    }

    #[test]
    fn report_commitment_detects_metric_tampering() {
        let reference = observation(vec![note(0, 0, 8, 60, None)]);
        let prediction = reference.clone();
        let mut report = evaluate_score_grid_v1(&reference, &prediction).unwrap();
        let original = score_grid_evaluation_commitment_v1(&report).unwrap();
        report.score_f1.f1 = 0.0;
        let tampered = score_grid_evaluation_commitment_v1(&report).unwrap();
        assert_ne!(original, tampered);
        assert!(!validate_score_grid_report_v1(&reference, &prediction, &report));
    }
}
