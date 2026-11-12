// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Model-agnostic beat/downbeat evaluation for Melothaea.
//!
//! The evaluator consumes timestamp sequences from any beat tracker and one
//! or more independently annotated reference sequences. It does not perform
//! beat tracking, audio decoding, or annotation consensus. Multiple reference
//! tracks remain separately visible so disagreement is not erased by choosing
//! one annotator as unquestioned truth.
//!
//! Strict beat F1 uses the mir_eval default 70 ms event tolerance. Cemgil
//! scoring and the five conventional metrical-level variants are additional
//! diagnostics; the best metrical-level score must never replace strict F1.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const BEAT_GRID_EVALUATION_SCHEMA_V1: &str = "muse-beat-grid-evaluation-v1";
pub const BEAT_GRID_EVALUATION_ALGORITHM_V1: &str =
    "sorted-one-to-one-event-match-cemgil-continuity-goto-p-score-information-gain-v1";
pub const BEAT_GRID_DEFAULT_TOLERANCE_MS_V1: u32 = 70;
pub const BEAT_GRID_DEFAULT_DOWNBEAT_TOLERANCE_MS_V1: u32 = 70;
pub const BEAT_GRID_DEFAULT_CEMGIL_SIGMA_MS_V1: u32 = 40;
pub const BEAT_GRID_MAX_REFERENCES_V1: usize = 64;
pub const BEAT_GRID_MAX_EVENTS_PER_SEQUENCE_V1: usize = 2_000_000;
pub const BEAT_GRID_MAX_TOTAL_EVENTS_V1: usize = 8_000_000;
pub const BEAT_GRID_MAX_DURATION_SECONDS_V1: f64 = 30_000.0;
pub const BEAT_GRID_DEFAULT_GOTO_THRESHOLD_BPS_V1: u16 = 3_500;
pub const BEAT_GRID_DEFAULT_GOTO_MU_BPS_V1: u16 = 2_000;
pub const BEAT_GRID_DEFAULT_GOTO_SIGMA_BPS_V1: u16 = 2_000;
pub const BEAT_GRID_DEFAULT_P_SCORE_THRESHOLD_BPS_V1: u16 = 2_000;
pub const BEAT_GRID_DEFAULT_INFORMATION_GAIN_BINS_V1: u16 = 41;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatGridReferenceV1 {
    pub reference_id: String,
    /// One coherent beat interpretation from one annotator or annotation source.
    pub beat_times_seconds: Vec<f64>,
    /// Empty means downbeat ground truth is unavailable, not that no downbeats exist.
    pub downbeat_times_seconds: Vec<f64>,
    pub annotation_source: String,
    pub annotation_group: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatGridPredictionV1 {
    pub model_id: String,
    pub model_revision: String,
    pub beat_times_seconds: Vec<f64>,
    pub downbeat_times_seconds: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BeatGridEvaluationConfigV1 {
    pub beat_tolerance_ms: u32,
    pub downbeat_tolerance_ms: u32,
    pub cemgil_sigma_ms: u32,
    /// Optional explicit trim applied equally to reference and prediction.
    /// None is the default: retain the complete annotated clip.
    pub minimum_evaluation_time_ms: Option<u32>,
    pub metric_level_variants_enabled: bool,
    pub continuity_phase_threshold_basis_points: u16,
    pub continuity_period_threshold_basis_points: u16,
    pub goto_threshold_basis_points: u16,
    pub goto_mu_basis_points: u16,
    pub goto_sigma_basis_points: u16,
    pub p_score_threshold_basis_points: u16,
    pub information_gain_bins: u16,
}

impl Default for BeatGridEvaluationConfigV1 {
    fn default() -> Self {
        Self {
            beat_tolerance_ms: BEAT_GRID_DEFAULT_TOLERANCE_MS_V1,
            downbeat_tolerance_ms: BEAT_GRID_DEFAULT_DOWNBEAT_TOLERANCE_MS_V1,
            cemgil_sigma_ms: BEAT_GRID_DEFAULT_CEMGIL_SIGMA_MS_V1,
            minimum_evaluation_time_ms: None,
            metric_level_variants_enabled: true,
            continuity_phase_threshold_basis_points: 1_750,
            continuity_period_threshold_basis_points: 1_750,
            goto_threshold_basis_points: BEAT_GRID_DEFAULT_GOTO_THRESHOLD_BPS_V1,
            goto_mu_basis_points: BEAT_GRID_DEFAULT_GOTO_MU_BPS_V1,
            goto_sigma_basis_points: BEAT_GRID_DEFAULT_GOTO_SIGMA_BPS_V1,
            p_score_threshold_basis_points: BEAT_GRID_DEFAULT_P_SCORE_THRESHOLD_BPS_V1,
            information_gain_bins: BEAT_GRID_DEFAULT_INFORMATION_GAIN_BINS_V1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatGridEventMetricV1 {
    pub true_positives: u32,
    pub false_positives: u32,
    pub false_negatives: u32,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatGridTimingErrorSummaryV1 {
    pub matched_event_count: u32,
    /// Signed prediction-minus-reference bias in milliseconds.
    pub mean_signed_error_ms: Option<f64>,
    pub mean_absolute_error_ms: Option<f64>,
    pub median_absolute_error_ms: Option<f64>,
    pub p95_absolute_error_ms: Option<f64>,
    pub p99_absolute_error_ms: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatGridIntervalErrorSummaryV1 {
    pub adjacent_interval_pair_count: u32,
    pub mean_absolute_error_ms: Option<f64>,
    pub median_absolute_error_ms: Option<f64>,
    pub p95_absolute_error_ms: Option<f64>,
    pub mean_absolute_relative_error_percent: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatGridContinuityMetricsV1 {
    pub correct_metric_level_continuous: f64,
    pub correct_metric_level_total: f64,
    pub any_metric_level_continuous: f64,
    pub any_metric_level_total: f64,
    pub evaluated_reference_variants: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatGridReferenceScoreV1 {
    pub reference_id: String,
    pub annotation_source: String,
    pub annotation_group: Option<String>,
    pub reference_beat_count: u32,
    pub predicted_beat_count: u32,
    /// Strict beat event F1 at the configured tolerance (70 ms by default).
    pub beat_f_measure: BeatGridEventMetricV1,
    /// Direct-reference Cemgil score at the configured sigma.
    pub cemgil_score: Option<f64>,
    /// Maximum Cemgil score over reference, offbeat, double, and half-time variants.
    /// This is a diagnostic for metrical-level ambiguity, not strict beat accuracy.
    pub cemgil_best_metric_level: Option<f64>,
    /// Goto's binary long-continuous-track correctness measure (0 or 1).
    pub goto_score: f64,
    /// McKinney P-score; absent only if 10 ms quantization collapses all
    /// reference intervals and the documented median period is undefined.
    pub p_score: Option<f64>,
    /// mir_eval beat-error information gain, normalized to [0, 1].
    pub information_gain: f64,
    /// mir_eval continuity: correct metric level continuous / total, and any
    /// metrical level continuous / total. These remain separate from strict F1.
    pub continuity_metrics: Option<BeatGridContinuityMetricsV1>,
    pub downbeat_metric: Option<BeatGridEventMetricV1>,
    pub beat_timing_error: BeatGridTimingErrorSummaryV1,
    pub downbeat_timing_error: Option<BeatGridTimingErrorSummaryV1>,
    /// Interval error uses only adjacent one-to-one beat matches, preventing
    /// missing/extra events from being silently interpreted as tempo drift.
    pub inter_beat_interval_error: BeatGridIntervalErrorSummaryV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatGridEvaluationReportV1 {
    pub schema: String,
    pub algorithm: String,
    pub config: BeatGridEvaluationConfigV1,
    pub prediction_sha256: String,
    pub reference_bundle_sha256: String,
    pub prediction_model_id: String,
    pub prediction_model_revision: String,
    pub reference_count: u32,
    pub reference_scores: Vec<BeatGridReferenceScoreV1>,
    /// Median strict beat F1 across reference interpretations.
    pub median_beat_f1_across_references: f64,
    /// Worst strict beat F1 across reference interpretations.
    pub minimum_beat_f1_across_references: f64,
    /// Median downbeat F1 over references that actually provide downbeat labels.
    pub median_downbeat_f1_across_references: Option<f64>,
    /// Pairwise beat F1 between reference tracks; measures annotation disagreement.
    pub median_pairwise_reference_agreement: Option<f64>,
    pub minimum_pairwise_reference_agreement: Option<f64>,
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeatGridEvaluationIssueV1 {
    InvalidConfig { field: String },
    NoReferences,
    EmptyReferenceBeats { index: usize },
    TooManyReferences { count: usize, maximum: usize },
    InvalidReferenceId { index: usize },
    InvalidAnnotationSource { index: usize },
    InvalidTimestampSequence { field: String, index: usize },
    TooManyEvents { count: usize, maximum: usize },
    TotalEventBudgetExceeded { count: usize, maximum: usize },
    InvalidModelIdentity,
    SerializationFailed,
    ReportInputMismatch,
}

#[derive(Serialize)]
struct EvaluationCommitmentV1<'a> {
    schema: &'a str,
    algorithm: &'a str,
    config: &'a BeatGridEvaluationConfigV1,
    prediction_sha256: &'a str,
    reference_bundle_sha256: &'a str,
    prediction_model_id: &'a str,
    prediction_model_revision: &'a str,
    reference_count: u32,
    reference_scores: &'a [BeatGridReferenceScoreV1],
    median_beat_f1_across_references: f64,
    minimum_beat_f1_across_references: f64,
    median_downbeat_f1_across_references: Option<f64>,
    median_pairwise_reference_agreement: Option<f64>,
    minimum_pairwise_reference_agreement: Option<f64>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn domain_separated_sha256(schema: &str, bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(schema.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn validate_config(
    config: &BeatGridEvaluationConfigV1,
) -> Result<(), BeatGridEvaluationIssueV1> {
    if config.beat_tolerance_ms == 0 || config.beat_tolerance_ms > 5_000 {
        return Err(BeatGridEvaluationIssueV1::InvalidConfig {
            field: "beat_tolerance_ms".into(),
        });
    }
    if config.downbeat_tolerance_ms == 0 || config.downbeat_tolerance_ms > 5_000 {
        return Err(BeatGridEvaluationIssueV1::InvalidConfig {
            field: "downbeat_tolerance_ms".into(),
        });
    }
    if config.cemgil_sigma_ms == 0 || config.cemgil_sigma_ms > 1_000 {
        return Err(BeatGridEvaluationIssueV1::InvalidConfig {
            field: "cemgil_sigma_ms".into(),
        });
    }
    if !(1..=5_000).contains(&config.continuity_phase_threshold_basis_points) {
        return Err(BeatGridEvaluationIssueV1::InvalidConfig {
            field: "continuity_phase_threshold_basis_points".into(),
        });
    }
    if !(1..=5_000).contains(&config.continuity_period_threshold_basis_points) {
        return Err(BeatGridEvaluationIssueV1::InvalidConfig {
            field: "continuity_period_threshold_basis_points".into(),
        });
    }
    if config.goto_threshold_basis_points >= 10_000
        || config.goto_mu_basis_points == 0
        || config.goto_mu_basis_points > 5_000
        || config.goto_sigma_basis_points == 0
        || config.goto_sigma_basis_points > 5_000
    {
        return Err(BeatGridEvaluationIssueV1::InvalidConfig {
            field: "goto_parameters".into(),
        });
    }
    if config.p_score_threshold_basis_points == 0
        || config.p_score_threshold_basis_points > 5_000
    {
        return Err(BeatGridEvaluationIssueV1::InvalidConfig {
            field: "p_score_threshold_basis_points".into(),
        });
    }
    if !(2..=257).contains(&config.information_gain_bins) {
        return Err(BeatGridEvaluationIssueV1::InvalidConfig {
            field: "information_gain_bins".into(),
        });
    }
    if config.minimum_evaluation_time_ms.is_some_and(|value| value > 30_000_000) {
        return Err(BeatGridEvaluationIssueV1::InvalidConfig {
            field: "minimum_evaluation_time_ms".into(),
        });
    }
    Ok(())
}

fn validate_times(
    values: &[f64],
    field: &str,
    allow_empty: bool,
) -> Result<(), BeatGridEvaluationIssueV1> {
    if !allow_empty && values.is_empty() {
        return Err(BeatGridEvaluationIssueV1::EmptyReferenceBeats {
            index: 0,
        });
    }
    if values.len() > BEAT_GRID_MAX_EVENTS_PER_SEQUENCE_V1 {
        return Err(BeatGridEvaluationIssueV1::TooManyEvents {
            count: values.len(),
            maximum: BEAT_GRID_MAX_EVENTS_PER_SEQUENCE_V1,
        });
    }
    let mut previous = None;
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite()
            || *value < 0.0
            || *value > BEAT_GRID_MAX_DURATION_SECONDS_V1
            || previous.is_some_and(|prior| *value <= prior)
        {
            return Err(BeatGridEvaluationIssueV1::InvalidTimestampSequence {
                field: field.into(),
                index,
            });
        }
        previous = Some(*value);
    }
    Ok(())
}

fn validate_inputs(
    prediction: &BeatGridPredictionV1,
    references: &[BeatGridReferenceV1],
    config: &BeatGridEvaluationConfigV1,
) -> Result<(), BeatGridEvaluationIssueV1> {
    validate_config(config)?;
    if references.is_empty() {
        return Err(BeatGridEvaluationIssueV1::NoReferences);
    }
    if references.len() > BEAT_GRID_MAX_REFERENCES_V1 {
        return Err(BeatGridEvaluationIssueV1::TooManyReferences {
            count: references.len(),
            maximum: BEAT_GRID_MAX_REFERENCES_V1,
        });
    }
    if prediction.model_id.trim().is_empty()
        || prediction.model_id.len() > 256
        || prediction.model_revision.trim().is_empty()
        || prediction.model_revision.len() > 256
    {
        return Err(BeatGridEvaluationIssueV1::InvalidModelIdentity);
    }
    validate_times(
        &prediction.beat_times_seconds,
        "prediction.beat_times_seconds",
        true,
    )?;
    validate_times(
        &prediction.downbeat_times_seconds,
        "prediction.downbeat_times_seconds",
        true,
    )?;

    let mut total_events = prediction.beat_times_seconds.len()
        + prediction.downbeat_times_seconds.len();
    for (index, reference) in references.iter().enumerate() {
        if reference.reference_id.trim().is_empty()
            || reference.reference_id.len() > 256
            || references[..index].iter().any(|prior| prior.reference_id == reference.reference_id)
        {
            return Err(BeatGridEvaluationIssueV1::InvalidReferenceId { index });
        }
        if reference.annotation_source.trim().is_empty()
            || reference.annotation_source.len() > 256
        {
            return Err(BeatGridEvaluationIssueV1::InvalidAnnotationSource { index });
        }
        if reference.annotation_group.as_ref().is_some_and(|value| {
            value.trim().is_empty() || value.len() > 256
        }) {
            return Err(BeatGridEvaluationIssueV1::InvalidAnnotationSource { index });
        }
        validate_times(
            &reference.beat_times_seconds,
            "reference.beat_times_seconds",
            true,
        )
        .map_err(|issue| match issue {
            BeatGridEvaluationIssueV1::EmptyReferenceBeats { .. } => {
                BeatGridEvaluationIssueV1::EmptyReferenceBeats { index }
            }
            other => other,
        })?;
        if reference.beat_times_seconds.is_empty() {
            return Err(BeatGridEvaluationIssueV1::EmptyReferenceBeats { index });
        }
        validate_times(
            &reference.downbeat_times_seconds,
            "reference.downbeat_times_seconds",
            true,
        )?;
        total_events = total_events
            .saturating_add(reference.beat_times_seconds.len())
            .saturating_add(reference.downbeat_times_seconds.len());
    }
    if total_events > BEAT_GRID_MAX_TOTAL_EVENTS_V1 {
        return Err(BeatGridEvaluationIssueV1::TotalEventBudgetExceeded {
            count: total_events,
            maximum: BEAT_GRID_MAX_TOTAL_EVENTS_V1,
        });
    }
    Ok(())
}

fn trimmed(values: &[f64], minimum_seconds: Option<f64>) -> Vec<f64> {
    match minimum_seconds {
        Some(minimum) => {
            let start = values.partition_point(|value| *value < minimum);
            values[start..].to_vec()
        }
        None => values.to_vec(),
    }
}

/// Sorted greedy matching is maximum-cardinality for one-dimensional points
/// under a fixed symmetric tolerance. It also guarantees one-to-one matches.
fn match_events(
    reference: &[f64],
    estimated: &[f64],
    tolerance_seconds: f64,
) -> Vec<(usize, usize)> {
    let mut matching = Vec::new();
    let mut reference_index = 0usize;
    let mut estimated_index = 0usize;

    while reference_index < reference.len() && estimated_index < estimated.len() {
        let error = estimated[estimated_index] - reference[reference_index];
        if error.abs() <= tolerance_seconds {
            matching.push((reference_index, estimated_index));
            reference_index += 1;
            estimated_index += 1;
        } else if error < -tolerance_seconds {
            estimated_index += 1;
        } else {
            reference_index += 1;
        }
    }
    matching
}

fn event_metric(
    reference_count: usize,
    estimated_count: usize,
    true_positives: usize,
) -> BeatGridEventMetricV1 {
    let false_positives = estimated_count.saturating_sub(true_positives);
    let false_negatives = reference_count.saturating_sub(true_positives);
    let precision = if estimated_count == 0 {
        0.0
    } else {
        true_positives as f64 / estimated_count as f64
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
    BeatGridEventMetricV1 {
        true_positives: u32::try_from(true_positives).unwrap_or(u32::MAX),
        false_positives: u32::try_from(false_positives).unwrap_or(u32::MAX),
        false_negatives: u32::try_from(false_negatives).unwrap_or(u32::MAX),
        precision,
        recall,
        f1,
    }
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(if values.len().is_multiple_of(2) {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    })
}

fn percentile(sorted: &[f64], fraction: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[index])
}

fn timing_error_summary(
    reference: &[f64],
    estimated: &[f64],
    matching: &[(usize, usize)],
) -> BeatGridTimingErrorSummaryV1 {
    let errors_ms: Vec<f64> = matching
        .iter()
        .map(|(reference_index, estimated_index)| {
            (estimated[*estimated_index] - reference[*reference_index]) * 1_000.0
        })
        .collect();
    let mut absolute: Vec<f64> = errors_ms.iter().map(|value| value.abs()).collect();
    let mean_signed = if errors_ms.is_empty() {
        None
    } else {
        Some(errors_ms.iter().sum::<f64>() / errors_ms.len() as f64)
    };
    let mean_absolute = if absolute.is_empty() {
        None
    } else {
        Some(absolute.iter().sum::<f64>() / absolute.len() as f64)
    };
    let median_absolute = median(&mut absolute);
    BeatGridTimingErrorSummaryV1 {
        matched_event_count: u32::try_from(errors_ms.len()).unwrap_or(u32::MAX),
        mean_signed_error_ms: mean_signed,
        mean_absolute_error_ms: mean_absolute,
        median_absolute_error_ms: median_absolute,
        p95_absolute_error_ms: percentile(&absolute, 0.95),
        p99_absolute_error_ms: percentile(&absolute, 0.99),
    }
}

fn interval_error_summary(
    reference: &[f64],
    estimated: &[f64],
    matching: &[(usize, usize)],
) -> BeatGridIntervalErrorSummaryV1 {
    let mut errors_ms = Vec::new();
    let mut relative_errors_percent = Vec::new();
    for pair in matching.windows(2) {
        let (reference_left, estimated_left) = pair[0];
        let (reference_right, estimated_right) = pair[1];
        if reference_right != reference_left + 1 || estimated_right != estimated_left + 1 {
            continue;
        }
        let reference_interval = reference[reference_right] - reference[reference_left];
        let estimated_interval = estimated[estimated_right] - estimated[estimated_left];
        let absolute_ms = (estimated_interval - reference_interval).abs() * 1_000.0;
        errors_ms.push(absolute_ms);
        if reference_interval > 0.0 {
            relative_errors_percent.push(
                (estimated_interval - reference_interval).abs() / reference_interval * 100.0,
            );
        }
    }
    let mut sorted_errors = errors_ms.clone();
    let median_error = median(&mut sorted_errors);
    BeatGridIntervalErrorSummaryV1 {
        adjacent_interval_pair_count: u32::try_from(errors_ms.len()).unwrap_or(u32::MAX),
        mean_absolute_error_ms: (!errors_ms.is_empty())
            .then_some(errors_ms.iter().sum::<f64>() / errors_ms.len() as f64),
        median_absolute_error_ms: median_error,
        p95_absolute_error_ms: percentile(&sorted_errors, 0.95),
        mean_absolute_relative_error_percent: (!relative_errors_percent.is_empty())
            .then_some(
                relative_errors_percent.iter().sum::<f64>()
                    / relative_errors_percent.len() as f64,
            ),
    }
}

fn cemgil_score(
    reference: &[f64],
    estimated: &[f64],
    sigma_seconds: f64,
) -> Option<f64> {
    if reference.is_empty() || estimated.is_empty() {
        return Some(0.0);
    }
    let gaussian_sum = reference
        .iter()
        .map(|reference_time| {
            let nearest_error = estimated
                .iter()
                .map(|estimated_time| (reference_time - estimated_time).abs())
                .fold(f64::INFINITY, f64::min);
            (-nearest_error.powi(2) / (2.0 * sigma_seconds.powi(2))).exp()
        })
        .sum::<f64>();
    Some(gaussian_sum / (0.5 * (estimated.len() + reference.len()) as f64))
}

fn metric_level_variants(reference: &[f64]) -> [Vec<f64>; 5] {
    if reference.len() < 2 {
        return [
            reference.to_vec(),
            Vec::new(),
            reference.to_vec(),
            reference.to_vec(),
            reference.to_vec(),
        ];
    }
    let mut offbeat = Vec::with_capacity(reference.len() - 1);
    let mut doubled = Vec::with_capacity(reference.len() * 2 - 1);
    for pair in reference.windows(2) {
        doubled.push(pair[0]);
        let midpoint = (pair[0] + pair[1]) * 0.5;
        offbeat.push(midpoint);
        doubled.push(midpoint);
    }
    doubled.push(*reference.last().unwrap_or(&reference[0]));
    [
        reference.to_vec(),
        offbeat,
        doubled,
        reference.iter().step_by(2).copied().collect(),
        reference.iter().skip(1).step_by(2).copied().collect(),
    ]
}

fn continuity_for_reference_variant(
    reference: &[f64],
    estimated: &[f64],
    phase_threshold: f64,
    period_threshold: f64,
) -> (f64, f64) {
    if estimated.len() <= 1 || reference.len() <= 1 {
        return (0.0, 0.0);
    }

    let annotation_count = reference.len().max(estimated.len());
    let mut used_annotations = vec![false; annotation_count];
    let mut beat_successes = vec![false; annotation_count];

    for estimated_index in 0..estimated.len() {
        let estimated_time = estimated[estimated_index];
        // Match numpy.argmin: select the first reference index on ties.
        let mut nearest_index = 0usize;
        let mut minimum_difference = (estimated_time - reference[0]).abs();
        for (reference_index, reference_time) in reference.iter().enumerate().skip(1) {
            let difference = (estimated_time - reference_time).abs();
            if difference < minimum_difference {
                minimum_difference = difference;
                nearest_index = reference_index;
            }
        }

        if used_annotations[nearest_index] {
            continue;
        }

        let (reference_interval, estimated_interval) =
            if estimated_index == 0 || nearest_index == 0 {
                let reference_interval = if nearest_index + 1 < reference.len() {
                    reference[nearest_index + 1] - reference[nearest_index]
                } else {
                    reference[nearest_index] - reference[nearest_index - 1]
                };
                let estimated_interval = if estimated_index + 1 < estimated.len() {
                    estimated[estimated_index + 1] - estimated[estimated_index]
                } else {
                    estimated[estimated_index] - estimated[estimated_index - 1]
                };
                (reference_interval, estimated_interval)
            } else {
                (
                    reference[nearest_index] - reference[nearest_index - 1],
                    estimated[estimated_index] - estimated[estimated_index - 1],
                )
            };

        let phase = if reference_interval == 0.0 {
            if minimum_difference == 0.0 { 1.0 } else { f64::INFINITY }
        } else {
            minimum_difference / reference_interval
        };
        let period = if reference_interval == 0.0 {
            if estimated_interval == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            (1.0 - estimated_interval / reference_interval).abs()
        };

        if phase < phase_threshold && period < period_threshold {
            used_annotations[nearest_index] = true;
            beat_successes[estimated_index] = true;
        }
    }

    let total_success = beat_successes.iter().filter(|success| **success).count();
    let mut longest_run = 0usize;
    let mut current_run = 0usize;
    for success in &beat_successes {
        if *success {
            current_run += 1;
            longest_run = longest_run.max(current_run);
        } else {
            current_run = 0;
        }
    }
    (
        longest_run as f64 / annotation_count as f64,
        total_success as f64 / annotation_count as f64,
    )
}

fn continuity_metrics(
    reference: &[f64],
    estimated: &[f64],
    config: &BeatGridEvaluationConfigV1,
) -> BeatGridContinuityMetricsV1 {
    let phase_threshold =
        f64::from(config.continuity_phase_threshold_basis_points) / 10_000.0;
    let period_threshold =
        f64::from(config.continuity_period_threshold_basis_points) / 10_000.0;
    let variants = if config.metric_level_variants_enabled {
        metric_level_variants(reference)
    } else {
        [reference.to_vec(), Vec::new(), Vec::new(), Vec::new(), Vec::new()]
    };

    let mut variant_scores = Vec::new();
    for variant in variants.iter().filter(|variant| !variant.is_empty()) {
        variant_scores.push(continuity_for_reference_variant(
            variant,
            estimated,
            phase_threshold,
            period_threshold,
        ));
    }
    let direct = variant_scores.first().copied().unwrap_or((0.0, 0.0));
    let any_continuous = variant_scores
        .iter()
        .map(|(continuous, _)| *continuous)
        .fold(0.0f64, f64::max);
    let any_total = variant_scores
        .iter()
        .map(|(_, total)| *total)
        .fold(0.0f64, f64::max);

    BeatGridContinuityMetricsV1 {
        correct_metric_level_continuous: direct.0,
        correct_metric_level_total: direct.1,
        any_metric_level_continuous: any_continuous,
        any_metric_level_total: any_total,
        evaluated_reference_variants: u32::try_from(variant_scores.len()).unwrap_or(u32::MAX),
    }
}

fn goto_score(
    reference: &[f64],
    estimated: &[f64],
    config: &BeatGridEvaluationConfigV1,
) -> f64 {
    if reference.is_empty() || estimated.is_empty() {
        return 0.0;
    }

    let threshold = f64::from(config.goto_threshold_basis_points) / 10_000.0;
    let mean_limit = f64::from(config.goto_mu_basis_points) / 10_000.0;
    let std_limit = f64::from(config.goto_sigma_basis_points) / 10_000.0;
    let mut beat_error = vec![1.0f64; reference.len()];

    let mut estimated_start = 0usize;
    let mut estimated_end = 0usize;
    for index in 1..reference.len().saturating_sub(1) {
        let previous_interval = 0.5 * (reference[index] - reference[index - 1]);
        let next_interval = 0.5 * (reference[index + 1] - reference[index]);
        let window_min = reference[index] - previous_interval;
        let window_max = reference[index] + next_interval;
        while estimated_start < estimated.len() && estimated[estimated_start] < window_min {
            estimated_start += 1;
        }
        if estimated_end < estimated_start {
            estimated_end = estimated_start;
        }
        while estimated_end < estimated.len() && estimated[estimated_end] < window_max {
            estimated_end += 1;
        }
        let count = estimated_end - estimated_start;
        if count == 1 {
            let offset = estimated[estimated_start] - reference[index];
            beat_error[index] = if offset < 0.0 {
                offset / previous_interval
            } else {
                offset / next_interval
            };
        }
    }

    let incorrect: Vec<usize> = beat_error
        .iter()
        .enumerate()
        .filter_map(|(index, error)| (error.abs() > threshold).then_some(index))
        .collect();

    let mut track: Vec<f64> = Vec::new();
    if incorrect.len() < 3 {
        if let (Some(first), Some(last)) = (incorrect.first(), incorrect.last()) {
            let start = first.saturating_add(1);
            let end = last.saturating_sub(1);
            if start < end {
                track.extend_from_slice(&beat_error[start..end]);
            }
        }
    } else {
        let mut best_gap = 0usize;
        let mut best_pair = None;
        for pair in incorrect.windows(2) {
            let gap = pair[1] - pair[0];
            if gap > best_gap {
                best_gap = gap;
                best_pair = Some((pair[0], pair[1]));
            }
        }
        if let Some((start, end)) = best_pair {
            let inner_count = reference.len().saturating_sub(2);
            if best_gap.saturating_sub(1) as f64 > 0.25 * inner_count as f64 {
                track.extend_from_slice(&beat_error[start..=end]);
            }
        }
    }

    if track.len() < 2 {
        return 0.0;
    }
    let mean_absolute = track.iter().map(|value| value.abs()).sum::<f64>()
        / track.len() as f64;
    let mean = track.iter().sum::<f64>() / track.len() as f64;
    let sample_variance = track
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (track.len() - 1) as f64;
    let sample_std = sample_variance.sqrt();

    if mean_absolute < mean_limit && sample_std < std_limit {
        1.0
    } else {
        0.0
    }
}

fn round_ties_even(value: f64) -> Option<i64> {
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        return None;
    }
    let floor = value.floor();
    let fraction = value - floor;
    let rounded = if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.0
    } else if (floor as i64).rem_euclid(2) == 0 {
        floor
    } else {
        floor + 1.0
    };
    Some(rounded as i64)
}

fn p_score(
    reference: &[f64],
    estimated: &[f64],
    config: &BeatGridEvaluationConfigV1,
) -> Option<f64> {
    if reference.len() <= 1 || estimated.len() <= 1 {
        return Some(0.0);
    }
    let offset = reference[0].min(estimated[0]);
    let quantize = |sequence: &[f64]| -> Option<Vec<i64>> {
        let mut output: Vec<i64> = sequence
            .iter()
            .map(|time| ((*time - offset) * 100.0).ceil() as i64)
            .collect();
        output.sort_unstable();
        output.dedup();
        Some(output)
    };
    let reference_bins = quantize(reference)?;
    let estimated_bins = quantize(estimated)?;
    if reference_bins.len() < 2 {
        // mir_eval's median inter-annotation period is undefined after
        // 10 ms impulse-train quantization collapses the reference to one cell.
        return None;
    }

    let intervals: Vec<i64> = reference_bins
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    let mut sorted_intervals = intervals;
    sorted_intervals.sort_unstable();
    let middle = sorted_intervals.len() / 2;
    let median_interval = if sorted_intervals.len() % 2 == 0 {
        (sorted_intervals[middle - 1] as f64 + sorted_intervals[middle] as f64) * 0.5
    } else {
        sorted_intervals[middle] as f64
    };
    let threshold = f64::from(config.p_score_threshold_basis_points) / 10_000.0;
    let Some(window_size) = round_ties_even(threshold * median_interval) else {
        return None;
    };
    let window_size = window_size.max(0);

    // Summing the valid-lag portion of the full impulse-train correlation is
    // exactly the count of reference/prediction impulse pairs whose quantized
    // separation lies inside the symmetric P-score window.
    let mut low = 0usize;
    let mut high = 0usize;
    let mut pair_count = 0u64;
    for &reference_bin in &reference_bins {
        let minimum = reference_bin.saturating_sub(window_size);
        let maximum = reference_bin.saturating_add(window_size);
        while low < estimated_bins.len() && estimated_bins[low] < minimum {
            low += 1;
        }
        if high < low {
            high = low;
        }
        while high < estimated_bins.len() && estimated_bins[high] <= maximum {
            high += 1;
        }
        pair_count = pair_count.saturating_add((high - low) as u64);
    }

    let denominator = reference.len().max(estimated.len());
    Some(pair_count as f64 / denominator as f64)
}

fn information_gain_entropy(
    reference: &[f64],
    estimated: &[f64],
    bins: usize,
) -> Option<f64> {
    if reference.len() <= 1 || estimated.is_empty() {
        return Some(0.0);
    }
    let mut histogram = vec![0usize; bins];
    for &estimated_time in estimated {
        let insertion = reference.partition_point(|time| *time < estimated_time);
        let nearest = if insertion == 0 {
            0
        } else if insertion >= reference.len() {
            reference.len() - 1
        } else {
            let left = insertion - 1;
            let left_error = (estimated_time - reference[left]).abs();
            let right_error = (reference[insertion] - estimated_time).abs();
            // Equality chooses the lower/earlier index, matching numpy.argmin.
            if left_error <= right_error { left } else { insertion }
        };
        let absolute_error = estimated_time - reference[nearest];

        // Mirror mir_eval 0.8.2's boundary handling, including its historical
        // nearest-first-beat / negative-error branch, for fixture parity.
        let interval = if nearest == reference.len() - 1 {
            0.5 * (reference[nearest] - reference[nearest - 1])
        } else if absolute_error < 0.0 {
            let previous_index = if nearest == 0 {
                reference.len() - 1
            } else {
                nearest - 1
            };
            0.5 * (reference[nearest] - reference[previous_index])
        } else {
            0.5 * (reference[nearest + 1] - reference[nearest])
        };
        if interval == 0.0 || !interval.is_finite() {
            return None;
        }

        let scaled_error = 0.5 * absolute_error / interval;
        // Equivalent to numpy.mod(error + 0.5, -1) + 0.5, yielding
        // the wrapped interval (-0.5, 0.5].
        let wrapped_error = scaled_error - (scaled_error - 0.5).ceil();
        let raw_bin = ((wrapped_error + 0.5) * bins as f64).floor();
        if !raw_bin.is_finite() {
            return None;
        }
        let bin = (raw_bin as isize).clamp(0, bins as isize - 1) as usize;
        histogram[bin] += 1;
    }

    let total = estimated.len() as f64;
    let entropy = histogram
        .iter()
        .filter(|count| **count > 0)
        .map(|count| {
            let probability = *count as f64 / total;
            -probability * probability.log2()
        })
        .sum::<f64>();
    Some(entropy)
}

fn information_gain(
    reference: &[f64],
    estimated: &[f64],
    bins: usize,
) -> f64 {
    if reference.len() <= 1 || estimated.len() <= 1 {
        return 0.0;
    }
    let forward = information_gain_entropy(reference, estimated, bins)
        .unwrap_or(f64::INFINITY);
    let backward = information_gain_entropy(estimated, reference, bins)
        .unwrap_or(f64::INFINITY);
    let normalizer = (bins as f64).log2();
    let entropy = forward.max(backward);
    if !normalizer.is_finite() || normalizer <= 0.0 || !entropy.is_finite() {
        0.0
    } else {
        (normalizer - entropy) / normalizer
    }
}

fn score_one_reference(
    reference: &BeatGridReferenceV1,
    prediction: &BeatGridPredictionV1,
    config: &BeatGridEvaluationConfigV1,
) -> BeatGridReferenceScoreV1 {
    let minimum_seconds = config.minimum_evaluation_time_ms.map(|value| {
        f64::from(value) / 1_000.0
    });
    let reference_beats = trimmed(&reference.beat_times_seconds, minimum_seconds);
    let estimated_beats = trimmed(&prediction.beat_times_seconds, minimum_seconds);
    let beat_matches = match_events(
        &reference_beats,
        &estimated_beats,
        f64::from(config.beat_tolerance_ms) / 1_000.0,
    );
    let beat_f_measure = event_metric(
        reference_beats.len(),
        estimated_beats.len(),
        beat_matches.len(),
    );
    let beat_timing_error =
        timing_error_summary(&reference_beats, &estimated_beats, &beat_matches);
    let inter_beat_interval_error =
        interval_error_summary(&reference_beats, &estimated_beats, &beat_matches);

    let cemgil = cemgil_score(
        &reference_beats,
        &estimated_beats,
        f64::from(config.cemgil_sigma_ms) / 1_000.0,
    );
    let goto = goto_score(&reference_beats, &estimated_beats, config);
    let p_score = p_score(&reference_beats, &estimated_beats, config);
    let information_gain = information_gain(
        &reference_beats,
        &estimated_beats,
        usize::from(config.information_gain_bins),
    );
    let cemgil_best = if config.metric_level_variants_enabled {
        metric_level_variants(&reference_beats)
            .iter()
            .filter_map(|variant| {
                cemgil_score(
                    variant,
                    &estimated_beats,
                    f64::from(config.cemgil_sigma_ms) / 1_000.0,
                )
            })
            .max_by(f64::total_cmp)
    } else {
        None
    };

    let reference_downbeats = trimmed(&reference.downbeat_times_seconds, minimum_seconds);
    let estimated_downbeats = trimmed(&prediction.downbeat_times_seconds, minimum_seconds);
    let (downbeat_metric, downbeat_timing_error) = if reference_downbeats.is_empty() {
        (None, None)
    } else {
        let matches = match_events(
            &reference_downbeats,
            &estimated_downbeats,
            f64::from(config.downbeat_tolerance_ms) / 1_000.0,
        );
        (
            Some(event_metric(
                reference_downbeats.len(),
                estimated_downbeats.len(),
                matches.len(),
            )),
            Some(timing_error_summary(
                &reference_downbeats,
                &estimated_downbeats,
                &matches,
            )),
        )
    };

    BeatGridReferenceScoreV1 {
        reference_id: reference.reference_id.clone(),
        annotation_source: reference.annotation_source.clone(),
        annotation_group: reference.annotation_group.clone(),
        reference_beat_count: u32::try_from(reference_beats.len()).unwrap_or(u32::MAX),
        predicted_beat_count: u32::try_from(estimated_beats.len()).unwrap_or(u32::MAX),
        beat_f_measure,
        cemgil_score: cemgil,
        cemgil_best_metric_level: cemgil_best,
        goto_score: goto,
        p_score,
        information_gain,
        continuity_metrics: Some(continuity_metrics(&reference_beats, &estimated_beats, config)),
        downbeat_metric,
        beat_timing_error,
        downbeat_timing_error,
        inter_beat_interval_error,
    }
}

fn median_of(mut values: Vec<f64>) -> Option<f64> {
    median(&mut values)
}

fn aggregate_scores(
    scores: &[BeatGridReferenceScoreV1],
    references: &[BeatGridReferenceV1],
) -> (f64, f64, Option<f64>, Option<f64>, Option<f64>) {
    let beat_scores: Vec<f64> = scores
        .iter()
        .map(|score| score.beat_f_measure.f1)
        .collect();
    let median_beat = median_of(beat_scores.clone()).unwrap_or(0.0);
    let minimum_beat = beat_scores.iter().copied().fold(f64::INFINITY, f64::min);

    let downbeat_scores: Vec<f64> = scores
        .iter()
        .filter_map(|score| score.downbeat_metric.as_ref().map(|metric| metric.f1))
        .collect();
    let median_downbeat = median_of(downbeat_scores);

    let mut pairwise = Vec::new();
    for left_index in 0..references.len() {
        for right_index in left_index + 1..references.len() {
            let Some(left_group) = references[left_index].annotation_group.as_ref() else {
                continue;
            };
            if references[right_index].annotation_group.as_ref() != Some(left_group) {
                continue;
            }
            let matches = match_events(
                &references[left_index].beat_times_seconds,
                &references[right_index].beat_times_seconds,
                f64::from(BEAT_GRID_DEFAULT_TOLERANCE_MS_V1) / 1_000.0,
            );
            pairwise.push(event_metric(
                references[left_index].beat_times_seconds.len(),
                references[right_index].beat_times_seconds.len(),
                matches.len(),
            ).f1);
        }
    }
    (
        median_beat,
        minimum_beat,
        median_downbeat,
        median_of(pairwise.clone()),
        pairwise.into_iter().min_by(f64::total_cmp),
    )
}

/// Evaluate a beat/downbeat tracker against one or more independent reference
/// interpretations. Reference tracks are scored separately; aggregation uses
/// the median and minimum so annotator disagreement is not hidden.
pub fn evaluate_beat_grid_v1(
    prediction: &BeatGridPredictionV1,
    references: &[BeatGridReferenceV1],
    config: BeatGridEvaluationConfigV1,
) -> Result<BeatGridEvaluationReportV1, BeatGridEvaluationIssueV1> {
    validate_inputs(prediction, references, &config)?;
    let reference_bundle_bytes = serde_json::to_vec(references)
        .map_err(|_| BeatGridEvaluationIssueV1::SerializationFailed)?;
    let prediction_bytes = serde_json::to_vec(prediction)
        .map_err(|_| BeatGridEvaluationIssueV1::SerializationFailed)?;
    let reference_bundle_sha256 = domain_separated_sha256(
        "muse-beat-grid-reference-bundle-v1",
        &reference_bundle_bytes,
    );
    let prediction_sha256 = domain_separated_sha256(
        "muse-beat-grid-prediction-v1",
        &prediction_bytes,
    );
    let reference_scores: Vec<_> = references
        .iter()
        .map(|reference| score_one_reference(reference, prediction, &config))
        .collect();
    let (
        median_beat_f1_across_references,
        minimum_beat_f1_across_references,
        median_downbeat_f1_across_references,
        median_pairwise_reference_agreement,
        minimum_pairwise_reference_agreement,
    ) = aggregate_scores(&reference_scores, references);

    let mut report = BeatGridEvaluationReportV1 {
        schema: BEAT_GRID_EVALUATION_SCHEMA_V1.into(),
        algorithm: BEAT_GRID_EVALUATION_ALGORITHM_V1.into(),
        config,
        prediction_sha256,
        reference_bundle_sha256,
        prediction_model_id: prediction.model_id.clone(),
        prediction_model_revision: prediction.model_revision.clone(),
        reference_count: u32::try_from(references.len()).unwrap_or(u32::MAX),
        reference_scores,
        median_beat_f1_across_references,
        minimum_beat_f1_across_references,
        median_downbeat_f1_across_references,
        median_pairwise_reference_agreement,
        minimum_pairwise_reference_agreement,
        report_sha256: String::new(),
    };
    report.report_sha256 = beat_grid_evaluation_commitment_v1(&report)
        .map_err(|_| BeatGridEvaluationIssueV1::SerializationFailed)?;
    Ok(report)
}

pub fn beat_grid_evaluation_commitment_v1(
    report: &BeatGridEvaluationReportV1,
) -> Result<String, serde_json::Error> {
    let commitment = EvaluationCommitmentV1 {
        schema: &report.schema,
        algorithm: &report.algorithm,
        config: &report.config,
        prediction_sha256: &report.prediction_sha256,
        reference_bundle_sha256: &report.reference_bundle_sha256,
        prediction_model_id: &report.prediction_model_id,
        prediction_model_revision: &report.prediction_model_revision,
        reference_count: report.reference_count,
        reference_scores: &report.reference_scores,
        median_beat_f1_across_references: report.median_beat_f1_across_references,
        minimum_beat_f1_across_references: report.minimum_beat_f1_across_references,
        median_downbeat_f1_across_references: report.median_downbeat_f1_across_references,
        median_pairwise_reference_agreement: report.median_pairwise_reference_agreement,
        minimum_pairwise_reference_agreement: report.minimum_pairwise_reference_agreement,
    };
    Ok(domain_separated_sha256(
        BEAT_GRID_EVALUATION_SCHEMA_V1,
        &serde_json::to_vec(&commitment)?,
    ))
}

pub fn validate_beat_grid_evaluation_report_v1(
    report: &BeatGridEvaluationReportV1,
    prediction: &BeatGridPredictionV1,
    references: &[BeatGridReferenceV1],
) -> Result<(), BeatGridEvaluationIssueV1> {
    let replay = evaluate_beat_grid_v1(prediction, references, report.config.clone())?;
    if replay != *report {
        return Err(BeatGridEvaluationIssueV1::ReportInputMismatch);
    }
    let digest = beat_grid_evaluation_commitment_v1(report)
        .map_err(|_| BeatGridEvaluationIssueV1::SerializationFailed)?;
    if digest != report.report_sha256 {
        return Err(BeatGridEvaluationIssueV1::ReportInputMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        id: &str,
        group: Option<&str>,
        beats: &[f64],
        downbeats: &[f64],
    ) -> BeatGridReferenceV1 {
        BeatGridReferenceV1 {
            reference_id: id.into(),
            beat_times_seconds: beats.to_vec(),
            downbeat_times_seconds: downbeats.to_vec(),
            annotation_source: "test-annotator".into(),
            annotation_group: group.map(str::to_owned),
        }
    }

    fn prediction(beats: &[f64], downbeats: &[f64]) -> BeatGridPredictionV1 {
        BeatGridPredictionV1 {
            model_id: "fixture-tracker".into(),
            model_revision: "fixture-v1".into(),
            beat_times_seconds: beats.to_vec(),
            downbeat_times_seconds: downbeats.to_vec(),
        }
    }

    #[test]
    fn exact_beat_and_downbeat_sequences_are_perfect() {
        let beats = [0.0, 0.5, 1.0, 1.5, 2.0];
        let downbeats = [0.0, 2.0];
        let prediction = prediction(&beats, &downbeats);
        let references = [reference("a", Some("clip-a"), &beats, &downbeats)];
        let report = evaluate_beat_grid_v1(
            &prediction,
            &references,
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        assert_eq!(report.reference_scores[0].beat_f_measure.f1, 1.0);
        assert_eq!(report.reference_scores[0].downbeat_metric.as_ref().unwrap().f1, 1.0);
        assert_eq!(report.median_beat_f1_across_references, 1.0);
        assert_eq!(report.median_downbeat_f1_across_references, Some(1.0));
    }

    #[test]
    fn default_seventy_millisecond_beat_tolerance_is_inclusive() {
        let reference = [0.0, 0.5, 1.0, 1.5];
        let predicted = [0.07, 0.57, 1.07, 1.57];
        let report = evaluate_beat_grid_v1(
            &prediction(&predicted, &[]),
            &[reference("a", None, &reference_times, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        assert_eq!(report.reference_scores[0].beat_f_measure.true_positives, 4);
        assert_eq!(report.reference_scores[0].beat_f_measure.f1, 1.0);
    }

    #[test]
    fn duplicated_estimated_beats_are_false_positives_not_double_matches() {
        let reference = [0.0, 0.5, 1.0];
        let predicted = [0.0, 0.01, 0.5, 1.0];
        let report = evaluate_beat_grid_v1(
            &prediction(&predicted, &[]),
            &[reference("a", None, &reference_times, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        assert_eq!(report.reference_scores[0].beat_f_measure.true_positives, 3);
        assert_eq!(report.reference_scores[0].beat_f_measure.false_positives, 1);
        assert!(report.reference_scores[0].beat_f_measure.f1 < 1.0);
    }

    #[test]
    fn missing_beats_reduce_recall() {
        let reference = [0.0, 0.5, 1.0, 1.5];
        let predicted = [0.0, 1.0];
        let report = evaluate_beat_grid_v1(
            &prediction(&predicted, &[]),
            &[reference("a", None, &reference, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        assert_eq!(report.reference_scores[0].beat_f_measure.recall, 0.5);
        assert_eq!(report.reference_scores[0].beat_f_measure.precision, 1.0);
    }

    #[test]
    fn half_time_interpretation_is_visible_as_strict_error_and_best_metric_level() {
        let reference = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let predicted = [0.0, 1.0, 2.0, 3.0];
        let report = evaluate_beat_grid_v1(
            &prediction(&predicted, &[]),
            &[reference("a", None, &reference, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let score = &report.reference_scores[0];
        assert!(score.beat_f_measure.f1 < 1.0);
        assert!(score.cemgil_best_metric_level.unwrap() >= score.cemgil_score.unwrap());
    }

    #[test]
    fn downbeats_are_scored_separately_from_beats() {
        let reference_beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let reference_downbeats = [0.0, 2.0];
        let predicted_beats = reference_beats;
        let predicted_downbeats = [1.0, 3.0];
        let report = evaluate_beat_grid_v1(
            &prediction(&predicted_beats, &predicted_downbeats),
            &[reference("a", None, &reference_beats, &reference_downbeats)],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        assert_eq!(report.reference_scores[0].beat_f_measure.f1, 1.0);
        assert_eq!(report.reference_scores[0].downbeat_metric.as_ref().unwrap().f1, 0.0);
    }

    #[test]
    fn missing_downbeat_annotation_is_unknown_not_zero() {
        let beats = [0.0, 0.5, 1.0];
        let report = evaluate_beat_grid_v1(
            &prediction(&beats, &[]),
            &[reference("a", None, &beats, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        assert!(report.reference_scores[0].downbeat_metric.is_none());
        assert!(report.median_downbeat_f1_across_references.is_none());
    }

    #[test]
    fn multiple_reference_interpretations_remain_separate_and_report_agreement() {
        let ref_a = [0.0, 0.5, 1.0, 1.5, 2.0];
        let ref_b = [0.0, 0.5, 1.02, 1.5, 2.0];
        let predicted = [0.0, 0.5, 1.0, 1.5, 2.0];
        let references = [
            reference("a", Some("same-clip"), &ref_a, &[]),
            reference("b", Some("same-clip"), &ref_b, &[]),
        ];
        let report = evaluate_beat_grid_v1(
            &prediction(&predicted, &[]),
            &references,
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        assert_eq!(report.reference_scores.len(), 2);
        assert_eq!(report.median_pairwise_reference_agreement, Some(1.0));
        assert!(report.reference_scores.iter().all(|score| score.beat_f_measure.f1 == 1.0));
    }

    #[test]
    fn reference_disagreement_is_not_computed_across_different_clips() {
        let ref_a = [0.0, 0.5, 1.0];
        let ref_b = [0.1, 0.6, 1.1];
        let references = [
            reference("a", Some("clip-a"), &ref_a, &[]),
            reference("b", Some("clip-b"), &ref_b, &[]),
        ];
        let report = evaluate_beat_grid_v1(
            &prediction(&ref_a, &[]),
            &references,
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        assert!(report.median_pairwise_reference_agreement.is_none());
    }

    #[test]
    fn explicit_five_second_trim_is_applied_to_both_sequences() {
        let reference_times = [0.0, 0.5, 1.0, 5.0, 5.5, 6.0];
        let predicted_times = [0.1, 0.6, 5.0, 5.5, 6.0];
        let report = evaluate_beat_grid_v1(
            &prediction(&predicted_times, &[]),
            &[reference("a", None, &reference_times, &[])],
            BeatGridEvaluationConfigV1 {
                minimum_evaluation_time_ms: Some(5_000),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(report.reference_scores[0].beat_f_measure.f1, 1.0);
        assert_eq!(report.reference_scores[0].reference_beat_count, 3);
    }

    #[test]
    fn empty_reference_beat_annotation_is_rejected() {
        let references = [reference("empty", None, &[], &[])];
        assert_eq!(
            evaluate_beat_grid_v1(
                &prediction(&[], &[]),
                &references,
                BeatGridEvaluationConfigV1::default(),
            )
            .unwrap_err(),
            BeatGridEvaluationIssueV1::EmptyReferenceBeats { index: 0 }
        );
    }

    #[test]
    fn inter_beat_interval_error_is_reported_only_for_adjacent_matches() {
        let reference = [0.0, 0.5, 1.0, 1.5, 2.0];
        let predicted = [0.0, 0.51, 1.02, 1.53, 2.04];
        let report = evaluate_beat_grid_v1(
            &prediction(&predicted, &[]),
            &[reference("a", None, &reference, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let summary = &report.reference_scores[0].inter_beat_interval_error;
        assert_eq!(summary.adjacent_interval_pair_count, 4);
        assert!((summary.mean_absolute_error_ms.unwrap() - 10.0).abs() < 1e-9);
        assert!((summary.mean_absolute_relative_error_percent.unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn references_without_explicit_shared_group_are_not_cross_compared() {
        let a = [0.0, 0.5, 1.0];
        let b = [0.0, 0.6, 1.2];
        let references = [
            reference("a", None, &a, &[]),
            reference("b", None, &b, &[]),
        ];
        let report = evaluate_beat_grid_v1(
            &prediction(&a, &[]),
            &references,
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        assert!(report.median_pairwise_reference_agreement.is_none());
    }

    #[test]
    fn continuity_metrics_expose_a_break_in_the_beat_sequence() {
        let reference_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let predicted_times = [0.0, 0.5, 1.0, 1.85, 2.35, 2.85];
        let report = evaluate_beat_grid_v1(
            &prediction(&predicted_times, &[]),
            &[reference("clip", None, &reference_times, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let continuity = report.reference_scores[0].continuity_metrics.as_ref().unwrap();
        assert!(continuity.correct_metric_level_total < 1.0);
        assert!(continuity.correct_metric_level_continuous < 1.0);
        assert!(continuity.any_metric_level_total >= continuity.correct_metric_level_total);
    }

    #[test]
    fn continuity_any_metric_level_detects_a_half_time_sequence() {
        let reference_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let predicted_times = [0.0, 1.0, 2.0, 3.0];
        let report = evaluate_beat_grid_v1(
            &prediction(&predicted_times, &[]),
            &[reference("clip", None, &reference_times, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let continuity = report.reference_scores[0].continuity_metrics.as_ref().unwrap();
        assert!(continuity.correct_metric_level_total < continuity.any_metric_level_total);
    }

    #[test]
    fn continuity_is_zero_when_only_one_beat_is_available() {
        let reference_times = [0.0];
        let predicted_times = [0.0];
        let report = evaluate_beat_grid_v1(
            &prediction(&predicted_times, &[]),
            &[reference("clip", None, &reference_times, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let continuity = report.reference_scores[0].continuity_metrics.as_ref().unwrap();
        assert_eq!(continuity.correct_metric_level_continuous, 0.0);
        assert_eq!(continuity.correct_metric_level_total, 0.0);
    }

    #[test]
    fn exact_and_offbeat_cemgil_values_match_mir_eval_metric_level_semantics() {
        let reference_times = [0.0, 0.5, 1.0, 1.5, 2.0];
        let exact_report = evaluate_beat_grid_v1(
            &prediction(&reference_times, &[]),
            &[reference("exact", None, &reference_times, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let exact = &exact_report.reference_scores[0];
        assert_eq!(exact.beat_f_measure.f1, 1.0);
        assert!((exact.cemgil_score.unwrap() - 1.0).abs() < 1e-12);
        assert!((exact.cemgil_best_metric_level.unwrap() - 1.0).abs() < 1e-12);
        let offbeat_times = [0.25, 0.75, 1.25, 1.75];
        let offbeat_report = evaluate_beat_grid_v1(
            &prediction(&offbeat_times, &[]),
            &[reference("offbeat", None, &reference_times, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let offbeat = &offbeat_report.reference_scores[0];
        assert_eq!(offbeat.beat_f_measure.f1, 0.0);
        assert!(offbeat.cemgil_score.unwrap() < 1e-6);
        assert!((offbeat.cemgil_best_metric_level.unwrap() - 1.0).abs() < 1e-12);
        let continuity = offbeat.continuity_metrics.as_ref().unwrap();
        assert_eq!(continuity.correct_metric_level_total, 0.0);
        assert!((continuity.any_metric_level_total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn doubled_pulse_is_strictly_scored_but_metric_level_tolerance_is_separate() {
        let reference_times = [0.0, 1.0, 2.0, 3.0];
        let doubled_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let report = evaluate_beat_grid_v1(
            &prediction(&doubled_times, &[]),
            &[reference("double", None, &reference_times, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let score = &report.reference_scores[0];
        assert!(score.beat_f_measure.f1 < 1.0);
        assert!(score.cemgil_best_metric_level.unwrap() > score.cemgil_score.unwrap());
        let continuity = score.continuity_metrics.as_ref().unwrap();
        assert!(continuity.any_metric_level_total >= continuity.correct_metric_level_total);
    }

    #[test]
    fn strict_beat_threshold_has_no_extra_epsilon() {
        let reference_times = [0.0, 1.0, 2.0];
        let just_outside = [0.070_001, 1.070_001, 2.070_001];
        let report = evaluate_beat_grid_v1(
            &prediction(&just_outside, &[]),
            &[reference("edge", None, &reference_times, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        assert_eq!(report.reference_scores[0].beat_f_measure.true_positives, 0);
    }

    #[test]
    fn pinned_mir_eval_0_8_2_reference_fixtures_match_beat_metric_semantics() {
        // Expected values were generated with the installed mir_eval 0.8.2
        // package using direct beat metric functions (not evaluate(), whose
        // default five-second trim would remove these short synthetic cases).
        let exact = [0.0, 0.5, 1.0, 1.5, 2.0];
        let exact_report = evaluate_beat_grid_v1(
            &prediction(&exact, &[]),
            &[reference("exact", None, &exact, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let exact_score = &exact_report.reference_scores[0];
        assert!((exact_score.beat_f_measure.f1 - 1.0).abs() < 1e-12);
        assert!((exact_score.cemgil_score.unwrap() - 1.0).abs() < 1e-12);
        assert!((exact_score.cemgil_best_metric_level.unwrap() - 1.0).abs() < 1e-12);
        let exact_cml = exact_score.continuity_metrics.as_ref().unwrap();
        assert!((exact_cml.correct_metric_level_continuous - 1.0).abs() < 1e-12);
        assert!((exact_cml.correct_metric_level_total - 1.0).abs() < 1e-12);
        assert!((exact_cml.any_metric_level_continuous - 1.0).abs() < 1e-12);
        assert!((exact_cml.any_metric_level_total - 1.0).abs() < 1e-12);

        let offbeat_ref = [0.0, 0.5, 1.0, 1.5, 2.0];
        let offbeat_est = [0.25, 0.75, 1.25, 1.75];
        let offbeat_report = evaluate_beat_grid_v1(
            &prediction(&offbeat_est, &[]),
            &[reference("offbeat", None, &offbeat_ref, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let offbeat = &offbeat_report.reference_scores[0];
        assert_eq!(offbeat.beat_f_measure.f1, 0.0);
        assert!((offbeat.cemgil_score.unwrap() - 3.66e-9).abs() < 1e-9);
        assert!((offbeat.cemgil_best_metric_level.unwrap() - 1.0).abs() < 1e-12);
        let offbeat_cml = offbeat.continuity_metrics.as_ref().unwrap();
        assert_eq!(offbeat_cml.correct_metric_level_total, 0.0);
        assert_eq!(offbeat_cml.correct_metric_level_continuous, 0.0);
        assert!((offbeat_cml.any_metric_level_total - 1.0).abs() < 1e-12);
        assert!((offbeat_cml.any_metric_level_continuous - 1.0).abs() < 1e-12);

        let half_ref = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let half_est = [0.0, 1.0, 2.0, 3.0];
        let half_report = evaluate_beat_grid_v1(
            &prediction(&half_est, &[]),
            &[reference("half", None, &half_ref, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let half = &half_report.reference_scores[0];
        assert!((half.beat_f_measure.f1 - 0.7272727272727273).abs() < 1e-12);
        assert!((half.cemgil_score.unwrap() - 0.7272727272727273).abs() < 1e-12);
        assert!((half.cemgil_best_metric_level.unwrap() - 1.0).abs() < 1e-12);
        let half_cml = half.continuity_metrics.as_ref().unwrap();
        assert_eq!(half_cml.correct_metric_level_continuous, 0.0);
        assert_eq!(half_cml.correct_metric_level_total, 0.0);
        assert!((half_cml.any_metric_level_continuous - 1.0).abs() < 1e-12);
        assert!((half_cml.any_metric_level_total - 1.0).abs() < 1e-12);

        let double_ref = [0.0, 1.0, 2.0, 3.0];
        let double_est = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let double_report = evaluate_beat_grid_v1(
            &prediction(&double_est, &[]),
            &[reference("double", None, &double_ref, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let double = &double_report.reference_scores[0];
        assert!((double.beat_f_measure.f1 - 0.7272727272727273).abs() < 1e-12);
        assert!((double.cemgil_score.unwrap() - 0.7272727272727273).abs() < 1e-12);
        assert!((double.cemgil_best_metric_level.unwrap() - 1.0).abs() < 1e-12);

        let broken_ref = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let broken_est = [0.0, 0.5, 1.0, 1.85, 2.35, 2.85];
        let broken_report = evaluate_beat_grid_v1(
            &prediction(&broken_est, &[]),
            &[reference("broken", None, &broken_ref, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let broken = &broken_report.reference_scores[0];
        assert!((broken.beat_f_measure.f1 - 0.5).abs() < 1e-12);
        assert!((broken.cemgil_score.unwrap() - 0.500294608769).abs() < 1e-9);
        let broken_cml = broken.continuity_metrics.as_ref().unwrap();
        assert!((broken_cml.correct_metric_level_continuous - 0.5).abs() < 1e-12);
        assert!((broken_cml.correct_metric_level_total - 0.5).abs() < 1e-12);

        let outside_ref = [0.0, 1.0, 2.0];
        let outside_est = [0.070001, 1.070001, 2.070001];
        let outside_report = evaluate_beat_grid_v1(
            &prediction(&outside_est, &[]),
            &[reference("outside", None, &outside_ref, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        let outside = &outside_report.reference_scores[0];
        assert_eq!(outside.beat_f_measure.f1, 0.0);
        let outside_cml = outside.continuity_metrics.as_ref().unwrap();
        // mir_eval's continuity criterion is phase/period based, not 70 ms.
        assert!((outside_cml.correct_metric_level_total - 1.0).abs() < 1e-12);
        assert!((outside_cml.any_metric_level_total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn malformed_non_monotonic_timestamp_sequences_are_rejected() {
        let prediction = prediction(&[0.0, 0.5, 0.4], &[]);
        let references = [reference("a", None, &[0.0, 0.5, 1.0], &[])];
        assert!(matches!(
            evaluate_beat_grid_v1(
                &prediction,
                &references,
                BeatGridEvaluationConfigV1::default()
            ),
            Err(BeatGridEvaluationIssueV1::InvalidTimestampSequence { .. })
        ));
    }

    #[test]
    fn report_replay_rejects_changed_prediction_or_reference_input() {
        let beats = [0.0, 0.5, 1.0];
        let prediction = prediction(&beats, &[]);
        let references = [reference("a", None, &beats, &[])];
        let report = evaluate_beat_grid_v1(
            &prediction,
            &references,
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        assert!(validate_beat_grid_evaluation_report_v1(
            &report,
            &prediction,
            &references
        )
        .is_ok());
        let changed_prediction = prediction(&[0.0, 0.51, 1.0], &[]);
        assert_eq!(
            validate_beat_grid_evaluation_report_v1(
                &report,
                &changed_prediction,
                &references
            )
            .unwrap_err(),
            BeatGridEvaluationIssueV1::ReportInputMismatch
        );
    }

    #[test]
    fn empty_prediction_is_a_valid_zero_score_not_a_parser_error() {
        let reference = [0.0, 0.5, 1.0];
        let report = evaluate_beat_grid_v1(
            &prediction(&[], &[]),
            &[reference("a", None, &reference, &[])],
            BeatGridEvaluationConfigV1::default(),
        )
        .unwrap();
        assert_eq!(report.reference_scores[0].beat_f_measure.f1, 0.0);
        assert_eq!(report.reference_scores[0].beat_f_measure.false_negatives, 3);
    }
}
