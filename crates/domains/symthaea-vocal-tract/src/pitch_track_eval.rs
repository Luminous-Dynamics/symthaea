// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Source- and clock-bound pitch-track evaluation.
//!
//! This module scores an estimator against an independent reference track. It
//! deliberately separates voiced/unvoiced decisions from F0 error and reports
//! missing-frame coverage so dropped observations cannot disappear from the
//! denominator. It does not implement a pitch estimator.

use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

/// Current pitch-track evaluation schema version.
pub const PITCH_TRACK_EVALUATION_SCHEMA_V1: u32 = 1;

/// Default relative-frequency threshold for a gross pitch error.
pub const DEFAULT_GROSS_PITCH_ERROR_RATIO: f64 = 0.20;

/// Default relative tolerance for identifying octave/sub-octave errors.
pub const DEFAULT_OCTAVE_RATIO_TOLERANCE: f64 = 0.05;

/// Immutable source and clock identity shared by reference and estimated tracks.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PitchTrackSourceV1 {
    /// SHA-256 digest of the exact source recording or immutable source segment.
    pub source_content_sha256: String,
    /// Source clock domain.
    pub clock_domain_id: String,
    /// Clock generation; must change after a source clock reset.
    pub clock_generation: u64,
    /// Inclusive start of the evaluated source interval, in clock nanoseconds.
    pub source_start_ns: u64,
    /// Exclusive end of the evaluated source interval, in clock nanoseconds.
    pub source_end_ns: u64,
}

/// One time-supported reference or estimated pitch observation.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PitchTrackFrameV1 {
    /// Monotonic frame index within this track.
    pub frame_index: u64,
    /// Frame reference time in the source clock, in nanoseconds.
    pub source_timestamp_ns: u64,
    /// F0 in Hz. None explicitly means unvoiced or no valid F0 estimate.
    pub f0_hz: Option<f64>,
    /// Optional periodicity/voicing evidence in [0, 1]. Not a calibrated probability.
    pub periodicity: Option<f64>,
}

/// A pitch track tied to one immutable source interval.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PitchTrackV1 {
    /// Schema version.
    pub schema_version: u32,
    /// Stable ID for this reference or estimator track.
    pub track_id: String,
    /// Extractor/annotator identity. This is descriptive, not a trust claim.
    pub producer_id: String,
    /// Shared source and clock identity.
    pub source: PitchTrackSourceV1,
    /// Ordered observations, including explicit unvoiced frames.
    pub frames: Vec<PitchTrackFrameV1>,
}

/// Frozen thresholds used by the pitch-track scorer.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PitchTrackScoreConfigV1 {
    /// Maximum absolute source-timestamp difference for one-to-one alignment.
    pub timestamp_tolerance_ns: u64,
    /// Relative F0 error above which a jointly voiced frame is a gross pitch error.
    pub gross_pitch_error_ratio: f64,
    /// Relative ratio tolerance around 2x or 0.5x for an octave error.
    pub octave_ratio_tolerance: f64,
}

impl Default for PitchTrackScoreConfigV1 {
    fn default() -> Self {
        Self {
            timestamp_tolerance_ns: 2_500_000,
            gross_pitch_error_ratio: DEFAULT_GROSS_PITCH_ERROR_RATIO,
            octave_ratio_tolerance: DEFAULT_OCTAVE_RATIO_TOLERANCE,
        }
    }
}

/// Typed scorer error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PitchTrackEvaluationError {
    message: String,
}

impl PitchTrackEvaluationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for PitchTrackEvaluationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for PitchTrackEvaluationError {}

/// Source-matched pitch and voicing metrics for one reference/estimated pair.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PitchTrackMetricsV1 {
    /// Number of reference observations.
    pub reference_frame_count: u64,
    /// Number of estimated observations.
    pub estimated_frame_count: u64,
    /// Number of source-time-aligned one-to-one frame pairs.
    pub aligned_frame_count: u64,
    /// Reference observations without a timestamp-matched estimate.
    pub missing_estimated_frame_count: u64,
    /// Estimated observations without a timestamp-matched reference.
    pub unmatched_estimated_frame_count: u64,
    /// Voiced/voiced pairs used for F0 error.
    pub jointly_voiced_frame_count: u64,
    /// True positive voiced decisions, including missing estimates as false negatives.
    pub voiced_true_positive: u64,
    /// False-positive voiced estimates, including unmatched estimated frames.
    pub voiced_false_positive: u64,
    /// Missed reference-voiced frames.
    pub voiced_false_negative: u64,
    /// Aligned reference/estimated pairs both marked unvoiced.
    pub unvoiced_true_negative: u64,
    /// Aligned-frame voicing precision.
    pub voicing_precision: Option<f64>,
    /// Reference-voiced recall.
    pub voicing_recall: Option<f64>,
    /// Voicing F1.
    pub voicing_f1: Option<f64>,
    /// Voicing decision error rate over scored voiced/unvoiced decisions.
    pub voicing_error_rate: Option<f64>,
    /// Fraction of reference frames with a time-matched estimate.
    pub frame_coverage: Option<f64>,
    /// Root-mean-square F0 error in Hz over jointly voiced frames.
    pub f0_rmse_hz: Option<f64>,
    /// Median absolute F0 error in cents over jointly voiced frames.
    pub median_absolute_f0_error_cents: Option<f64>,
    /// Mean signed F0 error in cents over jointly voiced frames.
    pub mean_signed_f0_error_cents: Option<f64>,
    /// Fraction of jointly voiced frames with relative error above the frozen GPE threshold.
    pub gross_pitch_error_rate: Option<f64>,
    /// Fraction of jointly voiced frames with a 2x or 0.5x octave/sub-octave error.
    pub octave_error_rate: Option<f64>,
}

/// Score an estimated pitch track against an independent reference.
///
/// Tracks must describe the exact same source digest, source clock domain and
/// generation, and source interval. Frames are aligned one-to-one by source
/// timestamp using the configured tolerance. Missing frames are reported and
/// missing estimates on reference-voiced frames count as voicing false negatives.
///
/// F0 RMSE, cents errors, gross pitch error and octave error are calculated
/// only where both tracks contain valid positive F0. They are not imputed for
/// unvoiced or missing frames.
pub fn score_pitch_tracks(
    reference: &PitchTrackV1,
    estimated: &PitchTrackV1,
    config: PitchTrackScoreConfigV1,
) -> Result<PitchTrackMetricsV1, PitchTrackEvaluationError> {
    validate_score_config(config)?;
    validate_pitch_track(reference, "reference")?;
    validate_pitch_track(estimated, "estimated")?;

    if reference.source != estimated.source {
        return Err(PitchTrackEvaluationError::new(
            "reference and estimated pitch tracks do not share the exact source/clock interval",
        ));
    }

    let mut metrics = PitchTrackMetricsV1 {
        reference_frame_count: reference.frames.len() as u64,
        estimated_frame_count: estimated.frames.len() as u64,
        aligned_frame_count: 0,
        missing_estimated_frame_count: 0,
        unmatched_estimated_frame_count: 0,
        jointly_voiced_frame_count: 0,
        voiced_true_positive: 0,
        voiced_false_positive: 0,
        voiced_false_negative: 0,
        unvoiced_true_negative: 0,
        voicing_precision: None,
        voicing_recall: None,
        voicing_f1: None,
        voicing_error_rate: None,
        frame_coverage: None,
        f0_rmse_hz: None,
        median_absolute_f0_error_cents: None,
        mean_signed_f0_error_cents: None,
        gross_pitch_error_rate: None,
        octave_error_rate: None,
    };

    let mut ref_index = 0usize;
    let mut est_index = 0usize;
    let mut f0_errors = F0ErrorAccumulator::default();

    while ref_index < reference.frames.len() && est_index < estimated.frames.len() {
        let ref_frame = reference.frames[ref_index];
        let est_frame = estimated.frames[est_index];
        let timestamp_delta = est_frame
            .source_timestamp_ns
            .abs_diff(ref_frame.source_timestamp_ns);

        if timestamp_delta <= config.timestamp_tolerance_ns {
            // When more than one observation lies inside the tolerance, prefer
            // the locally closest monotonic pair instead of accepting the first
            // in-tolerance frame. This prevents a jittered duplicate from
            // displacing an exact timestamp match.
            let next_estimated_delta = estimated
                .frames
                .get(est_index + 1)
                .map(|frame| {
                    frame
                        .source_timestamp_ns
                        .abs_diff(ref_frame.source_timestamp_ns)
                })
                .filter(|delta| *delta <= config.timestamp_tolerance_ns);
            let next_reference_delta = reference
                .frames
                .get(ref_index + 1)
                .map(|frame| {
                    frame
                        .source_timestamp_ns
                        .abs_diff(est_frame.source_timestamp_ns)
                })
                .filter(|delta| *delta <= config.timestamp_tolerance_ns);

            if next_estimated_delta.is_some_and(|delta| delta < timestamp_delta)
                && next_estimated_delta.unwrap() <= next_reference_delta.unwrap_or(u64::MAX)
            {
                metrics.unmatched_estimated_frame_count += 1;
                if est_frame.f0_hz.is_some() {
                    metrics.voiced_false_positive += 1;
                }
                est_index += 1;
                continue;
            }
            if next_reference_delta.is_some_and(|delta| delta < timestamp_delta) {
                metrics.missing_estimated_frame_count += 1;
                if ref_frame.f0_hz.is_some() {
                    metrics.voiced_false_negative += 1;
                }
                ref_index += 1;
                continue;
            }

            metrics.aligned_frame_count += 1;
            score_aligned_frame(ref_frame, est_frame, config, &mut metrics, &mut f0_errors);
            ref_index += 1;
            est_index += 1;
        } else if est_frame.source_timestamp_ns < ref_frame.source_timestamp_ns {
            metrics.unmatched_estimated_frame_count += 1;
            if est_frame.f0_hz.is_some() {
                metrics.voiced_false_positive += 1;
            }
            est_index += 1;
        } else {
            metrics.missing_estimated_frame_count += 1;
            if ref_frame.f0_hz.is_some() {
                metrics.voiced_false_negative += 1;
            }
            ref_index += 1;
        }
    }

    while ref_index < reference.frames.len() {
        metrics.missing_estimated_frame_count += 1;
        if reference.frames[ref_index].f0_hz.is_some() {
            metrics.voiced_false_negative += 1;
        }
        ref_index += 1;
    }

    while est_index < estimated.frames.len() {
        metrics.unmatched_estimated_frame_count += 1;
        if estimated.frames[est_index].f0_hz.is_some() {
            metrics.voiced_false_positive += 1;
        }
        est_index += 1;
    }

    let scored_decisions = metrics.voiced_true_positive
        + metrics.voiced_false_positive
        + metrics.voiced_false_negative
        + metrics.unvoiced_true_negative;
    let predicted_voiced = metrics.voiced_true_positive + metrics.voiced_false_positive;
    let reference_voiced = metrics.voiced_true_positive + metrics.voiced_false_negative;
    metrics.voicing_precision = if predicted_voiced > 0 {
        ratio(metrics.voiced_true_positive, predicted_voiced)
    } else if reference_voiced > 0 {
        Some(0.0)
    } else {
        None
    };
    metrics.voicing_recall = if reference_voiced > 0 {
        ratio(metrics.voiced_true_positive, reference_voiced)
    } else if predicted_voiced > 0 {
        Some(0.0)
    } else {
        None
    };
    metrics.voicing_f1 = harmonic_mean(metrics.voicing_precision, metrics.voicing_recall);
    metrics.voicing_error_rate = ratio(
        metrics.voiced_false_positive + metrics.voiced_false_negative,
        scored_decisions,
    );
    metrics.frame_coverage = ratio(metrics.aligned_frame_count, metrics.reference_frame_count);

    if metrics.jointly_voiced_frame_count > 0 {
        let count = metrics.jointly_voiced_frame_count as f64;
        metrics.f0_rmse_hz = Some((f0_errors.squared_hz_error / count).sqrt());
        metrics.median_absolute_f0_error_cents = Some(median(&mut f0_errors.absolute_cents_error));
        metrics.mean_signed_f0_error_cents =
            Some(f0_errors.signed_cents_error.iter().sum::<f64>() / count);
        metrics.gross_pitch_error_rate = Some(f0_errors.gross_pitch_errors as f64 / count);
        metrics.octave_error_rate = Some(f0_errors.octave_errors as f64 / count);
    }

    Ok(metrics)
}

#[derive(Default)]
struct F0ErrorAccumulator {
    squared_hz_error: f64,
    signed_cents_error: Vec<f64>,
    absolute_cents_error: Vec<f64>,
    gross_pitch_errors: u64,
    octave_errors: u64,
}

fn score_aligned_frame(
    reference: PitchTrackFrameV1,
    estimated: PitchTrackFrameV1,
    config: PitchTrackScoreConfigV1,
    metrics: &mut PitchTrackMetricsV1,
    errors: &mut F0ErrorAccumulator,
) {
    match (reference.f0_hz, estimated.f0_hz) {
        (Some(reference_f0), Some(estimated_f0)) => {
            metrics.voiced_true_positive += 1;
            metrics.jointly_voiced_frame_count += 1;
            let hz_error = estimated_f0 - reference_f0;
            errors.squared_hz_error += hz_error * hz_error;
            let cents_error =
                1200.0 * (estimated_f0.ln() - reference_f0.ln()) / std::f64::consts::LN_2;
            errors.signed_cents_error.push(cents_error);
            errors.absolute_cents_error.push(cents_error.abs());
            let relative_error = (estimated_f0 / reference_f0 - 1.0).abs();
            if relative_error > config.gross_pitch_error_ratio {
                errors.gross_pitch_errors += 1;
            }
            let ratio = estimated_f0 / reference_f0;
            let octave_error = (ratio / 2.0 - 1.0).abs() <= config.octave_ratio_tolerance
                || (ratio / 0.5 - 1.0).abs() <= config.octave_ratio_tolerance;
            if octave_error {
                errors.octave_errors += 1;
            }
        }
        (Some(_), None) => metrics.voiced_false_negative += 1,
        (None, Some(_)) => metrics.voiced_false_positive += 1,
        (None, None) => metrics.unvoiced_true_negative += 1,
    }
}

fn validate_score_config(config: PitchTrackScoreConfigV1) -> Result<(), PitchTrackEvaluationError> {
    if !config.gross_pitch_error_ratio.is_finite()
        || config.gross_pitch_error_ratio <= 0.0
        || config.gross_pitch_error_ratio >= 1.0
    {
        return Err(PitchTrackEvaluationError::new(
            "gross pitch error ratio must be finite and between zero and one",
        ));
    }
    if !config.octave_ratio_tolerance.is_finite()
        || config.octave_ratio_tolerance < 0.0
        || config.octave_ratio_tolerance >= 0.5
    {
        return Err(PitchTrackEvaluationError::new(
            "octave ratio tolerance must be finite and in [0, 0.5)",
        ));
    }
    Ok(())
}

fn validate_pitch_track(
    track: &PitchTrackV1,
    label: &str,
) -> Result<(), PitchTrackEvaluationError> {
    if track.schema_version != PITCH_TRACK_EVALUATION_SCHEMA_V1 {
        return Err(PitchTrackEvaluationError::new(format!(
            "{label} pitch track has unsupported schema version"
        )));
    }
    if track.track_id.trim().is_empty() || track.producer_id.trim().is_empty() {
        return Err(PitchTrackEvaluationError::new(format!(
            "{label} pitch track and producer IDs must be non-empty"
        )));
    }
    validate_sha256(&track.source.source_content_sha256)?;
    if track.source.clock_domain_id.trim().is_empty()
        || track.source.source_end_ns <= track.source.source_start_ns
    {
        return Err(PitchTrackEvaluationError::new(format!(
            "{label} pitch track has invalid source clock/interval metadata"
        )));
    }

    let mut previous_timestamp = None;
    let mut previous_index = None;
    for frame in &track.frames {
        if frame.source_timestamp_ns < track.source.source_start_ns
            || frame.source_timestamp_ns >= track.source.source_end_ns
        {
            return Err(PitchTrackEvaluationError::new(format!(
                "{label} frame timestamp is outside the declared source interval"
            )));
        }
        if previous_timestamp.is_some_and(|previous| frame.source_timestamp_ns <= previous) {
            return Err(PitchTrackEvaluationError::new(format!(
                "{label} frame timestamps must be strictly increasing"
            )));
        }
        if previous_index.is_some_and(|previous| frame.frame_index <= previous) {
            return Err(PitchTrackEvaluationError::new(format!(
                "{label} frame indices must be strictly increasing"
            )));
        }
        if frame
            .f0_hz
            .is_some_and(|f0| !f0.is_finite() || !(1.0..=20_000.0).contains(&f0))
        {
            return Err(PitchTrackEvaluationError::new(format!(
                "{label} frame F0 must be finite and in [1, 20,000] Hz when present"
            )));
        }
        if frame
            .periodicity
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(PitchTrackEvaluationError::new(format!(
                "{label} periodicity must be finite and in [0, 1] when present"
            )));
        }
        previous_timestamp = Some(frame.source_timestamp_ns);
        previous_index = Some(frame.frame_index);
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), PitchTrackEvaluationError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PitchTrackEvaluationError::new(
            "source content digest must be a 64-character SHA-256 hex digest",
        ));
    }
    Ok(())
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn harmonic_mean(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) if left + right > 0.0 => {
            Some(2.0 * left * right / (left + right))
        }
        (Some(0.0), Some(0.0)) => Some(0.0),
        _ => None,
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> PitchTrackSourceV1 {
        PitchTrackSourceV1 {
            source_content_sha256: "a".repeat(64),
            clock_domain_id: "mic-clock".to_string(),
            clock_generation: 3,
            source_start_ns: 0,
            source_end_ns: 100_000_000,
        }
    }

    fn track(track_id: &str, values: &[Option<f64>]) -> PitchTrackV1 {
        PitchTrackV1 {
            schema_version: PITCH_TRACK_EVALUATION_SCHEMA_V1,
            track_id: track_id.to_string(),
            producer_id: format!("producer-{track_id}"),
            source: source(),
            frames: values
                .iter()
                .enumerate()
                .map(|(index, f0_hz)| PitchTrackFrameV1 {
                    frame_index: index as u64,
                    source_timestamp_ns: index as u64 * 10_000_000,
                    f0_hz: *f0_hz,
                    periodicity: f0_hz.map(|_| 0.95),
                })
                .collect(),
        }
    }

    #[test]
    fn perfect_pitch_and_voicing_track_has_zero_error() {
        let reference = track("reference", &[Some(100.0), Some(110.0), None, Some(120.0)]);
        let estimated = track("estimated", &[Some(100.0), Some(110.0), None, Some(120.0)]);
        let metrics =
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).unwrap();
        assert_eq!(metrics.aligned_frame_count, 4);
        assert_eq!(metrics.frame_coverage, Some(1.0));
        assert_eq!(metrics.voicing_f1, Some(1.0));
        assert_eq!(metrics.f0_rmse_hz, Some(0.0));
        assert_eq!(metrics.median_absolute_f0_error_cents, Some(0.0));
        assert_eq!(metrics.gross_pitch_error_rate, Some(0.0));
        assert_eq!(metrics.octave_error_rate, Some(0.0));
    }

    #[test]
    fn octave_error_is_counted_separately_from_f0_error() {
        let reference = track("reference", &[Some(100.0), Some(200.0)]);
        let estimated = track("estimated", &[Some(200.0), Some(100.0)]);
        let metrics =
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).unwrap();
        assert_eq!(metrics.jointly_voiced_frame_count, 2);
        assert_eq!(metrics.octave_error_rate, Some(1.0));
        assert_eq!(metrics.gross_pitch_error_rate, Some(1.0));
    }

    #[test]
    fn voiced_unvoiced_confusion_is_scored_without_zero_hz_fabrication() {
        let reference = track("reference", &[Some(120.0), None, Some(150.0), None]);
        let estimated = track("estimated", &[None, Some(90.0), Some(150.0), None]);
        let metrics =
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).unwrap();
        assert_eq!(metrics.voiced_true_positive, 1);
        assert_eq!(metrics.voiced_false_positive, 1);
        assert_eq!(metrics.voiced_false_negative, 1);
        assert_eq!(metrics.unvoiced_true_negative, 1);
        assert_eq!(metrics.voicing_precision, Some(0.5));
        assert_eq!(metrics.voicing_recall, Some(0.5));
        assert_eq!(metrics.voicing_f1, Some(0.5));
        assert_eq!(metrics.f0_rmse_hz, Some(0.0));
    }

    #[test]
    fn missing_frames_reduce_coverage_and_count_voiced_misses() {
        let reference = track("reference", &[Some(100.0), Some(110.0), Some(120.0)]);
        let mut estimated = track("estimated", &[Some(100.0), Some(120.0)]);
        estimated.frames[1].source_timestamp_ns = 20_000_000;
        estimated.frames[1].frame_index = 2;
        let metrics =
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).unwrap();
        assert_eq!(metrics.aligned_frame_count, 2);
        assert_eq!(metrics.missing_estimated_frame_count, 1);
        assert_eq!(metrics.frame_coverage, Some(2.0 / 3.0));
        assert_eq!(metrics.voiced_false_negative, 1);
    }

    #[test]
    fn ambiguous_in_tolerance_frames_choose_the_closest_monotonic_pair() {
        let mut reference = track("reference", &[Some(100.0)]);
        reference.frames[0].source_timestamp_ns = 10_000_000;
        let mut estimated = track("estimated", &[Some(80.0), Some(100.0)]);
        estimated.frames[0].source_timestamp_ns = 9_000_000;
        estimated.frames[0].frame_index = 0;
        estimated.frames[1].source_timestamp_ns = 10_000_000;
        estimated.frames[1].frame_index = 1;

        let metrics =
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).unwrap();

        assert_eq!(metrics.aligned_frame_count, 1);
        assert_eq!(metrics.unmatched_estimated_frame_count, 1);
        assert_eq!(metrics.f0_rmse_hz, Some(0.0));
        assert_eq!(metrics.gross_pitch_error_rate, Some(0.0));
    }

    #[test]
    fn timestamp_jitter_within_tolerance_is_aligned() {
        let reference = track("reference", &[Some(100.0), Some(110.0)]);
        let mut estimated = track("estimated", &[Some(100.0), Some(110.0)]);
        estimated.frames[0].source_timestamp_ns += 2_000_000;
        estimated.frames[1].source_timestamp_ns += 2_000_000;
        let metrics =
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).unwrap();
        assert_eq!(metrics.aligned_frame_count, 2);
    }

    #[test]
    fn source_digest_clock_or_generation_mismatch_is_rejected() {
        let reference = track("reference", &[Some(100.0)]);
        let mut estimated = track("estimated", &[Some(100.0)]);
        estimated.source.source_content_sha256 = "b".repeat(64);
        assert!(
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).is_err()
        );

        let mut estimated = track("estimated", &[Some(100.0)]);
        estimated.source.clock_generation += 1;
        assert!(
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).is_err()
        );
    }

    #[test]
    fn invalid_f0_periodicity_timestamp_and_order_are_rejected() {
        let reference = track("reference", &[Some(100.0)]);
        let mut estimated = track("estimated", &[Some(100.0)]);
        estimated.frames[0].f0_hz = Some(0.0);
        assert!(
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).is_err()
        );

        let mut estimated = track("estimated", &[Some(100.0)]);
        estimated.frames[0].periodicity = Some(1.1);
        assert!(
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).is_err()
        );
    }

    #[test]
    fn all_missed_voiced_frames_produce_zero_recall_and_f1() {
        let reference = track("reference", &[Some(100.0), Some(110.0)]);
        let mut estimated = track("estimated", &[Some(100.0), Some(110.0)]);
        estimated.frames.clear();
        let metrics =
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).unwrap();
        assert_eq!(metrics.voicing_precision, Some(0.0));
        assert_eq!(metrics.voicing_recall, Some(0.0));
        assert_eq!(metrics.voicing_f1, Some(0.0));
        assert_eq!(metrics.frame_coverage, Some(0.0));
        assert_eq!(metrics.f0_rmse_hz, None);
    }

    #[test]
    fn non_monotonic_and_out_of_interval_timestamps_are_rejected() {
        let reference = track("reference", &[Some(100.0), Some(110.0)]);
        let mut estimated = track("estimated", &[Some(100.0), Some(110.0)]);
        estimated.frames[1].source_timestamp_ns = estimated.frames[0].source_timestamp_ns;
        assert!(
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).is_err()
        );

        let mut estimated = track("estimated", &[Some(100.0)]);
        estimated.frames[0].source_timestamp_ns = estimated.source.source_end_ns;
        assert!(
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).is_err()
        );
    }

    #[test]
    fn non_estimable_pitch_metrics_are_none_not_zero() {
        let reference = track("reference", &[None, None]);
        let estimated = track("estimated", &[None, None]);
        let metrics =
            score_pitch_tracks(&reference, &estimated, PitchTrackScoreConfigV1::default()).unwrap();
        assert_eq!(metrics.voicing_f1, None);
        assert_eq!(metrics.f0_rmse_hz, None);
        assert_eq!(metrics.gross_pitch_error_rate, None);
        assert_eq!(metrics.octave_error_rate, None);
        assert_eq!(metrics.unvoiced_true_negative, 2);
    }
}
