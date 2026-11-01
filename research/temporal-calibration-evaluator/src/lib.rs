//! Evaluator-only temporal action-value calibration diagnostics.
//!
//! Fit corrections use only finalized feedback from strictly earlier episodes.
//! This crate must not be linked into candidate execution: it consumes realized
//! evaluator outcomes and is not a candidate feature source.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExposureValues {
    pub wrong_belief_us: i64,
    pub abstention_us: i64,
    pub missing_output_us: i64,
}

impl ExposureValues {
    const fn as_array(self) -> [i64; 3] {
        [self.wrong_belief_us, self.abstention_us, self.missing_output_us]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationSample {
    /// Strictly increasing sequence assigned by the evaluator's finalized
    /// receipt stream, not by the candidate.
    pub feedback_sequence: u64,
    pub episode_id: u64,
    pub model_sha256: [u8; 32],
    /// Commitment to the evaluator-validated treatment/control pair and
    /// matched episode context.
    pub paired_intervention_sha256: [u8; 32],
    /// Digest of the finalized receipt whose outcome became eligible for use.
    pub finalized_receipt_sha256: [u8; 32],
    pub forecast_issued_at_us: u64,
    /// Time at which the outcome became eligible for release.
    pub feedback_available_at_us: u64,
    /// Candidate-estimated reduction relative to its declared paired baseline.
    pub predicted_reduction: ExposureValues,
    /// Evaluator-measured reduction from a valid paired intervention.
    pub realized_reduction: ExposureValues,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExposureAxis {
    WrongBelief,
    Abstention,
    MissingOutput,
}

impl ExposureAxis {
    const ALL: [Self; 3] = [
        Self::WrongBelief,
        Self::Abstention,
        Self::MissingOutput,
    ];

    const fn index(self) -> usize {
        match self {
            Self::WrongBelief => 0,
            Self::Abstention => 1,
            Self::MissingOutput => 2,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AxisCalibrationReport {
    pub axis: ExposureAxis,
    pub fit_sample_count: usize,
    pub holdout_sample_count: usize,
    /// Mean realized-minus-predicted error on the fit partition.
    pub fitted_bias_correction_us: i64,
    pub raw_holdout_bias_us: i64,
    pub calibrated_holdout_bias_us: i64,
    pub raw_holdout_mae_us: u64,
    pub calibrated_holdout_mae_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationAuditReport {
    pub model_sha256: [u8; 32],
    pub fit_first_sequence: u64,
    pub fit_last_sequence: u64,
    pub holdout_first_sequence: u64,
    pub holdout_last_sequence: u64,
    pub fit_feedback_available_through_us: u64,
    pub holdout_forecasts_begin_at_us: u64,
    pub axes: [AxisCalibrationReport; 3],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CalibrationError {
    EmptyFitPartition,
    EmptyHoldoutPartition,
    ModelDigestMismatch { sequence: u64 },
    MissingEvidenceCommitment { sequence: u64 },
    InvalidFeedbackTimestamp { sequence: u64 },
    NonIncreasingSequence { partition: &'static str, sequence: u64 },
    NonIncreasingForecastTime { partition: &'static str, sequence: u64 },
    DuplicateEpisode { episode_id: u64 },
    DuplicateEvidenceCommitment { sequence: u64 },
    TemporalLeakage {
        fit_feedback_available_through_us: u64,
        holdout_forecasts_begin_at_us: u64,
    },
    ArithmeticOverflow,
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFitPartition => write!(f, "calibration fit partition is empty"),
            Self::EmptyHoldoutPartition => write!(f, "calibration holdout partition is empty"),
            Self::ModelDigestMismatch { sequence } => write!(f, "model digest mismatch at feedback sequence {sequence}"),
            Self::MissingEvidenceCommitment { sequence } => write!(f, "paired intervention or finalized receipt commitment is missing at feedback sequence {sequence}"),
            Self::InvalidFeedbackTimestamp { sequence } => write!(f, "feedback at sequence {sequence} was available before its forecast was issued"),
            Self::NonIncreasingSequence { partition, sequence } => write!(f, "{partition} feedback sequence is not strictly increasing at {sequence}"),
            Self::NonIncreasingForecastTime { partition, sequence } => write!(f, "{partition} forecast issue time is not strictly increasing at sequence {sequence}"),
            Self::DuplicateEpisode { episode_id } => write!(f, "episode {episode_id} appears more than once in calibration partitions"),
            Self::DuplicateEvidenceCommitment { sequence } => write!(f, "paired intervention or finalized receipt commitment is reused at feedback sequence {sequence}"),
            Self::TemporalLeakage { fit_feedback_available_through_us, holdout_forecasts_begin_at_us } => write!(f, "calibration feedback through {fit_feedback_available_through_us} is not strictly earlier than holdout forecast origin {holdout_forecasts_begin_at_us}"),
            Self::ArithmeticOverflow => write!(f, "calibration audit arithmetic overflowed"),
        }
    }
}

impl std::error::Error for CalibrationError {}

/// Fit per-axis additive bias on earlier finalized episodes and report raw vs
/// bias-corrected metrics on later held-out forecasts. MAE is descriptive; it
/// is not a confidence interval, probability calibration score or causal claim.
pub fn audit_temporal_value_calibration(
    expected_model_sha256: [u8; 32],
    fit_samples: &[CalibrationSample],
    holdout_samples: &[CalibrationSample],
) -> Result<CalibrationAuditReport, CalibrationError> {
    if fit_samples.is_empty() {
        return Err(CalibrationError::EmptyFitPartition);
    }
    if holdout_samples.is_empty() {
        return Err(CalibrationError::EmptyHoldoutPartition);
    }

    validate_partition(fit_samples, expected_model_sha256, "fit")?;
    validate_partition(holdout_samples, expected_model_sha256, "holdout")?;

    let mut episode_ids = BTreeSet::new();
    let mut pair_commitments = BTreeSet::new();
    let mut receipt_commitments = BTreeSet::new();
    for sample in fit_samples.iter().chain(holdout_samples.iter()) {
        if !episode_ids.insert(sample.episode_id) {
            return Err(CalibrationError::DuplicateEpisode {
                episode_id: sample.episode_id,
            });
        }
        if !pair_commitments.insert(sample.paired_intervention_sha256)
            || !receipt_commitments.insert(sample.finalized_receipt_sha256)
        {
            return Err(CalibrationError::DuplicateEvidenceCommitment {
                sequence: sample.feedback_sequence,
            });
        }
    }

    let fit_last = fit_samples.last().expect("non-empty fit partition");
    let holdout_first = holdout_samples.first().expect("non-empty holdout partition");
    if fit_last.feedback_sequence >= holdout_first.feedback_sequence {
        return Err(CalibrationError::NonIncreasingSequence {
            partition: "cross-partition",
            sequence: holdout_first.feedback_sequence,
        });
    }

    let fit_feedback_available_through_us = fit_samples.iter()
        .map(|sample| sample.feedback_available_at_us)
        .max()
        .ok_or(CalibrationError::EmptyFitPartition)?;
    let holdout_forecasts_begin_at_us = holdout_samples.iter()
        .map(|sample| sample.forecast_issued_at_us)
        .min()
        .ok_or(CalibrationError::EmptyHoldoutPartition)?;
    if fit_feedback_available_through_us >= holdout_forecasts_begin_at_us {
        return Err(CalibrationError::TemporalLeakage {
            fit_feedback_available_through_us,
            holdout_forecasts_begin_at_us,
        });
    }

    let [wrong_axis, abstention_axis, missing_axis] = ExposureAxis::ALL;
    let axes = [
        audit_axis(fit_samples, holdout_samples, wrong_axis)?,
        audit_axis(fit_samples, holdout_samples, abstention_axis)?,
        audit_axis(fit_samples, holdout_samples, missing_axis)?,
    ];

    Ok(CalibrationAuditReport {
        model_sha256: expected_model_sha256,
        fit_first_sequence: fit_samples.first().expect("non-empty fit partition").feedback_sequence,
        fit_last_sequence: fit_last.feedback_sequence,
        holdout_first_sequence: holdout_first.feedback_sequence,
        holdout_last_sequence: holdout_samples.last().expect("non-empty holdout partition").feedback_sequence,
        fit_feedback_available_through_us,
        holdout_forecasts_begin_at_us,
        axes,
    })
}

fn validate_partition(
    samples: &[CalibrationSample],
    expected_model_sha256: [u8; 32],
    partition: &'static str,
) -> Result<(), CalibrationError> {
    for (index, sample) in samples.iter().enumerate() {
        if sample.model_sha256 != expected_model_sha256 {
            return Err(CalibrationError::ModelDigestMismatch {
                sequence: sample.feedback_sequence,
            });
        }
        if sample.paired_intervention_sha256 == [0; 32]
            || sample.finalized_receipt_sha256 == [0; 32]
        {
            return Err(CalibrationError::MissingEvidenceCommitment {
                sequence: sample.feedback_sequence,
            });
        }
        if sample.feedback_available_at_us <= sample.forecast_issued_at_us {
            return Err(CalibrationError::InvalidFeedbackTimestamp {
                sequence: sample.feedback_sequence,
            });
        }
        if index > 0 {
            let previous = &samples[index - 1];
            if sample.feedback_sequence <= previous.feedback_sequence {
                return Err(CalibrationError::NonIncreasingSequence {
                    partition,
                    sequence: sample.feedback_sequence,
                });
            }
            if sample.forecast_issued_at_us <= previous.forecast_issued_at_us {
                return Err(CalibrationError::NonIncreasingForecastTime {
                    partition,
                    sequence: sample.feedback_sequence,
                });
            }
        }
    }
    Ok(())
}

fn audit_axis(
    fit: &[CalibrationSample],
    holdout: &[CalibrationSample],
    axis: ExposureAxis,
) -> Result<AxisCalibrationReport, CalibrationError> {
    let index = axis.index();
    let fit_bias = mean_error(fit, index, 0)?;
    let raw_holdout = holdout_metrics(holdout, index, 0)?;
    let calibrated_holdout = holdout_metrics(holdout, index, fit_bias)?;

    Ok(AxisCalibrationReport {
        axis,
        fit_sample_count: fit.len(),
        holdout_sample_count: holdout.len(),
        fitted_bias_correction_us: fit_bias,
        raw_holdout_bias_us: raw_holdout.mean_bias_us,
        calibrated_holdout_bias_us: calibrated_holdout.mean_bias_us,
        raw_holdout_mae_us: raw_holdout.mean_absolute_error_us,
        calibrated_holdout_mae_us: calibrated_holdout.mean_absolute_error_us,
    })
}

#[derive(Clone, Copy)]
struct HoldoutMetrics {
    mean_bias_us: i64,
    mean_absolute_error_us: u64,
}

fn mean_error(
    samples: &[CalibrationSample],
    axis_index: usize,
    correction_us: i64,
) -> Result<i64, CalibrationError> {
    let mut sum = 0_i128;
    for sample in samples {
        let predicted = i128::from(sample.predicted_reduction.as_array()[axis_index])
            .checked_add(i128::from(correction_us))
            .ok_or(CalibrationError::ArithmeticOverflow)?;
        let realized = i128::from(sample.realized_reduction.as_array()[axis_index]);
        let error = realized.checked_sub(predicted).ok_or(CalibrationError::ArithmeticOverflow)?;
        sum = sum.checked_add(error).ok_or(CalibrationError::ArithmeticOverflow)?;
    }
    let count = i128::try_from(samples.len()).map_err(|_| CalibrationError::ArithmeticOverflow)?;
    let mean = sum / count;
    i64::try_from(mean).map_err(|_| CalibrationError::ArithmeticOverflow)
}

fn holdout_metrics(
    samples: &[CalibrationSample],
    axis_index: usize,
    correction_us: i64,
) -> Result<HoldoutMetrics, CalibrationError> {
    let mut bias_sum = 0_i128;
    let mut absolute_error_sum = 0_u128;
    for sample in samples {
        let predicted = i128::from(sample.predicted_reduction.as_array()[axis_index])
            .checked_add(i128::from(correction_us))
            .ok_or(CalibrationError::ArithmeticOverflow)?;
        let realized = i128::from(sample.realized_reduction.as_array()[axis_index]);
        let error = realized.checked_sub(predicted).ok_or(CalibrationError::ArithmeticOverflow)?;
        bias_sum = bias_sum.checked_add(error).ok_or(CalibrationError::ArithmeticOverflow)?;
        absolute_error_sum = absolute_error_sum
            .checked_add(error.unsigned_abs())
            .ok_or(CalibrationError::ArithmeticOverflow)?;
    }
    let count = i128::try_from(samples.len()).map_err(|_| CalibrationError::ArithmeticOverflow)?;
    let mean_bias = bias_sum / count;
    let mean_absolute = absolute_error_sum
        / u128::try_from(samples.len()).map_err(|_| CalibrationError::ArithmeticOverflow)?;
    Ok(HoldoutMetrics {
        mean_bias_us: i64::try_from(mean_bias).map_err(|_| CalibrationError::ArithmeticOverflow)?,
        mean_absolute_error_us: u64::try_from(mean_absolute)
            .map_err(|_| CalibrationError::ArithmeticOverflow)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(
        sequence: u64,
        episode_id: u64,
        issued: u64,
        available: u64,
        predicted: ExposureValues,
        realized: ExposureValues,
    ) -> CalibrationSample {
        CalibrationSample {
            feedback_sequence: sequence,
            episode_id,
            model_sha256: [7; 32],
            paired_intervention_sha256: [sequence as u8; 32],
            finalized_receipt_sha256: [episode_id as u8; 32],
            forecast_issued_at_us: issued,
            feedback_available_at_us: available,
            predicted_reduction: predicted,
            realized_reduction: realized,
        }
    }

    fn values(wrong: i64, abstention: i64, missing: i64) -> ExposureValues {
        ExposureValues {
            wrong_belief_us: wrong,
            abstention_us: abstention,
            missing_output_us: missing,
        }
    }

    fn partitions() -> (Vec<CalibrationSample>, Vec<CalibrationSample>) {
        let fit = vec![
            sample(1, 1, 10, 20, values(10, 5, 2), values(14, 3, 4)),
            sample(2, 2, 30, 40, values(10, 5, 2), values(14, 3, 4)),
        ];
        let holdout = vec![
            sample(3, 3, 50, 60, values(20, 8, 1), values(24, 6, 3)),
            sample(4, 4, 70, 80, values(30, 9, 2), values(34, 7, 4)),
        ];
        (fit, holdout)
    }

    #[test]
    fn fit_bias_is_evaluated_only_on_strictly_later_feedback() {
        let (fit, holdout) = partitions();
        let report = audit_temporal_value_calibration([7; 32], &fit, &holdout).unwrap();
        assert_eq!(report.fit_feedback_available_through_us, 40);
        assert_eq!(report.holdout_forecasts_begin_at_us, 50);
        assert_eq!(report.axes[0].fitted_bias_correction_us, 4);
        assert_eq!(report.axes[0].raw_holdout_mae_us, 4);
        assert_eq!(report.axes[0].calibrated_holdout_mae_us, 0);
        assert_eq!(report.axes[1].fitted_bias_correction_us, -2);
        assert_eq!(report.axes[1].calibrated_holdout_mae_us, 0);
    }

    #[test]
    fn current_or_future_feedback_is_rejected_as_temporal_leakage() {
        let (fit, mut holdout) = partitions();
        holdout[0].forecast_issued_at_us = 40;
        assert!(matches!(
            audit_temporal_value_calibration([7; 32], &fit, &holdout),
            Err(CalibrationError::TemporalLeakage { .. })
        ));
    }

    #[test]
    fn model_digest_must_match_across_both_partitions() {
        let (fit, mut holdout) = partitions();
        holdout[0].model_sha256 = [9; 32];
        assert!(matches!(
            audit_temporal_value_calibration([7; 32], &fit, &holdout),
            Err(CalibrationError::ModelDigestMismatch { sequence: 3 })
        ));
    }

    #[test]
    fn missing_pair_or_receipt_commitment_is_rejected() {
        let (mut fit, holdout) = partitions();
        fit[0].paired_intervention_sha256 = [0; 32];
        assert_eq!(
            audit_temporal_value_calibration([7; 32], &fit, &holdout),
            Err(CalibrationError::MissingEvidenceCommitment { sequence: 1 })
        );
    }

    #[test]
    fn outcome_must_become_available_after_its_forecast() {
        let (mut fit, holdout) = partitions();
        fit[0].feedback_available_at_us = fit[0].forecast_issued_at_us;
        assert!(matches!(
            audit_temporal_value_calibration([7; 32], &fit, &holdout),
            Err(CalibrationError::InvalidFeedbackTimestamp { sequence: 1 })
        ));
    }

    #[test]
    fn out_of_order_and_duplicate_sequences_are_rejected() {
        let (mut fit, holdout) = partitions();
        fit[1].feedback_sequence = 1;
        assert!(matches!(
            audit_temporal_value_calibration([7; 32], &fit, &holdout),
            Err(CalibrationError::NonIncreasingSequence { .. })
        ));
    }

    #[test]
    fn duplicate_episode_ids_are_rejected() {
        let (fit, mut holdout) = partitions();
        holdout[0].episode_id = fit[0].episode_id;
        assert_eq!(
            audit_temporal_value_calibration([7; 32], &fit, &holdout),
            Err(CalibrationError::DuplicateEpisode { episode_id: 1 })
        );
    }

    #[test]
    fn repeated_finalized_receipt_or_pair_commitment_is_rejected() {
        let (fit, mut holdout) = partitions();
        holdout[0].finalized_receipt_sha256 = fit[0].finalized_receipt_sha256;
        assert_eq!(
            audit_temporal_value_calibration([7; 32], &fit, &holdout),
            Err(CalibrationError::DuplicateEvidenceCommitment { sequence: 3 })
        );
    }

    #[test]
    fn empty_partitions_fail_closed() {
        let (fit, holdout) = partitions();
        assert_eq!(
            audit_temporal_value_calibration([7; 32], &[], &holdout),
            Err(CalibrationError::EmptyFitPartition)
        );
        assert_eq!(
            audit_temporal_value_calibration([7; 32], &fit, &[]),
            Err(CalibrationError::EmptyHoldoutPartition)
        );
    }

    #[test]
    fn calibration_reports_can_show_regression_instead_of_hiding_it() {
        let fit = vec![
            sample(1, 1, 10, 20, values(10, 5, 2), values(15, 5, 2)),
            sample(2, 2, 30, 40, values(10, 5, 2), values(15, 5, 2)),
        ];
        let holdout = vec![
            sample(3, 3, 50, 60, values(20, 8, 1), values(10, 8, 1)),
            sample(4, 4, 70, 80, values(30, 9, 2), values(20, 9, 2)),
        ];
        let report = audit_temporal_value_calibration([7; 32], &fit, &holdout).unwrap();
        assert!(report.axes[0].calibrated_holdout_mae_us
            > report.axes[0].raw_holdout_mae_us);
    }
}
