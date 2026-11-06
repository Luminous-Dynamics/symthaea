//! Paired probabilistic forecast skill versus simple, source-bound baselines.
//!
//! Candidate and baseline forecasts must target the same series/Trading Periods,
//! use the same historical decision timestamps, and be scored against identical
//! realized observations and source receipts. The baseline model must itself
//! be built only from data available at each decision time. This module compares
//! supplied forecasts; it does not construct persistence/seasonal/climatology
//! baselines or execute vendor models.

use crate::forecast_metrics::{
    ForecastCalibrationError, ForecastCalibrationObservation, evaluate_forecast_calibration,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastSkillError {
    EmptyCandidateSet,
    EmptyBaselineSet,
    TargetCountMismatch,
    TargetMissingFromBaseline,
    DuplicateTarget,
    DecisionTimestampMismatch,
    RealizedOutcomeMismatch,
    ModelIdentityNotDistinct,
    CandidateCalibrationInvalid(ForecastCalibrationError),
    BaselineCalibrationInvalid(ForecastCalibrationError),
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactLossSkill {
    /// (baseline loss - candidate loss) / baseline loss.
    /// Positive numerator means candidate improvement; negative means regression.
    Defined {
        improvement_numerator: i128,
        baseline_loss_denominator: u128,
    },
    BothZero,
    BaselineZeroCandidatePositive,
}

impl ExactLossSkill {
    pub fn candidate_is_better(self) -> Option<bool> {
        match self {
            Self::Defined {
                improvement_numerator,
                ..
            } => Some(improvement_numerator > 0),
            Self::BothZero => Some(false),
            Self::BaselineZeroCandidatePositive => Some(false),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairedForecastLossDelta<'a> {
    pub series_id: &'a str,
    pub valid_period_start_utc: i64,
    pub valid_period_end_utc: i64,
    /// Baseline minus candidate. Positive means candidate has lower loss.
    pub median_mae_improvement_minor_zar_per_mwh: i128,
    /// Baseline minus candidate WIS numerator. Positive means candidate is better.
    pub weighted_interval_improvement_numerator: i128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForecastSkillAssessment<'a> {
    pub evaluation_as_of_utc: i64,
    pub candidate_model_id: &'a str,
    pub candidate_model_sha256: &'a str,
    pub baseline_model_id: &'a str,
    pub baseline_model_sha256: &'a str,
    pub paired_sample_count: u64,
    pub candidate_mae_numerator_minor_zar_per_mwh: u128,
    pub baseline_mae_numerator_minor_zar_per_mwh: u128,
    pub median_mae_skill: ExactLossSkill,
    pub candidate_wis_numerator: u128,
    pub baseline_wis_numerator: u128,
    pub wis_denominator: u128,
    pub weighted_interval_skill: ExactLossSkill,
    pub candidate_pinball_q10_numerator: i128,
    pub baseline_pinball_q10_numerator: i128,
    pub candidate_pinball_q50_numerator: i128,
    pub baseline_pinball_q50_numerator: i128,
    pub candidate_pinball_q90_numerator: i128,
    pub baseline_pinball_q90_numerator: i128,
    pub pinball_denominator: i128,
    pub candidate_central_80_covered_count: u64,
    pub baseline_central_80_covered_count: u64,
    /// Per-target paired loss differentials in candidate array order.
    pub paired_loss_deltas: Vec<PairedForecastLossDelta<'a>>,
}

/// Exact relative improvement for lower-is-better losses.
/// The denominator is the baseline loss numerator; the scale cancels because
/// candidate and baseline share the same sample count and metric denominator.
fn exact_skill(candidate: u128, baseline: u128) -> Result<ExactLossSkill, ForecastSkillError> {
    if baseline == 0 {
        return Ok(if candidate == 0 {
            ExactLossSkill::BothZero
        } else {
            ExactLossSkill::BaselineZeroCandidatePositive
        });
    }
    let improvement = i128::try_from(baseline)
        .map_err(|_| ForecastSkillError::ArithmeticOverflow)?
        .checked_sub(i128::try_from(candidate).map_err(|_| ForecastSkillError::ArithmeticOverflow)?)
        .ok_or(ForecastSkillError::ArithmeticOverflow)?;
    Ok(ExactLossSkill::Defined {
        improvement_numerator: improvement,
        baseline_loss_denominator: baseline,
    })
}

fn per_observation_wis_numerator(
    observation: ForecastCalibrationObservation<'_>,
) -> Result<u128, ForecastSkillError> {
    let actual = i128::from(observation.realized_minor_zar_per_mwh);
    let q10 = i128::from(observation.forecast_q10_minor_zar_per_mwh);
    let q50 = i128::from(observation.forecast_q50_minor_zar_per_mwh);
    let q90 = i128::from(observation.forecast_q90_minor_zar_per_mwh);
    let median_error = (actual - q50)
        .checked_abs()
        .ok_or(ForecastSkillError::ArithmeticOverflow)?;
    let interval_width =
        u128::try_from(q90 - q10).map_err(|_| ForecastSkillError::ArithmeticOverflow)?;
    let outside_distance = if actual < q10 {
        q10 - actual
    } else if actual > q90 {
        actual - q90
    } else {
        0
    };
    interval_width
        .checked_add(
            u128::try_from(outside_distance)
                .map_err(|_| ForecastSkillError::ArithmeticOverflow)?
                .checked_mul(10)
                .ok_or(ForecastSkillError::ArithmeticOverflow)?,
        )
        .and_then(|interval_score| {
            u128::try_from(median_error)
                .ok()
                .and_then(|error| error.checked_mul(5))
                .and_then(|weighted_median| interval_score.checked_add(weighted_median))
        })
        .ok_or(ForecastSkillError::ArithmeticOverflow)
}

/// Compare candidate and baseline quantile forecasts on an exact paired target
/// set. Pairing is by series + valid interval, not array position. The realized
/// value, realized source ref/digest, observation timestamp, and available-at
/// timestamp must be byte-for-byte equivalent across the pair.
pub fn compare_forecast_skill<'a>(
    evaluation_as_of_utc: i64,
    candidate: &[ForecastCalibrationObservation<'a>],
    baseline: &[ForecastCalibrationObservation<'a>],
) -> Result<ForecastSkillAssessment<'a>, ForecastSkillError> {
    if candidate.is_empty() {
        return Err(ForecastSkillError::EmptyCandidateSet);
    }
    if baseline.is_empty() {
        return Err(ForecastSkillError::EmptyBaselineSet);
    }
    if candidate.len() != baseline.len() {
        return Err(ForecastSkillError::TargetCountMismatch);
    }

    let candidate_first = candidate[0];
    let baseline_first = baseline[0];
    if candidate_first.model_id == baseline_first.model_id
        && candidate_first.model_sha256 == baseline_first.model_sha256
    {
        return Err(ForecastSkillError::ModelIdentityNotDistinct);
    }

    let mut paired_loss_deltas = Vec::with_capacity(candidate.len());
    for (index, item) in candidate.iter().copied().enumerate() {
        if candidate[..index].iter().any(|previous| {
            previous.series_id == item.series_id
                && previous.valid_period_start_utc == item.valid_period_start_utc
                && previous.valid_period_end_utc == item.valid_period_end_utc
        }) {
            return Err(ForecastSkillError::DuplicateTarget);
        }

        let mut matches = baseline.iter().copied().filter(|other| {
            other.series_id == item.series_id
                && other.valid_period_start_utc == item.valid_period_start_utc
                && other.valid_period_end_utc == item.valid_period_end_utc
        });
        let Some(other) = matches.next() else {
            return Err(ForecastSkillError::TargetMissingFromBaseline);
        };
        if matches.next().is_some() {
            return Err(ForecastSkillError::DuplicateTarget);
        }

        if item.decision_time_utc != other.decision_time_utc {
            return Err(ForecastSkillError::DecisionTimestampMismatch);
        }
        if item.realized_minor_zar_per_mwh != other.realized_minor_zar_per_mwh
            || item.realized_source_ref != other.realized_source_ref
            || item.realized_source_sha256 != other.realized_source_sha256
            || item.realized_observed_at_utc != other.realized_observed_at_utc
            || item.realized_available_at_utc != other.realized_available_at_utc
        {
            return Err(ForecastSkillError::RealizedOutcomeMismatch);
        }

        let actual = i128::from(item.realized_minor_zar_per_mwh);
        let candidate_q50 = i128::from(item.forecast_q50_minor_zar_per_mwh);
        let baseline_q50 = i128::from(other.forecast_q50_minor_zar_per_mwh);
        let candidate_abs_error = (actual - candidate_q50)
            .checked_abs()
            .ok_or(ForecastSkillError::ArithmeticOverflow)?;
        let baseline_abs_error = (actual - baseline_q50)
            .checked_abs()
            .ok_or(ForecastSkillError::ArithmeticOverflow)?;

        let candidate_wis = per_observation_wis_numerator(item)?;
        let baseline_wis = per_observation_wis_numerator(other)?;
        let mae_improvement = baseline_abs_error
            .checked_sub(candidate_abs_error)
            .ok_or(ForecastSkillError::ArithmeticOverflow)?;
        let wis_improvement = i128::try_from(baseline_wis)
            .map_err(|_| ForecastSkillError::ArithmeticOverflow)?
            .checked_sub(
                i128::try_from(candidate_wis)
                    .map_err(|_| ForecastSkillError::ArithmeticOverflow)?,
            )
            .ok_or(ForecastSkillError::ArithmeticOverflow)?;
        paired_loss_deltas.push(PairedForecastLossDelta {
            series_id: item.series_id,
            valid_period_start_utc: item.valid_period_start_utc,
            valid_period_end_utc: item.valid_period_end_utc,
            median_mae_improvement_minor_zar_per_mwh: mae_improvement,
            weighted_interval_improvement_numerator: wis_improvement,
        });
    }

    for item in baseline {
        if !candidate.iter().any(|other| {
            other.series_id == item.series_id
                && other.valid_period_start_utc == item.valid_period_start_utc
                && other.valid_period_end_utc == item.valid_period_end_utc
        }) {
            return Err(ForecastSkillError::TargetMissingFromBaseline);
        }
    }

    let candidate_metrics = evaluate_forecast_calibration(evaluation_as_of_utc, candidate)
        .map_err(ForecastSkillError::CandidateCalibrationInvalid)?;
    let baseline_metrics = evaluate_forecast_calibration(evaluation_as_of_utc, baseline)
        .map_err(ForecastSkillError::BaselineCalibrationInvalid)?;

    if candidate_metrics.sample_count != baseline_metrics.sample_count
        || candidate_metrics.pinball_denominator != baseline_metrics.pinball_denominator
        || candidate_metrics.weighted_interval_score_80_denominator
            != baseline_metrics.weighted_interval_score_80_denominator
    {
        return Err(ForecastSkillError::TargetCountMismatch);
    }

    Ok(ForecastSkillAssessment {
        evaluation_as_of_utc,
        candidate_model_id: candidate_metrics.model_id,
        candidate_model_sha256: candidate_metrics.model_sha256,
        baseline_model_id: baseline_metrics.model_id,
        baseline_model_sha256: baseline_metrics.model_sha256,
        paired_sample_count: candidate_metrics.sample_count,
        candidate_mae_numerator_minor_zar_per_mwh: candidate_metrics
            .absolute_error_sum_minor_zar_per_mwh,
        baseline_mae_numerator_minor_zar_per_mwh: baseline_metrics
            .absolute_error_sum_minor_zar_per_mwh,
        median_mae_skill: exact_skill(
            candidate_metrics.absolute_error_sum_minor_zar_per_mwh,
            baseline_metrics.absolute_error_sum_minor_zar_per_mwh,
        )?,
        candidate_wis_numerator: candidate_metrics.weighted_interval_score_80_numerator,
        baseline_wis_numerator: baseline_metrics.weighted_interval_score_80_numerator,
        wis_denominator: candidate_metrics.weighted_interval_score_80_denominator,
        weighted_interval_skill: exact_skill(
            candidate_metrics.weighted_interval_score_80_numerator,
            baseline_metrics.weighted_interval_score_80_numerator,
        )?,
        candidate_pinball_q10_numerator: candidate_metrics.pinball_q10_numerator,
        baseline_pinball_q10_numerator: baseline_metrics.pinball_q10_numerator,
        candidate_pinball_q50_numerator: candidate_metrics.pinball_q50_numerator,
        baseline_pinball_q50_numerator: baseline_metrics.pinball_q50_numerator,
        candidate_pinball_q90_numerator: candidate_metrics.pinball_q90_numerator,
        baseline_pinball_q90_numerator: baseline_metrics.pinball_q90_numerator,
        pinball_denominator: candidate_metrics.pinball_denominator,
        candidate_central_80_covered_count: candidate_metrics.central_80_covered_count,
        baseline_central_80_covered_count: baseline_metrics.central_80_covered_count,
        paired_loss_deltas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const CANDIDATE_SHA: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const BASELINE_SHA: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

    fn forecast(
        id: &'static str,
        period_start: i64,
        period_end: i64,
        model_id: &'static str,
        model_sha256: &'static str,
        q10: i64,
        q50: i64,
        q90: i64,
        actual: i64,
    ) -> ForecastCalibrationObservation<'static> {
        ForecastCalibrationObservation {
            record_id: id,
            forecast_source_ref: id,
            forecast_source_sha256: SOURCE_SHA,
            realized_source_ref: "market/realized/SMP",
            realized_source_sha256: SOURCE_SHA,
            series_id: "SAWEM/SMP",
            model_id,
            model_sha256,
            decision_time_utc: period_start - 60,
            forecast_issued_at_utc: period_start - 120,
            forecast_available_at_utc: period_start - 90,
            valid_period_start_utc: period_start,
            valid_period_end_utc: period_end,
            forecast_q10_minor_zar_per_mwh: q10,
            forecast_q50_minor_zar_per_mwh: q50,
            forecast_q90_minor_zar_per_mwh: q90,
            realized_minor_zar_per_mwh: actual,
            realized_observed_at_utc: period_end + 5,
            realized_available_at_utc: period_end + 10,
        }
    }

    fn paired() -> (
        [ForecastCalibrationObservation<'static>; 2],
        [ForecastCalibrationObservation<'static>; 2],
    ) {
        (
            [
                forecast(
                    "candidate-1",
                    100,
                    110,
                    "candidate-v1",
                    CANDIDATE_SHA,
                    90,
                    100,
                    110,
                    105,
                ),
                forecast(
                    "candidate-2",
                    200,
                    210,
                    "candidate-v1",
                    CANDIDATE_SHA,
                    190,
                    200,
                    210,
                    205,
                ),
            ],
            [
                forecast(
                    "baseline-1",
                    100,
                    110,
                    "persistence-v1",
                    BASELINE_SHA,
                    70,
                    90,
                    110,
                    105,
                ),
                forecast(
                    "baseline-2",
                    200,
                    210,
                    "persistence-v1",
                    BASELINE_SHA,
                    170,
                    190,
                    210,
                    205,
                ),
            ],
        )
    }

    #[test]
    fn reports_exact_paired_mae_and_interval_skill_against_persistence() {
        let (candidate, baseline) = paired();
        let result = compare_forecast_skill(300, &candidate, &baseline).unwrap();
        assert_eq!(result.paired_sample_count, 2);
        assert_eq!(result.candidate_mae_numerator_minor_zar_per_mwh, 10);
        assert_eq!(result.baseline_mae_numerator_minor_zar_per_mwh, 30);
        assert_eq!(
            result.median_mae_skill,
            ExactLossSkill::Defined {
                improvement_numerator: 20,
                baseline_loss_denominator: 30,
            }
        );
        assert_eq!(
            result.weighted_interval_skill.candidate_is_better(),
            Some(true)
        );
    }

    #[test]
    fn candidate_regression_is_a_negative_skill_score_not_hidden_by_absolute_value() {
        let (mut candidate, baseline) = paired();
        candidate[0].forecast_q10_minor_zar_per_mwh = 130;
        candidate[0].forecast_q50_minor_zar_per_mwh = 145;
        candidate[0].forecast_q90_minor_zar_per_mwh = 160;
        candidate[1].forecast_q10_minor_zar_per_mwh = 230;
        candidate[1].forecast_q50_minor_zar_per_mwh = 245;
        candidate[1].forecast_q90_minor_zar_per_mwh = 260;
        let result = compare_forecast_skill(300, &candidate, &baseline).unwrap();
        assert_eq!(
            result.median_mae_skill,
            ExactLossSkill::Defined {
                improvement_numerator: -50,
                baseline_loss_denominator: 30,
            }
        );
        assert_eq!(result.median_mae_skill.candidate_is_better(), Some(false));
    }

    #[test]
    fn different_actuals_or_decision_frontiers_cannot_be_compared() {
        let (candidate, mut baseline) = paired();
        baseline[0].realized_minor_zar_per_mwh = 106;
        assert_eq!(
            compare_forecast_skill(300, &candidate, &baseline),
            Err(ForecastSkillError::RealizedOutcomeMismatch)
        );

        let (candidate, mut baseline) = paired();
        baseline[0].decision_time_utc -= 1;
        assert_eq!(
            compare_forecast_skill(300, &candidate, &baseline),
            Err(ForecastSkillError::DecisionTimestampMismatch)
        );
    }

    #[test]
    fn mismatched_targets_and_duplicate_periods_are_rejected() {
        let (candidate, mut baseline) = paired();
        baseline[0].valid_period_start_utc = 101;
        assert_eq!(
            compare_forecast_skill(300, &candidate, &baseline),
            Err(ForecastSkillError::TargetMissingFromBaseline)
        );

        let candidate = [
            candidate[0],
            ForecastCalibrationObservation {
                record_id: "candidate-duplicate",
                ..candidate[0]
            },
        ];
        let baseline = [
            baseline[0],
            ForecastCalibrationObservation {
                valid_period_start_utc: 100,
                valid_period_end_utc: 110,
                record_id: "baseline-duplicate",
                ..baseline[0]
            },
        ];
        assert_eq!(
            compare_forecast_skill(300, &candidate, &baseline),
            Err(ForecastSkillError::DuplicateTarget)
        );
    }

    #[test]
    fn exact_model_identity_and_zero_baseline_cases_are_explicit() {
        assert_eq!(
            compare_forecast_skill(300, &paired().0, &paired().0),
            Err(ForecastSkillError::ModelIdentityNotDistinct)
        );
        assert_eq!(exact_skill(0, 0), Ok(ExactLossSkill::BothZero));
        assert_eq!(
            exact_skill(1, 0),
            Ok(ExactLossSkill::BaselineZeroCandidatePositive)
        );
    }
}
