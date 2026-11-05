//! Leakage-safe, exact-accumulator forecast calibration scoring.
//!
//! This module scores a single immutable model build against observations that
//! became available after the historical decision time. It does not generate
//! forecasts, authenticate feeds, or imply that synthetic tests establish
//! calibration. Callers must split by project/time before evaluation and must
//! evaluate each model build separately.

const BASIS_POINTS: i128 = 10_000;
const Q10_BPS: i128 = 1_000;
const Q50_BPS: i128 = 5_000;
const Q90_BPS: i128 = 9_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForecastCalibrationObservation<'a> {
    pub record_id: &'a str,
    pub forecast_source_ref: &'a str,
    pub forecast_source_sha256: &'a str,
    pub realized_source_ref: &'a str,
    pub realized_source_sha256: &'a str,
    pub series_id: &'a str,
    pub model_id: &'a str,
    pub model_sha256: &'a str,
    /// Frozen historical decision timestamp; the forecast was available then.
    pub decision_time_utc: i64,
    pub forecast_issued_at_utc: i64,
    pub forecast_available_at_utc: i64,
    pub valid_period_start_utc: i64,
    pub valid_period_end_utc: i64,
    /// Forecast quantiles in minor ZAR/MWh.
    pub forecast_q10_minor_zar_per_mwh: i64,
    pub forecast_q50_minor_zar_per_mwh: i64,
    pub forecast_q90_minor_zar_per_mwh: i64,
    /// Realized price in minor ZAR/MWh.
    pub realized_minor_zar_per_mwh: i64,
    pub realized_observed_at_utc: i64,
    pub realized_available_at_utc: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastCalibrationError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidModelDigest,
    InvalidPeriod,
    ForecastWasNotAvailableAtDecision,
    DecisionAfterTargetStart,
    RealizedObservationBeforePeriodEnd,
    RealizedObservationNotAvailableAtEvaluation,
    QuantileCrossing,
    MixedModelBuild,
    DuplicateRecordId,
    DuplicateTargetPeriod,
    ArithmeticOverflow,
    EmptyEvaluationSet,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForecastCalibrationMetrics<'a> {
    pub model_id: &'a str,
    pub model_sha256: &'a str,
    pub evaluation_as_of_utc: i64,
    pub sample_count: u64,
    /// Sum of |actual - q50|; divide by sample_count for MAE.
    pub absolute_error_sum_minor_zar_per_mwh: u128,
    /// Sum of actual - q50; divide by sample_count for signed bias.
    pub signed_error_sum_minor_zar_per_mwh: i128,
    /// Sum of squared median errors; divide by sample_count and sqrt for RMSE.
    pub squared_error_sum_minor_zar2_per_mwh2: u128,
    /// Exact pinball-loss numerators, each with denominator BASIS_POINTS.
    pub pinball_q10_numerator: i128,
    pub pinball_q50_numerator: i128,
    pub pinball_q90_numerator: i128,
    pub pinball_denominator: i128,
    pub actual_at_or_below_q10_count: u64,
    pub actual_at_or_below_q50_count: u64,
    pub actual_at_or_below_q90_count: u64,
    pub central_80_covered_count: u64,
    pub below_q10_count: u64,
    pub above_q90_count: u64,
    /// Signed calibration error: observed_count * 10,000 - n * nominal_quantile_bps.
    /// Divide by n * 10,000 to express coverage error as a fraction.
    pub q10_coverage_error_numerator: i128,
    pub q50_coverage_error_numerator: i128,
    pub q90_coverage_error_numerator: i128,
    /// Sum of q90-q10 interval widths.
    pub central_80_interval_width_sum_minor_zar_per_mwh: u128,
    /// Weighted interval score for median + central 80% interval:
    /// sum(5*|actual-q50| + interval_score_80), denominator = 15*n.
    pub weighted_interval_score_80_numerator: u128,
    pub weighted_interval_score_80_denominator: u128,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn abs_i128(value: i128) -> Result<u128, ForecastCalibrationError> {
    u128::try_from(value.checked_abs().ok_or(ForecastCalibrationError::ArithmeticOverflow)?)
        .map_err(|_| ForecastCalibrationError::ArithmeticOverflow)
}

fn pinball_numerator(error: i128, quantile_bps: i128) -> Result<i128, ForecastCalibrationError> {
    let weight = if error >= 0 {
        quantile_bps
    } else {
        BASIS_POINTS - quantile_bps
    };
    error
        .checked_abs()
        .and_then(|magnitude| magnitude.checked_mul(weight))
        .ok_or(ForecastCalibrationError::ArithmeticOverflow)
}

fn validate_observation(
    observation: ForecastCalibrationObservation<'_>,
    evaluation_as_of_utc: i64,
) -> Result<(), ForecastCalibrationError> {
    if !nonempty(observation.record_id)
        || !nonempty(observation.forecast_source_ref)
        || !nonempty(observation.realized_source_ref)
        || !nonempty(observation.series_id)
        || !nonempty(observation.model_id)
    {
        return Err(ForecastCalibrationError::InvalidEvidenceReference);
    }
    if !valid_sha256(observation.forecast_source_sha256)
        || !valid_sha256(observation.realized_source_sha256)
    {
        return Err(ForecastCalibrationError::InvalidSourceDigest);
    }
    if !valid_sha256(observation.model_sha256) {
        return Err(ForecastCalibrationError::InvalidModelDigest);
    }
    if observation.valid_period_start_utc >= observation.valid_period_end_utc {
        return Err(ForecastCalibrationError::InvalidPeriod);
    }
    if observation.forecast_issued_at_utc > observation.forecast_available_at_utc
        || observation.forecast_available_at_utc > observation.decision_time_utc
    {
        return Err(ForecastCalibrationError::ForecastWasNotAvailableAtDecision);
    }
    if observation.decision_time_utc > observation.valid_period_start_utc {
        return Err(ForecastCalibrationError::DecisionAfterTargetStart);
    }
    if observation.realized_observed_at_utc < observation.valid_period_end_utc {
        return Err(ForecastCalibrationError::RealizedObservationBeforePeriodEnd);
    }
    if observation.realized_available_at_utc < observation.realized_observed_at_utc
        || observation.realized_available_at_utc > evaluation_as_of_utc
    {
        return Err(ForecastCalibrationError::RealizedObservationNotAvailableAtEvaluation);
    }
    if observation.forecast_q10_minor_zar_per_mwh
        > observation.forecast_q50_minor_zar_per_mwh
        || observation.forecast_q50_minor_zar_per_mwh
            > observation.forecast_q90_minor_zar_per_mwh
    {
        return Err(ForecastCalibrationError::QuantileCrossing);
    }
    Ok(())
}

/// Score a single model build against realized outcomes at a later evaluation
/// timestamp. Quantile forecasts must have been available by each decision
/// timestamp; realized outcomes must have been available by evaluation time.
/// Exact integer accumulators retain auditability and avoid floating-point
/// rounding in the benchmark receipt.
pub fn evaluate_forecast_calibration<'a>(
    evaluation_as_of_utc: i64,
    observations: &[ForecastCalibrationObservation<'a>],
) -> Result<ForecastCalibrationMetrics<'a>, ForecastCalibrationError> {
    if observations.is_empty() {
        return Err(ForecastCalibrationError::EmptyEvaluationSet);
    }

    let first = observations[0];
    let model_id = first.model_id;
    let model_sha256 = first.model_sha256;

    let mut absolute_error_sum = 0_u128;
    let mut signed_error_sum = 0_i128;
    let mut squared_error_sum = 0_u128;
    let mut pinball_q10 = 0_i128;
    let mut pinball_q50 = 0_i128;
    let mut pinball_q90 = 0_i128;
    let mut at_or_below_q10 = 0_u64;
    let mut at_or_below_q50 = 0_u64;
    let mut at_or_below_q90 = 0_u64;
    let mut central_80_covered = 0_u64;
    let mut below_q10 = 0_u64;
    let mut above_q90 = 0_u64;
    let mut interval_width_sum = 0_u128;
    let mut wis_numerator = 0_u128;

    for (index, observation) in observations.iter().copied().enumerate() {
        validate_observation(observation, evaluation_as_of_utc)?;
        if observation.model_id != model_id || observation.model_sha256 != model_sha256 {
            return Err(ForecastCalibrationError::MixedModelBuild);
        }
        for previous in &observations[..index] {
            if previous.record_id == observation.record_id {
                return Err(ForecastCalibrationError::DuplicateRecordId);
            }
            if previous.series_id == observation.series_id
                && previous.valid_period_start_utc == observation.valid_period_start_utc
                && previous.valid_period_end_utc == observation.valid_period_end_utc
            {
                return Err(ForecastCalibrationError::DuplicateTargetPeriod);
            }
        }

        let actual = i128::from(observation.realized_minor_zar_per_mwh);
        let q10 = i128::from(observation.forecast_q10_minor_zar_per_mwh);
        let q50 = i128::from(observation.forecast_q50_minor_zar_per_mwh);
        let q90 = i128::from(observation.forecast_q90_minor_zar_per_mwh);
        let error = actual - q50;
        let absolute_error = abs_i128(error)?;
        let squared_error = absolute_error
            .checked_mul(absolute_error)
            .ok_or(ForecastCalibrationError::ArithmeticOverflow)?;
        absolute_error_sum = absolute_error_sum
            .checked_add(absolute_error)
            .ok_or(ForecastCalibrationError::ArithmeticOverflow)?;
        signed_error_sum = signed_error_sum
            .checked_add(error)
            .ok_or(ForecastCalibrationError::ArithmeticOverflow)?;
        squared_error_sum = squared_error_sum
            .checked_add(squared_error)
            .ok_or(ForecastCalibrationError::ArithmeticOverflow)?;

        pinball_q10 = pinball_q10
            .checked_add(pinball_numerator(actual - q10, Q10_BPS)?)
            .ok_or(ForecastCalibrationError::ArithmeticOverflow)?;
        pinball_q50 = pinball_q50
            .checked_add(pinball_numerator(error, Q50_BPS)?)
            .ok_or(ForecastCalibrationError::ArithmeticOverflow)?;
        pinball_q90 = pinball_q90
            .checked_add(pinball_numerator(actual - q90, Q90_BPS)?)
            .ok_or(ForecastCalibrationError::ArithmeticOverflow)?;

        if actual <= q10 {
            at_or_below_q10 = at_or_below_q10.saturating_add(1);
            below_q10 = below_q10.saturating_add(1);
        }
        if actual <= q50 {
            at_or_below_q50 = at_or_below_q50.saturating_add(1);
        }
        if actual <= q90 {
            at_or_below_q90 = at_or_below_q90.saturating_add(1);
        }
        if actual >= q10 && actual <= q90 {
            central_80_covered = central_80_covered.saturating_add(1);
        }
        if actual > q90 {
            above_q90 = above_q90.saturating_add(1);
        }

        let interval_width = u128::try_from(q90 - q10)
            .map_err(|_| ForecastCalibrationError::ArithmeticOverflow)?;
        interval_width_sum = interval_width_sum
            .checked_add(interval_width)
            .ok_or(ForecastCalibrationError::ArithmeticOverflow)?;

        // Central 80% interval score: width + 10 * distance outside the interval.
        let outside_distance = if actual < q10 {
            q10 - actual
        } else if actual > q90 {
            actual - q90
        } else {
            0
        };
        let interval_score = interval_width
            .checked_add(
                u128::try_from(outside_distance)
                    .map_err(|_| ForecastCalibrationError::ArithmeticOverflow)?
                    .checked_mul(10)
                    .ok_or(ForecastCalibrationError::ArithmeticOverflow)?,
            )
            .ok_or(ForecastCalibrationError::ArithmeticOverflow)?;
        let weighted = absolute_error
            .checked_mul(5)
            .and_then(|value| value.checked_add(interval_score))
            .ok_or(ForecastCalibrationError::ArithmeticOverflow)?;
        wis_numerator = wis_numerator
            .checked_add(weighted)
            .ok_or(ForecastCalibrationError::ArithmeticOverflow)?;
    }

    let n = u64::try_from(observations.len())
        .map_err(|_| ForecastCalibrationError::ArithmeticOverflow)?;
    let calibration_error = |count: u64, nominal_bps: i128| {
        i128::from(count)
            .checked_mul(BASIS_POINTS)
            .and_then(|observed| {
                i128::from(n)
                    .checked_mul(nominal_bps)
                    .and_then(|expected| observed.checked_sub(expected))
            })
            .ok_or(ForecastCalibrationError::ArithmeticOverflow)
    };
    let wis_denominator = u128::from(n)
        .checked_mul(15)
        .ok_or(ForecastCalibrationError::ArithmeticOverflow)?;

    Ok(ForecastCalibrationMetrics {
        model_id,
        model_sha256,
        evaluation_as_of_utc,
        sample_count: n,
        absolute_error_sum_minor_zar_per_mwh: absolute_error_sum,
        signed_error_sum_minor_zar_per_mwh: signed_error_sum,
        squared_error_sum_minor_zar2_per_mwh2: squared_error_sum,
        pinball_q10_numerator: pinball_q10,
        pinball_q50_numerator: pinball_q50,
        pinball_q90_numerator: pinball_q90,
        pinball_denominator: BASIS_POINTS,
        actual_at_or_below_q10_count: at_or_below_q10,
        actual_at_or_below_q50_count: at_or_below_q50,
        actual_at_or_below_q90_count: at_or_below_q90,
        central_80_covered_count: central_80_covered,
        below_q10_count: below_q10,
        above_q90_count: above_q90,
        q10_coverage_error_numerator: calibration_error(at_or_below_q10, Q10_BPS)?,
        q50_coverage_error_numerator: calibration_error(at_or_below_q50, Q50_BPS)?,
        q90_coverage_error_numerator: calibration_error(at_or_below_q90, Q90_BPS)?,
        central_80_interval_width_sum_minor_zar_per_mwh: interval_width_sum,
        weighted_interval_score_80_numerator: wis_numerator,
        weighted_interval_score_80_denominator: wis_denominator,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const MODEL_SHA: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn observation(
        id: &'static str,
        q10: i64,
        q50: i64,
        q90: i64,
        actual: i64,
    ) -> ForecastCalibrationObservation<'static> {
        ForecastCalibrationObservation {
            record_id: id,
            forecast_source_ref: id,
            forecast_source_sha256: SOURCE_SHA,
            realized_source_ref: "market/realized/price",
            realized_source_sha256: SOURCE_SHA,
            series_id: "SAWEM/SMP/TP-001",
            model_id: "chronos-v1",
            model_sha256: MODEL_SHA,
            decision_time_utc: 50,
            forecast_issued_at_utc: 40,
            forecast_available_at_utc: 45,
            valid_period_start_utc: 100,
            valid_period_end_utc: 200,
            forecast_q10_minor_zar_per_mwh: q10,
            forecast_q50_minor_zar_per_mwh: q50,
            forecast_q90_minor_zar_per_mwh: q90,
            realized_minor_zar_per_mwh: actual,
            realized_observed_at_utc: 205,
            realized_available_at_utc: 210,
        }
    }

    #[test]
    fn known_answer_scores_mae_bias_pinball_coverage_and_weighted_interval() {
        let observations = [observation("obs-001", 100, 200, 300, 350)];
        let result = evaluate_forecast_calibration(220, &observations).unwrap();
        assert_eq!(result.sample_count, 1);
        assert_eq!(result.absolute_error_sum_minor_zar_per_mwh, 150);
        assert_eq!(result.signed_error_sum_minor_zar_per_mwh, 150);
        assert_eq!(result.squared_error_sum_minor_zar2_per_mwh2, 22_500);
        assert_eq!(result.pinball_q10_numerator, 250_000);
        assert_eq!(result.pinball_q50_numerator, 750_000);
        assert_eq!(result.pinball_q90_numerator, 450_000);
        assert_eq!(result.actual_at_or_below_q10_count, 0);
        assert_eq!(result.actual_at_or_below_q50_count, 0);
        assert_eq!(result.actual_at_or_below_q90_count, 0);
        assert_eq!(result.central_80_covered_count, 0);
        assert_eq!(result.above_q90_count, 1);
        assert_eq!(result.central_80_interval_width_sum_minor_zar_per_mwh, 200);
        assert_eq!(result.weighted_interval_score_80_numerator, 1_450);
        assert_eq!(result.weighted_interval_score_80_denominator, 15);
    }

    #[test]
    fn exact_empirical_coverage_and_calibration_error_are_count_based() {
        let observations = [
            observation("obs-low", 100, 200, 300, 50),
            observation("obs-middle", 100, 200, 300, 200),
            observation("obs-high", 100, 200, 300, 350),
            observation("obs-upper", 100, 200, 300, 250),
            observation("obs-lower", 100, 200, 300, 150),
            observation("obs-center", 100, 200, 300, 200),
            observation("obs-upper-2", 100, 200, 300, 250),
            observation("obs-center-2", 100, 200, 300, 200),
            observation("obs-high-2", 100, 200, 300, 350),
            observation("obs-low-2", 100, 200, 300, 50),
        ];
        let result = evaluate_forecast_calibration(220, &observations).unwrap();
        assert_eq!(result.sample_count, 10);
        assert_eq!(result.actual_at_or_below_q10_count, 2);
        assert_eq!(result.actual_at_or_below_q50_count, 6);
        assert_eq!(result.actual_at_or_below_q90_count, 8);
        assert_eq!(result.central_80_covered_count, 6);
        assert_eq!(result.below_q10_count, 2);
        assert_eq!(result.above_q90_count, 2);
        assert_eq!(result.q10_coverage_error_numerator, 10_000);
        assert_eq!(result.q50_coverage_error_numerator, 10_000);
        assert_eq!(result.q90_coverage_error_numerator, -10_000);
    }

    #[test]
    fn forecast_known_at_decision_and_realized_known_at_evaluation_are_both_required() {
        let late_forecast = [ForecastCalibrationObservation {
            forecast_available_at_utc: 51,
            ..observation("late-forecast", 100, 200, 300, 250)
        }];
        assert_eq!(
            evaluate_forecast_calibration(220, &late_forecast),
            Err(ForecastCalibrationError::ForecastWasNotAvailableAtDecision)
        );

        let late_actual = [ForecastCalibrationObservation {
            realized_available_at_utc: 221,
            ..observation("late-actual", 100, 200, 300, 250)
        }];
        assert_eq!(
            evaluate_forecast_calibration(220, &late_actual),
            Err(ForecastCalibrationError::RealizedObservationNotAvailableAtEvaluation)
        );
    }

    #[test]
    fn future_decision_target_and_realized_before_interval_end_fail_closed() {
        let after_start = [ForecastCalibrationObservation {
            decision_time_utc: 101,
            ..observation("after-start", 100, 200, 300, 250)
        }];
        assert_eq!(
            evaluate_forecast_calibration(220, &after_start),
            Err(ForecastCalibrationError::DecisionAfterTargetStart)
        );

        let early_actual = [ForecastCalibrationObservation {
            realized_observed_at_utc: 199,
            realized_available_at_utc: 210,
            ..observation("early-realized", 100, 200, 300, 250)
        }];
        assert_eq!(
            evaluate_forecast_calibration(220, &early_actual),
            Err(ForecastCalibrationError::RealizedObservationBeforePeriodEnd)
        );
    }

    #[test]
    fn quantile_crossing_mixed_model_builds_and_duplicate_targets_are_rejected() {
        let crossing = [observation("crossing", 300, 200, 400, 250)];
        assert_eq!(
            evaluate_forecast_calibration(220, &crossing),
            Err(ForecastCalibrationError::QuantileCrossing)
        );

        let other_model = [ForecastCalibrationObservation {
            model_id: "homer-v1",
            ..observation("other-model", 100, 200, 300, 250)
        }];
        assert_eq!(
            evaluate_forecast_calibration(220, &other_model),
            Err(ForecastCalibrationError::MixedModelBuild)
        );

        let duplicate = [
            observation("same-period-a", 100, 200, 300, 250),
            observation("same-period-b", 100, 200, 300, 260),
        ];
        assert_eq!(
            evaluate_forecast_calibration(220, &duplicate),
            Err(ForecastCalibrationError::DuplicateTargetPeriod)
        );
    }

    #[test]
    fn malformed_digests_and_empty_evaluation_are_rejected() {
        assert_eq!(
            evaluate_forecast_calibration(220, &[]),
            Err(ForecastCalibrationError::EmptyEvaluationSet)
        );
        let bad_source = [ForecastCalibrationObservation {
            forecast_source_sha256: "bad",
            ..observation("bad-source", 100, 200, 300, 250)
        }];
        assert_eq!(
            evaluate_forecast_calibration(220, &bad_source),
            Err(ForecastCalibrationError::InvalidSourceDigest)
        );
        let bad_model = [ForecastCalibrationObservation {
            model_sha256: "bad",
            ..observation("bad-model", 100, 200, 300, 250)
        }];
        assert_eq!(
            evaluate_forecast_calibration(220, &bad_model),
            Err(ForecastCalibrationError::InvalidModelDigest)
        );
    }
}
