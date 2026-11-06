//! Predeclared circular moving-block bootstrap for paired forecast skill.
//!
//! The plan contains exact block-start indices and a content digest. No RNG is
//! invoked inside the evaluator, so the resampling schedule is replayable.
//! The block length and confidence level must be preregistered before inspecting
//! the held-out results. Positive improvement favours the candidate forecast.

use crate::forecast_skill::{ForecastSkillAssessment, PairedForecastLossDelta};

const BPS: u128 = 10_000;
const MIN_BOOTSTRAP_DRAWS: u32 = 100;
const MAX_BOOTSTRAP_DRAWS: u32 = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CircularBlockBootstrapPlan<'a> {
    pub plan_id: &'a str,
    pub plan_source_ref: &'a str,
    pub plan_sha256: &'a str,
    pub block_length: u32,
    pub draw_count: u32,
    pub confidence_level_bps: u16,
    /// Flattened block starts, draw-major. Each draw contains ceil(n / block_length)
    /// start indexes. Samples within each block are circular and contiguous.
    pub block_start_indices: &'a [u32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastInferenceError {
    InvalidEvidenceReference,
    InvalidPlanDigest,
    InvalidBlockLength,
    InvalidDrawCount,
    InvalidConfidenceLevel,
    BlockStartCountMismatch,
    BlockStartOutsideTargetSet,
    EmptyPairedLossSet,
    MixedSeriesForBootstrap,
    NonChronologicalOrGappedTargets,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForecastSkillConfidenceInterval {
    /// Lower/upper numerators of the resampled mean improvement.
    /// Divide by paired_sample_count to obtain mean improvement.
    pub lower_improvement_numerator: i128,
    pub upper_improvement_numerator: i128,
    pub paired_sample_count: u64,
    pub draw_count: u32,
    pub block_length: u32,
    pub confidence_level_bps: u16,
    /// True only if the entire two-sided bootstrap interval is positive.
    pub improvement_supported: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForecastSkillInference {
    pub plan_id: String,
    pub plan_source_ref: String,
    pub plan_sha256: String,
    pub mae: ForecastSkillConfidenceInterval,
    pub weighted_interval: ForecastSkillConfidenceInterval,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn ceil_div(numerator: u128, denominator: u128) -> Result<u128, ForecastInferenceError> {
    numerator
        .checked_add(denominator - 1)
        .map(|value| value / denominator)
        .ok_or(ForecastInferenceError::ArithmeticOverflow)
}

fn confidence_interval(
    mut bootstrap_sums: Vec<i128>,
    paired_sample_count: u64,
    draw_count: u32,
    block_length: u32,
    confidence_level_bps: u16,
) -> Result<ForecastSkillConfidenceInterval, ForecastInferenceError> {
    bootstrap_sums.sort_unstable();
    let draws = u128::from(draw_count);
    let tail_bps = BPS
        .checked_sub(u128::from(confidence_level_bps))
        .ok_or(ForecastInferenceError::InvalidConfidenceLevel)?;
    let lower_rank = ceil_div(
        draws
            .checked_mul(tail_bps)
            .ok_or(ForecastInferenceError::ArithmeticOverflow)?,
        2 * BPS,
    )?
    .max(1);
    let upper_rank = ceil_div(
        draws
            .checked_mul(BPS + u128::from(confidence_level_bps))
            .ok_or(ForecastInferenceError::ArithmeticOverflow)?,
        2 * BPS,
    )?
    .max(1);

    let lower_index = usize::try_from(lower_rank - 1)
        .map_err(|_| ForecastInferenceError::ArithmeticOverflow)?;
    let upper_index = usize::try_from(upper_rank - 1)
        .map_err(|_| ForecastInferenceError::ArithmeticOverflow)?;
    let lower = *bootstrap_sums
        .get(lower_index)
        .ok_or(ForecastInferenceError::ArithmeticOverflow)?;
    let upper = *bootstrap_sums
        .get(upper_index)
        .ok_or(ForecastInferenceError::ArithmeticOverflow)?;

    Ok(ForecastSkillConfidenceInterval {
        lower_improvement_numerator: lower,
        upper_improvement_numerator: upper,
        paired_sample_count,
        draw_count,
        block_length,
        confidence_level_bps,
        improvement_supported: lower > 0,
    })
}

/// Re-evaluate the paired candidate-minus-baseline loss differences using a
/// frozen circular moving-block resampling plan. Each draw contains enough
/// contiguous circular blocks to resample exactly n paired Trading Periods.
pub fn evaluate_forecast_skill_inference(
    skill: &ForecastSkillAssessment<'_>,
    plan: CircularBlockBootstrapPlan<'_>,
) -> Result<ForecastSkillInference, ForecastInferenceError> {
    let deltas = &skill.paired_loss_deltas;
    if deltas.is_empty() {
        return Err(ForecastInferenceError::EmptyPairedLossSet);
    }
    if !nonempty(plan.plan_id) || !nonempty(plan.plan_source_ref) {
        return Err(ForecastInferenceError::InvalidEvidenceReference);
    }
    if !valid_sha256(plan.plan_sha256) {
        return Err(ForecastInferenceError::InvalidPlanDigest);
    }
    if plan.block_length == 0 || plan.block_length as usize > deltas.len() {
        return Err(ForecastInferenceError::InvalidBlockLength);
    }
    if plan.draw_count < MIN_BOOTSTRAP_DRAWS || plan.draw_count > MAX_BOOTSTRAP_DRAWS {
        return Err(ForecastInferenceError::InvalidDrawCount);
    }
    if !(9_000..10_000).contains(&plan.confidence_level_bps) {
        return Err(ForecastInferenceError::InvalidConfidenceLevel);
    }

    let sample_count = deltas.len();
    let first_series = deltas[0].series_id;
    for (index, delta) in deltas.iter().enumerate() {
        if delta.series_id != first_series {
            return Err(ForecastInferenceError::MixedSeriesForBootstrap);
        }
        if index > 0 {
            let previous = deltas[index - 1];
            if previous.valid_period_start_utc >= delta.valid_period_start_utc
                || previous.valid_period_end_utc != delta.valid_period_start_utc
            {
                return Err(ForecastInferenceError::NonChronologicalOrGappedTargets);
            }
        }
    }
    let block_length = plan.block_length as usize;
    let blocks_per_draw = sample_count.div_ceil(block_length);
    let expected_starts = (plan.draw_count as usize)
        .checked_mul(blocks_per_draw)
        .ok_or(ForecastInferenceError::ArithmeticOverflow)?;
    if plan.block_start_indices.len() != expected_starts {
        return Err(ForecastInferenceError::BlockStartCountMismatch);
    }
    if plan
        .block_start_indices
        .iter()
        .any(|start| *start as usize >= sample_count)
    {
        return Err(ForecastInferenceError::BlockStartOutsideTargetSet);
    }

    let mut mae_bootstrap = Vec::with_capacity(plan.draw_count as usize);
    let mut wis_bootstrap = Vec::with_capacity(plan.draw_count as usize);
    for draw_index in 0..plan.draw_count as usize {
        let mut mae_sum = 0_i128;
        let mut wis_sum = 0_i128;
        let mut sampled = 0_usize;
        let draw_start = draw_index
            .checked_mul(blocks_per_draw)
            .ok_or(ForecastInferenceError::ArithmeticOverflow)?;

        for block_index in 0..blocks_per_draw {
            if sampled == sample_count {
                break;
            }
            let start = plan.block_start_indices[draw_start + block_index] as usize;
            let take = block_length.min(sample_count - sampled);
            for offset in 0..take {
                let target_index = (start + offset) % sample_count;
                let delta: PairedForecastLossDelta<'_> = deltas[target_index];
                mae_sum = mae_sum
                    .checked_add(delta.median_mae_improvement_minor_zar_per_mwh)
                    .ok_or(ForecastInferenceError::ArithmeticOverflow)?;
                wis_sum = wis_sum
                    .checked_add(delta.weighted_interval_improvement_numerator)
                    .ok_or(ForecastInferenceError::ArithmeticOverflow)?;
            }
            sampled += take;
        }
        if sampled != sample_count {
            return Err(ForecastInferenceError::BlockStartCountMismatch);
        }
        mae_bootstrap.push(mae_sum);
        wis_bootstrap.push(wis_sum);
    }

    let paired_count = u64::try_from(sample_count)
        .map_err(|_| ForecastInferenceError::ArithmeticOverflow)?;
    let mae = confidence_interval(
        mae_bootstrap,
        paired_count,
        plan.draw_count,
        plan.block_length,
        plan.confidence_level_bps,
    )?;
    let weighted_interval = confidence_interval(
        wis_bootstrap,
        paired_count,
        plan.draw_count,
        plan.block_length,
        plan.confidence_level_bps,
    )?;

    Ok(ForecastSkillInference {
        plan_id: plan.plan_id.to_owned(),
        plan_source_ref: plan.plan_source_ref.to_owned(),
        plan_sha256: plan.plan_sha256.to_owned(),
        mae,
        weighted_interval,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast_skill::compare_forecast_skill;
    use crate::forecast_metrics::ForecastCalibrationObservation;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const CANDIDATE_SHA: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const BASELINE_SHA: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
    const PLAN_SHA: &str = "9988776655443322998877665544332299887766554433229988776655443322";

    fn forecast(
        id: &'static str,
        start: i64,
        model: &'static str,
        sha: &'static str,
        q50: i64,
        actual: i64,
    ) -> ForecastCalibrationObservation<'static> {
        ForecastCalibrationObservation {
            record_id: id,
            forecast_source_ref: id,
            forecast_source_sha256: SOURCE_SHA,
            realized_source_ref: "realized/common",
            realized_source_sha256: SOURCE_SHA,
            series_id: "SAWEM/SMP",
            model_id: model,
            model_sha256: sha,
            decision_time_utc: start - 60,
            forecast_issued_at_utc: start - 120,
            forecast_available_at_utc: start - 90,
            valid_period_start_utc: start,
            valid_period_end_utc: start + 15,
            forecast_q10_minor_zar_per_mwh: q50 - 10,
            forecast_q50_minor_zar_per_mwh: q50,
            forecast_q90_minor_zar_per_mwh: q50 + 10,
            realized_minor_zar_per_mwh: actual,
            realized_observed_at_utc: start + 20,
            realized_available_at_utc: start + 25,
        }
    }

    fn skill() -> ForecastSkillAssessment<'static> {
        let candidate = [
            forecast("candidate-1", 100, "candidate-v1", CANDIDATE_SHA, 100, 105),
            forecast("candidate-2", 115, "candidate-v1", CANDIDATE_SHA, 200, 205),
        ];
        let baseline = [
            forecast("baseline-1", 100, "persistence-v1", BASELINE_SHA, 90, 105),
            forecast("baseline-2", 115, "persistence-v1", BASELINE_SHA, 190, 205),
        ];
        compare_forecast_skill(300, &candidate, &baseline).unwrap()
    }

    fn plan(starts: &'static [u32], block_length: u32) -> CircularBlockBootstrapPlan<'static> {
        CircularBlockBootstrapPlan {
            plan_id: "bootstrap/predeclared/001",
            plan_source_ref: "benchmark/holdout/bootstrap-plan",
            plan_sha256: PLAN_SHA,
            block_length,
            draw_count: MIN_BOOTSTRAP_DRAWS,
            confidence_level_bps: 9_500,
            block_start_indices: starts,
        }
    }

    #[test]
    fn positive_lower_bound_supports_paired_improvement() {
        let starts = [0_u32, 1].repeat(MIN_BOOTSTRAP_DRAWS as usize);
        let starts: &'static [u32] = Box::leak(starts.into_boxed_slice());
        let result = evaluate_forecast_skill_inference(&skill(), plan(starts, 1)).unwrap();
        assert!(result.mae.improvement_supported);
        assert!(result.weighted_interval.improvement_supported);
        assert_eq!(result.mae.lower_improvement_numerator, 20);
        assert_eq!(result.mae.upper_improvement_numerator, 20);
        assert_eq!(result.mae.paired_sample_count, 2);
        assert_eq!(result.mae.confidence_level_bps, 9_500);
    }

    #[test]
    fn block_bootstrap_can_be_inconclusive_when_periods_disagree() {
        let candidate = [
            forecast("candidate-good", 100, "candidate-v1", CANDIDATE_SHA, 100, 105),
            forecast("candidate-bad", 115, "candidate-v1", CANDIDATE_SHA, 250, 205),
        ];
        let baseline = [
            forecast("baseline-good", 100, "persistence-v1", BASELINE_SHA, 90, 105),
            forecast("baseline-bad", 115, "persistence-v1", BASELINE_SHA, 190, 205),
        ];
        let assessment = compare_forecast_skill(300, &candidate, &baseline).unwrap();
        let mut starts = Vec::new();
        for _ in 0..(MIN_BOOTSTRAP_DRAWS / 2) {
            starts.extend_from_slice(&[0, 0]);
        }
        for _ in 0..(MIN_BOOTSTRAP_DRAWS / 2) {
            starts.extend_from_slice(&[1, 1]);
        }
        let result = evaluate_forecast_skill_inference(
            &assessment,
            CircularBlockBootstrapPlan {
                plan_id: "bootstrap/predeclared/001",
                plan_source_ref: "benchmark/holdout/bootstrap-plan",
                plan_sha256: PLAN_SHA,
                block_length: 1,
                draw_count: MIN_BOOTSTRAP_DRAWS,
                confidence_level_bps: 9_500,
                block_start_indices: &starts,
            },
        )
        .unwrap();
        assert!(!result.mae.improvement_supported);
        assert!(result.mae.lower_improvement_numerator < 0);
        assert!(result.mae.upper_improvement_numerator > 0);
    }

    #[test]
    fn bootstrap_rejects_multiple_series_and_time_gaps() {
        let mut assessment = skill();
        assessment.paired_loss_deltas[1].series_id = "OTHER/SERIES";
        let starts = [0_u32, 1].repeat(MIN_BOOTSTRAP_DRAWS as usize);
        let starts: &'static [u32] = Box::leak(starts.into_boxed_slice());
        assert_eq!(
            evaluate_forecast_skill_inference(&assessment, plan(starts, 1)),
            Err(ForecastInferenceError::MixedSeriesForBootstrap)
        );

        let mut assessment = skill();
        assessment.paired_loss_deltas[1].valid_period_start_utc = 201;
        assert_eq!(
            evaluate_forecast_skill_inference(&assessment, plan(starts, 1)),
            Err(ForecastInferenceError::NonChronologicalOrGappedTargets)
        );
    }

    #[test]
    fn invalid_bootstrap_plan_metadata_and_indexes_fail_closed() {
        let starts = [0_u32, 1].repeat(MIN_BOOTSTRAP_DRAWS as usize);
        let starts: &'static [u32] = Box::leak(starts.into_boxed_slice());
        let base = plan(starts, 1);
        assert_eq!(
            evaluate_forecast_skill_inference(
                &skill(),
                CircularBlockBootstrapPlan {
                    plan_sha256: "bad",
                    ..base
                }
            ),
            Err(ForecastInferenceError::InvalidPlanDigest)
        );
        assert_eq!(
            evaluate_forecast_skill_inference(
                &skill(),
                CircularBlockBootstrapPlan {
                    block_start_indices: &[0],
                    ..base
                }
            ),
            Err(ForecastInferenceError::BlockStartCountMismatch)
        );
        let invalid = vec![2_u32; 2 * MIN_BOOTSTRAP_DRAWS as usize];
        assert_eq!(
            evaluate_forecast_skill_inference(
                &skill(),
                CircularBlockBootstrapPlan {
                    block_start_indices: &invalid,
                    ..base
                }
            ),
            Err(ForecastInferenceError::BlockStartOutsideTargetSet)
        );
    }
}
