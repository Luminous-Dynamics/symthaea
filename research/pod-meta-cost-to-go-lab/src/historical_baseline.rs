use std::collections::{BTreeMap, BTreeSet};

use crate::{ForecastId, ForecastJournal, ResourceVector, TerminalOutcome};

const DIMENSION_COUNT: usize = 6;

/// How a frozen historical profile was selected for an evaluation prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoricalProfileSource {
    TaskFamily(u32),
    GlobalFallback,
}

/// Empirical order-statistic cost profile. q90/q95 are descriptive empirical
/// upper estimates, not conformal bounds or coverage guarantees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoricalCostProfile {
    pub source: HistoricalProfileSource,
    pub support_count: usize,
    pub q50: ResourceVector,
    pub q90: ResourceVector,
    pub q95: ResourceVector,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoricalBaselinePrediction {
    pub profile: HistoricalCostProfile,
}

/// Frozen family-conditional empirical baseline trained only from
/// independently verified sufficient-proof outcomes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoricalCostBaseline {
    pub training_partition_digest: [u8; 32],
    pub model_digest: [u8; 32],
    training_run_ids: BTreeSet<u64>,
    global_profile: HistoricalCostProfile,
    family_profiles: BTreeMap<u32, HistoricalCostProfile>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HistoricalDimensionMetrics {
    pub completed_labels: usize,
    pub q50_mean_absolute_error: Option<f64>,
    /// Positive bias means historical q50 over-predicts remaining cost.
    pub q50_mean_signed_bias: Option<f64>,
    pub q90_marginal_coverage: Option<f64>,
    pub q95_marginal_coverage: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct DimensionAccumulator {
    count: usize,
    absolute_error_sum: u128,
    signed_error_sum: i128,
    q90_hits: usize,
    q95_hits: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalBaselineRow {
    pub forecast_id: ForecastId,
    pub run_id: u64,
    pub decision_sequence: u64,
    pub task_family: u32,
    pub profile_source: HistoricalProfileSource,
    pub support_count: usize,
    pub prediction: HistoricalCostProfile,
    pub observed_work: Option<ResourceVector>,
    pub terminal: Option<TerminalOutcome>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalBaselineReport {
    pub total_prefixes: usize,
    pub completed_verified: usize,
    pub right_censored: usize,
    pub competing_terminal: usize,
    pub pending: usize,
    pub global_fallback_prefixes: usize,
    /// Descriptive all-six-dimension q90 coverage; not a joint guarantee.
    pub empirical_joint_q90_coverage: Option<f64>,
    pub dimensions: [HistoricalDimensionMetrics; DIMENSION_COUNT],
    pub by_task_family: Vec<HistoricalBaselineFamilyReport>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalBaselineFamilyReport {
    pub task_family: u32,
    pub prefixes: usize,
    pub completed_verified: usize,
    pub global_fallback_prefixes: usize,
    pub empirical_joint_q90_coverage: Option<f64>,
    pub dimensions: [HistoricalDimensionMetrics; DIMENSION_COUNT],
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalBaselineEvaluation {
    /// Stable order: run ID, decision sequence, then forecast ID.
    pub rows: Vec<HistoricalBaselineRow>,
    pub report: HistoricalBaselineReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoricalBaselineError {
    MissingTrainingPartitionDigest,
    MissingModelDigest,
    EmptyVerifiedTrainingCohort,
    TrainingRunOverlap,
    NumericOverflow,
}

impl std::fmt::Display for HistoricalBaselineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HistoricalBaselineError {}

impl HistoricalCostBaseline {
    /// Fit a task-family median/upper-quantile baseline. The partition/model
    /// digests are caller-declared identities, not cryptographic attestations.
    pub fn fit(
        training: &ForecastJournal,
        training_partition_digest: [u8; 32],
        model_digest: [u8; 32],
    ) -> Result<Self, HistoricalBaselineError> {
        if is_zero(training_partition_digest) {
            return Err(HistoricalBaselineError::MissingTrainingPartitionDigest);
        }
        if is_zero(model_digest) {
            return Err(HistoricalBaselineError::MissingModelDigest);
        }

        let mut training_run_ids = BTreeSet::new();
        let mut global_costs = Vec::<ResourceVector>::new();
        let mut family_costs = BTreeMap::<u32, Vec<ResourceVector>>::new();

        for forecast in training.iter_forecasts() {
            training_run_ids.insert(forecast.run_id);
            let Some(outcome) = training.outcome(forecast.id) else {
                continue;
            };
            if outcome.terminal != TerminalOutcome::SufficientProof
                || outcome.verifier_digest.is_none()
            {
                continue;
            }
            global_costs.push(outcome.observed_work);
            family_costs
                .entry(forecast.task_family)
                .or_default()
                .push(outcome.observed_work);
        }

        if global_costs.is_empty() {
            return Err(HistoricalBaselineError::EmptyVerifiedTrainingCohort);
        }

        let global_profile = build_profile(
            HistoricalProfileSource::GlobalFallback,
            &global_costs,
        )?;
        let family_profiles = family_costs
            .into_iter()
            .map(|(family, costs)| {
                build_profile(HistoricalProfileSource::TaskFamily(family), &costs)
                    .map(|profile| (family, profile))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;

        Ok(Self {
            training_partition_digest,
            model_digest,
            training_run_ids,
            global_profile,
            family_profiles,
        })
    }

    pub fn predict(&self, task_family: u32) -> HistoricalBaselinePrediction {
        let profile = self
            .family_profiles
            .get(&task_family)
            .copied()
            .unwrap_or(self.global_profile);
        HistoricalBaselinePrediction { profile }
    }

    pub fn training_run_count(&self) -> usize {
        self.training_run_ids.len()
    }

    pub fn evaluate(
        &self,
        evaluation: &ForecastJournal,
    ) -> Result<HistoricalBaselineEvaluation, HistoricalBaselineError> {
        let mut rows = Vec::new();
        for forecast in evaluation.iter_forecasts() {
            if self.training_run_ids.contains(&forecast.run_id) {
                return Err(HistoricalBaselineError::TrainingRunOverlap);
            }
            let prediction = self.predict(forecast.task_family);
            let outcome = evaluation.outcome(forecast.id);
            rows.push(HistoricalBaselineRow {
                forecast_id: forecast.id,
                run_id: forecast.run_id,
                decision_sequence: forecast.decision_sequence,
                task_family: forecast.task_family,
                profile_source: prediction.profile.source,
                support_count: prediction.profile.support_count,
                prediction: prediction.profile,
                observed_work: outcome.map(|value| value.observed_work),
                terminal: outcome.map(|value| value.terminal),
            });
        }
        rows.sort_by_key(|row| (row.run_id, row.decision_sequence, row.forecast_id));

        let mut overall = BaselineAccumulator::default();
        let mut by_family = BTreeMap::<u32, BaselineAccumulator>::new();
        for row in &rows {
            update_baseline_accumulator(&mut overall, row)?;
            update_baseline_accumulator(by_family.entry(row.task_family).or_default(), row)?;
        }

        let report = HistoricalBaselineReport {
            total_prefixes: overall.total_prefixes,
            completed_verified: overall.completed_verified,
            right_censored: overall.right_censored,
            competing_terminal: overall.competing_terminal,
            pending: overall.pending,
            global_fallback_prefixes: overall.global_fallback_prefixes,
            empirical_joint_q90_coverage: ratio(
                overall.joint_q90_hits,
                overall.completed_verified,
            ),
            dimensions: dimension_metrics(overall.dimensions),
            by_task_family: by_family
                .into_iter()
                .map(|(task_family, values)| HistoricalBaselineFamilyReport {
                    task_family,
                    prefixes: values.total_prefixes,
                    completed_verified: values.completed_verified,
                    global_fallback_prefixes: values.global_fallback_prefixes,
                    empirical_joint_q90_coverage: ratio(
                        values.joint_q90_hits,
                        values.completed_verified,
                    ),
                    dimensions: dimension_metrics(values.dimensions),
                })
                .collect(),
        };

        Ok(HistoricalBaselineEvaluation { rows, report })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct BaselineAccumulator {
    total_prefixes: usize,
    completed_verified: usize,
    right_censored: usize,
    competing_terminal: usize,
    pending: usize,
    global_fallback_prefixes: usize,
    joint_q90_hits: usize,
    dimensions: [DimensionAccumulator; DIMENSION_COUNT],
}

fn build_profile(
    source: HistoricalProfileSource,
    costs: &[ResourceVector],
) -> Result<HistoricalCostProfile, HistoricalBaselineError> {
    if costs.is_empty() {
        return Err(HistoricalBaselineError::EmptyVerifiedTrainingCohort);
    }
    let mut q50 = [0u128; DIMENSION_COUNT];
    let mut q90 = [0u128; DIMENSION_COUNT];
    let mut q95 = [0u128; DIMENSION_COUNT];
    for dimension in 0..DIMENSION_COUNT {
        let mut values = costs
            .iter()
            .map(|cost| resource_values(*cost)[dimension])
            .collect::<Vec<_>>();
        values.sort_unstable();
        q50[dimension] = median(&values)?;
        q90[dimension] = nearest_rank(&values, 9, 10)?;
        q95[dimension] = nearest_rank(&values, 95, 100)?;
    }
    Ok(HistoricalCostProfile {
        source,
        support_count: costs.len(),
        q50: resource_vector(q50),
        q90: resource_vector(q90),
        q95: resource_vector(q95),
    })
}

fn median(sorted: &[u128]) -> Result<u128, HistoricalBaselineError> {
    let len = sorted.len();
    if len == 0 {
        return Err(HistoricalBaselineError::EmptyVerifiedTrainingCohort);
    }
    if len % 2 == 1 {
        Ok(sorted[len / 2])
    } else {
        let lower = sorted[len / 2 - 1];
        let upper = sorted[len / 2];
        lower
            .checked_add((upper - lower) / 2)
            .ok_or(HistoricalBaselineError::NumericOverflow)
    }
}

fn nearest_rank(
    sorted: &[u128],
    numerator: u128,
    denominator: u128,
) -> Result<u128, HistoricalBaselineError> {
    if sorted.is_empty() || denominator == 0 || numerator == 0 || numerator > denominator {
        return Err(HistoricalBaselineError::EmptyVerifiedTrainingCohort);
    }
    let count = u128::try_from(sorted.len())
        .map_err(|_| HistoricalBaselineError::NumericOverflow)?;
    let rank_numerator = count
        .checked_mul(numerator)
        .ok_or(HistoricalBaselineError::NumericOverflow)?;
    let rank = rank_numerator
        .checked_add(denominator - 1)
        .ok_or(HistoricalBaselineError::NumericOverflow)?
        / denominator;
    let index = usize::try_from(rank.saturating_sub(1))
        .map_err(|_| HistoricalBaselineError::NumericOverflow)?;
    sorted
        .get(index)
        .copied()
        .ok_or(HistoricalBaselineError::NumericOverflow)
}

fn update_baseline_accumulator(
    accumulator: &mut BaselineAccumulator,
    row: &HistoricalBaselineRow,
) -> Result<(), HistoricalBaselineError> {
    increment(&mut accumulator.total_prefixes)?;
    if row.profile_source == HistoricalProfileSource::GlobalFallback {
        increment(&mut accumulator.global_fallback_prefixes)?;
    }
    match row.terminal {
        None => increment(&mut accumulator.pending)?,
        Some(TerminalOutcome::RightCensored(_)) => increment(&mut accumulator.right_censored)?,
        Some(TerminalOutcome::CompetingTerminal(_)) => {
            increment(&mut accumulator.competing_terminal)?
        }
        Some(TerminalOutcome::SufficientProof) => {
            let observed = row
                .observed_work
                .ok_or(HistoricalBaselineError::EmptyVerifiedTrainingCohort)?;
            increment(&mut accumulator.completed_verified)?;
            let actual = resource_values(observed);
            let q50 = resource_values(row.prediction.q50);
            let q90 = resource_values(row.prediction.q90);
            let q95 = resource_values(row.prediction.q95);
            let mut joint_covered = true;
            for index in 0..DIMENSION_COUNT {
                let dimension = &mut accumulator.dimensions[index];
                increment(&mut dimension.count)?;
                dimension.absolute_error_sum = dimension
                    .absolute_error_sum
                    .checked_add(q50[index].abs_diff(actual[index]))
                    .ok_or(HistoricalBaselineError::NumericOverflow)?;
                let signed = i128::try_from(q50[index])
                    .map_err(|_| HistoricalBaselineError::NumericOverflow)?
                    .checked_sub(
                        i128::try_from(actual[index])
                            .map_err(|_| HistoricalBaselineError::NumericOverflow)?,
                    )
                    .ok_or(HistoricalBaselineError::NumericOverflow)?;
                dimension.signed_error_sum = dimension
                    .signed_error_sum
                    .checked_add(signed)
                    .ok_or(HistoricalBaselineError::NumericOverflow)?;
                if actual[index] <= q90[index] {
                    increment(&mut dimension.q90_hits)?;
                } else {
                    joint_covered = false;
                }
                if actual[index] <= q95[index] {
                    increment(&mut dimension.q95_hits)?;
                }
            }
            if joint_covered {
                increment(&mut accumulator.joint_q90_hits)?;
            }
        }
    }
    Ok(())
}

fn dimension_metrics(
    dimensions: [DimensionAccumulator; DIMENSION_COUNT],
) -> [HistoricalDimensionMetrics; DIMENSION_COUNT] {
    std::array::from_fn(|index| {
        let value = dimensions[index];
        HistoricalDimensionMetrics {
            completed_labels: value.count,
            q50_mean_absolute_error: mean_u128(value.absolute_error_sum, value.count),
            q50_mean_signed_bias: mean_i128(value.signed_error_sum, value.count),
            q90_marginal_coverage: ratio(value.q90_hits, value.count),
            q95_marginal_coverage: ratio(value.q95_hits, value.count),
        }
    })
}

fn increment(value: &mut usize) -> Result<(), HistoricalBaselineError> {
    *value = value
        .checked_add(1)
        .ok_or(HistoricalBaselineError::NumericOverflow)?;
    Ok(())
}

fn mean_u128(sum: u128, count: usize) -> Option<f64> {
    (count > 0).then(|| sum as f64 / count as f64)
}

fn mean_i128(sum: i128, count: usize) -> Option<f64> {
    (count > 0).then(|| sum as f64 / count as f64)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn resource_values(vector: ResourceVector) -> [u128; DIMENSION_COUNT] {
    [
        vector.tokens,
        vector.tool_calls,
        vector.cpu_ms,
        vector.gpu_ms,
        vector.wall_ms,
        vector.micro_usd,
    ]
}

fn resource_vector(values: [u128; DIMENSION_COUNT]) -> ResourceVector {
    ResourceVector::new(
        values[0], values[1], values[2], values[3], values[4], values[5],
    )
}

fn is_zero(digest: [u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CensorReason, CompetingReason, ForecastOutcomeReceipt, ForecastReceipt,
        RemainingCostForecast,
    };

    fn forecast(id: u64, run_id: u64, family: u32) -> ForecastReceipt {
        ForecastReceipt {
            id: ForecastId(id),
            run_id,
            decision_sequence: 1,
            task_family: family,
            operation_family: 1,
            forecast: RemainingCostForecast {
                pre_state_digest: [1; 32],
                frontier_digest: [2; 32],
                model_digest: [3; 32],
                calibration_digest: [4; 32],
                completion_probability: 0.9,
                remaining_q50: ResourceVector::ZERO,
                remaining_q90: ResourceVector::ZERO,
                remaining_q95: ResourceVector::ZERO,
                mandatory_verification_reserve: ResourceVector::ZERO,
                support_count: 1,
                shift_status: crate::ShiftStatus::InDomain,
            },
        }
    }

    fn outcome(
        id: u64,
        work: ResourceVector,
        terminal: TerminalOutcome,
    ) -> ForecastOutcomeReceipt {
        ForecastOutcomeReceipt {
            forecast_id: ForecastId(id),
            observed_sequence: 2,
            observed_work: work,
            terminal,
            evidence_digest: [5; 32],
            meter_digest: [6; 32],
            verifier_digest: (terminal == TerminalOutcome::SufficientProof).then_some([7; 32]),
        }
    }

    #[test]
    fn training_uses_verified_completions_and_builds_family_and_global_profiles() {
        let mut training = ForecastJournal::default();
        training.append_forecast(forecast(1, 101, 7)).unwrap();
        training.append_forecast(forecast(2, 102, 7)).unwrap();
        training.append_forecast(forecast(3, 103, 8)).unwrap();
        training
            .append_outcome(outcome(
                1,
                ResourceVector::new(10, 1, 2, 0, 5, 8),
                TerminalOutcome::SufficientProof,
            ))
            .unwrap();
        training
            .append_outcome(outcome(
                2,
                ResourceVector::new(20, 2, 4, 1, 10, 16),
                TerminalOutcome::SufficientProof,
            ))
            .unwrap();
        training
            .append_outcome(outcome(
                3,
                ResourceVector::new(100, 10, 20, 5, 50, 80),
                TerminalOutcome::RightCensored(CensorReason::Timeout),
            ))
            .unwrap();

        let model = HistoricalCostBaseline::fit(&training, [8; 32], [9; 32]).unwrap();
        assert_eq!(model.training_run_count(), 3);
        let family = model.predict(7).profile;
        assert_eq!(family.source, HistoricalProfileSource::TaskFamily(7));
        assert_eq!(family.support_count, 2);
        assert_eq!(family.q50.tokens, 15);
        assert_eq!(family.q90.tokens, 20);
        assert_eq!(family.q95.tokens, 20);

        let fallback = model.predict(99).profile;
        assert_eq!(fallback.source, HistoricalProfileSource::GlobalFallback);
        assert_eq!(fallback.support_count, 2);
        assert_eq!(fallback.q50.tokens, 15);
    }

    #[test]
    fn evaluation_is_disjoint_from_training_and_scores_only_verified_labels() {
        let mut training = ForecastJournal::default();
        training.append_forecast(forecast(1, 101, 7)).unwrap();
        training
            .append_outcome(outcome(
                1,
                ResourceVector::new(10, 1, 2, 0, 5, 8),
                TerminalOutcome::SufficientProof,
            ))
            .unwrap();
        let model = HistoricalCostBaseline::fit(&training, [8; 32], [9; 32]).unwrap();

        let mut evaluation = ForecastJournal::default();
        evaluation.append_forecast(forecast(2, 201, 7)).unwrap();
        evaluation.append_forecast(forecast(3, 202, 99)).unwrap();
        evaluation
            .append_outcome(outcome(
                2,
                ResourceVector::new(12, 1, 2, 0, 5, 8),
                TerminalOutcome::SufficientProof,
            ))
            .unwrap();
        evaluation
            .append_outcome(outcome(
                3,
                ResourceVector::new(100, 10, 20, 5, 50, 80),
                TerminalOutcome::CompetingTerminal(CompetingReason::PermissionRevoked),
            ))
            .unwrap();

        let result = model.evaluate(&evaluation).unwrap();
        assert_eq!(result.rows[0].profile_source, HistoricalProfileSource::TaskFamily(7));
        assert_eq!(result.rows[1].profile_source, HistoricalProfileSource::GlobalFallback);
        assert_eq!(result.report.total_prefixes, 2);
        assert_eq!(result.report.completed_verified, 1);
        assert_eq!(result.report.competing_terminal, 1);
        assert_eq!(result.report.global_fallback_prefixes, 1);
        assert_eq!(result.report.dimensions[0].q50_mean_absolute_error, Some(2.0));
    }

    #[test]
    fn overlapping_training_and_evaluation_runs_fail_closed() {
        let mut training = ForecastJournal::default();
        training.append_forecast(forecast(1, 101, 7)).unwrap();
        training
            .append_outcome(outcome(
                1,
                ResourceVector::new(10, 1, 2, 0, 5, 8),
                TerminalOutcome::SufficientProof,
            ))
            .unwrap();
        let model = HistoricalCostBaseline::fit(&training, [8; 32], [9; 32]).unwrap();

        let mut evaluation = ForecastJournal::default();
        evaluation.append_forecast(forecast(2, 101, 7)).unwrap();
        assert_eq!(
            model.evaluate(&evaluation),
            Err(HistoricalBaselineError::TrainingRunOverlap)
        );
    }

    #[test]
    fn empty_verified_training_cohort_and_zero_digests_are_rejected() {
        let mut training = ForecastJournal::default();
        training.append_forecast(forecast(1, 101, 7)).unwrap();
        assert_eq!(
            HistoricalCostBaseline::fit(&training, [8; 32], [9; 32]),
            Err(HistoricalBaselineError::EmptyVerifiedTrainingCohort)
        );
        assert_eq!(
            HistoricalCostBaseline::fit(&training, [0; 32], [9; 32]),
            Err(HistoricalBaselineError::MissingTrainingPartitionDigest)
        );
    }
}
