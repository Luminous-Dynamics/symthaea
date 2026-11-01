use std::collections::BTreeMap;

use crate::{ForecastId, ForecastJournal, RemainingCostForecast, ResourceVector, TerminalOutcome};

const DIMENSION_COUNT: usize = 6;

/// One immutable forecast/outcome pair from a frozen execution trajectory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefixReplayRow {
    pub forecast_id: ForecastId,
    pub run_id: u64,
    pub decision_sequence: u64,
    pub task_family: u32,
    pub operation_family: u32,
    pub forecast: RemainingCostForecast,
    pub observed_work: Option<ResourceVector>,
    pub terminal: Option<TerminalOutcome>,
    pub verifier_present: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct DimensionAccumulator {
    completed_count: usize,
    absolute_error_sum: u128,
    signed_error_sum: i128,
    q90_hits: usize,
    q95_hits: usize,
    censored_q90_definite_exceedances: usize,
    censored_q95_definite_exceedances: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct FamilyAccumulator {
    forecasts: usize,
    completed_verified: usize,
    right_censored: usize,
    competing_terminal: usize,
    pending: usize,
    joint_q90_hits: usize,
    dimensions: [DimensionAccumulator; DIMENSION_COUNT],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DimensionReplayMetrics {
    pub completed_labels: usize,
    pub q50_mean_absolute_error: Option<f64>,
    /// Positive bias means the q50 forecast over-predicts remaining cost.
    pub q50_mean_signed_bias: Option<f64>,
    /// Marginal upper coverage of q90 on exact verified labels.
    pub q90_marginal_coverage: Option<f64>,
    /// Marginal upper coverage of q95 on exact verified labels.
    pub q95_marginal_coverage: Option<f64>,
    /// Censored lower bounds already strictly above q90.
    pub censored_q90_definite_exceedances: usize,
    /// Censored lower bounds already strictly above q95.
    pub censored_q95_definite_exceedances: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TaskFamilyReplayMetrics {
    pub task_family: u32,
    pub forecasts: usize,
    pub completed_verified: usize,
    pub right_censored: usize,
    pub competing_terminal: usize,
    pub pending: usize,
    pub empirical_joint_q90_coverage: Option<f64>,
    pub dimensions: [DimensionReplayMetrics; DIMENSION_COUNT],
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrefixReplayReport {
    pub total_forecasts: usize,
    pub completed_verified: usize,
    pub right_censored: usize,
    pub competing_terminal: usize,
    pub pending: usize,
    /// Empirical all-six-dimensions q90 coverage; not a simultaneous guarantee.
    pub empirical_joint_q90_coverage: Option<f64>,
    pub dimensions: [DimensionReplayMetrics; DIMENSION_COUNT],
    pub by_task_family: Vec<TaskFamilyReplayMetrics>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrefixReplayEvaluation {
    /// Stable order: run ID, decision sequence, then forecast ID.
    pub rows: Vec<PrefixReplayRow>,
    pub report: PrefixReplayReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixReplayError {
    MissingCompletedVerifier,
    MissingCompletedObservation,
    NumericOverflow,
    InconsistentJournalOutcome,
}

impl std::fmt::Display for PrefixReplayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for PrefixReplayError {}

/// Deterministic rollout-prefix evaluator over the immutable forecast journal.
///
/// Exact prediction errors and coverage are computed only for sufficient-proof
/// outcomes with an independent verifier receipt. Right-censored lower bounds
/// can establish definite upper-envelope exceedances, but are never treated as
/// exact completion labels. Competing and pending outcomes remain separate.
pub struct PrefixReplayEvaluator;

impl PrefixReplayEvaluator {
    pub fn evaluate(
        journal: &ForecastJournal,
    ) -> Result<PrefixReplayEvaluation, PrefixReplayError> {
        let mut rows = journal
            .iter_forecasts()
            .map(|forecast| {
                let outcome = journal.outcome(forecast.id);
                PrefixReplayRow {
                    forecast_id: forecast.id,
                    run_id: forecast.run_id,
                    decision_sequence: forecast.decision_sequence,
                    task_family: forecast.task_family,
                    operation_family: forecast.operation_family,
                    forecast: forecast.forecast,
                    observed_work: outcome.map(|value| value.observed_work),
                    terminal: outcome.map(|value| value.terminal),
                    verifier_present: outcome
                        .and_then(|value| value.verifier_digest)
                        .is_some(),
                }
            })
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| (row.run_id, row.decision_sequence, row.forecast_id));

        let mut overall = FamilyAccumulator::default();
        let mut by_family = BTreeMap::<u32, FamilyAccumulator>::new();
        for row in &rows {
            update_accumulator(&mut overall, row)?;
            update_accumulator(by_family.entry(row.task_family).or_default(), row)?;
        }

        let report = PrefixReplayReport {
            total_forecasts: overall.forecasts,
            completed_verified: overall.completed_verified,
            right_censored: overall.right_censored,
            competing_terminal: overall.competing_terminal,
            pending: overall.pending,
            empirical_joint_q90_coverage: ratio(
                overall.joint_q90_hits,
                overall.completed_verified,
            ),
            dimensions: dimension_metrics(overall.dimensions),
            by_task_family: by_family
                .into_iter()
                .map(|(task_family, values)| TaskFamilyReplayMetrics {
                    task_family,
                    forecasts: values.forecasts,
                    completed_verified: values.completed_verified,
                    right_censored: values.right_censored,
                    competing_terminal: values.competing_terminal,
                    pending: values.pending,
                    empirical_joint_q90_coverage: ratio(
                        values.joint_q90_hits,
                        values.completed_verified,
                    ),
                    dimensions: dimension_metrics(values.dimensions),
                })
                .collect(),
        };

        Ok(PrefixReplayEvaluation { rows, report })
    }
}

fn update_accumulator(
    accumulator: &mut FamilyAccumulator,
    row: &PrefixReplayRow,
) -> Result<(), PrefixReplayError> {
    accumulator.forecasts = accumulator
        .forecasts
        .checked_add(1)
        .ok_or(PrefixReplayError::NumericOverflow)?;

    match row.terminal {
        None => {
            accumulator.pending = accumulator
                .pending
                .checked_add(1)
                .ok_or(PrefixReplayError::NumericOverflow)?;
        }
        Some(TerminalOutcome::SufficientProof) => {
            if !row.verifier_present {
                return Err(PrefixReplayError::MissingCompletedVerifier);
            }
            let observed = row
                .observed_work
                .ok_or(PrefixReplayError::MissingCompletedObservation)?;
            accumulator.completed_verified = accumulator
                .completed_verified
                .checked_add(1)
                .ok_or(PrefixReplayError::NumericOverflow)?;

            let actual = resource_values(observed);
            let q50 = resource_values(row.forecast.remaining_q50);
            let q90 = resource_values(row.forecast.remaining_q90);
            let q95 = resource_values(row.forecast.remaining_q95);
            let mut joint_q90_covered = true;

            for index in 0..DIMENSION_COUNT {
                let metric = &mut accumulator.dimensions[index];
                metric.completed_count = metric
                    .completed_count
                    .checked_add(1)
                    .ok_or(PrefixReplayError::NumericOverflow)?;
                metric.absolute_error_sum = metric
                    .absolute_error_sum
                    .checked_add(q50[index].abs_diff(actual[index]))
                    .ok_or(PrefixReplayError::NumericOverflow)?;

                let signed_error = i128::try_from(q50[index])
                    .map_err(|_| PrefixReplayError::NumericOverflow)?
                    .checked_sub(
                        i128::try_from(actual[index])
                            .map_err(|_| PrefixReplayError::NumericOverflow)?,
                    )
                    .ok_or(PrefixReplayError::NumericOverflow)?;
                metric.signed_error_sum = metric
                    .signed_error_sum
                    .checked_add(signed_error)
                    .ok_or(PrefixReplayError::NumericOverflow)?;

                if actual[index] <= q90[index] {
                    metric.q90_hits = metric
                        .q90_hits
                        .checked_add(1)
                        .ok_or(PrefixReplayError::NumericOverflow)?;
                } else {
                    joint_q90_covered = false;
                }
                if actual[index] <= q95[index] {
                    metric.q95_hits = metric
                        .q95_hits
                        .checked_add(1)
                        .ok_or(PrefixReplayError::NumericOverflow)?;
                }
            }

            if joint_q90_covered {
                accumulator.joint_q90_hits = accumulator
                    .joint_q90_hits
                    .checked_add(1)
                    .ok_or(PrefixReplayError::NumericOverflow)?;
            }
        }
        Some(TerminalOutcome::RightCensored(_)) => {
            accumulator.right_censored = accumulator
                .right_censored
                .checked_add(1)
                .ok_or(PrefixReplayError::NumericOverflow)?;
            let lower_bound = row
                .observed_work
                .ok_or(PrefixReplayError::InconsistentJournalOutcome)?;
            let lower = resource_values(lower_bound);
            let q90 = resource_values(row.forecast.remaining_q90);
            let q95 = resource_values(row.forecast.remaining_q95);

            for index in 0..DIMENSION_COUNT {
                if lower[index] > q90[index] {
                    accumulator.dimensions[index].censored_q90_definite_exceedances = accumulator
                        .dimensions[index]
                        .censored_q90_definite_exceedances
                        .checked_add(1)
                        .ok_or(PrefixReplayError::NumericOverflow)?;
                }
                if lower[index] > q95[index] {
                    accumulator.dimensions[index].censored_q95_definite_exceedances = accumulator
                        .dimensions[index]
                        .censored_q95_definite_exceedances
                        .checked_add(1)
                        .ok_or(PrefixReplayError::NumericOverflow)?;
                }
            }
        }
        Some(TerminalOutcome::CompetingTerminal(_)) => {
            accumulator.competing_terminal = accumulator
                .competing_terminal
                .checked_add(1)
                .ok_or(PrefixReplayError::NumericOverflow)?;
        }
    }

    Ok(())
}

fn dimension_metrics(
    values: [DimensionAccumulator; DIMENSION_COUNT],
) -> [DimensionReplayMetrics; DIMENSION_COUNT] {
    std::array::from_fn(|index| {
        let value = values[index];
        DimensionReplayMetrics {
            completed_labels: value.completed_count,
            q50_mean_absolute_error: mean_u128(
                value.absolute_error_sum,
                value.completed_count,
            ),
            q50_mean_signed_bias: mean_i128(value.signed_error_sum, value.completed_count),
            q90_marginal_coverage: ratio(value.q90_hits, value.completed_count),
            q95_marginal_coverage: ratio(value.q95_hits, value.completed_count),
            censored_q90_definite_exceedances: value.censored_q90_definite_exceedances,
            censored_q95_definite_exceedances: value.censored_q95_definite_exceedances,
        }
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CensorReason, CompetingReason, ForecastOutcomeReceipt, ForecastReceipt, ShiftStatus,
    };

    fn forecast(id: u64, run_id: u64, sequence: u64, family: u32) -> ForecastReceipt {
        ForecastReceipt {
            id: ForecastId(id),
            run_id,
            decision_sequence: sequence,
            task_family: family,
            operation_family: 9,
            forecast: RemainingCostForecast {
                pre_state_digest: [1; 32],
                frontier_digest: [2; 32],
                model_digest: [3; 32],
                calibration_digest: [4; 32],
                completion_probability: 0.9,
                remaining_q50: ResourceVector::new(10, 2, 3, 4, 5, 6),
                remaining_q90: ResourceVector::new(20, 3, 5, 7, 8, 9),
                remaining_q95: ResourceVector::new(25, 4, 6, 8, 9, 10),
                mandatory_verification_reserve: ResourceVector::ZERO,
                support_count: 64,
                shift_status: ShiftStatus::InDomain,
            },
        }
    }

    fn outcome(
        id: u64,
        sequence: u64,
        work: ResourceVector,
        terminal: TerminalOutcome,
    ) -> ForecastOutcomeReceipt {
        ForecastOutcomeReceipt {
            forecast_id: ForecastId(id),
            observed_sequence: sequence,
            observed_work: work,
            terminal,
            evidence_digest: [5; 32],
            meter_digest: [6; 32],
            verifier_digest: (terminal == TerminalOutcome::SufficientProof)
                .then_some([7; 32]),
        }
    }

    #[test]
    fn replay_scores_only_independently_verified_exact_labels() {
        let mut journal = ForecastJournal::default();
        journal.append_forecast(forecast(1, 10, 1, 7)).unwrap();
        journal
            .append_outcome(outcome(
                1,
                2,
                ResourceVector::new(12, 2, 4, 4, 6, 6),
                TerminalOutcome::SufficientProof,
            ))
            .unwrap();

        let report = PrefixReplayEvaluator::evaluate(&journal).unwrap().report;
        assert_eq!(report.total_forecasts, 1);
        assert_eq!(report.completed_verified, 1);
        assert_eq!(report.pending, 0);
        assert_eq!(report.dimensions[0].q50_mean_absolute_error, Some(2.0));
        assert_eq!(report.dimensions[0].q90_marginal_coverage, Some(1.0));
        assert_eq!(report.empirical_joint_q90_coverage, Some(1.0));
    }

    #[test]
    fn censoring_and_competing_terminals_are_not_exact_labels() {
        let mut journal = ForecastJournal::default();
        journal.append_forecast(forecast(1, 10, 1, 7)).unwrap();
        journal.append_forecast(forecast(2, 10, 2, 7)).unwrap();
        journal.append_forecast(forecast(3, 10, 3, 8)).unwrap();
        journal
            .append_outcome(outcome(
                1,
                2,
                ResourceVector::new(21, 1, 2, 3, 4, 5),
                TerminalOutcome::RightCensored(CensorReason::Timeout),
            ))
            .unwrap();
        journal
            .append_outcome(outcome(
                2,
                3,
                ResourceVector::new(5, 1, 2, 3, 4, 5),
                TerminalOutcome::CompetingTerminal(CompetingReason::PermissionRevoked),
            ))
            .unwrap();

        let report = PrefixReplayEvaluator::evaluate(&journal).unwrap().report;
        assert_eq!(report.right_censored, 1);
        assert_eq!(report.competing_terminal, 1);
        assert_eq!(report.pending, 1);
        assert_eq!(report.completed_verified, 0);
        assert_eq!(report.dimensions[0].completed_labels, 0);
        assert_eq!(report.dimensions[0].censored_q90_definite_exceedances, 1);
        assert_eq!(report.dimensions[0].q50_mean_absolute_error, None);
    }

    #[test]
    fn task_family_summaries_and_row_order_are_deterministic() {
        let mut journal = ForecastJournal::default();
        journal.append_forecast(forecast(2, 20, 2, 8)).unwrap();
        journal.append_forecast(forecast(1, 10, 1, 7)).unwrap();
        let evaluation = PrefixReplayEvaluator::evaluate(&journal).unwrap();
        assert_eq!(evaluation.rows[0].forecast_id, ForecastId(1));
        assert_eq!(evaluation.rows[1].forecast_id, ForecastId(2));
        assert_eq!(evaluation.report.by_task_family.len(), 2);
        assert_eq!(evaluation.report.by_task_family[0].task_family, 7);
        assert_eq!(evaluation.report.by_task_family[1].task_family, 8);
    }
}
