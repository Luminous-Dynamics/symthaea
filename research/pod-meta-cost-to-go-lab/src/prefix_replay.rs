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
    pub remaining_budget: Option<ResourceVector>,
    pub predicted_feasible: Option<bool>,
    pub actual_feasible: Option<bool>,
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

/// Remaining hard budget captured at the same pre-action state as a forecast.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixBudgetReceipt {
    pub run_id: u64,
    pub decision_sequence: u64,
    pub pre_state_digest: [u8; 32],
    pub remaining_budget: ResourceVector,
    pub evidence_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FeasibilityReplayMetrics {
    pub budgeted_prefixes: usize,
    pub predicted_feasible: usize,
    pub predicted_infeasible: usize,
    pub resolved_predicted_feasible: usize,
    pub resolved_predicted_infeasible: usize,
    pub abstained_predictions: usize,
    pub abstained_with_known_actual: usize,
    pub actual_feasibility_known: usize,
    pub actual_feasible: usize,
    pub false_feasible: usize,
    pub false_infeasible: usize,
    pub true_feasible: usize,
    pub true_infeasible: usize,
    pub unknown_actual_outcomes: usize,
    /// False feasible / all predictions declared feasible.
    pub false_feasible_rate: Option<f64>,
    /// False infeasible / all predictions declared infeasible.
    pub false_infeasible_rate: Option<f64>,
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
    pub feasibility: FeasibilityReplayMetrics,
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
    InvalidFeasibilityThreshold,
    InvalidBudgetReceipt,
    DuplicateBudgetReceipt,
    BudgetForUnknownForecast,
    BudgetPreStateMismatch,
    BudgetArithmeticOverflow,
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
        Self::evaluate_with_budgets(journal, &[], 0.5)
    }

    /// Replay using state-bound remaining-budget receipts. A prefix is
    /// predicted feasible only when the probability threshold passes and the
    /// q90 vector plus mandatory verification reserve fits every budget
    /// dimension. Actual feasibility is known for exact sufficient-proof
    /// outcomes, or when a censored lower bound already exceeds the budget.
    pub fn evaluate_with_budgets(
        journal: &ForecastJournal,
        budgets: &[PrefixBudgetReceipt],
        minimum_completion_probability: f64,
    ) -> Result<PrefixReplayEvaluation, PrefixReplayError> {
        if !minimum_completion_probability.is_finite()
            || !(0.0..=1.0).contains(&minimum_completion_probability)
        {
            return Err(PrefixReplayError::InvalidFeasibilityThreshold);
        }

        let mut budgets_by_key = BTreeMap::new();
        for budget in budgets {
            if budget.run_id == 0
                || budget.decision_sequence == 0
                || is_zero(budget.pre_state_digest)
                || is_zero(budget.evidence_digest)
            {
                return Err(PrefixReplayError::InvalidBudgetReceipt);
            }
            if budgets_by_key
                .insert((budget.run_id, budget.decision_sequence), *budget)
                .is_some()
            {
                return Err(PrefixReplayError::DuplicateBudgetReceipt);
            }
        }

        let forecasts = journal.iter_forecasts().collect::<Vec<_>>();
        for budget in budgets {
            let forecast = forecasts
                .iter()
                .find(|forecast| {
                    forecast.run_id == budget.run_id
                        && forecast.decision_sequence == budget.decision_sequence
                })
                .ok_or(PrefixReplayError::BudgetForUnknownForecast)?;
            if forecast.forecast.pre_state_digest != budget.pre_state_digest {
                return Err(PrefixReplayError::BudgetPreStateMismatch);
            }
        }

        let mut rows = forecasts
            .into_iter()
            .map(|forecast| {
                let outcome = journal.outcome(forecast.id);
                let budget = budgets_by_key
                    .get(&(forecast.run_id, forecast.decision_sequence))
                    .copied();
                let predicted_feasible = budget
                    .map(|receipt| {
                        predicted_feasible(
                            forecast.forecast,
                            receipt.remaining_budget,
                            minimum_completion_probability,
                        )
                    })
                    .transpose()?
                    .flatten();
                let actual_feasible = budget
                    .and_then(|receipt| actual_feasible(outcome, receipt.remaining_budget));
                Ok(PrefixReplayRow {
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
                    remaining_budget: budget.map(|value| value.remaining_budget),
                    predicted_feasible,
                    actual_feasible,
                })
            })
            .collect::<Result<Vec<_>, PrefixReplayError>>()?;
        rows.sort_by_key(|row| (row.run_id, row.decision_sequence, row.forecast_id));

        let mut overall = FamilyAccumulator::default();
        let mut feasibility = FeasibilityReplayMetrics::default();
        let mut by_family = BTreeMap::<u32, FamilyAccumulator>::new();
        for row in &rows {
            update_accumulator(&mut overall, row)?;
            update_accumulator(by_family.entry(row.task_family).or_default(), row)?;
            update_feasibility(&mut feasibility, row)?;
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
            feasibility,
        };

        Ok(PrefixReplayEvaluation { rows, report })
    }
}

fn predicted_feasible(
    forecast: RemainingCostForecast,
    remaining_budget: ResourceVector,
    minimum_completion_probability: f64,
) -> Result<Option<bool>, PrefixReplayError> {
    if forecast.shift_status != crate::ShiftStatus::InDomain || forecast.support_count == 0 {
        return Ok(None);
    }
    let required = forecast
        .remaining_q90
        .checked_add(forecast.mandatory_verification_reserve)
        .ok_or(PrefixReplayError::BudgetArithmeticOverflow)?;
    Ok(Some(
        forecast.completion_probability >= minimum_completion_probability
            && within_budget(required, remaining_budget),
    ))
}

fn actual_feasible(
    outcome: Option<&crate::ForecastOutcomeReceipt>,
    remaining_budget: ResourceVector,
) -> Option<bool> {
    let outcome = outcome?;
    match outcome.terminal {
        TerminalOutcome::SufficientProof => Some(within_budget(
            outcome.observed_work,
            remaining_budget,
        )),
        TerminalOutcome::RightCensored(_) => {
            let lower_bound = outcome.observed_work;
            (!within_budget(lower_bound, remaining_budget)).then_some(false)
        }
        TerminalOutcome::CompetingTerminal(_) => None,
    }
}

fn within_budget(cost: ResourceVector, budget: ResourceVector) -> bool {
    let cost = resource_values(cost);
    let budget = resource_values(budget);
    (0..DIMENSION_COUNT).all(|index| cost[index] <= budget[index])
}

fn update_feasibility(
    metrics: &mut FeasibilityReplayMetrics,
    row: &PrefixReplayRow,
) -> Result<(), PrefixReplayError> {
    if row.remaining_budget.is_none() {
        return Ok(());
    }
    increment(&mut metrics.budgeted_prefixes)?;

    let Some(predicted) = row.predicted_feasible else {
        increment(&mut metrics.abstained_predictions)?;
        match row.actual_feasible {
            Some(actual) => {
                increment(&mut metrics.abstained_with_known_actual)?;
                increment(&mut metrics.actual_feasibility_known)?;
                if actual {
                    increment(&mut metrics.actual_feasible)?;
                }
            }
            None => increment(&mut metrics.unknown_actual_outcomes)?,
        }
        return Ok(());
    };

    if predicted {
        increment(&mut metrics.predicted_feasible)?;
    } else {
        increment(&mut metrics.predicted_infeasible)?;
    }

    let Some(actual) = row.actual_feasible else {
        increment(&mut metrics.unknown_actual_outcomes)?;
        return Ok(());
    };
    increment(&mut metrics.actual_feasibility_known)?;
    if predicted {
        increment(&mut metrics.resolved_predicted_feasible)?;
    } else {
        increment(&mut metrics.resolved_predicted_infeasible)?;
    }
    if actual {
        increment(&mut metrics.actual_feasible)?;
    }

    match (predicted, actual) {
        (true, true) => increment(&mut metrics.true_feasible)?,
        (true, false) => increment(&mut metrics.false_feasible)?,
        (false, true) => increment(&mut metrics.false_infeasible)?,
        (false, false) => increment(&mut metrics.true_infeasible)?,
    }

    metrics.false_feasible_rate = ratio(
        metrics.false_feasible,
        metrics.resolved_predicted_feasible,
    );
    metrics.false_infeasible_rate = ratio(
        metrics.false_infeasible,
        metrics.resolved_predicted_infeasible,
    );
    Ok(())
}

fn increment(value: &mut usize) -> Result<(), PrefixReplayError> {
    *value = value
        .checked_add(1)
        .ok_or(PrefixReplayError::NumericOverflow)?;
    Ok(())
}

fn is_zero(digest: [u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
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
    fn budget_bound_prefixes_score_false_feasible_and_false_infeasible() {
        let mut journal = ForecastJournal::default();
        for id in 1..=4 {
            journal
                .append_forecast(forecast(id, 100 + id, 1, 7))
                .unwrap();
        }
        journal
            .append_outcome(outcome(
                1,
                2,
                ResourceVector::new(12, 2, 4, 4, 6, 6),
                TerminalOutcome::SufficientProof,
            ))
            .unwrap();
        journal
            .append_outcome(outcome(
                2,
                2,
                ResourceVector::new(26, 1, 1, 1, 1, 1),
                TerminalOutcome::RightCensored(CensorReason::Timeout),
            ))
            .unwrap();
        journal
            .append_outcome(outcome(
                3,
                2,
                ResourceVector::new(9, 1, 2, 3, 4, 5),
                TerminalOutcome::SufficientProof,
            ))
            .unwrap();
        journal
            .append_outcome(outcome(
                4,
                2,
                ResourceVector::new(1, 1, 1, 1, 1, 1),
                TerminalOutcome::CompetingTerminal(CompetingReason::PermissionRevoked),
            ))
            .unwrap();

        let generous = ResourceVector::new(25, 4, 6, 8, 9, 10);
        let tight = ResourceVector::new(10, 2, 3, 4, 5, 6);
        let budgets = (1..=4)
            .map(|id| PrefixBudgetReceipt {
                run_id: 100 + id,
                decision_sequence: 1,
                pre_state_digest: [1; 32],
                remaining_budget: if id <= 2 { generous } else { tight },
                evidence_digest: [8; 32],
            })
            .collect::<Vec<_>>();

        let report = PrefixReplayEvaluator::evaluate_with_budgets(&journal, &budgets, 0.8)
            .unwrap()
            .report;
        assert_eq!(report.feasibility.budgeted_prefixes, 4);
        assert_eq!(report.feasibility.predicted_feasible, 2);
        assert_eq!(report.feasibility.predicted_infeasible, 2);
        assert_eq!(report.feasibility.actual_feasibility_known, 3);
        assert_eq!(report.feasibility.actual_feasible, 2);
        assert_eq!(report.feasibility.true_feasible, 1);
        assert_eq!(report.feasibility.false_feasible, 1);
        assert_eq!(report.feasibility.false_infeasible, 1);
        assert_eq!(report.feasibility.unknown_actual_outcomes, 1);
        assert_eq!(report.feasibility.false_feasible_rate, Some(0.5));
        assert_eq!(report.feasibility.resolved_predicted_feasible, 2);
        assert_eq!(report.feasibility.resolved_predicted_infeasible, 1);
        assert_eq!(report.feasibility.false_infeasible_rate, Some(1.0));
    }

    #[test]
    fn budget_receipts_must_bind_to_exact_forecast_state_and_prefix() {
        let mut journal = ForecastJournal::default();
        journal.append_forecast(forecast(1, 10, 1, 7)).unwrap();
        let wrong_state = [99; 32];
        let budget = PrefixBudgetReceipt {
            run_id: 10,
            decision_sequence: 1,
            pre_state_digest: wrong_state,
            remaining_budget: ResourceVector::new(100, 100, 100, 100, 100, 100),
            evidence_digest: [8; 32],
        };
        assert_eq!(
            PrefixReplayEvaluator::evaluate_with_budgets(&journal, &[budget], 0.8),
            Err(PrefixReplayError::BudgetPreStateMismatch)
        );

        let duplicate = PrefixBudgetReceipt {
            pre_state_digest: [1; 32],
            ..budget
        };
        assert_eq!(
            PrefixReplayEvaluator::evaluate_with_budgets(
                &journal,
                &[duplicate, duplicate],
                0.8
            ),
            Err(PrefixReplayError::DuplicateBudgetReceipt)
        );
    }

    #[test]
    fn out_of_domain_forecasts_abstain_instead_of_claiming_feasibility() {
        let mut journal = ForecastJournal::default();
        let mut shifted = forecast(1, 10, 1, 7);
        shifted.forecast.shift_status = crate::ShiftStatus::UnsupportedShift;
        journal.append_forecast(shifted).unwrap();
        journal
            .append_outcome(outcome(
                1,
                2,
                ResourceVector::new(8, 1, 2, 3, 4, 5),
                TerminalOutcome::SufficientProof,
            ))
            .unwrap();
        let budget = PrefixBudgetReceipt {
            run_id: 10,
            decision_sequence: 1,
            pre_state_digest: [1; 32],
            remaining_budget: ResourceVector::new(100, 100, 100, 100, 100, 100),
            evidence_digest: [8; 32],
        };
        let report = PrefixReplayEvaluator::evaluate_with_budgets(
            &journal,
            &[budget],
            0.8,
        )
        .unwrap()
        .report;
        assert_eq!(report.feasibility.budgeted_prefixes, 1);
        assert_eq!(report.feasibility.abstained_predictions, 1);
        assert_eq!(report.feasibility.abstained_with_known_actual, 1);
        assert_eq!(report.feasibility.predicted_feasible, 0);
        assert_eq!(report.feasibility.predicted_infeasible, 0);
        assert_eq!(report.feasibility.actual_feasibility_known, 1);
        assert_eq!(report.feasibility.false_feasible, 0);
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
