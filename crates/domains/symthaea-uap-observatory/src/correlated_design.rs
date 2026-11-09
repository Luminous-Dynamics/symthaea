// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Selected-set optimal design with correlated residual covariance.
//!
//! Unlike the independent-row design, this module stores the covariance over
//! the complete candidate observation pool. After actions are selected, it
//! extracts the selected principal covariance submatrix and jointly whitens
//! the corresponding raw Jacobian rows. This is required when residual errors
//! remain correlated after conditioning on explicitly represented nuisance
//! states.

use super::budgeted_design::{
    BudgetedDesignConfig, BudgetedDesignError, BudgetedDesignReport, DesignSelectionStep,
    ScenarioDesignScore, cholesky_factor, cholesky_log_determinant,
    marginal_target_log_determinant,
};
use crate::observability::analyze_jacobian;
use std::collections::BTreeSet;

/// One scenario with a fixed candidate-observation pool and its full covariance.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrelatedDesignScenario {
    pub scenario_id: String,
    /// Number of leading parameters whose marginal uncertainty is the objective.
    pub target_parameter_count: usize,
    /// Non-normalized positive scenario weight.
    pub weight: f64,
    /// Positive-definite prior precision for target and nuisance parameters.
    pub prior_precision: Vec<Vec<f64>>,
    /// Raw (not pre-whitened) measurement Jacobian, one row per pool observation.
    pub measurement_jacobian_rows: Vec<Vec<f64>>,
    /// Full residual covariance, in exactly the same row ordering.
    pub measurement_covariance: Vec<Vec<f64>>,
}

/// Candidate action selecting observations from one or more scenario pools.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrelatedExperimentOption {
    pub option_id: String,
    pub cost: f64,
    pub exclusive_group: Option<String>,
    /// Observation indices selected by the action in each scenario.
    /// An empty list means that the action is unavailable in that scenario.
    pub observation_indices_by_scenario: Vec<Vec<usize>>,
}

/// Maximum complete candidate-observation pool accepted by this dense reference.
pub const MAX_CORRELATED_DESIGN_OBSERVATIONS: usize = 512;

/// Greedy target-marginal D-optimal selection using selected-set covariance.
///
/// Candidate measurement rows are not independently whitened. The selected
/// covariance principal submatrix is factored and the corresponding Jacobian
/// rows are jointly whitened at each evaluation. The same budget, fidelity
/// exclusivity, scenario weights and report contract as the independent design
/// are retained.
pub fn greedy_correlated_budgeted_design(
    scenarios: &[CorrelatedDesignScenario],
    options: &[CorrelatedExperimentOption],
    config: &BudgetedDesignConfig,
) -> Result<BudgetedDesignReport, BudgetedDesignError> {
    let dimension = validate_correlated_inputs(scenarios, options, config)?;
    let weights = normalized_weights(scenarios);

    let mut selected = Vec::new();
    let mut selected_set = BTreeSet::new();
    let mut selected_groups = BTreeSet::new();
    let mut total_cost = 0.0;
    let mut steps = Vec::new();
    let mut current_scores =
        evaluate_correlated_selection(scenarios, options, &selected, config, dimension)?;
    let mut current_expected = weighted_target_gain(&current_scores, &weights);

    loop {
        let mut best: Option<(usize, f64, f64)> = None;
        for (index, option) in options.iter().enumerate() {
            if selected_set.contains(&index)
                || total_cost + option.cost > config.budget + 1.0e-12
                || option
                    .exclusive_group
                    .as_ref()
                    .is_some_and(|group| selected_groups.contains(group))
                || overlaps_selected_measurements(scenarios, options, &selected, index)
            {
                continue;
            }

            let mut trial = selected.clone();
            trial.push(index);
            let scores =
                evaluate_correlated_selection(scenarios, options, &trial, config, dimension)?;
            let marginal_gain = weighted_target_gain(&scores, &weights) - current_expected;
            let gain_per_cost = marginal_gain / option.cost;
            if !marginal_gain.is_finite() || !gain_per_cost.is_finite() {
                return Err(BudgetedDesignError::NumericalOverflow);
            }

            let replace = match best {
                None => true,
                Some((best_index, best_gain, best_density)) => {
                    gain_per_cost > best_density + 1.0e-12
                        || ((gain_per_cost - best_density).abs() <= 1.0e-12
                            && option.option_id < options[best_index].option_id)
                        || ((gain_per_cost - best_density).abs() <= 1.0e-12
                            && option.option_id == options[best_index].option_id
                            && marginal_gain > best_gain)
                }
            };
            if replace {
                best = Some((index, marginal_gain, gain_per_cost));
            }
        }

        let Some((best_index, marginal_gain, gain_per_cost)) = best else {
            break;
        };
        if marginal_gain <= 1.0e-12 {
            break;
        }

        let option = &options[best_index];
        selected.push(best_index);
        selected_set.insert(best_index);
        if let Some(group) = &option.exclusive_group {
            selected_groups.insert(group.clone());
        }
        total_cost += option.cost;
        steps.push(DesignSelectionStep {
            option_id: option.option_id.clone(),
            cost: option.cost,
            marginal_expected_gain_nats: marginal_gain,
            marginal_gain_per_cost: gain_per_cost,
        });

        current_scores =
            evaluate_correlated_selection(scenarios, options, &selected, config, dimension)?;
        current_expected = weighted_target_gain(&current_scores, &weights);
    }

    correlated_report(
        options,
        config,
        &selected,
        total_cost,
        steps,
        current_scores,
        &weights,
    )
}

/// Exhaustive subset oracle for small correlated-observation candidate sets.
pub fn exhaustive_correlated_budgeted_design(
    scenarios: &[CorrelatedDesignScenario],
    options: &[CorrelatedExperimentOption],
    config: &BudgetedDesignConfig,
) -> Result<BudgetedDesignReport, BudgetedDesignError> {
    let dimension = validate_correlated_inputs(scenarios, options, config)?;
    const MAX_EXHAUSTIVE_OPTIONS: usize = 20;
    if options.len() > MAX_EXHAUSTIVE_OPTIONS {
        return Err(BudgetedDesignError::TooManyOptionsForExhaustive {
            count: options.len(),
            maximum: MAX_EXHAUSTIVE_OPTIONS,
        });
    }

    let weights = normalized_weights(scenarios);
    let subset_count = 1_u64 << options.len();
    let mut best_selection = Vec::new();
    let mut best_cost = 0.0;
    let mut best_scores =
        evaluate_correlated_selection(scenarios, options, &[], config, dimension)?;
    let mut best_gain = weighted_target_gain(&best_scores, &weights);
    let mut best_ids: Vec<String> = Vec::new();

    for mask in 1..subset_count {
        let mut selected = Vec::new();
        let mut groups = BTreeSet::new();
        let mut cost = 0.0;
        let mut feasible = true;

        for (index, option) in options.iter().enumerate() {
            if mask & (1_u64 << index) == 0 {
                continue;
            }
            cost += option.cost;
            if cost > config.budget + 1.0e-12
                || option
                    .exclusive_group
                    .as_ref()
                    .is_some_and(|group| !groups.insert(group.clone()))
                || overlaps_selected_measurements(scenarios, options, &selected, index)
            {
                feasible = false;
                break;
            }
            selected.push(index);
        }
        if !feasible {
            continue;
        }

        let scores =
            evaluate_correlated_selection(scenarios, options, &selected, config, dimension)?;
        let gain = weighted_target_gain(&scores, &weights);
        let mut ids: Vec<String> = selected
            .iter()
            .map(|index| options[*index].option_id.clone())
            .collect();
        ids.sort();

        let replace = gain > best_gain + 1.0e-12
            || ((gain - best_gain).abs() <= 1.0e-12 && ids < best_ids);
        if replace {
            best_selection = selected;
            best_cost = cost;
            best_scores = scores;
            best_gain = gain;
            best_ids = ids;
        }
    }

    correlated_report(
        options,
        config,
        &best_selection,
        best_cost,
        Vec::new(),
        best_scores,
        &weights,
    )
}

fn validate_correlated_inputs(
    scenarios: &[CorrelatedDesignScenario],
    options: &[CorrelatedExperimentOption],
    config: &BudgetedDesignConfig,
) -> Result<usize, BudgetedDesignError> {
    if scenarios.is_empty() {
        return Err(BudgetedDesignError::NoScenarios);
    }
    if options.is_empty() {
        return Err(BudgetedDesignError::NoOptions);
    }
    if !config.budget.is_finite() || config.budget < 0.0 {
        return Err(BudgetedDesignError::InvalidBudget);
    }

    let dimension = scenarios[0].prior_precision.len();
    if dimension == 0 {
        return Err(BudgetedDesignError::InvalidPriorShape {
            scenario_id: scenarios[0].scenario_id.clone(),
        });
    }
    if dimension > crate::observability::MAX_OBSERVABILITY_PARAMETERS {
        return Err(BudgetedDesignError::TooManyParameters {
            count: dimension,
            maximum: crate::observability::MAX_OBSERVABILITY_PARAMETERS,
        });
    }
    if config.parameter_scales.len() != dimension {
        return Err(BudgetedDesignError::ParameterScaleCount {
            expected: dimension,
            actual: config.parameter_scales.len(),
        });
    }
    for (column, scale) in config.parameter_scales.iter().copied().enumerate() {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(BudgetedDesignError::InvalidParameterScale { column });
        }
    }

    let mut scenario_ids = BTreeSet::new();
    for (scenario_index, scenario) in scenarios.iter().enumerate() {
        if scenario.scenario_id.trim().is_empty() {
            return Err(BudgetedDesignError::InvalidScenarioId {
                index: scenario_index,
            });
        }
        if !scenario_ids.insert(scenario.scenario_id.as_str()) {
            return Err(BudgetedDesignError::DuplicateScenarioId(
                scenario.scenario_id.clone(),
            ));
        }
        if !scenario.weight.is_finite() || scenario.weight <= 0.0 {
            return Err(BudgetedDesignError::InvalidScenarioWeight {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        if scenario.target_parameter_count == 0 || scenario.target_parameter_count > dimension {
            return Err(BudgetedDesignError::InvalidTargetParameterCount {
                scenario_id: scenario.scenario_id.clone(),
                count: scenario.target_parameter_count,
                maximum: dimension,
            });
        }
        if scenario.prior_precision.len() != dimension
            || scenario
                .prior_precision
                .iter()
                .any(|row| row.len() != dimension)
        {
            return Err(BudgetedDesignError::InvalidPriorShape {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        let prior_scale = scenario
            .prior_precision
            .iter()
            .flatten()
            .fold(1.0_f64, |maximum, value| maximum.max(value.abs()));
        for row in 0..dimension {
            for column in 0..dimension {
                let value = scenario.prior_precision[row][column];
                if !value.is_finite() {
                    return Err(BudgetedDesignError::NumericalOverflow);
                }
                if (value - scenario.prior_precision[column][row]).abs() > 1.0e-12 * prior_scale {
                    return Err(BudgetedDesignError::InvalidPriorSymmetry {
                        scenario_id: scenario.scenario_id.clone(),
                    });
                }
            }
        }
        cholesky_factor(&scenario.prior_precision).map_err(|_| {
            BudgetedDesignError::PriorNotPositiveDefinite {
                scenario_id: scenario.scenario_id.clone(),
            }
        })?;

        let measurement_count = scenario.measurement_jacobian_rows.len();
        if measurement_count > MAX_CORRELATED_DESIGN_OBSERVATIONS {
            return Err(BudgetedDesignError::TooManyMeasurements {
                scenario_id: scenario.scenario_id.clone(),
                count: measurement_count,
                maximum: MAX_CORRELATED_DESIGN_OBSERVATIONS,
            });
        }
        if scenario.measurement_covariance.len() != measurement_count
            || scenario
                .measurement_covariance
                .iter()
                .any(|row| row.len() != measurement_count)
        {
            return Err(BudgetedDesignError::InvalidMeasurementCovarianceShape {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        for (row_index, row) in scenario.measurement_jacobian_rows.iter().enumerate() {
            if row.len() != dimension {
                return Err(BudgetedDesignError::MeasurementJacobianShape {
                    scenario_id: scenario.scenario_id.clone(),
                    row_index,
                    expected: dimension,
                    actual: row.len(),
                });
            }
            for (column, value) in row.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(BudgetedDesignError::NonFiniteJacobian {
                        option_id: scenario.scenario_id.clone(),
                        scenario_index,
                        row_index,
                        column,
                    });
                }
            }
        }
        let covariance_scale = scenario
            .measurement_covariance
            .iter()
            .flatten()
            .fold(1.0_f64, |maximum, value| maximum.max(value.abs()));
        for row in 0..measurement_count {
            for column in 0..measurement_count {
                let value = scenario.measurement_covariance[row][column];
                if !value.is_finite() {
                    return Err(BudgetedDesignError::NumericalOverflow);
                }
                if (value - scenario.measurement_covariance[column][row]).abs()
                    > 1.0e-12 * covariance_scale
                {
                    return Err(BudgetedDesignError::InvalidMeasurementCovarianceSymmetry {
                        scenario_id: scenario.scenario_id.clone(),
                    });
                }
            }
        }
        if measurement_count > 0 {
            cholesky_factor(&scenario.measurement_covariance).map_err(|_| {
                BudgetedDesignError::MeasurementCovarianceNotPositiveDefinite {
                    scenario_id: scenario.scenario_id.clone(),
                }
            })?;
        }
    }

    let mut option_ids = BTreeSet::new();
    for (option_index, option) in options.iter().enumerate() {
        if option.option_id.trim().is_empty() {
            return Err(BudgetedDesignError::InvalidOptionId { index: option_index });
        }
        if !option_ids.insert(option.option_id.as_str()) {
            return Err(BudgetedDesignError::DuplicateOptionId(
                option.option_id.clone(),
            ));
        }
        if !option.cost.is_finite() || option.cost <= 0.0 {
            return Err(BudgetedDesignError::InvalidOptionCost {
                option_id: option.option_id.clone(),
            });
        }
        if option
            .exclusive_group
            .as_ref()
            .is_some_and(|group| group.trim().is_empty())
        {
            return Err(BudgetedDesignError::EmptyExclusiveGroup {
                option_id: option.option_id.clone(),
            });
        }
        if option.observation_indices_by_scenario.len() != scenarios.len() {
            return Err(BudgetedDesignError::MeasurementIndexScenarioCountMismatch {
                option_id: option.option_id.clone(),
                expected: scenarios.len(),
                actual: option.observation_indices_by_scenario.len(),
            });
        }
        for (scenario_index, indices) in option
            .observation_indices_by_scenario
            .iter()
            .enumerate()
        {
            let measurement_count =
                scenarios[scenario_index].measurement_jacobian_rows.len();
            let mut seen = BTreeSet::new();
            for index in indices.iter().copied() {
                if index >= measurement_count {
                    return Err(BudgetedDesignError::MeasurementIndexOutOfRange {
                        option_id: option.option_id.clone(),
                        scenario_index,
                        index,
                        measurement_count,
                    });
                }
                if !seen.insert(index) {
                    return Err(BudgetedDesignError::DuplicateMeasurementIndexInOption {
                        option_id: option.option_id.clone(),
                        scenario_index,
                        index,
                    });
                }
            }
        }
    }

    let total_weight = scenarios.iter().map(|scenario| scenario.weight).sum::<f64>();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(BudgetedDesignError::NumericalOverflow);
    }

    Ok(dimension)
}

fn selected_observation_indices(
    scenario: &CorrelatedDesignScenario,
    options: &[CorrelatedExperimentOption],
    selected: &[usize],
    scenario_index: usize,
) -> Result<Vec<usize>, BudgetedDesignError> {
    let mut indices = Vec::new();
    for option_index in selected {
        indices.extend(
            options[*option_index].observation_indices_by_scenario[scenario_index]
                .iter()
                .copied(),
        );
    }
    indices.sort_unstable();
    for pair in indices.windows(2) {
        if pair[0] == pair[1] {
            return Err(BudgetedDesignError::DuplicateSelectedMeasurement {
                scenario_id: scenario.scenario_id.clone(),
                index: pair[0],
            });
        }
    }
    Ok(indices)
}

fn overlaps_selected_measurements(
    scenarios: &[CorrelatedDesignScenario],
    options: &[CorrelatedExperimentOption],
    selected: &[usize],
    candidate_index: usize,
) -> bool {
    for scenario_index in 0..scenarios.len() {
        let mut existing = BTreeSet::new();
        for selected_index in selected {
            existing.extend(
                options[*selected_index].observation_indices_by_scenario[scenario_index]
                    .iter()
                    .copied(),
            );
        }
        if options[candidate_index].observation_indices_by_scenario[scenario_index]
            .iter()
            .any(|index| existing.contains(index))
        {
            return true;
        }
    }
    false
}

fn evaluate_correlated_selection(
    scenarios: &[CorrelatedDesignScenario],
    options: &[CorrelatedExperimentOption],
    selected: &[usize],
    config: &BudgetedDesignConfig,
    dimension: usize,
) -> Result<Vec<ScenarioDesignScore>, BudgetedDesignError> {
    let mut scores = Vec::with_capacity(scenarios.len());
    for (scenario_index, scenario) in scenarios.iter().enumerate() {
        let selected_indices =
            selected_observation_indices(scenario, options, selected, scenario_index)?;
        let mut information = scenario.prior_precision.clone();
        let mut whitened_rows = Vec::new();

        if !selected_indices.is_empty() {
            let count = selected_indices.len();
            let mut selected_covariance = vec![vec![0.0; count]; count];
            let mut selected_jacobian = Vec::with_capacity(count);
            for (local_row, source_row) in selected_indices.iter().copied().enumerate() {
                selected_jacobian.push(
                    scenario.measurement_jacobian_rows[source_row].clone(),
                );
                for (local_column, source_column) in
                    selected_indices.iter().copied().enumerate()
                {
                    selected_covariance[local_row][local_column] =
                        scenario.measurement_covariance[source_row][source_column];
                }
            }
            let lower = cholesky_factor(&selected_covariance).map_err(|_| {
                BudgetedDesignError::MeasurementCovarianceNotPositiveDefinite {
                    scenario_id: scenario.scenario_id.clone(),
                }
            })?;
            whitened_rows = forward_whiten_jacobian(&lower, &selected_jacobian)?;

            for row in &whitened_rows {
                for left in 0..dimension {
                    for right in left..dimension {
                        information[left][right] += row[left] * row[right];
                        if !information[left][right].is_finite() {
                            return Err(BudgetedDesignError::NumericalOverflow);
                        }
                        information[right][left] = information[left][right];
                    }
                }
            }
        }

        let posterior_joint_logdet = cholesky_log_determinant(&information)
            .map_err(|_| BudgetedDesignError::CholeskyFailure)?;
        let prior_joint_logdet = cholesky_log_determinant(&scenario.prior_precision).map_err(|_| {
            BudgetedDesignError::PriorNotPositiveDefinite {
                scenario_id: scenario.scenario_id.clone(),
            }
        })?;
        let posterior_target_logdet =
            marginal_target_log_determinant(&information, scenario.target_parameter_count)
                .map_err(|_| BudgetedDesignError::CholeskyFailure)?;
        let prior_target_logdet = marginal_target_log_determinant(
            &scenario.prior_precision,
            scenario.target_parameter_count,
        )
        .map_err(|_| BudgetedDesignError::PriorNotPositiveDefinite {
            scenario_id: scenario.scenario_id.clone(),
        })?;

        let (
            data_only_rank,
            data_only_nullity,
            data_only_singular_values,
            data_only_condition_number,
        ) = if whitened_rows.is_empty() {
            (0, dimension, vec![0.0; dimension], None)
        } else {
            let report = analyze_jacobian(&whitened_rows, &config.parameter_scales, 1.0e-10)?;
            (
                report.numerical_rank,
                report.nullity,
                report.singular_values,
                report.condition_number,
            )
        };
        scores.push(ScenarioDesignScore {
            scenario_id: scenario.scenario_id.clone(),
            information_gain_nats: posterior_target_logdet - prior_target_logdet,
            joint_information_gain_nats: posterior_joint_logdet - prior_joint_logdet,
            data_only_rank,
            data_only_nullity,
            data_only_singular_values,
            data_only_condition_number,
        });
    }
    Ok(scores)
}

fn forward_whiten_jacobian(
    lower: &[Vec<f64>],
    rows: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, BudgetedDesignError> {
    let measurement_count = rows.len();
    let parameter_count = rows.first().map_or(0, Vec::len);
    if lower.len() != measurement_count
        || lower.iter().any(|row| row.len() != measurement_count)
        || rows.iter().any(|row| row.len() != parameter_count)
    {
        return Err(BudgetedDesignError::CholeskyFailure);
    }
    let mut whitened = vec![vec![0.0; parameter_count]; measurement_count];
    for column in 0..parameter_count {
        for row in 0..measurement_count {
            let diagonal = lower[row][row];
            if !diagonal.is_finite() || diagonal <= 0.0 {
                return Err(BudgetedDesignError::CholeskyFailure);
            }
            let previous = (0..row)
                .map(|index| lower[row][index] * whitened[index][column])
                .sum::<f64>();
            whitened[row][column] = (rows[row][column] - previous) / diagonal;
            if !whitened[row][column].is_finite() {
                return Err(BudgetedDesignError::NumericalOverflow);
            }
        }
    }
    Ok(whitened)
}

fn normalized_weights(scenarios: &[CorrelatedDesignScenario]) -> Vec<f64> {
    let total = scenarios.iter().map(|scenario| scenario.weight).sum::<f64>();
    scenarios
        .iter()
        .map(|scenario| scenario.weight / total)
        .collect()
}

fn weighted_target_gain(scores: &[ScenarioDesignScore], weights: &[f64]) -> f64 {
    scores
        .iter()
        .zip(weights)
        .map(|(score, weight)| score.information_gain_nats * weight)
        .sum()
}

fn weighted_joint_gain(scores: &[ScenarioDesignScore], weights: &[f64]) -> f64 {
    scores
        .iter()
        .zip(weights)
        .map(|(score, weight)| score.joint_information_gain_nats * weight)
        .sum()
}

fn correlated_report(
    options: &[CorrelatedExperimentOption],
    config: &BudgetedDesignConfig,
    selected: &[usize],
    total_cost: f64,
    steps: Vec<DesignSelectionStep>,
    scenario_scores: Vec<ScenarioDesignScore>,
    weights: &[f64],
) -> Result<BudgetedDesignReport, BudgetedDesignError> {
    let expected_information_gain_nats = weighted_target_gain(&scenario_scores, weights);
    let worst_case_information_gain_nats = scenario_scores
        .iter()
        .map(|score| score.information_gain_nats)
        .fold(f64::INFINITY, f64::min);
    let expected_joint_information_gain_nats = weighted_joint_gain(&scenario_scores, weights);
    let worst_case_joint_information_gain_nats = scenario_scores
        .iter()
        .map(|score| score.joint_information_gain_nats)
        .fold(f64::INFINITY, f64::min);
    Ok(BudgetedDesignReport {
        selected_option_ids: selected
            .iter()
            .map(|index| options[*index].option_id.clone())
            .collect(),
        total_cost,
        budget: config.budget,
        remaining_budget: (config.budget - total_cost).max(0.0),
        expected_information_gain_nats,
        worst_case_information_gain_nats,
        expected_joint_information_gain_nats,
        worst_case_joint_information_gain_nats,
        scenario_scores,
        steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budgeted_design::{
        DesignScenario, ExperimentOption, exhaustive_budgeted_design, greedy_budgeted_design,
    };

    fn scenario(covariance: Vec<Vec<f64>>, rows: Vec<Vec<f64>>) -> CorrelatedDesignScenario {
        CorrelatedDesignScenario {
            scenario_id: "scenario".to_string(),
            target_parameter_count: 1,
            weight: 1.0,
            prior_precision: vec![vec![1.0]],
            measurement_jacobian_rows: rows,
            measurement_covariance: covariance,
        }
    }

    fn option(id: &str, cost: f64, indices: Vec<usize>) -> CorrelatedExperimentOption {
        CorrelatedExperimentOption {
            option_id: id.to_string(),
            cost,
            exclusive_group: None,
            observation_indices_by_scenario: vec![indices],
        }
    }

    fn config(budget: f64, parameter_count: usize) -> BudgetedDesignConfig {
        BudgetedDesignConfig {
            budget,
            parameter_scales: vec![1.0; parameter_count],
        }
    }

    #[test]
    fn diagonal_covariance_matches_independent_whitened_design() {
        let correlated_scenarios = vec![scenario(
            vec![vec![1.0, 0.0], vec![0.0, 4.0]],
            vec![vec![1.0], vec![1.0]],
        )];
        let correlated_options = vec![option("sensor-a", 1.0, vec![0]), option("sensor-b", 1.0, vec![1])];
        let correlated = greedy_correlated_budgeted_design(
            &correlated_scenarios,
            &correlated_options,
            &config(2.0, 1),
        )
        .unwrap();

        let independent_scenarios = vec![DesignScenario {
            scenario_id: "scenario".to_string(),
            target_parameter_count: 1,
            weight: 1.0,
            prior_precision: vec![vec![1.0]],
        }];
        let independent_options = vec![
            ExperimentOption {
                option_id: "sensor-a".to_string(),
                cost: 1.0,
                exclusive_group: None,
                jacobian_rows_by_scenario: vec![vec![vec![1.0]]],
            },
            ExperimentOption {
                option_id: "sensor-b".to_string(),
                cost: 1.0,
                exclusive_group: None,
                jacobian_rows_by_scenario: vec![vec![vec![0.5]]],
            },
        ];
        let independent = greedy_budgeted_design(
            &independent_scenarios,
            &independent_options,
            &config(2.0, 1),
        )
        .unwrap();

        assert_eq!(correlated.selected_option_ids, independent.selected_option_ids);
        assert!(
            (correlated.expected_information_gain_nats
                - independent.expected_information_gain_nats)
                .abs()
                < 1.0e-12
        );
        assert!(
            (correlated.expected_joint_information_gain_nats
                - independent.expected_joint_information_gain_nats)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn common_mode_residual_correlation_limits_repeated_sensor_information() {
        let rho = 0.9;
        let scenarios = vec![scenario(
            vec![vec![1.0, rho], vec![rho, 1.0]],
            vec![vec![1.0], vec![1.0]],
        )];
        let options = vec![option("two-correlated-sensors", 1.0, vec![0, 1])];
        let report = greedy_correlated_budgeted_design(
            &scenarios,
            &options,
            &config(1.0, 1),
        )
        .unwrap();
        let expected_information = (1.0 + 2.0 / (1.0 + rho)).ln();
        let independent_assumption_information = 3.0_f64.ln();

        assert!(
            (report.expected_information_gain_nats - expected_information).abs() < 1.0e-12
        );
        assert!(report.expected_information_gain_nats < independent_assumption_information);
    }

    #[test]
    fn selected_set_uses_the_covariance_principal_submatrix() {
        let scenarios = vec![scenario(
            vec![vec![1.0, 0.8], vec![0.8, 1.0]],
            vec![vec![1.0], vec![1.0]],
        )];
        let options = vec![option("sensor-a", 1.0, vec![0]), option("sensor-b", 1.0, vec![1])];
        let report = greedy_correlated_budgeted_design(
            &scenarios,
            &options,
            &config(1.0, 1),
        )
        .unwrap();

        // Either one-observation option has information log(1 + 1/1).
        assert!((report.expected_information_gain_nats - 2.0_f64.ln()).abs() < 1.0e-12);
        assert_eq!(report.selected_option_ids.len(), 1);
    }

    #[test]
    fn row_and_covariance_permutation_preserves_selected_design_score() {
        let first_scenario = CorrelatedDesignScenario {
            scenario_id: "scenario".to_string(),
            target_parameter_count: 2,
            weight: 1.0,
            prior_precision: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            measurement_jacobian_rows: vec![vec![1.0, 0.2], vec![0.1, 1.0]],
            measurement_covariance: vec![vec![1.0, 0.4], vec![0.4, 1.0]],
        };
        let second_scenario = CorrelatedDesignScenario {
            scenario_id: "scenario".to_string(),
            target_parameter_count: 2,
            weight: 1.0,
            prior_precision: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            measurement_jacobian_rows: vec![vec![0.1, 1.0], vec![1.0, 0.2]],
            measurement_covariance: vec![vec![1.0, 0.4], vec![0.4, 1.0]],
        };
        let first_options = vec![option("both", 1.0, vec![0, 1])];
        let second_options = vec![option("both", 1.0, vec![1, 0])];
        let first = greedy_correlated_budgeted_design(
            &[first_scenario],
            &first_options,
            &config(1.0, 2),
        )
        .unwrap();
        let second = greedy_correlated_budgeted_design(
            &[second_scenario],
            &second_options,
            &config(1.0, 2),
        )
        .unwrap();

        assert!(
            (first.expected_information_gain_nats - second.expected_information_gain_nats).abs()
                < 1.0e-12
        );
        assert!(
            (first.expected_joint_information_gain_nats
                - second.expected_joint_information_gain_nats)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn invalid_covariance_and_duplicate_selected_observations_fail_closed() {
        let singular = vec![scenario(
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            vec![vec![1.0], vec![1.0]],
        )];
        assert!(matches!(
            greedy_correlated_budgeted_design(
                &singular,
                &[option("both", 1.0, vec![0, 1])],
                &config(1.0, 1)
            ),
            Err(BudgetedDesignError::MeasurementCovarianceNotPositiveDefinite { .. })
        ));

        let asymmetric = vec![scenario(
            vec![vec![1.0, 0.1], vec![0.2, 1.0]],
            vec![vec![1.0], vec![1.0]],
        )];
        assert!(matches!(
            greedy_correlated_budgeted_design(
                &asymmetric,
                &[option("both", 1.0, vec![0, 1])],
                &config(1.0, 1)
            ),
            Err(BudgetedDesignError::InvalidMeasurementCovarianceSymmetry { .. })
        ));

        let non_finite = vec![scenario(
            vec![vec![1.0, f64::NAN], vec![f64::NAN, 1.0]],
            vec![vec![1.0], vec![1.0]],
        )];
        assert!(matches!(
            greedy_correlated_budgeted_design(
                &non_finite,
                &[option("both", 1.0, vec![0, 1])],
                &config(1.0, 1)
            ),
            Err(BudgetedDesignError::NumericalOverflow)
        ));

        let valid = vec![scenario(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![1.0], vec![1.0]],
        )];
        let duplicate = vec![
            option("sensor-a", 1.0, vec![0]),
            option("sensor-a-repeated", 1.0, vec![0]),
        ];
        assert!(matches!(
            evaluate_correlated_selection(&valid, &duplicate, &[0, 1], &config(2.0, 1), 1),
            Err(BudgetedDesignError::DuplicateSelectedMeasurement { .. })
        ));
        assert!(matches!(
            validate_correlated_inputs(
                &valid,
                &[option("out-of-range", 1.0, vec![2])],
                &config(1.0, 1)
            ),
            Err(BudgetedDesignError::MeasurementIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn greedy_and_exhaustive_correlated_design_agree_on_one_action_case() {
        let scenarios = vec![scenario(
            vec![vec![1.0, 0.6], vec![0.6, 1.0]],
            vec![vec![1.0], vec![0.2]],
        )];
        let options = vec![
            option("first", 1.0, vec![0]),
            option("second", 1.0, vec![1]),
        ];
        let greedy =
            greedy_correlated_budgeted_design(&scenarios, &options, &config(2.0, 1)).unwrap();
        let exact =
            exhaustive_correlated_budgeted_design(&scenarios, &options, &config(2.0, 1)).unwrap();

        assert_eq!(greedy.selected_option_ids, exact.selected_option_ids);
        assert!(
            (greedy.expected_information_gain_nats - exact.expected_information_gain_nats).abs()
                < 1.0e-12
        );
    }
}
