// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Budgeted multi-scenario experimental design for small linearized systems.
//!
//! Each candidate contributes whitened Jacobian rows for each declared
//! scenario. The greedy selector maximizes target-state marginal D-optimal
//! information gain per unit cost. Nuisance parameters are eliminated through
//! a Schur complement; full joint-state gain is reported separately.
//! An exhaustive subset oracle is provided for small candidate sets.
//!
//! Important limits:
//! - This is a local linearized design kernel, not a nonlinear sensor planner.
//! - Candidate contributions are assumed conditionally independent after
//!   whitening. Shared calibration/cross-candidate covariance must be modelled
//!   jointly before constructing these rows; it must not be ignored.
//! - The greedy expected-logdet objective is useful for screening, but no
//!   guarantee is claimed for worst-case, nonlinear or correlated objectives.
//! - Data-only observability is reported separately from prior-regularized
//!   target-marginal and full joint-state Bayesian D-optimal utilities.

use crate::observability::{MAX_OBSERVABILITY_PARAMETERS, ObservabilityError, analyze_jacobian};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// One target/calibration hypothesis used during experimental design.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignScenario {
    pub scenario_id: String,
    /// Number of leading parameters whose marginal uncertainty is the design
    /// objective. Remaining parameters are nuisance variables.
    pub target_parameter_count: usize,
    /// Non-normalized positive scenario weight.
    pub weight: f64,
    /// Positive-definite prior precision for the full target+nuisance state.
    pub prior_precision: Vec<Vec<f64>>,
}

/// Scenario for selected-set correlated residual covariance design.
///
/// The Jacobian rows and covariance use one immutable observation ordering.
/// Candidate options select indices into that common pool. The covariance is
/// sliced and jointly whitened only after a candidate set has been selected.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrelatedDesignScenario {
    pub scenario_id: String,
    pub target_parameter_count: usize,
    pub weight: f64,
    pub prior_precision: Vec<Vec<f64>>,
    /// One raw (not pre-whitened) residual-Jacobian row per candidate observation.
    pub measurement_jacobian_rows: Vec<Vec<f64>>,
    /// Full covariance of the candidate observation pool in the same ordering.
    pub measurement_covariance: Vec<Vec<f64>>,
}

/// Candidate action selecting one or more rows from a correlated observation pool.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrelatedExperimentOption {
    pub option_id: String,
    pub cost: f64,
    pub exclusive_group: Option<String>,
    /// Observation indices selected by this action in each scenario.
    /// Empty indices mean that the action is unavailable in that scenario.
    pub observation_indices_by_scenario: Vec<Vec<usize>>,
}

/// Maximum observation-pool size accepted by the small dense covariance reference.
pub const MAX_CORRELATED_DESIGN_OBSERVATIONS: usize = 512;

/// One optional measurement action or sensor-fidelity choice.
///
/// `jacobian_rows_by_scenario[s]` contains the candidate's whitened
/// measurement-Jacobian rows under scenario s. An empty row vector means the
/// action is unavailable in that scenario. Two options sharing an
/// `exclusive_group` cannot both be selected (for example, two fidelity
/// choices for the same physical station).
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentOption {
    pub option_id: String,
    pub cost: f64,
    pub exclusive_group: Option<String>,
    pub jacobian_rows_by_scenario: Vec<Vec<Vec<f64>>>,
}

/// Constraints for one scalar-resource budget.
#[derive(Clone, Debug, PartialEq)]
pub struct BudgetedDesignConfig {
    pub budget: f64,
    /// Characteristic physical parameter displacements used to scale columns
    /// for data-only observability diagnostics.
    pub parameter_scales: Vec<f64>,
}

/// Data-only observability and prior-regularized utility for one scenario.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioDesignScore {
    pub scenario_id: String,
    /// Marginal target-state information gain after eliminating nuisance
    /// parameters with the Schur complement.
    pub information_gain_nats: f64,
    /// Full joint-state information gain, including nuisance parameters.
    pub joint_information_gain_nats: f64,
    pub data_only_rank: usize,
    pub data_only_nullity: usize,
    pub data_only_singular_values: Vec<f64>,
    pub data_only_condition_number: Option<f64>,
}

/// One deterministic greedy selection step.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignSelectionStep {
    pub option_id: String,
    pub cost: f64,
    pub marginal_expected_gain_nats: f64,
    pub marginal_gain_per_cost: f64,
}

/// Final selected design and scenario-level diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct BudgetedDesignReport {
    /// Greedy selection order; deterministic tie-breaking uses option ID.
    pub selected_option_ids: Vec<String>,
    pub total_cost: f64,
    pub budget: f64,
    pub remaining_budget: f64,
    /// Scenario-weighted target-state marginal information gain.
    pub expected_information_gain_nats: f64,
    /// Minimum target-state marginal information gain across scenarios.
    pub worst_case_information_gain_nats: f64,
    /// Scenario-weighted full joint-state information gain.
    pub expected_joint_information_gain_nats: f64,
    /// Minimum full joint-state information gain across scenarios.
    pub worst_case_joint_information_gain_nats: f64,
    pub scenario_scores: Vec<ScenarioDesignScore>,
    pub steps: Vec<DesignSelectionStep>,
}

/// Invalid input or numerical failure in budgeted experiment design.
#[derive(Clone, Debug, PartialEq)]
pub enum BudgetedDesignError {
    NoScenarios,
    NoOptions,
    InvalidBudget,
    InvalidScenarioId {
        index: usize,
    },
    DuplicateScenarioId(String),
    InvalidScenarioWeight {
        scenario_id: String,
    },
    InvalidTargetParameterCount {
        scenario_id: String,
        count: usize,
        maximum: usize,
    },
    InvalidPriorShape {
        scenario_id: String,
    },
    InvalidPriorSymmetry {
        scenario_id: String,
    },
    PriorNotPositiveDefinite {
        scenario_id: String,
    },
    InvalidOptionId {
        index: usize,
    },
    DuplicateOptionId(String),
    InvalidOptionCost {
        option_id: String,
    },
    EmptyExclusiveGroup {
        option_id: String,
    },
    ScenarioCountMismatch {
        option_id: String,
        expected: usize,
        actual: usize,
    },
    RaggedJacobian {
        option_id: String,
        scenario_index: usize,
        row_index: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteJacobian {
        option_id: String,
        scenario_index: usize,
        row_index: usize,
        column: usize,
    },
    ParameterScaleCount {
        expected: usize,
        actual: usize,
    },
    TooManyParameters {
        count: usize,
        maximum: usize,
    },
    InvalidParameterScale {
        column: usize,
    },
    TooManyOptionsForExhaustive {
        count: usize,
        maximum: usize,
    },
    TooManyMeasurements {
        scenario_id: String,
        count: usize,
        maximum: usize,
    },
    MeasurementJacobianShape {
        scenario_id: String,
        row_index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidMeasurementCovarianceShape {
        scenario_id: String,
    },
    InvalidMeasurementCovarianceSymmetry {
        scenario_id: String,
    },
    MeasurementCovarianceNotPositiveDefinite {
        scenario_id: String,
    },
    MeasurementIndexScenarioCountMismatch {
        option_id: String,
        expected: usize,
        actual: usize,
    },
    MeasurementIndexOutOfRange {
        option_id: String,
        scenario_index: usize,
        index: usize,
        measurement_count: usize,
    },
    DuplicateMeasurementIndexInOption {
        option_id: String,
        scenario_index: usize,
        index: usize,
    },
    DuplicateSelectedMeasurement {
        scenario_id: String,
        index: usize,
    },
    NumericalOverflow,
    CholeskyFailure,
    Observability(ObservabilityError),
}

impl fmt::Display for BudgetedDesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoScenarios => write!(f, "at least one design scenario is required"),
            Self::NoOptions => write!(f, "at least one experiment option is required"),
            Self::InvalidBudget => write!(f, "budget must be finite and non-negative"),
            Self::InvalidScenarioId { index } => {
                write!(f, "scenario ID at index {index} must not be empty")
            }
            Self::DuplicateScenarioId(id) => write!(f, "duplicate scenario ID {id}"),
            Self::InvalidScenarioWeight { scenario_id } => {
                write!(
                    f,
                    "scenario {scenario_id} must have a finite positive weight"
                )
            }
            Self::InvalidTargetParameterCount {
                scenario_id,
                count,
                maximum,
            } => write!(
                f,
                "scenario {scenario_id} has {count} target parameters; valid range is 1..={maximum}"
            ),
            Self::InvalidPriorShape { scenario_id } => {
                write!(
                    f,
                    "prior precision for scenario {scenario_id} is not square"
                )
            }
            Self::InvalidPriorSymmetry { scenario_id } => {
                write!(
                    f,
                    "prior precision for scenario {scenario_id} is not symmetric"
                )
            }
            Self::PriorNotPositiveDefinite { scenario_id } => {
                write!(
                    f,
                    "prior precision for scenario {scenario_id} is not positive definite"
                )
            }
            Self::InvalidOptionId { index } => {
                write!(f, "option ID at index {index} must not be empty")
            }
            Self::DuplicateOptionId(id) => write!(f, "duplicate option ID {id}"),
            Self::InvalidOptionCost { option_id } => {
                write!(f, "option {option_id} must have a finite positive cost")
            }
            Self::EmptyExclusiveGroup { option_id } => {
                write!(f, "option {option_id} has an empty exclusive group")
            }
            Self::ScenarioCountMismatch {
                option_id,
                expected,
                actual,
            } => write!(
                f,
                "option {option_id} has {actual} scenario Jacobians; expected {expected}"
            ),
            Self::RaggedJacobian {
                option_id,
                scenario_index,
                row_index,
                expected,
                actual,
            } => write!(
                f,
                "option {option_id}, scenario {scenario_index}, Jacobian row {row_index} has                  {actual} columns; expected {expected}"
            ),
            Self::NonFiniteJacobian {
                option_id,
                scenario_index,
                row_index,
                column,
            } => write!(
                f,
                "option {option_id}, scenario {scenario_index}, Jacobian entry ({row_index},                  {column}) is not finite"
            ),
            Self::ParameterScaleCount { expected, actual } => {
                write!(f, "received {actual} parameter scales; expected {expected}")
            }
            Self::InvalidParameterScale { column } => {
                write!(
                    f,
                    "parameter scale at column {column} is not finite and positive"
                )
            }
            Self::TooManyParameters { count, maximum } => {
                write!(
                    f,
                    "design has {count} parameters; maximum supported is {maximum}"
                )
            }
            Self::TooManyOptionsForExhaustive { count, maximum } => write!(
                f,
                "exhaustive design has {count} options; maximum supported is {maximum}"
            ),
            Self::TooManyMeasurements {
                scenario_id,
                count,
                maximum,
            } => write!(
                f,
                "scenario {scenario_id} has {count} candidate measurements; maximum supported is {maximum}"
            ),
            Self::MeasurementJacobianShape {
                scenario_id,
                row_index,
                expected,
                actual,
            } => write!(
                f,
                "scenario {scenario_id} measurement Jacobian row {row_index} has {actual} columns; expected {expected}"
            ),
            Self::InvalidMeasurementCovarianceShape { scenario_id } => write!(
                f,
                "measurement covariance for scenario {scenario_id} is not square or does not match the observation pool"
            ),
            Self::InvalidMeasurementCovarianceSymmetry { scenario_id } => write!(
                f,
                "measurement covariance for scenario {scenario_id} is not symmetric"
            ),
            Self::MeasurementCovarianceNotPositiveDefinite { scenario_id } => write!(
                f,
                "measurement covariance for scenario {scenario_id} is not positive definite"
            ),
            Self::MeasurementIndexScenarioCountMismatch {
                option_id,
                expected,
                actual,
            } => write!(
                f,
                "option {option_id} has {actual} scenario observation-index sets; expected {expected}"
            ),
            Self::MeasurementIndexOutOfRange {
                option_id,
                scenario_index,
                index,
                measurement_count,
            } => write!(
                f,
                "option {option_id}, scenario {scenario_index} selects measurement {index}; pool size is {measurement_count}"
            ),
            Self::DuplicateMeasurementIndexInOption {
                option_id,
                scenario_index,
                index,
            } => write!(
                f,
                "option {option_id}, scenario {scenario_index} selects measurement {index} more than once"
            ),
            Self::DuplicateSelectedMeasurement {
                scenario_id,
                index,
            } => write!(
                f,
                "selected actions duplicate measurement {index} in scenario {scenario_id}"
            ),
            Self::NumericalOverflow => write!(f, "information accumulation overflowed"),
            Self::CholeskyFailure => write!(f, "posterior precision Cholesky factorization failed"),
            Self::Observability(error) => write!(f, "data-only observability failed: {error}"),
        }
    }
}

impl Error for BudgetedDesignError {}

impl From<ObservabilityError> for BudgetedDesignError {
    fn from(value: ObservabilityError) -> Self {
        Self::Observability(value)
    }
}

/// Greedily select experiment options by marginal expected log-det gain per
/// unit cost.
///
/// Scenario weights are normalized internally. Posterior information is
/// `P0 + sum(J_option^T J_option)`; the prior contributes to the utility but
/// is never included in the data-only rank report. If the candidate actions
/// include shared/correlated measurement errors, their joint whitening must be
/// performed before using this additive contribution interface.
pub fn greedy_budgeted_design(
    scenarios: &[DesignScenario],
    options: &[ExperimentOption],
    config: &BudgetedDesignConfig,
) -> Result<BudgetedDesignReport, BudgetedDesignError> {
    let dimension = validate_inputs(scenarios, options, config)?;
    let weights = normalized_scenario_weights(scenarios);

    let mut selected = Vec::new();
    let mut selected_set = BTreeSet::new();
    let mut selected_groups = BTreeSet::new();
    let mut total_cost = 0.0;
    let mut steps = Vec::new();
    let mut current_scores = evaluate_selection(
        scenarios,
        options,
        &selected,
        &config.parameter_scales,
        dimension,
    )?;
    let mut current_expected = weighted_gain(&current_scores, &weights);

    loop {
        let mut best: Option<(usize, f64, f64)> = None;

        for (index, option) in options.iter().enumerate() {
            if selected_set.contains(&index)
                || total_cost + option.cost > config.budget + 1.0e-12
                || option
                    .exclusive_group
                    .as_ref()
                    .is_some_and(|group| selected_groups.contains(group))
            {
                continue;
            }

            let mut trial = selected.clone();
            trial.push(index);
            let trial_scores = evaluate_selection(
                scenarios,
                options,
                &trial,
                &config.parameter_scales,
                dimension,
            )?;
            let marginal_gain = weighted_gain(&trial_scores, &weights) - current_expected;
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
                        || (gain_per_cost - best_density).abs() <= 1.0e-12
                            && option.option_id == options[best_index].option_id
                            && marginal_gain > best_gain
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

        current_scores = evaluate_selection(
            scenarios,
            options,
            &selected,
            &config.parameter_scales,
            dimension,
        )?;
        current_expected = weighted_gain(&current_scores, &weights);
    }

    report_from_selection(
        options,
        config,
        &selected,
        total_cost,
        steps,
        current_scores,
        &weights,
    )
}

/// Exact subset oracle for small candidate sets (at most 20 options).
///
/// The objective is the normalized scenario-weighted expected information
/// gain. Ties are resolved lexicographically by sorted option-ID lists. This
/// is intended as a benchmark oracle for measuring greedy optimality gaps,
/// not as the scalable production selector.
pub fn exhaustive_budgeted_design(
    scenarios: &[DesignScenario],
    options: &[ExperimentOption],
    config: &BudgetedDesignConfig,
) -> Result<BudgetedDesignReport, BudgetedDesignError> {
    let dimension = validate_inputs(scenarios, options, config)?;
    const MAX_EXHAUSTIVE_OPTIONS: usize = 20;
    if options.len() > MAX_EXHAUSTIVE_OPTIONS {
        return Err(BudgetedDesignError::TooManyOptionsForExhaustive {
            count: options.len(),
            maximum: MAX_EXHAUSTIVE_OPTIONS,
        });
    }
    let weights = normalized_scenario_weights(scenarios);
    let subset_count = 1_u64 << options.len();

    let mut best_selection: Vec<usize> = Vec::new();
    let mut best_cost = 0.0;
    let mut best_scores = evaluate_selection(
        scenarios,
        options,
        &best_selection,
        &config.parameter_scales,
        dimension,
    )?;
    let mut best_gain = weighted_gain(&best_scores, &weights);
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
            if cost > config.budget + 1.0e-12 {
                feasible = false;
                break;
            }
            if let Some(group) = &option.exclusive_group {
                if !groups.insert(group.clone()) {
                    feasible = false;
                    break;
                }
            }
            selected.push(index);
        }
        if !feasible {
            continue;
        }

        let scores = evaluate_selection(
            scenarios,
            options,
            &selected,
            &config.parameter_scales,
            dimension,
        )?;
        let gain = weighted_gain(&scores, &weights);
        let mut ids: Vec<String> = selected
            .iter()
            .map(|index| options[*index].option_id.clone())
            .collect();
        ids.sort();

        if gain > best_gain + 1.0e-12 || ((gain - best_gain).abs() <= 1.0e-12 && ids < best_ids) {
            best_selection = selected;
            best_cost = cost;
            best_scores = scores;
            best_gain = gain;
            best_ids = ids;
        }
    }

    // The exhaustive subset is reported in sorted option-ID order, not in the
    // subset mask's input order.
    best_selection.sort_by(|left, right| options[*left].option_id.cmp(&options[*right].option_id));
    report_from_selection(
        options,
        config,
        &best_selection,
        best_cost,
        Vec::new(),
        best_scores,
        &weights,
    )
}

fn validate_inputs(
    scenarios: &[DesignScenario],
    options: &[ExperimentOption],
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
    if dimension > MAX_OBSERVABILITY_PARAMETERS {
        return Err(BudgetedDesignError::TooManyParameters {
            count: dimension,
            maximum: MAX_OBSERVABILITY_PARAMETERS,
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
    for (index, scenario) in scenarios.iter().enumerate() {
        if scenario.scenario_id.trim().is_empty() {
            return Err(BudgetedDesignError::InvalidScenarioId { index });
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
        for row in &scenario.prior_precision {
            if row.iter().any(|value| !value.is_finite()) {
                return Err(BudgetedDesignError::NumericalOverflow);
            }
        }
        let scale = scenario
            .prior_precision
            .iter()
            .flatten()
            .fold(1.0_f64, |maximum, value| maximum.max(value.abs()));
        for row in 0..dimension {
            for column in 0..dimension {
                if (scenario.prior_precision[row][column] - scenario.prior_precision[column][row])
                    .abs()
                    > 1.0e-12 * scale
                {
                    return Err(BudgetedDesignError::InvalidPriorSymmetry {
                        scenario_id: scenario.scenario_id.clone(),
                    });
                }
            }
        }
        if cholesky_log_determinant(&scenario.prior_precision).is_err() {
            return Err(BudgetedDesignError::PriorNotPositiveDefinite {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
    }

    let mut option_ids = BTreeSet::new();
    for (index, option) in options.iter().enumerate() {
        if option.option_id.trim().is_empty() {
            return Err(BudgetedDesignError::InvalidOptionId { index });
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
        if option.jacobian_rows_by_scenario.len() != scenarios.len() {
            return Err(BudgetedDesignError::ScenarioCountMismatch {
                option_id: option.option_id.clone(),
                expected: scenarios.len(),
                actual: option.jacobian_rows_by_scenario.len(),
            });
        }
        for (scenario_index, rows) in option.jacobian_rows_by_scenario.iter().enumerate() {
            for (row_index, row) in rows.iter().enumerate() {
                if row.len() != dimension {
                    return Err(BudgetedDesignError::RaggedJacobian {
                        option_id: option.option_id.clone(),
                        scenario_index,
                        row_index,
                        expected: dimension,
                        actual: row.len(),
                    });
                }
                for (column, value) in row.iter().copied().enumerate() {
                    if !value.is_finite() {
                        return Err(BudgetedDesignError::NonFiniteJacobian {
                            option_id: option.option_id.clone(),
                            scenario_index,
                            row_index,
                            column,
                        });
                    }
                }
            }
        }
    }

    Ok(dimension)
}

fn normalized_scenario_weights(scenarios: &[DesignScenario]) -> Vec<f64> {
    // Scale before summing so a set of individually finite weights cannot
    // overflow merely because their unnormalized total is large.
    let maximum = scenarios
        .iter()
        .map(|scenario| scenario.weight)
        .fold(0.0_f64, f64::max);
    let scaled_total: f64 = scenarios
        .iter()
        .map(|scenario| scenario.weight / maximum)
        .sum();
    scenarios
        .iter()
        .map(|scenario| (scenario.weight / maximum) / scaled_total)
        .collect()
}

fn evaluate_selection(
    scenarios: &[DesignScenario],
    options: &[ExperimentOption],
    selected: &[usize],
    parameter_scales: &[f64],
    dimension: usize,
) -> Result<Vec<ScenarioDesignScore>, BudgetedDesignError> {
    let mut scores = Vec::with_capacity(scenarios.len());

    for (scenario_index, scenario) in scenarios.iter().enumerate() {
        let mut information = scenario.prior_precision.clone();
        let mut data_rows = Vec::new();

        for option_index in selected {
            let rows = &options[*option_index].jacobian_rows_by_scenario[scenario_index];
            for row in rows {
                data_rows.push(row.clone());
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
        let prior_joint_logdet =
            cholesky_log_determinant(&scenario.prior_precision).map_err(|_| {
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
        ) = if data_rows.is_empty() {
            (0, dimension, vec![0.0; dimension], None)
        } else {
            let report = analyze_jacobian(&data_rows, parameter_scales, 1.0e-10)?;
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

fn weighted_gain(scores: &[ScenarioDesignScore], weights: &[f64]) -> f64 {
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

fn report_from_selection(
    options: &[ExperimentOption],
    config: &BudgetedDesignConfig,
    selected: &[usize],
    total_cost: f64,
    steps: Vec<DesignSelectionStep>,
    scenario_scores: Vec<ScenarioDesignScore>,
    weights: &[f64],
) -> Result<BudgetedDesignReport, BudgetedDesignError> {
    let expected_information_gain_nats = weighted_gain(&scenario_scores, weights);
    let worst_case_information_gain_nats = scenario_scores
        .iter()
        .map(|score| score.information_gain_nats)
        .fold(f64::INFINITY, f64::min);
    let expected_joint_information_gain_nats = weighted_joint_gain(&scenario_scores, weights);
    let worst_case_joint_information_gain_nats = scenario_scores
        .iter()
        .map(|score| score.joint_information_gain_nats)
        .fold(f64::INFINITY, f64::min);
    let selected_option_ids = selected
        .iter()
        .map(|index| options[*index].option_id.clone())
        .collect();

    Ok(BudgetedDesignReport {
        selected_option_ids,
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

/// Log-determinant of the precision marginal for the leading target block.
///
/// The target precision after nuisance elimination is the Schur complement
/// A - B C^-1 B^T of the full precision [[A, B], [B^T, C]]. Cholesky solves
/// are used for C^-1 B^T; no explicit inverse or pseudo-inverse is formed.
fn marginal_target_log_determinant(
    precision: &[Vec<f64>],
    target_parameter_count: usize,
) -> Result<f64, ()> {
    let dimension = precision.len();
    if target_parameter_count == 0
        || target_parameter_count > dimension
        || precision.iter().any(|row| row.len() != dimension)
    {
        return Err(());
    }
    if target_parameter_count == dimension {
        return cholesky_log_determinant(precision);
    }

    let nuisance_count = dimension - target_parameter_count;
    let nuisance_precision: Vec<Vec<f64>> = precision[target_parameter_count..]
        .iter()
        .map(|row| row[target_parameter_count..].to_vec())
        .collect();
    let nuisance_lower = cholesky_factor(&nuisance_precision)?;
    let mut schur: Vec<Vec<f64>> = precision[..target_parameter_count]
        .iter()
        .map(|row| row[..target_parameter_count].to_vec())
        .collect();

    for target_column in 0..target_parameter_count {
        let rhs: Vec<f64> = (0..nuisance_count)
            .map(|index| precision[target_parameter_count + index][target_column])
            .collect();
        let solved = cholesky_solve(&nuisance_lower, &rhs)?;
        for target_row in 0..target_parameter_count {
            let correction = (0..nuisance_count)
                .map(|index| precision[target_row][target_parameter_count + index] * solved[index])
                .sum::<f64>();
            schur[target_row][target_column] -= correction;
        }
    }

    for row in 0..target_parameter_count {
        for column in (row + 1)..target_parameter_count {
            let symmetric = 0.5 * (schur[row][column] + schur[column][row]);
            schur[row][column] = symmetric;
            schur[column][row] = symmetric;
        }
    }
    cholesky_log_determinant(&schur)
}

fn cholesky_solve(lower: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, ()> {
    let dimension = lower.len();
    if rhs.len() != dimension || lower.iter().any(|row| row.len() != dimension) {
        return Err(());
    }

    let mut forward = vec![0.0; dimension];
    for row in 0..dimension {
        let previous = (0..row)
            .map(|column| lower[row][column] * forward[column])
            .sum::<f64>();
        let diagonal = lower[row][row];
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return Err(());
        }
        forward[row] = (rhs[row] - previous) / diagonal;
        if !forward[row].is_finite() {
            return Err(());
        }
    }

    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let following = ((row + 1)..dimension)
            .map(|column| lower[column][row] * solution[column])
            .sum::<f64>();
        let diagonal = lower[row][row];
        solution[row] = (forward[row] - following) / diagonal;
        if !solution[row].is_finite() {
            return Err(());
        }
    }
    Ok(solution)
}

fn cholesky_factor(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ()> {
    let dimension = matrix.len();
    if dimension == 0 || matrix.iter().any(|row| row.len() != dimension) {
        return Err(());
    }
    let mut lower = vec![vec![0.0; dimension]; dimension];
    let scale = matrix
        .iter()
        .flatten()
        .fold(1.0_f64, |maximum, value| maximum.max(value.abs()));
    for row in 0..dimension {
        for column in 0..=row {
            if (matrix[row][column] - matrix[column][row]).abs() > 1.0e-12 * scale {
                return Err(());
            }
            let correction = (0..column)
                .map(|index| lower[row][index] * lower[column][index])
                .sum::<f64>();
            let residual = matrix[row][column] - correction;
            if row == column {
                if !residual.is_finite() || residual <= f64::MIN_POSITIVE {
                    return Err(());
                }
                lower[row][column] = residual.sqrt();
            } else {
                lower[row][column] = residual / lower[column][column];
                if !lower[row][column].is_finite() {
                    return Err(());
                }
            }
        }
    }
    Ok(lower)
}

/// Cholesky log-determinant for a symmetric positive-definite matrix.
fn cholesky_log_determinant(matrix: &[Vec<f64>]) -> Result<f64, ()> {
    let dimension = matrix.len();
    let mut lower = vec![vec![0.0; dimension]; dimension];
    let mut log_determinant = 0.0;

    for row in 0..dimension {
        for column in 0..=row {
            let mut value = matrix[row][column];
            for k in 0..column {
                value -= lower[row][k] * lower[column][k];
            }

            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(());
                }
                lower[row][column] = value.sqrt();
                log_determinant += 2.0 * lower[row][column].ln();
            } else {
                lower[row][column] = value / lower[column][column];
            }
        }
    }

    if log_determinant.is_finite() {
        Ok(log_determinant)
    } else {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(scenario_id: &str, weight: f64) -> DesignScenario {
        DesignScenario {
            scenario_id: scenario_id.to_string(),
            target_parameter_count: 2,
            weight,
            prior_precision: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        }
    }

    fn option(
        option_id: &str,
        cost: f64,
        group: Option<&str>,
        rows_by_scenario: Vec<Vec<Vec<f64>>>,
    ) -> ExperimentOption {
        ExperimentOption {
            option_id: option_id.to_string(),
            cost,
            exclusive_group: group.map(str::to_string),
            jacobian_rows_by_scenario: rows_by_scenario,
        }
    }

    fn config(budget: f64) -> BudgetedDesignConfig {
        BudgetedDesignConfig {
            budget,
            parameter_scales: vec![1.0, 1.0],
        }
    }

    #[test]
    fn greedy_selects_complementary_measurements_and_matches_exhaustive_oracle() {
        let scenarios = vec![scenario("near", 1.0), scenario("far", 2.0)];
        let options = vec![
            option(
                "measure-x",
                1.0,
                None,
                vec![vec![vec![1.0, 0.0]], vec![vec![1.0, 0.0]]],
            ),
            option(
                "measure-y",
                1.0,
                None,
                vec![vec![vec![0.0, 1.0]], vec![vec![0.0, 1.0]]],
            ),
            option(
                "expensive-x",
                2.1,
                None,
                vec![vec![vec![10.0, 0.0]], vec![vec![10.0, 0.0]]],
            ),
        ];

        let greedy = greedy_budgeted_design(&scenarios, &options, &config(2.0)).unwrap();
        let exact = exhaustive_budgeted_design(&scenarios, &options, &config(2.0)).unwrap();

        assert_eq!(greedy.selected_option_ids, vec!["measure-x", "measure-y"]);
        assert_eq!(greedy.selected_option_ids, exact.selected_option_ids);
        assert!((greedy.total_cost - 2.0).abs() < 1.0e-12);
        assert!(greedy.expected_information_gain_nats > 0.0);
        assert!(greedy.worst_case_information_gain_nats > 0.0);
        assert_eq!(greedy.scenario_scores.len(), 2);
        assert!(
            greedy
                .scenario_scores
                .iter()
                .all(|score| score.data_only_rank == 2)
        );
    }

    #[test]
    fn mutual_exclusion_prevents_double_counting_sensor_fidelity_options() {
        let scenarios = vec![scenario("target-a", 0.5), scenario("target-b", 0.5)];
        let options = vec![
            option(
                "station-low-fidelity",
                1.0,
                Some("station-1"),
                vec![vec![vec![1.0, 0.0]], vec![vec![1.0, 0.0]]],
            ),
            option(
                "station-high-fidelity",
                1.5,
                Some("station-1"),
                vec![vec![vec![2.0, 0.0]], vec![vec![2.0, 0.0]]],
            ),
            option(
                "independent-baseline",
                1.0,
                None,
                vec![vec![vec![0.0, 1.0]], vec![vec![0.0, 1.0]]],
            ),
        ];

        let report = greedy_budgeted_design(&scenarios, &options, &config(2.5)).unwrap();
        let station_choices = report
            .selected_option_ids
            .iter()
            .filter(|id| id.starts_with("station-"))
            .count();
        assert_eq!(station_choices, 1);
        assert!(report.total_cost <= report.budget + 1.0e-12);
    }

    #[test]
    fn prior_regularization_does_not_hide_data_only_null_directions() {
        let scenarios = vec![scenario("nominal", 1.0)];
        let options = vec![option("x-only", 1.0, None, vec![vec![vec![1.0, 0.0]]])];

        let report = greedy_budgeted_design(&scenarios, &options, &config(1.0)).unwrap();
        assert_eq!(report.scenario_scores[0].data_only_rank, 1);
        assert_eq!(report.scenario_scores[0].data_only_nullity, 1);
        assert!(report.scenario_scores[0].information_gain_nats.is_finite());
    }

    #[test]
    fn scenario_unavailability_is_visible_as_zero_gain_not_zero_noise() {
        let scenarios = vec![scenario("visible", 0.5), scenario("occluded", 0.5)];
        let options = vec![option(
            "bearing-sensor",
            1.0,
            None,
            vec![vec![vec![1.0, 0.0]], vec![]],
        )];

        let report = greedy_budgeted_design(&scenarios, &options, &config(1.0)).unwrap();
        assert!(report.scenario_scores[0].information_gain_nats > 0.0);
        assert!(report.scenario_scores[1].information_gain_nats.abs() < 1.0e-12);
        assert_eq!(report.scenario_scores[1].data_only_rank, 0);
        assert_eq!(report.scenario_scores[1].data_only_nullity, 2);
        assert!(report.worst_case_information_gain_nats.abs() < 1.0e-12);
    }

    #[test]
    fn option_order_permutation_keeps_ties_deterministic() {
        let scenarios = vec![scenario("single", 1.0)];
        let x = option("alpha", 1.0, None, vec![vec![vec![1.0, 0.0]]]);
        let y = option("beta", 1.0, None, vec![vec![vec![0.0, 1.0]]]);
        let first =
            greedy_budgeted_design(&scenarios, &[x.clone(), y.clone()], &config(1.0)).unwrap();
        let second = greedy_budgeted_design(&scenarios, &[y, x], &config(1.0)).unwrap();

        assert_eq!(first.selected_option_ids, vec!["alpha"]);
        assert_eq!(first.selected_option_ids, second.selected_option_ids);
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        let scenarios = vec![scenario("single", 1.0)];
        let good = option("sensor", 1.0, None, vec![vec![vec![1.0, 0.0]]]);

        assert_eq!(
            greedy_budgeted_design(&[], &[good.clone()], &config(1.0)),
            Err(BudgetedDesignError::NoScenarios)
        );
        assert_eq!(
            greedy_budgeted_design(&scenarios, &[], &config(1.0)),
            Err(BudgetedDesignError::NoOptions)
        );
        assert_eq!(
            greedy_budgeted_design(&scenarios, &[good.clone()], &config(f64::NAN)),
            Err(BudgetedDesignError::InvalidBudget)
        );
        assert!(matches!(
            greedy_budgeted_design(
                &scenarios,
                &[option("bad-cost", -1.0, None, vec![vec![vec![1.0, 0.0]]])],
                &config(1.0)
            ),
            Err(BudgetedDesignError::InvalidOptionCost { .. })
        ));
        assert!(matches!(
            greedy_budgeted_design(
                &scenarios,
                &[option("bad-row", 1.0, None, vec![vec![vec![1.0]]])],
                &config(1.0)
            ),
            Err(BudgetedDesignError::RaggedJacobian { .. })
        ));
        assert!(matches!(
            greedy_budgeted_design(
                &scenarios,
                &[option(
                    "bad-value",
                    1.0,
                    None,
                    vec![vec![vec![f64::NAN, 0.0]]]
                )],
                &config(1.0)
            ),
            Err(BudgetedDesignError::NonFiniteJacobian { .. })
        ));
    }

    #[test]
    fn target_marginal_information_equals_joint_information_without_nuisance() {
        let scenarios = vec![scenario("target-only", 1.0)];
        let options = vec![
            option("measure-x", 1.0, None, vec![vec![vec![1.0, 0.0]]]),
            option("measure-y", 1.0, None, vec![vec![vec![0.0, 2.0]]]),
        ];
        let scores = evaluate_selection(&scenarios, &options, &[0, 1], &[1.0, 1.0], 2).unwrap();

        assert!(
            (scores[0].information_gain_nats - scores[0].joint_information_gain_nats).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn uncoupled_nuisance_measurement_has_joint_gain_but_zero_target_gain() {
        let scenarios = vec![DesignScenario {
            scenario_id: "target-with-clock-nuisance".to_string(),
            target_parameter_count: 1,
            weight: 1.0,
            prior_precision: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        }];
        let options = vec![option(
            "clock-only-calibration",
            1.0,
            None,
            vec![vec![vec![0.0, 1.0]]],
        )];

        // Evaluate the action explicitly: a target-only optimizer should not
        // select it when its target marginal gain is zero, but the report must
        // still show that nuisance information was acquired.
        let scores = evaluate_selection(&scenarios, &options, &[0], &[1.0, 1.0], 2).unwrap();

        assert!(scores[0].information_gain_nats.abs() < 1.0e-12);
        assert!((scores[0].joint_information_gain_nats - 2.0_f64.ln()).abs() < 1.0e-12);
    }

    #[test]
    fn coupled_calibration_observation_improves_target_marginal_information() {
        let scenarios = vec![DesignScenario {
            scenario_id: "target-correlated-with-clock".to_string(),
            target_parameter_count: 1,
            weight: 1.0,
            prior_precision: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        }];
        let options = vec![
            option(
                "bearing-mixed-target-and-clock",
                1.0,
                None,
                vec![vec![vec![1.0, 1.0]]],
            ),
            option("clock-calibration", 1.0, None, vec![vec![vec![0.0, 1.0]]]),
        ];
        let bearing_only = evaluate_selection(&scenarios, &options, &[0], &[1.0, 1.0], 2).unwrap();
        let bearing_and_calibration =
            evaluate_selection(&scenarios, &options, &[0, 1], &[1.0, 1.0], 2).unwrap();

        let expected_bearing_target_gain = 1.5_f64.ln();
        let expected_combined_target_gain = (5.0_f64 / 3.0).ln();
        assert!(
            (bearing_only[0].information_gain_nats - expected_bearing_target_gain).abs() < 1.0e-12
        );
        assert!(
            (bearing_and_calibration[0].information_gain_nats - expected_combined_target_gain)
                .abs()
                < 1.0e-12
        );
        assert!(
            bearing_and_calibration[0].information_gain_nats
                > bearing_only[0].information_gain_nats
        );
        assert!(
            bearing_and_calibration[0].joint_information_gain_nats
                > bearing_only[0].joint_information_gain_nats
        );
    }

    #[test]
    fn schur_complement_matches_correlated_precision_reference() {
        let precision = vec![vec![2.0, 0.6], vec![0.6, 1.5]];
        let logdet = marginal_target_log_determinant(&precision, 1).unwrap();
        let expected_precision = 2.0 - 0.6_f64.powi(2) / 1.5;

        assert!((logdet - expected_precision.ln()).abs() < 1.0e-12);
    }

    #[test]
    fn singular_nuisance_precision_is_rejected_without_pseudoinverse() {
        let precision = vec![vec![2.0, 0.0], vec![0.0, 0.0]];

        assert!(marginal_target_log_determinant(&precision, 1).is_err());
    }

    #[test]
    fn invalid_target_parameter_partition_fails_closed() {
        let mut scenarios = vec![scenario("bad-target-partition", 1.0)];
        scenarios[0].target_parameter_count = 0;
        let options = vec![option("measure", 1.0, None, vec![vec![vec![1.0, 0.0]]])];

        assert!(matches!(
            greedy_budgeted_design(&scenarios, &options, &config(1.0)),
            Err(BudgetedDesignError::InvalidTargetParameterCount { .. })
        ));
    }

    #[test]
    fn greedy_selector_chooses_calibration_when_it_improves_target_marginal_eig() {
        let scenarios = vec![DesignScenario {
            scenario_id: "target-clock-coupled".to_string(),
            target_parameter_count: 1,
            weight: 1.0,
            prior_precision: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        }];
        let options = vec![
            option(
                "bearing-mixed-target-and-clock",
                1.0,
                None,
                vec![vec![vec![1.0, 1.0]]],
            ),
            option("clock-calibration", 1.0, None, vec![vec![vec![0.0, 1.0]]]),
        ];

        let greedy = greedy_budgeted_design(
            &scenarios,
            &options,
            &BudgetedDesignConfig {
                budget: 2.0,
                parameter_scales: vec![1.0, 1.0],
            },
        )
        .unwrap();
        let exact = exhaustive_budgeted_design(
            &scenarios,
            &options,
            &BudgetedDesignConfig {
                budget: 2.0,
                parameter_scales: vec![1.0, 1.0],
            },
        )
        .unwrap();

        assert_eq!(
            greedy.selected_option_ids,
            vec!["bearing-mixed-target-and-clock", "clock-calibration"]
        );
        assert_eq!(greedy.selected_option_ids, exact.selected_option_ids);
        assert!(
            (greedy.expected_information_gain_nats - (5.0_f64 / 3.0).ln()).abs()
                < 1.0e-12
        );
        assert!(greedy.expected_joint_information_gain_nats > greedy.expected_information_gain_nats);
    }

    #[test]
    fn nuisance_only_measurement_can_improve_target_when_prior_couples_them() {
        let scenarios = vec![DesignScenario {
            scenario_id: "correlated-target-clock-prior".to_string(),
            target_parameter_count: 1,
            weight: 1.0,
            prior_precision: vec![vec![2.0, 0.5], vec![0.5, 1.0]],
        }];
        let options = vec![option(
            "clock-only-calibration",
            1.0,
            None,
            vec![vec![vec![0.0, 1.0]]],
        )];
        let scores = evaluate_selection(
            &scenarios,
            &options,
            &[0],
            &[1.0, 1.0],
            2,
        )
        .unwrap();

        // Prior target marginal precision is 2 - 0.5^2/1 = 1.75.
        // After measuring only the nuisance parameter it becomes
        // 2 - 0.5^2/2 = 1.875, so target information legitimately improves.
        let expected_target_gain = (1.875_f64 / 1.75).ln();
        assert!(
            (scores[0].information_gain_nats - expected_target_gain).abs() < 1.0e-12
        );
        assert!(scores[0].joint_information_gain_nats > scores[0].information_gain_nats);
    }

    #[test]
    fn shared_clock_state_prevents_false_independent_sensor_information_gain() {
        let shared_clock_scenarios = vec![DesignScenario {
            scenario_id: "two-sensors-one-shared-clock".to_string(),
            target_parameter_count: 1,
            weight: 1.0,
            prior_precision: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        }];
        let shared_clock_options = vec![
            option("sensor-a", 1.0, None, vec![vec![vec![1.0, 1.0]]]),
            option("sensor-b", 1.0, None, vec![vec![vec![1.0, 1.0]]]),
        ];
        let shared_clock = evaluate_selection(
            &shared_clock_scenarios,
            &shared_clock_options,
            &[0, 1],
            &[1.0, 1.0],
            2,
        )
        .unwrap();

        // Deliberately misspecified alternative: each sensor is assigned an
        // independent clock-offset nuisance instead of the one shared clock.
        let independent_clock_scenarios = vec![DesignScenario {
            scenario_id: "two-sensors-two-independent-clocks".to_string(),
            target_parameter_count: 1,
            weight: 1.0,
            prior_precision: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        }];
        let independent_clock_options = vec![
            option("sensor-a", 1.0, None, vec![vec![vec![1.0, 1.0, 0.0]]]),
            option("sensor-b", 1.0, None, vec![vec![vec![1.0, 0.0, 1.0]]]),
        ];
        let independent_clocks = evaluate_selection(
            &independent_clock_scenarios,
            &independent_clock_options,
            &[0, 1],
            &[1.0, 1.0, 1.0],
            3,
        )
        .unwrap();

        let shared_expected = (5.0_f64 / 3.0).ln();
        let independent_expected = 2.0_f64.ln();
        assert!(
            (shared_clock[0].information_gain_nats - shared_expected).abs() < 1.0e-12
        );
        assert!(
            (independent_clocks[0].information_gain_nats - independent_expected).abs()
                < 1.0e-12
        );
        assert!(
            independent_clocks[0].information_gain_nats
                > shared_clock[0].information_gain_nats
        );
    }

    #[test]
    fn random_spd_blocks_satisfy_schur_determinant_identity() {
        // Construct lower-triangular factors with a deterministic sequence,
        // form Lambda = L L^T, then verify det(Schur) = det(Lambda) / det(C).
        for trial in 1..=64 {
            let mut lower = vec![vec![0.0; 4]; 4];
            for row in 0..4 {
                for column in 0..=row {
                    lower[row][column] = if row == column {
                        1.0 + ((trial * (row + 3) + column * 7) % 11) as f64 / 5.0
                    } else {
                        (((trial * (row + 5) + column * 13) % 17) as f64 - 8.0) / 23.0
                    };
                }
            }
            let mut precision = vec![vec![0.0; 4]; 4];
            for row in 0..4 {
                for column in 0..4 {
                    precision[row][column] = (0..4)
                        .map(|index| lower[row][index] * lower[column][index])
                        .sum();
                }
            }

            let full_logdet = cholesky_log_determinant(&precision).unwrap();
            let nuisance = vec![
                vec![precision[2][2], precision[2][3]],
                vec![precision[3][2], precision[3][3]],
            ];
            let nuisance_logdet = cholesky_log_determinant(&nuisance).unwrap();
            let target_schur_logdet =
                marginal_target_log_determinant(&precision, 2).unwrap();
            assert!(
                (target_schur_logdet - (full_logdet - nuisance_logdet)).abs() < 1.0e-10,
                "trial={trial}"
            );
        }
    }

    #[test]
    fn exhaustive_oracle_rejects_large_candidate_sets() {
        let scenarios = vec![scenario("single", 1.0)];
        let options = (0..21)
            .map(|index| {
                option(
                    &format!("sensor-{index:02}"),
                    1.0,
                    None,
                    vec![vec![vec![1.0, index as f64 + 1.0]]],
                )
            })
            .collect::<Vec<_>>();

        assert!(matches!(
            exhaustive_budgeted_design(&scenarios, &options, &config(2.0)),
            Err(BudgetedDesignError::TooManyOptionsForExhaustive {
                count: 21,
                maximum: 20
            })
        ));
    }
}
