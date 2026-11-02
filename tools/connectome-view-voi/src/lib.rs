#![forbid(unsafe_code)]

//! Decision-focused, robust value-of-information (VOI) selection for connectome evidence views.
//!
//! This crate evaluates whether an admissible evidence action is expected to change a
//! downstream decision enough to justify its cost. It deliberately does not rank evidence
//! by entropy. The input likelihoods and finite set of admissible belief scenarios must be
//! supplied with independent calibration/provenance by the caller.
//!
//! A positive *lower* net-VOI bound across every supplied admissible scenario is required
//! before the planner selects a view. If value is positive only in some scenarios, the
//! planner stops and abstains rather than averaging scenarios or inventing weights.

use std::collections::BTreeSet;
use std::fmt;

const PROBABILITY_SUM_TOLERANCE: f64 = 1.0e-9;
const POSITIVE_VOI_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionAction {
    pub action_id: String,
    /// Nonnegative loss for each state, in the order given by DecisionProblem.state_ids.
    pub losses_by_state: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefScenario {
    pub scenario_id: String,
    /// Prior probability for each state, in DecisionProblem.state_ids order.
    /// These are probabilities within one scenario; scenarios themselves are not weighted.
    pub prior_by_state: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioViewLikelihood {
    pub scenario_id: String,
    /// Stable signal identifiers. Rows in signal_likelihoods_by_state follow this order.
    pub signal_ids: Vec<String>,
    /// Matrix P(signal | state): rows are signals, columns are DecisionProblem.state_ids.
    /// For each state, likelihoods across all signal rows must sum to one.
    pub signal_likelihoods_by_state: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ViewOption {
    pub view_id: String,
    /// Evidence-acquisition cost expressed in the same loss units as DecisionAction.
    pub cost_units: f64,
    /// The upstream manifest must attest task compatibility before including this view.
    pub allowed_task_routes: Vec<String>,
    pub leakage_safe: bool,
    /// One likelihood model for every admissible belief scenario, keyed by scenario_id.
    pub scenario_likelihoods: Vec<ScenarioViewLikelihood>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionProblem {
    pub task_route: String,
    /// Stable state IDs. All probability and loss vectors use this ordering.
    pub state_ids: Vec<String>,
    pub actions: Vec<DecisionAction>,
    /// An unweighted finite set of admissible probability models. VOI is bounded over it.
    pub belief_scenarios: Vec<BeliefScenario>,
    pub view_options: Vec<ViewOption>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarBounds {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioVoi {
    pub scenario_id: String,
    pub risk_before: f64,
    pub risk_after: f64,
    pub net_voi: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ViewEvaluation {
    pub view_id: String,
    pub acquisition_cost_units: f64,
    pub risk_before_bounds: ScalarBounds,
    pub risk_after_bounds: ScalarBounds,
    pub net_voi_bounds: ScalarBounds,
    /// Scenario rows are sorted by scenario_id and retained for audit.
    pub scenario_results: Vec<ScenarioVoi>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// No candidate view was supplied (all unavailable or inadmissible upstream).
    NoAdmissibleViews,
    /// Every view has non-positive net VOI in every admissible scenario.
    NoPositiveNetValue,
    /// Some scenario yields positive value, but no view is positive in every scenario.
    NoRobustlyPositiveView,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ViewDecision {
    /// Selected only when its lower net-VOI bound is strictly positive above tolerance.
    pub selected_view_id: Option<String>,
    pub stop_reason: Option<StopReason>,
    /// Evaluations are sorted by view_id for stable receipts.
    pub evaluations: Vec<ViewEvaluation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiError {
    InvalidIdentifier {
        kind: &'static str,
        value: String,
    },
    DuplicateIdentifier {
        kind: &'static str,
        value: String,
    },
    EmptyCollection(&'static str),
    DimensionMismatch {
        object_id: String,
        expected: usize,
        actual: usize,
    },
    InvalidProbability {
        object_id: String,
        value_index: usize,
    },
    InvalidProbabilityDistribution {
        object_id: String,
        sum_bits: u64,
    },
    InvalidLoss {
        action_id: String,
        state_index: usize,
    },
    InvalidCost(String),
    LeakageForbidden(String),
    ViewForbiddenForTask {
        view_id: String,
        task_route: String,
    },
    UnknownScenario {
        view_id: String,
        scenario_id: String,
    },
    MissingScenario {
        view_id: String,
        scenario_id: String,
    },
    NonFiniteRisk(String),
}

impl fmt::Display for VoiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier { kind, value } => {
                write!(f, "invalid {kind} identifier: {value:?}")
            }
            Self::DuplicateIdentifier { kind, value } => {
                write!(f, "duplicate {kind} identifier: {value}")
            }
            Self::EmptyCollection(name) => write!(f, "{name} must not be empty"),
            Self::DimensionMismatch {
                object_id,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch for {object_id}: expected {expected}, got {actual}"
            ),
            Self::InvalidProbability {
                object_id,
                value_index,
            } => write!(
                f,
                "invalid probability in {object_id} at index {value_index}"
            ),
            Self::InvalidProbabilityDistribution { object_id, sum_bits } => {
                let sum = f64::from_bits(*sum_bits);
                write!(
                    f,
                    "probability distribution for {object_id} sums to {sum}, not one"
                )
            }
            Self::InvalidLoss {
                action_id,
                state_index,
            } => write!(
                f,
                "invalid loss for action {action_id} at state index {state_index}"
            ),
            Self::InvalidCost(id) => write!(f, "invalid acquisition cost for view {id}"),
            Self::LeakageForbidden(id) => {
                write!(f, "view {id} is not certified leakage-safe for this task")
            }
            Self::ViewForbiddenForTask {
                view_id,
                task_route,
            } => write!(
                f,
                "view {view_id} is not permitted for task route {task_route}"
            ),
            Self::UnknownScenario {
                view_id,
                scenario_id,
            } => write!(
                f,
                "view {view_id} references unknown scenario {scenario_id}"
            ),
            Self::MissingScenario {
                view_id,
                scenario_id,
            } => write!(
                f,
                "view {view_id} is missing scenario likelihoods for {scenario_id}"
            ),
            Self::NonFiniteRisk(id) => write!(f, "non-finite expected risk for {id}"),
        }
    }
}

impl std::error::Error for VoiError {}

/// Evaluate every admissible candidate view and choose the one with the largest robust
/// net-VOI lower bound. No averaging is performed across belief scenarios.
///
/// For scenario s and action a:
///
/// - before risk = min_a sum_y P_s(y) L(a,y)
/// - after risk = sum_v min_a sum_y P_s(y) P_s(v|y) L(a,y)
/// - net VOI = before risk - after risk - acquisition cost
///
/// Since the finite scenario set is an unweighted admissible uncertainty set, a view is
/// selected only when its minimum net VOI over all scenarios is greater than
/// POSITIVE_VOI_TOLERANCE. Otherwise the result is an explicit stop/abstain receipt.
///
/// Signals are processed in ascending signal_id order and view/scenario receipts are
/// canonicalized to make results independent of caller row order.
pub fn choose_next_view(problem: &DecisionProblem) -> Result<ViewDecision, VoiError> {
    validate_problem(problem)?;

    let mut evaluations = Vec::with_capacity(problem.view_options.len());
    for view in &problem.view_options {
        evaluations.push(evaluate_view(problem, view)?);
    }
    evaluations.sort_by(|left, right| left.view_id.cmp(&right.view_id));

    if evaluations.is_empty() {
        return Ok(ViewDecision {
            selected_view_id: None,
            stop_reason: Some(StopReason::NoAdmissibleViews),
            evaluations,
        });
    }

    let mut best_index: Option<usize> = None;
    for (index, candidate) in evaluations.iter().enumerate() {
        if candidate.net_voi_bounds.lower <= POSITIVE_VOI_TOLERANCE {
            continue;
        }

        match best_index {
            None => best_index = Some(index),
            Some(current_index) => {
                let current = &evaluations[current_index];
                let ordering = candidate
                    .net_voi_bounds
                    .lower
                    .total_cmp(&current.net_voi_bounds.lower);
                if ordering.is_gt()
                    || (ordering.is_eq() && candidate.view_id < current.view_id)
                {
                    best_index = Some(index);
                }
            }
        }
    }

    if let Some(index) = best_index {
        return Ok(ViewDecision {
            selected_view_id: Some(evaluations[index].view_id.clone()),
            stop_reason: None,
            evaluations,
        });
    }

    let max_upper = evaluations
        .iter()
        .map(|evaluation| evaluation.net_voi_bounds.upper)
        .fold(f64::NEG_INFINITY, f64::max);

    let stop_reason = if max_upper <= POSITIVE_VOI_TOLERANCE {
        StopReason::NoPositiveNetValue
    } else {
        StopReason::NoRobustlyPositiveView
    };

    Ok(ViewDecision {
        selected_view_id: None,
        stop_reason: Some(stop_reason),
        evaluations,
    })
}

fn validate_problem(problem: &DecisionProblem) -> Result<(), VoiError> {
    validate_id("task route", &problem.task_route)?;
    if problem.state_ids.is_empty() {
        return Err(VoiError::EmptyCollection("state_ids"));
    }
    if problem.actions.is_empty() {
        return Err(VoiError::EmptyCollection("actions"));
    }
    if problem.belief_scenarios.is_empty() {
        return Err(VoiError::EmptyCollection("belief_scenarios"));
    }

    validate_unique_ids(
        "state",
        problem.state_ids.iter().map(String::as_str),
    )?;
    validate_unique_ids(
        "action",
        problem.actions.iter().map(|action| action.action_id.as_str()),
    )?;
    validate_unique_ids(
        "belief scenario",
        problem
            .belief_scenarios
            .iter()
            .map(|scenario| scenario.scenario_id.as_str()),
    )?;
    validate_unique_ids(
        "view",
        problem.view_options.iter().map(|view| view.view_id.as_str()),
    )?;

    for action in &problem.actions {
        validate_id("action", &action.action_id)?;
        if action.losses_by_state.len() != problem.state_ids.len() {
            return Err(VoiError::DimensionMismatch {
                object_id: action.action_id.clone(),
                expected: problem.state_ids.len(),
                actual: action.losses_by_state.len(),
            });
        }
        for (state_index, loss) in action.losses_by_state.iter().enumerate() {
            if !loss.is_finite() || *loss < 0.0 {
                return Err(VoiError::InvalidLoss {
                    action_id: action.action_id.clone(),
                    state_index,
                });
            }
        }
    }

    for scenario in &problem.belief_scenarios {
        validate_id("belief scenario", &scenario.scenario_id)?;
        if scenario.prior_by_state.len() != problem.state_ids.len() {
            return Err(VoiError::DimensionMismatch {
                object_id: scenario.scenario_id.clone(),
                expected: problem.state_ids.len(),
                actual: scenario.prior_by_state.len(),
            });
        }
        validate_probability_distribution(
            &scenario.scenario_id,
            &scenario.prior_by_state,
        )?;
    }

    for view in &problem.view_options {
        validate_id("view", &view.view_id)?;
        if !view.cost_units.is_finite() || view.cost_units < 0.0 {
            return Err(VoiError::InvalidCost(view.view_id.clone()));
        }
        if !view.leakage_safe {
            return Err(VoiError::LeakageForbidden(view.view_id.clone()));
        }
        if !view
            .allowed_task_routes
            .iter()
            .any(|route| route == &problem.task_route)
        {
            return Err(VoiError::ViewForbiddenForTask {
                view_id: view.view_id.clone(),
                task_route: problem.task_route.clone(),
            });
        }
        validate_unique_ids(
            "allowed task route",
            view.allowed_task_routes.iter().map(String::as_str),
        )?;

        validate_unique_ids(
            "view scenario",
            view.scenario_likelihoods
                .iter()
                .map(|scenario| scenario.scenario_id.as_str()),
        )?;

        for scenario_view in &view.scenario_likelihoods {
            validate_id("view scenario", &scenario_view.scenario_id)?;
            if !problem
                .belief_scenarios
                .iter()
                .any(|scenario| scenario.scenario_id == scenario_view.scenario_id)
            {
                return Err(VoiError::UnknownScenario {
                    view_id: view.view_id.clone(),
                    scenario_id: scenario_view.scenario_id.clone(),
                });
            }
            validate_view_likelihoods(
                &view.view_id,
                scenario_view,
                problem.state_ids.len(),
            )?;
        }

        for scenario in &problem.belief_scenarios {
            if !view
                .scenario_likelihoods
                .iter()
                .any(|likelihood| likelihood.scenario_id == scenario.scenario_id)
            {
                return Err(VoiError::MissingScenario {
                    view_id: view.view_id.clone(),
                    scenario_id: scenario.scenario_id.clone(),
                });
            }
        }
    }

    Ok(())
}

fn validate_view_likelihoods(
    view_id: &str,
    likelihood: &ScenarioViewLikelihood,
    state_count: usize,
) -> Result<(), VoiError> {
    if likelihood.signal_ids.is_empty() {
        return Err(VoiError::EmptyCollection("signal_ids"));
    }
    validate_unique_ids(
        "signal",
        likelihood.signal_ids.iter().map(String::as_str),
    )?;
    if likelihood.signal_likelihoods_by_state.len() != likelihood.signal_ids.len() {
        return Err(VoiError::DimensionMismatch {
            object_id: format!("{view_id}:{}", likelihood.scenario_id),
            expected: likelihood.signal_ids.len(),
            actual: likelihood.signal_likelihoods_by_state.len(),
        });
    }

    for (signal_index, row) in likelihood
        .signal_likelihoods_by_state
        .iter()
        .enumerate()
    {
        if row.len() != state_count {
            return Err(VoiError::DimensionMismatch {
                object_id: format!("{view_id}:signal:{}", likelihood.signal_ids[signal_index]),
                expected: state_count,
                actual: row.len(),
            });
        }
        for (state_index, probability) in row.iter().enumerate() {
            if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                return Err(VoiError::InvalidProbability {
                    object_id: format!(
                        "{view_id}:signal:{}",
                        likelihood.signal_ids[signal_index]
                    ),
                    value_index: state_index,
                });
            }
        }
    }

    for state_index in 0..state_count {
        let column: Vec<f64> = likelihood
            .signal_likelihoods_by_state
            .iter()
            .map(|row| row[state_index])
            .collect();
        validate_probability_distribution(
            &format!("{view_id}:{}:state:{state_index}", likelihood.scenario_id),
            &column,
        )?;
    }

    Ok(())
}

fn evaluate_view(
    problem: &DecisionProblem,
    view: &ViewOption,
) -> Result<ViewEvaluation, VoiError> {
    let mut scenario_results = Vec::with_capacity(problem.belief_scenarios.len());
    for belief in &problem.belief_scenarios {
        let likelihood = view
            .scenario_likelihoods
            .iter()
            .find(|candidate| candidate.scenario_id == belief.scenario_id)
            .ok_or_else(|| VoiError::MissingScenario {
                view_id: view.view_id.clone(),
                scenario_id: belief.scenario_id.clone(),
            })?;

        let risk_before = bayes_risk(&belief.prior_by_state, &problem.actions);
        let risk_after = expected_post_view_risk(
            &belief.prior_by_state,
            likelihood,
            &problem.actions,
        );
        let net_voi = risk_before - risk_after - view.cost_units;

        if !risk_before.is_finite() || !risk_after.is_finite() || !net_voi.is_finite() {
            return Err(VoiError::NonFiniteRisk(view.view_id.clone()));
        }
        scenario_results.push(ScenarioVoi {
            scenario_id: belief.scenario_id.clone(),
            risk_before,
            risk_after,
            net_voi,
        });
    }

    scenario_results.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));

    let risk_before_bounds = bounds_from(
        scenario_results.iter().map(|result| result.risk_before),
    );
    let risk_after_bounds = bounds_from(
        scenario_results.iter().map(|result| result.risk_after),
    );
    let net_voi_bounds = bounds_from(
        scenario_results.iter().map(|result| result.net_voi),
    );

    Ok(ViewEvaluation {
        view_id: view.view_id.clone(),
        acquisition_cost_units: view.cost_units,
        risk_before_bounds,
        risk_after_bounds,
        net_voi_bounds,
        scenario_results,
    })
}

fn bayes_risk(prior: &[f64], actions: &[DecisionAction]) -> f64 {
    actions
        .iter()
        .map(|action| compensated_dot(prior, &action.losses_by_state))
        .fold(f64::INFINITY, f64::min)
}

fn expected_post_view_risk(
    prior: &[f64],
    likelihood: &ScenarioViewLikelihood,
    actions: &[DecisionAction],
) -> f64 {
    let mut signal_indices: Vec<usize> = (0..likelihood.signal_ids.len()).collect();
    signal_indices.sort_by(|left, right| {
        likelihood.signal_ids[*left].cmp(&likelihood.signal_ids[*right])
    });

    let mut risk_by_signal = Vec::with_capacity(signal_indices.len());
    for signal_index in signal_indices {
        let weighted_state_probabilities: Vec<f64> = prior
            .iter()
            .enumerate()
            .map(|(state_index, probability)| {
                probability * likelihood.signal_likelihoods_by_state[signal_index][state_index]
            })
            .collect();
        let risk_for_signal = actions
            .iter()
            .map(|action| {
                compensated_dot(&weighted_state_probabilities, &action.losses_by_state)
            })
            .fold(f64::INFINITY, f64::min);
        risk_by_signal.push(risk_for_signal);
    }

    compensated_sum(&risk_by_signal)
}

fn compensated_dot(left: &[f64], right: &[f64]) -> f64 {
    let products: Vec<f64> = left
        .iter()
        .zip(right)
        .map(|(left_value, right_value)| left_value * right_value)
        .collect();
    compensated_sum(&products)
}

fn compensated_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let adjusted = value - compensation;
        let next = sum + adjusted;
        compensation = (next - sum) - adjusted;
        sum = next;
    }
    sum
}

fn bounds_from(values: impl Iterator<Item = f64>) -> ScalarBounds {
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for value in values {
        lower = lower.min(value);
        upper = upper.max(value);
    }
    ScalarBounds { lower, upper }
}

fn validate_probability_distribution(
    object_id: &str,
    values: &[f64],
) -> Result<(), VoiError> {
    for (value_index, value) in values.iter().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(value) {
            return Err(VoiError::InvalidProbability {
                object_id: object_id.to_owned(),
                value_index,
            });
        }
    }
    let sum = compensated_sum(values);
    if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        return Err(VoiError::InvalidProbabilityDistribution {
            object_id: object_id.to_owned(),
            sum_bits: sum.to_bits(),
        });
    }
    Ok(())
}

fn validate_id(kind: &'static str, value: &str) -> Result<(), VoiError> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(VoiError::InvalidIdentifier {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_unique_ids<'a>(
    kind: &'static str,
    values: impl Iterator<Item = &'a str>,
) -> Result<(), VoiError> {
    let mut seen = BTreeSet::new();
    for value in values {
        validate_id(kind, value)?;
        if !seen.insert(value.to_owned()) {
            return Err(VoiError::DuplicateIdentifier {
                kind,
                value: value.to_owned(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_problem(
        prior_scenarios: Vec<BeliefScenario>,
        view_options: Vec<ViewOption>,
    ) -> DecisionProblem {
        DecisionProblem {
            task_route: "false_merge_identification".to_owned(),
            state_ids: vec!["error".to_owned(), "no_error".to_owned()],
            actions: vec![
                DecisionAction {
                    action_id: "accept".to_owned(),
                    losses_by_state: vec![10.0, 0.0],
                },
                DecisionAction {
                    action_id: "review".to_owned(),
                    losses_by_state: vec![1.0, 2.0],
                },
            ],
            belief_scenarios: prior_scenarios,
            view_options,
        }
    }

    fn view(
        view_id: &str,
        cost: f64,
        safe: bool,
        allowed_tasks: &[&str],
        scenario_likelihoods: Vec<ScenarioViewLikelihood>,
    ) -> ViewOption {
        ViewOption {
            view_id: view_id.to_owned(),
            cost_units: cost,
            allowed_task_routes: allowed_tasks.iter().map(|value| (*value).to_owned()).collect(),
            leakage_safe: safe,
            scenario_likelihoods,
        }
    }

    fn likelihood(
        scenario_id: &str,
        signal_ids: &[&str],
        rows: Vec<Vec<f64>>,
    ) -> ScenarioViewLikelihood {
        ScenarioViewLikelihood {
            scenario_id: scenario_id.to_owned(),
            signal_ids: signal_ids.iter().map(|value| (*value).to_owned()).collect(),
            signal_likelihoods_by_state: rows,
        }
    }

    fn scenario(id: &str, p_error: f64) -> BeliefScenario {
        BeliefScenario {
            scenario_id: id.to_owned(),
            prior_by_state: vec![p_error, 1.0 - p_error],
        }
    }

    #[test]
    fn perfect_view_is_selected_when_robust_net_value_is_positive() {
        let problem = standard_problem(
            vec![scenario("base", 0.5)],
            vec![view(
                "perfect",
                0.1,
                true,
                &["false_merge_identification"],
                vec![likelihood(
                    "base",
                    &["error_signal", "no_error_signal"],
                    vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                )],
            )],
        );

        let decision = choose_next_view(&problem).unwrap();

        assert_eq!(decision.selected_view_id.as_deref(), Some("perfect"));
        assert_eq!(decision.stop_reason, None);
        let evaluation = &decision.evaluations[0];
        assert!((evaluation.risk_before_bounds.lower - 1.5).abs() < 1e-12);
        assert!((evaluation.risk_after_bounds.upper - 0.5).abs() < 1e-12);
        assert!((evaluation.net_voi_bounds.lower - 0.9).abs() < 1e-12);
    }

    #[test]
    fn conditionally_redundant_view_has_zero_gross_value_and_is_rejected_after_cost() {
        let problem = standard_problem(
            vec![scenario("base", 0.5)],
            vec![view(
                "redundant",
                0.1,
                true,
                &["false_merge_identification"],
                vec![likelihood(
                    "base",
                    &["signal-a", "signal-b"],
                    vec![vec![0.5, 0.5], vec![0.5, 0.5]],
                )],
            )],
        );

        let decision = choose_next_view(&problem).unwrap();

        assert_eq!(decision.selected_view_id, None);
        assert_eq!(decision.stop_reason, Some(StopReason::NoPositiveNetValue));
        assert!((decision.evaluations[0].net_voi_bounds.lower + 0.1).abs() < 1e-12);
    }

    #[test]
    fn informative_view_is_not_selected_when_it_cannot_change_the_optimal_action() {
        let mut problem = standard_problem(
            vec![scenario("rare_error", 0.01)],
            vec![view(
                "perfect_but_irrelevant",
                0.1,
                true,
                &["false_merge_identification"],
                vec![likelihood(
                    "rare_error",
                    &["error_signal", "no_error_signal"],
                    vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                )],
            )],
        );
        problem.actions = vec![
            DecisionAction {
                action_id: "always_accept".to_owned(),
                losses_by_state: vec![1.0, 0.0],
            },
            DecisionAction {
                action_id: "always_review".to_owned(),
                losses_by_state: vec![100.0, 100.0],
            },
        ];

        let decision = choose_next_view(&problem).unwrap();

        assert_eq!(decision.selected_view_id, None);
        assert_eq!(decision.stop_reason, Some(StopReason::NoPositiveNetValue));
        assert!((decision.evaluations[0].risk_before_bounds.lower - 0.01).abs() < 1e-12);
        assert!((decision.evaluations[0].risk_after_bounds.upper - 0.01).abs() < 1e-12);
    }

    #[test]
    fn scenario_disagreement_forces_stop_instead_of_averaging() {
        let problem = standard_problem(
            vec![scenario("likely", 0.5), scenario("rare", 0.01)],
            vec![view(
                "perfect",
                0.2,
                true,
                &["false_merge_identification"],
                vec![
                    likelihood(
                        "likely",
                        &["error_signal", "no_error_signal"],
                        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                    ),
                    likelihood(
                        "rare",
                        &["error_signal", "no_error_signal"],
                        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                    ),
                ],
            )],
        );

        let decision = choose_next_view(&problem).unwrap();

        assert_eq!(decision.selected_view_id, None);
        assert_eq!(
            decision.stop_reason,
            Some(StopReason::NoRobustlyPositiveView)
        );
        assert!(decision.evaluations[0].net_voi_bounds.lower < 0.0);
        assert!(decision.evaluations[0].net_voi_bounds.upper > 0.0);
    }

    #[test]
    fn selects_view_with_highest_worst_case_net_value() {
        let problem = standard_problem(
            vec![scenario("s1", 0.5), scenario("s2", 0.4)],
            vec![
                view(
                    "view-b",
                    0.05,
                    true,
                    &["false_merge_identification"],
                    vec![
                        likelihood(
                            "s1",
                            &["error_signal", "no_error_signal"],
                            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                        ),
                        likelihood(
                            "s2",
                            &["error_signal", "no_error_signal"],
                            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                        ),
                    ],
                ),
                view(
                    "view-a",
                    0.1,
                    true,
                    &["false_merge_identification"],
                    vec![
                        likelihood(
                            "s1",
                            &["signal-a", "signal-b"],
                            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
                        ),
                        likelihood(
                            "s2",
                            &["signal-a", "signal-b"],
                            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
                        ),
                    ],
                ),
            ],
        );

        let decision = choose_next_view(&problem).unwrap();

        assert_eq!(decision.selected_view_id.as_deref(), Some("view-b"));
        assert!(decision.evaluations.iter().all(|evaluation| {
            evaluation.net_voi_bounds.lower > POSITIVE_VOI_TOLERANCE
        }));
    }

    #[test]
    fn no_views_returns_explicit_no_admissible_action() {
        let problem = standard_problem(vec![scenario("base", 0.5)], vec![]);

        let decision = choose_next_view(&problem).unwrap();

        assert_eq!(decision.selected_view_id, None);
        assert_eq!(decision.stop_reason, Some(StopReason::NoAdmissibleViews));
        assert!(decision.evaluations.is_empty());
    }

    #[test]
    fn rejects_view_with_target_leakage_before_evaluating_likelihoods() {
        let problem = standard_problem(
            vec![scenario("base", 0.5)],
            vec![view(
                "em_best",
                0.0,
                false,
                &["false_merge_identification"],
                vec![likelihood(
                    "base",
                    &["yes", "no"],
                    vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                )],
            )],
        );

        assert_eq!(
            choose_next_view(&problem),
            Err(VoiError::LeakageForbidden("em_best".to_owned()))
        );
    }

    #[test]
    fn rejects_view_not_allowed_for_current_task() {
        let problem = standard_problem(
            vec![scenario("base", 0.5)],
            vec![view(
                "dual_mask_only_view",
                0.0,
                true,
                &["false_split_correction"],
                vec![likelihood(
                    "base",
                    &["yes", "no"],
                    vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                )],
            )],
        );

        assert_eq!(
            choose_next_view(&problem),
            Err(VoiError::ViewForbiddenForTask {
                view_id: "dual_mask_only_view".to_owned(),
                task_route: "false_merge_identification".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_invalid_prior_and_non_normalized_signal_likelihoods() {
        let mut bad_prior = standard_problem(
            vec![BeliefScenario {
                scenario_id: "bad".to_owned(),
                prior_by_state: vec![0.8, 0.8],
            }],
            vec![],
        );
        assert!(matches!(
            choose_next_view(&bad_prior),
            Err(VoiError::InvalidProbabilityDistribution { .. })
        ));

        bad_prior.belief_scenarios = vec![scenario("base", 0.5)];
        bad_prior.view_options = vec![view(
            "bad-likelihood",
            0.0,
            true,
            &["false_merge_identification"],
            vec![likelihood(
                "base",
                &["signal-a", "signal-b"],
                vec![vec![0.9, 0.1], vec![0.0, 0.0]],
            )],
        )];
        assert!(matches!(
            choose_next_view(&bad_prior),
            Err(VoiError::InvalidProbabilityDistribution { .. })
        ));
    }

    #[test]
    fn rejects_missing_or_unknown_scenario_likelihoods() {
        let mut problem = standard_problem(
            vec![scenario("base", 0.5), scenario("shift", 0.4)],
            vec![view(
                "missing",
                0.0,
                true,
                &["false_merge_identification"],
                vec![likelihood(
                    "base",
                    &["yes", "no"],
                    vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                )],
            )],
        );
        assert_eq!(
            choose_next_view(&problem),
            Err(VoiError::MissingScenario {
                view_id: "missing".to_owned(),
                scenario_id: "shift".to_owned(),
            })
        );

        problem.view_options[0].scenario_likelihoods[0].scenario_id =
            "unregistered".to_owned();
        assert_eq!(
            choose_next_view(&problem),
            Err(VoiError::UnknownScenario {
                view_id: "missing".to_owned(),
                scenario_id: "unregistered".to_owned(),
            })
        );
    }

    #[test]
    fn input_view_and_signal_order_does_not_change_decision() {
        let base = scenario("base", 0.5);
        let view_a = view(
            "a",
            0.1,
            true,
            &["false_merge_identification"],
            vec![likelihood(
                "base",
                &["signal-a", "signal-b"],
                vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            )],
        );
        let view_b = view(
            "b",
            0.05,
            true,
            &["false_merge_identification"],
            vec![likelihood(
                "base",
                &["error_signal", "no_error_signal"],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            )],
        );

        let first = standard_problem(vec![base.clone()], vec![view_a.clone(), view_b.clone()]);
        let second = standard_problem(
            vec![base],
            vec![
                view(
                    "b",
                    0.05,
                    true,
                    &["false_merge_identification"],
                    vec![likelihood(
                        "base",
                        &["no_error_signal", "error_signal"],
                        vec![vec![0.0, 1.0], vec![1.0, 0.0]],
                    )],
                ),
                view_a,
            ],
        );

        assert_eq!(
            choose_next_view(&first).unwrap(),
            choose_next_view(&second).unwrap()
        );
    }

    #[test]
    fn equal_worst_case_value_uses_lexicographic_view_id() {
        let shared = vec![
            likelihood(
                "base",
                &["error_signal", "no_error_signal"],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            ),
        ];
        let problem = standard_problem(
            vec![scenario("base", 0.5)],
            vec![
                view("zeta", 0.1, true, &["false_merge_identification"], shared.clone()),
                view("alpha", 0.1, true, &["false_merge_identification"], shared),
            ],
        );

        let decision = choose_next_view(&problem).unwrap();

        assert_eq!(decision.selected_view_id.as_deref(), Some("alpha"));
    }
}
