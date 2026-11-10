// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Bounded finite-horizon decision-focused diagnostic planning.
//!
//! One-step expected value of sample information can miss complementary probes:
//! no individual observation need improve the immediate decision, while a
//! sequence of observations can jointly identify the decision-relevant cause.
//! This module performs finite-horizon Bayesian decision-tree search over
//! action-conditioned categorical likelihoods.
//!
//! Every node has an explicit stop option. A probe is removed after use, so
//! repeated measurements are not silently treated as conditionally independent.
//! The search has a hard expansion budget and returns an error rather than a
//! partial plan when that budget is exhausted. This is a finite-hypothesis
//! planner, not a general POMDP solver or an action-execution authority.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::decision_relevant_information::{
    DecisionDiagnosticConfig, DecisionDiagnosticError, DecisionHypothesis, DiagnosticAction,
    evaluate_decision_focused_diagnostics,
};

const RISK_TOLERANCE: f64 = 1e-10;
const MAX_DIAGNOSTIC_ACTIONS: usize = 64;
const MAX_HYPOTHESES: usize = 1024;
const MAX_OUTCOMES_PER_ACTION: usize = 256;

/// Provenance-bound cumulative safety budget for one adaptive diagnostic path.
///
/// Each action's safety_risk must be an externally calibrated upper bound on
/// the conditional probability of an adverse safety event if that action is
/// taken after any reachable observation history. The union bound then makes
/// the sum of these per-action bounds a conservative upper bound on the
/// probability of at least one adverse event along a path, without assuming
/// independence. The digest identifies the external risk semantics/calibration;
/// it does not authenticate the issuer or prove the calibration is correct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialDiagnosticSafetyBudget {
    pub maximum_cumulative_risk: f64,
    pub risk_semantics_digest: String,
}

/// Evaluation returned when a path-wise safety budget is enabled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetedSequentialDiagnosticEvaluation {
    pub safety_budget: SequentialDiagnosticSafetyBudget,
    /// Canonical SHA-256 over the exact numerical budget and risk-semantics digest.
    pub safety_budget_digest: String,
    pub evaluation: SequentialDiagnosticEvaluation,
}

/// Finite-horizon search limits.
///
/// The expansion budget counts belief-state nodes in the search tree. In
/// addition, explicit hypothesis, action, and outcome caps bound each node's
/// branching and validation work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequentialDiagnosticConfig {
    pub horizon: usize,
    pub max_policy_expansions: usize,
}

impl Default for SequentialDiagnosticConfig {
    fn default() -> Self {
        Self {
            horizon: 2,
            max_policy_expansions: 4096,
        }
    }
}

/// One possible observation and the optimal continuation after that outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialDiagnosticBranch {
    pub outcome_index: usize,
    pub outcome_probability: f64,
    pub posterior_model_probabilities: Vec<f64>,
    pub continuation: Box<SequentialDiagnosticPlanNode>,
}

/// A node in the selected adaptive policy tree.
///
/// A node with selected_action_index=None stops and takes the Bayes-optimal
/// downstream decision under its current posterior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialDiagnosticPlanNode {
    pub selected_action_index: Option<usize>,
    pub selected_action_id: Option<String>,
    pub expected_total_cost: f64,
    pub expected_terminal_bayes_risk: f64,
    pub expected_observation_cost: f64,
    /// Safety-risk penalty after applying the configured risk weight.
    pub expected_safety_penalty: f64,
    pub branches: Vec<SequentialDiagnosticBranch>,
}

/// Root-level value of taking one particular first probe and then planning
/// optimally for the remaining horizon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialDiagnosticActionValue {
    pub action_index: usize,
    pub action_id: String,
    pub eligible: bool,
    pub expected_total_cost: Option<f64>,
    pub expected_terminal_bayes_risk: Option<f64>,
    pub expected_observation_cost: Option<f64>,
    pub expected_safety_penalty: Option<f64>,
}

/// Full finite-horizon evaluation, including the no-probe alternative.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialDiagnosticEvaluation {
    pub horizon: usize,
    pub expansions: usize,
    pub no_probe_bayes_risk: f64,
    pub expected_total_cost: f64,
    pub selected_action_index: Option<usize>,
    pub first_action_values: Vec<SequentialDiagnosticActionValue>,
    pub plan: SequentialDiagnosticPlanNode,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum SequentialDiagnosticError {
    #[error(transparent)]
    DecisionDiagnostic(#[from] DecisionDiagnosticError),
    #[error("at most {maximum} diagnostic actions are supported, received {actual}")]
    TooManyActions { maximum: usize, actual: usize },
    #[error("at most {maximum} model hypotheses are supported, received {actual}")]
    TooManyHypotheses { maximum: usize, actual: usize },
    #[error("action {action_index} has {actual} possible outcomes; maximum is {maximum}")]
    TooManyOutcomes {
        action_index: usize,
        maximum: usize,
        actual: usize,
    },
    #[error("planning horizon {horizon} exceeds the {action_count} distinct actions")]
    HorizonExceedsActionCount { horizon: usize, action_count: usize },
    #[error("planning expansion budget must be positive")]
    EmptyExpansionBudget,
    #[error("sequential diagnostic planning exceeded its expansion budget of {budget}")]
    ExpansionBudgetExceeded { budget: usize },
    #[error("predictive outcome mass is non-finite or non-positive for action {action_index}")]
    InvalidPredictiveOutcomeMass { action_index: usize },
    #[error("decision-risk calculation produced a non-finite value")]
    NonFiniteRisk,
    #[error("cumulative safety-risk budget must be finite and in [0, 1]")]
    InvalidCumulativeSafetyRiskBudget,
    #[error("risk-semantics digest must be 64 lowercase hexadecimal characters")]
    InvalidRiskSemanticsDigest,
}

/// Evaluate an adaptive diagnostic policy up to the requested horizon.
///
/// The planner minimizes expected terminal Bayes risk plus weighted
/// observation costs and safety penalties. It can stop at the root or at any
/// subsequent branch. Each action can be used at most once on a path.
///
/// Horizon 1 is equivalent to the one-step EVSI selector (same objective,
/// action eligibility, tie behavior, and no-probe option).
pub fn evaluate_sequential_decision_diagnostics(
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    decision_config: DecisionDiagnosticConfig,
    planning_config: SequentialDiagnosticConfig,
) -> Result<SequentialDiagnosticEvaluation, SequentialDiagnosticError> {
    evaluate_sequential_decision_diagnostics_internal(
        hypotheses,
        actions,
        decision_config,
        planning_config,
        None,
    )
}

/// Evaluate a sequential diagnostic policy with a hard cumulative safety bound.
///
/// Unlike DecisionDiagnosticConfig::maximum_safety_risk, which applies to
/// each individual probe, this budget constrains the sum of conditional
/// per-action adverse-event probability upper bounds on every possible path.
/// A probe that is individually admissible can therefore become ineligible
/// after another probe has consumed part of the path budget. The risk-semantics
/// digest must identify the external calibration/meaning of safety_risk.
///
/// This is a conservative union-bound control, not an independence assumption
/// and not proof that the supplied per-action bounds are valid.
pub fn evaluate_sequential_decision_diagnostics_with_safety_budget(
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    decision_config: DecisionDiagnosticConfig,
    planning_config: SequentialDiagnosticConfig,
    safety_budget: SequentialDiagnosticSafetyBudget,
) -> Result<BudgetedSequentialDiagnosticEvaluation, SequentialDiagnosticError> {
    let safety_budget_digest =
        compute_sequential_diagnostic_safety_budget_digest(&safety_budget)?;
    let evaluation = evaluate_sequential_decision_diagnostics_internal(
        hypotheses,
        actions,
        decision_config,
        planning_config,
        Some(safety_budget.maximum_cumulative_risk),
    )?;
    Ok(BudgetedSequentialDiagnosticEvaluation {
        safety_budget,
        safety_budget_digest,
        evaluation,
    })
}

/// Compute a canonical, domain-separated SHA-256 digest for a path safety budget.
///
/// The digest binds the exact IEEE-754 bit pattern of the maximum cumulative
/// risk together with the external risk-semantics/calibration digest.
pub fn compute_sequential_diagnostic_safety_budget_digest(
    safety_budget: &SequentialDiagnosticSafetyBudget,
) -> Result<String, SequentialDiagnosticError> {
    validate_safety_budget(safety_budget)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-sequential-diagnostic-safety-budget-v1\\0");
    hasher.update(safety_budget.maximum_cumulative_risk.to_bits().to_le_bytes());
    hasher.update((safety_budget.risk_semantics_digest.len() as u64).to_le_bytes());
    hasher.update(safety_budget.risk_semantics_digest.as_bytes());
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(encoded)
}

fn validate_safety_budget(
    safety_budget: &SequentialDiagnosticSafetyBudget,
) -> Result<(), SequentialDiagnosticError> {
    if !safety_budget.maximum_cumulative_risk.is_finite()
        || !(0.0..=1.0).contains(&safety_budget.maximum_cumulative_risk)
    {
        return Err(SequentialDiagnosticError::InvalidCumulativeSafetyRiskBudget);
    }
    if safety_budget.risk_semantics_digest.len() != 64
        || !safety_budget
            .risk_semantics_digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SequentialDiagnosticError::InvalidRiskSemanticsDigest);
    }
    Ok(())
}

fn evaluate_sequential_decision_diagnostics_internal(
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    decision_config: DecisionDiagnosticConfig,
    planning_config: SequentialDiagnosticConfig,
    maximum_cumulative_safety_risk: Option<f64>,
) -> Result<SequentialDiagnosticEvaluation, SequentialDiagnosticError> {
    if actions.len() > MAX_DIAGNOSTIC_ACTIONS {
        return Err(SequentialDiagnosticError::TooManyActions {
            maximum: MAX_DIAGNOSTIC_ACTIONS,
            actual: actions.len(),
        });
    }
    if hypotheses.len() > MAX_HYPOTHESES {
        return Err(SequentialDiagnosticError::TooManyHypotheses {
            maximum: MAX_HYPOTHESES,
            actual: hypotheses.len(),
        });
    }
    if let Some(first) = hypotheses.first() {
        for (action_index, outcomes) in first.outcome_probabilities_by_action.iter().enumerate() {
            if outcomes.len() > MAX_OUTCOMES_PER_ACTION {
                return Err(SequentialDiagnosticError::TooManyOutcomes {
                    action_index,
                    maximum: MAX_OUTCOMES_PER_ACTION,
                    actual: outcomes.len(),
                });
            }
        }
    }
    if planning_config.horizon > actions.len() {
        return Err(SequentialDiagnosticError::HorizonExceedsActionCount {
            horizon: planning_config.horizon,
            action_count: actions.len(),
        });
    }
    if planning_config.max_policy_expansions == 0 {
        return Err(SequentialDiagnosticError::EmptyExpansionBudget);
    }
    // Reuse the one-step kernel as the canonical validator for priors,
    // likelihood shapes, losses, action descriptors, and policy weights.
    let one_step = evaluate_decision_focused_diagnostics(hypotheses, actions, decision_config)?;

    let prior: Vec<f64> = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .collect();
    let mut expansions = 0;
    let remaining_actions = if actions.len() == MAX_DIAGNOSTIC_ACTIONS {
        u64::MAX
    } else {
        (1_u64 << actions.len()) - 1
    };

    let (plan, first_action_values) = solve_node(
        hypotheses,
        actions,
        decision_config,
        planning_config,
        &prior,
        planning_config.horizon,
        remaining_actions,
        maximum_cumulative_safety_risk,
        &mut expansions,
    )?;

    Ok(SequentialDiagnosticEvaluation {
        horizon: planning_config.horizon,
        expansions,
        no_probe_bayes_risk: one_step.prior_bayes_risk,
        expected_total_cost: plan.expected_total_cost,
        selected_action_index: plan.selected_action_index,
        first_action_values,
        plan,
    })
}

fn solve_node(
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    decision_config: DecisionDiagnosticConfig,
    planning_config: SequentialDiagnosticConfig,
    posterior: &[f64],
    remaining_steps: usize,
    remaining_actions: u64,
    remaining_cumulative_safety_risk: Option<f64>,
    expansions: &mut usize,
) -> Result<
    (
        SequentialDiagnosticPlanNode,
        Vec<SequentialDiagnosticActionValue>,
    ),
    SequentialDiagnosticError,
> {
    if *expansions >= planning_config.max_policy_expansions {
        return Err(SequentialDiagnosticError::ExpansionBudgetExceeded {
            budget: planning_config.max_policy_expansions,
        });
    }
    *expansions += 1;
    let decision_count = hypotheses[0].loss_by_final_decision.len();
    let stop_risk = bayes_risk(posterior, hypotheses, decision_count)?;
    let mut best = stop_node(stop_risk);
    let mut first_action_values = Vec::with_capacity(actions.len());

    if remaining_steps == 0 || remaining_actions == 0 {
        return Ok((best, first_action_values));
    }

    for (action_index, action) in actions.iter().enumerate() {
        let action_bit = 1_u64 << action_index;
        if remaining_actions & action_bit == 0 {
            continue;
        }
        let eligible = action.safety_risk <= decision_config.maximum_safety_risk
            && remaining_cumulative_safety_risk.is_none_or(|remaining| {
                action.safety_risk <= remaining + RISK_TOLERANCE
            });
        if !eligible {
            first_action_values.push(SequentialDiagnosticActionValue {
                action_index,
                action_id: action.id.clone(),
                eligible: false,
                expected_total_cost: None,
                expected_terminal_bayes_risk: None,
                expected_observation_cost: None,
                expected_safety_penalty: None,
            });
            continue;
        }

        let candidate = plan_after_action(
            hypotheses,
            actions,
            decision_config,
            planning_config,
            posterior,
            action_index,
            remaining_steps - 1,
            remaining_actions & !action_bit,
            remaining_cumulative_safety_risk,
            expansions,
        )?;

        first_action_values.push(SequentialDiagnosticActionValue {
            action_index,
            action_id: action.id.clone(),
            eligible: true,
            expected_total_cost: Some(candidate.expected_total_cost),
            expected_terminal_bayes_risk: Some(candidate.expected_terminal_bayes_risk),
            expected_observation_cost: Some(candidate.expected_observation_cost),
            expected_safety_penalty: Some(candidate.expected_safety_penalty),
        });

        // Strict comparison means stopping wins exact ties. Earlier action
        // indices win ties between probes because iteration order is stable.
        if candidate.expected_total_cost + RISK_TOLERANCE < best.expected_total_cost {
            best = candidate;
        }
    }

    Ok((best, first_action_values))
}

fn plan_after_action(
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    decision_config: DecisionDiagnosticConfig,
    planning_config: SequentialDiagnosticConfig,
    posterior: &[f64],
    action_index: usize,
    remaining_steps: usize,
    remaining_actions: u64,
    remaining_cumulative_safety_risk: Option<f64>,
    expansions: &mut usize,
) -> Result<SequentialDiagnosticPlanNode, SequentialDiagnosticError> {
    let action = &actions[action_index];
    let outcome_count = hypotheses[0].outcome_probabilities_by_action[action_index].len();
    let mut branches = Vec::with_capacity(outcome_count);
    let mut expected_total_cost = decision_config.observation_cost_weight * action.observation_cost
        + decision_config.safety_risk_weight * action.safety_risk;
    let mut expected_terminal_bayes_risk = 0.0;
    let mut expected_observation_cost = action.observation_cost;
    let mut expected_safety_penalty = decision_config.safety_risk_weight * action.safety_risk;
    let mut predictive_mass = 0.0;
    let remaining_after_action = remaining_cumulative_safety_risk.map(|remaining| {
        (remaining - action.safety_risk).max(0.0)
    });

    for outcome_index in 0..outcome_count {
        let outcome_probability = hypotheses
            .iter()
            .zip(posterior)
            .map(|(hypothesis, probability)| {
                probability
                    * hypothesis.outcome_probabilities_by_action[action_index][outcome_index]
            })
            .sum::<f64>();

        if !outcome_probability.is_finite() || outcome_probability < 0.0 {
            return Err(SequentialDiagnosticError::InvalidPredictiveOutcomeMass { action_index });
        }
        if outcome_probability == 0.0 {
            continue;
        }

        predictive_mass += outcome_probability;
        let next_posterior = hypotheses
            .iter()
            .zip(posterior)
            .map(|(hypothesis, probability)| {
                probability
                    * hypothesis.outcome_probabilities_by_action[action_index][outcome_index]
                    / outcome_probability
            })
            .collect::<Vec<_>>();

        let (continuation, _) = solve_node(
            hypotheses,
            actions,
            decision_config,
            planning_config,
            &next_posterior,
            remaining_steps,
            remaining_actions,
            remaining_after_action,
            expansions,
        )?;

        expected_total_cost += outcome_probability * continuation.expected_total_cost;
        expected_terminal_bayes_risk +=
            outcome_probability * continuation.expected_terminal_bayes_risk;
        expected_observation_cost += outcome_probability * continuation.expected_observation_cost;
        expected_safety_penalty += outcome_probability * continuation.expected_safety_penalty;

        branches.push(SequentialDiagnosticBranch {
            outcome_index,
            outcome_probability,
            posterior_model_probabilities: next_posterior,
            continuation: Box::new(continuation),
        });
    }

    if !predictive_mass.is_finite() || (predictive_mass - 1.0).abs() > 1e-8 {
        return Err(SequentialDiagnosticError::InvalidPredictiveOutcomeMass { action_index });
    }
    if !expected_total_cost.is_finite()
        || !expected_terminal_bayes_risk.is_finite()
        || !expected_observation_cost.is_finite()
        || !expected_safety_penalty.is_finite()
    {
        return Err(SequentialDiagnosticError::NonFiniteRisk);
    }

    Ok(SequentialDiagnosticPlanNode {
        selected_action_index: Some(action_index),
        selected_action_id: Some(action.id.clone()),
        expected_total_cost,
        expected_terminal_bayes_risk,
        expected_observation_cost,
        expected_safety_penalty,
        branches,
    })
}

fn stop_node(bayes_risk: f64) -> SequentialDiagnosticPlanNode {
    SequentialDiagnosticPlanNode {
        selected_action_index: None,
        selected_action_id: None,
        expected_total_cost: bayes_risk,
        expected_terminal_bayes_risk: bayes_risk,
        expected_observation_cost: 0.0,
        expected_safety_penalty: 0.0,
        branches: Vec::new(),
    }
}

fn bayes_risk(
    posterior: &[f64],
    hypotheses: &[DecisionHypothesis],
    decision_count: usize,
) -> Result<f64, SequentialDiagnosticError> {
    let risk = (0..decision_count)
        .map(|decision_index| {
            hypotheses
                .iter()
                .zip(posterior)
                .map(|(hypothesis, probability)| {
                    probability * hypothesis.loss_by_final_decision[decision_index]
                })
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min);

    if risk.is_finite() {
        Ok(risk)
    } else {
        Err(SequentialDiagnosticError::NonFiniteRisk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xor_complementary_probes() -> (Vec<DecisionHypothesis>, Vec<DiagnosticAction>) {
        let hypotheses = (0..4)
            .map(|index| {
                let first_bit = index / 2;
                let second_bit = index % 2;
                let xor = first_bit ^ second_bit;
                DecisionHypothesis {
                    id: format!("fault-bits-{first_bit}-{second_bit}"),
                    prior_probability: 0.25,
                    outcome_probabilities_by_action: vec![
                        if first_bit == 0 {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        },
                        if second_bit == 0 {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        },
                    ],
                    loss_by_final_decision: if xor == 0 {
                        vec![0.0, 1.0]
                    } else {
                        vec![1.0, 0.0]
                    },
                }
            })
            .collect();

        let actions = vec![
            DiagnosticAction {
                id: "reveal-first-fault-bit".into(),
                observation_cost: 0.0,
                safety_risk: 0.0,
            },
            DiagnosticAction {
                id: "reveal-second-fault-bit".into(),
                observation_cost: 0.0,
                safety_risk: 0.0,
            },
        ];
        (hypotheses, actions)
    }

    fn decision_config() -> DecisionDiagnosticConfig {
        DecisionDiagnosticConfig {
            observation_cost_weight: 1.0,
            safety_risk_weight: 1.0,
            maximum_safety_risk: 1.0,
        }
    }

    #[test]
    fn two_complementary_probes_have_zero_myopic_evsi_but_positive_sequential_value() {
        let (hypotheses, actions) = xor_complementary_probes();
        let myopic =
            evaluate_decision_focused_diagnostics(&hypotheses, &actions, decision_config())
                .expect("one-step diagnostic evaluation");
        assert_eq!(myopic.selected_action_index, None);
        assert!(
            myopic
                .actions
                .iter()
                .all(|action| action.expected_value_of_sample_information.abs() < 1e-12)
        );

        let sequential = evaluate_sequential_decision_diagnostics(
            &hypotheses,
            &actions,
            decision_config(),
            SequentialDiagnosticConfig {
                horizon: 2,
                max_policy_expansions: 64,
            },
        )
        .expect("two-step diagnostic policy");
        assert!(sequential.selected_action_index.is_some());
        assert!(sequential.expected_total_cost < 1e-12);
        assert!(sequential.plan.expected_terminal_bayes_risk < 1e-12);
        assert_eq!(sequential.plan.branches.len(), 2);
        for branch in &sequential.plan.branches {
            let continuation = &branch.continuation;
            assert!(continuation.selected_action_index.is_some());
            assert_ne!(
                continuation.selected_action_index,
                sequential.selected_action_index
            );
            assert_eq!(continuation.branches.len(), 2);
            for terminal in &continuation.branches {
                assert_eq!(terminal.continuation.selected_action_index, None);
            }
        }
    }

    #[test]
    fn cumulative_safety_budget_blocks_individually_safe_but_jointly_unsafe_probes() {
        let (hypotheses, mut actions) = xor_complementary_probes();
        for action in &mut actions {
            action.safety_risk = 0.35;
        }
        let decision_config = DecisionDiagnosticConfig {
            safety_risk_weight: 0.1,
            maximum_safety_risk: 0.5,
            ..decision_config()
        };
        let result = evaluate_sequential_decision_diagnostics_with_safety_budget(
            &hypotheses,
            &actions,
            decision_config,
            SequentialDiagnosticConfig {
                horizon: 2,
                max_policy_expansions: 64,
            },
            SequentialDiagnosticSafetyBudget {
                maximum_cumulative_risk: 0.6,
                risk_semantics_digest: "a".repeat(64),
            },
        )
        .expect("valid pathwise safety budget");

        // Each probe is individually below 0.5, but two probes would consume
        // 0.7 of a 0.6 path budget. One probe has no decision value in this XOR
        // fixture, so the optimal policy is to stop without probing.
        assert_eq!(result.evaluation.selected_action_index, None);
        assert!((result.evaluation.expected_total_cost - 0.5).abs() < 1e-12);
        assert!(result.evaluation.first_action_values.iter().all(|value| value.eligible));
    }

    #[test]
    fn cumulative_safety_budget_allows_sequence_at_the_declared_path_bound() {
        let (hypotheses, mut actions) = xor_complementary_probes();
        for action in &mut actions {
            action.safety_risk = 0.35;
        }
        let decision_config = DecisionDiagnosticConfig {
            safety_risk_weight: 0.1,
            maximum_safety_risk: 0.5,
            ..decision_config()
        };
        let result = evaluate_sequential_decision_diagnostics_with_safety_budget(
            &hypotheses,
            &actions,
            decision_config,
            SequentialDiagnosticConfig {
                horizon: 2,
                max_policy_expansions: 64,
            },
            SequentialDiagnosticSafetyBudget {
                maximum_cumulative_risk: 0.7,
                risk_semantics_digest: "b".repeat(64),
            },
        )
        .expect("sequence exactly consumes the declared path budget");

        assert!(result.evaluation.selected_action_index.is_some());
        assert!(result.evaluation.plan.expected_terminal_bayes_risk < 1e-12);
        assert!((result.evaluation.plan.expected_safety_penalty - 0.07).abs() < 1e-10);
        assert!(result.evaluation.expected_total_cost < 0.1);
        assert_eq!(result.safety_budget.maximum_cumulative_risk, 0.7);
    }

    #[test]
    fn cumulative_safety_budget_digest_binds_limit_and_semantics() {
        let budget = SequentialDiagnosticSafetyBudget {
            maximum_cumulative_risk: 0.6,
            risk_semantics_digest: "d".repeat(64),
        };
        let original =
            compute_sequential_diagnostic_safety_budget_digest(&budget).expect("digest");
        assert_eq!(original.len(), 64);
        assert_eq!(
            original,
            compute_sequential_diagnostic_safety_budget_digest(&budget).expect("stable digest")
        );

        let changed_limit = SequentialDiagnosticSafetyBudget {
            maximum_cumulative_risk: 0.7,
            ..budget.clone()
        };
        let changed_semantics = SequentialDiagnosticSafetyBudget {
            risk_semantics_digest: "e".repeat(64),
            ..budget
        };
        assert_ne!(
            original,
            compute_sequential_diagnostic_safety_budget_digest(&changed_limit)
                .expect("changed limit digest")
        );
        assert_ne!(
            original,
            compute_sequential_diagnostic_safety_budget_digest(&changed_semantics)
                .expect("changed semantics digest")
        );
    }

    #[test]
    fn malformed_cumulative_safety_budget_fails_closed() {
        let (hypotheses, actions) = xor_complementary_probes();
        let invalid_digest = SequentialDiagnosticSafetyBudget {
            maximum_cumulative_risk: 0.5,
            risk_semantics_digest: "not-a-digest".into(),
        };
        assert!(matches!(
            evaluate_sequential_decision_diagnostics_with_safety_budget(
                &hypotheses,
                &actions,
                decision_config(),
                SequentialDiagnosticConfig::default(),
                invalid_digest
            ),
            Err(SequentialDiagnosticError::InvalidRiskSemanticsDigest)
        ));

        let invalid_bound = SequentialDiagnosticSafetyBudget {
            maximum_cumulative_risk: f64::NAN,
            risk_semantics_digest: "c".repeat(64),
        };
        assert!(matches!(
            evaluate_sequential_decision_diagnostics_with_safety_budget(
                &hypotheses,
                &actions,
                decision_config(),
                SequentialDiagnosticConfig::default(),
                invalid_bound
            ),
            Err(SequentialDiagnosticError::InvalidCumulativeSafetyRiskBudget)
        ));
    }

    #[test]
    fn horizon_one_matches_the_myopic_evsi_selector() {
        let (hypotheses, mut actions) = xor_complementary_probes();
        actions[0].observation_cost = 0.0;
        actions[1].observation_cost = 0.25;
        let myopic =
            evaluate_decision_focused_diagnostics(&hypotheses, &actions, decision_config())
                .expect("one-step evaluation");
        let sequential = evaluate_sequential_decision_diagnostics(
            &hypotheses,
            &actions,
            decision_config(),
            SequentialDiagnosticConfig {
                horizon: 1,
                max_policy_expansions: 16,
            },
        )
        .expect("horizon-one evaluation");
        assert_eq!(
            sequential.selected_action_index,
            myopic.selected_action_index
        );
        assert_eq!(sequential.selected_action_index, None);
        assert!((sequential.expected_total_cost - myopic.no_probe_score).abs() < 1e-12);
    }

    #[test]
    fn hard_safety_limit_applies_at_every_node() {
        let (hypotheses, mut actions) = xor_complementary_probes();
        actions[1].safety_risk = 0.9;
        let config = DecisionDiagnosticConfig {
            maximum_safety_risk: 0.2,
            ..decision_config()
        };
        let result = evaluate_sequential_decision_diagnostics(
            &hypotheses,
            &actions,
            config,
            SequentialDiagnosticConfig {
                horizon: 2,
                max_policy_expansions: 16,
            },
        )
        .expect("safe bounded plan");
        assert_eq!(result.selected_action_index, None);
        assert!(!result.first_action_values[1].eligible);
        assert!(result.plan.branches.is_empty());
    }

    #[test]
    fn planner_declines_a_sequence_whose_cost_exceeds_terminal_decision_value() {
        let (hypotheses, mut actions) = xor_complementary_probes();
        actions[0].observation_cost = 0.3;
        actions[1].observation_cost = 0.3;
        let result = evaluate_sequential_decision_diagnostics(
            &hypotheses,
            &actions,
            decision_config(),
            SequentialDiagnosticConfig {
                horizon: 2,
                max_policy_expansions: 64,
            },
        )
        .expect("cost-aware bounded plan");
        assert_eq!(result.selected_action_index, None);
        assert!((result.expected_total_cost - 0.5).abs() < 1e-12);
    }

    #[test]
    fn expansion_budget_exhaustion_fails_closed_without_a_partial_plan() {
        let (hypotheses, actions) = xor_complementary_probes();
        let result = evaluate_sequential_decision_diagnostics(
            &hypotheses,
            &actions,
            decision_config(),
            SequentialDiagnosticConfig {
                horizon: 2,
                max_policy_expansions: 1,
            },
        );
        assert!(matches!(
            result,
            Err(SequentialDiagnosticError::ExpansionBudgetExceeded { budget: 1 })
        ));
    }

    #[test]
    fn action_is_never_reused_on_a_single_plan_path() {
        let (hypotheses, actions) = xor_complementary_probes();
        let result = evaluate_sequential_decision_diagnostics(
            &hypotheses,
            &actions,
            decision_config(),
            SequentialDiagnosticConfig {
                horizon: 2,
                max_policy_expansions: 64,
            },
        )
        .expect("bounded plan");
        for branch in &result.plan.branches {
            assert_ne!(
                branch.continuation.selected_action_index,
                result.plan.selected_action_index
            );
        }
    }

    #[test]
    fn rejects_horizons_larger_than_the_distinct_action_set() {
        let (hypotheses, actions) = xor_complementary_probes();
        assert!(matches!(
            evaluate_sequential_decision_diagnostics(
                &hypotheses,
                &actions,
                decision_config(),
                SequentialDiagnosticConfig {
                    horizon: 3,
                    max_policy_expansions: 64,
                }
            ),
            Err(SequentialDiagnosticError::HorizonExceedsActionCount {
                horizon: 3,
                action_count: 2
            })
        ));
    }
}
