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

use super::causal_diagnostic_semantics::{
    CausalDiagnosticActionDescriptor, CausalDiagnosticActionError,
    CausalDiagnosticDecisionProblemReceipt, validate_causal_diagnostic_decision_problem,
};
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

/// Replayable identity binding for a budgeted adaptive diagnostic policy.
///
/// The receipt binds the causal decision-problem receipt, risk-budget semantics,
/// planner limits, complete selected policy tree, and evaluation summary. It is
/// an integrity/identity record, not an authenticated signature or proof that
/// externally supplied risk calibration and causal-identification claims are true.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialDiagnosticPolicyReceipt {
    pub schema_version: u32,
    pub decision_problem_digest: String,
    pub safety_budget_digest: String,
    pub risk_semantics_digest: String,
    pub planning_config_digest: String,
    pub policy_tree_digest: String,
    pub evaluation_digest: String,
    /// Maximum sum of conditional adverse-event probability upper bounds over
    /// any terminal path in the selected policy tree.
    pub maximum_path_risk_upper_bound: f64,
    /// Number of terminal paths represented by positive-probability model outcomes.
    pub modeled_terminal_path_count: usize,
    pub receipt_digest: String,
}

/// Result of budgeted sequential planning with a causal decision-problem receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetedSequentialDiagnosticEvaluationWithReceipt {
    pub budgeted_evaluation: BudgetedSequentialDiagnosticEvaluation,
    pub decision_problem_receipt: CausalDiagnosticDecisionProblemReceipt,
    pub policy_receipt: SequentialDiagnosticPolicyReceipt,
}

/// Result of structurally verifying a budgeted sequential diagnostic receipt.
///
/// This verifies exact decision-problem identity, branch support and Bayes
/// updates, recursive objective accounting, and the cumulative path-risk bound.
/// It deliberately does not claim that the selected policy is globally optimal:
/// that requires replay against a separately implemented optimizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialDiagnosticPolicyVerificationReport {
    pub schema_version: u32,
    pub receipt_digest: String,
    pub decision_problem_digest: String,
    pub policy_tree_digest: String,
    pub maximum_path_risk_upper_bound: f64,
    pub modeled_terminal_path_count: usize,
    pub branch_probabilities_verified: bool,
    pub posterior_updates_verified: bool,
    pub objective_accounting_verified: bool,
    pub pathwise_risk_budget_verified: bool,
    /// Exact replay against the canonical planner implementation used to create the receipt.
    pub canonical_solver_replay_match: bool,
    /// Always false for this verifier: an independent optimizer is still required.
    pub optimality_independently_reverified: bool,
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
    #[error(transparent)]
    CausalDiagnosticAction(#[from] CausalDiagnosticActionError),
    #[error("selected diagnostic policy tree is malformed or inconsistent with its action set")]
    InvalidPolicyTree,
    #[error("selected diagnostic policy path risk bound {bound} exceeds budget {budget}")]
    PolicyExceedsSafetyBudget { bound: f64, budget: f64 },
    #[error("policy receipt does not match the supplied problem, budget, evaluation, or policy tree")]
    PolicyReceiptMismatch,
    #[error("diagnostic evaluation summary is inconsistent with the selected policy tree")]
    InvalidEvaluationSummary,
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

/// Evaluate a budgeted sequential policy and bind it to the exact causal decision problem.
///
/// The decision-problem receipt covers typed action semantics, action-likelihood
/// tables, hypothesis priors, downstream losses, decision-policy configuration,
/// and task specification. The policy receipt additionally covers the cumulative
/// budget, planner limits, selected policy tree, and recomputed worst-path risk.
///
/// This does not authenticate the source of the causal/risk claims. Callers must
/// still independently authenticate the receipts and calibrations before any
/// physical action is considered.
pub fn evaluate_sequential_decision_diagnostics_with_policy_receipt(
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    causal_action_descriptors: &[CausalDiagnosticActionDescriptor],
    decision_config: DecisionDiagnosticConfig,
    planning_config: SequentialDiagnosticConfig,
    safety_budget: SequentialDiagnosticSafetyBudget,
    task_specification_digest: &str,
) -> Result<BudgetedSequentialDiagnosticEvaluationWithReceipt, SequentialDiagnosticError> {
    let decision_problem_receipt = validate_causal_diagnostic_decision_problem(
        actions,
        causal_action_descriptors,
        hypotheses,
        decision_config,
        task_specification_digest,
    )?;
    let budgeted_evaluation =
        evaluate_sequential_decision_diagnostics_with_safety_budget(
            hypotheses,
            actions,
            decision_config,
            planning_config,
            safety_budget,
        )?;
    let policy_receipt = compute_sequential_diagnostic_policy_receipt(
        hypotheses,
        actions,
        planning_config,
        decision_config,
        &decision_problem_receipt,
        &budgeted_evaluation,
    )?;

    Ok(BudgetedSequentialDiagnosticEvaluationWithReceipt {
        budgeted_evaluation,
        decision_problem_receipt,
        policy_receipt,
    })
}

/// Compute a canonical receipt over a freshly evaluated budgeted policy.
///
/// This remains private so callers cannot turn an arbitrary deserialized policy
/// tree into a trusted evaluation receipt. The public combined API above performs
/// causal validation and planning before receipt construction.
fn compute_sequential_diagnostic_policy_receipt(
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    planning_config: SequentialDiagnosticConfig,
    decision_config: DecisionDiagnosticConfig,
    decision_problem_receipt: &CausalDiagnosticDecisionProblemReceipt,
    budgeted_evaluation: &BudgetedSequentialDiagnosticEvaluation,
) -> Result<SequentialDiagnosticPolicyReceipt, SequentialDiagnosticError> {
    let budget_digest =
        compute_sequential_diagnostic_safety_budget_digest(&budgeted_evaluation.safety_budget)?;
    if budget_digest != budgeted_evaluation.safety_budget_digest
        || budgeted_evaluation.evaluation.plan.selected_action_index
            != budgeted_evaluation.evaluation.selected_action_index
    {
        return Err(SequentialDiagnosticError::InvalidPolicyTree);
    }

    let planning_config_digest = compute_planning_config_digest(planning_config);
    let (policy_tree_digest, maximum_path_risk_upper_bound, modeled_terminal_path_count) =
        compute_policy_tree_digest(
            &budgeted_evaluation.evaluation.plan,
            hypotheses,
            actions,
            decision_config,
            budgeted_evaluation.safety_budget.maximum_cumulative_risk,
            planning_config.horizon,
        )?;
    if maximum_path_risk_upper_bound
        > budgeted_evaluation.safety_budget.maximum_cumulative_risk + RISK_TOLERANCE
    {
        return Err(SequentialDiagnosticError::PolicyExceedsSafetyBudget {
            bound: maximum_path_risk_upper_bound,
            budget: budgeted_evaluation.safety_budget.maximum_cumulative_risk,
        });
    }
    let evaluation_digest = compute_policy_evaluation_digest(
        &budgeted_evaluation.evaluation,
        &policy_tree_digest,
    );

    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-budgeted-sequential-diagnostic-policy-receipt-v1\0");
    hash_receipt_field(
        &mut hasher,
        "decision_problem_digest",
        decision_problem_receipt.decision_problem_digest.as_bytes(),
    );
    hash_receipt_field(
        &mut hasher,
        "safety_budget_digest",
        budget_digest.as_bytes(),
    );
    hash_receipt_field(
        &mut hasher,
        "risk_semantics_digest",
        budgeted_evaluation.safety_budget.risk_semantics_digest.as_bytes(),
    );
    hash_receipt_field(
        &mut hasher,
        "planning_config_digest",
        planning_config_digest.as_bytes(),
    );
    hash_receipt_field(
        &mut hasher,
        "policy_tree_digest",
        policy_tree_digest.as_bytes(),
    );
    hash_receipt_field(
        &mut hasher,
        "evaluation_digest",
        evaluation_digest.as_bytes(),
    );
    hasher.update(maximum_path_risk_upper_bound.to_bits().to_le_bytes());
    hasher.update((modeled_terminal_path_count as u64).to_le_bytes());

    Ok(SequentialDiagnosticPolicyReceipt {
        schema_version: 1,
        decision_problem_digest: decision_problem_receipt.decision_problem_digest.clone(),
        safety_budget_digest: budget_digest,
        risk_semantics_digest: budgeted_evaluation.safety_budget.risk_semantics_digest.clone(),
        planning_config_digest,
        policy_tree_digest,
        evaluation_digest,
        maximum_path_risk_upper_bound,
        modeled_terminal_path_count,
        receipt_digest: format!("{:x}", hasher.finalize()),
    })
}

fn compute_planning_config_digest(config: SequentialDiagnosticConfig) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-sequential-diagnostic-planning-config-v1\0");
    hasher.update((config.horizon as u64).to_le_bytes());
    hasher.update((config.max_policy_expansions as u64).to_le_bytes());
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, Copy)]
struct PolicyAuditSummary {
    maximum_path_risk: f64,
    terminal_path_count: usize,
}

fn compute_policy_tree_digest(
    plan: &SequentialDiagnosticPlanNode,
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    decision_config: DecisionDiagnosticConfig,
    maximum_cumulative_risk: f64,
    planning_horizon: usize,
) -> Result<(String, f64, usize), SequentialDiagnosticError> {
    if hypotheses.is_empty() || actions.is_empty() || actions.len() > MAX_DIAGNOSTIC_ACTIONS {
        return Err(SequentialDiagnosticError::InvalidPolicyTree);
    }
    let posterior: Vec<f64> = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .collect();
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-sequential-diagnostic-policy-tree-v1\0");
    let audit = audit_policy_node(
        &mut hasher,
        plan,
        &posterior,
        hypotheses,
        actions,
        decision_config,
        maximum_cumulative_risk,
        planning_horizon,
        0.0,
        0,
    )?;
    Ok((
        format!("{:x}", hasher.finalize()),
        audit.maximum_path_risk,
        audit.terminal_path_count,
    ))
}

#[allow(clippy::too_many_arguments)]
fn audit_policy_node(
    hasher: &mut Sha256,
    node: &SequentialDiagnosticPlanNode,
    posterior: &[f64],
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    decision_config: DecisionDiagnosticConfig,
    maximum_cumulative_risk: f64,
    remaining_steps: usize,
    risk_before_node: f64,
    used_actions: u64,
) -> Result<PolicyAuditSummary, SequentialDiagnosticError> {
    hasher.update(b"node-v1\0");
    let hypothesis_count = hypotheses.len();
    if posterior.len() != hypothesis_count
        || posterior.iter().any(|probability| {
            !probability.is_finite() || !(0.0..=1.0).contains(probability)
        })
        || (posterior.iter().sum::<f64>() - 1.0).abs() > 1e-8
    {
        return Err(SequentialDiagnosticError::InvalidPolicyTree);
    }

    for value in [
        node.expected_total_cost,
        node.expected_terminal_bayes_risk,
        node.expected_observation_cost,
        node.expected_safety_penalty,
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(SequentialDiagnosticError::InvalidPolicyTree);
        }
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher.update((node.branches.len() as u64).to_le_bytes());

    let stop_risk = bayes_risk(
        posterior,
        hypotheses,
        hypotheses[0].loss_by_final_decision.len(),
    )?;
    let Some(action_index) = node.selected_action_index else {
        if node.selected_action_id.is_some()
            || !node.branches.is_empty()
            || !policy_values_match(node.expected_total_cost, stop_risk)
            || !policy_values_match(node.expected_terminal_bayes_risk, stop_risk)
            || !policy_values_match(node.expected_observation_cost, 0.0)
            || !policy_values_match(node.expected_safety_penalty, 0.0)
        {
            return Err(SequentialDiagnosticError::InvalidPolicyTree);
        }
        hasher.update([0]);
        return Ok(PolicyAuditSummary {
            maximum_path_risk: risk_before_node,
            terminal_path_count: 1,
        });
    };

    if remaining_steps == 0 {
        return Err(SequentialDiagnosticError::InvalidPolicyTree);
    }
    let action = actions
        .get(action_index)
        .ok_or(SequentialDiagnosticError::InvalidPolicyTree)?;
    if node.selected_action_id.as_deref() != Some(action.id.as_str())
        || action_index >= MAX_DIAGNOSTIC_ACTIONS
        || used_actions & (1_u64 << action_index) != 0
        || !action.safety_risk.is_finite()
        || !(0.0..=1.0).contains(&action.safety_risk)
        || action.safety_risk > decision_config.maximum_safety_risk
    {
        return Err(SequentialDiagnosticError::InvalidPolicyTree);
    }

    let risk_after_node = risk_before_node + action.safety_risk;
    if !risk_after_node.is_finite()
        || risk_after_node > maximum_cumulative_risk + RISK_TOLERANCE
    {
        return Err(SequentialDiagnosticError::PolicyExceedsSafetyBudget {
            bound: risk_after_node,
            budget: maximum_cumulative_risk,
        });
    }

    hasher.update([1]);
    hasher.update((action_index as u64).to_le_bytes());
    hash_receipt_field(hasher, "action_id", action.id.as_bytes());
    hasher.update(action.observation_cost.to_bits().to_le_bytes());
    hasher.update(action.safety_risk.to_bits().to_le_bytes());

    let outcome_count = hypotheses[0]
        .outcome_probabilities_by_action
        .get(action_index)
        .ok_or(SequentialDiagnosticError::InvalidPolicyTree)?
        .len();
    if outcome_count == 0 {
        return Err(SequentialDiagnosticError::InvalidPolicyTree);
    }
    let mut branch_cursor = 0_usize;
    let mut predictive_mass = 0.0_f64;
    let mut expected_total_cost = decision_config.observation_cost_weight
        * action.observation_cost
        + decision_config.safety_risk_weight * action.safety_risk;
    let mut expected_terminal_bayes_risk = 0.0_f64;
    let mut expected_observation_cost = action.observation_cost;
    let mut expected_safety_penalty =
        decision_config.safety_risk_weight * action.safety_risk;
    let mut maximum_path_risk = risk_after_node;
    let mut terminal_path_count = 0_usize;
    let next_used_actions = used_actions | (1_u64 << action_index);

    for outcome_index in 0..outcome_count {
        let outcome_probability = hypotheses
            .iter()
            .zip(posterior)
            .map(|(hypothesis, prior)| {
                prior
                    * hypothesis.outcome_probabilities_by_action[action_index][outcome_index]
            })
            .sum::<f64>();
        if !outcome_probability.is_finite()
            || outcome_probability < 0.0
            || outcome_probability > 1.0 + 1e-8
        {
            return Err(SequentialDiagnosticError::InvalidPolicyTree);
        }
        predictive_mass += outcome_probability;
        if outcome_probability == 0.0 {
            continue;
        }

        let branch = node
            .branches
            .get(branch_cursor)
            .ok_or(SequentialDiagnosticError::InvalidPolicyTree)?;
        if branch.outcome_index != outcome_index
            || !branch.outcome_probability.is_finite()
            || !policy_values_match(branch.outcome_probability, outcome_probability)
            || branch.posterior_model_probabilities.len() != hypothesis_count
        {
            return Err(SequentialDiagnosticError::InvalidPolicyTree);
        }

        let expected_posterior: Vec<f64> = hypotheses
            .iter()
            .zip(posterior)
            .map(|(hypothesis, prior)| {
                prior
                    * hypothesis.outcome_probabilities_by_action[action_index][outcome_index]
                    / outcome_probability
            })
            .collect();
        if expected_posterior
            .iter()
            .zip(&branch.posterior_model_probabilities)
            .any(|(expected, actual)| !policy_values_match(*expected, *actual))
        {
            return Err(SequentialDiagnosticError::InvalidPolicyTree);
        }

        hasher.update((branch.outcome_index as u64).to_le_bytes());
        hasher.update(branch.outcome_probability.to_bits().to_le_bytes());
        hasher.update((branch.posterior_model_probabilities.len() as u64).to_le_bytes());
        for probability in &branch.posterior_model_probabilities {
            hasher.update(probability.to_bits().to_le_bytes());
        }

        let child = audit_policy_node(
            hasher,
            &branch.continuation,
            &expected_posterior,
            hypotheses,
            actions,
            decision_config,
            maximum_cumulative_risk,
            remaining_steps - 1,
            risk_after_node,
            next_used_actions,
        )?;
        expected_total_cost += outcome_probability
            * branch.continuation.expected_total_cost;
        expected_terminal_bayes_risk += outcome_probability
            * branch.continuation.expected_terminal_bayes_risk;
        expected_observation_cost += outcome_probability
            * branch.continuation.expected_observation_cost;
        expected_safety_penalty += outcome_probability
            * branch.continuation.expected_safety_penalty;
        maximum_path_risk = maximum_path_risk.max(child.maximum_path_risk);
        terminal_path_count = terminal_path_count
            .checked_add(child.terminal_path_count)
            .ok_or(SequentialDiagnosticError::InvalidPolicyTree)?;
        branch_cursor += 1;
    }

    if !policy_values_match(predictive_mass, 1.0)
        || branch_cursor != node.branches.len()
        || !policy_values_match(node.expected_total_cost, expected_total_cost)
        || !policy_values_match(
            node.expected_terminal_bayes_risk,
            expected_terminal_bayes_risk,
        )
        || !policy_values_match(node.expected_observation_cost, expected_observation_cost)
        || !policy_values_match(node.expected_safety_penalty, expected_safety_penalty)
        || terminal_path_count == 0
    {
        return Err(SequentialDiagnosticError::InvalidPolicyTree);
    }

    Ok(PolicyAuditSummary {
        maximum_path_risk,
        terminal_path_count,
    })
}

fn policy_values_match(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs()
            <= 1e-8 * actual.abs().max(expected.abs()).max(1.0)
}

fn compute_policy_evaluation_digest(
    evaluation: &SequentialDiagnosticEvaluation,
    policy_tree_digest: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-sequential-diagnostic-evaluation-v1\0");
    hasher.update((evaluation.horizon as u64).to_le_bytes());
    hasher.update((evaluation.expansions as u64).to_le_bytes());
    hasher.update(evaluation.no_probe_bayes_risk.to_bits().to_le_bytes());
    hasher.update(evaluation.expected_total_cost.to_bits().to_le_bytes());
    match evaluation.selected_action_index {
        Some(index) => {
            hasher.update([1]);
            hasher.update((index as u64).to_le_bytes());
        }
        None => hasher.update([0]),
    }
    hasher.update((evaluation.first_action_values.len() as u64).to_le_bytes());
    for action in &evaluation.first_action_values {
        hasher.update((action.action_index as u64).to_le_bytes());
        hash_receipt_field(&mut hasher, "action_id", action.action_id.as_bytes());
        hasher.update([u8::from(action.eligible)]);
        for value in [
            action.expected_total_cost,
            action.expected_terminal_bayes_risk,
            action.expected_observation_cost,
            action.expected_safety_penalty,
        ] {
            match value {
                Some(value) => {
                    hasher.update([1]);
                    hasher.update(value.to_bits().to_le_bytes());
                }
                None => hasher.update([0]),
            }
        }
    }
    hash_receipt_field(&mut hasher, "policy_tree_digest", policy_tree_digest.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn hash_receipt_field(hasher: &mut Sha256, name: &str, value: &[u8]) {
    hasher.update((name.len() as u64).to_le_bytes());
    hasher.update(name.as_bytes());
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

/// Compute a canonical, domain-separated SHA-256 digest for a path safety budget.
///
/// The digest binds the exact IEEE-754 bit pattern of the maximum cumulative
/// risk together with the external risk-semantics/calibration digest.
/// Verify a serialized policy receipt against its original decision inputs.
///
/// This independently recomputes causal problem identity, every reachable
/// outcome probability and Bayes posterior, recursive objective accounting,
/// and the maximum cumulative path-risk upper bound. It does not prove that the
/// selected policy is globally optimal; use a separately implemented optimizer
/// for that claim. It also does not authenticate external calibration sources.
pub fn verify_sequential_diagnostic_policy_receipt(
    hypotheses: &[DecisionHypothesis],
    actions: &[DiagnosticAction],
    causal_action_descriptors: &[CausalDiagnosticActionDescriptor],
    decision_config: DecisionDiagnosticConfig,
    planning_config: SequentialDiagnosticConfig,
    expected_safety_budget: &SequentialDiagnosticSafetyBudget,
    task_specification_digest: &str,
    supplied: &BudgetedSequentialDiagnosticEvaluationWithReceipt,
) -> Result<SequentialDiagnosticPolicyVerificationReport, SequentialDiagnosticError> {
    let expected_problem_receipt = validate_causal_diagnostic_decision_problem(
        actions,
        causal_action_descriptors,
        hypotheses,
        decision_config,
        task_specification_digest,
    )?;
    if planning_config.horizon > actions.len()
        || planning_config.max_policy_expansions == 0
        || supplied.decision_problem_receipt != expected_problem_receipt
        || supplied.budgeted_evaluation.safety_budget != *expected_safety_budget
        || supplied.budgeted_evaluation.evaluation.horizon != planning_config.horizon
        || supplied.budgeted_evaluation.evaluation.expansions == 0
        || supplied.budgeted_evaluation.evaluation.expansions
            > planning_config.max_policy_expansions
    {
        return Err(SequentialDiagnosticError::PolicyReceiptMismatch);
    }

    let evaluation = &supplied.budgeted_evaluation.evaluation;
    let one_step = evaluate_decision_focused_diagnostics(
        hypotheses,
        actions,
        decision_config,
    )?;
    if evaluation.plan.selected_action_index != evaluation.selected_action_index
        || !policy_values_match(evaluation.expected_total_cost, evaluation.plan.expected_total_cost)
        || !policy_values_match(evaluation.no_probe_bayes_risk, one_step.prior_bayes_risk)
    {
        return Err(SequentialDiagnosticError::InvalidEvaluationSummary);
    }

    if planning_config.horizon == 0 {
        if !evaluation.first_action_values.is_empty() {
            return Err(SequentialDiagnosticError::InvalidEvaluationSummary);
        }
    } else {
        if evaluation.first_action_values.len() != actions.len() {
            return Err(SequentialDiagnosticError::InvalidEvaluationSummary);
        }
        for (index, (value, action)) in evaluation
            .first_action_values
            .iter()
            .zip(actions)
            .enumerate()
        {
            let expected_eligible = action.safety_risk
                <= decision_config.maximum_safety_risk
                && action.safety_risk
                    <= expected_safety_budget.maximum_cumulative_risk + RISK_TOLERANCE;
            if value.action_index != index
                || value.action_id != action.id
                || value.eligible != expected_eligible
            {
                return Err(SequentialDiagnosticError::InvalidEvaluationSummary);
            }
            let option_values = [
                value.expected_total_cost,
                value.expected_terminal_bayes_risk,
                value.expected_observation_cost,
                value.expected_safety_penalty,
            ];
            if (expected_eligible
                && option_values
                    .iter()
                    .any(|value| value.is_none_or(|x| !x.is_finite() || x < 0.0)))
                || (!expected_eligible && option_values.iter().any(|value| value.is_some()))
            {
                return Err(SequentialDiagnosticError::InvalidEvaluationSummary);
            }
        }
    }

    // Replay the exact bounded optimizer to validate root action-value
    // summaries and canonical policy selection. This is not an independent
    // optimizer; that stronger check belongs to an external verifier.
    let replayed_evaluation = evaluate_sequential_decision_diagnostics_with_safety_budget(
        hypotheses,
        actions,
        decision_config,
        planning_config,
        expected_safety_budget.clone(),
    )?;
    if replayed_evaluation.evaluation != supplied.budgeted_evaluation.evaluation {
        return Err(SequentialDiagnosticError::InvalidEvaluationSummary);
    }

    let expected_policy_receipt = compute_sequential_diagnostic_policy_receipt(
        hypotheses,
        actions,
        planning_config,
        decision_config,
        &expected_problem_receipt,
        &supplied.budgeted_evaluation,
    )?;
    if supplied.policy_receipt != expected_policy_receipt {
        return Err(SequentialDiagnosticError::PolicyReceiptMismatch);
    }

    Ok(SequentialDiagnosticPolicyVerificationReport {
        schema_version: 1,
        receipt_digest: expected_policy_receipt.receipt_digest,
        decision_problem_digest: expected_policy_receipt.decision_problem_digest,
        policy_tree_digest: expected_policy_receipt.policy_tree_digest,
        maximum_path_risk_upper_bound: expected_policy_receipt.maximum_path_risk_upper_bound,
        modeled_terminal_path_count: expected_policy_receipt.modeled_terminal_path_count,
        branch_probabilities_verified: true,
        posterior_updates_verified: true,
        objective_accounting_verified: true,
        pathwise_risk_budget_verified: true,
        canonical_solver_replay_match: true,
        optimality_independently_reverified: false,
    })
}

pub fn compute_sequential_diagnostic_safety_budget_digest(
    safety_budget: &SequentialDiagnosticSafetyBudget,
) -> Result<String, SequentialDiagnosticError> {
    validate_safety_budget(safety_budget)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-sequential-diagnostic-safety-budget-v1\0");
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

    // Small exhaustive reference implementation kept deliberately separate
    // from solve_node/plan_after_action. It returns only the optimum scalar
    // value, not a policy tree or receipt.
    fn exhaustive_reference_value(
        hypotheses: &[DecisionHypothesis],
        actions: &[DiagnosticAction],
        posterior: &[f64],
        config: DecisionDiagnosticConfig,
        remaining_steps: usize,
        used_actions: u64,
        cumulative_risk: f64,
        risk_budget: f64,
    ) -> f64 {
        let decision_count = hypotheses[0].loss_by_final_decision.len();
        let stop_value = (0..decision_count)
            .map(|decision| {
                hypotheses
                    .iter()
                    .zip(posterior)
                    .map(|(hypothesis, probability)| {
                        probability * hypothesis.loss_by_final_decision[decision]
                    })
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min);
        if remaining_steps == 0 {
            return stop_value;
        }

        let mut best = stop_value;
        for (action_index, action) in actions.iter().enumerate() {
            let action_bit = 1_u64 << action_index;
            let next_risk = cumulative_risk + action.safety_risk;
            if used_actions & action_bit != 0
                || action.safety_risk > config.maximum_safety_risk
                || next_risk > risk_budget + RISK_TOLERANCE
            {
                continue;
            }

            let outcome_count =
                hypotheses[0].outcome_probabilities_by_action[action_index].len();
            let mut candidate = config.observation_cost_weight * action.observation_cost
                + config.safety_risk_weight * action.safety_risk;
            for outcome_index in 0..outcome_count {
                let mut outcome_mass = 0.0;
                for (hypothesis, prior) in hypotheses.iter().zip(posterior) {
                    outcome_mass += prior
                        * hypothesis.outcome_probabilities_by_action[action_index][outcome_index];
                }
                if outcome_mass == 0.0 {
                    continue;
                }
                let next_posterior: Vec<f64> = hypotheses
                    .iter()
                    .zip(posterior)
                    .map(|(hypothesis, prior)| {
                        prior
                            * hypothesis.outcome_probabilities_by_action[action_index][outcome_index]
                            / outcome_mass
                    })
                    .collect();
                candidate += outcome_mass
                    * exhaustive_reference_value(
                        hypotheses,
                        actions,
                        &next_posterior,
                        config,
                        remaining_steps - 1,
                        used_actions | action_bit,
                        next_risk,
                        risk_budget,
                    );
            }
            best = best.min(candidate);
        }
        best
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
    fn cumulative_budget_planner_matches_separate_exhaustive_reference() {
        let (hypotheses, mut actions) = xor_complementary_probes();
        for action in &mut actions {
            action.safety_risk = 0.35;
        }
        let config = DecisionDiagnosticConfig {
            safety_risk_weight: 0.1,
            maximum_safety_risk: 0.5,
            ..decision_config()
        };
        let prior: Vec<f64> = hypotheses
            .iter()
            .map(|hypothesis| hypothesis.prior_probability)
            .collect();
        let planning = SequentialDiagnosticConfig {
            horizon: 2,
            max_policy_expansions: 64,
        };

        for (budget, expected) in [(0.6, 0.5), (0.7, 0.07)] {
            let reference = exhaustive_reference_value(
                &hypotheses,
                &actions,
                &prior,
                config,
                planning.horizon,
                0,
                0.0,
                budget,
            );
            let result = evaluate_sequential_decision_diagnostics_with_safety_budget(
                &hypotheses,
                &actions,
                config,
                planning,
                SequentialDiagnosticSafetyBudget {
                    maximum_cumulative_risk: budget,
                    risk_semantics_digest: "a".repeat(64),
                },
            )
            .expect("budgeted plan");
            assert!((reference - expected).abs() < 1e-12);
            assert!((result.evaluation.expected_total_cost - reference).abs() < 1e-12);
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

    fn independent_max_path_risk(
        node: &SequentialDiagnosticPlanNode,
        actions: &[DiagnosticAction],
    ) -> f64 {
        let action_risk = node
            .selected_action_index
            .map(|index| actions[index].safety_risk)
            .unwrap_or(0.0);
        let maximum_continuation = node
            .branches
            .iter()
            .map(|branch| independent_max_path_risk(&branch.continuation, actions))
            .fold(0.0_f64, f64::max);
        action_risk + maximum_continuation
    }

    fn causal_descriptors(
        hypotheses: &[DecisionHypothesis],
        actions: &[DiagnosticAction],
    ) -> Vec<CausalDiagnosticActionDescriptor> {
        use super::super::causal_diagnostic_semantics::{
            CausalDiagnosticActionKind, compute_action_likelihood_table_digest,
        };

        actions
            .iter()
            .enumerate()
            .map(|(action_index, action)| CausalDiagnosticActionDescriptor {
                action_id: action.id.clone(),
                action_likelihood_table_digest: compute_action_likelihood_table_digest(
                    hypotheses,
                    action_index,
                )
                .expect("valid action-likelihood table"),
                kind: CausalDiagnosticActionKind::PassiveObservation {
                    observation_model_digest: "a".repeat(64),
                    evidence_scope_digest: "b".repeat(64),
                },
            })
            .collect()
    }

    #[test]
    fn policy_receipt_binds_decision_problem_budget_and_worst_path_risk() {
        let (hypotheses, mut actions) = xor_complementary_probes();
        for action in &mut actions {
            action.safety_risk = 0.35;
        }
        let config = DecisionDiagnosticConfig {
            safety_risk_weight: 0.1,
            maximum_safety_risk: 0.5,
            ..decision_config()
        };
        let planning = SequentialDiagnosticConfig {
            horizon: 2,
            max_policy_expansions: 64,
        };
        let descriptors = causal_descriptors(&hypotheses, &actions);
        let result = evaluate_sequential_decision_diagnostics_with_policy_receipt(
            &hypotheses,
            &actions,
            &descriptors,
            config,
            planning,
            SequentialDiagnosticSafetyBudget {
                maximum_cumulative_risk: 0.7,
                risk_semantics_digest: "c".repeat(64),
            },
            &"d".repeat(64),
        )
        .expect("causally bound budgeted policy");

        assert_eq!(result.policy_receipt.schema_version, 1);
        assert_eq!(result.policy_receipt.decision_problem_digest,
            result.decision_problem_receipt.decision_problem_digest);
        assert_eq!(result.policy_receipt.maximum_path_risk_upper_bound, 0.7);
        assert_eq!(
            result.policy_receipt.maximum_path_risk_upper_bound,
            independent_max_path_risk(
                &result.budgeted_evaluation.evaluation.plan,
                &actions
            )
        );
        assert_eq!(result.policy_receipt.modeled_terminal_path_count, 4);
        assert_eq!(
            result.policy_receipt.receipt_digest.len(),
            64
        );

        let repeated = evaluate_sequential_decision_diagnostics_with_policy_receipt(
            &hypotheses,
            &actions,
            &descriptors,
            config,
            planning,
            SequentialDiagnosticSafetyBudget {
                maximum_cumulative_risk: 0.7,
                risk_semantics_digest: "c".repeat(64),
            },
            &"d".repeat(64),
        )
        .expect("deterministic replay");
        assert_eq!(
            result.policy_receipt.receipt_digest,
            repeated.policy_receipt.receipt_digest
        );

        let changed_budget = evaluate_sequential_decision_diagnostics_with_policy_receipt(
            &hypotheses,
            &actions,
            &descriptors,
            config,
            planning,
            SequentialDiagnosticSafetyBudget {
                maximum_cumulative_risk: 0.7,
                risk_semantics_digest: "e".repeat(64),
            },
            &"d".repeat(64),
        )
        .expect("changed risk semantics");
        assert_ne!(
            result.policy_receipt.receipt_digest,
            changed_budget.policy_receipt.receipt_digest
        );

        // The same tree shape with different externally supplied action-risk
        // bounds must produce a different causal problem and policy receipt.
        let mut changed_actions = actions.clone();
        changed_actions[1].safety_risk = 0.30;
        let changed_descriptors = causal_descriptors(&hypotheses, &changed_actions);
        let changed_action_risk = evaluate_sequential_decision_diagnostics_with_policy_receipt(
            &hypotheses,
            &changed_actions,
            &changed_descriptors,
            config,
            planning,
            SequentialDiagnosticSafetyBudget {
                maximum_cumulative_risk: 0.7,
                risk_semantics_digest: "c".repeat(64),
            },
            &"d".repeat(64),
        )
        .expect("changed action risk bound");
        assert!(
            (changed_action_risk.policy_receipt.maximum_path_risk_upper_bound - 0.65).abs()
                < 1e-12
        );
        assert_ne!(
            result.policy_receipt.receipt_digest,
            changed_action_risk.policy_receipt.receipt_digest
        );
    }

    #[test]
    fn public_policy_receipt_verifier_checks_branches_and_marks_optimality_unverified() {
        let (hypotheses, mut actions) = xor_complementary_probes();
        for action in &mut actions {
            action.safety_risk = 0.35;
        }
        let config = DecisionDiagnosticConfig {
            safety_risk_weight: 0.1,
            maximum_safety_risk: 0.5,
            ..decision_config()
        };
        let planning = SequentialDiagnosticConfig {
            horizon: 2,
            max_policy_expansions: 64,
        };
        let budget = SequentialDiagnosticSafetyBudget {
            maximum_cumulative_risk: 0.7,
            risk_semantics_digest: "c".repeat(64),
        };
        let descriptors = causal_descriptors(&hypotheses, &actions);
        let supplied = evaluate_sequential_decision_diagnostics_with_policy_receipt(
            &hypotheses,
            &actions,
            &descriptors,
            config,
            planning,
            budget.clone(),
            &"d".repeat(64),
        )
        .expect("receipt-bearing diagnostic policy");

        let report = verify_sequential_diagnostic_policy_receipt(
            &hypotheses,
            &actions,
            &descriptors,
            config,
            planning,
            &budget,
            &"d".repeat(64),
            &supplied,
        )
        .expect("structurally verified receipt");
        assert!(report.branch_probabilities_verified);
        assert!(report.posterior_updates_verified);
        assert!(report.objective_accounting_verified);
        assert!(report.pathwise_risk_budget_verified);
        assert!(!report.optimality_independently_reverified);
        assert_eq!(report.receipt_digest, supplied.policy_receipt.receipt_digest);

        let mut tampered = supplied.clone();
        tampered.budgeted_evaluation.evaluation.plan.branches[0]
            .outcome_probability += 0.01;
        assert!(verify_sequential_diagnostic_policy_receipt(
            &hypotheses,
            &actions,
            &descriptors,
            config,
            planning,
            &budget,
            &"d".repeat(64),
            &tampered,
        )
        .is_err());

        let changed_budget = SequentialDiagnosticSafetyBudget {
            maximum_cumulative_risk: 0.6,
            ..budget
        };
        assert!(matches!(
            verify_sequential_diagnostic_policy_receipt(
                &hypotheses,
                &actions,
                &descriptors,
                config,
                planning,
                &changed_budget,
                &"d".repeat(64),
                &supplied,
            ),
            Err(SequentialDiagnosticError::PolicyReceiptMismatch)
        ));
    }

    #[test]
    fn policy_receipt_rejects_a_stale_action_likelihood_descriptor() {
        let (hypotheses, actions) = xor_complementary_probes();
        let mut descriptors = causal_descriptors(&hypotheses, &actions);
        descriptors[0].action_likelihood_table_digest = "f".repeat(64);
        let result = evaluate_sequential_decision_diagnostics_with_policy_receipt(
            &hypotheses,
            &actions,
            &descriptors,
            decision_config(),
            SequentialDiagnosticConfig {
                horizon: 2,
                max_policy_expansions: 64,
            },
            SequentialDiagnosticSafetyBudget {
                maximum_cumulative_risk: 0.5,
                risk_semantics_digest: "c".repeat(64),
            },
            &"d".repeat(64),
        );
        assert!(matches!(
            result,
            Err(SequentialDiagnosticError::CausalDiagnosticAction(
                CausalDiagnosticActionError::ActionLikelihoodDigestMismatch(_)
            ))
        ));
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
