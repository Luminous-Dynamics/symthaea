//! Finite-horizon sequential decision policy for costed measurements.
//!
//! Unlike one-step EVSI-per-cost ranking, this module solves a bounded adaptive
//! measurement problem under a fixed integer budget. It enumerates observation
//! branches, updates the posterior after each observation, and chooses between
//! stopping and another measurement. The exactness claim is conditional on the
//! supplied finite model and is subject to f64 roundoff; it is not a scalable
//! optimizer or a guarantee that the supplied model is correct.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt;

use super::pareto_evsi_oracle::{
    ExactEvsiError, ExactEvsiProblem, EvsiReplicateAction,
};

const POLICY_SCHEMA: &str = "symthaea.architecture-engineering.sequential-decision-policy";
const POLICY_VERSION: &str = "0.1.0";
const RISK_TOLERANCE: f64 = 1e-9;
const MAX_POLICY_ACTIONS: usize = 16;
const MAX_EXPANDED_NODES: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequentialPolicyError {
    pub path: String,
    pub message: String,
}

impl SequentialPolicyError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for SequentialPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for SequentialPolicyError {}

impl From<ExactEvsiError> for SequentialPolicyError {
    fn from(error: ExactEvsiError) -> Self {
        Self { path: error.path, message: error.message }
    }
}

/// One reachable observation branch in an adaptive policy tree.
/// Zero-probability outcomes are retained with no fabricated posterior or child.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialPolicyBranch {
    pub outcome_id: String,
    pub predictive_probability: f64,
    pub posterior_sha256: Option<String>,
    pub child: Option<Box<SequentialPolicyNode>>,
}

/// A policy node makes an explicit stop-versus-measure decision after its visible history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialPolicyNode {
    pub remaining_budget_units: u64,
    pub remaining_action_ids: Vec<String>,
    pub posterior_sha256: String,
    pub current_bayes_risk: f64,
    /// Set when the optimal action at this node is to stop and take this terminal decision.
    pub terminal_decision_id: Option<String>,
    /// Set when the optimal action at this node is to perform another measurement.
    pub selected_action_id: Option<String>,
    /// Expected terminal Bayes risk after following this subtree.
    pub expected_terminal_bayes_risk: f64,
    /// Expected future spend, including the selected action at this node.
    pub expected_additional_cost_units: f64,
    /// Maximum possible spend along any reachable branch of this subtree.
    pub worst_case_additional_cost_units: u64,
    pub branches: Vec<SequentialPolicyBranch>,
}

/// Digest-bound optimal adaptive policy for one finite problem and integer budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialDecisionPolicy {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub problem_sha256: String,
    pub budget_units: u64,
    pub initial_bayes_risk: f64,
    pub expected_terminal_bayes_risk: f64,
    pub expected_risk_reduction: f64,
    pub expected_total_cost_units: f64,
    pub worst_case_total_cost_units: u64,
    pub expanded_policy_nodes: usize,
    pub root: SequentialPolicyNode,
    pub policy_sha256: String,
}

#[derive(Serialize)]
struct PolicyDigestPayload<'a> {
    schema: &'a str,
    version: &'a str,
    campaign_id: &'a str,
    problem_sha256: &'a str,
    budget_units: u64,
    initial_bayes_risk: f64,
    expected_terminal_bayes_risk: f64,
    expected_risk_reduction: f64,
    expected_total_cost_units: f64,
    worst_case_total_cost_units: u64,
    expanded_policy_nodes: usize,
    root: &'a SequentialPolicyNode,
}

/// Solve the finite-horizon adaptive measurement problem exactly by enumerating
/// every reachable posterior/action/budget state.
///
/// Every action can be used at most once. To model repeated measurements, provide
/// distinct action IDs (for example, candidate/objective/replicate slots). Their
/// likelihoods are conditional on the latent joint world. Shared latent noise or
/// dependence must be represented in that joint world rather than silently assumed
/// away. The decision set may include an explicit abstention decision with a declared
/// loss; stopping is always available and costs no additional measurement budget.
///
/// The action count is capped at 16 and the policy expansion at 100,000 nodes. This
/// is a small-problem reference solver, not the production controller.
pub fn solve_sequential_policy(
    problem: &ExactEvsiProblem,
    budget_units: u64,
) -> Result<SequentialDecisionPolicy, SequentialPolicyError> {
    let canonical = problem.clone().canonicalize()?;
    if canonical.actions.len() > MAX_POLICY_ACTIONS {
        return Err(SequentialPolicyError::new(
            "problem.actions",
            format!("sequential reference solver supports at most {MAX_POLICY_ACTIONS} distinct actions"),
        ));
    }

    let action_mask = if canonical.actions.is_empty() {
        0
    } else {
        (1_u64 << canonical.actions.len()) - 1
    };
    let posterior: Vec<f64> = canonical.worlds
        .iter()
        .map(|world| world.posterior_probability)
        .collect();
    let mut expanded_nodes = 0usize;
    let root = solve_node(
        &canonical,
        &posterior,
        action_mask,
        budget_units,
        &mut expanded_nodes,
    )?;
    let initial_bayes_risk = root.current_bayes_risk;
    let expected_terminal_bayes_risk = root.expected_terminal_bayes_risk;
    let expected_risk_reduction =
        (initial_bayes_risk - expected_terminal_bayes_risk).max(0.0);
    let expected_total_cost_units = root.expected_additional_cost_units;
    let worst_case_total_cost_units = root.worst_case_additional_cost_units;
    let problem_sha256 = canonical.problem_sha256()?;

    let payload = PolicyDigestPayload {
        schema: POLICY_SCHEMA,
        version: POLICY_VERSION,
        campaign_id: &canonical.campaign_id,
        problem_sha256: &problem_sha256,
        budget_units,
        initial_bayes_risk,
        expected_terminal_bayes_risk,
        expected_risk_reduction,
        expected_total_cost_units,
        worst_case_total_cost_units,
        expanded_policy_nodes: expanded_nodes,
        root: &root,
    };
    let policy_sha256 = digest_serializable(&payload)?;

    Ok(SequentialDecisionPolicy {
        schema: POLICY_SCHEMA.to_owned(),
        version: POLICY_VERSION.to_owned(),
        campaign_id: canonical.campaign_id,
        problem_sha256,
        budget_units,
        initial_bayes_risk,
        expected_terminal_bayes_risk,
        expected_risk_reduction,
        expected_total_cost_units,
        worst_case_total_cost_units,
        expanded_policy_nodes: expanded_nodes,
        root,
        policy_sha256,
    })
}

/// Re-solve the finite problem and reject any altered policy tree or summary.
pub fn verify_sequential_policy(
    problem: &ExactEvsiProblem,
    policy: &SequentialDecisionPolicy,
) -> Result<(), SequentialPolicyError> {
    let reconstructed = solve_sequential_policy(problem, policy.budget_units)?;
    if reconstructed != *policy {
        return Err(SequentialPolicyError::new(
            "policy",
            "stored sequential policy differs from reconstruction from the frozen finite problem",
        ));
    }
    Ok(())
}

fn solve_node(
    problem: &ExactEvsiProblem,
    posterior: &[f64],
    remaining_action_mask: u64,
    remaining_budget_units: u64,
    expanded_nodes: &mut usize,
) -> Result<SequentialPolicyNode, SequentialPolicyError> {
    *expanded_nodes = expanded_nodes.checked_add(1).ok_or_else(|| {
        SequentialPolicyError::new("policy.expanded_nodes", "policy-node counter overflow")
    })?;
    if *expanded_nodes > MAX_EXPANDED_NODES {
        return Err(SequentialPolicyError::new(
            "policy.expanded_nodes",
            format!("policy expansion exceeded the {MAX_EXPANDED_NODES}-node safety ceiling"),
        ));
    }

    let (stop_risk, stop_decision_id) = minimum_bayes_risk(problem, posterior)?;
    let available_action_ids: Vec<String> = problem.actions
        .iter()
        .enumerate()
        .filter(|(index, _)| (remaining_action_mask & (1_u64 << index)) != 0)
        .map(|(_, action)| action.action_id.clone())
        .collect();

    // Stopping is the incumbent. Ties deliberately prefer stopping to avoid spending
    // budget for no expected reduction in terminal decision risk.
    let mut best_expected_risk = stop_risk;
    let mut best_expected_cost = 0.0;
    let mut best_worst_case_cost = 0u64;
    let mut best_action_id: Option<String> = None;
    let mut best_terminal_decision_id = Some(stop_decision_id);
    let mut best_branches = Vec::new();

    for (action_index, action) in problem.actions.iter().enumerate() {
        let action_bit = 1_u64 << action_index;
        if remaining_action_mask & action_bit == 0
            || action.incremental_cost_units > remaining_budget_units
        {
            continue;
        }
        let (expected_risk, expected_cost, worst_case_cost, branches) = evaluate_action(
            problem,
            action,
            action_index,
            posterior,
            remaining_action_mask,
            remaining_budget_units,
            expanded_nodes,
        )?;
        if expected_risk > stop_risk + RISK_TOLERANCE {
            return Err(SequentialPolicyError::new(
                format!("actions.{}.expected_risk", action.action_id),
                "measurement increased expected optimal terminal risk; check likelihood normalization and numeric conditioning",
            ));
        }
        if expected_risk + RISK_TOLERANCE < best_expected_risk {
            best_expected_risk = expected_risk;
            best_expected_cost = expected_cost;
            best_worst_case_cost = worst_case_cost;
            best_action_id = Some(action.action_id.clone());
            best_terminal_decision_id = None;
            best_branches = branches;
        }
        // Actions are canonicalized by ID. Strict improvement preserves deterministic
        // lexical tie-breaking among actions with equivalent terminal risk.
    }

    let posterior_sha256 = digest_serializable(&posterior)?;
    Ok(SequentialPolicyNode {
        remaining_budget_units,
        remaining_action_ids: available_action_ids,
        posterior_sha256,
        current_bayes_risk: stop_risk,
        terminal_decision_id: best_terminal_decision_id,
        selected_action_id: best_action_id,
        expected_terminal_bayes_risk: best_expected_risk,
        expected_additional_cost_units: best_expected_cost,
        worst_case_additional_cost_units: best_worst_case_cost,
        branches: best_branches,
    })
}

fn evaluate_action(
    problem: &ExactEvsiProblem,
    action: &EvsiReplicateAction,
    action_index: usize,
    posterior: &[f64],
    remaining_action_mask: u64,
    remaining_budget_units: u64,
    expanded_nodes: &mut usize,
) -> Result<
    (f64, f64, u64, Vec<SequentialPolicyBranch>),
    SequentialPolicyError,
> {
    let mut expected_risk = 0.0;
    let mut risk_compensation = 0.0;
    let mut expected_cost = action.incremental_cost_units as f64;
    let mut cost_compensation = 0.0;
    let mut worst_case_cost = action.incremental_cost_units;
    let mut branches = Vec::with_capacity(action.outcomes.len());
    let child_mask = remaining_action_mask & !(1_u64 << action_index);
    let child_budget = remaining_budget_units - action.incremental_cost_units;

    for outcome in &action.outcomes {
        let mut predictive_probability = 0.0;
        let mut predictive_compensation = 0.0;
        let likelihoods: Vec<f64> = outcome.likelihoods_by_world
            .iter()
            .map(|row| row.probability)
            .collect();
        for (prior, likelihood) in posterior.iter().zip(&likelihoods) {
            kahan_add(
                &mut predictive_probability,
                &mut predictive_compensation,
                prior * likelihood,
            );
        }

        if predictive_probability <= 0.0 {
            branches.push(SequentialPolicyBranch {
                outcome_id: outcome.outcome_id.clone(),
                predictive_probability: 0.0,
                posterior_sha256: None,
                child: None,
            });
            continue;
        }

        let posterior_after_outcome: Vec<f64> = posterior
            .iter()
            .zip(&likelihoods)
            .map(|(prior, likelihood)| prior * likelihood / predictive_probability)
            .collect();
        let child = solve_node(
            problem,
            &posterior_after_outcome,
            child_mask,
            child_budget,
            expanded_nodes,
        )?;
        kahan_add(
            &mut expected_risk,
            &mut risk_compensation,
            predictive_probability * child.expected_terminal_bayes_risk,
        );
        kahan_add(
            &mut expected_cost,
            &mut cost_compensation,
            predictive_probability * child.expected_additional_cost_units,
        );
        worst_case_cost = worst_case_cost.max(
            action.incremental_cost_units
                .checked_add(child.worst_case_additional_cost_units)
                .ok_or_else(|| SequentialPolicyError::new(
                    "policy.worst_case_cost",
                    "worst-case cost addition overflow",
                ))?,
        );
        let posterior_sha256 = child.posterior_sha256.clone();
        branches.push(SequentialPolicyBranch {
            outcome_id: outcome.outcome_id.clone(),
            predictive_probability,
            posterior_sha256: Some(posterior_sha256),
            child: Some(Box::new(child)),
        });
    }

    if !expected_risk.is_finite() || !expected_cost.is_finite() {
        return Err(SequentialPolicyError::new(
            format!("actions.{}", action.action_id),
            "expected risk or cost became non-finite",
        ));
    }
    if worst_case_cost > remaining_budget_units {
        return Err(SequentialPolicyError::new(
            format!("actions.{}.worst_case_cost", action.action_id),
            "constructed policy exceeds the declared measurement budget",
        ));
    }
    Ok((expected_risk, expected_cost, worst_case_cost, branches))
}

fn minimum_bayes_risk(
    problem: &ExactEvsiProblem,
    posterior: &[f64],
) -> Result<(f64, String), SequentialPolicyError> {
    if posterior.len() != problem.worlds.len() {
        return Err(SequentialPolicyError::new(
            "posterior",
            "posterior vector length differs from the finite world space",
        ));
    }
    let mut best: Option<(f64, String)> = None;
    for decision in &problem.decisions {
        let mut risk = 0.0;
        let mut compensation = 0.0;
        for (index, loss) in decision.losses_by_world.iter().enumerate() {
            kahan_add(
                &mut risk,
                &mut compensation,
                posterior[index] * loss.loss,
            );
        }
        if !risk.is_finite() {
            return Err(SequentialPolicyError::new(
                format!("decisions.{}.expected_loss", decision.decision_id),
                "expected decision loss became non-finite",
            ));
        }
        match &best {
            None => best = Some((risk, decision.decision_id.clone())),
            Some((best_risk, best_id))
                if risk < *best_risk
                    || (risk == *best_risk && decision.decision_id < *best_id) =>
            {
                best = Some((risk, decision.decision_id.clone()));
            }
            _ => {}
        }
    }
    best.ok_or_else(|| SequentialPolicyError::new(
        "decisions",
        "no terminal decisions are available",
    ))
}

fn kahan_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let adjusted = value - *compensation;
    let next = *sum + adjusted;
    *compensation = (next - *sum) - adjusted;
    *sum = next;
}

fn digest_serializable<T: Serialize>(
    value: &T,
) -> Result<String, SequentialPolicyError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        SequentialPolicyError::new(
            "serialization",
            format!("cannot serialize sequential policy receipt: {error}"),
        )
    })?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pareto_evsi_oracle::{
        analyze_exact_evsi, EvsiActionOutcome, EvsiDecision, EvsiWorldLikelihood,
        EvsiWorldLoss, EvsiWorldProbability,
    };

    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn losses(a: f64, b: f64) -> Vec<EvsiWorldLoss> {
        vec![
            EvsiWorldLoss { world_id: "world-a".to_owned(), loss: a },
            EvsiWorldLoss { world_id: "world-b".to_owned(), loss: b },
        ]
    }

    fn likelihoods(a: f64, b: f64) -> Vec<EvsiWorldLikelihood> {
        vec![
            EvsiWorldLikelihood { world_id: "world-a".to_owned(), probability: a },
            EvsiWorldLikelihood { world_id: "world-b".to_owned(), probability: b },
        ]
    }

    fn signal_action(id: &str, cost: u64, p_a: f64) -> EvsiReplicateAction {
        EvsiReplicateAction {
            action_id: id.to_owned(),
            candidate_id: "candidate-a".to_owned(),
            objective_id: "quality".to_owned(),
            incremental_cost_units: cost,
            outcomes: vec![
                EvsiActionOutcome {
                    outcome_id: "signal-a".to_owned(),
                    likelihoods_by_world: likelihoods(p_a, 1.0 - p_a),
                },
                EvsiActionOutcome {
                    outcome_id: "signal-b".to_owned(),
                    likelihoods_by_world: likelihoods(1.0 - p_a, p_a),
                },
            ],
        }
    }

    fn problem() -> ExactEvsiProblem {
        ExactEvsiProblem {
            schema: "symthaea.architecture-engineering.exact-discrete-evsi-problem".to_owned(),
            version: "0.1.0".to_owned(),
            campaign_id: "sequential-budget-choice".to_owned(),
            posterior_model_id: "finite-posterior".to_owned(),
            posterior_model_sha256: H1.to_owned(),
            observation_model_id: "binary-signal-model".to_owned(),
            observation_model_sha256: H2.to_owned(),
            decision_loss_model_id: "symmetric-terminal-loss".to_owned(),
            decision_loss_model_sha256: H3.to_owned(),
            cost_model_id: "integer-cost-units".to_owned(),
            cost_model_sha256: H4.to_owned(),
            worlds: vec![
                EvsiWorldProbability { world_id: "world-a".to_owned(), posterior_probability: 0.5 },
                EvsiWorldProbability { world_id: "world-b".to_owned(), posterior_probability: 0.5 },
            ],
            decisions: vec![
                EvsiDecision {
                    decision_id: "choose-a".to_owned(),
                    losses_by_world: losses(0.0, 10.0),
                },
                EvsiDecision {
                    decision_id: "choose-b".to_owned(),
                    losses_by_world: losses(10.0, 0.0),
                },
            ],
            actions: vec![
                signal_action("measure-cheap", 1, 0.7),
                signal_action("measure-expensive", 2, 0.85),
            ],
        }
    }

    #[test]
    fn budgeted_dynamic_program_can_beat_greedy_evsi_per_cost() {
        let problem = problem();
        let greedy = analyze_exact_evsi(&problem).unwrap();
        assert_eq!(greedy.selected_action_id.as_deref(), Some("measure-cheap"));

        let policy = solve_sequential_policy(&problem, 2).unwrap();
        assert_eq!(policy.root.selected_action_id.as_deref(), Some("measure-expensive"));
        assert!((policy.initial_bayes_risk - 5.0).abs() < 1e-12);
        assert!((policy.expected_terminal_bayes_risk - 1.5).abs() < 1e-12);
        assert!((policy.expected_risk_reduction - 3.5).abs() < 1e-12);
        assert!((policy.expected_total_cost_units - 2.0).abs() < 1e-12);
        assert_eq!(policy.worst_case_total_cost_units, 2);
        verify_sequential_policy(&problem, &policy).unwrap();
    }

    #[test]
    fn zero_value_measurements_do_not_displace_the_stop_decision() {
        let mut model = problem();
        for action in &mut model.actions {
            action.outcomes = vec![
                EvsiActionOutcome {
                    outcome_id: "uninformative".to_owned(),
                    likelihoods_by_world: likelihoods(1.0, 1.0),
                },
            ];
        }
        let policy = solve_sequential_policy(&model, 3).unwrap();
        assert_eq!(policy.root.selected_action_id, None);
        assert!(policy.root.terminal_decision_id.is_some());
        assert_eq!(policy.expected_total_cost_units, 0.0);
        assert_eq!(policy.worst_case_total_cost_units, 0);
    }

    #[test]
    fn policy_tree_is_bound_to_the_problem_budget_and_decisions() {
        let problem = problem();
        let mut policy = solve_sequential_policy(&problem, 2).unwrap();
        policy.root.selected_action_id = Some("measure-cheap".to_owned());
        assert!(verify_sequential_policy(&problem, &policy).is_err());

        let baseline = solve_sequential_policy(&problem, 2).unwrap();
        let different_budget = solve_sequential_policy(&problem, 1).unwrap();
        assert_ne!(baseline.policy_sha256, different_budget.policy_sha256);
    }

    #[test]
    fn input_order_does_not_change_the_reconstructed_policy() {
        let baseline = problem();
        let mut reordered = baseline.clone();
        reordered.worlds.reverse();
        reordered.decisions.reverse();
        reordered.actions.reverse();
        for decision in &mut reordered.decisions {
            decision.losses_by_world.reverse();
        }
        for action in &mut reordered.actions {
            action.outcomes.reverse();
            for outcome in &mut action.outcomes {
                outcome.likelihoods_by_world.reverse();
            }
        }
        let left = solve_sequential_policy(&baseline, 2).unwrap();
        let right = solve_sequential_policy(&reordered, 2).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn zero_probability_outcomes_are_retained_without_fabricated_children() {
        let mut model = problem();
        for action in &mut model.actions {
            action.outcomes.push(EvsiActionOutcome {
                outcome_id: "impossible".to_owned(),
                likelihoods_by_world: likelihoods(0.0, 0.0),
            });
        }
        let policy = solve_sequential_policy(&model, 2).unwrap();
        assert!(policy.root.selected_action_id.is_some());
        let impossible = policy.root.branches.iter().find(|branch|
            branch.outcome_id == "impossible"
        ).unwrap();
        assert_eq!(impossible.predictive_probability, 0.0);
        assert_eq!(impossible.child, None);
        assert_eq!(impossible.posterior_sha256, None);
    }
}
