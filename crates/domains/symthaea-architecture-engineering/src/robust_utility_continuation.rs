//! Robust decision utility over a finite alternative family with a hard null-size constraint.
//!
//! One null P0 and multiple alternative kernels share a finite observable tree.
//! A common deterministic stop/continue policy is exhaustively enumerated.
//! Its exact type-I size is evaluated under P0, while its power and expected
//! evaluator cost are evaluated under every alternative P1_k.
//!
//! The selected policy maximizes the worst-case decision utility:
//!   benefit * power
//!   - false_negative_loss * (1 - power)
//!   - false_positive_loss * null_size
//!   - evaluator_cost_weight * expected_cost
//! subject to null_size <= alpha and an optional worst-case expected-cost cap.
//!
//! This is an exact finite reference, not a continuous robust MDP and not a
//! guarantee against alternatives omitted from the declared finite family.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

const PLAN_SCHEMA: &str = "symthaea.architecture-engineering.robust-utility-continuation-plan";
const PLAN_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.robust-utility-continuation-report";
const REPORT_VERSION: &str = "0.1.0";
const PROBABILITY_TOLERANCE: f64 = 1e-10;
const SUPERMARTINGALE_TOLERANCE: f64 = 1e-10;
const OBJECTIVE_TOLERANCE: f64 = 1e-12;
const MAX_ALTERNATIVE_MODELS: usize = 8;
const MAX_HISTORY_NODES: usize = 64;
const MAX_POLICY_DECISION_NODES: usize = 16;
const MAX_CONTINUATION_OUTCOMES_PER_HISTORY: usize = 64;
const MAX_TOTAL_CONTINUATION_OUTCOMES: usize = 4_096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustUtilityError {
    pub path: String,
    pub message: String,
}

impl RobustUtilityError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for RobustUtilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for RobustUtilityError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustAlternativeModel {
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlternativeTransitionProbability {
    pub model_id: String,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustContinuationOutcome {
    pub outcome_id: String,
    pub null_probability: f64,
    pub alternative_probabilities: Vec<AlternativeTransitionProbability>,
    /// Terminal randomized-test rejection probability for this observable outcome.
    pub rejection_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustConditionalTest {
    /// Cost paid when this confirmatory evaluation is launched.
    pub launch_cost: f64,
    pub outcomes: Vec<RobustContinuationOutcome>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustHistoryNode {
    pub node_id: String,
    pub parent_node_id: Option<String>,
    /// Conditional transition probability under the null P0.
    pub null_probability: f64,
    /// One probability for every declared alternative model.
    pub alternative_probabilities: Vec<AlternativeTransitionProbability>,
    /// [0,1] interim test value; the P0 conditional expectation of children
    /// may not exceed this value.
    pub interim_test_value: f64,
    /// Evaluator cost paid when this history is reached from its parent.
    pub observation_cost: f64,
    /// Confirmatory continuation available if the policy stops at this history.
    pub continuation_test: RobustConditionalTest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustDecisionLoss {
    pub true_rejection_benefit: f64,
    pub false_negative_loss: f64,
    pub false_positive_loss: f64,
    pub evaluator_cost_weight: f64,
    /// Optional hard cap on the maximum expected cost across alternatives.
    pub maximum_expected_cost: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustUtilityContinuationPlan {
    pub schema: String,
    pub version: String,
    pub plan_id: String,
    pub significance_level: f64,
    pub alternative_models: Vec<RobustAlternativeModel>,
    pub decision_loss: RobustDecisionLoss,
    pub history_nodes: Vec<RobustHistoryNode>,
}

impl RobustUtilityContinuationPlan {
    pub fn new(
        plan_id: impl Into<String>,
        significance_level: f64,
        alternative_models: Vec<RobustAlternativeModel>,
        decision_loss: RobustDecisionLoss,
        history_nodes: Vec<RobustHistoryNode>,
    ) -> Result<Self, RobustUtilityError> {
        Self {
            schema: PLAN_SCHEMA.into(),
            version: PLAN_VERSION.into(),
            plan_id: plan_id.into(),
            significance_level,
            alternative_models,
            decision_loss,
            history_nodes,
        }
        .canonicalize()
    }

    pub fn canonicalize(mut self) -> Result<Self, RobustUtilityError> {
        self.alternative_models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        self.history_nodes.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        for node in &mut self.history_nodes {
            node.alternative_probabilities.sort_by(|a, b| a.model_id.cmp(&b.model_id));
            node.continuation_test.outcomes.sort_by(|a, b| a.outcome_id.cmp(&b.outcome_id));
            for outcome in &mut node.continuation_test.outcomes {
                outcome.alternative_probabilities.sort_by(|a, b| a.model_id.cmp(&b.model_id));
            }
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), RobustUtilityError> {
        if self.schema != PLAN_SCHEMA || self.version != PLAN_VERSION {
            return Err(RobustUtilityError::new(
                "plan.schema",
                "unsupported robust-utility continuation schema/version",
            ));
        }
        if self.plan_id.trim().is_empty() {
            return Err(RobustUtilityError::new("plan.plan_id", "plan ID must be non-empty"));
        }
        if !self.significance_level.is_finite()
            || !(0.0..1.0).contains(&self.significance_level)
        {
            return Err(RobustUtilityError::new(
                "plan.significance_level",
                "alpha must be finite and strictly between zero and one",
            ));
        }
        if self.alternative_models.len() < 2
            || self.alternative_models.len() > MAX_ALTERNATIVE_MODELS
        {
            return Err(RobustUtilityError::new(
                "plan.alternative_models",
                "robust alternative family must contain between two and eight models",
            ));
        }
        let mut model_ids = BTreeSet::new();
        for (index, model) in self.alternative_models.iter().enumerate() {
            if model.model_id.trim().is_empty() || !model_ids.insert(model.model_id.as_str()) {
                return Err(RobustUtilityError::new(
                    format!("plan.alternative_models[{index}].model_id"),
                    "alternative model IDs must be non-empty and unique",
                ));
            }
        }
        validate_nonnegative(
            self.decision_loss.true_rejection_benefit,
            "plan.decision_loss.true_rejection_benefit",
        )?;
        validate_nonnegative(
            self.decision_loss.false_negative_loss,
            "plan.decision_loss.false_negative_loss",
        )?;
        validate_nonnegative(
            self.decision_loss.false_positive_loss,
            "plan.decision_loss.false_positive_loss",
        )?;
        validate_nonnegative(
            self.decision_loss.evaluator_cost_weight,
            "plan.decision_loss.evaluator_cost_weight",
        )?;
        if let Some(cap) = self.decision_loss.maximum_expected_cost {
            validate_nonnegative(cap, "plan.decision_loss.maximum_expected_cost")?;
        }

        if self.history_nodes.is_empty() || self.history_nodes.len() > MAX_HISTORY_NODES {
            return Err(RobustUtilityError::new(
                "plan.history_nodes",
                "history node count is empty or exceeds supported bounds",
            ));
        }

        let mut by_id = BTreeMap::new();
        let mut roots = Vec::new();
        let mut total_continuation_outcomes = 0usize;
        for (index, node) in self.history_nodes.iter().enumerate() {
            let path = format!("plan.history_nodes[{index}]");
            if node.node_id.trim().is_empty()
                || by_id.insert(node.node_id.as_str(), node).is_some()
            {
                return Err(RobustUtilityError::new(
                    format!("{path}.node_id"),
                    "history IDs must be non-empty and unique",
                ));
            }
            validate_probability(node.null_probability, &format!("{path}.null_probability"))?;
            validate_probability(
                node.interim_test_value,
                &format!("{path}.interim_test_value"),
            )?;
            validate_nonnegative(node.observation_cost, &format!("{path}.observation_cost"))?;
            validate_alternative_probabilities(
                &node.alternative_probabilities,
                &model_ids,
                &format!("{path}.alternative_probabilities"),
            )?;
            if node.parent_node_id.is_none() {
                roots.push(node);
            } else if node.parent_node_id.as_deref().is_some_and(str::is_empty) {
                return Err(RobustUtilityError::new(
                    format!("{path}.parent_node_id"),
                    "parent ID cannot be empty",
                ));
            }

            validate_nonnegative(
                node.continuation_test.launch_cost,
                &format!("{path}.continuation_test.launch_cost"),
            )?;
            if node.continuation_test.outcomes.is_empty()
                || node.continuation_test.outcomes.len() > MAX_CONTINUATION_OUTCOMES_PER_HISTORY
            {
                return Err(RobustUtilityError::new(
                    format!("{path}.continuation_test.outcomes"),
                    "continuation outcome count is empty or exceeds supported bounds",
                ));
            }
            total_continuation_outcomes = total_continuation_outcomes
                .checked_add(node.continuation_test.outcomes.len())
                .ok_or_else(|| RobustUtilityError::new(
                    "plan.history_nodes",
                    "total continuation outcome count overflow",
                ))?;
            let mut outcome_ids = BTreeSet::new();
            let mut null_mass = 0.0;
            let mut null_mass_compensation = 0.0;
            let mut conditional_size = 0.0;
            let mut size_compensation = 0.0;
            for (outcome_index, outcome) in node.continuation_test.outcomes.iter().enumerate() {
                let outcome_path = format!("{path}.continuation_test.outcomes[{outcome_index}]");
                if outcome.outcome_id.trim().is_empty()
                    || !outcome_ids.insert(outcome.outcome_id.as_str())
                {
                    return Err(RobustUtilityError::new(
                        format!("{outcome_path}.outcome_id"),
                        "continuation outcome IDs must be non-empty and unique",
                    ));
                }
                validate_probability(
                    outcome.null_probability,
                    &format!("{outcome_path}.null_probability"),
                )?;
                validate_probability(
                    outcome.rejection_probability,
                    &format!("{outcome_path}.rejection_probability"),
                )?;
                validate_alternative_probabilities(
                    &outcome.alternative_probabilities,
                    &model_ids,
                    &format!("{outcome_path}.alternative_probabilities"),
                )?;
                kahan_add(&mut null_mass, &mut null_mass_compensation, outcome.null_probability);
                kahan_add(
                    &mut conditional_size,
                    &mut size_compensation,
                    outcome.null_probability * outcome.rejection_probability,
                );
            }
            if (null_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(RobustUtilityError::new(
                    format!("{path}.continuation_test.outcomes.null_probability"),
                    format!("conditional null probabilities must sum to one; found {null_mass}"),
                ));
            }
            if conditional_size > node.interim_test_value + SUPERMARTINGALE_TOLERANCE {
                return Err(RobustUtilityError::new(
                    format!("{path}.continuation_test.outcomes"),
                    format!(
                        "conditional continuation size {conditional_size} exceeds interim test value {}",
                        node.interim_test_value
                    ),
                ));
            }
            for model in &self.alternative_models {
                let mass = compensated_sum(node.continuation_test.outcomes.iter().map(|outcome| {
                    alternative_probability(&outcome.alternative_probabilities, &model.model_id)
                        .unwrap_or(f64::NAN)
                }));
                if !mass.is_finite() || (mass - 1.0).abs() > PROBABILITY_TOLERANCE {
                    return Err(RobustUtilityError::new(
                        format!("{path}.continuation_test.outcomes.alternative_probabilities.{}", model.model_id),
                        format!("conditional alternative probabilities must sum to one; found {mass}"),
                    ));
                }
            }
        }
        if total_continuation_outcomes > MAX_TOTAL_CONTINUATION_OUTCOMES {
            return Err(RobustUtilityError::new(
                "plan.history_nodes",
                "total continuation outcome count exceeds supported bounds",
            ));
        }
        if roots.len() != 1 {
            return Err(RobustUtilityError::new(
                "plan.history_nodes.root",
                format!("expected exactly one root, found {}", roots.len()),
            ));
        }
        let root = roots[0];
        if (root.null_probability - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(RobustUtilityError::new(
                "plan.history_nodes.root.null_probability",
                "root null transition probability must equal one",
            ));
        }
        if root.observation_cost.abs() > PROBABILITY_TOLERANCE {
            return Err(RobustUtilityError::new(
                "plan.history_nodes.root.observation_cost",
                "root observation cost must be zero because no incoming edge is traversed",
            ));
        }
        if root.interim_test_value > self.significance_level + SUPERMARTINGALE_TOLERANCE {
            return Err(RobustUtilityError::new(
                "plan.history_nodes.root.interim_test_value",
                "root interim test value exceeds alpha",
            ));
        }
        for model in &self.alternative_models {
            let root_probability = alternative_probability(
                &root.alternative_probabilities,
                &model.model_id,
            )
            .ok_or_else(|| RobustUtilityError::new(
                "plan.history_nodes.root.alternative_probabilities",
                format!("root probability missing for model {}", model.model_id),
            ))?;
            if (root_probability - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(RobustUtilityError::new(
                    format!("plan.history_nodes.root.alternative_probabilities.{}", model.model_id),
                    "root alternative transition probability must equal one",
                ));
            }
        }

        let mut children: BTreeMap<String, Vec<&RobustHistoryNode>> = BTreeMap::new();
        for (index, node) in self.history_nodes.iter().enumerate() {
            if let Some(parent_id) = node.parent_node_id.as_deref() {
                if !by_id.contains_key(parent_id) {
                    return Err(RobustUtilityError::new(
                        format!("plan.history_nodes[{index}].parent_node_id"),
                        format!("unknown parent history {parent_id:?}"),
                    ));
                }
                children.entry(parent_id.to_owned()).or_default().push(node);
            }
        }

        let internal_decisions = children.len();
        if internal_decisions > MAX_POLICY_DECISION_NODES {
            return Err(RobustUtilityError::new(
                "plan.history_nodes",
                format!(
                    "policy has {internal_decisions} decision histories; exact enumeration is bounded at {MAX_POLICY_DECISION_NODES}"
                ),
            ));
        }

        for (parent_id, child_nodes) in &children {
            let parent = by_id.get(parent_id.as_str()).copied().ok_or_else(|| {
                RobustUtilityError::new("plan.history_nodes", "validated parent disappeared")
            })?;
            let mut null_mass = 0.0;
            let mut null_mass_compensation = 0.0;
            let mut expected_test = 0.0;
            let mut test_compensation = 0.0;
            for child in child_nodes {
                kahan_add(&mut null_mass, &mut null_mass_compensation, child.null_probability);
                kahan_add(
                    &mut expected_test,
                    &mut test_compensation,
                    child.null_probability * child.interim_test_value,
                );
            }
            if (null_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(RobustUtilityError::new(
                    format!("plan.history_nodes.{parent_id}.children.null_probability"),
                    format!("null child probabilities must sum to one; found {null_mass}"),
                ));
            }
            if expected_test > parent.interim_test_value + SUPERMARTINGALE_TOLERANCE {
                return Err(RobustUtilityError::new(
                    format!("plan.history_nodes.{parent_id}.children"),
                    format!(
                        "P0 conditional expected child test value {expected_test} exceeds parent budget {}",
                        parent.interim_test_value
                    ),
                ));
            }
            for model in &self.alternative_models {
                let mass = compensated_sum(child_nodes.iter().map(|child| {
                    alternative_probability(&child.alternative_probabilities, &model.model_id)
                        .unwrap_or(f64::NAN)
                }));
                if !mass.is_finite() || (mass - 1.0).abs() > PROBABILITY_TOLERANCE {
                    return Err(RobustUtilityError::new(
                        format!("plan.history_nodes.{parent_id}.children.alternative_probabilities.{}", model.model_id),
                        format!("alternative child probabilities must sum to one; found {mass}"),
                    ));
                }
            }
        }

        // Every history must descend from the unique root; detached cycles fail closed.
        for node in &self.history_nodes {
            let mut current = node;
            let mut seen = BTreeSet::new();
            seen.insert(current.node_id.as_str());
            while let Some(parent_id) = current.parent_node_id.as_deref() {
                if !seen.insert(parent_id) {
                    return Err(RobustUtilityError::new(
                        "plan.history_nodes",
                        format!("parent cycle detected from history {}", node.node_id),
                    ));
                }
                current = by_id.get(parent_id).copied().ok_or_else(|| {
                    RobustUtilityError::new("plan.history_nodes", "history parent disappeared")
                })?;
            }
            if current.node_id != root.node_id {
                return Err(RobustUtilityError::new(
                    "plan.history_nodes",
                    format!("history {} does not descend from the unique root", node.node_id),
                ));
            }
        }
        Ok(())
    }

    pub fn plan_sha256(&self) -> Result<String, RobustUtilityError> {
        let canonical = self.clone().canonicalize()?;
        digest(&canonical)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobustPolicyAction {
    LaunchConditionalContinuation,
    ContinueFirstStage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustPolicyDecision {
    pub node_id: String,
    pub action: RobustPolicyAction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustAlternativeEvaluation {
    pub model_id: String,
    pub rejection_power: f64,
    pub expected_evaluator_cost: f64,
    pub decision_utility: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustUtilityContinuationReport {
    pub schema: String,
    pub version: String,
    pub plan_id: String,
    pub plan_sha256: String,
    pub significance_level: f64,
    pub selected_policy_mask: u64,
    pub enumerated_policy_count: u64,
    pub feasible_policy_count: u64,
    pub selected_policy: Vec<RobustPolicyDecision>,
    pub selected_null_size: f64,
    pub worst_case_rejection_power: f64,
    pub worst_case_decision_utility: f64,
    pub maximum_expected_evaluator_cost: f64,
    pub remaining_alpha_budget: f64,
    pub alternative_evaluations: Vec<RobustAlternativeEvaluation>,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    plan_id: &'a str,
    plan_sha256: &'a str,
    significance_level: f64,
    selected_policy_mask: u64,
    enumerated_policy_count: u64,
    feasible_policy_count: u64,
    selected_policy: &'a [RobustPolicyDecision],
    selected_null_size: f64,
    worst_case_rejection_power: f64,
    worst_case_decision_utility: f64,
    maximum_expected_evaluator_cost: f64,
    remaining_alpha_budget: f64,
    alternative_evaluations: &'a [RobustAlternativeEvaluation],
}

#[derive(Debug, Clone, Copy, Default)]
struct PolicyMetrics {
    rejection_power: f64,
    null_size: f64,
    expected_cost: f64,
}

fn alternative_probability(
    probabilities: &[AlternativeTransitionProbability],
    model_id: &str,
) -> Option<f64> {
    probabilities.iter()
        .find(|entry| entry.model_id == model_id)
        .map(|entry| entry.probability)
}

fn validate_alternative_probabilities(
    probabilities: &[AlternativeTransitionProbability],
    expected_models: &BTreeSet<&str>,
    path: &str,
) -> Result<(), RobustUtilityError> {
    if probabilities.len() != expected_models.len() {
        return Err(RobustUtilityError::new(
            path,
            format!(
                "expected probabilities for {} alternatives, found {}",
                expected_models.len(),
                probabilities.len()
            ),
        ));
    }
    let mut seen = BTreeSet::new();
    for (index, entry) in probabilities.iter().enumerate() {
        if entry.model_id.trim().is_empty()
            || !expected_models.contains(entry.model_id.as_str())
            || !seen.insert(entry.model_id.as_str())
        {
            return Err(RobustUtilityError::new(
                format!("{path}[{index}].model_id"),
                "alternative probability must reference one unique declared model",
            ));
        }
        validate_probability(entry.probability, &format!("{path}[{index}].probability"))?;
    }
    if seen != *expected_models {
        return Err(RobustUtilityError::new(
            path,
            "one or more declared alternative models have no transition probability",
        ));
    }
    Ok(())
}

fn validate_probability(value: f64, path: &str) -> Result<(), RobustUtilityError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(RobustUtilityError::new(
            path,
            "probability or test value must be finite and lie in [0, 1]",
        ));
    }
    Ok(())
}

fn validate_nonnegative(value: f64, path: &str) -> Result<(), RobustUtilityError> {
    if !value.is_finite() || value < 0.0 {
        return Err(RobustUtilityError::new(
            path,
            "costs and loss weights must be finite and nonnegative",
        ));
    }
    Ok(())
}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for value in values {
        kahan_add(&mut sum, &mut compensation, value);
    }
    sum
}

fn kahan_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let adjusted = value - *compensation;
    let next = *sum + adjusted;
    *compensation = (next - *sum) - adjusted;
    *sum = next;
}

fn evaluate_policy_at(
    plan: &RobustUtilityContinuationPlan,
    children: &BTreeMap<String, Vec<&RobustHistoryNode>>,
    continue_policy: &BTreeMap<String, bool>,
    node: &RobustHistoryNode,
    model_id: &str,
) -> Result<PolicyMetrics, RobustUtilityError> {
    if let Some(child_nodes) = children.get(&node.node_id) {
        if continue_policy.get(&node.node_id).copied().unwrap_or(false) {
            let mut result = PolicyMetrics {
                expected_cost: node.observation_cost,
                ..PolicyMetrics::default()
            };
            let mut power_compensation = 0.0;
            let mut size_compensation = 0.0;
            let mut cost_compensation = 0.0;
            for child in child_nodes {
                let child_metrics = evaluate_policy_at(
                    plan,
                    children,
                    continue_policy,
                    child,
                    model_id,
                )?;
                let alt_probability = alternative_probability(
                    &child.alternative_probabilities,
                    model_id,
                )
                .ok_or_else(|| RobustUtilityError::new(
                    "evaluation.alternative_probabilities",
                    format!("missing transition for alternative {model_id}"),
                ))?;
                kahan_add(
                    &mut result.rejection_power,
                    &mut power_compensation,
                    alt_probability * child_metrics.rejection_power,
                );
                kahan_add(
                    &mut result.null_size,
                    &mut size_compensation,
                    child.null_probability * child_metrics.null_size,
                );
                kahan_add(
                    &mut result.expected_cost,
                    &mut cost_compensation,
                    alt_probability * child_metrics.expected_cost,
                );
            }
            Ok(result)
        } else {
            evaluate_continuation(node, model_id)
        }
    } else {
        evaluate_continuation(node, model_id)
    }
}

fn evaluate_continuation(
    node: &RobustHistoryNode,
    model_id: &str,
) -> Result<PolicyMetrics, RobustUtilityError> {
    let mut result = PolicyMetrics {
        expected_cost: node.observation_cost + node.continuation_test.launch_cost,
        ..PolicyMetrics::default()
    };
    let mut power_compensation = 0.0;
    let mut size_compensation = 0.0;
    for outcome in &node.continuation_test.outcomes {
        let alt_probability = alternative_probability(
            &outcome.alternative_probabilities,
            model_id,
        )
        .ok_or_else(|| RobustUtilityError::new(
            "evaluation.continuation_test",
            format!("missing outcome probability for alternative {model_id}"),
        ))?;
        kahan_add(
            &mut result.rejection_power,
            &mut power_compensation,
            alt_probability * outcome.rejection_probability,
        );
        kahan_add(
            &mut result.null_size,
            &mut size_compensation,
            outcome.null_probability * outcome.rejection_probability,
        );
    }
    Ok(result)
}

#[derive(Debug, Clone)]
struct CandidatePolicy {
    mask: u64,
    null_size: f64,
    worst_case_power: f64,
    worst_case_utility: f64,
    maximum_expected_cost: f64,
    alternative_evaluations: Vec<RobustAlternativeEvaluation>,
    policy: Vec<RobustPolicyDecision>,
}

fn compare_candidate(candidate: &CandidatePolicy, incumbent: &CandidatePolicy) -> bool {
    if candidate.worst_case_utility > incumbent.worst_case_utility + OBJECTIVE_TOLERANCE {
        return true;
    }
    if (candidate.worst_case_utility - incumbent.worst_case_utility).abs()
        > OBJECTIVE_TOLERANCE
    {
        return false;
    }
    if candidate.maximum_expected_cost
        < incumbent.maximum_expected_cost - OBJECTIVE_TOLERANCE
    {
        return true;
    }
    if (candidate.maximum_expected_cost - incumbent.maximum_expected_cost).abs()
        > OBJECTIVE_TOLERANCE
    {
        return false;
    }
    if candidate.null_size < incumbent.null_size - OBJECTIVE_TOLERANCE {
        return true;
    }
    if (candidate.null_size - incumbent.null_size).abs() > OBJECTIVE_TOLERANCE {
        return false;
    }
    candidate.mask < incumbent.mask
}

pub fn analyze_robust_utility_continuation(
    input: &RobustUtilityContinuationPlan,
) -> Result<RobustUtilityContinuationReport, RobustUtilityError> {
    let plan = input.clone().canonicalize()?;
    let mut children: BTreeMap<String, Vec<&RobustHistoryNode>> = BTreeMap::new();
    for node in &plan.history_nodes {
        if let Some(parent_id) = node.parent_node_id.as_deref() {
            children.entry(parent_id.to_owned()).or_default().push(node);
        }
    }
    for child_nodes in children.values_mut() {
        child_nodes.sort_by(|a, b| a.node_id.cmp(&b.node_id));
    }

    let root = plan.history_nodes.iter()
        .find(|node| node.parent_node_id.is_none())
        .ok_or_else(|| RobustUtilityError::new("plan.history_nodes", "root history missing"))?;
    let decision_nodes = children.keys().cloned().collect::<Vec<_>>();
    let policy_count = 1u64.checked_shl(decision_nodes.len() as u32)
        .ok_or_else(|| RobustUtilityError::new(
            "plan.history_nodes",
            "policy enumeration count overflow",
        ))?;

    let mut best: Option<CandidatePolicy> = None;
    let mut feasible_count = 0u64;
    for mask in 0..policy_count {
        let continue_policy = decision_nodes.iter().enumerate()
            .map(|(bit, node_id)| (node_id.clone(), mask & (1u64 << bit) != 0))
            .collect::<BTreeMap<_, _>>();
        let mut evaluations = Vec::with_capacity(plan.alternative_models.len());
        let mut worst_utility = f64::INFINITY;
        let mut worst_power = f64::INFINITY;
        let mut maximum_cost = 0.0_f64;
        let mut common_null_size = None;

        for model in &plan.alternative_models {
            let metrics = evaluate_policy_at(
                &plan,
                &children,
                &continue_policy,
                root,
                &model.model_id,
            )?;
            if !metrics.rejection_power.is_finite()
                || !metrics.null_size.is_finite()
                || !metrics.expected_cost.is_finite()
            {
                return Err(RobustUtilityError::new(
                    "evaluation",
                    "policy reconstruction yielded a non-finite metric",
                ));
            }
            if let Some(previous) = common_null_size {
                if (previous - metrics.null_size).abs() > PROBABILITY_TOLERANCE {
                    return Err(RobustUtilityError::new(
                        "evaluation.null_size",
                        "P0 size changed across alternatives despite a shared null kernel",
                    ));
                }
            } else {
                common_null_size = Some(metrics.null_size);
            }
            let utility =
                plan.decision_loss.true_rejection_benefit * metrics.rejection_power
                - plan.decision_loss.false_negative_loss * (1.0 - metrics.rejection_power)
                - plan.decision_loss.false_positive_loss * metrics.null_size
                - plan.decision_loss.evaluator_cost_weight * metrics.expected_cost;
            if !utility.is_finite() {
                return Err(RobustUtilityError::new(
                    "evaluation.utility",
                    "decision utility is non-finite",
                ));
            }
            worst_utility = worst_utility.min(utility);
            worst_power = worst_power.min(metrics.rejection_power);
            maximum_cost = maximum_cost.max(metrics.expected_cost);
            evaluations.push(RobustAlternativeEvaluation {
                model_id: model.model_id.clone(),
                rejection_power: metrics.rejection_power,
                expected_evaluator_cost: metrics.expected_cost,
                decision_utility: utility,
            });
        }

        let null_size = common_null_size.unwrap_or(f64::NAN);
        if !null_size.is_finite() {
            return Err(RobustUtilityError::new(
                "evaluation.null_size",
                "null-size reconstruction missing",
            ));
        }
        if null_size > plan.significance_level + SUPERMARTINGALE_TOLERANCE {
            continue;
        }
        if plan.decision_loss.maximum_expected_cost
            .is_some_and(|cap| maximum_cost > cap + OBJECTIVE_TOLERANCE)
        {
            continue;
        }
        feasible_count += 1;

        let policy = plan.history_nodes.iter().map(|node| {
            let action = if children.contains_key(&node.node_id)
                && continue_policy.get(&node.node_id).copied().unwrap_or(false)
            {
                RobustPolicyAction::ContinueFirstStage
            } else {
                RobustPolicyAction::LaunchConditionalContinuation
            };
            RobustPolicyDecision {
                node_id: node.node_id.clone(),
                action,
            }
        }).collect();

        let candidate = CandidatePolicy {
            mask,
            null_size,
            worst_case_power: worst_power,
            worst_case_utility: worst_utility,
            maximum_expected_cost: maximum_cost,
            alternative_evaluations: evaluations,
            policy,
        };
        if best.as_ref().is_none_or(|incumbent| compare_candidate(&candidate, incumbent)) {
            best = Some(candidate);
        }
    }

    let best = best.ok_or_else(|| RobustUtilityError::new(
        "evaluation.feasible_policies",
        "no policy satisfies the null-size and expected-cost constraints",
    ))?;
    let plan_sha256 = plan.plan_sha256()?;
    let payload = ReportPayload {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        plan_id: &plan.plan_id,
        plan_sha256: &plan_sha256,
        significance_level: plan.significance_level,
        selected_policy_mask: best.mask,
        enumerated_policy_count: policy_count,
        feasible_policy_count: feasible_count,
        selected_policy: &best.policy,
        selected_null_size: best.null_size,
        worst_case_rejection_power: best.worst_case_power,
        worst_case_decision_utility: best.worst_case_utility,
        maximum_expected_evaluator_cost: best.maximum_expected_cost,
        remaining_alpha_budget: plan.significance_level - best.null_size,
        alternative_evaluations: &best.alternative_evaluations,
    };
    let report_sha256 = digest(&payload)?;
    Ok(RobustUtilityContinuationReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        plan_id: plan.plan_id,
        plan_sha256,
        significance_level: plan.significance_level,
        selected_policy_mask: best.mask,
        enumerated_policy_count: policy_count,
        feasible_policy_count: feasible_count,
        selected_policy: best.policy,
        selected_null_size: best.null_size,
        worst_case_rejection_power: best.worst_case_power,
        worst_case_decision_utility: best.worst_case_utility,
        maximum_expected_evaluator_cost: best.maximum_expected_cost,
        remaining_alpha_budget: plan.significance_level - best.null_size,
        alternative_evaluations: best.alternative_evaluations,
        report_sha256,
    })
}

pub fn verify_robust_utility_continuation_report(
    plan: &RobustUtilityContinuationPlan,
    report: &RobustUtilityContinuationReport,
) -> Result<(), RobustUtilityError> {
    if analyze_robust_utility_continuation(plan)? != *report {
        return Err(RobustUtilityError::new(
            "report",
            "stored report differs from robust-utility policy reconstruction",
        ));
    }
    Ok(())
}

fn digest<T: Serialize>(value: &T) -> Result<String, RobustUtilityError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        RobustUtilityError::new("serialization", error.to_string())
    })?;
    Ok(Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alt(model_id: &str, probability: f64) -> AlternativeTransitionProbability {
        AlternativeTransitionProbability {
            model_id: model_id.into(),
            probability,
        }
    }

    fn continuation(
        launch_cost: f64,
        null_hit: f64,
        alt_a_hit: f64,
        alt_b_hit: f64,
        rejection: f64,
    ) -> RobustConditionalTest {
        RobustConditionalTest {
            launch_cost,
            outcomes: vec![
                RobustContinuationOutcome {
                    outcome_id: "hit".into(),
                    null_probability: null_hit,
                    alternative_probabilities: vec![
                        alt("alt-a", alt_a_hit),
                        alt("alt-b", alt_b_hit),
                    ],
                    rejection_probability: rejection,
                },
                RobustContinuationOutcome {
                    outcome_id: "miss".into(),
                    null_probability: 1.0 - null_hit,
                    alternative_probabilities: vec![
                        alt("alt-a", 1.0 - alt_a_hit),
                        alt("alt-b", 1.0 - alt_b_hit),
                    ],
                    rejection_probability: 0.0,
                },
            ],
        }
    }

    fn fixture() -> RobustUtilityContinuationPlan {
        RobustUtilityContinuationPlan::new(
            "robust-alternative-decision-utility",
            0.05,
            vec![
                RobustAlternativeModel { model_id: "alt-b".into() },
                RobustAlternativeModel { model_id: "alt-a".into() },
            ],
            RobustDecisionLoss {
                true_rejection_benefit: 10.0,
                false_negative_loss: 2.0,
                false_positive_loss: 100.0,
                evaluator_cost_weight: 0.1,
                maximum_expected_cost: None,
            },
            vec![
                RobustHistoryNode {
                    node_id: "root".into(),
                    parent_node_id: None,
                    null_probability: 1.0,
                    alternative_probabilities: vec![alt("alt-a", 1.0), alt("alt-b", 1.0)],
                    interim_test_value: 0.05,
                    observation_cost: 0.0,
                    continuation_test: continuation(2.0, 0.05, 0.5, 0.2, 1.0),
                },
                RobustHistoryNode {
                    node_id: "early".into(),
                    parent_node_id: Some("root".into()),
                    null_probability: 0.5,
                    alternative_probabilities: vec![alt("alt-a", 0.7), alt("alt-b", 0.3)],
                    interim_test_value: 0.08,
                    observation_cost: 1.0,
                    continuation_test: continuation(2.0, 0.06, 0.9, 0.4, 1.0),
                },
                RobustHistoryNode {
                    node_id: "late".into(),
                    parent_node_id: Some("root".into()),
                    null_probability: 0.5,
                    alternative_probabilities: vec![alt("alt-a", 0.3), alt("alt-b", 0.7)],
                    interim_test_value: 0.02,
                    observation_cost: 1.0,
                    continuation_test: continuation(2.0, 0.01, 0.1, 0.8, 1.0),
                },
                RobustHistoryNode {
                    node_id: "early-good".into(),
                    parent_node_id: Some("early".into()),
                    null_probability: 0.5,
                    alternative_probabilities: vec![alt("alt-a", 0.8), alt("alt-b", 0.2)],
                    interim_test_value: 0.05,
                    observation_cost: 1.0,
                    continuation_test: continuation(2.0, 0.05, 0.95, 0.25, 1.0),
                },
                RobustHistoryNode {
                    node_id: "early-bad".into(),
                    parent_node_id: Some("early".into()),
                    null_probability: 0.5,
                    alternative_probabilities: vec![alt("alt-a", 0.2), alt("alt-b", 0.8)],
                    interim_test_value: 0.02,
                    observation_cost: 1.0,
                    continuation_test: continuation(2.0, 0.02, 0.1, 0.75, 1.0),
                },
            ],
        ).unwrap()
    }

    #[test]
    fn robust_policy_optimizes_worst_case_utility_and_preserves_null_size() {
        let plan = fixture();
        let report = analyze_robust_utility_continuation(&plan).unwrap();
        assert_eq!(report.enumerated_policy_count, 4);
        assert_eq!(report.feasible_policy_count, 4);
        assert!((report.selected_null_size - 0.0225).abs() < 1e-12);
        assert!((report.worst_case_rejection_power - 0.576).abs() < 1e-12);
        assert!((report.maximum_expected_evaluator_cost - 3.7).abs() < 1e-12);
        assert!((report.worst_case_decision_utility - 2.292).abs() < 1e-12);
        let root = report.selected_policy.iter().find(|row| row.node_id == "root").unwrap();
        let early = report.selected_policy.iter().find(|row| row.node_id == "early").unwrap();
        assert_eq!(root.action, RobustPolicyAction::ContinueFirstStage);
        assert_eq!(early.action, RobustPolicyAction::ContinueFirstStage);
        verify_robust_utility_continuation_report(&plan, &report).unwrap();
    }

    #[test]
    fn hard_evaluator_cost_cap_selects_the_cheaper_feasible_policy() {
        let mut plan = fixture();
        plan.decision_loss.maximum_expected_cost = Some(3.5);
        let report = analyze_robust_utility_continuation(&plan).unwrap();
        assert!((report.maximum_expected_evaluator_cost - 3.0).abs() < 1e-12);
        assert!((report.selected_null_size - 0.035).abs() < 1e-12);
        let early = report.selected_policy.iter().find(|row| row.node_id == "early").unwrap();
        assert_eq!(early.action, RobustPolicyAction::LaunchConditionalContinuation);
    }

    #[test]
    fn alternative_misspecification_changes_policy_not_null_certificate() {
        let mut plan = fixture();
        let good = plan.history_nodes.iter_mut()
            .find(|node| node.node_id == "early-good").unwrap();
        good.alternative_probabilities.iter_mut()
            .find(|row| row.model_id == "alt-a").unwrap().probability = 0.1;
        let bad = plan.history_nodes.iter_mut()
            .find(|node| node.node_id == "early-bad").unwrap();
        bad.alternative_probabilities.iter_mut()
            .find(|row| row.model_id == "alt-a").unwrap().probability = 0.9;
        let report = analyze_robust_utility_continuation(&plan).unwrap();
        let early = report.selected_policy.iter().find(|row| row.node_id == "early").unwrap();
        assert_eq!(early.action, RobustPolicyAction::LaunchConditionalContinuation);
        assert!((report.selected_null_size - 0.035).abs() < 1e-12);
        assert!(report.selected_null_size <= report.significance_level);
    }

    #[test]
    fn no_policy_under_cost_cap_fails_closed() {
        let mut plan = fixture();
        plan.decision_loss.maximum_expected_cost = Some(1.0);
        assert!(analyze_robust_utility_continuation(&plan).is_err());
    }

    #[test]
    fn malformed_null_alternative_mass_and_conditional_budgets_fail_closed() {
        let mut null_mass = fixture();
        null_mass.history_nodes.iter_mut()
            .find(|node| node.node_id == "early").unwrap()
            .null_probability = 0.4;
        assert!(null_mass.validate().is_err());

        let mut alt_mass = fixture();
        alt_mass.history_nodes.iter_mut()
            .find(|node| node.node_id == "early").unwrap()
            .alternative_probabilities.iter_mut()
            .find(|row| row.model_id == "alt-b").unwrap().probability = 0.4;
        assert!(alt_mass.validate().is_err());

        let mut budget = fixture();
        budget.history_nodes.iter_mut()
            .find(|node| node.node_id == "early").unwrap()
            .continuation_test.outcomes[0].rejection_probability = 0.9;
        assert!(budget.validate().is_err());
    }

    #[test]
    fn missing_duplicate_models_support_and_bad_costs_fail_closed() {
        let mut missing = fixture();
        missing.history_nodes[1].alternative_probabilities.pop();
        assert!(missing.validate().is_err());

        let mut duplicate = fixture();
        duplicate.alternative_models[1].model_id = "alt-a".into();
        assert!(duplicate.validate().is_err());

        let mut cost = fixture();
        cost.history_nodes[1].observation_cost = f64::NAN;
        assert!(cost.validate().is_err());
    }

    #[test]
    fn canonical_order_and_report_tampering_are_detected() {
        let plan = fixture();
        let report = analyze_robust_utility_continuation(&plan).unwrap();
        let mut reordered = plan.clone();
        reordered.alternative_models.reverse();
        reordered.history_nodes.reverse();
        for node in &mut reordered.history_nodes {
            node.alternative_probabilities.reverse();
            node.continuation_test.outcomes.reverse();
            for outcome in &mut node.continuation_test.outcomes {
                outcome.alternative_probabilities.reverse();
            }
        }
        assert_eq!(report, analyze_robust_utility_continuation(&reordered).unwrap());
        let mut altered = report.clone();
        altered.selected_null_size += 0.01;
        assert!(verify_robust_utility_continuation_report(&plan, &altered).is_err());
    }
}
