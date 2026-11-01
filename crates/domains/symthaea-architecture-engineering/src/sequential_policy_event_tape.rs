//! Bind an adaptive finite-policy recommendation to an exact EventTape prefix.
//!
//! This module checks structural tape validity, posterior/model identity bindings and
//! deterministic policy reconstruction. It does not prove the inference receipt is authentic
//! or that its producer computed the posterior correctly; that requires artifact resolution
//! and an independently authenticated inference verifier.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{error::Error, fmt};

use super::{
    event_tape::EventTape,
    pareto_evsi_oracle::ExactEvsiProblem,
    pareto_sequential_policy::{verify_sequential_policy, SequentialDecisionPolicy},
};

const SCHEMA: &str = "symthaea.architecture-engineering.eventtape-sequential-recommendation";
const VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecommendationBindingError { pub path: String, pub message: String }
impl RecommendationBindingError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}
impl fmt::Display for RecommendationBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}
impl Error for RecommendationBindingError {}

/// Evidence identity for the posterior consumed by the sequential solver.
/// The inference receipt digest is a commitment, not a signature or proof of correctness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialPosteriorBinding {
    pub visible_history_sha256: String,
    pub posterior_model_sha256: String,
    pub posterior_distribution_sha256: String,
    pub inference_receipt_sha256: String,
}

impl SequentialPosteriorBinding {
    pub fn new(
        tape: &EventTape,
        problem: &ExactEvsiProblem,
        inference_receipt_sha256: impl Into<String>,
    ) -> Result<Self, RecommendationBindingError> {
        tape.validate().map_err(|e| RecommendationBindingError::new(
            "event_tape", e.to_string()
        ))?;
        let problem = problem.clone().canonicalize().map_err(|e| {
            RecommendationBindingError::new("problem", e.to_string())
        })?;
        if tape.header().campaign_id != problem.campaign_id {
            return Err(RecommendationBindingError::new(
                "problem.campaign_id",
                "posterior problem campaign differs from the EventTape campaign",
            ));
        }
        let binding = Self {
            visible_history_sha256: digest(tape.events())?,
            posterior_model_sha256: problem.posterior_model_sha256.clone(),
            posterior_distribution_sha256: digest(&problem.worlds)?,
            inference_receipt_sha256: inference_receipt_sha256.into(),
        };
        binding.validate()?;
        Ok(binding)
    }

    pub fn validate(&self) -> Result<(), RecommendationBindingError> {
        validate_sha(&self.visible_history_sha256, "posterior_binding.visible_history_sha256")?;
        validate_sha(&self.posterior_model_sha256, "posterior_binding.posterior_model_sha256")?;
        validate_sha(&self.posterior_distribution_sha256, "posterior_binding.posterior_distribution_sha256")?;
        validate_sha(&self.inference_receipt_sha256, "posterior_binding.inference_receipt_sha256")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialPolicyRecommendation {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub source_commit: String,
    pub policy_id: String,
    pub policy_algorithm_id: String,
    pub observed_through_sequence: u64,
    pub visible_history_sha256: String,
    pub event_tape_header_sha256: String,
    pub last_event_sha256: Option<String>,
    pub objective_contract_sha256: Option<String>,
    pub problem_sha256: String,
    pub policy_sha256: String,
    pub posterior_binding: SequentialPosteriorBinding,
    pub budget_units: u64,
    pub remaining_budget_units: u64,
    pub selected_action_id: Option<String>,
    pub terminal_decision_id: Option<String>,
    pub expected_terminal_bayes_risk: f64,
    pub expected_additional_cost_units: f64,
    pub worst_case_additional_cost_units: u64,
    pub recommendation_sha256: String,
}

#[derive(Serialize)]
struct RecommendationDigestPayload<'a> {
    schema: &'a str,
    version: &'a str,
    campaign_id: &'a str,
    source_commit: &'a str,
    policy_id: &'a str,
    policy_algorithm_id: &'a str,
    observed_through_sequence: u64,
    visible_history_sha256: &'a str,
    event_tape_header_sha256: &'a str,
    last_event_sha256: &'a Option<String>,
    objective_contract_sha256: &'a Option<String>,
    problem_sha256: &'a str,
    policy_sha256: &'a str,
    posterior_binding: &'a SequentialPosteriorBinding,
    budget_units: u64,
    remaining_budget_units: u64,
    selected_action_id: &'a Option<String>,
    terminal_decision_id: &'a Option<String>,
    expected_terminal_bayes_risk: f64,
    expected_additional_cost_units: f64,
    worst_case_additional_cost_units: u64,
}

/// Freeze a solver recommendation against the exact, fully-metered EventTape prefix.
///
/// The problem's posterior distribution must have been inferred from the visible prefix.
/// This API checks that the caller-supplied posterior binding names that exact prefix,
/// model digest, posterior distribution digest, and an inference receipt digest. It does
/// not authenticate or resolve the inference receipt itself.
pub fn bind_sequential_recommendation(
    tape: &EventTape,
    problem: &ExactEvsiProblem,
    policy: &SequentialDecisionPolicy,
    posterior_binding: &SequentialPosteriorBinding,
) -> Result<SequentialPolicyRecommendation, RecommendationBindingError> {
    tape.validate().map_err(|e| RecommendationBindingError::new(
        "event_tape", e.to_string()
    ))?;
    let canonical = problem.clone().canonicalize().map_err(|e| {
        RecommendationBindingError::new("problem", e.to_string())
    })?;
    verify_sequential_policy(&canonical, policy).map_err(|e| {
        RecommendationBindingError::new("policy", e.to_string())
    })?;
    posterior_binding.validate()?;

    if tape.header().campaign_id != canonical.campaign_id
        || policy.campaign_id != canonical.campaign_id
    {
        return Err(RecommendationBindingError::new(
            "campaign_id",
            "EventTape, finite problem, and sequential policy campaign IDs must match",
        ));
    }
    let visible_history_sha256 = digest(tape.events())?;
    let posterior_distribution_sha256 = digest(&canonical.worlds)?;
    if posterior_binding.visible_history_sha256 != visible_history_sha256 {
        return Err(RecommendationBindingError::new(
            "posterior_binding.visible_history_sha256",
            "posterior inference receipt is not bound to the exact current visible EventTape prefix",
        ));
    }
    if posterior_binding.posterior_model_sha256 != canonical.posterior_model_sha256 {
        return Err(RecommendationBindingError::new(
            "posterior_binding.posterior_model_sha256",
            "posterior model digest differs from the finite decision problem",
        ));
    }
    if posterior_binding.posterior_distribution_sha256 != posterior_distribution_sha256 {
        return Err(RecommendationBindingError::new(
            "posterior_binding.posterior_distribution_sha256",
            "posterior distribution digest differs from the finite decision problem",
        ));
    }

    if let Some(action_id) = policy.root.selected_action_id.as_deref() {
        if !canonical.actions.iter().any(|action| action.action_id == action_id)
            || !policy.root.remaining_action_ids.iter().any(|available| available == action_id)
        {
            return Err(RecommendationBindingError::new(
                "policy.selected_action_id",
                "selected action is not available in the frozen finite problem",
            ));
        }
    } else if policy.root.terminal_decision_id.is_none() {
        return Err(RecommendationBindingError::new(
            "policy.root",
            "policy must select a measurement action or an explicit terminal decision",
        ));
    }

    let event_count = u64::try_from(tape.events().len()).map_err(|_| {
        RecommendationBindingError::new("event_tape.events", "event count does not fit u64")
    })?;
    let last_event_sha256 = tape.events().last().map(|event| event.event_sha256.clone());
    let objective_contract_sha256 = tape.header().objective_contract_sha256.clone();
    let problem_sha256 = canonical.problem_sha256().map_err(|e| {
        RecommendationBindingError::new("problem", e.to_string())
    })?;
    let event_tape_header_sha256 = digest(tape.header())?;

    let payload = RecommendationDigestPayload {
        schema: SCHEMA, version: VERSION,
        campaign_id: &canonical.campaign_id,
        source_commit: &tape.header().source_commit,
        policy_id: &tape.header().policy.policy_id,
        policy_algorithm_id: &tape.header().policy.algorithm_id,
        observed_through_sequence: event_count,
        visible_history_sha256: &visible_history_sha256,
        event_tape_header_sha256: &event_tape_header_sha256,
        last_event_sha256: &last_event_sha256,
        objective_contract_sha256: &objective_contract_sha256,
        problem_sha256: &problem_sha256,
        policy_sha256: &policy.policy_sha256,
        posterior_binding,
        budget_units: policy.budget_units,
        remaining_budget_units: policy.root.remaining_budget_units,
        selected_action_id: &policy.root.selected_action_id,
        terminal_decision_id: &policy.root.terminal_decision_id,
        expected_terminal_bayes_risk: policy.expected_terminal_bayes_risk,
        expected_additional_cost_units: policy.expected_total_cost_units,
        worst_case_additional_cost_units: policy.worst_case_total_cost_units,
    };
    let recommendation_sha256 = digest(&payload)?;

    Ok(SequentialPolicyRecommendation {
        schema: SCHEMA.into(), version: VERSION.into(),
        campaign_id: canonical.campaign_id,
        source_commit: tape.header().source_commit.clone(),
        policy_id: tape.header().policy.policy_id.clone(),
        policy_algorithm_id: tape.header().policy.algorithm_id.clone(),
        observed_through_sequence: event_count,
        visible_history_sha256,
        event_tape_header_sha256,
        last_event_sha256,
        objective_contract_sha256,
        problem_sha256,
        policy_sha256: policy.policy_sha256.clone(),
        posterior_binding: posterior_binding.clone(),
        budget_units: policy.budget_units,
        remaining_budget_units: policy.root.remaining_budget_units,
        selected_action_id: policy.root.selected_action_id.clone(),
        terminal_decision_id: policy.root.terminal_decision_id.clone(),
        expected_terminal_bayes_risk: policy.expected_terminal_bayes_risk,
        expected_additional_cost_units: policy.expected_total_cost_units,
        worst_case_additional_cost_units: policy.worst_case_total_cost_units,
        recommendation_sha256,
    })
}

/// Reject a recommendation if the tape prefix, posterior, model, policy, or header changed.
pub fn verify_sequential_recommendation(
    tape: &EventTape,
    problem: &ExactEvsiProblem,
    policy: &SequentialDecisionPolicy,
    posterior_binding: &SequentialPosteriorBinding,
    recommendation: &SequentialPolicyRecommendation,
) -> Result<(), RecommendationBindingError> {
    let reconstructed = bind_sequential_recommendation(
        tape, problem, policy, posterior_binding
    )?;
    if reconstructed != *recommendation {
        return Err(RecommendationBindingError::new(
            "recommendation",
            "recommendation differs from reconstruction at the exact EventTape prefix",
        ));
    }
    Ok(())
}

fn validate_sha(value: &str, path: &str) -> Result<(), RecommendationBindingError> {
    if value.len() != 64 || !value.bytes().all(|b| {
        b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
    }) {
        return Err(RecommendationBindingError::new(path, "expected lowercase 64-character SHA-256"));
    }
    Ok(())
}

fn digest<T: Serialize>(value: &T) -> Result<String, RecommendationBindingError> {
    let bytes = serde_json::to_vec(value).map_err(|e| {
        RecommendationBindingError::new("serialization", e.to_string())
    })?;
    Ok(Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        event_tape::{
            ActionEvent, ActionIntent, ActionOutcome, BudgetModel, CandidateDisposition,
            ComparisonMode, EventTapeHeader, ExecutionContract, ExecutionMode, MeterValue,
            PolicySnapshot, ResourceReceipt,
        },
        pareto_evsi_oracle::{
            EvsiActionOutcome, EvsiDecision, EvsiReplicateAction, EvsiWorldLikelihood,
            EvsiWorldLoss, EvsiWorldProbability,
        },
        pareto_sequential_policy::solve_sequential_policy,
    };
    use serde_json::json;

    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const H5: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn finite_problem() -> ExactEvsiProblem {
        ExactEvsiProblem {
            schema: "symthaea.architecture-engineering.exact-discrete-evsi-problem".into(),
            version: "0.1.0".into(), campaign_id: "campaign-binding-test".into(),
            posterior_model_id: "frozen-posterior".into(), posterior_model_sha256: H1.into(),
            observation_model_id: "frozen-observation".into(), observation_model_sha256: H2.into(),
            decision_loss_model_id: "terminal-loss".into(), decision_loss_model_sha256: H3.into(),
            cost_model_id: "action-cost".into(), cost_model_sha256: H4.into(),
            worlds: vec![
                EvsiWorldProbability { world_id: "world-a".into(), posterior_probability: 0.5 },
                EvsiWorldProbability { world_id: "world-b".into(), posterior_probability: 0.5 }],
            decisions: vec![
                EvsiDecision { decision_id: "choose-a".into(), losses_by_world: vec![
                    EvsiWorldLoss { world_id: "world-a".into(), loss: 0.0 },
                    EvsiWorldLoss { world_id: "world-b".into(), loss: 10.0 }]},
                EvsiDecision { decision_id: "choose-b".into(), losses_by_world: vec![
                    EvsiWorldLoss { world_id: "world-a".into(), loss: 10.0 },
                    EvsiWorldLoss { world_id: "world-b".into(), loss: 0.0 }]}],
            actions: vec![EvsiReplicateAction { action_id: "measure-a".into(),
                candidate_id: "candidate-a".into(), objective_id: "quality".into(),
                incremental_cost_units: 1, outcomes: vec![
                    EvsiActionOutcome { outcome_id: "a".into(), likelihoods_by_world: vec![
                        EvsiWorldLikelihood { world_id: "world-a".into(), probability: 1.0 },
                        EvsiWorldLikelihood { world_id: "world-b".into(), probability: 0.0 }]},
                    EvsiActionOutcome { outcome_id: "b".into(), likelihoods_by_world: vec![
                        EvsiWorldLikelihood { world_id: "world-a".into(), probability: 0.0 },
                        EvsiWorldLikelihood { world_id: "world-b".into(), probability: 1.0 }]}]}],
        }.canonicalize().unwrap()
    }

    fn header() -> EventTapeHeader {
        let config = json!({"controller": "sequential-reference-v0.1"});
        EventTapeHeader {
            schema: "symthaea.ar5b4.event-tape".into(), version: "0.2.0".into(),
            campaign_id: "campaign-binding-test".into(),
            source_commit: "0123456789abcdef0123456789abcdef01234567".into(),
            execution_environment_sha256: H5.into(),
            comparison_mode: ComparisonMode::MechanismMatched,
            proposal_source_id: "fixed-proposals-v1".into(),
            proposal_stream_sha256: Some(H5.into()),
            task_recipe_sha256: H1.into(), comparison_profile_sha256: H2.into(),
            evaluator_sha256: H3.into(), verifier_sha256: H4.into(),
            budget_model_id: "one-unit-per-action".into(),
            budget_model: BudgetModel::ActionUnit { charge_per_action_microunits: 1 },
            budget_model_sha256: digest(&BudgetModel::ActionUnit { charge_per_action_microunits: 1 }).unwrap(),
            budget_limit_microunits: 100,
            initial_direction_ids: vec!["A".into()],
            execution: ExecutionContract { mode: ExecutionMode::Sequential, worker_count: 1,
                seats_per_round: 1, first_round_claim_cap: None, steady_state_claim_cap: None },
            policy: PolicySnapshot { policy_id: "sequential-controller".into(),
                algorithm_id: "finite-dp-v0.1".into(), config: config.clone(),
                config_sha256: digest(&config).unwrap(), has_hidden_oracle_access: false },
            objective_contract: None, objective_contract_sha256: None,
        }
    }

    fn resource_receipt() -> ResourceReceipt {
        let measured = || MeterValue::Measured { value: 0 };
        ResourceReceipt {
            host_cpu_millis: measured(), wall_millis: measured(), accelerator_millis: measured(),
            peak_memory_bytes: measured(), accelerator_peak_memory_bytes: measured(),
            energy_millijoules: measured(), input_tokens: measured(), output_tokens: measured(),
            tool_calls: measured(), evaluator_calls: MeterValue::Measured { value: 1 },
            verifier_calls: MeterValue::Measured { value: 1 },
            charged_budget_microunits: MeterValue::Measured { value: 1 },
        }
    }

    fn append_candidate_event(tape: &mut EventTape, id: &str, candidate: &str, round: u64) {
        tape.append(ActionEvent {
            event_id: id.into(), policy_id: "sequential-controller".into(),
            worker_id: 0, round_id: round, batch_id: None, observed_through_sequence: round - 1,
            visible_history_sha256: String::new(), retry_of_event_id: None,
            intent: ActionIntent::EvaluateCandidate {
                candidate_sha256: candidate.into(), parent_candidate_sha256: None,
                semantic_delta_sha256: H2.into(), direction_id: "A".into(),
            },
            outcome: ActionOutcome::CandidateEvaluation {
                task_recipe_sha256: H1.into(), comparison_profile_sha256: H2.into(),
                evaluator_sha256: H3.into(), verifier_sha256: H4.into(),
                verifier_receipt_sha256: H5.into(), disposition: CandidateDisposition::Verified,
                score: Some(0.9), uncertainty: Some(0.01), evidence_sha256: Some(H5.into()),
            },
            resources: resource_receipt(),
        }).unwrap();
    }

    fn make_binding(tape: &EventTape, problem: &ExactEvsiProblem) -> SequentialPosteriorBinding {
        SequentialPosteriorBinding::new(tape, problem, H5).unwrap()
    }

    #[test]
    fn recommendation_is_bound_to_exact_fully_metered_event_tape_prefix() {
        let mut tape = EventTape::new(header()).unwrap();
        append_candidate_event(&mut tape, "event-1", H1, 1);
        let problem = finite_problem();
        let policy = solve_sequential_policy(&problem, 1).unwrap();
        let binding = make_binding(&tape, &problem);
        let recommendation = bind_sequential_recommendation(&tape, &problem, &policy, &binding).unwrap();
        assert_eq!(recommendation.observed_through_sequence, 1);
        assert_eq!(recommendation.selected_action_id.as_deref(), Some("measure-a"));
        verify_sequential_recommendation(&tape, &problem, &policy, &binding, &recommendation).unwrap();

        append_candidate_event(&mut tape, "event-2", H2, 2);
        assert!(verify_sequential_recommendation(&tape, &problem, &policy, &binding, &recommendation).is_err());
    }

    #[test]
    fn posterior_binding_from_another_visible_prefix_is_rejected() {
        let mut first = EventTape::new(header()).unwrap();
        append_candidate_event(&mut first, "event-1", H1, 1);
        let mut second = EventTape::new(header()).unwrap();
        append_candidate_event(&mut second, "event-1", H2, 1);
        let problem = finite_problem();
        let policy = solve_sequential_policy(&problem, 1).unwrap();
        let stale = make_binding(&first, &problem);
        assert!(bind_sequential_recommendation(&second, &problem, &policy, &stale).is_err());
    }

    #[test]
    fn altered_inference_receipt_digest_is_rejected() {
        let mut tape = EventTape::new(header()).unwrap();
        append_candidate_event(&mut tape, "event-1", H1, 1);
        let problem = finite_problem();
        let policy = solve_sequential_policy(&problem, 1).unwrap();
        let binding = make_binding(&tape, &problem);
        let mut recommendation = bind_sequential_recommendation(&tape, &problem, &policy, &binding).unwrap();
        recommendation.posterior_binding.inference_receipt_sha256 = H4.into();
        assert!(verify_sequential_recommendation(&tape, &problem, &policy, &binding, &recommendation).is_err());
    }
}
