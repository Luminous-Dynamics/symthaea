//! Fail-closed mapping from sequential policy action IDs to EventTape intents.
//!
//! This module authorizes exactly one next measurement or explicit stop. The action map
//! binds solver action IDs to candidate/objective/replicate identity and a declared integer
//! cost crosswalk. It does not execute evaluators or authenticate the external dispatcher.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, fmt};

use super::{
    event_tape::{
        ActionIntent, EventTape, EventTapeObjectiveMode,
    },
    pareto_evsi_oracle::ExactEvsiProblem,
    pareto_sequential_policy::{verify_sequential_policy, SequentialDecisionPolicy},
    sequential_policy_event_tape::{
        verify_sequential_recommendation, SequentialPolicyRecommendation,
        SequentialPosteriorBinding,
    },
};

const SCHEMA: &str = "symthaea.architecture-engineering.sequential-action-dispatch";
const VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequentialDispatchError { pub path: String, pub message: String }
impl SequentialDispatchError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}
impl fmt::Display for SequentialDispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}
impl Error for SequentialDispatchError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequentialMeasurementKind {
    DeterministicVector,
    NoisyReplicate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialActionDescriptor {
    pub action_id: String,
    pub candidate_id: String,
    pub objective_id: String,
    pub kind: SequentialMeasurementKind,
    pub candidate_sha256: String,
    pub parent_candidate_sha256: Option<String>,
    pub semantic_delta_sha256: String,
    pub direction_id: String,
    pub objective_contract_sha256: String,
    pub decision_sha256: Option<String>,
    pub replicate_id: Option<String>,
    pub sample_seed: Option<u64>,
    /// Must equal the solver cost units times cost_unit_microunits.
    pub declared_incremental_cost_microunits: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialActionMappingContract {
    pub schema: String,
    pub version: String,
    pub mapping_id: String,
    /// Identity of the integer cost unit used by ExactEvsiProblem.
    pub decision_cost_model_id: String,
    pub decision_cost_model_sha256: String,
    /// Identity of EventTape's normalized budget formula.
    pub event_tape_budget_model_sha256: String,
    /// One solver cost unit equals this many EventTape budget microunits.
    pub cost_unit_microunits: u64,
    /// Exactly one mapping for every action in the finite decision problem.
    pub actions: Vec<SequentialActionDescriptor>,
}

impl SequentialActionMappingContract {
    pub fn validate(&self, problem: &ExactEvsiProblem) -> Result<(), SequentialDispatchError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(SequentialDispatchError::new("mapping.schema", "unsupported dispatch mapping schema/version"));
        }
        if self.mapping_id.trim().is_empty() || self.decision_cost_model_id.trim().is_empty() {
            return Err(SequentialDispatchError::new("mapping", "mapping and decision cost model IDs must be non-empty"));
        }
        validate_sha(&self.decision_cost_model_sha256, "mapping.decision_cost_model_sha256")?;
        validate_sha(&self.event_tape_budget_model_sha256, "mapping.event_tape_budget_model_sha256")?;
        if self.decision_cost_model_id != problem.cost_model_id
            || self.decision_cost_model_sha256 != problem.cost_model_sha256
        {
            return Err(SequentialDispatchError::new("mapping.decision_cost_model", "cost model identity differs from the finite decision problem"));
        }
        if self.cost_unit_microunits == 0 {
            return Err(SequentialDispatchError::new("mapping.cost_unit_microunits", "cost conversion must be positive"));
        }
        if self.actions.len() != problem.actions.len() {
            return Err(SequentialDispatchError::new("mapping.actions", "mapping must cover every finite problem action exactly once"));
        }

        let mut action_ids = BTreeSet::new();
        let mut deterministic_keys = BTreeSet::new();
        let mut replicate_keys = BTreeSet::new();
        for (index, descriptor) in self.actions.iter().enumerate() {
            let path = format!("mapping.actions[{index}]");
            if descriptor.action_id.trim().is_empty()
                || descriptor.candidate_id.trim().is_empty()
                || descriptor.objective_id.trim().is_empty()
                || descriptor.direction_id.trim().is_empty()
            {
                return Err(SequentialDispatchError::new(&path, "action, candidate, objective and direction IDs must be non-empty"));
            }
            if !action_ids.insert(descriptor.action_id.as_str()) {
                return Err(SequentialDispatchError::new(format!("{path}.action_id"), "action IDs must be unique"));
            }
            validate_sha(&descriptor.candidate_sha256, &format!("{path}.candidate_sha256"))?;
            validate_sha(&descriptor.semantic_delta_sha256, &format!("{path}.semantic_delta_sha256"))?;
            validate_sha(&descriptor.objective_contract_sha256, &format!("{path}.objective_contract_sha256"))?;
            if let Some(parent) = descriptor.parent_candidate_sha256.as_deref() {
                validate_sha(parent, &format!("{path}.parent_candidate_sha256"))?;
            }
            let problem_action = problem.actions.iter().find(|action| action.action_id == descriptor.action_id)
                .ok_or_else(|| SequentialDispatchError::new(format!("{path}.action_id"), "mapping references an unknown action"))?;
            if descriptor.candidate_id != problem_action.candidate_id
                || descriptor.objective_id != problem_action.objective_id
            {
                return Err(SequentialDispatchError::new(&path, "candidate/objective identity differs from the finite problem action"));
            }
            let expected_cost = problem_action.incremental_cost_units
                .checked_mul(self.cost_unit_microunits)
                .ok_or_else(|| SequentialDispatchError::new(format!("{path}.declared_incremental_cost_microunits"), "cost conversion overflow"))?;
            if expected_cost != descriptor.declared_incremental_cost_microunits {
                return Err(SequentialDispatchError::new(format!("{path}.declared_incremental_cost_microunits"), "declared EventTape charge differs from converted solver cost"));
            }

            match descriptor.kind {
                SequentialMeasurementKind::DeterministicVector => {
                    if descriptor.decision_sha256.is_some() || descriptor.replicate_id.is_some() || descriptor.sample_seed.is_some() {
                        return Err(SequentialDispatchError::new(&path, "deterministic-vector actions cannot carry noisy replicate identity"));
                    }
                    let key = (descriptor.candidate_sha256.as_str(), descriptor.objective_contract_sha256.as_str(), descriptor.direction_id.as_str());
                    if !deterministic_keys.insert(key) {
                        return Err(SequentialDispatchError::new(&path, "deterministic candidate/vector identity is mapped more than once"));
                    }
                }
                SequentialMeasurementKind::NoisyReplicate => {
                    let decision = descriptor.decision_sha256.as_deref().ok_or_else(|| SequentialDispatchError::new(
                        format!("{path}.decision_sha256"), "noisy replicate requires decision identity",
                    ))?;
                    validate_sha(decision, &format!("{path}.decision_sha256"))?;
                    let replicate = descriptor.replicate_id.as_deref().filter(|value| !value.trim().is_empty())
                        .ok_or_else(|| SequentialDispatchError::new(format!("{path}.replicate_id"), "noisy replicate requires a non-empty replicate ID"))?;
                    let seed = descriptor.sample_seed.ok_or_else(|| SequentialDispatchError::new(
                        format!("{path}.sample_seed"), "noisy replicate requires a seed",
                    ))?;
                    let key = (descriptor.candidate_sha256.as_str(), descriptor.objective_contract_sha256.as_str(), decision, replicate, seed);
                    if !replicate_keys.insert(key) {
                        return Err(SequentialDispatchError::new(&path, "noisy replicate identity is mapped more than once"));
                    }
                }
            }
        }
        for action in &problem.actions {
            if !action_ids.contains(action.action_id.as_str()) {
                return Err(SequentialDispatchError::new("mapping.actions", format!("missing mapping for action {}", action.action_id)));
            }
        }
        Ok(())
    }

    pub fn mapping_sha256(&self) -> Result<String, SequentialDispatchError> {
        digest(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialDispatchAuthorization {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub source_commit: String,
    pub policy_id: String,
    pub mapping_id: String,
    pub mapping_sha256: String,
    pub recommendation_sha256: String,
    pub visible_history_sha256: String,
    pub observed_through_sequence: u64,
    pub selected_action_id: String,
    pub intent: ActionIntent,
    pub expected_charge_microunits: u64,
    pub remaining_budget_microunits: u64,
    pub authorization_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialStopAuthorization {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub source_commit: String,
    pub policy_id: String,
    pub recommendation_sha256: String,
    pub visible_history_sha256: String,
    pub observed_through_sequence: u64,
    pub terminal_decision_id: String,
    pub budget_units: u64,
    pub stop_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum AuthorizedSequentialDecision {
    Measure { authorization: SequentialDispatchAuthorization },
    Stop { authorization: SequentialStopAuthorization },
}

#[derive(Serialize)]
struct DispatchDigestPayload<'a> {
    schema: &'a str, version: &'a str, campaign_id: &'a str,
    source_commit: &'a str, policy_id: &'a str, mapping_id: &'a str,
    mapping_sha256: &'a str, recommendation_sha256: &'a str,
    visible_history_sha256: &'a str, observed_through_sequence: u64,
    selected_action_id: &'a str, intent: &'a ActionIntent,
    expected_charge_microunits: u64, remaining_budget_microunits: u64,
}

#[derive(Serialize)]
struct StopDigestPayload<'a> {
    schema: &'a str, version: &'a str, campaign_id: &'a str,
    source_commit: &'a str, policy_id: &'a str,
    recommendation_sha256: &'a str, visible_history_sha256: &'a str,
    observed_through_sequence: u64, terminal_decision_id: &'a str,
    budget_units: u64,
}

/// Produce one admissible next action or an explicit terminal stop decision.
///
/// The mapping contract is a caller-supplied, digest-bound crosswalk between finite
/// solver action IDs and EventTape intent identities. The declared conversion unit is
/// checked against remaining EventTape budget, but its economic correctness still depends
/// on independent qualification of the cost-model crosswalk.
pub fn authorize_sequential_decision(
    tape: &EventTape,
    problem: &ExactEvsiProblem,
    policy: &SequentialDecisionPolicy,
    posterior_binding: &SequentialPosteriorBinding,
    recommendation: &SequentialPolicyRecommendation,
    mapping: &SequentialActionMappingContract,
) -> Result<AuthorizedSequentialDecision, SequentialDispatchError> {
    verify_sequential_recommendation(tape, problem, policy, posterior_binding, recommendation)
        .map_err(|error| SequentialDispatchError::new("recommendation", error.to_string()))?;
    verify_sequential_policy(problem, policy)
        .map_err(|error| SequentialDispatchError::new("policy", error.to_string()))?;
    let problem = problem.clone().canonicalize()
        .map_err(|error| SequentialDispatchError::new("problem", error.to_string()))?;
    mapping.validate(&problem)?;

    if mapping.event_tape_budget_model_sha256 != tape.header().budget_model_sha256 {
        return Err(SequentialDispatchError::new("mapping.event_tape_budget_model_sha256", "cost crosswalk is not bound to the EventTape budget model"));
    }
    let objective_contract = tape.header().objective_contract.as_ref().ok_or_else(|| {
        SequentialDispatchError::new("event_tape.header.objective_contract", "typed objective contract is required for sequential measurement dispatch")
    })?;
    let objective_contract_sha256 = tape.header().objective_contract_sha256.as_deref().ok_or_else(|| {
        SequentialDispatchError::new("event_tape.header.objective_contract_sha256", "objective-contract digest is required")
    })?;
    if objective_contract.contract_sha256().map_err(|error| SequentialDispatchError::new("objective_contract", error.to_string()))?
        != objective_contract_sha256
    {
        return Err(SequentialDispatchError::new("event_tape.header.objective_contract_sha256", "objective-contract digest does not reconstruct"));
    }

    let summary = tape.validate().map_err(|error| SequentialDispatchError::new("event_tape", error.to_string()))?;
    let spent = summary.total_charged_budget_microunits.ok_or_else(|| {
        SequentialDispatchError::new("event_tape.resources", "fully metered EventTape budget is required")
    })?;
    let remaining_microunits = tape.header().budget_limit_microunits.checked_sub(spent).ok_or_else(|| {
        SequentialDispatchError::new("event_tape.budget", "already-spent EventTape budget exceeds campaign limit")
    })?;
    let policy_budget_microunits = policy.budget_units.checked_mul(mapping.cost_unit_microunits)
        .ok_or_else(|| SequentialDispatchError::new("policy.budget_units", "cost conversion overflow"))?;
    if policy.root.remaining_budget_units != policy.budget_units
        || policy_budget_microunits > remaining_microunits
    {
        return Err(SequentialDispatchError::new("policy.budget_units", "sequential policy budget does not fit the remaining EventTape budget at this prefix"));
    }

    let mapping_sha = mapping.mapping_sha256()?;
    if let Some(action_id) = policy.root.selected_action_id.as_deref() {
        let descriptor = mapping.actions.iter().find(|descriptor| descriptor.action_id == action_id)
            .ok_or_else(|| SequentialDispatchError::new("mapping.selected_action_id", "selected solver action has no mapping"))?;
        if !policy.root.remaining_action_ids.iter().any(|available| available == action_id) {
            return Err(SequentialDispatchError::new("policy.selected_action_id", "selected action is not available at the root"));
        }
        if objective_contract_sha256 != descriptor.objective_contract_sha256 {
            return Err(SequentialDispatchError::new("mapping.objective_contract_sha256", "action mapping objective contract differs from EventTape"));
        }
        if !objective_contract.objective_specs.iter().any(|objective| objective.objective_id == descriptor.objective_id) {
            return Err(SequentialDispatchError::new("mapping.objective_id", "mapped objective is absent from the frozen objective contract"));
        }
        if !summary.registered_direction_ids.iter().any(|direction| direction == &descriptor.direction_id) {
            return Err(SequentialDispatchError::new("mapping.direction_id", "mapped direction is not registered in EventTape"));
        }
        let expected_mode = match descriptor.kind {
            SequentialMeasurementKind::DeterministicVector => EventTapeObjectiveMode::DeterministicVector,
            SequentialMeasurementKind::NoisyReplicate => EventTapeObjectiveMode::NoisyReplicates,
        };
        if objective_contract.mode != expected_mode {
            return Err(SequentialDispatchError::new("mapping.kind", "measurement kind is incompatible with the frozen objective mode"));
        }
        let expected_charge = descriptor.declared_incremental_cost_microunits;
        if expected_charge > remaining_microunits || expected_charge > policy_budget_microunits {
            return Err(SequentialDispatchError::new("mapping.expected_charge_microunits", "selected action exceeds remaining campaign or policy budget"));
        }
        let intent = match descriptor.kind {
            SequentialMeasurementKind::DeterministicVector => ActionIntent::EvaluateCandidateVector {
                candidate_sha256: descriptor.candidate_sha256.clone(),
                parent_candidate_sha256: descriptor.parent_candidate_sha256.clone(),
                semantic_delta_sha256: descriptor.semantic_delta_sha256.clone(),
                direction_id: descriptor.direction_id.clone(),
                objective_contract_sha256: descriptor.objective_contract_sha256.clone(),
            },
            SequentialMeasurementKind::NoisyReplicate => ActionIntent::ObserveCandidateReplicate {
                candidate_sha256: descriptor.candidate_sha256.clone(),
                parent_candidate_sha256: descriptor.parent_candidate_sha256.clone(),
                semantic_delta_sha256: descriptor.semantic_delta_sha256.clone(),
                direction_id: descriptor.direction_id.clone(),
                objective_contract_sha256: descriptor.objective_contract_sha256.clone(),
                decision_sha256: descriptor.decision_sha256.clone().ok_or_else(|| SequentialDispatchError::new("mapping.decision_sha256", "noisy decision digest missing"))?,
                replicate_id: descriptor.replicate_id.clone().ok_or_else(|| SequentialDispatchError::new("mapping.replicate_id", "replicate ID missing"))?,
                sample_seed: descriptor.sample_seed.ok_or_else(|| SequentialDispatchError::new("mapping.sample_seed", "sample seed missing"))?,
            },
        };
        if tape.events().iter().any(|event| same_measurement_identity(&event.payload.intent, &intent)) {
            return Err(SequentialDispatchError::new("mapping.selected_action_id", "selected measurement identity has already been attempted in this EventTape"));
        }

        let payload = DispatchDigestPayload {
            schema: SCHEMA, version: VERSION, campaign_id: &recommendation.campaign_id,
            source_commit: &recommendation.source_commit, policy_id: &recommendation.policy_id,
            mapping_id: &mapping.mapping_id, mapping_sha256: &mapping_sha,
            recommendation_sha256: &recommendation.recommendation_sha256,
            visible_history_sha256: &recommendation.visible_history_sha256,
            observed_through_sequence: recommendation.observed_through_sequence,
            selected_action_id: action_id, intent: &intent,
            expected_charge_microunits: expected_charge,
            remaining_budget_microunits: remaining_microunits,
        };
        let authorization_sha256 = digest(&payload)?;
        return Ok(AuthorizedSequentialDecision::Measure {
            authorization: SequentialDispatchAuthorization {
                schema: SCHEMA.into(), version: VERSION.into(),
                campaign_id: recommendation.campaign_id.clone(),
                source_commit: recommendation.source_commit.clone(),
                policy_id: recommendation.policy_id.clone(),
                mapping_id: mapping.mapping_id.clone(), mapping_sha256: mapping_sha,
                recommendation_sha256: recommendation.recommendation_sha256.clone(),
                visible_history_sha256: recommendation.visible_history_sha256.clone(),
                observed_through_sequence: recommendation.observed_through_sequence,
                selected_action_id: action_id.to_owned(), intent,
                expected_charge_microunits: expected_charge,
                remaining_budget_microunits: remaining_microunits,
                authorization_sha256,
            },
        });
    }

    let terminal_decision_id = policy.root.terminal_decision_id.as_deref().ok_or_else(|| {
        SequentialDispatchError::new("policy.root", "no action selected and no terminal decision specified")
    })?;
    let payload = StopDigestPayload {
        schema: SCHEMA, version: VERSION, campaign_id: &recommendation.campaign_id,
        source_commit: &recommendation.source_commit, policy_id: &recommendation.policy_id,
        recommendation_sha256: &recommendation.recommendation_sha256,
        visible_history_sha256: &recommendation.visible_history_sha256,
        observed_through_sequence: recommendation.observed_through_sequence,
        terminal_decision_id, budget_units: policy.budget_units,
    };
    Ok(AuthorizedSequentialDecision::Stop {
        authorization: SequentialStopAuthorization {
            schema: SCHEMA.into(), version: VERSION.into(),
            campaign_id: recommendation.campaign_id.clone(),
            source_commit: recommendation.source_commit.clone(),
            policy_id: recommendation.policy_id.clone(),
            recommendation_sha256: recommendation.recommendation_sha256.clone(),
            visible_history_sha256: recommendation.visible_history_sha256.clone(),
            observed_through_sequence: recommendation.observed_through_sequence,
            terminal_decision_id: terminal_decision_id.to_owned(),
            budget_units: policy.budget_units,
            stop_sha256: digest(&payload)?,
        },
    })
}

/// Recompute the next action/stop receipt; any stale prefix or changed mapping fails.
pub fn verify_sequential_dispatch_authorization(
    tape: &EventTape,
    problem: &ExactEvsiProblem,
    policy: &SequentialDecisionPolicy,
    posterior_binding: &SequentialPosteriorBinding,
    recommendation: &SequentialPolicyRecommendation,
    mapping: &SequentialActionMappingContract,
    authorization: &AuthorizedSequentialDecision,
) -> Result<(), SequentialDispatchError> {
    let reconstructed = authorize_sequential_decision(tape, problem, policy, posterior_binding, recommendation, mapping)?;
    if reconstructed != *authorization {
        return Err(SequentialDispatchError::new("authorization", "dispatch authorization differs from reconstruction at the exact visible prefix"));
    }
    Ok(())
}

fn same_measurement_identity(left: &ActionIntent, right: &ActionIntent) -> bool {
    match (left, right) {
        (
            ActionIntent::EvaluateCandidateVector {
                candidate_sha256: a, objective_contract_sha256: ac, direction_id: ad, ..
            },
            ActionIntent::EvaluateCandidateVector {
                candidate_sha256: b, objective_contract_sha256: bc, direction_id: bd, ..
            },
        ) => a == b && ac == bc && ad == bd,
        (
            ActionIntent::ObserveCandidateReplicate {
                candidate_sha256: a, objective_contract_sha256: ac,
                decision_sha256: ax, replicate_id: ar, sample_seed: asd, ..
            },
            ActionIntent::ObserveCandidateReplicate {
                candidate_sha256: b, objective_contract_sha256: bc,
                decision_sha256: bx, replicate_id: br, sample_seed: bsd, ..
            },
        ) => a == b && ac == bc && ax == bx && ar == br && asd == bsd,
        _ => false,
    }
}

fn validate_sha(value: &str, path: &str) -> Result<(), SequentialDispatchError> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(SequentialDispatchError::new(path, "expected lowercase 64-character SHA-256"));
    }
    Ok(())
}
fn digest<T: Serialize>(value: &T) -> Result<String, SequentialDispatchError> {
    let bytes = serde_json::to_vec(value).map_err(|e| SequentialDispatchError::new("serialization", e.to_string()))?;
    Ok(Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        event_tape::{
            ActionEvent, BudgetModel, CandidateDisposition, ComparisonMode,
            EventTapeHeader, EventTapeObjectiveContract, EventTapeObjectiveSpec,
            EventTapeObjectiveMode, ExecutionContract, ExecutionMode, MeterValue,
            PolicySnapshot, ResourceReceipt,
        },
        pareto_objectives::ObjectiveDirection,
        pareto_evsi_oracle::{
            EvsiActionOutcome, EvsiDecision, EvsiReplicateAction, EvsiWorldLikelihood,
            EvsiWorldLoss, EvsiWorldProbability,
        },
        pareto_sequential_policy::solve_sequential_policy,
        sequential_policy_event_tape::{bind_sequential_recommendation, SequentialPosteriorBinding},
    };
    use serde_json::json;

    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const H5: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const H6: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn digest_test<T: Serialize>(value: &T) -> String {
        Sha256::digest(serde_json::to_vec(value).unwrap()).iter().map(|b| format!("{b:02x}")).collect()
    }
    fn measured(value: u64) -> MeterValue { MeterValue::Measured { value } }
    fn resources() -> ResourceReceipt {
        ResourceReceipt {
            host_cpu_millis: measured(1), wall_millis: measured(1), accelerator_millis: measured(0),
            peak_memory_bytes: measured(1024), accelerator_peak_memory_bytes: measured(0),
            energy_millijoules: measured(0), input_tokens: measured(0), output_tokens: measured(0),
            tool_calls: measured(0), evaluator_calls: measured(1), verifier_calls: measured(1),
            charged_budget_microunits: measured(1),
        }
    }

    fn contract() -> EventTapeObjectiveContract {
        EventTapeObjectiveContract {
            schema: "symthaea.ar5b4.event-tape-objective-contract".into(), version: "0.1.0".into(),
            mode: EventTapeObjectiveMode::DeterministicVector,
            objective_specs: vec![
                EventTapeObjectiveSpec { objective_id: "quality".into(), direction: ObjectiveDirection::Maximize,
                    observation_lower_bound: None, observation_upper_bound: None, bounds_certificate_sha256: None },
                EventTapeObjectiveSpec { objective_id: "cost".into(), direction: ObjectiveDirection::Minimize,
                    observation_lower_bound: None, observation_upper_bound: None, bounds_certificate_sha256: None },
            ],
            reference_point_raw: vec![0.0,0.0], sampling_protocol_id: None,
            sampling_protocol_sha256: None, noise_model_id: None, noise_model_sha256: None,
            familywise_alpha: None, max_candidates: None, minimum_replicates: None,
        }
    }

    fn header(contract: &EventTapeObjectiveContract) -> EventTapeHeader {
        let config=json!({"controller":"sequential-dispatch-test"});
        let budget_model=BudgetModel::ActionUnit { charge_per_action_microunits: 1 };
        EventTapeHeader {
            schema:"symthaea.ar5b4.event-tape".into(),version:"0.2.0".into(),
            campaign_id:"dispatch-campaign".into(),
            source_commit:"0123456789abcdef0123456789abcdef01234567".into(),
            execution_environment_sha256:H5.into(),comparison_mode:ComparisonMode::MechanismMatched,
            proposal_source_id:"fixed-proposals".into(),proposal_stream_sha256:Some(H6.into()),
            task_recipe_sha256:H1.into(),comparison_profile_sha256:H2.into(),
            evaluator_sha256:H3.into(),verifier_sha256:H4.into(),budget_model_id:"one-unit".into(),
            budget_model_sha256:digest_test(&budget_model),budget_model,budget_limit_microunits:100,
            initial_direction_ids:vec!["direction-A".into()],
            execution:ExecutionContract { mode:ExecutionMode::Sequential,worker_count:1,seats_per_round:1,
                first_round_claim_cap:None,steady_state_claim_cap:None },
            policy:PolicySnapshot { policy_id:"sequential-controller".into(),algorithm_id:"finite-dp-v0.1".into(),
                config:config.clone(),config_sha256:digest_test(&config),has_hidden_oracle_access:false },
            objective_contract:Some(contract.clone()),objective_contract_sha256:Some(digest_test(contract)),
        }
    }

    fn finite_problem() -> ExactEvsiProblem {
        ExactEvsiProblem {
            schema:"symthaea.architecture-engineering.exact-discrete-evsi-problem".into(),version:"0.1.0".into(),
            campaign_id:"dispatch-campaign".into(),posterior_model_id:"posterior".into(),posterior_model_sha256:H1.into(),
            observation_model_id:"observation".into(),observation_model_sha256:H2.into(),
            decision_loss_model_id:"loss".into(),decision_loss_model_sha256:H3.into(),
            cost_model_id:"one-unit".into(),cost_model_sha256:H4.into(),
            worlds:vec![
                EvsiWorldProbability{world_id:"world-a".into(),posterior_probability:0.5},
                EvsiWorldProbability{world_id:"world-b".into(),posterior_probability:0.5}],
            decisions:vec![
                EvsiDecision{decision_id:"choose-a".into(),losses_by_world:vec![
                    EvsiWorldLoss{world_id:"world-a".into(),loss:0.0},EvsiWorldLoss{world_id:"world-b".into(),loss:10.0}]},
                EvsiDecision{decision_id:"choose-b".into(),losses_by_world:vec![
                    EvsiWorldLoss{world_id:"world-a".into(),loss:10.0},EvsiWorldLoss{world_id:"world-b".into(),loss:0.0}]}],
            actions:vec![EvsiReplicateAction { action_id:"measure-a".into(),candidate_id:"candidate-a".into(),
                objective_id:"quality".into(),incremental_cost_units:1,outcomes:vec![
                    EvsiActionOutcome{outcome_id:"signal-a".into(),likelihoods_by_world:vec![
                        EvsiWorldLikelihood{world_id:"world-a".into(),probability:1.0},
                        EvsiWorldLikelihood{world_id:"world-b".into(),probability:0.0}]},
                    EvsiActionOutcome{outcome_id:"signal-b".into(),likelihoods_by_world:vec![
                        EvsiWorldLikelihood{world_id:"world-a".into(),probability:0.0},
                        EvsiWorldLikelihood{world_id:"world-b".into(),probability:1.0}]}]}],
        }.canonicalize().unwrap()
    }

    fn tape() -> EventTape {
        let contract=contract();
        let contract_sha=digest_test(&contract);
        let mut tape=EventTape::new(header(&contract)).unwrap();
        let previous_intent=ActionIntent::EvaluateCandidateVector {
            candidate_sha256:H6.into(),parent_candidate_sha256:None,semantic_delta_sha256:H2.into(),
            direction_id:"direction-A".into(),objective_contract_sha256:contract_sha.clone()
        };
        let previous_outcome=crate::event_tape::ActionOutcome::CandidateMultiObjectiveEvaluation {
            task_recipe_sha256:H1.into(),comparison_profile_sha256:H2.into(),evaluator_sha256:H3.into(),
            verifier_sha256:H4.into(),verifier_receipt_sha256:H5.into(),
            objective_contract_sha256:contract_sha,disposition:CandidateDisposition::Verified,
            objective_values:Some(vec![0.8,0.2]),evidence_sha256:Some(H1.into()),
            evidence_manifest_sha256:Some(H2.into()),evidence_byte_receipt_sha256:Some(H3.into()),
            failure_reason:None
        };
        tape.append(ActionEvent {
            event_id:"previous-event".into(),policy_id:"sequential-controller".into(),worker_id:0,
            round_id:1,batch_id:None,observed_through_sequence:0,visible_history_sha256:String::new(),
            retry_of_event_id:None,intent:previous_intent,outcome:previous_outcome,resources:resources()
        }).unwrap();
        tape
    }

    fn mapping(problem:&ExactEvsiProblem,tape:&EventTape)->SequentialActionMappingContract {
        let contract_sha=tape.header().objective_contract_sha256.clone().unwrap();
        SequentialActionMappingContract {
            schema:SCHEMA.into(),version:VERSION.into(),mapping_id:"dispatch-map-v1".into(),
            decision_cost_model_id:problem.cost_model_id.clone(),decision_cost_model_sha256:problem.cost_model_sha256.clone(),
            event_tape_budget_model_sha256:tape.header().budget_model_sha256.clone(),cost_unit_microunits:1,
            actions:vec![SequentialActionDescriptor {
                action_id:"measure-a".into(),candidate_id:"candidate-a".into(),objective_id:"quality".into(),
                kind:SequentialMeasurementKind::DeterministicVector,candidate_sha256:H5.into(),
                parent_candidate_sha256:None,semantic_delta_sha256:H3.into(),direction_id:"direction-A".into(),
                objective_contract_sha256:contract_sha,decision_sha256:None,replicate_id:None,sample_seed:None,
                declared_incremental_cost_microunits:1,
            }]
        }
    }

    fn prepared() -> (EventTape,ExactEvsiProblem,SequentialDecisionPolicy,SequentialPosteriorBinding,SequentialPolicyRecommendation,SequentialActionMappingContract) {
        let tape=tape(); let problem=finite_problem(); let policy=solve_sequential_policy(&problem,1).unwrap();
        let binding=SequentialPosteriorBinding::new(&tape,&problem,H5).unwrap();
        let recommendation=bind_sequential_recommendation(&tape,&problem,&policy,&binding).unwrap();
        let map=mapping(&problem,&tape);
        (tape,problem,policy,binding,recommendation,map)
    }

    #[test]
    fn selected_measurement_maps_to_one_budget_checked_event_tape_intent() {
        let (tape,problem,policy,binding,recommendation,map)=prepared();
        let decision=authorize_sequential_decision(&tape,&problem,&policy,&binding,&recommendation,&map).unwrap();
        match &decision {
            AuthorizedSequentialDecision::Measure { authorization } => {
                assert_eq!(authorization.selected_action_id,"measure-a");
                assert_eq!(authorization.expected_charge_microunits,1);
                assert!(matches!(&authorization.intent,ActionIntent::EvaluateCandidateVector{..}));
            }
            AuthorizedSequentialDecision::Stop { .. } => panic!("expected the informative measurement"),
        }
        verify_sequential_dispatch_authorization(&tape,&problem,&policy,&binding,&recommendation,&map,&decision).unwrap();
    }

    #[test]
    fn no_information_policy_emits_explicit_stop_authorization() {
        let (tape,mut problem,_,_,_,_)=prepared();
        for action in &mut problem.actions {
            for outcome in &mut action.outcomes {
                for likelihood in &mut outcome.likelihoods_by_world {
                    likelihood.probability=0.5;
                }
            }
        }
        let problem=problem.canonicalize().unwrap();
        let policy=solve_sequential_policy(&problem,1).unwrap();
        assert!(policy.root.selected_action_id.is_none());
        let binding=SequentialPosteriorBinding::new(&tape,&problem,H5).unwrap();
        let recommendation=bind_sequential_recommendation(&tape,&problem,&policy,&binding).unwrap();
        let map=mapping(&problem,&tape);
        let decision=authorize_sequential_decision(&tape,&problem,&policy,&binding,&recommendation,&map).unwrap();
        match decision {
            AuthorizedSequentialDecision::Stop { authorization } => {
                assert_eq!(authorization.terminal_decision_id,policy.root.terminal_decision_id.unwrap());
                assert_eq!(authorization.observed_through_sequence,1);
            }
            AuthorizedSequentialDecision::Measure { .. } => panic!("uninformative actions must not be dispatched"),
        }
    }

    #[test]
    fn budget_overrun_and_already_attempted_measurement_are_rejected() {
        let (tape,problem,policy,binding,recommendation,mut map)=prepared();
        map.cost_unit_microunits=101;
        map.actions[0].declared_incremental_cost_microunits=101;
        assert!(authorize_sequential_decision(&tape,&problem,&policy,&binding,&recommendation,&map).is_err());

        let (tape,problem,policy,binding,recommendation,mut map)=prepared();
        map.actions[0].candidate_sha256=H6.into();
        assert!(authorize_sequential_decision(&tape,&problem,&policy,&binding,&recommendation,&map).is_err());
    }

    #[test]
    fn action_mapping_substitution_and_missing_mapping_fail_closed() {
        let (tape,problem,policy,binding,recommendation,mut map)=prepared();
        map.actions[0].candidate_id="other-candidate".into();
        assert!(authorize_sequential_decision(&tape,&problem,&policy,&binding,&recommendation,&map).is_err());

        let (tape,problem,policy,binding,recommendation,mut map)=prepared();
        map.actions.clear();
        assert!(authorize_sequential_decision(&tape,&problem,&policy,&binding,&recommendation,&map).is_err());
    }
}
