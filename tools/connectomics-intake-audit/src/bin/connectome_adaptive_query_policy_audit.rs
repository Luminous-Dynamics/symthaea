//! Exact worst-case adaptive query-directed observation policy.
//!
//! The planner filters the original joint tuple relations after each observed
//! action outcome. It stops when all still-compatible worlds agree on the
//! frozen query decision; it does not require identifying the exact world.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const MAX_ACTIONS: usize = 12;
const MAX_SCENARIOS: usize = 12;
const MAX_JOINT_TUPLES_PER_WORLD: usize = 64;
const MAX_POLICY_STATES: usize = 25_000;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-adaptive-query-policy-audit",
    version,
    about = "Find a minimum worst-case adaptive observation policy for a frozen query"
)]
struct Args {
    /// Frozen worlds, action universe and correlated joint observation supports.
    #[arg(long)]
    input: PathBuf,
    /// Optional deterministic JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    analysis_id: String,
    dataset_revision_commit_sha: String,
    query_id: String,
    query_spec_sha256: String,
    scenario_set_scope: String,
    scenario_set_sha256: String,
    action_set_sha256: String,
    joint_support_scope: String,
    joint_support_sha256: String,
    effort_unit: String,
    max_total_effort_units: u64,
    scenarios: Vec<Scenario>,
    actions: Vec<ObservationAction>,
    joint_world_supports: Vec<JointWorldSupport>,
    #[serde(default)]
    support_challenges: Vec<SupportChallenge>,
    #[serde(default)]
    scenario_challenges: Vec<ScenarioExpansionChallenge>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    graph_sha256: String,
    decision_met: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum ObservationActionKind {
    ImageInspection,
    MorphologyReview,
    SynapseAdjudication,
    OrthogonalMeasurement,
    Other,
}

impl ObservationActionKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::ImageInspection => "image_inspection",
            Self::MorphologyReview => "morphology_review",
            Self::SynapseAdjudication => "synapse_adjudication",
            Self::OrthogonalMeasurement => "orthogonal_measurement",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ObservationAction {
    action_id: String,
    action_kind: ObservationActionKind,
    effort_units: u64,
    source_evidence_sha256: String,
    observation_model_sha256: String,
    dependence_groups: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct JointWorldSupport {
    scenario_id: String,
    source_evidence_sha256: String,
    joint_observation_model_sha256: String,
    dependence_groups: Vec<String>,
    possible_joint_outcomes: Vec<JointOutcome>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SupportChallenge {
    challenge_id: String,
    challenger_evidence_sha256: String,
    challenger_model_sha256: String,
    additions: Vec<JointSupportAddition>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct JointSupportAddition {
    scenario_id: String,
    source_evidence_sha256: String,
    joint_observation_model_sha256: String,
    dependence_groups: Vec<String>,
    possible_joint_outcomes: Vec<JointOutcome>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ScenarioExpansionChallenge {
    challenge_id: String,
    challenger_evidence_sha256: String,
    challenger_model_sha256: String,
    additions: Vec<ScenarioWorldAddition>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ScenarioWorldAddition {
    scenario: Scenario,
    support: JointWorldSupport,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct JointOutcome {
    outcomes: Vec<ActionOutcome>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ActionOutcome {
    action_id: String,
    outcome: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct PolicyState {
    remaining_action_mask: u16,
    world_row_masks: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyNode {
    action_id: Option<String>,
    action_cost_units: u64,
    terminal_query_decision: Option<bool>,
    compatible_world_ids: Vec<String>,
    branches: Vec<PolicyBranch>,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyBranch {
    observed_outcome: String,
    child: Box<PolicyNode>,
}

#[derive(Debug, Clone)]
struct PolicySolution {
    worst_case_cost: u64,
    minimum_path_cost: u64,
    maximum_path_cost: u64,
    optimal_policy_count: u128,
    node: PolicyNode,
}

#[derive(Debug, Clone, Serialize)]
struct FixedPlanSummary {
    minimum_effort_units: Option<u64>,
    minimum_action_ids: Vec<String>,
    minimum_plan_count: u64,
    within_declared_budget: Option<bool>,
    adaptive_worst_case_savings_units: Option<i64>,
}

#[derive(Debug, Serialize)]
struct SupportChallengeResult {
    challenge_id: String,
    challenge_sha256: String,
    expanded_joint_support_sha256: String,
    added_joint_tuple_count: usize,
    adaptive_worst_case_effort_before: Option<u64>,
    adaptive_worst_case_effort_after: Option<u64>,
    adaptive_status_before: String,
    adaptive_status_after: String,
    adaptive_effort_non_decreasing: bool,
    fixed_minimum_effort_before: Option<u64>,
    fixed_minimum_effort_after: Option<u64>,
    fixed_status_before: String,
    fixed_status_after: String,
    fixed_effort_non_decreasing: bool,
    monotonicity_passed: bool,
}

#[derive(Debug, Serialize)]
struct ScenarioExpansionResult {
    challenge_id: String,
    challenge_sha256: String,
    expanded_scenario_set_sha256: String,
    expanded_joint_support_sha256: String,
    added_scenario_count: usize,
    adaptive_worst_case_effort_before: Option<u64>,
    adaptive_worst_case_effort_after: Option<u64>,
    adaptive_status_before: String,
    adaptive_status_after: String,
    adaptive_effort_non_decreasing: bool,
    fixed_minimum_effort_before: Option<u64>,
    fixed_minimum_effort_after: Option<u64>,
    fixed_status_before: String,
    fixed_status_after: String,
    fixed_effort_non_decreasing: bool,
    monotonicity_passed: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    tool: &'static str,
    analysis_id: String,
    input_sha256: String,
    dataset_revision_commit_sha: String,
    query_id: String,
    query_spec_sha256: String,
    scenario_set_sha256: String,
    action_set_sha256: String,
    joint_support_sha256: String,
    effort_unit: String,
    max_total_effort_units: u64,
    scenario_count: usize,
    observation_action_count: usize,
    joint_support_tuple_count: usize,
    exact_policy_state_count: usize,
    policy_node_count: usize,
    policy_branch_count: usize,
    worst_case_effort_units: Option<u64>,
    optimal_policy_count: u128,
    minimum_terminal_path_effort_units: Option<u64>,
    maximum_terminal_path_effort_units: Option<u64>,
    worst_case_policy_within_budget: Option<bool>,
    fixed_non_adaptive_optimum: FixedPlanSummary,
    support_challenge_count: usize,
    support_challenges: Vec<SupportChallengeResult>,
    support_expansion_monotonicity_passed: bool,
    scenario_challenge_count: usize,
    scenario_challenges: Vec<ScenarioExpansionResult>,
    hypothesis_expansion_monotonicity_passed: bool,
    policy: Option<PolicyNode>,
    out_of_model_observation_disposition: &'static str,
    expected_cost_claim: &'static str,
    probability_claim: &'static str,
    status: &'static str,
    search_complete: bool,
    scope_note: &'static str,
}

type TupleRowsByWorld = Vec<Vec<Vec<String>>>;
type OutcomeMasksByAction = Vec<Vec<BTreeMap<String, u64>>>;

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn sorted_scenarios(scenarios: &[Scenario]) -> Vec<Scenario> {
    let mut rows = scenarios.to_vec();
    rows.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    rows
}

fn sorted_actions(actions: &[ObservationAction]) -> Vec<ObservationAction> {
    let mut rows = actions.to_vec();
    rows.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    rows
}

fn canonical_scenario_set_sha256(scenarios: &[Scenario]) -> String {
    let rows = sorted_scenarios(scenarios);
    let mut hasher = Sha256::new();
    hasher.update(b"connectome-query-disambiguation-scenario-set-v1");
    hasher.update((rows.len() as u64).to_be_bytes());
    for scenario in rows {
        hasher.update(b"WORLD");
        hash_field(&mut hasher, &scenario.scenario_id);
        hasher.update(scenario.graph_sha256.as_bytes());
        hasher.update([u8::from(scenario.decision_met)]);
    }
    hex::encode(hasher.finalize())
}

fn canonical_action_set_sha256(actions: &[ObservationAction]) -> String {
    let rows = sorted_actions(actions);
    let mut hasher = Sha256::new();
    hasher.update(b"connectome-joint-observation-action-set-v1");
    hasher.update((rows.len() as u64).to_be_bytes());
    for mut action in rows {
        action.dependence_groups.sort();
        hasher.update(b"ACTION");
        hash_field(&mut hasher, &action.action_id);
        hash_field(&mut hasher, action.action_kind.as_str());
        hasher.update(action.effort_units.to_be_bytes());
        hasher.update(action.source_evidence_sha256.as_bytes());
        hasher.update(action.observation_model_sha256.as_bytes());
        hasher.update((action.dependence_groups.len() as u64).to_be_bytes());
        for group in action.dependence_groups {
            hasher.update(b"DEPENDENCE");
            hash_field(&mut hasher, &group);
        }
    }
    hex::encode(hasher.finalize())
}

fn normalized_joint_rows(
    support: &JointWorldSupport,
    action_ids: &[String],
) -> Result<Vec<Vec<String>>, String> {
    let known_actions = action_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut rows = Vec::with_capacity(support.possible_joint_outcomes.len());
    let mut unique_rows = BTreeSet::new();
    for (row_index, tuple) in support.possible_joint_outcomes.iter().enumerate() {
        if tuple.outcomes.len() != action_ids.len() {
            return Err(format!(
                "world {} tuple {row_index} must have exactly one outcome per action",
                support.scenario_id
            ));
        }
        let mut by_action = BTreeMap::new();
        for outcome in &tuple.outcomes {
            if !known_actions.contains(outcome.action_id.as_str()) || !nonempty(&outcome.outcome) {
                return Err(format!(
                    "world {} tuple {row_index} has unknown action or empty outcome",
                    support.scenario_id
                ));
            }
            if by_action
                .insert(outcome.action_id.as_str(), outcome.outcome.as_str())
                .is_some()
            {
                return Err(format!(
                    "world {} tuple {row_index} repeats action {}",
                    support.scenario_id, outcome.action_id
                ));
            }
        }
        let row = action_ids
            .iter()
            .map(|action_id| {
                by_action
                    .get(action_id.as_str())
                    .map(|value| (*value).to_string())
                    .ok_or_else(|| {
                        format!(
                            "world {} tuple {row_index} omits action {}",
                            support.scenario_id, action_id
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !unique_rows.insert(row.clone()) {
            return Err(format!(
                "world {} has duplicate joint tuples",
                support.scenario_id
            ));
        }
        rows.push(row);
    }
    rows.sort();
    Ok(rows)
}

fn canonical_joint_support_sha256(
    scenarios: &[Scenario],
    actions: &[ObservationAction],
    supports: &[JointWorldSupport],
) -> Result<String, String> {
    let worlds = sorted_scenarios(scenarios);
    let actions = sorted_actions(actions);
    let action_ids = actions
        .iter()
        .map(|action| action.action_id.clone())
        .collect::<Vec<_>>();
    let mut support_rows = supports.iter().collect::<Vec<_>>();
    support_rows.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut hasher = Sha256::new();
    hasher.update(b"connectome-joint-observation-support-set-v1");
    hasher.update((worlds.len() as u64).to_be_bytes());
    for world in worlds {
        let support = support_rows
            .iter()
            .find(|row| row.scenario_id == world.scenario_id)
            .ok_or_else(|| format!("missing support for {}", world.scenario_id))?;
        hasher.update(b"WORLD-SUPPORT");
        hash_field(&mut hasher, &world.scenario_id);
        hasher.update(support.source_evidence_sha256.as_bytes());
        hasher.update(support.joint_observation_model_sha256.as_bytes());
        let mut groups = support.dependence_groups.clone();
        groups.sort();
        hasher.update((groups.len() as u64).to_be_bytes());
        for group in groups {
            hasher.update(b"DEPENDENCE");
            hash_field(&mut hasher, &group);
        }
        let tuples = normalized_joint_rows(support, &action_ids)?;
        hasher.update((tuples.len() as u64).to_be_bytes());
        for tuple in tuples {
            hasher.update(b"JOINT-TUPLE");
            for (action_id, outcome) in action_ids.iter().zip(tuple) {
                hash_field(&mut hasher, action_id);
                hash_field(&mut hasher, &outcome);
            }
        }
    }
    Ok(hex::encode(hasher.finalize()))
}

fn validate(input: &Input) -> Result<(Vec<String>, TupleRowsByWorld), String> {
    if input.schema_version != 1 {
        return Err("schema_version must equal 1".into());
    }
    if !nonempty(&input.analysis_id) || !nonempty(&input.query_id) {
        return Err("analysis_id and query_id must be non-empty".into());
    }
    if !valid_hex(&input.dataset_revision_commit_sha, 40)
        || !valid_hex(&input.query_spec_sha256, 64)
        || !valid_hex(&input.scenario_set_sha256, 64)
        || !valid_hex(&input.action_set_sha256, 64)
        || !valid_hex(&input.joint_support_sha256, 64)
    {
        return Err(
            "dataset, query, scenario, action and joint-support digests are malformed".into(),
        );
    }
    if !matches!(
        input.scenario_set_scope.as_str(),
        "exhaustive_declared_scenario_space" | "sampled_sensitivity_set"
    ) {
        return Err("scenario_set_scope is invalid".into());
    }
    if !matches!(
        input.joint_support_scope.as_str(),
        "exhaustive_declared_joint_support" | "sampled_sensitivity_support"
    ) {
        return Err("joint_support_scope is invalid".into());
    }
    if !nonempty(&input.effort_unit) || input.max_total_effort_units == 0 {
        return Err("effort unit and budget must be positive".into());
    }
    if !(2..=MAX_SCENARIOS).contains(&input.scenarios.len()) {
        return Err(format!(
            "adaptive policy supports 2..={MAX_SCENARIOS} worlds"
        ));
    }
    if input.actions.is_empty() || input.actions.len() > MAX_ACTIONS {
        return Err(format!(
            "adaptive policy supports 1..={MAX_ACTIONS} actions"
        ));
    }
    let mut scenario_ids = BTreeSet::new();
    for scenario in &input.scenarios {
        if !nonempty(&scenario.scenario_id)
            || !scenario_ids.insert(scenario.scenario_id.clone())
            || !valid_hex(&scenario.graph_sha256, 64)
        {
            return Err("scenario IDs must be unique and graph digests valid".into());
        }
    }
    if canonical_scenario_set_sha256(&input.scenarios) != input.scenario_set_sha256 {
        return Err("scenario-set digest mismatch".into());
    }
    let mut action_id_set = BTreeSet::new();
    for action in &input.actions {
        if !nonempty(&action.action_id)
            || !action_id_set.insert(action.action_id.clone())
            || action.effort_units == 0
            || !valid_hex(&action.source_evidence_sha256, 64)
            || !valid_hex(&action.observation_model_sha256, 64)
        {
            return Err("action IDs, effort and provenance are invalid".into());
        }
        let mut groups = BTreeSet::new();
        for group in &action.dependence_groups {
            if !nonempty(group) || !groups.insert(group) {
                return Err(format!(
                    "action {} has duplicate/empty dependence group",
                    action.action_id
                ));
            }
        }
    }
    let sorted_action_rows = sorted_actions(&input.actions);
    let action_ids = sorted_action_rows
        .iter()
        .map(|action| action.action_id.clone())
        .collect::<Vec<_>>();
    if canonical_action_set_sha256(&input.actions) != input.action_set_sha256 {
        return Err("action-set digest mismatch".into());
    }

    let mut support_ids = BTreeSet::new();
    let mut supports_by_world = BTreeMap::new();
    for support in &input.joint_world_supports {
        if !scenario_ids.contains(&support.scenario_id)
            || !support_ids.insert(support.scenario_id.clone())
            || !valid_hex(&support.source_evidence_sha256, 64)
            || !valid_hex(&support.joint_observation_model_sha256, 64)
        {
            return Err("joint-world support identity/provenance invalid".into());
        }
        if support.possible_joint_outcomes.is_empty()
            || support.possible_joint_outcomes.len() > MAX_JOINT_TUPLES_PER_WORLD
        {
            return Err(format!(
                "world {} exceeds adaptive tuple limits",
                support.scenario_id
            ));
        }
        let mut groups = BTreeSet::new();
        for group in &support.dependence_groups {
            if !nonempty(group) || !groups.insert(group) {
                return Err(format!(
                    "world {} has duplicate/empty dependence group",
                    support.scenario_id
                ));
            }
        }
        supports_by_world.insert(
            support.scenario_id.clone(),
            normalized_joint_rows(support, &action_ids)?,
        );
    }
    if support_ids != scenario_ids {
        return Err("joint support must include every scenario exactly once".into());
    }
    if canonical_joint_support_sha256(
        &input.scenarios,
        &input.actions,
        &input.joint_world_supports,
    )? != input.joint_support_sha256
    {
        return Err("joint-support digest mismatch".into());
    }
    let worlds = sorted_scenarios(&input.scenarios);
    let tuples_by_world = worlds
        .iter()
        .map(|world| {
            supports_by_world
                .remove(&world.scenario_id)
                .ok_or_else(|| format!("missing normalized support for {}", world.scenario_id))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((action_ids, tuples_by_world))
}

fn build_outcome_masks(
    action_ids: &[String],
    tuples_by_world: &[Vec<Vec<String>>],
) -> OutcomeMasksByAction {
    (0..action_ids.len())
        .map(|action_index| {
            tuples_by_world
                .iter()
                .map(|tuples| {
                    let mut outcomes = BTreeMap::<String, u64>::new();
                    for (row_index, tuple) in tuples.iter().enumerate() {
                        *outcomes.entry(tuple[action_index].clone()).or_default() |=
                            1_u64 << row_index;
                    }
                    outcomes
                })
                .collect()
        })
        .collect()
}

struct Solver<'a> {
    action_ids: &'a [String],
    action_effort: &'a [u64],
    scenarios: &'a [Scenario],
    outcome_masks: OutcomeMasksByAction,
    state_count: usize,
    memo: BTreeMap<PolicyState, Option<PolicySolution>>,
}

impl<'a> Solver<'a> {
    fn compatible_world_indices(&self, state: &PolicyState) -> Vec<usize> {
        state
            .world_row_masks
            .iter()
            .enumerate()
            .filter(|(_, row_mask)| **row_mask != 0)
            .map(|(index, _)| index)
            .collect()
    }

    fn terminal_decision(&self, state: &PolicyState) -> Option<bool> {
        let active = self.compatible_world_indices(state);
        let first = *active.first()?;
        let decision = self.scenarios[first].decision_met;
        active
            .iter()
            .all(|index| self.scenarios[*index].decision_met == decision)
            .then_some(decision)
    }

    fn make_leaf(&self, state: &PolicyState, decision: bool) -> PolicySolution {
        let world_ids = self
            .compatible_world_indices(state)
            .iter()
            .map(|index| self.scenarios[*index].scenario_id.clone())
            .collect();
        PolicySolution {
            worst_case_cost: 0,
            minimum_path_cost: 0,
            maximum_path_cost: 0,
            optimal_policy_count: 1,
            node: PolicyNode {
                action_id: None,
                action_cost_units: 0,
                terminal_query_decision: Some(decision),
                compatible_world_ids: world_ids,
                branches: Vec::new(),
            },
        }
    }

    fn solve(&mut self, state: PolicyState) -> Result<Option<PolicySolution>, String> {
        if let Some(cached) = self.memo.get(&state) {
            return Ok(cached.clone());
        }
        self.state_count += 1;
        if self.state_count > MAX_POLICY_STATES {
            return Err(format!(
                "adaptive policy state cap {MAX_POLICY_STATES} exceeded; no optimality claim emitted"
            ));
        }

        if let Some(decision) = self.terminal_decision(&state) {
            let leaf = self.make_leaf(&state, decision);
            self.memo.insert(state, Some(leaf.clone()));
            return Ok(Some(leaf));
        }
        if state.remaining_action_mask == 0 {
            self.memo.insert(state, None);
            return Ok(None);
        }

        let active_worlds = self.compatible_world_indices(&state);
        let compatible_world_ids = active_worlds
            .iter()
            .map(|index| self.scenarios[*index].scenario_id.clone())
            .collect::<Vec<_>>();
        let mut best: Option<PolicySolution> = None;

        for action_index in 0..self.action_ids.len() {
            let action_bit = 1_u16 << action_index;
            if state.remaining_action_mask & action_bit == 0 {
                continue;
            }
            let mut observed_outcomes = BTreeSet::new();
            for world_index in &active_worlds {
                for (outcome, outcome_mask) in &self.outcome_masks[action_index][*world_index] {
                    if state.world_row_masks[*world_index] & outcome_mask != 0 {
                        observed_outcomes.insert(outcome.clone());
                    }
                }
            }
            if observed_outcomes.is_empty() {
                continue;
            }

            let child_remaining_mask = state.remaining_action_mask & !action_bit;
            let mut branches = Vec::new();
            let mut child_solutions = Vec::new();
            let mut action_is_valid = true;
            for observed_outcome in observed_outcomes {
                let mut child_world_masks = vec![0_u64; state.world_row_masks.len()];
                for world_index in &active_worlds {
                    let outcome_mask = self.outcome_masks[action_index][*world_index]
                        .get(&observed_outcome)
                        .copied()
                        .unwrap_or_default();
                    child_world_masks[*world_index] =
                        state.world_row_masks[*world_index] & outcome_mask;
                }
                let child_state = PolicyState {
                    remaining_action_mask: child_remaining_mask,
                    world_row_masks: child_world_masks,
                };
                let Some(child_solution) = self.solve(child_state)? else {
                    action_is_valid = false;
                    break;
                };
                branches.push(PolicyBranch {
                    observed_outcome,
                    child: Box::new(child_solution.node.clone()),
                });
                child_solutions.push(child_solution);
            }
            if !action_is_valid || branches.is_empty() {
                continue;
            }
            branches.sort_by(|left, right| left.observed_outcome.cmp(&right.observed_outcome));

            let child_worst = child_solutions
                .iter()
                .map(|child| child.worst_case_cost)
                .max()
                .unwrap_or_default();
            let child_min_path = child_solutions
                .iter()
                .map(|child| child.minimum_path_cost)
                .min()
                .unwrap_or_default();
            let child_max_path = child_solutions
                .iter()
                .map(|child| child.maximum_path_cost)
                .max()
                .unwrap_or_default();
            let action_cost = self.action_effort[action_index];
            let candidate_policy_count =
                child_solutions.iter().try_fold(1_u128, |count, child| {
                    count
                        .checked_mul(child.optimal_policy_count)
                        .ok_or_else(|| "adaptive optimal-policy count overflows u128".to_string())
                })?;
            let candidate = PolicySolution {
                worst_case_cost: action_cost
                    .checked_add(child_worst)
                    .ok_or_else(|| "adaptive worst-case policy effort overflows u64".to_string())?,
                minimum_path_cost: action_cost
                    .checked_add(child_min_path)
                    .ok_or_else(|| "adaptive minimum path effort overflows u64".to_string())?,
                maximum_path_cost: action_cost
                    .checked_add(child_max_path)
                    .ok_or_else(|| "adaptive maximum path effort overflows u64".to_string())?,
                optimal_policy_count: candidate_policy_count,
                node: PolicyNode {
                    action_id: Some(self.action_ids[action_index].clone()),
                    action_cost_units: action_cost,
                    terminal_query_decision: None,
                    compatible_world_ids: compatible_world_ids.clone(),
                    branches,
                },
            };
            if best.as_ref().is_none_or(|best_solution| {
                candidate.worst_case_cost < best_solution.worst_case_cost
            }) {
                best = Some(candidate);
            } else if best.as_ref().is_some_and(|best_solution| {
                candidate.worst_case_cost == best_solution.worst_case_cost
            }) {
                let total_count = best
                    .as_ref()
                    .expect("equal-cost best policy exists")
                    .optimal_policy_count
                    .checked_add(candidate.optimal_policy_count)
                    .ok_or_else(|| "adaptive optimal-policy count overflows u128".to_string())?;
                best.as_mut()
                    .expect("equal-cost best policy exists")
                    .optimal_policy_count = total_count;
            }
        }

        self.memo.insert(state, best.clone());
        Ok(best)
    }
}

fn resolve_policy_branch<'a>(
    node: &'a PolicyNode,
    observed_outcome: &str,
) -> Result<&'a PolicyNode, &'static str> {
    if node.action_id.is_none() {
        return Err("POLICY_ALREADY_TERMINAL");
    }
    node.branches
        .iter()
        .find(|branch| branch.observed_outcome == observed_outcome)
        .map(|branch| branch.child.as_ref())
        .ok_or("OUT_OF_MODEL_OBSERVATION")
}

fn validate_policy_tree(node: &PolicyNode) -> Result<(), String> {
    if node.action_id.is_none() {
        if node.terminal_query_decision.is_none() || !node.branches.is_empty() {
            return Err("terminal policy node has inconsistent decision/branch state".into());
        }
        return Ok(());
    }
    if node.terminal_query_decision.is_some() || node.branches.is_empty() {
        return Err("observation policy node must have branches and no terminal decision".into());
    }
    for branch in &node.branches {
        let child = resolve_policy_branch(node, &branch.observed_outcome)
            .map_err(|status| format!("policy branch resolution failed: {status}"))?;
        validate_policy_tree(child)?;
    }
    Ok(())
}

fn normalized_addition_rows(
    addition: &JointSupportAddition,
    action_ids: &[String],
) -> Result<Vec<Vec<String>>, String> {
    let support = JointWorldSupport {
        scenario_id: addition.scenario_id.clone(),
        source_evidence_sha256: addition.source_evidence_sha256.clone(),
        joint_observation_model_sha256: addition.joint_observation_model_sha256.clone(),
        dependence_groups: addition.dependence_groups.clone(),
        possible_joint_outcomes: addition.possible_joint_outcomes.clone(),
    };
    normalized_joint_rows(&support, action_ids)
}

fn canonical_support_challenge_sha256(
    base_support_sha256: &str,
    challenge: &SupportChallenge,
    action_ids: &[String],
) -> Result<String, String> {
    let mut additions = challenge.additions.iter().collect::<Vec<_>>();
    additions.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut hasher = Sha256::new();
    hasher.update(b"connectome-adaptive-support-expansion-challenge-v1");
    hasher.update(base_support_sha256.as_bytes());
    hash_field(&mut hasher, &challenge.challenge_id);
    hasher.update(challenge.challenger_evidence_sha256.as_bytes());
    hasher.update(challenge.challenger_model_sha256.as_bytes());
    hasher.update((additions.len() as u64).to_be_bytes());
    for addition in additions {
        hasher.update(b"WORLD-ADDITION");
        hash_field(&mut hasher, &addition.scenario_id);
        hasher.update(addition.source_evidence_sha256.as_bytes());
        hasher.update(addition.joint_observation_model_sha256.as_bytes());
        let mut groups = addition.dependence_groups.clone();
        groups.sort();
        hasher.update((groups.len() as u64).to_be_bytes());
        for group in groups {
            hasher.update(b"DEPENDENCE");
            hash_field(&mut hasher, &group);
        }
        let rows = normalized_addition_rows(addition, action_ids)?;
        hasher.update((rows.len() as u64).to_be_bytes());
        for row in rows {
            hasher.update(b"ADDED-JOINT-TUPLE");
            for (action_id, outcome) in action_ids.iter().zip(row) {
                hash_field(&mut hasher, action_id);
                hash_field(&mut hasher, &outcome);
            }
        }
    }
    Ok(hex::encode(hasher.finalize()))
}

fn expand_joint_support(
    base_sha256: &str,
    challenge: &SupportChallenge,
    action_ids: &[String],
    scenarios: &[Scenario],
    base_rows: &[Vec<Vec<String>>],
) -> Result<(String, String, usize, TupleRowsByWorld), String> {
    if !nonempty(&challenge.challenge_id) {
        return Err("support challenge ID must be non-empty".into());
    }
    if !valid_hex(&challenge.challenger_evidence_sha256, 64)
        || !valid_hex(&challenge.challenger_model_sha256, 64)
    {
        return Err(format!(
            "support challenge {} challenger evidence/model digests must be SHA-256",
            challenge.challenge_id
        ));
    }
    if challenge.additions.is_empty() || challenge.additions.len() > scenarios.len() {
        return Err(format!(
            "support challenge {} must add tuples to 1..={} worlds",
            challenge.challenge_id,
            scenarios.len()
        ));
    }
    let scenario_indices = scenarios
        .iter()
        .enumerate()
        .map(|(index, world)| (world.scenario_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut seen_worlds = BTreeSet::new();
    let mut expanded_rows = base_rows.to_vec();
    let mut added_tuple_count = 0_usize;
    for addition in &challenge.additions {
        let Some(world_index) = scenario_indices.get(addition.scenario_id.as_str()).copied() else {
            return Err(format!(
                "support challenge {} references unknown world {}",
                challenge.challenge_id, addition.scenario_id
            ));
        };
        if !seen_worlds.insert(addition.scenario_id.as_str()) {
            return Err(format!(
                "support challenge {} repeats world {}",
                challenge.challenge_id, addition.scenario_id
            ));
        }
        if !valid_hex(&addition.source_evidence_sha256, 64)
            || !valid_hex(&addition.joint_observation_model_sha256, 64)
        {
            return Err(format!(
                "support challenge {} world {} has invalid source/model digest",
                challenge.challenge_id, addition.scenario_id
            ));
        }
        let mut groups = BTreeSet::new();
        for group in &addition.dependence_groups {
            if !nonempty(group) || !groups.insert(group) {
                return Err(format!(
                    "support challenge {} world {} has duplicate/empty dependence group",
                    challenge.challenge_id, addition.scenario_id
                ));
            }
        }
        let rows = normalized_addition_rows(addition, action_ids)?;
        if rows.is_empty() {
            return Err(format!(
                "support challenge {} world {} contains no added tuples",
                challenge.challenge_id, addition.scenario_id
            ));
        }
        let existing = expanded_rows[world_index].iter().collect::<BTreeSet<_>>();
        if rows.iter().any(|row| existing.contains(row)) {
            return Err(format!(
                "support challenge {} world {} attempts to add a tuple already present",
                challenge.challenge_id, addition.scenario_id
            ));
        }
        let new_count = expanded_rows[world_index]
            .len()
            .checked_add(rows.len())
            .ok_or_else(|| "expanded joint tuple count overflow".to_string())?;
        if new_count > MAX_JOINT_TUPLES_PER_WORLD {
            return Err(format!(
                "support challenge {} world {} exceeds the {}-tuple expanded support cap",
                challenge.challenge_id, addition.scenario_id, MAX_JOINT_TUPLES_PER_WORLD
            ));
        }
        added_tuple_count = added_tuple_count
            .checked_add(rows.len())
            .ok_or_else(|| "added joint tuple count overflow".to_string())?;
        expanded_rows[world_index].extend(rows);
        expanded_rows[world_index].sort();
    }
    let challenge_sha256 = canonical_support_challenge_sha256(base_sha256, challenge, action_ids)?;
    let mut expanded_hasher = Sha256::new();
    expanded_hasher.update(b"connectome-adaptive-expanded-joint-support-v1");
    expanded_hasher.update(base_sha256.as_bytes());
    expanded_hasher.update(challenge_sha256.as_bytes());
    let expanded_sha256 = hex::encode(expanded_hasher.finalize());
    Ok((
        challenge_sha256,
        expanded_sha256,
        added_tuple_count,
        expanded_rows,
    ))
}

fn effort_is_non_decreasing(before: Option<u64>, after: Option<u64>) -> bool {
    match (before, after) {
        (Some(before), Some(after)) => after >= before,
        (Some(_), None) | (None, None) => true,
        (None, Some(_)) => false,
    }
}

fn solve_adaptive_policy(
    action_ids: &[String],
    action_effort: &[u64],
    scenarios: &[Scenario],
    tuples_by_world: &[Vec<Vec<String>>],
) -> Result<(Option<PolicySolution>, usize), String> {
    let full_world_masks = tuples_by_world
        .iter()
        .map(|tuples| {
            if tuples.len() == 64 {
                u64::MAX
            } else {
                (1_u64 << tuples.len()) - 1
            }
        })
        .collect::<Vec<_>>();
    let initial_state = PolicyState {
        remaining_action_mask: (1_u16 << action_ids.len()) - 1,
        world_row_masks: full_world_masks,
    };
    let outcome_masks = build_outcome_masks(action_ids, tuples_by_world);
    let mut solver = Solver {
        action_ids,
        action_effort,
        scenarios,
        outcome_masks,
        state_count: 0,
        memo: BTreeMap::new(),
    };
    let solution = solver.solve(initial_state)?;
    Ok((solution, solver.state_count))
}

fn adaptive_status(solution: &Option<PolicySolution>) -> &'static str {
    if solution.is_some() {
        "POLICY_EXISTS"
    } else {
        "NO_POLICY"
    }
}

fn fixed_status(summary: &FixedPlanSummary) -> &'static str {
    if summary.minimum_effort_units.is_some() {
        "FIXED_PLAN_EXISTS"
    } else {
        "NO_FIXED_PLAN"
    }
}

fn decision_conflict_pairs(scenarios: &[Scenario]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for left in 0..scenarios.len() {
        for right in (left + 1)..scenarios.len() {
            if scenarios[left].decision_met != scenarios[right].decision_met {
                pairs.push((left, right));
            }
        }
    }
    pairs
}

/// overlap[mask] is true when some cross-world tuple pair agrees on every
/// selected action in mask. The superset-OR transform fills all subsets of
/// each equality mask, so exact fixed-bundle separation is the negation.
fn pair_overlap_table(
    left_tuples: &[Vec<String>],
    right_tuples: &[Vec<String>],
    action_count: usize,
) -> Vec<bool> {
    let subset_count = 1_usize << action_count;
    let mut overlap = vec![false; subset_count];
    for left in left_tuples {
        for right in right_tuples {
            let mut equal_mask = 0_usize;
            for action_index in 0..action_count {
                if left[action_index] == right[action_index] {
                    equal_mask |= 1_usize << action_index;
                }
            }
            overlap[equal_mask] = true;
        }
    }
    for bit_index in 0..action_count {
        let bit = 1_usize << bit_index;
        for mask in 0..subset_count {
            if mask & bit == 0 && overlap[mask | bit] {
                overlap[mask] = true;
            }
        }
    }
    overlap
}

fn selected_action_ids(mask: usize, action_ids: &[String]) -> Vec<String> {
    action_ids
        .iter()
        .enumerate()
        .filter(|(index, _)| mask & (1_usize << index) != 0)
        .map(|(_, action_id)| action_id.clone())
        .collect()
}

fn minimum_fixed_plan(
    action_ids: &[String],
    action_effort: &[u64],
    scenarios: &[Scenario],
    tuple_rows: &[Vec<Vec<String>>],
    budget: u64,
) -> Result<FixedPlanSummary, String> {
    let pairs = decision_conflict_pairs(scenarios);
    if pairs.is_empty() {
        return Ok(FixedPlanSummary {
            minimum_effort_units: Some(0),
            minimum_action_ids: Vec::new(),
            minimum_plan_count: 1,
            within_declared_budget: Some(true),
            adaptive_worst_case_savings_units: None,
        });
    }
    let overlap_tables = pairs
        .iter()
        .map(|(left, right)| {
            pair_overlap_table(&tuple_rows[*left], &tuple_rows[*right], action_ids.len())
        })
        .collect::<Vec<_>>();
    let full_mask = (1_usize << action_ids.len()) - 1;
    if overlap_tables.iter().any(|table| table[full_mask]) {
        return Ok(FixedPlanSummary {
            minimum_effort_units: None,
            minimum_action_ids: Vec::new(),
            minimum_plan_count: 0,
            within_declared_budget: None,
            adaptive_worst_case_savings_units: None,
        });
    }

    let subset_count = 1_usize << action_ids.len();
    let mut best_cost: Option<u64> = None;
    let mut best_ids = Vec::new();
    let mut best_count = 0_u64;
    for mask in 1..subset_count {
        let mut cost = 0_u64;
        for (index, effort) in action_effort.iter().enumerate() {
            if mask & (1_usize << index) != 0 {
                cost = cost
                    .checked_add(*effort)
                    .ok_or_else(|| "fixed action-bundle effort overflows u64".to_string())?;
            }
        }
        if best_cost.is_some_and(|best| cost > best) {
            continue;
        }
        if overlap_tables.iter().any(|table| table[mask]) {
            continue;
        }
        let candidate_ids = selected_action_ids(mask, action_ids);
        if best_cost.is_none_or(|best| cost < best) {
            best_cost = Some(cost);
            best_ids = candidate_ids;
            best_count = 1;
        } else if best_cost == Some(cost) {
            best_count += 1;
            if candidate_ids < best_ids {
                best_ids = candidate_ids;
            }
        }
    }
    Ok(FixedPlanSummary {
        minimum_effort_units: best_cost,
        minimum_action_ids: best_ids,
        minimum_plan_count: best_count,
        within_declared_budget: best_cost.map(|cost| cost <= budget),
        adaptive_worst_case_savings_units: None,
    })
}

fn cost_delta(fixed: Option<u64>, adaptive: Option<u64>) -> Option<i64> {
    let fixed = i64::try_from(fixed?).ok()?;
    let adaptive = i64::try_from(adaptive?).ok()?;
    fixed.checked_sub(adaptive)
}

fn count_policy_nodes(node: &PolicyNode) -> (usize, usize) {
    let mut nodes = 1_usize;
    let mut branches = node.branches.len();
    for branch in &node.branches {
        let (child_nodes, child_branches) = count_policy_nodes(&branch.child);
        nodes += child_nodes;
        branches += child_branches;
    }
    (nodes, branches)
}

fn evaluate(input: &Input, input_bytes: &[u8]) -> Result<Report, String> {
    let (action_ids, tuples_by_world) = validate(input)?;
    let scenarios = sorted_scenarios(&input.scenarios);
    let actions = sorted_actions(&input.actions);
    let action_effort = actions
        .iter()
        .map(|action| action.effort_units)
        .collect::<Vec<_>>();
    let tuple_count = tuples_by_world.iter().map(Vec::len).sum::<usize>();
    let mut fixed_non_adaptive_optimum = minimum_fixed_plan(
        &action_ids,
        &action_effort,
        &scenarios,
        &tuples_by_world,
        input.max_total_effort_units,
    )?;
    let (solution, state_count) =
        solve_adaptive_policy(&action_ids, &action_effort, &scenarios, &tuples_by_world)?;
    fixed_non_adaptive_optimum.adaptive_worst_case_savings_units = cost_delta(
        fixed_non_adaptive_optimum.minimum_effort_units,
        solution.as_ref().map(|policy| policy.worst_case_cost),
    );

    let mut challenge_ids = BTreeSet::new();
    let mut support_challenge_results = Vec::with_capacity(input.support_challenges.len());
    for challenge in &input.support_challenges {
        if !challenge_ids.insert(challenge.challenge_id.as_str()) {
            return Err(format!(
                "duplicate support challenge ID {}",
                challenge.challenge_id
            ));
        }
        let (challenge_sha256, expanded_sha256, added_tuple_count, expanded_rows) =
            expand_joint_support(
                &input.joint_support_sha256,
                challenge,
                &action_ids,
                &scenarios,
                &tuples_by_world,
            )?;
        let (expanded_solution, _expanded_state_count) =
            solve_adaptive_policy(&action_ids, &action_effort, &scenarios, &expanded_rows)?;
        let expanded_fixed = minimum_fixed_plan(
            &action_ids,
            &action_effort,
            &scenarios,
            &expanded_rows,
            input.max_total_effort_units,
        )?;
        let adaptive_before = solution.as_ref().map(|policy| policy.worst_case_cost);
        let adaptive_after = expanded_solution
            .as_ref()
            .map(|policy| policy.worst_case_cost);
        let fixed_before = fixed_non_adaptive_optimum.minimum_effort_units;
        let fixed_after = expanded_fixed.minimum_effort_units;
        let adaptive_non_decreasing = effort_is_non_decreasing(adaptive_before, adaptive_after);
        let fixed_non_decreasing = effort_is_non_decreasing(fixed_before, fixed_after);
        support_challenge_results.push(SupportChallengeResult {
            challenge_id: challenge.challenge_id.clone(),
            challenge_sha256,
            expanded_joint_support_sha256: expanded_sha256,
            added_joint_tuple_count: added_tuple_count,
            adaptive_worst_case_effort_before: adaptive_before,
            adaptive_worst_case_effort_after: adaptive_after,
            adaptive_status_before: adaptive_status(&solution).to_string(),
            adaptive_status_after: adaptive_status(&expanded_solution).to_string(),
            adaptive_effort_non_decreasing: adaptive_non_decreasing,
            fixed_minimum_effort_before: fixed_before,
            fixed_minimum_effort_after: fixed_after,
            fixed_status_before: fixed_status(&fixed_non_adaptive_optimum).to_string(),
            fixed_status_after: fixed_status(&expanded_fixed).to_string(),
            fixed_effort_non_decreasing: fixed_non_decreasing,
            monotonicity_passed: adaptive_non_decreasing && fixed_non_decreasing,
        });
    }

    let (
        worst_case,
        min_path,
        max_path,
        within_budget,
        policy,
        policy_nodes,
        policy_branches,
        optimal_policy_count,
        status,
    ) = match solution.clone() {
        Some(solution) => {
            validate_policy_tree(&solution.node)?;
            let within = solution.worst_case_cost <= input.max_total_effort_units;
            let (nodes, branches) = count_policy_nodes(&solution.node);
            let status = if within {
                "MINIMUM_WORST_CASE_ADAPTIVE_POLICY"
            } else {
                "MINIMUM_WORST_CASE_ADAPTIVE_POLICY_EXCEEDS_BUDGET"
            };
            (
                Some(solution.worst_case_cost),
                Some(solution.minimum_path_cost),
                Some(solution.maximum_path_cost),
                Some(within),
                Some(solution.node),
                nodes,
                branches,
                solution.optimal_policy_count,
                status,
            )
        }
        None => (
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            0,
            "NO_ADAPTIVE_POLICY_WITHIN_DECLARED_ACTIONS",
        ),
    };

    let support_expansion_monotonicity_passed = support_challenge_results
        .iter()
        .all(|challenge| challenge.monotonicity_passed);
    let reported_status = if support_expansion_monotonicity_passed {
        status
    } else {
        "SUPPORT_EXPANSION_MONOTONICITY_VIOLATION"
    };

    Ok(Report {
        schema_version: 1,
        tool: "connectome-adaptive-query-policy-audit",
        analysis_id: input.analysis_id.clone(),
        input_sha256: sha256(input_bytes),
        dataset_revision_commit_sha: input.dataset_revision_commit_sha.clone(),
        query_id: input.query_id.clone(),
        query_spec_sha256: input.query_spec_sha256.clone(),
        scenario_set_sha256: input.scenario_set_sha256.clone(),
        action_set_sha256: input.action_set_sha256.clone(),
        joint_support_sha256: input.joint_support_sha256.clone(),
        effort_unit: input.effort_unit.clone(),
        max_total_effort_units: input.max_total_effort_units,
        scenario_count: input.scenarios.len(),
        observation_action_count: input.actions.len(),
        joint_support_tuple_count: tuple_count,
        exact_policy_state_count: state_count,
        policy_node_count: policy_nodes,
        policy_branch_count: policy_branches,
        worst_case_effort_units: worst_case,
        optimal_policy_count,
        minimum_terminal_path_effort_units: min_path,
        maximum_terminal_path_effort_units: max_path,
        worst_case_policy_within_budget: within_budget,
        fixed_non_adaptive_optimum,
        support_challenge_count: support_challenge_results.len(),
        support_challenges: support_challenge_results,
        support_expansion_monotonicity_passed,
        policy,
        out_of_model_observation_disposition: "FAIL_CLOSED_NO_QUERY_DECISION",
        expected_cost_claim: "NONE_NO_PRIOR_OR_EXPECTED_COST_COMPUTED",
        probability_claim: "NONE_FINITE_SET_VALUED_CORRELATED_SUPPORT_ONLY",
        status: reported_status,
        search_complete: true,
        scope_note: "The policy conditions the original joint tuple relations after each observed action outcome. It stops only when all compatible worlds agree on the frozen query decision, not when the exact world is identified. It does not factorize marginals, infer independence, assign probabilities, prove joint-support completeness, validate measurement models, or establish biological truth. Optional support challenges are additive-only sensitivity checks; a monotonicity pass does not prove challenge completeness. An observation category outside every declared compatible tuple is out-of-model and must fail closed.",
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let input_bytes = match fs::read(&args.input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("connectome-adaptive-query-policy-audit: could not read input: {error}");
            return ExitCode::from(2);
        }
    };
    let input: Input = match serde_json::from_slice(&input_bytes) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("connectome-adaptive-query-policy-audit: invalid JSON input: {error}");
            return ExitCode::from(2);
        }
    };
    let report = match evaluate(&input, &input_bytes) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("connectome-adaptive-query-policy-audit: {error}");
            return ExitCode::from(3);
        }
    };
    let rendered = match serde_json::to_string_pretty(&report) {
        Ok(json) => json + "\n",
        Err(error) => {
            eprintln!(
                "connectome-adaptive-query-policy-audit: could not serialize report: {error}"
            );
            return ExitCode::from(2);
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(path, rendered.as_bytes()) {
            eprintln!("connectome-adaptive-query-policy-audit: could not write report: {error}");
            return ExitCode::from(2);
        }
    } else {
        print!("{rendered}");
    }
    if !report.support_expansion_monotonicity_passed {
        return ExitCode::from(4);
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adaptive_fixture() -> Input {
        serde_json::from_str(include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_ADAPTIVE_QUERY_POLICY_SAVINGS.example.json"
        ))
        .expect("adaptive policy fixture must parse")
    }

    fn ambiguous_fixture() -> Input {
        serde_json::from_str(include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_ADAPTIVE_QUERY_POLICY_NO_POLICY.example.json"
        ))
        .expect("adaptive no-policy fixture must parse")
    }

    fn support_expansion_fixture() -> Input {
        serde_json::from_str(include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_ADAPTIVE_SUPPORT_EXPANSION.example.json"
        ))
        .expect("support-expansion fixture must parse")
    }

    fn no_policy_support_expansion_fixture() -> Input {
        serde_json::from_str(include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_ADAPTIVE_SUPPORT_EXPANSION_NO_POLICY.example.json"
        ))
        .expect("no-policy support-expansion fixture must parse")
    }

    fn bytes(input: &Input) -> Vec<u8> {
        serde_json::to_vec(input).expect("fixture serializes")
    }

    #[test]
    fn adaptive_policy_saves_worst_case_effort_and_stops_on_query_unanimity() {
        let input = adaptive_fixture();
        let report = evaluate(&input, &bytes(&input)).unwrap();
        assert_eq!(report.worst_case_effort_units, Some(4));
        assert_eq!(report.optimal_policy_count, 1);
        assert_eq!(report.minimum_terminal_path_effort_units, Some(3));
        assert_eq!(report.maximum_terminal_path_effort_units, Some(4));
        assert_eq!(report.worst_case_policy_within_budget, Some(true));
        assert_eq!(
            report.fixed_non_adaptive_optimum.minimum_effort_units,
            Some(5)
        );
        assert_eq!(
            report.fixed_non_adaptive_optimum.minimum_action_ids,
            vec!["inspect-y".to_string(), "inspect-z".to_string()]
        );
        assert_eq!(
            report
                .fixed_non_adaptive_optimum
                .adaptive_worst_case_savings_units,
            Some(1)
        );
        assert_eq!(report.status, "MINIMUM_WORST_CASE_ADAPTIVE_POLICY");
        let root = report.policy.unwrap();
        assert_eq!(root.action_id.as_deref(), Some("inspect-x"));
        assert_eq!(root.branches.len(), 2);
        assert!(
            root.branches
                .iter()
                .any(|branch| branch.child.action_id.as_deref() == Some("inspect-y"))
        );
        assert!(
            root.branches
                .iter()
                .any(|branch| branch.child.action_id.as_deref() == Some("inspect-z"))
        );
        let x_right = root
            .branches
            .iter()
            .find(|branch| branch.observed_outcome == "x-right")
            .unwrap();
        let shared_true_leaf = x_right
            .child
            .branches
            .iter()
            .find(|branch| branch.observed_outcome == "z-c")
            .unwrap();
        assert_eq!(shared_true_leaf.child.terminal_query_decision, Some(true));
    }

    #[test]
    fn different_worlds_with_same_query_decision_can_share_terminal_leaf() {
        let input = adaptive_fixture();
        let report = evaluate(&input, &bytes(&input)).unwrap();
        let root = report.policy.unwrap();
        let right_branch = root
            .branches
            .iter()
            .find(|branch| branch.observed_outcome == "x-right")
            .unwrap();
        assert_eq!(right_branch.child.action_id.as_deref(), Some("inspect-z"));
        let shared = right_branch
            .child
            .branches
            .iter()
            .find(|branch| branch.observed_outcome == "z-c")
            .unwrap();
        assert_eq!(shared.child.terminal_query_decision, Some(true));
        assert_eq!(shared.child.compatible_world_ids.len(), 2);
    }

    #[test]
    fn mixed_decision_state_with_no_remaining_action_has_no_policy() {
        let input = ambiguous_fixture();
        let report = evaluate(&input, &bytes(&input)).unwrap();
        assert_eq!(report.worst_case_effort_units, None);
        assert_eq!(report.fixed_non_adaptive_optimum.minimum_effort_units, None);
        assert_eq!(report.fixed_non_adaptive_optimum.minimum_plan_count, 0);
        assert_eq!(report.status, "NO_ADAPTIVE_POLICY_WITHIN_DECLARED_ACTIONS");
        assert!(report.policy.is_none());
    }

    #[test]
    fn joint_conditioning_removes_outcomes_incompatible_with_observed_prefix() {
        let input = adaptive_fixture();
        let (action_ids, tuples) = validate(&input).unwrap();
        let masks = build_outcome_masks(&action_ids, &tuples);
        let x_index = action_ids.iter().position(|id| id == "inspect-x").unwrap();
        let y_index = action_ids.iter().position(|id| id == "inspect-y").unwrap();
        let left_world = input
            .scenarios
            .iter()
            .position(|world| world.scenario_id == "world-a")
            .unwrap();
        let x_left_mask = masks[x_index][left_world]["x-left"];
        let y_after_x_left = masks[y_index][left_world]
            .iter()
            .filter(|(_, mask)| x_left_mask & **mask != 0)
            .map(|(outcome, _)| outcome.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(y_after_x_left, BTreeSet::from(["y-a".to_string()]));
    }

    #[test]
    fn observation_not_present_in_joint_support_fails_closed() {
        let input = adaptive_fixture();
        let report = evaluate(&input, &bytes(&input)).unwrap();
        let root = report.policy.unwrap();
        assert_eq!(
            resolve_policy_branch(&root, "not-in-declared-support").unwrap_err(),
            "OUT_OF_MODEL_OBSERVATION"
        );
    }

    #[test]
    fn exact_count_of_tied_optimal_policy_roots_is_reported() {
        let mut input = adaptive_fixture();
        input.scenarios.truncate(2);
        input.actions.truncate(2);
        input.actions[0].effort_units = 1;
        input.actions[1].effort_units = 1;
        input.joint_world_supports.truncate(2);
        input.joint_world_supports[1].possible_joint_outcomes[0].outcomes[0].outcome =
            "x-right".into();
        for support in &mut input.joint_world_supports {
            support.possible_joint_outcomes[0]
                .outcomes
                .retain(|outcome| outcome.action_id != "inspect-z");
        }
        input.scenario_set_sha256 = canonical_scenario_set_sha256(&input.scenarios);
        input.action_set_sha256 = canonical_action_set_sha256(&input.actions);
        input.joint_support_sha256 = canonical_joint_support_sha256(
            &input.scenarios,
            &input.actions,
            &input.joint_world_supports,
        )
        .unwrap();
        input.max_total_effort_units = 1;

        let report = evaluate(&input, &bytes(&input)).unwrap();
        assert_eq!(report.worst_case_effort_units, Some(1));
        assert_eq!(report.optimal_policy_count, 2);
        assert!(report.policy.is_some());
    }

    #[test]
    fn additive_support_expansion_cannot_lower_optimal_effort() {
        let input = support_expansion_fixture();
        let report = evaluate(&input, &bytes(&input)).unwrap();
        assert_eq!(report.worst_case_effort_units, Some(4));
        assert_eq!(
            report.fixed_non_adaptive_optimum.minimum_effort_units,
            Some(5)
        );
        assert_eq!(report.support_challenge_count, 1);
        let challenge = &report.support_challenges[0];
        assert_eq!(challenge.added_joint_tuple_count, 1);
        assert_eq!(challenge.adaptive_worst_case_effort_before, Some(4));
        assert_eq!(challenge.adaptive_worst_case_effort_after, Some(5));
        assert_eq!(challenge.fixed_minimum_effort_before, Some(5));
        assert_eq!(challenge.fixed_minimum_effort_after, Some(5));
        assert!(challenge.adaptive_effort_non_decreasing);
        assert!(challenge.fixed_effort_non_decreasing);
        assert!(challenge.monotonicity_passed);
    }

    #[test]
    fn additive_support_cannot_resurrect_a_previously_impossible_policy() {
        let input = no_policy_support_expansion_fixture();
        let report = evaluate(&input, &bytes(&input)).unwrap();
        assert_eq!(report.worst_case_effort_units, None);
        let challenge = &report.support_challenges[0];
        assert_eq!(challenge.adaptive_worst_case_effort_before, None);
        assert_eq!(challenge.adaptive_worst_case_effort_after, None);
        assert_eq!(challenge.adaptive_status_before, "NO_POLICY");
        assert_eq!(challenge.adaptive_status_after, "NO_POLICY");
        assert!(challenge.adaptive_effort_non_decreasing);
        assert!(challenge.monotonicity_passed);
    }

    #[test]
    fn support_expansion_rejects_a_tuple_already_in_base_support() {
        let mut input = support_expansion_fixture();
        input.support_challenges[0].additions[0].possible_joint_outcomes[0].outcomes[1].outcome =
            "y-same".into();
        input.support_challenges[0].additions[0].possible_joint_outcomes[0].outcomes[2].outcome =
            "z-d".into();
        let error = evaluate(&input, &bytes(&input)).unwrap_err();
        assert!(error.contains("already present"));
    }

    #[test]
    fn no_policy_cannot_be_resurrected_by_adding_support() {
        assert!(effort_is_non_decreasing(None, None));
        assert!(!effort_is_non_decreasing(None, Some(1)));
        assert!(effort_is_non_decreasing(Some(1), None));
    }

    #[test]
    fn action_order_does_not_change_policy_cost() {
        let mut input = adaptive_fixture();
        let before = evaluate(&input, &bytes(&input)).unwrap();
        input.actions.reverse();
        for support in &mut input.joint_world_supports {
            support.possible_joint_outcomes.reverse();
            for tuple in &mut support.possible_joint_outcomes {
                tuple.outcomes.reverse();
            }
        }
        let reordered = evaluate(&input, &bytes(&input)).unwrap();
        assert_eq!(
            before.worst_case_effort_units,
            reordered.worst_case_effort_units
        );
        assert_eq!(
            before.maximum_terminal_path_effort_units,
            reordered.maximum_terminal_path_effort_units
        );
    }
}
