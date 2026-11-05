//! Exact query disambiguation under correlated joint observation supports.
//!
//! This tool models admissible joint observation tuples directly. It never
//! factorizes marginal supports and never infers independence between actions.
//! A selected action subset separates two worlds only when their projected
//! joint-support relations are disjoint.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const MAX_ACTIONS: usize = 16;
const MAX_SCENARIOS: usize = 24;
const MAX_JOINT_TUPLES_PER_WORLD: usize = 128;
const MAX_RETURNED_PLANS: usize = 128;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-joint-observation-disambiguation-audit",
    version,
    about = "Find exact minimum-cost action sets that separate correlated joint observation supports"
)]
struct Args {
    /// Frozen worlds, action universe, and correlated joint observation supports.
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
struct JointOutcome {
    outcomes: Vec<ActionOutcome>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ActionOutcome {
    action_id: String,
    outcome: String,
}

#[derive(Debug, Clone, Serialize, Eq, PartialEq, Ord, PartialOrd)]
struct WorldPair {
    world_a: String,
    world_b: String,
}

#[derive(Debug, Serialize)]
struct ActionMarginalCoverage {
    action_id: String,
    effort_units: u64,
    marginally_separated_decision_conflict_pair_count: usize,
}

#[derive(Debug, Serialize)]
struct JointPlan {
    action_ids: Vec<String>,
    effort_units: u64,
    jointly_separated_decision_conflict_pair_count: usize,
}

#[derive(Debug, Serialize)]
struct PairDiagnostic {
    pair: WorldPair,
    marginally_separated_by_action_ids: Vec<String>,
    jointly_separable_with_full_action_universe: bool,
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
    scenario_set_scope: String,
    scenario_set_sha256: String,
    action_set_sha256: String,
    joint_support_scope: String,
    joint_support_sha256: String,
    effort_unit: String,
    max_total_effort_units: u64,
    scenario_count: usize,
    observation_action_count: usize,
    joint_support_tuple_count: usize,
    decision_conflict_pair_count: usize,
    marginally_separable_pair_count: usize,
    jointly_separable_pair_count: usize,
    jointly_undiscriminable_pairs: Vec<WorldPair>,
    pair_diagnostics: Vec<PairDiagnostic>,
    action_marginal_coverage: Vec<ActionMarginalCoverage>,
    subset_count_evaluated: u64,
    minimum_joint_disambiguation_effort_units: Option<u64>,
    minimum_joint_plan_count: u64,
    joint_plans_returned: usize,
    minimum_joint_plans_truncated: bool,
    minimum_joint_plan_within_budget: Option<bool>,
    joint_plans: Vec<JointPlan>,
    marginal_separation_rule: &'static str,
    joint_separation_rule: &'static str,
    probability_claim: &'static str,
    status: &'static str,
    search_complete: bool,
    scope_note: &'static str,
}

type JointTuplesByWorld = Vec<Vec<Vec<String>>>;
type PairOverlapTable = Vec<bool>;

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
    world_support: &JointWorldSupport,
    action_ids: &[String],
) -> Result<Vec<Vec<String>>, String> {
    let known_actions = action_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut rows = Vec::with_capacity(world_support.possible_joint_outcomes.len());
    let mut unique_rows = BTreeSet::new();
    for (row_index, tuple) in world_support.possible_joint_outcomes.iter().enumerate() {
        if tuple.outcomes.len() != action_ids.len() {
            return Err(format!(
                "world {} joint tuple {row_index} must contain exactly one outcome for every action",
                world_support.scenario_id
            ));
        }
        let mut by_action = BTreeMap::new();
        for outcome in &tuple.outcomes {
            if !known_actions.contains(outcome.action_id.as_str()) {
                return Err(format!(
                    "world {} joint tuple {row_index} references unknown action {}",
                    world_support.scenario_id, outcome.action_id
                ));
            }
            if !nonempty(&outcome.outcome) {
                return Err(format!(
                    "world {} joint tuple {row_index} has an empty outcome for {}",
                    world_support.scenario_id, outcome.action_id
                ));
            }
            if by_action
                .insert(outcome.action_id.as_str(), outcome.outcome.as_str())
                .is_some()
            {
                return Err(format!(
                    "world {} joint tuple {row_index} repeats action {}",
                    world_support.scenario_id, outcome.action_id
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
                            "world {} joint tuple {row_index} omits action {}",
                            world_support.scenario_id, action_id
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !unique_rows.insert(row.clone()) {
            return Err(format!(
                "world {} contains duplicate joint outcome tuples",
                world_support.scenario_id
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
    let scenarios = sorted_scenarios(scenarios);
    let actions = sorted_actions(actions);
    let action_ids = actions
        .iter()
        .map(|action| action.action_id.clone())
        .collect::<Vec<_>>();
    let mut support_rows = supports.iter().collect::<Vec<_>>();
    support_rows.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut hasher = Sha256::new();
    hasher.update(b"connectome-joint-observation-support-set-v1");
    hasher.update((scenarios.len() as u64).to_be_bytes());
    for scenario in scenarios {
        let support = support_rows
            .iter()
            .find(|row| row.scenario_id == scenario.scenario_id)
            .ok_or_else(|| format!("missing joint support for {}", scenario.scenario_id))?;
        hasher.update(b"WORLD-SUPPORT");
        hash_field(&mut hasher, &scenario.scenario_id);
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

fn validate(input: &Input) -> Result<(Vec<String>, JointTuplesByWorld), String> {
    if input.schema_version != 1 {
        return Err("schema_version must equal 1".into());
    }
    if !nonempty(&input.analysis_id) || !nonempty(&input.query_id) {
        return Err("analysis_id and query_id must be non-empty".into());
    }
    if !valid_hex(&input.dataset_revision_commit_sha, 40) {
        return Err("dataset_revision_commit_sha must be lowercase 40-character Git SHA".into());
    }
    if !valid_hex(&input.query_spec_sha256, 64)
        || !valid_hex(&input.scenario_set_sha256, 64)
        || !valid_hex(&input.action_set_sha256, 64)
        || !valid_hex(&input.joint_support_sha256, 64)
    {
        return Err(
            "query, scenario, action, and joint-support digests must be lowercase SHA-256".into(),
        );
    }
    if !matches!(
        input.scenario_set_scope.as_str(),
        "exhaustive_declared_scenario_space" | "sampled_sensitivity_set"
    ) {
        return Err("scenario_set_scope is not a recognized finite-world scope".into());
    }
    if !matches!(
        input.joint_support_scope.as_str(),
        "exhaustive_declared_joint_support" | "sampled_sensitivity_support"
    ) {
        return Err("joint_support_scope is not a recognized finite-support scope".into());
    }
    if !nonempty(&input.effort_unit) || input.max_total_effort_units == 0 {
        return Err("effort_unit must be non-empty and max_total_effort_units positive".into());
    }
    if !(2..=MAX_SCENARIOS).contains(&input.scenarios.len()) {
        return Err(format!(
            "scenarios must contain between 2 and {MAX_SCENARIOS} worlds"
        ));
    }
    if input.actions.is_empty() || input.actions.len() > MAX_ACTIONS {
        return Err(format!(
            "actions must contain between 1 and {MAX_ACTIONS} observation actions"
        ));
    }
    for action in &input.actions {
        if action.effort_units == 0 {
            return Err(format!(
                "action {} effort_units must be positive",
                action.action_id
            ));
        }
    }

    let mut scenario_ids = BTreeSet::new();
    for scenario in &input.scenarios {
        if !nonempty(&scenario.scenario_id) || !scenario_ids.insert(scenario.scenario_id.clone()) {
            return Err("scenario IDs must be non-empty and unique".into());
        }
        if !valid_hex(&scenario.graph_sha256, 64) {
            return Err(format!(
                "scenario {} graph_sha256 must be lowercase SHA-256",
                scenario.scenario_id
            ));
        }
    }
    let actual_scenario_digest = canonical_scenario_set_sha256(&input.scenarios);
    if actual_scenario_digest != input.scenario_set_sha256 {
        return Err(format!(
            "scenario-set digest mismatch: declared {}, computed {}",
            input.scenario_set_sha256, actual_scenario_digest
        ));
    }

    let mut action_ids_seen = BTreeSet::new();
    for action in &input.actions {
        if !nonempty(&action.action_id) || !action_ids_seen.insert(action.action_id.clone()) {
            return Err("action IDs must be non-empty and unique".into());
        }
        if !valid_hex(&action.source_evidence_sha256, 64)
            || !valid_hex(&action.observation_model_sha256, 64)
        {
            return Err(format!(
                "action {} source/model digests must be lowercase SHA-256",
                action.action_id
            ));
        }
        let mut groups = BTreeSet::new();
        for group in &action.dependence_groups {
            if !nonempty(group) || !groups.insert(group) {
                return Err(format!(
                    "action {} has empty or duplicate dependence groups",
                    action.action_id
                ));
            }
        }
    }
    let actions = sorted_actions(&input.actions);
    let action_ids = actions
        .iter()
        .map(|action| action.action_id.clone())
        .collect::<Vec<_>>();
    let action_digest = canonical_action_set_sha256(&input.actions);
    if action_digest != input.action_set_sha256 {
        return Err(format!(
            "action-set digest mismatch: declared {}, computed {}",
            input.action_set_sha256, action_digest
        ));
    }

    let mut support_ids = BTreeSet::new();
    let mut supports_by_world = BTreeMap::new();
    for support in &input.joint_world_supports {
        if !scenario_ids.contains(&support.scenario_id)
            || !support_ids.insert(support.scenario_id.clone())
        {
            return Err("joint supports must reference each scenario exactly once".into());
        }
        if !valid_hex(&support.source_evidence_sha256, 64)
            || !valid_hex(&support.joint_observation_model_sha256, 64)
        {
            return Err(format!(
                "world {} source/model digests must be lowercase SHA-256",
                support.scenario_id
            ));
        }
        if support.possible_joint_outcomes.is_empty()
            || support.possible_joint_outcomes.len() > MAX_JOINT_TUPLES_PER_WORLD
        {
            return Err(format!(
                "world {} must contain between 1 and {} joint outcome tuples",
                support.scenario_id, MAX_JOINT_TUPLES_PER_WORLD
            ));
        }
        let mut groups = BTreeSet::new();
        for group in &support.dependence_groups {
            if !nonempty(group) || !groups.insert(group) {
                return Err(format!(
                    "world {} has empty or duplicate dependence groups",
                    support.scenario_id
                ));
            }
        }
        let rows = normalized_joint_rows(support, &action_ids)?;
        supports_by_world.insert(support.scenario_id.clone(), rows);
    }
    if support_ids != scenario_ids {
        return Err(
            "joint_world_supports must contain exactly one support relation for every scenario"
                .into(),
        );
    }

    let actual_joint_digest = canonical_joint_support_sha256(
        &input.scenarios,
        &input.actions,
        &input.joint_world_supports,
    )?;
    if actual_joint_digest != input.joint_support_sha256 {
        return Err(format!(
            "joint-support digest mismatch: declared {}, computed {}",
            input.joint_support_sha256, actual_joint_digest
        ));
    }

    let sorted_worlds = sorted_scenarios(&input.scenarios);
    let tuples_by_world = sorted_worlds
        .iter()
        .map(|world| {
            supports_by_world.remove(&world.scenario_id).ok_or_else(|| {
                format!("missing normalized joint support for {}", world.scenario_id)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((action_ids, tuples_by_world))
}

fn decision_conflict_pairs(scenarios: &[Scenario]) -> Vec<WorldPair> {
    let worlds = sorted_scenarios(scenarios);
    let mut pairs = Vec::new();
    for left in 0..worlds.len() {
        for right in (left + 1)..worlds.len() {
            if worlds[left].decision_met != worlds[right].decision_met {
                pairs.push(WorldPair {
                    world_a: worlds[left].scenario_id.clone(),
                    world_b: worlds[right].scenario_id.clone(),
                });
            }
        }
    }
    pairs
}

fn marginal_sets(tuples: &[Vec<String>], action_index: usize) -> BTreeSet<String> {
    tuples
        .iter()
        .map(|tuple| tuple[action_index].clone())
        .collect()
}

/// For a world pair, overlap[mask] is true iff at least one cross-world tuple
/// pair agrees on every selected action in mask. A superset zeta transform
/// fills all subsets of each equality mask. Robust separation is then !overlap[mask].
fn pair_overlap_table(
    left_tuples: &[Vec<String>],
    right_tuples: &[Vec<String>],
    action_count: usize,
) -> PairOverlapTable {
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

fn marginal_action_coverage(
    action_ids: &[String],
    action_effort: &[u64],
    pairs: &[WorldPair],
    scenarios: &[Scenario],
    tuples_by_world: &[Vec<Vec<String>>],
) -> Vec<ActionMarginalCoverage> {
    let sorted_worlds = sorted_scenarios(scenarios);
    let world_index = sorted_worlds
        .iter()
        .enumerate()
        .map(|(index, world)| (world.scenario_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    action_ids
        .iter()
        .enumerate()
        .map(|(action_index, action_id)| {
            let count = pairs
                .iter()
                .filter(|pair| {
                    let left = world_index[pair.world_a.as_str()];
                    let right = world_index[pair.world_b.as_str()];
                    marginal_sets(&tuples_by_world[left], action_index)
                        .is_disjoint(&marginal_sets(&tuples_by_world[right], action_index))
                })
                .count();
            ActionMarginalCoverage {
                action_id: action_id.clone(),
                effort_units: action_effort[action_index],
                marginally_separated_decision_conflict_pair_count: count,
            }
        })
        .collect()
}

fn evaluate(input: &Input, input_bytes: &[u8]) -> Result<Report, String> {
    let (action_ids, tuples_by_world) = validate(input)?;
    let scenarios = sorted_scenarios(&input.scenarios);
    let actions = sorted_actions(&input.actions);
    let action_effort = actions
        .iter()
        .map(|action| action.effort_units)
        .collect::<Vec<_>>();
    let pairs = decision_conflict_pairs(&scenarios);
    let world_index = scenarios
        .iter()
        .enumerate()
        .map(|(index, world)| (world.scenario_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let subset_count = 1_usize << action_ids.len();
    let full_mask = subset_count - 1;

    let mut overlap_tables = Vec::with_capacity(pairs.len());
    let mut pair_diagnostics = Vec::with_capacity(pairs.len());
    let mut marginally_separable_pair_count = 0_usize;
    let mut jointly_undiscriminable_pairs = Vec::new();

    for pair in &pairs {
        let left = world_index[pair.world_a.as_str()];
        let right = world_index[pair.world_b.as_str()];
        let table = pair_overlap_table(
            &tuples_by_world[left],
            &tuples_by_world[right],
            action_ids.len(),
        );
        let marginal_action_ids = (0..action_ids.len())
            .filter(|action_index| !table[1_usize << action_index])
            .map(|action_index| action_ids[action_index].clone())
            .collect::<Vec<_>>();
        if !marginal_action_ids.is_empty() {
            marginally_separable_pair_count += 1;
        }
        let jointly_separable = !table[full_mask];
        if !jointly_separable {
            jointly_undiscriminable_pairs.push(pair.clone());
        }
        pair_diagnostics.push(PairDiagnostic {
            pair: pair.clone(),
            marginally_separated_by_action_ids: marginal_action_ids,
            jointly_separable_with_full_action_universe: jointly_separable,
        });
        overlap_tables.push(table);
    }

    let jointly_separable_pair_count = pairs.len() - jointly_undiscriminable_pairs.len();
    let marginal_coverage = marginal_action_coverage(
        &action_ids,
        &action_effort,
        &pairs,
        &scenarios,
        &tuples_by_world,
    );

    let total_subsets = (subset_count - 1) as u64;
    let mut best_cost: Option<u64> = None;
    let mut best_count = 0_u64;
    let mut plans = Vec::new();
    let evaluated_subsets;

    if pairs.is_empty() {
        best_cost = Some(0);
        best_count = 1;
        plans.push(JointPlan {
            action_ids: Vec::new(),
            effort_units: 0,
            jointly_separated_decision_conflict_pair_count: 0,
        });
        evaluated_subsets = 1;
    } else if !jointly_undiscriminable_pairs.is_empty() {
        evaluated_subsets = 0;
    } else {
        evaluated_subsets = total_subsets;
        for mask in 1..subset_count {
            let mut cost = 0_u64;
            for (index, effort) in action_effort.iter().enumerate() {
                if mask & (1_usize << index) != 0 {
                    cost = cost
                        .checked_add(*effort)
                        .ok_or_else(|| "action subset effort overflows u64".to_string())?;
                }
            }
            if best_cost.is_some_and(|best| cost > best) {
                continue;
            }
            let separates_every_pair = overlap_tables.iter().all(|table| !table[mask]);
            if !separates_every_pair {
                continue;
            }
            if best_cost.is_none_or(|best| cost < best) {
                best_cost = Some(cost);
                best_count = 0;
                plans.clear();
            }
            if best_cost == Some(cost) {
                best_count += 1;
                if plans.len() < MAX_RETURNED_PLANS {
                    plans.push(JointPlan {
                        action_ids: selected_action_ids(mask, &action_ids),
                        effort_units: cost,
                        jointly_separated_decision_conflict_pair_count: pairs.len(),
                    });
                }
            }
        }
    }
    plans.sort_by(|left, right| left.action_ids.cmp(&right.action_ids));
    let minimum_joint_plan_within_budget =
        best_cost.map(|cost| cost <= input.max_total_effort_units);
    let status = if pairs.is_empty() {
        "NO_DECISION_DISAGREEMENT_WITHIN_DECLARED_SCENARIOS"
    } else if !jointly_undiscriminable_pairs.is_empty() {
        "NO_JOINT_DISAMBIGUATION_WITHIN_DECLARED_SUPPORT"
    } else if minimum_joint_plan_within_budget == Some(true) {
        "MINIMUM_COST_JOINT_DISAMBIGUATION_PLAN"
    } else {
        "MINIMUM_COST_JOINT_DISAMBIGUATION_EXCEEDS_BUDGET"
    };
    let joint_tuple_count = tuples_by_world.iter().map(Vec::len).sum();

    Ok(Report {
        schema_version: 1,
        tool: "connectome-joint-observation-disambiguation-audit",
        analysis_id: input.analysis_id.clone(),
        input_sha256: sha256(input_bytes),
        dataset_revision_commit_sha: input.dataset_revision_commit_sha.clone(),
        query_id: input.query_id.clone(),
        query_spec_sha256: input.query_spec_sha256.clone(),
        scenario_set_scope: input.scenario_set_scope.clone(),
        scenario_set_sha256: input.scenario_set_sha256.clone(),
        action_set_sha256: input.action_set_sha256.clone(),
        joint_support_scope: input.joint_support_scope.clone(),
        joint_support_sha256: input.joint_support_sha256.clone(),
        effort_unit: input.effort_unit.clone(),
        max_total_effort_units: input.max_total_effort_units,
        scenario_count: input.scenarios.len(),
        observation_action_count: input.actions.len(),
        joint_support_tuple_count: joint_tuple_count,
        decision_conflict_pair_count: pairs.len(),
        marginally_separable_pair_count,
        jointly_separable_pair_count,
        jointly_undiscriminable_pairs,
        pair_diagnostics,
        action_marginal_coverage: marginal_coverage,
        subset_count_evaluated: evaluated_subsets,
        minimum_joint_disambiguation_effort_units: best_cost,
        minimum_joint_plan_count: best_count,
        joint_plans_returned: plans.len(),
        minimum_joint_plans_truncated: best_count > MAX_RETURNED_PLANS as u64,
        minimum_joint_plan_within_budget,
        joint_plans: plans,
        marginal_separation_rule: "SINGLE_ACTION_MARGINAL_SUPPORTS_DISJOINT",
        joint_separation_rule: "PROJECTED_JOINT_SUPPORT_RELATIONS_DISJOINT",
        probability_claim: "NONE_FINITE_SET_VALUED_CORRELATED_SUPPORT_ONLY",
        status,
        search_complete: true,
        scope_note: "The planner uses caller-supplied finite joint-support relations and exact projection onto selected action subsets. It does not factorize marginal supports, infer independence, assign probabilities, prove that joint-support rows are complete, validate measurement models, or establish biological truth. Results are scoped to the declared worlds, actions, and joint support relations.",
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let input_bytes = match fs::read(&args.input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!(
                "connectome-joint-observation-disambiguation-audit: could not read input: {error}"
            );
            return ExitCode::from(2);
        }
    };
    let input: Input = match serde_json::from_slice(&input_bytes) {
        Ok(input) => input,
        Err(error) => {
            eprintln!(
                "connectome-joint-observation-disambiguation-audit: invalid JSON input: {error}"
            );
            return ExitCode::from(2);
        }
    };
    let report = match evaluate(&input, &input_bytes) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("connectome-joint-observation-disambiguation-audit: {error}");
            return ExitCode::from(2);
        }
    };
    let rendered = match serde_json::to_string_pretty(&report) {
        Ok(json) => json + "\n",
        Err(error) => {
            eprintln!(
                "connectome-joint-observation-disambiguation-audit: could not serialize report: {error}"
            );
            return ExitCode::from(2);
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(path, rendered.as_bytes()) {
            eprintln!(
                "connectome-joint-observation-disambiguation-audit: could not write report: {error}"
            );
            return ExitCode::from(2);
        }
    } else {
        print!("{rendered}");
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xor_fixture() -> Input {
        serde_json::from_str(include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_JOINT_OBSERVATION_XOR.example.json"
        ))
        .expect("XOR joint-support fixture must parse")
    }

    fn overlap_fixture() -> Input {
        serde_json::from_str(include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_JOINT_OBSERVATION_SHARED_TUPLE.example.json"
        ))
        .expect("shared-tuple joint-support fixture must parse")
    }

    fn bytes(input: &Input) -> Vec<u8> {
        serde_json::to_vec(input).expect("fixture serializes")
    }

    #[test]
    fn joint_projection_recovers_complementarity_that_marginals_miss() {
        let input = xor_fixture();
        let report = evaluate(&input, &bytes(&input)).unwrap();
        assert_eq!(report.decision_conflict_pair_count, 1);
        assert_eq!(report.marginally_separable_pair_count, 0);
        assert_eq!(report.jointly_separable_pair_count, 1);
        assert_eq!(report.minimum_joint_disambiguation_effort_units, Some(3));
        assert_eq!(report.minimum_joint_plan_count, 1);
        assert_eq!(report.subset_count_evaluated, 3);
        assert_eq!(report.status, "MINIMUM_COST_JOINT_DISAMBIGUATION_PLAN");
        assert_eq!(
            report.joint_plans[0].action_ids,
            vec!["inspect-x".to_string(), "inspect-y".to_string()]
        );
    }

    #[test]
    fn shared_joint_tuple_blocks_false_separation() {
        let input = overlap_fixture();
        let report = evaluate(&input, &bytes(&input)).unwrap();
        assert_eq!(report.marginally_separable_pair_count, 0);
        assert_eq!(report.jointly_separable_pair_count, 0);
        assert_eq!(report.minimum_joint_disambiguation_effort_units, None);
        assert_eq!(
            report.status,
            "NO_JOINT_DISAMBIGUATION_WITHIN_DECLARED_SUPPORT"
        );
        assert_eq!(report.jointly_undiscriminable_pairs.len(), 1);
    }

    #[test]
    fn action_and_tuple_order_do_not_change_canonical_digests() {
        let mut input = xor_fixture();
        let action_digest = canonical_action_set_sha256(&input.actions);
        let joint_digest = canonical_joint_support_sha256(
            &input.scenarios,
            &input.actions,
            &input.joint_world_supports,
        )
        .unwrap();
        input.actions.reverse();
        for support in &mut input.joint_world_supports {
            support.possible_joint_outcomes.reverse();
            for tuple in &mut support.possible_joint_outcomes {
                tuple.outcomes.reverse();
            }
        }
        assert_eq!(action_digest, canonical_action_set_sha256(&input.actions));
        assert_eq!(
            joint_digest,
            canonical_joint_support_sha256(
                &input.scenarios,
                &input.actions,
                &input.joint_world_supports
            )
            .unwrap()
        );
    }

    #[test]
    fn malformed_joint_tuple_fails_closed_when_action_is_omitted() {
        let mut input = xor_fixture();
        input.joint_world_supports[0].possible_joint_outcomes[0]
            .outcomes
            .pop();
        assert!(
            validate(&input)
                .unwrap_err()
                .contains("exactly one outcome for every action")
        );
    }

    #[test]
    fn duplicated_joint_tuple_fails_closed() {
        let mut input = xor_fixture();
        let duplicate = input.joint_world_supports[0].possible_joint_outcomes[0].clone();
        input.joint_world_supports[0]
            .possible_joint_outcomes
            .push(duplicate);
        let action_ids = sorted_actions(&input.actions)
            .iter()
            .map(|action| action.action_id.clone())
            .collect::<Vec<_>>();
        assert!(
            normalized_joint_rows(&input.joint_world_supports[0], &action_ids)
                .unwrap_err()
                .contains("duplicate joint outcome tuples")
        );
    }

    #[test]
    fn support_digest_mismatch_fails_closed() {
        let mut input = xor_fixture();
        input.joint_support_sha256 = "0".repeat(64);
        assert!(
            validate(&input)
                .unwrap_err()
                .contains("joint-support digest mismatch")
        );
    }

    #[test]
    fn all_action_projection_is_monotone_with_selected_action_set_size() {
        let input = xor_fixture();
        let report = evaluate(&input, &bytes(&input)).unwrap();
        let (action_ids, tuples) = validate(&input).unwrap();
        let pairs = decision_conflict_pairs(&input.scenarios);
        let worlds = sorted_scenarios(&input.scenarios);
        let world_index = worlds
            .iter()
            .enumerate()
            .map(|(index, world)| (world.scenario_id.as_str(), index))
            .collect::<BTreeMap<_, _>>();
        let pair = &pairs[0];
        let left = world_index[pair.world_a.as_str()];
        let right = world_index[pair.world_b.as_str()];
        let table = pair_overlap_table(&tuples[left], &tuples[right], action_ids.len());
        let one_action_separations = (0..action_ids.len())
            .map(|index| !table[1_usize << index])
            .collect::<Vec<_>>();
        assert!(one_action_separations.iter().all(|separated| !separated));
        assert!(!table[(1_usize << action_ids.len()) - 1]);
        assert_eq!(report.jointly_separable_pair_count, 1);
    }

    #[test]
    fn support_tuple_limit_is_enforced() {
        let mut input = xor_fixture();
        input.joint_world_supports[0].possible_joint_outcomes = (0..=MAX_JOINT_TUPLES_PER_WORLD)
            .map(|index| JointOutcome {
                outcomes: vec![
                    ActionOutcome {
                        action_id: "inspect-x".into(),
                        outcome: format!("x-{index}"),
                    },
                    ActionOutcome {
                        action_id: "inspect-y".into(),
                        outcome: format!("y-{index}"),
                    },
                ],
            })
            .collect();
        assert!(
            validate(&input)
                .unwrap_err()
                .contains("must contain between 1 and")
        );
    }
}
