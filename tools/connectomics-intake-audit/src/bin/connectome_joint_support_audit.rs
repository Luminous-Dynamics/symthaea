//! Exact robust discrimination over declared joint observation-support relations.
//!
//! No independence or probability model is inferred. A selected action subset
//! separates two worlds only when their projected joint-support relations are disjoint.
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    process::ExitCode,
};

const MAX_ACTIONS: usize = 20;
const MAX_SCENARIOS: usize = 24;
const MAX_JOINT_ROWS_PER_WORLD: usize = 64;
const MAX_TOTAL_JOINT_ROWS: usize = 1536;
const MAX_RETURNED_PLANS: usize = 128;
const MAX_SUBSET_MARK_STEPS: u64 = 50_000_000;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-joint-support-audit",
    about = "Exact robust query discrimination over correlated joint observation supports"
)]
struct Args {
    #[arg(long)]
    input: PathBuf,
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
    joint_support_set_sha256: String,
    effort_unit: String,
    max_total_effort_units: u64,
    scenarios: Vec<Scenario>,
    actions: Vec<Action>,
    joint_supports: Vec<JointSupport>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    graph_sha256: String,
    decision_met: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Action {
    action_id: String,
    action_kind: String,
    effort_units: u64,
    source_evidence_sha256: String,
    dependence_groups: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct JointSupport {
    scenario_id: String,
    joint_observation_model_sha256: String,
    source_evidence_sha256: String,
    dependence_groups: Vec<String>,
    rows: Vec<JointRow>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct JointRow {
    outcomes: Vec<JointOutcome>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct JointOutcome {
    action_id: String,
    value: String,
}

#[derive(Debug, Clone, Serialize, Eq, PartialEq, Ord, PartialOrd)]
struct WorldPair {
    world_a: String,
    world_b: String,
}

#[derive(Debug, Serialize)]
struct MarginalActionCoverage {
    action_id: String,
    effort_units: u64,
    separated_conflict_pairs: Vec<WorldPair>,
}

#[derive(Debug, Serialize)]
struct JointPlan {
    action_ids: Vec<String>,
    effort_units: u64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    tool: &'static str,
    analysis_id: String,
    dataset_revision_commit_sha: String,
    query_id: String,
    query_spec_sha256: String,
    scenario_set_sha256: String,
    action_set_sha256: String,
    joint_support_set_sha256: String,
    input_sha256: String,
    scenario_count: usize,
    action_count: usize,
    total_joint_support_rows: usize,
    joint_support_scope: String,
    effort_unit: String,
    max_total_effort_units: u64,
    decision_conflict_pair_count: usize,
    marginal_action_coverage: Vec<MarginalActionCoverage>,
    marginally_separable_pair_count: usize,
    marginally_undiscriminable_pairs: Vec<WorldPair>,
    jointly_separable_pair_count: usize,
    jointly_undiscriminable_pairs: Vec<WorldPair>,
    unavoidable_shared_full_joint_tuple_pairs: Vec<WorldPair>,
    distinct_difference_constraint_count: usize,
    minimal_difference_constraint_count: usize,
    candidate_action_subsets: u64,
    subset_mark_steps: u64,
    minimum_joint_disambiguation_effort_units: Option<u64>,
    minimum_joint_plan_count: u64,
    minimum_joint_plans_returned: usize,
    minimum_joint_plans_truncated: bool,
    minimum_joint_plans: Vec<JointPlan>,
    minimum_joint_plan_within_budget: Option<bool>,
    search_complete: bool,
    status: &'static str,
    probability_claim: &'static str,
    scope_note: &'static str,
}

#[derive(Debug)]
struct NormalizedSupport {
    scenario_id: String,
    rows: Vec<Vec<String>>,
}

#[derive(Debug)]
struct Validated {
    actions: Vec<Action>,
    scenarios: Vec<Scenario>,
    supports: Vec<NormalizedSupport>,
    total_rows: usize,
}

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

fn canonical_scenario_set_sha256(scenarios: &[Scenario]) -> String {
    let mut rows = scenarios.to_vec();
    rows.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut hasher = Sha256::new();
    hasher.update(b"connectome-query-disambiguation-scenario-set-v1");
    hasher.update((rows.len() as u64).to_be_bytes());
    for row in rows {
        hasher.update(b"WORLD");
        hash_field(&mut hasher, &row.scenario_id);
        hasher.update(row.graph_sha256.as_bytes());
        hasher.update([u8::from(row.decision_met)]);
    }
    hex::encode(hasher.finalize())
}

fn canonical_action_set_sha256(actions: &[Action]) -> String {
    let mut rows = actions.to_vec();
    rows.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    let mut hasher = Sha256::new();
    hasher.update(b"connectome-joint-observation-action-set-v1");
    hasher.update((rows.len() as u64).to_be_bytes());
    for mut row in rows {
        row.dependence_groups.sort();
        hasher.update(b"ACTION");
        hash_field(&mut hasher, &row.action_id);
        hash_field(&mut hasher, &row.action_kind);
        hasher.update(row.effort_units.to_be_bytes());
        hasher.update(row.source_evidence_sha256.as_bytes());
        hasher.update((row.dependence_groups.len() as u64).to_be_bytes());
        for group in row.dependence_groups {
            hasher.update(b"DEPENDENCE");
            hash_field(&mut hasher, &group);
        }
    }
    hex::encode(hasher.finalize())
}

fn canonical_joint_support_set_sha256(supports: &[JointSupport]) -> String {
    let mut worlds = supports.to_vec();
    worlds.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut hasher = Sha256::new();
    hasher.update(b"connectome-joint-observation-support-set-v1");
    hasher.update((worlds.len() as u64).to_be_bytes());
    for mut world in worlds {
        world.dependence_groups.sort();
        hasher.update(b"SUPPORT");
        hash_field(&mut hasher, &world.scenario_id);
        hasher.update(world.joint_observation_model_sha256.as_bytes());
        hasher.update(world.source_evidence_sha256.as_bytes());
        hasher.update((world.dependence_groups.len() as u64).to_be_bytes());
        for group in world.dependence_groups {
            hasher.update(b"DEPENDENCE");
            hash_field(&mut hasher, &group);
        }

        let mut rows = world
            .rows
            .into_iter()
            .map(|mut row| {
                row.outcomes
                    .sort_by(|left, right| left.action_id.cmp(&right.action_id));
                row.outcomes
            })
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            left.iter()
                .map(|value| (&value.action_id, &value.value))
                .cmp(right.iter().map(|value| (&value.action_id, &value.value)))
        });
        hasher.update((rows.len() as u64).to_be_bytes());
        for row in rows {
            hasher.update(b"JOINT-ROW");
            hasher.update((row.len() as u64).to_be_bytes());
            for outcome in row {
                hasher.update(b"OUTCOME");
                hash_field(&mut hasher, &outcome.action_id);
                hash_field(&mut hasher, &outcome.value);
            }
        }
    }
    hex::encode(hasher.finalize())
}

fn validate(input: &Input) -> Result<Validated, String> {
    if input.schema_version != 1 || !nonempty(&input.analysis_id) || !nonempty(&input.query_id) {
        return Err("schema_version, analysis_id or query_id is invalid".into());
    }
    if !valid_hex(&input.dataset_revision_commit_sha, 40)
        || !valid_hex(&input.query_spec_sha256, 64)
        || !valid_hex(&input.scenario_set_sha256, 64)
        || !valid_hex(&input.action_set_sha256, 64)
        || !valid_hex(&input.joint_support_set_sha256, 64)
    {
        return Err("baseline identity or SHA-256 digest is malformed".into());
    }
    if !matches!(
        input.scenario_set_scope.as_str(),
        "exhaustive_declared_scenario_space" | "sampled_sensitivity_set"
    ) {
        return Err("scenario_set_scope is invalid".into());
    }
    if !matches!(
        input.joint_support_scope.as_str(),
        "exhaustive_declared_joint_support" | "sampled_joint_sensitivity_set"
    ) {
        return Err("joint_support_scope is invalid".into());
    }
    if !nonempty(&input.effort_unit) || input.max_total_effort_units == 0 {
        return Err("effort_unit and max_total_effort_units must be positive".into());
    }
    if !(2..=MAX_SCENARIOS).contains(&input.scenarios.len())
        || input.actions.is_empty()
        || input.actions.len() > MAX_ACTIONS
    {
        return Err(format!(
            "input requires 2..={MAX_SCENARIOS} worlds and 1..={MAX_ACTIONS} actions"
        ));
    }
    if input.joint_supports.len() != input.scenarios.len() {
        return Err("joint supports must cover every scenario exactly once".into());
    }
    if canonical_scenario_set_sha256(&input.scenarios) != input.scenario_set_sha256 {
        return Err("scenario-set digest mismatch".into());
    }
    if canonical_action_set_sha256(&input.actions) != input.action_set_sha256 {
        return Err("action-set digest mismatch".into());
    }
    if canonical_joint_support_set_sha256(&input.joint_supports) != input.joint_support_set_sha256 {
        return Err("joint-support-set digest mismatch".into());
    }

    let mut scenarios = input.scenarios.clone();
    scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut scenario_ids = BTreeSet::new();
    for scenario in &scenarios {
        if !nonempty(&scenario.scenario_id)
            || !scenario_ids.insert(scenario.scenario_id.clone())
            || !valid_hex(&scenario.graph_sha256, 64)
        {
            return Err("scenario IDs must be unique and graph digests valid".into());
        }
    }

    let mut actions = input.actions.clone();
    actions.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    let mut action_ids = BTreeSet::new();
    let mut total_effort = 0_u64;
    for action in &actions {
        if !nonempty(&action.action_id)
            || !action_ids.insert(action.action_id.clone())
            || action.effort_units == 0
        {
            return Err("action IDs must be unique and effort_units positive".into());
        }
        if !matches!(
            action.action_kind.as_str(),
            "image_inspection"
                | "morphology_review"
                | "synapse_adjudication"
                | "orthogonal_measurement"
                | "other"
        ) {
            return Err(format!(
                "action {} has invalid action_kind",
                action.action_id
            ));
        }
        if !valid_hex(&action.source_evidence_sha256, 64) {
            return Err(format!(
                "action {} source_evidence_sha256 must be lowercase SHA-256",
                action.action_id
            ));
        }
        total_effort = total_effort
            .checked_add(action.effort_units)
            .ok_or_else(|| "sum of action effort_units overflows u64".to_string())?;
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
    let _ = total_effort;

    let mut source_supports = input.joint_supports.clone();
    source_supports.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut normalized = Vec::new();
    let mut total_rows = 0_usize;
    for support in &source_supports {
        if !scenario_ids.contains(&support.scenario_id) {
            return Err(format!(
                "joint support references unknown scenario {}",
                support.scenario_id
            ));
        }
        if !valid_hex(&support.joint_observation_model_sha256, 64)
            || !valid_hex(&support.source_evidence_sha256, 64)
        {
            return Err(format!(
                "joint support {} has invalid source/model provenance digests",
                support.scenario_id
            ));
        }
        let mut groups = BTreeSet::new();
        for group in &support.dependence_groups {
            if !nonempty(group) || !groups.insert(group) {
                return Err(format!(
                    "joint support {} has empty or duplicate dependence groups",
                    support.scenario_id
                ));
            }
        }
        if support.rows.is_empty() || support.rows.len() > MAX_JOINT_ROWS_PER_WORLD {
            return Err(format!(
                "joint support {} must contain 1..={MAX_JOINT_ROWS_PER_WORLD} rows",
                support.scenario_id
            ));
        }
        total_rows = total_rows
            .checked_add(support.rows.len())
            .ok_or_else(|| "total joint support row count overflows usize".to_string())?;
        let mut row_set = BTreeSet::new();
        let mut rows = Vec::new();
        for row in &support.rows {
            if row.outcomes.len() != actions.len() {
                return Err(format!(
                    "joint row in {} must contain exactly one outcome for every declared action",
                    support.scenario_id
                ));
            }
            let mut values = BTreeMap::new();
            for outcome in &row.outcomes {
                if !action_ids.contains(&outcome.action_id) || !nonempty(&outcome.value) {
                    return Err(format!(
                        "joint row in {} has unknown action or empty outcome value",
                        support.scenario_id
                    ));
                }
                if values
                    .insert(outcome.action_id.clone(), outcome.value.clone())
                    .is_some()
                {
                    return Err(format!(
                        "joint row in {} repeats action {}",
                        support.scenario_id, outcome.action_id
                    ));
                }
            }
            let tuple = actions
                .iter()
                .map(|action| {
                    values
                        .get(&action.action_id)
                        .expect("exact action membership validated")
                        .clone()
                })
                .collect::<Vec<_>>();
            if !row_set.insert(tuple.clone()) {
                return Err(format!(
                    "joint support {} contains duplicate full joint rows",
                    support.scenario_id
                ));
            }
            rows.push(tuple);
        }
        rows.sort();
        normalized.push(NormalizedSupport {
            scenario_id: support.scenario_id.clone(),
            rows,
        });
    }
    if total_rows > MAX_TOTAL_JOINT_ROWS {
        return Err(format!(
            "total joint support rows exceed cap of {MAX_TOTAL_JOINT_ROWS}"
        ));
    }
    let normalized_ids = normalized
        .iter()
        .map(|support| support.scenario_id.clone())
        .collect::<BTreeSet<_>>();
    if normalized_ids != scenario_ids || normalized.len() != scenarios.len() {
        return Err("joint supports must cover every scenario exactly once".into());
    }

    Ok(Validated {
        actions,
        scenarios,
        supports: normalized,
        total_rows,
    })
}

fn decision_conflict_pairs(scenarios: &[Scenario]) -> Vec<WorldPair> {
    let mut pairs = Vec::new();
    for left in 0..scenarios.len() {
        for right in (left + 1)..scenarios.len() {
            if scenarios[left].decision_met != scenarios[right].decision_met {
                pairs.push(WorldPair {
                    world_a: scenarios[left].scenario_id.clone(),
                    world_b: scenarios[right].scenario_id.clone(),
                });
            }
        }
    }
    pairs
}

fn find_support<'a>(supports: &'a [NormalizedSupport], scenario_id: &str) -> &'a NormalizedSupport {
    supports
        .binary_search_by(|support| support.scenario_id.as_str().cmp(scenario_id))
        .map(|index| &supports[index])
        .expect("validated support set covers all scenarios")
}

fn pair_separated_for_mask(
    left: &NormalizedSupport,
    right: &NormalizedSupport,
    mask: usize,
) -> bool {
    for left_row in &left.rows {
        for right_row in &right.rows {
            let mut projected_equal = true;
            let mut remaining = mask;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                if left_row[bit] != right_row[bit] {
                    projected_equal = false;
                    break;
                }
                remaining &= remaining - 1;
            }
            if projected_equal {
                return false;
            }
        }
    }
    true
}

fn marginal_coverage(
    actions: &[Action],
    pairs: &[WorldPair],
    supports: &[NormalizedSupport],
) -> Vec<MarginalActionCoverage> {
    actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            let mask = 1_usize << index;
            let separated_conflict_pairs = pairs
                .iter()
                .filter(|pair| {
                    pair_separated_for_mask(
                        find_support(supports, &pair.world_a),
                        find_support(supports, &pair.world_b),
                        mask,
                    )
                })
                .cloned()
                .collect();
            MarginalActionCoverage {
                action_id: action.action_id.clone(),
                effort_units: action.effort_units,
                separated_conflict_pairs,
            }
        })
        .collect()
}

#[derive(Debug)]
struct PlanSearch {
    jointly_separable_pair_count: usize,
    jointly_undiscriminable_pairs: Vec<WorldPair>,
    unavoidable_shared_pairs: Vec<WorldPair>,
    distinct_constraints: usize,
    minimal_constraints: usize,
    candidate_subsets: u64,
    mark_steps: u64,
    min_cost: Option<u64>,
    min_count: u64,
    plans: Vec<JointPlan>,
    search_complete: bool,
    status: &'static str,
}

fn selected_action_ids(mask: usize, actions: &[Action]) -> Vec<String> {
    actions
        .iter()
        .enumerate()
        .filter(|(index, _)| mask & (1_usize << *index) != 0)
        .map(|(_, action)| action.action_id.clone())
        .collect()
}

fn exact_joint_plan(
    actions: &[Action],
    pairs: &[WorldPair],
    supports: &[NormalizedSupport],
    budget: u64,
) -> Result<PlanSearch, String> {
    let action_count = actions.len();
    let subset_space = 1_usize << action_count;
    let all_actions_mask = subset_space - 1;
    let candidate_subsets = all_actions_mask as u64;

    if pairs.is_empty() {
        return Ok(PlanSearch {
            jointly_separable_pair_count: 0,
            jointly_undiscriminable_pairs: Vec::new(),
            unavoidable_shared_pairs: Vec::new(),
            distinct_constraints: 0,
            minimal_constraints: 0,
            candidate_subsets,
            mark_steps: 0,
            min_cost: Some(0),
            min_count: 1,
            plans: vec![JointPlan {
                action_ids: Vec::new(),
                effort_units: 0,
            }],
            search_complete: true,
            status: "NO_DECISION_DISAGREEMENT_WITHIN_DECLARED_SCENARIOS",
        });
    }

    // If any cross-world row pair is identical over every action, no projection
    // can separate those worlds. Otherwise every cross-row pair contributes a
    // hitting-set constraint: the selected action mask must intersect its
    // differing-coordinate mask.
    let mut constraints = BTreeSet::new();
    let mut unavoidable_shared_pairs = Vec::new();
    let mut jointly_undiscriminable_pairs = Vec::new();
    for pair in pairs {
        let left = find_support(supports, &pair.world_a);
        let right = find_support(supports, &pair.world_b);
        let mut shared_full_tuple = false;
        let mut pair_constraints = BTreeSet::new();
        for left_row in &left.rows {
            for right_row in &right.rows {
                let mut difference_mask = 0_usize;
                for action_index in 0..action_count {
                    if left_row[action_index] != right_row[action_index] {
                        difference_mask |= 1_usize << action_index;
                    }
                }
                if difference_mask == 0 {
                    shared_full_tuple = true;
                } else {
                    pair_constraints.insert(difference_mask);
                }
            }
        }
        if shared_full_tuple {
            unavoidable_shared_pairs.push(pair.clone());
            jointly_undiscriminable_pairs.push(pair.clone());
        } else {
            constraints.extend(pair_constraints);
        }
    }

    if !unavoidable_shared_pairs.is_empty() {
        return Ok(PlanSearch {
            jointly_separable_pair_count: pairs.len() - jointly_undiscriminable_pairs.len(),
            jointly_undiscriminable_pairs,
            unavoidable_shared_pairs,
            distinct_constraints: constraints.len(),
            minimal_constraints: 0,
            candidate_subsets,
            mark_steps: 0,
            min_cost: None,
            min_count: 0,
            plans: Vec::new(),
            search_complete: true,
            status: "NO_JOINT_SUPPORT_PLAN_WITHIN_DECLARED_RELATIONS",
        });
    }

    // Exact reduction of the hitting-set constraints. A superset constraint is
    // redundant if one of its proper subsets is already present.
    let mut constraint_subsets = vec![0_u32; subset_space];
    for constraint in &constraints {
        constraint_subsets[*constraint] = 1;
    }
    for bit in 0..action_count {
        let bit_mask = 1_usize << bit;
        for mask in 0..subset_space {
            if mask & bit_mask != 0 {
                constraint_subsets[mask] += constraint_subsets[mask ^ bit_mask];
            }
        }
    }
    let minimal_constraints = constraints
        .iter()
        .copied()
        .filter(|constraint| constraint_subsets[*constraint] == 1)
        .collect::<Vec<_>>();

    // Mark every action subset that fails at least one hitting-set constraint.
    // For constraint C, those invalid selections are precisely all submasks of
    // the complement of C. A hard marking budget fails closed rather than
    // returning an approximate result as an exact minimum.
    let mut invalid = vec![false; subset_space];
    let mut mark_steps = 0_u64;
    for constraint in &minimal_constraints {
        let complement = all_actions_mask ^ constraint;
        let mut subset = complement;
        loop {
            if mark_steps >= MAX_SUBSET_MARK_STEPS {
                return Ok(PlanSearch {
                    jointly_separable_pair_count: pairs.len(),
                    jointly_undiscriminable_pairs: Vec::new(),
                    unavoidable_shared_pairs,
                    distinct_constraints: constraints.len(),
                    minimal_constraints: minimal_constraints.len(),
                    candidate_subsets,
                    mark_steps,
                    min_cost: None,
                    min_count: 0,
                    plans: Vec::new(),
                    search_complete: false,
                    status: "JOINT_SUPPORT_SEARCH_LIMIT_REACHED",
                });
            }
            invalid[subset] = true;
            mark_steps += 1;
            if subset == 0 {
                break;
            }
            subset = (subset - 1) & complement;
        }
    }

    let mut costs = vec![0_u64; subset_space];
    let mut min_cost: Option<u64> = None;
    let mut min_count = 0_u64;
    let mut plans = Vec::new();
    for mask in 1..subset_space {
        let bit = mask.trailing_zeros() as usize;
        let previous = mask & (mask - 1);
        costs[mask] = costs[previous]
            .checked_add(actions[bit].effort_units)
            .ok_or_else(|| "action subset effort overflows u64".to_string())?;
        if invalid[mask] {
            continue;
        }
        let cost = costs[mask];
        if min_cost.is_none_or(|best| cost < best) {
            min_cost = Some(cost);
            min_count = 0;
            plans.clear();
        }
        if min_cost == Some(cost) {
            min_count += 1;
            if plans.len() < MAX_RETURNED_PLANS {
                plans.push(JointPlan {
                    action_ids: selected_action_ids(mask, actions),
                    effort_units: cost,
                });
            }
        }
    }
    plans.sort_by(|left, right| left.action_ids.cmp(&right.action_ids));
    let jointly_undiscriminable_pairs = if min_cost.is_none() {
        pairs.to_vec()
    } else {
        Vec::new()
    };
    let jointly_separable_pair_count = pairs.len() - jointly_undiscriminable_pairs.len();
    let status = match min_cost {
        None => "NO_JOINT_SUPPORT_PLAN_WITHIN_DECLARED_RELATIONS",
        Some(cost) if cost <= budget => "MINIMUM_COST_JOINT_ROBUST_PLAN",
        Some(_) => "MINIMUM_COST_JOINT_ROBUST_PLAN_EXCEEDS_BUDGET",
    };

    Ok(PlanSearch {
        jointly_separable_pair_count,
        jointly_undiscriminable_pairs,
        unavoidable_shared_pairs,
        distinct_constraints: constraints.len(),
        minimal_constraints: minimal_constraints.len(),
        candidate_subsets,
        mark_steps,
        min_cost,
        min_count,
        plans,
        search_complete: true,
        status,
    })
}

fn evaluate(input: &Input, input_bytes: &[u8]) -> Result<Report, String> {
    let validated = validate(input)?;
    let pairs = decision_conflict_pairs(&validated.scenarios);
    let marginal = marginal_coverage(&validated.actions, &pairs, &validated.supports);
    let marginal_separable = pairs
        .iter()
        .filter(|pair| {
            marginal
                .iter()
                .any(|row| row.separated_conflict_pairs.contains(pair))
        })
        .count();
    let marginally_undiscriminable_pairs = pairs
        .iter()
        .filter(|pair| {
            !marginal
                .iter()
                .any(|row| row.separated_conflict_pairs.contains(pair))
        })
        .cloned()
        .collect::<Vec<_>>();

    let plan = exact_joint_plan(
        &validated.actions,
        &pairs,
        &validated.supports,
        input.max_total_effort_units,
    )?;
    let minimum_joint_plan_within_budget = plan
        .min_cost
        .map(|cost| cost <= input.max_total_effort_units);

    Ok(Report {
        schema_version: 1,
        tool: "connectome-joint-support-audit",
        analysis_id: input.analysis_id.clone(),
        dataset_revision_commit_sha: input.dataset_revision_commit_sha.clone(),
        query_id: input.query_id.clone(),
        query_spec_sha256: input.query_spec_sha256.clone(),
        scenario_set_sha256: input.scenario_set_sha256.clone(),
        action_set_sha256: input.action_set_sha256.clone(),
        joint_support_set_sha256: input.joint_support_set_sha256.clone(),
        input_sha256: sha256(input_bytes),
        scenario_count: validated.scenarios.len(),
        action_count: validated.actions.len(),
        total_joint_support_rows: validated.total_rows,
        joint_support_scope: input.joint_support_scope.clone(),
        effort_unit: input.effort_unit.clone(),
        max_total_effort_units: input.max_total_effort_units,
        decision_conflict_pair_count: pairs.len(),
        marginal_action_coverage: marginal,
        marginally_separable_pair_count: marginal_separable,
        marginally_undiscriminable_pairs,
        jointly_separable_pair_count: plan.jointly_separable_pair_count,
        jointly_undiscriminable_pairs: plan.jointly_undiscriminable_pairs,
        unavoidable_shared_full_joint_tuple_pairs: plan.unavoidable_shared_pairs,
        distinct_difference_constraint_count: plan.distinct_constraints,
        minimal_difference_constraint_count: plan.minimal_constraints,
        candidate_action_subsets: plan.candidate_subsets,
        subset_mark_steps: plan.mark_steps,
        minimum_joint_disambiguation_effort_units: plan.min_cost,
        minimum_joint_plan_count: plan.min_count,
        minimum_joint_plans_returned: plan.plans.len(),
        minimum_joint_plans_truncated: plan.min_count > MAX_RETURNED_PLANS as u64,
        minimum_joint_plans: plan.plans,
        minimum_joint_plan_within_budget,
        search_complete: plan.search_complete,
        status: plan.status,
        probability_claim: "NONE_JOINT_SET_VALUED_DISCRIMINATION_ONLY",
        scope_note: "The planner evaluates the declared finite joint-support relation directly. It does not factorize marginal supports, infer independence, infer likelihoods, or prove support completeness. Results are conditional on the supplied scenarios, action set, joint support rows, provenance digests and scope declaration. A digest binds a declaration; it does not prove biological truth or real-world completeness.",
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let input_bytes = match fs::read(&args.input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("connectome-joint-support-audit: could not read input: {error}");
            return ExitCode::from(2);
        }
    };
    let input: Input = match serde_json::from_slice(&input_bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("connectome-joint-support-audit: invalid JSON input: {error}");
            return ExitCode::from(2);
        }
    };
    let report = match evaluate(&input, &input_bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("connectome-joint-support-audit: {error}");
            return ExitCode::from(2);
        }
    };
    let rendered = match serde_json::to_string_pretty(&report) {
        Ok(value) => value + "\n",
        Err(error) => {
            eprintln!("connectome-joint-support-audit: could not serialize report: {error}");
            return ExitCode::from(2);
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(path, rendered.as_bytes()) {
            eprintln!("connectome-joint-support-audit: could not write report: {error}");
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

    fn fixture() -> Input {
        serde_json::from_str(include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_JOINT_SUPPORT_XOR.example.json"
        ))
        .expect("checked-in joint-support fixture must parse")
    }

    fn fixture_bytes() -> Vec<u8> {
        include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_JOINT_SUPPORT_XOR.example.json"
        )
        .as_bytes()
        .to_vec()
    }

    fn validated(input: &Input) -> Validated {
        validate(input).expect("fixture and mutation must remain valid")
    }

    fn support_for<'a>(validated: &'a Validated, world: &str) -> &'a NormalizedSupport {
        find_support(&validated.supports, world)
    }

    fn two_action_support_case(alpha_mask: u8, beta_mask: u8) -> Input {
        let mut input = fixture();
        input.actions.truncate(2);
        input.action_set_sha256 = canonical_action_set_sha256(&input.actions);
        for support in &mut input.joint_supports {
            let mask = if support.scenario_id == "world-alpha" {
                alpha_mask
            } else {
                beta_mask
            };
            support.rows = (0..4)
                .filter(|tuple| mask & (1_u8 << *tuple) != 0)
                .map(|tuple| JointRow {
                    outcomes: vec![
                        JointOutcome {
                            action_id: "inspect-x".into(),
                            value: if tuple & 1 == 0 { "0" } else { "1" }.into(),
                        },
                        JointOutcome {
                            action_id: "inspect-y".into(),
                            value: if tuple & 2 == 0 { "0" } else { "1" }.into(),
                        },
                    ],
                })
                .collect();
        }
        input.joint_support_set_sha256 =
            canonical_joint_support_set_sha256(&input.joint_supports);
        input
    }

    fn direct_projection_separates(
        left: &NormalizedSupport,
        right: &NormalizedSupport,
        mask: usize,
    ) -> bool {
        let project = |support: &NormalizedSupport| {
            support
                .rows
                .iter()
                .map(|row| {
                    (0..row.len())
                        .filter(|index| mask & (1_usize << index) != 0)
                        .map(|index| row[index].clone())
                        .collect::<Vec<_>>()
                })
                .collect::<BTreeSet<_>>()
        };
        let left_projection = project(left);
        let right_projection = project(right);
        left_projection.is_disjoint(&right_projection)
    }

    #[test]
    fn xor_complementarity_is_recovered_without_marginal_independence() {
        let input = fixture();
        let report = evaluate(&input, &fixture_bytes()).unwrap();
        assert_eq!(report.decision_conflict_pair_count, 1);
        assert_eq!(report.marginally_separable_pair_count, 0);
        assert_eq!(report.marginally_undiscriminable_pairs.len(), 1);
        assert_eq!(report.minimum_joint_disambiguation_effort_units, Some(5));
        assert_eq!(report.minimum_joint_plan_count, 1);
        assert_eq!(
            report.minimum_joint_plans[0].action_ids,
            vec!["inspect-x".to_string(), "inspect-y".to_string()]
        );
        assert_eq!(report.status, "MINIMUM_COST_JOINT_ROBUST_PLAN");
        assert!(report.search_complete);
    }

    #[test]
    fn x_and_y_marginals_overlap_but_their_joint_projection_is_disjoint() {
        let input = fixture();
        let validated = validated(&input);
        let left = support_for(&validated, "world-alpha");
        let right = support_for(&validated, "world-beta");
        assert!(!pair_separated_for_mask(left, right, 0b001));
        assert!(!pair_separated_for_mask(left, right, 0b010));
        assert!(pair_separated_for_mask(left, right, 0b011));
        assert!(!pair_separated_for_mask(left, right, 0b100));
        assert!(pair_separated_for_mask(left, right, 0b101));
        assert!(!pair_separated_for_mask(left, right, 0b110));
    }

    #[test]
    fn a_shared_full_joint_tuple_prevents_every_possible_projection() {
        let mut input = fixture();
        input.joint_supports[1].rows.push(JointRow {
            outcomes: vec![
                JointOutcome {
                    action_id: "inspect-x".into(),
                    value: "0".into(),
                },
                JointOutcome {
                    action_id: "inspect-y".into(),
                    value: "0".into(),
                },
                JointOutcome {
                    action_id: "orthogonal-z".into(),
                    value: "0".into(),
                },
            ],
        });
        input.joint_support_set_sha256 = canonical_joint_support_set_sha256(&input.joint_supports);
        let bytes = serde_json::to_vec(&input).unwrap();
        let report = evaluate(&input, &bytes).unwrap();
        assert_eq!(
            report.status,
            "NO_JOINT_SUPPORT_PLAN_WITHIN_DECLARED_RELATIONS"
        );
        assert_eq!(report.minimum_joint_disambiguation_effort_units, None);
        assert_eq!(
            report.unavoidable_shared_full_joint_tuple_pairs,
            vec![WorldPair {
                world_a: "world-alpha".into(),
                world_b: "world-beta".into(),
            }]
        );
    }

    #[test]
    fn add_only_joint_support_expansion_increases_minimum_effort() {
        let mut input = fixture();
        input.joint_supports[1].rows.push(JointRow {
            outcomes: vec![
                JointOutcome {
                    action_id: "inspect-x".into(),
                    value: "0".into(),
                },
                JointOutcome {
                    action_id: "inspect-y".into(),
                    value: "0".into(),
                },
                JointOutcome {
                    action_id: "orthogonal-z".into(),
                    value: "1".into(),
                },
            ],
        });
        input.joint_support_set_sha256 = canonical_joint_support_set_sha256(&input.joint_supports);
        let bytes = serde_json::to_vec(&input).unwrap();
        let report = evaluate(&input, &bytes).unwrap();
        assert_eq!(report.minimum_joint_disambiguation_effort_units, Some(7));
        assert_eq!(
            report.status,
            "MINIMUM_COST_JOINT_ROBUST_PLAN_EXCEEDS_BUDGET"
        );
        assert_eq!(
            report.minimum_joint_plans[0].action_ids,
            vec!["inspect-x".to_string(), "orthogonal-z".to_string()]
        );
    }

    #[test]
    fn support_expansion_never_creates_a_new_robust_projection() {
        let mut input = fixture();
        let baseline = validated(&input);
        let pair = WorldPair {
            world_a: "world-alpha".into(),
            world_b: "world-beta".into(),
        };
        input.joint_supports[1].rows.push(JointRow {
            outcomes: vec![
                JointOutcome {
                    action_id: "inspect-x".into(),
                    value: "0".into(),
                },
                JointOutcome {
                    action_id: "inspect-y".into(),
                    value: "0".into(),
                },
                JointOutcome {
                    action_id: "orthogonal-z".into(),
                    value: "B".into(),
                },
            ],
        });
        input.joint_support_set_sha256 = canonical_joint_support_set_sha256(&input.joint_supports);
        let expanded = validated(&input);
        let before_left = support_for(&baseline, &pair.world_a);
        let before_right = support_for(&baseline, &pair.world_b);
        let after_left = support_for(&expanded, &pair.world_a);
        let after_right = support_for(&expanded, &pair.world_b);
        for mask in 0..(1_usize << baseline.actions.len()) {
            let before = pair_separated_for_mask(before_left, before_right, mask);
            let after = pair_separated_for_mask(after_left, after_right, mask);
            assert!(
                !after || before,
                "support expansion gained separation for mask {mask:#b}"
            );
        }
    }

    #[test]
    fn full_row_requires_each_declared_action_once() {
        let mut input = fixture();
        input.joint_supports[0].rows[0].outcomes.pop();
        input.joint_support_set_sha256 = canonical_joint_support_set_sha256(&input.joint_supports);
        assert!(
            validate(&input)
                .unwrap_err()
                .contains("exactly one outcome for every declared action")
        );
    }

    #[test]
    fn action_and_row_order_do_not_change_canonical_digests() {
        let mut input = fixture();
        let action_digest = canonical_action_set_sha256(&input.actions);
        let support_digest = canonical_joint_support_set_sha256(&input.joint_supports);
        input.actions.reverse();
        for support in &mut input.joint_supports {
            support.rows.reverse();
            support.dependence_groups.reverse();
            for row in &mut support.rows {
                row.outcomes.reverse();
            }
        }
        assert_eq!(action_digest, canonical_action_set_sha256(&input.actions));
        assert_eq!(
            support_digest,
            canonical_joint_support_set_sha256(&input.joint_supports)
        );
    }

    #[test]
    fn exhaustive_two_action_two_world_supports_match_direct_projection_oracle() {
        let mut case_count = 0_usize;
        for alpha_mask in 1_u8..16 {
            for beta_mask in 1_u8..16 {
                let input = two_action_support_case(alpha_mask, beta_mask);
                let bytes = serde_json::to_vec(&input).unwrap();
                let report = evaluate(&input, &bytes).unwrap();
                let validated = validated(&input);
                let left = support_for(&validated, "world-alpha");
                let right = support_for(&validated, "world-beta");

                let valid_masks = (1_usize..4)
                    .filter(|mask| direct_projection_separates(left, right, *mask))
                    .collect::<Vec<_>>();
                let expected_minimum = valid_masks
                    .iter()
                    .map(|mask| {
                        validated
                            .actions
                            .iter()
                            .enumerate()
                            .filter(|(index, _)| *mask & (1_usize << *index) != 0)
                            .map(|(_, action)| action.effort_units)
                            .sum::<u64>()
                    })
                    .min();
                let expected_count = expected_minimum.map_or(0, |minimum| {
                    valid_masks
                        .iter()
                        .filter(|mask| {
                            validated
                                .actions
                                .iter()
                                .enumerate()
                                .filter(|(index, _)| **mask & (1_usize << *index) != 0)
                                .map(|(_, action)| action.effort_units)
                                .sum::<u64>()
                                == minimum
                        })
                        .count() as u64
                });
                let expected_marginally_separable =
                    direct_projection_separates(left, right, 0b01)
                        || direct_projection_separates(left, right, 0b10);
                let full_projection_separates =
                    direct_projection_separates(left, right, 0b11);

                assert_eq!(
                    report.minimum_joint_disambiguation_effort_units,
                    expected_minimum,
                    "wrong minimum for alpha={alpha_mask:04b}, beta={beta_mask:04b}"
                );
                assert_eq!(
                    report.minimum_joint_plan_count, expected_count,
                    "wrong tied-plan count for alpha={alpha_mask:04b}, beta={beta_mask:04b}"
                );
                assert_eq!(
                    report.marginally_separable_pair_count,
                    usize::from(expected_marginally_separable)
                );
                assert_eq!(
                    report.jointly_separable_pair_count,
                    usize::from(full_projection_separates)
                );
                assert_eq!(
                    report.unavoidable_shared_full_joint_tuple_pairs.is_empty(),
                    full_projection_separates
                );
                assert!(report.search_complete);

                let expected_ids = valid_masks
                    .iter()
                    .filter(|mask| {
                        expected_minimum.is_some_and(|minimum| {
                            validated
                                .actions
                                .iter()
                                .enumerate()
                                .filter(|(index, _)| **mask & (1_usize << *index) != 0)
                                .map(|(_, action)| action.effort_units)
                                .sum::<u64>()
                                == minimum
                        })
                    })
                    .map(|mask| selected_action_ids(*mask, &validated.actions))
                    .collect::<BTreeSet<_>>();
                let actual_ids = report
                    .minimum_joint_plans
                    .iter()
                    .map(|plan| plan.action_ids.clone())
                    .collect::<BTreeSet<_>>();
                assert_eq!(
                    actual_ids, expected_ids,
                    "wrong tied-plan membership for alpha={alpha_mask:04b}, beta={beta_mask:04b}"
                );
                case_count += 1;
            }
        }
        assert_eq!(case_count, 225);
    }

    #[test]
    fn exhaustive_small_support_expansions_never_create_projection_separation() {
        for alpha_mask in 1_u8..16 {
            for beta_mask in 1_u8..16 {
                let baseline_input = two_action_support_case(alpha_mask, beta_mask);
                let baseline_bytes = serde_json::to_vec(&baseline_input).unwrap();
                let baseline_report = evaluate(&baseline_input, &baseline_bytes).unwrap();
                let baseline = validated(&baseline_input);

                for (world_index, source_mask) in
                    [(0_usize, alpha_mask), (1_usize, beta_mask)]
                {
                    if source_mask == 0b1111 {
                        continue;
                    }
                    let missing_tuple = (0_u8..4)
                        .find(|tuple| source_mask & (1_u8 << *tuple) == 0)
                        .expect("a non-full support has a missing tuple");
                    let mut expanded_input = baseline_input.clone();
                    expanded_input.joint_supports[world_index].rows.push(JointRow {
                        outcomes: vec![
                            JointOutcome {
                                action_id: "inspect-x".into(),
                                value: if missing_tuple & 1 == 0 { "0" } else { "1" }.into(),
                            },
                            JointOutcome {
                                action_id: "inspect-y".into(),
                                value: if missing_tuple & 2 == 0 { "0" } else { "1" }.into(),
                            },
                        ],
                    });
                    expanded_input.joint_support_set_sha256 =
                        canonical_joint_support_set_sha256(&expanded_input.joint_supports);
                    let expanded_bytes = serde_json::to_vec(&expanded_input).unwrap();
                    let expanded_report = evaluate(&expanded_input, &expanded_bytes).unwrap();
                    let expanded = validated(&expanded_input);

                    let left_before = support_for(&baseline, "world-alpha");
                    let right_before = support_for(&baseline, "world-beta");
                    let left_after = support_for(&expanded, "world-alpha");
                    let right_after = support_for(&expanded, "world-beta");
                    for mask in 0..4 {
                        let before = direct_projection_separates(
                            left_before,
                            right_before,
                            mask,
                        );
                        let after = direct_projection_separates(
                            left_after,
                            right_after,
                            mask,
                        );
                        assert!(
                            !after || before,
                            "expansion created separation: alpha={alpha_mask:04b}, beta={beta_mask:04b}, added_world={world_index}, mask={mask:02b}"
                        );
                    }
                    if let (Some(before_cost), Some(after_cost)) = (
                        baseline_report.minimum_joint_disambiguation_effort_units,
                        expanded_report.minimum_joint_disambiguation_effort_units,
                    ) {
                        assert!(
                            after_cost >= before_cost,
                            "minimum cost decreased after support expansion: alpha={alpha_mask:04b}, beta={beta_mask:04b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn stale_joint_support_digest_fails_closed() {
        let mut input = fixture();
        input.joint_support_set_sha256 = "0".repeat(64);
        assert!(
            validate(&input)
                .unwrap_err()
                .contains("joint-support-set digest mismatch")
        );
    }
}
