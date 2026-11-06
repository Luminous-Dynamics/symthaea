//! Exact minimax adaptive intervention policies under coherent nuisance uncertainty.
//!
//! V1 uses one deterministic joint potential-outcome vector per biological world
//! and fixed nuisance state. It does not assume independence or assign probabilities.

use clap::Parser;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const MAX_WORLDS: usize = 12;
const MAX_NUISANCE_MODELS: usize = 8;
const MAX_ACTIONS: usize = 12;
const MAX_MEMO_STATES: usize = 2_000_000;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-adaptive-intervention-policy-audit",
    version,
    about = "Find exact worst-case outcome-contingent intervention policies"
)]
struct Args {
    /// Frozen biological worlds, nuisance models, actions and joint potential outcomes.
    #[arg(long)]
    input: PathBuf,
    /// Optional deterministic JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    analysis_id: String,
    claim_scope: String,
    dataset_revision_commit_sha: String,
    query: Query,
    worlds: Vec<World>,
    nuisance_models: Vec<NuisanceModel>,
    actions: Vec<Action>,
    potential_outcomes: Vec<PotentialOutcomeRow>,
    search_limits: SearchLimits,
    expected: Expected,
    #[serde(default)]
    notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    query_id: String,
    query_spec_sha256: String,
    decision_rule_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct World {
    world_id: String,
    graph_sha256: String,
    source_evidence_sha256: String,
    biological_disposition: String,
    query_decision: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct NuisanceModel {
    nuisance_model_id: String,
    model_sha256: String,
    description: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Action {
    action_id: String,
    intervention_kind: String,
    effort_units: u64,
    feasibility_status: String,
    feasibility_receipt_sha256: String,
    safety_profile_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PotentialOutcomeRow {
    world_id: String,
    nuisance_model_id: String,
    observations: Vec<Observation>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    action_id: String,
    outcome_id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchLimits {
    max_memo_states: usize,
    max_policy_depth: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    adaptive_status: String,
    adaptive_worst_case_effort_units: Option<u64>,
    adaptive_root_action_id: Option<String>,
    fixed_status: String,
    fixed_minimum_effort_units: Option<u64>,
    fixed_action_ids: Vec<String>,
    adaptivity_savings_units: Option<i64>,
}

#[derive(Debug, Clone)]
struct Prepared {
    input: Input,
    outcomes: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct PolicyState {
    latent_state_indices: Vec<usize>,
    used_action_mask: u16,
}

#[derive(Debug, Clone)]
struct Plan {
    worst_case_effort_units: u64,
    policy_tree: Value,
}

#[derive(Debug, Clone)]
struct FixedPlan {
    minimum_effort_units: u64,
    action_ids: Vec<String>,
}

#[derive(Debug)]
struct SearchContext {
    worlds: Vec<World>,
    nuisance_models: Vec<NuisanceModel>,
    actions: Vec<Action>,
    outcomes: Vec<Vec<String>>,
    max_memo_states: usize,
    max_policy_depth: usize,
    memo_state_count: usize,
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_token(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 128
        && bytes[0].is_ascii_lowercase().then_some(()).is_some()
            || (!bytes.is_empty() && bytes[0].is_ascii_digit())
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._:-".contains(byte))
}

fn validate_unique<I>(values: I, label: &str) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let mut seen = BTreeSet::new();
    for value in values {
        if !valid_token(&value) {
            return Err(format!("{label} contains a malformed identity: {value}"));
        }
        if !seen.insert(value.clone()) {
            return Err(format!("{label} contains a duplicate identity: {value}"));
        }
    }
    Ok(())
}

fn require_sha(value: &str, length: usize, label: &str) -> Result<(), String> {
    if valid_hex(value, length) {
        Ok(())
    } else {
        Err(format!(
            "{label} must be {length} lowercase hexadecimal characters"
        ))
    }
}

fn prepare(mut input: Input) -> Result<Prepared, String> {
    if input.schema_version != 1 {
        return Err("schema_version must equal 1".into());
    }
    if input.claim_scope != "synthetic_known_truth" {
        return Err("V1 accepts synthetic_known_truth fixtures only".into());
    }
    if !valid_token(&input.analysis_id) || !valid_token(&input.query.query_id) {
        return Err("analysis_id or query_id is malformed".into());
    }
    require_sha(
        &input.dataset_revision_commit_sha,
        40,
        "dataset revision commit SHA",
    )?;
    require_sha(
        &input.query.query_spec_sha256,
        64,
        "query specification digest",
    )?;
    require_sha(
        &input.query.decision_rule_sha256,
        64,
        "query decision-rule digest",
    )?;

    input.worlds.sort_by(|a, b| a.world_id.cmp(&b.world_id));
    input
        .nuisance_models
        .sort_by(|a, b| a.nuisance_model_id.cmp(&b.nuisance_model_id));
    input.actions.sort_by(|a, b| a.action_id.cmp(&b.action_id));

    if !(2..=MAX_WORLDS).contains(&input.worlds.len()) {
        return Err("world count is outside V1 bounds".into());
    }
    if !(1..=MAX_NUISANCE_MODELS).contains(&input.nuisance_models.len()) {
        return Err("nuisance-model count is outside V1 bounds".into());
    }
    if !(1..=MAX_ACTIONS).contains(&input.actions.len()) {
        return Err("action count is outside V1 bounds".into());
    }

    validate_unique(
        input.worlds.iter().map(|row| row.world_id.clone()),
        "world IDs",
    )?;
    validate_unique(
        input
            .nuisance_models
            .iter()
            .map(|row| row.nuisance_model_id.clone()),
        "nuisance-model IDs",
    )?;
    validate_unique(
        input.actions.iter().map(|row| row.action_id.clone()),
        "action IDs",
    )?;

    for world in &input.worlds {
        if world.biological_disposition != "ADMISSIBLE" {
            return Err(format!(
                "world {} is not biologically admissible",
                world.world_id
            ));
        }
        require_sha(&world.graph_sha256, 64, "world graph digest")?;
        require_sha(
            &world.source_evidence_sha256,
            64,
            "world source-evidence digest",
        )?;
    }

    for nuisance in &input.nuisance_models {
        require_sha(
            &nuisance.model_sha256,
            64,
            "nuisance-model digest",
        )?;
        if nuisance.description.trim().is_empty() || !nuisance.description.is_ascii() {
            return Err(format!(
                "nuisance model {} has an empty or non-ASCII description",
                nuisance.nuisance_model_id
            ));
        }
    }

    for action in &input.actions {
        if !matches!(
            action.intervention_kind.as_str(),
            "OPTOGENETIC"
                | "ELECTRICAL"
                | "PHARMACOLOGICAL"
                | "SENSORY"
                | "BEHAVIORAL"
                | "SIMULATED"
                | "OTHER"
        ) {
            return Err(format!(
                "action {} has an unsupported intervention kind",
                action.action_id
            ));
        }
        if action.effort_units == 0 {
            return Err(format!(
                "action {} has zero effort",
                action.action_id
            ));
        }
        if !matches!(
            action.feasibility_status.as_str(),
            "FEASIBLE" | "DISALLOWED" | "INACCESSIBLE" | "NOT_ASSESSED"
        ) {
            return Err(format!(
                "action {} has an unsupported feasibility status",
                action.action_id
            ));
        }
        require_sha(
            &action.feasibility_receipt_sha256,
            64,
            "action feasibility receipt",
        )?;
        require_sha(
            &action.safety_profile_sha256,
            64,
            "action safety-profile digest",
        )?;
    }

    if input.search_limits.max_memo_states == 0
        || input.search_limits.max_memo_states > MAX_MEMO_STATES
    {
        return Err("max_memo_states is outside V1 bounds".into());
    }
    if input.search_limits.max_policy_depth == 0
        || input.search_limits.max_policy_depth > input.actions.len()
    {
        return Err("max_policy_depth is outside V1 bounds".into());
    }

    let world_ids = input
        .worlds
        .iter()
        .map(|row| row.world_id.as_str())
        .collect::<BTreeSet<_>>();
    let nuisance_ids = input
        .nuisance_models
        .iter()
        .map(|row| row.nuisance_model_id.as_str())
        .collect::<BTreeSet<_>>();
    let action_ids = input
        .actions
        .iter()
        .map(|row| row.action_id.as_str())
        .collect::<BTreeSet<_>>();

    let mut row_map: BTreeMap<(String, String), BTreeMap<String, String>> =
        BTreeMap::new();
    for row in &input.potential_outcomes {
        if !world_ids.contains(row.world_id.as_str()) {
            return Err(format!(
                "potential-outcome row references unknown world {}",
                row.world_id
            ));
        }
        if !nuisance_ids.contains(row.nuisance_model_id.as_str()) {
            return Err(format!(
                "potential-outcome row references unknown nuisance model {}",
                row.nuisance_model_id
            ));
        }
        let key = (row.world_id.clone(), row.nuisance_model_id.clone());
        if row_map.contains_key(&key) {
            return Err(format!(
                "duplicate world/nuisance potential-outcome row for {}/{}",
                key.0, key.1
            ));
        }
        let mut outcomes = BTreeMap::new();
        for observation in &row.observations {
            if !action_ids.contains(observation.action_id.as_str()) {
                return Err(format!(
                    "potential outcome references unknown action {}",
                    observation.action_id
                ));
            }
            if !valid_token(&observation.outcome_id) {
                return Err(format!(
                    "potential outcome has malformed outcome ID {}",
                    observation.outcome_id
                ));
            }
            if outcomes
                .insert(
                    observation.action_id.clone(),
                    observation.outcome_id.clone(),
                )
                .is_some()
            {
                return Err(format!(
                    "duplicate action outcome for latent state {}/{}",
                    key.0, key.1
                ));
            }
        }
        if outcomes.len() != input.actions.len()
            || outcomes.keys().any(|action_id| !action_ids.contains(action_id.as_str()))
        {
            return Err(format!(
                "potential-outcome vector is incomplete for {}/{}",
                key.0, key.1
            ));
        }
        row_map.insert(key, outcomes);
    }

    let mut outcomes = Vec::with_capacity(input.worlds.len() * input.nuisance_models.len());
    for world in &input.worlds {
        for nuisance in &input.nuisance_models {
            let key = (world.world_id.clone(), nuisance.nuisance_model_id.clone());
            let row = row_map.get(&key).ok_or_else(|| {
                format!(
                    "potential-outcome matrix is missing {}/{}",
                    key.0, key.1
                )
            })?;
            outcomes.push(
                input
                    .actions
                    .iter()
                    .map(|action| row[&action.action_id].clone())
                    .collect(),
            );
        }
    }
    if row_map.len() != input.worlds.len() * input.nuisance_models.len() {
        return Err("potential-outcome matrix has extra latent states".into());
    }

    Ok(Prepared { input, outcomes })
}

fn solve_adaptive(
    latent_state_indices: &[usize],
    used_action_mask: u16,
    context: &mut SearchContext,
    memo: &mut HashMap<PolicyState, Option<Plan>>,
) -> Result<Option<Plan>, String> {
    let key = PolicyState {
        latent_state_indices: latent_state_indices.to_vec(),
        used_action_mask,
    };
    if let Some(cached) = memo.get(&key) {
        return Ok(cached.clone());
    }

    context.memo_state_count += 1;
    if context.memo_state_count > context.max_memo_states {
        return Err("INCOMPLETE_SEARCH_BUDGET_EXCEEDED".into());
    }

    let nuisance_count = context.nuisance_models.len();
    let mut decisions = BTreeSet::new();
    let mut world_ids = BTreeSet::new();
    for state_index in latent_state_indices {
        let world_index = state_index / nuisance_count;
        let world = &context.worlds[world_index];
        decisions.insert(world.query_decision);
        world_ids.insert(world.world_id.clone());
    }

    if decisions.len() == 1 {
        let decision = *decisions.iter().next().expect("non-empty latent state");
        let plan = Plan {
            worst_case_effort_units: 0,
            policy_tree: json!({
                "node_kind": "STOP",
                "query_decision": decision,
                "remaining_world_ids": world_ids.into_iter().collect::<Vec<_>>(),
                "remaining_latent_state_count": latent_state_indices.len(),
            }),
        };
        memo.insert(key, Some(plan.clone()));
        return Ok(Some(plan));
    }

    if used_action_mask.count_ones() as usize >= context.max_policy_depth {
        memo.insert(key, None);
        return Ok(None);
    }

    let mut best: Option<(u64, String, Plan)> = None;
    for action_index in 0..context.actions.len() {
        let bit = 1_u16 << action_index;
        let action = &context.actions[action_index];
        if used_action_mask & bit != 0 || action.feasibility_status != "FEASIBLE" {
            continue;
        }

        let mut branches: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for state_index in latent_state_indices {
            let outcome_id = context.outcomes[*state_index][action_index].clone();
            branches
                .entry(outcome_id)
                .or_default()
                .push(*state_index);
        }

        let next_mask = used_action_mask | bit;
        let mut child_rows = Vec::with_capacity(branches.len());
        let mut child_costs = Vec::with_capacity(branches.len());
        let mut valid_action = true;
        for (outcome_id, child_states) in branches {
            let child_plan = solve_adaptive(&child_states, next_mask, context, memo)?;
            let Some(child_plan) = child_plan else {
                valid_action = false;
                break;
            };
            child_costs.push(child_plan.worst_case_effort_units);
            child_rows.push(json!({
                "outcome_id": outcome_id,
                "child": child_plan.policy_tree,
            }));
        }
        if !valid_action {
            continue;
        }

        let child_max = child_costs.into_iter().max().unwrap_or(0);
        let worst_case = action
            .effort_units
            .checked_add(child_max)
            .ok_or_else(|| "adaptive policy effort overflow".to_string())?;
        let tree = json!({
            "node_kind": "INTERVENE",
            "action_id": action.action_id,
            "effort_units": action.effort_units,
            "worst_case_subtree_effort_units": worst_case,
            "branches": child_rows,
        });
        let candidate = Plan {
            worst_case_effort_units: worst_case,
            policy_tree: tree,
        };

        let is_better = match &best {
            None => true,
            Some((best_cost, best_action_id, _)) => {
                (worst_case, action.action_id.as_str())
                    < (*best_cost, best_action_id.as_str())
            }
        };
        if is_better {
            best = Some((worst_case, action.action_id.clone(), candidate));
        }
    }

    let result = best.map(|(_, _, plan)| plan);
    memo.insert(key, result.clone());
    Ok(result)
}

fn solve_fixed(
    worlds: &[World],
    nuisance_models: &[NuisanceModel],
    actions: &[Action],
    outcomes: &[Vec<String>],
) -> Result<Option<FixedPlan>, String> {
    let feasible_indices = actions
        .iter()
        .enumerate()
        .filter_map(|(index, action)| {
            (action.feasibility_status == "FEASIBLE").then_some(index)
        })
        .collect::<Vec<_>>();
    if feasible_indices.len() >= usize::BITS as usize {
        return Err("too many feasible actions for exact fixed subset enumeration".into());
    }

    let mut best: Option<(u64, usize, Vec<String>)> = None;
    let subset_count = 1_usize << feasible_indices.len();
    for subset_mask in 0..subset_count {
        let selected_indices = feasible_indices
            .iter()
            .enumerate()
            .filter_map(|(position, action_index)| {
                (subset_mask & (1_usize << position) != 0).then_some(*action_index)
            })
            .collect::<Vec<_>>();
        let mut effort = 0_u64;
        let mut selected_ids = Vec::with_capacity(selected_indices.len());
        for action_index in &selected_indices {
            effort = effort
                .checked_add(actions[*action_index].effort_units)
                .ok_or_else(|| "fixed-plan effort overflow".to_string())?;
            selected_ids.push(actions[*action_index].action_id.clone());
        }

        let mut decision_by_tuple: BTreeMap<Vec<String>, bool> = BTreeMap::new();
        let mut resolving = true;
        for (world_index, _world) in worlds.iter().enumerate() {
            for nuisance_index in 0..nuisance_models.len() {
                let state_index = world_index * nuisance_models.len() + nuisance_index;
                let joint_tuple = selected_indices
                    .iter()
                    .map(|action_index| outcomes[state_index][*action_index].clone())
                    .collect::<Vec<_>>();
                let decision = worlds[world_index].query_decision;
                if let Some(previous) = decision_by_tuple.insert(joint_tuple, decision)
                    && previous != decision
                {
                    resolving = false;
                    break;
                }
            }
            if !resolving {
                break;
            }
        }
        if !resolving {
            continue;
        }

        selected_ids.sort();
        let candidate = (effort, selected_ids.len(), selected_ids);
        if best.as_ref().is_none_or(|current| candidate < *current) {
            best = Some(candidate);
        }
    }

    Ok(best.map(|(minimum_effort_units, _count, action_ids)| FixedPlan {
        minimum_effort_units,
        action_ids,
    }))
}

fn sha256_json(value: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("canonical JSON serialization failed: {error}"))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn canonical_digests(prepared: &Prepared) -> Result<(String, String, String, String), String> {
    let world_rows = prepared
        .input
        .worlds
        .iter()
        .map(|world| {
            json!({
                "world_id": world.world_id,
                "graph_sha256": world.graph_sha256,
                "source_evidence_sha256": world.source_evidence_sha256,
                "biological_disposition": world.biological_disposition,
                "query_decision": world.query_decision,
            })
        })
        .collect::<Vec<_>>();
    let nuisance_rows = prepared
        .input
        .nuisance_models
        .iter()
        .map(|nuisance| {
            json!({
                "nuisance_model_id": nuisance.nuisance_model_id,
                "model_sha256": nuisance.model_sha256,
            })
        })
        .collect::<Vec<_>>();
    let action_rows = prepared
        .input
        .actions
        .iter()
        .map(|action| {
            json!({
                "action_id": action.action_id,
                "intervention_kind": action.intervention_kind,
                "effort_units": action.effort_units,
                "feasibility_status": action.feasibility_status,
                "feasibility_receipt_sha256": action.feasibility_receipt_sha256,
                "safety_profile_sha256": action.safety_profile_sha256,
            })
        })
        .collect::<Vec<_>>();
    let potential_rows = prepared
        .input
        .worlds
        .iter()
        .enumerate()
        .flat_map(|(world_index, world)| {
            prepared
                .input
                .nuisance_models
                .iter()
                .enumerate()
                .map(move |(nuisance_index, nuisance)| {
                    let state_index =
                        world_index * prepared.input.nuisance_models.len() + nuisance_index;
                    let observations = prepared.input.actions.iter().enumerate().map(
                        |(action_index, action)| {
                            json!({
                                "action_id": action.action_id,
                                "outcome_id": prepared.outcomes[state_index][action_index],
                            })
                        },
                    ).collect::<Vec<_>>();
                    json!({
                        "world_id": world.world_id,
                        "nuisance_model_id": nuisance.nuisance_model_id,
                        "observations": observations,
                    })
                })
        })
        .collect::<Vec<_>>();

    Ok((
        sha256_json(&json!(world_rows))?,
        sha256_json(&json!(nuisance_rows))?,
        sha256_json(&json!(action_rows))?,
        sha256_json(&json!(potential_rows))?,
    ))
}

fn report_for_input(raw: &[u8], input: Input) -> Result<Value, String> {
    let prepared = prepare(input)?;
    let mut context = SearchContext {
        worlds: prepared.input.worlds.clone(),
        nuisance_models: prepared.input.nuisance_models.clone(),
        actions: prepared.input.actions.clone(),
        outcomes: prepared.outcomes.clone(),
        max_memo_states: prepared.input.search_limits.max_memo_states,
        max_policy_depth: prepared.input.search_limits.max_policy_depth,
        memo_state_count: 0,
    };
    let initial_states = (0..prepared.input.worlds.len() * prepared.input.nuisance_models.len())
        .collect::<Vec<_>>();
    let mut memo = HashMap::new();
    let adaptive = match solve_adaptive(&initial_states, 0, &mut context, &mut memo) {
        Ok(result) => result,
        Err(error) if error == "INCOMPLETE_SEARCH_BUDGET_EXCEEDED" => {
            return Ok(json!({
                "schema_version": 1,
                "analysis_id": prepared.input.analysis_id,
                "input_sha256": hex::encode(Sha256::digest(raw)),
                "status": "INCOMPLETE_SEARCH_BUDGET_EXCEEDED",
                "search_complete": false,
                "claim_scope": "synthetic_known_truth",
            }));
        }
        Err(error) => return Err(error),
    };
    let fixed = solve_fixed(
        &prepared.input.worlds,
        &prepared.input.nuisance_models,
        &prepared.input.actions,
        &prepared.outcomes,
    )?;

    let adaptive_cost = adaptive
        .as_ref()
        .map(|plan| plan.worst_case_effort_units);
    let fixed_cost = fixed
        .as_ref()
        .map(|plan| plan.minimum_effort_units);
    let adaptivity_savings = match (adaptive_cost, fixed_cost) {
        (Some(adaptive), Some(fixed)) => Some(
            i64::try_from(fixed)
                .map_err(|_| "fixed effort exceeds signed report range".to_string())?
                - i64::try_from(adaptive)
                    .map_err(|_| "adaptive effort exceeds signed report range".to_string())?,
        ),
        _ => None,
    };
    if adaptivity_savings.is_some_and(|savings| savings < 0) {
        return Err("adaptive minimax effort exceeds the fixed-plan optimum".into());
    }
    if adaptive.is_some() != fixed.is_some() {
        return Err(
            "adaptive and fixed resolving-plan existence differ for a complete deterministic potential-outcome table"
                .into(),
        );
    }

    let (world_digest, nuisance_digest, action_digest, potential_digest) =
        canonical_digests(&prepared)?;
    let adaptive_status = if adaptive.is_some() {
        "EXACT_POLICY"
    } else {
        "NO_GUARANTEED_QUERY_RESOLUTION"
    };
    let fixed_status = if fixed.is_some() {
        "EXACT_FIXED_PLAN"
    } else {
        "NO_GUARANTEED_FIXED_PLAN"
    };
    let adaptive_tree = adaptive
        .as_ref()
        .map(|plan| plan.policy_tree.clone());
    let adaptive_root = adaptive_tree
        .as_ref()
        .and_then(|tree| tree.get("action_id"))
        .and_then(Value::as_str);
    let fixed_action_ids = fixed
        .as_ref()
        .map(|plan| plan.action_ids.clone())
        .unwrap_or_default();
    let adaptivity_basis_points = match (adaptivity_savings, fixed_cost) {
        (Some(savings), Some(cost)) if cost > 0 => Some(
            (u64::try_from(savings)
                .map_err(|_| "negative adaptivity savings".to_string())?
                * 10_000)
                / cost,
        ),
        _ => None,
    };

    let report = json!({
        "schema_version": 1,
        "tool": "connectome-adaptive-intervention-policy-audit",
        "analysis_id": prepared.input.analysis_id,
        "input_sha256": hex::encode(Sha256::digest(raw)),
        "dataset_revision_commit_sha": prepared.input.dataset_revision_commit_sha,
        "query_id": prepared.input.query.query_id,
        "world_set_sha256": world_digest,
        "nuisance_set_sha256": nuisance_digest,
        "action_set_sha256": action_digest,
        "potential_outcome_table_sha256": potential_digest,
        "biological_world_count": prepared.input.worlds.len(),
        "nuisance_model_count": prepared.input.nuisance_models.len(),
        "latent_state_count": prepared.input.worlds.len() * prepared.input.nuisance_models.len(),
        "action_count": prepared.input.actions.len(),
        "feasible_action_count": prepared.input.actions.iter().filter(|action| action.feasibility_status == "FEASIBLE").count(),
        "adaptive_status": adaptive_status,
        "adaptive_worst_case_effort_units": adaptive_cost,
        "adaptive_root_action_id": adaptive_root,
        "adaptive_policy_tree": adaptive_tree,
        "adaptive_memo_state_count": context.memo_state_count,
        "fixed_status": fixed_status,
        "fixed_minimum_effort_units": fixed_cost,
        "fixed_action_ids": fixed_action_ids,
        "adaptivity_savings_units": adaptivity_savings,
        "adaptivity_savings_basis_points": adaptivity_basis_points,
        "adaptive_not_worse_than_fixed": adaptivity_savings.is_none_or(|savings| savings >= 0),
        "search_complete": true,
        "status": if adaptive.is_some() { "SYNTHETIC_EXACT_POLICY_VALIDATED" } else { "SYNTHETIC_NO_GUARANTEED_RESOLUTION" },
        "probability_claim": "NONE_DETERMINISTIC_POTENTIAL_OUTCOMES_ONLY",
        "biological_truth_disposition": "NOT_ESTABLISHED_BY_THIS_ORACLE",
        "hypothesis_space_completeness": "NOT_ESTABLISHED_BY_THIS_ORACLE",
    });
    validate_expected(&report, &prepared.input.expected)?;
    Ok(report)
}

fn validate_expected(report: &Value, expected: &Expected) -> Result<(), String> {
    let observed = (
        report["adaptive_status"].as_str().unwrap_or_default(),
        report["adaptive_worst_case_effort_units"].as_u64(),
        report["adaptive_root_action_id"].as_str(),
        report["fixed_status"].as_str().unwrap_or_default(),
        report["fixed_minimum_effort_units"].as_u64(),
        report["fixed_action_ids"]
            .as_array()
            .map(|rows| rows.iter().filter_map(Value::as_str).map(str::to_string).collect::<Vec<_>>)
            .unwrap_or_default(),
        report["adaptivity_savings_units"].as_i64(),
    );
    let frozen = (
        expected.adaptive_status.as_str(),
        expected.adaptive_worst_case_effort_units,
        expected.adaptive_root_action_id.as_deref(),
        expected.fixed_status.as_str(),
        expected.fixed_minimum_effort_units,
        expected.fixed_action_ids.clone(),
        expected.adaptivity_savings_units,
    );
    if observed != frozen {
        return Err(format!(
            "known-answer mismatch: observed {:?}, expected {:?}",
            observed, frozen
        ));
    }
    Ok(())
}

fn run(input_bytes: &[u8]) -> Result<Value, String> {
    let input: Input = serde_json::from_slice(input_bytes)
        .map_err(|error| format!("input JSON/schema decoding failed: {error}"))?;
    report_for_input(input_bytes, input)
}

fn main() -> ExitCode {
    let args = Args::parse();
    let bytes = match fs::read(&args.input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read {}: {error}", args.input.display());
            return ExitCode::FAILURE;
        }
    };
    let report = match run(&bytes) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("adaptive intervention policy audit rejected input: {error}");
            return ExitCode::FAILURE;
        }
    };
    let rendered = match serde_json::to_string_pretty(&report) {
        Ok(value) => value + "\n",
        Err(error) => {
            eprintln!("failed to serialize report: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(output) = args.output {
        if let Err(error) = fs::write(&output, rendered) {
            eprintln!("failed to write {}: {error}", output.display());
            return ExitCode::FAILURE;
        }
    } else {
        print!("{rendered}");
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADAPTIVE: &str = include_str!(
        "../../../../docs/neuroscience/examples/CONNECTOME_ADAPTIVE_INTERVENTION_POLICY_ADAPTIVE_SAVINGS.example.json"
    );
    const ERASED: &str = include_str!(
        "../../../../docs/neuroscience/examples/CONNECTOME_ADAPTIVE_INTERVENTION_POLICY_NUISANCE_ERASURE.example.json"
    );
    const NO_RESOLUTION: &str = include_str!(
        "../../../../docs/neuroscience/examples/CONNECTOME_ADAPTIVE_INTERVENTION_POLICY_NO_RESOLUTION.example.json"
    );

    fn check_fixture(raw: &str) -> Value {
        run(raw.as_bytes()).unwrap()
    }

    #[test]
    fn adaptive_policy_saves_one_effort_unit() {
        let report = check_fixture(ADAPTIVE);
        assert_eq!(report["adaptive_worst_case_effort_units"], 3);
        assert_eq!(report["adaptive_root_action_id"], "screen-group");
        assert_eq!(report["fixed_minimum_effort_units"], 4);
        assert_eq!(report["adaptivity_savings_units"], 1);
        assert_eq!(report["adaptivity_savings_basis_points"], 2500);
    }

    #[test]
    fn nuisance_erasure_removes_the_adaptive_advantage() {
        let report = check_fixture(ERASED);
        assert_eq!(report["adaptive_worst_case_effort_units"], 4);
        assert_eq!(report["adaptive_root_action_id"], "resolve-a");
        assert_eq!(report["fixed_minimum_effort_units"], 4);
        assert_eq!(report["adaptivity_savings_units"], 0);
    }

    #[test]
    fn observational_overlap_returns_no_guaranteed_policy() {
        let report = check_fixture(NO_RESOLUTION);
        assert_eq!(report["adaptive_status"], "NO_GUARANTEED_QUERY_RESOLUTION");
        assert_eq!(report["fixed_status"], "NO_GUARANTEED_FIXED_PLAN");
        assert!(report["adaptive_worst_case_effort_units"].is_null());
    }

    #[test]
    fn incomplete_potential_outcome_matrix_is_rejected() {
        let mut value: Value = serde_json::from_str(ADAPTIVE).unwrap();
        value["potential_outcomes"].as_array_mut().unwrap().pop();
        let raw = serde_json::to_vec(&value).unwrap();
        assert!(run(&raw).unwrap_err().contains("missing"));
    }

    #[test]
    fn strict_deserialization_rejects_unknown_fields() {
        let mut value: Value = serde_json::from_str(ADAPTIVE).unwrap();
        value["unexpected_field"] = json!(true);
        let raw = serde_json::to_vec(&value).unwrap();
        assert!(run(&raw).unwrap_err().contains("unknown field"));
    }

    #[test]
    fn action_row_permutation_preserves_canonical_results() {
        let mut value: Value = serde_json::from_str(ADAPTIVE).unwrap();
        value["worlds"].as_array_mut().unwrap().reverse();
        value["nuisance_models"].as_array_mut().unwrap().reverse();
        value["actions"].as_array_mut().unwrap().reverse();
        value["potential_outcomes"].as_array_mut().unwrap().reverse();
        let raw = serde_json::to_vec(&value).unwrap();
        let report = run(&raw).unwrap();
        assert_eq!(report["adaptive_worst_case_effort_units"], 3);
        assert_eq!(report["fixed_minimum_effort_units"], 4);
        assert_eq!(report["world_set_sha256"], check_fixture(ADAPTIVE)["world_set_sha256"]);
        assert_eq!(report["action_set_sha256"], check_fixture(ADAPTIVE)["action_set_sha256"]);
    }

    #[test]
    fn cheap_disallowed_decoy_cannot_change_optimum() {
        let mut value: Value = serde_json::from_str(ADAPTIVE).unwrap();
        value["actions"].as_array_mut().unwrap().push(json!({
            "action_id": "perfect-decoy",
            "intervention_kind": "SIMULATED",
            "effort_units": 1,
            "feasibility_status": "DISALLOWED",
            "feasibility_receipt_sha256": "d".repeat(64),
            "safety_profile_sha256": "e".repeat(64),
        }));
        for row in value["potential_outcomes"].as_array_mut().unwrap() {
            row["observations"].as_array_mut().unwrap().push(json!({
                "action_id": "perfect-decoy",
                "outcome_id": format!("decoy-{}-{}", row["world_id"].as_str().unwrap(), row["nuisance_model_id"].as_str().unwrap()),
            }));
        }
        let raw = serde_json::to_vec(&value).unwrap();
        let report = run(&raw).unwrap();
        assert_eq!(report["adaptive_worst_case_effort_units"], 3);
        assert_eq!(report["fixed_minimum_effort_units"], 4);
        assert!(!report["fixed_action_ids"].as_array().unwrap().iter().any(|v| v == "perfect-decoy"));
    }

    #[test]
    fn exhausted_memo_state_budget_returns_incomplete_not_exact() {
        let mut value: Value = serde_json::from_str(ADAPTIVE).unwrap();
        value["search_limits"]["max_memo_states"] = json!(1);
        let raw = serde_json::to_vec(&value).unwrap();
        let report = run(&raw).unwrap();
        assert_eq!(report["status"], "INCOMPLETE_SEARCH_BUDGET_EXCEEDED");
        assert_eq!(report["search_complete"], false);
        assert!(report.get("adaptive_worst_case_effort_units").is_none());
    }
}
