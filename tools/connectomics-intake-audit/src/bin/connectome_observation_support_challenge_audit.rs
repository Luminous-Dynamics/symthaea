//! Adversarial expansion audit for robust connectome observation supports.
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
const MAX_RETURNED_PLANS: usize = 128;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-observation-support-challenge-audit",
    about = "Audit robust plans under adversarial support-set expansion"
)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    challenge: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BaselineInput {
    schema_version: u32,
    analysis_id: String,
    dataset_revision_commit_sha: String,
    query_id: String,
    query_spec_sha256: String,
    scenario_set_scope: String,
    scenario_set_sha256: String,
    effort_unit: String,
    max_total_effort_units: u64,
    scenarios: Vec<Scenario>,
    actions: Vec<Action>,
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
    observation_model_sha256: String,
    dependence_groups: Vec<String>,
    outcomes: Vec<WorldOutcomes>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WorldOutcomes {
    scenario_id: String,
    possible_outcomes: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ChallengeInput {
    schema_version: u32,
    analysis_id: String,
    dataset_revision_commit_sha: String,
    query_id: String,
    query_spec_sha256: String,
    scenario_set_sha256: String,
    baseline_action_set_sha256: String,
    challenger_protocol_sha256: String,
    records: Vec<ChallengeRecord>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ChallengeRecord {
    challenge_id: String,
    action_id: String,
    scenario_id: String,
    additional_possible_outcomes: Vec<String>,
    source_evidence_sha256: String,
    dependence_groups: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Eq, PartialEq, Ord, PartialOrd)]
struct WorldPair {
    world_a: String,
    world_b: String,
}

#[derive(Debug, Serialize)]
struct ActionCoverage {
    action_id: String,
    effort_units: u64,
    robustly_separated_pairs: Vec<WorldPair>,
}

#[derive(Debug, Serialize)]
struct ActionPlan {
    action_ids: Vec<String>,
    effort_units: u64,
}

#[derive(Debug, Serialize)]
struct PlanSummary {
    robustly_separable_pair_count: usize,
    robustly_undiscriminable_pairs: Vec<WorldPair>,
    minimum_effort_units: Option<u64>,
    minimum_plan_count: u64,
    minimum_plans_returned: usize,
    minimum_plans_truncated: bool,
    minimum_plans: Vec<ActionPlan>,
    subset_count_evaluated: u64,
    within_declared_budget: Option<bool>,
}

#[derive(Debug, Serialize)]
struct RecordImpact {
    challenge_id: String,
    action_id: String,
    scenario_id: String,
    newly_admitted_outcome_count: usize,
    newly_admitted_outcomes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct LostActionSeparation {
    action_id: String,
    pair: WorldPair,
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
    baseline_action_set_sha256: String,
    challenge_input_sha256: String,
    challenge_set_sha256: String,
    challenger_protocol_sha256: String,
    effort_unit: String,
    max_total_effort_units: u64,
    challenge_record_count: usize,
    newly_admitted_outcome_count: usize,
    baseline: PlanSummary,
    challenged: PlanSummary,
    baseline_action_coverage: Vec<ActionCoverage>,
    challenged_action_coverage: Vec<ActionCoverage>,
    lost_action_separations: Vec<LostActionSeparation>,
    lost_global_separations: Vec<WorldPair>,
    record_impacts: Vec<RecordImpact>,
    outcome: &'static str,
    support_expansion_monotone: bool,
    monotonicity_violations: Vec<String>,
    search_complete: bool,
    claim_boundary: &'static str,
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn scenario_digest(scenarios: &[Scenario]) -> String {
    let mut rows = scenarios.to_vec();
    rows.sort_by(|a, b| a.scenario_id.cmp(&b.scenario_id));
    let mut h = Sha256::new();
    h.update(b"connectome-query-disambiguation-scenario-set-v1");
    h.update((rows.len() as u64).to_be_bytes());
    for row in rows {
        h.update(b"WORLD");
        hash_field(&mut h, &row.scenario_id);
        h.update(row.graph_sha256.as_bytes());
        h.update([u8::from(row.decision_met)]);
    }
    hex::encode(h.finalize())
}

fn action_digest(actions: &[Action]) -> String {
    let mut rows = actions.to_vec();
    rows.sort_by(|a, b| a.action_id.cmp(&b.action_id));
    let mut h = Sha256::new();
    h.update(b"connectome-robust-query-disambiguation-action-set-v1");
    h.update((rows.len() as u64).to_be_bytes());
    for mut action in rows {
        action.dependence_groups.sort();
        action
            .outcomes
            .sort_by(|a, b| a.scenario_id.cmp(&b.scenario_id));
        h.update(b"ACTION");
        hash_field(&mut h, &action.action_id);
        hash_field(&mut h, &action.action_kind);
        h.update(action.effort_units.to_be_bytes());
        h.update(action.source_evidence_sha256.as_bytes());
        h.update(action.observation_model_sha256.as_bytes());
        h.update((action.dependence_groups.len() as u64).to_be_bytes());
        for group in action.dependence_groups {
            h.update(b"DEPENDENCE");
            hash_field(&mut h, &group);
        }
        h.update((action.outcomes.len() as u64).to_be_bytes());
        for mut outcome in action.outcomes {
            outcome.possible_outcomes.sort();
            h.update(b"WORLD-OUTCOMES");
            hash_field(&mut h, &outcome.scenario_id);
            h.update((outcome.possible_outcomes.len() as u64).to_be_bytes());
            for item in outcome.possible_outcomes {
                h.update(b"POSSIBLE");
                hash_field(&mut h, &item);
            }
        }
    }
    hex::encode(h.finalize())
}

fn challenge_digest(records: &[ChallengeRecord]) -> String {
    let mut rows = records.to_vec();
    rows.sort_by(|a, b| a.challenge_id.cmp(&b.challenge_id));
    let mut h = Sha256::new();
    h.update(b"connectome-observation-support-challenge-set-v1");
    h.update((rows.len() as u64).to_be_bytes());
    for mut row in rows {
        row.additional_possible_outcomes.sort();
        row.dependence_groups.sort();
        h.update(b"CHALLENGE");
        hash_field(&mut h, &row.challenge_id);
        hash_field(&mut h, &row.action_id);
        hash_field(&mut h, &row.scenario_id);
        h.update(row.source_evidence_sha256.as_bytes());
        h.update((row.dependence_groups.len() as u64).to_be_bytes());
        for group in row.dependence_groups {
            h.update(b"DEPENDENCE");
            hash_field(&mut h, &group);
        }
        h.update((row.additional_possible_outcomes.len() as u64).to_be_bytes());
        for outcome in row.additional_possible_outcomes {
            h.update(b"ADDITIONAL");
            hash_field(&mut h, &outcome);
        }
    }
    hex::encode(h.finalize())
}

fn validate_baseline(input: &BaselineInput) -> Result<(), String> {
    if input.schema_version != 1 || !nonempty(&input.analysis_id) || !nonempty(&input.query_id) {
        return Err("baseline schema version, analysis_id or query_id is invalid".into());
    }
    if !valid_hex(&input.dataset_revision_commit_sha, 40)
        || !valid_hex(&input.query_spec_sha256, 64)
        || !valid_hex(&input.scenario_set_sha256, 64)
    {
        return Err("baseline identity or query digest is malformed".into());
    }
    if !matches!(
        input.scenario_set_scope.as_str(),
        "exhaustive_declared_scenario_space" | "sampled_sensitivity_set"
    ) {
        return Err("baseline scenario_set_scope is invalid".into());
    }
    if !nonempty(&input.effort_unit) || input.max_total_effort_units == 0 {
        return Err("baseline effort unit and budget must be positive".into());
    }
    if !(2..=MAX_SCENARIOS).contains(&input.scenarios.len())
        || input.actions.is_empty()
        || input.actions.len() > MAX_ACTIONS
    {
        return Err(format!(
            "baseline requires 2..={MAX_SCENARIOS} worlds and 1..={MAX_ACTIONS} actions"
        ));
    }
    let mut worlds = BTreeSet::new();
    for world in &input.scenarios {
        if !nonempty(&world.scenario_id)
            || !worlds.insert(world.scenario_id.clone())
            || !valid_hex(&world.graph_sha256, 64)
        {
            return Err("baseline world IDs must be unique and graph digests valid".into());
        }
    }
    if scenario_digest(&input.scenarios) != input.scenario_set_sha256 {
        return Err("baseline scenario-set digest mismatch".into());
    }
    let mut ids = BTreeSet::new();
    for action in &input.actions {
        if !nonempty(&action.action_id)
            || !ids.insert(action.action_id.clone())
            || action.effort_units == 0
        {
            return Err("baseline action IDs must be unique and effort positive".into());
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
        if !valid_hex(&action.source_evidence_sha256, 64)
            || !valid_hex(&action.observation_model_sha256, 64)
        {
            return Err(format!(
                "action {} provenance digests are invalid",
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
        let mut seen_worlds = BTreeSet::new();
        for outcome in &action.outcomes {
            if !worlds.contains(&outcome.scenario_id)
                || !seen_worlds.insert(outcome.scenario_id.clone())
                || outcome.possible_outcomes.is_empty()
            {
                return Err(format!(
                    "action {} has invalid or incomplete world outcome rows",
                    action.action_id
                ));
            }
            let mut observed = BTreeSet::new();
            for value in &outcome.possible_outcomes {
                if !nonempty(value) || !observed.insert(value) {
                    return Err(format!(
                        "action {} has empty or duplicate baseline outcome",
                        action.action_id
                    ));
                }
            }
        }
        if seen_worlds != worlds {
            return Err(format!(
                "action {} does not cover every baseline world",
                action.action_id
            ));
        }
    }
    Ok(())
}

fn validate_challenge(base: &BaselineInput, challenge: &ChallengeInput) -> Result<(), String> {
    if challenge.schema_version != 1 {
        return Err("challenge schema_version must equal 1".into());
    }
    if challenge.analysis_id != base.analysis_id
        || challenge.dataset_revision_commit_sha != base.dataset_revision_commit_sha
        || challenge.query_id != base.query_id
        || challenge.query_spec_sha256 != base.query_spec_sha256
        || challenge.scenario_set_sha256 != base.scenario_set_sha256
        || challenge.baseline_action_set_sha256 != action_digest(&base.actions)
    {
        return Err("challenge binding does not match the exact baseline analysis/query/scenario/action set".into());
    }
    if !valid_hex(&challenge.challenger_protocol_sha256, 64) {
        return Err("challenger_protocol_sha256 must be lowercase SHA-256".into());
    }
    let worlds = base
        .scenarios
        .iter()
        .map(|w| w.scenario_id.as_str())
        .collect::<BTreeSet<_>>();
    let actions = base
        .actions
        .iter()
        .map(|a| a.action_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut record_ids = BTreeSet::new();
    for row in &challenge.records {
        if !nonempty(&row.challenge_id) || !record_ids.insert(row.challenge_id.clone()) {
            return Err("challenge IDs must be non-empty and unique".into());
        }
        if !actions.contains(row.action_id.as_str()) || !worlds.contains(row.scenario_id.as_str()) {
            return Err(format!(
                "challenge {} references an unknown action or world",
                row.challenge_id
            ));
        }
        if row.additional_possible_outcomes.is_empty()
            || !valid_hex(&row.source_evidence_sha256, 64)
        {
            return Err(format!(
                "challenge {} requires outcomes and a source-evidence digest",
                row.challenge_id
            ));
        }
        let mut values = BTreeSet::new();
        for value in &row.additional_possible_outcomes {
            if !nonempty(value) || !values.insert(value) {
                return Err(format!(
                    "challenge {} has empty or duplicate additional outcomes",
                    row.challenge_id
                ));
            }
        }
        let mut groups = BTreeSet::new();
        for group in &row.dependence_groups {
            if !nonempty(group) || !groups.insert(group) {
                return Err(format!(
                    "challenge {} has empty or duplicate dependence groups",
                    row.challenge_id
                ));
            }
        }
    }
    Ok(())
}

fn conflict_pairs(scenarios: &[Scenario]) -> Vec<WorldPair> {
    let mut worlds = scenarios.to_vec();
    worlds.sort_by(|a, b| a.scenario_id.cmp(&b.scenario_id));
    let mut pairs = Vec::new();
    for i in 0..worlds.len() {
        for j in (i + 1)..worlds.len() {
            if worlds[i].decision_met != worlds[j].decision_met {
                pairs.push(WorldPair {
                    world_a: worlds[i].scenario_id.clone(),
                    world_b: worlds[j].scenario_id.clone(),
                });
            }
        }
    }
    pairs
}

fn supports(action: &Action) -> BTreeMap<String, BTreeSet<String>> {
    action
        .outcomes
        .iter()
        .map(|row| {
            (
                row.scenario_id.clone(),
                row.possible_outcomes.iter().cloned().collect(),
            )
        })
        .collect()
}

fn separates(action: &Action, pair: &WorldPair) -> bool {
    let rows = supports(action);
    rows.get(&pair.world_a)
        .is_some_and(|a| rows.get(&pair.world_b).is_some_and(|b| a.is_disjoint(b)))
}

fn coverage(actions: &[Action], pairs: &[WorldPair]) -> Vec<ActionCoverage> {
    let mut rows = actions
        .iter()
        .map(|action| ActionCoverage {
            action_id: action.action_id.clone(),
            effort_units: action.effort_units,
            robustly_separated_pairs: pairs
                .iter()
                .filter(|pair| separates(action, pair))
                .cloned()
                .collect(),
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| a.action_id.cmp(&b.action_id));
    rows
}

fn selected_ids(mask: u64, actions: &[Action]) -> Vec<String> {
    let mut ids = actions
        .iter()
        .enumerate()
        .filter(|(i, _)| mask & (1_u64 << i) != 0)
        .map(|(_, action)| action.action_id.clone())
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

fn summarize(actions: &[Action], pairs: &[WorldPair], budget: u64) -> Result<PlanSummary, String> {
    let mut undiscriminable = pairs
        .iter()
        .filter(|pair| !actions.iter().any(|a| separates(a, pair)))
        .cloned()
        .collect::<Vec<_>>();
    undiscriminable.sort();
    let separable = pairs.len() - undiscriminable.len();
    if pairs.is_empty() {
        return Ok(PlanSummary {
            robustly_separable_pair_count: 0,
            robustly_undiscriminable_pairs: vec![],
            minimum_effort_units: Some(0),
            minimum_plan_count: 1,
            minimum_plans_returned: 1,
            minimum_plans_truncated: false,
            minimum_plans: vec![ActionPlan {
                action_ids: vec![],
                effort_units: 0,
            }],
            subset_count_evaluated: 1,
            within_declared_budget: Some(true),
        });
    }
    if !undiscriminable.is_empty() {
        return Ok(PlanSummary {
            robustly_separable_pair_count: separable,
            robustly_undiscriminable_pairs: undiscriminable,
            minimum_effort_units: None,
            minimum_plan_count: 0,
            minimum_plans_returned: 0,
            minimum_plans_truncated: false,
            minimum_plans: vec![],
            subset_count_evaluated: 0,
            within_declared_budget: None,
        });
    }
    let words = pairs.len().div_ceil(64);
    let mut masks = vec![vec![0_u64; words]; actions.len()];
    for (ai, action) in actions.iter().enumerate() {
        for (pi, pair) in pairs.iter().enumerate() {
            if separates(action, pair) {
                masks[ai][pi / 64] |= 1_u64 << (pi % 64);
            }
        }
    }
    let full = (0..words)
        .map(|word| {
            let remaining = pairs.len().saturating_sub(word * 64);
            if remaining >= 64 {
                u64::MAX
            } else {
                (1_u64 << remaining) - 1
            }
        })
        .collect::<Vec<_>>();
    let total = (1_u64 << actions.len()) - 1;
    let mut best: Option<u64> = None;
    let mut count = 0_u64;
    let mut plans = Vec::new();
    for mask in 1..=total {
        let mut cost = 0_u64;
        let mut found = vec![0_u64; words];
        for (i, action) in actions.iter().enumerate() {
            if mask & (1_u64 << i) == 0 {
                continue;
            }
            cost = cost
                .checked_add(action.effort_units)
                .ok_or_else(|| "subset effort overflow".to_string())?;
            for (w, bits) in masks[i].iter().enumerate() {
                found[w] |= bits;
            }
        }
        if found != full {
            continue;
        }
        if best.is_none_or(|v| cost < v) {
            best = Some(cost);
            count = 0;
            plans.clear();
        }
        if best == Some(cost) {
            count += 1;
            if plans.len() < MAX_RETURNED_PLANS {
                plans.push(ActionPlan {
                    action_ids: selected_ids(mask, actions),
                    effort_units: cost,
                });
            }
        }
    }
    plans.sort_by(|a, b| a.action_ids.cmp(&b.action_ids));
    let min = best.ok_or_else(|| {
        "internal planner inconsistency: no covering subset after per-pair cover check".to_string()
    })?;
    Ok(PlanSummary {
        robustly_separable_pair_count: separable,
        robustly_undiscriminable_pairs: vec![],
        minimum_effort_units: Some(min),
        minimum_plan_count: count,
        minimum_plans_returned: plans.len(),
        minimum_plans_truncated: count > MAX_RETURNED_PLANS as u64,
        minimum_plans: plans,
        subset_count_evaluated: total,
        within_declared_budget: Some(min <= budget),
    })
}

fn evaluate(
    base: &BaselineInput,
    challenge: &ChallengeInput,
    challenge_bytes: &[u8],
) -> Result<Report, String> {
    validate_baseline(base)?;
    validate_challenge(base, challenge)?;
    let pairs = conflict_pairs(&base.scenarios);
    let mut challenged = base.actions.clone();
    let mut impacts = Vec::new();
    let mut new_count = 0_usize;
    for record in &challenge.records {
        let action = challenged
            .iter_mut()
            .find(|a| a.action_id == record.action_id)
            .expect("validated action");
        let world = action
            .outcomes
            .iter_mut()
            .find(|o| o.scenario_id == record.scenario_id)
            .expect("validated world");
        let mut newly_added = Vec::new();
        for value in &record.additional_possible_outcomes {
            if !world.possible_outcomes.contains(value) {
                world.possible_outcomes.push(value.clone());
                newly_added.push(value.clone());
                new_count += 1;
            }
        }
        newly_added.sort();
        impacts.push(RecordImpact {
            challenge_id: record.challenge_id.clone(),
            action_id: record.action_id.clone(),
            scenario_id: record.scenario_id.clone(),
            newly_admitted_outcome_count: newly_added.len(),
            newly_admitted_outcomes: newly_added,
        });
    }
    impacts.sort_by(|a, b| a.challenge_id.cmp(&b.challenge_id));

    let baseline_summary = summarize(&base.actions, &pairs, base.max_total_effort_units)?;
    let challenged_summary = summarize(&challenged, &pairs, base.max_total_effort_units)?;
    let baseline_coverage = coverage(&base.actions, &pairs);
    let challenged_coverage = coverage(&challenged, &pairs);
    let mut lost_action = Vec::new();
    for before in &baseline_coverage {
        let after = challenged_coverage
            .iter()
            .find(|row| row.action_id == before.action_id)
            .expect("same actions");
        for pair in &before.robustly_separated_pairs {
            if !after.robustly_separated_pairs.contains(pair) {
                lost_action.push(LostActionSeparation {
                    action_id: before.action_id.clone(),
                    pair: pair.clone(),
                });
            }
        }
    }
    lost_action.sort_by(|a, b| (&a.action_id, &a.pair).cmp(&(&b.action_id, &b.pair)));
    let mut lost_global = baseline_summary.robustly_undiscriminable_pairs.clone();
    for pair in &pairs {
        let before = base.actions.iter().any(|a| separates(a, pair));
        let after = challenged.iter().any(|a| separates(a, pair));
        if before && !after {
            lost_global.push(pair.clone());
        }
    }
    lost_global.sort();
    lost_global.dedup();

    let mut violations = Vec::new();
    for row in &baseline_coverage {
        let after = challenged_coverage
            .iter()
            .find(|candidate| candidate.action_id == row.action_id)
            .expect("same actions");
        if after.robustly_separated_pairs.len() > row.robustly_separated_pairs.len() {
            violations.push(format!(
                "action {} gained robust coverage after support expansion",
                row.action_id
            ));
        }
    }
    if challenged_summary.robustly_separable_pair_count
        > baseline_summary.robustly_separable_pair_count
    {
        violations.push("global robust pair coverage increased after support expansion".into());
    }
    if let (Some(before), Some(after)) = (
        baseline_summary.minimum_effort_units,
        challenged_summary.minimum_effort_units,
    ) && after < before
    {
        violations.push(format!(
            "minimum effort decreased from {before} to {after} after support expansion"
        ));
    }
    if baseline_summary
        .robustly_undiscriminable_pairs
        .iter()
        .any(|p| {
            !challenged_summary
                .robustly_undiscriminable_pairs
                .contains(p)
        })
    {
        violations.push(
            "a baseline-undiscriminable pair became discriminable after support expansion".into(),
        );
    }

    let outcome = if challenge.records.is_empty() {
        "NO_CHALLENGE_ADDITIONS"
    } else if baseline_summary.minimum_effort_units.is_none() {
        "BASELINE_ALREADY_HAS_NO_ROBUST_PLAN"
    } else if challenged_summary.minimum_effort_units.is_none() {
        "ROBUST_PLAN_DESTROYED_BY_SUPPORT_EXPANSION"
    } else if challenged_summary.minimum_effort_units > baseline_summary.minimum_effort_units {
        "ROBUST_PLAN_COST_INCREASED"
    } else {
        "ROBUST_PLAN_PRESERVED"
    };

    Ok(Report {
        schema_version: 1,
        tool: "connectome-observation-support-challenge-audit",
        analysis_id: base.analysis_id.clone(),
        dataset_revision_commit_sha: base.dataset_revision_commit_sha.clone(),
        query_id: base.query_id.clone(),
        query_spec_sha256: base.query_spec_sha256.clone(),
        scenario_set_sha256: base.scenario_set_sha256.clone(),
        baseline_action_set_sha256: action_digest(&base.actions),
        challenge_input_sha256: sha256(challenge_bytes),
        challenge_set_sha256: challenge_digest(&challenge.records),
        challenger_protocol_sha256: challenge.challenger_protocol_sha256.clone(),
        effort_unit: base.effort_unit.clone(),
        max_total_effort_units: base.max_total_effort_units,
        challenge_record_count: challenge.records.len(),
        newly_admitted_outcome_count: new_count,
        baseline: baseline_summary,
        challenged: challenged_summary,
        baseline_action_coverage: baseline_coverage,
        challenged_action_coverage: challenged_coverage,
        lost_action_separations: lost_action,
        lost_global_separations: lost_global,
        record_impacts: impacts,
        outcome,
        support_expansion_monotone: violations.is_empty(),
        monotonicity_violations: violations,
        search_complete: true,
        claim_boundary: "Results are conditional on the baseline worlds, observation actions, and supplied challenge records. Challenge records only expand possible-outcome supports; they do not prove support completeness, authenticity, reviewer independence, or biological truth. No probabilities or likelihoods are inferred.",
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let base_bytes = match fs::read(&args.input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("could not read baseline: {e}");
            return ExitCode::from(2);
        }
    };
    let challenge_bytes = match fs::read(&args.challenge) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("could not read challenge: {e}");
            return ExitCode::from(2);
        }
    };
    let base: BaselineInput = match serde_json::from_slice(&base_bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("invalid baseline JSON: {e}");
            return ExitCode::from(2);
        }
    };
    let challenge: ChallengeInput = match serde_json::from_slice(&challenge_bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("invalid challenge JSON: {e}");
            return ExitCode::from(2);
        }
    };
    let report = match evaluate(&base, &challenge, &challenge_bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("connectome-observation-support-challenge-audit: {e}");
            return ExitCode::from(2);
        }
    };
    let rendered = match serde_json::to_string_pretty(&report) {
        Ok(v) => v + "\n",
        Err(e) => {
            eprintln!("could not serialize report: {e}");
            return ExitCode::from(2);
        }
    };
    if let Some(path) = args.output {
        if let Err(e) = fs::write(path, rendered.as_bytes()) {
            eprintln!("could not write report: {e}");
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

    fn base() -> BaselineInput {
        serde_json::from_str(include_str!("../../../../docs/neuroscience/examples/CONNECTOME_ROBUST_QUERY_DISAMBIGUATION.example.json")).unwrap()
    }

    fn challenge() -> ChallengeInput {
        serde_json::from_str(include_str!("../../../../docs/neuroscience/examples/CONNECTOME_SUPPORT_CHALLENGE_COST_INCREASE.example.json")).unwrap()
    }

    fn destroyed() -> ChallengeInput {
        serde_json::from_str(include_str!("../../../../docs/neuroscience/examples/CONNECTOME_SUPPORT_CHALLENGE_DESTROYED.example.json")).unwrap()
    }

    fn bytes<T: Serialize>(value: &T) -> Vec<u8> {
        serde_json::to_vec(value).unwrap()
    }

    #[test]
    fn broadened_support_increases_exact_minimum_from_four_to_five() {
        let b = base();
        let c = challenge();
        let r = evaluate(&b, &c, &bytes(&c)).unwrap();
        assert_eq!(r.baseline.minimum_effort_units, Some(4));
        assert_eq!(r.challenged.minimum_effort_units, Some(5));
        assert_eq!(r.outcome, "ROBUST_PLAN_COST_INCREASED");
        assert_eq!(r.challenged.within_declared_budget, Some(false));
        assert_eq!(r.lost_global_separations.len(), 0);
        assert!(r.support_expansion_monotone);
    }

    #[test]
    fn challenged_support_cannot_gain_pair_coverage_or_lower_cost() {
        let b = base();
        let c = challenge();
        let r = evaluate(&b, &c, &bytes(&c)).unwrap();
        assert!(r.support_expansion_monotone);
        assert_eq!(r.lost_action_separations.len(), 3);
        assert_eq!(r.newly_admitted_outcome_count, 3);
    }

    #[test]
    fn adversarial_expansion_can_destroy_every_robust_plan() {
        let b = base();
        let c = destroyed();
        let r = evaluate(&b, &c, &bytes(&c)).unwrap();
        assert_eq!(r.baseline.minimum_effort_units, Some(4));
        assert_eq!(r.challenged.minimum_effort_units, None);
        assert_eq!(r.outcome, "ROBUST_PLAN_DESTROYED_BY_SUPPORT_EXPANSION");
        assert_eq!(r.lost_global_separations.len(), 2);
        assert!(r.support_expansion_monotone);
    }

    #[test]
    fn wrong_baseline_binding_fails_closed() {
        let b = base();
        let mut c = challenge();
        c.baseline_action_set_sha256 = "0".repeat(64);
        assert!(validate_challenge(&b, &c).unwrap_err().contains("binding"));
    }

    #[test]
    fn challenge_must_have_provenance_and_nonempty_additions() {
        let b = base();
        let mut c = challenge();
        c.records[0].source_evidence_sha256 = "x".into();
        assert!(
            validate_challenge(&b, &c)
                .unwrap_err()
                .contains("source-evidence")
        );
    }

    #[test]
    fn challenge_record_order_does_not_change_canonical_digest() {
        let mut c = challenge();
        let digest = challenge_digest(&c.records);
        c.records.reverse();
        for r in &mut c.records {
            r.additional_possible_outcomes.reverse();
            r.dependence_groups.reverse();
        }
        assert_eq!(digest, challenge_digest(&c.records));
    }

    #[test]
    fn no_records_is_an_explicit_noop() {
        let b = base();
        let mut c = challenge();
        c.records.clear();
        let r = evaluate(&b, &c, &bytes(&c)).unwrap();
        assert_eq!(r.outcome, "NO_CHALLENGE_ADDITIONS");
        assert_eq!(
            r.baseline.minimum_effort_units,
            r.challenged.minimum_effort_units
        );
        assert!(r.lost_action_separations.is_empty());
    }
}
