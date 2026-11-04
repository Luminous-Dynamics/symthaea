//! Exact minimum-cost observation set that separates all declared
//! reconstruction-world pairs with opposing scientific-query decisions.
//!
//! V1 is deterministic set-valued experimental design. It does not compute
//! entropy, posterior probabilities or expected value of information.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const MAX_ACTIONS: usize = 20;
const MAX_SCENARIOS: usize = 24;
const MAX_RETURNED_MINIMAL_PLANS: usize = 128;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-query-disambiguation-audit",
    version,
    about = "Find minimum-cost deterministic observations that disambiguate query-conflicting connectome worlds"
)]
struct Args {
    /// Frozen world set, query decision labels and observation outcome mappings.
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
    effort_unit: String,
    max_total_effort_units: u64,
    scenarios: Vec<Scenario>,
    actions: Vec<ObservationAction>,
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
struct ObservationOutcome {
    scenario_id: String,
    outcome_id: String,
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
    outcomes: Vec<ObservationOutcome>,
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
    effort_unit: String,
    max_total_effort_units: u64,
    scenario_count: usize,
    observation_action_count: usize,
    decision_conflict_pair_count: usize,
    individually_discriminable_pair_count: usize,
    undiscriminable_pairs: Vec<WorldPair>,
    action_pair_coverage: Vec<ActionCoverage>,
    subset_count_evaluated: u64,
    minimum_disambiguation_effort_units: Option<u64>,
    minimum_plan_count: u64,
    plans_returned: usize,
    minimum_plans_truncated: bool,
    minimum_plan_within_budget: Option<bool>,
    plans: Vec<DisambiguationPlan>,
    greedy_comparator: GreedyPlan,
    status: &'static str,
    search_complete: bool,
    probability_claim: &'static str,
    scope_note: &'static str,
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
    separated_decision_conflict_pair_count: usize,
}

#[derive(Debug, Serialize)]
struct DisambiguationPlan {
    action_ids: Vec<String>,
    effort_units: u64,
    separated_decision_conflict_pair_count: usize,
}

#[derive(Debug, Serialize)]
struct GreedyPlan {
    action_ids: Vec<String>,
    effort_units: u64,
    separated_decision_conflict_pair_count: usize,
    remaining_pairs: Vec<WorldPair>,
    complete: bool,
    regret_to_exact_minimum_effort_units: Option<u64>,
}

type OutcomesByWorld = BTreeMap<String, String>;

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
    for scenario in rows {
        hasher.update(b"WORLD");
        hash_field(&mut hasher, &scenario.scenario_id);
        hasher.update(scenario.graph_sha256.as_bytes());
        hasher.update([u8::from(scenario.decision_met)]);
    }
    hex::encode(hasher.finalize())
}

fn canonical_action_set_sha256(actions: &[ObservationAction]) -> String {
    let mut rows = actions.to_vec();
    rows.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    let mut hasher = Sha256::new();
    hasher.update(b"connectome-query-disambiguation-action-set-v1");
    hasher.update((rows.len() as u64).to_be_bytes());
    for mut action in rows {
        action.dependence_groups.sort();
        action.outcomes.sort_by(|left, right| {
            left.scenario_id
                .cmp(&right.scenario_id)
                .then_with(|| left.outcome_id.cmp(&right.outcome_id))
        });
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
        hasher.update((action.outcomes.len() as u64).to_be_bytes());
        for outcome in action.outcomes {
            hasher.update(b"OUTCOME");
            hash_field(&mut hasher, &outcome.scenario_id);
            hash_field(&mut hasher, &outcome.outcome_id);
        }
    }
    hex::encode(hasher.finalize())
}

fn validate(input: &Input) -> Result<Vec<OutcomesByWorld>, String> {
    if input.schema_version != 1 {
        return Err("schema_version must equal 1".into());
    }
    if !nonempty(&input.analysis_id) || !nonempty(&input.query_id) {
        return Err("analysis_id and query_id must be non-empty".into());
    }
    if !valid_hex(&input.dataset_revision_commit_sha, 40) {
        return Err("dataset_revision_commit_sha must be lowercase 40-character Git SHA".into());
    }
    if !valid_hex(&input.query_spec_sha256, 64) {
        return Err("query_spec_sha256 must be lowercase SHA-256".into());
    }
    if !valid_hex(&input.scenario_set_sha256, 64) {
        return Err("scenario_set_sha256 must be lowercase SHA-256".into());
    }
    if !matches!(
        input.scenario_set_scope.as_str(),
        "exhaustive_declared_scenario_space" | "sampled_sensitivity_set"
    ) {
        return Err("scenario_set_scope is not a recognized finite-world scope".into());
    }
    if !nonempty(&input.effort_unit) || input.max_total_effort_units == 0 {
        return Err("effort_unit must be non-empty and max_total_effort_units positive".into());
    }
    if !(2..=MAX_SCENARIOS).contains(&input.scenarios.len()) {
        return Err(format!(
            "scenarios must contain between 2 and {MAX_SCENARIOS} worlds for exact pair analysis"
        ));
    }
    if input.actions.is_empty() || input.actions.len() > MAX_ACTIONS {
        return Err(format!(
            "actions must contain between 1 and {MAX_ACTIONS} observation actions for exact search"
        ));
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
    let actual_scenario_set_sha = canonical_scenario_set_sha256(&input.scenarios);
    if actual_scenario_set_sha != input.scenario_set_sha256 {
        return Err(format!(
            "scenario-set digest mismatch: declared {}, computed {}",
            input.scenario_set_sha256, actual_scenario_set_sha
        ));
    }

    let mut action_ids = BTreeSet::new();
    let mut total_effort = 0_u64;
    let mut outcomes_by_action = Vec::with_capacity(input.actions.len());
    for action in &input.actions {
        if !nonempty(&action.action_id) || !action_ids.insert(action.action_id.clone()) {
            return Err("action IDs must be non-empty and unique".into());
        }
        if action.effort_units == 0 {
            return Err(format!(
                "action {} effort_units must be positive",
                action.action_id
            ));
        }
        total_effort = total_effort
            .checked_add(action.effort_units)
            .ok_or_else(|| "sum of observation effort_units overflows u64".to_string())?;
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

        let mut outcomes = BTreeMap::new();
        for outcome in &action.outcomes {
            if !scenario_ids.contains(&outcome.scenario_id) || !nonempty(&outcome.outcome_id) {
                return Err(format!(
                    "action {} has an unknown world or empty outcome",
                    action.action_id
                ));
            }
            if outcomes
                .insert(outcome.scenario_id.clone(), outcome.outcome_id.clone())
                .is_some()
            {
                return Err(format!(
                    "action {} has duplicate outcome for world {}",
                    action.action_id, outcome.scenario_id
                ));
            }
        }
        if outcomes.len() != input.scenarios.len() {
            return Err(format!(
                "action {} must declare exactly one outcome for every scenario",
                action.action_id
            ));
        }
        outcomes_by_action.push(outcomes);
    }
    Ok(outcomes_by_action)
}

fn decision_conflict_pairs(scenarios: &[Scenario]) -> Vec<WorldPair> {
    let mut rows = scenarios.to_vec();
    rows.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut pairs = Vec::new();
    for left in 0..rows.len() {
        for right in (left + 1)..rows.len() {
            if rows[left].decision_met != rows[right].decision_met {
                pairs.push(WorldPair {
                    world_a: rows[left].scenario_id.clone(),
                    world_b: rows[right].scenario_id.clone(),
                });
            }
        }
    }
    pairs
}

fn action_separates_pair(outcomes: &OutcomesByWorld, pair: &WorldPair) -> bool {
    outcomes.get(&pair.world_a) != outcomes.get(&pair.world_b)
}

fn action_pair_coverage(
    actions: &[ObservationAction],
    outcomes_by_action: &[OutcomesByWorld],
    pairs: &[WorldPair],
) -> Vec<ActionCoverage> {
    actions
        .iter()
        .zip(outcomes_by_action)
        .map(|(action, outcomes)| ActionCoverage {
            action_id: action.action_id.clone(),
            effort_units: action.effort_units,
            separated_decision_conflict_pair_count: pairs
                .iter()
                .filter(|pair| action_separates_pair(outcomes, pair))
                .count(),
        })
        .collect()
}

fn greedy_disambiguation_plan(
    actions: &[ObservationAction],
    outcomes_by_action: &[OutcomesByWorld],
    pairs: &[WorldPair],
    exact_minimum_cost: Option<u64>,
) -> GreedyPlan {
    let mut uncovered = pairs.iter().cloned().collect::<BTreeSet<_>>();
    let mut selected_indices = BTreeSet::new();
    let mut selected_ids = Vec::new();
    let mut effort_units = 0_u64;

    loop {
        if uncovered.is_empty() {
            break;
        }
        let mut best: Option<(usize, usize)> = None;
        for (index, action) in actions.iter().enumerate() {
            if selected_indices.contains(&index) {
                continue;
            }
            let marginal = uncovered
                .iter()
                .filter(|pair| action_separates_pair(&outcomes_by_action[index], pair))
                .count();
            if marginal == 0 {
                continue;
            }
            let replace = match best {
                None => true,
                Some((best_index, best_marginal)) => {
                    let candidate_cross =
                        (marginal as u128) * (actions[best_index].effort_units as u128);
                    let best_cross = (best_marginal as u128) * (action.effort_units as u128);
                    candidate_cross > best_cross
                        || (candidate_cross == best_cross
                            && (marginal > best_marginal
                                || (marginal == best_marginal
                                    && action.action_id < actions[best_index].action_id)))
                }
            };
            if replace {
                best = Some((index, marginal));
            }
        }
        let Some((best_index, _)) = best else {
            break;
        };
        let action = &actions[best_index];
        effort_units = effort_units.saturating_add(action.effort_units);
        selected_indices.insert(best_index);
        selected_ids.push(action.action_id.clone());
        uncovered.retain(|pair| !action_separates_pair(&outcomes_by_action[best_index], pair));
    }

    selected_ids.sort();
    let complete = uncovered.is_empty();
    let separated = pairs.len() - uncovered.len();
    let regret = if complete {
        exact_minimum_cost.map(|minimum| effort_units.saturating_sub(minimum))
    } else {
        None
    };
    GreedyPlan {
        action_ids: selected_ids,
        effort_units,
        separated_decision_conflict_pair_count: separated,
        remaining_pairs: uncovered.into_iter().collect(),
        complete,
        regret_to_exact_minimum_effort_units: regret,
    }
}

fn selected_action_ids(mask: u64, actions: &[ObservationAction]) -> Vec<String> {
    let mut ids = actions
        .iter()
        .enumerate()
        .filter(|(index, _)| mask & (1_u64 << index) != 0)
        .map(|(_, action)| action.action_id.clone())
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

fn exact_minimum_disambiguation(
    input: &Input,
    pairs: &[WorldPair],
    outcomes_by_action: &[OutcomesByWorld],
) -> Result<(u64, u64, Vec<DisambiguationPlan>), String> {
    let pair_count = pairs.len();
    let word_count = pair_count.div_ceil(64);
    let mut action_masks = vec![vec![0_u64; word_count]; input.actions.len()];
    for (action_index, outcomes) in outcomes_by_action.iter().enumerate() {
        for (pair_index, pair) in pairs.iter().enumerate() {
            if action_separates_pair(outcomes, pair) {
                action_masks[action_index][pair_index / 64] |= 1_u64 << (pair_index % 64);
            }
        }
    }
    let complete_masks = (0..word_count)
        .map(|word| {
            let remaining = pair_count.saturating_sub(word * 64);
            if remaining >= 64 {
                u64::MAX
            } else {
                (1_u64 << remaining) - 1
            }
        })
        .collect::<Vec<_>>();

    let total_subsets = (1_u64 << input.actions.len()) - 1;
    let mut best_cost: Option<u64> = None;
    let mut best_count = 0_u64;
    let mut plans = Vec::new();

    for mask in 1..=total_subsets {
        let mut cost = 0_u64;
        let mut covered = vec![0_u64; word_count];
        for (index, action) in input.actions.iter().enumerate() {
            if mask & (1_u64 << index) == 0 {
                continue;
            }
            cost = cost
                .checked_add(action.effort_units)
                .ok_or_else(|| "action subset cost overflows u64".to_string())?;
            for (word, action_word) in action_masks[index].iter().enumerate() {
                covered[word] |= action_word;
            }
        }
        if covered != complete_masks {
            continue;
        }
        let replace_best = best_cost.is_none_or(|best| cost < best);
        if replace_best {
            best_cost = Some(cost);
            best_count = 0;
            plans.clear();
        }
        if best_cost == Some(cost) {
            best_count += 1;
            if plans.len() < MAX_RETURNED_MINIMAL_PLANS {
                plans.push(DisambiguationPlan {
                    action_ids: selected_action_ids(mask, &input.actions),
                    effort_units: cost,
                    separated_decision_conflict_pair_count: pair_count,
                });
            }
        }
    }
    plans.sort_by(|left, right| left.action_ids.cmp(&right.action_ids));
    Ok((total_subsets, best_count, plans))
}

fn evaluate(input: &Input, input_bytes: &[u8]) -> Result<Report, String> {
    let outcomes_by_action = validate(input)?;
    let mut scenarios = input.scenarios.clone();
    scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let pairs = decision_conflict_pairs(&scenarios);
    let pair_count = pairs.len();
    let mut coverage = action_pair_coverage(&input.actions, &outcomes_by_action, &pairs);
    coverage.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    let undiscriminable_pairs = pairs
        .iter()
        .filter(|pair| {
            !outcomes_by_action
                .iter()
                .any(|outcomes| action_separates_pair(outcomes, pair))
        })
        .cloned()
        .collect::<Vec<_>>();
    let individually_discriminable = pair_count - undiscriminable_pairs.len();

    let (subset_count, minimum_count, mut plans, min_cost) =
        if pair_count == 0 || !undiscriminable_pairs.is_empty() {
            (0, 0, Vec::new(), None)
        } else {
            let (subsets, count, plans) =
                exact_minimum_disambiguation(input, &pairs, &outcomes_by_action)?;
            let cost = plans.first().map(|plan| plan.effort_units);
            (subsets, count, plans, cost)
        };
    let status = if pair_count == 0 {
        "NO_DECISION_DISAGREEMENT_WITHIN_DECLARED_SCENARIOS"
    } else if !undiscriminable_pairs.is_empty() {
        "NO_COMPLETE_DISAMBIGUATION_WITHIN_DECLARED_ACTIONS"
    } else if min_cost.is_some_and(|cost| cost <= input.max_total_effort_units) {
        "MINIMUM_COST_DISAMBIGUATION_PLAN"
    } else {
        "MINIMUM_COST_DISAMBIGUATION_EXCEEDS_BUDGET"
    };
    let minimum_plan_within_budget = min_cost.map(|cost| cost <= input.max_total_effort_units);
    let greedy_comparator =
        greedy_disambiguation_plan(&input.actions, &outcomes_by_action, &pairs, min_cost);
    let plans_truncated = minimum_count > MAX_RETURNED_MINIMAL_PLANS as u64;
    if plans.len() > MAX_RETURNED_MINIMAL_PLANS {
        plans.truncate(MAX_RETURNED_MINIMAL_PLANS);
    }

    Ok(Report {
        schema_version: 1,
        tool: "connectome-query-disambiguation-audit",
        analysis_id: input.analysis_id.clone(),
        input_sha256: sha256(input_bytes),
        dataset_revision_commit_sha: input.dataset_revision_commit_sha.clone(),
        query_id: input.query_id.clone(),
        query_spec_sha256: input.query_spec_sha256.clone(),
        scenario_set_scope: input.scenario_set_scope.clone(),
        scenario_set_sha256: input.scenario_set_sha256.clone(),
        action_set_sha256: canonical_action_set_sha256(&input.actions),
        effort_unit: input.effort_unit.clone(),
        max_total_effort_units: input.max_total_effort_units,
        scenario_count: input.scenarios.len(),
        observation_action_count: input.actions.len(),
        decision_conflict_pair_count: pair_count,
        individually_discriminable_pair_count: individually_discriminable,
        undiscriminable_pairs,
        action_pair_coverage: coverage,
        subset_count_evaluated: subset_count,
        minimum_disambiguation_effort_units: min_cost,
        minimum_plan_count: minimum_count,
        plans_returned: plans.len(),
        minimum_plans_truncated: plans_truncated,
        minimum_plan_within_budget,
        plans: std::mem::take(&mut plans),
        greedy_comparator,
        status,
        search_complete: true,
        probability_claim: "NONE_DETERMINISTIC_WORLD_DISCRIMINATION_ONLY",
        scope_note: "This computes a minimum-cost set of declared deterministic observation actions that separates every pair of submitted worlds with opposing query decisions. Outcome mappings are caller-supplied hypotheses bound by their model/source digests; the tool does not verify measurement validity, independence, biological truth, observation noise or completeness of the world/action universe. It computes no entropy, posterior probability or expected value of information. No-disagreement and no-plan results are scoped to the exact submitted finite worlds and action mappings.",
    })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let input_bytes = match fs::read(&args.input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("connectome-query-disambiguation-audit: could not read input: {error}");
            return ExitCode::from(2);
        }
    };
    let input: Input = match serde_json::from_slice(&input_bytes) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("connectome-query-disambiguation-audit: invalid JSON input: {error}");
            return ExitCode::from(2);
        }
    };
    let report = match evaluate(&input, &input_bytes) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("connectome-query-disambiguation-audit: {error}");
            return ExitCode::from(2);
        }
    };
    let rendered = match serde_json::to_string_pretty(&report) {
        Ok(json) => json + "\n",
        Err(error) => {
            eprintln!("connectome-query-disambiguation-audit: could not serialize report: {error}");
            return ExitCode::from(2);
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(path, rendered.as_bytes()) {
            eprintln!("connectome-query-disambiguation-audit: could not write report: {error}");
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
            "../../../../docs/neuroscience/examples/CONNECTOME_QUERY_DISAMBIGUATION.example.json"
        ))
        .expect("checked-in observation disambiguation fixture must parse")
    }

    fn fixture_bytes() -> Vec<u8> {
        include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_QUERY_DISAMBIGUATION.example.json"
        )
        .as_bytes()
        .to_vec()
    }

    #[test]
    fn exact_plan_uses_two_actions_to_separate_all_conflicting_world_pairs() {
        let input = fixture();
        let report = evaluate(&input, &fixture_bytes()).unwrap();
        assert_eq!(report.decision_conflict_pair_count, 4);
        assert_eq!(report.individually_discriminable_pair_count, 4);
        assert_eq!(report.minimum_disambiguation_effort_units, Some(4));
        assert_eq!(report.minimum_plan_count, 2);
        assert_eq!(report.status, "MINIMUM_COST_DISAMBIGUATION_PLAN");
        assert_eq!(report.minimum_plan_within_budget, Some(true));
        assert_eq!(report.subset_count_evaluated, 63);
        assert!(
            report
                .plans
                .iter()
                .all(|plan| { plan.action_ids.len() == 2 && plan.effort_units == 4 })
        );
        assert!(!report.probability_claim.contains("POSTERIOR"));
    }

    #[test]
    fn greedy_comparator_reports_one_unit_of_regret_against_exact_optimum() {
        let input = fixture();
        let report = evaluate(&input, &fixture_bytes()).unwrap();
        assert_eq!(report.minimum_disambiguation_effort_units, Some(4));
        assert!(report.greedy_comparator.complete);
        assert_eq!(report.greedy_comparator.effort_units, 5);
        assert_eq!(
            report
                .greedy_comparator
                .regret_to_exact_minimum_effort_units,
            Some(1)
        );
        assert_eq!(
            report.greedy_comparator.action_ids,
            vec![
                "inspect-morphology".to_string(),
                "partial-broad-scan".to_string()
            ]
        );
    }

    #[test]
    fn one_observation_cannot_claim_to_separate_identical_outcomes() {
        let mut input = fixture();
        let action = input
            .actions
            .iter_mut()
            .find(|action| action.action_id == "inspect-synapse")
            .unwrap();
        let alpha_outcome = action
            .outcomes
            .iter()
            .find(|outcome| outcome.scenario_id == "world-alpha")
            .unwrap()
            .outcome_id
            .clone();
        let beta_outcome = action
            .outcomes
            .iter_mut()
            .find(|outcome| outcome.scenario_id == "world-beta")
            .unwrap();
        beta_outcome.outcome_id = alpha_outcome;
        let bytes = serde_json::to_vec(&input).unwrap();
        let report = evaluate(&input, &bytes).unwrap();
        let coverage = report
            .action_pair_coverage
            .iter()
            .find(|row| row.action_id == "inspect-synapse")
            .unwrap();
        assert_eq!(coverage.separated_decision_conflict_pair_count, 2);
        assert!(report.search_complete);
    }

    #[test]
    fn no_action_can_distinguish_a_pair_means_no_complete_plan() {
        let mut input = fixture();
        for action in &mut input.actions {
            let alpha = action
                .outcomes
                .iter()
                .find(|outcome| outcome.scenario_id == "world-alpha")
                .unwrap()
                .outcome_id
                .clone();
            let beta = action
                .outcomes
                .iter_mut()
                .find(|outcome| outcome.scenario_id == "world-beta")
                .unwrap();
            beta.outcome_id = alpha;
        }
        let bytes = serde_json::to_vec(&input).unwrap();
        let report = evaluate(&input, &bytes).unwrap();
        assert_eq!(
            report.status,
            "NO_COMPLETE_DISAMBIGUATION_WITHIN_DECLARED_ACTIONS"
        );
        assert!(!report.undiscriminable_pairs.is_empty());
        assert_eq!(report.minimum_disambiguation_effort_units, None);
    }

    #[test]
    fn greedy_comparator_reports_partial_coverage_when_no_action_can_finish() {
        let mut input = fixture();
        for action in &mut input.actions {
            let alpha = action
                .outcomes
                .iter()
                .find(|outcome| outcome.scenario_id == "world-alpha")
                .unwrap()
                .outcome_id
                .clone();
            let beta = action
                .outcomes
                .iter_mut()
                .find(|outcome| outcome.scenario_id == "world-beta")
                .unwrap();
            beta.outcome_id = alpha;
        }
        let bytes = serde_json::to_vec(&input).unwrap();
        let report = evaluate(&input, &bytes).unwrap();
        assert!(!report.greedy_comparator.complete);
        assert_eq!(
            report
                .greedy_comparator
                .separated_decision_conflict_pair_count,
            3
        );
        assert_eq!(report.greedy_comparator.remaining_pairs.len(), 1);
        assert_eq!(
            report
                .greedy_comparator
                .regret_to_exact_minimum_effort_units,
            None
        );
    }

    #[test]
    fn no_query_decision_disagreement_requires_no_disambiguation_plan() {
        let mut input = fixture();
        for scenario in &mut input.scenarios {
            scenario.decision_met = true;
        }
        input.scenario_set_sha256 = canonical_scenario_set_sha256(&input.scenarios);
        let bytes = serde_json::to_vec(&input).unwrap();
        let report = evaluate(&input, &bytes).unwrap();
        assert_eq!(
            report.status,
            "NO_DECISION_DISAGREEMENT_WITHIN_DECLARED_SCENARIOS"
        );
        assert_eq!(report.decision_conflict_pair_count, 0);
        assert_eq!(report.subset_count_evaluated, 0);
    }

    #[test]
    fn complete_plan_is_reported_but_marked_over_budget_when_cost_exceeds_limit() {
        let mut input = fixture();
        input.max_total_effort_units = 3;
        let bytes = serde_json::to_vec(&input).unwrap();
        let report = evaluate(&input, &bytes).unwrap();
        assert_eq!(report.minimum_disambiguation_effort_units, Some(4));
        assert_eq!(report.minimum_plan_count, 2);
        assert_eq!(report.minimum_plan_within_budget, Some(false));
        assert_eq!(report.status, "MINIMUM_COST_DISAMBIGUATION_EXCEEDS_BUDGET");
    }

    #[test]
    fn scenario_order_and_action_order_do_not_change_canonical_digests() {
        let mut input = fixture();
        let scenario_hash = canonical_scenario_set_sha256(&input.scenarios);
        let action_hash = canonical_action_set_sha256(&input.actions);
        input.scenarios.reverse();
        input.actions.reverse();
        assert_eq!(
            scenario_hash,
            canonical_scenario_set_sha256(&input.scenarios)
        );
        assert_eq!(action_hash, canonical_action_set_sha256(&input.actions));
    }

    #[test]
    fn outcome_coverage_must_be_exact_for_every_scenario() {
        let mut input = fixture();
        input.actions[0].outcomes.pop();
        assert!(
            validate(&input)
                .unwrap_err()
                .contains("exactly one outcome")
        );
    }

    #[test]
    fn action_provenance_digests_are_required() {
        let mut input = fixture();
        input.actions[0].observation_model_sha256 = "0".into();
        assert!(validate(&input).unwrap_err().contains("digests"));
    }
}
