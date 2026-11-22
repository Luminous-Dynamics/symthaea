use super::Resources;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const EPSILON: f64 = 1.0e-10;
const CORPUS_SCHEMA: &str = "luminous.orbital-joint-evidence-voi.sequential.v0.1";
const CLAIM_CEILING: &str = "finite_horizon_sequential_campaign_semantics_only";
type LossMatrix = BTreeMap<String, BTreeMap<String, f64>>;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialCorpus {
    pub schema: String,
    pub benchmark_id: String,
    pub evidence_class: String,
    pub claim_ceiling: String,
    pub resource_frequency_semantics: String,
    pub policies_to_compare: Vec<String>,
    pub scenarios: Vec<SequentialScenario>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialScenario {
    pub scenario_id: String,
    pub evidence_class: String,
    pub claim_ceiling: String,
    pub decision_context: SequentialDecisionContext,
    pub states: Vec<String>,
    pub decisions: Vec<String>,
    pub decision_obligations: BTreeMap<String, Vec<String>>,
    pub loss_matrix: LossMatrix,
    pub resource_frequency_model: ResourceFrequencyModel,
    pub budget: Resources,
    pub max_actions: usize,
    pub fixed_campaign_sequence: Vec<String>,
    pub actions: Vec<SequentialAction>,
    pub expected: SequentialExpected,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SequentialDecisionContext {
    pub profile_ref: String,
    pub article_subject_ref: String,
    pub configuration_id: String,
    pub icd_revision: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceFrequencyModel {
    pub model_ref: String,
    pub calibration_status: String,
    pub used_for: String,
    pub state_weights: BTreeMap<String, f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialAction {
    pub action_id: String,
    pub action_class: String,
    pub decision_context: SequentialDecisionContext,
    pub method_ref: String,
    pub admissibility: String,
    pub resource_plan: Resources,
    pub observation_by_state: BTreeMap<String, String>,
    pub obligations_by_observation: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExpectedResourceUse {
    pub engineering_hours: f64,
    pub compute_core_hours: f64,
    pub facility_hours: f64,
    pub specimen_count: f64,
    pub review_hours: f64,
    pub external_cost_units: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialExpected {
    pub baseline_worst_case_regret: f64,
    pub myopic_worst_case_regret: f64,
    pub myopic_stops_without_action: bool,
    pub fixed_campaign_worst_case_regret: f64,
    pub adaptive_worst_case_regret: f64,
    pub adaptive_first_action: String,
    pub adaptive_expected_resources: ExpectedResourceUse,
    pub fixed_expected_resources: ExpectedResourceUse,
    pub require_adaptive_resource_pareto_dominates_fixed: bool,
    pub notes: String,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
#[serde(tag = "node_kind")]
pub enum PolicyNode {
    Terminal {
        belief_states: Vec<String>,
        decision: String,
        worst_case_regret: f64,
        covered_obligations: Vec<String>,
    },
    Test {
        action_id: String,
        branches: BTreeMap<String, PolicyNode>,
    },
    HumanReview {
        candidate_first_actions: Vec<String>,
        worst_case_regret: f64,
    },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MyopicAudit {
    pub selected_action_id: Option<String>,
    pub worst_case_regret: f64,
    pub stops_without_action: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FixedCampaignAudit {
    pub sequence: Vec<String>,
    pub worst_case_regret: f64,
    pub expected_resources: ExpectedResourceUse,
    pub worst_path_resources: Resources,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AdaptiveCampaignAudit {
    pub first_action_id: Option<String>,
    pub worst_case_regret: f64,
    pub expected_resources: Option<ExpectedResourceUse>,
    pub worst_path_resources: Resources,
    pub policy: PolicyNode,
    pub human_review_required: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SequentialScenarioAudit {
    pub scenario_id: String,
    pub search_states_expanded: usize,
    pub search_memo_hits: usize,
    pub search_memo_entries: usize,
    pub policy_tree_nodes: usize,
    pub baseline_terminal_disposition: String,
    pub baseline_worst_case_regret: f64,
    pub myopic: MyopicAudit,
    pub fixed_campaign: FixedCampaignAudit,
    pub adaptive_campaign: AdaptiveCampaignAudit,
    pub expected_resource_savings: Option<ExpectedResourceUse>,
    pub oracle_matched: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SequentialReceipt {
    pub schema: String,
    pub benchmark_id: String,
    pub input_sha256: String,
    pub source_commit: String,
    pub evidence_class: String,
    pub claim_ceiling: String,
    pub authority_granted: bool,
    pub scenario_count: usize,
    pub oracle_matches: usize,
    pub oracle_mismatches: usize,
    pub scenarios: Vec<SequentialScenarioAudit>,
}

#[derive(Debug, Clone)]
struct Plan {
    worst_case_regret: f64,
    expected_resources: Option<ExpectedResourceUse>,
    worst_path_resources: Resources,
    node: PolicyNode,
}

#[derive(Debug, Default, Clone)]
struct SearchStats {
    states_expanded: usize,
    memo_hits: usize,
}

type MemoKey = (Vec<String>, [u64; 6], usize, Vec<String>, Vec<String>);

fn memo_key(
    belief: &[String],
    remaining_budget: &Resources,
    depth_remaining: usize,
    used: &BTreeSet<String>,
    covered_obligations: &BTreeSet<String>,
) -> MemoKey {
    let mut canonical_belief = belief.to_vec();
    canonical_belief.sort();
    canonical_belief.dedup();
    (
        canonical_belief,
        [
            remaining_budget.engineering_hours.to_bits(),
            remaining_budget.compute_core_hours.to_bits(),
            remaining_budget.facility_hours.to_bits(),
            remaining_budget.specimen_count as u64,
            remaining_budget.review_hours.to_bits(),
            remaining_budget.external_cost_units.to_bits(),
        ],
        depth_remaining,
        used.iter().cloned().collect(),
        covered_obligations.iter().cloned().collect(),
    )
}

fn policy_tree_nodes(node: &PolicyNode) -> usize {
    match node {
        PolicyNode::Terminal { .. } | PolicyNode::HumanReview { .. } => 1,
        PolicyNode::Test { branches, .. } => {
            1 + branches.values().map(policy_tree_nodes).sum::<usize>()
        }
    }
}

fn valid_commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn valid_resources(resources: &Resources) -> bool {
    [
        resources.engineering_hours,
        resources.compute_core_hours,
        resources.facility_hours,
        resources.review_hours,
        resources.external_cost_units,
    ]
    .iter()
    .all(|value| value.is_finite() && *value >= 0.0)
}

fn fits_budget(cost: &Resources, budget: &Resources) -> bool {
    cost.engineering_hours <= budget.engineering_hours + EPSILON
        && cost.compute_core_hours <= budget.compute_core_hours + EPSILON
        && cost.facility_hours <= budget.facility_hours + EPSILON
        && cost.specimen_count <= budget.specimen_count
        && cost.review_hours <= budget.review_hours + EPSILON
        && cost.external_cost_units <= budget.external_cost_units + EPSILON
}

fn add_resources(left: &Resources, right: &Resources) -> Result<Resources, String> {
    let sum = Resources {
        engineering_hours: left.engineering_hours + right.engineering_hours,
        compute_core_hours: left.compute_core_hours + right.compute_core_hours,
        facility_hours: left.facility_hours + right.facility_hours,
        specimen_count: left
            .specimen_count
            .checked_add(right.specimen_count)
            .ok_or_else(|| "sequential specimen count overflow".to_owned())?,
        review_hours: left.review_hours + right.review_hours,
        external_cost_units: left.external_cost_units + right.external_cost_units,
    };
    if !valid_resources(&sum) {
        return Err("sequential resource total is non-finite".to_owned());
    }
    Ok(sum)
}

fn max_resources(left: &Resources, right: &Resources) -> Resources {
    Resources {
        engineering_hours: left.engineering_hours.max(right.engineering_hours),
        compute_core_hours: left.compute_core_hours.max(right.compute_core_hours),
        facility_hours: left.facility_hours.max(right.facility_hours),
        specimen_count: left.specimen_count.max(right.specimen_count),
        review_hours: left.review_hours.max(right.review_hours),
        external_cost_units: left.external_cost_units.max(right.external_cost_units),
    }
}

fn zero_resources() -> Resources {
    Resources {
        engineering_hours: 0.0,
        compute_core_hours: 0.0,
        facility_hours: 0.0,
        specimen_count: 0,
        review_hours: 0.0,
        external_cost_units: 0.0,
    }
}

fn expected_from_resources(resources: &Resources) -> ExpectedResourceUse {
    ExpectedResourceUse {
        engineering_hours: resources.engineering_hours,
        compute_core_hours: resources.compute_core_hours,
        facility_hours: resources.facility_hours,
        specimen_count: resources.specimen_count as f64,
        review_hours: resources.review_hours,
        external_cost_units: resources.external_cost_units,
    }
}

fn expected_add(left: &ExpectedResourceUse, right: &ExpectedResourceUse) -> ExpectedResourceUse {
    ExpectedResourceUse {
        engineering_hours: left.engineering_hours + right.engineering_hours,
        compute_core_hours: left.compute_core_hours + right.compute_core_hours,
        facility_hours: left.facility_hours + right.facility_hours,
        specimen_count: left.specimen_count + right.specimen_count,
        review_hours: left.review_hours + right.review_hours,
        external_cost_units: left.external_cost_units + right.external_cost_units,
    }
}

fn expected_scale(value: &ExpectedResourceUse, factor: f64) -> ExpectedResourceUse {
    ExpectedResourceUse {
        engineering_hours: value.engineering_hours * factor,
        compute_core_hours: value.compute_core_hours * factor,
        facility_hours: value.facility_hours * factor,
        specimen_count: value.specimen_count * factor,
        review_hours: value.review_hours * factor,
        external_cost_units: value.external_cost_units * factor,
    }
}

fn expected_dominates(left: &ExpectedResourceUse, right: &ExpectedResourceUse) -> bool {
    let no_more = left.engineering_hours <= right.engineering_hours + EPSILON
        && left.compute_core_hours <= right.compute_core_hours + EPSILON
        && left.facility_hours <= right.facility_hours + EPSILON
        && left.specimen_count <= right.specimen_count + EPSILON
        && left.review_hours <= right.review_hours + EPSILON
        && left.external_cost_units <= right.external_cost_units + EPSILON;
    let strictly_less = left.engineering_hours < right.engineering_hours - EPSILON
        || left.compute_core_hours < right.compute_core_hours - EPSILON
        || left.facility_hours < right.facility_hours - EPSILON
        || left.specimen_count < right.specimen_count - EPSILON
        || left.review_hours < right.review_hours - EPSILON
        || left.external_cost_units < right.external_cost_units - EPSILON;
    no_more && strictly_less
}

fn expected_close(left: &ExpectedResourceUse, right: &ExpectedResourceUse) -> bool {
    [
        (left.engineering_hours, right.engineering_hours),
        (left.compute_core_hours, right.compute_core_hours),
        (left.facility_hours, right.facility_hours),
        (left.specimen_count, right.specimen_count),
        (left.review_hours, right.review_hours),
        (left.external_cost_units, right.external_cost_units),
    ]
    .iter()
    .all(|(a, b)| (a - b).abs() <= 1.0e-9)
}

fn regret(scenario: &SequentialScenario, decision: &str, state: &str) -> Result<f64, String> {
    let actual = *scenario
        .loss_matrix
        .get(decision)
        .and_then(|row| row.get(state))
        .ok_or_else(|| format!("{} missing loss {decision}/{state}", scenario.scenario_id))?;
    let best = scenario
        .decisions
        .iter()
        .map(|candidate| {
            scenario
                .loss_matrix
                .get(candidate)
                .and_then(|row| row.get(state))
                .copied()
                .ok_or_else(|| format!("{} missing loss {candidate}/{state}", scenario.scenario_id))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    Ok((actual - best).max(0.0))
}

fn terminal_plan(
    scenario: &SequentialScenario,
    belief: &[String],
    covered_obligations: &BTreeSet<String>,
) -> Result<Plan, String> {
    let mut candidates = Vec::new();
    for decision in &scenario.decisions {
        let required = scenario.decision_obligations.get(decision).ok_or_else(|| {
            format!(
                "{} decision {decision} lacks obligation contract",
                scenario.scenario_id
            )
        })?;
        if !required
            .iter()
            .all(|obligation| covered_obligations.contains(obligation))
        {
            continue;
        }
        let mut worst = 0.0_f64;
        for state in belief {
            worst = worst.max(regret(scenario, decision, state)?);
        }
        candidates.push((decision.clone(), worst));
    }
    candidates.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let (decision, risk) = candidates
        .into_iter()
        .next()
        .ok_or_else(|| "no terminal decision is supported by covered obligations".to_owned())?;
    Ok(Plan {
        worst_case_regret: risk,
        expected_resources: Some(expected_from_resources(&zero_resources())),
        worst_path_resources: zero_resources(),
        node: PolicyNode::Terminal {
            belief_states: belief.to_vec(),
            decision,
            worst_case_regret: risk,
            covered_obligations: covered_obligations.iter().cloned().collect(),
        },
    })
}

fn eligible_action(
    scenario: &SequentialScenario,
    action: &SequentialAction,
    budget: &Resources,
    used: &BTreeSet<String>,
) -> bool {
    action.admissibility == "ADMISSIBLE"
        && action.decision_context == scenario.decision_context
        && !used.contains(&action.action_id)
        && valid_resources(&action.resource_plan)
        && fits_budget(&action.resource_plan, budget)
}

fn action_partitions(
    belief: &[String],
    action: &SequentialAction,
) -> BTreeMap<String, Vec<String>> {
    let mut branches = BTreeMap::<String, Vec<String>>::new();
    for state in belief {
        if let Some(observation) = action.observation_by_state.get(state) {
            branches
                .entry(observation.clone())
                .or_default()
                .push(state.clone());
        }
    }
    branches
}

fn conditional_branch_probability(
    scenario: &SequentialScenario,
    belief: &[String],
    branch: &[String],
) -> Result<f64, String> {
    let total: f64 = belief
        .iter()
        .map(|state| scenario.resource_frequency_model.state_weights[state])
        .sum();
    let branch_total: f64 = branch
        .iter()
        .map(|state| scenario.resource_frequency_model.state_weights[state])
        .sum();
    if total <= 0.0 || !total.is_finite() || !branch_total.is_finite() {
        return Err("invalid conditional resource-frequency mass".to_owned());
    }
    Ok(branch_total / total)
}

fn select_plan(mut candidates: Vec<Plan>) -> Result<Plan, String> {
    let best_risk = candidates
        .iter()
        .map(|plan| plan.worst_case_regret)
        .reduce(f64::min)
        .ok_or_else(|| "no sequential plans to compare".to_owned())?;
    let top_indices: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, plan)| (plan.worst_case_regret - best_risk).abs() <= EPSILON)
        .map(|(index, _)| index)
        .collect();
    // An unresolved child policy has no defensible expected-resource vector.
    // It must remain on the review frontier; never silently drop it and select
    // another tied plan merely because that other plan has a known cost.
    let unresolved_cost = top_indices
        .iter()
        .any(|index| candidates[*index].expected_resources.is_none());
    let frontier: Vec<usize> = if unresolved_cost {
        Vec::new()
    } else {
        top_indices
            .iter()
            .copied()
            .filter(|candidate_index| {
                let candidate_cost = candidates[*candidate_index]
                    .expected_resources
                    .as_ref()
                    .expect("top-plan resource completeness checked");
                !top_indices.iter().any(|other_index| {
                    other_index != candidate_index
                        && candidates[*other_index]
                            .expected_resources
                            .as_ref()
                            .is_some_and(|other_cost| {
                                expected_dominates(other_cost, candidate_cost)
                            })
                })
            })
            .collect()
    };
    if !unresolved_cost && frontier.len() == 1 {
        return Ok(candidates.swap_remove(frontier[0]));
    }
    let review_indices: Vec<usize> = if unresolved_cost {
        top_indices
    } else {
        frontier
    };
    let mut candidate_first_actions = Vec::new();
    for index in &review_indices {
        match &candidates[*index].node {
            PolicyNode::Test { action_id, .. } => candidate_first_actions.push(action_id.clone()),
            PolicyNode::HumanReview {
                candidate_first_actions: nested,
                ..
            } => candidate_first_actions.extend(nested.iter().cloned()),
            PolicyNode::Terminal { .. } => {}
        }
    }
    candidate_first_actions.sort();
    candidate_first_actions.dedup();
    let worst_path_resources = review_indices
        .iter()
        .map(|index| candidates[*index].worst_path_resources.clone())
        .reduce(|left, right| max_resources(&left, &right))
        .unwrap_or_else(zero_resources);
    Ok(Plan {
        worst_case_regret: best_risk,
        expected_resources: None,
        worst_path_resources,
        node: PolicyNode::HumanReview {
            candidate_first_actions,
            worst_case_regret: best_risk,
        },
    })
}

fn solve(
    scenario: &SequentialScenario,
    belief: &[String],
    remaining_budget: &Resources,
    depth_remaining: usize,
    used: &BTreeSet<String>,
    covered_obligations: &BTreeSet<String>,
    memo: &mut BTreeMap<MemoKey, Plan>,
    stats: &mut SearchStats,
) -> Result<Plan, String> {
    let key = memo_key(
        belief,
        remaining_budget,
        depth_remaining,
        used,
        covered_obligations,
    );
    if let Some(cached) = memo.get(&key) {
        stats.memo_hits += 1;
        return Ok(cached.clone());
    }
    stats.states_expanded += 1;

    let mut candidates = vec![terminal_plan(scenario, belief, covered_obligations)?];
    if depth_remaining == 0 {
        let result = candidates.remove(0);
        memo.insert(key, result.clone());
        return Ok(result);
    }
    for action in &scenario.actions {
        if !eligible_action(scenario, action, remaining_budget, used) {
            continue;
        }
        let partitions = action_partitions(belief, action);
        if partitions.len() < 2 {
            continue;
        }
        let mut branch_nodes = BTreeMap::new();
        let mut worst_case_regret = 0.0_f64;
        let mut expected_future = ExpectedResourceUse {
            engineering_hours: 0.0,
            compute_core_hours: 0.0,
            facility_hours: 0.0,
            specimen_count: 0.0,
            review_hours: 0.0,
            external_cost_units: 0.0,
        };
        let mut worst_future_resources = zero_resources();
        let mut has_future_expected_cost = true;
        for (observation, branch_belief) in &partitions {
            let mut next_used = used.clone();
            next_used.insert(action.action_id.clone());
            let mut child_budget = remaining_budget.clone();
            child_budget.engineering_hours -= action.resource_plan.engineering_hours;
            child_budget.compute_core_hours -= action.resource_plan.compute_core_hours;
            child_budget.facility_hours -= action.resource_plan.facility_hours;
            child_budget.specimen_count -= action.resource_plan.specimen_count;
            child_budget.review_hours -= action.resource_plan.review_hours;
            child_budget.external_cost_units -= action.resource_plan.external_cost_units;
            let mut next_covered = covered_obligations.clone();
            let outcome_obligations = action
                .obligations_by_observation
                .get(observation)
                .ok_or_else(|| {
                    format!(
                        "{} has no obligation mapping for {observation}",
                        action.action_id
                    )
                })?;
            next_covered.extend(outcome_obligations.iter().cloned());
            let child = solve(
                scenario,
                branch_belief,
                &child_budget,
                depth_remaining - 1,
                &next_used,
                &next_covered,
                memo,
                stats,
            )?;
            worst_case_regret = worst_case_regret.max(child.worst_case_regret);
            if let Some(child_expected) = &child.expected_resources {
                let probability = conditional_branch_probability(scenario, belief, branch_belief)?;
                expected_future = expected_add(
                    &expected_future,
                    &expected_scale(child_expected, probability),
                );
            } else {
                has_future_expected_cost = false;
            }
            worst_future_resources =
                max_resources(&worst_future_resources, &child.worst_path_resources);
            branch_nodes.insert(observation.clone(), child.node);
        }
        let current_expected = expected_from_resources(&action.resource_plan);
        let expected_resources =
            has_future_expected_cost.then(|| expected_add(&current_expected, &expected_future));
        let worst_path_resources = add_resources(&action.resource_plan, &worst_future_resources)?;
        if !fits_budget(&worst_path_resources, &scenario.budget) {
            continue;
        }
        candidates.push(Plan {
            worst_case_regret,
            expected_resources,
            worst_path_resources,
            node: PolicyNode::Test {
                action_id: action.action_id.clone(),
                branches: branch_nodes,
            },
        });
    }
    let result = select_plan(candidates)?;
    memo.insert(key, result.clone());
    Ok(result)
}

fn myopic_policy(scenario: &SequentialScenario, baseline: f64) -> Result<MyopicAudit, String> {
    let mut scored = Vec::new();
    for action in &scenario.actions {
        if !eligible_action(scenario, action, &scenario.budget, &BTreeSet::new()) {
            continue;
        }
        let partitions = action_partitions(&scenario.states, action);
        if partitions.len() < 2 {
            continue;
        }
        let mut worst = 0.0_f64;
        for (observation, branch) in &partitions {
            let covered: BTreeSet<String> = action
                .obligations_by_observation
                .get(observation)
                .ok_or_else(|| {
                    format!(
                        "{} has no obligation mapping for {observation}",
                        action.action_id
                    )
                })?
                .iter()
                .cloned()
                .collect();
            worst = worst.max(terminal_plan(scenario, branch, &covered)?.worst_case_regret);
        }
        scored.push((action.action_id.clone(), worst));
    }
    scored.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    if let Some((action_id, risk)) = scored.first()
        && risk + EPSILON < baseline
    {
        return Ok(MyopicAudit {
            selected_action_id: Some(action_id.clone()),
            worst_case_regret: *risk,
            stops_without_action: false,
        });
    }
    Ok(MyopicAudit {
        selected_action_id: None,
        worst_case_regret: baseline,
        stops_without_action: true,
    })
}

fn fixed_policy(scenario: &SequentialScenario) -> Result<FixedCampaignAudit, String> {
    let actions_by_id: BTreeMap<&str, &SequentialAction> = scenario
        .actions
        .iter()
        .map(|action| (action.action_id.as_str(), action))
        .collect();
    let mut resources = zero_resources();
    for action_id in &scenario.fixed_campaign_sequence {
        let action = actions_by_id
            .get(action_id.as_str())
            .ok_or_else(|| format!("fixed sequence references missing action {action_id}"))?;
        if !eligible_action(scenario, action, &scenario.budget, &BTreeSet::new()) {
            return Err(format!("fixed sequence action {action_id} is inadmissible"));
        }
        resources = add_resources(&resources, &action.resource_plan)?;
    }
    if !fits_budget(&resources, &scenario.budget) {
        return Err("fixed campaign exceeds total resource budget".to_owned());
    }
    let mut groups = BTreeMap::<Vec<String>, (Vec<String>, BTreeSet<String>)>::new();
    for state in &scenario.states {
        let mut signature = Vec::new();
        let mut covered_obligations = BTreeSet::new();
        for action_id in &scenario.fixed_campaign_sequence {
            let action = actions_by_id[action_id.as_str()];
            let observation = action.observation_by_state[state].clone();
            let produced = action
                .obligations_by_observation
                .get(&observation)
                .ok_or_else(|| format!("{action_id} lacks obligation mapping for {observation}"))?;
            signature.push(observation);
            covered_obligations.extend(produced.iter().cloned());
        }
        let entry = groups
            .entry(signature)
            .or_insert_with(|| (Vec::new(), covered_obligations.clone()));
        if entry.1 != covered_obligations {
            return Err(
                "identical fixed-campaign outcome signature has inconsistent evidence".to_owned(),
            );
        }
        entry.0.push(state.clone());
    }
    let mut worst_case_regret = 0.0_f64;
    for (belief, covered_obligations) in groups.values() {
        worst_case_regret = worst_case_regret
            .max(terminal_plan(scenario, belief, covered_obligations)?.worst_case_regret);
    }
    Ok(FixedCampaignAudit {
        sequence: scenario.fixed_campaign_sequence.clone(),
        worst_case_regret,
        expected_resources: expected_from_resources(&resources),
        worst_path_resources: resources,
    })
}

fn validate_scenario(scenario: &SequentialScenario) -> Result<SequentialScenarioAudit, String> {
    if scenario.evidence_class != "synthetic" || scenario.claim_ceiling != CLAIM_CEILING {
        return Err(format!(
            "{} has unsupported evidence or claim ceiling",
            scenario.scenario_id
        ));
    }
    let context = &scenario.decision_context;
    if [
        &context.profile_ref,
        &context.article_subject_ref,
        &context.configuration_id,
        &context.icd_revision,
    ]
    .iter()
    .any(|value| value.trim().is_empty())
    {
        return Err(format!(
            "{} has incomplete decision context",
            scenario.scenario_id
        ));
    }
    if scenario.states.len() < 2
        || scenario.states.iter().collect::<BTreeSet<_>>().len() != scenario.states.len()
        || scenario.decisions.len() < 2
        || scenario.max_actions == 0
        || scenario.actions.is_empty()
        || !valid_resources(&scenario.budget)
    {
        return Err(format!(
            "{} has invalid basic scenario shape",
            scenario.scenario_id
        ));
    }
    let state_set: BTreeSet<&str> = scenario.states.iter().map(String::as_str).collect();
    let decision_set: BTreeSet<&str> = scenario.decisions.iter().map(String::as_str).collect();
    if !["ACCEPT", "REJECT", "HOLD"]
        .iter()
        .all(|decision| decision_set.contains(decision))
    {
        return Err(format!(
            "{} must declare ACCEPT, REJECT and HOLD dispositions",
            scenario.scenario_id
        ));
    }
    if decision_set.len() != scenario.decisions.len() {
        return Err(format!(
            "{} has duplicate decision labels",
            scenario.scenario_id
        ));
    }
    if scenario.loss_matrix.len() != decision_set.len()
        || scenario.decision_obligations.len() != decision_set.len()
        || scenario
            .decision_obligations
            .keys()
            .any(|decision| !decision_set.contains(decision.as_str()))
    {
        return Err(format!(
            "{} loss matrix has wrong decision count",
            scenario.scenario_id
        ));
    }
    for (decision, row) in &scenario.loss_matrix {
        if !decision_set.contains(decision.as_str())
            || row.len() != state_set.len()
            || row.keys().any(|state| !state_set.contains(state.as_str()))
            || row.values().any(|loss| !loss.is_finite() || *loss < 0.0)
        {
            return Err(format!(
                "{} loss row {decision} is incomplete or invalid",
                scenario.scenario_id
            ));
        }
    }
    let frequency = &scenario.resource_frequency_model;
    if frequency.model_ref.trim().is_empty()
        || frequency.calibration_status != "calibrated_for_fixture_only"
        || frequency.used_for != "EXPECTED_RESOURCE_USE_ONLY"
        || frequency.state_weights.len() != state_set.len()
        || frequency
            .state_weights
            .keys()
            .any(|state| !state_set.contains(state.as_str()))
        || frequency
            .state_weights
            .values()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
        || (frequency.state_weights.values().sum::<f64>() - 1.0).abs() > 1.0e-9
    {
        return Err(format!(
            "{} has invalid resource-frequency model",
            scenario.scenario_id
        ));
    }
    let declared_obligations: BTreeSet<&str> = scenario
        .decision_obligations
        .values()
        .flatten()
        .map(String::as_str)
        .collect();
    if scenario
        .decision_obligations
        .iter()
        .any(|(decision, obligations)| {
            obligations.iter().any(|value| value.trim().is_empty())
                || obligations.iter().collect::<BTreeSet<_>>().len() != obligations.len()
                || (matches!(decision.as_str(), "ACCEPT" | "REJECT") && obligations.is_empty())
        })
    {
        return Err(format!(
            "{} has an empty or duplicate decision obligation",
            scenario.scenario_id
        ));
    }
    let mut action_ids = BTreeSet::new();
    for action in &scenario.actions {
        let possible_outcomes: BTreeSet<&str> = action
            .observation_by_state
            .values()
            .map(String::as_str)
            .collect();
        if action.obligations_by_observation.len() != possible_outcomes.len()
            || action
                .obligations_by_observation
                .keys()
                .any(|outcome| !possible_outcomes.contains(outcome.as_str()))
            || action
                .obligations_by_observation
                .values()
                .any(|obligations| {
                    obligations.iter().collect::<BTreeSet<_>>().len() != obligations.len()
                        || obligations
                            .iter()
                            .any(|value| !declared_obligations.contains(value.as_str()))
                })
        {
            return Err(format!(
                "{} has malformed outcome-specific obligation map",
                action.action_id
            ));
        }
        if action.action_id.trim().is_empty()
            || action.action_class.trim().is_empty()
            || action.method_ref.trim().is_empty()
            || !action_ids.insert(action.action_id.as_str())
            || action.admissibility != "ADMISSIBLE" && action.admissibility != "INADMISSIBLE"
            || !valid_resources(&action.resource_plan)
            || action.observation_by_state.len() != state_set.len()
            || action
                .observation_by_state
                .keys()
                .any(|state| !state_set.contains(state.as_str()))
            || scenario
                .states
                .iter()
                .any(|state| !action.observation_by_state.contains_key(state))
        {
            return Err(format!("{} has malformed action", scenario.scenario_id));
        }
    }
    if scenario.fixed_campaign_sequence.is_empty()
        || scenario
            .fixed_campaign_sequence
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != scenario.fixed_campaign_sequence.len()
        || scenario
            .fixed_campaign_sequence
            .iter()
            .any(|action_id| !action_ids.contains(action_id.as_str()))
    {
        return Err(format!(
            "{} has invalid fixed campaign sequence",
            scenario.scenario_id
        ));
    }

    let no_obligations = BTreeSet::new();
    let baseline_plan = terminal_plan(scenario, &scenario.states, &no_obligations)?;
    let baseline_terminal_disposition = match &baseline_plan.node {
        PolicyNode::Terminal { decision, .. } => decision.clone(),
        _ => return Err("baseline policy must terminate immediately".to_owned()),
    };
    if baseline_terminal_disposition != "HOLD" {
        return Err(format!(
            "{} no-evidence baseline must be HOLD, not {baseline_terminal_disposition}",
            scenario.scenario_id
        ));
    }
    let baseline = baseline_plan.worst_case_regret;
    let myopic = myopic_policy(scenario, baseline)?;
    let fixed = fixed_policy(scenario)?;
    let mut memo = BTreeMap::new();
    let mut search_stats = SearchStats::default();
    let adaptive = solve(
        scenario,
        &scenario.states,
        &scenario.budget,
        scenario.max_actions,
        &BTreeSet::new(),
        &no_obligations,
        &mut memo,
        &mut search_stats,
    )?;
    let tree_nodes = policy_tree_nodes(&adaptive.node);
    let first_action_id = match &adaptive.node {
        PolicyNode::Test { action_id, .. } => Some(action_id.clone()),
        _ => None,
    };
    let human_review_required = matches!(&adaptive.node, PolicyNode::HumanReview { .. });
    let resource_savings = if let Some(expected) = &adaptive.expected_resources {
        let fixed_use = &fixed.expected_resources;
        Some(ExpectedResourceUse {
            engineering_hours: fixed_use.engineering_hours - expected.engineering_hours,
            compute_core_hours: fixed_use.compute_core_hours - expected.compute_core_hours,
            facility_hours: fixed_use.facility_hours - expected.facility_hours,
            specimen_count: fixed_use.specimen_count - expected.specimen_count,
            review_hours: fixed_use.review_hours - expected.review_hours,
            external_cost_units: fixed_use.external_cost_units - expected.external_cost_units,
        })
    } else {
        None
    };
    let expected = &scenario.expected;
    let mut oracle_matched = (baseline - expected.baseline_worst_case_regret).abs() <= 1.0e-9
        && (myopic.worst_case_regret - expected.myopic_worst_case_regret).abs() <= 1.0e-9
        && myopic.stops_without_action == expected.myopic_stops_without_action
        && (fixed.worst_case_regret - expected.fixed_campaign_worst_case_regret).abs() <= 1.0e-9
        && (adaptive.worst_case_regret - expected.adaptive_worst_case_regret).abs() <= 1.0e-9
        && first_action_id.as_deref() == Some(expected.adaptive_first_action.as_str());
    if let Some(actual) = &adaptive.expected_resources {
        oracle_matched &= expected_close(actual, &expected.adaptive_expected_resources);
    } else {
        oracle_matched = false;
    }
    oracle_matched &= expected_close(
        &fixed.expected_resources,
        &expected.fixed_expected_resources,
    );
    if expected.require_adaptive_resource_pareto_dominates_fixed {
        oracle_matched &= adaptive
            .expected_resources
            .as_ref()
            .is_some_and(|adaptive_use| {
                expected_dominates(adaptive_use, &fixed.expected_resources)
            });
    }

    if !oracle_matched {
        return Err(format!(
            "{} sequential policy result differs from known-answer oracle",
            scenario.scenario_id
        ));
    }
    if !fits_budget(&adaptive.worst_path_resources, &scenario.budget) {
        return Err(format!(
            "{} adaptive worst path exceeds budget",
            scenario.scenario_id
        ));
    }
    Ok(SequentialScenarioAudit {
        scenario_id: scenario.scenario_id.clone(),
        search_states_expanded: search_stats.states_expanded,
        search_memo_hits: search_stats.memo_hits,
        search_memo_entries: memo.len(),
        policy_tree_nodes: tree_nodes,
        baseline_terminal_disposition,
        baseline_worst_case_regret: baseline,
        myopic,
        fixed_campaign: fixed,
        adaptive_campaign: AdaptiveCampaignAudit {
            first_action_id,
            worst_case_regret: adaptive.worst_case_regret,
            expected_resources: adaptive.expected_resources,
            worst_path_resources: adaptive.worst_path_resources,
            policy: adaptive.node,
            human_review_required,
        },
        expected_resource_savings: resource_savings,
        oracle_matched,
    })
}

pub fn run_sequential_audit(
    input: &[u8],
    source_commit: &str,
) -> Result<SequentialReceipt, String> {
    if !valid_commit(source_commit) {
        return Err("source commit must be exactly 40 lowercase hexadecimal characters".to_owned());
    }
    let corpus: SequentialCorpus = serde_json::from_slice(input)
        .map_err(|error| format!("invalid sequential campaign corpus: {error}"))?;
    if corpus.schema != CORPUS_SCHEMA
        || corpus.benchmark_id != "ORBITAL-JOINT-SEQUENTIAL-PLANNING-001"
        || corpus.evidence_class != "synthetic"
        || corpus.claim_ceiling != CLAIM_CEILING
        || corpus.resource_frequency_semantics.trim().is_empty()
        || corpus.policies_to_compare.is_empty()
        || corpus.scenarios.is_empty()
    {
        return Err("unsupported sequential corpus or claim boundary".to_owned());
    }
    let mut ids = BTreeSet::new();
    let mut scenarios = Vec::new();
    for scenario in &corpus.scenarios {
        if !ids.insert(scenario.scenario_id.as_str()) {
            return Err(format!(
                "duplicate sequential scenario {}",
                scenario.scenario_id
            ));
        }
        scenarios.push(validate_scenario(scenario)?);
    }
    scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let oracle_matches = scenarios
        .iter()
        .filter(|scenario| scenario.oracle_matched)
        .count();
    let oracle_mismatches = scenarios.len() - oracle_matches;
    Ok(SequentialReceipt {
        schema: "luminous.orbital-joint-evidence-voi.sequential-receipt.v0.1".to_owned(),
        benchmark_id: corpus.benchmark_id,
        input_sha256: format!("{:x}", Sha256::digest(input)),
        source_commit: source_commit.to_owned(),
        evidence_class: "synthetic".to_owned(),
        claim_ceiling: CLAIM_CEILING.to_owned(),
        authority_granted: false,
        scenario_count: scenarios.len(),
        oracle_matches,
        oracle_mismatches,
        scenarios,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &[u8] = include_bytes!(
        "../../../fixtures/orbital-joint-evidence-voi/v0.1/sequential-campaign.synthetic.json"
    );
    const SOURCE: &str = "0123456789abcdef0123456789abcdef01234567";

    fn receipt() -> SequentialReceipt {
        run_sequential_audit(FIXTURE, SOURCE).expect("sequential campaign oracle must match")
    }

    #[test]
    fn unresolved_child_cost_keeps_parent_candidates_on_human_review_frontier() {
        let mut nested_branches = BTreeMap::new();
        nested_branches.insert(
            "PASS".to_owned(),
            PolicyNode::HumanReview {
                candidate_first_actions: vec![
                    "A-CHILD-LOW-ENGINEERING".to_owned(),
                    "A-CHILD-LOW-FACILITY".to_owned(),
                ],
                worst_case_regret: 0.0,
            },
        );
        let unresolved_parent = Plan {
            worst_case_regret: 0.0,
            expected_resources: None,
            worst_path_resources: Resources {
                engineering_hours: 2.0,
                compute_core_hours: 0.0,
                facility_hours: 2.0,
                specimen_count: 1,
                review_hours: 1.0,
                external_cost_units: 1.0,
            },
            node: PolicyNode::Test {
                action_id: "A-PARENT-WITH-UNRESOLVED-CHILD".to_owned(),
                branches: nested_branches,
            },
        };
        let known_parent = Plan {
            worst_case_regret: 0.0,
            expected_resources: Some(expected_from_resources(&Resources {
                engineering_hours: 0.5,
                compute_core_hours: 0.0,
                facility_hours: 0.5,
                specimen_count: 1,
                review_hours: 0.25,
                external_cost_units: 0.25,
            })),
            worst_path_resources: Resources {
                engineering_hours: 0.5,
                compute_core_hours: 0.0,
                facility_hours: 0.5,
                specimen_count: 1,
                review_hours: 0.25,
                external_cost_units: 0.25,
            },
            node: PolicyNode::Test {
                action_id: "B-KNOWN-COST-PARENT".to_owned(),
                branches: BTreeMap::new(),
            },
        };

        let selected = select_plan(vec![unresolved_parent, known_parent]).unwrap();
        let PolicyNode::HumanReview {
            candidate_first_actions,
            worst_case_regret,
        } = selected.node
        else {
            panic!("a known-cost plan must not displace a tied unresolved subtree");
        };
        assert_eq!(worst_case_regret, 0.0);
        assert_eq!(
            candidate_first_actions,
            vec![
                "A-PARENT-WITH-UNRESOLVED-CHILD".to_owned(),
                "B-KNOWN-COST-PARENT".to_owned(),
            ]
        );
        assert!(selected.expected_resources.is_none());
    }

    #[test]
    fn memoized_exact_state_replay_does_not_expand_the_search_twice() {
        let corpus: SequentialCorpus = serde_json::from_slice(FIXTURE).unwrap();
        let scenario = &corpus.scenarios[0];
        let mut memo = BTreeMap::new();
        let mut stats = SearchStats::default();
        let first = solve(
            scenario,
            &scenario.states,
            &scenario.budget,
            scenario.max_actions,
            &BTreeSet::new(),
            &BTreeSet::new(),
            &mut memo,
            &mut stats,
        )
        .unwrap();
        let expanded = stats.states_expanded;
        assert!(expanded > 0);
        assert_eq!(memo.len(), expanded);

        let second = solve(
            scenario,
            &scenario.states,
            &scenario.budget,
            scenario.max_actions,
            &BTreeSet::new(),
            &BTreeSet::new(),
            &mut memo,
            &mut stats,
        )
        .unwrap();
        assert_eq!(first.node, second.node);
        assert_eq!(first.worst_case_regret, second.worst_case_regret);
        assert_eq!(stats.states_expanded, expanded);
        assert_eq!(stats.memo_hits, 1);
    }

    #[test]
    fn exact_finite_horizon_policy_beats_myopic_stop_rule() {
        let result = receipt();
        let scenario = &result.scenarios[0];
        assert!(scenario.search_states_expanded > 0);
        assert_eq!(
            scenario.search_memo_entries,
            scenario.search_states_expanded
        );
        assert!(scenario.policy_tree_nodes > 0);
        assert_eq!(scenario.baseline_terminal_disposition, "HOLD");
        assert_eq!(scenario.baseline_worst_case_regret, 2.0);
        assert_eq!(scenario.myopic.worst_case_regret, 2.0);
        assert!(scenario.myopic.stops_without_action);
        assert_eq!(scenario.adaptive_campaign.worst_case_regret, 0.0);
    }

    #[test]
    fn adaptive_policy_uses_observation_contingent_stopping() {
        let result = receipt();
        let scenario = &result.scenarios[0];
        let PolicyNode::Test {
            action_id,
            branches,
        } = &scenario.adaptive_campaign.policy
        else {
            panic!("adaptive policy must start with a test");
        };
        assert_eq!(action_id, "A-COMBINED-LOAD-SCREEN");
        let Some(PolicyNode::Terminal {
            decision,
            covered_obligations,
            ..
        }) = branches.get("FAIL")
        else {
            panic!("FAIL observation must terminate with rejection basis");
        };
        assert_eq!(decision, "REJECT");
        assert_eq!(covered_obligations, &vec!["REJECTION-BASIS-001".to_owned()]);
        let Some(PolicyNode::Test {
            action_id,
            branches,
        }) = branches.get("PASS")
        else {
            panic!("PASS branch must run the targeted thermal inspection");
        };
        assert_eq!(action_id, "B-THERMAL-INSPECTION");
        assert!(matches!(
            branches.get("CLEAR"),
            Some(PolicyNode::Terminal { decision, .. }) if decision == "ACCEPT"
        ));
        let Some(PolicyNode::Terminal {
            decision,
            covered_obligations,
            ..
        }) = branches.get("INDICATION")
        else {
            panic!("INDICATION observation must terminate with rejection basis");
        };
        assert_eq!(decision, "REJECT");
        assert!(covered_obligations.contains(&"REJECTION-BASIS-001".to_owned()));
    }

    #[test]
    fn adaptive_and_fixed_campaigns_have_equal_worst_case_decision_quality() {
        let result = receipt();
        let scenario = &result.scenarios[0];
        assert_eq!(scenario.fixed_campaign.worst_case_regret, 0.0);
        assert_eq!(scenario.adaptive_campaign.worst_case_regret, 0.0);
    }

    #[test]
    fn contingent_campaign_reduces_every_expected_resource_dimension() {
        let result = receipt();
        let scenario = &result.scenarios[0];
        let adaptive = scenario
            .adaptive_campaign
            .expected_resources
            .as_ref()
            .unwrap();
        let fixed = &scenario.fixed_campaign.expected_resources;
        assert!(expected_dominates(adaptive, fixed));
        assert!((adaptive.engineering_hours - 1.2).abs() < 1.0e-9);
        assert!((adaptive.facility_hours - 1.55).abs() < 1.0e-9);
        assert!((adaptive.specimen_count - 1.7).abs() < 1.0e-9);
        let budget = Resources {
            engineering_hours: 2.0,
            compute_core_hours: 0.0,
            facility_hours: 2.5,
            specimen_count: 2,
            review_hours: 0.5,
            external_cost_units: 2.0,
        };
        assert!(fits_budget(
            &scenario.adaptive_campaign.worst_path_resources,
            &budget
        ));
    }

    #[test]
    fn resource_frequency_model_changes_cost_estimate_not_robust_risk() {
        let original = receipt();
        let mut value: Value = serde_json::from_slice(FIXTURE).unwrap();
        value["scenarios"][0]["resource_frequency_model"]["state_weights"] = serde_json::json!({
            "HEALTHY":0.5,
            "AXIAL_CAPACITY_DEFECT":0.2,
            "THERMAL_WEAR_DEFECT":0.1,
            "JOINT_RETENTION_DEFECT":0.2
        });
        value["scenarios"][0]["expected"]["adaptive_expected_resources"] = serde_json::json!({
            "engineering_hours":1.1,
            "compute_core_hours":0.0,
            "facility_hours":1.4,
            "specimen_count":1.6,
            "review_hours":0.22,
            "external_cost_units":0.78
        });
        let changed = serde_json::to_vec(&value).unwrap();
        let second = run_sequential_audit(&changed, SOURCE).unwrap();
        let original_scenario = &original.scenarios[0];
        let second_scenario = &second.scenarios[0];
        assert_eq!(original_scenario.adaptive_campaign.worst_case_regret, 0.0);
        assert_eq!(second_scenario.adaptive_campaign.worst_case_regret, 0.0);
        assert!(
            (second_scenario
                .adaptive_campaign
                .expected_resources
                .as_ref()
                .unwrap()
                .engineering_hours
                - 1.1)
                .abs()
                < 1.0e-9
        );
        assert!(
            (original_scenario
                .adaptive_campaign
                .expected_resources
                .as_ref()
                .unwrap()
                .engineering_hours
                - 1.2)
                .abs()
                < 1.0e-9
        );
    }

    #[test]
    fn frequency_profile_can_reverse_first_test_without_changing_robust_decision_risk() {
        let result = receipt();
        let seq_one = result
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "SEQ-001")
            .unwrap();
        let seq_two = result
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "SEQ-002")
            .unwrap();
        assert_eq!(
            seq_one.adaptive_campaign.first_action_id.as_deref(),
            Some("A-COMBINED-LOAD-SCREEN")
        );
        assert_eq!(
            seq_two.adaptive_campaign.first_action_id.as_deref(),
            Some("B-THERMAL-INSPECTION")
        );
        assert_eq!(seq_one.adaptive_campaign.worst_case_regret, 0.0);
        assert_eq!(seq_two.adaptive_campaign.worst_case_regret, 0.0);
        assert!(expected_dominates(
            seq_one
                .adaptive_campaign
                .expected_resources
                .as_ref()
                .unwrap(),
            &seq_one.fixed_campaign.expected_resources
        ));
        assert!(expected_dominates(
            seq_two
                .adaptive_campaign
                .expected_resources
                .as_ref()
                .unwrap(),
            &seq_two.fixed_campaign.expected_resources
        ));
    }

    #[test]
    fn wrong_article_or_configuration_context_is_not_selectable() {
        let mut value: Value = serde_json::from_slice(FIXTURE).unwrap();
        value["scenarios"][0]["actions"][0]["decision_context"]["article_subject_ref"] =
            Value::String("physical-unit://OTHER-SERIAL".to_owned());
        let changed = serde_json::to_vec(&value).unwrap();
        assert!(run_sequential_audit(&changed, SOURCE).is_err());
    }

    #[test]
    fn malformed_frequency_model_is_rejected() {
        let mut value: Value = serde_json::from_slice(FIXTURE).unwrap();
        value["scenarios"][0]["resource_frequency_model"]["used_for"] =
            Value::String("DECISION_RISK".to_owned());
        let changed = serde_json::to_vec(&value).unwrap();
        assert!(run_sequential_audit(&changed, SOURCE).is_err());
    }

    #[test]
    fn baseline_without_evidence_is_hold_not_reject_or_accept() {
        let corpus: SequentialCorpus = serde_json::from_slice(FIXTURE).unwrap();
        let scenario = &corpus.scenarios[0];
        let baseline = terminal_plan(scenario, &scenario.states, &BTreeSet::new()).unwrap();
        assert_eq!(baseline.worst_case_regret, 2.0);
        assert!(matches!(
            baseline.node,
            PolicyNode::Terminal { decision, .. } if decision == "HOLD"
        ));
    }

    #[test]
    fn accept_leaf_carries_every_required_path_obligation() {
        let result = receipt();
        let scenario = &result.scenarios[0];
        let PolicyNode::Test { branches, .. } = &scenario.adaptive_campaign.policy else {
            panic!("adaptive policy must start at the combined-load screen");
        };
        let Some(PolicyNode::Test { branches, .. }) = branches.get("PASS") else {
            panic!("PASS branch must execute targeted thermal evidence");
        };
        let Some(PolicyNode::Terminal {
            decision,
            covered_obligations,
            ..
        }) = branches.get("CLEAR")
        else {
            panic!("CLEAR branch must end in a terminal disposition");
        };
        assert_eq!(decision, "ACCEPT");
        assert_eq!(
            covered_obligations.iter().cloned().collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "COMBINED-LOAD-CAPABILITY-001".to_owned(),
                "THERMAL-WEAR-INSPECTABILITY-001".to_owned(),
            ])
        );
        let PolicyNode::Test { branches, .. } = &scenario.adaptive_campaign.policy else {
            unreachable!();
        };
        let Some(PolicyNode::Terminal {
            decision,
            covered_obligations,
            ..
        }) = branches.get("FAIL")
        else {
            panic!("FAIL branch must be a terminal rejection");
        };
        assert_eq!(decision, "REJECT");
        assert_eq!(covered_obligations, &vec!["REJECTION-BASIS-001".to_owned()]);
    }

    #[test]
    fn missing_thermal_coverage_cannot_produce_accept() {
        let mut value: Value = serde_json::from_slice(FIXTURE).unwrap();
        value["scenarios"][0]["actions"][1]["obligations_by_observation"]["CLEAR"] =
            serde_json::json!([]);
        let changed = serde_json::to_vec(&value).unwrap();
        assert!(run_sequential_audit(&changed, SOURCE).is_err());
    }

    #[test]
    fn unknown_action_obligation_is_rejected() {
        let mut value: Value = serde_json::from_slice(FIXTURE).unwrap();
        value["scenarios"][0]["actions"][0]["obligations_by_observation"]["PASS"] =
            serde_json::json!(["UNDECLARED-REQUIREMENT"]);
        let changed = serde_json::to_vec(&value).unwrap();
        assert!(run_sequential_audit(&changed, SOURCE).is_err());
    }

    #[test]
    fn action_order_does_not_change_contingent_policy() {
        let original = receipt();
        let mut value: Value = serde_json::from_slice(FIXTURE).unwrap();
        value["scenarios"][0]["actions"]
            .as_array_mut()
            .unwrap()
            .reverse();
        let changed = serde_json::to_vec(&value).unwrap();
        let second = run_sequential_audit(&changed, SOURCE).unwrap();
        assert_eq!(
            original.scenarios[0].adaptive_campaign.first_action_id,
            second.scenarios[0].adaptive_campaign.first_action_id
        );
        assert_eq!(
            original.scenarios[0].adaptive_campaign.policy,
            second.scenarios[0].adaptive_campaign.policy
        );
    }
}
