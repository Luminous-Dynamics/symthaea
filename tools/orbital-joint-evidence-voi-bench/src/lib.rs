use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CORPUS_SCHEMA: &str = "luminous.orbital-joint-evidence-voi.scenarios.v0.1";
pub const RECEIPT_SCHEMA: &str = "luminous.orbital-joint-evidence-voi.receipt.v0.1";
pub const CLAIM_CEILING: &str = "synthetic_decision_value_semantics_only";
const EPSILON: f64 = 1.0e-10;
type ObservationLikelihoods = BTreeMap<String, BTreeMap<String, f64>>;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Corpus {
    pub schema: String,
    pub benchmark_id: String,
    pub evidence_class: String,
    pub claim_ceiling: String,
    pub resource_dimensions: Vec<String>,
    pub loss_units: String,
    pub observation_semantics: String,
    pub policies_to_compare: Vec<String>,
    pub scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub scenario_id: String,
    pub claim_ceiling: String,
    pub evidence_class: String,
    pub decision_regime: String,
    pub target_claim_id: String,
    pub requires_independent_evidence: bool,
    pub decision_context: DecisionContext,
    pub states: Vec<String>,
    pub loss_matrix: BTreeMap<String, BTreeMap<String, f64>>,
    pub prior: Option<BTreeMap<String, f64>>,
    pub current_evidence_refs: Vec<String>,
    pub current_evidence_dependency_groups: BTreeMap<String, Vec<String>>,
    pub required_obligations: Vec<String>,
    pub budget: Resources,
    pub actions: Vec<Action>,
    pub expected: Expected,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionContext {
    pub profile_ref: String,
    pub article_subject_ref: String,
    pub configuration_id: String,
    pub icd_revision: String,
}

#[derive(Debug, Deserialize)]
pub struct Action {
    pub action_id: String,
    pub action_class: String,
    pub target_claim_id: String,
    pub profile_ref: String,
    pub article_subject_ref: String,
    pub configuration_id: String,
    pub icd_revision: String,
    pub method_ref: String,
    pub admissibility: Admissibility,
    pub resources: Resources,
    pub observation_by_state: BTreeMap<String, String>,
    pub dependencies: Vec<String>,
    #[serde(flatten)]
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Admissibility {
    pub state: String,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Resources {
    pub engineering_hours: f64,
    pub compute_core_hours: f64,
    pub facility_hours: f64,
    pub specimen_count: u64,
    pub review_hours: f64,
    pub external_cost_units: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expected {
    pub best_action_id: String,
    pub abstain: bool,
    pub must_not_select: Vec<String>,
    pub notes: String,
    pub baseline_minimax_regret: Option<f64>,
    pub post_action_worst_case_regret: Option<f64>,
    pub baseline_expected_loss: Option<f64>,
    pub post_action_expected_loss: Option<f64>,
    pub evsi: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ActionAudit {
    pub action_id: String,
    pub admissibility_state: String,
    pub planned_resources: Resources,
    pub actual_resources: Option<Resources>,
    pub execution_state: Option<String>,
    pub actual_exceeds_plan: bool,
    pub actual_exceeds_scenario_budget: bool,
    pub budget_fits: bool,
    pub target_claim_matches: bool,
    pub profile_matches: bool,
    pub article_subject_matches: bool,
    pub configuration_matches: bool,
    pub icd_revision_matches: bool,
    pub common_cause_conflict: bool,
    pub mandatory_obligations_covered: bool,
    pub decision_informative: bool,
    pub execution_outcome_usable: bool,
    pub policy_eligible: bool,
    pub decision_value_improvement: Option<f64>,
    pub post_action_regret_or_loss: Option<f64>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ScenarioAudit {
    pub scenario_id: String,
    pub decision_regime: String,
    pub baseline_regret_or_loss: f64,
    pub selected_action_id: Option<String>,
    pub selected_action_value: Option<f64>,
    pub selection_disposition: String,
    pub selection_matches_oracle: bool,
    pub expected_action_id: String,
    pub expected_action_eligible: bool,
    pub expected_action_value: Option<f64>,
    pub abstention_correct: bool,
    pub action_count: usize,
    pub eligible_action_count: usize,
    pub budget_blocked_count: usize,
    pub actual_resources_consumed: Resources,
    pub actual_resources_over_scenario_budget: bool,
    pub actual_over_plan_action_count: usize,
    pub actions_with_actual_resource_records: usize,
    pub oracle_matched: bool,
    pub actions: Vec<ActionAudit>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Receipt {
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
    pub admissible_action_count: usize,
    pub eligible_action_count: usize,
    pub budget_blocked_action_count: usize,
    pub scenarios: Vec<ScenarioAudit>,
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

fn dominates_resource_plan(left: &Resources, right: &Resources) -> bool {
    let no_more = left.engineering_hours <= right.engineering_hours + EPSILON
        && left.compute_core_hours <= right.compute_core_hours + EPSILON
        && left.facility_hours <= right.facility_hours + EPSILON
        && left.specimen_count <= right.specimen_count
        && left.review_hours <= right.review_hours + EPSILON
        && left.external_cost_units <= right.external_cost_units + EPSILON;
    let strictly_less = left.engineering_hours < right.engineering_hours - EPSILON
        || left.compute_core_hours < right.compute_core_hours - EPSILON
        || left.facility_hours < right.facility_hours - EPSILON
        || left.specimen_count < right.specimen_count
        || left.review_hours < right.review_hours - EPSILON
        || left.external_cost_units < right.external_cost_units - EPSILON;
    no_more && strictly_less
}

fn select_best_action(audits: &[ActionAudit]) -> (Option<String>, Option<f64>, &'static str) {
    let candidates: Vec<&ActionAudit> = audits
        .iter()
        .filter(|audit| {
            audit.policy_eligible
                && audit
                    .decision_value_improvement
                    .is_some_and(|value| value > EPSILON)
        })
        .collect();
    let Some(best_value) = candidates
        .iter()
        .filter_map(|audit| audit.decision_value_improvement)
        .reduce(f64::max)
    else {
        return (None, None, "ABSTAIN_NO_DECISION_VALUE");
    };
    let top_value: Vec<&ActionAudit> = candidates
        .into_iter()
        .filter(|audit| {
            audit
                .decision_value_improvement
                .is_some_and(|value| (value - best_value).abs() <= EPSILON)
        })
        .collect();
    let pareto_front: Vec<&ActionAudit> = top_value
        .iter()
        .copied()
        .filter(|candidate| {
            !top_value.iter().any(|other| {
                other.action_id != candidate.action_id
                    && dominates_resource_plan(
                        &other.planned_resources,
                        &candidate.planned_resources,
                    )
            })
        })
        .collect();
    if pareto_front.len() == 1 {
        let selected = pareto_front[0];
        return (
            Some(selected.action_id.clone()),
            selected.decision_value_improvement,
            "SELECTED_MAX_VALUE_UNIQUE_PARETO_PLAN",
        );
    }
    (
        None,
        Some(best_value),
        "HUMAN_REVIEW_REQUIRED_RESOURCE_PARETO_TIE",
    )
}

fn actual_resources(action: &Action) -> Result<Option<Resources>, String> {
    let observed_state = metadata_string(action, "observed_execution_state");
    let Some(value) = action.metadata.get("actual_resources") else {
        if observed_state.is_some() {
            return Err(format!(
                "{} has an execution state without an actual resource record",
                action.action_id
            ));
        }
        return Ok(None);
    };
    if observed_state.is_none() {
        return Err(format!(
            "{} has actual resources without an execution state",
            action.action_id
        ));
    }
    let parsed: Resources = serde_json::from_value(value.clone())
        .map_err(|error| format!("{} actual_resources invalid: {error}", action.action_id))?;
    if !valid_resources(&parsed) {
        return Err(format!(
            "{} actual resource record is invalid",
            action.action_id
        ));
    }
    Ok(Some(parsed))
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

fn accumulate_resources(total: &mut Resources, actual: &Resources) -> Result<(), String> {
    total.engineering_hours += actual.engineering_hours;
    total.compute_core_hours += actual.compute_core_hours;
    total.facility_hours += actual.facility_hours;
    total.specimen_count = total
        .specimen_count
        .checked_add(actual.specimen_count)
        .ok_or_else(|| "actual specimen count overflow".to_owned())?;
    total.review_hours += actual.review_hours;
    total.external_cost_units += actual.external_cost_units;
    if !valid_resources(total) {
        return Err("accumulated actual resource total is non-finite".to_owned());
    }
    Ok(())
}

fn metadata_string<'a>(action: &'a Action, key: &str) -> Option<&'a str> {
    action.metadata.get(key).and_then(Value::as_str)
}

fn metadata_strings(action: &Action, key: &str) -> BTreeSet<String> {
    action
        .metadata
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn is_terminal_nonresult(action: &Action) -> bool {
    matches!(
        metadata_string(action, "observed_execution_state"),
        Some("FAILED" | "ABORTED" | "INVALID" | "CENSORED" | "OUT_OF_DOMAIN")
    )
}

fn action_is_independent(scenario: &Scenario, action: &Action) -> bool {
    if !scenario.requires_independent_evidence {
        return true;
    }
    let dependencies: BTreeSet<&str> = action.dependencies.iter().map(String::as_str).collect();
    !scenario
        .current_evidence_dependency_groups
        .values()
        .flatten()
        .any(|reference| dependencies.contains(reference.as_str()))
}

fn obligations_covered(scenario: &Scenario, action: &Action) -> bool {
    if scenario.required_obligations.is_empty() {
        return true;
    }
    let covered = metadata_strings(action, "obligations_covered");
    scenario
        .required_obligations
        .iter()
        .all(|obligation| covered.contains(obligation))
}

fn decision_regret(
    loss_matrix: &BTreeMap<String, BTreeMap<String, f64>>,
    decisions: &[String],
    states: &[String],
    decision: &str,
    state: &str,
) -> Result<f64, String> {
    let actual = *loss_matrix
        .get(decision)
        .and_then(|row| row.get(state))
        .ok_or_else(|| format!("missing loss for decision {decision} in state {state}"))?;
    let best = decisions
        .iter()
        .map(|candidate| {
            loss_matrix
                .get(candidate)
                .and_then(|row| row.get(state))
                .copied()
                .ok_or_else(|| format!("missing loss for decision {candidate} in state {state}"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let _ = states;
    Ok((actual - best).max(0.0))
}

fn minimax_regret(scenario: &Scenario) -> Result<f64, String> {
    let decisions: Vec<String> = scenario.loss_matrix.keys().cloned().collect();
    if decisions.len() < 2 || scenario.states.len() < 2 {
        return Err(format!(
            "{} must have at least two decisions and states",
            scenario.scenario_id
        ));
    }
    let mut best_worst = f64::INFINITY;
    for decision in &decisions {
        let mut worst = 0.0_f64;
        for state in &scenario.states {
            worst = worst.max(decision_regret(
                &scenario.loss_matrix,
                &decisions,
                &scenario.states,
                decision,
                state,
            )?);
        }
        best_worst = best_worst.min(worst);
    }
    Ok(best_worst)
}

fn post_action_minimax_regret(scenario: &Scenario, action: &Action) -> Result<f64, String> {
    let decisions: Vec<String> = scenario.loss_matrix.keys().cloned().collect();
    let mut groups = BTreeMap::<String, Vec<String>>::new();
    for state in &scenario.states {
        let observation = action
            .observation_by_state
            .get(state)
            .ok_or_else(|| format!("{} missing observation for state {state}", action.action_id))?;
        groups
            .entry(observation.clone())
            .or_default()
            .push(state.clone());
    }
    let mut worst_group_regret = 0.0_f64;
    for group_states in groups.values() {
        let mut best_group_regret = f64::INFINITY;
        for decision in &decisions {
            let mut worst = 0.0_f64;
            for state in group_states {
                worst = worst.max(decision_regret(
                    &scenario.loss_matrix,
                    &decisions,
                    &scenario.states,
                    decision,
                    state,
                )?);
            }
            best_group_regret = best_group_regret.min(worst);
        }
        worst_group_regret = worst_group_regret.max(best_group_regret);
    }
    Ok(worst_group_regret)
}

fn expected_loss(scenario: &Scenario, prior: &BTreeMap<String, f64>) -> Result<f64, String> {
    let decisions: Vec<String> = scenario.loss_matrix.keys().cloned().collect();
    decisions
        .iter()
        .map(|decision| {
            scenario
                .states
                .iter()
                .map(|state| {
                    let probability = *prior.get(state).ok_or_else(|| {
                        format!("{} prior missing state {state}", scenario.scenario_id)
                    })?;
                    let loss = *scenario
                        .loss_matrix
                        .get(decision)
                        .and_then(|row| row.get(state))
                        .ok_or_else(|| {
                            format!("{} loss missing {decision}/{state}", scenario.scenario_id)
                        })?;
                    Ok(probability * loss)
                })
                .try_fold(0.0, |sum, value: Result<f64, String>| {
                    value.map(|v| sum + v)
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|values| values.into_iter().fold(f64::INFINITY, f64::min))
}

fn expected_post_action_loss(
    scenario: &Scenario,
    prior: &BTreeMap<String, f64>,
    likelihoods: &BTreeMap<String, BTreeMap<String, f64>>,
) -> Result<f64, String> {
    let mut outcome_probabilities = BTreeMap::<String, f64>::new();
    for state in &scenario.states {
        let probability = *prior
            .get(state)
            .ok_or_else(|| format!("{} prior missing state {state}", scenario.scenario_id))?;
        let state_likelihoods = likelihoods
            .get(state)
            .ok_or_else(|| format!("{} likelihoods missing state {state}", scenario.scenario_id))?;
        let total: f64 = state_likelihoods.values().sum();
        if !total.is_finite() || (total - 1.0).abs() > 1.0e-9 {
            return Err(format!(
                "{} observation likelihoods for {state} do not sum to one",
                scenario.scenario_id
            ));
        }
        for (outcome, likelihood) in state_likelihoods {
            if !likelihood.is_finite() || *likelihood < 0.0 {
                return Err(format!("{} has invalid likelihood", scenario.scenario_id));
            }
            *outcome_probabilities.entry(outcome.clone()).or_default() += probability * likelihood;
        }
    }

    let mut expected = 0.0;
    for (outcome, marginal_probability) in outcome_probabilities {
        if marginal_probability <= 0.0 {
            continue;
        }
        let mut posterior = BTreeMap::new();
        for state in &scenario.states {
            let prior_probability = prior[state];
            let likelihood = likelihoods[state].get(&outcome).copied().unwrap_or(0.0);
            posterior.insert(
                state.clone(),
                prior_probability * likelihood / marginal_probability,
            );
        }
        expected += marginal_probability * expected_loss(scenario, &posterior)?;
    }
    Ok(expected)
}

fn parse_likelihoods(action: &Action) -> Result<Option<ObservationLikelihoods>, String> {
    let Some(value) = action.metadata.get("observation_likelihoods") else {
        return Ok(None);
    };
    serde_json::from_value(value.clone())
        .map(Some)
        .map_err(|error| {
            format!(
                "{} observation_likelihoods invalid: {error}",
                action.action_id
            )
        })
}

fn validate_scenario(scenario: &Scenario) -> Result<ScenarioAudit, String> {
    if scenario.claim_ceiling != CLAIM_CEILING || scenario.evidence_class != "synthetic" {
        return Err(format!(
            "{} has unsupported evidence class or claim ceiling",
            scenario.scenario_id
        ));
    }
    if scenario.scenario_id.trim().is_empty()
        || scenario.target_claim_id.trim().is_empty()
        || scenario.decision_context.profile_ref.trim().is_empty()
        || scenario.decision_context.article_subject_ref.trim().is_empty()
        || scenario.decision_context.configuration_id.trim().is_empty()
        || scenario.decision_context.icd_revision.trim().is_empty()
        || scenario.states.len() < 2
        || scenario.actions.is_empty()
        || scenario.current_evidence_refs.is_empty()
    {
        return Err(format!(
            "{} is missing required scenario identity or content",
            scenario.scenario_id
        ));
    }
    let states: BTreeSet<&str> = scenario.states.iter().map(String::as_str).collect();
    if states.len() != scenario.states.len() {
        return Err(format!(
            "{} has duplicate hidden states",
            scenario.scenario_id
        ));
    }
    if !valid_resources(&scenario.budget) {
        return Err(format!("{} has invalid budget", scenario.scenario_id));
    }
    let decisions: BTreeSet<&str> = scenario.loss_matrix.keys().map(String::as_str).collect();
    if decisions.len() < 2 {
        return Err(format!(
            "{} has fewer than two decisions",
            scenario.scenario_id
        ));
    }
    for (decision, row) in &scenario.loss_matrix {
        if row.len() != states.len()
            || row.keys().any(|state| !states.contains(state.as_str()))
            || row.values().any(|loss| !loss.is_finite() || *loss < 0.0)
        {
            return Err(format!(
                "{} loss row {decision} is incomplete or invalid",
                scenario.scenario_id
            ));
        }
    }
    if scenario
        .current_evidence_refs
        .iter()
        .collect::<BTreeSet<_>>()
        .len()
        != scenario.current_evidence_refs.len()
    {
        return Err(format!(
            "{} has duplicate current evidence references",
            scenario.scenario_id
        ));
    }
    if scenario
        .current_evidence_dependency_groups
        .keys()
        .any(|key| {
            !scenario
                .current_evidence_refs
                .iter()
                .any(|reference| reference == key)
        })
    {
        return Err(format!(
            "{} has dependency groups for unknown evidence refs",
            scenario.scenario_id
        ));
    }

    match scenario.decision_regime.as_str() {
        "distribution_free" => {
            if scenario.prior.is_some() {
                return Err(format!(
                    "{} distribution-free regime must not contain a prior",
                    scenario.scenario_id
                ));
            }
        }
        "calibrated_probability" => {
            let prior = scenario.prior.as_ref().ok_or_else(|| {
                format!(
                    "{} calibrated-probability regime requires an explicit prior",
                    scenario.scenario_id
                )
            })?;
            if prior.len() != states.len()
                || prior.keys().any(|state| !states.contains(state.as_str()))
                || prior.values().any(|p| !p.is_finite() || *p <= 0.0)
            {
                return Err(format!(
                    "{} has invalid prior coverage or probabilities",
                    scenario.scenario_id
                ));
            }
            let total: f64 = prior.values().sum();
            if (total - 1.0).abs() > 1.0e-9 {
                return Err(format!(
                    "{} prior probabilities do not sum to one",
                    scenario.scenario_id
                ));
            }
        }
        _ => {
            return Err(format!(
                "{} has an unknown decision regime",
                scenario.scenario_id
            ));
        }
    }

    let baseline = if scenario.decision_regime == "distribution_free" {
        minimax_regret(scenario)?
    } else {
        expected_loss(scenario, scenario.prior.as_ref().expect("prior validated"))?
    };
    if scenario.decision_regime == "distribution_free" {
        if let Some(expected) = scenario.expected.baseline_minimax_regret {
            if (baseline - expected).abs() > 1.0e-9 {
                return Err(format!(
                    "{} baseline minimax regret differs from oracle",
                    scenario.scenario_id
                ));
            }
        } else {
            return Err(format!(
                "{} distribution-free oracle is missing baseline regret",
                scenario.scenario_id
            ));
        }
    } else if let Some(expected) = scenario.expected.baseline_expected_loss {
        if (baseline - expected).abs() > 1.0e-9 {
            return Err(format!(
                "{} baseline expected loss differs from oracle",
                scenario.scenario_id
            ));
        }
    } else {
        return Err(format!(
            "{} calibrated-probability oracle is missing baseline expected loss",
            scenario.scenario_id
        ));
    }

    let mut action_ids = BTreeSet::new();
    let mut audits = Vec::new();
    let mut action_value_by_id = BTreeMap::<String, Option<f64>>::new();
    let mut eligible_by_id = BTreeMap::<String, bool>::new();
    let mut budget_blocked_count = 0usize;
    let mut actual_resources_consumed = zero_resources();
    let mut actual_over_plan_action_count = 0usize;
    let mut actions_with_actual_resource_records = 0usize;

    for action in &scenario.actions {
        if action.action_id.trim().is_empty() || !action_ids.insert(action.action_id.as_str()) {
            return Err(format!(
                "{} has empty or duplicate action IDs",
                scenario.scenario_id
            ));
        }
        for identity in [
            &action.action_class,
            &action.target_claim_id,
            &action.profile_ref,
            &action.article_subject_ref,
            &action.configuration_id,
            &action.icd_revision,
            &action.method_ref,
        ] {
            if identity.trim().is_empty() {
                return Err(format!(
                    "{} / {} has empty action identity",
                    scenario.scenario_id, action.action_id
                ));
            }
        }
        if !valid_resources(&action.resources) {
            return Err(format!(
                "{} / {} has invalid resource vector",
                scenario.scenario_id, action.action_id
            ));
        }
        let budget_fits = fits_budget(&action.resources, &scenario.budget);
        if !budget_fits {
            budget_blocked_count += 1;
        }
        if !scenario
            .states
            .iter()
            .all(|state| action.observation_by_state.contains_key(state))
            || action.observation_by_state.len() != states.len()
        {
            return Err(format!(
                "{} / {} observation map does not cover exact state set",
                scenario.scenario_id, action.action_id
            ));
        }
        if action.dependencies.iter().collect::<BTreeSet<_>>().len() != action.dependencies.len() {
            return Err(format!(
                "{} / {} has duplicate dependency labels",
                scenario.scenario_id, action.action_id
            ));
        }
        match action.admissibility.state.as_str() {
            "ADMISSIBLE" if !action.admissibility.reason_codes.is_empty() => {
                return Err(format!(
                    "{} / {} admissible action has blocking reason codes",
                    scenario.scenario_id, action.action_id
                ));
            }
            "INADMISSIBLE" | "SOURCE_SCALE_ONLY"
                if action.admissibility.reason_codes.is_empty() =>
            {
                return Err(format!(
                    "{} / {} blocked/source-scale-only action lacks reason codes",
                    scenario.scenario_id, action.action_id
                ));
            }
            "ADMISSIBLE" | "INADMISSIBLE" | "SOURCE_SCALE_ONLY" => {}
            _ => {
                return Err(format!(
                    "{} / {} has unknown admissibility state",
                    scenario.scenario_id, action.action_id
                ));
            }
        }
        let target_claim_matches = action.target_claim_id == scenario.target_claim_id;
        let profile_matches = action.profile_ref == scenario.decision_context.profile_ref;
        let article_subject_matches =
            action.article_subject_ref == scenario.decision_context.article_subject_ref;
        let configuration_matches =
            action.configuration_id == scenario.decision_context.configuration_id;
        let icd_revision_matches = action.icd_revision == scenario.decision_context.icd_revision;
        let common_cause_conflict = !action_is_independent(scenario, action);
        let mandatory_obligations_covered = obligations_covered(scenario, action);
        let terminal_nonresult = is_terminal_nonresult(action);
        let actual = actual_resources(action)?;
        let execution_state =
            metadata_string(action, "observed_execution_state").map(str::to_owned);
        let actual_exceeds_plan = actual
            .as_ref()
            .is_some_and(|resources| !fits_budget(resources, &action.resources));
        let actual_exceeds_scenario_budget = actual
            .as_ref()
            .is_some_and(|resources| !fits_budget(resources, &scenario.budget));
        if let Some(actual) = &actual {
            accumulate_resources(&mut actual_resources_consumed, actual)?;
            actions_with_actual_resource_records += 1;
            if actual_exceeds_plan {
                actual_over_plan_action_count += 1;
            }
        }
        let mut reasons = action.admissibility.reason_codes.clone();
        if !budget_fits {
            reasons.push("BUDGET_EXCEEDED".to_owned());
        }
        if !target_claim_matches {
            reasons.push("TARGET_CLAIM_MISMATCH".to_owned());
        }
        if !profile_matches {
            reasons.push("PROFILE_REFERENCE_MISMATCH".to_owned());
        }
        if !article_subject_matches {
            reasons.push("ARTICLE_SUBJECT_MISMATCH".to_owned());
        }
        if !configuration_matches {
            reasons.push("CONFIGURATION_IDENTITY_MISMATCH".to_owned());
        }
        if !icd_revision_matches {
            reasons.push("ICD_REVISION_MISMATCH".to_owned());
        }
        if common_cause_conflict {
            reasons.push("COMMON_CAUSE_DEPENDENCY".to_owned());
        }
        if !mandatory_obligations_covered {
            reasons.push("MANDATORY_OBLIGATION_MISSING".to_owned());
        }
        if terminal_nonresult {
            reasons.push("EXECUTION_OUTCOME_UNUSABLE".to_owned());
        }
        if scenario.decision_regime == "calibrated_probability"
            && action.admissibility.state == "ADMISSIBLE"
            && (metadata_string(action, "probability_model_ref").is_none()
                || metadata_string(action, "calibration_status")
                    != Some("calibrated_for_fixture_only"))
        {
            reasons.push("PROBABILITY_MODEL_UNCALIBRATED".to_owned());
        }
        reasons.sort();
        reasons.dedup();

        let probability_model_calibrated = scenario.decision_regime != "calibrated_probability"
            || (metadata_string(action, "probability_model_ref").is_some()
                && metadata_string(action, "calibration_status")
                    == Some("calibrated_for_fixture_only"));
        let policy_eligible = action.admissibility.state == "ADMISSIBLE"
            && budget_fits
            && target_claim_matches
            && profile_matches
            && article_subject_matches
            && configuration_matches
            && icd_revision_matches
            && !common_cause_conflict
            && mandatory_obligations_covered
            && !terminal_nonresult
            && probability_model_calibrated;
        let decision_informative = action
            .observation_by_state
            .values()
            .collect::<BTreeSet<_>>()
            .len()
            > 1;

        let decision_value_improvement = if !policy_eligible {
            None
        } else if scenario.decision_regime == "distribution_free" {
            let post = post_action_minimax_regret(scenario, action)?;
            Some((baseline - post).max(0.0))
        } else {
            let likelihoods = parse_likelihoods(action)?;
            match likelihoods {
                Some(likelihoods) => {
                    let expected_post = expected_post_action_loss(
                        scenario,
                        scenario.prior.as_ref().expect("prior validated"),
                        &likelihoods,
                    )?;
                    Some((baseline - expected_post).max(0.0))
                }
                None => None,
            }
        };
        let post_action_score = if !policy_eligible {
            None
        } else if scenario.decision_regime == "distribution_free" {
            Some(post_action_minimax_regret(scenario, action)?)
        } else if let Some(likelihoods) = parse_likelihoods(action)? {
            Some(expected_post_action_loss(
                scenario,
                scenario.prior.as_ref().expect("prior validated"),
                &likelihoods,
            )?)
        } else {
            None
        };

        action_value_by_id.insert(action.action_id.clone(), decision_value_improvement);
        eligible_by_id.insert(action.action_id.clone(), policy_eligible);
        audits.push(ActionAudit {
            action_id: action.action_id.clone(),
            admissibility_state: action.admissibility.state.clone(),
            planned_resources: action.resources.clone(),
            actual_resources: actual.clone(),
            execution_state,
            actual_exceeds_plan,
            actual_exceeds_scenario_budget,
            budget_fits,
            target_claim_matches,
            profile_matches,
            article_subject_matches,
            configuration_matches,
            icd_revision_matches,
            common_cause_conflict,
            mandatory_obligations_covered,
            decision_informative,
            execution_outcome_usable: !terminal_nonresult,
            policy_eligible,
            decision_value_improvement,
            post_action_regret_or_loss: post_action_score,
            reasons,
        });
    }

    audits.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    let (selected_action_id, selected_action_value, selection_disposition) =
        select_best_action(&audits);
    let expected_action_id = scenario.expected.best_action_id.clone();
    let expected_action_eligible = eligible_by_id
        .get(&expected_action_id)
        .copied()
        .unwrap_or(false);
    let expected_action_value = action_value_by_id
        .get(&expected_action_id)
        .copied()
        .flatten();
    let abstain_correct = if scenario.expected.abstain {
        selected_action_id.is_none()
            && matches!(
                selection_disposition,
                "ABSTAIN_NO_DECISION_VALUE" | "HUMAN_REVIEW_REQUIRED_RESOURCE_PARETO_TIE"
            )
    } else {
        expected_action_eligible && expected_action_value.is_some_and(|value| value > EPSILON)
    };

    for must_not_select in &scenario.expected.must_not_select {
        if !action_ids.contains(must_not_select.as_str()) {
            return Err(format!(
                "{} must_not_select references unknown action {must_not_select}",
                scenario.scenario_id
            ));
        }
        let eligible = eligible_by_id
            .get(must_not_select)
            .copied()
            .unwrap_or(false);
        let value = action_value_by_id.get(must_not_select).copied().flatten();
        if eligible && value.is_some_and(|score| score > EPSILON) {
            return Err(format!(
                "{} incorrectly marks positive-value eligible action {must_not_select} as must_not_select",
                scenario.scenario_id
            ));
        }
    }

    let expected_action_is_sentinel = matches!(expected_action_id.as_str(), "ABSTAIN")
        || expected_action_id.starts_with("REPLAN-");
    let selection_matches_oracle = if scenario.expected.abstain {
        selected_action_id.is_none()
    } else {
        selected_action_id.as_deref() == Some(expected_action_id.as_str())
    };
    if !expected_action_is_sentinel && !expected_action_eligible {
        return Err(format!(
            "{} expected action is not eligible",
            scenario.scenario_id
        ));
    }
    if expected_action_is_sentinel && !scenario.expected.abstain {
        return Err(format!(
            "{} sentinel action must set abstain=true",
            scenario.scenario_id
        ));
    }
    if scenario.expected.abstain && !abstain_correct {
        return Err(format!(
            "{} abstention is not supported by action-value audit",
            scenario.scenario_id
        ));
    }

    let best_action_post = if scenario.decision_regime == "distribution_free" {
        scenario.expected.post_action_worst_case_regret
    } else {
        scenario.expected.post_action_expected_loss
    };
    if let Some(expected_post) = best_action_post {
        if expected_action_is_sentinel {
            if (expected_post - baseline).abs() > 1.0e-9 {
                return Err(format!(
                    "{} abstention expected post score differs from baseline",
                    scenario.scenario_id
                ));
            }
        } else if expected_action_value.is_none() {
            return Err(format!(
                "{} expected action has no computed decision value",
                scenario.scenario_id
            ));
        } else if let Some(actual_post) = audits
            .iter()
            .find(|audit| audit.action_id == expected_action_id)
            .and_then(|audit| audit.post_action_regret_or_loss)
            && (actual_post - expected_post).abs() > 1.0e-9
        {
            return Err(format!(
                "{} expected post-action score differs from oracle",
                scenario.scenario_id
            ));
        }
    }

    if scenario.decision_regime == "calibrated_probability"
        && let Some(expected_evsi) = scenario.expected.evsi
        && expected_action_value.is_none_or(|value| (value - expected_evsi).abs() > 1.0e-9)
    {
        return Err(format!(
            "{} EVSI differs from independent one-step oracle",
            scenario.scenario_id
        ));
    }

    let action_count = audits.len();
    let eligible_action_count = audits.iter().filter(|audit| audit.policy_eligible).count();
    let oracle_matched = selection_matches_oracle
        && if scenario.expected.abstain {
            abstain_correct
        } else {
            expected_action_eligible && expected_action_value.is_some_and(|value| value > EPSILON)
        };
    Ok(ScenarioAudit {
        scenario_id: scenario.scenario_id.clone(),
        decision_regime: scenario.decision_regime.clone(),
        baseline_regret_or_loss: baseline,
        selected_action_id,
        selected_action_value,
        selection_disposition: selection_disposition.to_owned(),
        selection_matches_oracle,
        expected_action_id,
        expected_action_eligible,
        expected_action_value,
        abstention_correct: abstain_correct,
        action_count,
        eligible_action_count,
        budget_blocked_count,
        actual_resources_over_scenario_budget: !fits_budget(
            &actual_resources_consumed,
            &scenario.budget,
        ),
        actual_resources_consumed,
        actual_over_plan_action_count,
        actions_with_actual_resource_records,
        oracle_matched,
        actions: audits,
    })
}

pub fn run_audit(input: &[u8], source_commit: &str) -> Result<Receipt, String> {
    if !valid_commit(source_commit) {
        return Err("source commit must be exactly 40 lowercase hexadecimal characters".to_owned());
    }
    let corpus: Corpus = serde_json::from_slice(input)
        .map_err(|error| format!("invalid scenario corpus: {error}"))?;
    if corpus.schema != CORPUS_SCHEMA
        || corpus.benchmark_id != "ORBITAL-JOINT-EVIDENCE-VOI-001"
        || corpus.evidence_class != "synthetic"
        || corpus.claim_ceiling != CLAIM_CEILING
    {
        return Err(
            "unsupported corpus schema, benchmark ID, evidence class or claim ceiling".to_owned(),
        );
    }
    if corpus.resource_dimensions.is_empty()
        || corpus.policies_to_compare.is_empty()
        || corpus.scenarios.is_empty()
    {
        return Err("corpus dimensions, policies and scenarios must be non-empty".to_owned());
    }
    let mut scenario_ids = BTreeSet::new();
    let mut scenario_audits = Vec::new();
    for scenario in &corpus.scenarios {
        if !scenario_ids.insert(scenario.scenario_id.as_str()) {
            return Err(format!("duplicate scenario ID: {}", scenario.scenario_id));
        }
        scenario_audits.push(validate_scenario(scenario)?);
    }
    scenario_audits.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let oracle_matches = scenario_audits
        .iter()
        .filter(|audit| audit.oracle_matched)
        .count();
    let oracle_mismatches = scenario_audits.len() - oracle_matches;
    let admissible_action_count = scenario_audits
        .iter()
        .flat_map(|scenario| scenario.actions.iter())
        .filter(|action| action.admissibility_state == "ADMISSIBLE")
        .count();
    let eligible_action_count = scenario_audits
        .iter()
        .flat_map(|scenario| scenario.actions.iter())
        .filter(|action| action.policy_eligible)
        .count();
    let budget_blocked_action_count = scenario_audits
        .iter()
        .map(|scenario| scenario.budget_blocked_count)
        .sum();
    Ok(Receipt {
        schema: RECEIPT_SCHEMA.to_owned(),
        benchmark_id: corpus.benchmark_id,
        input_sha256: format!("{:x}", Sha256::digest(input)),
        source_commit: source_commit.to_owned(),
        evidence_class: "synthetic".to_owned(),
        claim_ceiling: CLAIM_CEILING.to_owned(),
        authority_granted: false,
        scenario_count: scenario_audits.len(),
        oracle_matches,
        oracle_mismatches,
        admissible_action_count,
        eligible_action_count,
        budget_blocked_action_count,
        scenarios: scenario_audits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] =
        include_bytes!("../../../fixtures/orbital-joint-evidence-voi/v0.1/scenarios.json");
    const SOURCE: &str = "0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn all_frozen_scenarios_pass_the_independent_semantic_contract() {
        let receipt = run_audit(FIXTURE, SOURCE).expect("scenario catalogue must audit");
        assert_eq!(receipt.scenario_count, 15);
        assert_eq!(receipt.oracle_matches, 15);
        assert_eq!(receipt.oracle_mismatches, 0);
        assert!(!receipt.authority_granted);
    }

    #[test]
    fn source_commit_shape_is_required() {
        assert!(run_audit(FIXTURE, "main").is_err());
    }

    #[test]
    fn robust_value_distinguishes_relevant_from_irrelevant_actions() {
        let receipt = run_audit(FIXTURE, SOURCE).unwrap();
        let first = receipt
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "JEV-001")
            .unwrap();
        assert_eq!(first.expected_action_id, "A-COUPON-INTERACTION");
        let coupon = first
            .actions
            .iter()
            .find(|action| action.action_id == "A-COUPON-INTERACTION")
            .unwrap();
        let mass = first
            .actions
            .iter()
            .find(|action| action.action_id == "A-MASS-ONLY")
            .unwrap();
        assert_eq!(coupon.decision_value_improvement, Some(5.0));
        assert_eq!(mass.decision_value_improvement, Some(0.0));
    }

    #[test]
    fn calibrated_evsi_uses_explicit_prior_and_likelihoods() {
        let receipt = run_audit(FIXTURE, SOURCE).unwrap();
        let case = receipt
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "JEV-002")
            .unwrap();
        assert!((case.baseline_regret_or_loss - 4.5).abs() < 1.0e-12);
        assert_eq!(case.expected_action_value, Some(4.5));
    }

    #[test]
    fn no_information_control_requires_abstention() {
        let receipt = run_audit(FIXTURE, SOURCE).unwrap();
        let case = receipt
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "JEV-003")
            .unwrap();
        assert!(case.abstention_correct);
        assert_eq!(case.eligible_action_count, 1);
    }

    #[test]
    fn over_budget_action_is_never_policy_eligible() {
        let receipt = run_audit(FIXTURE, SOURCE).unwrap();
        let case = receipt
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "JEV-004")
            .unwrap();
        assert_eq!(case.eligible_action_count, 0);
        assert_eq!(case.budget_blocked_count, 1);
    }

    #[test]
    fn serialized_article_and_lifecycle_obligations_are_not_conflated() {
        let receipt = run_audit(FIXTURE, SOURCE).unwrap();
        let case = receipt
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "JEV-010")
            .unwrap();
        let repeated = case
            .actions
            .iter()
            .find(|action| action.action_id == "A-REPEATED-MATING-WEAR")
            .unwrap();
        let first = case
            .actions
            .iter()
            .find(|action| action.action_id == "A-FIRST-CYCLE-ONLY")
            .unwrap();
        assert!(repeated.policy_eligible);
        assert!(!first.policy_eligible);
    }

    #[test]
    fn aborted_actual_resources_are_preserved_and_variance_is_reported() {
        let receipt = run_audit(FIXTURE, SOURCE).unwrap();
        let over_plan = receipt
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "JEV-012")
            .unwrap();
        let over_plan_action = over_plan
            .actions
            .iter()
            .find(|action| action.action_id == "A-ABORTED-OVER-PLAN")
            .unwrap();
        assert!(over_plan_action.actual_exceeds_plan);
        assert!(!over_plan_action.actual_exceeds_scenario_budget);
        assert!(!over_plan.actual_resources_over_scenario_budget);
        assert_eq!(over_plan.actual_over_plan_action_count, 1);
        assert_eq!(over_plan.actions_with_actual_resource_records, 1);
        assert_eq!(over_plan.actual_resources_consumed.engineering_hours, 1.2);
        assert_eq!(over_plan.actual_resources_consumed.facility_hours, 0.8);
        assert_eq!(over_plan.actual_resources_consumed.external_cost_units, 0.5);

        let over_budget = receipt
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "JEV-013")
            .unwrap();
        let over_budget_action = over_budget
            .actions
            .iter()
            .find(|action| action.action_id == "A-ABORTED-OVER-BUDGET")
            .unwrap();
        assert!(over_budget_action.actual_exceeds_plan);
        assert!(over_budget_action.actual_exceeds_scenario_budget);
        assert!(over_budget.actual_resources_over_scenario_budget);
        assert_eq!(over_budget.actual_resources_consumed.engineering_hours, 2.4);
        assert_eq!(over_budget.actual_resources_consumed.specimen_count, 2);
    }

    #[test]
    fn execution_state_without_actual_cost_is_rejected() {
        let mut value: Value = serde_json::from_slice(FIXTURE).unwrap();
        let action = value["scenarios"][10]["actions"][0]
            .as_object_mut()
            .unwrap();
        action.remove("actual_resources");
        let changed = serde_json::to_vec(&value).unwrap();
        assert!(run_audit(&changed, SOURCE).is_err());
    }

    #[test]
    fn actual_cost_without_execution_state_is_rejected() {
        let mut value: Value = serde_json::from_slice(FIXTURE).unwrap();
        let action = value["scenarios"][10]["actions"][0]
            .as_object_mut()
            .unwrap();
        action.remove("observed_execution_state");
        let changed = serde_json::to_vec(&value).unwrap();
        assert!(run_audit(&changed, SOURCE).is_err());
    }

    #[test]
    fn one_step_selector_picks_decision_value_not_prediction_variance() {
        let receipt = run_audit(FIXTURE, SOURCE).unwrap();
        let first = receipt
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "JEV-001")
            .unwrap();
        assert_eq!(
            first.selected_action_id.as_deref(),
            Some("A-COUPON-INTERACTION")
        );
        assert_eq!(
            first.selection_disposition,
            "SELECTED_MAX_VALUE_UNIQUE_PARETO_PLAN"
        );
        assert!(first.selection_matches_oracle);

        let campaign = receipt
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "JEV-009")
            .unwrap();
        assert_eq!(
            campaign.selected_action_id.as_deref(),
            Some("A-CRITICAL-INTERACTION-COVERAGE")
        );
    }

    #[test]
    fn one_step_selector_abstains_when_every_admissible_action_has_zero_value() {
        let receipt = run_audit(FIXTURE, SOURCE).unwrap();
        let case = receipt
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "JEV-003")
            .unwrap();
        assert_eq!(case.selected_action_id, None);
        assert_eq!(case.selection_disposition, "ABSTAIN_NO_DECISION_VALUE");
        assert!(case.selection_matches_oracle);
    }

    #[test]
    fn unique_pareto_dominating_action_breaks_equal_value_tie_without_cost_scalarization() {
        let mut value: Value = serde_json::from_slice(FIXTURE).unwrap();
        let scenario = value["scenarios"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|item| item["scenario_id"] == "JEV-001")
            .unwrap();
        let mut cheaper = scenario["actions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["action_id"] == "A-COUPON-INTERACTION")
            .unwrap()
            .clone();
        cheaper["action_id"] = Value::String("A-CHEAPER-COUPON".to_owned());
        cheaper["resources"]["engineering_hours"] = Value::from(0.5);
        cheaper["resources"]["facility_hours"] = Value::from(0.5);
        scenario["actions"].as_array_mut().unwrap().push(cheaper);
        scenario["expected"]["best_action_id"] = Value::String("A-CHEAPER-COUPON".to_owned());
        let changed = serde_json::to_vec(&value).unwrap();
        let receipt = run_audit(&changed, SOURCE).unwrap();
        let first = receipt
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "JEV-001")
            .unwrap();
        assert_eq!(
            first.selected_action_id.as_deref(),
            Some("A-CHEAPER-COUPON")
        );
        assert!(first.selection_matches_oracle);
    }

    #[test]
    fn incomparable_equal_value_resource_plans_require_human_review() {
        let mut value: Value = serde_json::from_slice(FIXTURE).unwrap();
        let scenario = value["scenarios"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|item| item["scenario_id"] == "JEV-001")
            .unwrap();
        let original = scenario["actions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["action_id"] == "A-COUPON-INTERACTION")
            .unwrap()
            .clone();
        let mut lower_engineering = original.clone();
        lower_engineering["action_id"] = Value::String("A-LOW-ENG-HIGH-FACILITY".to_owned());
        lower_engineering["resources"]["engineering_hours"] = Value::from(0.5);
        lower_engineering["resources"]["facility_hours"] = Value::from(1.5);
        let mut lower_facility = original;
        lower_facility["action_id"] = Value::String("A-HIGH-ENG-LOW-FACILITY".to_owned());
        lower_facility["resources"]["engineering_hours"] = Value::from(1.5);
        lower_facility["resources"]["facility_hours"] = Value::from(0.5);
        let actions = scenario["actions"].as_array_mut().unwrap();
        actions.push(lower_engineering);
        actions.push(lower_facility);
        let changed = serde_json::to_vec(&value).unwrap();
        let receipt = run_audit(&changed, SOURCE).unwrap();
        let first = receipt
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "JEV-001")
            .unwrap();
        assert_eq!(first.selected_action_id, None);
        assert_eq!(
            first.selection_disposition,
            "HUMAN_REVIEW_REQUIRED_RESOURCE_PARETO_TIE"
        );
        assert!(!first.selection_matches_oracle);
    }

    #[test]
    fn action_order_does_not_change_selected_action() {
        let original = run_audit(FIXTURE, SOURCE).unwrap();
        let mut value: Value = serde_json::from_slice(FIXTURE).unwrap();
        let scenario = value["scenarios"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|item| item["scenario_id"] == "JEV-001")
            .unwrap();
        scenario["actions"].as_array_mut().unwrap().reverse();
        let reversed = serde_json::to_vec(&value).unwrap();
        let second = run_audit(&reversed, SOURCE).unwrap();
        let first_choice = original
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "JEV-001")
            .unwrap();
        let second_choice = second
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "JEV-001")
            .unwrap();
        assert_eq!(
            first_choice.selected_action_id,
            second_choice.selected_action_id
        );
    }

    #[test]
    fn exact_profile_serial_configuration_and_icd_context_block_false_joins() {
        let receipt = run_audit(FIXTURE, SOURCE).unwrap();
        let case = receipt
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "JEV-015")
            .unwrap();
        assert_eq!(case.eligible_action_count, 0);
        assert_eq!(case.selected_action_id, None);
        assert_eq!(case.selection_disposition, "ABSTAIN_NO_DECISION_VALUE");
        assert!(case.selection_matches_oracle);
        for action in &case.actions {
            assert!(!action.policy_eligible);
            assert_eq!(action.admissibility_state, "ADMISSIBLE");
            assert!(action.target_claim_matches);
            assert_eq!(
                action.profile_matches,
                action.action_id != "A-WRONG-PROFILE"
            );
            assert_eq!(
                action.article_subject_matches,
                action.action_id != "A-WRONG-SERIAL"
            );
            assert_eq!(
                action.configuration_matches,
                action.action_id != "A-WRONG-CONFIGURATION"
            );
            assert_eq!(
                action.icd_revision_matches,
                action.action_id != "A-WRONG-ICD-REVISION"
            );
        }
    }

    #[test]
    fn incomparable_resource_tradeoff_is_frozen_as_human_review_not_scalarized() {
        let receipt = run_audit(FIXTURE, SOURCE).unwrap();
        let case = receipt
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "JEV-014")
            .unwrap();
        assert_eq!(case.selected_action_id, None);
        assert_eq!(case.selected_action_value, Some(5.0));
        assert_eq!(
            case.selection_disposition,
            "HUMAN_REVIEW_REQUIRED_RESOURCE_PARETO_TIE"
        );
        assert!(case.selection_matches_oracle);
        assert!(case.abstention_correct);
    }

    #[test]
    fn output_is_deterministic() {
        let first = serde_json::to_vec(&run_audit(FIXTURE, SOURCE).unwrap()).unwrap();
        let second = serde_json::to_vec(&run_audit(FIXTURE, SOURCE).unwrap()).unwrap();
        assert_eq!(first, second);
    }
}
