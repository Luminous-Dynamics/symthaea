use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/transport/ev-diagnostic-voi-scenario/0.1";
const SCENARIO_VERSION: &str = "ev-diagnostic-voi-scenario/0.1";
const CORPUS_VERSION: &str = "ev-diagnostic-voi-corpus/0.1";
const EPSILON: f64 = 1e-9;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    corpus_version: String,
    scenarios: Vec<CorpusCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusCase {
    case_id: String,
    expected_status: String,
    expected_test_id: Option<String>,
    scenario: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_version: String,
    scenario_id: String,
    oracle_evidence_class: String,
    asset: AssetBinding,
    evidence: EvidenceBinding,
    context: Context,
    states: Vec<State>,
    actions: Vec<Action>,
    tests: Vec<DiagnosticTest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AssetBinding {
    vehicle_ref: String,
    vehicle_configuration_digest: String,
    pack_ref: String,
    pack_configuration_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceBinding {
    observation_batch_ref: String,
    observation_batch_digest: String,
    pack_capability_profile_ref: String,
    pack_capability_profile_digest: String,
    prior_model_ref: String,
    prior_model_digest: String,
    decision_loss_model_ref: String,
    decision_loss_model_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Context {
    chemistry: String,
    temperature_c: f64,
    soc_fraction: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    state_id: String,
    prior_probability: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Action {
    action_id: String,
    gate_status: String,
    losses: Vec<StateLoss>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StateLoss {
    state_id: String,
    loss_units: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticTest {
    test_id: String,
    vehicle_configuration_digest: String,
    pack_configuration_digest: String,
    calibration_ref: String,
    protocol_ref: String,
    equipment_ref: String,
    protocol_digest: String,
    equipment_digest: String,
    calibration_digest: String,
    interpretation_status: String,
    cost: TestCost,
    applicability: Applicability,
    preconditions: Preconditions,
    outcomes: Vec<TestOutcome>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestCost {
    test_cost_units: f64,
    downtime_cost_units: f64,
    cost_unit_ref: String,
    evidence_class: String,
    evidence_digest: String,
    source_ref: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Applicability {
    supported_chemistries: Vec<String>,
    minimum_temperature_c: f64,
    maximum_temperature_c: f64,
    minimum_soc_fraction: f64,
    maximum_soc_fraction: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Preconditions {
    safety_review: String,
    qualified_facility: String,
    protocol_approved: String,
    calibration_valid: String,
    authority_approved: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestOutcome {
    outcome_id: String,
    likelihoods: Vec<StateLikelihood>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StateLikelihood {
    state_id: String,
    probability: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct OutcomeValue {
    pub outcome_id: String,
    pub probability: f64,
    pub posterior_best_action: String,
    pub posterior_expected_loss: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TestEvaluation {
    pub test_id: String,
    pub eligible: bool,
    pub ineligibility_reasons: Vec<String>,
    pub expected_information_gain_nats: f64,
    pub expected_post_test_loss: Option<f64>,
    pub gross_evsi: Option<f64>,
    pub test_cost_units: f64,
    pub downtime_cost_units: f64,
    pub net_value_units: Option<f64>,
    pub outcomes: Vec<OutcomeValue>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ScenarioDecision {
    pub scenario_id: String,
    pub vehicle_ref: String,
    pub vehicle_configuration_digest: String,
    pub pack_ref: String,
    pub pack_configuration_digest: String,
    pub observation_batch_ref: String,
    pub observation_batch_digest: String,
    pub pack_capability_profile_ref: String,
    pub pack_capability_profile_digest: String,
    pub prior_model_ref: String,
    pub prior_model_digest: String,
    pub decision_loss_model_ref: String,
    pub decision_loss_model_digest: String,
    pub recommendation_status: String,
    pub current_best_action: Option<String>,
    pub prior_expected_loss: Option<f64>,
    pub clairvoyant_lower_bound: Option<f64>,
    pub no_test_expected_regret: Option<f64>,
    pub selected_test_id: Option<String>,
    pub selected_test_gross_evsi: Option<f64>,
    pub selected_test_cost_units: Option<f64>,
    pub selected_test_downtime_cost_units: Option<f64>,
    pub selected_test_net_value_units: Option<f64>,
    pub expected_post_test_loss: Option<f64>,
    pub expected_post_test_regret: Option<f64>,
    pub maximum_information_gain_test_id: Option<String>,
    pub maximum_gross_evsi_test_id: Option<String>,
    pub eligible_test_count: usize,
    pub tests: Vec<TestEvaluation>,
}

#[derive(Debug, Serialize)]
struct ReportPayload {
    corpus_version: String,
    schema_id: String,
    schema_sha256: String,
    corpus_sha256: String,
    scenario_count: usize,
    selected_test_count: usize,
    no_test_count: usize,
    abstain_count: usize,
    scenarios: Vec<ScenarioDecision>,
}

#[derive(Debug, Serialize)]
pub struct CorpusReport {
    #[serde(flatten)]
    payload: ReportPayload,
    report_sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn close_to_one(value: f64) -> bool {
    value.is_finite() && (value - 1.0).abs() <= EPSILON
}

fn schema_validator(schema_bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(schema_bytes).map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if schema.get("$id").and_then(Value::as_str) != Some(SCHEMA_ID) {
        return Err("schema identity does not match frozen VOI contract".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn validate_scenario(scenario: &Scenario) -> Vec<String> {
    let mut errors = Vec::new();
    if scenario.scenario_version != SCENARIO_VERSION || !nonempty(&scenario.scenario_id) {
        errors.push("scenario version or identity is invalid".to_owned());
    }
    if !matches!(
        scenario.oracle_evidence_class.as_str(),
        "synthetic_fixture" | "independently_adjudicated_reference"
    ) {
        errors.push("unknown oracle evidence class".to_owned());
    }
    if !nonempty(&scenario.asset.vehicle_ref)
        || !nonempty(&scenario.asset.pack_ref)
        || !is_digest(&scenario.asset.vehicle_configuration_digest)
        || !is_digest(&scenario.asset.pack_configuration_digest)
    {
        errors.push("asset configuration identity or digest is invalid".to_owned());
    }
    if !nonempty(&scenario.evidence.observation_batch_ref)
        || !is_digest(&scenario.evidence.observation_batch_digest)
        || !nonempty(&scenario.evidence.pack_capability_profile_ref)
        || !is_digest(&scenario.evidence.pack_capability_profile_digest)
        || !nonempty(&scenario.evidence.prior_model_ref)
        || !is_digest(&scenario.evidence.prior_model_digest)
        || !nonempty(&scenario.evidence.decision_loss_model_ref)
        || !is_digest(&scenario.evidence.decision_loss_model_digest)
    {
        errors.push("observation, capability, prior or decision-loss evidence binding is invalid".to_owned());
    }
    if !nonempty(&scenario.context.chemistry)
        || !scenario.context.temperature_c.is_finite()
        || !scenario.context.soc_fraction.is_finite()
        || !(0.0..=1.0).contains(&scenario.context.soc_fraction)
    {
        errors.push("scenario context is invalid".to_owned());
    }

    let mut state_ids = BTreeSet::new();
    let mut prior_sum = 0.0;
    for state in &scenario.states {
        if !nonempty(&state.state_id) || !state_ids.insert(state.state_id.clone()) {
            errors.push(format!(
                "state identity is empty or duplicated: {}",
                state.state_id
            ));
        }
        if !state.prior_probability.is_finite()
            || state.prior_probability < 0.0
            || state.prior_probability > 1.0
        {
            errors.push(format!(
                "state {} has invalid prior probability",
                state.state_id
            ));
        }
        prior_sum += state.prior_probability;
    }
    if scenario.states.len() < 2 || !close_to_one(prior_sum) {
        errors.push("state priors must contain at least two states and sum to one".to_owned());
    }

    let mut action_ids = BTreeSet::new();
    for action in &scenario.actions {
        if !nonempty(&action.action_id) || !action_ids.insert(action.action_id.clone()) {
            errors.push(format!(
                "action identity is empty or duplicated: {}",
                action.action_id
            ));
        }
        if !matches!(
            action.gate_status.as_str(),
            "satisfied" | "blocked" | "unknown"
        ) {
            errors.push(format!(
                "action {} has unknown gate status",
                action.action_id
            ));
        }
        let mut seen = BTreeSet::new();
        for loss in &action.losses {
            if !seen.insert(loss.state_id.clone()) {
                errors.push(format!(
                    "action {} repeats state loss {}",
                    action.action_id, loss.state_id
                ));
            }
            if !state_ids.contains(&loss.state_id)
                || !loss.loss_units.is_finite()
                || loss.loss_units < 0.0
            {
                errors.push(format!(
                    "action {} has invalid state loss",
                    action.action_id
                ));
            }
        }
        if seen != state_ids {
            errors.push(format!(
                "action {} must declare exactly one loss per state",
                action.action_id
            ));
        }
    }
    if scenario.actions.is_empty() {
        errors.push("scenario must declare at least one candidate action".to_owned());
    }

    let mut test_ids = BTreeSet::new();
    for test in &scenario.tests {
        if !nonempty(&test.test_id) || !test_ids.insert(test.test_id.clone()) {
            errors.push(format!(
                "test identity is empty or duplicated: {}",
                test.test_id
            ));
        }
        if !is_digest(&test.vehicle_configuration_digest)
            || !is_digest(&test.pack_configuration_digest)
            || !nonempty(&test.calibration_ref)
            || !nonempty(&test.protocol_ref)
            || !nonempty(&test.equipment_ref)
            || !is_digest(&test.protocol_digest)
            || !is_digest(&test.equipment_digest)
            || !is_digest(&test.calibration_digest)
        {
            errors.push(format!(
                "test {} has invalid protocol/equipment/calibration identity",
                test.test_id
            ));
        }
        if !matches!(
            test.interpretation_status.as_str(),
            "qualified_for_domain" | "unqualified" | "unknown"
        ) {
            errors.push(format!(
                "test {} has invalid interpretation status",
                test.test_id
            ));
        }
        if !test.cost.test_cost_units.is_finite()
            || test.cost.test_cost_units < 0.0
            || !test.cost.downtime_cost_units.is_finite()
            || test.cost.downtime_cost_units < 0.0
            || test.cost.cost_unit_ref != "decision-loss-unit/0.1"
            || !matches!(
                test.cost.evidence_class.as_str(),
                "declared_synthetic_cost" | "reference_quote" | "measured_cost"
            )
            || !is_digest(&test.cost.evidence_digest)
            || !nonempty(&test.cost.source_ref)
        {
            errors.push(format!(
                "test {} has invalid or untraceable cost",
                test.test_id
            ));
        }
        let app = &test.applicability;
        if app.supported_chemistries.is_empty()
            || !app.minimum_temperature_c.is_finite()
            || !app.maximum_temperature_c.is_finite()
            || app.minimum_temperature_c > app.maximum_temperature_c
            || !app.minimum_soc_fraction.is_finite()
            || !app.maximum_soc_fraction.is_finite()
            || app.minimum_soc_fraction < 0.0
            || app.maximum_soc_fraction > 1.0
            || app.minimum_soc_fraction > app.maximum_soc_fraction
        {
            errors.push(format!(
                "test {} has invalid applicability domain",
                test.test_id
            ));
        }
        for status in [
            test.preconditions.safety_review.as_str(),
            test.preconditions.qualified_facility.as_str(),
            test.preconditions.protocol_approved.as_str(),
            test.preconditions.calibration_valid.as_str(),
            test.preconditions.authority_approved.as_str(),
        ] {
            if !matches!(status, "satisfied" | "missing" | "unknown") {
                errors.push(format!(
                    "test {} has unknown precondition status",
                    test.test_id
                ));
            }
        }

        let mut outcome_ids = BTreeSet::new();
        let mut probability_sums: BTreeMap<String, f64> =
            state_ids.iter().map(|id| (id.clone(), 0.0)).collect();
        for outcome in &test.outcomes {
            if !nonempty(&outcome.outcome_id) || !outcome_ids.insert(outcome.outcome_id.clone()) {
                errors.push(format!(
                    "test {} has empty or duplicate outcome identity",
                    test.test_id
                ));
            }
            let mut likelihood_state_ids = BTreeSet::new();
            for likelihood in &outcome.likelihoods {
                if !likelihood_state_ids.insert(likelihood.state_id.clone())
                    || !state_ids.contains(&likelihood.state_id)
                    || !likelihood.probability.is_finite()
                    || likelihood.probability < 0.0
                    || likelihood.probability > 1.0
                {
                    errors.push(format!(
                        "test {} outcome {} has invalid likelihood entry",
                        test.test_id, outcome.outcome_id
                    ));
                } else if let Some(sum) = probability_sums.get_mut(&likelihood.state_id) {
                    *sum += likelihood.probability;
                }
            }
            if likelihood_state_ids != state_ids {
                errors.push(format!(
                    "test {} outcome {} must cover exactly every state",
                    test.test_id, outcome.outcome_id
                ));
            }
        }
        if test.outcomes.len() < 2 || probability_sums.values().any(|sum| !close_to_one(*sum)) {
            errors.push(format!(
                "test {} outcome likelihoods must sum to one for every state",
                test.test_id
            ));
        }
    }
    errors
}

fn sorted_states(scenario: &Scenario) -> Vec<&State> {
    let mut states: Vec<&State> = scenario.states.iter().collect();
    states.sort_by(|a, b| a.state_id.cmp(&b.state_id));
    states
}

fn sorted_actions(scenario: &Scenario) -> Vec<&Action> {
    let mut actions: Vec<&Action> = scenario.actions.iter().collect();
    actions.sort_by(|a, b| a.action_id.cmp(&b.action_id));
    actions
}

fn loss_map(action: &Action) -> BTreeMap<&str, f64> {
    action
        .losses
        .iter()
        .map(|loss| (loss.state_id.as_str(), loss.loss_units))
        .collect()
}

fn state_probabilities(scenario: &Scenario) -> BTreeMap<String, f64> {
    scenario
        .states
        .iter()
        .map(|state| (state.state_id.clone(), state.prior_probability))
        .collect()
}

fn admissible_actions(scenario: &Scenario) -> Vec<&Action> {
    sorted_actions(scenario)
        .into_iter()
        .filter(|action| action.gate_status == "satisfied")
        .collect()
}

fn expected_loss(action: &Action, probabilities: &BTreeMap<String, f64>) -> f64 {
    let losses = loss_map(action);
    probabilities
        .iter()
        .map(|(state, probability)| {
            probability * losses.get(state.as_str()).copied().unwrap_or(f64::INFINITY)
        })
        .sum()
}

fn best_action(
    actions: &[&Action],
    probabilities: &BTreeMap<String, f64>,
) -> Option<(String, f64)> {
    actions
        .iter()
        .map(|action| {
            (
                action.action_id.clone(),
                expected_loss(action, probabilities),
            )
        })
        .min_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        })
}

fn clairvoyant_loss(actions: &[&Action], probabilities: &BTreeMap<String, f64>) -> f64 {
    let states: Vec<String> = probabilities.keys().cloned().collect();
    states
        .iter()
        .map(|state| {
            let one_hot: BTreeMap<String, f64> = states
                .iter()
                .map(|id| (id.clone(), if id == state { 1.0 } else { 0.0 }))
                .collect();
            let minimum = actions
                .iter()
                .map(|action| expected_loss(action, &one_hot))
                .fold(f64::INFINITY, f64::min);
            probabilities.get(state).copied().unwrap_or(0.0) * minimum
        })
        .sum()
}

fn entropy(probabilities: impl Iterator<Item = f64>) -> f64 {
    probabilities
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

fn precondition_reasons(test: &DiagnosticTest) -> Vec<String> {
    let mut reasons = Vec::new();
    if test.interpretation_status != "qualified_for_domain" {
        reasons.push("interpretation_model_not_qualified".to_owned());
    }
    if test.preconditions.safety_review != "satisfied" {
        reasons.push("safety_review_not_satisfied".to_owned());
    }
    if test.preconditions.qualified_facility != "satisfied" {
        reasons.push("qualified_facility_not_confirmed".to_owned());
    }
    if test.preconditions.protocol_approved != "satisfied" {
        reasons.push("protocol_not_approved".to_owned());
    }
    if test.preconditions.calibration_valid != "satisfied" {
        reasons.push("calibration_not_valid".to_owned());
    }
    if test.preconditions.authority_approved != "satisfied" {
        reasons.push("responsible_authority_not_approved".to_owned());
    }
    reasons
}

fn eligibility_reasons(test: &DiagnosticTest, scenario: &Scenario) -> Vec<String> {
    let mut reasons = precondition_reasons(test);
    if test.vehicle_configuration_digest != scenario.asset.vehicle_configuration_digest
        || test.pack_configuration_digest != scenario.asset.pack_configuration_digest
    {
        reasons.push("vehicle_or_pack_configuration_mismatch".to_owned());
    }
    let context = &scenario.context;
    if !test
        .applicability
        .supported_chemistries
        .contains(&context.chemistry)
    {
        reasons.push("chemistry_out_of_domain".to_owned());
    }
    if context.temperature_c < test.applicability.minimum_temperature_c
        || context.temperature_c > test.applicability.maximum_temperature_c
    {
        reasons.push("temperature_out_of_domain".to_owned());
    }
    if context.soc_fraction < test.applicability.minimum_soc_fraction
        || context.soc_fraction > test.applicability.maximum_soc_fraction
    {
        reasons.push("soc_out_of_domain".to_owned());
    }
    reasons.sort();
    reasons
}

fn evaluate_test(
    test: &DiagnosticTest,
    scenario: &Scenario,
    actions: &[&Action],
    prior: &BTreeMap<String, f64>,
    prior_expected_loss: f64,
) -> TestEvaluation {
    let reasons = eligibility_reasons(test, scenario);
    let eligible = reasons.is_empty();
    let prior_entropy = entropy(prior.values().copied());
    let mut outcomes = Vec::new();
    let mut expected_post_loss = 0.0;
    let mut expected_posterior_entropy = 0.0;
    let mut information_gain = 0.0;

    if eligible {
        let mut sorted_outcomes: Vec<&TestOutcome> = test.outcomes.iter().collect();
        sorted_outcomes.sort_by(|a, b| a.outcome_id.cmp(&b.outcome_id));
        for outcome in sorted_outcomes {
            let likelihoods: BTreeMap<&str, f64> = outcome
                .likelihoods
                .iter()
                .map(|entry| (entry.state_id.as_str(), entry.probability))
                .collect();
            let outcome_probability = prior
                .iter()
                .map(|(state, probability)| {
                    probability * likelihoods.get(state.as_str()).copied().unwrap_or(0.0)
                })
                .sum::<f64>();
            let posterior: BTreeMap<String, f64> = prior
                .iter()
                .map(|(state, probability)| {
                    let likelihood = likelihoods.get(state.as_str()).copied().unwrap_or(0.0);
                    let posterior_probability = if outcome_probability > 0.0 {
                        probability * likelihood / outcome_probability
                    } else {
                        0.0
                    };
                    (state.clone(), posterior_probability)
                })
                .collect();
            let posterior_entropy = entropy(posterior.values().copied());
            expected_posterior_entropy += outcome_probability * posterior_entropy;
            let (action, loss) = best_action(actions, &posterior).unwrap_or_default();
            expected_post_loss += outcome_probability * loss;
            outcomes.push(OutcomeValue {
                outcome_id: outcome.outcome_id.clone(),
                probability: outcome_probability,
                posterior_best_action: action,
                posterior_expected_loss: loss,
            });
        }
        information_gain = (prior_entropy - expected_posterior_entropy).max(0.0);
    }

    let gross_evsi = if eligible {
        Some((prior_expected_loss - expected_post_loss).max(0.0))
    } else {
        None
    };
    let net_value =
        gross_evsi.map(|value| value - test.cost.test_cost_units - test.cost.downtime_cost_units);

    TestEvaluation {
        test_id: test.test_id.clone(),
        eligible,
        ineligibility_reasons: reasons,
        expected_information_gain_nats: information_gain,
        expected_post_test_loss: eligible.then_some(expected_post_loss),
        gross_evsi,
        test_cost_units: test.cost.test_cost_units,
        downtime_cost_units: test.cost.downtime_cost_units,
        net_value_units: net_value,
        outcomes,
    }
}

fn choose_test<'a>(
    tests: &'a [TestEvaluation],
    value: fn(&'a TestEvaluation) -> f64,
) -> Option<&'a TestEvaluation> {
    tests
        .iter()
        .filter(|test| test.eligible)
        .max_by(|left, right| {
            value(left)
                .total_cmp(&value(right))
                .then_with(|| right.test_id.cmp(&left.test_id))
        })
}

fn evaluate_scenario(scenario: &Scenario) -> ScenarioDecision {
    let actions = admissible_actions(scenario);
    let prior = state_probabilities(scenario);
    let current = best_action(&actions, &prior);
    let oracle = if actions.is_empty() {
        None
    } else {
        Some(clairvoyant_loss(&actions, &prior))
    };
    let prior_loss = current.as_ref().map(|(_, loss)| *loss);
    let no_test_regret = prior_loss
        .zip(oracle)
        .map(|(loss, lower)| (loss - lower).max(0.0));

    let mut tests: Vec<&DiagnosticTest> = scenario.tests.iter().collect();
    tests.sort_by(|a, b| a.test_id.cmp(&b.test_id));
    let evaluations: Vec<TestEvaluation> = if let Some((_, current_loss)) = current.as_ref() {
        tests
            .iter()
            .map(|test| evaluate_test(test, scenario, &actions, &prior, *current_loss))
            .collect()
    } else {
        tests
            .iter()
            .map(|test| TestEvaluation {
                test_id: test.test_id.clone(),
                eligible: false,
                ineligibility_reasons: vec!["no_admissible_current_action".to_owned()],
                expected_information_gain_nats: 0.0,
                expected_post_test_loss: None,
                gross_evsi: None,
                test_cost_units: test.cost.test_cost_units,
                downtime_cost_units: test.cost.downtime_cost_units,
                net_value_units: None,
                outcomes: Vec::new(),
            })
            .collect()
    };

    let best_info = choose_test(&evaluations, |test| test.expected_information_gain_nats);
    let best_gross = choose_test(&evaluations, |test| {
        test.gross_evsi.unwrap_or(f64::NEG_INFINITY)
    });
    let selected = choose_test(&evaluations, |test| {
        test.net_value_units.unwrap_or(f64::NEG_INFINITY)
    })
    .filter(|test| test.net_value_units.unwrap_or(f64::NEG_INFINITY) > EPSILON);

    let (status, selected_id) = if current.is_none() {
        ("abstain", None)
    } else if let Some(test) = selected {
        ("test_selected", Some(test.test_id.clone()))
    } else {
        ("no_test", None)
    };

    let selected_test = selected_id
        .as_ref()
        .and_then(|id| evaluations.iter().find(|test| &test.test_id == id));
    let expected_post = selected_test.and_then(|test| test.expected_post_test_loss);
    let expected_post_regret = expected_post.zip(oracle).map(|(loss, lower)| {
        (loss - lower).max(0.0)
            + selected_test
                .map(|test| test.test_cost_units + test.downtime_cost_units)
                .unwrap_or(0.0)
    });

    ScenarioDecision {
        scenario_id: scenario.scenario_id.clone(),
        vehicle_ref: scenario.asset.vehicle_ref.clone(),
        vehicle_configuration_digest: scenario.asset.vehicle_configuration_digest.clone(),
        pack_ref: scenario.asset.pack_ref.clone(),
        pack_configuration_digest: scenario.asset.pack_configuration_digest.clone(),
        observation_batch_ref: scenario.evidence.observation_batch_ref.clone(),
        observation_batch_digest: scenario.evidence.observation_batch_digest.clone(),
        pack_capability_profile_ref: scenario.evidence.pack_capability_profile_ref.clone(),
        pack_capability_profile_digest: scenario.evidence.pack_capability_profile_digest.clone(),
        prior_model_ref: scenario.evidence.prior_model_ref.clone(),
        prior_model_digest: scenario.evidence.prior_model_digest.clone(),
        decision_loss_model_ref: scenario.evidence.decision_loss_model_ref.clone(),
        decision_loss_model_digest: scenario.evidence.decision_loss_model_digest.clone(),
        recommendation_status: status.to_owned(),
        current_best_action: current.as_ref().map(|(action, _)| action.clone()),
        prior_expected_loss: prior_loss,
        clairvoyant_lower_bound: oracle,
        no_test_expected_regret: no_test_regret,
        selected_test_id: selected_id,
        selected_test_gross_evsi: selected_test.and_then(|test| test.gross_evsi),
        selected_test_cost_units: selected_test.map(|test| test.test_cost_units),
        selected_test_downtime_cost_units: selected_test.map(|test| test.downtime_cost_units),
        selected_test_net_value_units: selected_test.and_then(|test| test.net_value_units),
        expected_post_test_loss: expected_post,
        expected_post_test_regret: expected_post_regret,
        maximum_information_gain_test_id: best_info.map(|test| test.test_id.clone()),
        maximum_gross_evsi_test_id: best_gross.map(|test| test.test_id.clone()),
        eligible_test_count: evaluations.iter().filter(|test| test.eligible).count(),
        tests: evaluations,
    }
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<CorpusReport, String> {
    let validator = schema_validator(schema_bytes)?;
    let corpus: Corpus =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION {
        return Err("unsupported VOI corpus version".to_owned());
    }

    let mut seen_case_ids = BTreeSet::new();
    let mut decisions = Vec::new();
    let mut selected_count = 0;
    let mut no_test_count = 0;
    let mut abstain_count = 0;

    let mut cases: Vec<CorpusCase> = corpus.scenarios;
    cases.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    for case in cases {
        if !nonempty(&case.case_id) || !seen_case_ids.insert(case.case_id.clone()) {
            return Err(format!(
                "case identity is empty or duplicated: {}",
                case.case_id
            ));
        }
        if !matches!(
            case.expected_status.as_str(),
            "test_selected" | "no_test" | "abstain" | "reject"
        ) {
            return Err(format!("case {} has invalid expected status", case.case_id));
        }
        let schema_errors: Vec<String> = validator
            .iter_errors(&case.scenario)
            .map(|error| error.to_string())
            .collect();
        if !schema_errors.is_empty() {
            if case.expected_status == "reject" {
                continue;
            }
            return Err(format!(
                "case {} fails schema: {}",
                case.case_id,
                schema_errors.join("; ")
            ));
        }

        let scenario: Scenario = serde_json::from_value(case.scenario.clone())
            .map_err(|error| format!("case {} typed scenario parse: {error}", case.case_id))?;
        let semantic_errors = validate_scenario(&scenario);
        if !semantic_errors.is_empty() {
            if case.expected_status == "reject" {
                continue;
            }
            return Err(format!(
                "case {} invalid: {}",
                case.case_id,
                semantic_errors.join("; ")
            ));
        }
        if case.expected_status == "reject" {
            return Err(format!(
                "negative case {} was unexpectedly accepted",
                case.case_id
            ));
        }

        let decision = evaluate_scenario(&scenario);
        if decision.recommendation_status != case.expected_status
            || decision.selected_test_id != case.expected_test_id
        {
            return Err(format!(
                "case {} expected status/test {:?}/{:?}, got {:?}/{:?}",
                case.case_id,
                case.expected_status,
                case.expected_test_id,
                decision.recommendation_status,
                decision.selected_test_id
            ));
        }
        match decision.recommendation_status.as_str() {
            "test_selected" => selected_count += 1,
            "no_test" => no_test_count += 1,
            "abstain" => abstain_count += 1,
            _ => {}
        }
        decisions.push(decision);
    }

    let schema_digest = sha256(schema_bytes);
    let corpus_digest = sha256(corpus_bytes);
    let payload = ReportPayload {
        corpus_version: CORPUS_VERSION.to_owned(),
        schema_id: SCHEMA_ID.to_owned(),
        schema_sha256: schema_digest,
        corpus_sha256: corpus_digest,
        scenario_count: decisions.len(),
        selected_test_count: selected_count,
        no_test_count,
        abstain_count,
        scenarios: decisions,
    };
    let report_bytes =
        serde_json::to_vec(&payload).map_err(|error| format!("report serialization: {error}"))?;
    let report_sha256 = sha256(&report_bytes);
    Ok(CorpusReport {
        payload,
        report_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_diagnostic_voi_scenario_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/diagnostic_voi_scenarios_v0_1.json");

    #[test]
    fn frozen_corpus_selects_one_test_abstains_once_and_rejects_two_controls() {
        let report = audit_corpus(SCHEMA, CORPUS).expect("frozen decision corpus");
        assert_eq!(report.payload.scenario_count, 5);
        assert_eq!(report.payload.selected_test_count, 1);
        assert_eq!(report.payload.no_test_count, 3);
        assert_eq!(report.payload.abstain_count, 1);
    }

    #[test]
    fn high_information_without_decision_value_does_not_trigger_a_test() {
        let report = audit_corpus(SCHEMA, CORPUS).expect("frozen decision corpus");
        let case = report
            .payload
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "synthetic-high-information-no-decision-value")
            .expect("information-only case");
        let test = case.tests.first().expect("diagnostic test");
        assert!(test.expected_information_gain_nats > 0.5);
        assert_eq!(test.gross_evsi, Some(0.0));
        assert_eq!(case.recommendation_status, "no_test");
    }

    #[test]
    fn test_selection_has_positive_net_value_and_reduces_expected_regret() {
        let report = audit_corpus(SCHEMA, CORPUS).expect("frozen decision corpus");
        let case = report
            .payload
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "synthetic-diagnostic-positive-evsi")
            .expect("positive EVSI case");
        assert_eq!(
            case.selected_test_id.as_deref(),
            Some("capacity_reference_test")
        );
        assert!(case.selected_test_net_value_units.unwrap_or(0.0) > 0.0);
        assert!(
            case.expected_post_test_regret.unwrap_or(f64::INFINITY)
                < case.no_test_expected_regret.unwrap_or(0.0)
        );
    }

    #[test]
    fn permutation_of_states_actions_tests_and_outcomes_does_not_change_decision() {
        let corpus: Value = serde_json::from_slice(CORPUS).expect("corpus");
        let original_value = corpus["scenarios"][0]["scenario"].clone();
        let original: Scenario = serde_json::from_value(original_value.clone()).expect("scenario");
        let original_decision = evaluate_scenario(&original);
        let mut permuted_value = original_value;
        permuted_value["states"].as_array_mut().unwrap().reverse();
        permuted_value["actions"].as_array_mut().unwrap().reverse();
        permuted_value["tests"].as_array_mut().unwrap().reverse();
        for test in permuted_value["tests"].as_array_mut().unwrap() {
            test["outcomes"].as_array_mut().unwrap().reverse();
            for outcome in test["outcomes"].as_array_mut().unwrap() {
                outcome["likelihoods"].as_array_mut().unwrap().reverse();
            }
        }
        let permuted: Scenario = serde_json::from_value(permuted_value).expect("permuted scenario");
        assert_eq!(original_decision, evaluate_scenario(&permuted));
    }

    #[test]
    fn mismatched_pack_configuration_excludes_a_test() {
        let report = audit_corpus(SCHEMA, CORPUS).expect("frozen decision corpus");
        let case = report
            .payload
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "synthetic-wrong-pack-configuration-test-excluded")
            .expect("wrong-configuration case");
        assert_eq!(case.recommendation_status, "no_test");
        let test = case.tests.first().expect("candidate test");
        assert!(!test.eligible);
        assert!(test
            .ineligibility_reasons
            .contains(&"vehicle_or_pack_configuration_mismatch".to_owned()));
    }

    #[test]
    fn scenario_report_preserves_asset_and_evidence_identity() {
        let report = audit_corpus(SCHEMA, CORPUS).expect("frozen decision corpus");
        let case = report
            .payload
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "synthetic-diagnostic-positive-evsi")
            .expect("positive EVSI case");
        assert_eq!(case.vehicle_ref, "vehicle-synthetic-ev-01");
        assert_eq!(case.pack_ref, "pack-synthetic-ev-01");
        assert_eq!(
            case.observation_batch_ref,
            "synthetic-observation-batch-v0.1"
        );
        assert_eq!(
            case.pack_capability_profile_ref,
            "synthetic-pack-capability-profile-v0.1"
        );
    }

    #[test]
    fn no_admissible_action_returns_abstain() {
        let report = audit_corpus(SCHEMA, CORPUS).expect("frozen decision corpus");
        let case = report
            .payload
            .scenarios
            .iter()
            .find(|case| case.scenario_id == "synthetic-no-admissible-action-abstain")
            .expect("abstention case");
        assert_eq!(case.recommendation_status, "abstain");
        assert!(case.selected_test_id.is_none());
        assert_eq!(case.eligible_test_count, 0);
    }
}
