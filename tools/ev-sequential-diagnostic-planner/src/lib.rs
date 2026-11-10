use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/transport/ev-sequential-diagnostic-scenario/0.1";
const CORPUS_VERSION: &str = "ev-sequential-diagnostic-corpus/0.1";
const SCENARIO_VERSION: &str = "ev-sequential-diagnostic/0.1";
const EPSILON: f64 = 1e-12;
const MAX_HORIZON: usize = 8;
const MAX_OUTCOMES: usize = 8;
const MAX_POLICY_NODES: usize = 4096;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    corpus_version: String,
    schema_id: String,
    cases: Vec<CorpusCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusCase {
    case_id: String,
    expectation: String,
    expected_root_choice: Option<String>,
    expected_root_test_id: Option<String>,
    expected_net_value: Option<f64>,
    scenario: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    schema_version: String,
    scenario_id: String,
    decision_unit: String,
    horizon: usize,
    abstain_loss: f64,
    hypotheses: Vec<HypothesisPrior>,
    actions: Vec<DecisionAction>,
    tests: Vec<DiagnosticTest>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HypothesisPrior {
    id: String,
    prior: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionAction {
    id: String,
    admissible: bool,
    loss_by_hypothesis: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticTest {
    id: String,
    test_cost: f64,
    downtime_cost: f64,
    prerequisites: Prerequisites,
    requires: Option<TestCondition>,
    outcomes: Vec<TestOutcome>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Prerequisites {
    safety_approved: bool,
    calibration_valid: bool,
    domain_supported: bool,
    equipment_available: bool,
    protocol_digest: String,
    equipment_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestCondition {
    test_id: String,
    outcome: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestOutcome {
    id: String,
    likelihood_by_hypothesis: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PolicyNode {
    pub branch_probability: Option<f64>,
    pub posterior: BTreeMap<String, f64>,
    pub choice: String,
    pub test_id: Option<String>,
    pub terminal_action: Option<String>,
    pub expected_total_loss: f64,
    pub immediate_test_cost: f64,
    pub expected_tests: f64,
    pub max_tests_on_path: usize,
    pub branches: BTreeMap<String, PolicyNode>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct OneStepValue {
    pub test_id: String,
    pub admissible_at_root: bool,
    pub exclusion_reason: Option<String>,
    pub expected_total_loss: Option<f64>,
    pub net_value: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PlanResult {
    pub scenario_id: String,
    pub root_choice: String,
    pub root_test_id: Option<String>,
    pub baseline_no_test_loss: f64,
    pub optimized_expected_loss: f64,
    pub net_value: f64,
    pub expected_tests: f64,
    pub max_tests_on_path: usize,
    pub one_step_candidates: Vec<OneStepValue>,
    pub policy: PolicyNode,
}

#[derive(Debug, Serialize)]
struct CaseAudit {
    case_id: String,
    expectation: String,
    accepted: bool,
    errors: Vec<String>,
    plan: Option<PlanResult>,
}

#[derive(Debug, Serialize)]
struct ReportPayload {
    corpus_version: String,
    schema_id: String,
    schema_sha256: String,
    canonical_corpus_sha256: String,
    case_count: usize,
    planned_count: usize,
    rejected_count: usize,
    cases: Vec<CaseAudit>,
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

fn schema_validator(bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if schema.get("$id").and_then(Value::as_str) != Some(SCHEMA_ID) {
        return Err("schema identity does not match planner v0.1".to_owned());
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
    if scenario.schema_version != SCENARIO_VERSION
        || !nonempty(&scenario.scenario_id)
        || !nonempty(&scenario.decision_unit)
    {
        errors.push("scenario version, identity or decision unit is invalid".to_owned());
    }
    if scenario.tests.len() > MAX_HORIZON
        || scenario.horizon > MAX_HORIZON
        || scenario.horizon > scenario.tests.len()
    {
        errors.push("horizon exceeds the finite test catalog or v0.1 cap".to_owned());
    }
    if !scenario.abstain_loss.is_finite() || scenario.abstain_loss < 0.0 {
        errors.push("abstain loss must be finite and non-negative".to_owned());
    }

    let mut hypothesis_ids = BTreeSet::new();
    let mut prior_sum = 0.0;
    for hypothesis in &scenario.hypotheses {
        if !nonempty(&hypothesis.id) || !hypothesis_ids.insert(hypothesis.id.clone()) {
            errors.push(format!(
                "hypothesis identity is empty or duplicated: {}",
                hypothesis.id
            ));
        }
        if !hypothesis.prior.is_finite() || hypothesis.prior <= 0.0 || hypothesis.prior > 1.0 {
            errors.push(format!("hypothesis {} has invalid prior", hypothesis.id));
        }
        prior_sum += hypothesis.prior;
    }
    if scenario.hypotheses.is_empty() || (prior_sum - 1.0).abs() > 1e-9 {
        errors.push("hypothesis priors must be positive and sum to one".to_owned());
    }

    let mut action_ids = BTreeSet::new();
    for action in &scenario.actions {
        if !nonempty(&action.id) || !action_ids.insert(action.id.clone()) {
            errors.push(format!("action identity is empty or duplicated: {}", action.id));
        }
        if action.loss_by_hypothesis.len() != hypothesis_ids.len()
            || action
                .loss_by_hypothesis
                .keys()
                .any(|id| !hypothesis_ids.contains(id))
        {
            errors.push(format!(
                "action {} must define loss for every hypothesis",
                action.id
            ));
        }
        if action
            .loss_by_hypothesis
            .values()
            .any(|loss| !loss.is_finite() || *loss < 0.0)
        {
            errors.push(format!("action {} contains invalid loss", action.id));
        }
    }

    let mut test_ids = BTreeSet::new();
    for test in &scenario.tests {
        if !nonempty(&test.id) || !test_ids.insert(test.id.clone()) {
            errors.push(format!("test identity is empty or duplicated: {}", test.id));
        }
        if !test.test_cost.is_finite()
            || test.test_cost < 0.0
            || !test.downtime_cost.is_finite()
            || test.downtime_cost < 0.0
        {
            errors.push(format!("test {} has invalid cost", test.id));
        }
        if !is_digest(&test.prerequisites.protocol_digest)
            || !is_digest(&test.prerequisites.equipment_digest)
        {
            errors.push(format!(
                "test {} has invalid protocol/equipment digest",
                test.id
            ));
        }
        if test.outcomes.len() < 2 || test.outcomes.len() > MAX_OUTCOMES {
            errors.push(format!(
                "test {} must declare 2..={MAX_OUTCOMES} outcomes",
                test.id
            ));
        }
        let mut outcome_ids = BTreeSet::new();
        for outcome in &test.outcomes {
            if !nonempty(&outcome.id) || !outcome_ids.insert(outcome.id.clone()) {
                errors.push(format!("test {} has empty/duplicate outcome id", test.id));
            }
            if outcome.likelihood_by_hypothesis.len() != hypothesis_ids.len()
                || outcome
                    .likelihood_by_hypothesis
                    .keys()
                    .any(|id| !hypothesis_ids.contains(id))
            {
                errors.push(format!(
                    "test {} outcome {} must define every hypothesis likelihood",
                    test.id, outcome.id
                ));
            }
            if outcome
                .likelihood_by_hypothesis
                .values()
                .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
            {
                errors.push(format!(
                    "test {} outcome {} has invalid likelihood",
                    test.id, outcome.id
                ));
            }
        }
        for hypothesis_id in &hypothesis_ids {
            let sum: f64 = test
                .outcomes
                .iter()
                .filter_map(|outcome| outcome.likelihood_by_hypothesis.get(hypothesis_id))
                .sum();
            if (sum - 1.0).abs() > 1e-9 {
                errors.push(format!(
                    "test {} likelihoods for hypothesis {} sum to {sum}, not one",
                    test.id, hypothesis_id
                ));
            }
        }
    }

    for test in &scenario.tests {
        if let Some(condition) = &test.requires {
            if !test_ids.contains(&condition.test_id) || condition.test_id == test.id {
                errors.push(format!("test {} has invalid dependency", test.id));
                continue;
            }
            let dependency = scenario
                .tests
                .iter()
                .find(|candidate| candidate.id == condition.test_id);
            if dependency.is_none_or(|candidate| {
                !candidate.outcomes.iter().any(|outcome| outcome.id == condition.outcome)
            }) {
                errors.push(format!(
                    "test {} dependency references an unknown outcome",
                    test.id
                ));
            }
        }
    }
    if dependency_cycle(&scenario.tests) {
        errors.push("conditional test dependencies contain a cycle".to_owned());
    }
    errors
}

fn visit_dependency(
    id: &str,
    dependencies: &BTreeMap<String, String>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
) -> bool {
    if visited.contains(id) {
        return false;
    }
    if !visiting.insert(id.to_owned()) {
        return true;
    }
    if let Some(next) = dependencies.get(id) {
        if visit_dependency(next, dependencies, visiting, visited) {
            return true;
        }
    }
    visiting.remove(id);
    visited.insert(id.to_owned());
    false
}

fn dependency_cycle(tests: &[DiagnosticTest]) -> bool {
    let dependencies: BTreeMap<String, String> = tests
        .iter()
        .filter_map(|test| {
            test.requires
                .as_ref()
                .map(|condition| (test.id.clone(), condition.test_id.clone()))
        })
        .collect();
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    dependencies
        .keys()
        .any(|id| visit_dependency(id, &dependencies, &mut visiting, &mut visited))
}

fn terminal_action(scenario: &Scenario, posterior: &BTreeMap<String, f64>) -> (String, f64) {
    let mut best: Option<(String, f64)> = None;
    for action in scenario.actions.iter().filter(|action| action.admissible) {
        let expected = posterior
            .iter()
            .map(|(hypothesis, probability)| {
                probability * action.loss_by_hypothesis[hypothesis]
            })
            .sum::<f64>();
        if best
            .as_ref()
            .is_none_or(|(_, best_loss)| expected < *best_loss - EPSILON)
        {
            best = Some((action.id.clone(), expected));
        }
    }
    best.unwrap_or_else(|| ("ABSTAIN".to_owned(), scenario.abstain_loss))
}

fn test_is_admissible(
    test: &DiagnosticTest,
    history: &BTreeMap<String, String>,
    used: &BTreeSet<String>,
) -> bool {
    if used.contains(&test.id)
        || !test.prerequisites.safety_approved
        || !test.prerequisites.calibration_valid
        || !test.prerequisites.domain_supported
        || !test.prerequisites.equipment_available
    {
        return false;
    }
    match &test.requires {
        None => true,
        Some(condition) => history.get(&condition.test_id) == Some(&condition.outcome),
    }
}

fn optimize_node(
    scenario: &Scenario,
    posterior: &BTreeMap<String, f64>,
    history: &BTreeMap<String, String>,
    used: &BTreeSet<String>,
    remaining: usize,
    branch_probability: Option<f64>,
    node_budget: &mut usize,
) -> Result<PolicyNode, String> {
    if *node_budget == 0 {
        return Err(format!(
            "policy tree exceeded the v0.1 node cap of {MAX_POLICY_NODES}"
        ));
    }
    *node_budget -= 1;
    let (action, stop_loss) = terminal_action(scenario, posterior);
    let mut best = PolicyNode {
        branch_probability,
        posterior: posterior.clone(),
        choice: "NO_TEST".to_owned(),
        test_id: None,
        terminal_action: Some(action),
        expected_total_loss: stop_loss,
        immediate_test_cost: 0.0,
        expected_tests: 0.0,
        max_tests_on_path: 0,
        branches: BTreeMap::new(),
    };
    if remaining == 0 {
        return Ok(best);
    }

    for test in &scenario.tests {
        if !test_is_admissible(test, history, used) {
            continue;
        }
        let mut next_used = used.clone();
        next_used.insert(test.id.clone());
        let mut branches = BTreeMap::new();
        let mut expected_continuation_loss = 0.0;
        let mut expected_future_tests = 0.0;
        let mut maximum_future_tests = 0usize;

        for outcome in &test.outcomes {
            let marginal = posterior
                .iter()
                .map(|(hypothesis, probability)| {
                    probability * outcome.likelihood_by_hypothesis[hypothesis]
                })
                .sum::<f64>();
            if marginal <= EPSILON {
                continue;
            }
            let next_posterior: BTreeMap<String, f64> = posterior
                .iter()
                .map(|(hypothesis, probability)| {
                    (
                        hypothesis.clone(),
                        probability * outcome.likelihood_by_hypothesis[hypothesis] / marginal,
                    )
                })
                .collect();
            let mut next_history = history.clone();
            next_history.insert(test.id.clone(), outcome.id.clone());
            let child = optimize_node(
                scenario,
                &next_posterior,
                &next_history,
                &next_used,
                remaining - 1,
                Some(marginal),
                node_budget,
            )?;
            expected_continuation_loss += marginal * child.expected_total_loss;
            expected_future_tests += marginal * child.expected_tests;
            maximum_future_tests = maximum_future_tests.max(child.max_tests_on_path);
            branches.insert(outcome.id.clone(), child);
        }
        if branches.is_empty() {
            continue;
        }

        let immediate_cost = test.test_cost + test.downtime_cost;
        let candidate_loss = immediate_cost + expected_continuation_loss;
        if candidate_loss < best.expected_total_loss - EPSILON {
            best = PolicyNode {
                branch_probability,
                posterior: posterior.clone(),
                choice: "TEST".to_owned(),
                test_id: Some(test.id.clone()),
                terminal_action: None,
                expected_total_loss: candidate_loss,
                immediate_test_cost: immediate_cost,
                expected_tests: 1.0 + expected_future_tests,
                max_tests_on_path: 1 + maximum_future_tests,
                branches,
            };
        }
    }
    Ok(best)
}

fn one_step_candidates(
    scenario: &Scenario,
    prior: &BTreeMap<String, f64>,
    baseline: f64,
) -> Vec<OneStepValue> {
    let mut candidates = Vec::new();
    for test in &scenario.tests {
        let exclusion_reason = if !test.prerequisites.safety_approved {
            Some("SAFETY_APPROVAL_MISSING".to_owned())
        } else if !test.prerequisites.calibration_valid {
            Some("CALIBRATION_INVALID".to_owned())
        } else if !test.prerequisites.domain_supported {
            Some("DOMAIN_UNSUPPORTED".to_owned())
        } else if !test.prerequisites.equipment_available {
            Some("EQUIPMENT_UNAVAILABLE".to_owned())
        } else if test.requires.is_some() {
            Some("CONDITIONAL_DEPENDENCY_NOT_MET_AT_ROOT".to_owned())
        } else {
            None
        };
        if let Some(reason) = exclusion_reason {
            candidates.push(OneStepValue {
                test_id: test.id.clone(),
                admissible_at_root: false,
                exclusion_reason: Some(reason),
                expected_total_loss: None,
                net_value: None,
            });
            continue;
        }

        let mut expected_terminal_loss = 0.0;
        for outcome in &test.outcomes {
            let marginal = prior
                .iter()
                .map(|(hypothesis, probability)| {
                    probability * outcome.likelihood_by_hypothesis[hypothesis]
                })
                .sum::<f64>();
            if marginal <= EPSILON {
                continue;
            }
            let posterior = prior
                .iter()
                .map(|(hypothesis, probability)| {
                    (
                        hypothesis.clone(),
                        probability * outcome.likelihood_by_hypothesis[hypothesis] / marginal,
                    )
                })
                .collect::<BTreeMap<String, f64>>();
            let (_, risk) = terminal_action(scenario, &posterior);
            expected_terminal_loss += marginal * risk;
        }
        let total = test.test_cost + test.downtime_cost + expected_terminal_loss;
        candidates.push(OneStepValue {
            test_id: test.id.clone(),
            admissible_at_root: true,
            exclusion_reason: None,
            expected_total_loss: Some(total),
            net_value: Some(baseline - total),
        });
    }
    candidates.sort_by(|left, right| left.test_id.cmp(&right.test_id));
    candidates
}

fn plan_scenario(scenario: &Scenario) -> Result<PlanResult, String> {
    let errors = validate_scenario(scenario);
    if !errors.is_empty() {
        return Err(errors.join("; "));
    }
    let mut normalized = scenario.clone();
    normalized.hypotheses.sort_by(|left, right| left.id.cmp(&right.id));
    normalized.actions.sort_by(|left, right| left.id.cmp(&right.id));
    normalized.tests.sort_by(|left, right| left.id.cmp(&right.id));
    for test in &mut normalized.tests {
        test.outcomes.sort_by(|left, right| left.id.cmp(&right.id));
    }
    let prior: BTreeMap<String, f64> = normalized
        .hypotheses
        .iter()
        .map(|hypothesis| (hypothesis.id.clone(), hypothesis.prior))
        .collect();
    let (_, baseline) = terminal_action(&normalized, &prior);
    let one_step = one_step_candidates(&normalized, &prior, baseline);
    let mut node_budget = MAX_POLICY_NODES;
    let policy = optimize_node(
        &normalized,
        &prior,
        &BTreeMap::new(),
        &BTreeSet::new(),
        normalized.horizon,
        None,
        &mut node_budget,
    )?;
    let net_value = baseline - policy.expected_total_loss;
    if !net_value.is_finite() || !policy.expected_total_loss.is_finite() {
        return Err("planner produced non-finite expected loss".to_owned());
    }
    Ok(PlanResult {
        scenario_id: normalized.scenario_id,
        root_choice: policy.choice.clone(),
        root_test_id: policy.test_id.clone(),
        baseline_no_test_loss: baseline,
        optimized_expected_loss: policy.expected_total_loss,
        net_value,
        expected_tests: policy.expected_tests,
        max_tests_on_path: policy.max_tests_on_path,
        one_step_candidates: one_step,
        policy,
    })
}

fn canonicalize_corpus(value: &mut Value) {
    let Some(cases) = value.get_mut("cases").and_then(Value::as_array_mut) else {
        return;
    };
    cases.sort_by(|left, right| {
        left.get("case_id")
            .and_then(Value::as_str)
            .cmp(&right.get("case_id").and_then(Value::as_str))
    });
    for case in cases {
        if let Some(scenario) = case.get_mut("scenario") {
            if let Some(rows) = scenario.get_mut("hypotheses").and_then(Value::as_array_mut) {
                rows.sort_by(|left, right| {
                    left.get("id")
                        .and_then(Value::as_str)
                        .cmp(&right.get("id").and_then(Value::as_str))
                });
            }
            if let Some(rows) = scenario.get_mut("actions").and_then(Value::as_array_mut) {
                rows.sort_by(|left, right| {
                    left.get("id")
                        .and_then(Value::as_str)
                        .cmp(&right.get("id").and_then(Value::as_str))
                });
            }
            if let Some(tests) = scenario.get_mut("tests").and_then(Value::as_array_mut) {
                tests.sort_by(|left, right| {
                    left.get("id")
                        .and_then(Value::as_str)
                        .cmp(&right.get("id").and_then(Value::as_str))
                });
                for test in tests {
                    if let Some(outcomes) = test.get_mut("outcomes").and_then(Value::as_array_mut) {
                        outcomes.sort_by(|left, right| {
                            left.get("id")
                                .and_then(Value::as_str)
                                .cmp(&right.get("id").and_then(Value::as_str))
                        });
                    }
                }
            }
        }
    }
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<CorpusReport, String> {
    let validator = schema_validator(schema_bytes)?;
    let mut corpus_value: Value =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    canonicalize_corpus(&mut corpus_value);
    let canonical = serde_json::to_vec(&corpus_value)
        .map_err(|error| format!("canonical corpus: {error}"))?;
    let corpus: Corpus =
        serde_json::from_slice(&canonical).map_err(|error| format!("typed corpus: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION || corpus.schema_id != SCHEMA_ID {
        return Err("corpus version or schema identity mismatch".to_owned());
    }

    let mut seen = BTreeSet::new();
    let mut cases = Vec::new();
    let mut planned_count = 0usize;
    let mut rejected_count = 0usize;
    for case in corpus.cases {
        if !nonempty(&case.case_id) || !seen.insert(case.case_id.clone()) {
            return Err(format!("case identity is empty or duplicated: {}", case.case_id));
        }
        if !matches!(case.expectation.as_str(), "plan" | "reject") {
            return Err(format!("case {} has invalid expectation", case.case_id));
        }
        let schema_errors: Vec<String> = validator
            .iter_errors(&case.scenario)
            .map(|error| error.to_string())
            .collect();
        let outcome = if schema_errors.is_empty() {
            match serde_json::from_value::<Scenario>(case.scenario) {
                Ok(scenario) => match plan_scenario(&scenario) {
                    Ok(result) => Ok(result),
                    Err(error) => Err(vec![error]),
                },
                Err(error) => Err(vec![format!("typed scenario parse: {error}")]),
            }
        } else {
            Err(schema_errors)
        };
        let accepted = outcome.is_ok();
        if accepted != (case.expectation == "plan") {
            let detail = match &outcome {
                Ok(_) => "unexpected plan acceptance".to_owned(),
                Err(errors) => errors.join("; "),
            };
            return Err(format!(
                "case {} expectation mismatch: {}",
                case.case_id, detail
            ));
        }
        let (plan, errors) = match outcome {
            Ok(result) => {
                if case.expected_root_choice.as_deref() != Some(result.root_choice.as_str())
                    || case.expected_root_test_id != result.root_test_id
                    || case.expected_net_value.is_none_or(|expected| {
                        !expected.is_finite() || (result.net_value - expected).abs() > 1e-9
                    })
                {
                    return Err(format!(
                        "case {} known-truth planner disposition/value mismatch",
                        case.case_id
                    ));
                }
                planned_count += 1;
                (Some(result), Vec::new())
            }
            Err(errors) => {
                if case.expected_root_choice.is_some()
                    || case.expected_root_test_id.is_some()
                    || case.expected_net_value.is_some()
                {
                    return Err(format!(
                        "rejected case {} must not declare a plan result",
                        case.case_id
                    ));
                }
                rejected_count += 1;
                (None, errors)
            }
        };
        cases.push(CaseAudit {
            case_id: case.case_id,
            expectation: case.expectation,
            accepted,
            errors,
            plan,
        });
    }

    let payload = ReportPayload {
        corpus_version: CORPUS_VERSION.to_owned(),
        schema_id: SCHEMA_ID.to_owned(),
        schema_sha256: sha256(schema_bytes),
        canonical_corpus_sha256: sha256(&canonical),
        case_count: cases.len(),
        planned_count,
        rejected_count,
        cases,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("report serialization: {error}"))?;
    Ok(CorpusReport {
        payload,
        report_sha256: sha256(&bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_sequential_diagnostic_scenario_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/sequential_diagnostic_scenarios_v0_1.json");

    fn report() -> CorpusReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen sequential diagnostic corpus")
    }

    #[test]
    fn frozen_corpus_plans_five_cases_and_rejects_one() {
        let report = report();
        assert_eq!(report.payload.case_count, 7);
        assert_eq!(report.payload.planned_count, 6);
        assert_eq!(report.payload.rejected_count, 1);
    }

    #[test]
    fn complementary_tests_have_positive_sequence_value_when_each_is_greedy_negative() {
        let result = report();
        let planned = result
            .payload
            .cases
            .iter()
            .find(|case| case.case_id == "complementary-tests-positive-sequence")
            .and_then(|case| case.plan.as_ref())
            .expect("positive sequence plan");
        assert_eq!(planned.root_choice, "TEST");
        assert_eq!(planned.root_test_id.as_deref(), Some("test-a"));
        assert!((planned.net_value - 1.175).abs() < 1e-9);
        assert!((planned.expected_tests - 1.5).abs() < 1e-9);
        assert_eq!(planned.max_tests_on_path, 2);
        for test_id in ["test-a", "test-b"] {
            let candidate = planned.one_step_candidates.iter()
                .find(|candidate| candidate.test_id == test_id)
                .expect("one-step baseline candidate");
            assert!(candidate.admissible_at_root);
            assert!((candidate.net_value.unwrap() + 0.15).abs() < 1e-9);
        }
        assert_eq!(
            planned.policy.branches["a1"].branch_probability,
            Some(0.5)
        );
        assert!((planned.policy.branches["a1"].posterior["A1B1"] - 0.4).abs() < 1e-9);
    }

    #[test]
    fn conditional_test_only_appears_after_its_required_outcome() {
        let result = report();
        let planned = result
            .payload
            .cases
            .iter()
            .find(|case| case.case_id == "conditional-second-test-only-after-a1")
            .and_then(|case| case.plan.as_ref())
            .expect("conditional sequence plan");
        assert_eq!(planned.policy.test_id.as_deref(), Some("test-a"));
        assert_eq!(planned.policy.branches["a0"].choice, "NO_TEST");
        assert_eq!(planned.policy.branches["a1"].test_id.as_deref(), Some("test-b"));
    }

    #[test]
    fn high_information_with_zero_decision_value_selects_no_test() {
        let result = report();
        let planned = result
            .payload
            .cases
            .iter()
            .find(|case| case.case_id == "high-information-zero-decision-value")
            .and_then(|case| case.plan.as_ref())
            .expect("zero value plan");
        assert_eq!(planned.root_choice, "NO_TEST");
        assert_eq!(planned.root_test_id, None);
        assert_eq!(planned.net_value, 0.0);
    }

    #[test]
    fn missing_safety_approval_blocks_a_high_value_test() {
        let result = report();
        let planned = result
            .payload
            .cases
            .iter()
            .find(|case| case.case_id == "safety-prerequisite-blocks-high-value-test")
            .and_then(|case| case.plan.as_ref())
            .expect("safety control plan");
        assert_eq!(planned.root_choice, "NO_TEST");
        assert_eq!(planned.root_test_id, None);
    }

    #[test]
    fn no_admissible_terminal_action_returns_abstain() {
        let result = report();
        let planned = result.payload.cases.iter()
            .find(|case| case.case_id == "no-admissible-terminal-action-abstains")
            .and_then(|case| case.plan.as_ref())
            .expect("abstention plan");
        assert_eq!(planned.root_choice, "NO_TEST");
        assert_eq!(planned.policy.terminal_action.as_deref(), Some("ABSTAIN"));
        assert_eq!(planned.optimized_expected_loss, 12.0);
    }

    #[test]
    fn policy_is_invariant_to_hypothesis_action_test_and_outcome_order() {
        let original = report();
        let mut value: Value = serde_json::from_slice(CORPUS).expect("corpus JSON");
        value["cases"].as_array_mut().unwrap().reverse();
        for case in value["cases"].as_array_mut().unwrap() {
            case["scenario"]["hypotheses"].as_array_mut().unwrap().reverse();
            case["scenario"]["actions"].as_array_mut().unwrap().reverse();
            case["scenario"]["tests"].as_array_mut().unwrap().reverse();
            for test in case["scenario"]["tests"].as_array_mut().unwrap() {
                test["outcomes"].as_array_mut().unwrap().reverse();
            }
        }
        let bytes = serde_json::to_vec(&value).expect("permuted corpus");
        let replay = audit_corpus(SCHEMA, &bytes).expect("permuted audit");
        assert_eq!(
            original.payload.canonical_corpus_sha256,
            replay.payload.canonical_corpus_sha256
        );
        assert_eq!(original.report_sha256, replay.report_sha256);
    }
}
