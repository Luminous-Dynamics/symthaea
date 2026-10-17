use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use crate::{evaluate_trace, text_at, validate_trace};

const DECISION_SCHEMA_ID: &str =
    "urn:luminous-dynamics:appliance-decision-value-corpus:0.1.0";
const EXPECTED_CASE_IDS: [&str; 8] = [
    "DVAL-001",
    "DVAL-002",
    "DVAL-003",
    "DVAL-004",
    "DVAL-005",
    "DVAL-006",
    "DVAL-007",
    "DVAL-008",
];

type Errors = Vec<String>;

#[derive(Debug, Clone)]
struct Action {
    id: String,
    eligible: bool,
    losses: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct DecisionValueResult {
    decision: String,
    reason_code: String,
    selected_measurement_id: Option<String>,
    best_candidate_measurement_id: Option<String>,
    prior_action_id: Option<String>,
    prior_risk_units: Option<f64>,
    best_candidate_posterior_risk_units: Option<f64>,
    best_candidate_evsi_units: Option<f64>,
    best_candidate_cost_units: Option<f64>,
    best_candidate_net_value_units: Option<f64>,
    entropy_baseline_measurement_id: Option<String>,
}

impl DecisionValueResult {
    fn empty(decision: &str, reason_code: &str) -> Self {
        Self {
            decision: decision.to_owned(),
            reason_code: reason_code.to_owned(),
            selected_measurement_id: None,
            best_candidate_measurement_id: None,
            prior_action_id: None,
            prior_risk_units: None,
            best_candidate_posterior_risk_units: None,
            best_candidate_evsi_units: None,
            best_candidate_cost_units: None,
            best_candidate_net_value_units: None,
            entropy_baseline_measurement_id: None,
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "decision": self.decision,
            "reason_code": self.reason_code,
            "selected_measurement_id": self.selected_measurement_id,
            "best_candidate_measurement_id": self.best_candidate_measurement_id,
            "prior_action_id": self.prior_action_id,
            "prior_risk_units": self.prior_risk_units,
            "best_candidate_posterior_risk_units": self.best_candidate_posterior_risk_units,
            "best_candidate_evsi_units": self.best_candidate_evsi_units,
            "best_candidate_cost_units": self.best_candidate_cost_units,
            "best_candidate_net_value_units": self.best_candidate_net_value_units,
            "entropy_baseline_measurement_id": self.entropy_baseline_measurement_id
        })
    }
}

fn decision_schema_validator(schema: &Value) -> Result<jsonschema::Validator, String> {
    if text_at(schema, "$schema") != Some("https://json-schema.org/draft/2020-12/schema") {
        return Err("decision schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if text_at(schema, "$id") != Some(DECISION_SCHEMA_ID) {
        return Err(format!("decision schema $id must be {DECISION_SCHEMA_ID}"));
    }
    if !jsonschema::draft202012::meta::is_valid(schema) {
        return Err(
            "decision schema does not validate against the Draft 2020-12 meta-schema".to_owned(),
        );
    }
    jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(schema)
        .map_err(|error| format!("cannot compile decision schema: {error}"))
}

fn number_at(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn case_by_id<'a>(cases: &'a [Value], id: &str) -> Option<&'a Value> {
    cases
        .iter()
        .find(|case| text_at(case, "case_id") == Some(id))
}

fn trace_by_id<'a>(traces: &'a [Value], id: &str) -> Option<&'a Value> {
    traces
        .iter()
        .find(|trace| text_at(trace, "trace_id") == Some(id))
}

fn parse_actions(case: &Value, trace: &Value) -> Result<Vec<Action>, String> {
    let Some(actions) = case
        .get("decision_model")
        .and_then(|model| model.get("actions"))
        .and_then(Value::as_array)
    else {
        return Err("invalid_action_loss_matrix".to_owned());
    };
    let hypothesis_ids: BTreeSet<String> = trace["hypotheses"]
        .as_array()
        .ok_or_else(|| "invalid_action_loss_matrix".to_owned())?
        .iter()
        .filter_map(|hypothesis| text_at(hypothesis, "hypothesis_id").map(str::to_owned))
        .collect();
    if hypothesis_ids.is_empty() {
        return Err("invalid_action_loss_matrix".to_owned());
    }

    let mut action_ids = BTreeSet::new();
    let mut parsed = Vec::with_capacity(actions.len());
    for action in actions {
        let Some(id) = text_at(action, "action_id") else {
            return Err("invalid_action_loss_matrix".to_owned());
        };
        if !action_ids.insert(id.to_owned()) {
            return Err("duplicate_identity".to_owned());
        }
        let eligibility = text_at(action, "eligibility").unwrap_or("");
        if !matches!(
            eligibility,
            "eligible" | "requires_qualified_person" | "prohibited"
        ) {
            return Err("invalid_action_loss_matrix".to_owned());
        }
        let Some(losses) = action.get("losses").and_then(Value::as_array) else {
            return Err("invalid_action_loss_matrix".to_owned());
        };
        if losses.len() != hypothesis_ids.len() {
            return Err("invalid_action_loss_matrix".to_owned());
        }

        let mut loss_map = BTreeMap::new();
        for loss in losses {
            let Some(hypothesis_id) = text_at(loss, "hypothesis_id") else {
                return Err("invalid_action_loss_matrix".to_owned());
            };
            let Some(loss_units) = number_at(loss, "loss_units") else {
                return Err("invalid_action_loss_matrix".to_owned());
            };
            if !hypothesis_ids.contains(hypothesis_id)
                || !loss_units.is_finite()
                || loss_units < 0.0
                || loss_map
                    .insert(hypothesis_id.to_owned(), loss_units)
                    .is_some()
            {
                return Err("invalid_action_loss_matrix".to_owned());
            }
        }
        if loss_map.keys().cloned().collect::<BTreeSet<_>>() != hypothesis_ids {
            return Err("invalid_action_loss_matrix".to_owned());
        }
        parsed.push(Action {
            id: id.to_owned(),
            eligible: eligibility == "eligible",
            losses: loss_map,
        });
    }
    Ok(parsed)
}

fn parse_measurement_costs(
    case: &Value,
    trace: &Value,
) -> Result<BTreeMap<String, f64>, String> {
    let Some(costs) = case.get("measurement_costs").and_then(Value::as_array) else {
        return Err("invalid_measurement_cost".to_owned());
    };
    let expected_ids: BTreeSet<String> = trace["candidate_measurements"]
        .as_array()
        .ok_or_else(|| "invalid_measurement_cost".to_owned())?
        .iter()
        .filter_map(|measurement| {
            text_at(measurement, "measurement_id").map(str::to_owned)
        })
        .collect();
    let mut cost_map = BTreeMap::new();
    for cost in costs {
        let Some(id) = text_at(cost, "measurement_id") else {
            return Err("invalid_measurement_cost".to_owned());
        };
        let Some(loss_units) = number_at(cost, "loss_units") else {
            return Err("invalid_measurement_cost".to_owned());
        };
        if !loss_units.is_finite()
            || loss_units < 0.0
            || cost_map.insert(id.to_owned(), loss_units).is_some()
        {
            return Err("duplicate_identity".to_owned());
        }
    }
    if cost_map.keys().cloned().collect::<BTreeSet<_>>() != expected_ids {
        return Err("invalid_measurement_cost".to_owned());
    }
    Ok(cost_map)
}

fn expected_risk(
    actions: &[Action],
    probabilities: &BTreeMap<String, f64>,
) -> (String, f64) {
    let mut candidates = Vec::new();
    for action in actions.iter().filter(|action| action.eligible) {
        let risk = probabilities
            .iter()
            .map(|(hypothesis_id, probability)| {
                probability * action.losses.get(hypothesis_id).copied().unwrap_or(f64::INFINITY)
            })
            .sum::<f64>();
        candidates.push((action.id.as_str(), risk));
    }
    candidates.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(right.0))
    });
    let (action_id, risk) = candidates
        .first()
        .expect("the caller checks that one eligible action exists");
    let _ = trace;
    ((*action_id).to_owned(), *risk)
}

fn posterior_expected_risk(
    trace: &Value,
    actions: &[Action],
    measurement: &Value,
) -> Result<f64, String> {
    let hypotheses = trace["hypotheses"]
        .as_array()
        .ok_or_else(|| "invalid_prior_distribution".to_owned())?;
    let outcomes = measurement["outcome_ids"]
        .as_array()
        .ok_or_else(|| "invalid_likelihood_model".to_owned())?;
    let mut total = 0.0;

    for outcome in outcomes {
        let Some(outcome_id) = outcome.as_str() else {
            return Err("invalid_likelihood_model".to_owned());
        };
        let mut probability_of_outcome = 0.0;
        for hypothesis in hypotheses {
            let Some(hypothesis_id) = text_at(hypothesis, "hypothesis_id") else {
                return Err("invalid_prior_distribution".to_owned());
            };
            let Some(prior) = number_at(hypothesis, "prior_probability") else {
                return Err("invalid_prior_distribution".to_owned());
            };
            let likelihood = measurement["likelihoods"]
                .as_array()
                .and_then(|rows| {
                    rows.iter().find(|row| {
                        text_at(row, "hypothesis_id") == Some(hypothesis_id)
                    })
                })
                .and_then(|row| row["probabilities"].as_array())
                .and_then(|probabilities| {
                    probabilities
                        .iter()
                        .find(|probability| {
                            text_at(probability, "outcome_id") == Some(outcome_id)
                        })
                })
                .and_then(|probability| number_at(probability, "probability"))
                .ok_or_else(|| "invalid_likelihood_model".to_owned())?;
            probability_of_outcome += prior * likelihood;
        }
        if probability_of_outcome <= 0.0 {
            continue;
        }

        let mut posterior = BTreeMap::new();
        for hypothesis in hypotheses {
            let hypothesis_id = text_at(hypothesis, "hypothesis_id")
                .ok_or_else(|| "invalid_prior_distribution".to_owned())?;
            let prior = number_at(hypothesis, "prior_probability")
                .ok_or_else(|| "invalid_prior_distribution".to_owned())?;
            let likelihood = measurement["likelihoods"]
                .as_array()
                .and_then(|rows| {
                    rows.iter().find(|row| {
                        text_at(row, "hypothesis_id") == Some(hypothesis_id)
                    })
                })
                .and_then(|row| row["probabilities"].as_array())
                .and_then(|probabilities| {
                    probabilities
                        .iter()
                        .find(|probability| {
                            text_at(probability, "outcome_id") == Some(outcome_id)
                        })
                })
                .and_then(|probability| number_at(probability, "probability"))
                .ok_or_else(|| "invalid_likelihood_model".to_owned())?;
            posterior.insert(
                hypothesis_id.to_owned(),
                prior * likelihood / probability_of_outcome,
            );
        }
        let (_, outcome_risk) = expected_risk(actions, &posterior);
        total += probability_of_outcome * outcome_risk;
    }

    if total.is_finite() {
        Ok(total)
    } else {
        Err("invalid_action_loss_matrix".to_owned())
    }
}

fn base_entropy_choice(trace: &Value) -> Option<String> {
    let decision = evaluate_trace(trace);
    if decision.decision == "select_measurement" {
        decision.next_measurement_id
    } else {
        None
    }
}

fn evaluate_case(trace: &Value, case: &Value) -> Result<DecisionValueResult, String> {
    validate_trace(trace).map_err(|_| "invalid_base_trace".to_owned())?;
    let actions = parse_actions(case, trace)?;
    let costs = parse_measurement_costs(case, trace)?;
    let entropy_choice = base_entropy_choice(trace);

    if text_at(trace, "protection_state") == Some("tripped") {
        return Ok(DecisionValueResult::empty(
            "abstain_and_review",
            "protection_tripped",
        ));
    }
    if text_at(trace, "protection_state") == Some("unknown") {
        return Ok(DecisionValueResult::empty(
            "abstain_and_review",
            "protection_unknown",
        ));
    }
    if text_at(trace, "observation_status") != Some("complete") {
        return Ok(DecisionValueResult::empty(
            "abstain_and_review",
            "observations_unreliable",
        ));
    }

    let eligible_actions: Vec<Action> = actions
        .into_iter()
        .filter(|action| action.eligible)
        .collect();
    if eligible_actions.is_empty() {
        let mut result = DecisionValueResult::empty("no_safe_action", "no_safe_action");
        result.entropy_baseline_measurement_id = entropy_choice;
        return Ok(result);
    }

    let priors: BTreeMap<String, f64> = trace["hypotheses"]
        .as_array()
        .ok_or_else(|| "invalid_prior_distribution".to_owned())?
        .iter()
        .map(|hypothesis| {
            Ok((
                text_at(hypothesis, "hypothesis_id")
                    .ok_or_else(|| "invalid_prior_distribution".to_owned())?
                    .to_owned(),
                number_at(hypothesis, "prior_probability")
                    .ok_or_else(|| "invalid_prior_distribution".to_owned())?,
            ))
        })
        .collect::<Result<_, String>>()?;
    let (prior_action, prior_risk) = expected_risk(&eligible_actions, &priors);
    let available_tests: Vec<&Value> = trace["candidate_measurements"]
        .as_array()
        .ok_or_else(|| "invalid_likelihood_model".to_owned())?
        .iter()
        .filter(|measurement| {
            text_at(measurement, "risk_class") == Some("low")
                && measurement.get("available").and_then(Value::as_bool) == Some(true)
        })
        .collect();

    if available_tests.is_empty() {
        let mut result = DecisionValueResult::empty("no_safe_candidate", "no_safe_candidate");
        result.prior_action_id = Some(prior_action);
        result.prior_risk_units = Some(prior_risk);
        return Ok(result);
    }

    let mut candidates = Vec::new();
    for measurement in available_tests {
        let id = text_at(measurement, "measurement_id")
            .ok_or_else(|| "invalid_likelihood_model".to_owned())?;
        let cost = costs
            .get(id)
            .copied()
            .ok_or_else(|| "invalid_measurement_cost".to_owned())?;
        let posterior_risk =
            posterior_expected_risk(trace, &eligible_actions, measurement)?;
        let raw_evsi = prior_risk - posterior_risk;
        if raw_evsi < -1e-8 {
            return Err("computed expected posterior risk exceeds prior risk".to_owned());
        }
        let evsi = raw_evsi.max(0.0);
        candidates.push((
            id.to_owned(),
            posterior_risk,
            evsi,
            cost,
            evsi - cost,
        ));
    }
    candidates.sort_by(|left, right| {
        right
            .4
            .total_cmp(&left.4)
            .then_with(|| left.0.cmp(&right.0))
    });
    let (best_id, posterior_risk, evsi, cost, net_value) = candidates
        .first()
        .expect("the caller checks that one test is available")
        .clone();

    let decision = if net_value > 1e-9 {
        "select_measurement"
    } else {
        "choose_action_without_test"
    };
    let reason = if net_value > 1e-9 {
        "positive_net_decision_value"
    } else {
        "all_tests_nonpositive_net_value"
    };

    Ok(DecisionValueResult {
        decision: decision.to_owned(),
        reason_code: reason.to_owned(),
        selected_measurement_id: (decision == "select_measurement").then(|| best_id.clone()),
        best_candidate_measurement_id: Some(best_id),
        prior_action_id: Some(prior_action),
        prior_risk_units: Some(prior_risk),
        best_candidate_posterior_risk_units: Some(posterior_risk),
        best_candidate_evsi_units: Some(evsi),
        best_candidate_cost_units: Some(cost),
        best_candidate_net_value_units: Some(net_value),
        entropy_baseline_measurement_id: entropy_choice,
    })
}

fn expected_matches(actual: &DecisionValueResult, expected: &Value) -> Result<(), String> {
    fn expected_string(value: &Value, key: &str) -> Result<Option<String>, String> {
        match value.get(key) {
            Some(item) if item.is_null() => Ok(None),
            Some(item) => item
                .as_str()
                .map(|text| Some(text.to_owned()))
                .ok_or_else(|| format!("expected.{key} must be a string or null")),
            None => Err(format!("expected.{key} is missing")),
        }
    }
    fn expected_number(value: &Value, key: &str) -> Result<Option<f64>, String> {
        match value.get(key) {
            Some(item) if item.is_null() => Ok(None),
            Some(item) => item
                .as_f64()
                .map(Some)
                .ok_or_else(|| format!("expected.{key} must be a number or null")),
            None => Err(format!("expected.{key} is missing")),
        }
    }
    fn compare_number(
        key: &str,
        actual: Option<f64>,
        expected: Option<f64>,
    ) -> Result<(), String> {
        match (actual, expected) {
            (None, None) => Ok(()),
            (Some(a), Some(e)) if a.is_finite() && e.is_finite() && (a - e).abs() <= 1e-8 => {
                Ok(())
            }
            _ => Err(format!("{key}: actual {actual:?} does not match expected {expected:?}")),
        }
    }

    let decision = text_at(expected, "decision")
        .ok_or_else(|| "expected.decision is missing".to_owned())?;
    let reason = text_at(expected, "reason_code")
        .ok_or_else(|| "expected.reason_code is missing".to_owned())?;
    if decision != actual.decision {
        return Err(format!(
            "decision expected {decision}, actual {}",
            actual.decision
        ));
    }
    if reason != actual.reason_code {
        return Err(format!(
            "reason_code expected {reason}, actual {}",
            actual.reason_code
        ));
    }

    for (key, actual_value) in [
        ("selected_measurement_id", &actual.selected_measurement_id),
        (
            "best_candidate_measurement_id",
            &actual.best_candidate_measurement_id,
        ),
        ("prior_action_id", &actual.prior_action_id),
        (
            "entropy_baseline_measurement_id",
            &actual.entropy_baseline_measurement_id,
        ),
    ] {
        let expected_value = expected_string(expected, key)?;
        if *actual_value != expected_value {
            return Err(format!(
                "{key}: actual {actual_value:?} does not match expected {expected_value:?}"
            ));
        }
    }
    for (key, actual_value) in [
        ("prior_risk_units", actual.prior_risk_units),
        (
            "best_candidate_posterior_risk_units",
            actual.best_candidate_posterior_risk_units,
        ),
        ("best_candidate_evsi_units", actual.best_candidate_evsi_units),
        ("best_candidate_cost_units", actual.best_candidate_cost_units),
        (
            "best_candidate_net_value_units",
            actual.best_candidate_net_value_units,
        ),
    ] {
        compare_number(key, actual_value, expected_number(expected, key)?)?;
    }
    Ok(())
}

fn audit_values(
    base_schema: &Value,
    base_corpus: &Value,
    decision_schema: &Value,
    decision_corpus: &Value,
    base_corpus_raw: &str,
    decision_schema_raw: &str,
    decision_corpus_raw: &str,
) -> Result<Value, Errors> {
    let _ = base_schema;
    let actual_base_sha = format!("{:x}", Sha256::digest(base_corpus_raw.as_bytes()));
    if text_at(decision_corpus, "base_corpus_sha256") != Some(actual_base_sha.as_str()) {
        return Err(vec![format!(
            "base corpus SHA-256 mismatch: fixture declares {:?}, actual is {actual_base_sha}",
            text_at(decision_corpus, "base_corpus_sha256")
        )]);
    }

    let decision_validator =
        decision_schema_validator(decision_schema).map_err(|error| vec![error])?;
    let schema_errors: Vec<String> = decision_validator
        .iter_errors(decision_corpus)
        .map(|error| error.to_string())
        .collect();
    if !schema_errors.is_empty() {
        return Err(schema_errors);
    }
    let traces = base_corpus
        .get("traces")
        .and_then(Value::as_array)
        .ok_or_else(|| vec!["base corpus.traces must be an array".to_owned()])?;
    let cases = decision_corpus
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| vec!["decision corpus.cases must be an array".to_owned()])?;

    let expected_ids: BTreeSet<String> = EXPECTED_CASE_IDS
        .iter()
        .map(|id| (*id).to_owned())
        .collect();
    let mut seen_ids = BTreeSet::new();
    let mut selected = 0usize;
    let mut chose_without_test = 0usize;
    let mut abstained = 0usize;
    let mut no_safe_candidate = 0usize;
    let mut no_safe_action = 0usize;
    let mut rejected = 0usize;
    let mut errors = Vec::new();

    for case in cases {
        let Some(case_id) = text_at(case, "case_id") else {
            errors.push("decision case_id must be a string".to_owned());
            continue;
        };
        if !seen_ids.insert(case_id.to_owned()) {
            errors.push(format!("duplicate decision case_id {case_id}"));
            continue;
        }
        let Some(base_trace_id) = text_at(case, "base_trace_id") else {
            errors.push(format!("{case_id}: base_trace_id is missing"));
            continue;
        };
        let Some(trace) = trace_by_id(traces, base_trace_id) else {
            errors.push(format!("{case_id}: base trace {base_trace_id} does not exist"));
            continue;
        };
        let actual = match evaluate_case(trace, case) {
            Ok(result) => result,
            Err(reason_code) => DecisionValueResult::empty("reject_invalid_model", &reason_code),
        };
        let Some(expected) = case.get("expected") else {
            errors.push(format!("{case_id}: expected object is missing"));
            continue;
        };
        if let Err(error) = expected_matches(&actual, expected) {
            errors.push(format!("{case_id}: {error}"));
        }

        match actual.decision.as_str() {
            "select_measurement" => selected += 1,
            "choose_action_without_test" => chose_without_test += 1,
            "abstain_and_review" => abstained += 1,
            "no_safe_candidate" => no_safe_candidate += 1,
            "no_safe_action" => no_safe_action += 1,
            "reject_invalid_model" => rejected += 1,
            other => errors.push(format!("{case_id}: unsupported decision {other}")),
        }
    }

    for missing in expected_ids.difference(&seen_ids) {
        errors.push(format!("required decision case {missing} is missing"));
    }
    for unexpected in seen_ids.difference(&expected_ids) {
        errors.push(format!("unexpected decision case {unexpected}"));
    }
    if cases.len() != EXPECTED_CASE_IDS.len() {
        errors.push(format!(
            "decision corpus has {} cases; expected {}",
            cases.len(),
            EXPECTED_CASE_IDS.len()
        ));
    }

    if selected != 2
        || chose_without_test != 1
        || abstained != 2
        || no_safe_candidate != 1
        || no_safe_action != 1
        || rejected != 1
    {
        errors.push(format!(
            "decision census changed: selected={selected}, no_test={chose_without_test}, abstained={abstained}, no_safe_candidate={no_safe_candidate}, no_safe_action={no_safe_action}, rejected={rejected}"
        ));
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(json!({
        "report_version": "appliance-decision-value-audit/0.1",
        "status": "PASS_SYNTHETIC_DECISION_VALUE_ONLY",
        "evidence_class": "synthetic_decision_model_only",
        "physical_decision_benefit_established": false,
        "product_safety_assessed": false,
        "case_count": cases.len(),
        "selected_measurement": selected,
        "choose_action_without_test": chose_without_test,
        "abstained": abstained,
        "no_safe_candidate": no_safe_candidate,
        "no_safe_action": no_safe_action,
        "rejected_model": rejected,
        "base_corpus_sha256": actual_base_sha,
        "decision_schema_sha256": format!("{:x}", Sha256::digest(decision_schema_raw.as_bytes())),
        "decision_corpus_sha256": format!("{:x}", Sha256::digest(decision_corpus_raw.as_bytes()))
    }))
}

pub(super) fn audit_decision_texts(
    base_schema_raw: &str,
    base_corpus_raw: &str,
    decision_schema_raw: &str,
    decision_corpus_raw: &str,
) -> Result<Value, Errors> {
    crate::audit_texts(base_schema_raw, base_corpus_raw).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| format!("base corpus: {error}"))
            .collect::<Vec<_>>()
    })?;
    let base_schema: Value = serde_json::from_str(base_schema_raw)
        .map_err(|error| vec![format!("base schema JSON parse failed: {error}")])?;
    let base_corpus: Value = serde_json::from_str(base_corpus_raw)
        .map_err(|error| vec![format!("base corpus JSON parse failed: {error}")])?;
    let decision_schema: Value = serde_json::from_str(decision_schema_raw)
        .map_err(|error| vec![format!("decision schema JSON parse failed: {error}")])?;
    let decision_corpus: Value = serde_json::from_str(decision_corpus_raw)
        .map_err(|error| vec![format!("decision corpus JSON parse failed: {error}")])?;
    audit_values(
        &base_schema,
        &base_corpus,
        &decision_schema,
        &decision_corpus,
        base_corpus_raw,
        decision_schema_raw,
        decision_corpus_raw,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_SCHEMA: &str = include_str!(
        "../../../schemas/diagnostics/appliance_active_diagnostic_corpus_v0_1.schema.json"
    );
    const BASE_CORPUS: &str = include_str!(
        "../../../fixtures/diagnostics/appliance_active_diagnostic_corpus_v0_1.json"
    );
    const DECISION_SCHEMA: &str = include_str!(
        "../../../schemas/diagnostics/appliance_decision_value_corpus_v0_1.schema.json"
    );
    const DECISION_CORPUS: &str = include_str!(
        "../../../fixtures/diagnostics/appliance_decision_value_corpus_v0_1.json"
    );

    #[test]
    fn canonical_decision_corpus_matches_expected_actions_and_losses() {
        let report = audit_decision_texts(
            BASE_SCHEMA,
            BASE_CORPUS,
            DECISION_SCHEMA,
            DECISION_CORPUS,
        )
        .expect("decision corpus must qualify");
        assert_eq!(report["case_count"], 8);
        assert_eq!(report["selected_measurement"], 2);
        assert_eq!(report["choose_action_without_test"], 1);
        assert_eq!(report["abstained"], 2);
        assert_eq!(report["no_safe_candidate"], 1);
        assert_eq!(report["no_safe_action"], 1);
        assert_eq!(report["rejected_model"], 1);
    }

    #[test]
    fn decision_value_can_choose_a_different_test_than_entropy() {
        let corpus: Value = serde_json::from_str(DECISION_CORPUS).expect("decision corpus");
        let cases = corpus["cases"].as_array().expect("cases");
        let case = case_by_id(cases, "DVAL-001").expect("DVAL-001");
        let trace_id = text_at(case, "base_trace_id").expect("base trace id");
        let base: Value = serde_json::from_str(BASE_CORPUS).expect("base corpus");
        let trace = trace_by_id(base["traces"].as_array().expect("traces"), trace_id)
            .expect("base trace");
        let result = evaluate_case(trace, case).expect("decision case");
        assert_eq!(result.selected_measurement_id.as_deref(), Some("vibration_check"));
        assert_eq!(
            result.entropy_baseline_measurement_id.as_deref(),
            Some("rpm_check")
        );
        assert!((result.prior_risk_units.unwrap() - 2.25).abs() < 1e-8);
        assert!((result.best_candidate_posterior_risk_units.unwrap() - 1.5).abs() < 1e-8);
        assert!((result.best_candidate_evsi_units.unwrap() - 0.75).abs() < 1e-8);
        assert!((result.best_candidate_cost_units.unwrap() - 0.5).abs() < 1e-8);
        assert!((result.best_candidate_net_value_units.unwrap() - 0.25).abs() < 1e-8);
    }

    #[test]
    fn nonpositive_net_value_selects_no_test() {
        let corpus: Value = serde_json::from_str(DECISION_CORPUS).expect("decision corpus");
        let case = case_by_id(corpus["cases"].as_array().expect("cases"), "DVAL-002")
            .expect("DVAL-002");
        let base: Value = serde_json::from_str(BASE_CORPUS).expect("base corpus");
        let trace = trace_by_id(
            base["traces"].as_array().expect("traces"),
            text_at(case, "base_trace_id").expect("base trace id"),
        )
        .expect("base trace");
        let result = evaluate_case(trace, case).expect("decision case");
        assert_eq!(result.decision, "choose_action_without_test");
        assert_eq!(result.selected_measurement_id, None);
        assert!((result.best_candidate_net_value_units.unwrap() + 1.25).abs() < 1e-8);
    }

    #[test]
    fn invalid_base_digest_blocks_decision_scoring() {
        let mut corpus: Value = serde_json::from_str(DECISION_CORPUS).expect("decision corpus");
        corpus["base_corpus_sha256"] = json!("00");
        let raw = serde_json::to_string(&corpus).expect("serialize corpus");
        let err = audit_decision_texts(BASE_SCHEMA, BASE_CORPUS, DECISION_SCHEMA, &raw)
            .expect_err("baseline digest mismatch must fail");
        assert!(err.iter().any(|error| error.contains("base corpus SHA-256 mismatch")));
    }

    #[test]
    fn negative_action_loss_is_rejected_by_semantic_validation() {
        let mut corpus: Value = serde_json::from_str(DECISION_CORPUS).expect("decision corpus");
        corpus["cases"][0]["decision_model"]["actions"][0]["losses"][0]["loss_units"] =
            json!(-1.0);
        let raw = serde_json::to_string(&corpus).expect("serialize corpus");
        let err = audit_decision_texts(BASE_SCHEMA, BASE_CORPUS, DECISION_SCHEMA, &raw)
            .expect_err("negative loss must fail");
        assert!(err.iter().any(|error| error.contains("invalid_action_loss_matrix")));
    }

    #[test]
    fn duplicate_measurement_cost_identity_is_rejected() {
        let mut corpus: Value = serde_json::from_str(DECISION_CORPUS).expect("decision corpus");
        corpus["cases"][0]["measurement_costs"][1]["measurement_id"] =
            corpus["cases"][0]["measurement_costs"][0]["measurement_id"].clone();
        let raw = serde_json::to_string(&corpus).expect("serialize corpus");
        let err = audit_decision_texts(BASE_SCHEMA, BASE_CORPUS, DECISION_SCHEMA, &raw)
            .expect_err("duplicate cost identity must fail");
        assert!(err.iter().any(|error| error.contains("duplicate_identity")));
    }

    #[test]
    fn protection_trip_overrides_positive_decision_value() {
        let base: Value = serde_json::from_str(BASE_CORPUS).expect("base corpus");
        let corpus: Value = serde_json::from_str(DECISION_CORPUS).expect("decision corpus");
        let case = case_by_id(corpus["cases"].as_array().expect("cases"), "DVAL-003")
            .expect("DVAL-003");
        let trace = trace_by_id(
            base["traces"].as_array().expect("traces"),
            text_at(case, "base_trace_id").expect("base trace id"),
        )
        .expect("base trace");
        let result = evaluate_case(trace, case).expect("decision case");
        assert_eq!(result.decision, "abstain_and_review");
        assert_eq!(result.reason_code, "protection_tripped");
        assert_eq!(result.selected_measurement_id, None);
    }

    #[test]
    fn tied_perfect_measurements_use_lexical_id() {
        let base: Value = serde_json::from_str(BASE_CORPUS).expect("base corpus");
        let corpus: Value = serde_json::from_str(DECISION_CORPUS).expect("decision corpus");
        let case = case_by_id(corpus["cases"].as_array().expect("cases"), "DVAL-006")
            .expect("DVAL-006");
        let trace = trace_by_id(
            base["traces"].as_array().expect("traces"),
            text_at(case, "base_trace_id").expect("base trace id"),
        )
        .expect("base trace");
        let result = evaluate_case(trace, case).expect("decision case");
        assert_eq!(result.selected_measurement_id.as_deref(), Some("a_rpm_check"));
        assert!((result.best_candidate_evsi_units.unwrap() - 5.0).abs() < 1e-8);
    }
}
