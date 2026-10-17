use jsonschema::Validator;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const SCHEMA_ID: &str = "urn:luminous-dynamics:appliance-active-diagnostics:0.1.0";
const EXPECTED_TRACE_IDS: [&str; 21] = [
    "ADT-001", "ADT-002", "ADT-003", "ADT-004", "ADT-005", "ADT-006", "ADT-007", "ADT-008",
    "ADT-009", "ADT-010", "ADT-011", "ADT-012", "ADT-013", "ADT-014", "ADT-015", "ADT-016",
    "ADT-017", "ADT-018", "ADT-019", "ADT-020", "ADT-021",
];

type Errors = Vec<String>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Decision {
    decision: String,
    next_measurement_id: Option<String>,
    reason_code: String,
}

#[derive(Debug, PartialEq, Eq)]
struct AuditSummary {
    trace_count: usize,
    selected: usize,
    abstained: usize,
    no_safe_candidate: usize,
    no_discriminating_measurement: usize,
    invalid_contract: usize,
    schema_sha256: String,
    corpus_sha256: String,
}

fn text_at<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn number_at(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn default_schema_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../schemas/diagnostics/appliance_active_diagnostic_corpus_v0_1.schema.json")
}

fn default_corpus_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/diagnostics/appliance_active_diagnostic_corpus_v0_1.json")
}

fn schema_validator(schema: &Value) -> Result<Validator, String> {
    if text_at(schema, "$schema") != Some("https://json-schema.org/draft/2020-12/schema") {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if text_at(schema, "$id") != Some(SCHEMA_ID) {
        return Err(format!("schema $id must be {SCHEMA_ID}"));
    }
    if !jsonschema::draft202012::meta::is_valid(schema) {
        return Err("schema does not validate against the Draft 2020-12 meta-schema".to_owned());
    }
    jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(schema)
        .map_err(|error| format!("cannot compile schema: {error}"))
}

fn schema_errors(validator: &Validator, value: &Value) -> Vec<String> {
    validator.iter_errors(value).map(|error| error.to_string()).collect()
}

fn invalid_decision(reason_code: &str) -> Decision {
    Decision {
        decision: "reject_invalid_contract".to_owned(),
        next_measurement_id: None,
        reason_code: reason_code.to_owned(),
    }
}

fn validate_trace(trace: &Value) -> Result<(), String> {
    let Some(hypotheses) = trace.get("hypotheses").and_then(Value::as_array) else {
        return Err("invalid_prior_distribution".to_owned());
    };
    if hypotheses.len() < 2 {
        return Err("invalid_prior_distribution".to_owned());
    }

    let mut hypothesis_ids = BTreeSet::new();
    let mut prior_sum = 0.0_f64;
    for hypothesis in hypotheses {
        let Some(id) = text_at(hypothesis, "hypothesis_id") else {
            return Err("invalid_prior_distribution".to_owned());
        };
        let Some(prior) = number_at(hypothesis, "prior_probability") else {
            return Err("invalid_prior_distribution".to_owned());
        };
        if !hypothesis_ids.insert(id.to_owned()) {
            return Err("duplicate_identity".to_owned());
        }
        if !prior.is_finite() || !(0.0..=1.0).contains(&prior) {
            return Err("invalid_prior_distribution".to_owned());
        }
        prior_sum += prior;
    }
    if !prior_sum.is_finite() || (prior_sum - 1.0).abs() > 1e-9 {
        return Err("invalid_prior_distribution".to_owned());
    }

    let Some(measurements) = trace
        .get("candidate_measurements")
        .and_then(Value::as_array)
    else {
        return Err("invalid_likelihood_model".to_owned());
    };
    let mut measurement_ids = BTreeSet::new();

    for measurement in measurements {
        let Some(id) = text_at(measurement, "measurement_id") else {
            return Err("duplicate_identity".to_owned());
        };
        if !measurement_ids.insert(id.to_owned()) {
            return Err("duplicate_identity".to_owned());
        }

        let Some(cost) = number_at(measurement, "cost_units") else {
            return Err("invalid_cost".to_owned());
        };
        if !cost.is_finite() || cost <= 0.0 {
            return Err("invalid_cost".to_owned());
        }

        let Some(outcomes) = measurement.get("outcome_ids").and_then(Value::as_array) else {
            return Err("invalid_likelihood_model".to_owned());
        };
        if outcomes.len() < 2 {
            return Err("invalid_likelihood_model".to_owned());
        }
        let mut outcome_ids = BTreeSet::new();
        for outcome in outcomes {
            let Some(outcome_id) = outcome.as_str() else {
                return Err("invalid_likelihood_model".to_owned());
            };
            if outcome_id.is_empty() || !outcome_ids.insert(outcome_id.to_owned()) {
                return Err("invalid_likelihood_model".to_owned());
            }
        }

        let Some(likelihoods) = measurement.get("likelihoods").and_then(Value::as_array) else {
            return Err("invalid_likelihood_model".to_owned());
        };
        let mut row_ids = BTreeSet::new();
        if likelihoods.len() != hypothesis_ids.len() {
            return Err("invalid_likelihood_model".to_owned());
        }

        for row in likelihoods {
            let Some(hypothesis_id) = text_at(row, "hypothesis_id") else {
                return Err("invalid_likelihood_model".to_owned());
            };
            if !hypothesis_ids.contains(hypothesis_id)
                || !row_ids.insert(hypothesis_id.to_owned())
            {
                return Err("invalid_likelihood_model".to_owned());
            }

            let Some(probabilities) = row.get("probabilities").and_then(Value::as_array) else {
                return Err("invalid_likelihood_model".to_owned());
            };
            if probabilities.len() != outcome_ids.len() {
                return Err("invalid_likelihood_model".to_owned());
            }

            let mut probability_ids = BTreeSet::new();
            let mut row_sum = 0.0_f64;
            for item in probabilities {
                let Some(outcome_id) = text_at(item, "outcome_id") else {
                    return Err("invalid_likelihood_model".to_owned());
                };
                let Some(probability) = number_at(item, "probability") else {
                    return Err("invalid_likelihood_model".to_owned());
                };
                if !outcome_ids.contains(outcome_id)
                    || !probability_ids.insert(outcome_id.to_owned())
                    || !probability.is_finite()
                    || !(0.0..=1.0).contains(&probability)
                {
                    return Err("invalid_likelihood_model".to_owned());
                }
                row_sum += probability;
            }
            if probability_ids != outcome_ids
                || !row_sum.is_finite()
                || (row_sum - 1.0).abs() > 1e-9
            {
                return Err("invalid_likelihood_model".to_owned());
            }
        }

        if row_ids != hypothesis_ids {
            return Err("invalid_likelihood_model".to_owned());
        }
    }

    Ok(())
}

fn entropy(probabilities: impl Iterator<Item = f64>) -> f64 {
    -probabilities
        .map(|probability| {
            if probability > 0.0 {
                probability * probability.log2()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

fn likelihood<'a>(measurement: &'a Value, hypothesis_id: &str, outcome_id: &str) -> Option<f64> {
    measurement
        .get("likelihoods")?
        .as_array()?
        .iter()
        .find(|row| text_at(row, "hypothesis_id") == Some(hypothesis_id))?
        .get("probabilities")?
        .as_array()?
        .iter()
        .find(|item| text_at(item, "outcome_id") == Some(outcome_id))?
        .get("probability")?
        .as_f64()
}

fn information_gain(trace: &Value, measurement: &Value) -> f64 {
    let hypotheses = trace
        .get("hypotheses")
        .and_then(Value::as_array)
        .expect("validated hypotheses");
    let prior_entropy = entropy(
        hypotheses
            .iter()
            .map(|hypothesis| number_at(hypothesis, "prior_probability").expect("validated prior")),
    );
    let outcomes = measurement
        .get("outcome_ids")
        .and_then(Value::as_array)
        .expect("validated outcome ids");

    let mut expected_posterior_entropy = 0.0_f64;
    for outcome in outcomes {
        let outcome_id = outcome.as_str().expect("validated outcome id");
        let probability_of_outcome = hypotheses
            .iter()
            .map(|hypothesis| {
                let id = text_at(hypothesis, "hypothesis_id").expect("validated hypothesis id");
                number_at(hypothesis, "prior_probability").expect("validated prior")
                    * likelihood(measurement, id, outcome_id).expect("validated likelihood")
            })
            .sum::<f64>();

        if probability_of_outcome <= 0.0 {
            continue;
        }

        let posterior_entropy = entropy(hypotheses.iter().map(|hypothesis| {
            let id = text_at(hypothesis, "hypothesis_id").expect("validated hypothesis id");
            number_at(hypothesis, "prior_probability").expect("validated prior")
                * likelihood(measurement, id, outcome_id).expect("validated likelihood")
                / probability_of_outcome
        }));
        expected_posterior_entropy += probability_of_outcome * posterior_entropy;
    }

    (prior_entropy - expected_posterior_entropy).max(0.0)
}

fn evaluate_trace(trace: &Value) -> Decision {
    if let Err(reason_code) = validate_trace(trace) {
        return invalid_decision(&reason_code);
    }

    match text_at(trace, "protection_state") {
        Some("tripped") => {
            return Decision {
                decision: "abstain_and_review".to_owned(),
                next_measurement_id: None,
                reason_code: "protection_tripped".to_owned(),
            };
        }
        Some("unknown") => {
            return Decision {
                decision: "abstain_and_review".to_owned(),
                next_measurement_id: None,
                reason_code: "protection_unknown".to_owned(),
            };
        }
        Some("clear") => {}
        _ => return invalid_decision("invalid_likelihood_model"),
    }

    if text_at(trace, "observation_status") != Some("complete") {
        return Decision {
            decision: "abstain_and_review".to_owned(),
            next_measurement_id: None,
            reason_code: "observations_unreliable".to_owned(),
        };
    }

    let candidates: Vec<&Value> = trace
        .get("candidate_measurements")
        .and_then(Value::as_array)
        .expect("validated candidate measurements")
        .iter()
        .filter(|measurement| {
            text_at(measurement, "risk_class") == Some("low")
                && measurement.get("available").and_then(Value::as_bool) == Some(true)
        })
        .collect();

    if candidates.is_empty() {
        return Decision {
            decision: "no_safe_candidate".to_owned(),
            next_measurement_id: None,
            reason_code: "no_safe_candidate".to_owned(),
        };
    }

    let mut scored: Vec<(&str, f64)> = candidates
        .iter()
        .map(|measurement| {
            let id = text_at(measurement, "measurement_id").expect("validated measurement id");
            let cost = number_at(measurement, "cost_units").expect("validated cost");
            (id, information_gain(trace, measurement) / cost)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    let Some((measurement_id, score)) = scored.first() else {
        return Decision {
            decision: "no_safe_candidate".to_owned(),
            next_measurement_id: None,
            reason_code: "no_safe_candidate".to_owned(),
        };
    };
    if *score <= 1e-9 {
        return Decision {
            decision: "no_discriminating_measurement".to_owned(),
            next_measurement_id: None,
            reason_code: "no_discriminating_measurement".to_owned(),
        };
    }

    Decision {
        decision: "select_measurement".to_owned(),
        next_measurement_id: Some((*measurement_id).to_owned()),
        reason_code: "information_gain_per_cost".to_owned(),
    }
}

fn expected_decision(trace: &Value) -> Result<Decision, String> {
    let Some(expected) = trace.get("expected") else {
        return Err("expected decision is missing".to_owned());
    };
    let Some(decision) = text_at(expected, "decision") else {
        return Err("expected.decision is missing".to_owned());
    };
    let Some(reason_code) = text_at(expected, "reason_code") else {
        return Err("expected.reason_code is missing".to_owned());
    };
    let next_measurement_id = match expected.get("next_measurement_id") {
        Some(value) if value.is_null() => None,
        Some(value) => value.as_str().map(str::to_owned),
        None => return Err("expected.next_measurement_id is missing".to_owned()),
    };

    Ok(Decision {
        decision: decision.to_owned(),
        next_measurement_id,
        reason_code: reason_code.to_owned(),
    })
}

fn audit_values(
    schema: &Value,
    corpus: &Value,
    schema_raw: &str,
    corpus_raw: &str,
) -> Result<AuditSummary, Errors> {
    let validator = schema_validator(schema).map_err(|error| vec![error])?;
    let schema_errors = schema_errors(&validator, corpus);
    if !schema_errors.is_empty() {
        return Err(schema_errors);
    }

    let Some(traces) = corpus.get("traces").and_then(Value::as_array) else {
        return Err(vec!["corpus.traces must be an array".to_owned()]);
    };
    let expected_ids: BTreeSet<String> =
        EXPECTED_TRACE_IDS.iter().map(|id| (*id).to_owned()).collect();
    let mut seen_ids = BTreeSet::new();
    let mut selected = 0;
    let mut abstained = 0;
    let mut no_safe_candidate = 0;
    let mut no_discriminating_measurement = 0;
    let mut invalid_contract = 0;
    let mut errors = Vec::new();

    for trace in traces {
        let Some(trace_id) = text_at(trace, "trace_id") else {
            errors.push("trace_id must be a string".to_owned());
            continue;
        };
        if !seen_ids.insert(trace_id.to_owned()) {
            errors.push(format!("duplicate trace_id {trace_id}"));
            continue;
        }
        let actual = evaluate_trace(trace);
        let expected = match expected_decision(trace) {
            Ok(expected) => expected,
            Err(error) => {
                errors.push(format!("{trace_id}: {error}"));
                continue;
            }
        };
        if actual != expected {
            errors.push(format!(
                "{trace_id}: expected {expected:?}, evaluator produced {actual:?}"
            ));
        }

        match actual.decision.as_str() {
            "select_measurement" => selected += 1,
            "abstain_and_review" => abstained += 1,
            "no_safe_candidate" => no_safe_candidate += 1,
            "no_discriminating_measurement" => no_discriminating_measurement += 1,
            "reject_invalid_contract" => invalid_contract += 1,
            other => errors.push(format!("{trace_id}: unexpected decision {other}")),
        }
    }

    for missing in expected_ids.difference(&seen_ids) {
        errors.push(format!("required trace {missing} is missing"));
    }
    for unexpected in seen_ids.difference(&expected_ids) {
        errors.push(format!("unexpected trace_id {unexpected}"));
    }
    if traces.len() != EXPECTED_TRACE_IDS.len() {
        errors.push(format!(
            "trace count is {}, expected {}",
            traces.len(),
            EXPECTED_TRACE_IDS.len()
        ));
    }
    if selected != 4
        || abstained != 7
        || no_safe_candidate != 3
        || no_discriminating_measurement != 2
        || invalid_contract != 5
    {
        errors.push(format!(
            "decision census changed: selected={selected}, abstained={abstained}, no_safe={no_safe_candidate}, no_discrimination={no_discriminating_measurement}, invalid={invalid_contract}"
        ));
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(AuditSummary {
        trace_count: traces.len(),
        selected,
        abstained,
        no_safe_candidate,
        no_discriminating_measurement,
        invalid_contract,
        schema_sha256: format!("{:x}", Sha256::digest(schema_raw.as_bytes())),
        corpus_sha256: format!("{:x}", Sha256::digest(corpus_raw.as_bytes())),
    })
}

fn audit_texts(schema_raw: &str, corpus_raw: &str) -> Result<AuditSummary, Errors> {
    let schema: Value = serde_json::from_str(schema_raw)
        .map_err(|error| vec![format!("schema JSON parse failed: {error}")])?;
    let corpus: Value = serde_json::from_str(corpus_raw)
        .map_err(|error| vec![format!("corpus JSON parse failed: {error}")])?;
    audit_values(&schema, &corpus, schema_raw, corpus_raw)
}

fn read_arg_path(flag: &str) -> Option<PathBuf> {
    let mut arguments = env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == flag {
            return arguments.next().map(PathBuf::from);
        }
    }
    None
}

fn run() -> Result<(), String> {
    let schema_path = read_arg_path("--schema").unwrap_or_else(default_schema_path);
    let corpus_path = read_arg_path("--corpus").unwrap_or_else(default_corpus_path);
    let schema_raw = fs::read_to_string(&schema_path)
        .map_err(|error| format!("cannot read {}: {error}", schema_path.display()))?;
    let corpus_raw = fs::read_to_string(&corpus_path)
        .map_err(|error| format!("cannot read {}: {error}", corpus_path.display()))?;
    let summary = audit_texts(&schema_raw, &corpus_raw).map_err(|errors| errors.join("\n"))?;

    let report = json!({
        "report_version": "appliance-active-diagnostics-audit/0.1",
        "status": "PASS_SYNTHETIC_CONTRACT_ONLY",
        "evidence_class": "synthetic_contract_fixture",
        "physical_diagnostic_accuracy_established": false,
        "product_safety_assessed": false,
        "trace_count": summary.trace_count,
        "selected": summary.selected,
        "abstained": summary.abstained,
        "no_safe_candidate": summary.no_safe_candidate,
        "no_discriminating_measurement": summary.no_discriminating_measurement,
        "invalid_contract": summary.invalid_contract,
        "schema_sha256": summary.schema_sha256,
        "corpus_sha256": summary.corpus_sha256,
        "schema_path": schema_path.display().to_string(),
        "corpus_path": corpus_path.display().to_string()
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("appliance-active-diagnostics-audit: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical_schema() -> Value {
        serde_json::from_str(include_str!(
            "../../../schemas/diagnostics/appliance_active_diagnostic_corpus_v0_1.schema.json"
        ))
        .expect("canonical schema JSON must parse")
    }

    fn canonical_corpus() -> Value {
        serde_json::from_str(include_str!(
            "../../../fixtures/diagnostics/appliance_active_diagnostic_corpus_v0_1.json"
        ))
        .expect("canonical corpus JSON must parse")
    }

    fn audit_mutation(corpus: &Value) -> Result<AuditSummary, Errors> {
        let schema = canonical_schema();
        let schema_raw = serde_json::to_string(&schema).expect("schema serialize");
        let corpus_raw = serde_json::to_string(corpus).expect("corpus serialize");
        audit_values(&schema, corpus, &schema_raw, &corpus_raw)
    }

    #[test]
    fn canonical_corpus_matches_all_expected_decisions() {
        let schema_raw = include_str!(
            "../../../schemas/diagnostics/appliance_active_diagnostic_corpus_v0_1.schema.json"
        );
        let corpus_raw = include_str!(
            "../../../fixtures/diagnostics/appliance_active_diagnostic_corpus_v0_1.json"
        );
        let summary = audit_texts(schema_raw, corpus_raw).expect("canonical corpus must qualify");
        assert_eq!(summary.trace_count, 21);
        assert_eq!(summary.selected, 4);
        assert_eq!(summary.abstained, 7);
        assert_eq!(summary.no_safe_candidate, 3);
        assert_eq!(summary.no_discriminating_measurement, 2);
        assert_eq!(summary.invalid_contract, 5);
    }

    #[test]
    fn bad_prior_mass_is_rejected_without_renormalization() {
        let mut trace = canonical_corpus()["traces"][0].clone();
        trace["hypotheses"][0]["prior_probability"] = json!(0.55);
        assert_eq!(
            validate_trace(&trace).expect_err("prior mass must not be silently normalized"),
            "invalid_prior_distribution"
        );
    }

    #[test]
    fn invalid_likelihood_mass_is_rejected() {
        let mut trace = canonical_corpus()["traces"][0].clone();
        trace["candidate_measurements"][0]["likelihoods"][0]["probabilities"][0]["probability"] =
            json!(0.4);
        trace["candidate_measurements"][0]["likelihoods"][0]["probabilities"][1]["probability"] =
            json!(0.4);
        assert_eq!(
            validate_trace(&trace).expect_err("likelihood mass must sum to one"),
            "invalid_likelihood_model"
        );
    }

    #[test]
    fn tripped_protection_blocks_measurement_selection() {
        let mut trace = canonical_corpus()["traces"][0].clone();
        trace["protection_state"] = json!("tripped");
        let decision = evaluate_trace(&trace);
        assert_eq!(decision.decision, "abstain_and_review");
        assert_eq!(decision.next_measurement_id, None);
        assert_eq!(decision.reason_code, "protection_tripped");
    }

    #[test]
    fn stale_observations_abstain_before_fault_ranking() {
        let mut trace = canonical_corpus()["traces"][0].clone();
        trace["observation_status"] = json!("stale");
        let decision = evaluate_trace(&trace);
        assert_eq!(decision.decision, "abstain_and_review");
        assert_eq!(decision.next_measurement_id, None);
        assert_eq!(decision.reason_code, "observations_unreliable");
    }

    #[test]
    fn equal_scores_use_stable_lexical_tie_breaking() {
        let trace = canonical_corpus()["traces"][2].clone();
        let decision = evaluate_trace(&trace);
        assert_eq!(decision.decision, "select_measurement");
        assert_eq!(decision.next_measurement_id.as_deref(), Some("a_rpm_check"));
    }

    #[test]
    fn zero_cost_measurement_is_rejected() {
        let mut trace = canonical_corpus()["traces"][16].clone();
        trace["candidate_measurements"][0]["cost_units"] = json!(0.0);
        assert_eq!(
            validate_trace(&trace).expect_err("zero cost must be rejected"),
            "invalid_cost"
        );
    }

    #[test]
    fn perfect_binary_measurement_yields_one_bit_information_gain() {
        let trace = canonical_corpus()["traces"][2].clone();
        let measurement = &trace["candidate_measurements"][0];
        let gain = information_gain(&trace, measurement);
        assert!((gain - 1.0).abs() <= 1e-12, "expected 1 bit, got {gain}");
    }

    #[test]
    fn non_discriminating_measurement_yields_zero_information_gain() {
        let trace = canonical_corpus()["traces"][11].clone();
        let measurement = &trace["candidate_measurements"][0];
        let gain = information_gain(&trace, measurement);
        assert!(gain.abs() <= 1e-12, "expected 0 bits, got {gain}");
    }

    #[test]
    fn information_gain_is_finite_and_bounded_by_prior_entropy() {
        for trace in canonical_corpus()["traces"].as_array().expect("trace array") {
            if validate_trace(trace).is_err() {
                continue;
            }
            let prior_entropy = entropy(
                trace["hypotheses"]
                    .as_array()
                    .expect("hypothesis array")
                    .iter()
                    .map(|hypothesis| {
                        number_at(hypothesis, "prior_probability").expect("prior probability")
                    }),
            );
            for measurement in trace["candidate_measurements"]
                .as_array()
                .expect("measurement array")
            {
                let gain = information_gain(trace, measurement);
                assert!(
                    gain.is_finite(),
                    "non-finite information gain in {}",
                    text_at(trace, "trace_id").unwrap_or("?")
                );
                assert!(
                    gain >= -1e-12,
                    "negative information gain in {}",
                    text_at(trace, "trace_id").unwrap_or("?")
                );
                assert!(
                    gain <= prior_entropy + 1e-12,
                    "information gain exceeds prior entropy in {}",
                    text_at(trace, "trace_id").unwrap_or("?")
                );
            }
        }
    }

    #[test]
    fn canonical_corpus_replay_preserves_identical_digests_and_summary() {
        let schema_raw = include_str!(
            "../../../schemas/diagnostics/appliance_active_diagnostic_corpus_v0_1.schema.json"
        );
        let corpus_raw = include_str!(
            "../../../fixtures/diagnostics/appliance_active_diagnostic_corpus_v0_1.json"
        );
        let first = audit_texts(schema_raw, corpus_raw).expect("first audit");
        let second = audit_texts(schema_raw, corpus_raw).expect("second audit");
        assert_eq!(first, second);
    }

    #[test]
    fn missing_hypothesis_likelihood_is_rejected() {
        let mut trace = canonical_corpus()["traces"][14].clone();
        trace["candidate_measurements"][0]["likelihoods"]
            .as_array_mut()
            .expect("likelihood array")
            .pop();
        assert_eq!(
            validate_trace(&trace).expect_err("all hypotheses must be covered"),
            "invalid_likelihood_model"
        );
    }
}
