// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic interval-aware measurement acceptance qualifier.
//!
//! This is a deterministic decision-rule challenge, not a metrology system.
//! It does not verify calibration certificates, establish traceability to SI,
//! estimate statistical uncertainty, or admit physical observations.

use std::collections::HashSet;
use serde_json::Value;

pub const EMBEDDED_MEASUREMENT_GATE: &str =
    include_str!("../../../../docs/research/fixtures/habitat-measurement-gate-v0.1.json");

const DISPOSITIONS: [&str; 5] = [
    "Conforming", "Nonconforming", "Indeterminate", "Blocked", "NotEvaluable",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementCaseResult {
    pub case_id: String,
    pub disposition: String,
    pub expected_disposition: String,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementGateReport {
    pub issues: Vec<MeasurementGateIssue>,
    pub case_count: usize,
    pub results: Vec<MeasurementCaseResult>,
}
impl MeasurementGateReport {
    /// A valid result qualifies only the synthetic known-answer decision contract.
    pub fn is_valid(&self) -> bool { self.issues.is_empty() }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementGateIssue {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

pub fn validate_measurement_gate_str(input: &str) -> Result<MeasurementGateReport, serde_json::Error> {
    let root: Value = serde_json::from_str(input)?;
    Ok(validate_measurement_gate(&root))
}

pub fn validate_measurement_gate(root: &Value) -> MeasurementGateReport {
    let mut issues = Vec::new();
    expect_string(root, "schema_version", "0.1.0", &mut issues);
    expect_string(root, "campaign_id", "HABITAT-MEASUREMENT-GATE-001", &mut issues);
    expect_string(root, "program_id", "HABITAT-VERTICAL-001", &mut issues);
    expect_string(root, "evidence_status", "synthetic_decision_rule_challenge_only", &mut issues);
    if !root.get("claim_ceiling").and_then(Value::as_str)
        .is_some_and(|value| value.contains("No measurement here is a physical observation"))
    {
        push(&mut issues, "CLAIM_CEILING_MISSING", "claim_ceiling", "physical-measurement and standards-conformity limits must be explicit");
    }
    if root.pointer("/decision_rule/calibration_gate").and_then(Value::as_str)
        .is_none_or(|text| !text.contains("never treat these strings as actual calibration evidence"))
    {
        push(&mut issues, "CALIBRATION_CEILING_MISSING", "decision_rule.calibration_gate", "synthetic calibration labels must not be mistaken for verified calibration evidence");
    }
    if root.pointer("/decision_rule/uncertainty_semantics").and_then(Value::as_str)
        .is_none_or(|text| !text.contains("not a probability distribution"))
    {
        push(&mut issues, "UNCERTAINTY_SEMANTICS_MISSING", "decision_rule.uncertainty_semantics", "interval bounds must not be converted silently into probabilities");
    }
    reject_probability_fields(root, "$", &mut issues);

    let Some(cases) = root.get("cases").and_then(Value::as_array) else {
        push(&mut issues, "CASES_ARRAY_MISSING", "cases", "cases must be a JSON array");
        return report(issues, Vec::new());
    };
    let mut ids = HashSet::new();
    let mut results = Vec::with_capacity(cases.len());
    for (index, case) in cases.iter().enumerate() {
        let path = format!("cases[{index}]");
        let Some(id) = case.get("case_id").and_then(Value::as_str).filter(|id| !id.trim().is_empty()) else {
            push(&mut issues, "CASE_ID_MISSING", format!("{path}.case_id"), "case_id must be a non-empty string");
            continue;
        };
        if !ids.insert(id.to_owned()) {
            push(&mut issues, "DUPLICATE_CASE_ID", format!("{path}.case_id"), format!("case ID {id:?} is repeated"));
        }
        if case.get("evidence_status").and_then(Value::as_str) != Some("synthetic_only") {
            push(&mut issues, "EVIDENCE_STATUS_OVERCLAIM", format!("{path}.evidence_status"), "all cases must remain synthetic-only");
        }
        let disposition = evaluate_case(case);
        let expected = case.get("expected_disposition").and_then(Value::as_str).unwrap_or_default().to_owned();
        if !DISPOSITIONS.contains(&expected.as_str()) {
            push(&mut issues, "UNKNOWN_EXPECTED_DISPOSITION", format!("{path}.expected_disposition"), format!("unknown expected disposition {expected:?}"));
        }
        if disposition != expected {
            push(&mut issues, "KNOWN_ANSWER_DISPOSITION_MISMATCH", format!("{path}.expected_disposition"), format!("case {id:?} evaluates to {disposition:?}, expected {expected:?}"));
        }
        results.push(MeasurementCaseResult { case_id: id.to_owned(), disposition, expected_disposition: expected });
    }

    if root.pointer("/expected_known_answer/case_count").and_then(Value::as_u64) != Some(results.len() as u64) {
        push(&mut issues, "KNOWN_ANSWER_CASE_COUNT_MISMATCH", "expected_known_answer.case_count", "declared case count does not match evaluated cases");
    }
    if let Some(counts) = root.pointer("/expected_known_answer/disposition_counts").and_then(Value::as_object) {
        for disposition in DISPOSITIONS {
            let actual = results.iter().filter(|result| result.disposition == disposition).count() as u64;
            if counts.get(disposition).and_then(Value::as_u64) != Some(actual) {
                push(&mut issues, "KNOWN_ANSWER_DISPOSITION_COUNT_MISMATCH", format!("expected_known_answer.disposition_counts.{disposition}"), format!("expected count does not match evaluated count {actual}"));
            }
        }
    } else {
        push(&mut issues, "DISPOSITION_COUNTS_MISSING", "expected_known_answer.disposition_counts", "frozen per-disposition counts are required");
    }
    report(issues, results)
}

/// Evaluate one case. Calibration labels here are synthetic oracle state only.
pub fn evaluate_case(case: &Value) -> String {
    let Some(observation) = case.get("observation") else { return "NotEvaluable".to_owned(); };
    let Some(spec) = case.get("specification") else { return "NotEvaluable".to_owned(); };
    let Some(instrument) = case.get("instrument") else { return "NotEvaluable".to_owned(); };
    if observation.get("unit").and_then(Value::as_str) != Some("mg")
        || spec.get("unit").and_then(Value::as_str) != Some("mg")
    { return "Blocked".to_owned(); }
    let Some(minimum) = spec.get("minimum_mg").and_then(Value::as_u64) else { return "Blocked".to_owned(); };
    let Some(maximum) = spec.get("maximum_mg").and_then(Value::as_u64) else { return "Blocked".to_owned(); };
    if minimum > maximum { return "Blocked".to_owned(); }
    let Some(value) = observation.get("measured_value_mg").and_then(Value::as_u64) else { return "NotEvaluable".to_owned(); };
    let Some(uncertainty) = observation.get("expanded_uncertainty_mg").and_then(Value::as_u64) else { return "NotEvaluable".to_owned(); };
    if uncertainty == 0 { return "NotEvaluable".to_owned(); }
    let Some(lower) = value.checked_sub(uncertainty) else { return "NotEvaluable".to_owned(); };
    let Some(upper) = value.checked_add(uncertainty) else { return "NotEvaluable".to_owned(); };
    let Some(measured_date) = observation.get("measurement_date_utc").and_then(Value::as_str) else { return "NotEvaluable".to_owned(); };
    if !valid_utc_date(measured_date) { return "NotEvaluable".to_owned(); }
    let Some(certificate_status) = instrument.get("certificate_status").and_then(Value::as_str) else { return "NotEvaluable".to_owned(); };
    if certificate_status != "synthetic_reference_only" { return "NotEvaluable".to_owned(); }
    let Some(traceability_status) = instrument.get("traceability_status").and_then(Value::as_str) else { return "NotEvaluable".to_owned(); };
    if traceability_status != "declared_synthetic_chain" { return "Blocked".to_owned(); }
    let Some(valid_from) = instrument.get("calibration_valid_from_utc_date").and_then(Value::as_str) else { return "NotEvaluable".to_owned(); };
    let Some(valid_until) = instrument.get("calibration_valid_until_utc_date").and_then(Value::as_str) else { return "NotEvaluable".to_owned(); };
    if !valid_utc_date(valid_from) || !valid_utc_date(valid_until) || valid_from > valid_until { return "NotEvaluable".to_owned(); }
    if measured_date < valid_from || measured_date > valid_until { return "Blocked".to_owned(); }
    if upper < minimum || lower > maximum {
        "Nonconforming".to_owned()
    } else if lower >= minimum && upper <= maximum {
        "Conforming".to_owned()
    } else {
        "Indeterminate".to_owned()
    }
}

fn reject_probability_fields(value: &Value, path: &str, issues: &mut Vec<MeasurementGateIssue>) {
    const FORBIDDEN_KEYS: [&str; 6] = [
        "probability",
        "scenario_probability",
        "failure_probability",
        "probability_of_conformity",
        "likelihood",
        "confidence_level",
    ];
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                let child_path = format!("{path}.{key}");
                if FORBIDDEN_KEYS.contains(&key.as_str()) && !child.is_null() {
                    push(
                        issues,
                        "PROBABILITY_CLAIM_NOT_SUPPORTED",
                        &child_path,
                        "a deterministic interval fixture cannot supply a probability or likelihood",
                    );
                }
                reject_probability_fields(child, &child_path, issues);
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                reject_probability_fields(child, &format!("{path}[{index}]"), issues);
            }
        }
        _ => {}
    }
}

fn valid_utc_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-'
        || !bytes.iter().enumerate().all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
    { return false; }
    let Some(year) = value[0..4].parse::<u32>().ok() else { return false; };
    let Some(month) = value[5..7].parse::<u32>().ok() else { return false; };
    let Some(day) = value[8..10].parse::<u32>().ok() else { return false; };
    if year == 0 || !(1..=12).contains(&month) { return false; }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month { 1 | 3 | 5 | 7 | 8 | 10 | 12 => 31, 4 | 6 | 9 | 11 => 30, 2 if leap => 29, 2 => 28, _ => return false };
    (1..=days).contains(&day)
}
fn expect_string(root: &Value, field: &str, expected: &str, issues: &mut Vec<MeasurementGateIssue>) {
    if root.get(field).and_then(Value::as_str) != Some(expected) { push(issues, "EXPECTED_STRING_MISMATCH", field, format!("required value is {expected:?}")); }
}
fn push(issues: &mut Vec<MeasurementGateIssue>, code: &'static str, path: impl Into<String>, message: impl Into<String>) {
    issues.push(MeasurementGateIssue { code, path: path.into(), message: message.into() });
}
fn report(mut issues: Vec<MeasurementGateIssue>, mut results: Vec<MeasurementCaseResult>) -> MeasurementGateReport {
    issues.sort_by(|a,b| a.code.cmp(b.code).then_with(||a.path.cmp(&b.path)).then_with(||a.message.cmp(&b.message)));
    results.sort_by(|a,b| a.case_id.cmp(&b.case_id));
    MeasurementGateReport { case_count: results.len(), issues, results }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Value { serde_json::from_str(EMBEDDED_MEASUREMENT_GATE).expect("measurement gate fixture JSON") }
    fn get_case(root: &mut Value, id: &str) -> &mut Value {
        root["cases"].as_array_mut().expect("cases").iter_mut().find(|case| case["case_id"].as_str() == Some(id)).expect("case exists")
    }
    fn has_code(report: &MeasurementGateReport, code: &str) -> bool { report.issues.iter().any(|issue| issue.code == code) }

    #[test] fn embedded_known_answer_dispositions_match() {
        let report = validate_measurement_gate(&fixture());
        assert!(report.is_valid(), "{:?}", report.issues);
        assert_eq!(report.case_count, 13);
    }
    #[test] fn cases_are_sorted_independently_of_fixture_order() {
        let mut root = fixture(); let before = validate_measurement_gate(&root);
        root["cases"].as_array_mut().expect("cases").reverse();
        let after = validate_measurement_gate(&root);
        assert!(before.is_valid(), "{:?}", before.issues); assert!(after.is_valid(), "{:?}", after.issues);
        assert_eq!(before.results, after.results);
    }
    #[test] fn fully_contained_interval_is_conforming() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-01-CONFORMING").unwrap();
        assert_eq!(evaluate_case(c), "Conforming");
    }
    #[test] fn disjoint_interval_is_nonconforming() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-02-NONCONFORMING-HIGH").unwrap();
        assert_eq!(evaluate_case(c), "Nonconforming");
    }
    #[test] fn boundary_overlap_is_indeterminate_not_pass() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-03-BOUNDARY-INDETERMINATE").unwrap();
        assert_eq!(evaluate_case(c), "Indeterminate");
    }
    #[test] fn expired_calibration_blocks_conformity_claim() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-04-CALIBRATION-EXPIRED").unwrap();
        assert_eq!(evaluate_case(c), "Blocked");
    }
    #[test] fn missing_uncertainty_is_not_zero() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-05-MISSING-UNCERTAINTY").unwrap();
        assert_eq!(evaluate_case(c), "NotEvaluable");
    }
    #[test] fn unit_mismatch_blocks_silent_conversion() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-06-UNIT-MISMATCH").unwrap();
        assert_eq!(evaluate_case(c), "Blocked");
    }
    #[test] fn missing_certificate_is_not_accepted() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-07-CERTIFICATE-MISSING").unwrap();
        assert_eq!(evaluate_case(c), "NotEvaluable");
    }
    #[test] fn broken_traceability_chain_blocks() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-08-TRACEABILITY-BROKEN").unwrap();
        assert_eq!(evaluate_case(c), "Blocked");
    }
    #[test] fn interval_overflow_is_not_clamped() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-09-UPPER-OVERFLOW").unwrap();
        assert_eq!(evaluate_case(c), "NotEvaluable");
    }
    #[test] fn interval_underflow_is_not_clamped() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-10-LOWER-UNDERFLOW").unwrap();
        assert_eq!(evaluate_case(c), "NotEvaluable");
    }
    #[test] fn impossible_date_is_rejected_as_not_evaluable() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-11-INVALID-MEASUREMENT-DATE").unwrap();
        assert_eq!(evaluate_case(c), "NotEvaluable");
    }
    #[test] fn inverted_specification_is_blocked() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-12-INVALID-SPECIFICATION").unwrap();
        assert_eq!(evaluate_case(c), "Blocked");
    }
    #[test] fn future_calibration_date_blocks() {
        let root=fixture(); let c=root["cases"].as_array().unwrap().iter().find(|c|c["case_id"]=="MM-13-CALIBRATION-NOT-YET-VALID").unwrap();
        assert_eq!(evaluate_case(c), "Blocked");
    }
    #[test] fn probability_claim_from_interval_only_is_rejected() {
        let mut root = fixture();
        root["decision_rule"]["probability_of_conformity"] = Value::String("0.99".to_owned());
        let report = validate_measurement_gate(&root);
        assert!(has_code(&report, "PROBABILITY_CLAIM_NOT_SUPPORTED"));
    }
    #[test] fn changed_expected_disposition_is_detected() {
        let mut root=fixture(); get_case(&mut root, "MM-03-BOUNDARY-INDETERMINATE")["expected_disposition"]=Value::String("Conforming".to_owned());
        let report=validate_measurement_gate(&root); assert!(has_code(&report,"KNOWN_ANSWER_DISPOSITION_MISMATCH"));
    }
}
