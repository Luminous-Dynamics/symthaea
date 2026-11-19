use marine_sail_liveaboard_audit::{audit_case, compare_pair};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process::ExitCode;

const REQUIRED_FAMILIES: &[(&str, bool, &str)] = &[
    ("fixed_conservative", true, "as_of_decision_time"),
    ("polar_vmg", true, "as_of_decision_time"),
    ("transition_aware_vmg", true, "as_of_decision_time"),
    ("weather_routing", true, "as_of_decision_time"),
    ("reserve_aware_weather_routing", true, "as_of_decision_time"),
    ("integrated_advisory", true, "as_of_decision_time"),
    ("perfect_foresight_oracle", false, "realized_environment_evaluator_only"),
];

const REQUIRED_THREATS: &[&str] = &[
    "forecast_temporal_leakage",
    "realized_truth_leakage",
    "hard_gate_failure",
    "hard_gate_omission",
    "resource_dimension_omission",
    "resource_dimension_duplication",
    "comfort_constraint_bypass",
    "comfort_fallback_without_provenance",
    "identity_corruption",
    "matched_occupancy_mismatch",
    "matched_configuration_mismatch",
    "perfect_foresight_comparison",
    "matched_service_mismatch",
];

fn string_at<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn valid_pointer(pointer: &str) -> bool {
    if !pointer.starts_with('/') || pointer == "/" {
        return false;
    }
    pointer.split('/').skip(1).all(|part| {
        let bytes = part.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] == b'~' {
                if index + 1 >= bytes.len() || !matches!(bytes[index + 1], b'0' | b'1') {
                    return false;
                }
                index += 2;
            } else {
                index += 1;
            }
        }
        true
    })
}

fn validate(corpus: &Value) -> Result<Value, Vec<String>> {
    let mut errors = Vec::new();
    if !corpus.is_object() {
        return Err(vec!["corpus root must be an object".to_owned()]);
    }
    if string_at(corpus, "schema_version") != Some("0.1.0") {
        errors.push("unsupported schema_version".to_owned());
    }
    if string_at(corpus, "status") != Some("declarative_synthetic_only") {
        errors.push("corpus status must remain declarative_synthetic_only".to_owned());
    }
    if string_at(corpus, "claim_ceiling") != Some("test-vector-contract_only") {
        errors.push("claim ceiling is missing or too broad".to_owned());
    }
    if string_at(corpus, "source_fixture").is_none_or(str::is_empty) {
        errors.push("source_fixture is required".to_owned());
    }

    let mut arms = BTreeMap::new();
    match corpus.get("policy_arms").and_then(Value::as_array) {
        Some(items) => {
            for arm in items {
                let family = string_at(arm, "family").unwrap_or("<missing>");
                if arms.insert(family.to_owned(), arm).is_some() {
                    errors.push(format!("duplicate policy family: {family}"));
                }
            }
        }
        None => errors.push("policy_arms must be an array".to_owned()),
    }
    for (family, deployable, info) in REQUIRED_FAMILIES {
        match arms.get(*family) {
            Some(arm) => {
                if arm.get("deployable").and_then(Value::as_bool) != Some(*deployable) {
                    errors.push(format!("wrong deployable flag for {family}"));
                }
                if string_at(arm, "information_policy") != Some(*info) {
                    errors.push(format!("wrong information policy for {family}"));
                }
                if string_at(arm, "budget_profile_id").is_none_or(str::is_empty) {
                    errors.push(format!("missing budget profile for {family}"));
                }
            }
            None => errors.push(format!("required policy family missing: {family}")),
        }
    }

    let declared_threats: BTreeSet<String> = corpus
        .get("required_threat_classes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    for required in REQUIRED_THREATS {
        if !declared_threats.contains(*required) {
            errors.push(format!("required threat class missing: {required}"));
        }
    }

    let cases = match corpus.get("scenario_cases").and_then(Value::as_array) {
        Some(cases) if cases.len() >= 13 => cases,
        Some(_) => {
            errors.push("at least 13 scenario cases are required".to_owned());
            return Err(errors);
        }
        None => {
            errors.push("scenario_cases must be an array".to_owned());
            return Err(errors);
        }
    };

    let mut ids = BTreeSet::new();
    let mut seen_threats = BTreeSet::new();
    let mut command_counts = BTreeMap::<String, usize>::new();
    for (index, scenario) in cases.iter().enumerate() {
        let id = string_at(scenario, "scenario_id").unwrap_or("");
        if id.is_empty() {
            errors.push(format!("scenario[{index}] has no scenario_id"));
        } else if !ids.insert(id.to_owned()) {
            errors.push(format!("duplicate scenario_id: {id}"));
        }
        let threat = string_at(scenario, "threat_class").unwrap_or("");
        if threat.is_empty() {
            errors.push(format!("scenario[{index}] has no threat_class"));
        } else {
            seen_threats.insert(threat.to_owned());
        }
        let command = string_at(scenario, "command").unwrap_or("");
        if !matches!(command, "audit_case" | "compare_pair") {
            errors.push(format!("scenario[{index}] has invalid command"));
        } else {
            *command_counts.entry(command.to_owned()).or_default() += 1;
        }
        let target = string_at(scenario, "mutation_target").unwrap_or("");
        if (command == "audit_case" && target != "case")
            || (command == "compare_pair" && target != "candidate")
        {
            errors.push(format!("scenario[{index}] has a mutation_target/command mismatch"));
        }
        let disposition = string_at(scenario, "expected_disposition").unwrap_or("");
        let valid_disposition = if command == "audit_case" {
            matches!(disposition, "INVALID" | "BLOCKED" | "INCOMPLETE" | "SYNTHETIC_ONLY")
        } else {
            matches!(disposition, "MATCHED_SYNTHETIC_CASES_ONLY" | "INCOMPARABLE")
        };
        if !valid_disposition {
            errors.push(format!("scenario[{index}] has invalid expected disposition"));
        }
        let has_expected = scenario.get("expected_finding_codes").and_then(Value::as_array)
            .is_some_and(|items| !items.is_empty())
            || scenario.get("expected_mismatched_fields").and_then(Value::as_array).is_some();
        if !has_expected {
            errors.push(format!("scenario[{index}] needs expected finding codes or mismatch fields"));
        }
        if string_at(scenario, "purpose").is_none_or(str::is_empty) {
            errors.push(format!("scenario[{index}] purpose is required"));
        }

        match scenario.get("mutations").and_then(Value::as_array) {
            Some(mutations) => {
                if command == "audit_case" && id != "MSL-CORPUS-000" && mutations.is_empty() {
                    errors.push(format!("scenario[{index}] requires at least one mutation"));
                }
                for (mutation_index, mutation) in mutations.iter().enumerate() {
                    let op = string_at(mutation, "operation").unwrap_or("");
                    let path = string_at(mutation, "path").unwrap_or("");
                    if !matches!(op, "set" | "remove" | "append" | "copy") {
                        errors.push(format!("scenario[{index}] mutation[{mutation_index}] has invalid operation"));
                    }
                    if !valid_pointer(path) {
                        errors.push(format!("scenario[{index}] mutation[{mutation_index}] has invalid JSON pointer"));
                    }
                    if op == "copy" {
                        if string_at(mutation, "from_path").is_none_or(|from| !valid_pointer(from)) {
                            errors.push(format!("scenario[{index}] mutation[{mutation_index}] copy needs a valid from_path"));
                        }
                    } else if op != "remove" && mutation.get("value").is_none() {
                        errors.push(format!("scenario[{index}] mutation[{mutation_index}] needs a value"));
                    }
                }
            }
            None => errors.push(format!("scenario[{index}] mutations must be an array")),
        }
    }

    for required in REQUIRED_THREATS {
        if !seen_threats.contains(*required) {
            errors.push(format!("scenario vector missing required threat: {required}"));
        }
    }
    if command_counts.get("audit_case").copied().unwrap_or(0) == 0
        || command_counts.get("compare_pair").copied().unwrap_or(0) == 0
    {
        errors.push("corpus must cover both audit_case and compare_pair".to_owned());
    }

    let metrics = corpus.get("metric_contract").unwrap_or(&Value::Null);
    if metrics.get("target_improvement_fraction").and_then(Value::as_f64) != Some(0.10) {
        errors.push("the preregistered target must remain 10% pending prospective power analysis".to_owned());
    }
    if metrics.get("minimum_environment_strata").and_then(Value::as_u64).unwrap_or(0) < 3 {
        errors.push("at least three environment strata are required".to_owned());
    }
    if metrics.get("hard_constraints").and_then(Value::as_array).is_none_or(Vec::is_empty) {
        errors.push("hard constraint metrics must be explicit".to_owned());
    }
    if metrics.get("report_separately").and_then(Value::as_array).is_none_or(Vec::is_empty) {
        errors.push("resource and service metrics must be reported separately".to_owned());
    }

    if errors.is_empty() {
        Ok(serde_json::json!({
            "corpus_id": string_at(corpus, "corpus_id"),
            "status": "DECLARATIVE_CORPUS_VALID",
            "scenario_count": cases.len(),
            "policy_arm_count": arms.len(),
            "audit_case_count": command_counts.get("audit_case").copied().unwrap_or(0),
            "compare_pair_count": command_counts.get("compare_pair").copied().unwrap_or(0),
            "required_threat_classes_covered": REQUIRED_THREATS.len(),
            "claim_ceiling": "test_vector_contract_only",
            "execution_note": "Shape and coverage verified; scenario outcomes are not yet executed by this command."
        }))
    } else {
        Err(errors)
    }
}


fn decode_pointer(pointer: &str) -> Result<Vec<String>, String> {
    if !valid_pointer(pointer) {
        return Err(format!("invalid JSON Pointer: {pointer}"));
    }
    Ok(pointer
        .split('/')
        .skip(1)
        .map(|segment| segment.replace("~1", "/").replace("~0", "~"))
        .collect())
}

fn descend_mut<'a>(root: &'a mut Value, segments: &[String]) -> Result<&'a mut Value, String> {
    let mut current = root;
    for segment in segments {
        current = match current {
            Value::Object(map) => map
                .get_mut(segment)
                .ok_or_else(|| format!("JSON Pointer segment not found: {segment}"))?,
            Value::Array(items) => {
                let index = segment.parse::<usize>()
                    .map_err(|_| format!("array pointer is not an index: {segment}"))?;
                items.get_mut(index)
                    .ok_or_else(|| format!("array pointer index out of range: {index}"))?
            }
            _ => return Err(format!("cannot descend through scalar at {segment}")),
        };
    }
    Ok(current)
}

fn set_pointer(root: &mut Value, pointer: &str, value: Value) -> Result<(), String> {
    let segments = decode_pointer(pointer)?;
    let (leaf, parents) = segments.split_last()
        .ok_or_else(|| "root replacement is not allowed in this corpus".to_owned())?;
    let parent = descend_mut(root, parents)?;
    match parent {
        Value::Object(map) => {
            map.insert(leaf.clone(), value);
            Ok(())
        }
        Value::Array(items) => {
            let index = leaf.parse::<usize>()
                .map_err(|_| format!("array pointer is not an index: {leaf}"))?;
            let slot = items.get_mut(index)
                .ok_or_else(|| format!("array pointer index out of range: {index}"))?;
            *slot = value;
            Ok(())
        }
        _ => Err(format!("cannot set child of scalar at {pointer}")),
    }
}

fn remove_pointer(root: &mut Value, pointer: &str) -> Result<(), String> {
    let segments = decode_pointer(pointer)?;
    let (leaf, parents) = segments.split_last()
        .ok_or_else(|| "root removal is not allowed in this corpus".to_owned())?;
    let parent = descend_mut(root, parents)?;
    match parent {
        Value::Object(map) => map
            .remove(leaf)
            .map(|_| ())
            .ok_or_else(|| format!("object key not found: {leaf}")),
        Value::Array(items) => {
            let index = leaf.parse::<usize>()
                .map_err(|_| format!("array pointer is not an index: {leaf}"))?;
            if index >= items.len() {
                return Err(format!("array pointer index out of range: {index}"));
            }
            items.remove(index);
            Ok(())
        }
        _ => Err(format!("cannot remove child of scalar at {pointer}")),
    }
}

fn apply_mutation(target: &mut Value, mutation: &Value) -> Result<(), String> {
    let operation = string_at(mutation, "operation").unwrap_or("");
    let path = string_at(mutation, "path").unwrap_or("");
    match operation {
        "set" => set_pointer(
            target,
            path,
            mutation.get("value").cloned()
                .ok_or_else(|| format!("set mutation has no value at {path}"))?,
        ),
        "remove" => remove_pointer(target, path),
        "append" => {
            let value = mutation.get("value").cloned()
                .ok_or_else(|| format!("append mutation has no value at {path}"))?;
            let segments = decode_pointer(path)?;
            match descend_mut(target, &segments)? {
                Value::Array(items) => {
                    items.push(value);
                    Ok(())
                }
                _ => Err(format!("append target is not an array: {path}")),
            }
        }
        "copy" => {
            let from_path = string_at(mutation, "from_path")
                .ok_or_else(|| format!("copy mutation has no from_path at {path}"))?;
            let value = target.pointer(from_path).cloned()
                .ok_or_else(|| format!("copy source not found: {from_path}"))?;
            set_pointer(target, path, value)
        }
        _ => Err(format!("unknown mutation operation: {operation}")),
    }
}

fn execute_corpus(corpus: &Value, source_fixture: &Value) -> Result<Vec<Value>, Vec<String>> {
    let mut errors = Vec::new();
    let mut results = Vec::new();
    let cases = corpus.get("scenario_cases").and_then(Value::as_array)
        .ok_or_else(|| vec!["scenario_cases is not an array".to_owned()])?;

    for scenario in cases {
        let id = string_at(scenario, "scenario_id").unwrap_or("<missing>").to_owned();
        let command = string_at(scenario, "command").unwrap_or("");
        let mutations = scenario.get("mutations").and_then(Value::as_array)
            .ok_or_else(|| vec![format!("{id}: mutations is not an array")])?;
        let mut case = source_fixture.clone();
        let mut candidate = source_fixture.clone();
        let target = if command == "audit_case" { &mut case } else { &mut candidate };

        for mutation in mutations {
            if let Err(error) = apply_mutation(target, mutation) {
                errors.push(format!("{id}: {error}"));
            }
        }
        if errors.iter().any(|error| error.starts_with(&format!("{id}:"))) {
            continue;
        }

        let expected = string_at(scenario, "expected_disposition").unwrap_or("");
        let (actual, actual_findings, actual_mismatches) = match command {
            "audit_case" => {
                let report = audit_case(&case).to_value();
                let codes = report.get("findings").and_then(Value::as_array)
                    .into_iter().flatten()
                    .filter_map(|finding| finding.get("code").and_then(Value::as_str))
                    .map(str::to_owned).collect::<Vec<_>>();
                (
                    string_at(&report, "disposition").unwrap_or("").to_owned(),
                    codes,
                    Value::Null
                )
            }
            "compare_pair" => {
                let report = compare_pair(source_fixture, &candidate);
                (
                    string_at(&report, "disposition").unwrap_or("").to_owned(),
                    Vec::new(),
                    report.get("mismatched_fields").cloned().unwrap_or(Value::Null)
                )
            }
            _ => {
                errors.push(format!("{id}: unsupported command {command}"));
                continue;
            }
        };

        let mut case_passed = actual == expected;
        let mut missing_codes = Vec::new();
        if let Some(required) = scenario.get("expected_finding_codes").and_then(Value::as_array) {
            for code in required.iter().filter_map(Value::as_str) {
                if !actual_findings.iter().any(|actual_code| actual_code == code) {
                    case_passed = false;
                    missing_codes.push(code.to_owned());
                }
            }
        }
        let mut mismatched_fields_pass = true;
        if let Some(expected_fields) = scenario.get("expected_mismatched_fields") {
            mismatched_fields_pass = actual_mismatches == *expected_fields;
            case_passed &= mismatched_fields_pass;
        }
        if !case_passed {
            errors.push(format!(
                "{id}: expected disposition {expected}, got {actual}; missing codes={missing_codes:?}; mismatch fields matched={mismatched_fields_pass}"
            ));
        }

        results.push(json!({
            "scenario_id": id,
            "expected_disposition": expected,
            "actual_disposition": actual,
            "missing_expected_finding_codes": missing_codes,
            "expected_mismatched_fields_matched": mismatched_fields_pass,
            "passed": case_passed
        }));
    }
    if errors.is_empty() { Ok(results) } else { Err(errors) }
}

fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() != 2 {
        return Err("Usage: validate-corpus <corpus.json>".to_owned());
    }
    let text = fs::read_to_string(&arguments[1])
        .map_err(|error| format!("{}: {error}", arguments[1]))?;
    let corpus: Value = serde_json::from_str(&text)
        .map_err(|error| format!("{}: invalid JSON: {error}", arguments[1]))?;
    let mut report = match validate(&corpus) {
        Ok(report) => report,
        Err(errors) => {
            for error in errors {
                eprintln!("CORPUS_INVALID: {error}");
            }
            return Err("corpus validation failed".to_owned());
        }
    };
    let fixture_path = string_at(&corpus, "source_fixture")
        .ok_or_else(|| "source_fixture is required".to_owned())?;
    let fixture_text = fs::read_to_string(fixture_path)
        .map_err(|error| format!("{fixture_path}: {error}"))?;
    let fixture: Value = serde_json::from_str(&fixture_text)
        .map_err(|error| format!("{fixture_path}: invalid JSON: {error}"))?;
    let results = execute_corpus(&corpus, &fixture).map_err(|errors| errors.join("\\n"))?;
    report["status"] = json!("CORPUS_EXECUTED");
    report["executed_scenario_count"] = json!(results.len());
    report["scenario_results"] = Value::Array(results);
    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../docs/research/marine/fixtures/marine-sail-liveaboard-corpus-0001.json"
        )).expect("corpus fixture must parse")
    }

    #[test]
    fn frozen_corpus_has_required_policy_arms_and_threat_coverage() {
        let report = validate(&fixture()).expect("corpus should validate");
        assert_eq!(report["status"], "DECLARATIVE_CORPUS_VALID");
        assert_eq!(report["scenario_count"], 16);
        assert_eq!(report["policy_arm_count"], 7);
    }

    #[test]
    fn frozen_mutation_vectors_execute_against_real_audit_engine() {
        let corpus = fixture();
        let source_fixture: Value = serde_json::from_str(include_str!(
            "../../../../docs/research/marine/fixtures/synthetic-sail-liveaboard-case-0001.json"
        )).expect("base case must parse");
        let results = execute_corpus(&corpus, &source_fixture)
            .expect("all frozen mutation vectors should match the current audit engine");
        assert_eq!(results.len(), 16);
        assert!(results.iter().all(|result| result["passed"] == true));
    }

    #[test]
    fn duplicate_scenario_ids_are_rejected() {
        let mut value = fixture();
        value["scenario_cases"][1]["scenario_id"] = value["scenario_cases"][0]["scenario_id"].clone();
        let errors = validate(&value).expect_err("duplicate id must fail");
        assert!(errors.iter().any(|error| error.contains("duplicate scenario_id")));
    }

    #[test]
    fn missing_required_threat_family_is_rejected() {
        let mut value = fixture();
        value["required_threat_classes"] = serde_json::json!([]);
        let errors = validate(&value).expect_err("missing threat coverage must fail");
        assert!(errors.iter().any(|error| error.contains("required threat class missing")));
    }

    #[test]
    fn deployable_arm_cannot_consume_realized_environment_information() {
        let mut value = fixture();
        value["policy_arms"][0]["information_policy"] =
            serde_json::json!("realized_environment_evaluator_only");
        let errors = validate(&value).expect_err("deployable information mismatch must fail");
        assert!(errors.iter().any(|error| error.contains("wrong information policy")));
    }

    #[test]
    fn malformed_json_pointer_is_rejected() {
        assert!(!valid_pointer("/environment/invalid~2pointer"));
        assert!(valid_pointer("/environment/decision_snapshots/0/available_at"));
    }

    #[test]
    fn target_improvement_is_not_silently_reduced() {
        let mut value = fixture();
        value["metric_contract"]["target_improvement_fraction"] = serde_json::json!(0.01);
        let errors = validate(&value).expect_err("target change must require review");
        assert!(errors.iter().any(|error| error.contains("target must remain 10%")));
    }
}
