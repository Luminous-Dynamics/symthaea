use serde_json::Value;
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
                    } else if op != "remove" && !mutation.get("value").is_some() {
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

fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() != 2 {
        return Err("Usage: validate-corpus <corpus.json>".to_owned());
    }
    let text = fs::read_to_string(&arguments[1])
        .map_err(|error| format!("{}: {error}", arguments[1]))?;
    let corpus: Value = serde_json::from_str(&text)
        .map_err(|error| format!("{}: invalid JSON: {error}", arguments[1]))?;
    match validate(&corpus) {
        Ok(report) => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
        Err(errors) => {
            for error in errors {
                eprintln!("CORPUS_INVALID: {error}");
            }
            return Err("corpus validation failed".to_owned());
        }
    }
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
