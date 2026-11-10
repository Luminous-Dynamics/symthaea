use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    process::ExitCode,
};

const EXPECTED_SCENARIOS: usize = 40;
const EXPECTED_NEGATIVE_CONTROLS: usize = 24;
const AGREEMENT_THRESHOLD: f64 = 0.8;

fn field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|value| !value.trim().is_empty())
}

fn string_set(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn add(errors: &mut Vec<String>, code: &str) {
    errors.push(code.to_string());
}

fn allowed_action(action: &str) -> bool {
    matches!(
        action,
        "ContinueDebate"
            | "DiversifyHypotheses"
            | "RequestEvidence"
            | "VerifyClaim"
            | "AdversarialChallenge"
            | "PreserveMinority"
            | "IndependentAdjudication"
            | "FinalizeSupported"
            | "FinalizeIndeterminate"
            | "PluralisticMultiplePositions"
            | "StopBudgetExhausted"
            | "StopInvalid"
    )
}

fn reference_action(state: &Value) -> &'static str {
    // Integrity and budget gates precede epistemic optimization.
    if state["input_integrity_valid"].as_bool() != Some(true) {
        return "StopInvalid";
    }
    if state["budget_remaining"].as_i64() == Some(0) {
        return "StopBudgetExhausted";
    }
    if state["orchestrator_compromise"].as_bool() == Some(true)
        || state["source_lineage_compromised"].as_bool() == Some(true)
    {
        return "IndependentAdjudication";
    }
    if state["common_mode_signal"].as_bool() == Some(true) {
        return "AdversarialChallenge";
    }

    let unresolved = state["high_impact_unresolved_count"].as_u64().unwrap_or(0);
    let contradictions = state["contradictory_source_count"].as_u64().unwrap_or(0);
    let diverse = state["hypothesis_diversity_count"].as_u64().unwrap_or(0) >= 2;
    let evidence_complete = state["evidence_complete"].as_bool() == Some(true);
    let minority_verified = field(state, "minority_candidate_state") == Some("Verified");
    let minority_exists = field(state, "minority_candidate_state") != Some("None");
    let minority_preserved = state["minority_preserved"].as_bool() == Some(true);

    match field(state, "task_truth_profile") {
        Some("SyntheticKnownAnswer") => {
            if minority_verified && !minority_preserved {
                return "PreserveMinority";
            }
            if unresolved > 0 || contradictions > 0 {
                return "VerifyClaim";
            }
            if !evidence_complete {
                return if diverse {
                    "RequestEvidence"
                } else {
                    "DiversifyHypotheses"
                };
            }
            "FinalizeSupported"
        }
        Some("OpenWorldIndeterminate") => {
            if unresolved > 0 || contradictions > 0 {
                return "VerifyClaim";
            }
            if !evidence_complete {
                return if diverse {
                    "RequestEvidence"
                } else {
                    "DiversifyHypotheses"
                };
            }
            "FinalizeIndeterminate"
        }
        Some("NormativePluralism") => {
            if unresolved > 0 || contradictions > 0 {
                return "VerifyClaim";
            }
            if !evidence_complete {
                return "RequestEvidence";
            }
            if minority_exists && !minority_preserved {
                return "PreserveMinority";
            }
            "PluralisticMultiplePositions"
        }
        _ => "StopInvalid",
    }
}

fn validate_scenario(scenario: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    if field(scenario, "fixture_id").is_none() {
        add(&mut errors, "fixture_id_invalid");
    }
    let state = &scenario["state"];
    if !matches!(
        field(state, "task_truth_profile"),
        Some("SyntheticKnownAnswer" | "OpenWorldIndeterminate" | "NormativePluralism")
    ) {
        add(&mut errors, "task_truth_profile_invalid");
    }
    if state["input_integrity_valid"].as_bool().is_none() {
        add(&mut errors, "input_integrity_flag_invalid");
    }
    if state["budget_remaining"].as_i64().is_none_or(|value| value < 0) {
        add(&mut errors, "budget_remaining_invalid");
    }
    if state["rounds_used"].as_i64().is_none_or(|value| value < 0) {
        add(&mut errors, "rounds_used_invalid");
    }
    if state["round_limit"].as_i64().is_none_or(|value| value <= 0) {
        add(&mut errors, "round_limit_invalid");
    }
    let agreement = state["agent_agreement"].as_f64();
    if agreement.is_none_or(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        add(&mut errors, "agent_agreement_out_of_range");
    }
    if state["evidence_complete"].as_bool().is_none() {
        add(&mut errors, "evidence_complete_invalid");
    }
    if state["high_impact_unresolved_count"].as_i64().is_none_or(|value| value < 0) {
        add(&mut errors, "unresolved_claim_count_invalid");
    }
    if state["contradictory_source_count"].as_i64().is_none_or(|value| value < 0) {
        add(&mut errors, "contradiction_count_invalid");
    }
    if state["hypothesis_diversity_count"].as_i64().is_none_or(|value| value < 0) {
        add(&mut errors, "hypothesis_diversity_count_invalid");
    }
    if !matches!(
        field(state, "minority_candidate_state"),
        Some("None" | "Unverified" | "Verified")
    ) {
        add(&mut errors, "minority_state_invalid");
    }
    if state["minority_preserved"].as_bool().is_none() {
        add(&mut errors, "minority_preserved_invalid");
    }
    if state["common_mode_signal"].as_bool().is_none() {
        add(&mut errors, "common_mode_signal_invalid");
    }
    if state["orchestrator_compromise"].as_bool().is_none() {
        add(&mut errors, "orchestrator_compromise_invalid");
    }
    if state["source_lineage_compromised"].as_bool().is_none() {
        add(&mut errors, "source_lineage_flag_invalid");
    }

    let confidence = &state["confidence_signal"];
    match field(confidence, "status") {
        Some("NotProvided") => {
            if !confidence["probability_correct"].is_null()
                || !confidence["calibration_profile_sha256"].is_null()
            {
                add(&mut errors, "confidence_not_provided_not_empty");
            }
        }
        Some("SelfReported" | "OutOfDomain") => {
            if confidence["probability_correct"]
                .as_f64()
                .is_none_or(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
            {
                add(&mut errors, "confidence_probability_out_of_range");
            }
            if !confidence["calibration_profile_sha256"].is_null() {
                add(&mut errors, "confidence_profile_not_applicable");
            }
        }
        Some("CalibratedInScope") => {
            if confidence["probability_correct"]
                .as_f64()
                .is_none_or(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
            {
                add(&mut errors, "confidence_probability_out_of_range");
            }
            let digest = field(confidence, "calibration_profile_sha256").unwrap_or_default();
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                add(&mut errors, "confidence_profile_missing");
            }
        }
        Some(_) | None => add(&mut errors, "confidence_status_invalid"),
    }

    let expected_action = field(scenario, "expected_action");
    if expected_action.is_none_or(|action| !allowed_action(action)) {
        add(&mut errors, "expected_action_invalid");
    }
    // Invalid values do not get a second, misleading baseline-consistency diagnosis.
    if let Some(value) = agreement.filter(|value| value.is_finite() && (0.0..=1.0).contains(value)) {
        if scenario["expected_agreement_baseline_stop"].as_bool()
            != Some(value >= AGREEMENT_THRESHOLD)
        {
            add(&mut errors, "agreement_baseline_mismatch");
        }
    }
    if let (Some(rounds), Some(limit)) = (
        state["rounds_used"].as_i64().filter(|value| *value >= 0),
        state["round_limit"].as_i64().filter(|value| *value > 0),
    ) {
        if scenario["expected_fixed_round_baseline_stop"].as_bool() != Some(rounds >= limit) {
            add(&mut errors, "fixed_round_baseline_mismatch");
        }
    }

    if errors.is_empty() && expected_action != Some(reference_action(state)) {
        add(&mut errors, "reference_action_mismatch");
    }
    errors.sort();
    errors.dedup();
    errors
}

fn set_at_path(root: &mut Value, path: &str, replacement: Value) -> bool {
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    if parts.is_empty() || parts.iter().any(|part| part.is_empty()) {
        return false;
    }
    let mut current = root;
    for part in &parts[..parts.len() - 1] {
        current = match current {
            Value::Object(map) => match map.get_mut(*part) {
                Some(value) => value,
                None => return false,
            },
            Value::Array(items) => {
                match part.parse::<usize>().ok().and_then(|index| items.get_mut(index)) {
                    Some(value) => value,
                    None => return false,
                }
            },
            _ => return false,
        };
    }
    let last = parts[parts.len() - 1];
    match current {
        Value::Object(map) => {
            if !map.contains_key(last) { return false; }
            map.insert(last.to_string(), replacement);
            true
        }
        Value::Array(items) => match last.parse::<usize>() {
            Ok(index) if index < items.len() => { items[index] = replacement; true }
            _ => false,
        },
        _ => false,
    }
}

fn remove_at_path(root: &mut Value, path: &str) -> bool {
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    if parts.is_empty() || parts.iter().any(|part| part.is_empty()) {
        return false;
    }
    let mut current = root;
    for part in &parts[..parts.len() - 1] {
        current = match current {
            Value::Object(map) => match map.get_mut(*part) {
                Some(value) => value,
                None => return false,
            },
            Value::Array(items) => {
                match part.parse::<usize>().ok().and_then(|index| items.get_mut(index)) {
                    Some(value) => value,
                    None => return false,
                }
            },
            _ => return false,
        };
    }
    let last = parts[parts.len() - 1];
    match current {
        Value::Object(map) => map.remove(last).is_some(),
        Value::Array(items) => match last.parse::<usize>() {
            Ok(index) if index < items.len() => { items.remove(index); true }
            _ => false,
        },
        _ => false,
    }
}

fn apply_mutation(scenario: &mut Value, mutation: &Value) -> bool {
    let path = field(mutation, "path").unwrap_or_default();
    match field(mutation, "operation") {
        Some("set") => set_at_path(scenario, path, mutation["value"].clone()),
        Some("remove") => remove_at_path(scenario, path),
        _ => false,
    }
}

fn validate_manifest(scenarios: &Value, controls: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let Some(rows) = scenarios["scenarios"].as_array() else {
        return vec!["scenario_array_missing".to_string()];
    };
    if rows.len() != EXPECTED_SCENARIOS
        || scenarios["fixture_count"].as_u64() != Some(EXPECTED_SCENARIOS as u64)
    {
        add(&mut errors, "scenario_count_invalid");
    }
    if scenarios["family_count"].as_u64() != Some(8) {
        add(&mut errors, "scenario_family_count_invalid");
    }
    let mut ids = BTreeSet::new();
    let mut family_counts = BTreeMap::<String, usize>::new();
    let mut action_counts = BTreeMap::<String, usize>::new();
    for row in rows {
        if !ids.insert(field(row, "fixture_id").unwrap_or_default().to_string()) {
            add(&mut errors, "duplicate_fixture_id");
        }
        let family = field(row, "family_id").unwrap_or_default();
        if !families_contains(family) {
            add(&mut errors, "scenario_family_invalid");
        }
        *family_counts.entry(family.to_string()).or_default() += 1;
        for code in validate_scenario(row) {
            errors.push(format!("{}:{code}", field(row, "fixture_id").unwrap_or("unknown")));
        }
        if let Some(action) = field(row, "expected_action") {
            *action_counts.entry(action.to_string()).or_default() += 1;
        }
    }
    for family in [
        "factual_evidence_closure",
        "false_consensus",
        "minority_preservation",
        "open_world",
        "normative_pluralism",
        "controller_adversary",
        "budget_and_cost",
        "trajectory_value",
    ] {
        if family_counts.get(family) != Some(&5) {
            add(&mut errors, "scenario_family_coverage_invalid");
        }
    }
    for action in [
        "FinalizeSupported",
        "FinalizeIndeterminate",
        "PluralisticMultiplePositions",
        "StopBudgetExhausted",
    ] {
        if !action_counts.contains_key(action) {
            add(&mut errors, "required_terminal_actions_missing");
        }
    }

    let negative = controls["fixtures"].as_array().cloned().unwrap_or_default();
    if negative.len() != EXPECTED_NEGATIVE_CONTROLS
        || controls["fixture_count"].as_u64() != Some(EXPECTED_NEGATIVE_CONTROLS as u64)
    {
        add(&mut errors, "negative_control_count_invalid");
    }
    for fixture in &negative {
        let target_id = field(fixture, "target_fixture_id").unwrap_or_default();
        let Some(target) = rows.iter().find(|row| field(row, "fixture_id") == Some(target_id)) else {
            errors.push(format!(
                "{}:negative_control_target_missing",
                field(fixture, "fixture_id").unwrap_or("unknown")
            ));
            continue;
        };
        let mut mutated = target.clone();
        if !apply_mutation(&mut mutated, &fixture["mutation"]) {
            errors.push(format!(
                "{}:negative_control_mutation_invalid",
                field(fixture, "fixture_id").unwrap_or("unknown")
            ));
            continue;
        }
        let actual: BTreeSet<String> = validate_scenario(&mutated).into_iter().collect();
        let expected = string_set(&fixture["expected_violations"]);
        if actual != expected {
            errors.push(format!(
                "{}:negative_control_oracle_mismatch:{actual:?}:{expected:?}",
                field(fixture, "fixture_id").unwrap_or("unknown")
            ));
        }
    }
    errors.sort();
    errors.dedup();
    errors
}

fn families_contains(value: &str) -> bool {
    matches!(
        value,
        "factual_evidence_closure"
            | "false_consensus"
            | "minority_preservation"
            | "open_world"
            | "normative_pluralism"
            | "controller_adversary"
            | "budget_and_cost"
            | "trajectory_value"
    )
}

fn read_json(path: &str) -> Result<Value, String> {
    let contents = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    serde_json::from_str(&contents).map_err(|error| format!("{path}: invalid JSON: {error}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: ig-delib-control-qualifier <scenarios.json> <negative-controls.json>");
        return ExitCode::from(2);
    }
    let scenarios = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => { eprintln!("{error}"); return ExitCode::FAILURE; }
    };
    let controls = match read_json(&args[2]) {
        Ok(value) => value,
        Err(error) => { eprintln!("{error}"); return ExitCode::FAILURE; }
    };
    let errors = validate_manifest(&scenarios, &controls);
    if !errors.is_empty() {
        eprintln!("DELIBERATION CONTROL QUALIFICATION FAILED:");
        for error in errors { eprintln!(" - {error}"); }
        return ExitCode::FAILURE;
    }
    println!(
        "QUALIFIED DETERMINISTIC REFERENCE CONTROLLER: scenarios=40 families=8 negative_controls=24"
    );
    println!(
        "BASELINE WARNING: agreement-stop and fixed-round-stop are comparators, not correctness criteria."
    );
    println!("NOT A LEARNED MODERATOR: no model policy has been trained or evaluated.");
    println!("NOT EXECUTED: no LLM debate or human deliberation was run.");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCENARIOS: &str = include_str!("../../DELIB_CONTROL_SCENARIOS_V0_1.json");
    const CONTROLS: &str = include_str!("../../DELIB_CONTROL_NEGATIVE_CONTROLS_V0_1.json");

    fn parsed(text: &str) -> Value {
        serde_json::from_str(text).expect("checked-in JSON must parse")
    }

    #[test]
    fn checked_in_forty_scenarios_and_negative_controls_validate() {
        let errors = validate_manifest(&parsed(SCENARIOS), &parsed(CONTROLS));
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn near_unanimity_does_not_override_missing_evidence() {
        let rows = parsed(SCENARIOS);
        let row = &rows["scenarios"][5];
        assert_eq!(field(row, "expected_action"), Some("RequestEvidence"));
        assert!(row["expected_agreement_baseline_stop"].as_bool().unwrap());
    }

    #[test]
    fn confidence_and_agreement_are_not_controller_authority() {
        let rows = parsed(SCENARIOS);
        let mut row = rows["scenarios"][5].clone();
        let before = reference_action(&row["state"]);
        row["state"]["agent_agreement"] = Value::from(0.99);
        row["state"]["confidence_signal"]["status"] = Value::String("SelfReported".into());
        row["state"]["confidence_signal"]["probability_correct"] = Value::from(0.99);
        assert_eq!(before, reference_action(&row["state"]));
    }

    #[test]
    fn verified_minority_answer_is_preserved_before_final_selection() {
        let rows = parsed(SCENARIOS);
        let row = &rows["scenarios"][10];
        assert_eq!(reference_action(&row["state"]), "PreserveMinority");
    }

    #[test]
    fn open_world_without_resolution_ends_indeterminate() {
        let rows = parsed(SCENARIOS);
        let row = &rows["scenarios"][12];
        assert_eq!(reference_action(&row["state"]), "FinalizeIndeterminate");
    }

    #[test]
    fn normative_pluralism_never_gets_a_single_correctness_action() {
        let rows = parsed(SCENARIOS);
        let plural = rows["scenarios"].as_array().unwrap()
            .iter().filter(|row| row["state"]["task_truth_profile"] == "NormativePluralism");
        for row in plural {
            assert_ne!(reference_action(&row["state"]), "FinalizeSupported");
            assert_ne!(reference_action(&row["state"]), "FinalizeIndeterminate");
        }
    }

    #[test]
    fn budget_exhaustion_is_never_reported_as_success() {
        let rows = parsed(SCENARIOS);
        for row in rows["scenarios"].as_array().unwrap() {
            if row["state"]["budget_remaining"] == 0 {
                assert_eq!(reference_action(&row["state"]), "StopBudgetExhausted");
            }
        }
    }

    #[test]
    fn early_evidence_closure_can_stop_before_round_limit() {
        let rows = parsed(SCENARIOS);
        let row = rows["scenarios"].as_array().unwrap()
            .iter().find(|row| row["expected_action"] == "FinalizeSupported").unwrap();
        assert_eq!(row["expected_fixed_round_baseline_stop"], false);
        assert_eq!(reference_action(&row["state"]), "FinalizeSupported");
    }

    #[test]
    fn all_negative_controls_match_exact_violation_codes() {
        let scenarios = parsed(SCENARIOS);
        let controls = parsed(CONTROLS);
        let rows = scenarios["scenarios"].as_array().unwrap();
        for fixture in controls["fixtures"].as_array().unwrap() {
            let target = field(fixture, "target_fixture_id").unwrap();
            let mut scenario = rows
                .iter()
                .find(|row| field(row, "fixture_id") == Some(target))
                .unwrap()
                .clone();
            assert!(apply_mutation(&mut scenario, &fixture["mutation"]));
            let actual: BTreeSet<String> = validate_scenario(&scenario).into_iter().collect();
            assert_eq!(
                actual,
                string_set(&fixture["expected_violations"]),
                "{}",
                field(fixture, "fixture_id").unwrap()
            );
        }
    }
}
