//! Offline evaluator for Culinary Decision Quality / VOI Benchmark v1.
//!
//! The candidate policy must run separately and receive only each case's
//! `policy_input`. This program reads the corpus and frozen policy outputs after
//! inference, then scores them against oracle-only expected fields. Do not use
//! this executable as a policy input adapter or expose its report during tuning.

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process;

fn as_string_set(value: Option<&Value>) -> BTreeSet<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn output_string(value: &Value, key: &str) -> Option<&str> {
    value.get(key).and_then(Value::as_str)
}

fn output_string_set(value: &Value, key: &str) -> BTreeSet<String> {
    as_string_set(value.get(key))
}

fn candidate_id_set(value: &Value) -> BTreeSet<String> {
    as_string_set(value.get("selected_candidate_ids"))
}

fn expected_field_match(
    case_id: &str,
    key: &str,
    expected: &Value,
    actual: &Value,
    mismatches: &mut Vec<String>,
) {
    if expected.get(key).is_none() {
        return;
    }

    let matches = match key {
        "selected_candidate_ids" => {
            as_string_set(expected.get(key)) == candidate_id_set(actual)
        }
        "required_flags" => {
            as_string_set(expected.get(key)).is_subset(&output_string_set(actual, "flags"))
        }
        "eligible_source_refs" | "ineligible_source_refs" => {
            as_string_set(expected.get(key)) == output_string_set(actual, key)
        }
        "consumer_output_allowed_fields" => {
            output_string_set(actual, "consumer_output_fields")
                .is_subset(&as_string_set(expected.get(key)))
        }
        "canonical_output_equality" => permutation_outputs_equal(actual),
        _ => expected.get(key) == actual.get(key),
    };

    if !matches {
        mismatches.push(format!("{case_id}: field contract mismatch: {key}"));
    }
}

fn permutation_outputs_equal(output: &Value) -> bool {
    let Some(items) = output.get("permutation_outputs").and_then(Value::as_array) else {
        return false;
    };
    if items.len() != 2 {
        return false;
    }
    let (Some(first), Some(second)) = (items[0].get("decision"), items[1].get("decision")) else {
        return false;
    };
    first == second
}

fn is_unresolved_gate(value: &Value) -> bool {
    match value {
        Value::String(text) => matches!(
            text.to_ascii_lowercase().as_str(),
            "unknown" | "unresolved" | "indeterminate" | "failed" | "fail" | "blocked"
        ),
        Value::Object(map) => map.values().any(is_unresolved_gate),
        Value::Array(items) => items.iter().any(is_unresolved_gate),
        _ => false,
    }
}

fn selected_ids(output: &Value) -> BTreeSet<String> {
    candidate_id_set(output)
}

fn candidate_by_id<'a>(case: &'a Value, id: &str) -> Option<&'a Value> {
    case.get("policy_input")?
        .get("candidates")?
        .as_array()?
        .iter()
        .find(|candidate| candidate.get("id").and_then(Value::as_str) == Some(id))
}

fn recursively_has_key_value(value: &Value, key_name: &str, expected_value: &str) -> bool {
    match value {
        Value::Object(map) => {
            map.iter().any(|(key, child)| {
                (key == "status" && child.as_str() == Some(status))
                    || recursively_has_status(child, status)
            })
        }
        Value::Array(items) => items.iter().any(|child| recursively_has_status(child, status)),
        _ => false,
    }
}

fn hard_gate_or_missing_data_violations(case: &Value, output: &Value) -> Vec<String> {
    let case_id = case.get("id").and_then(Value::as_str).unwrap_or("<unknown-case>");
    let disposition = output_string(output, "disposition").unwrap_or("");
    let selected = selected_ids(output);
    let mut violations = Vec::new();

    // A selected candidate cannot pass through a failed or unresolved hard gate.
    for id in &selected {
        if let Some(candidate) = candidate_by_id(case, id) {
            if let Some(gates) = candidate.get("hard_gates") {
                if is_unresolved_gate(gates) {
                    violations.push(format!(
                        "{case_id}: selected candidate {id} has failed/unresolved hard gate"
                    ));
                }
            }
            if disposition == "recommend"
                && candidate
                    .get("nutrition")
                    .is_some_and(|nutrition| recursively_has_key_value(nutrition, "status", "missing"))
            {
                violations.push(format!(
                    "{case_id}: recommendation selects {id} with missing nutrition evidence"
                ));
            }
            if disposition == "recommend"
                && candidate
                    .get("cost")
                    .is_some_and(|cost| (recursively_has_key_value(cost, "status", "stale")\n                    || recursively_has_key_value(cost, "freshness", "stale")))
            {
                violations.push(format!(
                    "{case_id}: recommendation selects {id} with stale cost evidence"
                ));
            }
        }
    }

    // Enforce declared per-group limits on any selected portfolio.
    for id in &selected {
        if let Some(candidate) = candidate_by_id(case, id) {
            let (Some(costs), Some(limits)) = (
                candidate.get("group_costs_zar").and_then(Value::as_object),
                candidate.get("maximum_per_group_zar").and_then(Value::as_object),
            ) else {
                continue;
            };
            for (group, cost) in costs {
                let (Some(cost), Some(limit)) =
                    (cost.as_f64(), limits.get(group).and_then(Value::as_f64))
                else {
                    continue;
                };
                if cost > limit {
                    violations.push(format!(
                        "{case_id}: selected candidate {id} exceeds the declared affordability limit for {group}"
                    ));
                }
            }
        }
    }

    // The scorer checks source-time eligibility independently from the stated disposition.
    let expected_ineligible = as_string_set(case.get("oracle_only_expected").and_then(|v| v.get("ineligible_source_refs")));
    let source_refs_used = output_string_set(output, "source_refs_used");
    for ref_id in expected_ineligible.intersection(&source_refs_used) {
        violations.push(format!(
            "{case_id}: policy used a source ref that is ineligible at the replay cutoff: {ref_id}"
        ));
    }

    // Consumer projection must never expose fields excluded by the oracle contract.
    if let Some(allowed) = case
        .get("oracle_only_expected")
        .and_then(|v| v.get("consumer_output_allowed_fields"))
    {
        let allowed = as_string_set(Some(allowed));
        for field in output_string_set(output, "consumer_output_fields").difference(&allowed) {
            violations.push(format!(
                "{case_id}: consumer projection contains disallowed field: {field}"
            ));
        }
    }

    // Simulated observations remain simulated in the policy's emitted claims.
    let simulated_refs = collect_simulated_refs(case);
    if let Some(claims) = output.get("evidence_claims").and_then(Value::as_array) {
        for claim in claims {
            let ref_id = output_string(claim, "source_ref").unwrap_or("");
            if simulated_refs.contains(ref_id)
                && output_string(claim, "status") == Some("observed")
            {
                violations.push(format!(
                    "{case_id}: simulated evidence {ref_id} was relabelled as observed"
                ));
            }
        }
    }

    // A recalled lot cannot stay eligible in the selected recommendation.
    if let Some(update) = case.get("policy_input").and_then(|v| v.get("source_update")) {
        let recalled = output_string(update, "lot_ref");
        if output_string(update, "state") == Some("recalled") {
            if let Some(recalled_lot) = recalled {
                for id in &selected {
                    if candidate_by_id(case, id)
                        .and_then(|candidate| candidate.get("ingredient_lot_ref"))
                        .and_then(Value::as_str)
                        == Some(recalled_lot)
                    {
                        violations.push(format!(
                            "{case_id}: selected candidate {id} references recalled lot {recalled_lot}"
                        ));
                    }
                }
            }
        }
    }

    violations
}

fn collect_simulated_refs(case: &Value) -> BTreeSet<String> {
    fn visit(value: &Value, found: &mut BTreeSet<String>) {
        match value {
            Value::Object(map) => {
                if map.get("status").and_then(Value::as_str) == Some("simulated") {
                    if let Some(source) = map.get("source_ref").and_then(Value::as_str) {
                        found.insert(source.to_owned());
                    }
                }
                for child in map.values() {
                    visit(child, found);
                }
            }
            Value::Array(items) => {
                for item in items {
                    visit(item, found);
                }
            }
            _ => {}
        }
    }
    let mut found = BTreeSet::new();
    if let Some(input) = case.get("policy_input") {
        visit(input, &mut found);
    }
    found
}

fn score_case(case: &Value, output: Option<&Value>) -> Value {
    let case_id = case.get("id").and_then(Value::as_str).unwrap_or("<unknown-case>");
    let expected = case.get("oracle_only_expected").unwrap_or(&Value::Null);

    let Some(output) = output else {
        return json!({
            "case_id": case_id,
            "status": "missing_output",
            "disposition_match": false,
            "contract_fields_match": false,
            "semantic_violations": [],
            "mismatches": ["no policy output was supplied for this case"]
        });
    };

    let expected_disposition = expected.get("disposition").and_then(Value::as_str);
    let actual_disposition = output.get("disposition").and_then(Value::as_str);
    let disposition_match = expected_disposition.is_some() && expected_disposition == actual_disposition;
    let mut mismatches = Vec::new();
    if !disposition_match {
        mismatches.push(format!(
            "{case_id}: expected disposition {:?}, received {:?}",
            expected_disposition, actual_disposition
        ));
    }

    let mut scored_fields = Vec::new();
    for key in [
        "selected_candidate_ids",
        "measurement_id",
        "required_flags",
        "eligible_source_refs",
        "ineligible_source_refs",
        "consumer_output_allowed_fields",
        "canonical_output_equality",
    ] {
        if expected.get(key).is_some() {
            expected_field_match(case_id, key, expected, output, &mut mismatches);
            scored_fields.push(key);
        }
    }
    let violations = hard_gate_or_missing_data_violations(case, output);
    let contract_fields_match = mismatches.is_empty();
    let success = disposition_match && contract_fields_match && violations.is_empty();

    json!({
        "case_id": case_id,
        "status": if success { "match" } else { "mismatch" },
        "expected_disposition": expected_disposition,
        "actual_disposition": actual_disposition,
        "disposition_match": disposition_match,
        "scored_expected_fields": scored_fields,
        "contract_fields_match": contract_fields_match,
        "semantic_violations": violations,
        "mismatches": mismatches
    })
}

fn evaluate(corpus: &Value, policy_report: &Value) -> Result<Value, String> {
    let corpus_id = corpus.get("corpus_id").and_then(Value::as_str);
    let corpus_version = corpus.get("corpus_version").and_then(Value::as_str);
    if corpus_id != Some("culinary-decision-quality-voi-v1")
        || corpus_version != Some("1.0.0")
        || corpus.get("not_real_world_data").and_then(Value::as_bool) != Some(true)
    {
        return Err("unsupported corpus identity/version or missing synthetic-data marker".into());
    }
    if policy_report.get("corpus_id").and_then(Value::as_str) != corpus_id
        || policy_report.get("corpus_version").and_then(Value::as_str) != corpus_version
    {
        return Err("policy output corpus_id/corpus_version does not match the supplied corpus".into());
    }

    let cases = corpus
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| "corpus.cases must be an array".to_owned())?;
    let outputs = policy_report
        .get("outputs")
        .and_then(Value::as_array)
        .ok_or_else(|| "policy report outputs must be an array".to_owned())?;

    let expected_ids: BTreeSet<String> = cases
        .iter()
        .filter_map(|case| case.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect();
    let mut outputs_by_id: BTreeMap<String, &Value> = BTreeMap::new();
    let mut duplicate_ids = BTreeSet::new();
    let mut unknown_ids = BTreeSet::new();
    for output in outputs {
        let Some(id) = output.get("case_id").and_then(Value::as_str) else {
            unknown_ids.insert("<missing-case-id>".to_owned());
            continue;
        };
        if !expected_ids.contains(id) {
            unknown_ids.insert(id.to_owned());
        }
        if outputs_by_id.insert(id.to_owned(), output).is_some() {
            duplicate_ids.insert(id.to_owned());
        }
    }

    let mut per_case = Vec::new();
    let mut disposition_matches = 0usize;
    let mut exact_matches = 0usize;
    let mut violation_count = 0usize;
    for case in cases {
        let id = case.get("id").and_then(Value::as_str).unwrap_or("<unknown-case>");
        let scored = score_case(case, outputs_by_id.get(id).copied());
        if scored["disposition_match"] == true {
            disposition_matches += 1;
        }
        if scored["status"] == "match" {
            exact_matches += 1;
        }
        violation_count += scored["semantic_violations"]
            .as_array()
            .map_or(0, Vec::len);
        per_case.push(scored);
    }

    let missing_ids: BTreeSet<String> = expected_ids
        .difference(&outputs_by_id.keys().cloned().collect())
        .cloned()
        .collect();
    let structural_errors = missing_ids.len() + duplicate_ids.len() + unknown_ids.len();
    let passed = structural_errors == 0 && exact_matches == cases.len() && violation_count == 0;
    let case_count = cases.len();
    Ok(json!({
        "benchmark_id": corpus_id,
        "benchmark_version": corpus_version,
        "evidence_class": "synthetic_contract_fixture",
        "policy_outputs_are_complete": missing_ids.is_empty() && duplicate_ids.is_empty() && unknown_ids.is_empty(),
        "summary": {
            "case_count": case_count,
            "outputs_received": outputs.len(),
            "disposition_matches": disposition_matches,
            "exact_contract_matches": exact_matches,
            "semantic_violation_count": violation_count,
            "missing_case_ids": missing_ids,
            "duplicate_case_ids": duplicate_ids,
            "unknown_or_malformed_output_ids": unknown_ids,
            "structural_error_count": structural_errors,
            "pass": passed
        },
        "per_case": per_case,
        "claim_boundary": "A pass establishes only agreement on this finite synthetic corpus. It does not establish real food data truth, probability calibration, food safety, household benefit, or external validity."
    }))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 && args.len() != 4 {
        return Err(format!(
            "usage: {} <corpus.json> <frozen-policy-outputs.json> [report.json]",
            args.first().map(String::as_str).unwrap_or("culinary_decision_quality_eval")
        ));
    }

    let corpus_text = fs::read_to_string(&args[1])
        .map_err(|error| format!("cannot read corpus {}: {error}", args[1]))?;
    let policy_text = fs::read_to_string(&args[2])
        .map_err(|error| format!("cannot read policy report {}: {error}", args[2]))?;
    let corpus: Value = serde_json::from_str(&corpus_text)
        .map_err(|error| format!("invalid corpus JSON: {error}"))?;
    let policy: Value = serde_json::from_str(&policy_text)
        .map_err(|error| format!("invalid policy output JSON: {error}"))?;
    let report = evaluate(&corpus, &policy)?;
    let rendered = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("cannot render evaluation report: {error}"))?;

    if args.len() == 4 {
        fs::write(&args[3], format!("{rendered}\n"))
            .map_err(|error| format!("cannot write evaluation report {}: {error}", args[3]))?;
    } else {
        println!("{rendered}");
    }

    if report["summary"]["pass"] != true {
        process::exit(1);
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("culinary decision-quality evaluation failed: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn missing_and_duplicate_outputs_fail_closed() {
        let corpus = json!({
            "corpus_id": "culinary-decision-quality-voi-v1",
            "corpus_version": "1.0.0",
            "not_real_world_data": true,
            "cases": [
                {"id":"case-a","oracle_only_expected":{"disposition":"recommend"}}
            ]
        });
        let missing = json!({
            "corpus_id":"culinary-decision-quality-voi-v1",
            "corpus_version":"1.0.0",
            "outputs":[]
        });
        let report = evaluate(&corpus, &missing).unwrap();
        assert_eq!(report["summary"]["pass"], false);
        assert_eq!(report["summary"]["missing_case_ids"][0], "case-a");

        let duplicated = json!({
            "corpus_id":"culinary-decision-quality-voi-v1",
            "corpus_version":"1.0.0",
            "outputs":[
                {"case_id":"case-a","disposition":"recommend"},
                {"case_id":"case-a","disposition":"recommend"}
            ]
        });
        let report = evaluate(&corpus, &duplicated).unwrap();
        assert_eq!(report["summary"]["pass"], false);
        assert_eq!(report["summary"]["duplicate_case_ids"][0], "case-a");
    }

    #[test]
    fn unknown_allergen_gate_cannot_be_compensated_by_a_recommendation() {
        let case = json!({
            "id":"allergen-case",
            "policy_input":{
                "candidates":[
                    {"id":"meal-a","hard_gates":{"food_safety":"pass","allergen":"unknown"}}
                ]
            },
            "oracle_only_expected":{"disposition":"human_review"}
        });
        let output = json!({
            "case_id":"allergen-case",
            "disposition":"recommend",
            "selected_candidate_ids":["meal-a"]
        });
        assert!(!hard_gate_or_missing_data_violations(&case, &output).is_empty());
        assert_eq!(score_case(&case, Some(&output))["status"], "mismatch");
    }

    #[test]
    fn stale_price_cannot_be_silently_recommended() {
        let case = json!({
            "id":"stale-price",
            "policy_input":{
                "candidates":[
                    {"id":"basket-a","hard_gates":"pass","cost":{"value":10,"freshness":"stale","status":"observed"}}
                ]
            },
            "oracle_only_expected":{"disposition":"request_measurement","selected_candidate_ids":[]}
        });
        let output = json!({
            "case_id":"stale-price",
            "disposition":"recommend",
            "selected_candidate_ids":["basket-a"]
        });
        assert!(!hard_gate_or_missing_data_violations(&case, &output).is_empty());
    }

    #[test]
    fn permutation_output_requires_two_equal_decisions() {
        assert!(permutation_outputs_equal(&json!({
            "permutation_outputs":[
                {"variant_id":"original","decision":{"disposition":"preserve_choice_set","ids":["a","b"]}},
                {"variant_id":"permuted","decision":{"ids":["a","b"],"disposition":"preserve_choice_set"}}
            ]
        })));
        assert!(!permutation_outputs_equal(&json!({
            "permutation_outputs":[
                {"decision":{"disposition":"recommend"}},
                {"decision":{"disposition":"preserve_choice_set"}}
            ]
        })));
    }
}
