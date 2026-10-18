//! Semantic contract tests for CulinaryDecisionBundleV1 fixtures.
//!
//! This is a narrow consumer-side semantic checker, not a general JSON Schema
//! implementation. JSON shape is separately described by the versioned schema.
//! These tests must pass on an exact head before the cross-system contract is
//! qualified; source existence alone is not validation evidence.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA: &str =
    include_str!("../schemas/CulinaryDecisionBundleV1.schema.json");
const VALID: &str =
    include_str!("../fixtures/culinary-decision-bundle-v1/rank-reversal.json");
const PRIVATE_COORDINATES: &str =
    include_str!("../fixtures/culinary-decision-bundle-v1/schema-rejects-private-coordinates.json");
const BAD_SCENARIO_DIGEST: &str =
    include_str!("../fixtures/culinary-decision-bundle-v1/semantic-rejects-scenario-digest-mismatch.json");
const SIMULATED_AS_OBSERVED: &str =
    include_str!("../fixtures/culinary-decision-bundle-v1/semantic-rejects-simulation-relabelled-observation.json");
const UNKNOWN_GATE_ON_FRONTIER: &str =
    include_str!("../fixtures/culinary-decision-bundle-v1/semantic-rejects-unknown-gate-on-frontier.json");

fn parse_fixture(raw: &str) -> Value {
    serde_json::from_str(raw).expect("fixture is valid JSON")
}

/// Return semantic violations that an independent bundle consumer must reject.
/// This deliberately covers only the v1 cross-system safety invariants.
fn semantic_violations(bundle: &Value) -> Vec<String> {
    let mut errors = Vec::new();

    let Some(scenarios) = bundle.get("scenarios").and_then(Value::as_array) else {
        return vec!["scenarios must be an array".to_owned()];
    };
    let mut scenario_digest_by_id = BTreeMap::<String, String>::new();
    let mut baseline_count = 0usize;

    for scenario in scenarios {
        let id = scenario.get("id").and_then(Value::as_str).unwrap_or("");
        let digest = scenario.get("digest").and_then(Value::as_str).unwrap_or("");
        if id.trim().is_empty() || digest.trim().is_empty() {
            errors.push("scenario ID and digest must be non-empty".to_owned());
            continue;
        }
        if scenario_digest_by_id.insert(id.to_owned(), digest.to_owned()).is_some() {
            errors.push(format!("duplicate scenario ID: {id}"));
        }
        if scenario.get("is_baseline").and_then(Value::as_bool) == Some(true) {
            baseline_count += 1;
        }
    }
    if baseline_count != 1 {
        errors.push(format!("exactly one baseline is required; got {baseline_count}"));
    }

    let geography = bundle
        .get("scope")
        .and_then(|scope| scope.get("geographic_scope"));
    if let Some(geography) = geography {
        let allowed = ["label", "granularity", "source_refs"]
            .into_iter()
            .collect::<BTreeSet<_>>();
        if let Some(object) = geography.as_object() {
            for key in object.keys() {
                if !allowed.contains(key.as_str()) {
                    errors.push(format!("geographic field is not permitted: {key}"));
                }
            }
        }
    }

    let Some(candidates) = bundle.get("candidate_results").and_then(Value::as_array) else {
        return errors
            .into_iter()
            .chain(std::iter::once("candidate_results must be an array".to_owned()))
            .collect();
    };
    let mut candidate_ids = BTreeSet::new();
    let mut robust_frontier_calculated = BTreeSet::new();
    let objective_definitions = bundle
        .get("objective_definitions")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut objective_keys = BTreeSet::new();
    for definition in &objective_definitions {
        let key = definition.get("key").and_then(Value::as_str).unwrap_or("");
        if key.trim().is_empty() || !objective_keys.insert(key.to_owned()) {
            errors.push(format!("objective definition key is empty or duplicated: {key}"));
        }
    }
    let declared_objectives: BTreeMap<String, (String, String)> = bundle
        .get("objective_definitions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            Some((
                item.get("key")?.as_str()?.to_owned(),
                (
                    item.get("direction")?.as_str()?.to_owned(),
                    item.get("unit")?.as_str()?.to_owned(),
                ),
            ))
        })
        .collect();

    for candidate in candidates {
        let id = candidate
            .get("candidate_id")
            .and_then(Value::as_str)
            .unwrap_or("");
        if id.trim().is_empty() || !candidate_ids.insert(id.to_owned()) {
            errors.push(format!("candidate ID is empty or duplicated: {id}"));
        }

        let gate_disposition = candidate
            .get("gate_disposition")
            .and_then(Value::as_str)
            .unwrap_or("");
        let ranking_disposition = candidate
            .get("ranking_disposition")
            .and_then(Value::as_str)
            .unwrap_or("");
        let mut has_failed_gate = false;
        let mut has_unknown_gate = false;
        if let Some(gates) = candidate.get("gate_evidence").and_then(Value::as_array) {
            if gates.is_empty() {
                errors.push(format!("{id}: at least one hard-gate evidence row is required"));
            }
            for gate in gates {
                let key = gate.get("key").and_then(Value::as_str).unwrap_or("");
                let status = gate.get("status").and_then(Value::as_str).unwrap_or("");
                let refs = gate.get("evidence_refs").and_then(Value::as_array);
                match status {
                    "fail" => {
                        has_failed_gate = true;
                        if refs.is_none_or(Vec::is_empty) {
                            errors.push(format!("{id}: failed gate {key} lacks evidence reference"));
                        }
                    }
                    "unknown" => has_unknown_gate = true,
                    "pass" => {
                        if refs.is_none_or(Vec::is_empty) {
                            errors.push(format!("{id}: passed gate {key} lacks evidence reference"));
                        }
                    }
                    _ => errors.push(format!("{id}: gate {key} has unknown status {status}")),
                }
            }
        }
        let expected_gate_disposition = if has_failed_gate {
            "blocked"
        } else if has_unknown_gate {
            "indeterminate"
        } else {
            "feasible"
        };
        if gate_disposition != expected_gate_disposition {
            errors.push(format!("{id}: gate disposition conflicts with gate evidence"));
        }
        if gate_disposition != "feasible" && ranking_disposition != "not_ranked" {
            errors.push(format!("{id}: non-feasible candidate must be not_ranked"));
        }

        let Some(results) = candidate.get("scenario_results").and_then(Value::as_array) else {
            errors.push(format!("{id}: scenario_results must be an array"));
            continue;
        };
        let result_scenario_ids: BTreeSet<String> = results
            .iter()
            .filter_map(|result| result.get("scenario_id")?.as_str().map(str::to_owned))
            .collect();
        let expected_scenario_ids: BTreeSet<String> =
            scenario_digest_by_id.keys().cloned().collect();
        if result_scenario_ids != expected_scenario_ids || result_scenario_ids.len() != results.len() {
            errors.push(format!("{id}: scenario result coverage is missing, extra, or duplicated"));
        }

        let mut all_scenarios_rankable = gate_disposition == "feasible";
        let mut all_metrics_complete = true;
        for result in results {
            let scenario_id = result.get("scenario_id").and_then(Value::as_str).unwrap_or("");
            let recorded_digest = result
                .get("scenario_digest")
                .and_then(Value::as_str)
                .unwrap_or("");
            if scenario_digest_by_id.get(scenario_id).map(String::as_str) != Some(recorded_digest) {
                errors.push(format!("{id}: scenario {scenario_id} digest mismatch"));
            }

            let is_frontier = result.get("frontier").and_then(Value::as_bool) == Some(true);
            let mut scenario_metrics_complete = true;
            let mut seen_keys = BTreeSet::new();

            if let Some(objectives) = result.get("objectives").and_then(Value::as_array) {
                if objectives.is_empty() {
                    scenario_metrics_complete = false;
                    errors.push(format!("{id}: scenario {scenario_id} has no objective values"));
                }
                for objective in objectives {
                    let key = objective.get("key").and_then(Value::as_str).unwrap_or("");
                    let direction = objective.get("direction").and_then(Value::as_str).unwrap_or("");
                    let unit = objective.get("unit").and_then(Value::as_str).unwrap_or("");
                    let status = objective.get("status").and_then(Value::as_str).unwrap_or("");
                    let value = objective.get("value");
                    let evidence_refs = objective.get("evidence_refs").and_then(Value::as_array);
                    if !seen_keys.insert(key.to_owned()) {
                        errors.push(format!("{id}: duplicate objective key {key} in {scenario_id}"));
                    }
                    if declared_objectives.get(key) != Some(&(direction.to_owned(), unit.to_owned())) {
                        errors.push(format!("{id}: objective {key} definition/unit mismatch"));
                    }
                    let value_is_number = value.is_some_and(Value::is_number);
                    let value_is_null = value.is_some_and(Value::is_null);
                    let non_rankable_status = matches!(
                        status,
                        "missing" | "stale" | "conflicted" | "disputed" | "suppressed"
                    );
                    if non_rankable_status {
                        scenario_metrics_complete = false;
                        if !value_is_null {
                            errors.push(format!("{id}: {status} objective {key} must not expose a numeric value"));
                        }
                    } else if !value_is_number {
                        scenario_metrics_complete = false;
                        errors.push(format!("{id}: objective {key} status {status} needs a numeric value"));
                    } else if evidence_refs.is_none_or(Vec::is_empty) {
                        errors.push(format!("{id}: numeric objective {key} lacks evidence reference"));
                    }
                    if status == "observed"
                        && objective
                            .get("quality_flags")
                            .and_then(Value::as_array)
                            .is_some_and(|flags| {
                                flags.iter().any(|flag| {
                                    flag.as_str().is_some_and(|flag| {
                                        flag.contains("synthetic") || flag.contains("simulated")
                                    })
                                })
                            })
                    {
                        errors.push(format!("{id}: objective {key} relabels synthetic/simulated evidence as observed"));
                    }
                }
                let expected_keys: BTreeSet<String> =
                    declared_objectives.keys().cloned().collect();
                if seen_keys != expected_keys {
                    scenario_metrics_complete = false;
                    errors.push(format!(
                        "{id}: objective keys do not match the declared objective schema in {scenario_id}"
                    ));
                }
            } else {
                scenario_metrics_complete = false;
                errors.push(format!("{id}: objectives must be an array"));
            }

            all_metrics_complete &= scenario_metrics_complete;
            if is_frontier
                && (gate_disposition != "feasible"
                    || ranking_disposition != "ranked"
                    || !scenario_metrics_complete)
            {
                errors.push(format!("{id}: blocked, indeterminate, or incomplete candidate appears on a frontier"));
            }
            all_scenarios_rankable &= scenario_metrics_complete;
        }

        let expected_ranking_disposition = if gate_disposition != "feasible" {
            "not_ranked"
        } else if all_metrics_complete {
            "ranked"
        } else {
            "incomplete_metrics"
        };
        if ranking_disposition != expected_ranking_disposition {
            errors.push(format!("{id}: ranking disposition conflicts with gate/metric completeness"));
        }
        let robustly_rankable = candidate.get("robustly_rankable").and_then(Value::as_bool) == Some(true);
        if robustly_rankable != (all_scenarios_rankable && all_metrics_complete) {
            errors.push(format!("{id}: robustly_rankable conflicts with gate/scenario completeness"));
        }
        if robustly_rankable
            && candidate
                .get("robust_dominated_by")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
        {
            robust_frontier_calculated.insert(id.to_owned());
        }
    }

    let declared_robust_frontier: BTreeSet<String> = bundle
        .get("robust_frontier_ids")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    if declared_robust_frontier != robust_frontier_calculated {
        errors.push("robust_frontier_ids do not match complete, non-robustly-dominated candidates".to_owned());
    }

    errors
}

#[test]
fn schema_document_is_parseable_and_declares_v1() {
    let schema = parse_fixture(SCHEMA);
    assert_eq!(schema["title"], "CulinaryDecisionBundleV1");
    assert_eq!(schema["properties"]["schema_version"]["const"], "1.0.0");
    assert_eq!(
        schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
}

#[test]
fn positive_synthetic_rank_reversal_fixture_is_semantically_valid() {
    let bundle = parse_fixture(VALID);
    assert!(
        semantic_violations(&bundle).is_empty(),
        "unexpected semantic violations: {:?}",
        semantic_violations(&bundle)
    );
    let candidates = bundle["candidate_results"].as_array().unwrap();
    assert!(candidates[0]["scenario_results"][0]["frontier"].as_bool().unwrap());
    assert!(!candidates[0]["scenario_results"][1]["frontier"].as_bool().unwrap());
    assert!(!candidates[1]["scenario_results"][0]["frontier"].as_bool().unwrap());
    assert!(candidates[1]["scenario_results"][1]["frontier"].as_bool().unwrap());
    assert_eq!(bundle["robust_frontier_ids"], serde_json::json!(["alpha", "beta"]));
}

#[test]
fn exact_private_coordinates_are_rejected() {
    let bundle = parse_fixture(PRIVATE_COORDINATES);
    let violations = semantic_violations(&bundle);
    assert!(violations.iter().any(|error| error.contains("exact_coordinates")));
}

#[test]
fn scenario_digest_mismatch_is_rejected() {
    let bundle = parse_fixture(BAD_SCENARIO_DIGEST);
    let violations = semantic_violations(&bundle);
    assert!(violations.iter().any(|error| error.contains("digest mismatch")));
}

#[test]
fn simulated_evidence_cannot_be_relabelled_as_observed() {
    let bundle = parse_fixture(SIMULATED_AS_OBSERVED);
    let violations = semantic_violations(&bundle);
    assert!(violations.iter().any(|error| error.contains("relabels synthetic/simulated")));
}

#[test]
fn unknown_hard_gate_cannot_enter_a_frontier() {
    let bundle = parse_fixture(UNKNOWN_GATE_ON_FRONTIER);
    let violations = semantic_violations(&bundle);
    assert!(violations.iter().any(|error| error.contains("gate disposition conflicts")));
    assert!(violations.iter().any(|error| error.contains("non-feasible candidate must be not_ranked")));
    assert!(violations.iter().any(|error| error.contains("appears on a frontier")));
}
