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
const FALSE_SCENARIO_FRONTIER: &str =
    include_str!("../fixtures/culinary-decision-bundle-v1/semantic-rejects-false-scenario-frontier.json");
const FALSE_ROBUST_DOMINANCE: &str =
    include_str!("../fixtures/culinary-decision-bundle-v1/semantic-rejects-false-robust-dominance.json");
const PARTIAL_SCENARIO_RANKABILITY: &str =
    include_str!("../fixtures/culinary-decision-bundle-v1/valid-scenario-frontier-with-other-scenario-missing.json");

fn parse_fixture(raw: &str) -> Value {
    serde_json::from_str(raw).expect("fixture is valid JSON")
}

type ObjectiveValues = BTreeMap<String, (String, String, f64)>;

fn complete_objectives(result: &Value) -> Option<ObjectiveValues> {
    let objectives = result.get("objectives")?.as_array()?;
    if objectives.is_empty() {
        return None;
    }

    let mut values = ObjectiveValues::new();
    for objective in objectives {
        let key = objective.get("key")?.as_str()?;
        let direction = objective.get("direction")?.as_str()?;
        let unit = objective.get("unit")?.as_str()?;
        let status = objective.get("status")?.as_str()?;
        if !matches!(status, "observed" | "derived" | "estimated" | "simulated" | "synthetic") {
            return None;
        }
        if !matches!(direction, "minimize" | "maximize") {
            return None;
        }
        let value = objective.get("value")?.as_f64()?;
        if !value.is_finite() {
            return None;
        }
        if objective.get("evidence_refs")?.as_array()?.is_empty() {
            return None;
        }
        if values
            .insert(key.to_owned(), (direction.to_owned(), unit.to_owned(), value))
            .is_some()
        {
            return None;
        }
    }
    Some(values)
}

fn dominates(left: &ObjectiveValues, right: &ObjectiveValues) -> bool {
    if left.len() != right.len() || left.keys().ne(right.keys()) {
        return false;
    }
    let mut strictly_better = false;
    for (key, (direction, unit, left_value)) in left {
        let Some((right_direction, right_unit, right_value)) = right.get(key) else {
            return false;
        };
        if direction != right_direction || unit != right_unit {
            return false;
        }
        let (no_worse, better) = match direction.as_str() {
            "minimize" => (left_value <= right_value, left_value < right_value),
            "maximize" => (left_value >= right_value, left_value > right_value),
            _ => return false,
        };
        if !no_worse {
            return false;
        }
        strictly_better |= better;
    }
    strictly_better
}

fn robustly_dominates(
    left_by_scenario: &BTreeMap<String, ObjectiveValues>,
    right_by_scenario: &BTreeMap<String, ObjectiveValues>,
) -> bool {
    if left_by_scenario.len() != right_by_scenario.len()
        || left_by_scenario.keys().ne(right_by_scenario.keys())
    {
        return false;
    }

    let mut strictly_better_somewhere = false;
    for (scenario_id, left) in left_by_scenario {
        let Some(right) = right_by_scenario.get(scenario_id) else {
            return false;
        };
        if left.len() != right.len() || left.keys().ne(right.keys()) {
            return false;
        }
        for (key, (direction, unit, left_value)) in left {
            let Some((right_direction, right_unit, right_value)) = right.get(key) else {
                return false;
            };
            if direction != right_direction || unit != right_unit {
                return false;
            }
            let (no_worse, better) = match direction.as_str() {
                "minimize" => (left_value <= right_value, left_value < right_value),
                "maximize" => (left_value >= right_value, left_value > right_value),
                _ => return false,
            };
            if !no_worse {
                return false;
            }
            strictly_better_somewhere |= better;
        }
    }
    strictly_better_somewhere
}

fn scenario_result<'a>(candidate: &'a Value, scenario_id: &str) -> Option<&'a Value> {
    candidate
        .get("scenario_results")?
        .as_array()?
        .iter()
        .find(|result| result.get("scenario_id").and_then(Value::as_str) == Some(scenario_id))
}

fn candidate_scenario_values(
    candidate: &Value,
    scenario_ids: &BTreeSet<String>,
) -> Option<BTreeMap<String, ObjectiveValues>> {
    if candidate.get("gate_disposition")?.as_str()? != "feasible" {
        return None;
    }
    let mut values_by_scenario = BTreeMap::new();
    for scenario_id in scenario_ids {
        let result = scenario_result(candidate, scenario_id)?;
        values_by_scenario.insert(scenario_id.clone(), complete_objectives(result)?);
    }
    Some(values_by_scenario)
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
        let mut gate_keys = BTreeSet::new();
        if let Some(gates) = candidate.get("gate_evidence").and_then(Value::as_array) {
            if gates.is_empty() {
                errors.push(format!("{id}: at least one hard-gate evidence row is required"));
            }
            for gate in gates {
                let key = gate.get("key").and_then(Value::as_str).unwrap_or("");
                let status = gate.get("status").and_then(Value::as_str).unwrap_or("");
                let refs = gate.get("evidence_refs").and_then(Value::as_array);
                if key.trim().is_empty() || !gate_keys.insert(key.to_owned()) {
                    errors.push(format!("{id}: gate key is empty or duplicated: {key}"));
                }
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
        } else {
            errors.push(format!("{id}: gate_evidence must be an array"));
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
        let scenario_coverage_complete =
            result_scenario_ids == expected_scenario_ids && result_scenario_ids.len() == results.len();
        if !scenario_coverage_complete {
            errors.push(format!("{id}: scenario result coverage is missing, extra, or duplicated"));
        }

        let mut all_scenarios_rankable =
            gate_disposition == "feasible" && scenario_coverage_complete;
        let mut all_metrics_complete = scenario_coverage_complete;
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
                    let known_status = matches!(
                        status,
                        "observed"
                            | "derived"
                            | "estimated"
                            | "simulated"
                            | "synthetic"
                            | "missing"
                            | "stale"
                            | "conflicted"
                            | "disputed"
                            | "suppressed"
                    );
                    if !known_status {
                        scenario_metrics_complete = false;
                        errors.push(format!("{id}: objective {key} has unknown evidence status {status}"));
                    }
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
                && (gate_disposition != "feasible" || !scenario_metrics_complete)
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
    }

    let candidate_by_id: BTreeMap<String, &Value> = candidates
        .iter()
        .filter_map(|candidate| {
            Some((
                candidate.get("candidate_id")?.as_str()?.to_owned(),
                candidate,
            ))
        })
        .collect();
    let expected_scenario_ids: BTreeSet<String> =
        scenario_digest_by_id.keys().cloned().collect();

    // Recompute every scenario frontier from the candidate objective vectors.
    // Do not trust the frontier booleans authored by the producer.
    for scenario_id in &expected_scenario_ids {
        let eligible: Vec<(&str, ObjectiveValues)> = candidates
            .iter()
            .filter_map(|candidate| {
                if candidate.get("gate_disposition").and_then(Value::as_str) != Some("feasible") {
                    return None;
                }
                let id = candidate.get("candidate_id")?.as_str()?;
                let result = scenario_result(candidate, scenario_id)?;
                Some((id, complete_objectives(result)?))
            })
            .collect();

        let mut expected_frontier = BTreeSet::new();
        for (candidate_id, candidate_values) in &eligible {
            let is_dominated = eligible.iter().any(|(other_id, other_values)| {
                other_id != candidate_id && dominates(other_values, candidate_values)
            });
            if !is_dominated {
                expected_frontier.insert((*candidate_id).to_owned());
            }
        }

        for candidate in candidates {
            let id = candidate.get("candidate_id").and_then(Value::as_str).unwrap_or("");
            let result = scenario_result(candidate, scenario_id);
            let declared_frontier = result
                .and_then(|value| value.get("frontier"))
                .and_then(Value::as_bool);
            let expected = expected_frontier.contains(id);
            if declared_frontier != Some(expected) {
                errors.push(format!(
                    "{id}: frontier claim in scenario {scenario_id} differs from independently recomputed frontier"
                ));
            }
        }
    }

    // Independently recompute robust dominance over all scenarios; mere reciprocal
    // claims are insufficient if both candidates can reverse order under a shock.
    let values_by_candidate: BTreeMap<String, Option<BTreeMap<String, ObjectiveValues>>> =
        candidates
            .iter()
            .filter_map(|candidate| {
                let id = candidate.get("candidate_id")?.as_str()?.to_owned();
                Some((
                    id,
                    candidate_scenario_values(candidate, &expected_scenario_ids),
                ))
            })
            .collect();
    let mut expected_dominates: BTreeMap<String, BTreeSet<String>> = candidate_ids
        .iter()
        .map(|id| (id.clone(), BTreeSet::new()))
        .collect();
    let mut expected_dominated_by: BTreeMap<String, BTreeSet<String>> = candidate_ids
        .iter()
        .map(|id| (id.clone(), BTreeSet::new()))
        .collect();

    let eligible_ids: Vec<String> = values_by_candidate
        .iter()
        .filter_map(|(id, values)| values.as_ref().map(|_| id.clone()))
        .collect();

    for left_index in 0..eligible_ids.len() {
        for right_index in (left_index + 1)..eligible_ids.len() {
            let left_id = &eligible_ids[left_index];
            let right_id = &eligible_ids[right_index];
            let left_values = values_by_candidate[left_id].as_ref().unwrap();
            let right_values = values_by_candidate[right_id].as_ref().unwrap();

            if robustly_dominates(left_values, right_values) {
                expected_dominates
                    .get_mut(left_id)
                    .unwrap()
                    .insert(right_id.clone());
                expected_dominated_by
                    .get_mut(right_id)
                    .unwrap()
                    .insert(left_id.clone());
            } else if robustly_dominates(right_values, left_values) {
                expected_dominates
                    .get_mut(right_id)
                    .unwrap()
                    .insert(left_id.clone());
                expected_dominated_by
                    .get_mut(left_id)
                    .unwrap()
                    .insert(right_id.clone());
            }
        }
    }

    for (candidate_id, candidate) in &candidate_by_id {
        for (field, expected) in [
            (
                "robust_dominates",
                expected_dominates.get(candidate_id).cloned().unwrap_or_default(),
            ),
            (
                "robust_dominated_by",
                expected_dominated_by.get(candidate_id).cloned().unwrap_or_default(),
            ),
        ] {
            let declared: BTreeSet<String> = candidate
                .get(field)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
            if declared.len()
                != candidate
                    .get(field)
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len)
            {
                errors.push(format!("{candidate_id}: {field} contains duplicate or invalid entries"));
            }
            if declared != expected {
                errors.push(format!(
                    "{candidate_id}: {field} differs from independently recomputed robust dominance"
                ));
            }
        }
    }

    let expected_robust_frontier: BTreeSet<String> = eligible_ids
        .iter()
        .filter(|id| {
            expected_dominated_by
                .get(*id)
                .is_some_and(BTreeSet::is_empty)
        })
        .cloned()
        .collect();
    let declared_robust_frontier: BTreeSet<String> = bundle
        .get("robust_frontier_ids")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    if declared_robust_frontier != expected_robust_frontier {
        errors.push(
            "robust_frontier_ids differ from independently recomputed robust frontier".to_owned(),
        );
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
fn falsely_declared_scenario_frontier_is_rejected() {
    let bundle = parse_fixture(FALSE_SCENARIO_FRONTIER);
    let violations = semantic_violations(&bundle);
    assert!(violations.iter().any(|error| {
        error.contains("frontier claim in scenario observed_basket")
    }));
}

#[test]
fn falsely_declared_robust_dominance_is_rejected_even_when_reciprocal() {
    let bundle = parse_fixture(FALSE_ROBUST_DOMINANCE);
    let violations = semantic_violations(&bundle);
    assert!(violations.iter().any(|error| {
        error.contains("differs from independently recomputed robust dominance")
    }));
    assert!(violations.iter().any(|error| {
        error.contains("robust_frontier_ids differ from independently recomputed robust frontier")
    }));
}

#[test]
fn scenario_frontier_can_be_known_when_another_scenario_is_incomplete() {
    let bundle = parse_fixture(PARTIAL_SCENARIO_RANKABILITY);
    let violations = semantic_violations(&bundle);
    assert!(
        violations.is_empty(),
        "unexpected semantic violations: {:?}",
        violations
    );
    let alpha = bundle["candidate_results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["candidate_id"] == "alpha")
        .unwrap();
    assert_eq!(alpha["ranking_disposition"], "incomplete_metrics");
    assert!(!alpha["robustly_rankable"].as_bool().unwrap());
    assert!(alpha["scenario_results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|scenario| scenario["scenario_id"] == "observed_basket")
        .unwrap()["frontier"]
        .as_bool()
        .unwrap());
    assert_eq!(bundle["robust_frontier_ids"], serde_json::json!(["beta"]));
}

#[test]
fn unknown_hard_gate_cannot_enter_a_frontier() {
    let bundle = parse_fixture(UNKNOWN_GATE_ON_FRONTIER);
    let violations = semantic_violations(&bundle);
    assert!(violations.iter().any(|error| error.contains("gate disposition conflicts")));
    assert!(violations.iter().any(|error| error.contains("non-feasible candidate must be not_ranked")));
    assert!(violations.iter().any(|error| error.contains("appears on a frontier")));
}
