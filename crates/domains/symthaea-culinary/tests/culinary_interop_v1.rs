//! Semantic contract tests for CulinaryDecisionBundleV1 fixtures.
//!
//! This is a narrow consumer-side semantic checker, not a general JSON Schema
//! implementation. JSON shape is separately described by the versioned schema.
//! These tests must pass on an exact head before the cross-system contract is
//! qualified; source existence alone is not validation evidence.

use serde_json::Value;
use symthaea_culinary::interop::validate_decision_bundle_semantics;

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
const FALSE_RANK_INSTABILITY: &str =
    include_str!("../fixtures/culinary-decision-bundle-v1/semantic-rejects-false-rank-instability-counts.json");
const PARTIAL_SCENARIO_RANKABILITY: &str =
    include_str!("../fixtures/culinary-decision-bundle-v1/valid-scenario-frontier-with-other-scenario-missing.json");

fn parse_fixture(raw: &str) -> Value {
    serde_json::from_str(raw).expect("fixture is valid JSON")
}

fn semantic_violations(bundle: &Value) -> Vec<String> {
    validate_decision_bundle_semantics(bundle).err().unwrap_or_default()
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
fn false_rank_instability_counts_are_rejected() {
    let bundle = parse_fixture(FALSE_RANK_INSTABILITY);
    let violations = semantic_violations(&bundle);
    assert!(violations.iter().any(|error| {
        error.contains("rank_instability.ranked_scenarios does not match independently recomputed count")
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
