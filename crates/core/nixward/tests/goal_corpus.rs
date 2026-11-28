//! Synthetic NixWard user-goal corpus conformance checks.
//!
//! These tests validate goal decomposition and clarification expectations.
//! They do not prove that a plan is generated correctly or is safe to execute.

use std::collections::HashSet;

use serde_json::Value;

const CORPUS_JSON: &str = include_str!(
    "../../../../research/nixward-capability-advisory-v0.1/scenarios/goal-corpus-v0.1.json"
);
const SCHEMA_JSON: &str = include_str!(
    "../../../../research/nixward-capability-advisory-v0.1/schemas/goal-corpus-v0.1.schema.json"
);

#[test]
fn goal_corpus_is_versioned_and_contains_independent_clusters() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("goal corpus must be valid JSON");
    let schema: Value = serde_json::from_str(SCHEMA_JSON).expect("goal schema must be valid JSON");

    assert_eq!(
        schema["$schema"].as_str(),
        Some("https://json-schema.org/draft/2020-12/schema")
    );
    assert_eq!(corpus["schema_version"].as_str(), Some("0.1"));
    assert_eq!(
        corpus["profile_id"].as_str(),
        Some("NIXWARD-GOAL-CORPUS-V0.1")
    );
    assert_eq!(corpus["synthetic_only"].as_bool(), Some(true));

    let cases = corpus["cases"].as_array().expect("cases array");
    assert!(
        cases.len() >= 12,
        "frozen goal corpus needs at least 12 cases"
    );

    let mut case_ids = HashSet::new();
    for case in cases {
        let id = case["case_id"].as_str().expect("case_id");
        assert!(case_ids.insert(id), "duplicate case identity: {id}");
        assert!(
            case["cluster_id"]
                .as_str()
                .is_some_and(|value| !value.is_empty()),
            "{id}: independent task cluster is required"
        );
    }
}

#[test]
fn goal_requirements_have_verification_obligations_and_unique_ids() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("valid goal corpus");
    for case in corpus["cases"].as_array().expect("cases") {
        let case_id = case["case_id"].as_str().expect("case_id");
        let requirements = case["requirements"].as_array().expect("requirements");
        assert!(
            !requirements.is_empty(),
            "{case_id}: at least one requirement is needed"
        );

        let mut requirement_ids = HashSet::new();
        for requirement in requirements {
            let id = requirement["requirement_id"]
                .as_str()
                .expect("requirement_id");
            assert!(
                requirement_ids.insert(id),
                "{case_id}: duplicate requirement id {id}"
            );
            assert!(
                requirement["verification_obligations"]
                    .as_array()
                    .is_some_and(|items| !items.is_empty()),
                "{case_id}/{id}: every goal needs an explicit verification obligation"
            );
        }
    }
}

#[test]
fn irreversible_or_high_risk_decisions_cannot_be_assumed_by_the_planner() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("valid goal corpus");
    for case in corpus["cases"].as_array().expect("cases") {
        let case_id = case["case_id"].as_str().expect("case_id");
        let decisions = case["unresolved_decisions"].as_array().expect("decisions");
        let blocking = case["expected"]["blocking_decisions"]
            .as_array()
            .expect("blocking decisions");
        let classification = case["expected"]["classification"]
            .as_str()
            .expect("classification");

        for decision in decisions {
            let id = decision["decision_id"].as_str().expect("decision_id");
            let risk = decision["risk_if_assumed"].as_str().expect("risk");
            if risk == "irreversible" || risk == "high" {
                assert_ne!(
                    classification, "plan_candidate",
                    "{case_id}/{id}: high-risk decisions cannot be silently assumed"
                );
                assert!(
                    blocking.iter().any(|item| item.as_str() == Some(id)),
                    "{case_id}/{id}: high-risk decision must block plan promotion"
                );
            }
        }
    }
}

#[test]
fn plan_candidates_are_bound_to_a_complete_target_identity() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("valid goal corpus");
    for case in corpus["cases"].as_array().expect("cases") {
        if case["expected"]["classification"].as_str() != Some("plan_candidate") {
            continue;
        }

        let id = case["case_id"].as_str().expect("case_id");
        let target = &case["target"];
        for field in [
            "host_id",
            "nixos_release",
            "system",
            "flake_ref",
            "nixpkgs_revision",
            "flake_lock_digest",
            "configuration_identity",
        ] {
            assert!(
                target[field]
                    .as_str()
                    .is_some_and(|value| !value.is_empty()),
                "{id}: plan candidate requires target.{field}"
            );
        }
    }
}

#[test]
fn destructive_disk_goal_is_not_promoted_to_a_plan_candidate() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("valid goal corpus");
    let destructive = corpus["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["case_id"].as_str() == Some("NWG-005"))
        .expect("NWG-005 destructive disk case");

    assert_eq!(
        destructive["expected"]["classification"].as_str(),
        Some("abstain")
    );
    assert_eq!(
        destructive["expected"]["blocking_decisions"]
            .as_array()
            .map(Vec::len),
        Some(3)
    );
    assert!(
        destructive["expected"]["must_not_assume"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|value| value.as_str() == Some("guess_disk_device"))
    );
}
