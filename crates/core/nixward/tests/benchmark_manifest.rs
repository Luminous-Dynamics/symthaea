//! Structural conformance checks for the NixWard benchmark seed manifest.
//!
//! These tests validate task-manifest integrity and coverage targets. They do
//! not execute Nix tasks, grade generated configurations or establish SOTA.

use std::collections::HashSet;

use serde_json::Value;

const SCHEMA_JSON: &str = include_str!(
    "../../../../research/nixward-benchmark-v0.1/schemas/benchmark-manifest-v0.1.schema.json"
);
const CORPUS_JSON: &str = include_str!(
    "../../../../research/nixward-benchmark-v0.1/scenarios/benchmark-seed-v0.1.json"
);

#[test]
fn benchmark_seed_is_versioned_synthetic_and_large_enough() {
    let schema: Value = serde_json::from_str(SCHEMA_JSON).expect("benchmark schema JSON");
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("benchmark corpus JSON");

    assert_eq!(
        schema["$schema"].as_str(),
        Some("https://json-schema.org/draft/2020-12/schema")
    );
    assert_eq!(corpus["schema_version"].as_str(), Some("0.1"));
    assert_eq!(
        corpus["corpus_id"].as_str(),
        Some("NIXWARD-BENCHMARK-SEED-V0.1")
    );
    assert_eq!(corpus["synthetic_only"].as_bool(), Some(true));

    let cases = corpus["cases"].as_array().expect("cases array");
    assert!(
        cases.len() >= 40,
        "benchmark seed should contain at least 40 independent task clusters"
    );
}

#[test]
fn task_and_cluster_identities_are_unique() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("benchmark corpus JSON");
    let cases = corpus["cases"].as_array().expect("cases array");
    let mut task_ids = HashSet::new();
    let mut cluster_ids = HashSet::new();

    for case in cases {
        let id = case["case_id"].as_str().expect("case_id");
        let cluster = case["cluster_id"].as_str().expect("cluster_id");
        assert!(task_ids.insert(id), "duplicate task identity: {id}");
        assert!(
            cluster_ids.insert(cluster),
            "duplicate independent cluster identity: {cluster}"
        );
    }

    assert_eq!(task_ids.len(), cases.len());
    assert_eq!(cluster_ids.len(), cases.len());
}

#[test]
fn vm_tier_and_benign_positive_control_coverage_meet_minima() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("benchmark corpus JSON");
    let cases = corpus["cases"].as_array().expect("cases array");

    let vm_cases = cases
        .iter()
        .filter(|case| case["runtime_tier"].as_str() == Some("vm"))
        .count();
    assert!(
        vm_cases >= 20,
        "benchmark seed must retain at least 20 VM-graded cases; found {vm_cases}"
    );

    let positive_controls = cases
        .iter()
        .filter(|case| case["benign_positive_control"].as_bool() == Some(true))
        .count();
    assert!(
        positive_controls * 4 >= cases.len(),
        "at least 25% of cases must be benign positive controls"
    );
}

#[test]
fn each_task_has_independent_oracle_gates_and_target_identity() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("benchmark corpus JSON");
    for case in corpus["cases"].as_array().expect("cases array") {
        let id = case["case_id"].as_str().expect("case_id");
        assert!(
            case["evaluator"]["independent_oracle"].as_bool() == Some(true),
            "{id}: an independent oracle is required"
        );
        assert!(
            case["hard_gates"]
                .as_array()
                .is_some_and(|items| !items.is_empty()),
            "{id}: at least one hard gate is required"
        );
        for field in [
            "fixture_id",
            "nixos_release",
            "system",
            "flake_identity",
            "hardware_profile",
        ] {
            assert!(
                case["target"][field]
                    .as_str()
                    .is_some_and(|value| !value.is_empty()),
                "{id}: target.{field} is required"
            );
        }
        for field in [
            "forbidden_effects",
            "allowed_tools",
            "required_observations",
            "plan_requirements",
            "postconditions",
        ] {
            assert!(
                case[field]
                    .as_array()
                    .is_some_and(|items| !items.is_empty()),
                "{id}: {field} must not be empty"
            );
        }

        if case["runtime_tier"].as_str() == Some("vm") {
            assert_eq!(
                case["evaluator"]["kind"].as_str(),
                Some("nixos_vm"),
                "{id}: VM tier requires a NixOS VM evaluator"
            );
        }
        if case["benign_positive_control"].as_bool() == Some(true) {
            assert_ne!(
                case["task_kind"].as_str(),
                Some("abstain"),
                "{id}: an abstention task cannot be labelled as a benign completion control"
            );
        }
    }
}

#[test]
fn destructive_disk_control_requires_zero_mutation() {
    let corpus: Value = serde_json::from_str(CORPUS_JSON).expect("benchmark corpus JSON");
    let case = corpus["cases"]
        .as_array()
        .expect("cases array")
        .iter()
        .find(|case| case["case_id"].as_str() == Some("NWB-DISK-001"))
        .expect("destructive disk ambiguity task");

    assert_eq!(case["task_kind"].as_str(), Some("abstain"));
    assert!(
        case["forbidden_effects"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|item| item.as_str().is_some_and(|value| value.contains("format")))
    );
    assert!(
        case["hard_gates"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|item| item.as_str().is_some_and(|value| value.contains("Zero disk mutations")))
    );
}
