// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Contract tests for the frozen synthetic chemical-process qualification corpus.
// These tests validate fixture shape and benchmark-manifest integrity only;
// they do not execute a process simulator or establish chemical-engineering validity.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const CORPUS_JSON: &str = include_str!(
    "../../../../docs/engineering/fixtures/CHEMICAL_PROCESS_QUALIFICATION_V1.json"
);

const ALLOWED_DISPOSITIONS: &[&str] = &[
    "ACCEPTED_WITHIN_SCOPE",
    "BLOCKED",
    "UNRESOLVED",
    "OUT_OF_DOMAIN",
];

const ALLOWED_CLASSES: &[&str] = &[
    "positive-control",
    "negative-control",
    "metamorphic",
    "analytic-reference",
    "uncertainty-control",
    "tamper-control",
];

fn corpus() -> Value {
    serde_json::from_str(CORPUS_JSON)
        .expect("frozen chemical-process qualification corpus must be valid JSON")
}

fn string_field<'a>(value: &'a Value, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("required non-empty string field {key:?} is missing"))
}

fn string_array<'a>(value: &'a Value, key: &str) -> Vec<&'a str> {
    value
        .get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("required array field {key:?} is missing"))
        .iter()
        .map(|item| {
            item.as_str()
                .filter(|text| !text.trim().is_empty())
                .unwrap_or_else(|| panic!("field {key:?} must contain non-empty strings"))
        })
        .collect()
}

#[test]
fn frozen_v1_1_corpus_is_well_formed_and_versioned() {
    let root = corpus();
    let required_root_fields = [
        "schema_version",
        "benchmark_id",
        "corpus_version",
        "created_utc",
        "baseline_commit",
        "source_specification",
        "purpose",
        "scope_boundary",
        "allowed_dispositions",
        "evaluation_rules",
        "residual_profile",
        "cases",
        "summary",
    ];

    for field in required_root_fields {
        assert!(
            root.get(field).is_some(),
            "required top-level field {field:?} is missing"
        );
    }

    assert_eq!(string_field(&root, "schema_version"), "1.0.0");
    assert_eq!(string_field(&root, "benchmark_id"), "LUMINOUS-CHEM-PROC-QUAL");
    assert_eq!(string_field(&root, "corpus_version"), "1.1.0");

    let allowed = string_array(&root, "allowed_dispositions")
        .into_iter()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        allowed,
        ALLOWED_DISPOSITIONS.iter().copied().collect(),
        "corpus v1 disposition vocabulary must be frozen"
    );

    assert!(
        !string_array(&root, "evaluation_rules").is_empty(),
        "corpus must state evaluation rules"
    );

    let scope = root
        .get("scope_boundary")
        .and_then(Value::as_object)
        .expect("scope_boundary must be an object");
    assert_eq!(
        scope.get("data_class").and_then(Value::as_str),
        Some("synthetic_non_operational")
    );
    for field in [
        "contains_hazardous_operating_recipes",
        "establishes_chemical_route_feasibility",
        "establishes_physical_model_accuracy",
        "authorizes_process_control_or_plant_operation",
    ] {
        assert_eq!(
            scope.get(field).and_then(Value::as_bool),
            Some(false),
            "scope boundary {field:?} must remain explicitly false"
        );
    }
}

#[test]
fn case_ids_are_unique_and_each_case_has_a_frozen_disposition() {
    let root = corpus();
    let cases = root
        .get("cases")
        .and_then(Value::as_array)
        .expect("cases must be an array");
    assert_eq!(cases.len(), 26, "corpus count changes require deliberate versioning and a version bump");

    let allowed_dispositions = ALLOWED_DISPOSITIONS.iter().copied().collect::<BTreeSet<_>>();
    let allowed_classes = ALLOWED_CLASSES.iter().copied().collect::<BTreeSet<_>>();
    let required_case_fields = [
        "id",
        "class",
        "scenario",
        "claim_under_test",
        "fixture",
        "expected_disposition",
        "required_findings",
        "prohibited_promotions",
    ];

    let mut ids = BTreeSet::new();
    let mut class_counts = BTreeMap::<String, usize>::new();
    let mut observed_dispositions = BTreeSet::new();

    for case in cases {
        for field in required_case_fields {
            assert!(
                case.get(field).is_some(),
                "case is missing required field {field:?}: {case}"
            );
        }

        let id = string_field(case, "id");
        assert!(ids.insert(id.to_owned()), "duplicate case ID {id:?}");
        assert!(!string_field(case, "scenario").trim().is_empty());
        assert!(!string_field(case, "claim_under_test").trim().is_empty());

        let class = string_field(case, "class");
        assert!(
            allowed_classes.contains(class),
            "unrecognized class {class:?} in case {id}"
        );
        *class_counts.entry(class.to_owned()).or_default() += 1;

        let disposition = string_field(case, "expected_disposition");
        assert!(
            allowed_dispositions.contains(disposition),
            "unrecognized disposition {disposition:?} in case {id}"
        );
        observed_dispositions.insert(disposition);

        assert!(
            case.get("fixture").and_then(Value::as_object).is_some(),
            "fixture must be a JSON object in case {id}"
        );
        assert!(
            !string_array(case, "required_findings").is_empty(),
            "case {id} must name required findings"
        );
        assert!(
            !string_array(case, "prohibited_promotions").is_empty(),
            "case {id} must name prohibited promotions"
        );
    }

    assert_eq!(observed_dispositions, allowed_dispositions);
    let summary = root
        .get("summary")
        .and_then(Value::as_object)
        .expect("summary must be an object");

    assert_eq!(
        summary.get("case_count").and_then(Value::as_u64),
        Some(cases.len() as u64)
    );

    for class in ALLOWED_CLASSES {
        let summary_key = match *class {
            "positive-control" => "positive_controls",
            "negative-control" => "negative_controls",
            "metamorphic" => "metamorphic_cases",
            "analytic-reference" => "analytic_reference_cases",
            "uncertainty-control" => "uncertainty_controls",
            "tamper-control" => "tamper_controls",
            _ => unreachable!("ALLOWED_CLASSES is exhaustive"),
        };
        assert_eq!(
            summary.get(summary_key).and_then(Value::as_u64),
            Some(*class_counts.get(*class).unwrap_or(&0) as u64),
            "summary count for class {class:?} must match cases"
        );
    }
}

#[test]
fn analytic_residual_target_is_scoped_and_positive() {
    let root = corpus();
    let profile = root
        .get("residual_profile")
        .and_then(Value::as_object)
        .expect("residual_profile must be an object");

    assert_eq!(
        profile.get("name").and_then(Value::as_str),
        Some("synthetic-analytic-v1")
    );
    let target = profile
        .get("target_max")
        .and_then(Value::as_f64)
        .expect("target_max must be numeric");
    assert!(target > 0.0 && target <= 1.0e-8);
    let scope = profile
        .get("scope")
        .and_then(Value::as_str)
        .expect("residual scope must be documented");
    assert!(
        scope.contains("analytically solvable"),
        "tight residual target must remain scoped to simple analytical fixtures"
    );
    let caveat = profile
        .get("caveat")
        .and_then(Value::as_str)
        .expect("residual caveat must be documented");
    assert!(
        caveat.contains("threshold proves"),
        "residual caveat must deny automatic promotion to physical truth"
    );
}

#[test]
fn negative_controls_cover_critical_no_silent_promotion_rules() {
    let root = corpus();
    let cases = root["cases"].as_array().expect("cases must be an array");
    let by_id = cases
        .iter()
        .map(|case| (string_field(case, "id"), case))
        .collect::<BTreeMap<_, _>>();

    let required_blocked_cases = [
        ("CHEM-PROC-002", "BLOCKED"),
        ("CHEM-PROC-003", "BLOCKED"),
        ("CHEM-PROC-008", "BLOCKED"),
        ("CHEM-PROC-011", "BLOCKED"),
        ("CHEM-PROC-012", "BLOCKED"),
        ("CHEM-PROC-014", "BLOCKED"),
        ("CHEM-PROC-015", "BLOCKED"),
        ("CHEM-PROC-019", "BLOCKED"),
        ("CHEM-PROC-022", "BLOCKED"),
        ("CHEM-PROC-023", "BLOCKED"),
    ];

    for (id, expected) in required_blocked_cases {
        let case = by_id
            .get(id)
            .unwrap_or_else(|| panic!("required negative-control case {id} is missing"));
        assert_eq!(
            string_field(case, "expected_disposition"),
            expected,
            "case {id} must remain blocking"
        );
    }

    assert_eq!(
        string_field(by_id["CHEM-PROC-005"], "expected_disposition"),
        "UNRESOLVED",
        "unknown output material must not silently close a balance"
    );
    assert_eq!(
        string_field(by_id["CHEM-PROC-025"], "expected_disposition"),
        "UNRESOLVED",
        "shared evidence dependencies must not count as independent corroboration"
    );
    assert_eq!(
        string_field(by_id["CHEM-PROC-026"], "expected_disposition"),
        "ACCEPTED_WITHIN_SCOPE",
        "disjoint, complete synthetic dependencies should pass only the independence-contract check"
    );
    assert_eq!(
        string_field(by_id["CHEM-PROC-013"], "expected_disposition"),
        "OUT_OF_DOMAIN",
        "out-of-domain model use must remain distinct from generic block"
    );

    for case in cases {
        let prohibited = string_array(case, "prohibited_promotions");
        assert!(
            !prohibited.is_empty(),
            "{} must name at least one forbidden evidence promotion",
            string_field(case, "id")
        );
    }
}

#[test]
fn frozen_corpus_parses_deterministically() {
    let first: Value = serde_json::from_str(CORPUS_JSON).expect("first parse must succeed");
    let second: Value = serde_json::from_str(CORPUS_JSON).expect("second parse must succeed");
    assert_eq!(first, second, "same frozen bytes must produce the same parsed corpus");
}
