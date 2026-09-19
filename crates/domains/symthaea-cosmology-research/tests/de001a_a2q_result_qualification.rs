// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;

const SPEC: &str = include_str!("../references/de001a_a2q_result_qualification_v1.json");
const SOURCE: &str = include_str!("../src/a2q_qualify.rs");

#[test]
fn a2q_criteria_are_frozen_before_optimizer_output() {
    let value: Value = serde_json::from_str(SPEC).expect("valid A2Q spec JSON");
    assert_eq!(
        value["status"].as_str(),
        Some("preregistered-before-optimizer-output")
    );
    let criteria = value["frozen_reproduction_criteria"]
        .as_array()
        .expect("criteria array");
    assert_eq!(criteria.len(), 3);

    let expected = [
        ("chi2__BAO", 10.282299_f64, 0.01_f64),
        ("omegam", 0.29717936_f64, 0.001_f64),
        ("hrdrag", 101.54786_f64, 0.1_f64),
    ];
    for (name, reference, tolerance) in expected {
        let matches: Vec<_> = criteria
            .iter()
            .filter(|criterion| criterion["statistic"].as_str() == Some(name))
            .collect();
        assert_eq!(matches.len(), 1, "criterion {name}");
        assert_eq!(matches[0]["reference_value"].as_f64(), Some(reference));
        assert_eq!(matches[0]["max_absolute_delta"].as_f64(), Some(tolerance));
    }
}

#[test]
fn a2q_preserves_execution_error_as_unassessed_evidence() {
    let value: Value = serde_json::from_str(SPEC).expect("valid A2Q spec JSON");
    let classification = &value["classification"];
    assert_eq!(
        classification["valid_execution_error"]["qualification_verdict"].as_str(),
        Some("PASS")
    );
    assert_eq!(
        classification["valid_execution_error"]["reproduction_verdict"].as_str(),
        Some("UNASSESSED")
    );
    assert!(SOURCE.contains("if state == \"EXECUTION_ERROR\""));
    assert!(SOURCE.contains("(\"UNASSESSED\", Vec::new())"));
}

#[test]
fn a2q_negative_is_not_invalid_and_never_promotes_a3() {
    let value: Value = serde_json::from_str(SPEC).expect("valid A2Q spec JSON");
    let classification = &value["classification"];
    assert_eq!(
        classification["valid_executed_any_criterion_outside_tolerance"]
            ["qualification_verdict"]
            .as_str(),
        Some("PASS")
    );
    assert_eq!(
        classification["valid_executed_any_criterion_outside_tolerance"]
            ["reproduction_verdict"]
            .as_str(),
        Some("NEGATIVE")
    );
    assert_eq!(
        value["promotion_boundary"]["a3_execution_authorized"].as_bool(),
        Some(false)
    );
    assert_eq!(
        value["promotion_boundary"]["scientific_claim"].as_str(),
        Some("NONE")
    );
    assert!(SOURCE.contains("a3_execution_authorized: false"));
}

#[test]
fn a2q_does_not_reclassify_sampled_and_derived_roles() {
    let value: Value = serde_json::from_str(SPEC).expect("valid A2Q spec JSON");
    assert_eq!(
        value["role_semantics"]["criterion_membership_does_not_imply_sampled_parameter"]
            .as_bool(),
        Some(true)
    );
    assert_eq!(
        value["role_semantics"]["sampled_vs_derived_roles_come_from_qualified_A2R"]
            .as_bool(),
        Some(true)
    );
}
