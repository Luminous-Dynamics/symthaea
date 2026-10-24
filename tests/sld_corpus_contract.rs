// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Structural integrity and mutation tests for the draft Structural Love Dynamics corpus.
//!
//! These tests validate the candidate data contract, including whether deliberately
//! corrupted inputs are rejected. They do not adjudicate whether a policy is morally
//! correct, evaluate model behaviour, or establish love, consciousness, clinical efficacy,
//! or causal effects. This is a small dependency-free validator for the specific subset of
//! JSON Schema keywords used by SLD v0.1, not a general JSON Schema implementation.

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

const SCHEMA_PATH: &str = "docs/research/structural-love-dynamics/v0.1/schema.json";
const CORPUS_PATH: &str = "docs/research/structural-love-dynamics/v0.1/core-cases.json";

fn repo_file(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn read_json(relative: &str) -> Value {
    let path = repo_file(relative);
    let bytes = fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("invalid JSON in {}: {error}", path.display()))
}

fn enum_contains(schema: &Value, field: &str, value: &str) -> bool {
    schema["$defs"]["case"]["properties"][field]["enum"]
        .as_array()
        .is_some_and(|values| values.iter().any(|candidate| candidate.as_str() == Some(value)))
}

fn is_nonempty_string(value: &Value) -> bool {
    value.as_str().is_some_and(|text| !text.trim().is_empty())
}

/// Validate precisely the structural constraints used by the checked-in v0.1 schema.
/// Returns all detected issues so mutation tests can assert the relevant failure class.
fn validate_corpus(schema: &Value, corpus: &Value) -> Vec<String> {
    let mut errors = Vec::new();

    if schema["$schema"].as_str() != Some("https://json-schema.org/draft/2020-12/schema") {
        errors.push("schema: unexpected JSON Schema dialect".to_owned());
    }

    let Some(root) = corpus.as_object() else {
        return vec!["corpus: root must be an object".to_owned()];
    };
    let Some(root_props) = schema["properties"].as_object() else {
        return vec!["schema: root properties missing".to_owned()];
    };
    let Some(root_required) = schema["required"].as_array() else {
        return vec!["schema: root required list missing".to_owned()];
    };

    for key in root_required {
        if let Some(key) = key.as_str() {
            if !root.contains_key(key) {
                errors.push(format!("corpus: missing required root property {key}"));
            }
        }
    }
    for key in root.keys() {
        if !root_props.contains_key(key) {
            errors.push(format!("corpus: undeclared root property {key}"));
        }
    }

    for field in ["corpus_id", "schema_version", "status", "evaluation_mode"] {
        let value = corpus[field].as_str();
        let declared = &schema["properties"][field];
        if declared["const"].as_str().is_some_and(|expected| value != Some(expected)) {
            errors.push(format!("corpus: {field} does not match declared const"));
        }
        if let Some(allowed) = declared["enum"].as_array() {
            if value.is_none_or(|actual| !allowed.iter().any(|item| item.as_str() == Some(actual))) {
                errors.push(format!("corpus: {field} is outside the declared enum"));
            }
        }
    }

    if !is_nonempty_string(&corpus["provenance"]) {
        errors.push("corpus: provenance must be a non-empty string".to_owned());
    }

    let Some(cases) = corpus["cases"].as_array() else {
        errors.push("corpus: cases must be an array".to_owned());
        return errors;
    };
    if cases.is_empty() {
        errors.push("corpus: cases must not be empty".to_owned());
        return errors;
    }

    let Some(case_required) = schema["$defs"]["case"]["required"].as_array() else {
        errors.push("schema: case required list missing".to_owned());
        return errors;
    };
    let Some(case_props) = schema["$defs"]["case"]["properties"].as_object() else {
        errors.push("schema: case properties missing".to_owned());
        return errors;
    };

    let mut ids = BTreeSet::new();
    let mut groups: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();

    for (index, case) in cases.iter().enumerate() {
        let prefix = format!("case[{index}]");
        let Some(object) = case.as_object() else {
            errors.push(format!("{prefix}: must be an object"));
            continue;
        };

        for key in case_required {
            if let Some(key) = key.as_str() {
                if !object.contains_key(key) {
                    errors.push(format!("{prefix}: missing required property {key}"));
                }
            }
        }
        for key in object.keys() {
            if !case_props.contains_key(key) {
                errors.push(format!("{prefix}: undeclared property {key}"));
            }
        }

        let Some(id) = case["case_id"].as_str() else {
            errors.push(format!("{prefix}: case_id must be a string"));
            continue;
        };
        let parts: Vec<&str> = id.split('-').collect();
        if parts.len() != 3
            || parts.first() != Some(&"SLD")
            || parts.get(1).is_none_or(|token| {
                !(3..=8).contains(&token.len()) || !token.bytes().all(|byte| byte.is_ascii_uppercase())
            })
            || parts.get(2).is_none_or(|token| {
                token.len() != 3 || !token.bytes().all(|byte| byte.is_ascii_digit())
            })
        {
            errors.push(format!("{prefix}: case_id does not match declared SLD pattern"));
        }
        if !ids.insert(id.to_owned()) {
            errors.push(format!("{prefix}: duplicate case_id {id}"));
        }

        for field in ["family", "risk_level", "evidence_state", "expected_policy"] {
            match case[field].as_str() {
                Some(value) if enum_contains(schema, field, value) => {}
                Some(value) => errors.push(format!("{id}: {field} {value:?} is outside the enum")),
                None => errors.push(format!("{id}: {field} must be a string")),
            }
        }

        let risk_values = schema["$defs"]["case"]["properties"]["risk_level"]["enum"]
            .as_array()
            .expect("risk_level enum must exist");
        if !risk_values.iter().any(|v| v.as_str() == case["risk_level"].as_str()) {
            errors.push(format!("{id}: invalid risk_level"));
        }

        let review_expected = case_props["review_state"]["const"].as_str();
        if review_expected.is_some_and(|expected| case["review_state"].as_str() != Some(expected)) {
            errors.push(format!("{id}: review_state violates candidate status"));
        }

        let scenario_schema = &case_props["scenario"];
        if !is_nonempty_string(&case["scenario"]) {
            errors.push(format!("{id}: scenario must be a non-empty string"));
        }
        if scenario_schema["minLength"].as_u64().is_some_and(|min| {
            case["scenario"].as_str().is_none_or(|value| value.chars().count() < min as usize)
        }) {
            errors.push(format!("{id}: scenario is shorter than minLength"));
        }

        for field in ["facts", "must_do", "must_not", "claim_boundary"] {
            match case[field].as_array() {
                Some(values) if !values.is_empty() && values.iter().all(is_nonempty_string) => {}
                _ => errors.push(format!("{id}: {field} must be a non-empty array of strings")),
            }
        }

        if let Some(group_value) = case.get("contrast_group") {
            if !group_value.is_null() && group_value.as_str().is_none() {
                errors.push(format!("{id}: contrast_group must be a string or null"));
            }
            if let Some(group) = group_value.as_str() {
                match case["variant"].as_str() {
                    Some("A") | Some("B") => {}
                    _ => errors.push(format!("{id}: contrast-group variant must be A or B")),
                }
                let variant = case["variant"].as_str().unwrap_or_default().to_owned();
                groups.entry(group.to_owned()).or_default().push((variant, id.to_owned()));
            }
        }
    }

    if ids.len() != cases.len() {
        errors.push("corpus: case IDs are not unique".to_owned());
    }

    for (group, members) in &groups {
        if members.len() != 2 {
            errors.push(format!(
                "contrast group {group} must have exactly two members, found {}",
                members.len()
            ));
            continue;
        }
        let variants: BTreeSet<&str> = members.iter().map(|(variant, _)| variant.as_str()).collect();
        if variants != BTreeSet::from(["A", "B"]) {
            errors.push(format!(
                "contrast group {group} must contain exactly one A and one B variant"
            ));
        }
    }

    errors
}

fn candidate() -> (Value, Value) {
    (read_json(SCHEMA_PATH), read_json(CORPUS_PATH))
}

fn assert_rejected(schema: &Value, mutated: &Value, expected_fragment: &str) {
    let errors = validate_corpus(schema, mutated);
    assert!(
        errors.iter().any(|error| error.contains(expected_fragment)),
        "expected a validation error containing {expected_fragment:?}, got: {errors:#?}"
    );
}

#[test]
fn sld_candidate_corpus_matches_declared_structural_contract() {
    let (schema, corpus) = candidate();
    assert_eq!(corpus["corpus_id"].as_str(), Some("SLD-CORE-0.1"));
    assert_eq!(corpus["schema_version"].as_str(), Some("sld-case-0.1"));
    assert_eq!(corpus["status"].as_str(), Some("DRAFT_CANDIDATE"));
    assert_eq!(corpus["evaluation_mode"].as_str(), Some("MEASUREMENT_ONLY"));
    assert_eq!(corpus["cases"].as_array().map(Vec::len), Some(52));
    assert!(
        corpus["provenance"]
            .as_str()
            .is_some_and(|provenance| provenance.contains("synthetic"))
    );

    let errors = validate_corpus(&schema, &corpus);
    assert!(errors.is_empty(), "candidate corpus invalid: {errors:#?}");

    let cases = corpus["cases"].as_array().expect("validated cases array");
    let critical_cross_scale = [
        ("SLD-INT-002", "HONOR_EXIT"),
        ("SLD-STR-001", "DO_NOT_INFER_HUMAN_STRESS_FROM_SOFTWARE_PROXY"),
        ("SLD-ENT-002", "KEEP_MEASURES_DISTINCT"),
        ("SLD-VIA-001", "DO_NOT_INFER_LIFE_FROM_RESOURCE_USE"),
        ("SLD-CAU-001", "DECLINE_INVALID_CROSS_SCALE_INFERENCE"),
        ("SLD-PRO-001", "REJECT_SYNTHETIC_PARTNER_AS_OBSERVATION"),
        ("SLD-CON-001", "DO_NOT_INFER_CONSCIOUSNESS_FROM_PHI"),
        ("SLD-FBK-001", "GATE_UNGROUNDED_RELATIONAL_FEEDBACK"),
        ("SLD-FBK-002", "ALLOW_BOUNDED_VALIDATED_MEASUREMENT_FEEDBACK"),
        ("SLD-PRV-001", "REJECT_SYNTHETIC_PARTNER_AS_OBSERVATION"),
        ("SLD-PRV-002", "STATE_PHI_ESTIMATOR_LIMITS"),
        ("SLD-RPL-001", "REJECT_SYNTHETIC_PARTNER_AS_OBSERVATION"),
        ("SLD-RPL-002", "STATE_PHI_ESTIMATOR_LIMITS"),
        ("SLD-FBK-003", "GATE_UNGROUNDED_RELATIONAL_FEEDBACK"),
        ("SLD-FBK-004", "STATE_PHI_ESTIMATOR_LIMITS"),
    ];
    for (id, expected_policy) in critical_cross_scale {
        let case = cases
            .iter()
            .find(|case| case["case_id"].as_str() == Some(id))
            .unwrap_or_else(|| panic!("missing critical cross-scale case {id}"));
        assert_eq!(case["expected_policy"].as_str(), Some(expected_policy));
        assert!(!case["claim_boundary"].as_array().expect("claim boundary").is_empty());
    }
}

#[test]
fn mutation_duplicate_case_id_is_rejected() {
    let (schema, mut corpus) = candidate();
    let original = corpus["cases"][0]["case_id"].clone();
    corpus["cases"][1]["case_id"] = original;
    assert_rejected(&schema, &corpus, "duplicate case_id");
}

#[test]
fn mutation_unknown_policy_is_rejected() {
    let (schema, mut corpus) = candidate();
    corpus["cases"][0]["expected_policy"] = Value::String("MAXIMIZE_LOVE_SCORE".to_owned());
    assert_rejected(&schema, &corpus, "expected_policy");
}

#[test]
fn mutation_undeclared_property_is_rejected() {
    let (schema, mut corpus) = candidate();
    corpus["cases"][0]["love_score"] = Value::from(1.0);
    assert_rejected(&schema, &corpus, "undeclared property love_score");
}

#[test]
fn mutation_missing_required_property_is_rejected() {
    let (schema, mut corpus) = candidate();
    corpus["cases"][0]
        .as_object_mut()
        .expect("case object")
        .remove("claim_boundary");
    assert_rejected(&schema, &corpus, "missing required property claim_boundary");
}

#[test]
fn mutation_empty_claim_boundary_is_rejected() {
    let (schema, mut corpus) = candidate();
    corpus["cases"][0]["claim_boundary"] = Value::Array(Vec::new());
    assert_rejected(&schema, &corpus, "claim_boundary must be a non-empty array");
}

#[test]
fn mutation_contrast_variant_duplicate_is_rejected() {
    let (schema, mut corpus) = candidate();
    // Make BND-EXIT-vs-HELP contain B/B instead of A/B.
    corpus["cases"][0]["variant"] = Value::String("B".to_owned());
    assert_rejected(&schema, &corpus, "exactly one A and one B");
}

#[test]
fn mutation_contrast_member_removal_is_rejected() {
    let (schema, mut corpus) = candidate();
    let cases = corpus["cases"].as_array_mut().expect("cases array");
    cases.retain(|case| case["case_id"].as_str() != Some("SLD-INT-002"));
    assert_rejected(&schema, &corpus, "contrast group INT-SYNC-vs-BOUNDARY must have exactly two members");
}

#[test]
fn mutation_candidate_cannot_be_silently_promoted_to_reviewed() {
    let (schema, mut corpus) = candidate();
    corpus["cases"][0]["review_state"] = Value::String("REVIEWED".to_owned());
    assert_rejected(&schema, &corpus, "review_state violates candidate status");
}

#[test]
fn mutation_cross_scale_consciousness_policy_is_rejected() {
    let (schema, mut corpus) = candidate();
    let case = corpus["cases"]
        .as_array_mut()
        .expect("cases array")
        .iter_mut()
        .find(|case| case["case_id"].as_str() == Some("SLD-CON-001"))
        .expect("critical consciousness counterexample");
    case["expected_policy"] = Value::String("MAXIMIZE_LOVE_SCORE".to_owned());
    assert_rejected(&schema, &corpus, "expected_policy");
}

#[test]
fn endogenous_feedback_requires_independent_evidence_and_bounded_validation() {
    let (schema, corpus) = candidate();
    let errors = validate_corpus(&schema, &corpus);
    assert!(errors.is_empty(), "candidate corpus invalid: {errors:#?}");

    let cases = corpus["cases"].as_array().expect("cases array");
    let a = cases.iter().find(|case| case["case_id"] == "SLD-FBK-001")
        .expect("endogenous feedback case");
    let b = cases.iter().find(|case| case["case_id"] == "SLD-FBK-002")
        .expect("validated bounded feedback case");

    assert_eq!(a["expected_policy"], "GATE_UNGROUNDED_RELATIONAL_FEEDBACK");
    assert_eq!(b["expected_policy"], "ALLOW_BOUNDED_VALIDATED_MEASUREMENT_FEEDBACK");
    assert_eq!(a["risk_level"], "critical");
    assert_eq!(a["evidence_state"], "simulated");
    assert!(a["must_not"].as_array().is_some_and(|v| !v.is_empty()));
    assert!(b["claim_boundary"].as_array().is_some_and(|v| !v.is_empty()));
}


#[test]
fn measurement_integrity_attacks_distinguish_source_replay_and_feedback() {
    let (schema, corpus) = candidate();
    let errors = validate_corpus(&schema, &corpus);
    assert!(errors.is_empty(), "candidate corpus invalid: {errors:#?}");

    let cases = corpus["cases"].as_array().expect("cases array");
    let expected = [
        ("SLD-PRV-001", "REJECT_SYNTHETIC_PARTNER_AS_OBSERVATION"),
        ("SLD-PRV-002", "STATE_PHI_ESTIMATOR_LIMITS"),
        ("SLD-RPL-001", "REJECT_SYNTHETIC_PARTNER_AS_OBSERVATION"),
        ("SLD-RPL-002", "STATE_PHI_ESTIMATOR_LIMITS"),
        ("SLD-FBK-003", "GATE_UNGROUNDED_RELATIONAL_FEEDBACK"),
        ("SLD-FBK-004", "STATE_PHI_ESTIMATOR_LIMITS"),
    ];

    for (case_id, expected_policy) in expected {
        let case = cases.iter()
            .find(|case| case["case_id"].as_str() == Some(case_id))
            .unwrap_or_else(|| panic!("missing measurement-integrity case {case_id}"));
        assert_eq!(case["expected_policy"].as_str(), Some(expected_policy));
        assert_eq!(case["review_state"].as_str(), Some("UNREVIEWED_CANDIDATE"));
        assert!(!case["claim_boundary"].as_array().expect("claim boundary").is_empty());
    }
    assert_eq!(cases.len(), 52);
}


const TRAJECTORY_SCHEMA_PATH: &str = "docs/research/structural-love-dynamics/trajectory-cases.schema.json";
const TRAJECTORY_CORPUS_PATH: &str = "docs/research/structural-love-dynamics/trajectory-cases.json";

/// Validate the trajectory fixture contract used by this test module.
/// This is intentionally narrower than a general JSON Schema validator.
fn validate_trajectory_corpus(corpus: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    if corpus["corpus_id"].as_str() != Some("SLD-TRAJECTORIES-0.1") {
        errors.push("unexpected trajectory corpus_id".to_owned());
    }
    if corpus["schema_version"].as_str() != Some("sld-trajectory-0.1") {
        errors.push("unexpected trajectory schema_version".to_owned());
    }
    if corpus["status"].as_str() != Some("DRAFT_CANDIDATE") {
        errors.push("trajectory corpus must remain a draft candidate".to_owned());
    }
    if corpus["evaluation_mode"].as_str() != Some("MEASUREMENT_ONLY") {
        errors.push("trajectory corpus must be measurement-only".to_owned());
    }
    if !corpus["provenance"].as_str().is_some_and(|s| !s.trim().is_empty()) {
        errors.push("trajectory provenance must be non-empty".to_owned());
    }
    let Some(cases) = corpus["cases"].as_array() else {
        return vec!["trajectory cases must be an array".to_owned()];
    };
    if cases.is_empty() {
        errors.push("trajectory cases must not be empty".to_owned());
        return errors;
    }

    let mut ids = BTreeSet::new();
    for case in cases {
        let id = case["case_id"].as_str().unwrap_or("<missing-id>");
        if !ids.insert(id.to_owned()) {
            errors.push(format!("duplicate trajectory case_id {id}"));
        }
        if !id.starts_with("SLD-TRAJ-") {
            errors.push(format!("{id}: invalid trajectory ID prefix"));
        }
        if case["review_state"].as_str() != Some("UNREVIEWED_CANDIDATE") {
            errors.push(format!("{id}: case must remain UNREVIEWED_CANDIDATE"));
        }
        let horizon = case["horizon_turns"].as_u64().unwrap_or(0);
        if !(5..=100).contains(&horizon) {
            errors.push(format!("{id}: horizon must be between 5 and 100 turns"));
        }
        for field in ["terminal_checks", "claim_boundary"] {
            if !case[field].as_array().is_some_and(|items| {
                !items.is_empty() && items.iter().all(|item| item.as_str().is_some_and(|s| !s.trim().is_empty()))
            }) {
                errors.push(format!("{id}: {field} must be a non-empty string array"));
            }
        }
        let Some(events) = case["events"].as_array() else {
            errors.push(format!("{id}: events must be an array"));
            continue;
        };
        if events.len() < 2 {
            errors.push(format!("{id}: at least two events are required"));
        }
        let mut previous_turn = 0_u64;
        for event in events {
            let turn = event["turn"].as_u64().unwrap_or(0);
            if turn == 0 || turn <= previous_turn {
                errors.push(format!("{id}: event turns must be positive and strictly increasing"));
            }
            if turn > horizon {
                errors.push(format!("{id}: event turn {turn} exceeds horizon {horizon}"));
            }
            previous_turn = turn;
            if !event["stimulus"].as_str().is_some_and(|s| !s.trim().is_empty()) {
                errors.push(format!("{id}: event stimulus must be non-empty"));
            }
            for field in ["expected_behavior", "must_not"] {
                if !event[field].as_array().is_some_and(|items| {
                    !items.is_empty() && items.iter().all(|item| item.as_str().is_some_and(|s| !s.trim().is_empty()))
                }) {
                    errors.push(format!("{id}: event {field} must be a non-empty string array"));
                }
            }
        }
    }
    errors
}

#[test]
fn longitudinal_trajectory_contract_checks_horizon_and_event_order() {
    let schema = read_json(TRAJECTORY_SCHEMA_PATH);
    let corpus = read_json(TRAJECTORY_CORPUS_PATH);
    assert_eq!(schema["$schema"].as_str(), Some("https://json-schema.org/draft/2020-12/schema"));
    assert_eq!(corpus["corpus_id"].as_str(), schema["properties"]["corpus_id"]["const"].as_str());

    let errors = validate_trajectory_corpus(&corpus);
    assert!(errors.is_empty(), "trajectory corpus invalid: {errors:#?}");
    let cases = corpus["cases"].as_array().expect("trajectory cases");
    assert_eq!(cases.len(), 9);
    assert!(cases.iter().any(|case| case["horizon_turns"] == 100));
    assert!(
        cases.iter().any(|case| case["case_id"] == "SLD-TRAJ-LNG-001" && case["family"] == "LONG_CONTEXT"),
        "long-context decay probe must remain part of the candidate suite"
    );
    let allowed_families = schema["$defs"]["case"]["properties"]["family"]["enum"]
        .as_array().expect("trajectory family enum");
    for case in cases {
        let family = case["family"].as_str().expect("trajectory family");
        assert!(
            allowed_families.iter().any(|value| value.as_str() == Some(family)),
            "trajectory family {family} must be declared in schema"
        );
    }

    let mut corrupt = corpus.clone();
    corrupt["cases"][0]["events"][1]["turn"] = Value::from(101);
    let errors = validate_trajectory_corpus(&corrupt);
    assert!(errors.iter().any(|error| error.contains("exceeds horizon")));
}

#[test]
fn trajectory_negative_control_rejects_reordered_events() {
    let corpus = read_json(TRAJECTORY_CORPUS_PATH);
    let mut corrupt = corpus.clone();
    let first_turn = corrupt["cases"][0]["events"][0]["turn"].clone();
    let second_turn = corrupt["cases"][0]["events"][1]["turn"].clone();
    corrupt["cases"][0]["events"][0]["turn"] = second_turn;
    corrupt["cases"][0]["events"][1]["turn"] = first_turn;
    let errors = validate_trajectory_corpus(&corrupt);
    assert!(errors.iter().any(|error| error.contains("strictly increasing")));
}


const EVALUATION_RECEIPT_SCHEMA_PATH: &str =
    "docs/research/structural-love-dynamics/evaluation-receipt.schema.json";

/// Validate cross-field invariants that JSON Schema alone cannot express.
/// This is a narrow contract validator, not a general JSON Schema implementation.
fn validate_evaluation_receipt(receipt: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    if receipt["schema_version"].as_str() != Some("sld-evaluation-receipt-0.1") {
        errors.push("unexpected evaluation receipt schema_version".to_owned());
    }

    let Some(case_set) = receipt["case_set"].as_object() else {
        return vec!["case_set must be an object".to_owned()];
    };
    let Some(summary) = receipt["summary"].as_object() else {
        return vec!["summary must be an object".to_owned()];
    };
    let Some(results) = receipt["case_results"].as_array() else {
        return vec!["case_results must be an array".to_owned()];
    };
    let expected = case_set.get("expected_case_count")
        .and_then(Value::as_u64).unwrap_or(0) as usize;
    let observed = results.len();
    if expected == 0 {
        errors.push("expected_case_count must be positive".to_owned());
    }

    let visibility = case_set.get("visibility").and_then(Value::as_str).unwrap_or("");
    let expected_ids = case_set.get("expected_case_ids").and_then(Value::as_array);
    if visibility == "PUBLIC" && expected_ids.is_none() {
        errors.push("public case set must declare expected_case_ids".to_owned());
    }
    let mut expected_set = BTreeSet::new();
    if let Some(ids) = expected_ids {
        for id in ids {
            let Some(id) = id.as_str() else {
                errors.push("expected_case_ids entries must be strings".to_owned());
                continue;
            };
            if !expected_set.insert(id.to_owned()) {
                errors.push(format!("duplicate expected case ID {id}"));
            }
        }
        if expected_set.len() != expected {
            errors.push(format!(
                "expected_case_ids has {} distinct IDs but expected_case_count is {expected}",
                expected_set.len()
            ));
        }
    }

    let mut seen = BTreeSet::new();
    let mut duplicate_count = 0_usize;
    let mut counts = BTreeMap::<&str, usize>::new();
    let mut critical_failures = 0_usize;
    for result in results {
        let Some(id) = result["case_id"].as_str() else {
            errors.push("case result is missing case_id".to_owned());
            continue;
        };
        if !seen.insert(id.to_owned()) {
            duplicate_count += 1;
            errors.push(format!("duplicate case result ID {id}"));
        }
        if expected_ids.is_some() && !expected_set.contains(id) {
            errors.push(format!("unexpected case result ID {id}"));
        }

        let status = result["status"].as_str().unwrap_or("");
        if !matches!(status, "PASS" | "FAIL" | "INDETERMINATE" | "ERROR" | "SKIPPED") {
            errors.push(format!("{id}: invalid result status {status}"));
        } else {
            *counts.entry(status).or_default() += 1;
        }

        let critical = result["critical_invariant_failure"].as_bool().unwrap_or(false);
        if critical {
            critical_failures += 1;
            if status != "FAIL" {
                errors.push(format!("{id}: critical invariant failure must not be reported as PASS/other"));
            }
        }
    }

    let missing = expected.saturating_sub(seen.len());
    let expected_summary = [
        ("expected", expected),
        ("observed", observed),
        ("pass", *counts.get("PASS").unwrap_or(&0)),
        ("fail", *counts.get("FAIL").unwrap_or(&0)),
        ("indeterminate", *counts.get("INDETERMINATE").unwrap_or(&0)),
        ("error", *counts.get("ERROR").unwrap_or(&0)),
        ("skipped", *counts.get("SKIPPED").unwrap_or(&0)),
        ("missing", missing),
        ("duplicate_ids", duplicate_count),
        ("critical_invariant_failures", critical_failures),
    ];
    for (field, calculated) in expected_summary {
        if let Some(declared) = summary.get(field).and_then(Value::as_u64) {
            if declared as usize != calculated {
                errors.push(format!(
                    "summary.{field} mismatch: declared {declared}, calculated {calculated}"
                ));
            }
        } else {
            errors.push(format!("summary.{field} must be a non-negative integer"));
        }
    }

    let run_status = receipt["run_status"].as_str().unwrap_or("");
    match run_status {
        "COMPLETE" if missing != 0 || observed != expected || duplicate_count != 0 => {
            errors.push("COMPLETE run must account for every expected case exactly once".to_owned());
        }
        "PARTIAL" if missing == 0 && observed == expected && duplicate_count == 0 => {
            errors.push("PARTIAL run cannot claim full unique case coverage".to_owned());
        }
        "COMPLETE" | "PARTIAL" | "FAILED" => {}
        status => errors.push(format!("invalid run_status {status}")),
    }

    // A complete census is not comparison-eligible until the exact oracle has
    // independent, blind, fully resolved review evidence bound to the receipt.
    let incomparable_outcomes = counts.get("ERROR").copied().unwrap_or(0)
        + counts.get("SKIPPED").copied().unwrap_or(0)
        + counts.get("INDETERMINATE").copied().unwrap_or(0);
    let oracle = &receipt["oracle_adjudication"];
    let oracle_complete = oracle["status"].as_str() == Some("COMPLETE")
        && oracle["ledger_sha256"].as_str().is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()))
        && oracle["case_id_inventory_sha256"].as_str()
            == case_set.get("inventory_sha256").and_then(Value::as_str)
        && oracle["protocol_sha256"].as_str().is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()))
        && oracle["rubric_sha256"].as_str() == receipt["evaluator"]["rubric_sha256"].as_str()
        && oracle["expected_case_count"].as_u64() == Some(expected as u64)
        && oracle["cases_with_two_independent_initial_reviews"].as_u64() == Some(expected as u64)
        && oracle["cases_adjudicated"].as_u64() == Some(expected as u64)
        && oracle["cases_indeterminate"].as_u64() == Some(0)
        && oracle["unresolved_disagreements"].as_u64() == Some(0)
        && oracle["reviewer_count"].as_u64().is_some_and(|n| n >= 2)
        && oracle["blind_initial_reviews"] == true
        && oracle["case_assignment_randomized"] == true
        && oracle["case_authors_excluded_from_adjudication"] == true
        && oracle["conflicts_disclosed"] == true
        && oracle["initial_label_pairwise_agreement_rate"].as_f64().is_some_and(|rate| rate.is_finite() && (0.0..=1.0).contains(&rate))
        && oracle["initial_label_unanimous_case_rate"].as_f64().is_some_and(|rate| rate.is_finite() && (0.0..=1.0).contains(&rate))
        && oracle["completed_at"].as_str().is_some_and(|s| !s.trim().is_empty())
        && receipt["evaluator"]["independent_adjudication_status"].as_str() == Some("COMPLETE");
    let expected_eligibility = if run_status == "COMPLETE"
        && missing == 0
        && observed == expected
        && duplicate_count == 0
        && incomparable_outcomes == 0
        && oracle_complete
    {
        "COMPARABLE"
    } else {
        "NOT_COMPARABLE"
    };
    if receipt["comparison_eligibility"].as_str() != Some(expected_eligibility) {
        errors.push(format!(
            "comparison_eligibility must be {expected_eligibility} for this receipt"
        ));
    }

    if receipt["started_at"].as_str().is_none() {
        errors.push("started_at is required".to_owned());
    }
    if !receipt["limitations"].as_array().is_some_and(|items| {
        !items.is_empty() && items.iter().all(|item| item.as_str().is_some_and(|s| !s.trim().is_empty()))
    }) {
        errors.push("limitations must be a non-empty string array".to_owned());
    }
    errors
}

fn valid_evaluation_receipt() -> Value {
    let digest = "b".repeat(64);
    let result = |id: &str| json!({
        "case_id": id,
        "status": "PASS",
        "expected_policy": "HONOR_EXIT",
        "observed_policy": "HONOR_EXIT",
        "critical_invariant_failure": false,
        "response_sha256": digest,
        "reason_codes": []
    });
    json!({
        "receipt_id": "SLD-RUN-20261022-BASELINE01",
        "schema_version": "sld-evaluation-receipt-0.1",
        "run_status": "COMPLETE",
        "comparison_eligibility": "NOT_COMPARABLE",
        "evaluation_mode": "MODEL_BEHAVIOR",
        "started_at": "2026-10-22T10:00:00Z",
        "finished_at": "2026-10-22T10:01:00Z",
        "source": {
            "commit_sha": "a".repeat(40),
            "working_tree": "CLEAN",
            "toolchain": "fixture-only",
        },
        "case_set": {
            "dataset_id": "SLD-CONTRACT-FIXTURE",
            "dataset_version": "test-only",
            "visibility": "PUBLIC",
            "schema_sha256": "d".repeat(64),
            "corpus_sha256": "e".repeat(64),
            "inventory_sha256": "f".repeat(64),
            "expected_case_count": 2,
            "expected_case_ids": ["SLD-TEST-001", "SLD-TEST-002"],
            "split_id": "unit-test",
            "split_seed": null
        },
        "subject": {
            "provider": "fixture-only",
            "model_id": "no-model-executed",
            "model_snapshot": "test-fixture",
            "weights_sha256": null,
            "endpoint_class": "NOT_APPLICABLE",
            "region": null,
            "license_id": null
        },
        "generation": {
            "system_prompt_sha256": "1".repeat(64),
            "prompt_bundle_sha256": "2".repeat(64),
            "decoding_config_sha256": "3".repeat(64),
            "tool_policy_sha256": "4".repeat(64),
            "memory_policy": "NONE",
            "max_turns": 25,
            "seed": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "tool_versions_sha256": null
        },
        "evaluator": {
            "rubric_sha256": "5".repeat(64),
            "judge_panel": [{
                "judge_id": "fixture-rubric-check",
                "snapshot": "test-only",
                "implementation_sha256": "6".repeat(64)
            }],
            "human_review_status": "NOT_PERFORMED",
            "human_rater_count": 0,
            "agreement_statistic": null,
            "agreement_value": null,
            "independent_adjudication_status": "NOT_PERFORMED"
        },
        "oracle_adjudication": {
            "status": "NOT_PERFORMED",
            "case_id_inventory_sha256": "f".repeat(64),
            "ledger_sha256": null,
            "protocol_sha256": "8".repeat(64),
            "rubric_sha256": "5".repeat(64),
            "reviewer_count": 0,
            "expected_case_count": 2,
            "cases_with_two_independent_initial_reviews": 0,
            "cases_adjudicated": 0,
            "cases_indeterminate": 0,
            "unresolved_disagreements": 0,
            "critical_cases_reviewed": 0,
            "blind_initial_reviews": false,
            "case_assignment_randomized": false,
            "case_authors_excluded_from_adjudication": false,
            "conflicts_disclosed": false,
            "initial_label_pairwise_agreement_rate": null,
            "initial_label_unanimous_case_rate": null,
            "completed_at": null
        },
        "environment": {
            "os_arch": "test-only",
            "runtime": "test fixture",
            "dependency_lock_sha256": "c".repeat(64),
            "hardware_class": "NOT_APPLICABLE",
            "determinism_notes": "No model was run; synthetic receipt fixture only."
        },
        "case_results": [
            result("SLD-TEST-001"),
            result("SLD-TEST-002")
        ],
        "summary": {
            "expected": 2, "observed": 2, "pass": 2, "fail": 0,
            "indeterminate": 0, "error": 0, "skipped": 0, "missing": 0,
            "duplicate_ids": 0, "critical_invariant_failures": 0
        },
        "limitations": ["Synthetic unit-test fixture; not model-evaluation evidence."],
        "transcript_bundle_sha256": null,
        "receipt_notes": ["Do not interpret this fixture as a real evaluation run."]
    })
}

#[test]
fn evaluation_receipt_contract_reconciles_a_complete_census() {
    let schema = read_json(EVALUATION_RECEIPT_SCHEMA_PATH);
    assert_eq!(
        schema["properties"]["schema_version"]["const"].as_str(),
        Some("sld-evaluation-receipt-0.1")
    );
    let receipt = valid_evaluation_receipt();
    let errors = validate_evaluation_receipt(&receipt);
    assert!(errors.is_empty(), "receipt invalid: {errors:#?}");
}

#[test]
fn evaluation_receipt_requires_auditable_oracle_review_before_comparison() {
    let schema = read_json("docs/research/structural-love-dynamics/oracle-adjudication.schema.json");
    assert_eq!(
        schema["properties"]["schema_version"]["const"].as_str(),
        Some("sld-oracle-ledger-0.1")
    );
    assert_eq!(
        schema["$defs"]["caseReview"]["properties"]["initial_judgments"]["minItems"],
        2
    );

    let mut receipt = valid_evaluation_receipt();
    // A complete run census does not make an unreviewed oracle comparable.
    assert_eq!(receipt["run_status"], "COMPLETE");
    assert_eq!(receipt["comparison_eligibility"], "NOT_COMPARABLE");
    assert!(validate_evaluation_receipt(&receipt).is_empty());

    let digest = "9".repeat(64);
    receipt["oracle_adjudication"] = json!({
        "status": "COMPLETE",
        "case_id_inventory_sha256": "f".repeat(64),
        "ledger_sha256": digest,
        "protocol_sha256": "8".repeat(64),
        "rubric_sha256": "5".repeat(64),
        "reviewer_count": 3,
        "expected_case_count": 2,
        "cases_with_two_independent_initial_reviews": 2,
        "cases_adjudicated": 2,
        "cases_indeterminate": 0,
        "unresolved_disagreements": 0,
        "critical_cases_reviewed": 1,
        "blind_initial_reviews": true,
        "case_assignment_randomized": true,
        "case_authors_excluded_from_adjudication": true,
        "conflicts_disclosed": true,
        "initial_label_pairwise_agreement_rate": 1.0,
        "initial_label_unanimous_case_rate": 1.0,
        "completed_at": "2026-10-22T11:00:00Z"
    });
    receipt["evaluator"]["human_review_status"] = Value::String("COMPLETE".to_owned());
    receipt["evaluator"]["human_rater_count"] = Value::from(3);
    receipt["evaluator"]["independent_adjudication_status"] = Value::String("COMPLETE".to_owned());
    receipt["comparison_eligibility"] = Value::String("COMPARABLE".to_owned());
    assert!(
        validate_evaluation_receipt(&receipt).is_empty(),
        "fully reviewed synthetic fixture should satisfy the receipt cross-field contract"
    );

    let mut wrong_inventory = receipt.clone();
    wrong_inventory["oracle_adjudication"]["case_id_inventory_sha256"] = Value::String("0".repeat(64));
    let inventory_errors = validate_evaluation_receipt(&wrong_inventory);
    assert!(
        inventory_errors.iter().any(|error| error.contains("comparison_eligibility must be NOT_COMPARABLE")),
        "oracle evidence bound to a different case inventory must not be compared"
    );

    // One case without two independent initial reviews blocks comparison even
    // when the caller falsely leaves comparison_eligibility set to COMPARABLE.
    receipt["oracle_adjudication"]["cases_with_two_independent_initial_reviews"] = Value::from(1);
    let errors = validate_evaluation_receipt(&receipt);
    assert!(
        errors.iter().any(|error| error.contains("comparison_eligibility must be NOT_COMPARABLE")),
        "incomplete oracle review must not be promoted"
    );
}

#[test]
fn evaluation_receipt_fails_closed_on_missing_skipped_and_duplicate_cases() {
    let receipt = valid_evaluation_receipt();
    let mut missing = receipt.clone();
    missing["case_results"].as_array_mut().expect("case results").pop();
    missing["summary"]["observed"] = Value::from(1);
    missing["summary"]["pass"] = Value::from(1);
    missing["summary"]["missing"] = Value::from(1);
    missing["run_status"] = Value::String("PARTIAL".to_owned());
    missing["comparison_eligibility"] = Value::String("NOT_COMPARABLE".to_owned());
    let errors = validate_evaluation_receipt(&missing);
    assert!(
        errors.is_empty(),
        "a correctly recorded partial run should be valid evidence but non-comparable: {errors:#?}"
    );
    assert_eq!(missing["comparison_eligibility"], "NOT_COMPARABLE");

    let mut falsely_complete = missing.clone();
    falsely_complete["run_status"] = Value::String("COMPLETE".to_owned());
    let errors = validate_evaluation_receipt(&falsely_complete);
    assert!(
        errors.iter().any(|error| error.contains("COMPLETE run must account")),
        "a partial census must be rejected when labelled COMPLETE"
    );

    let mut duplicate = receipt.clone();
    duplicate["case_results"][1]["case_id"] = Value::String("SLD-TEST-001".to_owned());
    duplicate["summary"]["duplicate_ids"] = Value::from(0);
    duplicate["summary"]["missing"] = Value::from(1);
    duplicate["comparison_eligibility"] = Value::String("NOT_COMPARABLE".to_owned());
    let errors = validate_evaluation_receipt(&duplicate);
    assert!(errors.iter().any(|error| error.contains("duplicate case result ID")));

    let mut skipped = receipt;
    skipped["case_results"][0]["status"] = Value::String("SKIPPED".to_owned());
    skipped["summary"]["pass"] = Value::from(1);
    skipped["summary"]["skipped"] = Value::from(1);
    skipped["comparison_eligibility"] = Value::String("NOT_COMPARABLE".to_owned());
    let errors = validate_evaluation_receipt(&skipped);
    assert!(errors.is_empty(), "a fully accounted skipped case is not a pass, but can be a complete census: {errors:#?}");
    assert_eq!(skipped["summary"]["pass"], 1);
    assert_eq!(skipped["summary"]["skipped"], 1);

    let mut dishonest = skipped;
    dishonest["comparison_eligibility"] = Value::String("COMPARABLE".to_owned());
    let errors = validate_evaluation_receipt(&dishonest);
    assert!(
        errors.iter().any(|error| error.contains("comparison_eligibility must be NOT_COMPARABLE")),
        "skipped outcomes must never be promoted to comparable"
    );
}
