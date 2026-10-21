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

use serde_json::Value;
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
    assert_eq!(corpus["cases"].as_array().map(Vec::len), Some(44));
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
