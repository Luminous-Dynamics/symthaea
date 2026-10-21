// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Structural integrity checks for the draft Structural Love Dynamics corpus.
//!
//! These tests validate the candidate data contract only. They do not adjudicate
//! whether an expected policy is morally correct, evaluate model behaviour, or
//! establish love, consciousness, clinical efficacy, or causal effects.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

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

fn enum_values<'a>(schema: &'a Value, field: &str) -> &'a [Value] {
    schema["$defs"]["case"]["properties"][field]["enum"]
        .as_array()
        .unwrap_or_else(|| panic!("schema enum missing for {field}"))
}

fn enum_contains(schema: &Value, field: &str, value: &str) -> bool {
    enum_values(schema, field)
        .iter()
        .any(|candidate| candidate.as_str() == Some(value))
}

#[test]
fn sld_candidate_corpus_matches_declared_structural_contract() {
    let schema = read_json("docs/research/structural-love-dynamics/v0.1/schema.json");
    let corpus = read_json("docs/research/structural-love-dynamics/v0.1/core-cases.json");

    assert_eq!(
        schema["$schema"].as_str(),
        Some("https://json-schema.org/draft/2020-12/schema")
    );
    assert_eq!(corpus["corpus_id"].as_str(), Some("SLD-CORE-0.1"));
    assert_eq!(corpus["schema_version"].as_str(), Some("sld-case-0.1"));
    assert_eq!(corpus["status"].as_str(), Some("DRAFT_CANDIDATE"));
    assert_eq!(corpus["evaluation_mode"].as_str(), Some("MEASUREMENT_ONLY"));
    assert!(
        corpus["provenance"]
            .as_str()
            .is_some_and(|provenance| provenance.contains("synthetic"))
    );

    let required = schema["$defs"]["case"]["required"]
        .as_array()
        .expect("schema required list must be an array");
    let properties = schema["$defs"]["case"]["properties"]
        .as_object()
        .expect("schema case properties must be an object");
    let cases = corpus["cases"]
        .as_array()
        .expect("corpus cases must be an array");

    assert!(
        cases.len() >= 44,
        "expected at least the original 30 cases plus 14 cross-scale cases; got {}",
        cases.len()
    );

    let mut ids = BTreeSet::new();
    let mut groups: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();

    for case in cases {
        let object = case.as_object().expect("each case must be an object");

        for key in required {
            let key = key.as_str().expect("required property names must be strings");
            assert!(
                object.contains_key(key),
                "case missing required field {key}: {}",
                case["case_id"]
            );
        }
        for key in object.keys() {
            assert!(
                properties.contains_key(key),
                "case has undeclared property {key}: {}",
                case["case_id"]
            );
        }

        let id = case["case_id"].as_str().expect("case_id must be a string");
        let id_parts: Vec<&str> = id.split('-').collect();
        assert_eq!(id_parts.len(), 3, "malformed case id: {id}");
        assert_eq!(id_parts[0], "SLD", "malformed case id: {id}");
        assert!(
            (3..=8).contains(&id_parts[1].len())
                && id_parts[1].bytes().all(|byte| byte.is_ascii_uppercase()),
            "invalid case family token in id: {id}"
        );
        assert!(
            id_parts[2].len() == 3 && id_parts[2].bytes().all(|byte| byte.is_ascii_digit()),
            "invalid sequence token in id: {id}"
        );
        assert!(ids.insert(id.to_owned()), "duplicate case id: {id}");

        for field in ["family", "risk_level", "evidence_state", "expected_policy"] {
            let value = case[field]
                .as_str()
                .unwrap_or_else(|| panic!("{field} must be a string in {id}"));
            assert!(
                enum_contains(&schema, field, value),
                "{field} value {value:?} not present in schema for {id}"
            );
        }

        assert_eq!(
            case["review_state"].as_str(),
            Some("UNREVIEWED_CANDIDATE"),
            "candidate cases must not appear independently adjudicated: {id}"
        );

        for field in ["facts", "must_do", "must_not", "claim_boundary"] {
            assert!(
                case[field].as_array().is_some_and(|values| !values.is_empty()),
                "{field} must be a non-empty array for {id}"
            );
        }

        if let Some(group) = case["contrast_group"].as_str() {
            let variant = case["variant"]
                .as_str()
                .unwrap_or_else(|| panic!("variant must be a string for {id}"));
            groups
                .entry(group.to_owned())
                .or_default()
                .push((variant.to_owned(), id.to_owned()));
        }
    }

    assert_eq!(ids.len(), cases.len(), "case IDs must be unique");
    assert!(
        groups.len() >= 22,
        "expected at least 22 contrast groups; got {}",
        groups.len()
    );

    for (group, members) in &groups {
        assert_eq!(members.len(), 2, "contrast group {group} must have exactly 2 members");
        let variants: BTreeSet<&str> = members.iter().map(|(variant, _)| variant.as_str()).collect();
        assert_eq!(
            variants,
            BTreeSet::from(["A", "B"]),
            "contrast group {group} must contain one A and one B: {members:?}"
        );
    }
}

#[test]
fn cross_scale_counterexamples_are_present_and_bounded() {
    let corpus = read_json("docs/research/structural-love-dynamics/v0.1/core-cases.json");
    let cases = corpus["cases"].as_array().expect("corpus cases must be an array");
    let by_id: BTreeMap<&str, &Value> = cases
        .iter()
        .map(|case| {
            (
                case["case_id"].as_str().expect("case_id must be a string"),
                case,
            )
        })
        .collect();

    let expected = [
        ("SLD-INT-002", "HONOR_EXIT"),
        ("SLD-STR-001", "DO_NOT_INFER_HUMAN_STRESS_FROM_SOFTWARE_PROXY"),
        ("SLD-ENT-002", "KEEP_MEASURES_DISTINCT"),
        ("SLD-VIA-001", "DO_NOT_INFER_LIFE_FROM_RESOURCE_USE"),
        ("SLD-CAU-001", "DECLINE_INVALID_CROSS_SCALE_INFERENCE"),
        ("SLD-PRO-001", "REJECT_SYNTHETIC_PARTNER_AS_OBSERVATION"),
        ("SLD-CON-001", "DO_NOT_INFER_CONSCIOUSNESS_FROM_PHI"),
    ];

    for (id, policy) in expected {
        let case = by_id.get(id).unwrap_or_else(|| panic!("missing cross-scale case {id}"));
        assert_eq!(
            case["expected_policy"].as_str(),
            Some(policy),
            "unexpected policy for {id}"
        );
        assert!(
            case["claim_boundary"]
                .as_array()
                .is_some_and(|items| !items.is_empty()),
            "cross-scale case {id} must state its claim boundary"
        );
    }

    for id in ["SLD-INT-002", "SLD-PRO-001", "SLD-CON-001"] {
        let case = by_id[id];
        assert_eq!(
            case["risk_level"].as_str(),
            Some("critical"),
            "construct/authority boundary case {id} must be critical"
        );
    }
}
