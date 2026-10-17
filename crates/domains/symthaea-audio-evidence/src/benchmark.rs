// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed validation for grouped train/validation/calibration/test splits.
//!
//! This guards against identity leakage in audio-preference benchmarks. It does
//! not validate model quality, randomization, consent, or the scientific design
//! of a study; those require separate review and evidence.

use serde_json::Value;
use std::collections::{BTreeSet, HashMap, HashSet};

const SCHEMA_VERSION: &str = "hifi-split-manifest-v1";
const ROLES: [&str; 4] = ["train", "validation", "calibration", "test"];
const GROUP_DIMENSIONS: [&str; 5] = [
    "listener_id",
    "excerpt_id",
    "product_revision_id",
    "measurement_session_id",
    "fixture_id",
];

/// Validates a split manifest and its declared group-disjointness rules.
///
/// A rule declares one identity dimension and the roles from which the test
/// split must be disjoint. This permits valid personalization designs where a
/// listener appears in both calibration and test, while preventing that
/// listener from leaking into population training data.
pub fn validate_split_manifest(manifest: &Value) -> Result<usize, String> {
    let root = manifest
        .as_object()
        .ok_or_else(|| "manifest root must be an object".to_owned())?;

    require_string(root.get("schema_version"), "schema_version")?;
    if root.get("schema_version").and_then(Value::as_str) != Some(SCHEMA_VERSION) {
        return Err(format!("schema_version must be {SCHEMA_VERSION}"));
    }
    require_string(root.get("benchmark_id"), "benchmark_id")?;
    require_string(root.get("partition_algorithm"), "partition_algorithm")?;
    require_string(root.get("source_revision"), "source_revision")?;
    let evaluation_regime = root
        .get("evaluation_regime")
        .and_then(Value::as_str)
        .ok_or_else(|| "evaluation_regime must be a non-empty string".to_owned())?;
    if ![
        "unseen_listener",
        "personalization",
        "unseen_product_revision",
        "fixture_transfer",
        "session_transfer",
    ]
    .contains(&evaluation_regime)
    {
        return Err("evaluation_regime is not a supported benchmark regime".to_owned());
    }

    let observations = root
        .get("observations")
        .and_then(Value::as_array)
        .ok_or_else(|| "observations must be an array".to_owned())?;
    if observations.is_empty() {
        return Err("observations must not be empty".to_owned());
    }

    let mut sample_ids = HashSet::new();
    let mut role_counts: HashMap<&str, usize> = HashMap::new();

    for (index, row) in observations.iter().enumerate() {
        let row = row
            .as_object()
            .ok_or_else(|| format!("observation {index} must be an object"))?;
        let sample_id = required_string(row.get("sample_id"), index, "sample_id")?;

        for dimension in GROUP_DIMENSIONS {
            required_string(row.get(dimension), index, dimension)?;
        }

        let role = required_string(row.get("role"), index, "role")?;
        if !ROLES.contains(&role) {
            return Err(format!(
                "observation {index} has unknown role; use train/validation/calibration/test"
            ));
        }
        if !sample_ids.insert(sample_id) {
            return Err(format!(
                "sample_id values must be globally unique (observation {index})"
            ));
        }
        *role_counts.entry(role).or_insert(0) += 1;
    }

    if role_counts.get("train").copied().unwrap_or(0) == 0 {
        return Err("at least one train observation is required".to_owned());
    }
    if role_counts.get("test").copied().unwrap_or(0) == 0 {
        return Err("at least one test observation is required".to_owned());
    }

    let rules = root
        .get("holdout_rules")
        .and_then(Value::as_array)
        .ok_or_else(|| "holdout_rules must be an array".to_owned())?;
    if rules.is_empty() {
        return Err(
            "holdout_rules must not be empty; split integrity must be explicit".to_owned(),
        );
    }

    let mut seen_dimensions = HashSet::new();
    let mut declared_roles: HashMap<&str, HashSet<&str>> = HashMap::new();
    for (rule_index, rule) in rules.iter().enumerate() {
        let rule = rule
            .as_object()
            .ok_or_else(|| format!("holdout rule {rule_index} must be an object"))?;
        let dimension = rule
            .get("dimension")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("holdout rule {rule_index} requires a dimension"))?;
        if !GROUP_DIMENSIONS.contains(&dimension) {
            return Err(format!(
                "holdout rule {rule_index} uses unsupported dimension {dimension}"
            ));
        }
        if !seen_dimensions.insert(dimension) {
            return Err(format!("duplicate holdout rule for dimension {dimension}"));
        }

        let disjoint_roles = rule
            .get("disjoint_from_roles")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                format!("holdout rule {rule_index} requires disjoint_from_roles array")
            })?;
        if disjoint_roles.is_empty() {
            return Err(format!(
                "holdout rule {rule_index} must declare at least one role"
            ));
        }

        let mut compared_roles = HashSet::new();
        for requested_role in disjoint_roles {
            let requested_role = requested_role.as_str().ok_or_else(|| {
                format!("holdout rule {rule_index} contains a non-string role")
            })?;
            if !ROLES.contains(&requested_role) || requested_role == "test" {
                return Err(format!(
                    "holdout rule {rule_index} may compare test with fit roles only"
                ));
            }
            if !compared_roles.insert(requested_role) {
                return Err(format!(
                    "holdout rule {rule_index} repeats role {requested_role}"
                ));
            }
            declared_roles
                .entry(dimension)
                .or_default()
                .insert(requested_role);
            if role_counts.get(requested_role).copied().unwrap_or(0) == 0 {
                return Err(format!(
                    "holdout rule {rule_index} compares against empty role {requested_role}"
                ));
            }

            let test_values = identity_values(observations, "test", dimension)?;
            let comparison_values = identity_values(observations, requested_role, dimension)?;
            if test_values.iter().any(|value| comparison_values.contains(value)) {
                return Err(format!(
                    "group leakage on {dimension}: test overlaps {requested_role}"
                ));
            }
        }
    }

    if evaluation_regime == "unseen_listener"
        && role_counts.get("calibration").copied().unwrap_or(0) > 0
    {
        return Err(
            "unseen_listener regime must not include calibration observations".to_owned(),
        );
    }

    let required_rules: &[(&str, &[&str])] = match evaluation_regime {
        "unseen_listener" => &[("listener_id", &["train", "validation"])],
        "personalization" => &[
            ("listener_id", &["train", "validation"]),
            ("excerpt_id", &["train", "validation", "calibration"]),
            (
                "measurement_session_id",
                &["train", "validation", "calibration"],
            ),
        ],
        "unseen_product_revision" => &[(
            "product_revision_id",
            &["train", "validation", "calibration"],
        )],
        "fixture_transfer" => &[(
            "fixture_id",
            &["train", "validation", "calibration"],
        )],
        "session_transfer" => &[(
            "measurement_session_id",
            &["train", "validation", "calibration"],
        )],
        _ => unreachable!("evaluation regime was checked above"),
    };
    for (dimension, required_roles) in required_rules {
        let declared = declared_roles.get(*dimension).ok_or_else(|| {
            format!("regime {evaluation_regime} requires a holdout rule for {dimension}")
        })?;
        for required_role in required_roles.iter().copied() {
            if !declared.contains(required_role) {
                return Err(format!(
                    "regime {evaluation_regime}: {dimension} must exclude {required_role}"
                ));
            }
        }
    }

    Ok(observations.len())
}

fn identity_values(
    observations: &[Value],
    role: &str,
    dimension: &str,
) -> Result<BTreeSet<String>, String> {
    let mut values = BTreeSet::new();
    for (index, row) in observations.iter().enumerate() {
        let row = row
            .as_object()
            .ok_or_else(|| format!("observation {index} must be an object"))?;
        if row.get("role").and_then(Value::as_str) == Some(role) {
            let identity = row
                .get(dimension)
                .and_then(Value::as_str)
                .ok_or_else(|| format!("observation {index} requires {dimension}"))?;
            values.insert(identity.to_owned());
        }
    }
    Ok(values)
}

fn require_string(value: Option<&Value>, field: &str) -> Result<(), String> {
    match value.and_then(Value::as_str) {
        Some(value) if !value.trim().is_empty() => Ok(()),
        _ => Err(format!("{field} must be a non-empty string")),
    }
}

fn required_string<'a>(
    value: Option<&'a Value>,
    row_index: usize,
    field: &str,
) -> Result<&'a str, String> {
    let value = value
        .and_then(Value::as_str)
        .ok_or_else(|| format!("observation {row_index} requires string {field}"))?;
    if value.trim().is_empty() {
        return Err(format!("observation {row_index} requires non-empty {field}"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::validate_split_manifest;
    use serde_json::Value;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../schemas/examples/hp-pref-001-split-manifest.json"
        ))
        .expect("checked-in split fixture is valid JSON")
    }

    #[test]
    fn accepts_declared_personalization_split() {
        let manifest = fixture();
        assert_eq!(validate_split_manifest(&manifest), Ok(7));
    }

    #[test]
    fn rejects_reused_held_out_excerpt() {
        let mut manifest = fixture();
        manifest["observations"][6]["excerpt_id"] = serde_json::json!("excerpt-train-1");
        let error = validate_split_manifest(&manifest).expect_err("excerpt leakage must fail");
        assert!(error.contains("group leakage on excerpt_id"));
    }

    #[test]
    fn rejects_reused_test_session() {
        let mut manifest = fixture();
        manifest["observations"][5]["measurement_session_id"] =
            serde_json::json!("session-calibration-01");
        let error = validate_split_manifest(&manifest).expect_err("session leakage must fail");
        assert!(error.contains("group leakage on measurement_session_id"));
    }

    #[test]
    fn unseen_listener_regime_rejects_calibration_rows() {
        let mut manifest = fixture();
        manifest["evaluation_regime"] = serde_json::json!("unseen_listener");
        let error = validate_split_manifest(&manifest)
            .expect_err("unseen-listener benchmark must not use calibration");
        assert!(error.contains("must not include calibration observations"));
    }

    #[test]
    fn rejects_duplicate_sample_ids() {
        let mut manifest = fixture();
        let duplicate = manifest["observations"][0]["sample_id"].clone();
        manifest["observations"][1]["sample_id"] = duplicate;
        let error = validate_split_manifest(&manifest).expect_err("duplicate IDs must fail");
        assert!(error.contains("globally unique"));
    }

    #[test]
    fn rejects_empty_holdout_policy() {
        let mut manifest = fixture();
        manifest["holdout_rules"] = serde_json::json!([]);
        let error = validate_split_manifest(&manifest).expect_err("policy cannot be implicit");
        assert!(error.contains("must not be empty"));
    }

    #[test]
    fn rejects_unknown_roles() {
        let mut manifest = fixture();
        manifest["observations"][0]["role"] = serde_json::json!("test-ish");
        let error = validate_split_manifest(&manifest).expect_err("unknown role must fail");
        assert!(error.contains("unknown role"));
    }

    #[test]
    fn rejects_unknown_evaluation_regime() {
        let mut manifest = fixture();
        manifest["evaluation_regime"] = serde_json::json!("random-row-split");
        let error = validate_split_manifest(&manifest).expect_err("unknown regime must fail");
        assert!(error.contains("evaluation_regime"));
    }

    #[test]
    fn regime_requires_its_declared_holdout_dimensions() {
        let mut manifest = fixture();
        manifest["holdout_rules"] = serde_json::json!([
            { "dimension": "listener_id", "disjoint_from_roles": ["train", "validation"] }
        ]);
        let error = validate_split_manifest(&manifest)
            .expect_err("required dimensions cannot be omitted");
        assert!(error.contains("requires a holdout rule for excerpt_id"));
    }

    #[test]
    fn rejects_product_revision_leakage_in_product_transfer_regime() {
        let mut manifest = fixture();
        manifest["evaluation_regime"] = serde_json::json!("unseen_product_revision");
        for row in manifest["observations"].as_array_mut().expect("rows") {
            if row.get("role").and_then(Value::as_str) == Some("test") {
                row["product_revision_id"] = serde_json::json!("headphone-h1-rev2");
            }
        }
        manifest["holdout_rules"].as_array_mut().expect("rules").push(serde_json::json!({
            "dimension": "product_revision_id",
            "disjoint_from_roles": ["train", "validation", "calibration"]
        }));
        assert!(validate_split_manifest(&manifest).is_ok(), "clean product holdout should pass");
        manifest["observations"][5]["product_revision_id"] = serde_json::json!("headphone-h1-rev1");
        let error = validate_split_manifest(&manifest).expect_err("product leakage must fail");
        assert!(error.contains("group leakage on product_revision_id"));
    }

    #[test]
    fn rejects_fixture_leakage_in_fixture_transfer_regime() {
        let mut manifest = fixture();
        manifest["evaluation_regime"] = serde_json::json!("fixture_transfer");
        for row in manifest["observations"].as_array_mut().expect("rows") {
            if row["role"] == "test" {
                row["fixture_id"] = serde_json::json!("fixture-b");
            }
        }
        manifest["holdout_rules"].as_array_mut().expect("rules").push(serde_json::json!({
            "dimension": "fixture_id",
            "disjoint_from_roles": ["train", "validation", "calibration"]
        }));
        assert!(validate_split_manifest(&manifest).is_ok(), "clean fixture transfer should pass");
        manifest["observations"][5]["fixture_id"] = serde_json::json!("fixture-a");
        let error = validate_split_manifest(&manifest).expect_err("fixture leakage must fail");
        assert!(error.contains("group leakage on fixture_id"));
    }
}
