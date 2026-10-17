// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Structural audit for the public language/code capability smoke corpus.
//!
//! This checks fixture integrity only. It does not run a model, compile the
//! per-case Rust snippets, validate the JSON Schema, or produce a benchmark
//! score. A structural-audit result must never be reported as task success.

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::process;

const CORPUS: &str =
    include_str!("../fixtures/language-code-capability/smoke-v1.json");

const EXPECTED_LANES: [(&str, usize); 4] = [
    ("language", 8),
    ("rust_synthesis", 8),
    ("repair", 8),
    ("verifier_integrity", 8),
];

#[derive(Debug, PartialEq, Eq)]
struct CatalogSummary {
    total_cases: usize,
    lane_counts: BTreeMap<String, usize>,
}

fn nonempty_string<'a>(
    value: &'a Value,
    key: &str,
    context: &str,
) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|item| !item.trim().is_empty())
        .ok_or_else(|| format!("{context}: missing or empty string field `{key}`"))
}

fn require_object<'a>(
    value: &'a Value,
    key: &str,
    context: &str,
) -> Result<&'a Value, String> {
    value
        .get(key)
        .filter(|item| item.is_object())
        .ok_or_else(|| format!("{context}: missing object field `{key}`"))
}

fn validate_catalog(catalog: &Value) -> Result<CatalogSummary, String> {
    if !catalog.is_object() {
        return Err("catalog root must be an object".to_owned());
    }

    for (key, expected) in [
        ("schema_version", "1.0.0"),
        ("benchmark_id", "symthaea-language-code-capability-smoke"),
        ("dataset_version", "smoke-v1"),
        ("split", "public_smoke"),
    ] {
        let actual = nonempty_string(catalog, key, "catalog")?;
        if actual != expected {
            return Err(format!(
                "catalog: field `{key}` must be `{expected}`, got `{actual}`"
            ));
        }
    }

    if catalog.get("synthetic").and_then(Value::as_bool) != Some(true) {
        return Err("catalog must be explicitly labelled synthetic=true".to_owned());
    }

    if catalog
        .get("expected_case_count")
        .and_then(Value::as_u64)
        != Some(32)
    {
        return Err("expected_case_count must be exactly 32 for smoke-v1".to_owned());
    }

    nonempty_string(catalog, "created_date", "catalog")?;
    nonempty_string(catalog, "intended_use", "catalog")?;
    nonempty_string(catalog, "claim_ceiling", "catalog")?;

    let cases = catalog
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| "catalog: `cases` must be an array".to_owned())?;
    if cases.len() != 32 {
        return Err(format!(
            "catalog: expected 32 cases, found {}",
            cases.len()
        ));
    }

    let mut ids = BTreeSet::new();
    let mut lane_counts = BTreeMap::<String, usize>::new();

    for case in cases {
        let id = nonempty_string(case, "id", "case")?;
        if !ids.insert(id.to_owned()) {
            return Err(format!("duplicate case id: {id}"));
        }

        let lane = nonempty_string(case, "lane", id)?;
        if !EXPECTED_LANES.iter().any(|(expected, _)| *expected == lane) {
            return Err(format!("{id}: unknown lane `{lane}`"));
        }
        *lane_counts.entry(lane.to_owned()).or_default() += 1;

        nonempty_string(case, "kind", id)?;
        nonempty_string(case, "prompt", id)?;
        nonempty_string(case, "claim_ceiling", id)?;

        let expected = require_object(case, "expected", id)?;
        if expected.as_object().is_none_or(|fields| fields.is_empty()) {
            return Err(format!("{id}: `expected` must not be empty"));
        }

        let tags = case
            .get("tags")
            .and_then(Value::as_array)
            .filter(|items| !items.is_empty())
            .ok_or_else(|| format!("{id}: `tags` must be a non-empty array"))?;
        for tag in tags {
            if tag.as_str().is_none_or(|value| value.trim().is_empty()) {
                return Err(format!("{id}: tags must contain non-empty strings"));
            }
        }

        match lane {
            "language" => {
                nonempty_string(case, "semantic_contract", id)?;
            }
            "rust_synthesis" => {
                nonempty_string(case, "signature", id)?;
                nonempty_string(case, "test_source", id)?;
            }
            "repair" => {
                nonempty_string(case, "signature", id)?;
                nonempty_string(case, "buggy_source", id)?;
                nonempty_string(case, "repair_contract", id)?;
                nonempty_string(case, "test_source", id)?;
            }
            "verifier_integrity" => {
                require_object(case, "receipt_fixture", id)?;
                let outcome = nonempty_string(case, "expected_outcome", id)?;
                if outcome != "reject" {
                    return Err(format!(
                        "{id}: smoke-v1 verifier negative controls must expect reject"
                    ));
                }
                nonempty_string(case, "expected_reason", id)?;
            }
            _ => unreachable!("lane was checked against EXPECTED_LANES"),
        }
    }

    for (lane, expected_count) in EXPECTED_LANES {
        let actual = lane_counts.get(lane).copied().unwrap_or(0);
        if actual != expected_count {
            return Err(format!(
                "lane `{lane}`: expected {expected_count} cases, found {actual}"
            ));
        }
    }

    Ok(CatalogSummary {
        total_cases: cases.len(),
        lane_counts,
    })
}

fn run_audit() -> Result<(), String> {
    let catalog: Value = serde_json::from_str(CORPUS)
        .map_err(|error| format!("failed to parse embedded corpus JSON: {error}"))?;
    let summary = validate_catalog(&catalog)?;
    let report = json!({
        "status": "CATALOG_STRUCTURE_VALID",
        "benchmark_evaluated": false,
        "synthetic": true,
        "total_cases": summary.total_cases,
        "lane_counts": summary.lane_counts,
        "claim_ceiling": "Fixture structure only; no model task was executed."
    });
    let rendered = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed to serialize audit report: {error}"))?;
    println!("{rendered}");
    Ok(())
}

fn main() {
    if let Err(error) = run_audit() {
        eprintln!("language-code-capability catalog audit failed: {error}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(CORPUS).expect("checked-in smoke corpus must be valid JSON")
    }

    #[test]
    fn checked_in_catalog_has_expected_shape() {
        let summary = validate_catalog(&fixture()).expect("catalog structure");
        assert_eq!(summary.total_cases, 32);
        for (lane, count) in EXPECTED_LANES {
            assert_eq!(summary.lane_counts.get(lane), Some(&count));
        }
    }

    #[test]
    fn duplicate_case_ids_are_rejected() {
        let mut catalog = fixture();
        let duplicate = catalog["cases"][0]["id"].clone();
        catalog["cases"][1]["id"] = duplicate;
        assert!(validate_catalog(&catalog)
            .expect_err("duplicate ID must fail")
            .contains("duplicate case id"));
    }

    #[test]
    fn unlabelled_synthetic_data_is_rejected() {
        let mut catalog = fixture();
        catalog["synthetic"] = json!(false);
        assert!(validate_catalog(&catalog)
            .expect_err("unlabelled corpus must fail")
            .contains("synthetic=true"));
    }

    #[test]
    fn missing_synthesis_test_source_is_rejected() {
        let mut catalog = fixture();
        catalog["cases"][8].as_object_mut().unwrap().remove("test_source");
        assert!(validate_catalog(&catalog)
            .expect_err("synthesis case without oracle must fail")
            .contains("test_source"));
    }

    #[test]
    fn missing_repair_source_is_rejected() {
        let mut catalog = fixture();
        catalog["cases"][16].as_object_mut().unwrap().remove("buggy_source");
        assert!(validate_catalog(&catalog)
            .expect_err("repair case without broken source must fail")
            .contains("buggy_source"));
    }

    #[test]
    fn verifier_control_cannot_expect_acceptance() {
        let mut catalog = fixture();
        catalog["cases"][24]["expected_outcome"] = json!("accept");
        assert!(validate_catalog(&catalog)
            .expect_err("negative control cannot pass")
            .contains("must expect reject"));
    }

    #[test]
    fn lane_count_drift_is_rejected() {
        let mut catalog = fixture();
        catalog["cases"][0]["lane"] = json!("rust_synthesis");
        assert!(validate_catalog(&catalog)
            .expect_err("lane counts must remain balanced")
            .contains("lane `language`"));
    }

    #[test]
    fn empty_expected_contract_is_rejected() {
        let mut catalog = fixture();
        catalog["cases"][0]["expected"] = json!({});
        assert!(validate_catalog(&catalog)
            .expect_err("empty expectation must fail")
            .contains("expected"));
    }
}
