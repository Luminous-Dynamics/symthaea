// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic structural auditor for Broca counterfactual benchmark fixtures.
//!
//! This executable validates that evaluator-only oracle labels are kept outside
//! reader input and that every five-arm counterfactual has the declared causal
//! shape: one decisive-field flip, one irrelevant-field change, an exact repair,
//! and a matched irrelevant repair. It performs no model calls and computes no
//! answer-quality score. It is a benchmark-integrity gate, not a benchmark run.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_FIXTURES: &[&str] = &[
    "proof-budget-fixtures-v1.json",
    "bitemporal-query-fixtures-v1.json",
    "decision-use-validity-fixtures-v1.json",
    "timlmembench-governance-fixtures-v1.json",
];

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn string_field<'a>(value: &'a Value, field: &str, context: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{context}: missing string field {field:?}"))
}

fn path_segments(path: &str) -> Result<Vec<&str>, String> {
    let segments: Vec<&str> = path.split('.').collect();
    if path.trim().is_empty() || segments.iter().any(|segment| segment.is_empty()) {
        return Err(format!("invalid dotted JSON path {path:?}"));
    }
    Ok(segments)
}

fn get_path<'a>(root: &'a Value, path: &str) -> Result<&'a Value, String> {
    let segments = path_segments(path)?;
    let mut current = root;
    for segment in segments {
        current = match current {
            Value::Object(object) => object.get(segment),
            Value::Array(array) => segment.parse::<usize>().ok().and_then(|index| array.get(index)),
            _ => None,
        }
        .ok_or_else(|| format!("JSON path {path:?} does not exist"))?;
    }
    Ok(current)
}

fn remove_path(root: &mut Value, path: &str) -> Result<(), String> {
    fn remove_at(current: &mut Value, segments: &[&str], full_path: &str) -> Result<(), String> {
        if segments.len() == 1 {
            match current {
                Value::Object(object) => {
                    if object.remove(segments[0]).is_none() {
                        return Err(format!("JSON path {full_path:?} does not exist"));
                    }
                    return Ok(());
                }
                Value::Array(array) => {
                    let index = segments[0]
                        .parse::<usize>()
                        .map_err(|_| format!("array path segment {:?} is not an index", segments[0]))?;
                    if index >= array.len() {
                        return Err(format!("JSON path {full_path:?} does not exist"));
                    }
                    array.remove(index);
                    return Ok(());
                }
                _ => return Err(format!("JSON path {full_path:?} traverses a scalar")),
            }
        }

        let next = match current {
            Value::Object(object) => object.get_mut(segments[0]),
            Value::Array(array) => {
                let index = segments[0]
                    .parse::<usize>()
                    .map_err(|_| format!("array path segment {:?} is not an index", segments[0]))?;
                array.get_mut(index)
            }
            _ => None,
        }
        .ok_or_else(|| format!("JSON path {full_path:?} does not exist"))?;
        remove_at(next, &segments[1..], full_path)
    }

    let segments = path_segments(path)?;
    remove_at(root, &segments, path)
}

fn path_changed(left: &Value, right: &Value, path: &str) -> Result<bool, String> {
    Ok(get_path(left, path)? != get_path(right, path)?)
}

/// True when the two JSON documents are identical after removing one declared
/// intervention field from each. This enforces a one-field counterfactual.
fn equal_except_path(left: &Value, right: &Value, path: &str) -> Result<bool, String> {
    let mut left = left.clone();
    let mut right = right.clone();
    remove_path(&mut left, path)?;
    remove_path(&mut right, path)?;
    Ok(left == right)
}

fn contains_key_recursive(value: &Value, forbidden: &str) -> bool {
    match value {
        Value::Object(object) => {
            object.contains_key(forbidden)
                || object.values().any(|child| contains_key_recursive(child, forbidden))
        }
        Value::Array(array) => array.iter().any(|child| contains_key_recursive(child, forbidden)),
        _ => false,
    }
}

fn variants_by_intervention<'a>(
    variants: &'a [Value],
    case_id: &str,
) -> Result<BTreeMap<&'a str, &'a Value>, String> {
    let mut by_kind = BTreeMap::new();
    let mut seen_ids = BTreeSet::new();
    for variant in variants {
        let context = format!("{case_id} variant");
        let variant_id = string_field(variant, "variant_id", &context)?;
        let intervention = string_field(variant, "intervention", &context)?;
        if !seen_ids.insert(variant_id) {
            return Err(format!("{case_id}: duplicate variant_id {variant_id:?}"));
        }
        if by_kind.insert(intervention, variant).is_some() {
            return Err(format!("{case_id}: duplicate intervention {intervention:?}"));
        }
        if variant.get("reader_input").is_none() || variant.get("oracle").is_none() {
            return Err(format!("{case_id}/{variant_id}: reader_input and oracle must be separate fields"));
        }
        if contains_key_recursive(&variant["reader_input"], "oracle") {
            return Err(format!("{case_id}/{variant_id}: evaluator oracle leaked into reader_input"));
        }
    }

    let required = [
        "baseline",
        "decisive_flip",
        "irrelevant_change",
        "exact_repair",
        "matched_irrelevant_repair",
    ];
    if by_kind.len() != required.len() {
        return Err(format!(
            "{case_id}: expected exactly {} intervention arms, found {}",
            required.len(),
            by_kind.len()
        ));
    }
    for intervention in required {
        if !by_kind.contains_key(intervention) {
            return Err(format!("{case_id}: missing intervention arm {intervention:?}"));
        }
    }
    Ok(by_kind)
}

fn audit_case(case: &Value) -> Result<Value, String> {
    let case_id = string_field(case, "case_id", "case")?;
    let contract = case
        .get("intervention_contract")
        .ok_or_else(|| format!("{case_id}: missing intervention_contract"))?;
    let decisive_path = string_field(contract, "decisive_path", case_id)?;
    let invariant_path = string_field(contract, "invariant_path", case_id)?;
    if decisive_path == invariant_path {
        return Err(format!("{case_id}: decisive_path and invariant_path must differ"));
    }

    let variants = case
        .get("variants")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{case_id}: variants must be an array"))?;
    let arms = variants_by_intervention(variants, case_id)?;

    let baseline = &arms["baseline"]["reader_input"];
    let decisive = &arms["decisive_flip"]["reader_input"];
    let invariant = &arms["irrelevant_change"]["reader_input"];
    let repair = &arms["exact_repair"]["reader_input"];
    let false_repair = &arms["matched_irrelevant_repair"]["reader_input"];

    if !path_changed(baseline, decisive, decisive_path)? {
        return Err(format!("{case_id}: decisive_flip did not change {decisive_path:?}"));
    }
    if !equal_except_path(baseline, decisive, decisive_path)? {
        return Err(format!("{case_id}: decisive_flip changes more than {decisive_path:?}"));
    }

    if !path_changed(baseline, invariant, invariant_path)? {
        return Err(format!("{case_id}: irrelevant_change did not change {invariant_path:?}"));
    }
    if !equal_except_path(baseline, invariant, invariant_path)? {
        return Err(format!("{case_id}: irrelevant_change changes more than {invariant_path:?}"));
    }
    if arms["baseline"]["oracle"] != arms["irrelevant_change"]["oracle"] {
        return Err(format!("{case_id}: irrelevant presentation change altered oracle outcome"));
    }

    if !equal_except_path(baseline, repair, decisive_path)? {
        return Err(format!("{case_id}: exact_repair changes fields beyond {decisive_path:?}"));
    }
    if !equal_except_path(decisive, repair, decisive_path)? {
        return Err(format!("{case_id}: exact_repair is not aligned to either baseline or decisive-flip control"));
    }
    if arms["exact_repair"]["oracle"] != arms["baseline"]["oracle"]
        && arms["exact_repair"]["oracle"] != arms["decisive_flip"]["oracle"]
    {
        return Err(format!("{case_id}: exact_repair oracle is not aligned to either causal control"));
    }

    let false_repair_matches_baseline =
        path_changed(baseline, false_repair, invariant_path)?
            && equal_except_path(baseline, false_repair, invariant_path)?
            && arms["matched_irrelevant_repair"]["oracle"] == arms["baseline"]["oracle"];
    let false_repair_matches_flip =
        path_changed(decisive, false_repair, invariant_path)?
            && equal_except_path(decisive, false_repair, invariant_path)?
            && arms["matched_irrelevant_repair"]["oracle"] == arms["decisive_flip"]["oracle"];
    if !false_repair_matches_baseline && !false_repair_matches_flip {
        return Err(format!(
            "{case_id}: matched_irrelevant_repair must preserve one causal arm's oracle while changing only {invariant_path:?}"
        ));
    }

    if arms["baseline"]["oracle"] == arms["decisive_flip"]["oracle"] {
        return Err(format!("{case_id}: decisive intervention does not change the evaluator oracle"));
    }

    Ok(json!({
        "case_id": case_id,
        "variant_count": variants.len(),
        "decisive_path": decisive_path,
        "invariant_path": invariant_path,
        "reader_oracle_separation": true,
        "single_field_flip": true,
        "single_field_invariance_control": true,
        "repair_control": true,
        "matched_irrelevant_repair_control": true
    }))
}

fn audit_fixture(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("{}: cannot read fixture: {error}", path.display()))?;
    let raw: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{}: invalid JSON: {error}", path.display()))?;
    let schema_version = string_field(&raw, "schema_version", &path.display().to_string())?;
    let cases = raw
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{}: cases must be an array", path.display()))?;
    if cases.is_empty() {
        return Err(format!("{}: cases array must not be empty", path.display()));
    }

    let mut case_ids = BTreeSet::new();
    let mut audits = Vec::with_capacity(cases.len());
    let mut variant_count = 0usize;
    for case in cases {
        let case_id = string_field(case, "case_id", "case")?;
        if !case_ids.insert(case_id) {
            return Err(format!("{}: duplicate case_id {case_id:?}", path.display()));
        }
        let audit = audit_case(case)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        variant_count += audit["variant_count"]
            .as_u64()
            .ok_or_else(|| format!("{case_id}: audit variant count is invalid"))? as usize;
        audits.push(audit);
    }

    Ok(json!({
        "file": path.display().to_string(),
        "schema_version": schema_version,
        "sha256": sha256(&bytes),
        "case_count": cases.len(),
        "variant_count": variant_count,
        "counterfactual_audit": audits
    }))
}

fn default_fixture_paths() -> Result<Vec<PathBuf>, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../research/broca-bitemporal-evaluation-v1");
    Ok(DEFAULT_FIXTURES.iter().map(|name| root.join(name)).collect())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let paths: Vec<PathBuf> = if args.is_empty() {
        default_fixture_paths()?
    } else {
        args.into_iter().map(PathBuf::from).collect()
    };

    let mut reports = Vec::with_capacity(paths.len());
    let mut total_cases = 0usize;
    let mut total_variants = 0usize;
    for path in &paths {
        let report = audit_fixture(path)?;
        total_cases += report["case_count"].as_u64().unwrap_or(0) as usize;
        total_variants += report["variant_count"].as_u64().unwrap_or(0) as usize;
        reports.push(report);
    }

    let output = json!({
        "auditor": "broca-memory-bench-audit-v1",
        "status": "fixture_integrity_pass",
        "files": reports.len(),
        "cases": total_cases,
        "variants": total_variants,
        "files_audited": reports
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| format!("cannot serialize audit report: {error}"))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("broca-memory-bench-audit: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_case() -> Value {
        json!({
            "case_id": "D01",
            "intervention_contract": {
                "decisive_path": "state.value",
                "invariant_path": "presentation.label"
            },
            "variants": [
                {
                    "variant_id": "D01-base",
                    "intervention": "baseline",
                    "reader_input": {"state":{"value":"old","other":1},"presentation":{"label":"normal"}},
                    "oracle": {"status":"old"}
                },
                {
                    "variant_id": "D01-flip",
                    "intervention": "decisive_flip",
                    "reader_input": {"state":{"value":"new","other":1},"presentation":{"label":"normal"}},
                    "oracle": {"status":"new"}
                },
                {
                    "variant_id": "D01-invariant",
                    "intervention": "irrelevant_change",
                    "reader_input": {"state":{"value":"old","other":1},"presentation":{"label":"compact"}},
                    "oracle": {"status":"old"}
                },
                {
                    "variant_id": "D01-repair",
                    "intervention": "exact_repair",
                    "reader_input": {"state":{"value":"new","other":1},"presentation":{"label":"normal"}},
                    "oracle": {"status":"new"}
                },
                {
                    "variant_id": "D01-false-repair",
                    "intervention": "matched_irrelevant_repair",
                    "reader_input": {"state":{"value":"old","other":1},"presentation":{"label":"compact"}},
                    "oracle": {"status":"old"}
                }
            ]
        })
    }

    #[test]
    fn detects_a_valid_single_field_counterfactual_family() {
        assert!(audit_case(&valid_case()).is_ok());
    }

    #[test]
    fn rejects_oracle_leakage_into_reader_input() {
        let mut case = valid_case();
        case["variants"][0]["reader_input"]["oracle"] = json!({"gold":"old"});
        let error = audit_case(&case).unwrap_err();
        assert!(error.contains("oracle leaked"));
    }

    #[test]
    fn rejects_a_decisive_flip_that_changes_an_unrelated_field() {
        let mut case = valid_case();
        case["variants"][1]["reader_input"]["state"]["other"] = json!(2);
        let error = audit_case(&case).unwrap_err();
        assert!(error.contains("changes more than"));
    }

    #[test]
    fn rejects_an_irrelevant_change_that_changes_the_gold_oracle() {
        let mut case = valid_case();
        case["variants"][2]["oracle"]["status"] = json!("new");
        let error = audit_case(&case).unwrap_err();
        assert!(error.contains("irrelevant presentation change altered oracle"));
    }

    #[test]
    fn nested_json_path_removal_is_exact() {
        let mut value = json!({"a":{"b":1,"c":2},"label":"test"});
        remove_path(&mut value, "a.b").unwrap();
        assert_eq!(value, json!({"a":{"c":2},"label":"test"}));
    }
}
