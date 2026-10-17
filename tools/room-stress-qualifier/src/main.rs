// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent schema + semantic checks for the room-imagination stress corpus.
//!
//! This tool qualifies the manifest structure only. It does not execute room
//! geometry, simulation, accessibility, safety, building-code, or human-outcome
//! tests. Those require separate qualified providers and evidence.

use jsonschema::draft202012;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::Path;

const EXPECTED_SCHEMA_VERSION: &str = "room-imagination-stress-benchmark-v0.1";
const EXPECTED_CASE_COUNT: usize = 30;

const EMBEDDED_BENCHMARK: &str =
    include_str!("../../../docs/research/spatial-design/ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.json");
const EMBEDDED_SCHEMA: &str =
    include_str!("../../../docs/research/spatial-design/ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.schema.json");

#[derive(Debug, PartialEq, Eq)]
struct ValidationSummary {
    case_count: usize,
    family_count: usize,
    risk_counts: BTreeMap<String, usize>,
}

impl ValidationSummary {
    fn print(&self) {
        println!("ROOM IMAGINATION STRESS MANIFEST: VALID");
        println!("schema_version: {EXPECTED_SCHEMA_VERSION}");
        println!("cases: {}", self.case_count);
        println!("distinct families: {}", self.family_count);
        println!("risk counts:");
        for (risk, count) in &self.risk_counts {
            println!("  {risk}: {count}");
        }
        println!();
        println!("Claim ceiling: manifest/schema integrity only; no geometry or building qualification.");
    }
}

fn parse_json(label: &str, input: &str) -> Result<Value, String> {
    serde_json::from_str(input).map_err(|error| format!("{label} is not valid JSON: {error}"))
}

fn validate_documents(benchmark_text: &str, schema_text: &str) -> Result<ValidationSummary, String> {
    let benchmark = parse_json("benchmark", benchmark_text)?;
    let schema = parse_json("schema", schema_text)?;
    validate_values(benchmark, schema)
}

fn validate_values(benchmark: Value, schema: Value) -> Result<ValidationSummary, String> {
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema is invalid under the JSON Schema Draft 2020-12 meta-schema".into());
    }

    // The schema uses only local references. Remote/file reference resolution is
    // not enabled in Cargo.toml, so validation cannot fetch arbitrary resources.
    let validator = draft202012::new(&schema)
        .map_err(|error| format!("could not compile Draft 2020-12 schema: {error}"))?;

    let errors: Vec<String> = validator
        .iter_errors(&benchmark)
        .take(20)
        .map(|error| format!("{} at {}", error, error.instance_path()))
        .collect();
    if !errors.is_empty() {
        return Err(format!(
            "benchmark failed JSON Schema validation (showing at most 20 errors):\n{}",
            errors.join("\n")
        ));
    }

    let version = benchmark
        .get("schema_version")
        .and_then(Value::as_str)
        .ok_or_else(|| "schema_version must be a string".to_string())?;
    if version != EXPECTED_SCHEMA_VERSION {
        return Err(format!("unsupported schema_version: {version}"));
    }

    let cases = benchmark
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| "cases must be an array".to_string())?;
    if cases.len() != EXPECTED_CASE_COUNT {
        return Err(format!(
            "expected exactly {EXPECTED_CASE_COUNT} cases, found {}",
            cases.len()
        ));
    }

    let declared_count = benchmark
        .get("count")
        .and_then(Value::as_u64)
        .ok_or_else(|| "count must be a non-negative integer".to_string())?;
    if declared_count != cases.len() as u64 {
        return Err(format!(
            "declared count {declared_count} does not match {} case records",
            cases.len()
        ));
    }

    let mut ids = BTreeSet::new();
    let mut families = BTreeSet::new();
    let mut risk_counts = BTreeMap::<String, usize>::new();

    for (index, case) in cases.iter().enumerate() {
        let case_label = format!("cases[{index}]");
        let id = case
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{case_label}.id must be a string"))?;

        // Schema validation checks the lexical pattern; this additional oracle
        // requires the initial sealed corpus to enumerate exactly 001..030.
        let expected_id = format!("ROOM-STRESS-{:03}", index + 1);
        if id != expected_id {
            return Err(format!(
                "{case_label}.id must be {expected_id} at this position, found {id}"
            ));
        }
        if !ids.insert(id.to_owned()) {
            return Err(format!("duplicate case id: {id}"));
        }

        let family = case
            .get("family")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{case_label}.family must be a string"))?;
        families.insert(family.to_owned());

        let risk = case
            .get("risk")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{case_label}.risk must be a string"))?;
        *risk_counts.entry(risk.to_owned()).or_default() += 1;

        for field in ["known_facts", "unknowns", "required_behaviors", "blocking_failures"] {
            let values = case
                .get(field)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{case_label}.{field} must be an array"))?;
            if values.is_empty() {
                return Err(format!("{case_label}.{field} must not be empty"));
            }
            if values.iter().any(|value| {
                value.as_str().map(str::trim).is_none_or(str::is_empty)
            }) {
                return Err(format!(
                    "{case_label}.{field} must contain only non-empty strings"
                ));
            }
        }

        let intervention = case
            .get("intervention")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{case_label}.intervention must be a string"))?;
        if intervention.trim().is_empty() {
            return Err(format!("{case_label}.intervention must not be empty"));
        }
    }

    Ok(ValidationSummary {
        case_count: cases.len(),
        family_count: families.len(),
        risk_counts,
    })
}

fn run(args: &[String]) -> Result<(), String> {
    let (benchmark_path, schema_path) = match args {
        [] => (
            "docs/research/spatial-design/ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.json",
            "docs/research/spatial-design/ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.schema.json",
        ),
        [benchmark, schema] => (benchmark.as_str(), schema.as_str()),
        _ => {
            return Err(
                "usage: room-stress-qualifier [BENCHMARK.json SCHEMA.json]".to_string(),
            );
        }
    };

    let benchmark_text = fs::read_to_string(Path::new(benchmark_path))
        .map_err(|error| format!("cannot read benchmark {benchmark_path}: {error}"))?;
    let schema_text = fs::read_to_string(Path::new(schema_path))
        .map_err(|error| format!("cannot read schema {schema_path}: {error}"))?;

    validate_documents(&benchmark_text, &schema_text)?.print();
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if let Err(error) = run(&args) {
        eprintln!("ROOM IMAGINATION STRESS MANIFEST: INVALID");
        eprintln!("{error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_values() -> (Value, Value) {
        let benchmark: Value =
            serde_json::from_str(EMBEDDED_BENCHMARK).expect("benchmark fixture parses");
        let schema: Value =
            serde_json::from_str(EMBEDDED_SCHEMA).expect("schema fixture parses");
        (benchmark, schema)
    }

    #[test]
    fn committed_manifest_passes_schema_and_semantic_checks() {
        let summary = validate_documents(EMBEDDED_BENCHMARK, EMBEDDED_SCHEMA)
            .expect("committed manifest should validate");
        assert_eq!(summary.case_count, EXPECTED_CASE_COUNT);
        assert_eq!(summary.family_count, EXPECTED_CASE_COUNT);
        assert_eq!(summary.risk_counts.values().sum::<usize>(), EXPECTED_CASE_COUNT);
    }

    #[test]
    fn malformed_json_is_rejected() {
        let error = validate_documents("{not json", EMBEDDED_SCHEMA).unwrap_err();
        assert!(error.contains("not valid JSON"));
    }

    #[test]
    fn duplicate_case_ids_are_rejected() {
        let (mut benchmark, schema) = fixture_values();
        let cases = benchmark["cases"].as_array_mut().unwrap();
        cases[1]["id"] = cases[0]["id"].clone();
        let error = validate_values(benchmark, schema).unwrap_err();
        assert!(error.contains("must be ROOM-STRESS-002") || error.contains("duplicate case id"));
    }

    #[test]
    fn missing_required_field_is_rejected() {
        let (mut benchmark, schema) = fixture_values();
        benchmark["cases"][0].as_object_mut().unwrap().remove("blocking_failures");
        let error = validate_values(benchmark, schema).unwrap_err();
        assert!(error.contains("blocking_failures"));
    }

    #[test]
    fn unexpected_top_level_field_is_rejected() {
        let (mut benchmark, schema) = fixture_values();
        benchmark["unexpected"] = Value::Bool(true);
        assert!(validate_values(benchmark, schema).is_err());
    }

    #[test]
    fn unexpected_case_field_is_rejected() {
        let (mut benchmark, schema) = fixture_values();
        benchmark["cases"][0]["unreviewed_override"] = Value::Bool(true);
        assert!(validate_values(benchmark, schema).is_err());
    }

    #[test]
    fn unsupported_risk_label_is_rejected() {
        let (mut benchmark, schema) = fixture_values();
        benchmark["cases"][0]["risk"] = Value::String("low-risk".into());
        assert!(validate_values(benchmark, schema).is_err());
    }

    #[test]
    fn missing_known_fact_is_rejected() {
        let (mut benchmark, schema) = fixture_values();
        benchmark["cases"][0]["known_facts"] = Value::Array(vec![]);
        assert!(validate_values(benchmark, schema).is_err());
    }

    #[test]
    fn nonsequential_case_id_is_rejected() {
        let (mut benchmark, schema) = fixture_values();
        benchmark["cases"][12]["id"] = Value::String("ROOM-STRESS-999".into());
        let error = validate_values(benchmark, schema).unwrap_err();
        assert!(error.contains("must be ROOM-STRESS-013"));
    }

    #[test]
    fn wrong_declared_count_is_rejected() {
        let (mut benchmark, schema) = fixture_values();
        benchmark["count"] = Value::from(29);
        assert!(validate_values(benchmark, schema).is_err());
    }

    #[test]
    fn invalid_schema_is_rejected_before_instance_validation() {
        let (benchmark, _) = fixture_values();
        let invalid_schema = serde_json::json!({"type": "not-a-json-type"});
        let error = validate_values(benchmark, invalid_schema).unwrap_err();
        assert!(error.contains("Draft 2020-12 meta-schema"));
    }
}
