// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent schema + semantic checks for the room-imagination stress corpus.
//!
//! This tool qualifies the manifest structure only. It does not execute room
//! geometry, simulation, accessibility, safety, building-code, or human-outcome
//! tests. Those require separate qualified providers and evidence.

use jsonschema::draft202012;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Component, Path};

const EXPECTED_SCHEMA_VERSION: &str = "room-imagination-stress-benchmark-v0.1";
const EXPECTED_CASE_COUNT: usize = 30;
const MAX_EVIDENCE_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;

const EMBEDDED_BENCHMARK: &str = include_str!(
    "../../../docs/research/spatial-design/ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.json"
);
const EMBEDDED_SCHEMA: &str = include_str!(
    "../../../docs/research/spatial-design/ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.schema.json"
);
const EMBEDDED_RUN_REPORT_SCHEMA: &str = include_str!(
    "../../../docs/research/spatial-design/ROOM_IMAGINATION_STRESS_RUN_REPORT_V0_1.schema.json"
);
const EMBEDDED_CARGO_MANIFEST: &str = include_str!("../Cargo.toml");
const EMBEDDED_VALIDATOR_SOURCE: &str = include_str!("main.rs");
const CLAIM_CEILING: &str = concat!(
    "Manifest/schema and semantic integrity only. This receipt does not qualify geometry, ",
    "simulations, accessibility, comfort, safety, buildability, code compliance, ",
    "professional approval, or human outcomes."
);

#[derive(Debug, PartialEq, Eq)]
struct ValidationSummary {
    case_count: usize,
    family_count: usize,
    risk_counts: BTreeMap<String, usize>,
}

#[derive(Debug, PartialEq, Eq)]
struct EvaluationFailure {
    case_id: String,
    assertion_id: String,
    severity: String,
    failure_code: String,
    reasons: Vec<String>,
}

impl EvaluationFailure {
    fn display_line(&self) -> String {
        format!(
            "{} {} [{}:{}]: {}",
            self.case_id,
            self.assertion_id,
            self.severity,
            self.failure_code,
            self.reasons.join("; ")
        )
    }

    fn to_value(&self) -> Value {
        serde_json::json!({
            "case_id": self.case_id,
            "assertion_id": self.assertion_id,
            "severity": self.severity,
            "failure_code": self.failure_code,
            "reasons": self.reasons
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
struct EvaluationSummary {
    cases_evaluated: usize,
    assertions_checked: usize,
    assertions_passed: usize,
    failures: Vec<EvaluationFailure>,
}

impl EvaluationSummary {
    fn print(&self) {
        let status = if self.failures.is_empty() { "PASS" } else { "FAIL" };
        println!("ROOM STRESS RESULT REPORT: {status}");
        println!("cases evaluated: {}", self.cases_evaluated);
        println!("assertions checked: {}", self.assertions_checked);
        println!("assertions passed: {}", self.assertions_passed);
        println!(
            "assertions failed: {}",
            self.assertions_checked.saturating_sub(self.assertions_passed)
        );
        for failure in &self.failures {
            println!("FAIL: {}", failure.display_line());
        }
        println!(concat!(
            "Claim ceiling: reported observations are compared with synthetic oracles; ",
            "artifact paths and hashes are checked but artifact semantics are not qualified."
        ));
    }

    fn to_value(&self) -> Value {
        serde_json::json!({
            "result_version": "room-stress-evaluation-result-v0.1",
            "status": if self.failures.is_empty() { "pass" } else { "fail" },
            "cases_evaluated": self.cases_evaluated,
            "assertions_checked": self.assertions_checked,
            "assertions_passed": self.assertions_passed,
            "assertions_failed": self.assertions_checked.saturating_sub(self.assertions_passed),
            "failures": self.failures.iter().map(EvaluationFailure::to_value).collect::<Vec<_>>(),
            "claim_ceiling": concat!(
                "Reports are checked against synthetic oracles; evidence path/digest checks ",
                "do not qualify artifact semantics."
            )
        })
    }
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
        println!(
            "Claim ceiling: manifest/schema integrity only; no geometry or building qualification."
        );
    }
}

fn sha256_bytes_hex(input: &[u8]) -> String {
    Sha256::digest(input)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256_hex(input: &str) -> String {
    sha256_bytes_hex(input.as_bytes())
}

fn receipt_value(
    summary: &ValidationSummary,
    benchmark_text: &str,
    schema_text: &str,
    run_report_schema_text: &str,
    cargo_manifest_text: &str,
    validator_source_text: &str,
) -> Value {
    serde_json::json!({
        "receipt_version": "room-stress-manifest-receipt-v0.1",
        "status": "valid_manifest",
        "validator": {
            "name": "room-stress-qualifier",
            "version": env!("CARGO_PKG_VERSION"),
            "source_sha256": sha256_hex(validator_source_text),
            "cargo_manifest_sha256": sha256_hex(cargo_manifest_text)
        },
        "inputs": {
            "benchmark_sha256": sha256_hex(benchmark_text),
            "schema_sha256": sha256_hex(schema_text),
            "run_report_schema_sha256": sha256_hex(run_report_schema_text)
        },
        "summary": {
            "schema_version": EXPECTED_SCHEMA_VERSION,
            "case_count": summary.case_count,
            "distinct_family_count": summary.family_count,
            "risk_counts": &summary.risk_counts
        },
        "claim_ceiling": CLAIM_CEILING
    })
}

fn parse_json(label: &str, input: &str) -> Result<Value, String> {
    serde_json::from_str(input).map_err(|error| format!("{label} is not valid JSON: {error}"))
}

fn validate_documents(
    benchmark_text: &str,
    schema_text: &str,
) -> Result<ValidationSummary, String> {
    let benchmark = parse_json("benchmark", benchmark_text)?;
    let schema = parse_json("schema", schema_text)?;
    validate_values(benchmark, schema)
}

fn verify_evidence_reference(reference: &Value, evidence_root: &Path) -> Result<(), String> {
    let relative = reference
        .get("artifact_path")
        .and_then(Value::as_str)
        .ok_or_else(|| "artifact_path must be a string".to_string())?;

    if relative.trim().is_empty() || relative.contains('\\') {
        return Err("artifact_path must be a non-empty portable relative path".into());
    }

    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err("artifact_path cannot contain absolute paths or dot segments".into());
    }

    let claimed_hash = reference
        .get("artifact_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "artifact_sha256 must be a string".to_string())?;
    let artifact_path = evidence_root.join(relative_path);
    let canonical_artifact = fs::canonicalize(&artifact_path).map_err(|error| {
        format!("evidence artifact {} is unavailable: {error}", relative_path.display())
    })?;
    if !canonical_artifact.starts_with(evidence_root) {
        return Err("evidence artifact resolves outside the declared evidence root".into());
    }

    let metadata = fs::metadata(&canonical_artifact)
        .map_err(|error| format!("cannot inspect evidence artifact: {error}"))?;
    if !metadata.is_file() {
        return Err("evidence artifact is not a regular file".into());
    }
    if metadata.len() > MAX_EVIDENCE_ARTIFACT_BYTES {
        return Err(format!(
            "evidence artifact exceeds the {MAX_EVIDENCE_ARTIFACT_BYTES}-byte limit"
        ));
    }

    let file = fs::File::open(&canonical_artifact)
        .map_err(|error| format!("cannot open evidence artifact: {error}"))?;
    let mut bounded_reader = file.take(MAX_EVIDENCE_ARTIFACT_BYTES + 1);
    let mut bytes = Vec::new();
    bounded_reader
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read evidence artifact: {error}"))?;
    if bytes.len() as u64 > MAX_EVIDENCE_ARTIFACT_BYTES {
        return Err(format!(
            "evidence artifact exceeds the {MAX_EVIDENCE_ARTIFACT_BYTES}-byte limit"
        ));
    }
    let actual_hash = sha256_bytes_hex(&bytes);
    if actual_hash != claimed_hash {
        return Err(format!(
            "evidence artifact digest mismatch: declared {claimed_hash}, actual {actual_hash}"
        ));
    }
    Ok(())
}

fn evaluate_run_report(
    benchmark_text: &str,
    benchmark_schema_text: &str,
    report_schema_text: &str,
    report_text: &str,
    evidence_root: &Path,
) -> Result<EvaluationSummary, String> {
    // Validate the two contracts before scoring any candidate report.
    validate_documents(benchmark_text, benchmark_schema_text)?;
    let evidence_root = fs::canonicalize(evidence_root)
        .map_err(|error| format!("cannot resolve evidence root: {error}"))?;
    if !evidence_root.is_dir() {
        return Err("evidence root must be a directory".into());
    }
    let benchmark = parse_json("benchmark", benchmark_text)?;
    let report = parse_json("run report", report_text)?;
    let report_schema = parse_json("run report schema", report_schema_text)?;

    if !draft202012::meta::is_valid(&report_schema) {
        return Err(
            "run report schema is invalid under the JSON Schema Draft 2020-12 meta-schema".into(),
        );
    }
    let report_validator = draft202012::new(&report_schema)
        .map_err(|error| format!("could not compile run report schema: {error}"))?;
    let report_errors: Vec<String> = report_validator
        .iter_errors(&report)
        .take(20)
        .map(|error| format!("{} at {}", error, error.instance_path()))
        .collect();
    if !report_errors.is_empty() {
        return Err(format!(
            "run report failed JSON Schema validation (showing at most 20 errors):\n{}",
            report_errors.join("\n")
        ));
    }

    let supplied_hash = report
        .get("benchmark_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "run report benchmark_sha256 must be a string".to_string())?;
    let actual_hash = sha256_hex(benchmark_text);
    if supplied_hash != actual_hash {
        return Err(format!(concat!(
            "benchmark digest mismatch: report binds {supplied_hash}, ",
            "exact benchmark bytes hash to {actual_hash}"
        )));
    }

    let expected_cases = benchmark
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| "benchmark cases must be an array".to_string())?;
    let report_cases = report
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| "run report results must be an array".to_string())?;

    let expected_ids: BTreeSet<String> = expected_cases
        .iter()
        .filter_map(|case| case.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect();
    let mut report_by_id = BTreeMap::<String, &Value>::new();

    for result in report_cases {
        let case_id = result
            .get("case_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "each run result must have a case_id string".to_string())?;
        if !expected_ids.contains(case_id) {
            return Err(format!("run report contains unknown case_id {case_id}"));
        }
        if report_by_id.insert(case_id.to_owned(), result).is_some() {
            return Err(format!("duplicate run result for case {case_id}"));
        }
    }

    for case_id in &expected_ids {
        if !report_by_id.contains_key(case_id) {
            return Err(format!("run report is missing case result {case_id}"));
        }
    }

    let mut summary = EvaluationSummary {
        cases_evaluated: expected_cases.len(),
        assertions_checked: 0,
        assertions_passed: 0,
        failures: Vec::new(),
    };

    for case in expected_cases {
        let case_id = case
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "benchmark case id must be a string".to_string())?;
        let result = report_by_id
            .get(case_id)
            .ok_or_else(|| format!("missing result for {case_id}"))?;
        let assertions = case
            .get("assertions")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{case_id}.assertions must be an array"))?;
        let observations = result
            .get("observations")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{case_id} observations must be an object"))?;
        let evidence_refs = result
            .get("evidence_refs")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{case_id} evidence_refs must be an array"))?;

        for assertion in assertions {
            summary.assertions_checked += 1;
            let assertion_id = assertion
                .get("assertion_id")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{case_id} assertion_id must be a string"))?;
            let observation_key = assertion
                .get("observation_key")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{assertion_id} observation_key must be a string"))?;
            let failure_code = assertion
                .get("failure_code")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{assertion_id} failure_code must be a string"))?;
            let severity = assertion
                .get("severity")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{assertion_id} severity must be a string"))?;
            let expected = assertion
                .get("expected")
                .ok_or_else(|| format!("{assertion_id} expected value is missing"))?;

            let actual = observations.get(observation_key);
            let matching_refs: Vec<&Value> = evidence_refs
                .iter()
                .filter(|reference| {
                    reference.get("observation_key").and_then(Value::as_str)
                        == Some(observation_key)
                })
                .collect();

            let mut reasons = Vec::new();
            match actual {
                Some(value) if value == expected => {}
                Some(value) => reasons.push(format!(
                    "observation {observation_key} expected {expected}, observed {value}"
                )),
                None => reasons.push(format!("observation {observation_key} is missing")),
            }

            if assertion
                .get("evidence_required")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                if matching_refs.len() != 1 {
                    reasons.push(format!(
                        "expected exactly one evidence reference for {observation_key}, found {}",
                        matching_refs.len()
                    ));
                } else if let Err(error) =
                    verify_evidence_reference(matching_refs[0], &evidence_root)
                {
                    reasons.push(error);
                }
            }

            if reasons.is_empty() {
                summary.assertions_passed += 1;
            } else {
                summary.failures.push(EvaluationFailure {
                    case_id: case_id.to_owned(),
                    assertion_id: assertion_id.to_owned(),
                    severity: severity.to_owned(),
                    failure_code: failure_code.to_owned(),
                    reasons,
                });
            }
        }
    }

    Ok(summary)
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

    let manifest_fields = benchmark
        .get("case_fields")
        .ok_or_else(|| "case_fields is missing".to_string())?;
    let schema_fields = schema
        .pointer("/properties/case_fields/const")
        .ok_or_else(|| "schema case_fields const is missing".to_string())?;
    if manifest_fields != schema_fields {
        return Err("manifest case_fields registry does not match schema contract".into());
    }

    let mut ids = BTreeSet::new();
    let mut assertion_ids = BTreeSet::new();
    let mut families = BTreeSet::new();
    let mut risk_counts = BTreeMap::<String, usize>::new();

    for (index, case) in cases.iter().enumerate() {
        let case_label = format!("cases[{index}]");
        let id = case
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{case_label}.id must be a string"))?;

        if !ids.insert(id.to_owned()) {
            return Err(format!("duplicate case id: {id}"));
        }

        // Schema validation checks the lexical pattern; this additional oracle
        // requires the initial sealed corpus to enumerate exactly 001..030.
        let expected_id = format!("ROOM-STRESS-{:03}", index + 1);
        if id != expected_id {
            return Err(format!(
                "{case_label}.id must be {expected_id} at this position, found {id}"
            ));
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

        for field in [
            "known_facts",
            "unknowns",
            "required_behaviors",
            "blocking_failures",
        ] {
            let values = case
                .get(field)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{case_label}.{field} must be an array"))?;
            if values.is_empty() {
                return Err(format!("{case_label}.{field} must not be empty"));
            }
            if values
                .iter()
                .any(|value| value.as_str().map(str::trim).is_none_or(str::is_empty))
            {
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

        let assertions = case
            .get("assertions")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{case_label}.assertions must be an array"))?;
        let mut observation_keys = BTreeSet::new();
        for (assertion_index, assertion) in assertions.iter().enumerate() {
            let assertion_label = format!("{case_label}.assertions[{assertion_index}]");
            let assertion_id = assertion
                .get("assertion_id")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{assertion_label}.assertion_id must be a string"))?;
            if !assertion_ids.insert(assertion_id.to_owned()) {
                return Err(format!("duplicate assertion id: {assertion_id}"));
            }
            let observation_key = assertion
                .get("observation_key")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    format!("{assertion_label}.observation_key must be a string")
                })?;
            if !observation_keys.insert(observation_key.to_owned()) {
                return Err(format!(
                    "duplicate observation key {observation_key} in {case_label}"
                ));
            }
        }
    }

    Ok(ValidationSummary {
        case_count: cases.len(),
        family_count: families.len(),
        risk_counts,
    })
}

fn run(args: &[String]) -> Result<bool, String> {
    if args.first().map(String::as_str) == Some("--evaluate-json") {
        if args.len() != 3 {
            return Err(
                concat!(
                    "usage: room-stress-qualifier --evaluate-json ",
                    "RUN_REPORT.json EVIDENCE_ROOT"
                ).to_string(),
            );
        }
        let report_path = &args[1];
        let report_text = fs::read_to_string(Path::new(report_path))
            .map_err(|error| format!("cannot read run report {report_path}: {error}"))?;
        let summary = evaluate_run_report(
            EMBEDDED_BENCHMARK,
            EMBEDDED_SCHEMA,
            EMBEDDED_RUN_REPORT_SCHEMA,
            &report_text,
            Path::new(&args[2]),
        )?;
        let passes = summary.failures.is_empty();
        let rendered = serde_json::to_string_pretty(&summary.to_value())
            .map_err(|error| format!("could not serialize evaluation result: {error}"))?;
        println!("{rendered}");
        return Ok(passes);
    }

    if args.first().map(String::as_str) == Some("--evaluate") {
        if args.len() != 3 {
            return Err(
                "usage: room-stress-qualifier --evaluate RUN_REPORT.json EVIDENCE_ROOT".to_string(),
            );
        }
        let report_path = &args[1];
        let report_text = fs::read_to_string(Path::new(report_path))
            .map_err(|error| format!("cannot read run report {report_path}: {error}"))?;
        let summary = evaluate_run_report(
            EMBEDDED_BENCHMARK,
            EMBEDDED_SCHEMA,
            EMBEDDED_RUN_REPORT_SCHEMA,
            &report_text,
            Path::new(&args[2]),
        )?;
        summary.print();
        return Ok(summary.assertions_passed == summary.assertions_checked);
    }

    let (json_mode, path_args) = match args.first().map(String::as_str) {
        Some("--json") => (true, &args[1..]),
        _ => (false, args),
    };

    let (benchmark_path, schema_path) = match path_args {
        [] => (
            "docs/research/spatial-design/ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.json",
            "docs/research/spatial-design/ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.schema.json",
        ),
        [benchmark, schema] => (benchmark.as_str(), schema.as_str()),
        _ => {
            return Err(
                "usage: room-stress-qualifier [--json] [BENCHMARK.json SCHEMA.json]".to_string(),
            );
        }
    };

    let benchmark_text = fs::read_to_string(Path::new(benchmark_path))
        .map_err(|error| format!("cannot read benchmark {benchmark_path}: {error}"))?;
    let schema_text = fs::read_to_string(Path::new(schema_path))
        .map_err(|error| format!("cannot read schema {schema_path}: {error}"))?;

    let summary = validate_documents(&benchmark_text, &schema_text)?;
    if json_mode {
        let receipt = receipt_value(
            &summary,
            &benchmark_text,
            &schema_text,
            EMBEDDED_RUN_REPORT_SCHEMA,
            EMBEDDED_CARGO_MANIFEST,
            EMBEDDED_VALIDATOR_SOURCE,
        );
        let rendered = serde_json::to_string_pretty(&receipt)
            .map_err(|error| format!("could not serialize receipt: {error}"))?;
        println!("{rendered}");
    } else {
        summary.print();
    }
    Ok(true)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    match run(&args) {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(error) => {
            eprintln!("ROOM IMAGINATION STRESS VALIDATION: INVALID");
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_values() -> (Value, Value) {
        let benchmark: Value =
            serde_json::from_str(EMBEDDED_BENCHMARK).expect("benchmark fixture parses");
        let schema: Value = serde_json::from_str(EMBEDDED_SCHEMA).expect("schema fixture parses");
        (benchmark, schema)
    }

    #[test]
    fn committed_manifest_passes_schema_and_semantic_checks() {
        let summary = validate_documents(EMBEDDED_BENCHMARK, EMBEDDED_SCHEMA)
            .expect("committed manifest should validate");
        assert_eq!(summary.case_count, EXPECTED_CASE_COUNT);
        assert_eq!(summary.family_count, EXPECTED_CASE_COUNT);
        assert_eq!(
            summary.risk_counts.values().sum::<usize>(),
            EXPECTED_CASE_COUNT
        );
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
        assert!(error.contains("duplicate case id"));
    }

    #[test]
    fn missing_required_field_is_rejected() {
        let (mut benchmark, schema) = fixture_values();
        benchmark["cases"][0]
            .as_object_mut()
            .unwrap()
            .remove("blocking_failures");
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
    fn blank_required_item_is_rejected() {
        let (mut benchmark, schema) = fixture_values();
        benchmark["cases"][0]["known_facts"] = Value::Array(vec![Value::String("  ".into())]);
        let error = validate_values(benchmark, schema).unwrap_err();
        assert!(error.contains("non-empty strings"));
    }

    #[test]
    fn wrong_declared_count_is_rejected() {
        let (mut benchmark, schema) = fixture_values();
        benchmark["count"] = Value::from(29);
        assert!(validate_values(benchmark, schema).is_err());
    }

    #[test]
    fn receipt_binds_exact_input_bytes_and_states_claim_ceiling() {
        let summary = validate_documents(EMBEDDED_BENCHMARK, EMBEDDED_SCHEMA).unwrap();
        let receipt = receipt_value(
            &summary,
            EMBEDDED_BENCHMARK,
            EMBEDDED_SCHEMA,
            EMBEDDED_RUN_REPORT_SCHEMA,
            EMBEDDED_CARGO_MANIFEST,
            EMBEDDED_VALIDATOR_SOURCE,
        );
        assert_eq!(receipt["status"], "valid_manifest");
        assert_eq!(
            receipt["inputs"]["benchmark_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert_eq!(
            receipt["inputs"]["schema_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert_eq!(
            receipt["inputs"]["run_report_schema_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert!(
            receipt["claim_ceiling"]
                .as_str()
                .unwrap()
                .contains("does not qualify geometry")
        );
        assert_eq!(
            receipt,
            receipt_value(
                &summary,
                EMBEDDED_BENCHMARK,
                EMBEDDED_SCHEMA,
                EMBEDDED_RUN_REPORT_SCHEMA,
                EMBEDDED_CARGO_MANIFEST,
                EMBEDDED_VALIDATOR_SOURCE,
            ),
            "the receipt must be deterministic for identical input bytes"
        );
    }

    fn golden_report_value(benchmark_text: &str) -> Value {
        let benchmark: Value = serde_json::from_str(benchmark_text).unwrap();
        let benchmark_evidence_path = concat!(
            "docs/research/spatial-design/",
            "ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.json"
        );
        let mut results = Vec::new();
        for case in benchmark["cases"].as_array().unwrap() {
            let mut observations = serde_json::Map::new();
            let mut evidence_refs = Vec::new();
            for assertion in case["assertions"].as_array().unwrap() {
                let key = assertion["observation_key"].as_str().unwrap();
                observations.insert(key.to_owned(), assertion["expected"].clone());
                evidence_refs.push(serde_json::json!({
                    "observation_key": key,
                    "artifact_sha256": sha256_hex(benchmark_text),
                    "artifact_revision": "synthetic-golden-fixture-r1",
                    "artifact_path": benchmark_evidence_path
                }));
            }
            results.push(serde_json::json!({
                "case_id": case["id"],
                "observations": observations,
                "evidence_refs": evidence_refs
            }));
        }
        serde_json::json!({
            "schema_version": "room-stress-run-report-v0.1",
            "implementation": {
                "name": "synthetic-golden-fixture",
                "revision": "fixture-r1"
            },
            "benchmark_sha256": sha256_hex(benchmark_text),
            "results": results
        })
    }

    fn repository_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root exists")
    }

    #[test]
    fn golden_run_report_passes_all_machine_checkable_assertions() {
        let report = golden_report_value(EMBEDDED_BENCHMARK);
        let report_text = serde_json::to_string(&report).unwrap();
        let summary = evaluate_run_report(
            EMBEDDED_BENCHMARK,
            EMBEDDED_SCHEMA,
            EMBEDDED_RUN_REPORT_SCHEMA,
            &report_text,
            &repository_root(),
        )
        .expect("golden result report must evaluate");
        assert_eq!(summary.cases_evaluated, EXPECTED_CASE_COUNT);
        assert_eq!(summary.assertions_checked, 47);
        assert_eq!(summary.assertions_passed, 47);
        assert!(summary.failures.is_empty());
    }

    #[test]
    fn evaluation_summary_json_is_structured_and_stable() {
        let summary = EvaluationSummary {
            cases_evaluated: 1,
            assertions_checked: 1,
            assertions_passed: 0,
            failures: vec![EvaluationFailure {
                case_id: "ROOM-STRESS-001".into(),
                assertion_id: "ROOM-ASSERT-001-01".into(),
                severity: "blocking".into(),
                failure_code: "HARD_CONSTRAINT_DROPPED".into(),
                reasons: vec!["fixture failure".into()],
            }],
        };
        let value = summary.to_value();
        assert_eq!(value["status"], "fail");
        assert_eq!(value["assertions_failed"], 1);
        assert_eq!(value["failures"][0]["failure_code"], "HARD_CONSTRAINT_DROPPED");
        assert_eq!(value["failures"][0]["case_id"], "ROOM-STRESS-001");
    }

    #[test]
    fn wrong_observation_value_fails_with_stable_reason_code() {
        let mut report = golden_report_value(EMBEDDED_BENCHMARK);
        report["results"][0]["observations"]["hard_constraints_survive_edit_sequence"] =
            Value::Bool(false);
        let report_text = serde_json::to_string(&report).unwrap();
        let summary = evaluate_run_report(
            EMBEDDED_BENCHMARK,
            EMBEDDED_SCHEMA,
            EMBEDDED_RUN_REPORT_SCHEMA,
            &report_text,
            &repository_root(),
        )
        .unwrap();
        assert_eq!(summary.assertions_passed, 46);
        assert_eq!(summary.failures[0].failure_code, "HARD_CONSTRAINT_DROPPED");
    }

    #[test]
    fn missing_evidence_reference_fails_closed() {
        let mut report = golden_report_value(EMBEDDED_BENCHMARK);
        report["results"][0]["evidence_refs"]
            .as_array_mut()
            .unwrap()
            .remove(0);
        let report_text = serde_json::to_string(&report).unwrap();
        let summary = evaluate_run_report(
            EMBEDDED_BENCHMARK,
            EMBEDDED_SCHEMA,
            EMBEDDED_RUN_REPORT_SCHEMA,
            &report_text,
            &repository_root(),
        )
        .unwrap();
        assert!(summary.failures.iter().any(|failure| {
            failure.reasons.iter().any(|reason| {
                reason.contains("expected exactly one evidence reference")
            })
        }));
    }

    #[test]
    fn report_bound_to_different_benchmark_bytes_is_rejected() {
        let mut report = golden_report_value(EMBEDDED_BENCHMARK);
        report["benchmark_sha256"] = Value::String(
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
        );
        let report_text = serde_json::to_string(&report).unwrap();
        let error = evaluate_run_report(
            EMBEDDED_BENCHMARK,
            EMBEDDED_SCHEMA,
            EMBEDDED_RUN_REPORT_SCHEMA,
            &report_text,
            &repository_root(),
        )
        .unwrap_err();
        assert!(error.contains("benchmark digest mismatch"));
    }

    #[test]
    fn evidence_artifact_hash_mismatch_fails_closed() {
        let mut report = golden_report_value(EMBEDDED_BENCHMARK);
        report["results"][0]["evidence_refs"][0]["artifact_sha256"] =
            Value::String("b".repeat(64));
        let report_text = serde_json::to_string(&report).unwrap();
        let summary = evaluate_run_report(
            EMBEDDED_BENCHMARK,
            EMBEDDED_SCHEMA,
            EMBEDDED_RUN_REPORT_SCHEMA,
            &report_text,
            &repository_root(),
        )
        .unwrap();
        assert!(summary.failures.iter().any(|failure| {
            failure.reasons.iter().any(|reason| {
                reason.contains("evidence artifact digest mismatch")
            })
        }));
    }

    #[test]
    fn evidence_path_traversal_is_rejected() {
        let mut report = golden_report_value(EMBEDDED_BENCHMARK);
        report["results"][0]["evidence_refs"][0]["artifact_path"] =
            Value::String("../Cargo.toml".into());
        let report_text = serde_json::to_string(&report).unwrap();
        let summary = evaluate_run_report(
            EMBEDDED_BENCHMARK,
            EMBEDDED_SCHEMA,
            EMBEDDED_RUN_REPORT_SCHEMA,
            &report_text,
            &repository_root(),
        )
        .unwrap();
        assert!(summary.failures.iter().any(|failure| {
            failure.reasons.iter().any(|reason| {
                reason.contains("artifact_path cannot contain absolute paths or dot segments")
            })
        }));
    }

    #[test]
    fn missing_evidence_artifact_is_reported_as_failure() {
        let mut report = golden_report_value(EMBEDDED_BENCHMARK);
        report["results"][0]["evidence_refs"][0]["artifact_path"] =
            Value::String("docs/research/spatial-design/not-present.bin".into());
        let report_text = serde_json::to_string(&report).unwrap();
        let summary = evaluate_run_report(
            EMBEDDED_BENCHMARK,
            EMBEDDED_SCHEMA,
            EMBEDDED_RUN_REPORT_SCHEMA,
            &report_text,
            &repository_root(),
        )
        .unwrap();
        assert!(summary.failures.iter().any(|failure| {
            failure.reasons.iter().any(|reason| {
                reason.contains("evidence artifact") && reason.contains("is unavailable")
            })
        }));
    }

    #[test]
    fn duplicate_case_result_is_rejected() {
        let mut report = golden_report_value(EMBEDDED_BENCHMARK);
        let first_case_id = report["results"][0]["case_id"].clone();
        report["results"][1]["case_id"] = first_case_id;
        let report_text = serde_json::to_string(&report).unwrap();
        let error = evaluate_run_report(
            EMBEDDED_BENCHMARK,
            EMBEDDED_SCHEMA,
            EMBEDDED_RUN_REPORT_SCHEMA,
            &report_text,
            &repository_root(),
        )
        .unwrap_err();
        assert!(error.contains("duplicate run result"));
    }

    #[test]
    fn duplicate_assertion_ids_are_rejected() {
        let (mut benchmark, schema) = fixture_values();
        let duplicate = benchmark["cases"][0]["assertions"][0]["assertion_id"].clone();
        benchmark["cases"][1]["assertions"][0]["assertion_id"] = duplicate;
        let error = validate_values(benchmark, schema).unwrap_err();
        assert!(error.contains("duplicate assertion id"));
    }

    #[test]
    fn duplicate_observation_keys_within_a_case_are_rejected() {
        let (mut benchmark, schema) = fixture_values();
        let duplicate = benchmark["cases"][0]["assertions"][0]["observation_key"].clone();
        benchmark["cases"][0]["assertions"][1]["observation_key"] = duplicate;
        let error = validate_values(benchmark, schema).unwrap_err();
        assert!(error.contains("duplicate observation key"));
    }

    #[test]
    fn manifest_case_field_registry_must_match_schema() {
        let (mut benchmark, schema) = fixture_values();
        benchmark["case_fields"].as_array_mut().unwrap().pop();
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
