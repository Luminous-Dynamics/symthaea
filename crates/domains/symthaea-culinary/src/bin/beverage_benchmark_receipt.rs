// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed receipt generator for the synthetic beverage-engineering corpus.
//!
//! This qualifies file/contract integrity only. It does not evaluate beverage
//! quality, fermentation kinetics, food safety, sensory outcomes, or SOTA claims.

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fs,
    path::PathBuf,
    process::{Command, ExitCode},
};

const CORPUS_VERSION: &str = "beverage-engineering-scenarios-v1";
const CORPUS_STATUS: &str = "synthetic_contract_fixtures_not_beverage_measurements";
const SCHEMA_URI: &str = "https://json-schema.org/draft/2020-12/schema";
const EXPECTED_FIELDS: &[&str] = &[
    "id",
    "name",
    "product",
    "task",
    "evidence_class",
    "expected_disposition",
    "guards",
    "primary_metrics",
    "claim_ceiling",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    schema_version: String,
    status: String,
    description: String,
    source_spec: String,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    id: String,
    name: String,
    product: String,
    task: String,
    evidence_class: String,
    expected_disposition: String,
    guards: Vec<String>,
    primary_metrics: Vec<String>,
    claim_ceiling: String,
}

#[derive(Debug)]
struct CheckResult {
    errors: Vec<String>,
    scenario_count: usize,
    corpus_sha256: String,
    schema_sha256: String,
}

impl CheckResult {
    fn passed(&self) -> bool {
        self.errors.is_empty()
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn string_set(value: Option<&Value>) -> Option<BTreeSet<String>> {
    let items = value?.as_array()?;
    let mut set = BTreeSet::new();
    for item in items {
        set.insert(item.as_str()?.to_owned());
    }
    Some(set)
}

fn schema_enum(schema: &Value, field: &str) -> Option<BTreeSet<String>> {
    string_set(
        schema
            .get("properties")?
            .get("scenarios")?
            .get("items")?
            .get("properties")?
            .get(field)?
            .get("enum"),
    )
}

fn validate_inputs(corpus_bytes: &[u8], schema_bytes: &[u8]) -> CheckResult {
    let corpus_sha256 = hex_sha256(corpus_bytes);
    let schema_sha256 = hex_sha256(schema_bytes);
    let mut errors = Vec::new();

    let schema: Value = match serde_json::from_slice(schema_bytes) {
        Ok(value) => value,
        Err(error) => {
            errors.push(format!("schema_json_invalid: {error}"));
            return CheckResult {
                errors,
                scenario_count: 0,
                corpus_sha256,
                schema_sha256,
            };
        }
    };

    if schema.get("$schema").and_then(Value::as_str) != Some(SCHEMA_URI) {
        errors.push("schema_dialect_mismatch".to_owned());
    }
    if schema.get("additionalProperties").and_then(Value::as_bool) != Some(false) {
        errors.push("schema_root_must_reject_unknown_fields".to_owned());
    }

    let declared_required = schema
        .get("properties")
        .and_then(|value| value.get("scenarios"))
        .and_then(|value| value.get("items"))
        .and_then(|value| value.get("required"))
        .and_then(Value::as_array);
    match declared_required {
        Some(fields) => {
            let actual: BTreeSet<&str> = fields.iter().filter_map(Value::as_str).collect();
            let expected: BTreeSet<&str> = EXPECTED_FIELDS.iter().copied().collect();
            if actual != expected {
                errors.push("schema_required_fields_mismatch".to_owned());
            }
        }
        None => errors.push("schema_required_fields_missing".to_owned()),
    }

    let id_pattern = schema
        .get("properties")
        .and_then(|value| value.get("scenarios"))
        .and_then(|value| value.get("items"))
        .and_then(|value| value.get("properties"))
        .and_then(|value| value.get("id"))
        .and_then(|value| value.get("pattern"))
        .and_then(Value::as_str);
    if id_pattern != Some("^BEV-[0-9]{3}$") {
        errors.push("schema_id_pattern_mismatch".to_owned());
    }

    let product_values = schema_enum(&schema, "product");
    let disposition_values = schema_enum(&schema, "expected_disposition");
    if product_values.is_none() {
        errors.push("schema_product_enum_missing_or_invalid".to_owned());
    }
    if disposition_values.is_none() {
        errors.push("schema_disposition_enum_missing_or_invalid".to_owned());
    }

    let corpus: Corpus = match serde_json::from_slice(corpus_bytes) {
        Ok(value) => value,
        Err(error) => {
            errors.push(format!("corpus_json_or_shape_invalid: {error}"));
            return CheckResult {
                errors,
                scenario_count: 0,
                corpus_sha256,
                schema_sha256,
            };
        }
    };

    if corpus.schema_version != CORPUS_VERSION {
        errors.push("corpus_version_mismatch".to_owned());
    }
    if corpus.status != CORPUS_STATUS {
        errors.push("corpus_evidence_status_mismatch".to_owned());
    }
    if corpus.description.trim().is_empty() {
        errors.push("corpus_description_empty".to_owned());
    }
    if corpus.source_spec.trim().is_empty() {
        errors.push("corpus_source_spec_empty".to_owned());
    }
    if corpus.scenarios.len() < 16 {
        errors.push("scenario_count_below_minimum_16".to_owned());
    }

    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    for (index, scenario) in corpus.scenarios.iter().enumerate() {
        let prefix = format!("scenario_{}", index + 1);
        let valid_id = scenario
            .id
            .strip_prefix("BEV-")
            .is_some_and(|suffix| suffix.len() == 3 && suffix.bytes().all(|b| b.is_ascii_digit()));
        if !valid_id {
            errors.push(format!("{prefix}_id_invalid"));
        }
        if !ids.insert(scenario.id.as_str()) {
            errors.push(format!("{prefix}_id_duplicate"));
        }
        if scenario.name.trim().is_empty() || !names.insert(scenario.name.as_str()) {
            errors.push(format!("{prefix}_name_empty_or_duplicate"));
        }
        if !product_values
            .as_ref()
            .is_some_and(|values| values.contains(&scenario.product))
        {
            errors.push(format!("{prefix}_product_not_in_schema_enum"));
        }
        if !disposition_values
            .as_ref()
            .is_some_and(|values| values.contains(&scenario.expected_disposition))
        {
            errors.push(format!("{prefix}_disposition_not_in_schema_enum"));
        }
        if scenario.task.trim().is_empty() {
            errors.push(format!("{prefix}_task_empty"));
        }
        if scenario.evidence_class.trim().is_empty() {
            errors.push(format!("{prefix}_evidence_class_empty"));
        }
        if scenario.guards.is_empty() || scenario.guards.iter().any(|item| item.trim().is_empty()) {
            errors.push(format!("{prefix}_guards_empty_or_blank"));
        }
        if scenario
            .primary_metrics
            .iter()
            .any(|item| item.trim().is_empty())
        {
            errors.push(format!("{prefix}_primary_metric_blank"));
        }
        if scenario.claim_ceiling.trim().is_empty() {
            errors.push(format!("{prefix}_claim_ceiling_empty"));
        }
    }

    CheckResult {
        errors,
        scenario_count: corpus.scenarios.len(),
        corpus_sha256,
        schema_sha256,
    }
}

fn head_matches(expected_head: &str, actual_head: &str) -> bool {
    expected_head.len() == 40
        && expected_head.bytes().all(|byte| byte.is_ascii_hexdigit())
        && expected_head == actual_head
}

fn git_head() -> Result<String, String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|error| format!("git_rev_parse_unavailable: {error}"))?;
    if !output.status.success() {
        return Err("git_rev_parse_failed".to_owned());
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|error| format!("git_head_not_utf8: {error}"))
}

fn parse_args() -> Result<(PathBuf, PathBuf, String, PathBuf), String> {
    let mut corpus = None;
    let mut schema = None;
    let mut expected_head = None;
    let mut output = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing_value_for_{flag}"))?;
        match flag.as_str() {
            "--corpus" => corpus = Some(PathBuf::from(value)),
            "--schema" => schema = Some(PathBuf::from(value)),
            "--expected-head" => expected_head = Some(value),
            "--out" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown_argument_{flag}")),
        }
    }

    Ok((
        corpus.ok_or_else(|| "missing_--corpus".to_owned())?,
        schema.ok_or_else(|| "missing_--schema".to_owned())?,
        expected_head.ok_or_else(|| "missing_--expected-head".to_owned())?,
        output.ok_or_else(|| "missing_--out".to_owned())?,
    ))
}

fn run() -> Result<ExitCode, String> {
    let (corpus_path, schema_path, expected_head, output_path) = parse_args()?;
    let actual_head = git_head()?;
    let corpus_bytes =
        fs::read(&corpus_path).map_err(|error| format!("corpus_read_failed: {error}"))?;
    let schema_bytes =
        fs::read(&schema_path).map_err(|error| format!("schema_read_failed: {error}"))?;

    let mut check = validate_inputs(&corpus_bytes, &schema_bytes);
    let valid_head = head_matches(&expected_head, &actual_head);
    if !valid_head {
        check
            .errors
            .push("exact_head_mismatch_or_invalid_expected_head".to_owned());
    }

    let passed = check.passed();
    let receipt = json!({
        "receipt_schema": "beverage-benchmark-receipt-v1",
        "status": if passed { "PASS" } else { "FAIL" },
        "qualification_scope": "synthetic_contract_structure_and_exact_git_head_only",
        "exact_head_match": valid_head,
        "expected_head": expected_head,
        "actual_head": actual_head,
        "corpus_file": corpus_path.file_name().and_then(|value| value.to_str()).unwrap_or("<non-utf8>"),
        "schema_file": schema_path.file_name().and_then(|value| value.to_str()).unwrap_or("<non-utf8>"),
        "corpus_sha256": check.corpus_sha256,
        "schema_sha256": check.schema_sha256,
        "scenario_count": check.scenario_count,
        "errors": check.errors,
        "claim_ceiling": "This receipt does not validate the JSON Schema with a full Draft 2020-12 engine, evaluate a beverage model, establish dataset rights, or qualify beverage quality, safety, sensory outcomes, or industrial performance."
    });
    let serialized = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("receipt_serialization_failed: {error}"))?;
    fs::write(&output_path, serialized)
        .map_err(|error| format!("receipt_write_failed: {error}"))?;

    if passed {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::FAILURE)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("beverage-benchmark-receipt: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_corpus() -> &'static [u8] {
        include_bytes!("../../../../../docs/research/beverage-engineering-scenarios-v1.json")
    }

    fn real_schema() -> &'static [u8] {
        include_bytes!("../../../../../docs/research/beverage-engineering-scenarios-v1.schema.json")
    }

    #[test]
    fn checked_in_corpus_passes_contract_checks() {
        let result = validate_inputs(real_corpus(), real_schema());
        assert!(result.errors.is_empty(), "{:?}", result.errors);
        assert_eq!(result.scenario_count, 16);
        assert_eq!(result.corpus_sha256.len(), 64);
        assert_eq!(result.schema_sha256.len(), 64);
    }

    #[test]
    fn duplicate_ids_fail_closed() {
        let mut corpus: Value = serde_json::from_slice(real_corpus()).expect("valid corpus json");
        let first = corpus["scenarios"][0]["id"].clone();
        corpus["scenarios"][1]["id"] = first;
        let bytes = serde_json::to_vec(&corpus).expect("serialize modified corpus");
        let result = validate_inputs(&bytes, real_schema());
        assert!(result.errors.iter().any(|e| e.ends_with("_id_duplicate")));
    }

    #[test]
    fn missing_claim_ceiling_fails_closed() {
        let mut corpus: Value = serde_json::from_slice(real_corpus()).expect("valid corpus json");
        corpus["scenarios"][0]
            .as_object_mut()
            .expect("scenario object")
            .remove("claim_ceiling");
        let bytes = serde_json::to_vec(&corpus).expect("serialize modified corpus");
        let result = validate_inputs(&bytes, real_schema());
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.starts_with("corpus_json_or_shape_invalid"))
        );
    }

    #[test]
    fn unknown_fields_fail_closed() {
        let mut corpus: Value = serde_json::from_slice(real_corpus()).expect("valid corpus json");
        corpus
            .as_object_mut()
            .expect("corpus object")
            .insert("injected_pass".to_owned(), json!(true));
        let bytes = serde_json::to_vec(&corpus).expect("serialize modified corpus");
        let result = validate_inputs(&bytes, real_schema());
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.starts_with("corpus_json_or_shape_invalid"))
        );
    }

    #[test]
    fn schema_enum_drift_fails_closed() {
        let mut schema: Value = serde_json::from_slice(real_schema()).expect("valid schema json");
        schema["properties"]["scenarios"]["items"]["properties"]["product"]["enum"] =
            json!(["beer"]);
        let result = validate_inputs(
            real_corpus(),
            &serde_json::to_vec(&schema).expect("serialize schema"),
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.ends_with("_not_in_schema_enum"))
        );
    }

    #[test]
    fn malformed_json_has_a_non_pass_result() {
        let result = validate_inputs(b"{invalid", real_schema());
        assert!(!result.passed());
        assert_eq!(result.scenario_count, 0);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.starts_with("corpus_json_or_shape_invalid"))
        );
    }

    #[test]
    fn exact_head_gate_rejects_wrong_or_malformed_head() {
        let head = "0123456789abcdef0123456789abcdef01234567";
        assert!(head_matches(head, head));
        assert!(!head_matches("not-a-commit", head));
        assert!(!head_matches(
            "1123456789abcdef0123456789abcdef01234567",
            head
        ));
    }

    #[test]
    fn corpus_status_must_remain_synthetic() {
        let mut corpus: Value = serde_json::from_slice(real_corpus()).expect("valid corpus json");
        corpus["status"] = json!("measured_industrial_data");
        let bytes = serde_json::to_vec(&corpus).expect("serialize modified corpus");
        let result = validate_inputs(&bytes, real_schema());
        assert!(
            result
                .errors
                .iter()
                .any(|e| e == "corpus_evidence_status_mismatch")
        );
    }
}
