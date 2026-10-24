#![deny(unsafe_code)]
// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed validation for the versioned conventional-display workload corpus.
//!
//! This tool validates the checked-in JSON Schema and corpus, then applies
//! semantic checks which are important to this research contract but should
//! not be inferred from generic JSON Schema structure alone:
//! - v0.1 must contain exactly 32 stable case IDs;
//! - source references must resolve and be unique;
//! - lane and minimum evidence class must agree;
//! - every case stays UNQUALIFIED until a future reviewed schema/version admits
//!   a different evidence status;
//! - the declared and schema-fixed corpus counts must agree.
//!
//! A valid corpus contract is not proof that a display, algorithm, measurement,
//! or human study has passed any benchmark.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const EXPECTED_CORPUS_ID: &str = "VIS-DISP-CORPUS-001";
pub const EXPECTED_SCHEMA_VERSION: &str = "0.1.0";
pub const EXPECTED_CASE_COUNT: usize = 32;
pub const UNQUALIFIED: &str = "UNQUALIFIED";
pub const EXPECTED_JSON_SCHEMA_DRAFT: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub corpus_id: String,
    pub schema_version: String,
    pub claim_status: String,
    pub case_count: usize,
    pub lane_counts: BTreeMap<String, usize>,
    pub manifest_sha256: String,
    pub schema_sha256: String,
}

impl ValidationReport {
    /// Converts the report into stable JSON suitable for logs and evidence capture.
    #[must_use]
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "verifier": "visual-display-corpus-verify",
            "verifier_version": env!("CARGO_PKG_VERSION"),
            "schema_draft": EXPECTED_JSON_SCHEMA_DRAFT,
            "corpus_id": self.corpus_id,
            "schema_version": self.schema_version,
            "claim_status": self.claim_status,
            "case_count": self.case_count,
            "lane_counts": self.lane_counts,
            "manifest_sha256": self.manifest_sha256,
            "schema_sha256": self.schema_sha256,
            "qualification_boundary": "Contract valid only; all corpus cases remain UNQUALIFIED. No physical display, certification, power, motion, or human-perception result is asserted."
        })
    }
}

/// Validates raw JSON strings for the schema and the cross-record semantic contract.
///
/// Returns a report only when every gate passes. A returned report deliberately
/// says CONTRACT_VALID_UNQUALIFIED; it must never be described as benchmark PASS.
///
/// # Errors
///
/// Returns an ordered list of validation failures. Schema errors and independent
/// semantic errors are both preserved when available.
pub fn validate_corpus_str(
    manifest_text: &str,
    schema_text: &str,
) -> Result<ValidationReport, Vec<String>> {
    let schema: Value = match serde_json::from_str(schema_text) {
        Ok(value) => value,
        Err(error) => return Err(vec![format!("schema JSON parse error: {error}")]),
    };
    let manifest: Value = match serde_json::from_str(manifest_text) {
        Ok(value) => value,
        Err(error) => return Err(vec![format!("manifest JSON parse error: {error}")]),
    };

    let mut errors = Vec::new();

    if schema.get("$schema").and_then(Value::as_str) != Some(EXPECTED_JSON_SCHEMA_DRAFT) {
        errors.push(format!(
            "schema $schema must be {EXPECTED_JSON_SCHEMA_DRAFT}"
        ));
    }

    let validator = match jsonschema::draft202012::new(&schema) {
        Ok(validator) => Some(validator),
        Err(error) => {
            errors.push(format!("schema compilation failed: {error}"));
            None
        }
    };

    if let Some(validator) = validator {
        for error in validator.iter_errors(&manifest) {
            errors.push(format!("JSON Schema violation: {error}"));
        }
    }

    errors.extend(validate_semantics(&manifest, &schema));

    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }

    let cases = manifest
        .get("cases")
        .and_then(Value::as_array)
        .expect("schema and semantic checks guarantee cases is an array");

    let mut lane_counts = BTreeMap::new();
    for case in cases {
        let lane = case
            .get("lane")
            .and_then(Value::as_str)
            .expect("schema and semantic checks guarantee a string lane");
        *lane_counts.entry(lane.to_owned()).or_insert(0) += 1;
    }

    Ok(ValidationReport {
        corpus_id: EXPECTED_CORPUS_ID.to_owned(),
        schema_version: EXPECTED_SCHEMA_VERSION.to_owned(),
        claim_status: "CONTRACT_VALID_UNQUALIFIED".to_owned(),
        case_count: cases.len(),
        lane_counts,
        manifest_sha256: sha256_hex(manifest_text.as_bytes()),
        schema_sha256: sha256_hex(schema_text.as_bytes()),
    })
}

fn validate_semantics(manifest: &Value, schema: &Value) -> Vec<String> {
    let mut errors = Vec::new();

    expect_string(
        manifest,
        "corpus_id",
        EXPECTED_CORPUS_ID,
        "manifest",
        &mut errors,
    );
    expect_string(
        manifest,
        "schema_version",
        EXPECTED_SCHEMA_VERSION,
        "manifest",
        &mut errors,
    );
    expect_string(manifest, "status", UNQUALIFIED, "manifest", &mut errors);

    match manifest.get("created_date").and_then(Value::as_str) {
        Some(date) if is_valid_iso_date(date) => {}
        Some(date) => errors.push(format!(
            "manifest created_date is not a real YYYY-MM-DD date: {date}"
        )),
        None => errors.push("manifest created_date must be a string".to_owned()),
    }

    let declared_count = manifest.get("case_count").and_then(Value::as_u64);
    if declared_count != Some(EXPECTED_CASE_COUNT as u64) {
        errors.push(format!(
            "manifest case_count must remain frozen at {EXPECTED_CASE_COUNT}"
        ));
    }

    let schema_count = schema
        .pointer("/properties/case_count/const")
        .and_then(Value::as_u64);
    if schema_count != Some(EXPECTED_CASE_COUNT as u64) {
        errors.push(format!(
            "schema properties.case_count.const must remain {EXPECTED_CASE_COUNT}"
        ));
    }

    let min_items = schema
        .pointer("/properties/cases/minItems")
        .and_then(Value::as_u64);
    let max_items = schema
        .pointer("/properties/cases/maxItems")
        .and_then(Value::as_u64);
    if min_items != Some(EXPECTED_CASE_COUNT as u64)
        || max_items != Some(EXPECTED_CASE_COUNT as u64)
    {
        errors.push(format!(
            "schema cases minItems and maxItems must both remain {EXPECTED_CASE_COUNT}"
        ));
    }

    let cases = match manifest.get("cases").and_then(Value::as_array) {
        Some(cases) => cases,
        None => {
            errors.push("manifest cases must be an array".to_owned());
            return errors;
        }
    };

    if cases.len() != EXPECTED_CASE_COUNT {
        errors.push(format!(
            "manifest must contain exactly {EXPECTED_CASE_COUNT} cases, found {}",
            cases.len()
        ));
    }

    let allowed_evidence_classes = [
        "ANALYTIC_REFERENCE",
        "SYNTHETIC_MODEL",
        "BENCH_MEASURED",
        "HUMAN_STUDY",
    ];
    let lane_evidence = [
        ("software_math", "ANALYTIC_REFERENCE"),
        ("software_synthetic", "SYNTHETIC_MODEL"),
        ("physical_measurement", "BENCH_MEASURED"),
        ("human_task_study", "HUMAN_STUDY"),
    ];

    let sources = match manifest.get("reference_sources").and_then(Value::as_array) {
        Some(sources) => sources,
        None => {
            errors.push("manifest reference_sources must be an array".to_owned());
            return errors;
        }
    };

    let mut source_ids = BTreeSet::new();
    for (index, source) in sources.iter().enumerate() {
        let id = source.get("id").and_then(Value::as_str);
        match id {
            Some(id) => {
                if !source_ids.insert(id.to_owned()) {
                    errors.push(format!("duplicate reference source id: {id}"));
                }
            }
            None => errors.push(format!("reference_sources[{index}].id must be a string")),
        }

        if let Some(url) = source.get("url").and_then(Value::as_str)
            && !url.starts_with("https://")
        {
            errors.push(format!("reference_sources[{index}].url must use HTTPS"));
        }
    }

    if let Some(owner_issue) = manifest.get("owner_issue").and_then(Value::as_str)
        && !owner_issue.starts_with("https://github.com/Luminous-Dynamics/symthaea/issues/")
    {
        errors.push("owner_issue must remain a canonical Symthaea issue URL".to_owned());
    }

    let schema_reference_ids: BTreeSet<String> = schema
        .pointer("/properties/cases/items/properties/reference_ids/items/enum")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();

    if schema_reference_ids.is_empty() {
        errors.push(
            "schema must enumerate allowed reference_ids at cases/items/properties/reference_ids/items/enum"
                .to_owned(),
        );
    } else if schema_reference_ids != source_ids {
        errors.push(
            "manifest source IDs must exactly match the schema's allowed reference ID set"
                .to_owned(),
        );
    }

    let mut case_ids = BTreeSet::new();
    for (index, case) in cases.iter().enumerate() {
        let path = format!("cases[{index}]");
        let id = case.get("id").and_then(Value::as_str);

        match id {
            Some(id) => {
                if !is_case_id(id) {
                    errors.push(format!(
                        "{path}.id has invalid VIS-DISP case ID shape: {id}"
                    ));
                }
                if !case_ids.insert(id.to_owned()) {
                    errors.push(format!("duplicate case ID: {id}"));
                }
            }
            None => errors.push(format!("{path}.id must be a string")),
        }

        expect_string(case, "status", UNQUALIFIED, &path, &mut errors);

        let lane = case.get("lane").and_then(Value::as_str);
        let evidence = case.get("minimum_evidence_class").and_then(Value::as_str);

        match (lane, evidence) {
            (Some(lane), Some(evidence)) => {
                let expected = lane_evidence
                    .iter()
                    .find_map(|(known_lane, expected_evidence)| {
                        (*known_lane == lane).then_some(*expected_evidence)
                    });

                match expected {
                    Some(expected) if expected == evidence => {}
                    Some(expected) => errors.push(format!(
                        "{path} lane {lane} requires evidence class {expected}, got {evidence}"
                    )),
                    None => errors.push(format!("{path} has unknown lane: {lane}")),
                }

                if !allowed_evidence_classes.contains(&evidence) {
                    errors.push(format!("{path} has unknown evidence class: {evidence}"));
                }
            }
            _ => errors.push(format!(
                "{path} lane and minimum_evidence_class must both be strings"
            )),
        }

        if let Some(reference_ids) = case.get("reference_ids").and_then(Value::as_array) {
            let mut used_here = BTreeSet::new();
            for reference in reference_ids {
                match reference.as_str() {
                    Some(reference) => {
                        if !source_ids.contains(reference) {
                            errors
                                .push(format!("{path} references unknown source ID: {reference}"));
                        }
                        if !used_here.insert(reference.to_owned()) {
                            errors.push(format!("{path} repeats reference source ID: {reference}"));
                        }
                    }
                    None => {
                        errors.push(format!("{path}.reference_ids entries must all be strings"))
                    }
                }
            }
        }

        for field in ["title", "expected_behavior", "claim_limit"] {
            match case.get(field).and_then(Value::as_str) {
                Some(value) if !value.trim().is_empty() => {}
                _ => errors.push(format!("{path}.{field} must be a non-empty string")),
            }
        }

        for field in ["metrics", "failure_modes"] {
            let values = match case.get(field).and_then(Value::as_array) {
                Some(values) => values,
                None => {
                    errors.push(format!("{path}.{field} must be an array"));
                    continue;
                }
            };
            let mut unique = BTreeSet::new();
            for value in values {
                match value.as_str() {
                    Some(value) if !value.trim().is_empty() => {
                        if !unique.insert(value.to_owned()) {
                            errors
                                .push(format!("{path}.{field} contains duplicate value: {value}"));
                        }
                    }
                    _ => errors.push(format!("{path}.{field} entries must be non-empty strings")),
                }
            }
        }
    }

    errors
}

fn expect_string(value: &Value, key: &str, expected: &str, path: &str, errors: &mut Vec<String>) {
    match value.get(key).and_then(Value::as_str) {
        Some(actual) if actual == expected => {}
        Some(actual) => errors.push(format!("{path}.{key} must remain {expected}, got {actual}")),
        None => errors.push(format!("{path}.{key} must be a string")),
    }
}

fn is_valid_iso_date(value: &str) -> bool {
    let mut parts = value.split('-');
    let (Some(year), Some(month), Some(day), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };

    if year.len() != 4
        || month.len() != 2
        || day.len() != 2
        || !year.bytes().all(|byte| byte.is_ascii_digit())
        || !month.bytes().all(|byte| byte.is_ascii_digit())
        || !day.bytes().all(|byte| byte.is_ascii_digit())
    {
        return false;
    }

    let (Ok(year), Ok(month), Ok(day)) = (
        year.parse::<u32>(),
        month.parse::<u32>(),
        day.parse::<u32>(),
    ) else {
        return false;
    };

    if !(1..=12).contains(&month) {
        return false;
    }

    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return false,
    };

    (1..=days_in_month).contains(&day)
}

fn is_case_id(value: &str) -> bool {
    let Some(suffix) = value.strip_prefix("VIS-DISP-") else {
        return false;
    };
    !suffix.is_empty()
        && suffix.chars().all(|character| {
            character.is_ascii_uppercase() || character.is_ascii_digit() || character == '-'
        })
        && !suffix.starts_with('-')
        && !suffix.ends_with('-')
}

fn sha256_hex(input: &[u8]) -> String {
    let digest = Sha256::digest(input);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push_str(&format!("{byte:02x}"));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str =
        include_str!("../../../docs/engineering/visual-display-workload-corpus-v0.1.json");
    const SCHEMA: &str =
        include_str!("../../../docs/engineering/visual-display-workload-corpus.schema.json");

    fn values() -> (Value, Value) {
        (
            serde_json::from_str(MANIFEST).expect("checked-in manifest JSON must parse"),
            serde_json::from_str(SCHEMA).expect("checked-in schema JSON must parse"),
        )
    }

    fn revalidate(manifest: &Value, schema: &Value) -> Result<ValidationReport, Vec<String>> {
        validate_corpus_str(
            &serde_json::to_string(manifest).expect("manifest serializes"),
            &serde_json::to_string(schema).expect("schema serializes"),
        )
    }

    #[test]
    fn checked_in_corpus_validates_but_remains_unqualified() {
        let report = validate_corpus_str(MANIFEST, SCHEMA)
            .expect("the checked-in corpus should satisfy its schema and semantics");

        assert_eq!(report.corpus_id, EXPECTED_CORPUS_ID);
        assert_eq!(report.schema_version, EXPECTED_SCHEMA_VERSION);
        assert_eq!(report.claim_status, "CONTRACT_VALID_UNQUALIFIED");
        assert_eq!(report.case_count, EXPECTED_CASE_COUNT);
        assert_eq!(
            report.lane_counts.values().sum::<usize>(),
            EXPECTED_CASE_COUNT
        );
        assert_eq!(report.manifest_sha256.len(), 64);
        assert_eq!(report.schema_sha256.len(), 64);
    }

    #[test]
    fn rejects_invalid_calendar_date() {
        let (mut manifest, schema) = values();
        manifest["created_date"] = Value::String("2026-02-30".to_owned());

        let errors = revalidate(&manifest, &schema).expect_err("impossible date must reject");
        assert!(errors.iter().any(|error| error.contains("created_date")));
    }

    #[test]
    fn rejects_duplicate_case_id() {
        let (mut manifest, schema) = values();
        let first_id = manifest["cases"][0]["id"].clone();
        manifest["cases"][1]["id"] = first_id;

        let errors = revalidate(&manifest, &schema).expect_err("duplicate case ID must reject");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("duplicate case ID"))
        );
    }

    #[test]
    fn rejects_attempt_to_promote_case_status() {
        let (mut manifest, schema) = values();
        manifest["cases"][0]["status"] = Value::String("PASS".to_owned());

        let errors = revalidate(&manifest, &schema).expect_err("self-promotion must reject");
        assert!(errors.iter().any(|error| error.contains("status")));
    }

    #[test]
    fn rejects_lane_evidence_class_mismatch() {
        let (mut manifest, schema) = values();
        manifest["cases"][0]["minimum_evidence_class"] = Value::String("BENCH_MEASURED".to_owned());

        let errors = revalidate(&manifest, &schema).expect_err("lane mismatch must reject");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("requires evidence class"))
        );
    }

    #[test]
    fn rejects_unresolved_reference_id() {
        let (mut manifest, schema) = values();
        manifest["cases"][0]["reference_ids"][0] = Value::String("invented-source".to_owned());

        let errors = revalidate(&manifest, &schema).expect_err("unknown source must reject");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("unknown source ID"))
        );
    }

    #[test]
    fn rejects_mutated_schema_count() {
        let (manifest, mut schema) = values();
        schema["properties"]["case_count"]["const"] = Value::from(31);

        let errors = revalidate(&manifest, &schema).expect_err("mutated schema count must reject");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("schema properties.case_count.const"))
        );
    }

    #[test]
    fn rejects_missing_required_case_field() {
        let (mut manifest, schema) = values();
        manifest["cases"][0]
            .as_object_mut()
            .expect("case is an object")
            .remove("claim_limit");

        let errors = revalidate(&manifest, &schema).expect_err("missing field must reject");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("claim_limit") || error.contains("required"))
        );
    }
}
