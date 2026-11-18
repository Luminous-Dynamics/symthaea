// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::json;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::process::ExitCode;
use symthaea_landscape_review::{REVIEW_SCHEMA, REVIEW_SCHEMA_URI, validate_review};

const REPORT_SCHEMA: &str = "symthaea.landscape-review-validation-report.v0";
const REPORT_SCHEMA_URI: &str =
    "https://luminousdynamics.io/schemas/landscape-review-validation-report-v0.schema.json";
const MAX_INPUT_BYTES: usize = 8 * 1024 * 1024;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("Usage: landscape-review-check REVIEW.json");
        return ExitCode::from(64);
    };
    if args.next().is_some() {
        eprintln!("landscape-review-check accepts exactly one review file");
        return ExitCode::from(64);
    }
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("Could not read review artifact: {error}");
            return ExitCode::from(66);
        }
    };
    let report = validate_bytes(&bytes);
    match serde_json::to_string_pretty(&report) {
        Ok(value) => println!("{value}"),
        Err(error) => {
            eprintln!("Could not serialize validation report: {error}");
            return ExitCode::from(70);
        }
    }
    if report["valid"].as_bool() == Some(true) {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}

fn validate_bytes(bytes: &[u8]) -> serde_json::Value {
    let input_sha256 = format!("{:x}", Sha256::digest(bytes));
    let mut errors = Vec::new();
    if bytes.len() > MAX_INPUT_BYTES {
        errors.push(format!("input exceeds maximum of {MAX_INPUT_BYTES} bytes"));
    } else {
        match serde_json::from_slice(bytes) {
            Ok(value) => {
                if let Err(violations) = validate_review(&value) {
                    errors = violations;
                }
            }
            Err(error) => errors.push(format!("invalid JSON: {error}")),
        }
    }
    let valid = errors.is_empty();
    json!({
        "report_schema": REPORT_SCHEMA,
        "schema_uri": REPORT_SCHEMA_URI,
        "review_schema": REVIEW_SCHEMA,
        "review_schema_uri": REVIEW_SCHEMA_URI,
        "input_sha256": input_sha256,
        "input_bytes": bytes.len(),
        "valid": valid,
        "errors": errors,
        "interpretation_limit": "structural-policy-validation-does-not-prove-scientific-validity-or-impact"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_unranked_fixture_produces_repeatable_valid_report() {
        let bytes = include_bytes!("../../../../../fixtures/landscape/review-unranked-v0.json");
        let first = validate_bytes(bytes);
        let second = validate_bytes(bytes);
        assert_eq!(first, second);
        assert_eq!(first["valid"], true);
        assert_eq!(first["report_schema"], REPORT_SCHEMA);
        assert_eq!(first["schema_uri"], REPORT_SCHEMA_URI);
        assert_eq!(first["errors"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn invalid_json_emits_deterministic_hashed_failure_report() {
        let report = validate_bytes(b"{invalid json");
        assert_eq!(report["valid"], false);
        assert_eq!(report["input_sha256"].as_str().unwrap().len(), 64);
        assert!(
            report["errors"][0]
                .as_str()
                .unwrap()
                .contains("invalid JSON")
        );
    }

    #[test]
    fn overlarge_input_is_rejected_before_parsing() {
        let bytes = vec![b' '; MAX_INPUT_BYTES + 1];
        let report = validate_bytes(&bytes);
        assert_eq!(report["valid"], false);
        assert!(
            report["errors"][0]
                .as_str()
                .unwrap()
                .contains("exceeds maximum")
        );
    }
}

#[cfg(test)]
mod schema_tests {
    use super::*;

    #[test]
    fn report_schema_uri_matches_schema() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../schemas/landscape-review-validation-report-v0.schema.json"
        ))
        .expect("report JSON Schema must be valid JSON");
        assert_eq!(schema["$id"], REPORT_SCHEMA_URI);
        assert_eq!(
            schema["properties"]["report_schema"]["const"],
            REPORT_SCHEMA
        );
    }
}
