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
    let input_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let mut errors = Vec::new();
    if bytes.len() > MAX_INPUT_BYTES {
        errors.push(format!("input exceeds maximum of {MAX_INPUT_BYTES} bytes"));
    } else {
        match serde_json::from_slice(&bytes) {
            Ok(value) => {
                if let Err(violations) = validate_review(&value) {
                    errors = violations;
                }
            }
            Err(error) => errors.push(format!("invalid JSON: {error}")),
        }
    }
    let valid = errors.is_empty();
    let report = json!({
        "report_schema": REPORT_SCHEMA,
        "schema_uri": REPORT_SCHEMA_URI,
        "review_schema": REVIEW_SCHEMA,
        "review_schema_uri": REVIEW_SCHEMA_URI,
        "input_sha256": input_sha256,
        "input_bytes": bytes.len(),
        "valid": errors.is_empty(),
        "errors": errors,
        "interpretation_limit": "structural-policy-validation-does-not-prove-scientific-validity-or-impact"
    });
    match serde_json::to_string_pretty(&report) {
        Ok(value) => println!("{value}"),
        Err(error) => {
            eprintln!("Could not serialize validation report: {error}");
            return ExitCode::from(70);
        }
    }
    if valid { ExitCode::SUCCESS } else { ExitCode::from(2) }
}
