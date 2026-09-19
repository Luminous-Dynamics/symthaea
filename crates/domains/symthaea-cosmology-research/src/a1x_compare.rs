// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A1X cross-implementation fixed-point agreement comparator.
//!
//! This program performs no new cosmological fit. It compares a qualified A1R
//! Rust result with a qualified A1C released-Cobaya result for the same frozen
//! subject under preregistered implementation-agreement thresholds.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_JSON_BYTES: u64 = 8 * 1024 * 1024;
const PROTOCOL: &str = "DE-001A1X-CROSS-IMPLEMENTATION-v1";
const AUTHORITY: &str = "cross-implementation-fixed-point-agreement-only";
const SCIENTIFIC_CLAIM: &str = "NONE";
const A1Q_PROTOCOL: &str = "DE-001A1Q-BUNDLE-INTEGRITY-v1";
const A1R_PROTOCOL: &str = "DE-001A1R-RUST-ORACLE-v1";
const A1C_PROTOCOL: &str = "DE-001A1C-COBAYA-RESULT-v1";
const A1C_QUAL_PROTOCOL: &str = "DE-001A1C-RESULT-QUALIFICATION-v1";
const SPEC_RELATIVE: &str =
    "crates/domains/symthaea-cosmology-research/references/de001a_a1x_agreement_v1.json";

const ROW_COUNT: usize = 13;
const OMEGA_M: f64 = 0.297_177_87;
const H_R_D_MPC: f64 = 101.547_86;
const REFERENCE_CHI2: f64 = 10.282_299;
const REPRODUCTION_TOLERANCE: f64 = 0.01;
const MAX_PREDICTION_ABS_DELTA: f64 = 1.0e-8;
const MAX_PREDICTION_REL_DELTA: f64 = 1.0e-10;
const MAX_CHI2_ABS_DELTA: f64 = 1.0e-8;
const MEAN_SHA256: &str = "9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585";
const COV_SHA256: &str = "252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509";

#[derive(Debug, Serialize)]
struct RowDelta {
    index: usize,
    a1r_prediction: f64,
    a1c_prediction: f64,
    absolute_delta: f64,
    relative_delta: f64,
}

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    interpretation_authority: &'static str,
    a2_execution_authorized: bool,
    comparator_head: String,
    comparator_tree: String,
    subject_head: String,
    subject_tree: String,
    agreement_spec_sha256: String,
    a1q_receipt_sha256: String,
    a1r_receipt_sha256: String,
    a1c_result_sha256: String,
    a1c_qualification_sha256: String,
    a1r_reproduction_verdict: String,
    a1c_reproduction_verdict: String,
    a1r_chi2_bao: f64,
    a1c_chi2_bao: f64,
    chi2_absolute_delta: f64,
    max_prediction_absolute_delta: f64,
    max_prediction_relative_delta: f64,
    prediction_absolute_threshold: f64,
    prediction_relative_threshold: f64,
    chi2_absolute_threshold: f64,
    same_reproduction_verdict: bool,
    rows: Vec<RowDelta>,
    disagreement_reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct InvalidReceipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    a2_execution_authorized: bool,
    error: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read_regular_file(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{}: metadata failed: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!("{}: symlinks are forbidden", path.display()));
    }
    if !metadata.file_type().is_file() {
        return Err(format!("{}: not a regular file", path.display()));
    }
    if metadata.len() > MAX_JSON_BYTES {
        return Err(format!("{}: file too large", path.display()));
    }
    let bytes = fs::read(path).map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{}: file changed size while reading", path.display()));
    }
    Ok(bytes)
}

fn parse_json(bytes: &[u8], label: &str) -> Result<Value, String> {
    serde_json::from_slice(bytes).map_err(|error| format!("invalid {label} JSON: {error}"))
}

fn run_text(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map(|text| text.trim().to_owned())
        .map_err(|error| format!("{program} output was not UTF-8: {error}"))
}

fn comparator_identity() -> Result<(PathBuf, String, String), String> {
    let root = run_text("git", &["rev-parse", "--show-toplevel"])?;
    let head = run_text("git", &["rev-parse", "HEAD"])?;
    let tree = run_text("git", &["rev-parse", "HEAD^{tree}"])?;
    if root.is_empty() || head.is_empty() || tree.is_empty() {
        return Err("git returned an empty comparator identity".into());
    }
    Ok((PathBuf::from(root), head, tree))
}

fn field_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or non-string field: {key}"))
}

fn field_bool(value: &Value, key: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing or non-boolean field: {key}"))
}

fn field_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or non-u64 field: {key}"))
}

fn field_f64(value: &Value, key: &str) -> Result<f64, String> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| format!("missing or non-finite numeric field: {key}"))
}

fn nested<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value
        .get(key)
        .ok_or_else(|| format!("missing nested field: {key}"))
}

fn ordered_strings_equal(value: &Value, expected: &[&str]) -> bool {
    let Some(array) = value.as_array() else {
        return false;
    };
    array.len() == expected.len()
        && array
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.as_str() == Some(*expected))
}

fn exact_f64(left: f64, right: f64) -> bool {
    left.to_bits() == right.to_bits()
}

fn valid_lane_verdict(value: &str) -> bool {
    matches!(value, "PASS" | "NEGATIVE")
}

fn rederive_verdict(chi2: f64) -> &'static str {
    if (chi2 - REFERENCE_CHI2).abs() <= REPRODUCTION_TOLERANCE {
        "PASS"
    } else {
        "NEGATIVE"
    }
}

fn relative_delta(left: f64, right: f64) -> f64 {
    let absolute = (left - right).abs();
    let denominator = left.abs().max(right.abs());
    if denominator == 0.0 {
        0.0
    } else {
        absolute / denominator
    }
}

fn validate_spec(spec: &Value) -> Result<(), String> {
    if spec.get("schema_version").and_then(Value::as_u64) != Some(1)
        || field_str(spec, "protocol")? != PROTOCOL
        || field_str(spec, "status")? != "preregistered-comparator-only"
        || field_str(spec, "scientific_claim")? != SCIENTIFIC_CLAIM
        || field_str(spec, "authority")? != AUTHORITY
    {
        return Err("A1X specification identity drifted".into());
    }

    let inputs = nested(spec, "inputs")?;
    if field_str(inputs, "a1q_protocol")? != A1Q_PROTOCOL
        || field_str(inputs, "a1r_protocol")? != A1R_PROTOCOL
        || field_str(inputs, "a1c_protocol")? != A1C_PROTOCOL
        || field_str(inputs, "a1c_qualification_protocol")? != A1C_QUAL_PROTOCOL
        || !ordered_strings_equal(nested(inputs, "accepted_lane_verdicts")?, &["PASS", "NEGATIVE"])
    {
        return Err("A1X input contract drifted".into());
    }

    let lineage = nested(spec, "lineage")?;
    for key in [
        "a1r_sha_must_equal_a1q_primary_receipt_sha",
        "a1c_qualification_must_bind_a1c_result_sha",
        "a1c_qualification_must_bind_same_a1q_receipt",
        "a1c_subject_must_equal_a1q_subject",
        "a1c_qualification_subject_must_equal_a1q_subject",
        "a1q_reproduction_verdict_must_equal_a1r_verdict",
        "a1c_embedded_a1q_verdict_must_equal_a1q_verdict",
        "a1r_and_a1c_must_share_a0_receipt",
        "a1r_and_a1c_must_share_point_manifest",
    ] {
        if !field_bool(lineage, key)? {
            return Err(format!("A1X lineage rule {key} is not frozen true"));
        }
    }

    let subject = nested(spec, "subject")?;
    if field_u64(subject, "row_count")? != ROW_COUNT as u64
        || !exact_f64(field_f64(subject, "omega_m")?, OMEGA_M)
        || !exact_f64(field_f64(subject, "h_r_d_mpc")?, H_R_D_MPC)
        || !exact_f64(field_f64(subject, "reference_chi2_bao")?, REFERENCE_CHI2)
        || !exact_f64(
            field_f64(subject, "absolute_reproduction_tolerance")?,
            REPRODUCTION_TOLERANCE,
        )
        || field_str(subject, "dataset_mean_sha256")? != MEAN_SHA256
        || field_str(subject, "dataset_covariance_sha256")? != COV_SHA256
    {
        return Err("A1X fixed subject drifted".into());
    }

    let thresholds = nested(spec, "agreement_thresholds")?;
    if !exact_f64(
        field_f64(thresholds, "max_prediction_absolute_delta")?,
        MAX_PREDICTION_ABS_DELTA,
    ) || !exact_f64(
        field_f64(thresholds, "max_prediction_relative_delta")?,
        MAX_PREDICTION_REL_DELTA,
    ) || !exact_f64(
        field_f64(thresholds, "max_chi2_absolute_delta")?,
        MAX_CHI2_ABS_DELTA,
    ) || !field_bool(thresholds, "require_same_reproduction_verdict")?
        || field_str(thresholds, "relative_delta_definition")?
            != "abs(a-b)/max(abs(a),abs(b)); zero when both values are zero"
    {
        return Err("A1X agreement thresholds drifted".into());
    }

    let outcomes = nested(spec, "outcomes")?;
    for key in ["AGREE", "DISAGREE", "INVALID"] {
        if field_str(outcomes, key)?.trim().is_empty() {
            return Err(format!("A1X outcome description {key} is empty"));
        }
    }

    let promotion = nested(spec, "promotion")?;
    let excluded = [
        "optimizer reproduction",
        "LambdaCDM validity",
        "dynamic dark energy",
        "observational anomaly",
        "phenomenological dark-energy dynamics",
        "physical mechanism",
    ];
    if field_str(promotion, "agree_authority")? != "fixed-point implementation agreement only"
        || field_str(promotion, "disagree_authority")?
            != "implementation-disagreement diagnosis only"
        || field_bool(promotion, "a2_execution_authorized")?
        || !ordered_strings_equal(nested(promotion, "does_not_establish")?, &excluded)
    {
        return Err("A1X promotion boundary drifted".into());
    }
    Ok(())
}

fn a1r_predictions(value: &Value) -> Result<Vec<f64>, String> {
    let rows = value
        .get("predictions")
        .and_then(Value::as_array)
        .ok_or_else(|| "A1R predictions is not an array".to_owned())?;
    if rows.len() != ROW_COUNT {
        return Err("A1R prediction row count is not 13".into());
    }
    rows.iter()
        .map(|row| {
            row.get("predicted")
                .and_then(Value::as_f64)
                .filter(|number| number.is_finite())
                .ok_or_else(|| "A1R prediction contains a non-finite value".to_owned())
        })
        .collect()
}

fn a1c_predictions(value: &Value) -> Result<Vec<f64>, String> {
    let rows = value
        .get("prediction_vector")
        .and_then(Value::as_array)
        .ok_or_else(|| "A1C prediction_vector is not an array".to_owned())?;
    if rows.len() != ROW_COUNT {
        return Err("A1C prediction row count is not 13".into());
    }
    rows.iter()
        .map(|row| {
            row.as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| "A1C prediction contains a non-finite value".to_owned())
        })
        .collect()
}

fn execute(
    a1q_path: &Path,
    a1r_path: &Path,
    a1c_path: &Path,
    a1c_qualification_path: &Path,
) -> Result<Receipt, String> {
    let (root, comparator_head, comparator_tree) = comparator_identity()?;
    let spec_bytes = read_regular_file(&root.join(SPEC_RELATIVE))?;
    let a1q_bytes = read_regular_file(a1q_path)?;
    let a1r_bytes = read_regular_file(a1r_path)?;
    let a1c_bytes = read_regular_file(a1c_path)?;
    let qualification_bytes = read_regular_file(a1c_qualification_path)?;

    let spec = parse_json(&spec_bytes, "A1X specification")?;
    let a1q = parse_json(&a1q_bytes, "A1Q receipt")?;
    let a1r = parse_json(&a1r_bytes, "A1R receipt")?;
    let a1c = parse_json(&a1c_bytes, "A1C result")?;
    let qualification = parse_json(&qualification_bytes, "A1C qualification receipt")?;
    validate_spec(&spec)?;

    if field_str(&a1q, "protocol")? != A1Q_PROTOCOL
        || field_str(&a1q, "verdict")? != "PASS"
        || field_str(&a1q, "scientific_claim")? != SCIENTIFIC_CLAIM
        || field_str(&a1q, "authority")? != "evidence-bundle-integrity-only"
    {
        return Err("A1Q input is not qualified evidence".into());
    }
    let subject_head = field_str(&a1q, "subject_head")?.to_owned();
    let subject_tree = field_str(&a1q, "subject_tree")?.to_owned();
    let a1q_reproduction = field_str(&a1q, "reproduction_verdict")?;
    if !valid_lane_verdict(a1q_reproduction) {
        return Err("A1Q reproduction verdict is not PASS/NEGATIVE".into());
    }

    let a1r_sha256 = sha256_hex(&a1r_bytes);
    if field_str(&a1q, "a1r_primary_receipt_sha256")? != a1r_sha256 {
        return Err("A1R receipt SHA-256 is not the primary receipt qualified by A1Q".into());
    }
    if field_str(&a1r, "protocol")? != A1R_PROTOCOL
        || field_str(&a1r, "scientific_claim")? != SCIENTIFIC_CLAIM
        || field_str(&a1r, "authority")? != "fixed-point-reproduction-sanity-only"
        || !valid_lane_verdict(field_str(&a1r, "verdict")?)
        || field_str(&a1r, "verdict")? != a1q_reproduction
    {
        return Err("A1R receipt does not match the qualified A1Q reproduction lane".into());
    }

    let a1q_sha256 = sha256_hex(&a1q_bytes);
    let a1c_sha256 = sha256_hex(&a1c_bytes);
    let qualification_sha256 = sha256_hex(&qualification_bytes);
    if field_str(&qualification, "protocol")? != A1C_QUAL_PROTOCOL
        || field_str(&qualification, "verdict")? != "PASS"
        || field_str(&qualification, "scientific_claim")? != SCIENTIFIC_CLAIM
        || field_str(&qualification, "authority")? != "a1c-result-integrity-only"
        || !field_bool(&qualification, "a1x_comparison_authorized")?
        || field_bool(&qualification, "a2_execution_authorized")?
        || field_str(&qualification, "a1c_result_sha256")? != a1c_sha256
        || field_str(&qualification, "a1q_receipt_sha256")? != a1q_sha256
        || field_str(&qualification, "subject_head")? != subject_head
        || field_str(&qualification, "subject_tree")? != subject_tree
    {
        return Err("A1C qualification does not authorize this result for A1X".into());
    }

    if field_str(&a1c, "protocol")? != A1C_PROTOCOL
        || field_str(&a1c, "scientific_claim")? != SCIENTIFIC_CLAIM
        || field_str(&a1c, "authority")? != "released-likelihood-fixed-point-reproduction-only"
        || !valid_lane_verdict(field_str(&a1c, "verdict")?)
        || field_str(&a1c, "subject_head")? != subject_head
        || field_str(&a1c, "subject_tree")? != subject_tree
        || field_str(&a1c, "a1q_receipt_sha256")? != a1q_sha256
        || field_str(&a1c, "a1q_reproduction_verdict")? != a1q_reproduction
        || field_str(&qualification, "reproduction_verdict")? != field_str(&a1c, "verdict")?
    {
        return Err("A1C result does not match its qualified A1Q subject".into());
    }

    if field_str(&a1r, "a0_receipt_sha256")? != field_str(&a1c, "a0_receipt_sha256")?
        || field_str(&qualification, "a0_receipt_sha256")? != field_str(&a1c, "a0_receipt_sha256")?
        || field_str(&a1r, "point_manifest_sha256")?
            != field_str(&a1c, "point_manifest_sha256")?
        || field_str(&qualification, "point_manifest_sha256")?
            != field_str(&a1c, "point_manifest_sha256")?
    {
        return Err("A1R and A1C are not bound to the same A0/point lineage".into());
    }

    let a1r_parameters = nested(&a1r, "parameters")?;
    if !exact_f64(field_f64(a1r_parameters, "omega_m")?, OMEGA_M)
        || !exact_f64(field_f64(a1r_parameters, "h_r_d_mpc")?, H_R_D_MPC)
        || !exact_f64(field_f64(&a1r, "reference_chi2_bao")?, REFERENCE_CHI2)
        || !exact_f64(field_f64(&a1r, "absolute_tolerance")?, REPRODUCTION_TOLERANCE)
        || !exact_f64(field_f64(&a1c, "omega_m")?, OMEGA_M)
        || !exact_f64(field_f64(&a1c, "h_r_d_mpc")?, H_R_D_MPC)
        || !exact_f64(field_f64(&a1c, "reference_chi2_bao")?, REFERENCE_CHI2)
        || !exact_f64(field_f64(&a1c, "absolute_tolerance")?, REPRODUCTION_TOLERANCE)
        || field_str(&a1c, "dataset_mean_sha256")? != MEAN_SHA256
        || field_str(&a1c, "dataset_covariance_sha256")? != COV_SHA256
    {
        return Err("A1R/A1C fixed subjects are not the preregistered DE-001A1 subject".into());
    }

    let a1r_chi2 = field_f64(&a1r, "computed_chi2_bao")?;
    let a1c_chi2 = field_f64(&a1c, "chi2_bao")?;
    let a1r_delta = field_f64(&a1r, "absolute_delta_chi2")?;
    let a1c_delta = field_f64(&a1c, "absolute_delta_chi2")?;
    if !exact_f64(a1r_delta, (a1r_chi2 - REFERENCE_CHI2).abs())
        || !exact_f64(a1c_delta, (a1c_chi2 - REFERENCE_CHI2).abs())
        || field_str(&a1r, "verdict")? != rederive_verdict(a1r_chi2)
        || field_str(&a1c, "verdict")? != rederive_verdict(a1c_chi2)
        || !exact_f64(field_f64(&qualification, "chi2_bao")?, a1c_chi2)
    {
        return Err("A1R/A1C stored reproduction decisions do not rederive".into());
    }

    let a1r_vector = a1r_predictions(&a1r)?;
    let a1c_vector = a1c_predictions(&a1c)?;
    let mut rows = Vec::with_capacity(ROW_COUNT);
    let mut max_abs: f64 = 0.0;
    let mut max_rel: f64 = 0.0;
    for (index, (&rust_value, &cobaya_value)) in a1r_vector.iter().zip(&a1c_vector).enumerate() {
        let absolute = (rust_value - cobaya_value).abs();
        let relative = relative_delta(rust_value, cobaya_value);
        max_abs = max_abs.max(absolute);
        max_rel = max_rel.max(relative);
        rows.push(RowDelta {
            index,
            a1r_prediction: rust_value,
            a1c_prediction: cobaya_value,
            absolute_delta: absolute,
            relative_delta: relative,
        });
    }

    let chi2_delta = (a1r_chi2 - a1c_chi2).abs();
    let same_verdict = field_str(&a1r, "verdict")? == field_str(&a1c, "verdict")?;
    let mut disagreement_reasons = Vec::new();
    if !same_verdict {
        disagreement_reasons.push("reproduction-verdict-mismatch".to_owned());
    }
    if max_abs > MAX_PREDICTION_ABS_DELTA {
        disagreement_reasons.push("prediction-absolute-threshold-exceeded".to_owned());
    }
    if max_rel > MAX_PREDICTION_REL_DELTA {
        disagreement_reasons.push("prediction-relative-threshold-exceeded".to_owned());
    }
    if chi2_delta > MAX_CHI2_ABS_DELTA {
        disagreement_reasons.push("chi2-threshold-exceeded".to_owned());
    }

    let (verdict, interpretation_authority, exit_agree) = if disagreement_reasons.is_empty() {
        ("AGREE", "fixed-point implementation agreement only", true)
    } else {
        (
            "DISAGREE",
            "implementation-disagreement diagnosis only",
            false,
        )
    };

    let receipt = Receipt {
        protocol: PROTOCOL,
        verdict,
        scientific_claim: SCIENTIFIC_CLAIM,
        authority: AUTHORITY,
        interpretation_authority,
        a2_execution_authorized: false,
        comparator_head,
        comparator_tree,
        subject_head,
        subject_tree,
        agreement_spec_sha256: sha256_hex(&spec_bytes),
        a1q_receipt_sha256: a1q_sha256,
        a1r_receipt_sha256: a1r_sha256,
        a1c_result_sha256: a1c_sha256,
        a1c_qualification_sha256: qualification_sha256,
        a1r_reproduction_verdict: field_str(&a1r, "verdict")?.to_owned(),
        a1c_reproduction_verdict: field_str(&a1c, "verdict")?.to_owned(),
        a1r_chi2_bao: a1r_chi2,
        a1c_chi2_bao: a1c_chi2,
        chi2_absolute_delta: chi2_delta,
        max_prediction_absolute_delta: max_abs,
        max_prediction_relative_delta: max_rel,
        prediction_absolute_threshold: MAX_PREDICTION_ABS_DELTA,
        prediction_relative_threshold: MAX_PREDICTION_REL_DELTA,
        chi2_absolute_threshold: MAX_CHI2_ABS_DELTA,
        same_reproduction_verdict: same_verdict,
        rows,
        disagreement_reasons,
    };

    if exit_agree {
        Ok(receipt)
    } else {
        Err(serde_json::to_string(&receipt)
            .map_err(|error| format!("failed to serialize DISAGREE receipt: {error}"))?)
    }
}

fn write_json<T: Serialize>(value: &T) -> Result<(), String> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer_pretty(&mut out, value)
        .map_err(|error| format!("failed to serialize A1X receipt: {error}"))?;
    writeln!(out).map_err(|error| format!("failed to write A1X receipt: {error}"))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        let invalid = InvalidReceipt {
            protocol: PROTOCOL,
            verdict: "INVALID",
            scientific_claim: SCIENTIFIC_CLAIM,
            authority: AUTHORITY,
            a2_execution_authorized: false,
            error: "usage: de001a-a1x-compare A1Q.json A1R.json A1C.json A1C-QUALIFICATION.json".into(),
        };
        let _ = write_json(&invalid);
        std::process::exit(2);
    }

    match execute(
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
    ) {
        Ok(receipt) => {
            let _ = write_json(&receipt);
            std::process::exit(0);
        }
        Err(error) => {
            if let Ok(value) = serde_json::from_str::<Value>(&error) {
                if value.get("verdict").and_then(Value::as_str) == Some("DISAGREE") {
                    let stdout = io::stdout();
                    let mut out = stdout.lock();
                    let _ = serde_json::to_writer_pretty(&mut out, &value);
                    let _ = writeln!(out);
                    std::process::exit(1);
                }
            }
            let invalid = InvalidReceipt {
                protocol: PROTOCOL,
                verdict: "INVALID",
                scientific_claim: SCIENTIFIC_CLAIM,
                authority: AUTHORITY,
                a2_execution_authorized: false,
                error,
            };
            let _ = write_json(&invalid);
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_relative_delta_handles_zero() {
        assert_eq!(relative_delta(0.0, 0.0), 0.0);
        assert_eq!(relative_delta(2.0, 2.0), 0.0);
        assert!((relative_delta(2.0, 1.0) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn reproduction_verdict_boundary_is_frozen() {
        assert_eq!(rederive_verdict(REFERENCE_CHI2), "PASS");
        assert_eq!(rederive_verdict(REFERENCE_CHI2 + REPRODUCTION_TOLERANCE), "PASS");
        assert_eq!(
            rederive_verdict(REFERENCE_CHI2 + REPRODUCTION_TOLERANCE + 1.0e-6),
            "NEGATIVE"
        );
    }
}
