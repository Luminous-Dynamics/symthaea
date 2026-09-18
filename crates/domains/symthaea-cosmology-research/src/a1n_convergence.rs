// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A1N numerical-convergence qualifier.
//!
//! This program performs no optimization and introduces no new cosmological
//! parameters. It compares two already-produced A1R fixed-point receipts whose
//! manifests differ only in Simpson integration resolution.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

const MAX_SMALL_FILE_BYTES: u64 = 1024 * 1024;
const EXPECTED_PREDICTIONS: usize = 13;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConvergenceSpec {
    schema_version: u32,
    protocol: String,
    scientific_claim: String,
    authority: String,
    primary_simpson_subdivisions: u64,
    refined_simpson_subdivisions: u64,
    max_absolute_prediction_delta: f64,
    max_relative_prediction_delta: f64,
    max_absolute_chi2_delta: f64,
    development_history: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
#[serde(deny_unknown_fields)]
struct Parameters {
    omega_m: f64,
    h_r_d_mpc: f64,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
struct Independence {
    measurement_data_independent: bool,
    covariance_independent: bool,
    background_implementation_independent: bool,
    gaussian_likelihood_implementation_independent: bool,
    scope: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
struct Prediction {
    z: f64,
    observable: String,
    observed: f64,
    predicted: f64,
    residual: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct A1rReceipt {
    protocol: String,
    point_manifest_sha256: String,
    a0_receipt_sha256: String,
    verdict: String,
    scientific_claim: String,
    authority: String,
    model: String,
    parameters: Parameters,
    reference_chi2_bao: f64,
    computed_chi2_bao: f64,
    absolute_delta_chi2: f64,
    absolute_tolerance: f64,
    predictions: Vec<Prediction>,
    independence: Independence,
}

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: String,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: String,
    spec_sha256: String,
    primary_manifest_sha256: String,
    refined_manifest_sha256: String,
    primary_receipt_sha256: String,
    refined_receipt_sha256: String,
    a0_receipt_sha256: String,
    primary_prediction_sha256: String,
    refined_prediction_sha256: String,
    primary_chi2_bao: f64,
    refined_chi2_bao: f64,
    absolute_chi2_delta: f64,
    max_absolute_prediction_delta: f64,
    max_relative_prediction_delta: f64,
    allowed_absolute_chi2_delta: f64,
    allowed_absolute_prediction_delta: f64,
    allowed_relative_prediction_delta: f64,
    primary_simpson_subdivisions: u64,
    refined_simpson_subdivisions: u64,
    development_history: String,
}

#[derive(Debug, Serialize)]
struct InvalidReceipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    error: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
    if metadata.len() > MAX_SMALL_FILE_BYTES {
        return Err(format!("{}: file exceeds size limit", path.display()));
    }
    let bytes = fs::read(path).map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{}: size changed while reading", path.display()));
    }
    Ok(bytes)
}

fn validate_spec(spec: &ConvergenceSpec) -> Result<(), String> {
    if spec.schema_version != 1
        || spec.protocol != "DE-001A1N-NUMERICAL-CONVERGENCE-v1"
        || spec.scientific_claim != "NONE"
        || spec.authority != "numerical-convergence-qualification-only"
    {
        return Err("invalid A1N identity/authority contract".into());
    }
    if spec.primary_simpson_subdivisions < 32
        || spec.primary_simpson_subdivisions % 2 != 0
        || spec.refined_simpson_subdivisions != spec.primary_simpson_subdivisions * 2
    {
        return Err("invalid primary/refined quadrature contract".into());
    }
    for (name, value) in [
        ("max_absolute_prediction_delta", spec.max_absolute_prediction_delta),
        ("max_relative_prediction_delta", spec.max_relative_prediction_delta),
        ("max_absolute_chi2_delta", spec.max_absolute_chi2_delta),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{name} must be finite and positive"));
        }
    }
    if spec.development_history.trim().is_empty() {
        return Err("development_history must not be empty".into());
    }
    Ok(())
}

fn normalized_a1r_manifest(bytes: &[u8], expected_subdivisions: u64) -> Result<Value, String> {
    let mut value: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid A1R manifest JSON: {error}"))?;
    if value["schema_version"] != 1
        || value["protocol"] != "DE-001A1R-RUST-ORACLE-v1"
        || value["scientific_claim"] != "NONE"
    {
        return Err("unexpected A1R manifest identity".into());
    }
    let subdivisions = value["numerics"]["simpson_subdivisions"]
        .as_u64()
        .ok_or_else(|| "A1R manifest lacks integer Simpson subdivisions".to_owned())?;
    if subdivisions != expected_subdivisions {
        return Err(format!(
            "expected {expected_subdivisions} Simpson subdivisions, got {subdivisions}"
        ));
    }
    value["numerics"]["simpson_subdivisions"] = Value::from(0_u64);
    Ok(value)
}

fn validate_a1r_receipt(receipt: &A1rReceipt) -> Result<(), String> {
    if receipt.protocol != "DE-001A1R-RUST-ORACLE-v1"
        || receipt.scientific_claim != "NONE"
        || receipt.authority != "fixed-point-reproduction-sanity-only"
        || receipt.model != "flat-lambda-cdm-bao-only-late-background"
        || !matches!(receipt.verdict.as_str(), "PASS" | "NEGATIVE")
        || !is_lower_hex_sha256(&receipt.point_manifest_sha256)
        || !is_lower_hex_sha256(&receipt.a0_receipt_sha256)
        || receipt.predictions.len() != EXPECTED_PREDICTIONS
    {
        return Err("invalid A1R receipt identity or shape".into());
    }
    if !receipt.parameters.omega_m.is_finite()
        || !receipt.parameters.h_r_d_mpc.is_finite()
        || !receipt.reference_chi2_bao.is_finite()
        || !receipt.computed_chi2_bao.is_finite()
        || !receipt.absolute_delta_chi2.is_finite()
        || !receipt.absolute_tolerance.is_finite()
        || receipt.absolute_tolerance <= 0.0
    {
        return Err("A1R receipt contains invalid numerical values".into());
    }
    if receipt.independence.measurement_data_independent
        || receipt.independence.covariance_independent
        || !receipt.independence.background_implementation_independent
        || !receipt.independence.gaussian_likelihood_implementation_independent
        || receipt.independence.scope.trim().is_empty()
    {
        return Err("A1R receipt has invalid independence classification".into());
    }
    for prediction in &receipt.predictions {
        if !prediction.z.is_finite()
            || !prediction.observed.is_finite()
            || !prediction.predicted.is_finite()
            || !prediction.residual.is_finite()
            || prediction.observable.trim().is_empty()
        {
            return Err("A1R prediction contains invalid values".into());
        }
        let recomputed_residual = prediction.predicted - prediction.observed;
        if (recomputed_residual - prediction.residual).abs() > 1e-12 {
            return Err("A1R prediction residual is internally inconsistent".into());
        }
    }
    Ok(())
}

fn prediction_digest(predictions: &[Prediction]) -> Result<String, String> {
    let bytes = serde_json::to_vec(predictions)
        .map_err(|error| format!("prediction serialization failed: {error}"))?;
    Ok(sha256_hex(&bytes))
}

fn execute(
    spec_path: &Path,
    primary_manifest_path: &Path,
    refined_manifest_path: &Path,
    primary_receipt_path: &Path,
    refined_receipt_path: &Path,
) -> Result<Receipt, String> {
    let spec_bytes = read_regular_file(spec_path)?;
    let spec: ConvergenceSpec = serde_json::from_slice(&spec_bytes)
        .map_err(|error| format!("invalid convergence spec JSON: {error}"))?;
    validate_spec(&spec)?;

    let primary_manifest_bytes = read_regular_file(primary_manifest_path)?;
    let refined_manifest_bytes = read_regular_file(refined_manifest_path)?;
    let primary_manifest_sha256 = sha256_hex(&primary_manifest_bytes);
    let refined_manifest_sha256 = sha256_hex(&refined_manifest_bytes);
    let primary_normalized = normalized_a1r_manifest(
        &primary_manifest_bytes,
        spec.primary_simpson_subdivisions,
    )?;
    let refined_normalized = normalized_a1r_manifest(
        &refined_manifest_bytes,
        spec.refined_simpson_subdivisions,
    )?;
    if primary_normalized != refined_normalized {
        return Err("primary/refined A1R manifests differ beyond Simpson resolution".into());
    }

    let primary_receipt_bytes = read_regular_file(primary_receipt_path)?;
    let refined_receipt_bytes = read_regular_file(refined_receipt_path)?;
    let primary_receipt_sha256 = sha256_hex(&primary_receipt_bytes);
    let refined_receipt_sha256 = sha256_hex(&refined_receipt_bytes);
    let primary: A1rReceipt = serde_json::from_slice(&primary_receipt_bytes)
        .map_err(|error| format!("invalid primary A1R receipt JSON: {error}"))?;
    let refined: A1rReceipt = serde_json::from_slice(&refined_receipt_bytes)
        .map_err(|error| format!("invalid refined A1R receipt JSON: {error}"))?;
    validate_a1r_receipt(&primary)?;
    validate_a1r_receipt(&refined)?;

    if primary.point_manifest_sha256 != primary_manifest_sha256
        || refined.point_manifest_sha256 != refined_manifest_sha256
    {
        return Err("A1R receipt is not bound to the supplied point manifest".into());
    }
    if primary.a0_receipt_sha256 != refined.a0_receipt_sha256
        || primary.parameters != refined.parameters
        || primary.reference_chi2_bao.to_bits() != refined.reference_chi2_bao.to_bits()
        || primary.absolute_tolerance.to_bits() != refined.absolute_tolerance.to_bits()
        || primary.model != refined.model
        || primary.authority != refined.authority
        || primary.independence != refined.independence
    {
        return Err("primary/refined A1R receipts do not describe the same scientific subject".into());
    }

    let mut max_abs_prediction_delta = 0.0_f64;
    let mut max_rel_prediction_delta = 0.0_f64;
    for (left, right) in primary.predictions.iter().zip(&refined.predictions) {
        if left.z.to_bits() != right.z.to_bits()
            || left.observable != right.observable
            || left.observed.to_bits() != right.observed.to_bits()
        {
            return Err("primary/refined prediction vectors are not aligned".into());
        }
        let absolute = (left.predicted - right.predicted).abs();
        let relative = absolute / right.predicted.abs().max(f64::MIN_POSITIVE);
        max_abs_prediction_delta = max_abs_prediction_delta.max(absolute);
        max_rel_prediction_delta = max_rel_prediction_delta.max(relative);
    }

    let absolute_chi2_delta = (primary.computed_chi2_bao - refined.computed_chi2_bao).abs();
    let converged = max_abs_prediction_delta <= spec.max_absolute_prediction_delta
        && max_rel_prediction_delta <= spec.max_relative_prediction_delta
        && absolute_chi2_delta <= spec.max_absolute_chi2_delta;

    Ok(Receipt {
        protocol: spec.protocol,
        verdict: if converged { "PASS" } else { "INVALID" },
        scientific_claim: "NONE",
        authority: spec.authority,
        spec_sha256: sha256_hex(&spec_bytes),
        primary_manifest_sha256,
        refined_manifest_sha256,
        primary_receipt_sha256,
        refined_receipt_sha256,
        a0_receipt_sha256: primary.a0_receipt_sha256,
        primary_prediction_sha256: prediction_digest(&primary.predictions)?,
        refined_prediction_sha256: prediction_digest(&refined.predictions)?,
        primary_chi2_bao: primary.computed_chi2_bao,
        refined_chi2_bao: refined.computed_chi2_bao,
        absolute_chi2_delta,
        max_absolute_prediction_delta: max_abs_prediction_delta,
        max_relative_prediction_delta: max_rel_prediction_delta,
        allowed_absolute_chi2_delta: spec.max_absolute_chi2_delta,
        allowed_absolute_prediction_delta: spec.max_absolute_prediction_delta,
        allowed_relative_prediction_delta: spec.max_relative_prediction_delta,
        primary_simpson_subdivisions: spec.primary_simpson_subdivisions,
        refined_simpson_subdivisions: spec.refined_simpson_subdivisions,
        development_history: spec.development_history,
    })
}

fn write_json<T: Serialize>(value: &T) -> io::Result<()> {
    let stdout = io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, value)?;
    writeln!(&mut lock)?;
    Ok(())
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 5 {
        return Err(
            "usage: de001a-a1n-convergence SPEC.json PRIMARY_POINT.json REFINED_POINT.json PRIMARY_RECEIPT.json REFINED_RECEIPT.json".into(),
        );
    }
    let receipt = execute(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
    )?;
    let code = if receipt.verdict == "PASS" { 0 } else { 2 };
    write_json(&receipt).map_err(|error| format!("failed to write receipt: {error}"))?;
    Ok(code)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            let receipt = InvalidReceipt {
                protocol: "DE-001A1N-NUMERICAL-CONVERGENCE-v1",
                verdict: "INVALID",
                scientific_claim: "NONE",
                error,
            };
            let _ = write_json(&receipt);
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_rejects_non_doubling_refinement() {
        let spec = ConvergenceSpec {
            schema_version: 1,
            protocol: "DE-001A1N-NUMERICAL-CONVERGENCE-v1".into(),
            scientific_claim: "NONE".into(),
            authority: "numerical-convergence-qualification-only".into(),
            primary_simpson_subdivisions: 1024,
            refined_simpson_subdivisions: 3072,
            max_absolute_prediction_delta: 1e-8,
            max_relative_prediction_delta: 1e-10,
            max_absolute_chi2_delta: 1e-8,
            development_history: "test".into(),
        };
        assert!(validate_spec(&spec).is_err());
    }

    #[test]
    fn digest_validator_rejects_uppercase() {
        assert!(!is_lower_hex_sha256(&"A".repeat(64)));
        assert!(is_lower_hex_sha256(&"a".repeat(64)));
    }
}
