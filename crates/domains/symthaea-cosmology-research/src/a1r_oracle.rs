// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A1R fixed-point Rust oracle.
//!
//! This is deliberately not the Cobaya execution lane. It independently
//! computes flat-LambdaCDM BAO distance ratios and the Gaussian chi-square at
//! one preregistered point, using the same released DESI measurement vector
//! and covariance. It has no sampler and no minimizer.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

const MAX_SMALL_FILE_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    protocol: String,
    scientific_claim: String,
    authority: String,
    model: String,
    source: SourceSpec,
    data: DataSpec,
    parameters: Parameters,
    reference: Reference,
    numerics: Numerics,
    independence: Independence,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceSpec {
    statement: String,
    bestfit_role: String,
    bestfit_size: u64,
    bestfit_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DataSpec {
    mean_role: String,
    mean_size: u64,
    mean_sha256: String,
    covariance_role: String,
    covariance_size: u64,
    covariance_sha256: String,
    row_count: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
#[serde(deny_unknown_fields)]
struct Parameters {
    omega_m: f64,
    h_r_d_mpc: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    chi2_bao: f64,
    absolute_tolerance: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Numerics {
    speed_of_light_km_s: f64,
    simpson_subdivisions: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
struct Independence {
    measurement_data_independent: bool,
    covariance_independent: bool,
    background_implementation_independent: bool,
    gaussian_likelihood_implementation_independent: bool,
    scope: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct A0Receipt {
    protocol: String,
    manifest_sha256: String,
    verdict: String,
    scientific_claim: String,
    artifacts: Vec<A0ArtifactReceipt>,
    errors: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct A0ArtifactReceipt {
    role: String,
    authority: String,
    locator: String,
    path: String,
    expected_size: u64,
    actual_size: u64,
    expected_sha256: String,
    actual_sha256: String,
    status: String,
}

#[derive(Debug, Clone, Copy)]
enum Observable {
    DvOverRd,
    DmOverRd,
    DhOverRd,
}

impl Observable {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "DV_over_rs" => Ok(Self::DvOverRd),
            "DM_over_rs" => Ok(Self::DmOverRd),
            "DH_over_rs" => Ok(Self::DhOverRd),
            other => Err(format!("unsupported BAO observable {other:?}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::DvOverRd => "DV_over_rs",
            Self::DmOverRd => "DM_over_rs",
            Self::DhOverRd => "DH_over_rs",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Measurement {
    z: f64,
    value: f64,
    observable: Observable,
}

#[derive(Debug, Serialize)]
struct PredictionReceipt {
    z: f64,
    observable: &'static str,
    observed: f64,
    predicted: f64,
    residual: f64,
}

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: String,
    point_manifest_sha256: String,
    a0_receipt_sha256: String,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: String,
    model: String,
    parameters: Parameters,
    reference_chi2_bao: f64,
    computed_chi2_bao: f64,
    absolute_delta_chi2: f64,
    absolute_tolerance: f64,
    predictions: Vec<PredictionReceipt>,
    independence: Independence,
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

fn read_regular_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{}: metadata failed: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!("{}: symlinks are forbidden", path.display()));
    }
    if !metadata.file_type().is_file() {
        return Err(format!("{}: not a regular file", path.display()));
    }
    if metadata.len() > max_bytes {
        return Err(format!(
            "{}: file size {} exceeds limit {max_bytes}",
            path.display(),
            metadata.len()
        ));
    }
    let bytes = fs::read(path).map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{}: file changed size while reading", path.display()));
    }
    Ok(bytes)
}

fn verify_file(path: &Path, expected_size: u64, expected_sha256: &str) -> Result<Vec<u8>, String> {
    if !is_lower_hex_sha256(expected_sha256) {
        return Err("manifest contains malformed SHA-256".into());
    }
    let bytes = read_regular_file(path, MAX_SMALL_FILE_BYTES)?;
    if bytes.len() as u64 != expected_size {
        return Err(format!(
            "{}: expected {expected_size} bytes, got {}",
            path.display(),
            bytes.len()
        ));
    }
    let actual = sha256_hex(&bytes);
    if actual != expected_sha256 {
        return Err(format!(
            "{}: SHA-256 mismatch: expected {expected_sha256}, got {actual}",
            path.display()
        ));
    }
    Ok(bytes)
}

fn validate_manifest(manifest: &Manifest) -> Result<(), String> {
    if manifest.schema_version != 1 {
        return Err(format!("unsupported schema_version={}", manifest.schema_version));
    }
    if manifest.protocol != "DE-001A1R-RUST-ORACLE-v1" {
        return Err(format!("unexpected protocol {:?}", manifest.protocol));
    }
    if manifest.scientific_claim != "NONE" {
        return Err("A1R must declare scientific_claim=NONE".into());
    }
    if manifest.authority != "fixed-point-reproduction-sanity-only" {
        return Err(format!("unexpected authority {:?}", manifest.authority));
    }
    if manifest.model != "flat-lambda-cdm-bao-only-late-background" {
        return Err(format!("unexpected model {:?}", manifest.model));
    }
    if manifest.source.statement.trim().is_empty()
        || manifest.source.bestfit_role != "reference-bestfit-text"
        || manifest.source.bestfit_size != 902
        || !is_lower_hex_sha256(&manifest.source.bestfit_sha256)
    {
        return Err("invalid fixed-point source identity".into());
    }
    if manifest.data.mean_role != "dataset-mean"
        || manifest.data.covariance_role != "dataset-covariance"
        || manifest.data.row_count != 13
        || !is_lower_hex_sha256(&manifest.data.mean_sha256)
        || !is_lower_hex_sha256(&manifest.data.covariance_sha256)
    {
        return Err("invalid data identity contract".into());
    }
    if !manifest.parameters.omega_m.is_finite()
        || !(0.0..1.0).contains(&manifest.parameters.omega_m)
        || !manifest.parameters.h_r_d_mpc.is_finite()
        || manifest.parameters.h_r_d_mpc <= 0.0
    {
        return Err("invalid fixed cosmological point".into());
    }
    if !manifest.reference.chi2_bao.is_finite()
        || manifest.reference.chi2_bao < 0.0
        || !manifest.reference.absolute_tolerance.is_finite()
        || manifest.reference.absolute_tolerance <= 0.0
    {
        return Err("invalid reference comparison contract".into());
    }
    if !manifest.numerics.speed_of_light_km_s.is_finite()
        || manifest.numerics.speed_of_light_km_s <= 0.0
        || manifest.numerics.simpson_subdivisions < 32
        || manifest.numerics.simpson_subdivisions > 1_000_000
        || manifest.numerics.simpson_subdivisions % 2 != 0
    {
        return Err("invalid numerical integration contract".into());
    }
    if manifest.independence.measurement_data_independent
        || manifest.independence.covariance_independent
        || !manifest.independence.background_implementation_independent
        || !manifest.independence.gaussian_likelihood_implementation_independent
        || manifest.independence.scope.trim().is_empty()
    {
        return Err("invalid independence classification".into());
    }
    Ok(())
}

fn require_a0_artifact(
    receipt: &A0Receipt,
    role: &str,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<(), String> {
    let matches: Vec<_> = receipt.artifacts.iter().filter(|artifact| artifact.role == role).collect();
    if matches.len() != 1 {
        return Err(format!("A0 receipt must contain exactly one {role:?} artifact"));
    }
    let artifact = matches[0];
    if artifact.status != "PASS"
        || artifact.expected_size != expected_size
        || artifact.actual_size != expected_size
        || artifact.expected_sha256 != expected_sha256
        || artifact.actual_sha256 != expected_sha256
        || artifact.authority.trim().is_empty()
        || artifact.locator.trim().is_empty()
        || artifact.path.trim().is_empty()
    {
        return Err(format!("A0 artifact {role:?} does not satisfy the frozen identity"));
    }
    Ok(())
}

fn validate_a0_receipt(receipt: &A0Receipt, manifest: &Manifest) -> Result<(), String> {
    if receipt.protocol != "DE-001A0-BYTE-INTEGRITY-v1"
        || receipt.verdict != "PASS"
        || receipt.scientific_claim != "NONE"
        || !receipt.errors.is_empty()
        || !is_lower_hex_sha256(&receipt.manifest_sha256)
    {
        return Err("A1R requires a clean DE-001A0 PASS receipt".into());
    }
    require_a0_artifact(
        receipt,
        &manifest.data.mean_role,
        manifest.data.mean_size,
        &manifest.data.mean_sha256,
    )?;
    require_a0_artifact(
        receipt,
        &manifest.data.covariance_role,
        manifest.data.covariance_size,
        &manifest.data.covariance_sha256,
    )?;
    require_a0_artifact(
        receipt,
        &manifest.source.bestfit_role,
        manifest.source.bestfit_size,
        &manifest.source.bestfit_sha256,
    )?;
    Ok(())
}

fn parse_measurements(bytes: &[u8], expected_rows: usize) -> Result<Vec<Measurement>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| format!("mean file is not UTF-8: {error}"))?;
    let mut rows = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 3 {
            return Err(format!("mean line {} must have exactly 3 fields", line_number + 1));
        }
        let z: f64 = fields[0]
            .parse()
            .map_err(|_| format!("mean line {} has invalid redshift", line_number + 1))?;
        let value: f64 = fields[1]
            .parse()
            .map_err(|_| format!("mean line {} has invalid value", line_number + 1))?;
        if !z.is_finite() || z <= 0.0 || !value.is_finite() || value <= 0.0 {
            return Err(format!("mean line {} contains non-physical values", line_number + 1));
        }
        rows.push(Measurement {
            z,
            value,
            observable: Observable::parse(fields[2])?,
        });
    }
    if rows.len() != expected_rows {
        return Err(format!("expected {expected_rows} BAO rows, got {}", rows.len()));
    }
    Ok(rows)
}

fn parse_covariance(bytes: &[u8], dimension: usize) -> Result<Vec<Vec<f64>>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| format!("covariance is not UTF-8: {error}"))?;
    let mut rows = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values: Result<Vec<f64>, _> = line.split_whitespace().map(str::parse::<f64>).collect();
        let values = values.map_err(|_| format!("covariance line {} contains invalid float", line_number + 1))?;
        if values.len() != dimension || values.iter().any(|value| !value.is_finite()) {
            return Err(format!("covariance line {} must contain {dimension} finite values", line_number + 1));
        }
        rows.push(values);
    }
    if rows.len() != dimension {
        return Err(format!("covariance must contain {dimension} rows, got {}", rows.len()));
    }
    for i in 0..dimension {
        for j in 0..dimension {
            let a = rows[i][j];
            let b = rows[j][i];
            let scale = 1.0_f64.max(a.abs()).max(b.abs());
            if (a - b).abs() > 1e-12 * scale {
                return Err(format!("covariance is asymmetric at ({i},{j})"));
            }
        }
    }
    Ok(rows)
}

fn inv_e(z: f64, omega_m: f64) -> f64 {
    1.0 / (omega_m * (1.0 + z).powi(3) + (1.0 - omega_m)).sqrt()
}

fn simpson_integral(z: f64, omega_m: f64, subdivisions: usize) -> f64 {
    let step = z / subdivisions as f64;
    let mut sum = inv_e(0.0, omega_m) + inv_e(z, omega_m);
    for index in 1..subdivisions {
        let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * inv_e(index as f64 * step, omega_m);
    }
    sum * step / 3.0
}

fn predict(measurement: Measurement, manifest: &Manifest) -> f64 {
    let scale = manifest.numerics.speed_of_light_km_s / (100.0 * manifest.parameters.h_r_d_mpc);
    let dm_over_rd = scale
        * simpson_integral(
            measurement.z,
            manifest.parameters.omega_m,
            manifest.numerics.simpson_subdivisions,
        );
    let dh_over_rd = scale * inv_e(measurement.z, manifest.parameters.omega_m);
    match measurement.observable {
        Observable::DmOverRd => dm_over_rd,
        Observable::DhOverRd => dh_over_rd,
        Observable::DvOverRd => (measurement.z * dm_over_rd * dm_over_rd * dh_over_rd).cbrt(),
    }
}

fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = matrix.len();
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i][j];
            for k in 0..j {
                value -= lower[i][k] * lower[j][k];
            }
            if i == j {
                if !value.is_finite() || value <= 0.0 {
                    return Err(format!("covariance is not positive definite at pivot {i}"));
                }
                lower[i][j] = value.sqrt();
            } else {
                lower[i][j] = value / lower[j][j];
            }
        }
    }
    Ok(lower)
}

fn gaussian_chi2(covariance: &[Vec<f64>], residuals: &[f64]) -> Result<f64, String> {
    if covariance.len() != residuals.len() {
        return Err("covariance/residual dimension mismatch".into());
    }
    let lower = cholesky(covariance)?;
    let mut whitened = vec![0.0; residuals.len()];
    for i in 0..residuals.len() {
        let correction: f64 = (0..i).map(|j| lower[i][j] * whitened[j]).sum();
        whitened[i] = (residuals[i] - correction) / lower[i][i];
        if !whitened[i].is_finite() {
            return Err(format!("non-finite whitened residual at index {i}"));
        }
    }
    Ok(whitened.iter().map(|value| value * value).sum())
}

fn execute(
    manifest_path: &Path,
    a0_receipt_path: &Path,
    mean_path: &Path,
    covariance_path: &Path,
) -> Result<Receipt, String> {
    let manifest_bytes = read_regular_file(manifest_path, MAX_SMALL_FILE_BYTES)?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("invalid A1R manifest JSON: {error}"))?;
    validate_manifest(&manifest)?;

    let a0_bytes = read_regular_file(a0_receipt_path, MAX_SMALL_FILE_BYTES)?;
    let a0_sha256 = sha256_hex(&a0_bytes);
    let a0_receipt: A0Receipt = serde_json::from_slice(&a0_bytes)
        .map_err(|error| format!("invalid A0 receipt JSON: {error}"))?;
    validate_a0_receipt(&a0_receipt, &manifest)?;

    let mean_bytes = verify_file(mean_path, manifest.data.mean_size, &manifest.data.mean_sha256)?;
    let covariance_bytes = verify_file(
        covariance_path,
        manifest.data.covariance_size,
        &manifest.data.covariance_sha256,
    )?;
    let measurements = parse_measurements(&mean_bytes, manifest.data.row_count)?;
    let covariance = parse_covariance(&covariance_bytes, manifest.data.row_count)?;

    let mut predictions = Vec::with_capacity(measurements.len());
    let mut residuals = Vec::with_capacity(measurements.len());
    for measurement in measurements {
        let predicted = predict(measurement, &manifest);
        if !predicted.is_finite() || predicted <= 0.0 {
            return Err("oracle produced a non-finite or non-positive prediction".into());
        }
        let residual = predicted - measurement.value;
        residuals.push(residual);
        predictions.push(PredictionReceipt {
            z: measurement.z,
            observable: measurement.observable.as_str(),
            observed: measurement.value,
            predicted,
            residual,
        });
    }

    let chi2 = gaussian_chi2(&covariance, &residuals)?;
    if !chi2.is_finite() || chi2 < 0.0 {
        return Err("oracle produced invalid chi-square".into());
    }
    let delta = (chi2 - manifest.reference.chi2_bao).abs();
    let verdict = if delta <= manifest.reference.absolute_tolerance {
        "PASS"
    } else {
        "NEGATIVE"
    };

    Ok(Receipt {
        protocol: manifest.protocol,
        point_manifest_sha256: manifest_sha256,
        a0_receipt_sha256: a0_sha256,
        verdict,
        scientific_claim: "NONE",
        authority: manifest.authority,
        model: manifest.model,
        parameters: manifest.parameters,
        reference_chi2_bao: manifest.reference.chi2_bao,
        computed_chi2_bao: chi2,
        absolute_delta_chi2: delta,
        absolute_tolerance: manifest.reference.absolute_tolerance,
        predictions,
        independence: manifest.independence,
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
    if args.len() != 4 {
        return Err(
            "usage: de001a-a1r-oracle POINT.json A0_RECEIPT.json MEAN.txt COV.txt".into(),
        );
    }
    let receipt = execute(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
    )?;
    let code = if receipt.verdict == "PASS" { 0 } else { 1 };
    write_json(&receipt).map_err(|error| format!("failed to write receipt: {error}"))?;
    Ok(code)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            let receipt = InvalidReceipt {
                protocol: "DE-001A1R-RUST-ORACLE-v1",
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
    fn simpson_matches_eds_closed_form() {
        let z = 1.0;
        let numerical = simpson_integral(z, 1.0, 1024);
        let exact = 2.0 * (1.0 - 1.0 / (1.0 + z).sqrt());
        assert!((numerical - exact).abs() < 1e-12);
    }

    #[test]
    fn identity_covariance_reduces_to_sum_of_squares() {
        let covariance = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let chi2 = gaussian_chi2(&covariance, &[3.0, 4.0]).unwrap();
        assert!((chi2 - 25.0).abs() < 1e-12);
    }

    #[test]
    fn observable_parser_rejects_unknown_quantities() {
        assert!(Observable::parse("not-a-bao-observable").is_err());
    }
}
