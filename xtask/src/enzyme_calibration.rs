//! Reproduce and verify the explicitly declared EnzymeML calibration fit.
//!
//! This checks one calibration JSON artifact, not a biochemical kinetic model or a predictor.
//! The caller must supply the expected SHA-256 of the exact local source bytes.
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const REL_TOLERANCE: f64 = 1e-8;
const ABS_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Deserialize)]
struct CalibrationArtifact {
    samples: Vec<CalibrationSample>,
    model_result: CalibrationModel,
}

#[derive(Debug, Deserialize)]
struct CalibrationSample {
    id: String,
    concentration: f64,
    conc_unit: String,
    signal: f64,
}

#[derive(Debug, Deserialize)]
struct CalibrationModel {
    equation: String,
    parameters: Vec<CalibrationParameter>,
    calibration_range: CalibrationRange,
    statistics: CalibrationStatistics,
}

#[derive(Debug, Deserialize)]
struct CalibrationParameter {
    name: String,
    value: f64,
}

#[derive(Debug, Deserialize)]
struct CalibrationRange {
    conc_lower: f64,
    conc_upper: f64,
}

#[derive(Debug, Deserialize)]
struct CalibrationStatistics {
    r2: f64,
    rmsd: f64,
}

#[derive(Debug, PartialEq)]
struct CalibrationReport {
    sha256: String,
    selected_samples: usize,
    excluded_samples: usize,
    fitted_slope: f64,
    reported_slope: f64,
    fitted_r2: f64,
    reported_r2: f64,
    fitted_rmsd: f64,
    reported_rmsd: f64,
}

/// Verify source bytes and independently reproduce the zero-intercept linear calibration.
pub fn check(input: &Path, expected_sha256: &str) -> Result<()> {
    ensure!(
        is_sha256(expected_sha256),
        "--sha256 must be 64 hexadecimal characters"
    );
    let bytes = fs::read(input).with_context(|| format!("read calibration JSON {}", input.display()))?;
    let report = validate_bytes(&bytes, expected_sha256)?;
    println!(
        "ENZYME_CALIBRATION_CHECK=PASS sha256={} selected_samples={} excluded_samples={} fitted_slope={:.15} reported_slope={:.15} fitted_r2={:.15} reported_r2={:.15} fitted_rmsd={:.15} reported_rmsd={:.15}",
        report.sha256,
        report.selected_samples,
        report.excluded_samples,
        report.fitted_slope,
        report.reported_slope,
        report.fitted_r2,
        report.reported_r2,
        report.fitted_rmsd,
        report.reported_rmsd
    );
    Ok(())
}

fn validate_bytes(bytes: &[u8], expected_sha256: &str) -> Result<CalibrationReport> {
    ensure!(is_sha256(expected_sha256), "expected SHA-256 must be 64 hexadecimal characters");
    let actual_sha256 = format!("{:x}", Sha256::digest(bytes));
    ensure!(
        actual_sha256.eq_ignore_ascii_case(expected_sha256),
        "source SHA-256 mismatch: expected {}, got {}",
        expected_sha256,
        actual_sha256
    );

    let artifact: CalibrationArtifact =
        serde_json::from_slice(bytes).context("parse calibration JSON")?;
    ensure!(!artifact.samples.is_empty(), "calibration contains no samples");
    ensure!(
        artifact.model_result.equation.trim() == "a * concentration = signal",
        "only the declared zero-intercept linear calibration is supported"
    );

    let lower = artifact.model_result.calibration_range.conc_lower;
    let upper = artifact.model_result.calibration_range.conc_upper;
    ensure!(
        lower.is_finite() && upper.is_finite() && lower <= upper,
        "calibration range must have finite lower/upper bounds with lower <= upper"
    );

    let parameters: Vec<&CalibrationParameter> = artifact
        .model_result
        .parameters
        .iter()
        .filter(|parameter| parameter.name == "a")
        .collect();
    ensure!(parameters.len() == 1, "model must contain exactly one parameter named 'a'");
    let reported_slope = parameters[0].value;
    let reported_r2 = artifact.model_result.statistics.r2;
    let reported_rmsd = artifact.model_result.statistics.rmsd;
    ensure!(
        reported_slope.is_finite() && reported_slope > 0.0
            && reported_r2.is_finite()
            && reported_rmsd.is_finite() && reported_rmsd >= 0.0,
        "reported calibration parameters/statistics must be finite and physically meaningful"
    );

    let mut sample_ids = BTreeSet::new();
    let mut unit: Option<&str> = None;
    let mut selected: Vec<(f64, f64)> = Vec::new();
    let mut excluded_samples = 0usize;
    for sample in &artifact.samples {
        ensure!(!sample.id.trim().is_empty(), "calibration sample ID must be nonempty");
        ensure!(
            sample_ids.insert(sample.id.as_str()),
            "duplicate calibration sample ID: {}",
            sample.id
        );
        ensure!(
            !sample.conc_unit.trim().is_empty(),
            "calibration concentration unit must be nonempty"
        );
        if let Some(previous) = unit {
            ensure!(
                previous == sample.conc_unit,
                "calibration samples mix concentration units"
            );
        } else {
            unit = Some(&sample.conc_unit);
        }
        ensure!(
            sample.concentration.is_finite() && sample.concentration >= 0.0,
            "calibration sample {} has invalid concentration",
            sample.id
        );
        ensure!(
            sample.signal.is_finite(),
            "calibration sample {} has non-finite signal",
            sample.id
        );
        if (lower..=upper).contains(&sample.concentration) {
            selected.push((sample.concentration, sample.signal));
        } else {
            excluded_samples += 1;
        }
    }
    ensure!(selected.len() >= 2, "declared calibration range selects fewer than two samples");

    // Reproduce the source-declared equation y = a*x (no free intercept).
    let sum_x2: f64 = selected.iter().map(|(x, _)| x * x).sum();
    let sum_xy: f64 = selected.iter().map(|(x, y)| x * y).sum();
    ensure!(sum_x2.is_finite() && sum_x2 > 0.0 && sum_xy.is_finite(), "invalid calibration sums");
    let fitted_slope = sum_xy / sum_x2;
    ensure!(fitted_slope.is_finite() && fitted_slope > 0.0, "fitted calibration slope is invalid");

    let mean_y = selected.iter().map(|(_, y)| y).sum::<f64>() / selected.len() as f64;
    let sse: f64 = selected
        .iter()
        .map(|(x, y)| {
            let residual = y - fitted_slope * x;
            residual * residual
        })
        .sum();
    let sst: f64 = selected.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
    ensure!(
        sse.is_finite() && sst.is_finite() && sst > 0.0,
        "calibration residual/statistics are degenerate"
    );
    let fitted_r2 = 1.0 - sse / sst;
    let fitted_rmsd = (sse / selected.len() as f64).sqrt();
    ensure!(fitted_r2.is_finite() && fitted_rmsd.is_finite(), "fit statistics are non-finite");

    ensure!(
        close(fitted_slope, reported_slope),
        "fitted slope disagrees with source metadata: fitted {}, reported {}",
        fitted_slope,
        reported_slope
    );
    ensure!(
        close(fitted_r2, reported_r2),
        "fitted R2 disagrees with source metadata: fitted {}, reported {}",
        fitted_r2,
        reported_r2
    );
    ensure!(
        close(fitted_rmsd, reported_rmsd),
        "fitted RMSD disagrees with source metadata: fitted {}, reported {}",
        fitted_rmsd,
        reported_rmsd
    );

    Ok(CalibrationReport {
        sha256: actual_sha256,
        selected_samples: selected.len(),
        excluded_samples,
        fitted_slope,
        reported_slope,
        fitted_r2,
        reported_r2,
        fitted_rmsd,
        reported_rmsd,
    })
}

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= ABS_TOLERANCE + REL_TOLERANCE * expected.abs().max(1e-12)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn fixture() -> Value {
        json!({
            "samples": [
                {"id":"s0","concentration":0.0,"conc_unit":"umol / l","signal":0.0},
                {"id":"s1","concentration":1.0,"conc_unit":"umol / l","signal":2.0},
                {"id":"s2","concentration":2.0,"conc_unit":"umol / l","signal":4.0},
                {"id":"s3","concentration":3.0,"conc_unit":"umol / l","signal":6.0},
                {"id":"outside","concentration":4.0,"conc_unit":"umol / l","signal":100.0}
            ],
            "model_result": {
                "equation":"a * concentration = signal",
                "parameters":[{"name":"a","value":2.0}],
                "calibration_range":{"conc_lower":0.0,"conc_upper":3.0},
                "statistics":{"r2":1.0,"rmsd":0.0}
            }
        })
    }

    fn validate_fixture(value: &Value) -> anyhow::Result<CalibrationReport> {
        let bytes = serde_json::to_vec(value)?;
        let expected = format!("{:x}", Sha256::digest(&bytes));
        validate_bytes(&bytes, &expected)
    }

    #[test]
    fn reproduces_fit_and_respects_declared_calibration_range() {
        let report = validate_fixture(&fixture()).unwrap();
        assert_eq!(report.selected_samples, 4);
        assert_eq!(report.excluded_samples, 1);
        assert!((report.fitted_slope - 2.0).abs() < 1e-12);
        assert!((report.fitted_r2 - 1.0).abs() < 1e-12);
        assert_eq!(report.fitted_rmsd, 0.0);
    }

    #[test]
    fn rejects_wrong_source_sha256() {
        let bytes = serde_json::to_vec(&fixture()).unwrap();
        let error = validate_bytes(&bytes, &"0".repeat(64)).unwrap_err().to_string();
        assert!(error.contains("source SHA-256 mismatch"));
    }

    #[test]
    fn rejects_reported_fit_statistics_that_do_not_match_reproduction() {
        let mut value = fixture();
        value["model_result"]["parameters"][0]["value"] = json!(2.5);
        let error = validate_fixture(&value).unwrap_err().to_string();
        assert!(error.contains("fitted slope disagrees with source metadata"));
    }

    #[test]
    fn rejects_duplicate_calibration_sample_ids() {
        let mut value = fixture();
        value["samples"][1]["id"] = json!("s0");
        let error = validate_fixture(&value).unwrap_err().to_string();
        assert!(error.contains("duplicate calibration sample ID"));
    }

    #[test]
    fn rejects_mixed_concentration_units() {
        let mut value = fixture();
        value["samples"][1]["conc_unit"] = json!("mmol / l");
        let error = validate_fixture(&value).unwrap_err().to_string();
        assert!(error.contains("calibration samples mix concentration units"));
    }

    #[test]
    fn rejects_reversed_calibration_range() {
        let mut value = fixture();
        value["model_result"]["calibration_range"]["conc_lower"] = json!(4.0);
        value["model_result"]["calibration_range"]["conc_upper"] = json!(3.0);
        let error = validate_fixture(&value).unwrap_err().to_string();
        assert!(error.contains("calibration range must have finite lower/upper bounds"));
    }
}
