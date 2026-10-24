//! Reproduce and verify the explicitly declared EnzymeML calibration fit.
//!
//! This checks one calibration JSON artifact, not a biochemical kinetic model or a predictor.
//! The caller must supply the expected SHA-256 of the exact local source bytes.
use anyhow::{ensure, Context, Result};
use md5::Md5;
use sha1::Sha1;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use url::Url;

const REL_TOLERANCE: f64 = 1e-8;
const ABS_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Deserialize)]
struct CalibrationArtifact {
    ph: f64,
    temperature: f64,
    temperature_unit: String,
    wavelength: f64,
    signal_type: String,
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
    ph: f64,
    temperature_c: f64,
    temperature_unit: String,
    wavelength_nm: f64,
    signal_type: String,
    concentration_unit: String,
    range_lower: f64,
    range_upper: f64,
    total_samples: usize,
    selected_samples: usize,
    excluded_samples: usize,
    fitted_slope: f64,
    reported_slope: f64,
    fitted_r2: f64,
    reported_r2: f64,
    fitted_rmsd: f64,
    reported_rmsd: f64,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationSuiteManifest {
    schema_version: u32,
    status: String,
    source: CalibrationSuiteSource,
    files: Vec<CalibrationSuiteFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationSuiteSource {
    dataset_doi: String,
    dataverse_url: String,
    license: String,
    mirror_repository: String,
    mirror_commit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationSuiteFile {
    file_name: String,
    local_path: String,
    github_blob_sha: String,
    byte_length: usize,
    sha256: String,
    canonical_dataverse_md5: String,
    expected_fit: ExpectedCalibrationFit,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedCalibrationFit {
    ph: f64,
    temperature_c: f64,
    temperature_unit: String,
    wavelength_nm: f64,
    signal_type: String,
    concentration_unit: String,
    range_lower: f64,
    range_upper: f64,
    total_samples: usize,
    selected_samples: usize,
    excluded_samples: usize,
    slope: f64,
    r2: f64,
    rmsd: f64,
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
        artifact.ph.is_finite()
            && artifact.temperature.is_finite()
            && !artifact.temperature_unit.trim().is_empty()
            && artifact.wavelength.is_finite()
            && artifact.wavelength > 0.0
            && !artifact.signal_type.trim().is_empty(),
        "calibration pH, temperature, wavelength, unit, and signal type must be valid"
    );
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

    let concentration_unit = unit.context("calibration has no samples")?.to_string();
    Ok(CalibrationReport {
        sha256: actual_sha256,
        ph: artifact.ph,
        temperature_c: artifact.temperature,
        temperature_unit: artifact.temperature_unit,
        wavelength_nm: artifact.wavelength,
        signal_type: artifact.signal_type,
        concentration_unit,
        range_lower: lower,
        range_upper: upper,
        total_samples: artifact.samples.len(),
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

/// Validate all five source calibration artifacts listed in a metadata-only suite manifest.
pub fn check_suite(workspace_root: &Path, manifest_path: &Path) -> Result<()> {
    let canonical_root = fs::canonicalize(workspace_root)
        .with_context(|| format!("canonicalize workspace root {}", workspace_root.display()))?;
    let requested_manifest = if manifest_path.is_absolute() {
        manifest_path.to_path_buf()
    } else {
        canonical_root.join(manifest_path)
    };
    let canonical_manifest = fs::canonicalize(&requested_manifest)
        .with_context(|| format!("canonicalize suite manifest {}", requested_manifest.display()))?;
    ensure!(canonical_manifest.starts_with(&canonical_root), "suite manifest path escapes workspace root");
    let bytes = fs::read(&canonical_manifest)
        .with_context(|| format!("read suite manifest {}", canonical_manifest.display()))?;
    let manifest: CalibrationSuiteManifest =
        serde_json::from_slice(&bytes).context("parse calibration suite manifest JSON")?;
    let reports = validate_suite(&canonical_root, &manifest)?;
    println!(
        "ENZYME_CALIBRATION_SUITE=PASS status={} files={} source_doi={} mirror_commit={}",
        manifest.status, reports.len(), manifest.source.dataset_doi, manifest.source.mirror_commit
    );
    for (file_name, report, md5) in reports {
        println!(
            "ENZYME_CALIBRATION_FILE=PASS name={} samples={} selected={} excluded={} sha256={} md5={} slope_delta={:.3e} r2_delta={:.3e} rmsd_delta={:.3e}",
            file_name, report.total_samples, report.selected_samples, report.excluded_samples, report.sha256, md5,
            (report.fitted_slope - report.reported_slope).abs(),
            (report.fitted_r2 - report.reported_r2).abs(),
            (report.fitted_rmsd - report.reported_rmsd).abs()
        );
    }
    Ok(())
}

fn validate_suite(
    workspace_root: &Path,
    manifest: &CalibrationSuiteManifest,
) -> Result<Vec<(String, CalibrationReport, String)>> {
    ensure!(manifest.schema_version == 1, "unsupported calibration suite schema_version");
    ensure!(
        manifest.status == "reference_only_not_admitted",
        "suite manifest must remain explicitly reference_only_not_admitted"
    );
    ensure!(
        manifest.source.dataset_doi == "10.18419/DARUS-3867",
        "unexpected dataset DOI for the SLAC calibration suite"
    );
    let dataverse_url = Url::parse(&manifest.source.dataverse_url)
        .context("source.dataverse_url must be a valid URL")?;
    ensure!(
        dataverse_url.scheme() == "https"
            && dataverse_url.host_str() == Some("darus.uni-stuttgart.de")
            && dataverse_url.path() == "/dataset.xhtml"
            && dataverse_url.query()
                == Some("persistentId=doi%3A10.18419%2Fdarus-3867"),
        "source.dataverse_url must identify the canonical DaRUS-3867 record over HTTPS"
    );
    ensure!(
        manifest.source.license == "CC-BY-4.0",
        "unexpected source license label; re-review before updating this suite"
    );
    ensure!(
        manifest.source.mirror_repository == "EnzymeML/slac_modeling",
        "unexpected mirror repository for the SLAC calibration suite"
    );
    ensure!(
        manifest.source.mirror_commit == "737c57b54f3cf9c9914c53f3d8b7f7790c76b5c5",
        "unexpected mirror commit for the pinned SLAC calibration suite"
    );
    ensure!(manifest.files.len() == 5, "SLAC calibration reference suite requires exactly five temperature files");

    let root = fs::canonicalize(workspace_root)
        .with_context(|| format!("canonicalize workspace root {}", workspace_root.display()))?;
    let mut seen_names = BTreeSet::new();
    let mut seen_paths = BTreeSet::new();
    let mut reports = Vec::with_capacity(manifest.files.len());
    for file in &manifest.files {
        ensure!(!file.file_name.trim().is_empty(), "calibration file_name must be nonempty");
        ensure!(seen_names.insert(file.file_name.as_str()), "duplicate calibration file_name: {}", file.file_name);
        ensure!(seen_paths.insert(file.local_path.as_str()), "duplicate calibration local_path: {}", file.local_path);
        ensure!(
            Path::new(&file.file_name).file_name().and_then(|value| value.to_str()) == Some(file.file_name.as_str()),
            "file_name must be a filename without path components"
        );
        ensure!(
            Path::new(&file.local_path).file_name().and_then(|value| value.to_str()) == Some(file.file_name.as_str()),
            "local_path filename must equal file_name for {}",
            file.file_name
        );
        let expected_temperature = match file.file_name.as_str() {
            "ABTS_standard_25.0C.json" => 25.0,
            "ABTS_standard_30.0C.json" => 30.0,
            "ABTS_standard_35.5C.json" => 35.5,
            "ABTS_standard_40.0C.json" => 40.0,
            "ABTS_standard_45.0C.json" => 45.0,
            _ => anyhow::bail!("unexpected SLAC calibration filename: {}", file.file_name),
        };
        ensure!(
            close(file.expected_fit.temperature_c, expected_temperature),
            "filename/temperature reference mismatch for {}",
            file.file_name
        );
        ensure!(is_sha1(&file.github_blob_sha), "github_blob_sha must be a 40-character hexadecimal Git blob SHA");
        ensure!(is_sha256(&file.sha256), "file SHA-256 must be 64 hexadecimal characters");
        ensure!(is_md5(&file.canonical_dataverse_md5), "canonical DaRUS MD5 must be 32 hexadecimal characters");
        ensure!(
            file.expected_fit.range_lower.is_finite()
                && file.expected_fit.range_upper.is_finite()
                && file.expected_fit.range_lower <= file.expected_fit.range_upper,
            "invalid expected concentration range for {}",
            file.file_name
        );
        ensure!(
            file.expected_fit.slope.is_finite() && file.expected_fit.slope > 0.0
                && file.expected_fit.r2.is_finite()
                && file.expected_fit.rmsd.is_finite() && file.expected_fit.rmsd >= 0.0,
            "invalid expected fit statistics for {}",
            file.file_name
        );

        let relative_path = safe_relative_artifact_path(&file.local_path)
            .with_context(|| format!("invalid local_path for {}", file.file_name))?;
        let candidate = root.join(relative_path);
        let canonical_path = fs::canonicalize(&candidate)
            .with_context(|| format!("canonicalize calibration artifact {}", candidate.display()))?;
        ensure!(
            canonical_path.starts_with(&root),
            "calibration artifact escapes workspace root through a symlink: {}",
            file.local_path
        );
        let source_bytes = fs::read(&canonical_path)
            .with_context(|| format!("read calibration artifact {}", canonical_path.display()))?;
        ensure!(
            source_bytes.len() == file.byte_length,
            "byte length mismatch for {}: expected {}, got {}",
            file.file_name,
            file.byte_length,
            source_bytes.len()
        );
        let actual_git_blob_sha = git_blob_sha1(&source_bytes);
        ensure!(
            actual_git_blob_sha.eq_ignore_ascii_case(&file.github_blob_sha),
            "Git blob SHA mismatch for {}: expected {}, got {}",
            file.file_name,
            file.github_blob_sha,
            actual_git_blob_sha
        );
        let actual_md5 = format!("{:x}", Md5::digest(&source_bytes));
        ensure!(
            actual_md5.eq_ignore_ascii_case(&file.canonical_dataverse_md5),
            "canonical DaRUS MD5 mismatch for {}: expected {}, got {}",
            file.file_name,
            file.canonical_dataverse_md5,
            actual_md5
        );
        let report = validate_bytes(&source_bytes, &file.sha256)
            .with_context(|| format!("verify/reproduce calibration {}", file.file_name))?;
        ensure!(close(report.ph, file.expected_fit.ph), "pH mismatch for {}", file.file_name);
        ensure!(
            close(report.temperature_c, file.expected_fit.temperature_c)
                && report.temperature_unit == file.expected_fit.temperature_unit,
            "temperature mismatch for {}",
            file.file_name
        );
        ensure!(
            close(report.wavelength_nm, file.expected_fit.wavelength_nm),
            "wavelength mismatch for {}",
            file.file_name
        );
        ensure!(
            report.signal_type == file.expected_fit.signal_type,
            "signal type mismatch for {}",
            file.file_name
        );
        ensure!(report.concentration_unit == file.expected_fit.concentration_unit, "concentration unit mismatch for {}", file.file_name);
        ensure!(
            close(report.range_lower, file.expected_fit.range_lower)
                && close(report.range_upper, file.expected_fit.range_upper),
            "declared fit range mismatch for {}",
            file.file_name
        );
        ensure!(
            report.total_samples == file.expected_fit.total_samples
                && report.selected_samples == file.expected_fit.selected_samples
                && report.excluded_samples == file.expected_fit.excluded_samples,
            "sample-count mismatch for {}: total={} selected={} excluded={}",
            file.file_name, report.total_samples, report.selected_samples, report.excluded_samples
        );
        ensure!(close(report.fitted_slope, file.expected_fit.slope), "frozen reference slope mismatch for {}", file.file_name);
        ensure!(close(report.fitted_r2, file.expected_fit.r2), "frozen reference R2 mismatch for {}", file.file_name);
        ensure!(close(report.fitted_rmsd, file.expected_fit.rmsd), "frozen reference RMSD mismatch for {}", file.file_name);
        reports.push((file.file_name.clone(), report, actual_md5));
    }
    Ok(reports)
}

fn safe_relative_artifact_path(path: &str) -> Result<PathBuf> {
    let parsed = Path::new(path);
    ensure!(!path.trim().is_empty(), "artifact path is empty");
    ensure!(!parsed.is_absolute(), "absolute artifact paths are prohibited");
    for component in parsed.components() {
        ensure!(
            matches!(component, Component::Normal(_)),
            "artifact path must not contain '.' or parent traversal"
        );
    }
    Ok(parsed.to_path_buf())
}

fn git_blob_sha1(bytes: &[u8]) -> String {
    let header = format!("blob {}\0", bytes.len());
    let mut hasher = Sha1::new();
    hasher.update(header.as_bytes());
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn is_sha1(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_md5(value: &str) -> bool {
    value.len() == 32 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn fixture() -> Value {
        json!({
            "ph": 3.0,
            "temperature": 25.0,
            "temperature_unit": "C",
            "wavelength": 340.0,
            "signal_type": "Absorbance",
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

    fn fixture_at_temperature(temperature: f64) -> Value {
        let mut value = fixture();
        value["temperature"] = json!(temperature);
        value
    }

    fn validate_fixture(value: &Value) -> anyhow::Result<CalibrationReport> {
        let bytes = serde_json::to_vec(value)?;
        let expected = format!("{:x}", Sha256::digest(&bytes));
        validate_bytes(&bytes, &expected)
    }

    fn build_suite_fixture(root: &Path, corrupt_first_md5: bool) -> CalibrationSuiteManifest {
        let temperatures = [
            ("ABTS_standard_25.0C.json", 25.0),
            ("ABTS_standard_30.0C.json", 30.0),
            ("ABTS_standard_35.5C.json", 35.5),
            ("ABTS_standard_40.0C.json", 40.0),
            ("ABTS_standard_45.0C.json", 45.0),
        ];
        let mut files = Vec::new();
        for (index, (name, temperature)) in temperatures.into_iter().enumerate() {
            let value = fixture_at_temperature(temperature);
            let bytes = serde_json::to_vec(&value).unwrap();
            let sha256 = format!("{:x}", Sha256::digest(&bytes));
            let report = validate_bytes(&bytes, &sha256).unwrap();
            let md5 = if corrupt_first_md5 && index == 0 { "0".repeat(32) } else { format!("{:x}", Md5::digest(&bytes)) };
            let blob = git_blob_sha1(&bytes);
            let local_path = format!("research-data/enzyme-benchmark/slac-calibration/{name}");
            let full_path = root.join(&local_path);
            fs::create_dir_all(full_path.parent().unwrap()).unwrap();
            fs::write(&full_path, &bytes).unwrap();
            files.push(json!({
                "file_name":name,"local_path":local_path,"github_blob_sha":blob,
                "byte_length":bytes.len(),"sha256":sha256,"canonical_dataverse_md5":md5,
                "expected_fit":{
                    "ph":report.ph,"temperature_c":report.temperature_c,
                    "temperature_unit":report.temperature_unit,"wavelength_nm":report.wavelength_nm,
                    "signal_type":report.signal_type,"concentration_unit":report.concentration_unit,
                    "range_lower":report.range_lower,"range_upper":report.range_upper,
                    "total_samples":report.total_samples,"selected_samples":report.selected_samples,
                    "excluded_samples":report.excluded_samples,"slope":report.reported_slope,
                    "r2":report.reported_r2,"rmsd":report.reported_rmsd
                }
            }));
        }
        serde_json::from_value(json!({
            "schema_version":1,"status":"reference_only_not_admitted",
            "source":{
                "dataset_doi":"10.18419/DARUS-3867",
                "dataverse_url":"https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-3867",
                "license":"CC-BY-4.0","mirror_repository":"EnzymeML/slac_modeling",
                "mirror_commit":"737c57b54f3cf9c9914c53f3d8b7f7790c76b5c5"
            },"files":files
        })).unwrap()
    }

    #[test]
    fn calibration_suite_fixture_checks_hash_git_blob_md5_range_counts_fits_and_identity() {
        let root = std::env::temp_dir().join(format!("symthaea-calibration-suite-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let manifest = build_suite_fixture(&root, false);
        let reports = validate_suite(&root, &manifest).unwrap();
        assert_eq!(reports.len(), 5);
        assert!(reports.iter().all(|(_, report, md5)| report.selected_samples == 4 && md5.len() == 32));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn calibration_suite_rejects_md5_mismatch() {
        let root = std::env::temp_dir().join(format!("symthaea-calibration-suite-md5-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let manifest = build_suite_fixture(&root, true);
        let error = validate_suite(&root, &manifest).unwrap_err().to_string();
        assert!(error.contains("canonical DaRUS MD5 mismatch"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn calibration_suite_rejects_temperature_mismatch() {
        let root = std::env::temp_dir().join(format!("symthaea-calibration-suite-temp-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut manifest = build_suite_fixture(&root, false);
        manifest.files[0].expected_fit.temperature_c = 24.0;
        let error = validate_suite(&root, &manifest).unwrap_err().to_string();
        assert!(error.contains("filename/temperature reference mismatch"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn calibration_suite_rejects_source_temperature_mismatch_even_when_hashes_match() {
        let root = std::env::temp_dir().join(format!("symthaea-calibration-suite-source-temp-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut manifest = build_suite_fixture(&root, false);
        let path = root.join(&manifest.files[0].local_path);
        let original = fs::read(&path).unwrap();
        let mut value: Value = serde_json::from_slice(&original).unwrap();
        value["temperature"] = json!(24.0);
        let altered = serde_json::to_vec(&value).unwrap();
        fs::write(&path, &altered).unwrap();
        manifest.files[0].byte_length = altered.len();
        manifest.files[0].sha256 = format!("{:x}", Sha256::digest(&altered));
        manifest.files[0].github_blob_sha = git_blob_sha1(&altered);
        manifest.files[0].canonical_dataverse_md5 = format!("{:x}", Md5::digest(&altered));
        let error = validate_suite(&root, &manifest).unwrap_err().to_string();
        assert!(error.contains("temperature mismatch"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn calibration_suite_rejects_wavelength_mismatch() {
        let root = std::env::temp_dir().join(format!("symthaea-calibration-suite-wavelength-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut manifest = build_suite_fixture(&root, false);
        manifest.files[0].expected_fit.wavelength_nm = 420.0;
        let error = validate_suite(&root, &manifest).unwrap_err().to_string();
        assert!(error.contains("wavelength mismatch"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn calibration_suite_rejects_parent_traversal_paths() {
        assert!(safe_relative_artifact_path("../outside.json").is_err());
        assert!(safe_relative_artifact_path("/absolute/file.json").is_err());
    }

    #[test]
    fn git_blob_sha1_matches_known_git_object() {
        assert_eq!(git_blob_sha1(b"hello world\n"), "3b18e512dba79e4c725f2e5c0a9d42fe66a2d7e1");
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
