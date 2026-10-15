//! Fail-closed admission checker for the first offline enzyme-turnover benchmark.
//!
//! This is research tooling, not a biological predictor. It checks rights/provenance,
//! artifact identities, row completeness, and study/sequence-family split isolation before
//! a benchmark may be called frozen. The source dataset itself is never bundled here.
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

const REQUIRED_COLUMNS: &[&str] = &[
    "observation_id",
    "source_record_id",
    "source_study_id",
    "protein_id",
    "sequence_family_id",
    "endpoint",
    "reported_value",
    "reported_unit",
    "normalized_value",
    "normalized_unit",
    "unit_conversion_rule_id",
    "unit_conversion_factor",
    "assay_conditions_json",
];

const REQUIRED_BASELINES: &[&str] = &[
    "constant_median",
    "context_group_median",
    "descriptor_knn",
    "symthaea_assisted",
];

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u32,
    status: String,
    source: Source,
    target: Target,
    dataset_artifact: DatasetArtifact,
    split_manifest_artifact: Artifact,
    preregistration_artifact: Artifact,
    evaluation: Evaluation,
}

#[derive(Debug, Deserialize)]
struct Source {
    source_id: String,
    release: String,
    url: String,
    license_id: String,
    license_accepted: bool,
    rights_review_status: String,
    attribution: String,
    access_date: String,
}

#[derive(Debug, Deserialize)]
struct Target {
    endpoint: String,
    normalized_unit: String,
}

#[derive(Debug, Deserialize)]
struct Artifact {
    /// Workspace-relative path; absolute paths and parent traversal are rejected.
    path: String,
    /// Hex-encoded SHA-256 of the exact bytes at path.
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct DatasetArtifact {
    #[serde(flatten)]
    artifact: Artifact,
    record_count: usize,
}

#[derive(Debug, Deserialize)]
struct Evaluation {
    random_seed: u64,
    baseline_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SplitManifest {
    schema_version: u32,
    split_mode: String,
    train_study_ids: Vec<String>,
    test_study_ids: Vec<String>,
    train_sequence_family_ids: Vec<String>,
    test_sequence_family_ids: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct CheckReport {
    pub observations: usize,
    pub train_studies: usize,
    pub test_studies: usize,
    pub train_sequence_families: usize,
    pub test_sequence_families: usize,
    /// Sequence-family overlap across train/test in a study-held-out track.
    pub cross_split_families: usize,
    /// Study overlap across train/test in a sequence-family-held-out track.
    pub cross_split_studies: usize,
}

pub fn check(workspace_root: &Path, manifest_path: &Path) -> Result<()> {
    let manifest_bytes = fs::read(manifest_path)
        .with_context(|| format!("read manifest {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parse manifest {}", manifest_path.display()))?;
    let report = validate(workspace_root, &manifest)?;
    println!(
        "ENZYME_BENCH_ADMISSION=PASS observations={} train_studies={} test_studies={} train_families={} test_families={} cross_split_families={} cross_split_studies={} target={}",
        report.observations,
        report.train_studies,
        report.test_studies,
        report.train_sequence_families,
        report.test_sequence_families,
        report.cross_split_families,
        report.cross_split_studies,
        manifest.target.endpoint
    );
    Ok(())
}

fn validate(workspace_root: &Path, manifest: &Manifest) -> Result<CheckReport> {
    ensure!(manifest.schema_version == 1, "unsupported manifest schema_version");
    ensure!(manifest.status == "frozen", "manifest status must be 'frozen'");
    ensure!(!manifest.source.source_id.trim().is_empty(), "source.source_id is required");
    ensure!(!manifest.source.release.trim().is_empty(), "source.release is required");
    ensure!(
        manifest.source.url.starts_with("https://"),
        "source.url must be an https URL"
    );
    ensure!(!manifest.source.license_id.trim().is_empty(), "source.license_id is required");
    ensure!(manifest.source.license_accepted, "source license acceptance is not recorded");
    ensure!(
        manifest.source.rights_review_status == "cleared",
        "source rights review is not cleared (includes any applicable sequence-data benefit-sharing review)"
    );
    ensure!(!manifest.source.attribution.trim().is_empty(), "source attribution is required");
    ensure!(is_date_shape(&manifest.source.access_date), "source.access_date must be YYYY-MM-DD");
    ensure!(manifest.target.endpoint == "kcat", "v0 target endpoint must be exactly 'kcat'");
    ensure!(
        manifest.target.normalized_unit == "s^-1",
        "v0 target must be normalized to s^-1"
    );
    ensure!(manifest.dataset_artifact.record_count > 0, "dataset record_count must be positive");

    let baselines: BTreeSet<&str> = manifest.evaluation.baseline_ids.iter().map(String::as_str).collect();
    for required in REQUIRED_BASELINES {
        ensure!(baselines.contains(required), "missing required baseline: {required}");
    }
    ensure!(
        baselines.len() == manifest.evaluation.baseline_ids.len(),
        "evaluation.baseline_ids contains duplicates"
    );

    let dataset_bytes = read_verified_artifact(workspace_root, &manifest.dataset_artifact.artifact, "dataset")?;
    let split_bytes = read_verified_artifact(workspace_root, &manifest.split_manifest_artifact, "split manifest")?;
    let _preregistration_bytes =
        read_verified_artifact(workspace_root, &manifest.preregistration_artifact, "preregistration")?;
    let split: SplitManifest =
        serde_json::from_slice(&split_bytes).context("parse split manifest JSON")?;
    ensure!(split.schema_version == 1, "unsupported split manifest schema_version");
    let train_studies = unique_allow_empty(split.train_study_ids, "train_study_ids")?;
    let test_studies = unique_allow_empty(split.test_study_ids, "test_study_ids")?;
    let train_families =
        unique_allow_empty(split.train_sequence_family_ids, "train_sequence_family_ids")?;
    let test_families =
        unique_allow_empty(split.test_sequence_family_ids, "test_sequence_family_ids")?;

    match split.split_mode.as_str() {
        "study_holdout" => {
            ensure!(!train_studies.is_empty() && !test_studies.is_empty(), "study_holdout requires train and test studies");
            ensure!(train_families.is_empty() && test_families.is_empty(), "study_holdout must not claim a family partition");
            ensure!(train_studies.is_disjoint(&test_studies), "study leakage: train/test study IDs overlap");
        }
        "sequence_family_holdout" => {
            ensure!(!train_families.is_empty() && !test_families.is_empty(), "sequence_family_holdout requires train and test families");
            ensure!(train_studies.is_empty() && test_studies.is_empty(), "sequence_family_holdout must not claim a study partition");
            ensure!(train_families.is_disjoint(&test_families), "sequence-family leakage: train/test family IDs overlap");
        }
        "joint_study_and_sequence_family_holdout" => {
            ensure!(!train_studies.is_empty() && !test_studies.is_empty(), "joint holdout requires train and test studies");
            ensure!(!train_families.is_empty() && !test_families.is_empty(), "joint holdout requires train and test families");
            ensure!(train_studies.is_disjoint(&test_studies), "study leakage: train/test study IDs overlap");
            ensure!(
                train_families.is_disjoint(&test_families),
                "sequence-family leakage: train/test family IDs overlap"
            );
        }
        other => bail!("unsupported split_mode: {other}"),
    }

    let mut reader = csv::ReaderBuilder::new()
        .flexible(false)
        .from_reader(dataset_bytes.as_slice());
    let headers = reader.headers().context("read dataset CSV header")?.clone();
    let header_index: BTreeMap<&str, usize> = headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name, index))
        .collect();
    for required in REQUIRED_COLUMNS {
        ensure!(header_index.contains_key(required), "dataset CSV missing column: {required}");
    }

    let mut seen_observations = BTreeSet::new();
    let mut observed_train_studies = BTreeSet::new();
    let mut observed_test_studies = BTreeSet::new();
    let mut observed_train_families = BTreeSet::new();
    let mut observed_test_families = BTreeSet::new();
    let mut observations = 0usize;

    for (row_index, row) in reader.records().enumerate() {
        let row = row.with_context(|| format!("parse dataset CSV record {}", row_index + 2))?;
        let observation_id = field(&row, &header_index, "observation_id");
        ensure!(!observation_id.is_empty(), "empty observation_id at CSV record {}", row_index + 2);
        ensure!(
            seen_observations.insert(observation_id.to_string()),
            "duplicate observation_id: {observation_id}"
        );
        for required in [
            "source_record_id",
            "source_study_id",
            "protein_id",
            "sequence_family_id",
            "endpoint",
            "reported_value",
            "reported_unit",
            "normalized_value",
            "normalized_unit",
            "assay_conditions_json",
        ] {
            ensure!(
                !field(&row, &header_index, required).is_empty(),
                "empty {required} at CSV record {}",
                row_index + 2
            );
        }
        ensure!(
            field(&row, &header_index, "endpoint") == manifest.target.endpoint,
            "mixed endpoints at CSV record {}",
            row_index + 2
        );
        ensure!(
            field(&row, &header_index, "normalized_unit") == manifest.target.normalized_unit,
            "wrong normalized_unit at CSV record {}",
            row_index + 2
        );
        let reported_value: f64 = field(&row, &header_index, "reported_value")
            .parse()
            .with_context(|| format!("invalid reported_value at CSV record {}", row_index + 2))?;
        let normalized_value: f64 = field(&row, &header_index, "normalized_value")
            .parse()
            .with_context(|| format!("invalid normalized_value at CSV record {}", row_index + 2))?;
        let conversion_factor: f64 = field(&row, &header_index, "unit_conversion_factor")
            .parse()
            .with_context(|| format!("invalid unit_conversion_factor at CSV record {}", row_index + 2))?;
        ensure!(
            reported_value.is_finite() && reported_value > 0.0
                && normalized_value.is_finite() && normalized_value > 0.0
                && conversion_factor.is_finite() && conversion_factor > 0.0,
            "reported/normalized values and conversion factor must be finite and positive at CSV record {}",
            row_index + 2
        );
        ensure!(
            !field(&row, &header_index, "unit_conversion_rule_id").is_empty(),
            "unit conversion rule ID is required at CSV record {}",
            row_index + 2
        );
        let converted = reported_value * conversion_factor;
        let tolerance = 1e-9 * converted.abs().max(normalized_value.abs()).max(1.0);
        ensure!(
            converted.is_finite() && (converted - normalized_value).abs() <= tolerance,
            "unit conversion mismatch at CSV record {}: reported_value * factor != normalized_value",
            row_index + 2
        );

        let conditions: Value = serde_json::from_str(field(&row, &header_index, "assay_conditions_json"))
            .with_context(|| format!("invalid assay_conditions_json at CSV record {}", row_index + 2))?;
        validate_condition(&conditions, "ph", row_index + 2)?;
        validate_condition(&conditions, "temperature_c", row_index + 2)?;
        validate_condition(&conditions, "substrate_id", row_index + 2)?;

        let study = field(&row, &header_index, "source_study_id");
        let family = field(&row, &header_index, "sequence_family_id");
        let in_train_study = train_studies.contains(study);
        let in_test_study = test_studies.contains(study);
        let in_train_family = train_families.contains(family);
        let in_test_family = test_families.contains(family);
        let (in_train, in_test) = match split.split_mode.as_str() {
            "study_holdout" => (in_train_study, in_test_study),
            "sequence_family_holdout" => (in_train_family, in_test_family),
            "joint_study_and_sequence_family_holdout" => {
                ensure!(
                    in_train_study ^ in_test_study,
                    "study absent from or duplicated across frozen splits: {study}"
                );
                ensure!(
                    in_train_family ^ in_test_family,
                    "sequence family absent from or duplicated across frozen splits: {family}"
                );
                ensure!(
                    in_train_study == in_train_family,
                    "study/family split mismatch at observation {observation_id}"
                );
                (in_train_study, in_test_study)
            }
            _ => unreachable!("split mode validated above"),
        };
        ensure!(
            in_train ^ in_test,
            "observation {observation_id} is absent from or duplicated across frozen splits"
        );
        if in_train {
            observed_train_studies.insert(study.to_string());
            observed_train_families.insert(family.to_string());
        } else {
            observed_test_studies.insert(study.to_string());
            observed_test_families.insert(family.to_string());
        }
        observations += 1;
    }

    ensure!(
        observations == manifest.dataset_artifact.record_count,
        "record_count mismatch: manifest says {}, CSV contains {}",
        manifest.dataset_artifact.record_count,
        observations
    );
    ensure!(observations > 0, "dataset contains no observations");
    match split.split_mode.as_str() {
        "study_holdout" => {
            ensure!(observed_train_studies == train_studies, "train_study_ids contains unused or absent studies");
            ensure!(observed_test_studies == test_studies, "test_study_ids contains unused or absent studies");
        }
        "sequence_family_holdout" => {
            ensure!(
                observed_train_families == train_families,
                "train_sequence_family_ids contains unused or absent families"
            );
            ensure!(
                observed_test_families == test_families,
                "test_sequence_family_ids contains unused or absent families"
            );
        }
        "joint_study_and_sequence_family_holdout" => {
            ensure!(observed_train_studies == train_studies, "train_study_ids contains unused or absent studies");
            ensure!(observed_test_studies == test_studies, "test_study_ids contains unused or absent studies");
            ensure!(
                observed_train_families == train_families,
                "train_sequence_family_ids contains unused or absent families"
            );
            ensure!(
                observed_test_families == test_families,
                "test_sequence_family_ids contains unused or absent families"
            );
        }
        _ => unreachable!("split mode validated above"),
    }
    let cross_split_families = observed_train_families.intersection(&observed_test_families).count();
    let cross_split_studies = observed_train_studies.intersection(&observed_test_studies).count();
    if split.split_mode == "joint_study_and_sequence_family_holdout" {
        ensure!(cross_split_families == 0, "joint holdout contains cross-split sequence families");
        ensure!(cross_split_studies == 0, "joint holdout contains cross-split study IDs");
    }

    Ok(CheckReport {
        observations,
        train_studies: observed_train_studies.len(),
        test_studies: observed_test_studies.len(),
        train_sequence_families: observed_train_families.len(),
        test_sequence_families: observed_test_families.len(),
        cross_split_families,
        cross_split_studies,
    })
}

fn field<'a>(row: &'a csv::StringRecord, headers: &BTreeMap<&str, usize>, name: &str) -> &'a str {
    row.get(*headers.get(name).expect("required header checked"))
        .unwrap_or("")
        .trim()
}

fn validate_condition(conditions: &Value, key: &str, csv_record: usize) -> Result<()> {
    let condition = conditions
        .as_object()
        .and_then(|object| object.get(key))
        .and_then(Value::as_object)
        .with_context(|| format!("assay context must explicitly record {key} status at CSV record {csv_record}"))?;
    let status = condition.get("status").and_then(Value::as_str).unwrap_or("");
    match status {
        "reported" => {
            let value = condition.get("value").filter(|value| {
                !value.is_null()
                    && value.as_str().map(|s| !s.trim().is_empty()).unwrap_or(true)
            });
            ensure!(value.is_some(), "{key} reported without a value at CSV record {csv_record}");
        }
        "not_reported" => {
            ensure!(
                condition.get("value").is_none_or(Value::is_null),
                "{key} marked not_reported but has a value at CSV record {csv_record}"
            );
        }
        "not_applicable" => {
            ensure!(
                condition.get("reason").and_then(Value::as_str).is_some_and(|s| !s.trim().is_empty()),
                "{key} not_applicable requires a reason at CSV record {csv_record}"
            );
            ensure!(
                condition.get("value").is_none_or(Value::is_null),
                "{key} marked not_applicable but has a value at CSV record {csv_record}"
            );
        }
        _ => bail!(
            "{key} has unknown/missing status at CSV record {csv_record}; expected reported, not_reported, or not_applicable"
        ),
    }
    Ok(())
}

fn read_verified_artifact(root: &Path, artifact: &Artifact, label: &str) -> Result<Vec<u8>> {
    ensure!(is_sha256(&artifact.sha256), "{label} SHA-256 must be 64 hexadecimal characters");
    let path = safe_workspace_path(root, &artifact.path)
        .with_context(|| format!("invalid {label} path"))?;
    let bytes = fs::read(&path).with_context(|| format!("read {label} {}", path.display()))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    ensure!(
        actual.eq_ignore_ascii_case(&artifact.sha256),
        "{label} SHA-256 mismatch for {}: expected {}, got {}",
        path.display(),
        artifact.sha256,
        actual
    );
    Ok(bytes)
}

fn safe_workspace_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    ensure!(!relative.trim().is_empty(), "artifact path is empty");
    ensure!(!path.is_absolute(), "absolute artifact paths are prohibited");
    for component in path.components() {
        ensure!(
            matches!(component, Component::Normal(_)),
            "artifact path must not contain '.' or parent traversal"
        );
    }
    Ok(root.join(path))
}

fn unique_allow_empty(values: Vec<String>, label: &str) -> Result<BTreeSet<String>> {
    let original_len = values.len();
    let mut result = BTreeSet::new();
    for value in values {
        ensure!(!value.trim().is_empty(), "{label} contains an empty ID");
        ensure!(result.insert(value.clone()), "{label} contains duplicate ID: {value}");
    }
    ensure!(result.len() == original_len, "{label} contains duplicate IDs");
    Ok(result)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_date_shape(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn fixture_root() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "symthaea-enzyme-bench-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_hashed(root: &Path, path: &str, bytes: &[u8]) -> Value {
        let full_path = root.join(path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&full_path, bytes).unwrap();
        json!({
            "path": path,
            "sha256": format!("{:x}", Sha256::digest(bytes))
        })
    }

    fn make_manifest(root: &Path) -> Value {
        let csv_path = "data/synthetic_enzyme_observations.csv";
        let mut writer = csv::Writer::from_writer(Vec::new());
        writer.write_record(REQUIRED_COLUMNS).unwrap();
        let context_a = json!({
            "ph": {"status":"reported", "value":7.0},
            "temperature_c": {"status":"reported", "value":25.0},
            "substrate_id": {"status":"reported", "value":"SUBSTRATE-A"}
        });
        let context_b = json!({
            "ph": {"status":"not_reported"},
            "temperature_c": {"status":"reported", "value":30.0},
            "substrate_id": {"status":"reported", "value":"SUBSTRATE-B"}
        });
        let context_a_json = context_a.to_string();
        let context_b_json = context_b.to_string();
        writer.write_record([
            "obs-a", "source-a", "study-a", "protein-a", "family-a", "kcat",
            "2.0", "s^-1", "2.0", "s^-1", "identity_s_per_s_v1", "1.0", context_a_json.as_str()
        ]).unwrap();
        writer.write_record([
            "obs-b", "source-b", "study-b", "protein-b", "family-b", "kcat",
            "3.0", "s^-1", "3.0", "s^-1", "identity_s_per_s_v1", "1.0", context_b_json.as_str()
        ]).unwrap();
        let csv_bytes = writer.into_inner().unwrap();
        let dataset_artifact = write_hashed(root, csv_path, &csv_bytes);
        let splits = json!({
            "schema_version": 1,
            "split_mode": "joint_study_and_sequence_family_holdout",
            "train_study_ids": ["study-a"],
            "test_study_ids": ["study-b"],
            "train_sequence_family_ids": ["family-a"],
            "test_sequence_family_ids": ["family-b"]
        });
        let split_bytes = serde_json::to_vec_pretty(&splits).unwrap();
        let split_artifact = write_hashed(root, "data/splits.json", &split_bytes);
        let prereg_artifact = write_hashed(
            root,
            "research/preregistration.md",
            b"synthetic preregistration fixture; not research evidence",
        );
        json!({
            "schema_version": 1,
            "status": "frozen",
            "source": {
                "source_id": "SYNTHETIC_TEST_FIXTURE",
                "release": "test-v1",
                "url": "https://example.invalid/synthetic-fixture",
                "license_id": "internal-test-fixture",
                "license_accepted": true,
                "rights_review_status": "cleared",
                "attribution": "Synthetic test-only rows; not biological measurements.",
                "access_date": "2026-10-15"
            },
            "target": {"endpoint":"kcat", "normalized_unit":"s^-1"},
            "dataset_artifact": {
                "path": dataset_artifact["path"],
                "sha256": dataset_artifact["sha256"],
                "record_count": 2
            },
            "split_manifest_artifact": split_artifact,
            "preregistration_artifact": prereg_artifact,
            "evaluation": {
                "random_seed": 17,
                "baseline_ids": [
                    "constant_median",
                    "context_group_median",
                    "descriptor_knn",
                    "symthaea_assisted"
                ]
            }
        })
    }

    #[test]
    fn accepts_complete_synthetic_fixture_and_reports_counts() {
        let root = fixture_root();
        let manifest = make_manifest(&root);
        let parsed: Manifest = serde_json::from_value(manifest).unwrap();
        let report = validate(&root, &parsed).unwrap();
        assert_eq!(
            report,
            CheckReport {
                observations: 2,
                train_studies: 1,
                test_studies: 1,
                train_sequence_families: 1,
                test_sequence_families: 1,
                cross_split_families: 0,
                cross_split_studies: 0
            }
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_uncleared_rights_even_when_license_is_named() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        value["source"]["rights_review_status"] = json!("review_required");
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("rights review is not cleared"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_dataset_hash_mismatch() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        value["dataset_artifact"]["sha256"] = json!("0".repeat(64));
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("SHA-256 mismatch"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_study_and_family_leakage() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        let split_path = root.join("data/splits.json");
        let mut split: Value = serde_json::from_slice(&fs::read(&split_path).unwrap()).unwrap();
        split["test_study_ids"] = json!(["study-a"]);
        split["test_sequence_family_ids"] = json!(["family-a"]);
        let bytes = serde_json::to_vec_pretty(&split).unwrap();
        fs::write(&split_path, &bytes).unwrap();
        value["split_manifest_artifact"]["sha256"] =
            json!(format!("{:x}", Sha256::digest(&bytes)));
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("study leakage") || error.contains("sequence-family leakage"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_missing_explicit_assay_context() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        let csv_path = root.join("data/synthetic_enzyme_observations.csv");
        let bytes = fs::read(&csv_path).unwrap();
        let mut reader = csv::Reader::from_reader(bytes.as_slice());
        let headers = reader.headers().unwrap().clone();
        let mut writer = csv::Writer::from_writer(Vec::new());
        writer.write_record(&headers).unwrap();
        for row in reader.records() {
            let mut fields: Vec<String> = row.unwrap().iter().map(str::to_string).collect();
            if fields.first().is_some_and(|value| value.as_str() == "obs-a") {
                let context_index = headers
                    .iter()
                    .position(|name| name == "assay_conditions_json")
                    .unwrap();
                fields[context_index] = r#"{"ph":{"status":"unknown"},"temperature_c":{"status":"reported","value":25},"substrate_id":{"status":"reported","value":"X"}}"#.to_string();
            }
            writer.write_record(fields).unwrap();
        }
        let changed = writer.into_inner().unwrap();
        fs::write(&csv_path, &changed).unwrap();
        value["dataset_artifact"]["sha256"] = json!(format!("{:x}", Sha256::digest(&changed)));
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("unknown/missing status"));
        fs::remove_dir_all(root).unwrap();
    }
}
