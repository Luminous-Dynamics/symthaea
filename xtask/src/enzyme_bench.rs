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
use url::Url;

const REQUIRED_COLUMNS: &[&str] = &[
    "observation_id",
    "source_id",
    "source_release",
    "source_record_id",
    "source_record_url",
    "source_locator",
    "source_study_id",
    "measurement_method_id",
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

// Optional STRENDA-inspired reporting fields. These are audited, not admission gates:
// incomplete historical records remain eligible for review with their missingness visible.
const OPTIONAL_METADATA_FIELDS: &[&str] = &[
    "enzyme_purity",
    "enzyme_form",
    "enzyme_concentration",
    "buffer_system",
    "buffer_concentration",
    "assay_pressure",
    "ph_measurement_temperature_c",
    "substrate_concentration_range",
    "kinetic_equation",
    "replicate_count",
    "independent_experiment_count",
    "replicate_structure",
    "measurement_precision",
    "detection_limit",
    "reaction_equation",
    "analysis_method",
    "raw_measurement_data",
    "raw_measurement_data_sha256",
    "initial_rate_method",
    "controls_description",
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
    /// Descriptive STRENDA-inspired metadata coverage; this is not a data-quality score.
    pub metadata_coverage: BTreeMap<String, MetadataCoverage>,
    /// Nested candidate counts under the proposed kcat admission profile; not an admission result.
    pub kcat_evidence_tiers: KcatEvidenceTiers,
    /// One highest-tier label per observation, in CSV row order.
    pub row_evidence_tiers: Vec<RowEvidenceTier>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct KcatEvidenceTiers {
    pub context_qualified: usize,
    pub primary_comparison_candidates: usize,
    pub kinetic_reproduction_candidates: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub struct RowEvidenceTier {
    pub observation_id: String,
    /// The highest metadata-evidence tier reached; not a scientific-validity judgment.
    pub tier: &'static str,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct MetadataCoverage {
    pub reported: usize,
    pub not_reported: usize,
    pub not_applicable: usize,
    /// The key was absent, distinct from an explicit not_reported status.
    pub unrecorded: usize,
    /// A present field had an unrecognized or internally inconsistent status/value.
    pub invalid: usize,
}

impl MetadataCoverage {
    /// Number of observations accounted for by exactly one coverage outcome.
    fn accounted_observations(&self) -> usize {
        self.reported + self.not_reported + self.not_applicable + self.unrecorded + self.invalid
    }
}

pub fn check(workspace_root: &Path, manifest_path: &Path) -> Result<()> {
    let canonical_root = fs::canonicalize(workspace_root)
        .with_context(|| format!("canonicalize workspace root {}", workspace_root.display()))?;
    let requested_manifest = if manifest_path.is_absolute() {
        manifest_path.to_path_buf()
    } else {
        canonical_root.join(manifest_path)
    };
    let canonical_manifest = fs::canonicalize(&requested_manifest)
        .with_context(|| format!("canonicalize manifest {}", requested_manifest.display()))?;
    ensure!(
        canonical_manifest.starts_with(&canonical_root),
        "manifest path escapes workspace root: {}",
        manifest_path.display()
    );
    let manifest_bytes = fs::read(&canonical_manifest)
        .with_context(|| format!("read manifest {}", canonical_manifest.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parse manifest {}", canonical_manifest.display()))?;
    let report = validate(&canonical_root, &manifest)?;
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
    println!(
        "ENZYME_KCAT_EVIDENCE_TIERS observations={} context_qualified={} primary_comparison_candidates={} kinetic_reproduction_candidates={}",
        report.observations,
        report.kcat_evidence_tiers.context_qualified,
        report.kcat_evidence_tiers.primary_comparison_candidates,
        report.kcat_evidence_tiers.kinetic_reproduction_candidates
    );
    for row_tier in &report.row_evidence_tiers {
        let observation_id_json =
            serde_json::to_string(&row_tier.observation_id).expect("JSON serialization of a string is infallible");
        println!(
            "ENZYME_KCAT_ROW_TIER observation_id_json={} tier={}",
            observation_id_json, row_tier.tier
        );
    }
    for (field_name, coverage) in &report.metadata_coverage {
        println!(
            "ENZYME_METADATA_COVERAGE field={} observations={} reported={} not_reported={} not_applicable={} unrecorded={} invalid={}",
            field_name,
            report.observations,
            coverage.reported,
            coverage.not_reported,
            coverage.not_applicable,
            coverage.unrecorded,
            coverage.invalid
        );
    }
    Ok(())
}

fn validate(workspace_root: &Path, manifest: &Manifest) -> Result<CheckReport> {
    ensure!(manifest.schema_version == 1, "unsupported manifest schema_version");
    ensure!(manifest.status == "frozen", "manifest status must be 'frozen'");
    ensure!(!manifest.source.source_id.trim().is_empty(), "source.source_id is required");
    ensure!(!manifest.source.release.trim().is_empty(), "source.release is required");
    ensure!(
        is_valid_https_url(&manifest.source.url),
        "source.url must be a valid HTTPS URL with a host and no embedded credentials"
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
        ensure!(baselines.contains(*required), "missing required baseline: {required}");
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
    ensure!(
        header_index.len() == headers.len(),
        "dataset CSV contains duplicate column names"
    );
    for required in REQUIRED_COLUMNS {
        ensure!(header_index.contains_key(*required), "dataset CSV missing column: {required}");
    }

    let mut seen_observations = BTreeSet::new();
    let mut observed_train_studies = BTreeSet::new();
    let mut observed_test_studies = BTreeSet::new();
    let mut observed_train_families = BTreeSet::new();
    let mut observed_test_families = BTreeSet::new();
    let mut metadata_coverage = empty_metadata_coverage();
    let mut kcat_evidence_tiers = KcatEvidenceTiers::default();
    let mut row_evidence_tiers = Vec::new();
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
            "source_id",
            "source_release",
            "source_record_id",
            "source_record_url",
            "source_locator",
            "source_study_id",
            "measurement_method_id",
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
            field(&row, &header_index, "source_id") == manifest.source.source_id,
            "source_id differs from manifest source.source_id at CSV record {}",
            row_index + 2
        );
        ensure!(
            field(&row, &header_index, "source_release") == manifest.source.release,
            "source_release differs from manifest source.release at CSV record {}",
            row_index + 2
        );
        let source_record_url = field(&row, &header_index, "source_record_url");
        ensure!(
            is_valid_https_url(source_record_url),
            "source_record_url must be a valid HTTPS URL with a host and no embedded credentials at CSV record {}",
            row_index + 2
        );
        ensure!(
            !field(&row, &header_index, "source_locator").is_empty(),
            "source_locator must identify the exact record/table/figure at CSV record {}",
            row_index + 2
        );
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
        let (expected_factor, expected_rule_id) =
            match field(&row, &header_index, "reported_unit") {
                "s^-1" | "1/s" | "s-1" => (1.0, "kcat_s-1_identity_v1"),
                "min^-1" | "1/min" | "min-1" => (1.0 / 60.0, "kcat_min-1_to_s-1_v1"),
                "h^-1" | "1/h" | "h-1" => (1.0 / 3600.0, "kcat_h-1_to_s-1_v1"),
                other => bail!("unsupported kcat reported_unit at CSV record {}: {other}", row_index + 2),
            };
        ensure!(
            field(&row, &header_index, "unit_conversion_rule_id") == expected_rule_id,
            "unit conversion rule does not match reported_unit at CSV record {}",
            row_index + 2
        );
        ensure!(
            (conversion_factor - expected_factor).abs() <= 1e-12 * expected_factor.max(1.0),
            "unit conversion factor does not match reported_unit at CSV record {}",
            row_index + 2
        );
        let converted = reported_value * conversion_factor;
        let tolerance = 1e-9 * converted.abs().max(normalized_value.abs()).max(1e-12);
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
        record_optional_metadata_coverage(&conditions, &mut metadata_coverage);
        let mut tier_name = "review_only";
        if is_context_qualified(&conditions) {
            kcat_evidence_tiers.context_qualified += 1;
            tier_name = "context_qualified";
            if is_primary_comparison_candidate(&conditions) {
                kcat_evidence_tiers.primary_comparison_candidates += 1;
                tier_name = "primary_comparison_candidate";
                if is_kinetic_reproduction_candidate(&conditions) {
                    kcat_evidence_tiers.kinetic_reproduction_candidates += 1;
                    tier_name = "kinetic_reproduction_candidate";
                }
            }
        }
        row_evidence_tiers.push(RowEvidenceTier {
            observation_id: observation_id.to_string(),
            tier: tier_name,
        });

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
    for (field_name, coverage) in &metadata_coverage {
        ensure!(
            coverage.accounted_observations() == observations,
            "metadata coverage accounting mismatch for {field_name}: expected {observations}, accounted {}",
            coverage.accounted_observations()
        );
    }
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
        metadata_coverage,
        kcat_evidence_tiers,
        row_evidence_tiers,
    })
}

fn metadata_field_is_reported(conditions: &Value, field: &str) -> bool {
    conditions
        .as_object()
        .and_then(|object| object.get(field))
        .and_then(Value::as_object)
        .and_then(|object| {
            let status = object.get("status").and_then(Value::as_str)?;
            let value = object.get("value").filter(|value| !value.is_null())?;
            (status == "reported" && optional_metadata_value_is_valid(field, value)).then_some(())
        })
        .is_some()
}

fn metadata_field_is_reasoned_not_applicable(conditions: &Value, field: &str) -> bool {
    conditions
        .as_object()
        .and_then(|object| object.get(field))
        .and_then(Value::as_object)
        .is_some_and(|object| {
            object.get("status").and_then(Value::as_str) == Some("not_applicable")
                && object.get("value").is_none_or(Value::is_null)
                && object
                    .get("reason")
                    .and_then(Value::as_str)
                    .is_some_and(|reason| !reason.trim().is_empty())
        })
}

fn metadata_field_is_reported_or_reasoned_na(conditions: &Value, field: &str) -> bool {
    metadata_field_is_reported(conditions, field)
        || metadata_field_is_reasoned_not_applicable(conditions, field)
}

fn core_condition_is_reported(conditions: &Value, field: &str) -> bool {
    let Some(object) = conditions.as_object().and_then(|object| object.get(field)).and_then(Value::as_object) else {
        return false;
    };
    if object.get("status").and_then(Value::as_str) != Some("reported") {
        return false;
    }
    let Some(value) = object.get("value").filter(|value| !value.is_null()) else {
        return false;
    };
    match field {
        "ph" | "temperature_c" => value.as_f64().is_some_and(f64::is_finite),
        "substrate_id" => value.as_str().is_some_and(|value| !value.trim().is_empty()),
        _ => false,
    }
}

fn is_context_qualified(conditions: &Value) -> bool {
    core_condition_is_reported(conditions, "ph")
        && core_condition_is_reported(conditions, "temperature_c")
        && core_condition_is_reported(conditions, "substrate_id")
        && metadata_field_is_reported(conditions, "enzyme_form")
        && metadata_field_is_reported(conditions, "substrate_concentration_range")
        && metadata_field_is_reported(conditions, "reaction_equation")
        && metadata_field_is_reported(conditions, "kinetic_equation")
        && metadata_field_is_reported_or_reasoned_na(conditions, "assay_pressure")
        && (if metadata_field_is_reported(conditions, "buffer_system") {
            metadata_field_is_reported(conditions, "buffer_concentration")
        } else {
            metadata_field_is_reasoned_not_applicable(conditions, "buffer_system")
                && metadata_field_is_reasoned_not_applicable(conditions, "buffer_concentration")
        })
}

fn is_primary_comparison_candidate(conditions: &Value) -> bool {
    is_context_qualified(conditions)
        && metadata_field_is_reported(conditions, "enzyme_purity")
        && metadata_field_is_reported(conditions, "enzyme_concentration")
        && metadata_field_has_molar_concentration(conditions, "enzyme_concentration")
        && metadata_field_is_reported(conditions, "replicate_count")
        && metadata_field_is_reported(conditions, "independent_experiment_count")
        && metadata_field_is_reported(conditions, "replicate_structure")
        && metadata_field_is_reported(conditions, "measurement_precision")
        && metadata_field_is_reported(conditions, "analysis_method")
}

fn is_kinetic_reproduction_candidate(conditions: &Value) -> bool {
    is_primary_comparison_candidate(conditions)
        && metadata_field_is_reported(conditions, "raw_measurement_data")
        && metadata_field_is_reported(conditions, "raw_measurement_data_sha256")
        && metadata_field_is_reported(conditions, "initial_rate_method")
        && metadata_field_is_reported(conditions, "controls_description")
}

fn metadata_field_has_molar_concentration(conditions: &Value, field: &str) -> bool {
    let unit = conditions
        .as_object()
        .and_then(|object| object.get(field))
        .and_then(Value::as_object)
        .and_then(|object| object.get("value"))
        .and_then(Value::as_object)
        .and_then(|value| value.get("unit"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let normalized = unit
        .trim()
        .replace('μ', "u")
        .replace('µ', "u")
        .replace(' ', "")
        .to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "m" | "mol/l" | "moll-1" | "mol/liter" | "mol/litre"
            | "mm" | "mmol/l" | "mmoll-1"
            | "um" | "umol/l" | "umoll-1"
            | "nm" | "nmol/l" | "nmoll-1"
    )
}

fn empty_metadata_coverage() -> BTreeMap<String, MetadataCoverage> {
    OPTIONAL_METADATA_FIELDS
        .iter()
        .map(|field| ((*field).to_string(), MetadataCoverage::default()))
        .collect()
}

/// Count declared metadata statuses without treating missing context as zero or as a gate failure.
fn record_optional_metadata_coverage(
    conditions: &Value,
    coverage: &mut BTreeMap<String, MetadataCoverage>,
) {
    for field in OPTIONAL_METADATA_FIELDS {
        let counts = coverage.get_mut(*field).expect("all optional fields preinitialized");
        let Some(value) = conditions.as_object().and_then(|object| object.get(*field)) else {
            counts.unrecorded += 1;
            continue;
        };
        let Some(object) = value.as_object() else {
            counts.invalid += 1;
            continue;
        };
        let status = object.get("status").and_then(Value::as_str).unwrap_or("");
        let value = object.get("value").filter(|value| !value.is_null());
        match status {
            "reported" if value.is_some_and(|value| optional_metadata_value_is_valid(field, value)) => {
                counts.reported += 1
            }
            "not_reported" if value.is_none() => counts.not_reported += 1,
            "not_applicable"
                if value.is_none()
                    && object
                        .get("reason")
                        .and_then(Value::as_str)
                        .is_some_and(|reason| !reason.trim().is_empty()) =>
            {
                counts.not_applicable += 1
            }
            _ => counts.invalid += 1,
        }
    }
}

/// A typed coverage audit: validate the shape and basic plausibility of supplied metadata,
/// not the scientific truth of the value or the quality of the underlying experiment.
fn optional_metadata_value_is_valid(field: &str, value: &Value) -> bool {
    let nonempty_string = |candidate: &Value| {
        candidate.as_str().is_some_and(|text| !text.trim().is_empty())
    };
    let numeric_with_unit = |candidate: &Value, allow_zero: bool| {
        let Some(object) = candidate.as_object() else {
            return false;
        };
        let number = object.get("value").and_then(Value::as_f64);
        number.is_some_and(|number| {
            number.is_finite() && if allow_zero { number >= 0.0 } else { number > 0.0 }
        }) && object.get("unit").is_some_and(nonempty_string)
    };

    match field {
        "enzyme_purity" | "enzyme_form" | "buffer_system" | "replicate_structure"
        | "reaction_equation" | "kinetic_equation" => nonempty_string(value),
        "enzyme_concentration" | "buffer_concentration" | "assay_pressure" => {
            numeric_with_unit(value, false)
        }
        "ph_measurement_temperature_c" => value
            .as_f64()
            .is_some_and(|temperature| temperature.is_finite() && temperature > -273.15),
        "substrate_concentration_range" => {
            let Some(object) = value.as_object() else {
                return false;
            };
            let lower = object.get("lower").and_then(Value::as_f64);
            let upper = object.get("upper").and_then(Value::as_f64);
            lower.is_some_and(|number| number.is_finite() && number >= 0.0)
                && upper.is_some_and(|number| {
                    number.is_finite() && lower.is_some_and(|lower| number > lower)
                })
                && object.get("unit").is_some_and(nonempty_string)
        }
        "replicate_count" | "independent_experiment_count" => {
            value.as_u64().is_some_and(|count| count > 0)
        }
        "measurement_precision" => {
            let Some(object) = value.as_object() else {
                return false;
            };
            object.get("measure").is_some_and(nonempty_string)
                && object.get("value").and_then(Value::as_f64).is_some_and(|number| number.is_finite() && number >= 0.0)
                && object.get("unit").is_some_and(nonempty_string)
        }
        "detection_limit" => numeric_with_unit(value, true),
        "analysis_method" => {
            let Some(object) = value.as_object() else {
                return false;
            };
            ["software", "version", "method"]
                .iter()
                .all(|key| object.get(*key).is_some_and(nonempty_string))
        }
        "raw_measurement_data" => value
            .as_str()
            .is_some_and(|url| is_valid_https_url(url) && !url.trim().is_empty()),
        "raw_measurement_data_sha256" => value
            .as_str()
            .is_some_and(|digest| digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())),
        "initial_rate_method" | "controls_description" => nonempty_string(value),
        _ => false,
    }
}

fn is_valid_https_url(value: &str) -> bool {
    let Ok(url) = Url::parse(value) else {
        return false;
    };
    url.scheme() == "https"
        && url.host_str().is_some_and(|host| !host.trim().is_empty())
        && url.username().is_empty()
        && url.password().is_none()
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
            let value = condition
                .get("value")
                .filter(|value| !value.is_null())
                .with_context(|| format!("{key} reported without a value at CSV record {csv_record}"))?;
            match key {
                "ph" | "temperature_c" => ensure!(
                    value.as_f64().is_some_and(f64::is_finite),
                    "{key} must be a finite number at CSV record {csv_record}"
                ),
                "substrate_id" => ensure!(
                    value.as_str().is_some_and(|s| !s.trim().is_empty()),
                    "substrate_id must be a nonempty string at CSV record {csv_record}"
                ),
                _ => bail!("unsupported assay context key: {key}"),
            }
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
    let canonical_root = fs::canonicalize(root)
        .with_context(|| format!("canonicalize workspace root {}", root.display()))?;
    let candidate = safe_workspace_path(&canonical_root, &artifact.path)
        .with_context(|| format!("invalid {label} path"))?;
    let path = fs::canonicalize(&candidate)
        .with_context(|| format!("canonicalize {label} path {}", candidate.display()))?;
    ensure!(
        path.starts_with(&canonical_root),
        "{label} path escapes workspace root through a symlink: {}",
        artifact.path
    );
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
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || !bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit())
    {
        return false;
    }

    let Ok(year) = value[0..4].parse::<u32>() else {
        return false;
    };
    let Ok(month) = value[5..7].parse::<u32>() else {
        return false;
    };
    let Ok(day) = value[8..10].parse::<u32>() else {
        return false;
    };
    if year == 0 || !(1..=12).contains(&month) {
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

    #[test]
    fn kcat_profile_covers_core_and_all_audited_optional_fields() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask should be one level below workspace root");
        let profile_path = workspace_root.join("docs/research/KCAT_ADMISSION_PROFILE_V0.json");
        let profile: Value = serde_json::from_slice(&fs::read(profile_path).unwrap()).unwrap();
        assert_eq!(profile["profile_id"], "kcat-admission-profile-v0");
        assert_eq!(profile["status"], "tier_counter_implemented_policy_review_required");
        assert_eq!(profile["tiers"].as_array().unwrap().len(), 4);

        let configured: BTreeSet<String> = profile["field_rules"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|field| field["field"].as_str().map(str::to_string))
            .collect();
        for required in ["ph", "temperature_c", "substrate_id"] {
            assert!(configured.contains(required), "profile missing core field {required}");
        }
        for required in OPTIONAL_METADATA_FIELDS {
            assert!(configured.contains(*required), "profile missing audited field {required}");
        }
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
        writer.write_record(REQUIRED_COLUMNS.iter().copied()).unwrap();
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
            "obs-a", "SYNTHETIC_TEST_FIXTURE", "test-v1", "source-a",
            "https://example.invalid/source-a", "synthetic:record-a", "study-a",
            "synthetic:assay-method-v1", "protein-a", "family-a", "kcat",
            "2.0", "s^-1", "2.0", "s^-1", "kcat_s-1_identity_v1", "1.0", context_a_json.as_str()
        ]).unwrap();
        writer.write_record([
            "obs-b", "SYNTHETIC_TEST_FIXTURE", "test-v1", "source-b",
            "https://example.invalid/source-b", "synthetic:record-b", "study-b",
            "synthetic:assay-method-v1", "protein-b", "family-b", "kcat",
            "3.0", "s^-1", "3.0", "s^-1", "kcat_s-1_identity_v1", "1.0", context_b_json.as_str()
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
    fn rejects_manifest_path_outside_workspace() {
        let root = fixture_root();
        let value = make_manifest(&root);
        let external_manifest = root.with_extension("outside-manifest.json");
        fs::write(&external_manifest, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

        let error = check(&root, &external_manifest).unwrap_err().to_string();
        assert!(error.contains("manifest path escapes workspace root"));

        fs::remove_file(external_manifest).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_embedded_credentials_in_manifest_source_url() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        value["source"]["url"] = json!("https://user:secret@example.invalid/dataset");
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("source.url must be a valid HTTPS URL"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_duplicate_csv_header_names() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        let csv_path = root.join("data/synthetic_enzyme_observations.csv");
        let original = fs::read(&csv_path).unwrap();
        let mut reader = csv::Reader::from_reader(original.as_slice());
        let headers = reader.headers().unwrap().clone();
        let mut header_fields: Vec<String> = headers.iter().map(str::to_string).collect();
        let source_id_index = headers.iter().position(|name| name == "source_record_id").unwrap();
        header_fields[source_id_index] = "source_record_url".to_string();
        let mut writer = csv::Writer::from_writer(Vec::new());
        writer.write_record(&header_fields).unwrap();
        for row in reader.records() {
            writer.write_record(row.unwrap().iter()).unwrap();
        }
        let changed = writer.into_inner().unwrap();
        fs::write(&csv_path, &changed).unwrap();
        value["dataset_artifact"]["sha256"] = json!(format!("{:x}", Sha256::digest(&changed)));
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("dataset CSV contains duplicate column names"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_row_source_id_that_differs_from_manifest_rights_review() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        let dataset_bytes = rewrite_csv_field(
            &root,
            "obs-a",
            "source_id",
            "UNREVIEWED_SECOND_SOURCE",
        );
        value["dataset_artifact"]["sha256"] = json!(format!("{:x}", Sha256::digest(&dataset_bytes)));
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("source_id differs from manifest"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_row_source_release_that_differs_from_manifest() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        let dataset_bytes = rewrite_csv_field(&root, "obs-a", "source_release", "unreviewed-new-release");
        value["dataset_artifact"]["sha256"] = json!(format!("{:x}", Sha256::digest(&dataset_bytes)));
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("source_release differs from manifest"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reported_optional_metadata_requires_field_specific_shapes() {
        assert!(!optional_metadata_value_is_valid("enzyme_purity", &json!("  ")));
        assert!(optional_metadata_value_is_valid("enzyme_purity", &json!("95% by SDS-PAGE")));
        assert!(!optional_metadata_value_is_valid("enzyme_concentration", &json!({"value":0,"unit":"mg/mL"})));
        assert!(optional_metadata_value_is_valid("enzyme_concentration", &json!({"value":2.5,"unit":"mg/mL"})));
        assert!(!optional_metadata_value_is_valid("assay_pressure", &json!({"value":101.3})));
        assert!(!optional_metadata_value_is_valid("replicate_count", &json!(0)));
        assert!(!optional_metadata_value_is_valid("independent_experiment_count", &json!(-1)));
        assert!(optional_metadata_value_is_valid("replicate_structure", &json!("three independent enzyme preparations")));
        assert!(!optional_metadata_value_is_valid("measurement_precision", &json!({"measure":"SD","value":-0.1,"unit":"s^-1"})));
        assert!(!optional_metadata_value_is_valid("detection_limit", &json!({"value":0.2,"unit":""})));
        assert!(!optional_metadata_value_is_valid("analysis_method", &json!({"software":"fit"})));
        assert!(!optional_metadata_value_is_valid("raw_measurement_data", &json!("file:///etc/passwd")));
        assert!(optional_metadata_value_is_valid("raw_measurement_data", &json!("https://example.org/raw-data.csv")));
    }

    #[test]
    fn metadata_coverage_accounts_for_every_observation_exactly_once() {
        let mut coverage = empty_metadata_coverage();
        let rows = [
            json!({
                "enzyme_purity":{"status":"reported","value":"purified"},
                "enzyme_concentration":{"status":"not_reported"},
                "buffer_system":{"status":"not_applicable","reason":"no buffer used"}
            }),
            json!({
                "enzyme_purity":{"status":"reported"},
                "enzyme_concentration":"malformed",
                "buffer_system":{"status":"not_reported"}
            }),
            json!({}),
        ];
        for row in &rows {
            record_optional_metadata_coverage(row, &mut coverage);
        }
        for (field, counts) in &coverage {
            assert_eq!(
                counts.accounted_observations(),
                rows.len(),
                "coverage must account for each row once for {field}"
            );
        }
        assert_eq!(coverage["enzyme_purity"].reported, 2);
        assert_eq!(coverage["enzyme_purity"].invalid, 1);
        assert_eq!(coverage["enzyme_concentration"].not_reported, 1);
        assert_eq!(coverage["enzyme_concentration"].invalid, 1);
        assert_eq!(coverage["enzyme_concentration"].unrecorded, 1);
        assert_eq!(coverage["buffer_system"].not_applicable, 1);
        assert_eq!(coverage["buffer_system"].not_reported, 1);
        assert_eq!(coverage["buffer_system"].unrecorded, 1);
    }

    #[test]
    fn kcat_context_metadata_validates_range_and_measurement_temperature_shapes() {
        assert!(optional_metadata_value_is_valid(
            "substrate_concentration_range",
            &json!({"lower":0,"upper":150,"unit":"umol/L"})
        ));
        assert!(!optional_metadata_value_is_valid(
            "substrate_concentration_range",
            &json!({"lower":150,"upper":150,"unit":"umol/L"})
        ));
        assert!(!optional_metadata_value_is_valid(
            "substrate_concentration_range",
            &json!({"lower":0,"upper":150,"unit":""})
        ));
        assert!(optional_metadata_value_is_valid("ph_measurement_temperature_c", &json!(25.0)));
        assert!(!optional_metadata_value_is_valid("ph_measurement_temperature_c", &json!(-300.0)));
        assert!(optional_metadata_value_is_valid("kinetic_equation", &json!("Michaelis-Menten; initial rate vs substrate concentration")));
        assert!(!optional_metadata_value_is_valid("enzyme_form", &json!(" ")));
        assert!(optional_metadata_value_is_valid("buffer_concentration", &json!({"value":50,"unit":"mmol/L"})));
    }

    #[test]
    fn evidence_tiers_are_nested_and_require_context_and_raw_replay_evidence() {
        let mut context = json!({
            "ph":{"status":"reported","value":7.0},
            "temperature_c":{"status":"reported","value":25.0},
            "substrate_id":{"status":"reported","value":"substrate-x"},
            "enzyme_form":{"status":"reported","value":"homodimer; kcat relative to dimer"},
            "substrate_concentration_range":{"status":"reported","value":{"lower":0.0,"upper":2.0,"unit":"mM"}},
            "reaction_equation":{"status":"reported","value":"S + O2 -> P"},
            "kinetic_equation":{"status":"reported","value":"Michaelis-Menten; initial rate as a function of [S]"},
            "buffer_system":{"status":"reported","value":"50 mM phosphate buffer"},
            "buffer_concentration":{"status":"reported","value":{"value":50.0,"unit":"mM"}},
            "assay_pressure":{"status":"not_applicable","reason":"ambient atmospheric pressure"}
        });
        assert!(is_context_qualified(&context));
        assert!(!is_primary_comparison_candidate(&context));
        assert!(!is_kinetic_reproduction_candidate(&context));

        let object = context.as_object_mut().unwrap();
        object.insert("enzyme_purity".to_string(),json!({"status":"reported","value":"apparently homogeneous by PAGE"}));
        object.insert("enzyme_concentration".to_string(),json!({"status":"reported","value":{"value":0.5,"unit":"uM"}}));
        object.insert("replicate_count".to_string(),json!({"status":"reported","value":3}));
        object.insert("independent_experiment_count".to_string(),json!({"status":"reported","value":3}));
        object.insert("replicate_structure".to_string(),json!({"status":"reported","value":"three independent preparations"}));
        object.insert("measurement_precision".to_string(),json!({"status":"reported","value":{"measure":"SD","value":0.02,"unit":"s^-1"}}));
        object.insert("analysis_method".to_string(),json!({"status":"reported","value":{"software":"fit-tool","version":"1.2.0","method":"nonlinear least squares"}}));
        assert!(is_primary_comparison_candidate(&context));
        assert!(!is_kinetic_reproduction_candidate(&context));

        object.insert("raw_measurement_data".to_string(),json!({"status":"reported","value":"https://example.org/raw.csv"}));
        object.insert("raw_measurement_data_sha256".to_string(),json!({"status":"reported","value":"a".repeat(64)}));
        object.insert("initial_rate_method".to_string(),json!({"status":"reported","value":"linear slope over preregistered initial window"}));
        object.insert("controls_description".to_string(),json!({"status":"reported","value":"blank and no-enzyme controls applied"}));
        assert!(is_kinetic_reproduction_candidate(&context));

        object.insert("enzyme_concentration".to_string(),json!({"status":"reported","value":{"value":0.5,"unit":"mg/mL"}}));
        assert!(!is_primary_comparison_candidate(&context));
    }

    #[test]
    fn metadata_coverage_separates_missingness_from_invalid_statuses() {
        let mut coverage = empty_metadata_coverage();
        let row_a = json!({
            "enzyme_purity": {"status":"reported", "value":"apparent homogeneity by PAGE"},
            "enzyme_form": {"status":"reported", "value":"homodimer; concentration relative to dimer"},
            "enzyme_concentration": {"status":"not_reported"},
            "buffer_system": {"status":"not_applicable", "reason":"not specified in source record"},
            "buffer_concentration": {"status":"reported", "value":{"value":50,"unit":"mmol/L"}},
            "assay_pressure": {"status":"reported", "value":{"value":101.325,"unit":"kPa"}},
            "ph_measurement_temperature_c": {"status":"reported", "value":25},
            "substrate_concentration_range": {"status":"reported", "value":{"lower":0,"upper":150,"unit":"umol/L"}},
            "kinetic_equation": {"status":"reported", "value":"Michaelis-Menten; initial rate vs substrate concentration"},
            "replicate_count": {"status":"reported", "value":3},
            "independent_experiment_count": {"status":"reported", "value":2},
            "replicate_structure": {"status":"reported", "value":"two independent enzyme preparations"},
            "measurement_precision": {"status":"not_reported", "value":null},
            "detection_limit": {"status":"not_applicable", "reason":"positive kcat estimate reported"},
            "reaction_equation": {"status":"unknown", "value":"A -> B"},
            "analysis_method": {"status":"reported"},
            "raw_measurement_data": "malformed"
        });
        let row_b = json!({
            "enzyme_purity": {"status":"reported"},
            "enzyme_concentration": {"status":"unknown"}
        });
        record_optional_metadata_coverage(&row_a, &mut coverage);
        record_optional_metadata_coverage(&row_b, &mut coverage);

        assert_eq!(coverage["enzyme_purity"].reported, 1);
        assert_eq!(coverage["enzyme_purity"].invalid, 1);
        assert_eq!(coverage["enzyme_concentration"].not_reported, 1);
        assert_eq!(coverage["enzyme_concentration"].invalid, 1);
        assert_eq!(coverage["buffer_system"].not_applicable, 1);
        assert_eq!(coverage["buffer_system"].unrecorded, 1);
        assert_eq!(coverage["assay_pressure"].reported, 1);
        assert_eq!(coverage["replicate_count"].reported, 1);
        assert_eq!(coverage["independent_experiment_count"].reported, 1);
        assert_eq!(coverage["replicate_structure"].reported, 1);
        assert_eq!(coverage["measurement_precision"].not_reported, 1);
        assert_eq!(coverage["detection_limit"].not_applicable, 1);
        assert_eq!(coverage["reaction_equation"].invalid, 1);
        assert_eq!(coverage["analysis_method"].invalid, 1);
        assert_eq!(coverage["raw_measurement_data"].invalid, 1);
    }

    #[test]
    fn accepts_complete_synthetic_fixture_and_reports_counts() {
        let root = fixture_root();
        let manifest = make_manifest(&root);
        let parsed: Manifest = serde_json::from_value(manifest).unwrap();
        let report = validate(&root, &parsed).unwrap();
        assert_eq!(report.observations, 2);
        assert_eq!(report.train_studies, 1);
        assert_eq!(report.test_studies, 1);
        assert_eq!(report.train_sequence_families, 1);
        assert_eq!(report.test_sequence_families, 1);
        assert_eq!(report.cross_split_families, 0);
        assert_eq!(report.cross_split_studies, 0);
        assert_eq!(report.metadata_coverage.len(), OPTIONAL_METADATA_FIELDS.len());
        assert_eq!(report.metadata_coverage["enzyme_purity"].unrecorded, 2);
        assert_eq!(report.kcat_evidence_tiers.context_qualified, 0);
        assert_eq!(report.kcat_evidence_tiers.primary_comparison_candidates, 0);
        assert_eq!(report.kcat_evidence_tiers.kinetic_reproduction_candidates, 0);
        assert_eq!(report.row_evidence_tiers.len(), report.observations);
        assert!(report.row_evidence_tiers.iter().all(|row| row.tier == "review_only"));
        fs::remove_dir_all(root).unwrap();
    }

    fn rewrite_csv_field(root: &Path, observation_id: &str, field_name: &str, value: &str) -> Vec<u8> {
        let csv_path = root.join("data/synthetic_enzyme_observations.csv");
        let original = fs::read(&csv_path).unwrap();
        let mut reader = csv::Reader::from_reader(original.as_slice());
        let headers = reader.headers().unwrap().clone();
        let observation_index = headers.iter().position(|name| name == "observation_id").unwrap();
        let field_index = headers.iter().position(|name| name == field_name).unwrap();
        let mut writer = csv::Writer::from_writer(Vec::new());
        writer.write_record(&headers).unwrap();
        for row in reader.records() {
            let mut fields: Vec<String> = row.unwrap().iter().map(str::to_string).collect();
            if fields[observation_index].as_str() == observation_id {
                fields[field_index] = value.to_string();
            }
            writer.write_record(fields).unwrap();
        }
        let changed = writer.into_inner().unwrap();
        fs::write(csv_path, &changed).unwrap();
        changed
    }

    fn replace_split_manifest(root: &Path, manifest: &mut Value, split: Value) {
        let bytes = serde_json::to_vec_pretty(&split).unwrap();
        fs::write(root.join("data/splits.json"), &bytes).unwrap();
        manifest["split_manifest_artifact"]["sha256"] =
            json!(format!("{:x}", Sha256::digest(&bytes)));
    }

    #[test]
    fn study_holdout_reports_family_overlap_instead_of_silently_claiming_family_generalization() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        let dataset_bytes = rewrite_csv_field(&root, "obs-b", "sequence_family_id", "family-a");
        value["dataset_artifact"]["sha256"] = json!(format!("{:x}", Sha256::digest(&dataset_bytes)));
        let split = json!({
            "schema_version": 1,
            "split_mode": "study_holdout",
            "train_study_ids": ["study-a"],
            "test_study_ids": ["study-b"],
            "train_sequence_family_ids": [],
            "test_sequence_family_ids": []
        });
        replace_split_manifest(&root, &mut value, split);
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let report = validate(&root, &parsed).unwrap();
        assert_eq!(report.cross_split_families, 1);
        assert_eq!(report.cross_split_studies, 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sequence_family_holdout_reports_study_overlap_instead_of_silently_claiming_study_independence() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        let dataset_bytes = rewrite_csv_field(&root, "obs-b", "source_study_id", "study-a");
        value["dataset_artifact"]["sha256"] = json!(format!("{:x}", Sha256::digest(&dataset_bytes)));
        let split = json!({
            "schema_version": 1,
            "split_mode": "sequence_family_holdout",
            "train_study_ids": [],
            "test_study_ids": [],
            "train_sequence_family_ids": ["family-a"],
            "test_sequence_family_ids": ["family-b"]
        });
        replace_split_manifest(&root, &mut value, split);
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let report = validate(&root, &parsed).unwrap();
        assert_eq!(report.cross_split_studies, 1);
        assert_eq!(report.cross_split_families, 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_artifact_symlink_escape_even_when_external_bytes_match_hash() {
        use std::os::unix::fs::symlink;

        let root = fixture_root();
        let mut value = make_manifest(&root);
        let external = root.with_extension("outside.csv");
        let external_bytes = b"not in the benchmark workspace";
        fs::write(&external, external_bytes).unwrap();
        symlink(&external, root.join("data/escape.csv")).unwrap();
        value["dataset_artifact"]["path"] = json!("data/escape.csv");
        value["dataset_artifact"]["sha256"] =
            json!(format!("{:x}", Sha256::digest(external_bytes)));

        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("escapes workspace root through a symlink"));
        fs::remove_file(external).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn access_date_must_be_a_real_calendar_date() {
        assert!(is_date_shape("2026-10-15"));
        assert!(is_date_shape("2024-02-29"));
        assert!(!is_date_shape("2026-02-29"));
        assert!(!is_date_shape("2026-04-31"));
        assert!(!is_date_shape("2026-13-01"));
        assert!(!is_date_shape("0000-01-01"));
    }

    #[test]
    fn rejects_malformed_https_source_url() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        value["source"]["url"] = json!("https://");
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("source.url must be a valid HTTPS URL"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_malformed_https_source_record_url() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        let dataset_bytes = rewrite_csv_field(&root, "obs-a", "source_record_url", "https://");
        value["dataset_artifact"]["sha256"] = json!(format!("{:x}", Sha256::digest(&dataset_bytes)));
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("source_record_url must be a valid HTTPS URL"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_non_https_source_record_locator() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        let dataset_bytes = rewrite_csv_field(
            &root,
            "obs-a",
            "source_record_url",
            "http://example.invalid/source-a",
        );
        value["dataset_artifact"]["sha256"] = json!(format!("{:x}", Sha256::digest(&dataset_bytes)));
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("source_record_url must be HTTPS"));
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
    fn rejects_unit_conversion_factor_without_matching_reviewed_rule() {
        let root = fixture_root();
        let mut value = make_manifest(&root);
        let dataset_bytes = rewrite_csv_field(&root, "obs-a", "unit_conversion_factor", "2.0");
        value["dataset_artifact"]["sha256"] = json!(format!("{:x}", Sha256::digest(&dataset_bytes)));
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        let error = validate(&root, &parsed).unwrap_err().to_string();
        assert!(error.contains("unit conversion factor does not match reported_unit"));
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
