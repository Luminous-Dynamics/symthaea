// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed audit of fermentation evaluation split manifests.
//!
//! This checks split integrity and time availability, not model quality or the
//! biological independence of group IDs. Group IDs must be justified from the
//! underlying experimental design, not inferred from technical replicates.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const SCHEMA_VERSION: &str = "beverage-split-manifest-v1";
const MINIMUM_REQUIRED_GROUPS: usize = 2;
const CLAIM_CEILING: &str = "This receipt checks manifest structure, declared source-digest format, sample-ID uniqueness, declared group isolation across train/validation/test, pre-cutoff feature availability, future target timing, and the declared minimum group count. It does not authenticate the dataset digest, establish that group IDs are truly independent biological units, validate a model, establish transfer, or prove any physical beverage outcome.";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "lowercase")]
enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    const ALL: [Self; 3] = [Self::Train, Self::Validation, Self::Test];
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitManifest {
    schema_version: String,
    evidence_class: String,
    task_id: String,
    dataset_id: String,
    dataset_sha256: String,
    time_coordinate: String,
    minimum_groups_per_split: usize,
    records: Vec<SplitRecord>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitRecord {
    sample_id: String,
    group_id: String,
    split: Split,
    max_feature_available_time: f64,
    prediction_cutoff_time: f64,
    target_observed_time: f64,
}

#[derive(Debug, Serialize)]
struct SplitAuditReceipt {
    receipt_schema: &'static str,
    status: &'static str,
    manifest_sha256: String,
    manifest_parse_valid: bool,
    schema_version: Option<String>,
    evidence_class: Option<String>,
    task_id: Option<String>,
    dataset_id: Option<String>,
    dataset_sha256: Option<String>,
    dataset_sha256_format_valid: bool,
    record_count: usize,
    records_per_split: BTreeMap<Split, usize>,
    declared_groups_per_split: BTreeMap<Split, usize>,
    required_splits_present: bool,
    sample_ids_unique: bool,
    group_assignments_isolated: bool,
    features_respect_prediction_cutoff: bool,
    targets_observed_after_prediction_cutoff: bool,
    minimum_group_count_met: bool,
    duplicate_sample_ids: Vec<String>,
    groups_crossing_splits: Vec<String>,
    post_cutoff_feature_records: Vec<String>,
    non_future_target_records: Vec<String>,
    groups_below_minimum: Vec<String>,
    validation_errors: Vec<String>,
    claim_ceiling: &'static str,
}

fn hex_sha256(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}")
            .expect("writing to a String cannot fail");
    }
    output
}

fn digest_is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn clean_required_value(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

fn empty_split_counts() -> BTreeMap<Split, usize> {
    Split::ALL.into_iter().map(|split| (split, 0)).collect()
}

fn parse_failure_receipt(bytes: &[u8], error: String) -> SplitAuditReceipt {
    SplitAuditReceipt {
        receipt_schema: "beverage-split-audit-receipt-v1",
        status: "FAIL",
        manifest_sha256: hex_sha256(bytes),
        manifest_parse_valid: false,
        schema_version: None,
        evidence_class: None,
        task_id: None,
        dataset_id: None,
        dataset_sha256: None,
        dataset_sha256_format_valid: false,
        record_count: 0,
        records_per_split: empty_split_counts(),
        declared_groups_per_split: empty_split_counts(),
        required_splits_present: false,
        sample_ids_unique: false,
        group_assignments_isolated: false,
        features_respect_prediction_cutoff: false,
        targets_observed_after_prediction_cutoff: false,
        minimum_group_count_met: false,
        duplicate_sample_ids: Vec::new(),
        groups_crossing_splits: Vec::new(),
        post_cutoff_feature_records: Vec::new(),
        non_future_target_records: Vec::new(),
        groups_below_minimum: Vec::new(),
        validation_errors: vec![format!("manifest_parse_error:{error}")],
        claim_ceiling: CLAIM_CEILING,
    }
}

fn audit_manifest_bytes(bytes: &[u8]) -> SplitAuditReceipt {
    let manifest: SplitManifest = match serde_json::from_slice(bytes) {
        Ok(manifest) => manifest,
        Err(error) => return parse_failure_receipt(bytes, error.to_string()),
    };

    let mut validation_errors = Vec::new();
    if manifest.schema_version != SCHEMA_VERSION {
        validation_errors.push("schema_version_mismatch".to_owned());
    }
    if ![
        "SYNTHETIC_CONTRACT_FIXTURE",
        "MEASURED_SOURCE_SPLIT",
        "SIMULATED_SOURCE_SPLIT",
    ]
    .contains(&manifest.evidence_class.as_str())
    {
        validation_errors.push("unsupported_evidence_class".to_owned());
    }
    if !clean_required_value(&manifest.task_id) {
        validation_errors.push("invalid_task_id".to_owned());
    }
    if !clean_required_value(&manifest.dataset_id) {
        validation_errors.push("invalid_dataset_id".to_owned());
    }
    if !clean_required_value(&manifest.time_coordinate) {
        validation_errors.push("invalid_time_coordinate".to_owned());
    }
    let dataset_sha256_format_valid =
        digest_is_lower_hex_sha256(&manifest.dataset_sha256);
    if !dataset_sha256_format_valid {
        validation_errors.push(
            "dataset_sha256_must_be_64_lowercase_hex_characters".to_owned(),
        );
    }

    let mut records_per_split = empty_split_counts();
    let mut groups_per_split: BTreeMap<Split, BTreeSet<String>> = Split::ALL
        .into_iter()
        .map(|split| (split, BTreeSet::new()))
        .collect();
    let mut seen_sample_ids = HashSet::new();
    let mut duplicate_sample_ids = BTreeSet::new();
    let mut group_assignments: HashMap<String, Split> = HashMap::new();
    let mut groups_crossing_splits = BTreeSet::new();
    let mut post_cutoff_feature_records = BTreeSet::new();
    let mut non_future_target_records = BTreeSet::new();

    for record in &manifest.records {
        *records_per_split.entry(record.split).or_insert(0) += 1;

        if !clean_required_value(&record.sample_id) {
            validation_errors.push("invalid_sample_id".to_owned());
        }
        if !clean_required_value(&record.group_id) {
            validation_errors.push("invalid_group_id".to_owned());
        }
        if !seen_sample_ids.insert(record.sample_id.clone()) {
            duplicate_sample_ids.insert(record.sample_id.clone());
        }

        if clean_required_value(&record.group_id) {
            groups_per_split
                .entry(record.split)
                .or_default()
                .insert(record.group_id.clone());

            match group_assignments.get(&record.group_id) {
                Some(previous_split) if *previous_split != record.split => {
                    groups_crossing_splits.insert(record.group_id.clone());
                }
                Some(_) => {}
                None => {
                    group_assignments
                        .insert(record.group_id.clone(), record.split);
                }
            }
        }

        if !record.max_feature_available_time.is_finite()
            || !record.prediction_cutoff_time.is_finite()
            || !record.target_observed_time.is_finite()
        {
            validation_errors.push(format!(
                "non_finite_time_value:{}",
                record.sample_id
            ));
            continue;
        }

        if record.max_feature_available_time > record.prediction_cutoff_time {
            post_cutoff_feature_records.insert(record.sample_id.clone());
        }
        if record.target_observed_time <= record.prediction_cutoff_time {
            non_future_target_records.insert(record.sample_id.clone());
        }
    }

    let declared_groups_per_split = groups_per_split
        .iter()
        .map(|(split, groups)| (*split, groups.len()))
        .collect::<BTreeMap<_, _>>();

    let required_splits_present = Split::ALL
        .into_iter()
        .all(|split| records_per_split.get(&split).copied().unwrap_or(0) > 0);
    if !required_splits_present {
        validation_errors
            .push("train_validation_test_splits_are_all_required".to_owned());
    }

    let minimum_group_count_met =
        manifest.minimum_groups_per_split >= MINIMUM_REQUIRED_GROUPS
        && Split::ALL.into_iter().all(|split| {
            declared_groups_per_split
                .get(&split)
                .copied()
                .unwrap_or(0)
                >= manifest.minimum_groups_per_split
        });
    if manifest.minimum_groups_per_split < MINIMUM_REQUIRED_GROUPS {
        validation_errors.push(format!(
            "minimum_groups_per_split_must_be_at_least_{MINIMUM_REQUIRED_GROUPS}"
        ));
    }
    let groups_below_minimum = Split::ALL
        .into_iter()
        .filter(|split| {
            declared_groups_per_split
                .get(split)
                .copied()
                .unwrap_or(0)
                < manifest.minimum_groups_per_split
        })
        .map(|split| format!("{split:?}"))
        .collect::<Vec<_>>();
    if !groups_below_minimum.is_empty() {
        validation_errors.push(format!(
            "insufficient_declared_groups_in_splits:{}",
            groups_below_minimum.join(",")
        ));
    }

    let sample_ids_unique = duplicate_sample_ids.is_empty();
    if !sample_ids_unique {
        validation_errors.push("duplicate_sample_ids_detected".to_owned());
    }
    let group_assignments_isolated = groups_crossing_splits.is_empty();
    if !group_assignments_isolated {
        validation_errors.push(
            "group_ids_cross_train_validation_test_boundaries".to_owned(),
        );
    }
    let features_respect_prediction_cutoff =
        post_cutoff_feature_records.is_empty();
    if !features_respect_prediction_cutoff {
        validation_errors.push(
            "features_available_after_prediction_cutoff_detected".to_owned(),
        );
    }
    let targets_observed_after_prediction_cutoff =
        non_future_target_records.is_empty();
    if !targets_observed_after_prediction_cutoff {
        validation_errors.push(
            "target_not_strictly_after_prediction_cutoff_detected".to_owned(),
        );
    }

    let passed = validation_errors.is_empty()
        && dataset_sha256_format_valid
        && required_splits_present
        && sample_ids_unique
        && group_assignments_isolated
        && features_respect_prediction_cutoff
        && targets_observed_after_prediction_cutoff
        && minimum_group_count_met;

    SplitAuditReceipt {
        receipt_schema: "beverage-split-audit-receipt-v1",
        status: if passed { "PASS" } else { "FAIL" },
        manifest_sha256: hex_sha256(bytes),
        manifest_parse_valid: true,
        schema_version: Some(manifest.schema_version),
        evidence_class: Some(manifest.evidence_class),
        task_id: Some(manifest.task_id),
        dataset_id: Some(manifest.dataset_id),
        dataset_sha256: Some(manifest.dataset_sha256),
        dataset_sha256_format_valid,
        record_count: manifest.records.len(),
        records_per_split,
        declared_groups_per_split,
        required_splits_present,
        sample_ids_unique,
        group_assignments_isolated,
        features_respect_prediction_cutoff,
        targets_observed_after_prediction_cutoff,
        minimum_group_count_met,
        duplicate_sample_ids: duplicate_sample_ids.into_iter().collect(),
        groups_crossing_splits: groups_crossing_splits.into_iter().collect(),
        post_cutoff_feature_records: post_cutoff_feature_records
            .into_iter()
            .collect(),
        non_future_target_records: non_future_target_records
            .into_iter()
            .collect(),
        groups_below_minimum,
        validation_errors,
        claim_ceiling: CLAIM_CEILING,
    }
}

fn parse_args() -> Result<(PathBuf, PathBuf), String> {
    let mut input = None;
    let mut output = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing_value_for_{flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--out" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown_argument_{flag}")),
        }
    }

    Ok((
        input.ok_or_else(|| "missing_--input".to_owned())?,
        output.ok_or_else(|| "missing_--out".to_owned())?,
    ))
}

fn run() -> Result<ExitCode, String> {
    let (input_path, output_path) = parse_args()?;
    let input_bytes = fs::read(&input_path)
        .map_err(|error| format!("manifest_read_failed:{error}"))?;
    let receipt = audit_manifest_bytes(&input_bytes);
    let passed = receipt.status == "PASS";
    let serialized = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("receipt_serialization_failed:{error}"))?;
    fs::write(&output_path, serialized)
        .map_err(|error| format!("receipt_write_failed:{error}"))?;
    if passed {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::FAILURE)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("beverage-split-leakage-audit:{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &[u8] =
        include_bytes!("../../../../docs/research/beverage-split-manifest-synthetic-v1.json");

    fn fixture_value() -> Value {
        serde_json::from_slice(FIXTURE)
            .expect("checked-in fixture must be valid JSON")
    }

    fn audit_value(value: Value) -> SplitAuditReceipt {
        let bytes = serde_json::to_vec(&value)
            .expect("test fixture serialization must succeed");
        audit_manifest_bytes(&bytes)
    }

    #[test]
    fn checked_in_synthetic_manifest_passes() {
        let receipt = audit_manifest_bytes(FIXTURE);
        assert_eq!(receipt.status, "PASS", "{:?}", receipt.validation_errors);
        assert_eq!(receipt.record_count, 6);
        assert!(receipt.required_splits_present);
        assert!(receipt.sample_ids_unique);
        assert!(receipt.group_assignments_isolated);
        assert!(receipt.features_respect_prediction_cutoff);
        assert!(receipt.targets_observed_after_prediction_cutoff);
        assert!(receipt.minimum_group_count_met);
        assert_eq!(receipt.manifest_sha256.len(), 64);
    }

    #[test]
    fn same_declared_group_across_splits_fails() {
        let mut value = fixture_value();
        value["records"][2]["group_id"] = "batch-001".into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.group_assignments_isolated);
        assert_eq!(receipt.groups_crossing_splits, vec!["batch-001"]);
    }

    #[test]
    fn feature_after_prediction_cutoff_fails() {
        let mut value = fixture_value();
        value["records"][0]["max_feature_available_time"] = 24.5.into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.features_respect_prediction_cutoff);
        assert_eq!(receipt.post_cutoff_feature_records, vec!["sample-001"]);
    }

    #[test]
    fn target_must_be_strictly_after_prediction_cutoff() {
        for target_time in [24.0, 18.0] {
            let mut value = fixture_value();
            value["records"][0]["target_observed_time"] = target_time.into();
            let receipt = audit_value(value);
            assert_eq!(receipt.status, "FAIL");
            assert!(!receipt.targets_observed_after_prediction_cutoff);
        }
    }

    #[test]
    fn duplicate_sample_id_fails() {
        let mut value = fixture_value();
        let first_id = value["records"][0]["sample_id"].clone();
        value["records"][5]["sample_id"] = first_id;
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.sample_ids_unique);
        assert_eq!(receipt.duplicate_sample_ids, vec!["sample-001"]);
    }

    #[test]
    fn too_few_declared_groups_fails() {
        let mut value = fixture_value();
        value["minimum_groups_per_split"] = 3.into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.minimum_group_count_met);
        assert_eq!(receipt.groups_below_minimum.len(), 3);
    }

    #[test]
    fn malformed_dataset_digest_fails() {
        let mut value = fixture_value();
        value["dataset_sha256"] = "z".repeat(64).into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.dataset_sha256_format_valid);
    }

    #[test]
    fn missing_validation_split_fails() {
        let mut value = fixture_value();
        value["records"].as_array_mut().unwrap().retain(|record| {
            record["split"].as_str() != Some("validation")
        });
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.required_splits_present);
        assert!(!receipt.minimum_group_count_met);
    }

    #[test]
    fn malformed_json_emits_a_fail_receipt() {
        let receipt = audit_manifest_bytes(b"{");
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.manifest_parse_valid);
        assert!(
            receipt
                .validation_errors
                .iter()
                .any(|error| error.starts_with("manifest_parse_error:"))
        );
    }

    #[test]
    fn unknown_fields_fail_closed_during_deserialization() {
        let mut value = fixture_value();
        value["optimistically_impute_missing"] = true.into();
        let receipt = audit_value(value);
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.manifest_parse_valid);
    }
}
