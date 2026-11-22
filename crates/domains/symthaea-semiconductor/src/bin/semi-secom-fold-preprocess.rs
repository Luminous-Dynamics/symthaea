// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rebuild train-only preprocessing reports for every frozen stratified SECOM repeat.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use serde::Serialize;
use symthaea_semiconductor::benchmark::secom_raw::{
    SecomRawExpectationsV1, audit_secom_raw_v1, parse_features, parse_labels,
};
use symthaea_semiconductor::benchmark::split_manifest::{
    SplitManifestV1, validate_split_manifest_v1,
};
use symthaea_semiconductor::benchmark::stratified_repeat::StratifiedRepeatReportV1;
use symthaea_semiconductor::benchmark::{
    SecomFoldPreprocessorReportV1, SecomPreprocessorConfigV1, prepare_secom_stratified_fold_v1,
};

#[derive(Serialize)]
struct FiveRepeatPreprocessingReportV1 {
    schema: String,
    dataset_id: String,
    feature_file_sha256: String,
    label_file_sha256: String,
    dataset_snapshot_sha256: String,
    source_identity_manifest_sha256: String,
    preprocessing_config: SecomPreprocessorConfigV1,
    repeat_count: usize,
    repeats: Vec<SecomFoldPreprocessorReportV1>,
    claim_limits: Vec<String>,
}

fn run() -> Result<(), String> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 5 {
        return Err(
            "usage: semi-secom-fold-preprocess <secom.data> <secom_labels.data> <expectations.json> <stratified-repeat-report.json> <source-identity-manifest.json>"
                .into(),
        );
    }

    let feature_path = PathBuf::from(&arguments[0]);
    let label_path = PathBuf::from(&arguments[1]);
    let expectations_path = PathBuf::from(&arguments[2]);
    let repeats_path = PathBuf::from(&arguments[3]);
    let identity_manifest_path = PathBuf::from(&arguments[4]);

    let feature_bytes = fs::read(&feature_path)
        .map_err(|error| format!("cannot read {}: {error}", feature_path.display()))?;
    let label_bytes = fs::read(&label_path)
        .map_err(|error| format!("cannot read {}: {error}", label_path.display()))?;
    let expectations_text = fs::read_to_string(&expectations_path)
        .map_err(|error| format!("cannot read {}: {error}", expectations_path.display()))?;
    let repeats_text = fs::read_to_string(&repeats_path)
        .map_err(|error| format!("cannot read {}: {error}", repeats_path.display()))?;
    let identity_manifest_text = fs::read_to_string(&identity_manifest_path)
        .map_err(|error| format!("cannot read {}: {error}", identity_manifest_path.display()))?;

    let expectations: SecomRawExpectationsV1 = serde_json::from_str(&expectations_text)
        .map_err(|error| format!("invalid expectations JSON: {error}"))?;
    let repeats: StratifiedRepeatReportV1 = serde_json::from_str(&repeats_text)
        .map_err(|error| format!("invalid repeated split report JSON: {error}"))?;
    let identity_manifest: SplitManifestV1 = serde_json::from_str(&identity_manifest_text)
        .map_err(|error| format!("invalid source identity manifest JSON: {error}"))?;

    let raw_report = audit_secom_raw_v1(&feature_bytes, &label_bytes, &expectations)
        .map_err(|error| format!("raw SECOM audit rejected input: {error}"))?;
    if repeats.dataset_id != expectations.dataset_id
        || identity_manifest.dataset_id != expectations.dataset_id
        || repeats.dataset_snapshot_sha256 != identity_manifest.dataset_snapshot_sha256
    {
        return Err("dataset IDs or snapshot digests disagree across inputs".into());
    }
    let identity_report = validate_split_manifest_v1(&identity_manifest)
        .map_err(|error| format!("source identity manifest rejected input: {error}"))?;
    if repeats.source_identity_manifest_sha256 != identity_report.split_manifest_sha256 {
        return Err(
            "stratified repeat source identity digest does not match supplied manifest".into(),
        );
    }

    let features = parse_features(&feature_bytes)
        .map_err(|error| format!("SECOM feature parsing failed: {error}"))?;
    let labels = parse_labels(&label_bytes)
        .map_err(|error| format!("SECOM label parsing failed: {error}"))?;
    if features.len() != raw_report.rows
        || labels.len() != raw_report.rows
        || identity_manifest.assignments.len() != raw_report.rows
        || repeats.total_samples != raw_report.rows
    {
        return Err("raw feature/label/snapshot/repeat row support mismatch".into());
    }

    let mut sample_ids = Vec::with_capacity(raw_report.rows);
    for (index, (assignment, label)) in identity_manifest
        .assignments
        .iter()
        .zip(&labels)
        .enumerate()
    {
        if assignment.positive != (label.label == 1) {
            return Err(format!("source identity assignment {index} class mismatch"));
        }
        let source_timestamp = label.observed_at.format("%Y-%m-%dT%H:%M:%S").to_string();
        if assignment.observed_at.as_deref() != Some(source_timestamp.as_str()) {
            return Err(format!(
                "source identity assignment {index} timestamp mismatch"
            ));
        }
        sample_ids.push(assignment.sample_id.clone());
    }

    let config = SecomPreprocessorConfigV1::default();
    let mut reports = Vec::with_capacity(repeats.manifests.len());
    for manifest in &repeats.manifests {
        let prepared =
            prepare_secom_stratified_fold_v1(&sample_ids, &features, &labels, manifest, &config)
                .map_err(|error| {
                    format!(
                        "repeat {} train-only preprocessing rejected input: {error}",
                        manifest.repeat_index
                    )
                })?;
        reports.push(prepared.report);
    }

    if reports.len() != 5 {
        return Err(format!(
            "expected five preprocessing repeats, observed {}",
            reports.len()
        ));
    }
    let report = FiveRepeatPreprocessingReportV1 {
        schema: "luminous.semi-bench.secom-five-repeat-preprocessing-report.v1".into(),
        dataset_id: expectations.dataset_id,
        feature_file_sha256: raw_report.feature_sha256,
        label_file_sha256: raw_report.label_sha256,
        dataset_snapshot_sha256: identity_manifest.dataset_snapshot_sha256,
        source_identity_manifest_sha256: identity_report.split_manifest_sha256,
        preprocessing_config: config,
        repeat_count: reports.len(),
        repeats: reports,
        claim_limits: vec![
            "Filtering, medians, variance and quantile references are fitted only on each repeat's train partition".into(),
            "Quantile output and near-zero-variance thresholds are Luminous reconstruction settings; exact paper defaults are not disclosed".into(),
            "Feature Set B is an optional Luminous values-plus-mask ablation, not claimed as the cited paper's exact pipeline".into(),
            "SECOM columns are anonymous source columns, not physical sensor identities".into(),
            "Stratified repeats establish neither temporal nor independent-lot/fab generalization".into(),
        ],
    };

    let canonical = serde_json::to_value(&report)
        .map_err(|error| format!("cannot canonicalize preprocessing report: {error}"))?;
    let rendered = serde_json::to_string_pretty(&canonical)
        .map_err(|error| format!("cannot render preprocessing report: {error}"))?;
    println!("{rendered}");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}
