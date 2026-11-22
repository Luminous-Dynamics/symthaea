// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generate repeated stratified 80/20 SECOM manifests after raw source audit.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use symthaea_semiconductor::benchmark::secom_raw::{
    SecomRawExpectationsV1, audit_secom_raw_v1, parse_labels,
};
use symthaea_semiconductor::benchmark::split_manifest::{
    SplitManifestV1, validate_split_manifest_v1,
};

use symthaea_semiconductor::benchmark::{
    StratifiedRepeatReportV1, StratifiedSampleV1, StratifiedSplitSubjectV1,
    generate_repeated_stratified_splits_v1,
};

fn run() -> Result<(), String> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 4 {
        return Err(
            "usage: semi-secom-stratified-splits <secom.data> <secom_labels.data> <expectations.json> <dataset-snapshot-manifest.json>"
                .into(),
        );
    }

    let feature_path = PathBuf::from(&arguments[0]);
    let label_path = PathBuf::from(&arguments[1]);
    let expectation_path = PathBuf::from(&arguments[2]);
    let snapshot_path = PathBuf::from(&arguments[3]);

    let feature_bytes = fs::read(&feature_path)
        .map_err(|error| format!("cannot read {}: {error}", feature_path.display()))?;
    let label_bytes = fs::read(&label_path)
        .map_err(|error| format!("cannot read {}: {error}", label_path.display()))?;
    let expectation_text = fs::read_to_string(&expectation_path)
        .map_err(|error| format!("cannot read {}: {error}", expectation_path.display()))?;
    let snapshot_text = fs::read_to_string(&snapshot_path)
        .map_err(|error| format!("cannot read {}: {error}", snapshot_path.display()))?;

    let expectations: SecomRawExpectationsV1 = serde_json::from_str(&expectation_text)
        .map_err(|error| format!("invalid SECOM expectations JSON: {error}"))?;
    let snapshot_manifest: SplitManifestV1 = serde_json::from_str(&snapshot_text)
        .map_err(|error| format!("invalid dataset snapshot manifest JSON: {error}"))?;

    audit_secom_raw_v1(&feature_bytes, &label_bytes, &expectations)
        .map_err(|error| format!("raw SECOM audit rejected input: {error}"))?;
    if snapshot_manifest.dataset_id != expectations.dataset_id {
        return Err("dataset snapshot manifest dataset_id does not match raw expectations".into());
    }
    if snapshot_manifest.split_scheme != "forward_time"
        || snapshot_manifest.time_basis.as_deref() != Some("source_local_naive")
        || snapshot_manifest.clock_domain_id.as_deref()
            != Some(expectations.clock_domain_id.as_str())
    {
        return Err(
            "dataset snapshot manifest does not bind the expected source-local chronology".into(),
        );
    }
    let snapshot_report = validate_split_manifest_v1(&snapshot_manifest)
        .map_err(|error| format!("dataset snapshot manifest rejected input: {error}"))?;

    let labels = parse_labels(&label_bytes)
        .map_err(|error| format!("SECOM label parsing failed: {error}"))?;
    if snapshot_manifest.assignments.len() != labels.len() {
        return Err(format!(
            "snapshot manifest row support mismatch: expected {}, observed {}",
            labels.len(),
            snapshot_manifest.assignments.len()
        ));
    }
    let mut samples = Vec::with_capacity(labels.len());
    for (index, (assignment, label)) in snapshot_manifest
        .assignments
        .iter()
        .zip(&labels)
        .enumerate()
    {
        if assignment.positive != (label.label == 1) {
            return Err(format!(
                "snapshot manifest assignment {index} class does not match source label"
            ));
        }
        let source_timestamp = label.observed_at.format("%Y-%m-%dT%H:%M:%S").to_string();
        if assignment.observed_at.as_deref() != Some(source_timestamp.as_str()) {
            return Err(format!(
                "snapshot manifest assignment {index} timestamp does not match source label"
            ));
        }
        samples.push(StratifiedSampleV1 {
            sample_id: assignment.sample_id.clone(),
            positive: label.label == 1,
        });
    }

    let subject = StratifiedSplitSubjectV1 {
        schema: "luminous.semi-bench.stratified-repeat-subject.v1".into(),
        dataset_id: expectations.dataset_id,
        dataset_snapshot_sha256: snapshot_manifest.dataset_snapshot_sha256,
        source_identity_manifest_sha256: snapshot_report.split_manifest_sha256,
        test_fraction: 0.2,
        seeds: vec![42, 43, 44, 45, 46],
        seed_provenance:
            "Luminous protocol reconstruction; published article does not disclose its exact SECOM seeds"
                .into(),
        samples,
    };
    let report: StratifiedRepeatReportV1 = generate_repeated_stratified_splits_v1(&subject)
        .map_err(|error| format!("stratified split generation rejected input: {error}"))?;
    let canonical = serde_json::to_value(&report)
        .map_err(|error| format!("cannot canonicalize stratified split report: {error}"))?;
    let rendered = serde_json::to_string_pretty(&canonical)
        .map_err(|error| format!("cannot render stratified split report: {error}"))?;
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
