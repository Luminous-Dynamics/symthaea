// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run the train-only B2/B3/B4 SECOM value and missingness baselines.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use symthaea_semiconductor::benchmark::{
    SecomRawExpectationsV1, SecomTabularBaselinesReportV1, SplitManifestV1,
    evaluate_secom_tabular_baselines_v1,
};

fn run() -> Result<(), String> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 4 {
        return Err(
            "usage: semi-secom-tabular-baselines <secom.data> <secom_labels.data> <expectations.json> <split-manifest.json>"
                .to_string(),
        );
    }
    let feature_path = PathBuf::from(&arguments[0]);
    let label_path = PathBuf::from(&arguments[1]);
    let expectation_path = PathBuf::from(&arguments[2]);
    let split_path = PathBuf::from(&arguments[3]);

    let feature_bytes = fs::read(&feature_path)
        .map_err(|error| format!("cannot read {}: {error}", feature_path.display()))?;
    let label_bytes = fs::read(&label_path)
        .map_err(|error| format!("cannot read {}: {error}", label_path.display()))?;
    let expected_text = fs::read_to_string(&expectation_path)
        .map_err(|error| format!("cannot read {}: {error}", expectation_path.display()))?;
    let split_text = fs::read_to_string(&split_path)
        .map_err(|error| format!("cannot read {}: {error}", split_path.display()))?;

    let expectations: SecomRawExpectationsV1 = serde_json::from_str(&expected_text)
        .map_err(|error| format!("invalid expectations JSON: {error}"))?;
    let split_manifest: SplitManifestV1 = serde_json::from_str(&split_text)
        .map_err(|error| format!("invalid split manifest JSON: {error}"))?;
    let report: SecomTabularBaselinesReportV1 = evaluate_secom_tabular_baselines_v1(
        &feature_bytes,
        &label_bytes,
        &expectations,
        &split_manifest,
    )
    .map_err(|error| format!("SECOM B2/B3/B4 evaluation rejected input: {error}"))?;
    let rendered = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("cannot render tabular baseline report: {error}"))?;
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
