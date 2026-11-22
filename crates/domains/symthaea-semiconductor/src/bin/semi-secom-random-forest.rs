// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Train and evaluate deterministic Random Forest baselines on five frozen
//! stratified SECOM repeats. Physical sensor semantics are not inferred.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use serde::Serialize;
use sha2::Digest;
use symthaea_semiconductor::benchmark::random_forest::RandomForestModelV1;
use symthaea_semiconductor::benchmark::secom_fold_preprocessor::{
    SecomFoldPreprocessorReportV1, SecomPreprocessorConfigV1,
};
use symthaea_semiconductor::benchmark::secom_raw::{
    SecomRawExpectationsV1, audit_secom_raw_v1, parse_features, parse_labels,
};
use symthaea_semiconductor::benchmark::split_manifest::{
    SplitManifestV1, validate_split_manifest_v1,
};
use symthaea_semiconductor::benchmark::{
    ForestClassWeightingV1, RandomForestInputSampleV1, RandomForestParametersV1,
    RandomForestPredictionReportV1, RandomForestTrainingSampleV1, fit_random_forest_v1,
    predict_random_forest_v1, prepare_secom_stratified_fold_v1, score_random_forest_predictions_v1,
};
use symthaea_semiconductor::benchmark::stratified_repeat::StratifiedRepeatReportV1;

#[derive(Serialize)]
struct RandomForestArmReportV1 {
    model_id: String,
    parameters: RandomForestParametersV1,
    model_sha256: String,
    training_rows: usize,
    training_positive: usize,
    training_negative: usize,
    fit_milliseconds: u64,
    prediction_milliseconds: u64,
    prediction_report: RandomForestPredictionReportV1,
    metrics: symthaea_semiconductor::benchmark::binary_metrics::BinaryMetricReportV1,
}

#[derive(Serialize)]
struct RandomForestFoldReportV1 {
    repeat_index: usize,
    seed: u64,
    split_manifest_sha256: String,
    preprocessing: SecomFoldPreprocessorReportV1,
    arms: Vec<RandomForestArmReportV1>,
}

#[derive(Serialize)]
struct FiveRepeatRandomForestReportV1 {
    schema: String,
    dataset_id: String,
    feature_file_sha256: String,
    label_file_sha256: String,
    dataset_snapshot_sha256: String,
    source_identity_manifest_sha256: String,
    preprocessing_config: SecomPreprocessorConfigV1,
    repeat_count: usize,
    forest_protocol: String,
    repeats: Vec<RandomForestFoldReportV1>,
    claim_limits: Vec<String>,
}

fn run() -> Result<(), String> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 5 {
        return Err(
            "usage: semi-secom-random-forest <secom.data> <secom_labels.data> <expectations.json> <stratified-repeat-report.json> <source-identity-manifest.json>"
                .into(),
        );
    }
    let feature_path = PathBuf::from(&arguments[0]);
    let label_path = PathBuf::from(&arguments[1]);
    let expectations_path = PathBuf::from(&arguments[2]);
    let repeats_path = PathBuf::from(&arguments[3]);
    let identity_path = PathBuf::from(&arguments[4]);

    let feature_bytes = fs::read(&feature_path)
        .map_err(|error| format!("cannot read {}: {error}", feature_path.display()))?;
    let label_bytes = fs::read(&label_path)
        .map_err(|error| format!("cannot read {}: {error}", label_path.display()))?;
    let expectations_text = fs::read_to_string(&expectations_path)
        .map_err(|error| format!("cannot read {}: {error}", expectations_path.display()))?;
    let repeats_text = fs::read_to_string(&repeats_path)
        .map_err(|error| format!("cannot read {}: {error}", repeats_path.display()))?;
    let identity_text = fs::read_to_string(&identity_path)
        .map_err(|error| format!("cannot read {}: {error}", identity_path.display()))?;

    let expectations: SecomRawExpectationsV1 = serde_json::from_str(&expectations_text)
        .map_err(|error| format!("invalid expectations JSON: {error}"))?;
    let repeats: StratifiedRepeatReportV1 = serde_json::from_str(&repeats_text)
        .map_err(|error| format!("invalid stratified repeat report JSON: {error}"))?;
    let identity_manifest: SplitManifestV1 = serde_json::from_str(&identity_text)
        .map_err(|error| format!("invalid source identity manifest JSON: {error}"))?;

    let raw_report = audit_secom_raw_v1(&feature_bytes, &label_bytes, &expectations)
        .map_err(|error| format!("raw SECOM integrity audit rejected input: {error}"))?;
    if repeats.dataset_id != expectations.dataset_id
        || identity_manifest.dataset_id != expectations.dataset_id
        || repeats.dataset_snapshot_sha256 != identity_manifest.dataset_snapshot_sha256
    {
        return Err("dataset identifiers or snapshot digests disagree".into());
    }
    let identity_report = validate_split_manifest_v1(&identity_manifest)
        .map_err(|error| format!("source identity manifest rejected input: {error}"))?;
    if repeats.source_identity_manifest_sha256 != identity_report.split_manifest_sha256 {
        return Err(
            "stratified manifest does not bind the supplied source identity manifest".into(),
        );
    }
    if identity_manifest.split_scheme != "forward_time"
        || identity_manifest.time_basis.as_deref() != Some("source_local_naive")
        || identity_manifest.clock_domain_id.as_deref()
            != Some(expectations.clock_domain_id.as_str())
    {
        return Err(
            "source identity manifest does not bind expected source-local chronology".into(),
        );
    }

    let features = parse_features(&feature_bytes)
        .map_err(|error| format!("SECOM feature parse failed: {error}"))?;
    let labels =
        parse_labels(&label_bytes).map_err(|error| format!("SECOM label parse failed: {error}"))?;
    if features.len() != raw_report.rows
        || labels.len() != raw_report.rows
        || identity_manifest.assignments.len() != raw_report.rows
        || repeats.total_samples != raw_report.rows
    {
        return Err("raw feature, label, identity and repeat row support mismatch".into());
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
        let source_time = label.observed_at.format("%Y-%m-%dT%H:%M:%S").to_string();
        if assignment.observed_at.as_deref() != Some(source_time.as_str()) {
            return Err(format!(
                "source identity assignment {index} timestamp mismatch"
            ));
        }
        sample_ids.push(assignment.sample_id.clone());
    }

    let preprocessing_config = SecomPreprocessorConfigV1::default();
    let mut fold_reports = Vec::with_capacity(repeats.manifests.len());
    for manifest in &repeats.manifests {
        let prepared = prepare_secom_stratified_fold_v1(
            &sample_ids,
            &features,
            &labels,
            manifest,
            &preprocessing_config,
        )
        .map_err(|error| {
            format!(
                "repeat {} train-only preprocessing rejected input: {error}",
                manifest.repeat_index
            )
        })?;

        let training: Vec<RandomForestTrainingSampleV1> = prepared
            .training
            .iter()
            .map(|row| RandomForestTrainingSampleV1 {
                sample_id: row.sample_id.clone(),
                positive: row.positive,
                features: row.feature_set_a.clone(),
            })
            .collect();
        let test: Vec<RandomForestInputSampleV1> = prepared
            .test
            .iter()
            .map(|row| RandomForestInputSampleV1 {
                sample_id: row.sample_id.clone(),
                positive: row.positive,
                features: row.feature_set_a.clone(),
            })
            .collect();

        let mut arms = Vec::new();
        for (model_id, class_weighting) in [
            ("rf-200-unweighted", ForestClassWeightingV1::None),
            ("rf-200-balanced", ForestClassWeightingV1::Balanced),
        ] {
            let parameters = RandomForestParametersV1 {
                n_trees: 200,
                max_depth: 12,
                min_samples_leaf: 4,
                mtry: 0,
                seed: manifest.seed,
                class_weighting,
            };
            let fit_start = Instant::now();
            let model: RandomForestModelV1 =
                fit_random_forest_v1(&training, &parameters).map_err(|error| {
                    format!(
                        "repeat {} {model_id} fit failed: {error}",
                        manifest.repeat_index
                    )
                })?;
            let fit_milliseconds = fit_start.elapsed().as_millis() as u64;

            let prediction_start = Instant::now();
            let prediction_report = predict_random_forest_v1(&model, &test).map_err(|error| {
                format!(
                    "repeat {} {model_id} prediction failed: {error}",
                    manifest.repeat_index
                )
            })?;
            let prediction_milliseconds = prediction_start.elapsed().as_millis() as u64;
            let metrics = score_random_forest_predictions_v1(&prediction_report, 0.5, 10).map_err(
                |error| {
                    format!(
                        "repeat {} {model_id} metrics failed: {error}",
                        manifest.repeat_index
                    )
                },
            )?;

            arms.push(RandomForestArmReportV1 {
                model_id: model_id.into(),
                parameters,
                model_sha256: model.model_sha256.clone(),
                training_rows: model.training_rows,
                training_positive: model.training_positive,
                training_negative: model.training_negative,
                fit_milliseconds,
                prediction_milliseconds,
                prediction_report,
                metrics,
            });
        }

        fold_reports.push(RandomForestFoldReportV1 {
            repeat_index: manifest.repeat_index,
            seed: manifest.seed,
            split_manifest_sha256: serde_json::to_value(manifest)
                .ok()
                .and_then(|value| serde_json::to_vec(&value).ok())
                .map(|bytes| format!("{:x}", sha2::Sha256::digest(bytes)))
                .unwrap_or_default(),
            preprocessing: prepared.report,
            arms,
        });
    }

    if fold_reports.len() != 5 {
        return Err(format!(
            "expected five folds, observed {}",
            fold_reports.len()
        ));
    }

    let report = FiveRepeatRandomForestReportV1 {
        schema: "luminous.semi-bench.secom-five-repeat-random-forest-report.v1".into(),
        dataset_id: expectations.dataset_id,
        feature_file_sha256: raw_report.feature_sha256,
        label_file_sha256: raw_report.label_sha256,
        dataset_snapshot_sha256: identity_manifest.dataset_snapshot_sha256,
        source_identity_manifest_sha256: identity_report.split_manifest_sha256,
        preprocessing_config,
        repeat_count: fold_reports.len(),
        forest_protocol:
            "200 bootstrap CART trees; gini; mtry=floor(sqrt(p)) sampled at each node; max_depth=12; min_leaf=4; CPU; raw unweighted leaf positive fractions averaged across trees; balanced variant applies inverse-frequency weights only to split impurity"
                .into(),
        repeats: fold_reports,
        claim_limits: vec![
            "Stratified repeated splits measure interpolation within the historic public dataset, not chronological generalization".into(),
            "The five seeds are Luminous reconstruction seeds, not claimed to be authors original seeds".into(),
            "Tree leaf frequencies are scores, not calibrated defect probabilities".into(),
            "SECOM columns are anonymous source columns, not physical sensor identities".into(),
            "No tool, lot, fab, recipe, hold/release or APC action is authorized".into(),
        ],
    };

    let canonical = serde_json::to_value(&report)
        .map_err(|error| format!("cannot canonicalize Random Forest report: {error}"))?;
    let rendered = serde_json::to_string_pretty(&canonical)
        .map_err(|error| format!("cannot render Random Forest report: {error}"))?;
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
