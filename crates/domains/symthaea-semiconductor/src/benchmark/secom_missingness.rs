// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Missingness-only shortcut baseline for the SECOM observation-regime audit.
//!
//! The model sees only the Option::None mask. It never reads measured sensor
//! values, and it is fitted only on rows assigned to the training partition.
//! This is a diagnostic baseline, not a process-quality or defect oracle.

use crate::benchmark::binary_metrics::{
    BinaryMetricReportV1, BinaryScoreSampleV1, BinaryScoreSubjectV1,
    INPUT_SCHEMA_V1 as BINARY_INPUT_SCHEMA_V1, score_binary_subject_v1,
};
use crate::benchmark::secom_raw::{
    SecomLabelRowV1, SecomRawDataError, SecomRawExpectationsV1, SecomSensorRowV1,
    audit_secom_raw_v1, parse_features, parse_labels,
};
use crate::benchmark::split_manifest::{SplitManifestV1, validate_split_manifest_v1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

pub const OUTPUT_SCHEMA_V1: &str = "luminous.semi-bench.secom-missingness-report.v1";
const SMOOTHING_ALPHA: f64 = 1.0;
const HELD_OUT_SPLITS: [&str; 2] = ["validation", "test"];

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct SecomMissingnessError(pub String);

impl From<SecomRawDataError> for SecomMissingnessError {
    fn from(error: SecomRawDataError) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomClassMissingnessProfileV1 {
    pub rows: usize,
    pub missing_cells: usize,
    pub sensor_cells: usize,
    pub missing_fraction: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomObservationProfileV1 {
    pub rows: usize,
    pub failures: usize,
    pub passes: usize,
    pub missing_cells: usize,
    pub sensor_cells: usize,
    pub missing_fraction: f64,
    pub mean_missing_features_per_row: f64,
    pub missing_fraction_by_feature: Vec<f64>,
    pub by_class: BTreeMap<String, SecomClassMissingnessProfileV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomMissingnessSampleScoreV1 {
    pub sample_id: String,
    pub positive: bool,
    pub missing_features: usize,
    pub missing_fraction: f64,
    pub prevalence_only_probability: f64,
    pub missingness_only_probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomMissingnessHeldOutV1 {
    pub observation_profile: SecomObservationProfileV1,
    pub prevalence_only_metrics: BinaryMetricReportV1,
    pub missingness_only_metrics: BinaryMetricReportV1,
    pub sample_scores: Vec<SecomMissingnessSampleScoreV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomMissingnessClaimLimitsV1 {
    pub model_uses_missingness_mask_only: bool,
    pub model_fitted_on_training_partition_only: bool,
    pub held_out_labels_not_used_for_model_fitting: bool,
    pub does_not_establish_sensor_fault_or_process_causality: bool,
    pub does_not_establish_current_fab_transfer_or_yield_improvement: bool,
    pub source_mirror_official_uci_byte_identity_unverified: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomMissingnessReportV1 {
    pub schema: String,
    pub dataset_id: String,
    pub feature_file_sha256: String,
    pub label_file_sha256: String,
    pub dataset_snapshot_sha256: String,
    pub split_manifest_sha256: String,
    pub split_scheme: String,
    pub clock_domain_id: Option<String>,
    pub feature_count: usize,
    pub training_rows: usize,
    pub training_failures: usize,
    pub training_passes: usize,
    pub bernoulli_smoothing_alpha: f64,
    pub missingness_model_sha256: String,
    pub held_out: BTreeMap<String, SecomMissingnessHeldOutV1>,
    pub claim_limits: SecomMissingnessClaimLimitsV1,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BernoulliMissingnessModelV1 {
    feature_count: usize,
    training_rows: usize,
    training_failures: usize,
    training_passes: usize,
    smoothing_alpha: f64,
    prior_log_odds: f64,
    log_odds_missing_by_feature: Vec<f64>,
    log_odds_observed_by_feature: Vec<f64>,
}

impl BernoulliMissingnessModelV1 {
    fn predict_probability(&self, row: &SecomSensorRowV1) -> f64 {
        let mut log_odds = self.prior_log_odds;
        for (index, value) in row.values.iter().enumerate() {
            log_odds += if value.is_none() {
                self.log_odds_missing_by_feature[index]
            } else {
                self.log_odds_observed_by_feature[index]
            };
        }
        // Stable logistic transform: neither branch exponentiates a large
        // positive argument.
        if log_odds >= 0.0 {
            1.0 / (1.0 + (-log_odds).exp())
        } else {
            let exp = log_odds.exp();
            exp / (1.0 + exp)
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fit_missingness_model(
    features: &[SecomSensorRowV1],
    labels: &[SecomLabelRowV1],
    assignments: &[crate::benchmark::split_manifest::SplitAssignmentV1],
    smoothing_alpha: f64,
) -> Result<BernoulliMissingnessModelV1, SecomMissingnessError> {
    let feature_count = features
        .first()
        .map(|row| row.values.len())
        .ok_or_else(|| SecomMissingnessError("features: expected non-empty rows".into()))?;
    let mut training_rows = 0usize;
    let mut training_failures = 0usize;
    let mut training_passes = 0usize;
    let mut missing_when_fail = vec![0usize; feature_count];
    let mut missing_when_pass = vec![0usize; feature_count];

    for index in 0..features.len() {
        if assignments[index].split != "train" {
            continue;
        }
        training_rows += 1;
        let missing_counts = if labels[index].label == 1 {
            training_failures += 1;
            &mut missing_when_fail
        } else {
            training_passes += 1;
            &mut missing_when_pass
        };
        for (feature_index, value) in features[index].values.iter().enumerate() {
            if value.is_none() {
                missing_counts[feature_index] += 1;
            }
        }
    }

    if training_failures == 0 || training_passes == 0 {
        return Err(SecomMissingnessError(
            "training split must contain both fail and pass samples".into(),
        ));
    }

    let alpha = smoothing_alpha;
    let mut log_odds_missing_by_feature = Vec::with_capacity(feature_count);
    let mut log_odds_observed_by_feature = Vec::with_capacity(feature_count);
    for feature_index in 0..feature_count {
        let fail_missing = (missing_when_fail[feature_index] as f64 + alpha)
            / (training_failures as f64 + 2.0 * alpha);
        let pass_missing = (missing_when_pass[feature_index] as f64 + alpha)
            / (training_passes as f64 + 2.0 * alpha);
        log_odds_missing_by_feature.push((fail_missing / pass_missing).ln());
        log_odds_observed_by_feature.push(((1.0 - fail_missing) / (1.0 - pass_missing)).ln());
    }

    Ok(BernoulliMissingnessModelV1 {
        feature_count,
        training_rows,
        training_failures,
        training_passes,
        smoothing_alpha: alpha,
        prior_log_odds: (training_failures as f64 / training_passes as f64).ln(),
        log_odds_missing_by_feature,
        log_odds_observed_by_feature,
    })
}

fn observation_profile(
    indices: &[usize],
    features: &[SecomSensorRowV1],
    labels: &[SecomLabelRowV1],
) -> SecomObservationProfileV1 {
    let feature_count = features[indices[0]].values.len();
    let mut missing_by_feature = vec![0usize; feature_count];
    let mut by_class_acc: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut failures = 0usize;
    let mut passes = 0usize;
    let mut missing_cells = 0usize;

    for &index in indices {
        let class_key = if labels[index].label == 1 {
            failures += 1;
            "fail"
        } else {
            passes += 1;
            "pass"
        };
        let class = by_class_acc.entry(class_key.to_string()).or_default();
        class.0 += 1;
        let row_missing = features[index]
            .values
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_none())
            .map(|(feature_index, _)| {
                missing_by_feature[feature_index] += 1;
                1usize
            })
            .sum::<usize>();
        missing_cells += row_missing;
        class.1 += row_missing;
    }

    let rows = indices.len();
    let sensor_cells = rows * feature_count;
    let by_class = by_class_acc
        .into_iter()
        .map(|(class, (class_rows, class_missing))| {
            (
                class,
                SecomClassMissingnessProfileV1 {
                    rows: class_rows,
                    missing_cells: class_missing,
                    sensor_cells: class_rows * feature_count,
                    missing_fraction: class_missing as f64 / (class_rows * feature_count) as f64,
                },
            )
        })
        .collect();

    SecomObservationProfileV1 {
        rows,
        failures,
        passes,
        missing_cells,
        sensor_cells,
        missing_fraction: missing_cells as f64 / sensor_cells as f64,
        mean_missing_features_per_row: missing_cells as f64 / rows as f64,
        missing_fraction_by_feature: missing_by_feature
            .into_iter()
            .map(|count| count as f64 / rows as f64)
            .collect(),
        by_class,
    }
}

fn binary_subject(
    case_id: String,
    indices: &[usize],
    assignments: &[crate::benchmark::split_manifest::SplitAssignmentV1],
    labels: &[SecomLabelRowV1],
    scores: &[f64],
) -> BinaryScoreSubjectV1 {
    BinaryScoreSubjectV1 {
        schema: BINARY_INPUT_SCHEMA_V1.to_string(),
        case_id: Some(case_id),
        decision_threshold: 0.5,
        calibration_bins: 10,
        recall_target: 0.8,
        positive_class_semantics: Some(
            "SECOM in-house fail label (+1); observation-only diagnostic".into(),
        ),
        samples: indices
            .iter()
            .enumerate()
            .map(|(position, &index)| BinaryScoreSampleV1 {
                sample_id: assignments[index].sample_id.clone(),
                positive: labels[index].label == 1,
                score: scores[position],
            })
            .collect(),
    }
}

/// Computes a train-only Bernoulli missingness baseline against a prevalence
/// control on validation and test. The digest-bound raw audit is performed
/// before fitting, and row labels/timestamps must align with the split rows.
pub fn evaluate_secom_missingness_v1(
    feature_bytes: &[u8],
    label_bytes: &[u8],
    expectations: &SecomRawExpectationsV1,
    split_manifest: &SplitManifestV1,
) -> Result<SecomMissingnessReportV1, SecomMissingnessError> {
    let raw_report = audit_secom_raw_v1(feature_bytes, label_bytes, expectations)?;
    if split_manifest.dataset_id != expectations.dataset_id {
        return Err(SecomMissingnessError(
            "split_manifest.dataset_id does not match raw expectations".into(),
        ));
    }
    let split_report = validate_split_manifest_v1(split_manifest)
        .map_err(|error| SecomMissingnessError(error.to_string()))?;
    if split_manifest.assignments.len() != raw_report.rows {
        return Err(SecomMissingnessError(format!(
            "split assignments: expected {} rows, observed {}",
            raw_report.rows,
            split_manifest.assignments.len()
        )));
    }
    if split_manifest.split_scheme == "forward_time"
        && split_manifest.clock_domain_id.as_deref() != Some(expectations.clock_domain_id.as_str())
    {
        return Err(SecomMissingnessError(
            "forward-time split clock_domain_id does not match raw expectations".into(),
        ));
    }

    let features = parse_features(feature_bytes)?;
    let labels = parse_labels(label_bytes)?;
    for (index, (assignment, label)) in split_manifest.assignments.iter().zip(&labels).enumerate() {
        if assignment.positive != (label.label == 1) {
            return Err(SecomMissingnessError(format!(
                "split assignment {index} label does not match source label row"
            )));
        }
        if let Some(observed_at) = assignment.observed_at.as_deref() {
            let source_time = label.observed_at.format("%Y-%m-%dT%H:%M:%S").to_string();
            if observed_at != source_time {
                return Err(SecomMissingnessError(format!(
                    "split assignment {index} observed_at does not match source timestamp"
                )));
            }
        }
    }

    let train_indices: Vec<usize> = split_manifest
        .assignments
        .iter()
        .enumerate()
        .filter_map(|(index, assignment)| (assignment.split == "train").then_some(index))
        .collect();
    let training_rows = train_indices.len();
    let model = fit_missingness_model(
        &features,
        &labels,
        &split_manifest.assignments,
        SMOOTHING_ALPHA,
    )?;
    let model_bytes = serde_json::to_vec(&model)
        .map_err(|error| SecomMissingnessError(format!("model serialization failed: {error}")))?;
    let missingness_model_sha256 = sha256_hex(&model_bytes);
    let prevalence = model.training_failures as f64 / model.training_rows as f64;

    let mut held_out = BTreeMap::new();
    for split in HELD_OUT_SPLITS {
        let indices: Vec<usize> = split_manifest
            .assignments
            .iter()
            .enumerate()
            .filter_map(|(index, assignment)| (assignment.split == split).then_some(index))
            .collect();
        if indices.is_empty() {
            return Err(SecomMissingnessError(format!(
                "{split}: expected held-out samples"
            )));
        }

        let prevalence_scores = vec![prevalence; indices.len()];
        let missingness_scores: Vec<f64> = indices
            .iter()
            .map(|&index| model.predict_probability(&features[index]))
            .collect();
        let prevalence_subject = binary_subject(
            format!("secom-b0-prevalence-only-{split}"),
            &indices,
            &split_manifest.assignments,
            &labels,
            &prevalence_scores,
        );
        let missingness_subject = binary_subject(
            format!("secom-b1-missingness-only-{split}"),
            &indices,
            &split_manifest.assignments,
            &labels,
            &missingness_scores,
        );
        let prevalence_only_metrics = score_binary_subject_v1(&prevalence_subject)
            .map_err(|error| SecomMissingnessError(error.to_string()))?;
        let missingness_only_metrics = score_binary_subject_v1(&missingness_subject)
            .map_err(|error| SecomMissingnessError(error.to_string()))?;
        let observation_profile = observation_profile(&indices, &features, &labels);

        let sample_scores = indices
            .iter()
            .enumerate()
            .map(|(position, &index)| {
                let missing_features = features[index]
                    .values
                    .iter()
                    .filter(|value| value.is_none())
                    .count();
                SecomMissingnessSampleScoreV1 {
                    sample_id: split_manifest.assignments[index].sample_id.clone(),
                    positive: labels[index].label == 1,
                    missing_features,
                    missing_fraction: missing_features as f64 / model.feature_count as f64,
                    prevalence_only_probability: prevalence_scores[position],
                    missingness_only_probability: missingness_scores[position],
                }
            })
            .collect();

        held_out.insert(
            split.to_string(),
            SecomMissingnessHeldOutV1 {
                observation_profile,
                prevalence_only_metrics,
                missingness_only_metrics,
                sample_scores,
            },
        );
    }

    Ok(SecomMissingnessReportV1 {
        schema: OUTPUT_SCHEMA_V1.to_string(),
        dataset_id: expectations.dataset_id.clone(),
        feature_file_sha256: raw_report.feature_sha256,
        label_file_sha256: raw_report.label_sha256,
        dataset_snapshot_sha256: split_manifest.dataset_snapshot_sha256.clone(),
        split_manifest_sha256: split_report.split_manifest_sha256,
        split_scheme: split_manifest.split_scheme.clone(),
        clock_domain_id: split_manifest.clock_domain_id.clone(),
        feature_count: model.feature_count,
        training_rows,
        training_failures: model.training_failures,
        training_passes: model.training_passes,
        bernoulli_smoothing_alpha: model.smoothing_alpha,
        missingness_model_sha256,
        held_out,
        claim_limits: SecomMissingnessClaimLimitsV1 {
            model_uses_missingness_mask_only: true,
            model_fitted_on_training_partition_only: true,
            held_out_labels_not_used_for_model_fitting: true,
            does_not_establish_sensor_fault_or_process_causality: true,
            does_not_establish_current_fab_transfer_or_yield_improvement: true,
            source_mirror_official_uci_byte_identity_unverified: true,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::split_manifest::{
        INPUT_SCHEMA_V1 as SPLIT_INPUT_SCHEMA_V1, SplitAssignmentV1,
    };

    const FEATURES: &[u8] =
        b"1.0 2.0\n1.1 2.1\nNaN NaN\nNaN NaN\n1.2 2.2\nNaN NaN\n1.3 2.3\nNaN NaN\n";
    const LABELS: &[u8] = b"-1 \"19/07/2008 11:55:00\"\n-1 \"19/07/2008 12:00:00\"\n1 \"19/07/2008 12:05:00\"\n1 \"19/07/2008 12:10:00\"\n-1 \"19/07/2008 12:15:00\"\n1 \"19/07/2008 12:20:00\"\n-1 \"19/07/2008 12:25:00\"\n1 \"19/07/2008 12:30:00\"\n";

    fn expectations(label_bytes: &[u8], feature_bytes: &[u8]) -> SecomRawExpectationsV1 {
        SecomRawExpectationsV1 {
            schema: crate::benchmark::secom_raw::EXPECTATIONS_SCHEMA_V1.into(),
            dataset_id: "synthetic-secom-missingness-v1".into(),
            expected_feature_sha256: sha256_hex(feature_bytes),
            expected_label_sha256: sha256_hex(label_bytes),
            expected_rows: 8,
            expected_features_per_row: 2,
            clock_domain_id: "synthetic-secom-clock".into(),
        }
    }

    fn manifest(labels: &[i8]) -> SplitManifestV1 {
        let splits = [
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "test",
            "test",
        ];
        let assignments = labels
            .iter()
            .enumerate()
            .map(|(index, label)| SplitAssignmentV1 {
                sample_id: format!("synthetic-row-{index:04}"),
                split: splits[index].into(),
                positive: *label == 1,
                group_id: None,
                observed_at: None,
            })
            .collect();
        SplitManifestV1 {
            schema: SPLIT_INPUT_SCHEMA_V1.into(),
            dataset_id: "synthetic-secom-missingness-v1".into(),
            dataset_snapshot_sha256: "a".repeat(64),
            split_scheme: "stratified_sample".into(),
            time_basis: None,
            clock_domain_id: None,
            assignments,
        }
    }

    fn run(feature_bytes: &[u8], label_bytes: &[u8], labels: &[i8]) -> SecomMissingnessReportV1 {
        evaluate_secom_missingness_v1(
            feature_bytes,
            label_bytes,
            &expectations(label_bytes, feature_bytes),
            &manifest(labels),
        )
        .expect("synthetic matched dataset must evaluate")
    }

    #[test]
    fn detects_a_missingness_shortcut_against_prevalence_control() {
        let report = run(FEATURES, LABELS, &[-1, -1, 1, 1, -1, 1, -1, 1]);
        let validation = &report.held_out["validation"];
        let test = &report.held_out["test"];
        assert_eq!(report.training_rows, 4);
        assert_eq!(report.training_failures, 2);
        assert_eq!(report.training_passes, 2);
        assert_eq!(
            validation.prevalence_only_metrics.ranking_metrics.auroc,
            0.5
        );
        assert_eq!(test.prevalence_only_metrics.ranking_metrics.auroc, 0.5);
        assert_eq!(
            validation.missingness_only_metrics.ranking_metrics.auroc,
            1.0
        );
        assert_eq!(test.missingness_only_metrics.ranking_metrics.auroc, 1.0);
        assert_eq!(test.observation_profile.rows, 2);
        assert_eq!(test.observation_profile.missing_fraction, 0.5);
        assert!(report.claim_limits.model_uses_missingness_mask_only);
    }

    #[test]
    fn held_out_label_changes_cannot_change_the_fitted_model_or_scores() {
        let original = run(FEATURES, LABELS, &[-1, -1, 1, 1, -1, 1, -1, 1]);
        let changed_labels: &[u8] = b"-1 \"19/07/2008 11:55:00\"\n-1 \"19/07/2008 12:00:00\"\n1 \"19/07/2008 12:05:00\"\n1 \"19/07/2008 12:10:00\"\n1 \"19/07/2008 12:15:00\"\n-1 \"19/07/2008 12:20:00\"\n1 \"19/07/2008 12:25:00\"\n-1 \"19/07/2008 12:30:00\"\n";
        let changed = run(FEATURES, changed_labels, &[-1, -1, 1, 1, 1, -1, 1, -1]);
        assert_eq!(
            original.missingness_model_sha256,
            changed.missingness_model_sha256
        );
        let original_scores = &original.held_out["test"].sample_scores;
        let changed_scores = &changed.held_out["test"].sample_scores;
        for (left, right) in original_scores.iter().zip(changed_scores) {
            assert_eq!(left.sample_id, right.sample_id);
            assert_eq!(
                left.missingness_only_probability,
                right.missingness_only_probability
            );
        }
    }

    #[test]
    fn measured_value_changes_cannot_change_missingness_model_or_scores() {
        let changed_features: &[u8] =
            b"100.0 -20.0\n101.1 -21.0\nNaN NaN\nNaN NaN\n102.2 -22.0\nNaN NaN\n103.3 -23.0\nNaN NaN\n";
        let original = run(FEATURES, LABELS, &[-1, -1, 1, 1, -1, 1, -1, 1]);
        let changed = run(changed_features, LABELS, &[-1, -1, 1, 1, -1, 1, -1, 1]);
        assert_eq!(
            original.missingness_model_sha256,
            changed.missingness_model_sha256
        );
        assert_eq!(
            original.held_out["test"].sample_scores,
            changed.held_out["test"].sample_scores
        );
    }

    #[test]
    fn rejects_a_split_manifest_that_misaligns_source_labels() {
        let mut subject = manifest(&[-1, -1, 1, 1, -1, 1, -1, 1]);
        subject.assignments[0].positive = true;
        let error = evaluate_secom_missingness_v1(
            FEATURES,
            LABELS,
            &expectations(LABELS, FEATURES),
            &subject,
        )
        .unwrap_err();
        assert!(error.to_string().contains("label does not match"));
    }
}
