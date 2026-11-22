// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Train-only discriminative logistic regression baselines for SECOM.
//!
//! B5 is unweighted logistic regression; B5-balanced uses inverse-class-frequency
//! loss weights and then restores the training class prior to emitted scores.
//! Both use FTRL-Proximal, train-only median imputation and train-only scaling.
//! No validation or test values/labels participate in fitting.

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

pub const OUTPUT_SCHEMA_V1: &str = "luminous.semi-bench.secom-logistic-report.v1";
const HELD_OUT_SPLITS: [&str; 2] = ["validation", "test"];
const FTRL_ALPHA: f64 = 0.05;
const FTRL_BETA: f64 = 1.0;
const FTRL_L1: f64 = 0.0;
const FTRL_L2: f64 = 1.0;
const FTRL_EPOCHS: usize = 80;

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct SecomLogisticError(pub String);

impl From<SecomRawDataError> for SecomLogisticError {
    fn from(error: SecomRawDataError) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomLogisticScoreV1 {
    pub sample_id: String,
    pub positive: bool,
    pub failure_probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomLogisticHeldOutV1 {
    pub rows: usize,
    pub failures: usize,
    pub passes: usize,
    pub metrics: BinaryMetricReportV1,
    pub scores: Vec<SecomLogisticScoreV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomLogisticModelConfigV1 {
    pub epochs: usize,
    pub ftrl_alpha: f64,
    pub ftrl_beta: f64,
    pub l1: f64,
    pub l2: f64,
    pub input_feature_count: usize,
    pub input_description: String,
    pub class_balanced_loss: bool,
    pub class_weight_positive: f64,
    pub class_weight_negative: f64,
    pub output_prior_correction_log_odds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomLogisticModelResultV1 {
    pub model_id: String,
    pub config: SecomLogisticModelConfigV1,
    pub missing_features_without_observed_training_values: usize,
    pub model_sha256: String,
    pub held_out: BTreeMap<String, SecomLogisticHeldOutV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomLogisticClaimLimitsV1 {
    pub imputation_normalization_and_model_fit_on_train_only: bool,
    pub held_out_labels_not_used_for_fit_or_prior: bool,
    pub held_out_features_not_used_for_fit: bool,
    pub training_rows_sorted_by_sample_identity: bool,
    pub does_not_establish_feature_semantics_or_process_causality: bool,
    pub does_not_establish_current_fab_transfer_or_yield_improvement: bool,
    pub source_mirror_official_uci_byte_identity_unverified: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomLogisticReportV1 {
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
    pub models: BTreeMap<String, SecomLogisticModelResultV1>,
    pub claim_limits: SecomLogisticClaimLimitsV1,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FtrlLogisticModelV1 {
    model_id: String,
    medians: Vec<f64>,
    means: Vec<f64>,
    scales: Vec<f64>,
    weights: Vec<f64>,
    intercept: f64,
    config: SecomLogisticModelConfigV1,
    missing_features_without_observed_training_values: usize,
}

impl FtrlLogisticModelV1 {
    fn transform(&self, row: &SecomSensorRowV1, sample_id: &str) -> Result<Vec<f64>, SecomLogisticError> {
        if row.values.len() != self.medians.len() {
            return Err(SecomLogisticError(format!(
                "sample {sample_id:?}: expected {} sensor features, observed {}",
                self.medians.len(),
                row.values.len()
            )));
        }
        let mut transformed = Vec::with_capacity(self.config.input_feature_count);
        for (index, value) in row.values.iter().enumerate() {
            let imputed = value.unwrap_or(self.medians[index]);
            let standardized = (imputed - self.means[index]) / self.scales[index];
            if !standardized.is_finite() {
                return Err(SecomLogisticError(format!(
                    "sample {sample_id:?}, feature {index}: non-finite standardized value"
                )));
            }
            transformed.push(standardized);
        }
        for (index, value) in row.values.iter().enumerate() {
            let missing = if value.is_none() { 1.0 } else { 0.0 };
            let standardized = (missing - self.means[self.medians.len() + index])
                / self.scales[self.medians.len() + index];
            if !standardized.is_finite() {
                return Err(SecomLogisticError(format!(
                    "sample {sample_id:?}, missingness feature {index}: non-finite standardized value"
                )));
            }
            transformed.push(standardized);
        }
        Ok(transformed)
    }

    fn logit(&self, row: &SecomSensorRowV1, sample_id: &str) -> Result<f64, SecomLogisticError> {
        let transformed = self.transform(row, sample_id)?;
        let logit = self.intercept
            + self
                .weights
                .iter()
                .zip(&transformed)
                .map(|(weight, value)| weight * value)
                .sum::<f64>()
            + self.config.output_prior_correction_log_odds;
        if !logit.is_finite() {
            return Err(SecomLogisticError(format!(
                "sample {sample_id:?}: non-finite logistic score"
            )));
        }
        Ok(logit)
    }

    fn predict_probability(
        &self,
        row: &SecomSensorRowV1,
        sample_id: &str,
    ) -> Result<f64, SecomLogisticError> {
        let logit = self.logit(row, sample_id)?;
        let probability = sigmoid(logit);
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(SecomLogisticError(format!(
                "sample {sample_id:?}: invalid failure probability {probability}"
            )));
        }
        Ok(probability)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FtrlStateV1 {
    z: Vec<f64>,
    n: Vec<f64>,
    intercept_z: f64,
    intercept_n: f64,
}

impl FtrlStateV1 {
    fn new(feature_count: usize) -> Self {
        Self {
            z: vec![0.0; feature_count],
            n: vec![0.0; feature_count],
            intercept_z: 0.0,
            intercept_n: 0.0,
        }
    }

    fn weight(&self, index: usize) -> f64 {
        ftrl_weight(
            self.z[index],
            self.n[index],
            FTRL_ALPHA,
            FTRL_BETA,
            FTRL_L1,
            FTRL_L2,
        )
    }

    fn intercept(&self) -> f64 {
        ftrl_weight(
            self.intercept_z,
            self.intercept_n,
            FTRL_ALPHA,
            FTRL_BETA,
            0.0,
            0.0,
        )
    }

    fn update(
        &mut self,
        values: &[f64],
        target: f64,
        sample_weight: f64,
    ) -> Result<(), SecomLogisticError> {
        let logit = self.intercept()
            + self
                .z
                .iter()
                .enumerate()
                .map(|(index, _)| self.weight(index) * values[index])
                .sum::<f64>();
        let probability = sigmoid(logit);
        let gradient = (probability - target) * sample_weight;
        if !gradient.is_finite() {
            return Err(SecomLogisticError(
                "FTRL update: non-finite sample gradient".into(),
            ));
        }

        for (index, value) in values.iter().enumerate() {
            let feature_gradient = gradient * value;
            let previous_n = self.n[index];
            let next_n = previous_n + feature_gradient * feature_gradient;
            let sigma = (next_n.sqrt() - previous_n.sqrt()) / FTRL_ALPHA;
            let current_weight = self.weight(index);
            self.z[index] += feature_gradient - sigma * current_weight;
            self.n[index] = next_n;
            if !self.z[index].is_finite() || !self.n[index].is_finite() {
                return Err(SecomLogisticError(format!(
                    "FTRL update: non-finite state at feature {index}"
                )));
            }
        }

        let previous_n = self.intercept_n;
        let next_n = previous_n + gradient * gradient;
        let sigma = (next_n.sqrt() - previous_n.sqrt()) / FTRL_ALPHA;
        let current_intercept = self.intercept();
        self.intercept_z += gradient - sigma * current_intercept;
        self.intercept_n = next_n;
        if !self.intercept_z.is_finite() || !self.intercept_n.is_finite() {
            return Err(SecomLogisticError(
                "FTRL update: non-finite intercept state".into(),
            ));
        }
        Ok(())
    }

    fn weights(&self) -> Vec<f64> {
        (0..self.z.len()).map(|index| self.weight(index)).collect()
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn ftrl_weight(z: f64, n: f64, alpha: f64, beta: f64, l1: f64, l2: f64) -> f64 {
    if z.abs() <= l1 {
        0.0
    } else {
        -(z - z.signum() * l1) / ((beta + n.sqrt()) / alpha + l2)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn fit_model_v1(
    model_id: &str,
    class_balanced: bool,
    features: &[SecomSensorRowV1],
    labels: &[SecomLabelRowV1],
    assignments: &[crate::benchmark::split_manifest::SplitAssignmentV1],
) -> Result<FtrlLogisticModelV1, SecomLogisticError> {
    if features.is_empty() || features.len() != labels.len() || features.len() != assignments.len() {
        return Err(SecomLogisticError(
            "features, labels and assignments must have equal non-zero row counts".into(),
        ));
    }
    let feature_count = features[0].values.len();
    if feature_count == 0 {
        return Err(SecomLogisticError(
            "features: expected at least one sensor feature".into(),
        ));
    }
    for (index, row) in features.iter().enumerate() {
        if row.values.len() != feature_count {
            return Err(SecomLogisticError(format!(
                "row {index}: inconsistent sensor feature count"
            )));
        }
    }

    let mut training_indices: Vec<usize> = assignments
        .iter()
        .enumerate()
        .filter_map(|(index, assignment)| (assignment.split == "train").then_some(index))
        .collect();
    training_indices.sort_by(|&left, &right| {
        assignments[left]
            .sample_id
            .cmp(&assignments[right].sample_id)
    });
    let mut class_counts = [0usize; 2];
    for &index in &training_indices {
        let class_index = if labels[index].label == 1 { 1 } else { 0 };
        class_counts[class_index] += 1;
    }
    let training_rows = training_indices.len();
    if training_rows == 0 || class_counts[0] == 0 || class_counts[1] == 0 {
        return Err(SecomLogisticError(
            "training split must be non-empty and contain both pass and fail".into(),
        ));
    }

    let mut observed_by_feature = vec![Vec::<f64>::new(); feature_count];
    for &row_index in &training_indices {
        for (feature_index, value) in features[row_index].values.iter().enumerate() {
            if let Some(observed) = value {
                observed_by_feature[feature_index].push(*observed);
            }
        }
    }
    let missing_features_without_observed_training_values = observed_by_feature
        .iter()
        .filter(|values| values.is_empty())
        .count();
    let medians: Vec<f64> = observed_by_feature
        .iter_mut()
        .map(|values| {
            if values.is_empty() {
                0.0
            } else {
                median(values)
            }
        })
        .collect();

    let input_feature_count = feature_count * 2;
    let mut means = vec![0.0; input_feature_count];
    let mut transformed_training = Vec::with_capacity(training_rows);
    for &row_index in &training_indices {
        let mut vector = Vec::with_capacity(input_feature_count);
        for (feature_index, value) in features[row_index].values.iter().enumerate() {
            vector.push(value.unwrap_or(medians[feature_index]));
        }
        for value in &features[row_index].values {
            vector.push(if value.is_none() { 1.0 } else { 0.0 });
        }
        for (index, value) in vector.iter().enumerate() {
            means[index] += *value;
        }
        transformed_training.push((row_index, vector));
    }
    for mean in &mut means {
        *mean /= training_rows as f64;
        if !mean.is_finite() {
            return Err(SecomLogisticError(
                "training feature mean is non-finite".into(),
            ));
        }
    }
    let mut scales = vec![0.0; input_feature_count];
    for (_, vector) in &transformed_training {
        for (index, value) in vector.iter().enumerate() {
            let delta = value - means[index];
            scales[index] += delta * delta;
        }
    }
    for (index, scale) in scales.iter_mut().enumerate() {
        let variance = *scale / training_rows as f64;
        if !variance.is_finite() {
            return Err(SecomLogisticError(format!(
                "feature {index}: non-finite training variance"
            )));
        }
        let standard_deviation = variance.max(0.0).sqrt();
        *scale = if standard_deviation > 1.0e-12 {
            standard_deviation
        } else {
            1.0
        };
    }
    for (_, vector) in &mut transformed_training {
        for (index, value) in vector.iter_mut().enumerate() {
            *value = (*value - means[index]) / scales[index];
        }
    }

    let class_weight_negative = if class_balanced {
        training_rows as f64 / (2.0 * class_counts[0] as f64)
    } else {
        1.0
    };
    let class_weight_positive = if class_balanced {
        training_rows as f64 / (2.0 * class_counts[1] as f64)
    } else {
        1.0
    };
    let prior_correction = if class_balanced {
        (class_weight_negative / class_weight_positive).ln()
    } else {
        0.0
    };

    let mut state = FtrlStateV1::new(input_feature_count);
    for _epoch in 0..FTRL_EPOCHS {
        for (row_index, vector) in &transformed_training {
            let positive = labels[*row_index].label == 1;
            let (target, sample_weight) = if positive {
                (1.0, class_weight_positive)
            } else {
                (0.0, class_weight_negative)
            };
            state.update(vector, target, sample_weight)?;
        }
    }

    let config = SecomLogisticModelConfigV1 {
        epochs: FTRL_EPOCHS,
        ftrl_alpha: FTRL_ALPHA,
        ftrl_beta: FTRL_BETA,
        l1: FTRL_L1,
        l2: FTRL_L2,
        input_feature_count,
        input_description: "train-standardized numeric values with train-median imputation followed by train-standardized missingness indicators".into(),
        class_balanced_loss: class_balanced,
        class_weight_positive,
        class_weight_negative,
        output_prior_correction_log_odds: prior_correction,
    };

    Ok(FtrlLogisticModelV1 {
        model_id: model_id.into(),
        medians,
        means,
        scales,
        weights: state.weights(),
        intercept: state.intercept(),
        config,
        missing_features_without_observed_training_values,
    })
}

fn held_out_report(
    model: &FtrlLogisticModelV1,
    split: &str,
    indices: &[usize],
    assignments: &[crate::benchmark::split_manifest::SplitAssignmentV1],
    labels: &[SecomLabelRowV1],
    features: &[SecomSensorRowV1],
) -> Result<SecomLogisticHeldOutV1, SecomLogisticError> {
    let samples: Vec<BinaryScoreSampleV1> = indices
        .iter()
        .map(|&index| {
            Ok(BinaryScoreSampleV1 {
                sample_id: assignments[index].sample_id.clone(),
                positive: labels[index].label == 1,
                score: model.predict_probability(&features[index], &assignments[index].sample_id)?,
            })
        })
        .collect::<Result<Vec<_>, SecomLogisticError>>()?;
    let failures = samples.iter().filter(|sample| sample.positive).count();
    let passes = samples.len() - failures;
    let subject = BinaryScoreSubjectV1 {
        schema: BINARY_INPUT_SCHEMA_V1.into(),
        case_id: Some(format!("{}-{split}", model.model_id)),
        decision_threshold: 0.5,
        calibration_bins: 10,
        recall_target: 0.8,
        positive_class_semantics: Some(
            "SECOM in-house fail label (+1); train-only FTRL logistic baseline".into(),
        ),
        samples,
    };
    let metrics = score_binary_subject_v1(&subject)
        .map_err(|error| SecomLogisticError(error.to_string()))?;
    let scores = subject.samples.iter().map(|sample| SecomLogisticScoreV1 {
        sample_id: sample.sample_id.clone(),
        positive: sample.positive,
        failure_probability: sample.score,
    }).collect();
    Ok(SecomLogisticHeldOutV1 {
        rows: indices.len(),
        failures,
        passes,
        metrics,
        scores,
    })
}

fn evaluate_model(
    model_id: &str,
    class_balanced: bool,
    features: &[SecomSensorRowV1],
    labels: &[SecomLabelRowV1],
    assignments: &[crate::benchmark::split_manifest::SplitAssignmentV1],
) -> Result<SecomLogisticModelResultV1, SecomLogisticError> {
    let model = fit_model_v1(model_id, class_balanced, features, labels, assignments)?;
    let model_bytes = serde_json::to_vec(&model)
        .map_err(|error| SecomLogisticError(format!("model serialization failed: {error}")))?;
    let model_sha256 = sha256_hex(&model_bytes);
    let mut held_out = BTreeMap::new();
    for split in HELD_OUT_SPLITS {
        let indices: Vec<usize> = assignments.iter().enumerate()
            .filter_map(|(index, assignment)| (assignment.split == split).then_some(index))
            .collect();
        if indices.is_empty() {
            return Err(SecomLogisticError(format!("{split}: expected non-empty held-out rows")));
        }
        held_out.insert(
            split.into(),
            held_out_report(&model, split, &indices, assignments, labels, features)?,
        );
    }
    Ok(SecomLogisticModelResultV1 {
        model_id: model_id.into(),
        config: model.config.clone(),
        missing_features_without_observed_training_values: model.missing_features_without_observed_training_values,
        model_sha256,
        held_out,
    })
}

/// Evaluate unweighted and class-balanced train-only FTRL logistic regression
/// against the validation/test partitions in a digest-bound split manifest.
pub fn evaluate_secom_logistic_v1(
    feature_bytes: &[u8],
    label_bytes: &[u8],
    expectations: &SecomRawExpectationsV1,
    split_manifest: &SplitManifestV1,
) -> Result<SecomLogisticReportV1, SecomLogisticError> {
    let raw_report = audit_secom_raw_v1(feature_bytes, label_bytes, expectations)?;
    if split_manifest.dataset_id != expectations.dataset_id {
        return Err(SecomLogisticError(
            "split_manifest.dataset_id does not match raw expectations".into(),
        ));
    }
    let split_report = validate_split_manifest_v1(split_manifest)
        .map_err(|error| SecomLogisticError(error.to_string()))?;
    if split_manifest.assignments.len() != raw_report.rows {
        return Err(SecomLogisticError(format!(
            "split assignments: expected {} rows, observed {}",
            raw_report.rows,
            split_manifest.assignments.len()
        )));
    }
    if split_manifest.split_scheme == "forward_time"
        && split_manifest.clock_domain_id.as_deref() != Some(expectations.clock_domain_id.as_str())
    {
        return Err(SecomLogisticError(
            "forward-time split clock_domain_id does not match raw expectations".into(),
        ));
    }
    let features = parse_features(feature_bytes)?;
    let labels = parse_labels(label_bytes)?;
    for (index, (assignment, label)) in split_manifest.assignments.iter().zip(&labels).enumerate() {
        if assignment.positive != (label.label == 1) {
            return Err(SecomLogisticError(format!(
                "split assignment {index} label does not match source label row"
            )));
        }
        if let Some(observed_at) = assignment.observed_at.as_deref() {
            let source_time = label.observed_at.format("%Y-%m-%dT%H:%M:%S").to_string();
            if observed_at != source_time {
                return Err(SecomLogisticError(format!(
                    "split assignment {index} observed_at does not match source timestamp"
                )));
            }
        }
    }
    let training_rows = split_manifest
        .assignments
        .iter()
        .filter(|assignment| assignment.split == "train")
        .count();
    let training_failures = split_manifest
        .assignments
        .iter()
        .filter(|assignment| assignment.split == "train" && assignment.positive)
        .count();
    let training_passes = training_rows - training_failures;
    let mut models = BTreeMap::new();
    models.insert(
        "B5".into(),
        evaluate_model("B5-ftrl-logistic-unweighted", false, &features, &labels, &split_manifest.assignments)?,
    );
    models.insert(
        "B5-balanced".into(),
        evaluate_model("B5-ftrl-logistic-class-balanced", true, &features, &labels, &split_manifest.assignments)?,
    );
    Ok(SecomLogisticReportV1 {
        schema: OUTPUT_SCHEMA_V1.into(),
        dataset_id: expectations.dataset_id.clone(),
        feature_file_sha256: raw_report.feature_sha256,
        label_file_sha256: raw_report.label_sha256,
        dataset_snapshot_sha256: split_manifest.dataset_snapshot_sha256.clone(),
        split_manifest_sha256: split_report.split_manifest_sha256,
        split_scheme: split_manifest.split_scheme.clone(),
        clock_domain_id: split_manifest.clock_domain_id.clone(),
        feature_count: raw_report.features_per_row,
        training_rows,
        training_failures,
        training_passes,
        models,
        claim_limits: SecomLogisticClaimLimitsV1 {
            imputation_normalization_and_model_fit_on_train_only: true,
            held_out_labels_not_used_for_fit_or_prior: true,
            held_out_features_not_used_for_fit: true,
            training_rows_sorted_by_sample_identity: true,
            does_not_establish_feature_semantics_or_process_causality: true,
            does_not_establish_current_fab_transfer_or_yield_improvement: true,
            source_mirror_official_uci_byte_identity_unverified: true,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::split_manifest::SplitManifestV1;
    use sha2::{Digest, Sha256};

    const FEATURES: &[u8] = include_bytes!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/secom-missingness-mini-features-v1.data"
    );
    const LABELS: &[u8] = include_bytes!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/secom-missingness-mini-labels-v1.data"
    );
    const EXPECTATIONS: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/secom-missingness-mini-expectations-v1.json"
    );
    const SPLIT: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/secom-missingness-mini-split-v1.json"
    );

    fn evaluate(features: &[u8], labels: &[u8]) -> SecomLogisticReportV1 {
        let mut expectations: SecomRawExpectationsV1 =
            serde_json::from_str(EXPECTATIONS).expect("expectations fixture parses");
        expectations.expected_feature_sha256 = sha256_hex(features);
        expectations.expected_label_sha256 = sha256_hex(labels);
        let mut split: SplitManifestV1 = serde_json::from_str(SPLIT).expect("split fixture parses");
        let parsed_labels = parse_labels(labels).expect("fixture labels parse");
        for (assignment, label) in split.assignments.iter_mut().zip(&parsed_labels) {
            assignment.positive = label.label == 1;
        }
        evaluate_secom_logistic_v1(features, labels, &expectations, &split)
            .expect("synthetic logistic fixture evaluates")
    }

    #[test]
    fn emits_unweighted_and_balanced_train_only_models() {
        let report = evaluate(FEATURES, LABELS);
        assert_eq!(report.training_rows, 4);
        assert_eq!(report.training_failures, 2);
        assert_eq!(report.training_passes, 2);
        assert!(report.models.contains_key("B5"));
        assert!(report.models.contains_key("B5-balanced"));
        for model in report.models.values() {
            for split in ["validation", "test"] {
                assert_eq!(model.held_out[split].rows, 2);
                for score in &model.held_out[split].scores {
                    assert!(score.failure_probability.is_finite());
                    assert!((0.0..=1.0).contains(&score.failure_probability));
                }
            }
        }
    }

    #[test]
    fn changing_held_out_labels_does_not_change_fitted_models() {
        let changed = String::from_utf8(LABELS.to_vec())
            .unwrap()
            .replace("-1 \"19/07/2008 12:15:00\"", "1 \"19/07/2008 12:15:00\"")
            .replace("1 \"19/07/2008 12:20:00\"", "-1 \"19/07/2008 12:20:00\"");
        let original = evaluate(FEATURES, LABELS);
        let modified = evaluate(FEATURES, changed.as_bytes());
        for key in ["B5", "B5-balanced"] {
            assert_eq!(original.models[key].model_sha256, modified.models[key].model_sha256);
        }
    }

    #[test]
    fn held_out_numeric_changes_do_not_change_fitted_model_digests() {
        let changed = String::from_utf8(FEATURES.to_vec())
            .unwrap()
            .replace("1.2 2.2", "9999.0 -9999.0");
        let original = evaluate(FEATURES, LABELS);
        let modified = evaluate(changed.as_bytes(), LABELS);
        for key in ["B5", "B5-balanced"] {
            assert_eq!(
                original.models[key].model_sha256,
                modified.models[key].model_sha256
            );
        }
    }

    #[test]
    fn all_missing_training_sensor_uses_explicit_zero_median_fallback() {
        let all_missing = b"1.0 NaN\n1.1 NaN\nNaN NaN\nNaN NaN\n1.2 NaN\nNaN NaN\n1.3 NaN\nNaN NaN\n";
        let report = evaluate(all_missing, LABELS);
        for model in report.models.values() {
            assert_eq!(model.missing_features_without_observed_training_values, 1);
        }
    }

    #[test]
    fn training_row_permutation_preserves_fitted_model_digest() {
        let features = parse_features(FEATURES).expect("fixture features parse");
        let labels = parse_labels(LABELS).expect("fixture labels parse");
        let split: SplitManifestV1 = serde_json::from_str(SPLIT).expect("split fixture parses");
        let mut permuted_features = features.clone();
        let mut permuted_labels = labels.clone();
        let mut permuted_assignments = split.assignments.clone();
        permuted_features[..4].reverse();
        permuted_labels[..4].reverse();
        permuted_assignments[..4].reverse();

        for balanced in [false, true] {
            let original = fit_model_v1(
                "sample-order",
                balanced,
                &features,
                &labels,
                &split.assignments,
            )
            .expect("original fit succeeds");
            let permuted = fit_model_v1(
                "sample-order",
                balanced,
                &permuted_features,
                &permuted_labels,
                &permuted_assignments,
            )
            .expect("permuted fit succeeds");
            assert_eq!(
                sha256_hex(&serde_json::to_vec(&original).unwrap()),
                sha256_hex(&serde_json::to_vec(&permuted).unwrap())
            );
        }
    }

    #[test]
    fn stable_sigmoid_and_ftrl_weights_handle_extreme_values() {
        assert!(sigmoid(10_000.0) > 0.999);
        assert!(sigmoid(-10_000.0) < 0.001);
        assert_eq!(ftrl_weight(0.0, 0.0, FTRL_ALPHA, FTRL_BETA, FTRL_L1, FTRL_L2), 0.0);
    }
}
