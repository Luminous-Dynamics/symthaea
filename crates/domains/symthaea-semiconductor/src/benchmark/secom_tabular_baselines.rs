// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Train-only value and missingness-indicator baselines for SECOM.
//!
//! B2 is median-imputed values-only Gaussian naive Bayes.
//! B3 adds Bernoulli missingness-mask likelihoods to B2.
//! B4 uses zero-imputed values plus the same mask likelihoods.
//! All fitted quantities are derived from the training partition only.

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

pub const OUTPUT_SCHEMA_V1: &str = "luminous.semi-bench.secom-tabular-baselines-report.v1";
const HELD_OUT_SPLITS: [&str; 2] = ["validation", "test"];
const BERNOULLI_SMOOTHING_ALPHA: f64 = 1.0;
const GAUSSIAN_VARIANCE_FLOOR: f64 = 1.0e-6;
const LN_2_PI: f64 = 1.8378770664093453;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SecomImputationPolicyV1 {
    TrainingMedian,
    Zero,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct SecomTabularBaselineError(pub String);

impl From<SecomRawDataError> for SecomTabularBaselineError {
    fn from(error: SecomRawDataError) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomTabularScoreV1 {
    pub sample_id: String,
    pub positive: bool,
    pub failure_probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomTabularHeldOutV1 {
    pub rows: usize,
    pub failures: usize,
    pub passes: usize,
    pub metrics: BinaryMetricReportV1,
    pub scores: Vec<SecomTabularScoreV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomTabularBaselineResultV1 {
    pub baseline_id: String,
    pub imputation_policy: SecomImputationPolicyV1,
    pub uses_missingness_indicators: bool,
    pub missing_features_without_observed_training_values: usize,
    pub model_sha256: String,
    pub variance_floor: f64,
    pub bernoulli_smoothing_alpha: Option<f64>,
    pub held_out: BTreeMap<String, SecomTabularHeldOutV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomTabularClaimLimitsV1 {
    pub medians_and_model_parameters_fit_on_train_only: bool,
    pub held_out_labels_not_used_for_fitting: bool,
    pub held_out_features_not_used_for_fitting: bool,
    pub missingness_indicators_are_distinct_from_imputed_values: bool,
    pub does_not_establish_feature_semantics_or_process_causality: bool,
    pub does_not_establish_current_fab_transfer_or_yield_improvement: bool,
    pub source_mirror_official_uci_byte_identity_unverified: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomTabularBaselinesReportV1 {
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
    pub baselines: BTreeMap<String, SecomTabularBaselineResultV1>,
    pub claim_limits: SecomTabularClaimLimitsV1,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GaussianBernoulliModelV1 {
    baseline_id: String,
    imputation_policy: SecomImputationPolicyV1,
    uses_missingness_indicators: bool,
    feature_count: usize,
    training_rows: usize,
    training_failures: usize,
    training_passes: usize,
    missing_features_without_observed_training_values: usize,
    medians: Vec<f64>,
    normalization_means: Vec<f64>,
    normalization_scales: Vec<f64>,
    class_log_priors: [f64; 2],
    class_means: Vec<[f64; 2]>,
    class_variances: Vec<[f64; 2]>,
    missing_probability_by_class: Vec<[f64; 2]>,
    variance_floor: f64,
    bernoulli_smoothing_alpha: f64,
}

impl GaussianBernoulliModelV1 {
    fn impute(&self, value: Option<f64>, feature_index: usize) -> f64 {
        match (self.imputation_policy, value) {
            (_, Some(observed)) => observed,
            (SecomImputationPolicyV1::TrainingMedian, None) => self.medians[feature_index],
            (SecomImputationPolicyV1::Zero, None) => 0.0,
        }
    }

    fn predict_probability(
        &self,
        row: &SecomSensorRowV1,
        sample_id: &str,
    ) -> Result<f64, SecomTabularBaselineError> {
        if row.values.len() != self.feature_count {
            return Err(SecomTabularBaselineError(format!(
                "sample {sample_id:?}: expected {} features, observed {}",
                self.feature_count,
                row.values.len()
            )));
        }

        // Class index 0 is pass; class index 1 is fail.
        let mut log_probability = self.class_log_priors;
        for feature_index in 0..self.feature_count {
            let standardized = (self.impute(row.values[feature_index], feature_index)
                - self.normalization_means[feature_index])
                / self.normalization_scales[feature_index];
            if !standardized.is_finite() {
                return Err(SecomTabularBaselineError(format!(
                    "sample {sample_id:?}, feature {feature_index}: non-finite standardized value"
                )));
            }

            for (class_index, class_log_probability) in log_probability.iter_mut().enumerate() {
                let variance = self.class_variances[feature_index][class_index];
                let delta = standardized - self.class_means[feature_index][class_index];
                let gaussian = -0.5 * (LN_2_PI + variance.ln() + delta * delta / variance);
                if !gaussian.is_finite() {
                    return Err(SecomTabularBaselineError(format!(
                        "sample {sample_id:?}, feature {feature_index}: non-finite Gaussian likelihood"
                    )));
                }
                *class_log_probability += gaussian;

                if self.uses_missingness_indicators {
                    let probability_missing =
                        self.missing_probability_by_class[feature_index][class_index];
                    *class_log_probability += if row.values[feature_index].is_none() {
                        probability_missing.ln()
                    } else {
                        (1.0 - probability_missing).ln()
                    };
                }
            }
        }

        let log_odds = log_probability[1] - log_probability[0];
        if !log_odds.is_finite() {
            return Err(SecomTabularBaselineError(format!(
                "sample {sample_id:?}: non-finite posterior log-odds"
            )));
        }
        let probability = if log_odds >= 0.0 {
            1.0 / (1.0 + (-log_odds).exp())
        } else {
            let exponential = log_odds.exp();
            exponential / (1.0 + exponential)
        };
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(SecomTabularBaselineError(format!(
                "sample {sample_id:?}: invalid failure probability {probability}"
            )));
        }
        Ok(probability)
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

fn impute_value(value: Option<f64>, policy: SecomImputationPolicyV1, median: f64) -> f64 {
    match (policy, value) {
        (_, Some(observed)) => observed,
        (SecomImputationPolicyV1::TrainingMedian, None) => median,
        (SecomImputationPolicyV1::Zero, None) => 0.0,
    }
}

fn fit_model_v1(
    baseline_id: &str,
    policy: SecomImputationPolicyV1,
    uses_missingness_indicators: bool,
    features: &[SecomSensorRowV1],
    labels: &[SecomLabelRowV1],
    assignments: &[crate::benchmark::split_manifest::SplitAssignmentV1],
) -> Result<GaussianBernoulliModelV1, SecomTabularBaselineError> {
    let feature_count = features
        .first()
        .map(|row| row.values.len())
        .ok_or_else(|| SecomTabularBaselineError("features: expected non-empty rows".into()))?;
    if feature_count == 0 || features.len() != labels.len() || features.len() != assignments.len() {
        return Err(SecomTabularBaselineError(
            "features, labels and split assignments must have equal non-zero row/feature dimensions"
                .into(),
        ));
    }

    let mut training_indices: Vec<usize> = assignments
        .iter()
        .enumerate()
        .filter_map(|(index, assignment)| (assignment.split == "train").then_some(index))
        .collect();
    // Canonical identity ordering makes floating-point accumulation independent
    // of manifest row order while preserving feature/label/sample alignment.
    training_indices.sort_by(|&left, &right| {
        assignments[left].sample_id.cmp(&assignments[right].sample_id)
    });
    if training_indices.is_empty() {
        return Err(SecomTabularBaselineError(
            "training split: expected non-empty rows".into(),
        ));
    }

    let mut class_counts = [0usize; 2];
    let mut missing_counts = vec![[0usize; 2]; feature_count];
    let mut observed_by_feature = vec![Vec::<f64>::new(); feature_count];

    for &row_index in &training_indices {
        let class_index = if labels[row_index].label == 1 { 1 } else { 0 };
        class_counts[class_index] += 1;
        if features[row_index].values.len() != feature_count {
            return Err(SecomTabularBaselineError(format!(
                "training row {row_index}: inconsistent feature count"
            )));
        }
        for (feature_index, value) in features[row_index].values.iter().enumerate() {
            if let Some(observed) = value {
                observed_by_feature[feature_index].push(*observed);
            } else {
                missing_counts[feature_index][class_index] += 1;
            }
        }
    }
    if class_counts[0] == 0 || class_counts[1] == 0 {
        return Err(SecomTabularBaselineError(
            "training split must contain both pass and fail samples".into(),
        ));
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

    // Fit normalization solely on imputed training rows.
    let mut normalization_means = vec![0.0; feature_count];
    let mut imputed_training = Vec::with_capacity(training_indices.len());
    for &row_index in &training_indices {
        let row: Vec<f64> = features[row_index]
            .values
            .iter()
            .enumerate()
            .map(|(feature_index, value)| impute_value(*value, policy, medians[feature_index]))
            .collect();
        for (feature_index, value) in row.iter().enumerate() {
            normalization_means[feature_index] += *value;
        }
        imputed_training.push(row);
    }
    for mean in &mut normalization_means {
        *mean /= training_indices.len() as f64;
        if !mean.is_finite() {
            return Err(SecomTabularBaselineError(
                "normalization: non-finite training mean".into(),
            ));
        }
    }

    let mut normalization_scales = vec![0.0; feature_count];
    for row in &imputed_training {
        for feature_index in 0..feature_count {
            let delta = row[feature_index] - normalization_means[feature_index];
            normalization_scales[feature_index] += delta * delta;
        }
    }
    for (feature_index, sum_squared_deviations) in normalization_scales.iter_mut().enumerate() {
        let variance = *sum_squared_deviations / training_indices.len() as f64;
        if !variance.is_finite() {
            return Err(SecomTabularBaselineError(format!(
                "feature {feature_index}: non-finite normalization variance"
            )));
        }
        let scale = variance.max(0.0).sqrt();
        *sum_squared_deviations = if scale > 1.0e-12 { scale } else { 1.0 };
    }

    let mut class_means = vec![[0.0; 2]; feature_count];
    let mut normalized_training = Vec::with_capacity(imputed_training.len());
    for (position, values) in imputed_training.iter().enumerate() {
        let row_index = training_indices[position];
        let class_index = if labels[row_index].label == 1 { 1 } else { 0 };
        let normalized: Vec<f64> = values
            .iter()
            .enumerate()
            .map(|(feature_index, value)| {
                (value - normalization_means[feature_index]) / normalization_scales[feature_index]
            })
            .collect();
        for (feature_index, value) in normalized.iter().enumerate() {
            class_means[feature_index][class_index] += *value;
        }
        normalized_training.push((class_index, normalized));
    }
    for (feature_index, means_by_class) in class_means.iter_mut().enumerate() {
        for (class_index, mean) in means_by_class.iter_mut().enumerate() {
            *mean /= class_counts[class_index] as f64;
            if !mean.is_finite() {
                return Err(SecomTabularBaselineError(format!(
                    "feature {feature_index}: non-finite class mean"
                )));
            }
        }
    }

    let mut class_variances = vec![[0.0; 2]; feature_count];
    for (class_index, row) in &normalized_training {
        for (feature_index, value) in row.iter().enumerate() {
            let delta = *value - class_means[feature_index][*class_index];
            class_variances[feature_index][*class_index] += delta * delta;
        }
    }
    for (feature_index, variances_by_class) in class_variances.iter_mut().enumerate() {
        for (class_index, variance_slot) in variances_by_class.iter_mut().enumerate() {
            let variance = *variance_slot / class_counts[class_index] as f64;
            if !variance.is_finite() {
                return Err(SecomTabularBaselineError(format!(
                    "feature {feature_index}: non-finite class variance"
                )));
            }
            *variance_slot = variance.max(GAUSSIAN_VARIANCE_FLOOR);
        }
    }

    let mut missing_probability_by_class = vec![[0.5; 2]; feature_count];
    if uses_missingness_indicators {
        for feature_index in 0..feature_count {
            for class_index in 0..2 {
                missing_probability_by_class[feature_index][class_index] =
                    (missing_counts[feature_index][class_index] as f64 + BERNOULLI_SMOOTHING_ALPHA)
                        / (class_counts[class_index] as f64 + 2.0 * BERNOULLI_SMOOTHING_ALPHA);
            }
        }
    }

    Ok(GaussianBernoulliModelV1 {
        baseline_id: baseline_id.into(),
        imputation_policy: policy,
        uses_missingness_indicators,
        feature_count,
        training_rows: training_indices.len(),
        training_failures: class_counts[1],
        training_passes: class_counts[0],
        missing_features_without_observed_training_values,
        medians,
        normalization_means,
        normalization_scales,
        class_log_priors: [
            (class_counts[0] as f64 / training_indices.len() as f64).ln(),
            (class_counts[1] as f64 / training_indices.len() as f64).ln(),
        ],
        class_means,
        class_variances,
        missing_probability_by_class,
        variance_floor: GAUSSIAN_VARIANCE_FLOOR,
        bernoulli_smoothing_alpha: BERNOULLI_SMOOTHING_ALPHA,
    })
}

fn held_out_subject(
    baseline_id: &str,
    split: &str,
    indices: &[usize],
    assignments: &[crate::benchmark::split_manifest::SplitAssignmentV1],
    labels: &[SecomLabelRowV1],
    scores: &[f64],
) -> Result<SecomTabularHeldOutV1, SecomTabularBaselineError> {
    if indices.len() != scores.len() {
        return Err(SecomTabularBaselineError(
            "held-out indices and scores differ in length".into(),
        ));
    }
    let samples: Vec<BinaryScoreSampleV1> = indices
        .iter()
        .enumerate()
        .map(|(position, &index)| BinaryScoreSampleV1 {
            sample_id: assignments[index].sample_id.clone(),
            positive: labels[index].label == 1,
            score: scores[position],
        })
        .collect();
    let failures = samples.iter().filter(|sample| sample.positive).count();
    let passes = samples.len() - failures;
    let subject = BinaryScoreSubjectV1 {
        schema: BINARY_INPUT_SCHEMA_V1.into(),
        case_id: Some(format!("{baseline_id}-{split}")),
        decision_threshold: 0.5,
        calibration_bins: 10,
        recall_target: 0.8,
        positive_class_semantics: Some(
            "SECOM in-house fail label (+1); values/imputation baseline".into(),
        ),
        samples,
    };
    let metrics = score_binary_subject_v1(&subject)
        .map_err(|error| SecomTabularBaselineError(error.to_string()))?;
    let scores = subject
        .samples
        .iter()
        .map(|sample| SecomTabularScoreV1 {
            sample_id: sample.sample_id.clone(),
            positive: sample.positive,
            failure_probability: sample.score,
        })
        .collect();
    Ok(SecomTabularHeldOutV1 {
        rows: indices.len(),
        failures,
        passes,
        metrics,
        scores,
    })
}

fn evaluate_baseline(
    baseline_id: &str,
    policy: SecomImputationPolicyV1,
    uses_missingness_indicators: bool,
    features: &[SecomSensorRowV1],
    labels: &[SecomLabelRowV1],
    assignments: &[crate::benchmark::split_manifest::SplitAssignmentV1],
) -> Result<SecomTabularBaselineResultV1, SecomTabularBaselineError> {
    let model = fit_model_v1(
        baseline_id,
        policy,
        uses_missingness_indicators,
        features,
        labels,
        assignments,
    )?;
    let model_bytes = serde_json::to_vec(&model).map_err(|error| {
        SecomTabularBaselineError(format!("model serialization failed: {error}"))
    })?;
    let model_sha256 = sha256_hex(&model_bytes);

    let mut held_out = BTreeMap::new();
    for split in HELD_OUT_SPLITS {
        let indices: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter_map(|(index, assignment)| (assignment.split == split).then_some(index))
            .collect();
        if indices.is_empty() {
            return Err(SecomTabularBaselineError(format!(
                "{split}: expected non-empty held-out rows"
            )));
        }
        let scores = indices
            .iter()
            .map(|&index| {
                model.predict_probability(&features[index], &assignments[index].sample_id)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let report = held_out_subject(baseline_id, split, &indices, assignments, labels, &scores)?;
        held_out.insert(split.into(), report);
    }

    Ok(SecomTabularBaselineResultV1 {
        baseline_id: baseline_id.into(),
        imputation_policy: policy,
        uses_missingness_indicators,
        missing_features_without_observed_training_values: model
            .missing_features_without_observed_training_values,
        model_sha256,
        variance_floor: model.variance_floor,
        bernoulli_smoothing_alpha: uses_missingness_indicators
            .then_some(model.bernoulli_smoothing_alpha),
        held_out,
    })
}

/// Fits B2, B3 and B4 from train rows only, then scores validation and test.
pub fn evaluate_secom_tabular_baselines_v1(
    feature_bytes: &[u8],
    label_bytes: &[u8],
    expectations: &SecomRawExpectationsV1,
    split_manifest: &SplitManifestV1,
) -> Result<SecomTabularBaselinesReportV1, SecomTabularBaselineError> {
    let raw_report = audit_secom_raw_v1(feature_bytes, label_bytes, expectations)?;
    if split_manifest.dataset_id != expectations.dataset_id {
        return Err(SecomTabularBaselineError(
            "split_manifest.dataset_id does not match raw expectations".into(),
        ));
    }
    let split_report = validate_split_manifest_v1(split_manifest)
        .map_err(|error| SecomTabularBaselineError(error.to_string()))?;
    if split_manifest.assignments.len() != raw_report.rows {
        return Err(SecomTabularBaselineError(format!(
            "split assignments: expected {} rows, observed {}",
            raw_report.rows,
            split_manifest.assignments.len()
        )));
    }
    if split_manifest.split_scheme == "forward_time"
        && split_manifest.clock_domain_id.as_deref() != Some(expectations.clock_domain_id.as_str())
    {
        return Err(SecomTabularBaselineError(
            "forward-time split clock_domain_id does not match raw expectations".into(),
        ));
    }

    let features = parse_features(feature_bytes)?;
    let labels = parse_labels(label_bytes)?;
    for (index, (assignment, label)) in split_manifest.assignments.iter().zip(&labels).enumerate() {
        if assignment.positive != (label.label == 1) {
            return Err(SecomTabularBaselineError(format!(
                "split assignment {index} label does not match source label row"
            )));
        }
        if let Some(observed_at) = assignment.observed_at.as_deref() {
            let source_time = label.observed_at.format("%Y-%m-%dT%H:%M:%S").to_string();
            if observed_at != source_time {
                return Err(SecomTabularBaselineError(format!(
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
    if training_rows == 0 {
        return Err(SecomTabularBaselineError(
            "training split: expected non-empty rows".into(),
        ));
    }

    let mut baselines = BTreeMap::new();
    baselines.insert(
        "B2".into(),
        evaluate_baseline(
            "B2-values-only-median",
            SecomImputationPolicyV1::TrainingMedian,
            false,
            &features,
            &labels,
            &split_manifest.assignments,
        )?,
    );
    baselines.insert(
        "B3".into(),
        evaluate_baseline(
            "B3-values-plus-mask-median",
            SecomImputationPolicyV1::TrainingMedian,
            true,
            &features,
            &labels,
            &split_manifest.assignments,
        )?,
    );
    baselines.insert(
        "B4".into(),
        evaluate_baseline(
            "B4-values-plus-mask-zero",
            SecomImputationPolicyV1::Zero,
            true,
            &features,
            &labels,
            &split_manifest.assignments,
        )?,
    );

    Ok(SecomTabularBaselinesReportV1 {
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
        baselines,
        claim_limits: SecomTabularClaimLimitsV1 {
            medians_and_model_parameters_fit_on_train_only: true,
            held_out_labels_not_used_for_fitting: true,
            held_out_features_not_used_for_fitting: true,
            missingness_indicators_are_distinct_from_imputed_values: true,
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

    fn evaluate(feature_bytes: &[u8], label_bytes: &[u8]) -> SecomTabularBaselinesReportV1 {
        let mut expectations: SecomRawExpectationsV1 =
            serde_json::from_str(EXPECTATIONS).expect("expectations fixture parses");
        expectations.expected_feature_sha256 = format!("{:x}", Sha256::digest(feature_bytes));
        expectations.expected_label_sha256 = format!("{:x}", Sha256::digest(label_bytes));
        let mut split: SplitManifestV1 = serde_json::from_str(SPLIT).expect("split fixture parses");
        let parsed_labels = parse_labels(label_bytes).expect("fixture labels parse");
        for (assignment, label) in split.assignments.iter_mut().zip(&parsed_labels) {
            assignment.positive = label.label == 1;
        }
        evaluate_secom_tabular_baselines_v1(feature_bytes, label_bytes, &expectations, &split)
            .expect("synthetic fixture evaluates")
    }

    #[test]
    fn emits_b2_b3_b4_metrics_on_the_synthetic_fixture() {
        let report = evaluate(FEATURES, LABELS);
        assert_eq!(report.training_rows, 4);
        assert_eq!(report.training_failures, 2);
        assert_eq!(report.training_passes, 2);
        for baseline in ["B2", "B3", "B4"] {
            assert_eq!(report.baselines[baseline].held_out["validation"].rows, 2);
            assert_eq!(report.baselines[baseline].held_out["test"].rows, 2);
            for split in ["validation", "test"] {
                for score in &report.baselines[baseline].held_out[split].scores {
                    assert!(score.failure_probability.is_finite());
                    assert!((0.0..=1.0).contains(&score.failure_probability));
                }
            }
        }
        assert!(!report.baselines["B2"].uses_missingness_indicators);
        assert!(report.baselines["B3"].uses_missingness_indicators);
        assert!(report.baselines["B4"].uses_missingness_indicators);
    }

    #[test]
    fn b2_ignores_mask_when_imputed_value_is_identical_but_b3_uses_it() {
        let base = String::from_utf8(FEATURES.to_vec())
            .unwrap()
            .replace("1.2 2.2", "1.05 2.05");
        let changed_mask = base.replace("1.05 2.05", "NaN 2.05");
        let original = evaluate(base.as_bytes(), LABELS);
        let changed = evaluate(changed_mask.as_bytes(), LABELS);

        let original_b2 = &original.baselines["B2"].held_out["validation"].scores[0];
        let changed_b2 = &changed.baselines["B2"].held_out["validation"].scores[0];
        assert_eq!(
            original_b2.failure_probability,
            changed_b2.failure_probability
        );

        let original_b3 = &original.baselines["B3"].held_out["validation"].scores[0];
        let changed_b3 = &changed.baselines["B3"].held_out["validation"].scores[0];
        assert_ne!(
            original_b3.failure_probability,
            changed_b3.failure_probability
        );
    }

    #[test]
    fn held_out_label_changes_do_not_change_any_fitted_model_digest() {
        let changed_labels = String::from_utf8(LABELS.to_vec())
            .unwrap()
            .replace("-1 \"19/07/2008 12:15:00\"", "1 \"19/07/2008 12:15:00\"")
            .replace("1 \"19/07/2008 12:20:00\"", "-1 \"19/07/2008 12:20:00\"");
        let original = evaluate(FEATURES, LABELS);
        let changed = evaluate(FEATURES, changed_labels.as_bytes());
        for baseline in ["B2", "B3", "B4"] {
            assert_eq!(
                original.baselines[baseline].model_sha256,
                changed.baselines[baseline].model_sha256
            );
        }
    }

    #[test]
    fn held_out_feature_changes_do_not_change_fitted_model_digests() {
        let changed = String::from_utf8(FEATURES.to_vec())
            .unwrap()
            .replace("1.2 2.2", "9999.0 -9999.0");
        let original = evaluate(FEATURES, LABELS);
        let modified = evaluate(changed.as_bytes(), LABELS);
        for baseline in ["B2", "B3", "B4"] {
            assert_eq!(
                original.baselines[baseline].model_sha256,
                modified.baselines[baseline].model_sha256
            );
        }
    }

    #[test]
    fn all_missing_training_feature_uses_explicit_zero_fallback() {
        let all_missing =
            b"1.0 NaN\n1.1 NaN\nNaN NaN\nNaN NaN\n1.2 NaN\nNaN NaN\n1.3 NaN\nNaN NaN\n";
        let report = evaluate(all_missing, LABELS);
        for baseline in ["B2", "B3", "B4"] {
            assert_eq!(
                report.baselines[baseline].missing_features_without_observed_training_values,
                1
            );
        }
    }

    #[test]
    fn sample_input_order_does_not_affect_fitted_model_digest() {
        let features = parse_features(FEATURES).expect("fixture features parse");
        let labels = parse_labels(LABELS).expect("fixture labels parse");
        let split: SplitManifestV1 = serde_json::from_str(SPLIT).expect("split fixture parses");

        let digest_for =
            |baseline_id: &str,
             policy: SecomImputationPolicyV1,
             uses_missingness_indicators: bool,
             features: &[SecomSensorRowV1],
             labels: &[SecomLabelRowV1],
             assignments: &[crate::benchmark::split_manifest::SplitAssignmentV1]| {
                let model = fit_model_v1(
                    baseline_id,
                    policy,
                    uses_missingness_indicators,
                    features,
                    labels,
                    assignments,
                )
                .expect("training-only fit succeeds");
                sha256_hex(&serde_json::to_vec(&model).expect("model serializes"))
            };

        let configurations = [
            ("B2", SecomImputationPolicyV1::TrainingMedian, false),
            ("B3", SecomImputationPolicyV1::TrainingMedian, true),
            ("B4", SecomImputationPolicyV1::Zero, true),
        ];

        let mut permuted_features = features.clone();
        let mut permuted_labels = labels.clone();
        let mut permuted_assignments = split.assignments.clone();
        // Reorder paired training rows inside the model-fitting boundary.
        // The raw-file auditor still enforces the source's chronological order.
        permuted_features[..4].reverse();
        permuted_labels[..4].reverse();
        permuted_assignments[..4].reverse();

        for (baseline_id, policy, with_mask) in configurations {
            let original = digest_for(
                baseline_id,
                policy,
                with_mask,
                &features,
                &labels,
                &split.assignments,
            );
            let permuted = digest_for(
                baseline_id,
                policy,
                with_mask,
                &permuted_features,
                &permuted_labels,
                &permuted_assignments,
            );
            assert_eq!(
                original, permuted,
                "{baseline_id} must be row-order invariant"
            );
        }
    }
}
