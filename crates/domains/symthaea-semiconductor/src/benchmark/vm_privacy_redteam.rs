// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Synthetic model-update inversion red-team harness for VM privacy studies.
//!
//! This module demonstrates a known single-record logistic-gradient leakage
//! channel. It is an attack fixture, not a privacy defence or a formal DP proof.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

pub const VM_PRIVACY_ATTACK_SCHEMA_V1: &str = "luminous.semi-bench.vm-update-inversion-attack.v1";
pub const VM_PRIVACY_REPORT_SCHEMA_V1: &str = "luminous.semi-bench.vm-update-inversion-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateVisibilityV1 {
    PerRecord,
    PerClientBatch,
    SecureAggregateOnly,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticPrivacyRecordV1 {
    pub record_id: String,
    pub client_id: String,
    pub features: Vec<f64>,
    pub label: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmPrivacyAttackConfigV1 {
    pub schema: String,
    pub run_id: String,
    pub visibility: UpdateVisibilityV1,
    /// Per-example gradient clipping norm. Zero disables clipping.
    pub clipping_norm: f64,
    /// Per-coordinate Gaussian noise standard deviation. Zero disables noise.
    /// This is not an epsilon-DP parameter without a sampling/accounting model.
    pub gaussian_noise_stddev: f64,
    pub seed: u64,
    pub minimum_bias_gradient_magnitude: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyAttackStatusV1 {
    ExactRecordReconstructionDemonstrated,
    PartialRecordReconstruction,
    PerRecordAttackNotApplicableToAggregateVisibility,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmPrivacyAttackReportV1 {
    pub schema: String,
    pub run_id: String,
    pub visibility: UpdateVisibilityV1,
    pub attack_status: PrivacyAttackStatusV1,
    pub training_record_count: usize,
    pub client_count: usize,
    pub observed_update_count: usize,
    pub reconstructed_record_count: usize,
    pub feature_dimension: usize,
    pub mean_feature_rmse: Option<f64>,
    pub maximum_feature_rmse: Option<f64>,
    pub label_inference_accuracy: Option<f64>,
    pub clipping_norm: f64,
    pub gaussian_noise_stddev: f64,
    pub update_payload_sha256: String,
    pub report_sha256: String,
    pub formal_differential_privacy_claim: bool,
    pub claim_limits: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct VmPrivacyAttackErrorV1(pub String);

#[derive(Clone, Debug)]
struct PrivacySplitMix64 {
    state: u64,
}

impl PrivacySplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit_open(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) / ((1_u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let first = self.unit_open();
        let second = self.unit_open();
        (-2.0 * first.ln()).sqrt() * (std::f64::consts::TAU * second).cos()
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, VmPrivacyAttackErrorV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| VmPrivacyAttackErrorV1(format!("serialization failed: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn validate_inputs(
    records: &[SyntheticPrivacyRecordV1],
    model_weights: &[f64],
    model_bias: f64,
    config: &VmPrivacyAttackConfigV1,
) -> Result<(), VmPrivacyAttackErrorV1> {
    if config.schema != VM_PRIVACY_ATTACK_SCHEMA_V1 {
        return Err(VmPrivacyAttackErrorV1(
            "attack_config.schema: unsupported".into(),
        ));
    }
    if config.run_id.trim().is_empty() {
        return Err(VmPrivacyAttackErrorV1("run_id must be non-empty".into()));
    }
    if records.is_empty()
        || model_weights.is_empty()
        || model_bias.is_nan()
        || !model_bias.is_finite()
    {
        return Err(VmPrivacyAttackErrorV1(
            "records, model weights and finite bias are required".into(),
        ));
    }
    if config.clipping_norm < 0.0
        || !config.clipping_norm.is_finite()
        || config.gaussian_noise_stddev < 0.0
        || !config.gaussian_noise_stddev.is_finite()
        || config.minimum_bias_gradient_magnitude <= 0.0
        || !config.minimum_bias_gradient_magnitude.is_finite()
    {
        return Err(VmPrivacyAttackErrorV1(
            "clipping, noise and inversion-threshold parameters are invalid".into(),
        ));
    }
    if model_weights.iter().any(|weight| !weight.is_finite()) {
        return Err(VmPrivacyAttackErrorV1(
            "model weights must be finite".into(),
        ));
    }

    let mut record_ids = std::collections::BTreeSet::new();
    for (index, record) in records.iter().enumerate() {
        if record.record_id.trim().is_empty()
            || record.client_id.trim().is_empty()
            || record.features.len() != model_weights.len()
            || record.features.iter().any(|feature| !feature.is_finite())
            || !record_ids.insert(record.record_id.as_str())
        {
            return Err(VmPrivacyAttackErrorV1(format!(
                "records[{index}]: invalid identity, duplicate ID or feature vector"
            )));
        }
    }
    Ok(())
}

fn logistic_gradient(
    features: &[f64],
    label: bool,
    model_weights: &[f64],
    model_bias: f64,
) -> Vec<f64> {
    let logit = model_weights
        .iter()
        .zip(features)
        .map(|(weight, feature)| weight * feature)
        .sum::<f64>()
        + model_bias;
    let residual = sigmoid(logit) - if label { 1.0 } else { 0.0 };
    let mut gradient: Vec<f64> = features.iter().map(|feature| residual * feature).collect();
    gradient.push(residual);
    gradient
}

fn clip_gradient(gradient: &mut [f64], clipping_norm: f64) {
    if clipping_norm <= 0.0 {
        return;
    }
    let norm = gradient
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm > clipping_norm && norm > 0.0 {
        let scale = clipping_norm / norm;
        for value in gradient {
            *value *= scale;
        }
    }
}

fn add_noise(gradient: &mut [f64], noise_stddev: f64, rng: &mut PrivacySplitMix64) {
    if noise_stddev <= 0.0 {
        return;
    }
    for value in gradient {
        *value += noise_stddev * rng.normal();
    }
}

fn aggregate_by_client(
    records: &[SyntheticPrivacyRecordV1],
    gradients: &[Vec<f64>],
) -> Vec<(String, Vec<f64>)> {
    let mut sums = BTreeMap::<String, (Vec<f64>, usize)>::new();
    for (record, gradient) in records.iter().zip(gradients) {
        let entry = sums
            .entry(record.client_id.clone())
            .or_insert_with(|| (vec![0.0; gradient.len()], 0));
        for (sum, value) in entry.0.iter_mut().zip(gradient) {
            *sum += *value;
        }
        entry.1 += 1;
    }
    sums.into_iter()
        .map(|(client_id, (mut sum, count))| {
            for value in &mut sum {
                *value /= count as f64;
            }
            (client_id, sum)
        })
        .collect()
}

/// Generates the update payload visible to the declared attacker and attempts
/// the exact single-record logistic-gradient inversion when that visibility
/// model exposes per-record updates.
///
/// For a single record, logistic regression emits:
///   grad_w = (sigmoid(w·x+b) - y) * x
///   grad_b = sigmoid(w·x+b) - y
/// Therefore x = grad_w / grad_b and sign(grad_b) reveals y, except at
/// numerically degenerate points. Multiplicative norm clipping preserves this
/// ratio and is not, by itself, a defence against this attack.
pub fn evaluate_vm_update_inversion_v1(
    records: &[SyntheticPrivacyRecordV1],
    model_weights: &[f64],
    model_bias: f64,
    config: &VmPrivacyAttackConfigV1,
) -> Result<VmPrivacyAttackReportV1, VmPrivacyAttackErrorV1> {
    validate_inputs(records, model_weights, model_bias, config)?;

    let mut rng = PrivacySplitMix64::new(config.seed);
    let mut per_record_gradients = Vec::with_capacity(records.len());
    for record in records {
        let mut gradient =
            logistic_gradient(&record.features, record.label, model_weights, model_bias);
        clip_gradient(&mut gradient, config.clipping_norm);
        per_record_gradients.push(gradient);
    }

    let (observed_updates, attack_attempted): (Vec<Vec<f64>>, bool) = match config.visibility {
        UpdateVisibilityV1::PerRecord => {
            let updates = per_record_gradients
                .iter()
                .cloned()
                .map(|mut gradient| {
                    add_noise(&mut gradient, config.gaussian_noise_stddev, &mut rng);
                    gradient
                })
                .collect();
            (updates, true)
        }
        UpdateVisibilityV1::PerClientBatch => {
            let updates = aggregate_by_client(records, &per_record_gradients)
                .into_iter()
                .map(|(_, mut gradient)| {
                    add_noise(&mut gradient, config.gaussian_noise_stddev, &mut rng);
                    gradient
                })
                .collect();
            (updates, false)
        }
        UpdateVisibilityV1::SecureAggregateOnly => {
            let mut aggregate = vec![0.0; model_weights.len() + 1];
            for gradient in &per_record_gradients {
                for (sum, value) in aggregate.iter_mut().zip(gradient) {
                    *sum += *value;
                }
            }
            for value in &mut aggregate {
                *value /= records.len() as f64;
            }
            add_noise(&mut aggregate, config.gaussian_noise_stddev, &mut rng);
            (vec![aggregate], false)
        }
    };

    let update_payload_sha256 = json_sha256(&observed_updates)?;
    let mut reconstructed_record_count = 0_usize;
    let mut feature_squared_errors = Vec::new();
    let mut correct_labels = 0_usize;

    if attack_attempted {
        for (record, update) in records.iter().zip(&observed_updates) {
            let bias_gradient = update[model_weights.len()];
            if bias_gradient.abs() < config.minimum_bias_gradient_magnitude {
                continue;
            }
            reconstructed_record_count += 1;
            for (index, feature) in record.features.iter().enumerate() {
                let reconstructed = update[index] / bias_gradient;
                feature_squared_errors.push((reconstructed - feature).powi(2));
            }
            let inferred_label = bias_gradient < 0.0;
            if inferred_label == record.label {
                correct_labels += 1;
            }
        }
    }

    let mean_feature_rmse = (!feature_squared_errors.is_empty()).then(|| {
        (feature_squared_errors.iter().sum::<f64>() / feature_squared_errors.len() as f64).sqrt()
    });
    let maximum_feature_rmse = if attack_attempted && reconstructed_record_count > 0 {
        let mut per_record_rmse = Vec::with_capacity(reconstructed_record_count);
        for (record, update) in records.iter().zip(&observed_updates) {
            let bias_gradient = update[model_weights.len()];
            if bias_gradient.abs() < config.minimum_bias_gradient_magnitude {
                continue;
            }
            let squared_error = record
                .features
                .iter()
                .enumerate()
                .map(|(index, feature)| (update[index] / bias_gradient - feature).powi(2))
                .sum::<f64>()
                / model_weights.len() as f64;
            per_record_rmse.push(squared_error.sqrt());
        }
        per_record_rmse.into_iter().reduce(f64::max)
    } else {
        None
    };
    let label_inference_accuracy = (attack_attempted && reconstructed_record_count > 0)
        .then(|| correct_labels as f64 / reconstructed_record_count as f64);

    let attack_status = if !attack_attempted {
        PrivacyAttackStatusV1::PerRecordAttackNotApplicableToAggregateVisibility
    } else if reconstructed_record_count == records.len() {
        PrivacyAttackStatusV1::ExactRecordReconstructionDemonstrated
    } else {
        PrivacyAttackStatusV1::PartialRecordReconstruction
    };

    let client_count = records
        .iter()
        .map(|record| record.client_id.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let mut report = VmPrivacyAttackReportV1 {
        schema: VM_PRIVACY_REPORT_SCHEMA_V1.into(),
        run_id: config.run_id.clone(),
        visibility: config.visibility,
        attack_status,
        training_record_count: records.len(),
        client_count,
        observed_update_count: observed_updates.len(),
        reconstructed_record_count,
        feature_dimension: model_weights.len(),
        mean_feature_rmse,
        maximum_feature_rmse,
        label_inference_accuracy,
        clipping_norm: config.clipping_norm,
        gaussian_noise_stddev: config.gaussian_noise_stddev,
        update_payload_sha256,
        report_sha256: String::new(),
        formal_differential_privacy_claim: false,
        claim_limits: vec![
            "Synthetic logistic-model attack only; not a general federated VM privacy proof".into(),
            "Aggregate-visibility modes are not attacked by this single-record inversion routine".into(),
            "Noise standard deviation alone is not an epsilon/delta differential-privacy guarantee".into(),
            "Per-record norm clipping preserves the feature-to-bias gradient ratio in this attack".into(),
            "A failed attack does not establish privacy; threat-model and auxiliary-data coverage matter".into(),
        ],
    };
    report.report_sha256.clear();
    report.report_sha256 = json_sha256(&report)?;
    Ok(report)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuxiliaryInversionOutcomeV1 {
    SingleUnknownRecordReconstructed,
    NoAggregateWithExactlyOneUnknownRecord,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmAuxiliaryInversionReportV1 {
    pub schema: String,
    pub run_id: String,
    pub visibility: UpdateVisibilityV1,
    pub outcome: AuxiliaryInversionOutcomeV1,
    pub total_record_count: usize,
    pub known_auxiliary_record_count: usize,
    pub unknown_record_count: usize,
    pub reconstructed_record_count: usize,
    pub reconstructed_record_ids_for_evaluation: Vec<String>,
    pub mean_feature_rmse: Option<f64>,
    pub label_inference_accuracy: Option<f64>,
    pub update_payload_sha256: String,
    pub report_sha256: String,
    pub formal_differential_privacy_claim: bool,
    pub claim_limits: Vec<String>,
}

fn sum_gradients(
    gradients: impl Iterator<Item = Vec<f64>>,
    dimension: usize,
) -> Vec<f64> {
    let mut sum = vec![0.0; dimension];
    for gradient in gradients {
        for (target, value) in sum.iter_mut().zip(gradient) {
            *target += value;
        }
    }
    sum
}

fn attempt_single_record_inversion(
    residual_gradient: &[f64],
    model_dimension: usize,
    true_record: &SyntheticPrivacyRecordV1,
    minimum_bias_gradient_magnitude: f64,
) -> Option<(f64, bool)> {
    let bias_gradient = residual_gradient[model_dimension];
    if bias_gradient.abs() < minimum_bias_gradient_magnitude {
        return None;
    }
    let squared_error = true_record
        .features
        .iter()
        .enumerate()
        .map(|(index, feature)| {
            (residual_gradient[index] / bias_gradient - feature).powi(2)
        })
        .sum::<f64>()
        / model_dimension as f64;
    let inferred_label = bias_gradient < 0.0;
    Some((squared_error.sqrt(), inferred_label == true_record.label))
}

/// Simulates a stronger auxiliary-knowledge attacker against aggregate updates.
/// The attacker knows the exact features and labels of all auxiliary records,
/// subtracts their per-record gradients from a visible batch/secure aggregate,
/// and attempts inversion when exactly one unknown record remains in a group.
///
/// This tests an important limit: aggregation hides individual messages, but
/// does not automatically prevent deconvolution when the attacker knows all
/// but one contributing record. The assumption that cohort membership/counts
/// are visible is explicit in this synthetic protocol.
pub fn evaluate_vm_known_auxiliary_batch_inversion_v1(
    records: &[SyntheticPrivacyRecordV1],
    known_auxiliary_records: &[SyntheticPrivacyRecordV1],
    model_weights: &[f64],
    model_bias: f64,
    config: &VmPrivacyAttackConfigV1,
) -> Result<VmAuxiliaryInversionReportV1, VmPrivacyAttackErrorV1> {
    validate_inputs(records, model_weights, model_bias, config)?;
    if config.visibility == UpdateVisibilityV1::PerRecord {
        return Err(VmPrivacyAttackErrorV1(
            "auxiliary batch inversion requires per-client or secure aggregate visibility"
                .into(),
        ));
    }

    let records_by_id: BTreeMap<_, _> = records
        .iter()
        .map(|record| (record.record_id.as_str(), record))
        .collect();
    let mut known_ids = std::collections::BTreeSet::new();
    for (index, known) in known_auxiliary_records.iter().enumerate() {
        let Some(actual) = records_by_id.get(known.record_id.as_str()) else {
            return Err(VmPrivacyAttackErrorV1(format!(
                "known_auxiliary_records[{index}]: record ID is not in the contributing population"
            )));
        };
        if *actual != known || !known_ids.insert(known.record_id.as_str()) {
            return Err(VmPrivacyAttackErrorV1(format!(
                "known_auxiliary_records[{index}]: record is inconsistent or duplicated"
            )));
        }
    }
    if known_auxiliary_records.len() >= records.len() {
        return Err(VmPrivacyAttackErrorV1(
            "at least one contributing record must remain unknown".into(),
        ));
    }

    let dimension = model_weights.len() + 1;
    let mut gradients_by_id = BTreeMap::<&str, Vec<f64>>::new();
    for record in records {
        let mut gradient =
            logistic_gradient(&record.features, record.label, model_weights, model_bias);
        clip_gradient(&mut gradient, config.clipping_norm);
        gradients_by_id.insert(record.record_id.as_str(), gradient);
    }

    let mut rng = PrivacySplitMix64::new(config.seed);
    let mut payload = Vec::<(String, Vec<f64>, usize)>::new();
    match config.visibility {
        UpdateVisibilityV1::PerClientBatch => {
            let mut grouped_ids = BTreeMap::<String, Vec<&SyntheticPrivacyRecordV1>>::new();
            for record in records {
                grouped_ids
                    .entry(record.client_id.clone())
                    .or_default()
                    .push(record);
            }
            for (client_id, group) in grouped_ids {
                let gradient_sum = sum_gradients(
                    group
                        .iter()
                        .map(|record| gradients_by_id[record.record_id.as_str()].clone()),
                    dimension,
                );
                let mut mean_gradient: Vec<f64> = gradient_sum
                    .into_iter()
                    .map(|value| value / group.len() as f64)
                    .collect();
                add_noise(&mut mean_gradient, config.gaussian_noise_stddev, &mut rng);
                payload.push((client_id, mean_gradient, group.len()));
            }
        }
        UpdateVisibilityV1::SecureAggregateOnly => {
            let gradient_sum = sum_gradients(
                records
                    .iter()
                    .map(|record| gradients_by_id[record.record_id.as_str()].clone()),
                dimension,
            );
            let mut mean_gradient: Vec<f64> = gradient_sum
                .into_iter()
                .map(|value| value / records.len() as f64)
                .collect();
            add_noise(&mut mean_gradient, config.gaussian_noise_stddev, &mut rng);
            payload.push(("secure-aggregate".into(), mean_gradient, records.len()));
        }
        UpdateVisibilityV1::PerRecord => unreachable!("rejected above"),
    }

    let update_payload_sha256 = json_sha256(&payload)?;
    let mut reconstructed_record_ids = Vec::new();
    let mut feature_squared_errors = Vec::new();
    let mut correct_labels = 0_usize;

    for (group_id, observed_mean, group_count) in &payload {
        let group_records: Vec<_> = records
            .iter()
            .filter(|record| {
                config.visibility == UpdateVisibilityV1::SecureAggregateOnly
                    || &record.client_id == group_id
            })
            .collect();
        let known_in_group: Vec<_> = group_records
            .iter()
            .filter(|record| known_ids.contains(record.record_id.as_str()))
            .collect();
        let unknown_in_group: Vec<_> = group_records
            .iter()
            .filter(|record| !known_ids.contains(record.record_id.as_str()))
            .collect();
        if unknown_in_group.len() != 1 {
            continue;
        }

        // Undo mean aggregation using the visible cohort count, then subtract
        // known per-record gradients. Scalar multiplicative normalization
        // would cancel in the feature/bias ratio, but sum form is explicit.
        let mut residual = observed_mean
            .iter()
            .map(|value| value * *group_count as f64)
            .collect::<Vec<_>>();
        let known_gradient_sum = sum_gradients(
            known_in_group
                .iter()
                .map(|record| gradients_by_id[record.record_id.as_str()].clone()),
            dimension,
        );
        for (value, known_value) in residual.iter_mut().zip(known_gradient_sum) {
            *value -= known_value;
        }

        let target = unknown_in_group[0];
        if let Some((rmse, label_correct)) = attempt_single_record_inversion(
            &residual,
            model_weights.len(),
            target,
            config.minimum_bias_gradient_magnitude,
        ) {
            reconstructed_record_ids.push(target.record_id.clone());
            feature_squared_errors.push(rmse.powi(2));
            if label_correct {
                correct_labels += 1;
            }
        }
    }

    let reconstructed_record_count = reconstructed_record_ids.len();
    let outcome = if reconstructed_record_count > 0 {
        AuxiliaryInversionOutcomeV1::SingleUnknownRecordReconstructed
    } else {
        AuxiliaryInversionOutcomeV1::NoAggregateWithExactlyOneUnknownRecord
    };
    let mut report = VmAuxiliaryInversionReportV1 {
        schema: "luminous.semi-bench.vm-auxiliary-batch-inversion-report.v1".into(),
        run_id: config.run_id.clone(),
        visibility: config.visibility,
        outcome,
        total_record_count: records.len(),
        known_auxiliary_record_count: known_auxiliary_records.len(),
        unknown_record_count: records.len() - known_auxiliary_records.len(),
        reconstructed_record_count,
        reconstructed_record_ids_for_evaluation: reconstructed_record_ids,
        mean_feature_rmse: (!feature_squared_errors.is_empty()).then(|| {
            (feature_squared_errors.iter().sum::<f64>()
                / feature_squared_errors.len() as f64)
                .sqrt()
        }),
        label_inference_accuracy: (reconstructed_record_count > 0).then(|| {
            correct_labels as f64 / reconstructed_record_count as f64
        }),
        update_payload_sha256,
        report_sha256: String::new(),
        formal_differential_privacy_claim: false,
        claim_limits: vec![
            "Synthetic logistic-model known-auxiliary attack only".into(),
            "Assumes attacker knows exact features/labels of auxiliary records and visible cohort counts".into(),
            "Only groups with exactly one unknown record are inverted".into(),
            "A failed attack does not establish privacy for other auxiliary knowledge or attack families".into(),
            "No formal differential-privacy guarantee is computed".into(),
        ],
    };
    report.report_sha256 = json_sha256(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records() -> Vec<SyntheticPrivacyRecordV1> {
        vec![
            SyntheticPrivacyRecordV1 {
                record_id: "wafer-001".into(),
                client_id: "fab-a".into(),
                features: vec![0.2, -0.4, 0.7],
                label: true,
            },
            SyntheticPrivacyRecordV1 {
                record_id: "wafer-002".into(),
                client_id: "fab-a".into(),
                features: vec![-0.3, 0.5, 0.1],
                label: false,
            },
            SyntheticPrivacyRecordV1 {
                record_id: "wafer-003".into(),
                client_id: "fab-b".into(),
                features: vec![0.6, 0.2, -0.5],
                label: true,
            },
            SyntheticPrivacyRecordV1 {
                record_id: "wafer-004".into(),
                client_id: "fab-b".into(),
                features: vec![0.1, -0.8, 0.3],
                label: false,
            },
        ]
    }

    fn config(visibility: UpdateVisibilityV1) -> VmPrivacyAttackConfigV1 {
        VmPrivacyAttackConfigV1 {
            schema: VM_PRIVACY_ATTACK_SCHEMA_V1.into(),
            run_id: "synthetic-privacy-redteam-001".into(),
            visibility,
            clipping_norm: 0.0,
            gaussian_noise_stddev: 0.0,
            seed: 17,
            minimum_bias_gradient_magnitude: 1.0e-8,
        }
    }

    #[test]
    fn per_record_logistic_updates_reveal_features_and_labels_exactly() {
        let report = evaluate_vm_update_inversion_v1(
            &records(),
            &[0.15, -0.2, 0.3],
            0.1,
            &config(UpdateVisibilityV1::PerRecord),
        )
        .unwrap();
        assert_eq!(
            report.attack_status,
            PrivacyAttackStatusV1::ExactRecordReconstructionDemonstrated
        );
        assert_eq!(report.reconstructed_record_count, 4);
        assert_eq!(report.mean_feature_rmse, Some(0.0));
        assert_eq!(report.label_inference_accuracy, Some(1.0));
        assert!(!report.formal_differential_privacy_claim);
    }

    #[test]
    fn per_example_norm_clipping_does_not_defeat_ratio_inversion() {
        let mut attack = config(UpdateVisibilityV1::PerRecord);
        attack.clipping_norm = 0.05;
        let report =
            evaluate_vm_update_inversion_v1(&records(), &[0.15, -0.2, 0.3], 0.1, &attack).unwrap();
        assert_eq!(
            report.attack_status,
            PrivacyAttackStatusV1::ExactRecordReconstructionDemonstrated
        );
        assert!(report.mean_feature_rmse.unwrap() < 1.0e-12);
    }

    #[test]
    fn gaussian_noise_perturbs_reconstruction_but_is_not_reported_as_dp() {
        let mut attack = config(UpdateVisibilityV1::PerRecord);
        attack.gaussian_noise_stddev = 0.02;
        let first =
            evaluate_vm_update_inversion_v1(&records(), &[0.15, -0.2, 0.3], 0.1, &attack).unwrap();
        let second =
            evaluate_vm_update_inversion_v1(&records(), &[0.15, -0.2, 0.3], 0.1, &attack).unwrap();
        assert_ne!(first.mean_feature_rmse, Some(0.0));
        assert_eq!(first.report_sha256, second.report_sha256);
        assert!(!first.formal_differential_privacy_claim);
    }

    #[test]
    fn aggregate_visibility_does_not_mislabel_unattempted_attack_as_privacy_proof() {
        let batch = evaluate_vm_update_inversion_v1(
            &records(),
            &[0.15, -0.2, 0.3],
            0.1,
            &config(UpdateVisibilityV1::PerClientBatch),
        )
        .unwrap();
        let secure_aggregate = evaluate_vm_update_inversion_v1(
            &records(),
            &[0.15, -0.2, 0.3],
            0.1,
            &config(UpdateVisibilityV1::SecureAggregateOnly),
        )
        .unwrap();
        assert_eq!(batch.observed_update_count, 2);
        assert_eq!(secure_aggregate.observed_update_count, 1);
        assert_eq!(
            batch.attack_status,
            PrivacyAttackStatusV1::PerRecordAttackNotApplicableToAggregateVisibility
        );
        assert_eq!(
            secure_aggregate.attack_status,
            PrivacyAttackStatusV1::PerRecordAttackNotApplicableToAggregateVisibility
        );
        assert_eq!(batch.reconstructed_record_count, 0);
        assert_eq!(secure_aggregate.reconstructed_record_count, 0);
        assert!(!batch.formal_differential_privacy_claim);
        assert!(!secure_aggregate.formal_differential_privacy_claim);
    }

    #[test]
    fn known_auxiliary_records_reveal_one_unknown_in_per_client_batch() {
        let population = records();
        let auxiliary = population[..3].to_vec();
        let report = evaluate_vm_known_auxiliary_batch_inversion_v1(
            &population,
            &auxiliary,
            &[0.15, -0.2, 0.3],
            0.1,
            &config(UpdateVisibilityV1::PerClientBatch),
        )
        .unwrap();

        assert_eq!(report.unknown_record_count, 1);
        assert_eq!(report.reconstructed_record_count, 1);
        assert_eq!(report.reconstructed_record_ids_for_evaluation, vec!["wafer-004"]);
        assert_eq!(report.mean_feature_rmse, Some(0.0));
        assert_eq!(report.label_inference_accuracy, Some(1.0));
        assert!(!report.formal_differential_privacy_claim);
    }

    #[test]
    fn known_auxiliary_records_can_reveal_one_unknown_from_secure_aggregate() {
        let population = records();
        let auxiliary = population[..3].to_vec();
        let report = evaluate_vm_known_auxiliary_batch_inversion_v1(
            &population,
            &auxiliary,
            &[0.15, -0.2, 0.3],
            0.1,
            &config(UpdateVisibilityV1::SecureAggregateOnly),
        )
        .unwrap();

        assert_eq!(report.unknown_record_count, 1);
        assert_eq!(report.reconstructed_record_count, 1);
        assert_eq!(report.reconstructed_record_ids_for_evaluation, vec!["wafer-004"]);
        assert_eq!(report.mean_feature_rmse, Some(0.0));
        assert_eq!(report.label_inference_accuracy, Some(1.0));
    }

    #[test]
    fn auxiliary_attack_requires_exact_membership_and_values() {
        let population = records();
        let mut forged_auxiliary = population[..3].to_vec();
        forged_auxiliary[0].features[0] += 0.01;
        assert!(
            evaluate_vm_known_auxiliary_batch_inversion_v1(
                &population,
                &forged_auxiliary,
                &[0.15, -0.2, 0.3],
                0.1,
                &config(UpdateVisibilityV1::SecureAggregateOnly),
            )
            .is_err()
        );
    }

    #[test]
    fn duplicate_record_identity_and_non_finite_features_are_rejected() {
        let mut duplicated = records();
        duplicated[1].record_id = duplicated[0].record_id.clone();
        assert!(
            evaluate_vm_update_inversion_v1(
                &duplicated,
                &[0.15, -0.2, 0.3],
                0.1,
                &config(UpdateVisibilityV1::PerRecord)
            )
            .is_err()
        );

        let mut non_finite = records();
        non_finite[0].features[0] = f64::NAN;
        assert!(
            evaluate_vm_update_inversion_v1(
                &non_finite,
                &[0.15, -0.2, 0.3],
                0.1,
                &config(UpdateVisibilityV1::PerRecord)
            )
            .is_err()
        );
    }
}
