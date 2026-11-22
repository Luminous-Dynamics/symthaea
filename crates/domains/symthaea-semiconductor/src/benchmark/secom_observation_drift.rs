// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unsupervised observation-regime drift diagnostics for SECOM.
//!
//! The diagnostic compares only missingness masks and row-level missing counts.
//! It never uses pass/fail labels to compute drift and never changes model
//! parameters. It is a monitoring signal, not proof of process drift or cause.

use crate::benchmark::secom_raw::{
    SecomRawDataError, SecomRawExpectationsV1, audit_secom_raw_v1, parse_features, parse_labels,
};
use crate::benchmark::split_manifest::{SplitManifestV1, validate_split_manifest_v1};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub const OUTPUT_SCHEMA_V1: &str = "luminous.semi-bench.secom-observation-drift-report.v1";
const HELD_OUT_SPLITS: [&str; 2] = ["validation", "test"];
const FEATURE_RATE_ALERT_DELTA: f64 = 0.05;
const LN_2: f64 = std::f64::consts::LN_2;

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct SecomObservationDriftError(pub String);

impl From<SecomRawDataError> for SecomObservationDriftError {
    fn from(error: SecomRawDataError) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomObservationProfileV1 {
    pub rows: usize,
    pub feature_count: usize,
    pub missing_cells: usize,
    pub sensor_cells: usize,
    pub missing_fraction: f64,
    pub mean_missing_features_per_row: f64,
    pub missing_fraction_by_feature: Vec<f64>,
    /// Counts for rows with exactly 0, 1, ..., feature_count missing values.
    pub row_missing_count_histogram: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomObservationDriftV1 {
    pub reference_split: String,
    pub compared_split: String,
    pub reference_missing_fraction: f64,
    pub compared_missing_fraction: f64,
    /// Compared minus reference, in absolute probability units.
    pub pooled_missing_fraction_delta: f64,
    pub mean_absolute_feature_rate_delta: f64,
    pub maximum_absolute_feature_rate_delta: f64,
    /// Jensen-Shannon divergence of per-feature Bernoulli mask marginals.
    /// Values are in [0, 1] bits; this is marginal, not joint-mask divergence.
    pub mean_feature_js_divergence_bits: f64,
    pub maximum_feature_js_divergence_bits: f64,
    pub features_with_rate_delta_at_least_5pp: usize,
    pub mean_missing_features_per_row_delta: f64,
    /// Jensen-Shannon divergence of the row missing-count histogram.
    pub row_missing_count_js_divergence_bits: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomObservationDriftClaimLimitsV1 {
    pub drift_uses_missingness_mask_only: bool,
    pub drift_does_not_use_labels: bool,
    pub does_not_establish_process_drift_or_root_cause: bool,
    pub does_not_establish_sensor_fault_or_calibration_failure: bool,
    pub does_not_establish_predictive_performance_or_fab_transfer: bool,
    pub source_mirror_official_uci_byte_identity_unverified: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomObservationDriftReportV1 {
    pub schema: String,
    pub dataset_id: String,
    pub feature_file_sha256: String,
    pub label_file_sha256: String,
    pub dataset_snapshot_sha256: String,
    pub split_manifest_sha256: String,
    pub split_scheme: String,
    pub clock_domain_id: Option<String>,
    pub feature_count: usize,
    pub training_profile: SecomObservationProfileV1,
    pub held_out_profiles: BTreeMap<String, SecomObservationProfileV1>,
    pub drift_vs_training: BTreeMap<String, SecomObservationDriftV1>,
    pub feature_rate_alert_delta: f64,
    pub claim_limits: SecomObservationDriftClaimLimitsV1,
}

fn rounded(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}

fn binary_entropy(probability: f64) -> f64 {
    let term = |value: f64| {
        if value == 0.0 {
            0.0
        } else {
            -value * value.ln()
        }
    };
    term(probability) + term(1.0 - probability)
}

fn bernoulli_js_divergence_bits(left: f64, right: f64) -> f64 {
    let midpoint = 0.5 * (left + right);
    ((binary_entropy(midpoint) - 0.5 * binary_entropy(left) - 0.5 * binary_entropy(right)) / LN_2)
        .clamp(0.0, 1.0)
}

fn discrete_js_divergence_bits(left_counts: &[usize], right_counts: &[usize]) -> f64 {
    debug_assert_eq!(left_counts.len(), right_counts.len());
    let left_total: usize = left_counts.iter().sum();
    let right_total: usize = right_counts.iter().sum();
    if left_total == 0 || right_total == 0 {
        return 0.0;
    }
    let mut divergence = 0.0;
    for index in 0..left_counts.len() {
        let left = left_counts[index] as f64 / left_total as f64;
        let right = right_counts[index] as f64 / right_total as f64;
        let midpoint = 0.5 * (left + right);
        if left > 0.0 {
            divergence += 0.5 * left * (left / midpoint).ln();
        }
        if right > 0.0 {
            divergence += 0.5 * right * (right / midpoint).ln();
        }
    }
    (divergence / LN_2).clamp(0.0, 1.0)
}

fn observation_profile(
    indices: &[usize],
    features: &[crate::benchmark::secom_raw::SecomSensorRowV1],
) -> Result<SecomObservationProfileV1, SecomObservationDriftError> {
    if indices.is_empty() {
        return Err(SecomObservationDriftError(
            "observation profile: split must contain at least one row".into(),
        ));
    }
    let feature_count = features[indices[0]].values.len();
    if feature_count == 0 {
        return Err(SecomObservationDriftError(
            "observation profile: expected at least one sensor feature".into(),
        ));
    }

    let mut missing_by_feature = vec![0usize; feature_count];
    let mut row_missing_count_histogram = vec![0usize; feature_count + 1];
    let mut missing_cells = 0usize;
    for &row_index in indices {
        let row = &features[row_index];
        if row.values.len() != feature_count {
            return Err(SecomObservationDriftError(format!(
                "observation profile: row {row_index} has inconsistent feature count"
            )));
        }
        let mut row_missing = 0usize;
        for (feature_index, value) in row.values.iter().enumerate() {
            if value.is_none() {
                missing_by_feature[feature_index] += 1;
                row_missing += 1;
            }
        }
        missing_cells += row_missing;
        row_missing_count_histogram[row_missing] += 1;
    }

    let rows = indices.len();
    let sensor_cells = rows * feature_count;
    Ok(SecomObservationProfileV1 {
        rows,
        feature_count,
        missing_cells,
        sensor_cells,
        missing_fraction: rounded(missing_cells as f64 / sensor_cells as f64),
        mean_missing_features_per_row: rounded(missing_cells as f64 / rows as f64),
        missing_fraction_by_feature: missing_by_feature
            .into_iter()
            .map(|count| rounded(count as f64 / rows as f64))
            .collect(),
        row_missing_count_histogram,
    })
}

fn compare_profiles(
    reference_split: &str,
    compared_split: &str,
    reference: &SecomObservationProfileV1,
    compared: &SecomObservationProfileV1,
) -> Result<SecomObservationDriftV1, SecomObservationDriftError> {
    if reference.feature_count != compared.feature_count {
        return Err(SecomObservationDriftError(
            "observation drift: feature counts differ".into(),
        ));
    }
    let mut absolute_deltas = Vec::with_capacity(reference.feature_count);
    let mut feature_js = Vec::with_capacity(reference.feature_count);
    let mut features_over_alert = 0usize;
    for (left, right) in reference
        .missing_fraction_by_feature
        .iter()
        .zip(&compared.missing_fraction_by_feature)
    {
        let delta = (right - left).abs();
        absolute_deltas.push(delta);
        feature_js.push(bernoulli_js_divergence_bits(*left, *right));
        if delta >= FEATURE_RATE_ALERT_DELTA {
            features_over_alert += 1;
        }
    }
    let mean_absolute_feature_rate_delta =
        absolute_deltas.iter().sum::<f64>() / absolute_deltas.len() as f64;
    let mean_feature_js_divergence_bits = feature_js.iter().sum::<f64>() / feature_js.len() as f64;
    Ok(SecomObservationDriftV1 {
        reference_split: reference_split.into(),
        compared_split: compared_split.into(),
        reference_missing_fraction: reference.missing_fraction,
        compared_missing_fraction: compared.missing_fraction,
        pooled_missing_fraction_delta: rounded(
            compared.missing_fraction - reference.missing_fraction,
        ),
        mean_absolute_feature_rate_delta: rounded(mean_absolute_feature_rate_delta),
        maximum_absolute_feature_rate_delta: rounded(
            absolute_deltas.iter().copied().fold(0.0, f64::max),
        ),
        mean_feature_js_divergence_bits: rounded(mean_feature_js_divergence_bits),
        maximum_feature_js_divergence_bits: rounded(feature_js.iter().copied().fold(0.0, f64::max)),
        features_with_rate_delta_at_least_5pp: features_over_alert,
        mean_missing_features_per_row_delta: rounded(
            compared.mean_missing_features_per_row - reference.mean_missing_features_per_row,
        ),
        row_missing_count_js_divergence_bits: rounded(discrete_js_divergence_bits(
            &reference.row_missing_count_histogram,
            &compared.row_missing_count_histogram,
        )),
    })
}

/// Compares validation and test observation masks with the training mask.
///
/// Raw data digests, source labels, split assignment labels and source-clock
/// alignment are checked before calculating the unsupervised drift metrics.
/// Labels are not consumed by any drift statistic.
pub fn evaluate_secom_observation_drift_v1(
    feature_bytes: &[u8],
    label_bytes: &[u8],
    expectations: &SecomRawExpectationsV1,
    split_manifest: &SplitManifestV1,
) -> Result<SecomObservationDriftReportV1, SecomObservationDriftError> {
    let raw_report = audit_secom_raw_v1(feature_bytes, label_bytes, expectations)?;
    if split_manifest.dataset_id != expectations.dataset_id {
        return Err(SecomObservationDriftError(
            "split_manifest.dataset_id does not match raw expectations".into(),
        ));
    }
    let split_report = validate_split_manifest_v1(split_manifest)
        .map_err(|error| SecomObservationDriftError(error.to_string()))?;
    if split_manifest.assignments.len() != raw_report.rows {
        return Err(SecomObservationDriftError(format!(
            "split assignments: expected {} rows, observed {}",
            raw_report.rows,
            split_manifest.assignments.len()
        )));
    }
    if split_manifest.split_scheme == "forward_time"
        && split_manifest.clock_domain_id.as_deref() != Some(expectations.clock_domain_id.as_str())
    {
        return Err(SecomObservationDriftError(
            "forward-time split clock_domain_id does not match raw expectations".into(),
        ));
    }

    let features = parse_features(feature_bytes)?;
    let labels = parse_labels(label_bytes)?;
    for (index, (assignment, label)) in split_manifest.assignments.iter().zip(&labels).enumerate() {
        if assignment.positive != (label.label == 1) {
            return Err(SecomObservationDriftError(format!(
                "split assignment {index} label does not match source label row"
            )));
        }
        if let Some(observed_at) = assignment.observed_at.as_deref() {
            let source_time = label.observed_at.format("%Y-%m-%dT%H:%M:%S").to_string();
            if observed_at != source_time {
                return Err(SecomObservationDriftError(format!(
                    "split assignment {index} observed_at does not match source timestamp"
                )));
            }
        }
    }

    let indices_for = |split: &str| {
        split_manifest
            .assignments
            .iter()
            .enumerate()
            .filter_map(|(index, assignment)| (assignment.split == split).then_some(index))
            .collect::<Vec<_>>()
    };

    let training_indices = indices_for("train");
    let training_profile = observation_profile(&training_indices, &features)?;
    let mut held_out_profiles = BTreeMap::new();
    let mut drift_vs_training = BTreeMap::new();
    for split in HELD_OUT_SPLITS {
        let indices = indices_for(split);
        let profile = observation_profile(&indices, &features)?;
        let drift = compare_profiles("train", split, &training_profile, &profile)?;
        held_out_profiles.insert(split.into(), profile);
        drift_vs_training.insert(split.into(), drift);
    }

    Ok(SecomObservationDriftReportV1 {
        schema: OUTPUT_SCHEMA_V1.into(),
        dataset_id: expectations.dataset_id.clone(),
        feature_file_sha256: raw_report.feature_sha256,
        label_file_sha256: raw_report.label_sha256,
        dataset_snapshot_sha256: split_manifest.dataset_snapshot_sha256.clone(),
        split_manifest_sha256: split_report.split_manifest_sha256,
        split_scheme: split_manifest.split_scheme.clone(),
        clock_domain_id: split_manifest.clock_domain_id.clone(),
        feature_count: training_profile.feature_count,
        training_profile,
        held_out_profiles,
        drift_vs_training,
        feature_rate_alert_delta: FEATURE_RATE_ALERT_DELTA,
        claim_limits: SecomObservationDriftClaimLimitsV1 {
            drift_uses_missingness_mask_only: true,
            drift_does_not_use_labels: true,
            does_not_establish_process_drift_or_root_cause: true,
            does_not_establish_sensor_fault_or_calibration_failure: true,
            does_not_establish_predictive_performance_or_fab_transfer: true,
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

    fn subject(
        feature_bytes: &[u8],
        label_bytes: &[u8],
    ) -> (SecomRawExpectationsV1, SplitManifestV1) {
        let mut expectations: SecomRawExpectationsV1 =
            serde_json::from_str(EXPECTATIONS).expect("expectations fixture parses");
        expectations.expected_feature_sha256 = format!("{:x}", Sha256::digest(feature_bytes));
        expectations.expected_label_sha256 = format!("{:x}", Sha256::digest(label_bytes));
        let split = serde_json::from_str(SPLIT).expect("split fixture parses");
        (expectations, split)
    }

    fn evaluate(feature_bytes: &[u8], label_bytes: &[u8]) -> SecomObservationDriftReportV1 {
        let (expectations, mut split) = subject(feature_bytes, label_bytes);
        let parsed_labels = parse_labels(label_bytes).expect("synthetic labels parse");
        for (assignment, label) in split.assignments.iter_mut().zip(&parsed_labels) {
            assignment.positive = label.label == 1;
        }
        evaluate_secom_observation_drift_v1(feature_bytes, label_bytes, &expectations, &split)
            .expect("synthetic fixture evaluates")
    }

    #[test]
    fn identical_train_and_held_out_mask_marginals_have_zero_drift() {
        let report = evaluate(FEATURES, LABELS);
        for split in HELD_OUT_SPLITS {
            let drift = &report.drift_vs_training[split];
            assert_eq!(drift.pooled_missing_fraction_delta, 0.0);
            assert_eq!(drift.mean_absolute_feature_rate_delta, 0.0);
            assert_eq!(drift.maximum_absolute_feature_rate_delta, 0.0);
            assert_eq!(drift.mean_feature_js_divergence_bits, 0.0);
            assert_eq!(drift.maximum_feature_js_divergence_bits, 0.0);
            assert_eq!(drift.features_with_rate_delta_at_least_5pp, 0);
            assert_eq!(drift.row_missing_count_js_divergence_bits, 0.0);
        }
    }

    #[test]
    fn one_sensor_shift_is_detected_even_when_labels_are_unchanged() {
        let shifted = FEATURES.split(|byte| *byte == b'\n').collect::<Vec<_>>();
        assert_eq!(shifted.len(), 9);
        let mut modified = shifted.iter().map(|row| row.to_vec()).collect::<Vec<_>>();
        modified[4] = b"NaN 2.2".to_vec();
        let mut bytes = Vec::new();
        for (index, row) in modified.iter().enumerate() {
            bytes.extend_from_slice(row);
            if index + 1 < modified.len() {
                bytes.push(b'\n');
            }
        }
        let report = evaluate(&bytes, LABELS);
        let validation = &report.drift_vs_training["validation"];
        assert!(validation.mean_absolute_feature_rate_delta > 0.0);
        assert!(validation.maximum_absolute_feature_rate_delta >= 0.5);
        assert!(validation.mean_feature_js_divergence_bits > 0.0);
        assert!(validation.row_missing_count_js_divergence_bits > 0.0);
    }

    #[test]
    fn labels_do_not_change_observation_drift_statistics() {
        let changed_labels = String::from_utf8(LABELS.to_vec())
            .unwrap()
            .replace("-1 \"19/07/2008 12:15:00\"", "1 \"19/07/2008 12:15:00\"");
        let original = evaluate(FEATURES, LABELS);
        let changed = evaluate(FEATURES, changed_labels.as_bytes());
        assert_eq!(original.training_profile, changed.training_profile);
        assert_eq!(original.held_out_profiles, changed.held_out_profiles);
        assert_eq!(original.drift_vs_training, changed.drift_vs_training);
    }

    #[test]
    fn feature_permutation_preserves_aggregate_drift_statistics() {
        let permuted = std::str::from_utf8(FEATURES)
            .unwrap()
            .lines()
            .map(|row| {
                let mut cells: Vec<&str> = row.split_whitespace().collect();
                cells.reverse();
                format!("{} {}", cells[0], cells[1])
            })
            .collect::<Vec<_>>()
            .join("\n");
        let original = evaluate(FEATURES, LABELS);
        let changed = evaluate(permuted.as_bytes(), LABELS);
        for split in HELD_OUT_SPLITS {
            assert_eq!(
                original.drift_vs_training[split].mean_absolute_feature_rate_delta,
                changed.drift_vs_training[split].mean_absolute_feature_rate_delta
            );
            assert_eq!(
                original.drift_vs_training[split].maximum_absolute_feature_rate_delta,
                changed.drift_vs_training[split].maximum_absolute_feature_rate_delta
            );
            assert_eq!(
                original.drift_vs_training[split].row_missing_count_js_divergence_bits,
                changed.drift_vs_training[split].row_missing_count_js_divergence_bits
            );
        }
    }

    #[test]
    fn divergence_functions_are_symmetric_bounded_and_handle_extremes() {
        for (left, right) in [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.25, 0.75)] {
            let forward = bernoulli_js_divergence_bits(left, right);
            let reverse = bernoulli_js_divergence_bits(right, left);
            assert!((forward - reverse).abs() < 1e-12);
            assert!((0.0..=1.0).contains(&forward));
        }
        assert_eq!(bernoulli_js_divergence_bits(0.0, 1.0), 1.0);
        assert_eq!(discrete_js_divergence_bits(&[1, 0], &[0, 1]), 1.0);
        assert_eq!(discrete_js_divergence_bits(&[2, 0], &[2, 0]), 0.0);
    }
}
