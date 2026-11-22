// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Train-fold-only SECOM missingness filtering, median imputation and
//! empirical quantile transformation. All reconstructed defaults are explicit.

use crate::benchmark::secom_raw::{SecomLabelRowV1, SecomSensorRowV1};
use crate::benchmark::stratified_repeat::{MANIFEST_SCHEMA_V1, StratifiedRepeatManifestV1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const CONFIG_SCHEMA_V1: &str = "luminous.semi-bench.secom-fold-preprocessor-config.v1";
pub const REPORT_SCHEMA_V1: &str = "luminous.semi-bench.secom-fold-preprocessor-report.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantileOutputV1 {
    Uniform,
    NormalScore,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomPreprocessorConfigV1 {
    pub schema: String,
    pub maximum_training_missing_fraction: f64,
    pub minimum_training_variance: f64,
    pub minimum_distinct_training_values: usize,
    pub quantile_output: QuantileOutputV1,
    pub quantile_tie_policy: String,
    pub emit_missingness_indicator_ablation: bool,
}

impl Default for SecomPreprocessorConfigV1 {
    fn default() -> Self {
        Self {
            schema: CONFIG_SCHEMA_V1.into(),
            maximum_training_missing_fraction: 0.50,
            minimum_training_variance: 1.0e-12,
            minimum_distinct_training_values: 2,
            quantile_output: QuantileOutputV1::NormalScore,
            quantile_tie_policy: "empirical midrank; tails clipped to half-rank".into(),
            emit_missingness_indicator_ablation: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomPreprocessingRowV1 {
    pub sample_id: String,
    pub positive: bool,
    pub values: Vec<Option<f64>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomDroppedFeatureV1 {
    pub feature_index: usize,
    pub feature_id: String,
    pub reason: String,
    pub training_missing_fraction: f64,
    pub training_variance: Option<f64>,
    pub training_distinct_count: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomFittedFeatureV1 {
    pub feature_index: usize,
    pub feature_id: String,
    pub training_missing_fraction: f64,
    pub training_median: f64,
    pub training_variance: f64,
    pub training_distinct_count: usize,
    pub quantile_reference: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedSecomPreprocessorV1 {
    pub schema: String,
    pub config: SecomPreprocessorConfigV1,
    pub input_feature_count: usize,
    pub training_row_count: usize,
    pub training_sample_ids_sha256: String,
    pub retained_features: Vec<SecomFittedFeatureV1>,
    pub dropped_features: Vec<SecomDroppedFeatureV1>,
    pub pipeline_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomTransformedSampleV1 {
    pub sample_id: String,
    pub positive: bool,
    pub feature_set_a: Vec<f64>,
    pub feature_set_b: Option<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomFoldPreprocessorClaimLimitsV1 {
    pub fitting_is_train_only: bool,
    pub held_out_labels_not_used_for_fit: bool,
    pub held_out_features_not_used_for_fit: bool,
    pub column_ids_are_anonymous_not_physical_sensor_names: bool,
    pub preprocessing_defaults_are_reconstruction_not_full_paper_defaults: bool,
    pub feature_set_b_is_a_luminous_ablation: bool,
    pub does_not_establish_sensor_semantics_or_fab_transfer: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecomFoldPreprocessorReportV1 {
    pub schema: String,
    pub dataset_id: String,
    pub dataset_snapshot_sha256: String,
    pub source_identity_manifest_sha256: String,
    pub repeat_index: usize,
    pub seed: u64,
    pub training_rows: usize,
    pub test_rows: usize,
    pub training_failures: usize,
    pub test_failures: usize,
    pub input_feature_count: usize,
    pub retained_feature_count: usize,
    pub dropped_missingness_count: usize,
    pub dropped_low_variance_count: usize,
    pub pipeline_sha256: String,
    pub training_feature_set_a_sha256: String,
    pub test_feature_set_a_sha256: String,
    pub training_feature_set_b_sha256: Option<String>,
    pub test_feature_set_b_sha256: Option<String>,
    pub training_sample_ids_sha256: String,
    pub test_sample_ids_sha256: String,
    pub claim_limits: SecomFoldPreprocessorClaimLimitsV1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedSecomFoldV1 {
    pub fitted_preprocessor: FittedSecomPreprocessorV1,
    pub training: Vec<SecomTransformedSampleV1>,
    pub test: Vec<SecomTransformedSampleV1>,
    pub report: SecomFoldPreprocessorReportV1,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct SecomFoldPreprocessorError(pub String);

fn require_text(value: &str, field: &str) -> Result<(), SecomFoldPreprocessorError> {
    if value.trim().is_empty() {
        Err(SecomFoldPreprocessorError(format!("{field}: expected non-empty string")))
    } else {
        Ok(())
    }
}

fn finite(value: f64, field: &str) -> Result<(), SecomFoldPreprocessorError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SecomFoldPreprocessorError(format!("{field}: expected finite number")))
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
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

fn population_variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
}

fn distinct_count(values: &[f64]) -> usize {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.dedup_by(|a, b| *a == *b);
    sorted.len()
}

fn inverse_standard_normal(p: f64) -> f64 {
    const A: [f64; 6] = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
    const B: [f64; 5] = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
    const C: [f64; 6] = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
    const D: [f64; 4] = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
    let p = p.clamp(1.0e-15, 1.0 - 1.0e-15);
    if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p > 0.97575 {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

fn quantile_value(reference: &[f64], value: f64, output: &QuantileOutputV1) -> f64 {
    let lower = reference.partition_point(|x| *x < value);
    let upper = reference.partition_point(|x| *x <= value);
    let n = reference.len() as f64;
    let p = (((lower + upper) as f64) * 0.5 / n).clamp(0.5 / n, 1.0 - 0.5 / n);
    match output {
        QuantileOutputV1::Uniform => p,
        QuantileOutputV1::NormalScore => inverse_standard_normal(p),
    }
}

fn validate_config(config: &SecomPreprocessorConfigV1) -> Result<(), SecomFoldPreprocessorError> {
    if config.schema != CONFIG_SCHEMA_V1 {
        return Err(SecomFoldPreprocessorError("unsupported preprocessor config schema".into()));
    }
    if !config.maximum_training_missing_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.maximum_training_missing_fraction)
    {
        return Err(SecomFoldPreprocessorError("missingness threshold must be in [0,1]".into()));
    }
    finite(config.minimum_training_variance, "minimum_training_variance")?;
    if config.minimum_training_variance < 0.0 || config.minimum_distinct_training_values < 2 {
        return Err(SecomFoldPreprocessorError("invalid near-zero variance settings".into()));
    }
    require_text(&config.quantile_tie_policy, "quantile_tie_policy")?;
    Ok(())
}

fn ids_digest(rows: &[SecomPreprocessingRowV1]) -> String {
    let mut ids: Vec<&str> = rows.iter().map(|r| r.sample_id.as_str()).collect();
    ids.sort_unstable();
    sha256_hex(&serde_json::to_vec(&ids).expect("sample IDs serialize"))
}

pub fn fit_secom_preprocessor_v1(
    training_rows: &[SecomPreprocessingRowV1],
    config: &SecomPreprocessorConfigV1,
) -> Result<FittedSecomPreprocessorV1, SecomFoldPreprocessorError> {
    validate_config(config)?;
    if training_rows.is_empty() || training_rows[0].values.is_empty() {
        return Err(SecomFoldPreprocessorError("training rows/features must be non-empty".into()));
    }
    let feature_count = training_rows[0].values.len();
    let mut rows = training_rows.to_vec();
    rows.sort_by(|a, b| a.sample_id.cmp(&b.sample_id));
    let mut ids = BTreeSet::new();
    for (ri, row) in rows.iter().enumerate() {
        require_text(&row.sample_id, &format!("training_rows[{ri}].sample_id"))?;
        if !ids.insert(row.sample_id.as_str()) {
            return Err(SecomFoldPreprocessorError(format!("duplicate training sample ID {}", row.sample_id)));
        }
        if row.values.len() != feature_count {
            return Err(SecomFoldPreprocessorError(format!("training row {ri} feature dimension mismatch")));
        }
        for (fi, value) in row.values.iter().enumerate() {
            if let Some(value) = value { finite(*value, &format!("training_rows[{ri}].values[{fi}]"))?; }
        }
    }

    let mut retained = Vec::new();
    let mut dropped = Vec::new();
    for fi in 0..feature_count {
        let observed: Vec<f64> = rows.iter().filter_map(|r| r.values[fi]).collect();
        let missing_fraction = (rows.len() - observed.len()) as f64 / rows.len() as f64;
        if missing_fraction > config.maximum_training_missing_fraction {
            dropped.push(SecomDroppedFeatureV1 {
                feature_index: fi,
                feature_id: format!("secom-anonymous-column-{fi:04}"),
                reason: "training_missing_fraction_exceeds_threshold".into(),
                training_missing_fraction: missing_fraction,
                training_variance: None,
                training_distinct_count: None,
            });
            continue;
        }
        if observed.is_empty() {
            dropped.push(SecomDroppedFeatureV1 {
                feature_index: fi,
                feature_id: format!("secom-anonymous-column-{fi:04}"),
                reason: "no_observed_training_values".into(),
                training_missing_fraction: missing_fraction,
                training_variance: None,
                training_distinct_count: None,
            });
            continue;
        }
        let mut observed_for_median = observed;
        let feature_median = median(&mut observed_for_median);
        let imputed: Vec<f64> = rows.iter().map(|r| r.values[fi].unwrap_or(feature_median)).collect();
        let variance = population_variance(&imputed);
        let distinct = distinct_count(&imputed);
        finite(variance, &format!("feature {fi} training variance"))?;
        if variance <= config.minimum_training_variance || distinct < config.minimum_distinct_training_values {
            dropped.push(SecomDroppedFeatureV1 {
                feature_index: fi,
                feature_id: format!("secom-anonymous-column-{fi:04}"),
                reason: "training_near_zero_variance".into(),
                training_missing_fraction: missing_fraction,
                training_variance: Some(variance),
                training_distinct_count: Some(distinct),
            });
            continue;
        }
        let mut reference = imputed;
        reference.sort_by(f64::total_cmp);
        retained.push(SecomFittedFeatureV1 {
            feature_index: fi,
            feature_id: format!("secom-anonymous-column-{fi:04}"),
            training_missing_fraction: missing_fraction,
            training_median: feature_median,
            training_variance: variance,
            training_distinct_count: distinct,
            quantile_reference: reference,
        });
    }
    if retained.is_empty() {
        return Err(SecomFoldPreprocessorError("all features removed by train-only filtering".into()));
    }

    let mut fitted = FittedSecomPreprocessorV1 {
        schema: "luminous.semi-bench.fitted-secom-preprocessor.v1".into(),
        config: config.clone(),
        input_feature_count: feature_count,
        training_row_count: rows.len(),
        training_sample_ids_sha256: ids_digest(&rows),
        retained_features: retained,
        dropped_features: dropped,
        pipeline_sha256: String::new(),
    };
    let serialized = serde_json::to_vec(&fitted)
        .map_err(|e| SecomFoldPreprocessorError(format!("pipeline serialization failed: {e}")))?;
    fitted.pipeline_sha256 = sha256_hex(&serialized);
    Ok(fitted)
}

pub fn transform_secom_rows_v1(
    fitted: &FittedSecomPreprocessorV1,
    rows: &[SecomPreprocessingRowV1],
) -> Result<Vec<SecomTransformedSampleV1>, SecomFoldPreprocessorError> {
    if rows.is_empty() {
        return Err(SecomFoldPreprocessorError("transform rows must be non-empty".into()));
    }
    let mut ordered = rows.to_vec();
    ordered.sort_by(|a, b| a.sample_id.cmp(&b.sample_id));
    let mut ids = BTreeSet::new();
    let mut output = Vec::with_capacity(ordered.len());
    for (ri, row) in ordered.iter().enumerate() {
        require_text(&row.sample_id, &format!("rows[{ri}].sample_id"))?;
        if !ids.insert(row.sample_id.as_str()) {
            return Err(SecomFoldPreprocessorError(format!("duplicate transform sample ID {}", row.sample_id)));
        }
        if row.values.len() != fitted.input_feature_count {
            return Err(SecomFoldPreprocessorError(format!("transform row {ri} feature dimension mismatch")));
        }
        for (fi, value) in row.values.iter().enumerate() {
            if let Some(value) = value { finite(*value, &format!("rows[{ri}].values[{fi}]"))?; }
        }
        let mut a = Vec::with_capacity(fitted.retained_features.len());
        let mut mask = Vec::with_capacity(fitted.retained_features.len());
        for feature in &fitted.retained_features {
            let raw = row.values[feature.feature_index];
            a.push(quantile_value(
                &feature.quantile_reference,
                raw.unwrap_or(feature.training_median),
                &fitted.config.quantile_output,
            ));
            mask.push(if raw.is_some() { 0.0 } else { 1.0 });
        }
        let b = if fitted.config.emit_missingness_indicator_ablation {
            let missing_ratio = row.values.iter().filter(|v| v.is_none()).count() as f64
                / fitted.input_feature_count as f64;
            let mut joined = a.clone();
            joined.extend(mask);
            joined.push(missing_ratio);
            Some(joined)
        } else {
            None
        };
        output.push(SecomTransformedSampleV1 {
            sample_id: row.sample_id.clone(),
            positive: row.positive,
            feature_set_a: a,
            feature_set_b: b,
        });
    }
    Ok(output)
}

fn digest_matrix(rows: &[SecomTransformedSampleV1], use_b: bool) -> Result<String, SecomFoldPreprocessorError> {
    let payload: Vec<(&str, &[f64])> = rows.iter().map(|row| {
        let values = if use_b { row.feature_set_b.as_deref().unwrap_or(&row.feature_set_a) } else { &row.feature_set_a };
        (row.sample_id.as_str(), values)
    }).collect();
    let bytes = serde_json::to_vec(&payload)
        .map_err(|e| SecomFoldPreprocessorError(format!("matrix serialization failed: {e}")))?;
    Ok(sha256_hex(&bytes))
}

pub fn prepare_secom_stratified_fold_v1(
    sample_ids: &[String],
    features: &[SecomSensorRowV1],
    labels: &[SecomLabelRowV1],
    manifest: &StratifiedRepeatManifestV1,
    config: &SecomPreprocessorConfigV1,
) -> Result<PreparedSecomFoldV1, SecomFoldPreprocessorError> {
    if manifest.schema != MANIFEST_SCHEMA_V1 || !valid_sha256(&manifest.dataset_snapshot_sha256)
        || !valid_sha256(&manifest.source_identity_manifest_sha256)
    {
        return Err(SecomFoldPreprocessorError("invalid stratified manifest schema/digest".into()));
    }
    if sample_ids.len() != features.len() || features.len() != labels.len()
        || manifest.assignments.len() != sample_ids.len()
    {
        return Err(SecomFoldPreprocessorError("source rows and manifest assignments differ in length".into()));
    }
    let mut index_by_id = BTreeMap::new();
    for (index, id) in sample_ids.iter().enumerate() {
        require_text(id, &format!("sample_ids[{index}]"))?;
        if index_by_id.insert(id.as_str(), index).is_some() {
            return Err(SecomFoldPreprocessorError(format!("duplicate source sample ID {id}")));
        }
    }

    let mut assigned = BTreeSet::new();
    let mut training_rows = Vec::new();
    let mut test_rows = Vec::new();
    for assignment in &manifest.assignments {
        if !assigned.insert(assignment.sample_id.as_str()) {
            return Err(SecomFoldPreprocessorError(format!("duplicate manifest sample ID {}", assignment.sample_id)));
        }
        let index = *index_by_id.get(assignment.sample_id.as_str()).ok_or_else(|| {
            SecomFoldPreprocessorError(format!("manifest sample {} missing from source", assignment.sample_id))
        })?;
        if assignment.positive != (labels[index].label == 1) {
            return Err(SecomFoldPreprocessorError(format!("source label mismatch for {}", assignment.sample_id)));
        }
        let row = SecomPreprocessingRowV1 {
            sample_id: assignment.sample_id.clone(),
            positive: assignment.positive,
            values: features[index].values.clone(),
        };
        match assignment.partition.as_str() {
            "train" => training_rows.push(row),
            "test" => test_rows.push(row),
            _ => return Err(SecomFoldPreprocessorError(format!("invalid partition for {}", assignment.sample_id))),
        }
    }
    if assigned.len() != index_by_id.len() || training_rows.is_empty() || test_rows.is_empty() {
        return Err(SecomFoldPreprocessorError("manifest must cover all rows and have train/test partitions".into()));
    }

    let fitted = fit_secom_preprocessor_v1(&training_rows, config)?;
    let training = transform_secom_rows_v1(&fitted, &training_rows)?;
    let test = transform_secom_rows_v1(&fitted, &test_rows)?;
    let train_failures = training.iter().filter(|r| r.positive).count();
    let test_failures = test.iter().filter(|r| r.positive).count();
    let train_ids: Vec<&str> = training.iter().map(|r| r.sample_id.as_str()).collect();
    let test_ids: Vec<&str> = test.iter().map(|r| r.sample_id.as_str()).collect();
    let training_ids_sha = sha256_hex(&serde_json::to_vec(&train_ids).expect("sample ID list serializes"));
    let test_ids_sha = sha256_hex(&serde_json::to_vec(&test_ids).expect("sample ID list serializes"));
    let missing_dropped = fitted.dropped_features.iter().filter(|r| {
        r.reason == "training_missing_fraction_exceeds_threshold" || r.reason == "no_observed_training_values"
    }).count();
    let variance_dropped = fitted.dropped_features.iter().filter(|r| r.reason == "training_near_zero_variance").count();
    let train_a = digest_matrix(&training, false)?;
    let test_a = digest_matrix(&test, false)?;
    let (train_b, test_b) = if fitted.config.emit_missingness_indicator_ablation {
        (Some(digest_matrix(&training, true)?), Some(digest_matrix(&test, true)?))
    } else {
        (None, None)
    };

    let report = SecomFoldPreprocessorReportV1 {
        schema: REPORT_SCHEMA_V1.into(),
        dataset_id: manifest.dataset_id.clone(),
        dataset_snapshot_sha256: manifest.dataset_snapshot_sha256.clone(),
        source_identity_manifest_sha256: manifest.source_identity_manifest_sha256.clone(),
        repeat_index: manifest.repeat_index,
        seed: manifest.seed,
        training_rows: training.len(),
        test_rows: test.len(),
        training_failures: train_failures,
        test_failures,
        input_feature_count: fitted.input_feature_count,
        retained_feature_count: fitted.retained_features.len(),
        dropped_missingness_count: missing_dropped,
        dropped_low_variance_count: variance_dropped,
        pipeline_sha256: fitted.pipeline_sha256.clone(),
        training_feature_set_a_sha256: train_a,
        test_feature_set_a_sha256: test_a,
        training_feature_set_b_sha256: train_b,
        test_feature_set_b_sha256: test_b,
        training_sample_ids_sha256: training_ids_sha,
        test_sample_ids_sha256: test_ids_sha,
        claim_limits: SecomFoldPreprocessorClaimLimitsV1 {
            fitting_is_train_only: true,
            held_out_labels_not_used_for_fit: true,
            held_out_features_not_used_for_fit: true,
            column_ids_are_anonymous_not_physical_sensor_names: true,
            preprocessing_defaults_are_reconstruction_not_full_paper_defaults: true,
            feature_set_b_is_a_luminous_ablation: true,
            does_not_establish_sensor_semantics_or_fab_transfer: true,
        },
    };
    Ok(PreparedSecomFoldV1 { fitted_preprocessor: fitted, training, test, report })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, positive: bool, values: &[Option<f64>]) -> SecomPreprocessingRowV1 {
        SecomPreprocessingRowV1 { sample_id: id.into(), positive, values: values.to_vec() }
    }

    fn config() -> SecomPreprocessorConfigV1 {
        SecomPreprocessorConfigV1 {
            maximum_training_missing_fraction: 0.5,
            minimum_training_variance: 1.0e-12,
            minimum_distinct_training_values: 2,
            quantile_output: QuantileOutputV1::Uniform,
            emit_missingness_indicator_ablation: true,
            ..SecomPreprocessorConfigV1::default()
        }
    }

    #[test]
    fn filtering_is_train_only_and_drops_constant_feature() {
        let training = vec![
            row("a", false, &[Some(1.0), None, Some(5.0), Some(2.0)]),
            row("b", false, &[Some(2.0), None, Some(5.0), Some(2.0)]),
            row("c", true, &[Some(3.0), Some(4.0), Some(5.0), Some(2.0)]),
            row("d", true, &[Some(4.0), Some(6.0), Some(5.0), Some(2.0)]),
        ];
        let fitted = fit_secom_preprocessor_v1(&training, &config()).unwrap();
        assert_eq!(fitted.retained_features.len(), 2);
        assert_eq!(fitted.retained_features[0].feature_index, 0);
        assert_eq!(fitted.retained_features[1].feature_index, 1);
        assert_eq!(fitted.retained_features[1].training_median, 5.0);
        assert_eq!(fitted.dropped_features.len(), 2);
    }

    #[test]
    fn test_only_extremes_do_not_change_fitted_pipeline_digest() {
        let training = vec![
            row("a", false, &[Some(1.0), Some(4.0)]),
            row("b", false, &[Some(2.0), Some(5.0)]),
            row("c", true, &[Some(3.0), Some(6.0)]),
            row("d", true, &[Some(4.0), Some(7.0)]),
        ];
        let fitted = fit_secom_preprocessor_v1(&training, &config()).unwrap();
        let ordinary = vec![row("test", false, &[Some(5.0), Some(8.0)])];
        let extreme = vec![row("test", false, &[Some(1.0e100), Some(-1.0e100)])];
        let _ = transform_secom_rows_v1(&fitted, &ordinary).unwrap();
        let _ = transform_secom_rows_v1(&fitted, &extreme).unwrap();
        assert_eq!(fitted.pipeline_sha256, fit_secom_preprocessor_v1(&training, &config()).unwrap().pipeline_sha256);
    }

    #[test]
    fn training_row_order_does_not_change_fitted_digest() {
        let mut training = vec![
            row("a", false, &[Some(1.0), Some(4.0)]),
            row("b", false, &[Some(2.0), None]),
            row("c", true, &[Some(3.0), Some(6.0)]),
            row("d", true, &[Some(4.0), Some(7.0)]),
        ];
        let first = fit_secom_preprocessor_v1(&training, &config()).unwrap();
        training.reverse();
        let second = fit_secom_preprocessor_v1(&training, &config()).unwrap();
        assert_eq!(first.pipeline_sha256, second.pipeline_sha256);
    }

    #[test]
    fn exactly_half_missing_is_kept() {
        let training = vec![
            row("a", false, &[Some(1.0), None]),
            row("b", false, &[Some(2.0), None]),
            row("c", true, &[Some(3.0), Some(5.0)]),
            row("d", true, &[Some(4.0), Some(6.0)]),
        ];
        let fitted = fit_secom_preprocessor_v1(&training, &config()).unwrap();
        assert_eq!(fitted.retained_features.len(), 2);
        assert_eq!(fitted.retained_features[1].training_missing_fraction, 0.5);
    }

    #[test]
    fn all_missing_and_zero_variance_features_are_dropped() {
        let training = vec![
            row("a", false, &[Some(1.0), None, Some(0.0)]),
            row("b", false, &[Some(2.0), None, Some(0.0)]),
            row("c", true, &[Some(3.0), None, Some(0.0)]),
            row("d", true, &[Some(4.0), None, Some(0.0)]),
        ];
        let fitted = fit_secom_preprocessor_v1(&training, &config()).unwrap();
        assert_eq!(fitted.retained_features.len(), 1);
        assert_eq!(fitted.dropped_features.len(), 2);
    }

    #[test]
    fn normal_score_transform_stays_finite_for_extreme_test_value() {
        let training = vec![
            row("a", false, &[Some(1.0)]),
            row("b", false, &[Some(2.0)]),
            row("c", true, &[Some(3.0)]),
            row("d", true, &[Some(4.0)]),
        ];
        let fitted = fit_secom_preprocessor_v1(&training, &SecomPreprocessorConfigV1 {
            quantile_output: QuantileOutputV1::NormalScore,
            ..config()
        }).unwrap();
        let test = vec![row("test", false, &[Some(1.0e12)])];
        let transformed = transform_secom_rows_v1(&fitted, &test).unwrap();
        assert!(transformed[0].feature_set_a[0].is_finite());
        assert!(transformed[0].feature_set_a[0] > 0.0);
    }

    #[test]
    fn feature_set_b_is_values_plus_mask_plus_global_missing_ratio() {
        let training = vec![
            row("a", false, &[Some(1.0), None]),
            row("b", false, &[Some(2.0), Some(3.0)]),
            row("c", true, &[Some(3.0), Some(4.0)]),
            row("d", true, &[Some(4.0), Some(5.0)]),
        ];
        let fitted = fit_secom_preprocessor_v1(&training, &config()).unwrap();
        let transformed = transform_secom_rows_v1(&fitted, &training[..1]).unwrap();
        let b = transformed[0].feature_set_b.as_ref().unwrap();
        assert_eq!(b.len(), 5);
        assert_eq!(&b[2..4], &[0.0, 1.0]);
        assert_eq!(b[4], 0.5);
    }

    #[test]
    fn duplicate_ids_and_invalid_dimensions_reject() {
        let duplicate = vec![row("a", false, &[Some(1.0)]), row("a", true, &[Some(2.0)])];
        assert!(fit_secom_preprocessor_v1(&duplicate, &config()).is_err());
        let mismatched = vec![row("a", false, &[Some(1.0)]), row("b", true, &[Some(2.0), Some(3.0)])];
        assert!(fit_secom_preprocessor_v1(&mismatched, &config()).is_err());
    }
}
