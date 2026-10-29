// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paired, cluster-bootstrap inference for aviation forecast comparisons.
//!
//! The comparison is only defined for the exact same scored records, targets,
//! target windows, decision times, data snapshot, and evaluation protocol.
//! Dependence clusters (e.g. airport-day or disruption episode) are resampled
//! as blocks to avoid treating every flight row as independent.
//!
//! Percentile intervals are descriptive cluster-bootstrap uncertainty under
//! exchangeable clusters. They do not establish causal policy value, protect
//! against arbitrary temporal drift, or replace a locked time-held-out test.

use crate::aviation_calibration_protocol::EvaluationPartition;
use crate::aviation_forecast_metrics::{
    evaluate_forecasts, ForecastEvaluationContext, ForecastEvaluationMode,
    ForecastObservation, ForecastScoringError,
};
use std::collections::{BTreeMap, BTreeSet};

/// Which forecast policy failed validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastPolicySide {
    Baseline,
    Candidate,
}

/// Preregistered deterministic cluster-bootstrap settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForecastComparisonOptions {
    /// Number of cluster-resampling replicates. Must be at least 100.
    pub bootstrap_replicates: usize,
    /// Central confidence level, for example 0.95.
    pub confidence_level: f64,
    /// Public deterministic seed. It is part of the reproducibility record.
    pub seed: u64,
}

/// Why the candidate and baseline are not comparable under the declared task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForecastComparisonError {
    InvalidBootstrapReplicateCount { requested: usize },
    InvalidConfidenceLevel,
    BaselineScoringFailed { error: ForecastScoringError },
    CandidateScoringFailed { error: ForecastScoringError },
    DatasetSnapshotMismatch,
    EvaluationProtocolMismatch,
    EvaluatorCommitMismatch,
    EvaluationModeMismatch,
    EvaluationPartitionMismatch,
    RecordSetLengthMismatch { baseline: usize, candidate: usize },
    RecordMissingFromBaseline { record_id: String },
    RecordMissingFromCandidate { record_id: String },
    TargetOrTimingMismatch { record_id: String, field: &'static str },
    ClusterMismatch { record_id: String },
    CohortMismatch { record_id: String },
    InsufficientClusters { actual: usize, minimum: usize },
    NonFiniteMetric,
}

/// Paired baseline-versus-candidate point metrics and cluster-bootstrap intervals.
///
/// Differences are candidate minus baseline, so negative values favor the
/// candidate for RMSE and MAE. The intervals describe uncertainty in predictive
/// error differences on this scored partition, not uncertainty about airline
/// cost savings or intervention effects.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastComparisonReport {
    pub baseline_run_id: String,
    pub candidate_run_id: String,
    pub data_snapshot_sha256: String,
    pub evaluation_protocol_sha256: String,
    pub evaluator_source_commit_sha: String,
    pub evaluation_mode: ForecastEvaluationMode,
    pub partition: EvaluationPartition,
    pub observation_count: usize,
    pub event_cluster_count: usize,
    pub baseline_rmse_seconds: f64,
    pub candidate_rmse_seconds: f64,
    pub rmse_delta_seconds: f64,
    pub rmse_delta_ci_lower_seconds: f64,
    pub rmse_delta_ci_upper_seconds: f64,
    pub baseline_mae_seconds: f64,
    pub candidate_mae_seconds: f64,
    pub mae_delta_seconds: f64,
    pub mae_delta_ci_lower_seconds: f64,
    pub mae_delta_ci_upper_seconds: f64,
    /// Fraction of bootstrap replicates where candidate RMSE is strictly lower.
    /// This is a descriptive bootstrap fraction, not a p-value.
    pub bootstrap_fraction_candidate_rmse_better: f64,
    pub bootstrap_replicates: usize,
    pub confidence_level: f64,
    pub seed: u64,
}

#[derive(Debug, Clone)]
struct PairedRow {
    cluster_id: String,
    baseline_error: f64,
    candidate_error: f64,
}

/// Compare two forecasting systems on identical records and bootstrap whole
/// event clusters rather than individual flights.
///
/// Baseline/candidate feature schemas and model artifacts may differ, but their
/// source snapshot, evaluation protocol, evaluator commit, evaluation mode,
/// record IDs, targets, target windows, decision times, cohorts, and cluster IDs
/// must match. The method refuses row-level comparisons with missing/misaligned
/// labels rather than silently dropping observations.
pub fn compare_forecasts_cluster_bootstrap(
    baseline_context: &ForecastEvaluationContext,
    baseline: &[ForecastObservation],
    candidate_context: &ForecastEvaluationContext,
    candidate: &[ForecastObservation],
    mode: ForecastEvaluationMode,
    options: ForecastComparisonOptions,
) -> Result<ForecastComparisonReport, ForecastComparisonError> {
    if options.bootstrap_replicates < 100 {
        return Err(ForecastComparisonError::InvalidBootstrapReplicateCount {
            requested: options.bootstrap_replicates,
        });
    }
    if !options.confidence_level.is_finite()
        || options.confidence_level <= 0.0
        || options.confidence_level >= 1.0
    {
        return Err(ForecastComparisonError::InvalidConfidenceLevel);
    }

    let baseline_score = evaluate_forecasts(baseline_context, baseline, mode)
        .map_err(|error| ForecastComparisonError::BaselineScoringFailed { error })?;
    let candidate_score = evaluate_forecasts(candidate_context, candidate, mode)
        .map_err(|error| ForecastComparisonError::CandidateScoringFailed { error })?;

    if baseline_context.data_snapshot_sha256 != candidate_context.data_snapshot_sha256 {
        return Err(ForecastComparisonError::DatasetSnapshotMismatch);
    }
    if baseline_context.evaluation_protocol_sha256 != candidate_context.evaluation_protocol_sha256 {
        return Err(ForecastComparisonError::EvaluationProtocolMismatch);
    }
    if baseline_context.evaluator_source_commit_sha != candidate_context.evaluator_source_commit_sha {
        return Err(ForecastComparisonError::EvaluatorCommitMismatch);
    }
    if baseline_score.mode != candidate_score.mode || baseline_score.mode != mode {
        return Err(ForecastComparisonError::EvaluationModeMismatch);
    }
    if baseline_score.partition != candidate_score.partition {
        return Err(ForecastComparisonError::EvaluationPartitionMismatch);
    }
    if baseline.len() != candidate.len() {
        return Err(ForecastComparisonError::RecordSetLengthMismatch {
            baseline: baseline.len(),
            candidate: candidate.len(),
        });
    }

    let baseline_by_id: BTreeMap<&str, &ForecastObservation> = baseline
        .iter()
        .map(|row| (row.record_id.as_str(), row))
        .collect();
    let candidate_by_id: BTreeMap<&str, &ForecastObservation> = candidate
        .iter()
        .map(|row| (row.record_id.as_str(), row))
        .collect();

    for record_id in baseline_by_id.keys() {
        if !candidate_by_id.contains_key(record_id) {
            return Err(ForecastComparisonError::RecordMissingFromCandidate {
                record_id: (*record_id).into(),
            });
        }
    }
    for record_id in candidate_by_id.keys() {
        if !baseline_by_id.contains_key(record_id) {
            return Err(ForecastComparisonError::RecordMissingFromBaseline {
                record_id: (*record_id).into(),
            });
        }
    }

    // BTreeMap iteration fixes row ordering independently of input order.
    let mut paired_rows = Vec::with_capacity(baseline.len());
    let mut clusters = BTreeSet::<String>::new();
    for (record_id, base) in &baseline_by_id {
        let cand = candidate_by_id.get(*record_id).copied().ok_or_else(|| {
            ForecastComparisonError::RecordMissingFromCandidate {
                record_id: (*record_id).into(),
            }
        })?;
        for (field, left, right) in [
            ("decision_time_epoch_seconds", base.decision_time_epoch_seconds, cand.decision_time_epoch_seconds),
            ("target_window_start_epoch_seconds", base.target_window_start_epoch_seconds, cand.target_window_start_epoch_seconds),
            ("target_window_end_epoch_seconds", base.target_window_end_epoch_seconds, cand.target_window_end_epoch_seconds),
            ("target_observed_at_epoch_seconds", base.target_observed_at_epoch_seconds, cand.target_observed_at_epoch_seconds),
        ] {
            if left != right {
                return Err(ForecastComparisonError::TargetOrTimingMismatch {
                    record_id: (*record_id).into(),
                    field,
                });
            }
        }
        if base.actual_seconds.to_bits() != cand.actual_seconds.to_bits() {
            return Err(ForecastComparisonError::TargetOrTimingMismatch {
                record_id: (*record_id).into(),
                field: "actual_seconds",
            });
        }
        if base.event_cluster_id != cand.event_cluster_id {
            return Err(ForecastComparisonError::ClusterMismatch {
                record_id: (*record_id).into(),
            });
        }
        if base.cohort_id != cand.cohort_id {
            return Err(ForecastComparisonError::CohortMismatch {
                record_id: (*record_id).into(),
            });
        }
        clusters.insert(base.event_cluster_id.clone());
        paired_rows.push(PairedRow {
            cluster_id: base.event_cluster_id.clone(),
            baseline_error: base.predicted_seconds - base.actual_seconds,
            candidate_error: cand.predicted_seconds - cand.actual_seconds,
        });
    }

    if clusters.len() < 2 {
        return Err(ForecastComparisonError::InsufficientClusters {
            actual: clusters.len(),
            minimum: 2,
        });
    }

    let baseline_rmse = baseline_score.overall.rmse_seconds;
    let candidate_rmse = candidate_score.overall.rmse_seconds;
    let baseline_mae = baseline_score.overall.mae_seconds;
    let candidate_mae = candidate_score.overall.mae_seconds;
    let rmse_delta = candidate_rmse - baseline_rmse;
    let mae_delta = candidate_mae - baseline_mae;

    let mut rows_by_cluster = BTreeMap::<String, Vec<usize>>::new();
    for (index, row) in paired_rows.iter().enumerate() {
        rows_by_cluster.entry(row.cluster_id.clone()).or_default().push(index);
    }
    let cluster_ids: Vec<String> = rows_by_cluster.keys().cloned().collect();
    let mut rng = SplitMix64::new(options.seed);
    let mut rmse_deltas = Vec::with_capacity(options.bootstrap_replicates);
    let mut mae_deltas = Vec::with_capacity(options.bootstrap_replicates);
    let mut candidate_rmse_better = 0_usize;

    for _ in 0..options.bootstrap_replicates {
        let mut sampled_indices = Vec::with_capacity(paired_rows.len());
        for _ in 0..cluster_ids.len() {
            let selected = rng.bounded_index(cluster_ids.len());
            sampled_indices.extend(rows_by_cluster[&cluster_ids[selected]].iter().copied());
        }
        if sampled_indices.is_empty() {
            return Err(ForecastComparisonError::NonFiniteMetric);
        }

        let mut baseline_squared = 0.0;
        let mut candidate_squared = 0.0;
        let mut baseline_absolute = 0.0;
        let mut candidate_absolute = 0.0;
        for index in &sampled_indices {
            let row = &paired_rows[*index];
            baseline_squared += row.baseline_error * row.baseline_error;
            candidate_squared += row.candidate_error * row.candidate_error;
            baseline_absolute += row.baseline_error.abs();
            candidate_absolute += row.candidate_error.abs();
        }
        let count = sampled_indices.len() as f64;
        let base_rmse = (baseline_squared / count).sqrt();
        let candidate_rmse = (candidate_squared / count).sqrt();
        let base_mae = baseline_absolute / count;
        let candidate_mae = candidate_absolute / count;
        let rmse_delta = candidate_rmse - base_rmse;
        let mae_delta = candidate_mae - base_mae;
        if !rmse_delta.is_finite() || !mae_delta.is_finite() {
            return Err(ForecastComparisonError::NonFiniteMetric);
        }
        if rmse_delta < 0.0 {
            candidate_rmse_better += 1;
        }
        rmse_deltas.push(rmse_delta);
        mae_deltas.push(mae_delta);
    }
    rmse_deltas.sort_by(f64::total_cmp);
    mae_deltas.sort_by(f64::total_cmp);
    let alpha = 1.0 - options.confidence_level;
    let lower_q = alpha / 2.0;
    let upper_q = 1.0 - lower_q;
    let rmse_ci_lower = linear_quantile_sorted(&rmse_deltas, lower_q)
        .ok_or(ForecastComparisonError::NonFiniteMetric)?;
    let rmse_ci_upper = linear_quantile_sorted(&rmse_deltas, upper_q)
        .ok_or(ForecastComparisonError::NonFiniteMetric)?;
    let mae_ci_lower = linear_quantile_sorted(&mae_deltas, lower_q)
        .ok_or(ForecastComparisonError::NonFiniteMetric)?;
    let mae_ci_upper = linear_quantile_sorted(&mae_deltas, upper_q)
        .ok_or(ForecastComparisonError::NonFiniteMetric)?;

    Ok(ForecastComparisonReport {
        baseline_run_id: baseline_context.run_id.clone(),
        candidate_run_id: candidate_context.run_id.clone(),
        data_snapshot_sha256: baseline_context.data_snapshot_sha256.clone(),
        evaluation_protocol_sha256: baseline_context.evaluation_protocol_sha256.clone(),
        evaluator_source_commit_sha: baseline_context.evaluator_source_commit_sha.clone(),
        evaluation_mode: mode,
        partition: baseline_score.partition,
        observation_count: paired_rows.len(),
        event_cluster_count: clusters.len(),
        baseline_rmse_seconds: baseline_rmse,
        candidate_rmse_seconds: candidate_rmse,
        rmse_delta_seconds: rmse_delta,
        rmse_delta_ci_lower_seconds: rmse_ci_lower,
        rmse_delta_ci_upper_seconds: rmse_ci_upper,
        baseline_mae_seconds: baseline_mae,
        candidate_mae_seconds: candidate_mae,
        mae_delta_seconds: mae_delta,
        mae_delta_ci_lower_seconds: mae_ci_lower,
        mae_delta_ci_upper_seconds: mae_ci_upper,
        bootstrap_fraction_candidate_rmse_better: candidate_rmse_better as f64
            / options.bootstrap_replicates as f64,
        bootstrap_replicates: options.bootstrap_replicates,
        confidence_level: options.confidence_level,
        seed: options.seed,
    })
}

/// Small deterministic PRNG used only to make the bootstrap reproducible.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
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

    fn bounded_index(&mut self, upper_exclusive: usize) -> usize {
        // Rejection avoids the small modulo bias from u64 % upper.
        let upper = upper_exclusive as u64;
        let zone = u64::MAX - (u64::MAX % upper);
        loop {
            let value = self.next_u64();
            if value < zone {
                return (value % upper) as usize;
            }
        }
    }
}

fn linear_quantile_sorted(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() || !quantile.is_finite() || !(0.0..=1.0).contains(&quantile) {
        return None;
    }
    if values.len() == 1 {
        return Some(values[0]);
    }
    let position = quantile * (values.len() as f64 - 1.0);
    let low = position.floor() as usize;
    let fraction = position - low as f64;
    if low + 1 < values.len() {
        Some(values[low] + fraction * (values[low + 1] - values[low]))
    } else {
        Some(values[low])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aviation_data_contract::FieldRole;

    fn context(run_id: &str, data: &str) -> ForecastEvaluationContext {
        ForecastEvaluationContext {
            run_id: run_id.into(),
            data_snapshot_sha256: data.into(),
            prediction_artifact_sha256: format!("{:064x}", if run_id == "base" { 1 } else { 2 }),
            model_config_sha256: format!("{:064x}", if run_id == "base" { 3 } else { 4 }),
            feature_schema_sha256: "5".repeat(64),
            evaluation_protocol_sha256: "6".repeat(64),
            evaluator_source_commit_sha: "a".repeat(40),
        }
    }

    fn observation(id: &str, cluster: &str, actual: f64, predicted: f64) -> ForecastObservation {
        let start = 100_000 + (id.as_bytes()[0] as i64) * 1_000;
        ForecastObservation {
            record_id: id.into(),
            event_cluster_id: cluster.into(),
            cohort_id: "EDDF".into(),
            partition: EvaluationPartition::Test,
            decision_time_epoch_seconds: start - 60,
            target_window_start_epoch_seconds: start,
            target_window_end_epoch_seconds: start + actual as i64,
            target_observed_at_epoch_seconds: start + actual as i64 + 1,
            actual_seconds: actual,
            predicted_seconds: predicted,
            feature_schema_declared: true,
            features: vec![FeatureSpec {
                name: "forecast".into(),
                role: FieldRole::ModelInput,
                available_at_epoch_seconds: Some(start - 60),
            }],
            interval: None,
        }
    }

    fn options(seed: u64) -> ForecastComparisonOptions {
        ForecastComparisonOptions {
            bootstrap_replicates: 200,
            confidence_level: 0.95,
            seed,
        }
    }

    #[test]
    fn homogeneous_improvement_has_exact_paired_delta_and_degenerate_ci() {
        let data = "d".repeat(64);
        let baseline_context = context("base", &data);
        let candidate_context = context("candidate", &data);
        let baseline = vec![
            observation("a1", "episode-a", 10.0, 20.0),
            observation("a2", "episode-a", 20.0, 30.0),
            observation("b1", "episode-b", 30.0, 40.0),
            observation("b2", "episode-b", 40.0, 50.0),
        ];
        let candidate = baseline.iter().map(|row| {
            let mut value = row.clone();
            value.predicted_seconds = row.actual_seconds + 5.0;
            value
        }).collect::<Vec<_>>();
        let report = compare_forecasts_cluster_bootstrap(
            &baseline_context, &baseline, &candidate_context, &candidate,
            ForecastEvaluationMode::ProspectiveOperational, options(7),
        ).unwrap();
        assert_eq!(report.event_cluster_count, 2);
        assert!((report.rmse_delta_seconds + 5.0).abs() < 1e-12);
        assert!((report.mae_delta_seconds + 5.0).abs() < 1e-12);
        assert!((report.rmse_delta_ci_lower_seconds + 5.0).abs() < 1e-12);
        assert!((report.rmse_delta_ci_upper_seconds + 5.0).abs() < 1e-12);
        assert_eq!(report.bootstrap_fraction_candidate_rmse_better, 1.0);
    }

    #[test]
    fn identical_models_have_zero_delta_and_deterministic_intervals() {
        let data = "d".repeat(64);
        let context_a = context("base", &data);
        let context_b = context("candidate", &data);
        let rows = vec![
            observation("a1", "episode-a", 10.0, 12.0),
            observation("a2", "episode-a", 20.0, 19.0),
            observation("b1", "episode-b", 30.0, 27.0),
        ];
        let first = compare_forecasts_cluster_bootstrap(
            &context_a, &rows, &context_b, &rows,
            ForecastEvaluationMode::ProspectiveOperational, options(99),
        ).unwrap();
        let second = compare_forecasts_cluster_bootstrap(
            &context_a, &rows, &context_b, &rows,
            ForecastEvaluationMode::ProspectiveOperational, options(99),
        ).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.rmse_delta_seconds, 0.0);
        assert_eq!(first.rmse_delta_ci_lower_seconds, 0.0);
        assert_eq!(first.rmse_delta_ci_upper_seconds, 0.0);
    }

    #[test]
    fn row_order_does_not_change_paired_result() {
        let data = "d".repeat(64);
        let base_ctx = context("base", &data);
        let cand_ctx = context("candidate", &data);
        let baseline = vec![
            observation("a1", "episode-a", 10.0, 12.0),
            observation("a2", "episode-a", 20.0, 19.0),
            observation("b1", "episode-b", 30.0, 27.0),
        ];
        let mut candidate = baseline.clone();
        candidate[0].predicted_seconds = 11.0;
        let first = compare_forecasts_cluster_bootstrap(
            &base_ctx, &baseline, &cand_ctx, &candidate,
            ForecastEvaluationMode::ProspectiveOperational, options(17),
        ).unwrap();
        let mut baseline_reversed = baseline.clone();
        baseline_reversed.reverse();
        let mut candidate_reversed = candidate.clone();
        candidate_reversed.reverse();
        let second = compare_forecasts_cluster_bootstrap(
            &base_ctx, &baseline_reversed, &cand_ctx, &candidate_reversed,
            ForecastEvaluationMode::ProspectiveOperational, options(17),
        ).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn refuses_different_data_snapshots() {
        let baseline_context = context("base", &"d".repeat(64));
        let candidate_context = context("candidate", &"e".repeat(64));
        let baseline = vec![
            observation("a1", "episode-a", 10.0, 11.0),
            observation("b1", "episode-b", 20.0, 21.0),
        ];
        let candidate = baseline.clone();
        assert_eq!(
            compare_forecasts_cluster_bootstrap(
                &baseline_context, &baseline, &candidate_context, &candidate,
                ForecastEvaluationMode::ProspectiveOperational, options(1),
            ),
            Err(ForecastComparisonError::DatasetSnapshotMismatch)
        );
    }

    #[test]
    fn refuses_label_mismatch_for_same_record_id() {
        let data = "d".repeat(64);
        let base_ctx = context("base", &data);
        let cand_ctx = context("candidate", &data);
        let baseline = vec![
            observation("a1", "episode-a", 10.0, 11.0),
            observation("b1", "episode-b", 20.0, 21.0),
        ];
        let mut candidate = baseline.clone();
        candidate[0].actual_seconds = 11.0;
        candidate[0].target_window_end_epoch_seconds += 1;
        candidate[0].target_observed_at_epoch_seconds += 1;
        assert!(matches!(
            compare_forecasts_cluster_bootstrap(
                &base_ctx, &baseline, &cand_ctx, &candidate,
                ForecastEvaluationMode::ProspectiveOperational, options(1),
            ),
            Err(ForecastComparisonError::TargetOrTimingMismatch { field: "actual_seconds", .. })
        ));
    }

    #[test]
    fn refuses_single_cluster_and_invalid_bootstrap_options() {
        let data = "d".repeat(64);
        let base_ctx = context("base", &data);
        let cand_ctx = context("candidate", &data);
        let single_cluster = vec![
            observation("a1", "episode-a", 10.0, 11.0),
            observation("a2", "episode-a", 20.0, 21.0),
        ];
        assert!(matches!(
            compare_forecasts_cluster_bootstrap(
                &base_ctx, &single_cluster, &cand_ctx, &single_cluster,
                ForecastEvaluationMode::ProspectiveOperational, options(1),
            ),
            Err(ForecastComparisonError::InsufficientClusters { .. })
        ));
        let mut bad_replicates = options(1);
        bad_replicates.bootstrap_replicates = 99;
        assert!(matches!(
            compare_forecasts_cluster_bootstrap(
                &base_ctx, &single_cluster, &cand_ctx, &single_cluster,
                ForecastEvaluationMode::ProspectiveOperational, bad_replicates,
            ),
            Err(ForecastComparisonError::InvalidBootstrapReplicateCount { .. })
        ));
        let mut bad_confidence = options(1);
        bad_confidence.confidence_level = f64::NAN;
        assert!(matches!(
            compare_forecasts_cluster_bootstrap(
                &base_ctx, &single_cluster, &cand_ctx, &single_cluster,
                ForecastEvaluationMode::ProspectiveOperational, bad_confidence,
            ),
            Err(ForecastComparisonError::InvalidConfidenceLevel)
        ));
    }
}
