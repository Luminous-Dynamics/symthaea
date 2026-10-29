// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent raw-metric checks for the forecast comparison bootstrap.

use symthaea_operations_research::aviation_calibration_protocol::EvaluationPartition;
use symthaea_operations_research::aviation_data_contract::{FieldRole, FeatureSpec};
use symthaea_operations_research::aviation_forecast_comparison::{
    compare_forecasts_cluster_bootstrap, ForecastComparisonError, ForecastComparisonOptions,
};
use symthaea_operations_research::aviation_forecast_metrics::{
    ForecastEvaluationContext, ForecastEvaluationMode, ForecastObservation,
};

fn context(run_id: &str, data_hash: &str) -> ForecastEvaluationContext {
    ForecastEvaluationContext {
        run_id: run_id.into(),
        data_snapshot_sha256: data_hash.into(),
        prediction_artifact_sha256: (if run_id == "baseline" { "1" } else { "2" }).repeat(64),
        model_config_sha256: (if run_id == "baseline" { "3" } else { "4" }).repeat(64),
        feature_schema_sha256: "5".repeat(64),
        evaluation_protocol_sha256: "6".repeat(64),
        evaluator_source_commit_sha: "a".repeat(40),
    }
}

fn row(id: &str, cluster: &str, actual: f64, predicted: f64) -> ForecastObservation {
    let start = 50_000 + (id.as_bytes()[0] as i64) * 1_000;
    ForecastObservation {
        record_id: id.into(),
        event_cluster_id: cluster.into(),
        cohort_id: "EDDF".into(),
        partition: EvaluationPartition::Test,
        decision_time_epoch_seconds: start - 10,
        target_window_start_epoch_seconds: start,
        target_window_end_epoch_seconds: start + actual as i64,
        target_observed_at_epoch_seconds: start + actual as i64 + 1,
        actual_seconds: actual,
        predicted_seconds: predicted,
        feature_schema_declared: true,
        features: vec![FeatureSpec {
            name: "public_forecast".into(),
            role: FieldRole::ModelInput,
            available_at_epoch_seconds: Some(start - 10),
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
fn homogeneous_errors_produce_independently_expected_paired_differences() {
    let data = "d".repeat(64);
    let baseline_context = context("baseline", &data);
    let candidate_context = context("candidate", &data);
    let baseline = vec![
        row("a1", "airport-day-a", 10.0, 20.0),
        row("a2", "airport-day-a", 20.0, 30.0),
        row("b1", "airport-day-b", 30.0, 40.0),
        row("b2", "airport-day-b", 40.0, 50.0),
    ];
    let candidate = baseline.iter().map(|r| {
        let mut x = r.clone();
        x.predicted_seconds = x.actual_seconds + 5.0;
        x
    }).collect::<Vec<_>>();

    // Independently calculated directly from the raw fixture:
    // baseline absolute errors are all 10, candidate absolute errors all 5.
    let report = compare_forecasts_cluster_bootstrap(
        &baseline_context, &baseline,
        &candidate_context, &candidate,
        ForecastEvaluationMode::ProspectiveOperational, options(7),
    ).unwrap();
    assert_eq!(report.observation_count, 4);
    assert_eq!(report.event_cluster_count, 2);
    assert_eq!(report.baseline_rmse_seconds, 10.0);
    assert_eq!(report.candidate_rmse_seconds, 5.0);
    assert_eq!(report.rmse_delta_seconds, -5.0);
    assert_eq!(report.baseline_mae_seconds, 10.0);
    assert_eq!(report.candidate_mae_seconds, 5.0);
    assert_eq!(report.mae_delta_seconds, -5.0);
    assert_eq!(report.rmse_delta_ci_lower_seconds, -5.0);
    assert_eq!(report.rmse_delta_ci_upper_seconds, -5.0);
    assert_eq!(report.mae_delta_ci_lower_seconds, -5.0);
    assert_eq!(report.mae_delta_ci_upper_seconds, -5.0);
    assert_eq!(report.bootstrap_fraction_candidate_rmse_better, 1.0);
    assert_eq!(report.baseline_prediction_artifact_sha256, baseline_context.prediction_artifact_sha256);
    assert_eq!(report.candidate_prediction_artifact_sha256, candidate_context.prediction_artifact_sha256);
    assert_eq!(report.baseline_model_config_sha256, baseline_context.model_config_sha256);
    assert_eq!(report.candidate_model_config_sha256, candidate_context.model_config_sha256);
    assert_eq!(report.baseline_feature_schema_sha256, baseline_context.feature_schema_sha256);
    assert_eq!(report.candidate_feature_schema_sha256, candidate_context.feature_schema_sha256);
}

#[test]
fn same_snapshot_and_same_policy_are_order_invariant_and_reproducible() {
    let data = "d".repeat(64);
    let baseline_context = context("baseline", &data);
    let candidate_context = context("candidate", &data);
    let rows = vec![
        row("a1", "airport-day-a", 10.0, 12.0),
        row("a2", "airport-day-a", 20.0, 19.0),
        row("b1", "airport-day-b", 30.0, 27.0),
    ];
    let first = compare_forecasts_cluster_bootstrap(
        &baseline_context, &rows, &candidate_context, &rows,
        ForecastEvaluationMode::ProspectiveOperational, options(99),
    ).unwrap();
    let mut reversed = rows.clone();
    reversed.reverse();
    let second = compare_forecasts_cluster_bootstrap(
        &baseline_context, &reversed, &candidate_context, &reversed,
        ForecastEvaluationMode::ProspectiveOperational, options(99),
    ).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.rmse_delta_seconds, 0.0);
    assert_eq!(first.rmse_delta_ci_lower_seconds, 0.0);
    assert_eq!(first.rmse_delta_ci_upper_seconds, 0.0);
    assert_eq!(first.mae_delta_seconds, 0.0);
}

#[test]
fn mismatched_input_snapshot_is_not_ranked() {
    let baseline_context = context("baseline", &"d".repeat(64));
    let candidate_context = context("candidate", &"e".repeat(64));
    let rows = vec![
        row("a1", "airport-day-a", 10.0, 11.0),
        row("b1", "airport-day-b", 20.0, 21.0),
    ];
    assert_eq!(
        compare_forecasts_cluster_bootstrap(
            &baseline_context, &rows, &candidate_context, &rows,
            ForecastEvaluationMode::ProspectiveOperational, options(1),
        ),
        Err(ForecastComparisonError::DatasetSnapshotMismatch)
    );
}

#[test]
fn timing_and_cluster_mismatches_fail_closed() {
    let data = "d".repeat(64);
    let baseline_context = context("baseline", &data);
    let candidate_context = context("candidate", &data);
    let baseline = vec![
        row("a1", "airport-day-a", 10.0, 11.0),
        row("b1", "airport-day-b", 20.0, 21.0),
    ];
    let mut candidate = baseline.clone();
    candidate[0].event_cluster_id = "other-cluster".into();
    assert!(matches!(
        compare_forecasts_cluster_bootstrap(
            &baseline_context, &baseline, &candidate_context, &candidate,
            ForecastEvaluationMode::ProspectiveOperational, options(1),
        ),
        Err(ForecastComparisonError::ClusterMismatch { .. })
    ));

    let mut candidate = baseline.clone();
    candidate[0].decision_time_epoch_seconds -= 1;
    assert!(matches!(
        compare_forecasts_cluster_bootstrap(
            &baseline_context, &baseline, &candidate_context, &candidate,
            ForecastEvaluationMode::ProspectiveOperational, options(1),
        ),
        Err(ForecastComparisonError::TargetOrTimingMismatch { field: "decision_time_epoch_seconds", .. })
    ));
}
