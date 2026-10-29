// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independently calculated scorecard oracles for aviation forecast metrics.
//! Expected values are computed from raw predictions here, not by calling
//! production scoring helpers.

use symthaea_operations_research::aviation_calibration_protocol::EvaluationPartition;
use symthaea_operations_research::aviation_data_contract::{FieldRole, FeatureSpec};
use symthaea_operations_research::aviation_forecast_metrics::{
    evaluate_forecasts, ForecastEvaluationContext, ForecastEvaluationMode, ForecastObservation,
    ForecastScoringError,
};

fn context() -> ForecastEvaluationContext {
    ForecastEvaluationContext {
        run_id: "oracle-run".into(),
        data_snapshot_sha256: "1".repeat(64),
        prediction_artifact_sha256: "2".repeat(64),
        model_config_sha256: "3".repeat(64),
        feature_schema_sha256: "4".repeat(64),
        evaluation_protocol_sha256: "5".repeat(64),
        evaluator_source_commit_sha: "a".repeat(40),
    }
}

fn score(
    observations: &[ForecastObservation],
    mode: ForecastEvaluationMode,
) -> Result<symthaea_operations_research::aviation_forecast_metrics::ForecastEvaluationReport, ForecastScoringError> {
    evaluate_forecasts(&context(), observations, mode)
}

fn row(
    id: &str,
    cohort: &str,
    actual: f64,
    predicted: f64,
    start: i64,
    decision: i64,
) -> ForecastObservation {
    ForecastObservation {
        record_id: id.into(),
        event_cluster_id: format!("cluster-{id}"),
        cohort_id: cohort.into(),
        partition: EvaluationPartition::Test,
        decision_time_epoch_seconds: decision,
        target_window_start_epoch_seconds: start,
        target_window_end_epoch_seconds: start + actual as i64,
        target_observed_at_epoch_seconds: start + actual as i64 + 1,
        actual_seconds: actual,
        predicted_seconds: predicted,
        feature_schema_declared: true,
        features: vec![FeatureSpec {
            name: "weather_forecast".into(),
            role: FieldRole::ModelInput,
            available_at_epoch_seconds: Some(decision),
        }],
        interval: None,
    }
}

#[test]
fn independent_rmse_mae_bias_and_quantile_match_the_scorecard() {
    let rows = vec![
        row("a", "EDDF", 10.0, 12.0, 1_000, 1_000),
        row("b", "EGLL", 20.0, 18.0, 2_000, 2_000),
        row("c", "EGLL", 30.0, 33.0, 3_000, 3_000),
    ];
    let report = score(&rows, ForecastEvaluationMode::ProspectiveOperational).unwrap();

    let actual = [10.0_f64, 20.0, 30.0];
    let predicted = [12.0_f64, 18.0, 33.0];
    let errors: Vec<f64> = predicted.iter().zip(actual).map(|(p, a)| p - a).collect();
    let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
    let mae = errors.iter().map(|e| e.abs()).sum::<f64>() / errors.len() as f64;
    let bias = errors.iter().sum::<f64>() / errors.len() as f64;
    let mut abs_errors: Vec<f64> = errors.iter().map(|e| e.abs()).collect();
    abs_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let position = 0.9 * (abs_errors.len() as f64 - 1.0);
    let low = position.floor() as usize;
    let p90 = abs_errors[low] + (position - low as f64) * (abs_errors[low + 1] - abs_errors[low]);

    assert!((report.overall.rmse_seconds - rmse).abs() < 1e-12);
    assert!((report.overall.mae_seconds - mae).abs() < 1e-12);
    assert!((report.overall.mean_signed_error_seconds - bias).abs() < 1e-12);
    assert!((report.overall.p90_absolute_error_seconds - p90).abs() < 1e-12);
}

#[test]
fn retrospective_challenge_time_is_not_accepted_as_operational_forecast_time() {
    let row = row("a", "EDDF", 600.0, 590.0, 10_000, 10_600);
    assert!(matches!(
        score(&[row.clone()], ForecastEvaluationMode::ProspectiveOperational),
        Err(ForecastScoringError::PredictionAfterTargetStart { .. })
    ));
    let report = score(&[row], ForecastEvaluationMode::RetrospectiveChallenge).unwrap();
    assert_eq!(report.mode, ForecastEvaluationMode::RetrospectiveChallenge);
}

#[test]
fn a_feature_available_after_decision_time_cannot_be_scored_as_valid() {
    let mut row = row("a", "EDDF", 10.0, 10.0, 1_000, 1_000);
    row.features[0].available_at_epoch_seconds = Some(1_001);
    assert!(matches!(
        score(&[row], ForecastEvaluationMode::ProspectiveOperational),
        Err(ForecastScoringError::FeatureAvailabilityFailed { .. })
    ));
}

#[test]
fn cohort_order_is_deterministic_and_test_validation_are_not_mixed() {
    let rows = vec![
        row("b", "ZAAA", 10.0, 10.0, 1_000, 1_000),
        row("a", "AAAA", 10.0, 11.0, 2_000, 2_000),
    ];
    let report = score(&rows, ForecastEvaluationMode::ProspectiveOperational).unwrap();
    assert_eq!(report.by_cohort[0].cohort_id, "AAAA");
    assert_eq!(report.by_cohort[1].cohort_id, "ZAAA");

    let mut mixed = rows;
    mixed[1].partition = EvaluationPartition::Validation;
    assert_eq!(
        score(&mixed, ForecastEvaluationMode::ProspectiveOperational),
        Err(ForecastScoringError::MixedEvaluationPartitions)
    );
}


#[test]
fn score_report_preserves_all_run_provenance_digests() {
    let rows = vec![row("a", "EDDF", 10.0, 11.0, 1_000, 1_000)];
    let context = context();
    let report = evaluate_forecasts(&context, &rows, ForecastEvaluationMode::ProspectiveOperational).unwrap();
    assert_eq!(report.run_id, context.run_id);
    assert_eq!(report.data_snapshot_sha256, context.data_snapshot_sha256);
    assert_eq!(report.prediction_artifact_sha256, context.prediction_artifact_sha256);
    assert_eq!(report.model_config_sha256, context.model_config_sha256);
    assert_eq!(report.feature_schema_sha256, context.feature_schema_sha256);
    assert_eq!(report.evaluation_protocol_sha256, context.evaluation_protocol_sha256);
    assert_eq!(report.evaluator_source_commit_sha, context.evaluator_source_commit_sha);
}

#[test]
fn malformed_provenance_digest_blocks_scoring() {
    let mut context = context();
    context.feature_schema_sha256 = "not-a-digest".into();
    let rows = vec![row("a", "EDDF", 10.0, 11.0, 1_000, 1_000)];
    assert_eq!(
        evaluate_forecasts(&context, &rows, ForecastEvaluationMode::ProspectiveOperational),
        Err(ForecastScoringError::InvalidProvenanceField { field: "feature_schema_sha256" })
    );
}
