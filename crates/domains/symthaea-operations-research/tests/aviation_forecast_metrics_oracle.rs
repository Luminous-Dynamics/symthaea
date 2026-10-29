// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independently calculated scorecard oracles for aviation forecast metrics.
//! Expected values are computed from raw predictions here, not by calling
//! production scoring helpers.

use symthaea_operations_research::aviation_calibration_protocol::EvaluationPartition;
use symthaea_operations_research::aviation_data_contract::{FieldRole, FeatureSpec};
use symthaea_operations_research::aviation_forecast_metrics::{
    evaluate_forecasts, ForecastEvaluationMode, ForecastObservation, ForecastScoringError,
};

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
    let report = evaluate_forecasts(&rows, ForecastEvaluationMode::ProspectiveOperational).unwrap();

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
        evaluate_forecasts(&[row.clone()], ForecastEvaluationMode::ProspectiveOperational),
        Err(ForecastScoringError::PredictionAfterTargetStart { .. })
    ));
    let report = evaluate_forecasts(&[row], ForecastEvaluationMode::RetrospectiveChallenge).unwrap();
    assert_eq!(report.mode, ForecastEvaluationMode::RetrospectiveChallenge);
}

#[test]
fn a_feature_available_after_decision_time_cannot_be_scored_as_valid() {
    let mut row = row("a", "EDDF", 10.0, 10.0, 1_000, 1_000);
    row.features[0].available_at_epoch_seconds = Some(1_001);
    assert!(matches!(
        evaluate_forecasts(&[row], ForecastEvaluationMode::ProspectiveOperational),
        Err(ForecastScoringError::FeatureAvailabilityFailed { .. })
    ));
}

#[test]
fn cohort_order_is_deterministic_and_test_validation_are_not_mixed() {
    let rows = vec![
        row("b", "ZAAA", 10.0, 10.0, 1_000, 1_000),
        row("a", "AAAA", 10.0, 11.0, 2_000, 2_000),
    ];
    let report = evaluate_forecasts(&rows, ForecastEvaluationMode::ProspectiveOperational).unwrap();
    assert_eq!(report.by_cohort[0].cohort_id, "AAAA");
    assert_eq!(report.by_cohort[1].cohort_id, "ZAAA");

    let mut mixed = rows;
    mixed[1].partition = EvaluationPartition::Validation;
    assert_eq!(
        evaluate_forecasts(&mixed, ForecastEvaluationMode::ProspectiveOperational),
        Err(ForecastScoringError::MixedEvaluationPartitions)
    );
}
