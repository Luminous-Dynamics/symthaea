// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Auditable point/interval forecasting metrics for aviation public-data work.
//!
//! The important contract is task timing: an operational forecast must be
//! produced no later than the start of the target window. A retrospective
//! challenge may intentionally score a reconstruction task at/after the
//! window; it is labeled distinctly and cannot be described as an actionable
//! pre-event forecast. This module scores predictions; it does not prove that
//! a dataset is representative or that a policy change causes operational gain.

use crate::aviation_calibration_protocol::EvaluationPartition;
use crate::aviation_data_contract::{
    audit_feature_availability, AuditStatus, FeatureSpec,
};
use std::collections::{BTreeMap, BTreeSet};

/// Whether prediction time precedes the forecast target window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastEvaluationMode {
    /// Recommendation-time prediction: features and prediction must be available
    /// no later than the beginning of the target window.
    ProspectiveOperational,
    /// Retrospective benchmark that may reveal the window endpoint at scoring time.
    /// This mode does not establish operational usefulness for that same event.
    RetrospectiveChallenge,
}

/// Optional central prediction interval. Coverage must be in (0, 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictionInterval {
    pub lower_seconds: f64,
    pub upper_seconds: f64,
    pub nominal_coverage: f64,
}

/// One scored forecast row with explicit event-time, prediction-time, target-time,
/// feature-availability, event-cluster and evaluation-partition metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastObservation {
    pub record_id: String,
    /// Dependence cluster such as a disruption episode or airport-day.
    pub event_cluster_id: String,
    /// Cohort for stratified reporting, usually an airport or airport-season.
    pub cohort_id: String,
    pub partition: EvaluationPartition,
    pub decision_time_epoch_seconds: i64,
    pub target_window_start_epoch_seconds: i64,
    pub target_window_end_epoch_seconds: i64,
    pub target_observed_at_epoch_seconds: i64,
    pub actual_seconds: f64,
    pub predicted_seconds: f64,
    /// True only when an explicit model-input schema was supplied. An explicitly
    /// declared empty schema is permitted for an intercept-only baseline.
    pub feature_schema_declared: bool,
    /// Each included field must be a ModelInput with evidenced availability time.
    pub features: Vec<FeatureSpec>,
    pub interval: Option<PredictionInterval>,
}

/// Input contract failures. Invalid or incomplete records are never silently
/// dropped, clipped, or converted into a favorable score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForecastScoringError {
    EmptyObservationSet,
    InvalidIdentity { record_id: String, field: &'static str },
    DuplicateRecordId { record_id: String },
    InvalidEvaluationPartition { partition: EvaluationPartition },
    MixedEvaluationPartitions,
    InvalidTimestamp { record_id: String, field: &'static str },
    InvalidTargetWindow { record_id: String },
    TargetWindowDurationMismatch { record_id: String },
    TargetObservedBeforeWindowEnd { record_id: String },
    PredictionAfterTargetStart { record_id: String },
    FeatureSchemaNotDeclared { record_id: String },
    FeatureAvailabilityFailed { record_id: String, status: AuditStatus },
    InvalidActual { record_id: String },
    InvalidPrediction { record_id: String },
    InvalidInterval { record_id: String },
    MixedIntervalPresence,
    MixedIntervalCoverage,
    NonFiniteMetric,
}

/// Descriptive point and interval forecast metrics for one group.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastScoreSummary {
    pub observation_count: usize,
    pub event_cluster_count: usize,
    pub rmse_seconds: f64,
    pub mae_seconds: f64,
    /// Mean prediction minus observed value; positive means over-prediction.
    pub mean_signed_error_seconds: f64,
    pub p90_absolute_error_seconds: f64,
    pub nominal_interval_coverage: Option<f64>,
    pub empirical_interval_coverage: Option<f64>,
    pub mean_interval_width_seconds: Option<f64>,
    /// Mean central-interval Winkler score, lower is better.
    pub mean_winkler_score_seconds: Option<f64>,
}

/// Cohort-specific result, sorted by cohort ID.
#[derive(Debug, Clone, PartialEq)]
pub struct CohortForecastScore {
    pub cohort_id: String,
    pub metrics: ForecastScoreSummary,
}

/// Complete forecast scorecard. The partition and task mode are included so a
/// retrospective score cannot be mislabeled as a prospective operational forecast.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastEvaluationReport {
    pub mode: ForecastEvaluationMode,
    pub partition: EvaluationPartition,
    pub overall: ForecastScoreSummary,
    pub by_cohort: Vec<CohortForecastScore>,
}

#[derive(Debug, Clone)]
struct ScoredRow {
    event_cluster_id: String,
    cohort_id: String,
    signed_error: f64,
    absolute_error: f64,
    squared_error: f64,
    interval_covered: Option<bool>,
    interval_width: Option<f64>,
    winkler_score: Option<f64>,
    nominal_interval_coverage: Option<f64>,
}

/// Validate and score one frozen Validation or Test partition.
///
/// The function refuses Training/Calibration rows, mixed partitions, duplicate
/// record IDs, future feature values, unknown feature availability, invalid
/// target chronology, negative/non-finite predictions, and partially populated
/// interval outputs. It reports aggregate and cohort metrics without silently
/// removing difficult rows.
pub fn evaluate_forecasts(
    observations: &[ForecastObservation],
    mode: ForecastEvaluationMode,
) -> Result<ForecastEvaluationReport, ForecastScoringError> {
    if observations.is_empty() {
        return Err(ForecastScoringError::EmptyObservationSet);
    }

    let partition = observations[0].partition;
    if !matches!(partition, EvaluationPartition::Validation | EvaluationPartition::Test) {
        return Err(ForecastScoringError::InvalidEvaluationPartition { partition });
    }

    let has_intervals = observations[0].interval.is_some();
    let nominal_coverage = observations[0].interval.map(|value| value.nominal_coverage);
    let mut seen_record_ids = BTreeSet::new();
    let mut scored = Vec::with_capacity(observations.len());

    for observation in observations {
        if observation.partition != partition {
            return Err(ForecastScoringError::MixedEvaluationPartitions);
        }
        for (field, value) in [
            ("record_id", observation.record_id.as_str()),
            ("event_cluster_id", observation.event_cluster_id.as_str()),
            ("cohort_id", observation.cohort_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ForecastScoringError::InvalidIdentity {
                    record_id: observation.record_id.clone(),
                    field,
                });
            }
        }
        if !seen_record_ids.insert(observation.record_id.as_str()) {
            return Err(ForecastScoringError::DuplicateRecordId {
                record_id: observation.record_id.clone(),
            });
        }

        for (field, timestamp) in [
            ("decision_time_epoch_seconds", observation.decision_time_epoch_seconds),
            ("target_window_start_epoch_seconds", observation.target_window_start_epoch_seconds),
            ("target_window_end_epoch_seconds", observation.target_window_end_epoch_seconds),
            ("target_observed_at_epoch_seconds", observation.target_observed_at_epoch_seconds),
        ] {
            if timestamp < 0 {
                return Err(ForecastScoringError::InvalidTimestamp {
                    record_id: observation.record_id.clone(),
                    field,
                });
            }
        }
        if observation.target_window_start_epoch_seconds >= observation.target_window_end_epoch_seconds {
            return Err(ForecastScoringError::InvalidTargetWindow {
                record_id: observation.record_id.clone(),
            });
        }
        if observation.target_observed_at_epoch_seconds < observation.target_window_end_epoch_seconds {
            return Err(ForecastScoringError::TargetObservedBeforeWindowEnd {
                record_id: observation.record_id.clone(),
            });
        }
        if !observation.actual_seconds.is_finite() || observation.actual_seconds < 0.0 {
            return Err(ForecastScoringError::InvalidActual {
                record_id: observation.record_id.clone(),
            });
        }
        if !observation.predicted_seconds.is_finite() || observation.predicted_seconds < 0.0 {
            return Err(ForecastScoringError::InvalidPrediction {
                record_id: observation.record_id.clone(),
            });
        }
        let target_duration = (
            observation.target_window_end_epoch_seconds - observation.target_window_start_epoch_seconds
        ) as f64;
        if (observation.actual_seconds - target_duration).abs() > 1.0 {
            return Err(ForecastScoringError::TargetWindowDurationMismatch {
                record_id: observation.record_id.clone(),
            });
        }
        if mode == ForecastEvaluationMode::ProspectiveOperational
            && observation.decision_time_epoch_seconds > observation.target_window_start_epoch_seconds
        {
            return Err(ForecastScoringError::PredictionAfterTargetStart {
                record_id: observation.record_id.clone(),
            });
        }
        if !observation.feature_schema_declared {
            return Err(ForecastScoringError::FeatureSchemaNotDeclared {
                record_id: observation.record_id.clone(),
            });
        }
        // An explicit empty schema is a valid intercept-only baseline. When
        // fields exist, each must be an eligible model input and available by
        // the prediction decision time.
        if !observation.features.is_empty() {
            let audit = audit_feature_availability(
                observation.decision_time_epoch_seconds,
                &observation.features,
            );
            if audit.status != AuditStatus::Accept {
                return Err(ForecastScoringError::FeatureAvailabilityFailed {
                    record_id: observation.record_id.clone(),
                    status: audit.status,
                });
            }
        }

        if observation.interval.is_some() != has_intervals {
            return Err(ForecastScoringError::MixedIntervalPresence);
        }
        if let Some(interval) = observation.interval {
            if !interval.lower_seconds.is_finite()
                || !interval.upper_seconds.is_finite()
                || !interval.nominal_coverage.is_finite()
                || interval.lower_seconds < 0.0
                || interval.upper_seconds < interval.lower_seconds
                || interval.nominal_coverage <= 0.0
                || interval.nominal_coverage >= 1.0
            {
                return Err(ForecastScoringError::InvalidInterval {
                    record_id: observation.record_id.clone(),
                });
            }
            if nominal_coverage != Some(interval.nominal_coverage) {
                return Err(ForecastScoringError::MixedIntervalCoverage);
            }
        }

        let signed_error = observation.predicted_seconds - observation.actual_seconds;
        let absolute_error = signed_error.abs();
        let squared_error = signed_error * signed_error;
        if !signed_error.is_finite() || !absolute_error.is_finite() || !squared_error.is_finite() {
            return Err(ForecastScoringError::NonFiniteMetric);
        }

        let interval_metrics = observation.interval.map(|interval| {
            let width = interval.upper_seconds - interval.lower_seconds;
            let alpha = 1.0 - interval.nominal_coverage;
            let penalty = if observation.actual_seconds < interval.lower_seconds {
                (2.0 / alpha) * (interval.lower_seconds - observation.actual_seconds)
            } else if observation.actual_seconds > interval.upper_seconds {
                (2.0 / alpha) * (observation.actual_seconds - interval.upper_seconds)
            } else {
                0.0
            };
            (
                observation.actual_seconds >= interval.lower_seconds
                    && observation.actual_seconds <= interval.upper_seconds,
                width,
                width + penalty,
            )
        });

        scored.push(ScoredRow {
            event_cluster_id: observation.event_cluster_id.clone(),
            cohort_id: observation.cohort_id.clone(),
            signed_error,
            absolute_error,
            squared_error,
            interval_covered: interval_metrics.map(|value| value.0),
            interval_width: interval_metrics.map(|value| value.1),
            winkler_score: interval_metrics.map(|value| value.2),
            nominal_interval_coverage: observation.interval.map(|value| value.nominal_coverage),
        });
    }

    let overall_rows: Vec<&ScoredRow> = scored.iter().collect();
    let overall = summarize(&overall_rows)?;
    let mut cohort_rows = BTreeMap::<String, Vec<&ScoredRow>>::new();
    for row in &scored {
        cohort_rows.entry(row.cohort_id.clone()).or_default().push(row);
    }
    let mut by_cohort = Vec::with_capacity(cohort_rows.len());
    for (cohort_id, rows) in cohort_rows {
        by_cohort.push(CohortForecastScore {
            cohort_id,
            metrics: summarize(&rows)?,
        });
    }

    Ok(ForecastEvaluationReport { mode, partition, overall, by_cohort })
}

fn summarize(rows: &[&ScoredRow]) -> Result<ForecastScoreSummary, ForecastScoringError> {
    if rows.is_empty() {
        return Err(ForecastScoringError::EmptyObservationSet);
    }
    let count = rows.len() as f64;
    let mut squared_error_sum = 0.0;
    let mut absolute_error_sum = 0.0;
    let mut signed_error_sum = 0.0;
    let mut absolute_errors = Vec::with_capacity(rows.len());
    let mut clusters = BTreeSet::new();
    let nominal_interval_coverage = rows[0].nominal_interval_coverage;
    let intervals_present = rows[0].interval_covered.is_some();
    let mut covered = 0_usize;
    let mut width_sum = 0.0;
    let mut winkler_sum = 0.0;

    for row in rows {
        squared_error_sum += row.squared_error;
        absolute_error_sum += row.absolute_error;
        signed_error_sum += row.signed_error;
        absolute_errors.push(row.absolute_error);
        clusters.insert(row.event_cluster_id.as_str());
        if let Some(is_covered) = row.interval_covered {
            if is_covered { covered += 1; }
        }
        if let Some(width) = row.interval_width {
            width_sum += width;
        }
        if let Some(score) = row.winkler_score {
            winkler_sum += score;
        }
    }
    if !squared_error_sum.is_finite()
        || !absolute_error_sum.is_finite()
        || !signed_error_sum.is_finite()
        || (intervals_present && (!width_sum.is_finite() || !winkler_sum.is_finite()))
    {
        return Err(ForecastScoringError::NonFiniteMetric);
    }
    absolute_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p90_absolute_error_seconds = linear_quantile_sorted(&absolute_errors, 0.90)
        .ok_or(ForecastScoringError::NonFiniteMetric)?;

    Ok(ForecastScoreSummary {
        observation_count: rows.len(),
        event_cluster_count: clusters.len(),
        rmse_seconds: (squared_error_sum / count).sqrt(),
        mae_seconds: absolute_error_sum / count,
        mean_signed_error_seconds: signed_error_sum / count,
        p90_absolute_error_seconds,
        nominal_interval_coverage,
        empirical_interval_coverage: if intervals_present {
            Some(covered as f64 / count)
        } else {
            None
        },
        mean_interval_width_seconds: if intervals_present { Some(width_sum / count) } else { None },
        mean_winkler_score_seconds: if intervals_present { Some(winkler_sum / count) } else { None },
    })
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

    fn row(
        id: &str,
        cohort: &str,
        actual: f64,
        predicted: f64,
        start: i64,
        decision: i64,
        partition: EvaluationPartition,
    ) -> ForecastObservation {
        ForecastObservation {
            record_id: id.into(),
            event_cluster_id: format!("cluster-{id}"),
            cohort_id: cohort.into(),
            partition,
            decision_time_epoch_seconds: decision,
            target_window_start_epoch_seconds: start,
            target_window_end_epoch_seconds: start + actual as i64,
            target_observed_at_epoch_seconds: start + actual as i64 + 2,
            actual_seconds: actual,
            predicted_seconds: predicted,
            feature_schema_declared: true,
            features: vec![FeatureSpec {
                name: "weather_forecast".into(),
                role: crate::aviation_data_contract::FieldRole::ModelInput,
                available_at_epoch_seconds: Some(decision),
            }],
            interval: None,
        }
    }

    #[test]
    fn computes_rmse_mae_bias_p90_and_cohort_slices() {
        let observations = vec![
            row("r1", "ZAAA", 10.0, 12.0, 1_000, 1_000, EvaluationPartition::Test),
            row("r2", "AAAA", 20.0, 18.0, 2_000, 2_000, EvaluationPartition::Test),
            row("r3", "AAAA", 30.0, 33.0, 3_000, 3_000, EvaluationPartition::Test),
        ];
        let report = evaluate_forecasts(&observations, ForecastEvaluationMode::ProspectiveOperational).unwrap();
        assert_eq!(report.partition, EvaluationPartition::Test);
        assert_eq!(report.mode, ForecastEvaluationMode::ProspectiveOperational);
        assert!((report.overall.rmse_seconds - (17.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((report.overall.mae_seconds - 7.0 / 3.0).abs() < 1e-12);
        assert!((report.overall.mean_signed_error_seconds - 1.0).abs() < 1e-12);
        assert!((report.overall.p90_absolute_error_seconds - 2.8).abs() < 1e-12);
        assert_eq!(report.overall.event_cluster_count, 3);
        assert_eq!(report.by_cohort[0].cohort_id, "AAAA");
        assert_eq!(report.by_cohort[0].metrics.observation_count, 2);
        assert_eq!(report.by_cohort[1].cohort_id, "ZAAA");
    }

    #[test]
    fn retrospective_challenge_is_scored_but_not_mislabeled_operational() {
        let observation = row("r1", "EDDF", 600.0, 590.0, 10_000, 10_600, EvaluationPartition::Test);
        assert!(matches!(
            evaluate_forecasts(&[observation.clone()], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::PredictionAfterTargetStart { .. })
        ));
        let report = evaluate_forecasts(&[observation], ForecastEvaluationMode::RetrospectiveChallenge).unwrap();
        assert_eq!(report.mode, ForecastEvaluationMode::RetrospectiveChallenge);
        assert_eq!(report.overall.observation_count, 1);
    }

    #[test]
    fn rejects_future_feature_availability() {
        let mut observation = row("r1", "EDDF", 10.0, 10.0, 1_000, 1_000, EvaluationPartition::Test);
        observation.features[0].available_at_epoch_seconds = Some(1_001);
        assert!(matches!(
            evaluate_forecasts(&[observation], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::FeatureAvailabilityFailed { status: AuditStatus::Reject, .. })
        ));
    }

    #[test]
    fn rejects_missing_feature_availability_evidence() {
        let mut observation = row("r1", "EDDF", 10.0, 10.0, 1_000, 1_000, EvaluationPartition::Test);
        observation.features[0].available_at_epoch_seconds = None;
        assert!(matches!(
            evaluate_forecasts(&[observation], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::FeatureAvailabilityFailed { status: AuditStatus::Indeterminate, .. })
        ));
    }

    #[test]
    fn rejects_non_input_feature_roles_even_when_timestamp_is_early() {
        let mut observation = row("r1", "EDDF", 10.0, 10.0, 1_000, 1_000, EvaluationPartition::Test);
        observation.features[0].role = crate::aviation_data_contract::FieldRole::TargetOnly;
        assert!(matches!(
            evaluate_forecasts(&[observation], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::FeatureAvailabilityFailed { status: AuditStatus::Reject, .. })
        ));
    }

    #[test]
    fn rejects_training_and_calibration_rows_for_scoring() {
        for partition in [EvaluationPartition::Training, EvaluationPartition::Calibration] {
            let observation = row("r1", "EDDF", 10.0, 10.0, 1_000, 1_000, partition);
            assert!(matches!(
                evaluate_forecasts(&[observation], ForecastEvaluationMode::ProspectiveOperational),
                Err(ForecastScoringError::InvalidEvaluationPartition { .. })
            ));
        }
    }

    #[test]
    fn refuses_to_mix_validation_and_test_partitions() {
        let observations = vec![
            row("r1", "EDDF", 10.0, 10.0, 1_000, 1_000, EvaluationPartition::Validation),
            row("r2", "EDDF", 10.0, 10.0, 2_000, 2_000, EvaluationPartition::Test),
        ];
        assert_eq!(
            evaluate_forecasts(&observations, ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::MixedEvaluationPartitions)
        );
    }

    #[test]
    fn rejects_duplicate_records_and_empty_clusters() {
        let mut duplicated = row("r1", "EDDF", 10.0, 10.0, 1_000, 1_000, EvaluationPartition::Test);
        let mut second = duplicated.clone();
        second.cohort_id = "EGLL".into();
        assert!(matches!(
            evaluate_forecasts(&[duplicated.clone(), second], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::DuplicateRecordId { .. })
        ));
        duplicated.event_cluster_id = " ".into();
        assert!(matches!(
            evaluate_forecasts(&[duplicated], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::InvalidIdentity { field: "event_cluster_id", .. })
        ));
    }

    #[test]
    fn interval_metrics_cover_width_and_winkler_score() {
        let mut a = row("r1", "EDDF", 10.0, 10.0, 1_000, 1_000, EvaluationPartition::Test);
        let mut b = row("r2", "EDDF", 20.0, 20.0, 2_000, 2_000, EvaluationPartition::Test);
        a.interval = Some(PredictionInterval { lower_seconds: 8.0, upper_seconds: 12.0, nominal_coverage: 0.90 });
        b.interval = Some(PredictionInterval { lower_seconds: 18.0, upper_seconds: 22.0, nominal_coverage: 0.90 });
        let report = evaluate_forecasts(&[a, b], ForecastEvaluationMode::ProspectiveOperational).unwrap();
        assert_eq!(report.overall.nominal_interval_coverage, Some(0.90));
        assert_eq!(report.overall.empirical_interval_coverage, Some(1.0));
        assert_eq!(report.overall.mean_interval_width_seconds, Some(4.0));
        assert_eq!(report.overall.mean_winkler_score_seconds, Some(4.0));
    }

    #[test]
    fn interval_penalty_increases_when_observation_misses_bounds() {
        let mut a = row("r1", "EDDF", 20.0, 20.0, 1_000, 1_000, EvaluationPartition::Test);
        a.interval = Some(PredictionInterval { lower_seconds: 8.0, upper_seconds: 12.0, nominal_coverage: 0.90 });
        let report = evaluate_forecasts(&[a], ForecastEvaluationMode::ProspectiveOperational).unwrap();
        assert_eq!(report.overall.empirical_interval_coverage, Some(0.0));
        assert_eq!(report.overall.mean_winkler_score_seconds, Some(164.0));
    }

    #[test]
    fn rejects_mixed_interval_presence_or_coverage() {
        let mut a = row("r1", "EDDF", 10.0, 10.0, 1_000, 1_000, EvaluationPartition::Test);
        let mut b = row("r2", "EDDF", 10.0, 10.0, 2_000, 2_000, EvaluationPartition::Test);
        a.interval = Some(PredictionInterval { lower_seconds: 8.0, upper_seconds: 12.0, nominal_coverage: 0.90 });
        assert!(matches!(
            evaluate_forecasts(&[a.clone(), b.clone()], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::MixedIntervalPresence)
        ));
        b.interval = Some(PredictionInterval { lower_seconds: 8.0, upper_seconds: 12.0, nominal_coverage: 0.95 });
        assert!(matches!(
            evaluate_forecasts(&[a, b], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::MixedIntervalCoverage)
        ));
    }

    #[test]
    fn explicit_constant_baseline_can_have_empty_feature_schema() {
        let mut observation = row("r1", "EDDF", 10.0, 11.0, 1_000, 1_000, EvaluationPartition::Validation);
        observation.features.clear();
        let report = evaluate_forecasts(&[observation], ForecastEvaluationMode::ProspectiveOperational).unwrap();
        assert_eq!(report.overall.observation_count, 1);
    }

    #[test]
    fn missing_feature_schema_is_not_confused_with_constant_baseline() {
        let mut observation = row("r1", "EDDF", 10.0, 11.0, 1_000, 1_000, EvaluationPartition::Validation);
        observation.feature_schema_declared = false;
        assert!(matches!(
            evaluate_forecasts(&[observation], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::FeatureSchemaNotDeclared { .. })
        ));
    }

    #[test]
    fn rejects_negative_or_non_finite_predictions() {
        let mut negative = row("r1", "EDDF", 10.0, 11.0, 1_000, 1_000, EvaluationPartition::Test);
        negative.predicted_seconds = -1.0;
        assert!(matches!(
            evaluate_forecasts(&[negative], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::InvalidPrediction { .. })
        ));
        let mut non_finite = row("r2", "EDDF", 10.0, 11.0, 2_000, 2_000, EvaluationPartition::Test);
        non_finite.predicted_seconds = f64::NAN;
        assert!(matches!(
            evaluate_forecasts(&[non_finite], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::InvalidPrediction { .. })
        ));
    }

    #[test]
    fn rejects_window_or_actual_mismatch() {
        let mut bad_window = row("r1", "EDDF", 10.0, 10.0, 1_000, 1_000, EvaluationPartition::Test);
        bad_window.target_window_end_epoch_seconds = bad_window.target_window_start_epoch_seconds;
        assert!(matches!(
            evaluate_forecasts(&[bad_window], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::InvalidTargetWindow { .. })
        ));
        let mut bad_label = row("r2", "EDDF", 10.0, 10.0, 2_000, 2_000, EvaluationPartition::Test);
        bad_label.actual_seconds = 30.0;
        assert!(matches!(
            evaluate_forecasts(&[bad_label], ForecastEvaluationMode::ProspectiveOperational),
            Err(ForecastScoringError::TargetWindowDurationMismatch { .. })
        ));
    }
}
