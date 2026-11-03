// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Family-aware coverage and sharpness evaluation for vehicle trajectory sets.
//!
//! The caller supplies nonconformity scores and a scalar measure of each
//! prediction set. This module does not generate predictions, construct
//! geometric tubes, validate labels, or establish exchangeability.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub const VEHICLE_TRAJECTORY_COVERAGE_SCHEMA: &str =
    "symthaea.vehicle.trajectory-coverage.v0.1";
pub const TRAJECTORY_COVERAGE_TARGET_P95: f64 = 0.95;
pub const MIN_FAMILY_COVERAGE_SAMPLES: usize = 20;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryCalibrationObservation {
    pub failure_family_id: String,
    pub scenario_id: String,
    /// Nonconformity score produced by the frozen prediction/evaluation stack.
    pub nonconformity_score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryEvaluationObservation {
    pub failure_family_id: String,
    pub scenario_id: String,
    /// Score under the exact same score function used for calibration.
    pub nonconformity_score: f64,
    /// Caller-defined measure of this prediction set (e.g. tube area or mean
    /// radius). Its semantics are fixed by measure_id and measure_unit_id.
    pub prediction_set_measure: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryCoverageResult {
    pub failure_family_id: String,
    pub scenario_id: String,
    pub nonconformity_score: f64,
    pub prediction_set_measure: f64,
    pub covered: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryFamilyCoverage {
    pub failure_family_id: String,
    pub observation_count: usize,
    pub covered_count: usize,
    pub empirical_coverage: f64,
    pub mean_prediction_set_measure: f64,
    pub p95_prediction_set_measure: f64,
    /// Descriptive flag only; it is not a theorem-backed minimum sample size.
    pub underpowered_for_group_claim: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrajectoryCoverageError {
    EmptyIdentity,
    InvalidDigest,
    InvalidHorizon,
    InvalidNominalCoverage,
    EmptyCalibrationSet,
    EmptyEvaluationSet,
    InsufficientCalibrationSamples,
    DuplicateScenarioId,
    CalibrationEvaluationOverlap,
    InvalidNonconformityScore,
    InvalidPredictionSetMeasure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VehicleTrajectoryCoverageReport {
    pub schema_version: String,
    pub target_id: String,
    pub horizon_ms: u64,
    pub nominal_coverage: f64,
    pub predictor_artifact_digest: String,
    pub set_construction_artifact_digest: String,
    pub outcome_evaluator_digest: String,
    pub calibration_sampling_profile_digest: String,
    pub evaluation_sampling_profile_digest: String,
    pub measure_id: String,
    pub measure_unit_id: String,
    pub calibration_observations: Vec<TrajectoryCalibrationObservation>,
    pub evaluation_observations: Vec<TrajectoryEvaluationObservation>,
    pub evaluation_results: Vec<TrajectoryCoverageResult>,
    pub calibration_sample_count: usize,
    pub evaluation_sample_count: usize,
    /// One-based order statistic rank: ceil((n_cal + 1) * nominal_coverage).
    pub calibration_quantile_rank: usize,
    pub calibration_threshold: f64,
    pub empirical_coverage: f64,
    /// Empirical coverage minus the declared nominal target.
    pub signed_coverage_error: f64,
    pub mean_prediction_set_measure: f64,
    pub p95_prediction_set_measure: f64,
    /// Equal-weight average of family coverage rates; descriptive only.
    pub macro_family_coverage: f64,
    /// Minimum empirical coverage among observed failure families.
    pub worst_family_coverage: f64,
    pub family_results: Vec<TrajectoryFamilyCoverage>,
    /// Equal profile digests do not prove exchangeability.
    pub sampling_profile_digests_match: bool,
    /// BLAKE3 over the complete report with this field empty.
    pub report_digest: String,
}

impl VehicleTrajectoryCoverageReport {
    /// Rebuild all statistics from retained rows and validate the report digest.
    pub fn identity_is_valid(&self) -> bool {
        if self.schema_version != VEHICLE_TRAJECTORY_COVERAGE_SCHEMA
            || !digest_is_valid(&self.report_digest)
        {
            return false;
        }

        let Ok(expected) = derive_report(
            &self.target_id,
            self.horizon_ms,
            self.nominal_coverage,
            &self.predictor_artifact_digest,
            &self.set_construction_artifact_digest,
            &self.outcome_evaluator_digest,
            &self.calibration_sampling_profile_digest,
            &self.evaluation_sampling_profile_digest,
            &self.measure_id,
            &self.measure_unit_id,
            &self.calibration_observations,
            &self.evaluation_observations,
        ) else {
            return false;
        };

        let mut actual = self.clone();
        actual.report_digest.clear();
        let Ok(actual_bytes) = serde_json::to_vec(&actual) else {
            return false;
        };
        let Ok(expected_bytes) = serde_json::to_vec(&expected) else {
            return false;
        };
        actual_bytes == expected_bytes
            && blake3::hash(&actual_bytes).to_hex().to_string() == self.report_digest
    }
}

/// Evaluate a frozen split-conformal calibration sample against a disjoint
/// evaluation sample. Scores must be produced by the same frozen score
/// function and model/set identities declared in this report.
///
/// The finite-sample threshold uses the one-based order statistic
/// ceil((n_cal + 1) * nominal_coverage). If that rank exceeds n_cal, the
/// mathematically correct threshold is unbounded; this evaluator rejects the
/// input instead of silently substituting a finite value.
///
/// The resulting coverage is empirical. A conformal marginal-coverage
/// guarantee additionally requires exchangeability and a valid split protocol;
/// matching sampling-profile digests are not evidence that those assumptions
/// hold. Group results are diagnostics, not conditional-coverage guarantees.
#[allow(clippy::too_many_arguments)]
pub fn summarize_vehicle_trajectory_coverage(
    target_id: &str,
    horizon_ms: u64,
    nominal_coverage: f64,
    predictor_artifact_digest: &str,
    set_construction_artifact_digest: &str,
    outcome_evaluator_digest: &str,
    calibration_sampling_profile_digest: &str,
    evaluation_sampling_profile_digest: &str,
    measure_id: &str,
    measure_unit_id: &str,
    calibration_observations: &[TrajectoryCalibrationObservation],
    evaluation_observations: &[TrajectoryEvaluationObservation],
) -> Result<VehicleTrajectoryCoverageReport, TrajectoryCoverageError> {
    let mut report = derive_report(
        target_id,
        horizon_ms,
        nominal_coverage,
        predictor_artifact_digest,
        set_construction_artifact_digest,
        outcome_evaluator_digest,
        calibration_sampling_profile_digest,
        evaluation_sampling_profile_digest,
        measure_id,
        measure_unit_id,
        calibration_observations,
        evaluation_observations,
    )?;
    let bytes = serde_json::to_vec(&report)
        .map_err(|_| TrajectoryCoverageError::InvalidNonconformityScore)?;
    report.report_digest = blake3::hash(&bytes).to_hex().to_string();
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn derive_report(
    target_id: &str,
    horizon_ms: u64,
    nominal_coverage: f64,
    predictor_artifact_digest: &str,
    set_construction_artifact_digest: &str,
    outcome_evaluator_digest: &str,
    calibration_sampling_profile_digest: &str,
    evaluation_sampling_profile_digest: &str,
    measure_id: &str,
    measure_unit_id: &str,
    calibration_observations: &[TrajectoryCalibrationObservation],
    evaluation_observations: &[TrajectoryEvaluationObservation],
) -> Result<VehicleTrajectoryCoverageReport, TrajectoryCoverageError> {
    if [
        target_id,
        measure_id,
        measure_unit_id,
    ]
    .iter()
    .any(|value| value.trim().is_empty())
    {
        return Err(TrajectoryCoverageError::EmptyIdentity);
    }
    if horizon_ms == 0 {
        return Err(TrajectoryCoverageError::InvalidHorizon);
    }
    if !nominal_coverage.is_finite() || !(0.0..1.0).contains(&nominal_coverage) {
        return Err(TrajectoryCoverageError::InvalidNominalCoverage);
    }
    if ![
        predictor_artifact_digest,
        set_construction_artifact_digest,
        outcome_evaluator_digest,
        calibration_sampling_profile_digest,
        evaluation_sampling_profile_digest,
    ]
    .iter()
    .all(|digest| digest_is_valid(digest))
    {
        return Err(TrajectoryCoverageError::InvalidDigest);
    }
    if calibration_observations.is_empty() {
        return Err(TrajectoryCoverageError::EmptyCalibrationSet);
    }
    if evaluation_observations.is_empty() {
        return Err(TrajectoryCoverageError::EmptyEvaluationSet);
    }

    let mut calibration = calibration_observations.to_vec();
    calibration.sort_by(|left, right| {
        left.failure_family_id
            .cmp(&right.failure_family_id)
            .then_with(|| left.scenario_id.cmp(&right.scenario_id))
    });
    let mut evaluation = evaluation_observations.to_vec();
    evaluation.sort_by(|left, right| {
        left.failure_family_id
            .cmp(&right.failure_family_id)
            .then_with(|| left.scenario_id.cmp(&right.scenario_id))
    });

    let mut calibration_ids = BTreeSet::new();
    for row in &calibration {
        if row.failure_family_id.trim().is_empty() || row.scenario_id.trim().is_empty() {
            return Err(TrajectoryCoverageError::EmptyIdentity);
        }
        if !calibration_ids.insert(row.scenario_id.as_str()) {
            return Err(TrajectoryCoverageError::DuplicateScenarioId);
        }
        if !row.nonconformity_score.is_finite() || row.nonconformity_score < 0.0 {
            return Err(TrajectoryCoverageError::InvalidNonconformityScore);
        }
    }

    let mut evaluation_ids = BTreeSet::new();
    for row in &evaluation {
        if row.failure_family_id.trim().is_empty() || row.scenario_id.trim().is_empty() {
            return Err(TrajectoryCoverageError::EmptyIdentity);
        }
        if !evaluation_ids.insert(row.scenario_id.as_str()) {
            return Err(TrajectoryCoverageError::DuplicateScenarioId);
        }
        if calibration_ids.contains(row.scenario_id.as_str()) {
            return Err(TrajectoryCoverageError::CalibrationEvaluationOverlap);
        }
        if !row.nonconformity_score.is_finite() || row.nonconformity_score < 0.0 {
            return Err(TrajectoryCoverageError::InvalidNonconformityScore);
        }
        if !row.prediction_set_measure.is_finite() || row.prediction_set_measure < 0.0 {
            return Err(TrajectoryCoverageError::InvalidPredictionSetMeasure);
        }
    }

    let calibration_sample_count = calibration.len();
    let quantile_rank =
        (((calibration_sample_count as f64) + 1.0) * nominal_coverage).ceil() as usize;
    if quantile_rank == 0 || quantile_rank > calibration_sample_count {
        return Err(TrajectoryCoverageError::InsufficientCalibrationSamples);
    }
    let mut sorted_scores = calibration
        .iter()
        .map(|row| row.nonconformity_score)
        .collect::<Vec<_>>();
    sorted_scores.sort_by(f64::total_cmp);
    let calibration_threshold = sorted_scores[quantile_rank - 1];

    let mut evaluation_results = Vec::with_capacity(evaluation.len());
    let mut overall_covered = 0usize;
    let mut overall_measure_mean = 0.0;
    let mut all_set_measures = Vec::with_capacity(evaluation.len());
    let mut grouped: BTreeMap<&str, Vec<&TrajectoryEvaluationObservation>> = BTreeMap::new();

    for row in &evaluation {
        let covered = row.nonconformity_score <= calibration_threshold;
        if covered {
            overall_covered += 1;
        }
        overall_measure_mean = online_mean(
            overall_measure_mean,
            row.prediction_set_measure,
            all_set_measures.len() + 1,
        );
        all_set_measures.push(row.prediction_set_measure);
        evaluation_results.push(TrajectoryCoverageResult {
            failure_family_id: row.failure_family_id.clone(),
            scenario_id: row.scenario_id.clone(),
            nonconformity_score: row.nonconformity_score,
            prediction_set_measure: row.prediction_set_measure,
            covered,
        });
        grouped
            .entry(row.failure_family_id.as_str())
            .or_default()
            .push(row);
    }

    let evaluation_sample_count = evaluation.len();
    let empirical_coverage = overall_covered as f64 / evaluation_sample_count as f64;
    let p95_prediction_set_measure = nearest_rank_quantile(&mut all_set_measures, TRAJECTORY_COVERAGE_TARGET_P95);

    let mut family_results = Vec::with_capacity(grouped.len());
    for (family_id, family_rows) in grouped {
        let family_covered = family_rows
            .iter()
            .filter(|row| row.nonconformity_score <= calibration_threshold)
            .count();
        let mut family_measures = family_rows
            .iter()
            .map(|row| row.prediction_set_measure)
            .collect::<Vec<_>>();
        let family_mean = mean(&family_measures);
        let family_p95 = nearest_rank_quantile(
            &mut family_measures,
            TRAJECTORY_COVERAGE_TARGET_P95,
        );
        family_results.push(TrajectoryFamilyCoverage {
            failure_family_id: family_id.to_owned(),
            observation_count: family_rows.len(),
            covered_count: family_covered,
            empirical_coverage: family_covered as f64 / family_rows.len() as f64,
            mean_prediction_set_measure: family_mean,
            p95_prediction_set_measure: family_p95,
            underpowered_for_group_claim: family_rows.len() < MIN_FAMILY_COVERAGE_SAMPLES,
        });
    }

    let macro_family_coverage = mean(
        &family_results
            .iter()
            .map(|family| family.empirical_coverage)
            .collect::<Vec<_>>(),
    );
    let worst_family_coverage = family_results
        .iter()
        .map(|family| family.empirical_coverage)
        .fold(1.0_f64, f64::min);

    Ok(VehicleTrajectoryCoverageReport {
        schema_version: VEHICLE_TRAJECTORY_COVERAGE_SCHEMA.to_owned(),
        target_id: target_id.to_owned(),
        horizon_ms,
        nominal_coverage,
        predictor_artifact_digest: predictor_artifact_digest.to_owned(),
        set_construction_artifact_digest: set_construction_artifact_digest.to_owned(),
        outcome_evaluator_digest: outcome_evaluator_digest.to_owned(),
        calibration_sampling_profile_digest: calibration_sampling_profile_digest.to_owned(),
        evaluation_sampling_profile_digest: evaluation_sampling_profile_digest.to_owned(),
        measure_id: measure_id.to_owned(),
        measure_unit_id: measure_unit_id.to_owned(),
        calibration_observations: calibration,
        evaluation_observations: evaluation,
        evaluation_results,
        calibration_sample_count,
        evaluation_sample_count,
        calibration_quantile_rank: quantile_rank,
        calibration_threshold,
        empirical_coverage,
        signed_coverage_error: empirical_coverage - nominal_coverage,
        mean_prediction_set_measure: overall_measure_mean,
        p95_prediction_set_measure,
        macro_family_coverage,
        worst_family_coverage,
        family_results,
        sampling_profile_digests_match: calibration_sampling_profile_digest
            == evaluation_sampling_profile_digest,
        report_digest: String::new(),
    })
}

fn nearest_rank_quantile(values: &mut [f64], probability: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let rank = ((values.len() as f64) * probability).ceil() as usize;
    values[rank.saturating_sub(1).min(values.len() - 1)]
}

fn mean(values: &[f64]) -> f64 {
    let mut result = 0.0;
    for (index, value) in values.iter().enumerate() {
        result = online_mean(result, *value, index + 1);
    }
    result
}

fn online_mean(previous_mean: f64, value: f64, count: usize) -> f64 {
    if count == 1 {
        value
    } else {
        previous_mean + (value - previous_mean) / count as f64
    }
}

fn digest_is_valid(digest: &str) -> bool {
    digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration_rows() -> Vec<TrajectoryCalibrationObservation> {
        (0..19)
            .map(|index| TrajectoryCalibrationObservation {
                failure_family_id: if index % 2 == 0 { "pedestrian" } else { "cyclist" }.into(),
                scenario_id: format!("cal-{index:02}"),
                nonconformity_score: (index + 1) as f64 / 100.0,
            })
            .collect()
    }

    fn evaluation_rows() -> Vec<TrajectoryEvaluationObservation> {
        vec![
            TrajectoryEvaluationObservation {
                failure_family_id: "pedestrian".into(),
                scenario_id: "eval-ped-01".into(),
                nonconformity_score: 0.10,
                prediction_set_measure: 1.0,
            },
            TrajectoryEvaluationObservation {
                failure_family_id: "pedestrian".into(),
                scenario_id: "eval-ped-02".into(),
                nonconformity_score: 0.18,
                prediction_set_measure: 2.0,
            },
            TrajectoryEvaluationObservation {
                failure_family_id: "cyclist".into(),
                scenario_id: "eval-cyc-01".into(),
                nonconformity_score: 0.20,
                prediction_set_measure: 3.0,
            },
            TrajectoryEvaluationObservation {
                failure_family_id: "cyclist".into(),
                scenario_id: "eval-cyc-02".into(),
                nonconformity_score: 0.19,
                prediction_set_measure: 4.0,
            },
        ]
    }

    fn report() -> VehicleTrajectoryCoverageReport {
        summarize_vehicle_trajectory_coverage(
            "actor-trajectory-position",
            2_000,
            0.90,
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
            &"d".repeat(64),
            &"d".repeat(64),
            "mean-tube-radius",
            "meters",
            &calibration_rows(),
            &evaluation_rows(),
        )
        .unwrap()
    }

    #[test]
    fn uses_the_finite_sample_split_conformal_order_statistic() {
        let sample = report();
        assert_eq!(sample.calibration_sample_count, 19);
        assert_eq!(sample.calibration_quantile_rank, 18);
        assert!((sample.calibration_threshold - 0.18).abs() < f64::EPSILON);
        assert!(sample.identity_is_valid());
    }

    #[test]
    fn reports_coverage_and_set_sharpness_separately() {
        let sample = report();
        assert_eq!(sample.evaluation_sample_count, 4);
        assert!((sample.empirical_coverage - 0.5).abs() < f64::EPSILON);
        assert!((sample.signed_coverage_error + 0.4).abs() < f64::EPSILON);
        assert!((sample.mean_prediction_set_measure - 2.5).abs() < f64::EPSILON);
        assert!((sample.p95_prediction_set_measure - 4.0).abs() < f64::EPSILON);
        assert_eq!(sample.worst_family_coverage, 0.5);
        assert_eq!(sample.macro_family_coverage, 0.5);
    }

    #[test]
    fn insufficient_calibration_rejects_an_unbounded_finite_sample_quantile() {
        let mut calibration = calibration_rows();
        calibration.truncate(8);
        assert_eq!(
            summarize_vehicle_trajectory_coverage(
                "actor-trajectory-position",
                2_000,
                0.90,
                &"a".repeat(64),
                &"b".repeat(64),
                &"c".repeat(64),
                &"d".repeat(64),
                &"d".repeat(64),
                "mean-tube-radius",
                "meters",
                &calibration,
                &evaluation_rows(),
            )
            .unwrap_err(),
            TrajectoryCoverageError::InsufficientCalibrationSamples
        );
    }

    #[test]
    fn calibration_and_evaluation_ids_must_be_disjoint() {
        let mut evaluation = evaluation_rows();
        evaluation[0].scenario_id = "cal-00".into();
        assert_eq!(
            summarize_vehicle_trajectory_coverage(
                "actor-trajectory-position",
                2_000,
                0.90,
                &"a".repeat(64),
                &"b".repeat(64),
                &"c".repeat(64),
                &"d".repeat(64),
                &"d".repeat(64),
                "mean-tube-radius",
                "meters",
                &calibration_rows(),
                &evaluation,
            )
            .unwrap_err(),
            TrajectoryCoverageError::CalibrationEvaluationOverlap
        );
    }

    #[test]
    fn duplicate_ids_within_a_split_are_rejected() {
        let mut evaluation = evaluation_rows();
        evaluation[1].scenario_id = evaluation[0].scenario_id.clone();
        assert_eq!(
            summarize_vehicle_trajectory_coverage(
                "actor-trajectory-position",
                2_000,
                0.90,
                &"a".repeat(64),
                &"b".repeat(64),
                &"c".repeat(64),
                &"d".repeat(64),
                &"d".repeat(64),
                "mean-tube-radius",
                "meters",
                &calibration_rows(),
                &evaluation,
            )
            .unwrap_err(),
            TrajectoryCoverageError::DuplicateScenarioId
        );
    }

    #[test]
    fn family_results_retain_small_sample_warning() {
        let sample = report();
        assert!(sample
            .family_results
            .iter()
            .all(|family| family.underpowered_for_group_claim));
        assert_eq!(sample.family_results.len(), 2);
    }

    #[test]
    fn differing_sampling_profiles_are_reported_without_a_guarantee_claim() {
        let result = summarize_vehicle_trajectory_coverage(
            "actor-trajectory-position",
            2_000,
            0.90,
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
            &"d".repeat(64),
            &"e".repeat(64),
            "mean-tube-radius",
            "meters",
            &calibration_rows(),
            &evaluation_rows(),
        )
        .unwrap();
        assert!(!result.sampling_profile_digests_match);
        assert!(result.identity_is_valid());
    }

    #[test]
    fn report_digest_is_independent_of_input_order() {
        let mut calibration = calibration_rows();
        let mut evaluation = evaluation_rows();
        let first = summarize_vehicle_trajectory_coverage(
            "actor-trajectory-position",
            2_000,
            0.90,
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
            &"d".repeat(64),
            &"d".repeat(64),
            "mean-tube-radius",
            "meters",
            &calibration,
            &evaluation,
        )
        .unwrap();
        calibration.reverse();
        evaluation.reverse();
        let second = summarize_vehicle_trajectory_coverage(
            "actor-trajectory-position",
            2_000,
            0.90,
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
            &"d".repeat(64),
            &"d".repeat(64),
            "mean-tube-radius",
            "meters",
            &calibration,
            &evaluation,
        )
        .unwrap();
        assert_eq!(first.report_digest, second.report_digest);
    }

    #[test]
    fn detects_tampering_even_when_the_digest_is_recomputed() {
        let mut sample = report();
        sample.empirical_coverage = 1.0;
        sample.report_digest.clear();
        let bytes = serde_json::to_vec(&sample).unwrap();
        sample.report_digest = blake3::hash(&bytes).to_hex().to_string();
        assert!(!sample.identity_is_valid());
    }

    #[test]
    fn rejects_non_finite_and_negative_scores_and_set_measures() {
        let mut evaluation = evaluation_rows();
        evaluation[0].nonconformity_score = f64::NAN;
        let result = summarize_vehicle_trajectory_coverage(
            "actor-trajectory-position",
            2_000,
            0.90,
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
            &"d".repeat(64),
            &"d".repeat(64),
            "mean-tube-radius",
            "meters",
            &calibration_rows(),
            &evaluation,
        );
        assert_eq!(result.unwrap_err(), TrajectoryCoverageError::InvalidNonconformityScore);

        let mut evaluation = evaluation_rows();
        evaluation[0].prediction_set_measure = -1.0;
        let result = summarize_vehicle_trajectory_coverage(
            "actor-trajectory-position",
            2_000,
            0.90,
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
            &"d".repeat(64),
            &"d".repeat(64),
            "mean-tube-radius",
            "meters",
            &calibration_rows(),
            &evaluation,
        );
        assert_eq!(result.unwrap_err(), TrajectoryCoverageError::InvalidPredictionSetMeasure);
    }
}
