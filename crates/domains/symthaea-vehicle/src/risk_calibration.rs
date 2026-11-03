// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Family-aware calibration metrics for vehicle event-risk forecasts.
//!
//! This module evaluates probabilities against fully observed outcomes for one
//! event definition and one forecast horizon. It does not estimate event
//! prevalence from an adversarial scenario corpus and does not authenticate
//! labels or predictor artifacts. Censored episodes are retained and counted,
//! but never treated as negative outcomes.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub const VEHICLE_RISK_CALIBRATION_SCHEMA: &str =
    "symthaea.vehicle.risk-calibration.v0.1";
pub const DEFAULT_RISK_CALIBRATION_BINS: usize = 10;
pub const MIN_RISK_CALIBRATION_BINS: usize = 2;
pub const MAX_RISK_CALIBRATION_BINS: usize = 100;
pub const RISK_LOG_LOSS_EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskForecastOutcome {
    /// Event status is fully observed at the declared forecast horizon.
    Occurred,
    /// Event absence is fully observed through the declared horizon.
    NotOccurred,
    /// The episode ended, was interrupted, or lost observability before the
    /// declared horizon. This is not a negative label.
    Censored,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskForecastObservation {
    pub failure_family_id: String,
    pub scenario_id: String,
    /// Probability of the declared event occurring within the report horizon.
    pub predicted_probability: f64,
    pub outcome: RiskForecastOutcome,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskCalibrationBin {
    pub bin_index: usize,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub observation_count: usize,
    pub mean_predicted_probability: Option<f64>,
    pub observed_event_frequency: Option<f64>,
    pub absolute_calibration_gap: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskFamilyCalibration {
    pub failure_family_id: String,
    pub observed_count: usize,
    pub event_count: usize,
    pub event_rate: f64,
    pub brier_score: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RiskCalibrationError {
    EmptyObservationSet,
    NoObservedOutcomes,
    EmptyIdentity,
    InvalidDigest,
    InvalidHorizon,
    InvalidBinCount,
    InvalidProbability,
    DuplicateScenarioId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VehicleRiskCalibrationReport {
    pub schema_version: String,
    pub event_definition_id: String,
    pub horizon_ms: u64,
    pub predictor_artifact_digest: String,
    pub outcome_evaluator_digest: String,
    pub sampling_profile_digest: String,
    pub calibration_bin_count: usize,
    /// Canonically ordered raw observations, including censored episodes.
    pub observations: Vec<RiskForecastObservation>,
    pub observed_count: usize,
    pub censored_count: usize,
    pub event_count: usize,
    pub event_rate: f64,
    /// Mean squared probability error over fully observed labels only.
    pub brier_score: f64,
    /// Bernoulli log loss over fully observed labels only, clipped at 1e-12.
    pub log_loss: f64,
    /// Equal-width reliability-bin ECE. Publish bin counts alongside it.
    pub expected_calibration_error: f64,
    /// Largest absolute observed-vs-predicted gap among non-empty bins.
    pub maximum_calibration_error: f64,
    /// Equal-weight mean of per-family Brier scores.
    pub macro_family_brier_score: f64,
    /// Highest per-family Brier score; exposes a weak family hidden by the mean.
    pub worst_family_brier_score: f64,
    pub bins: Vec<RiskCalibrationBin>,
    pub family_results: Vec<RiskFamilyCalibration>,
    /// BLAKE3 over all retained inputs and metrics with this field empty.
    pub report_digest: String,
}

impl VehicleRiskCalibrationReport {
    /// Reconstruct metrics from retained observations and validate the digest.
    pub fn identity_is_valid(&self) -> bool {
        if self.schema_version != VEHICLE_RISK_CALIBRATION_SCHEMA
            || self.report_digest.len() != 64
            || !self.report_digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return false;
        }

        let Ok(expected) = derive_report(
            &self.event_definition_id,
            self.horizon_ms,
            &self.predictor_artifact_digest,
            &self.outcome_evaluator_digest,
            &self.sampling_profile_digest,
            self.calibration_bin_count,
            &self.observations,
        ) else {
            return false;
        };

        if expected.observed_count != self.observed_count
            || expected.censored_count != self.censored_count
            || expected.event_count != self.event_count
            || !approximately_equal(expected.event_rate, self.event_rate)
            || !approximately_equal(expected.brier_score, self.brier_score)
            || !approximately_equal(expected.log_loss, self.log_loss)
            || !approximately_equal(
                expected.expected_calibration_error,
                self.expected_calibration_error,
            )
            || !approximately_equal(
                expected.maximum_calibration_error,
                self.maximum_calibration_error,
            )
            || !approximately_equal(
                expected.macro_family_brier_score,
                self.macro_family_brier_score,
            )
            || !approximately_equal(
                expected.worst_family_brier_score,
                self.worst_family_brier_score,
            )
            || expected.bins != self.bins
            || expected.family_results != self.family_results
            || expected.observations != self.observations
        {
            return false;
        }

        let mut report = self.clone();
        let expected_digest = report.report_digest.clone();
        report.report_digest.clear();
        serde_json::to_vec(&report)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string() == expected_digest)
            .unwrap_or(false)
    }
}

/// Evaluate event-probability forecasts for one named event and horizon.
///
/// Only `Occurred` and `NotOccurred` labels enter proper scoring rules.
/// `Censored` records remain in the report and denominator disclosure, but
/// are excluded from scores rather than mislabeled as no-event outcomes.
///
/// Family-macro Brier gives each declared failure family equal weight so a
/// large family of sibling variants cannot dominate the summary.
pub fn summarize_vehicle_risk_forecasts(
    event_definition_id: &str,
    horizon_ms: u64,
    predictor_artifact_digest: &str,
    outcome_evaluator_digest: &str,
    sampling_profile_digest: &str,
    calibration_bin_count: usize,
    observations: &[RiskForecastObservation],
) -> Result<VehicleRiskCalibrationReport, RiskCalibrationError> {
    let mut report = derive_report(
        event_definition_id,
        horizon_ms,
        predictor_artifact_digest,
        outcome_evaluator_digest,
        sampling_profile_digest,
        calibration_bin_count,
        observations,
    )?;
    let bytes = serde_json::to_vec(&report).map_err(|_| RiskCalibrationError::InvalidProbability)?;
    report.report_digest = blake3::hash(&bytes).to_hex().to_string();
    Ok(report)
}

fn derive_report(
    event_definition_id: &str,
    horizon_ms: u64,
    predictor_artifact_digest: &str,
    outcome_evaluator_digest: &str,
    sampling_profile_digest: &str,
    calibration_bin_count: usize,
    observations: &[RiskForecastObservation],
) -> Result<VehicleRiskCalibrationReport, RiskCalibrationError> {
    if observations.is_empty() {
        return Err(RiskCalibrationError::EmptyObservationSet);
    }
    if event_definition_id.trim().is_empty() {
        return Err(RiskCalibrationError::EmptyIdentity);
    }
    if horizon_ms == 0 {
        return Err(RiskCalibrationError::InvalidHorizon);
    }
    if !(MIN_RISK_CALIBRATION_BINS..=MAX_RISK_CALIBRATION_BINS)
        .contains(&calibration_bin_count)
    {
        return Err(RiskCalibrationError::InvalidBinCount);
    }
    if !digest_is_valid(predictor_artifact_digest)
        || !digest_is_valid(outcome_evaluator_digest)
        || !digest_is_valid(sampling_profile_digest)
    {
        return Err(RiskCalibrationError::InvalidDigest);
    }

    let mut canonical = observations.to_vec();
    canonical.sort_by(|left, right| {
        left.failure_family_id
            .cmp(&right.failure_family_id)
            .then_with(|| left.scenario_id.cmp(&right.scenario_id))
    });

    let mut scenario_ids = BTreeSet::new();
    for observation in &canonical {
        if observation.failure_family_id.trim().is_empty()
            || observation.scenario_id.trim().is_empty()
        {
            return Err(RiskCalibrationError::EmptyIdentity);
        }
        if !scenario_ids.insert(observation.scenario_id.as_str()) {
            return Err(RiskCalibrationError::DuplicateScenarioId);
        }
        if !observation.predicted_probability.is_finite()
            || !(0.0..=1.0).contains(&observation.predicted_probability)
        {
            return Err(RiskCalibrationError::InvalidProbability);
        }
    }

    let observed = canonical
        .iter()
        .filter(|observation| observation.outcome != RiskForecastOutcome::Censored)
        .collect::<Vec<_>>();
    let censored_count = canonical.len() - observed.len();
    if observed.is_empty() {
        return Err(RiskCalibrationError::NoObservedOutcomes);
    }

    let observed_count = observed.len();
    let event_count = observed
        .iter()
        .filter(|observation| observation.outcome == RiskForecastOutcome::Occurred)
        .count();
    let event_rate = event_count as f64 / observed_count as f64;

    let mut brier_sum = 0.0;
    let mut log_loss_sum = 0.0;
    let mut bins = (0..calibration_bin_count)
        .map(|bin_index| RiskCalibrationBin {
            bin_index,
            lower_bound: bin_index as f64 / calibration_bin_count as f64,
            upper_bound: (bin_index + 1) as f64 / calibration_bin_count as f64,
            observation_count: 0,
            mean_predicted_probability: None,
            observed_event_frequency: None,
            absolute_calibration_gap: None,
        })
        .collect::<Vec<_>>();

    let mut grouped: BTreeMap<&str, Vec<&RiskForecastObservation>> = BTreeMap::new();
    for observation in &observed {
        let target = if observation.outcome == RiskForecastOutcome::Occurred {
            1.0
        } else {
            0.0
        };
        let probability = observation.predicted_probability;
        brier_sum += (probability - target).powi(2);
        let clipped = probability.clamp(RISK_LOG_LOSS_EPSILON, 1.0 - RISK_LOG_LOSS_EPSILON);
        log_loss_sum += -(target * clipped.ln() + (1.0 - target) * (1.0 - clipped).ln());

        let bin_index = ((probability * calibration_bin_count as f64) as usize)
            .min(calibration_bin_count - 1);
        let bin = &mut bins[bin_index];
        bin.observation_count += 1;
        bin.mean_predicted_probability =
            Some(bin.mean_predicted_probability.unwrap_or(0.0) + probability);
        bin.observed_event_frequency =
            Some(bin.observed_event_frequency.unwrap_or(0.0) + target);

        grouped
            .entry(observation.failure_family_id.as_str())
            .or_default()
            .push(observation);
    }

    let mut expected_calibration_error = 0.0;
    let mut maximum_calibration_error: f64 = 0.0;
    for bin in &mut bins {
        if bin.observation_count == 0 {
            continue;
        }
        let count = bin.observation_count as f64;
        let mean_probability = bin.mean_predicted_probability.unwrap_or(0.0) / count;
        let event_frequency = bin.observed_event_frequency.unwrap_or(0.0) / count;
        let gap = (mean_probability - event_frequency).abs();
        bin.mean_predicted_probability = Some(mean_probability);
        bin.observed_event_frequency = Some(event_frequency);
        bin.absolute_calibration_gap = Some(gap);
        expected_calibration_error += gap * (count / observed_count as f64);
        maximum_calibration_error = maximum_calibration_error.max(gap);
    }

    let mut family_results = Vec::with_capacity(grouped.len());
    for (family_id, family_observations) in grouped {
        let family_event_count = family_observations
            .iter()
            .filter(|observation| observation.outcome == RiskForecastOutcome::Occurred)
            .count();
        let family_observed_count = family_observations.len();
        let family_brier = family_observations
            .iter()
            .map(|observation| {
                let target = if observation.outcome == RiskForecastOutcome::Occurred {
                    1.0
                } else {
                    0.0
                };
                (observation.predicted_probability - target).powi(2)
            })
            .sum::<f64>()
            / family_observed_count as f64;
        family_results.push(RiskFamilyCalibration {
            failure_family_id: family_id.to_owned(),
            observed_count: family_observed_count,
            event_count: family_event_count,
            event_rate: family_event_count as f64 / family_observed_count as f64,
            brier_score: family_brier,
        });
    }

    let macro_family_brier_score = family_results
        .iter()
        .map(|family| family.brier_score)
        .sum::<f64>()
        / family_results.len() as f64;
    let worst_family_brier_score = family_results
        .iter()
        .map(|family| family.brier_score)
        .fold(0.0, f64::max);

    Ok(VehicleRiskCalibrationReport {
        schema_version: VEHICLE_RISK_CALIBRATION_SCHEMA.into(),
        event_definition_id: event_definition_id.into(),
        horizon_ms,
        predictor_artifact_digest: predictor_artifact_digest.into(),
        outcome_evaluator_digest: outcome_evaluator_digest.into(),
        sampling_profile_digest: sampling_profile_digest.into(),
        calibration_bin_count,
        observations: canonical,
        observed_count,
        censored_count,
        event_count,
        event_rate,
        brier_score: brier_sum / observed_count as f64,
        log_loss: log_loss_sum / observed_count as f64,
        expected_calibration_error,
        maximum_calibration_error,
        macro_family_brier_score,
        worst_family_brier_score,
        bins,
        family_results,
        report_digest: String::new(),
    })
}

fn digest_is_valid(digest: &str) -> bool {
    digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn approximately_equal(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 1e-10 * left.abs().max(right.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        family: &str,
        scenario: &str,
        probability: f64,
        outcome: RiskForecastOutcome,
    ) -> RiskForecastObservation {
        RiskForecastObservation {
            failure_family_id: family.into(),
            scenario_id: scenario.into(),
            predicted_probability: probability,
            outcome,
        }
    }

    fn sample() -> Vec<RiskForecastObservation> {
        vec![
            // Family A: 1/4 observed events in the 0.25 forecast bin.
            observation("crossing-a", "a-1", 0.25, RiskForecastOutcome::Occurred),
            observation("crossing-a", "a-2", 0.25, RiskForecastOutcome::NotOccurred),
            observation("crossing-a", "a-3", 0.25, RiskForecastOutcome::NotOccurred),
            observation("crossing-a", "a-4", 0.25, RiskForecastOutcome::NotOccurred),
            // Family B: 3/4 observed events in the 0.75 forecast bin.
            observation("crossing-b", "b-1", 0.75, RiskForecastOutcome::Occurred),
            observation("crossing-b", "b-2", 0.75, RiskForecastOutcome::Occurred),
            observation("crossing-b", "b-3", 0.75, RiskForecastOutcome::Occurred),
            observation("crossing-b", "b-4", 0.75, RiskForecastOutcome::NotOccurred),
        ]
    }

    fn summarize(
        observations: &[RiskForecastObservation],
    ) -> Result<VehicleRiskCalibrationReport, RiskCalibrationError> {
        summarize_vehicle_risk_forecasts(
            "pedestrian-contact-within-horizon",
            2_000,
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
            DEFAULT_RISK_CALIBRATION_BINS,
            observations,
        )
    }

    #[test]
    fn perfect_probability_grouping_has_zero_ece_and_correct_brier() {
        let report = summarize(&sample()).unwrap();
        assert!(report.expected_calibration_error.abs() < 1e-12);
        assert!((report.brier_score - 0.1875).abs() < 1e-12);
        assert_eq!(report.event_count, 2);
        assert_eq!(report.observed_count, 8);
        assert!(report.identity_is_valid());
    }

    #[test]
    fn overconfident_forecasts_increase_proper_scoring_losses() {
        let overconfident = vec![
            observation("crossing-a", "a-1", 0.01, RiskForecastOutcome::Occurred),
            observation("crossing-a", "a-2", 0.99, RiskForecastOutcome::NotOccurred),
            observation("crossing-b", "b-1", 0.01, RiskForecastOutcome::Occurred),
            observation("crossing-b", "b-2", 0.99, RiskForecastOutcome::NotOccurred),
        ];
        let report = summarize(&overconfident).unwrap();
        assert!(report.brier_score > 0.9);
        assert!(report.log_loss > 3.0);
        assert!(report.expected_calibration_error > 0.9);
    }

    #[test]
    fn censored_episodes_are_counted_but_never_labeled_no_event() {
        let mut observations = sample();
        observations.push(observation(
            "crossing-b",
            "b-censored",
            0.99,
            RiskForecastOutcome::Censored,
        ));
        let report = summarize(&observations).unwrap();
        assert_eq!(report.censored_count, 1);
        assert_eq!(report.observed_count, 8);
        assert_eq!(report.event_count, 4);
        assert!(report.identity_is_valid());
    }

    #[test]
    fn family_macro_brier_prevents_large_family_dominance() {
        let mut observations = sample();
        for index in 0..20 {
            observations.push(observation(
                "crossing-a",
                &format!("a-extra-{index}"),
                0.1,
                RiskForecastOutcome::NotOccurred,
            ));
        }
        let report = summarize(&observations).unwrap();
        let family_a = report
            .family_results
            .iter()
            .find(|family| family.failure_family_id == "crossing-a")
            .unwrap();
        let family_b = report
            .family_results
            .iter()
            .find(|family| family.failure_family_id == "crossing-b")
            .unwrap();
        assert_ne!(family_a.brier_score, family_b.brier_score);
        assert!(
            (report.macro_family_brier_score - (family_a.brier_score + family_b.brier_score) / 2.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn input_order_does_not_change_report_digest() {
        let observations = sample();
        let mut reversed = observations.clone();
        reversed.reverse();
        let first = summarize(&observations).unwrap();
        let second = summarize(&reversed).unwrap();
        assert_eq!(first.report_digest, second.report_digest);
    }

    #[test]
    fn rejects_invalid_probability_digest_horizon_and_bin_count() {
        let mut invalid = sample();
        invalid[0].predicted_probability = 1.1;
        assert_eq!(
            summarize(&invalid).unwrap_err(),
            RiskCalibrationError::InvalidProbability
        );
        assert_eq!(
            summarize_vehicle_risk_forecasts(
                "",
                2_000,
                &"a".repeat(64),
                &"b".repeat(64),
                &"c".repeat(64),
                10,
                &sample(),
            )
            .unwrap_err(),
            RiskCalibrationError::EmptyIdentity
        );
        assert_eq!(
            summarize_vehicle_risk_forecasts(
                "event",
                0,
                &"a".repeat(64),
                &"b".repeat(64),
                &"c".repeat(64),
                10,
                &sample(),
            )
            .unwrap_err(),
            RiskCalibrationError::InvalidHorizon
        );
        assert_eq!(
            summarize_vehicle_risk_forecasts(
                "event",
                2_000,
                &"a".repeat(64),
                &"b".repeat(64),
                &"c".repeat(64),
                1,
                &sample(),
            )
            .unwrap_err(),
            RiskCalibrationError::InvalidBinCount
        );
        assert_eq!(
            summarize_vehicle_risk_forecasts(
                "event",
                2_000,
                "not-a-digest",
                &"b".repeat(64),
                &"c".repeat(64),
                10,
                &sample(),
            )
            .unwrap_err(),
            RiskCalibrationError::InvalidDigest
        );
    }

    #[test]
    fn rejects_duplicate_scenarios_and_all_censored_inputs() {
        let mut duplicate = sample();
        duplicate.push(duplicate[0].clone());
        assert_eq!(
            summarize(&duplicate).unwrap_err(),
            RiskCalibrationError::DuplicateScenarioId
        );
        let censored = vec![
            observation("crossing-a", "a-1", 0.5, RiskForecastOutcome::Censored),
            observation("crossing-b", "b-1", 0.5, RiskForecastOutcome::Censored),
        ];
        assert_eq!(
            summarize(&censored).unwrap_err(),
            RiskCalibrationError::NoObservedOutcomes
        );
    }

    #[test]
    fn report_validator_detects_metrics_and_family_tampering() {
        let mut report = summarize(&sample()).unwrap();
        report.brier_score = 0.99;
        report.report_digest.clear();
        let bytes = serde_json::to_vec(&report).unwrap();
        report.report_digest = blake3::hash(&bytes).to_hex().to_string();
        assert!(!report.identity_is_valid());
    }

    #[test]
    fn empty_reliability_bins_are_retained_explicitly() {
        let report = summarize(&sample()).unwrap();
        assert_eq!(report.bins.len(), DEFAULT_RISK_CALIBRATION_BINS);
        assert!(report.bins.iter().any(|bin| bin.observation_count == 0));
        assert!(report
            .bins
            .iter()
            .filter(|bin| bin.observation_count == 0)
            .all(|bin| bin.mean_predicted_probability.is_none()
                && bin.observed_event_frequency.is_none()
                && bin.absolute_calibration_gap.is_none()));
    }
}
