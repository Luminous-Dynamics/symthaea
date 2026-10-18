// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Held-out residual-quantile intervals for the synthetic spatial FDM harness.
//!
//! This module reserves three disjoint roles: discovery observations fit a
//! predictor, calibration targets select an absolute-residual quantile, and
//! confirmation targets evaluate interval coverage. It performs no machine
//! I/O. Spatial dependence can violate exchangeability, so this benchmark
//! reports empirical coverage and does not claim a formal conformal guarantee.

use crate::crypto_digest::sha256;
use crate::fdm_spatial_benchmark::{
    NormalizedPosition, SpatialConfirmationPoint, SpatialEvidenceClass, SpatialObservation,
    SpatialPolicy, SyntheticSpatialCorpus, demo_synthetic_spatial_corpus, synthetic_response,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const FDM_SPATIAL_INTERVAL_SCHEMA: &str = "symthaea.fdm-spatial-interval-synthetic.v1";
const RESPONSE_FUNCTION_VERSION: &str = "smooth-quadratic-v1";

/// A target reserved exclusively for choosing an interval residual quantile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialIntervalCalibrationPoint {
    pub point_id: String,
    pub position: NormalizedPosition,
    /// Synthetic oracle value in millimetres; never passed to the predictor.
    pub oracle_signed_error_mm: f64,
}

/// Three-way split: fit, interval calibration, and final confirmation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticSpatialIntervalCorpus {
    pub schema_version: String,
    pub evidence_class: SpatialEvidenceClass,
    pub seed: u64,
    pub response_function_version: String,
    pub discovery: Vec<SpatialObservation>,
    pub calibration: Vec<SpatialIntervalCalibrationPoint>,
    pub confirmation: Vec<SpatialConfirmationPoint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialIntervalStatus {
    Predicted,
    Unknown,
    OutOfDomain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialIntervalPointResult {
    pub point_id: String,
    pub position: NormalizedPosition,
    pub status: SpatialIntervalStatus,
    pub predicted_signed_error_mm: Option<f64>,
    pub lower_bound_mm: Option<f64>,
    pub upper_bound_mm: Option<f64>,
    /// Whether the final held-out oracle lies inside the reported interval.
    pub contains_oracle: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialIntervalMetrics {
    pub discovery_count: usize,
    pub calibration_point_count: usize,
    pub calibration_residual_count: usize,
    pub calibration_unknown_count: usize,
    /// One-based order-statistic rank used for the residual quantile.
    pub quantile_rank: usize,
    pub nominal_coverage: f64,
    pub confirmation_count: usize,
    pub predicted_count: usize,
    pub unknown_count: usize,
    pub out_of_domain_count: usize,
    /// Predicted / all confirmation points; unknown and out-of-domain points
    /// are not silently discarded from this metric.
    pub coverage: f64,
    /// Empirical interval coverage among predicted confirmation points only.
    pub empirical_interval_coverage: Option<f64>,
    pub mean_interval_width_mm: Option<f64>,
    pub half_width_mm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialIntervalPolicyReport {
    pub policy: SpatialPolicy,
    pub evidence_class: SpatialEvidenceClass,
    pub corpus_digest_sha256: String,
    pub metrics: SpatialIntervalMetrics,
    pub points: Vec<SpatialIntervalPointResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpatialIntervalError {
    InvalidCorpus(Vec<String>),
    InvalidPolicy(&'static str),
    InvalidMiscoverage,
    InsufficientCalibration { usable: usize, required_rank: usize },
    SerializationFailure,
}

impl std::fmt::Display for SpatialIntervalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCorpus(issues) => write!(
                formatter,
                "invalid synthetic spatial interval corpus: {}",
                issues.join("; ")
            ),
            Self::InvalidPolicy(reason) => write!(formatter, "invalid spatial policy: {reason}"),
            Self::InvalidMiscoverage => {
                formatter.write_str("nominal miscoverage alpha must be finite and in (0, 1)")
            }
            Self::InsufficientCalibration { usable, required_rank } => write!(
                formatter,
                "insufficient usable calibration residuals: {usable} available, rank {required_rank} required"
            ),
            Self::SerializationFailure => {
                formatter.write_str("could not serialize the validated interval corpus")
            }
        }
    }
}

impl std::error::Error for SpatialIntervalError {}

/// Generate a deterministic synthetic corpus with a disjoint 16-point
/// calibration grid. Confirmation targets are inherited unchanged from v1.
pub fn demo_synthetic_spatial_interval_corpus(seed: u64) -> SyntheticSpatialIntervalCorpus {
    let base: SyntheticSpatialCorpus = demo_synthetic_spatial_corpus(seed);
    SyntheticSpatialIntervalCorpus {
        schema_version: FDM_SPATIAL_INTERVAL_SCHEMA.to_owned(),
        evidence_class: SpatialEvidenceClass::Synthetic,
        seed,
        response_function_version: RESPONSE_FUNCTION_VERSION.to_owned(),
        discovery: base.discovery,
        calibration: demo_interval_calibration_points(),
        confirmation: base.confirmation,
    }
}

fn demo_interval_calibration_points() -> Vec<SpatialIntervalCalibrationPoint> {
    let coordinates = [0.15, 0.30, 0.70, 0.85];
    let mut points = Vec::with_capacity(coordinates.len() * coordinates.len());
    let mut index = 0usize;
    for y in coordinates {
        for x in coordinates {
            let position = NormalizedPosition { x, y };
            points.push(SpatialIntervalCalibrationPoint {
                point_id: format!("calibration-{index:03}"),
                position,
                oracle_signed_error_mm: synthetic_response(position),
            });
            index += 1;
        }
    }
    points
}

/// Fail closed on schema, split overlap, identity collision, or a mutated
/// seed-generated synthetic oracle. This validator does not accept physical
/// measurements; a separate schema and provenance path is required for those.
pub fn validate_spatial_interval_corpus(
    corpus: &SyntheticSpatialIntervalCorpus,
) -> Result<(), SpatialIntervalError> {
    let mut issues = Vec::new();
    if corpus.schema_version != FDM_SPATIAL_INTERVAL_SCHEMA {
        issues.push("unsupported interval corpus schema version".to_owned());
    }
    if corpus.response_function_version != RESPONSE_FUNCTION_VERSION {
        issues.push("unsupported synthetic response function version".to_owned());
    }
    if corpus.evidence_class != SpatialEvidenceClass::Synthetic {
        issues.push("interval harness accepts synthetic evidence only".to_owned());
    }

    let expected = demo_synthetic_spatial_interval_corpus(corpus.seed);
    if corpus.discovery != expected.discovery {
        issues.push("discovery records differ from the frozen seed-generated corpus".to_owned());
    }
    if corpus.calibration != expected.calibration {
        issues.push("calibration records differ from the frozen calibration split".to_owned());
    }
    if corpus.confirmation != expected.confirmation {
        issues.push("confirmation records differ from the frozen held-out oracle".to_owned());
    }
    if corpus.discovery.is_empty() || corpus.calibration.is_empty() || corpus.confirmation.is_empty()
    {
        issues.push("discovery, calibration, and confirmation splits must all be non-empty".to_owned());
    }

    let mut ids = HashSet::new();
    let mut positions: Vec<NormalizedPosition> = Vec::new();
    for observation in &corpus.discovery {
        if observation.observation_id.trim().is_empty() || !ids.insert(observation.observation_id.as_str()) {
            issues.push(format!("empty or duplicate discovery ID: {}", observation.observation_id));
        }
        validate_position(observation.position, &observation.observation_id, &mut issues);
        if !observation.signed_error_mm.is_finite() {
            issues.push(format!("non-finite discovery target: {}", observation.observation_id));
        }
        add_unique_position(observation.position, &observation.observation_id, &mut positions, &mut issues);
    }
    for point in &corpus.calibration {
        if point.point_id.trim().is_empty() || !ids.insert(point.point_id.as_str()) {
            issues.push(format!("empty or duplicate calibration ID: {}", point.point_id));
        }
        validate_position(point.position, &point.point_id, &mut issues);
        if !point.oracle_signed_error_mm.is_finite() {
            issues.push(format!("non-finite calibration target: {}", point.point_id));
        }
        add_unique_position(point.position, &point.point_id, &mut positions, &mut issues);
    }
    for point in &corpus.confirmation {
        if point.point_id.trim().is_empty() || !ids.insert(point.point_id.as_str()) {
            issues.push(format!("empty or cross-split duplicate confirmation ID: {}", point.point_id));
        }
        validate_position(point.position, &point.point_id, &mut issues);
        if !point.oracle_signed_error_mm.is_finite() {
            issues.push(format!("non-finite confirmation target: {}", point.point_id));
        }
        add_unique_position(point.position, &point.point_id, &mut positions, &mut issues);
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(SpatialIntervalError::InvalidCorpus(issues))
    }
}

fn validate_position(position: NormalizedPosition, id: &str, issues: &mut Vec<String>) {
    if !position.x.is_finite()
        || !position.y.is_finite()
        || !(0.0..=1.0).contains(&position.x)
        || !(0.0..=1.0).contains(&position.y)
    {
        issues.push(format!("position outside finite normalized [0,1] domain: {id}"));
    }
}

fn add_unique_position(
    position: NormalizedPosition,
    id: &str,
    positions: &mut Vec<NormalizedPosition>,
    issues: &mut Vec<String>,
) {
    if positions
        .iter()
        .any(|existing| existing.x == position.x && existing.y == position.y)
    {
        issues.push(format!("position overlaps another split: {id}"));
    }
    positions.push(position);
}

fn validate_policy(policy: &SpatialPolicy) -> Result<(), SpatialIntervalError> {
    match policy {
        SpatialPolicy::GlobalMean | SpatialPolicy::NearestNeighbor => Ok(()),
        SpatialPolicy::InverseDistanceWeighted {
            neighbors,
            max_distance,
        } => {
            if *neighbors == 0 {
                return Err(SpatialIntervalError::InvalidPolicy(
                    "inverse-distance neighbour count must be positive",
                ));
            }
            if !max_distance.is_finite() || *max_distance <= 0.0 {
                return Err(SpatialIntervalError::InvalidPolicy(
                    "maximum prediction distance must be finite and positive",
                ));
            }
            Ok(())
        }
    }
}

fn inside_observed_rectangle(
    observations: &[SpatialObservation],
    target: NormalizedPosition,
) -> bool {
    if observations.is_empty() {
        return false;
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for observation in observations {
        min_x = min_x.min(observation.position.x);
        min_y = min_y.min(observation.position.y);
        max_x = max_x.max(observation.position.x);
        max_y = max_y.max(observation.position.y);
    }
    target.x >= min_x && target.x <= max_x && target.y >= min_y && target.y <= max_y
}

fn distance(left: NormalizedPosition, right: NormalizedPosition) -> f64 {
    let dx = left.x - right.x;
    let dy = left.y - right.y;
    (dx * dx + dy * dy).sqrt()
}

/// Equivalent prediction semantics to the v1 harness, duplicated locally so
/// the first interval increment can be reviewed independently. A parity test
/// compares these values with the v1 evaluator on its held-out fixture points.
fn predict(
    policy: &SpatialPolicy,
    observations: &[SpatialObservation],
    target: NormalizedPosition,
) -> Option<f64> {
    if observations.is_empty() {
        return None;
    }
    match policy {
        SpatialPolicy::GlobalMean => Some(
            observations
                .iter()
                .map(|observation| observation.signed_error_mm)
                .sum::<f64>()
                / observations.len() as f64,
        ),
        SpatialPolicy::NearestNeighbor => {
            if !inside_observed_rectangle(observations, target) {
                return None;
            }
            nearest(observations, target).map(|observation| observation.signed_error_mm)
        }
        SpatialPolicy::InverseDistanceWeighted { neighbors, max_distance } => {
            if !inside_observed_rectangle(observations, target) {
                return None;
            }
            let mut ranked: Vec<(&SpatialObservation, f64)> = observations
                .iter()
                .map(|observation| (observation, distance(observation.position, target)))
                .filter(|(_, distance)| *distance <= *max_distance)
                .collect();
            ranked.sort_by(|(left, left_distance), (right, right_distance)| {
                left_distance
                    .total_cmp(right_distance)
                    .then_with(|| left.observation_id.cmp(&right.observation_id))
            });
            ranked.truncate(*neighbors);
            if let Some((observation, distance)) = ranked.first() {
                if *distance == 0.0 {
                    return Some(observation.signed_error_mm);
                }
            }
            if ranked.is_empty() {
                return None;
            }
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;
            for (observation, distance) in ranked {
                let weight = 1.0 / (distance * distance);
                weighted_sum += weight * observation.signed_error_mm;
                weight_total += weight;
            }
            (weight_total.is_finite() && weight_total > 0.0)
                .then_some(weighted_sum / weight_total)
        }
    }
}

fn nearest(
    observations: &[SpatialObservation],
    target: NormalizedPosition,
) -> Option<&SpatialObservation> {
    observations.iter().min_by(|left, right| {
        distance(left.position, target)
            .total_cmp(&distance(right.position, target))
            .then_with(|| left.observation_id.cmp(&right.observation_id))
    })
}

/// Domain-separated SHA-256 identity of the exact validated three-way split.
pub fn digest_spatial_interval_corpus(
    corpus: &SyntheticSpatialIntervalCorpus,
) -> Result<String, SpatialIntervalError> {
    validate_spatial_interval_corpus(corpus)?;
    let serialized = serde_json::to_vec(corpus).map_err(|_| SpatialIntervalError::SerializationFailure)?;
    let domain = b"symthaea.fdm-spatial-interval-synthetic-corpus.v1\0";
    let mut bytes = Vec::with_capacity(domain.len() + serialized.len());
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&serialized);
    Ok(sha256(&bytes).to_hex())
}

/// Evaluate one policy with absolute-residual quantile intervals.
///
/// alpha is nominal miscoverage (for example, 0.10 means a nominal 90%
/// interval). The finite-sample order-statistic rank is ceil((n+1)*(1-alpha)).
/// If the required rank exceeds usable calibration residuals, evaluation fails
/// closed instead of silently substituting a smaller, under-covering quantile.
pub fn evaluate_spatial_intervals(
    corpus: &SyntheticSpatialIntervalCorpus,
    policies: &[SpatialPolicy],
    alpha: f64,
) -> Result<Vec<SpatialIntervalPolicyReport>, SpatialIntervalError> {
    validate_spatial_interval_corpus(corpus)?;
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(SpatialIntervalError::InvalidMiscoverage);
    }
    for policy in policies {
        validate_policy(policy)?;
    }
    let digest = digest_spatial_interval_corpus(corpus)?;
    let mut reports = Vec::with_capacity(policies.len());

    for policy in policies {
        let calibration_residuals: Vec<f64> = corpus
            .calibration
            .iter()
            .filter_map(|target| {
                predict(policy, &corpus.discovery, target.position)
                    .map(|prediction| (prediction - target.oracle_signed_error_mm).abs())
            })
            .collect();
        let calibration_unknown_count = corpus.calibration.len() - calibration_residuals.len();
        let n = calibration_residuals.len();
        let required_rank = (((n + 1) as f64) * (1.0 - alpha)).ceil() as usize;
        if n == 0 || required_rank == 0 || required_rank > n {
            return Err(SpatialIntervalError::InsufficientCalibration {
                usable: n,
                required_rank,
            });
        }
        let mut sorted = calibration_residuals;
        sorted.sort_by(f64::total_cmp);
        let half_width = sorted[required_rank - 1];

        let mut points = Vec::with_capacity(corpus.confirmation.len());
        let mut predicted_count = 0usize;
        let mut unknown_count = 0usize;
        let mut out_of_domain_count = 0usize;
        let mut covered_count = 0usize;
        let mut total_width = 0.0;

        for target in &corpus.confirmation {
            let result = if !inside_observed_rectangle(&corpus.discovery, target.position) {
                out_of_domain_count += 1;
                SpatialIntervalPointResult {
                    point_id: target.point_id.clone(),
                    position: target.position,
                    status: SpatialIntervalStatus::OutOfDomain,
                    predicted_signed_error_mm: None,
                    lower_bound_mm: None,
                    upper_bound_mm: None,
                    contains_oracle: None,
                }
            } else if let Some(prediction) = predict(policy, &corpus.discovery, target.position) {
                predicted_count += 1;
                let lower = prediction - half_width;
                let upper = prediction + half_width;
                let covered = (lower..=upper).contains(&target.oracle_signed_error_mm);
                if covered {
                    covered_count += 1;
                }
                total_width += upper - lower;
                SpatialIntervalPointResult {
                    point_id: target.point_id.clone(),
                    position: target.position,
                    status: SpatialIntervalStatus::Predicted,
                    predicted_signed_error_mm: Some(prediction),
                    lower_bound_mm: Some(lower),
                    upper_bound_mm: Some(upper),
                    contains_oracle: Some(covered),
                }
            } else {
                unknown_count += 1;
                SpatialIntervalPointResult {
                    point_id: target.point_id.clone(),
                    position: target.position,
                    status: SpatialIntervalStatus::Unknown,
                    predicted_signed_error_mm: None,
                    lower_bound_mm: None,
                    upper_bound_mm: None,
                    contains_oracle: None,
                }
            };
            points.push(result);
        }

        let confirmation_count = points.len();
        reports.push(SpatialIntervalPolicyReport {
            policy: policy.clone(),
            evidence_class: corpus.evidence_class,
            corpus_digest_sha256: digest.clone(),
            metrics: SpatialIntervalMetrics {
                discovery_count: corpus.discovery.len(),
                calibration_point_count: corpus.calibration.len(),
                calibration_residual_count: n,
                calibration_unknown_count,
                quantile_rank: required_rank,
                nominal_coverage: 1.0 - alpha,
                confirmation_count,
                predicted_count,
                unknown_count,
                out_of_domain_count,
                coverage: if confirmation_count == 0 {
                    0.0
                } else {
                    predicted_count as f64 / confirmation_count as f64
                },
                empirical_interval_coverage: (predicted_count > 0)
                    .then_some(covered_count as f64 / predicted_count as f64),
                mean_interval_width_mm: (predicted_count > 0)
                    .then_some(total_width / predicted_count as f64),
                half_width_mm: half_width,
            },
            points,
        });
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdm_spatial_benchmark::evaluate_spatial_policies;

    fn policies() -> Vec<SpatialPolicy> {
        vec![
            SpatialPolicy::GlobalMean,
            SpatialPolicy::NearestNeighbor,
            SpatialPolicy::InverseDistanceWeighted {
                neighbors: 4,
                max_distance: 0.5,
            },
        ]
    }

    #[test]
    fn interval_corpus_is_reproducible_and_has_three_disjoint_splits() {
        let first = demo_synthetic_spatial_interval_corpus(42);
        assert_eq!(first, demo_synthetic_spatial_interval_corpus(42));
        assert_eq!(first.calibration.len(), 16);
        validate_spatial_interval_corpus(&first).unwrap();
        let digest = digest_spatial_interval_corpus(&first).unwrap();
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn mutated_or_overlapping_split_fails_closed() {
        let mut changed_oracle = demo_synthetic_spatial_interval_corpus(42);
        changed_oracle.calibration[0].oracle_signed_error_mm += 0.01;
        assert!(validate_spatial_interval_corpus(&changed_oracle).is_err());

        let mut overlap = demo_synthetic_spatial_interval_corpus(42);
        overlap.calibration[0].position = overlap.confirmation[0].position;
        assert!(validate_spatial_interval_corpus(&overlap).is_err());

        let mut id_collision = demo_synthetic_spatial_interval_corpus(42);
        id_collision.calibration[0].point_id = id_collision.discovery[0].observation_id.clone();
        assert!(validate_spatial_interval_corpus(&id_collision).is_err());
    }

    #[test]
    fn interval_report_separates_coverage_unknown_and_out_of_domain() {
        let corpus = demo_synthetic_spatial_interval_corpus(42);
        let reports = evaluate_spatial_intervals(&corpus, &policies(), 0.10).unwrap();
        assert_eq!(reports.len(), 3);
        for report in &reports {
            let metrics = &report.metrics;
            assert_eq!(metrics.calibration_point_count, 16);
            assert_eq!(metrics.calibration_residual_count, 16);
            assert_eq!(metrics.calibration_unknown_count, 0);
            assert_eq!(metrics.quantile_rank, 16);
            assert_eq!(metrics.confirmation_count, 20);
            assert_eq!(metrics.predicted_count + metrics.unknown_count + metrics.out_of_domain_count, 20);
            assert_eq!(metrics.out_of_domain_count, 4);
            assert_eq!(metrics.predicted_count, 16);
            assert_eq!(metrics.unknown_count, 0);
            assert!(metrics.half_width_mm.is_finite());
            assert!(metrics.half_width_mm >= 0.0);
            assert!(metrics.empirical_interval_coverage.unwrap() >= 0.0);
            assert!(metrics.empirical_interval_coverage.unwrap() <= 1.0);
        }
        assert!(reports.iter().all(|report| {
            report.points.iter().filter(|point| point.status == SpatialIntervalStatus::OutOfDomain)
                .all(|point| point.lower_bound_mm.is_none() && point.contains_oracle.is_none())
        }));
    }

    #[test]
    fn interval_predictions_match_v1_policy_predictions_on_in_domain_targets() {
        let corpus = demo_synthetic_spatial_interval_corpus(42);
        let base = SyntheticSpatialCorpus {
            schema_version: crate::fdm_spatial_benchmark::FDM_SPATIAL_CORPUS_SCHEMA.to_owned(),
            evidence_class: SpatialEvidenceClass::Synthetic,
            seed: corpus.seed,
            response_function_version: RESPONSE_FUNCTION_VERSION.to_owned(),
            discovery: corpus.discovery.clone(),
            confirmation: corpus.confirmation.clone(),
        };
        // The v1 digest validator intentionally freezes the original oracle.
        let expected_base = demo_synthetic_spatial_corpus(corpus.seed);
        assert_eq!(base.discovery, expected_base.discovery);
        assert_eq!(base.confirmation, expected_base.confirmation);
        let reports = evaluate_spatial_policies(&expected_base, &policies()).unwrap();
        for (policy_index, policy) in policies().iter().enumerate() {
            for target in corpus.confirmation.iter().filter(|target| {
                inside_observed_rectangle(&corpus.discovery, target.position)
            }) {
                let local = predict(policy, &corpus.discovery, target.position);
                let reference = reports[policy_index]
                    .points
                    .iter()
                    .find(|point| point.point_id == target.point_id)
                    .unwrap()
                    .predicted_signed_error_mm;
                assert_eq!(local, reference);
            }
        }
    }

    #[test]
    fn invalid_alpha_and_too_small_calibration_fail_closed() {
        let corpus = demo_synthetic_spatial_interval_corpus(42);
        assert_eq!(
            evaluate_spatial_intervals(&corpus, &policies(), 1.0).unwrap_err(),
            SpatialIntervalError::InvalidMiscoverage
        );
        let error = evaluate_spatial_intervals(
            &corpus,
            &[SpatialPolicy::InverseDistanceWeighted {
                neighbors: 4,
                max_distance: 0.0001,
            }],
            0.10,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            SpatialIntervalError::InsufficientCalibration { usable: 0, .. }
        ));
    }

    #[test]
    fn deterministic_replay_and_digest_tamper_detection() {
        let corpus = demo_synthetic_spatial_interval_corpus(7);
        let first = evaluate_spatial_intervals(&corpus, &policies(), 0.10).unwrap();
        let second = evaluate_spatial_intervals(&corpus, &policies(), 0.10).unwrap();
        assert_eq!(first, second);

        let digest = digest_spatial_interval_corpus(&corpus).unwrap();
        let mut altered = corpus;
        altered.discovery[0].signed_error_mm += 0.001;
        assert_ne!(digest, serde_json::to_vec(&altered).map(|bytes| sha256(&bytes).to_hex()).unwrap());
        assert!(digest_spatial_interval_corpus(&altered).is_err());
    }
}
