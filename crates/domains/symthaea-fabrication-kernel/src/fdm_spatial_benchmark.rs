// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic synthetic spatial FDM calibration benchmark.
//!
//! The harness compares simple location-aware estimators against a global
//! baseline on held-out points. All observations are synthetic. This module
//! performs no machine I/O and does not model real printer physics.

use crate::crypto_digest::sha256;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const FDM_SPATIAL_CORPUS_SCHEMA: &str = "symthaea.fdm-spatial-synthetic.v1";

/// Normalized build-platform coordinate. Each axis is in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormalizedPosition {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialEvidenceClass {
    Synthetic,
}

/// A calibration observation available to the candidate policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialObservation {
    pub observation_id: String,
    pub position: NormalizedPosition,
    /// Synthetic signed dimensional error in millimetres.
    pub signed_error_mm: f64,
}

/// A held-out target. Candidate predictors never receive these targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialConfirmationPoint {
    pub point_id: String,
    pub position: NormalizedPosition,
    /// Synthetic oracle value, available to the evaluator only.
    pub oracle_signed_error_mm: f64,
}

/// Fixed synthetic response and a frozen discovery/confirmation split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticSpatialCorpus {
    pub schema_version: String,
    pub evidence_class: SpatialEvidenceClass,
    pub seed: u64,
    pub response_function_version: String,
    pub discovery: Vec<SpatialObservation>,
    pub confirmation: Vec<SpatialConfirmationPoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialPolicy {
    /// Predict every point with the mean of admitted discovery observations.
    GlobalMean,
    /// Return the nearest observed value, but do not extrapolate beyond the
    /// axis-aligned rectangle covered by discovery observations.
    NearestNeighbor,
    /// Inverse-distance-squared interpolation within the observed rectangle;
    /// abstain if no discovery point lies within max_distance.
    InverseDistanceWeighted {
        neighbors: usize,
        max_distance: f64,
    },
}

impl SpatialPolicy {
    pub fn name(&self) -> &'static str {
        match self {
            Self::GlobalMean => "global_mean",
            Self::NearestNeighbor => "nearest_neighbor",
            Self::InverseDistanceWeighted { .. } => "inverse_distance_weighted",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialPointResult {
    pub point_id: String,
    pub position: NormalizedPosition,
    pub oracle_signed_error_mm: f64,
    pub predicted_signed_error_mm: Option<f64>,
    /// prediction - oracle; None when the policy abstains.
    pub signed_prediction_error_mm: Option<f64>,
    pub absolute_prediction_error_mm: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialPolicyMetrics {
    pub discovery_count: usize,
    pub confirmation_count: usize,
    pub predicted_count: usize,
    pub unknown_count: usize,
    /// Coverage is kept separate from predictive error; abstention is visible.
    pub coverage: f64,
    pub mean_signed_prediction_error_mm: Option<f64>,
    pub mean_absolute_prediction_error_mm: Option<f64>,
    pub root_mean_squared_prediction_error_mm: Option<f64>,
    pub maximum_absolute_prediction_error_mm: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialPolicyReport {
    pub policy: SpatialPolicy,
    pub evidence_class: SpatialEvidenceClass,
    /// Domain-separated SHA-256 of the validated serialized synthetic corpus.
    pub corpus_digest_sha256: String,
    pub metrics: SpatialPolicyMetrics,
    pub points: Vec<SpatialPointResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpatialBenchmarkError {
    InvalidCorpus(Vec<String>),
    InvalidPolicy(&'static str),
    SerializationFailure,
}

impl std::fmt::Display for SpatialBenchmarkError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCorpus(issues) => write!(
                formatter,
                "invalid synthetic spatial corpus: {}",
                issues.join("; ")
            ),
            Self::InvalidPolicy(reason) => write!(formatter, "invalid spatial policy: {reason}"),
            Self::SerializationFailure => {
                formatter.write_str("could not serialize the validated spatial corpus")
            }
        }
    }
}

impl std::error::Error for SpatialBenchmarkError {}

/// Deterministic demo corpus: 3x3 discovery grid; off-grid and out-of-coverage
/// confirmation points. The synthetic response law is an experiment fixture,
/// not a validated approximation of physical printer behavior.
pub fn demo_synthetic_spatial_corpus(seed: u64) -> SyntheticSpatialCorpus {
    let mut rng = seed;
    let mut discovery = Vec::with_capacity(9);
    let grid = [0.1, 0.5, 0.9];
    let mut index = 0usize;
    for y in grid {
        for x in grid {
            // Integer-only PRNG state update and bounded conversion keep the
            // noise generation reproducible without external dependencies.
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise_code = ((rng >> 32) % 2001) as i64;
            let noise_mm = (noise_code as f64 - 1000.0) / 1_000_000.0;
            let position = NormalizedPosition { x, y };
            discovery.push(SpatialObservation {
                observation_id: format!("discovery-{index:03}"),
                position,
                signed_error_mm: synthetic_response(position) + noise_mm,
            });
            index += 1;
        }
    }

    let mut confirmation = Vec::with_capacity(20);
    let interior = [0.2, 0.35, 0.65, 0.8];
    let mut point_index = 0usize;
    for y in interior {
        for x in interior {
            let position = NormalizedPosition { x, y };
            confirmation.push(SpatialConfirmationPoint {
                point_id: format!("confirm-{point_index:03}"),
                position,
                oracle_signed_error_mm: synthetic_response(position),
            });
            point_index += 1;
        }
    }

    // These probes deliberately lie outside the measured rectangle, while
    // remaining inside the normalized platform; spatial policies must abstain.
    for position in [
        NormalizedPosition { x: 0.02, y: 0.5 },
        NormalizedPosition { x: 0.98, y: 0.5 },
        NormalizedPosition { x: 0.5, y: 0.02 },
        NormalizedPosition { x: 0.5, y: 0.98 },
    ] {
        confirmation.push(SpatialConfirmationPoint {
            point_id: format!("confirm-{point_index:03}"),
            position,
            oracle_signed_error_mm: synthetic_response(position),
        });
        point_index += 1;
    }

    SyntheticSpatialCorpus {
        schema_version: FDM_SPATIAL_CORPUS_SCHEMA.to_owned(),
        evidence_class: SpatialEvidenceClass::Synthetic,
        seed,
        response_function_version: "smooth-quadratic-v1".to_owned(),
        discovery,
        confirmation,
    }
}

/// Synthetic spatial error surface, defined explicitly for falsifiable tests.
pub fn synthetic_response(position: NormalizedPosition) -> f64 {
    let x = position.x - 0.5;
    let y = position.y - 0.5;
    0.18 * x - 0.13 * y + 0.21 * x * y + 0.32 * x * x - 0.17 * y * y
}

/// Validate provenance, schema, ranges, identity and the held-out position split.
pub fn validate_spatial_corpus(
    corpus: &SyntheticSpatialCorpus,
) -> Result<(), SpatialBenchmarkError> {
    let mut issues = Vec::new();
    if corpus.schema_version != FDM_SPATIAL_CORPUS_SCHEMA {
        issues.push("unsupported corpus schema version".to_owned());
    }
    if corpus.response_function_version != "smooth-quadratic-v1" {
        issues.push("unsupported synthetic response function version".to_owned());
    }
    if corpus.evidence_class != SpatialEvidenceClass::Synthetic {
        issues.push("this harness accepts synthetic evidence only".to_owned());
    }
    if corpus.discovery.is_empty() {
        issues.push("discovery observations are empty".to_owned());
    }
    if corpus.confirmation.is_empty() {
        issues.push("confirmation set is empty".to_owned());
    }

    // v1 is a frozen, seed-generated corpus definition. Reconstruct it to
    // prevent a mutated target/oracle from being relabeled as the same fixture.
    let expected = demo_synthetic_spatial_corpus(corpus.seed);
    if corpus.discovery != expected.discovery {
        issues.push("discovery records do not match the seed-generated v1 corpus".to_owned());
    }
    if corpus.confirmation != expected.confirmation {
        issues.push("confirmation records do not match the seed-generated v1 oracle".to_owned());
    }

    let mut observation_ids = HashSet::new();
    let mut positions = Vec::new();
    for observation in &corpus.discovery {
        if observation.observation_id.trim().is_empty() {
            issues.push("discovery observation ID is empty".to_owned());
        }
        if !observation_ids.insert(observation.observation_id.as_str()) {
            issues.push(format!("duplicate observation ID: {}", observation.observation_id));
        }
        validate_position(observation.position, &observation.observation_id, &mut issues);
        if !observation.signed_error_mm.is_finite() {
            issues.push(format!("non-finite discovery error: {}", observation.observation_id));
        }
        if positions.iter().any(|position: &NormalizedPosition| {
            same_position(*position, observation.position)
        }) {
            issues.push(format!(
                "duplicate discovery position: {}",
                observation.observation_id
            ));
        }
        positions.push(observation.position);
    }

    let mut confirmation_ids = HashSet::new();
    let mut confirmation_positions = Vec::new();
    for point in &corpus.confirmation {
        if point.point_id.trim().is_empty() {
            issues.push("confirmation point ID is empty".to_owned());
        }
        if !confirmation_ids.insert(point.point_id.as_str()) {
            issues.push(format!("duplicate confirmation point ID: {}", point.point_id));
        }
        if observation_ids.contains(point.point_id.as_str()) {
            issues.push(format!(
                "confirmation point ID leaks discovery identity: {}",
                point.point_id
            ));
        }
        validate_position(point.position, &point.point_id, &mut issues);
        if !point.oracle_signed_error_mm.is_finite() {
            issues.push(format!("non-finite confirmation target: {}", point.point_id));
        }
        if positions.iter().any(|position| same_position(*position, point.position)) {
            issues.push(format!(
                "confirmation position overlaps discovery position: {}",
                point.point_id
            ));
        }
        if confirmation_positions.iter().any(|position: &NormalizedPosition| {
            same_position(*position, point.position)
        }) {
            issues.push(format!(
                "duplicate confirmation position: {}",
                point.point_id
            ));
        }
        confirmation_positions.push(point.position);
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(SpatialBenchmarkError::InvalidCorpus(issues))
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

fn same_position(left: NormalizedPosition, right: NormalizedPosition) -> bool {
    left.x == right.x && left.y == right.y
}

fn validate_policy(policy: &SpatialPolicy) -> Result<(), SpatialBenchmarkError> {
    match policy {
        SpatialPolicy::GlobalMean | SpatialPolicy::NearestNeighbor => Ok(()),
        SpatialPolicy::InverseDistanceWeighted {
            neighbors,
            max_distance,
        } => {
            if *neighbors == 0 {
                return Err(SpatialBenchmarkError::InvalidPolicy(
                    "inverse-distance neighbour count must be positive",
                ));
            }
            if !max_distance.is_finite() || *max_distance <= 0.0 {
                return Err(SpatialBenchmarkError::InvalidPolicy(
                    "maximum prediction distance must be finite and positive",
                ));
            }
            Ok(())
        }
    }
}

/// Domain-separated identity of the exact validated corpus used for evaluation.
pub fn digest_spatial_corpus(
    corpus: &SyntheticSpatialCorpus,
) -> Result<String, SpatialBenchmarkError> {
    validate_spatial_corpus(corpus)?;
    let serialized =
        serde_json::to_vec(corpus).map_err(|_| SpatialBenchmarkError::SerializationFailure)?;
    let domain = b"symthaea.fdm-spatial-synthetic-corpus.v1\0";
    let mut bytes = Vec::with_capacity(domain.len() + serialized.len());
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&serialized);
    Ok(sha256(&bytes).to_hex())
}

/// Run policies against held-out targets. Prediction receives discovery only;
/// the confirmation oracle is read solely to recompute independent metrics.
pub fn evaluate_spatial_policies(
    corpus: &SyntheticSpatialCorpus,
    policies: &[SpatialPolicy],
) -> Result<Vec<SpatialPolicyReport>, SpatialBenchmarkError> {
    let corpus_digest_sha256 = digest_spatial_corpus(corpus)?;
    for policy in policies {
        validate_policy(policy)?;
    }

    let mut reports = Vec::with_capacity(policies.len());
    for policy in policies {
        let mut points = Vec::with_capacity(corpus.confirmation.len());
        for target in &corpus.confirmation {
            let predicted = predict(policy, &corpus.discovery, target.position);
            let signed_error = predicted.map(|value| value - target.oracle_signed_error_mm);
            points.push(SpatialPointResult {
                point_id: target.point_id.clone(),
                position: target.position,
                oracle_signed_error_mm: target.oracle_signed_error_mm,
                predicted_signed_error_mm: predicted,
                signed_prediction_error_mm: signed_error,
                absolute_prediction_error_mm: signed_error.map(f64::abs),
            });
        }
        let metrics = summarize(corpus.discovery.len(), &points);
        reports.push(SpatialPolicyReport {
            policy: policy.clone(),
            evidence_class: corpus.evidence_class,
            corpus_digest_sha256: corpus_digest_sha256.clone(),
            metrics,
            points,
        });
    }
    Ok(reports)
}

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
        SpatialPolicy::InverseDistanceWeighted {
            neighbors,
            max_distance,
        } => {
            if !inside_observed_rectangle(observations, target) {
                return None;
            }
            let mut ranked: Vec<(&SpatialObservation, f64)> = observations
                .iter()
                .map(|observation| (observation, distance(observation.position, target)))
                .filter(|(_, distance)| *distance <= *max_distance)
                .collect();
            ranked.sort_by(|(a_obs, a_dist), (b_obs, b_dist)| {
                a_dist
                    .total_cmp(b_dist)
                    .then_with(|| a_obs.observation_id.cmp(&b_obs.observation_id))
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
            if !weight_total.is_finite() || weight_total <= 0.0 {
                None
            } else {
                Some(weighted_sum / weight_total)
            }
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

fn inside_observed_rectangle(
    observations: &[SpatialObservation],
    target: NormalizedPosition,
) -> bool {
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

fn summarize(discovery_count: usize, points: &[SpatialPointResult]) -> SpatialPolicyMetrics {
    let absolute: Vec<f64> = points
        .iter()
        .filter_map(|point| point.absolute_prediction_error_mm)
        .collect();
    let signed: Vec<f64> = points
        .iter()
        .filter_map(|point| point.signed_prediction_error_mm)
        .collect();
    let predicted_count = signed.len();
    let unknown_count = points.len() - predicted_count;
    let mean = |values: &[f64]| {
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    };
    let rmse = if signed.is_empty() {
        None
    } else {
        Some(
            (signed.iter().map(|error| error * error).sum::<f64>() / signed.len() as f64)
                .sqrt(),
        )
    };
    SpatialPolicyMetrics {
        discovery_count,
        confirmation_count: points.len(),
        predicted_count,
        unknown_count,
        coverage: if points.is_empty() {
            0.0
        } else {
            predicted_count as f64 / points.len() as f64
        },
        mean_signed_prediction_error_mm: mean(&signed),
        mean_absolute_prediction_error_mm: mean(&absolute),
        root_mean_squared_prediction_error_mm: rmse,
        maximum_absolute_prediction_error_mm: absolute.iter().copied().reduce(f64::max),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn demo_corpus_is_deterministic_and_synthetic() {
        let first = demo_synthetic_spatial_corpus(42);
        let second = demo_synthetic_spatial_corpus(42);
        assert_eq!(first, second);
        assert_eq!(first.evidence_class, SpatialEvidenceClass::Synthetic);
        validate_spatial_corpus(&first).unwrap();
    }

    #[test]
    fn different_seed_changes_discovery_noise_but_not_oracle_surface() {
        let first = demo_synthetic_spatial_corpus(42);
        let second = demo_synthetic_spatial_corpus(43);
        assert_ne!(first.discovery, second.discovery);
        assert_eq!(first.confirmation, second.confirmation);
    }

    #[test]
    fn policies_report_separate_error_and_coverage_metrics() {
        let corpus = demo_synthetic_spatial_corpus(42);
        let reports = evaluate_spatial_policies(&corpus, &policies()).unwrap();
        assert_eq!(reports.len(), 3);
        for report in &reports {
            assert_eq!(report.evidence_class, SpatialEvidenceClass::Synthetic);
            assert_eq!(report.metrics.confirmation_count, 20);
            assert_eq!(
                report.metrics.predicted_count + report.metrics.unknown_count,
                report.metrics.confirmation_count
            );
        }
        assert_eq!(reports[0].metrics.coverage, 1.0);
        assert_eq!(reports[1].metrics.unknown_count, 4);
        assert_eq!(reports[2].metrics.unknown_count, 4);
    }

    #[test]
    fn exact_observation_position_is_recovered_by_spatial_estimators() {
        let corpus = demo_synthetic_spatial_corpus(42);
        let point = corpus.discovery[0].position;
        assert_eq!(
            predict(&SpatialPolicy::NearestNeighbor, &corpus.discovery, point),
            Some(corpus.discovery[0].signed_error_mm)
        );
        assert_eq!(
            predict(
                &SpatialPolicy::InverseDistanceWeighted {
                    neighbors: 4,
                    max_distance: 0.5,
                },
                &corpus.discovery,
                point
            ),
            Some(corpus.discovery[0].signed_error_mm)
        );
    }

    #[test]
    fn corrupted_or_leaked_confirmation_corpus_fails_closed() {
        let mut corpus = demo_synthetic_spatial_corpus(42);
        corpus.confirmation[0].point_id = corpus.discovery[0].observation_id.clone();
        assert!(validate_spatial_corpus(&corpus).is_err());

        let mut out_of_domain = demo_synthetic_spatial_corpus(42);
        out_of_domain.confirmation[0].position.x = 1.1;
        assert!(validate_spatial_corpus(&out_of_domain).is_err());

        let mut duplicated = demo_synthetic_spatial_corpus(42);
        duplicated.confirmation[1].position = duplicated.confirmation[0].position;
        assert!(validate_spatial_corpus(&duplicated).is_err());

        let mut altered_oracle = demo_synthetic_spatial_corpus(42);
        altered_oracle.confirmation[0].oracle_signed_error_mm += 0.01;
        assert!(validate_spatial_corpus(&altered_oracle).is_err());
    }

    #[test]
    fn invalid_interpolation_parameters_fail_closed() {
        let corpus = demo_synthetic_spatial_corpus(42);
        let error = evaluate_spatial_policies(
            &corpus,
            &[SpatialPolicy::InverseDistanceWeighted {
                neighbors: 0,
                max_distance: 0.5,
            }],
        )
        .unwrap_err();
        assert_eq!(
            error,
            SpatialBenchmarkError::InvalidPolicy(
                "inverse-distance neighbour count must be positive"
            )
        );
    }

    #[derive(serde::Deserialize)]
    struct FrozenSpatialCorpusSummary {
        schema_version: String,
        seed: u64,
        policy_metrics: Vec<FrozenSpatialPolicyMetrics>,
    }

    #[derive(serde::Deserialize)]
    struct FrozenSpatialPolicyMetrics {
        policy: String,
        predicted_count: usize,
        unknown_count: usize,
        coverage: f64,
        mean_signed_prediction_error_mm: f64,
        mean_absolute_prediction_error_mm: f64,
        root_mean_squared_prediction_error_mm: f64,
        maximum_absolute_prediction_error_mm: f64,
    }

    #[test]
    fn frozen_reference_metrics_match_independent_expected_values() {
        let fixture: FrozenSpatialCorpusSummary = serde_json::from_str(include_str!(
            "../tests/fixtures/fdm_spatial_v1.json"
        ))
        .expect("frozen spatial metric fixture must parse");
        assert_eq!(fixture.schema_version, FDM_SPATIAL_CORPUS_SCHEMA);
        let corpus = demo_synthetic_spatial_corpus(fixture.seed);
        let reports = evaluate_spatial_policies(&corpus, &policies()).unwrap();
        assert_eq!(reports.len(), fixture.policy_metrics.len());

        for expected in fixture.policy_metrics {
            let actual = reports
                .iter()
                .find(|report| report.policy.name() == expected.policy)
                .expect("fixture names a policy that is evaluated");
            assert_eq!(actual.metrics.predicted_count, expected.predicted_count);
            assert_eq!(actual.metrics.unknown_count, expected.unknown_count);
            let within = |actual: f64, expected: f64| (actual - expected).abs() < 1.0e-11;
            assert!(within(actual.metrics.coverage, expected.coverage));
            assert!(within(
                actual.metrics.mean_signed_prediction_error_mm.unwrap(),
                expected.mean_signed_prediction_error_mm
            ));
            assert!(within(
                actual.metrics.mean_absolute_prediction_error_mm.unwrap(),
                expected.mean_absolute_prediction_error_mm
            ));
            assert!(within(
                actual.metrics.root_mean_squared_prediction_error_mm.unwrap(),
                expected.root_mean_squared_prediction_error_mm
            ));
            assert!(within(
                actual.metrics.maximum_absolute_prediction_error_mm.unwrap(),
                expected.maximum_absolute_prediction_error_mm
            ));
            assert_eq!(actual.corpus_digest_sha256.len(), 64);
            assert_eq!(
                actual.corpus_digest_sha256,
                digest_spatial_corpus(&corpus).unwrap()
            );
        }
    }

    #[test]
    fn corpus_digest_changes_when_discovery_observation_changes() {
        let mut corpus = demo_synthetic_spatial_corpus(42);
        let first = digest_spatial_corpus(&corpus).unwrap();
        corpus.discovery[0].signed_error_mm += 0.001;
        let second = digest_spatial_corpus(&corpus).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn replaying_identical_inputs_produces_identical_reports() {
        let corpus = demo_synthetic_spatial_corpus(7);
        assert_eq!(
            evaluate_spatial_policies(&corpus, &policies()).unwrap(),
            evaluate_spatial_policies(&corpus, &policies()).unwrap()
        );
    }
}
