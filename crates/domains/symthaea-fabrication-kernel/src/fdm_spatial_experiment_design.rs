// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Equal-budget, deterministic spatial experiment-design benchmark for synthetic FDM.
//!
//! Candidate selectors receive only the candidate pool, the observations measured
//! so far, a seed, and the selection step. The dense evaluation grid and its oracle
//! values are used only after a complete sampling trace has been produced.
//!
//! This is an experiment-design harness, not a physical printer model or control loop.

use crate::crypto_digest::sha256;
use crate::fdm_spatial_benchmark::NormalizedPosition;
use serde::{Deserialize, Serialize};

pub const FDM_EXPERIMENT_DESIGN_SCHEMA: &str = "symthaea.fdm-experiment-design-synthetic.v1";
pub const INITIAL_OBSERVATION_COUNT: usize = 5;
pub const ADDITIONAL_OBSERVATION_BUDGET: usize = 8;
pub const TOTAL_OBSERVATION_BUDGET: usize =
    INITIAL_OBSERVATION_COUNT + ADDITIONAL_OBSERVATION_BUDGET;
const MEASURED_GRID_TICKS: [u32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const EVALUATION_GRID_DIVISIONS: usize = 20;
const CRITICAL_ERROR_THRESHOLD_MM: f64 = 0.10;
const CRITICAL_REGION_DETECTION_RADIUS: f64 = 0.08;

/// Frozen synthetic stress cases. These labels are never evidence of physical capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticSpatialSurface {
    SmoothQuadratic,
    Anisotropic,
    EdgeRollOff,
    LocalizedHotspot,
    PiecewiseBoundary,
    HeteroskedasticNoise,
    NullSpatialEffect,
}

impl SyntheticSpatialSurface {
    pub fn name(self) -> &'static str {
        match self {
            Self::SmoothQuadratic => "smooth_quadratic",
            Self::Anisotropic => "anisotropic",
            Self::EdgeRollOff => "edge_roll_off",
            Self::LocalizedHotspot => "localized_hotspot",
            Self::PiecewiseBoundary => "piecewise_boundary",
            Self::HeteroskedasticNoise => "heteroskedastic_noise",
            Self::NullSpatialEffect => "null_spatial_effect",
        }
    }

    pub fn all() -> [Self; 7] {
        [
            Self::SmoothQuadratic,
            Self::Anisotropic,
            Self::EdgeRollOff,
            Self::LocalizedHotspot,
            Self::PiecewiseBoundary,
            Self::HeteroskedasticNoise,
            Self::NullSpatialEffect,
        ]
    }
}

/// Competing selectors. Every policy starts from exactly the same five observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategy {
    FixedStratified,
    FarthestPoint,
    ModelDisagreement,
    SeededRandom,
}

impl SamplingStrategy {
    pub fn name(self) -> &'static str {
        match self {
            Self::FixedStratified => "fixed_stratified",
            Self::FarthestPoint => "farthest_point",
            Self::ModelDisagreement => "model_disagreement",
            Self::SeededRandom => "seeded_random",
        }
    }

    pub fn all() -> [Self; 4] {
        [
            Self::FixedStratified,
            Self::FarthestPoint,
            Self::ModelDisagreement,
            Self::SeededRandom,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialDesignObservation {
    pub observation_id: String,
    pub acquisition_step: usize,
    pub position: NormalizedPosition,
    /// Synthetic measured signed error, in millimetres.
    pub measured_error_mm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialDesignMetrics {
    pub observation_count: usize,
    pub additional_observation_budget: usize,
    /// Mean distance from every fixed evaluation-grid point to its nearest observation.
    pub mean_nearest_sample_distance: f64,
    /// Worst evaluation-grid distance to any observation.
    pub maximum_nearest_sample_distance: f64,
    /// MAE for a simple nearest-measured-value estimator over the evaluation grid.
    pub nearest_sample_estimator_mae_mm: f64,
    pub nearest_sample_estimator_max_absolute_error_mm: f64,
    pub normalized_travel_distance: f64,
    pub critical_region_evaluation_points: usize,
    /// Fraction of true critical grid points near at least one observation whose
    /// measured magnitude exceeded the predeclared threshold. None when no critical
    /// points exist on the synthetic evaluation grid.
    pub critical_region_recall: Option<f64>,
    /// Observations which cross the measurement threshold despite a sub-threshold
    /// latent surface value at that position.
    pub false_alarm_observation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialDesignReport {
    pub schema_version: String,
    pub evidence_class: String,
    pub surface: SyntheticSpatialSurface,
    pub strategy: SamplingStrategy,
    pub seed: u64,
    pub metrics: SpatialDesignMetrics,
    /// Ordered trace: every measured position and the value revealed to the selector.
    pub observations: Vec<SpatialDesignObservation>,
    /// Identity of schema, surface, seed, strategy, trace and recomputed metrics.
    pub report_digest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpatialDesignError {
    InvalidBudget,
    InvalidTrace(String),
    SerializationFailure,
}

impl std::fmt::Display for SpatialDesignError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBudget => formatter.write_str("sampling budget is incompatible with the frozen strategy protocol"),
            Self::InvalidTrace(reason) => write!(formatter, "invalid spatial design trace: {reason}"),
            Self::SerializationFailure => formatter.write_str("could not serialize spatial design report"),
        }
    }
}

impl std::error::Error for SpatialDesignError {}

fn tick(value: u32) -> f64 {
    f64::from(value) / 10.0
}

fn position(x_tick: u32, y_tick: u32) -> NormalizedPosition {
    NormalizedPosition { x: tick(x_tick), y: tick(y_tick) }
}

fn same_position(left: NormalizedPosition, right: NormalizedPosition) -> bool {
    left.x == right.x && left.y == right.y
}

fn initial_positions() -> Vec<NormalizedPosition> {
    vec![
        position(1, 1),
        position(9, 1),
        position(1, 9),
        position(9, 9),
        position(5, 5),
    ]
}

fn candidate_positions() -> Vec<NormalizedPosition> {
    let mut candidates = Vec::with_capacity(81 - INITIAL_OBSERVATION_COUNT);
    for y in MEASURED_GRID_TICKS {
        for x in MEASURED_GRID_TICKS {
            let candidate = position(x, y);
            if !initial_positions().iter().any(|initial| same_position(*initial, candidate)) {
                candidates.push(candidate);
            }
        }
    }
    candidates
}

fn fixed_stratified_priority() -> Vec<NormalizedPosition> {
    let mut result = Vec::with_capacity(8);
    for y in [3, 5, 7] {
        for x in [3, 5, 7] {
            let candidate = position(x, y);
            if !initial_positions().iter().any(|initial| same_position(*initial, candidate)) {
                result.push(candidate);
            }
        }
    }
    result
}

fn distance(left: NormalizedPosition, right: NormalizedPosition) -> f64 {
    let dx = left.x - right.x;
    let dy = left.y - right.y;
    (dx * dx + dy * dy).sqrt()
}

fn min_observation_distance(candidate: NormalizedPosition, observations: &[SpatialDesignObservation]) -> f64 {
    observations
        .iter()
        .map(|observation| distance(candidate, observation.position))
        .reduce(f64::min)
        .unwrap_or(f64::INFINITY)
}

fn predict_nearest(observations: &[SpatialDesignObservation], target: NormalizedPosition) -> Option<f64> {
    observations.iter().min_by(|left, right| {
        distance(left.position, target)
            .total_cmp(&distance(right.position, target))
            .then_with(|| left.observation_id.cmp(&right.observation_id))
    }).map(|observation| observation.measured_error_mm)
}

fn predict_idw(observations: &[SpatialDesignObservation], target: NormalizedPosition) -> Option<f64> {
    let mut ranked: Vec<(&SpatialDesignObservation, f64)> = observations
        .iter()
        .map(|observation| (observation, distance(observation.position, target)))
        .collect();
    ranked.sort_by(|(left, left_distance), (right, right_distance)| {
        left_distance.total_cmp(right_distance)
            .then_with(|| left.observation_id.cmp(&right.observation_id))
    });
    ranked.truncate(4);
    if let Some((observation, separation)) = ranked.first() {
        if *separation == 0.0 {
            return Some(observation.measured_error_mm);
        }
    }
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for (observation, separation) in ranked {
        let weight = 1.0 / (separation * separation);
        weighted_sum += weight * observation.measured_error_mm;
        weight_sum += weight;
    }
    (weight_sum.is_finite() && weight_sum > 0.0).then_some(weighted_sum / weight_sum)
}

/// Select the next location without access to surface type, seed-generated oracle
/// values, or the confirmation grid. Only prior measurements and candidates enter.
pub fn choose_next_location(
    strategy: SamplingStrategy,
    candidates: &[NormalizedPosition],
    observations: &[SpatialDesignObservation],
    seed: u64,
    selection_step: usize,
) -> Option<NormalizedPosition> {
    if candidates.is_empty() {
        return None;
    }
    match strategy {
        SamplingStrategy::FixedStratified => {
            fixed_stratified_priority().into_iter()
                .find(|preferred| candidates.iter().any(|candidate| same_position(*candidate, *preferred)))
                .or_else(|| candidates.first().copied())
        }
        SamplingStrategy::FarthestPoint => candidates.iter().copied().max_by(|left, right| {
            min_observation_distance(*left, observations)
                .total_cmp(&min_observation_distance(*right, observations))
                // Stable tie-break: prefer the lexicographically smaller location.
                .then_with(|| right.y.total_cmp(&left.y))
                .then_with(|| right.x.total_cmp(&left.x))
        }),
        SamplingStrategy::ModelDisagreement => candidates.iter().copied().max_by(|left, right| {
            let score = |target| {
                match (predict_nearest(observations, target), predict_idw(observations, target)) {
                    (Some(nearest), Some(weighted)) => (nearest - weighted).abs(),
                    _ => 0.0,
                }
            };
            score(*left).total_cmp(&score(*right))
                .then_with(|| right.y.total_cmp(&left.y))
                .then_with(|| right.x.total_cmp(&left.x))
        }),
        SamplingStrategy::SeededRandom => {
            let mut state = seed
                ^ (selection_step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ 0xD1B5_4A32_D192_ED03;
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            state = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            state = (state ^ (state >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            state ^= state >> 31;
            candidates.get((state as usize) % candidates.len()).copied()
        }
    }
}

fn latent_surface(surface: SyntheticSpatialSurface, p: NormalizedPosition) -> f64 {
    let x = p.x - 0.5;
    let y = p.y - 0.5;
    match surface {
        SyntheticSpatialSurface::SmoothQuadratic => {
            0.18 * x - 0.13 * y + 0.21 * x * y + 0.32 * x * x - 0.17 * y * y
        }
        SyntheticSpatialSurface::Anisotropic => {
            0.22 * x * x + 0.025 * y * y + 0.09 * x * y + 0.035 * (7.0 * p.y).sin()
        }
        SyntheticSpatialSurface::EdgeRollOff => {
            let edge_distance = p.x.min(1.0 - p.x).min(p.y).min(1.0 - p.y);
            0.22 * (1.0 - (edge_distance / 0.5).clamp(0.0, 1.0))
        }
        SyntheticSpatialSurface::LocalizedHotspot => {
            let dx = (p.x - 0.73) / 0.07;
            let dy = (p.y - 0.27) / 0.07;
            0.30 * (-0.5 * (dx * dx + dy * dy)).exp()
        }
        SyntheticSpatialSurface::PiecewiseBoundary => {
            if p.x < 0.55 { -0.075 + 0.02 * p.y } else { 0.12 - 0.025 * p.y }
        }
        SyntheticSpatialSurface::HeteroskedasticNoise => {
            0.18 * x - 0.13 * y + 0.21 * x * y + 0.32 * x * x - 0.17 * y * y
        }
        SyntheticSpatialSurface::NullSpatialEffect => 0.0,
    }
}

fn deterministic_noise(seed: u64, p: NormalizedPosition) -> f64 {
    let x = (p.x * 1000.0).round() as u64;
    let y = (p.y * 1000.0).round() as u64;
    let mut value = seed ^ x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ y.wrapping_mul(0xD1B5_4A32_D192_ED03);
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    let unit = ((value >> 11) as f64) / 9_007_199_254_740_992.0;
    2.0 * unit - 1.0
}

fn measure(surface: SyntheticSpatialSurface, position: NormalizedPosition, seed: u64) -> f64 {
    let noise_scale = match surface {
        SyntheticSpatialSurface::HeteroskedasticNoise => {
            if position.x >= 0.55 { 0.035 } else { 0.003 }
        }
        SyntheticSpatialSurface::NullSpatialEffect => 0.004,
        _ => 0.0002,
    };
    latent_surface(surface, position) + noise_scale * deterministic_noise(seed, position)
}

fn run_one_strategy(
    surface: SyntheticSpatialSurface,
    strategy: SamplingStrategy,
    seed: u64,
) -> Result<SpatialDesignReport, SpatialDesignError> {
    let mut observations = Vec::with_capacity(TOTAL_OBSERVATION_BUDGET);
    for (index, position) in initial_positions().into_iter().enumerate() {
        observations.push(SpatialDesignObservation {
            observation_id: format!("obs-{index:03}"),
            acquisition_step: index,
            position,
            measured_error_mm: measure(surface, position, seed),
        });
    }

    let mut candidates = candidate_positions();
    for selection_step in 0..ADDITIONAL_OBSERVATION_BUDGET {
        // Selection sees only measurements acquired before this step.
        let selected = choose_next_location(strategy, &candidates, &observations, seed, selection_step)
            .ok_or(SpatialDesignError::InvalidBudget)?;
        let measured = measure(surface, selected, seed);
        let next_index = observations.len();
        observations.push(SpatialDesignObservation {
            observation_id: format!("obs-{next_index:03}"),
            acquisition_step: next_index,
            position: selected,
            measured_error_mm: measured,
        });
        candidates.retain(|candidate| !same_position(*candidate, selected));
    }

    validate_trace(&observations)?;
    let metrics = evaluate_design(surface, &observations, seed);
    let serialized_identity = serde_json::to_vec(&(
        FDM_EXPERIMENT_DESIGN_SCHEMA,
        "synthetic",
        surface,
        strategy,
        seed,
        &observations,
        &metrics,
    )).map_err(|_| SpatialDesignError::SerializationFailure)?;
    let domain = b"symthaea.fdm-experiment-design-report.v1\0";
    let mut digest_input = Vec::with_capacity(domain.len() + serialized_identity.len());
    digest_input.extend_from_slice(domain);
    digest_input.extend_from_slice(&serialized_identity);
    let digest = sha256(&digest_input).to_hex();

    Ok(SpatialDesignReport {
        schema_version: FDM_EXPERIMENT_DESIGN_SCHEMA.to_owned(),
        evidence_class: "synthetic".to_owned(),
        surface,
        strategy,
        seed,
        metrics,
        observations,
        report_digest_sha256: digest,
    })
}

fn validate_trace(observations: &[SpatialDesignObservation]) -> Result<(), SpatialDesignError> {
    if observations.len() != TOTAL_OBSERVATION_BUDGET {
        return Err(SpatialDesignError::InvalidTrace("unexpected observation count".to_owned()));
    }
    for (index, observation) in observations.iter().enumerate() {
        if observation.acquisition_step != index || observation.observation_id != format!("obs-{index:03}") {
            return Err(SpatialDesignError::InvalidTrace("trace identity or acquisition order mismatch".to_owned()));
        }
        if !observation.position.x.is_finite()
            || !observation.position.y.is_finite()
            || !(0.0..=1.0).contains(&observation.position.x)
            || !(0.0..=1.0).contains(&observation.position.y)
            || !observation.measured_error_mm.is_finite()
        {
            return Err(SpatialDesignError::InvalidTrace("non-finite or out-of-domain observation".to_owned()));
        }
        if observations[..index].iter().any(|previous| same_position(previous.position, observation.position)) {
            return Err(SpatialDesignError::InvalidTrace("duplicate measured position".to_owned()));
        }
    }
    Ok(())
}

fn evaluate_design(
    surface: SyntheticSpatialSurface,
    observations: &[SpatialDesignObservation],
    seed: u64,
) -> SpatialDesignMetrics {
    let mut distance_sum = 0.0;
    let mut max_distance: f64 = 0.0;
    let mut prediction_abs_error_sum = 0.0;
    let mut prediction_max_abs_error: f64 = 0.0;
    let mut evaluation_count = 0usize;
    let mut critical_count = 0usize;
    let mut detected_critical_count = 0usize;
    let mut false_alarm_count = 0usize;
    let risky_sample_positions: Vec<NormalizedPosition> = observations
        .iter()
        .filter(|observation| observation.measured_error_mm.abs() >= CRITICAL_ERROR_THRESHOLD_MM)
        .map(|observation| observation.position)
        .collect();

    for observation in observations {
        let latent = latent_surface(surface, observation.position);
        if observation.measured_error_mm.abs() >= CRITICAL_ERROR_THRESHOLD_MM
            && latent.abs() < CRITICAL_ERROR_THRESHOLD_MM
        {
            false_alarm_count += 1;
        }
    }

    for y_index in 0..=EVALUATION_GRID_DIVISIONS {
        for x_index in 0..=EVALUATION_GRID_DIVISIONS {
            let target = NormalizedPosition {
                x: x_index as f64 / EVALUATION_GRID_DIVISIONS as f64,
                y: y_index as f64 / EVALUATION_GRID_DIVISIONS as f64,
            };
            let closest_distance = min_observation_distance(target, observations);
            distance_sum += closest_distance;
            max_distance = max_distance.max(closest_distance);

            let oracle = latent_surface(surface, target);
            let predicted = predict_nearest(observations, target).unwrap_or(0.0);
            let absolute_error = (predicted - oracle).abs();
            prediction_abs_error_sum += absolute_error;
            prediction_max_abs_error = prediction_max_abs_error.max(absolute_error);
            evaluation_count += 1;

            if oracle.abs() >= CRITICAL_ERROR_THRESHOLD_MM {
                critical_count += 1;
                let region_detected = risky_sample_positions.iter().any(|sample| {
                    distance(*sample, target) <= CRITICAL_REGION_DETECTION_RADIUS
                });
                if region_detected {
                    detected_critical_count += 1;
                }
            }
        }
    }

    let travel_distance = observations.windows(2)
        .map(|pair| distance(pair[0].position, pair[1].position))
        .sum();
    SpatialDesignMetrics {
        observation_count: observations.len(),
        additional_observation_budget: observations.len().saturating_sub(INITIAL_OBSERVATION_COUNT),
        mean_nearest_sample_distance: distance_sum / evaluation_count as f64,
        maximum_nearest_sample_distance: max_distance,
        nearest_sample_estimator_mae_mm: prediction_abs_error_sum / evaluation_count as f64,
        nearest_sample_estimator_max_absolute_error_mm: prediction_max_abs_error,
        normalized_travel_distance: travel_distance,
        critical_region_evaluation_points: critical_count,
        critical_region_recall: (critical_count > 0)
            .then_some(detected_critical_count as f64 / critical_count as f64),
        false_alarm_observation_count: false_alarm_count,
    }
}

/// Run the frozen surface panel against all selectors at equal measurement budget.
/// All summaries are synthetic; callers must not interpret them as printer results.
pub fn run_spatial_experiment_design_benchmark(
    seed: u64,
) -> Result<Vec<SpatialDesignReport>, SpatialDesignError> {
    let mut reports = Vec::with_capacity(SyntheticSpatialSurface::all().len() * SamplingStrategy::all().len());
    for surface in SyntheticSpatialSurface::all() {
        for strategy in SamplingStrategy::all() {
            reports.push(run_one_strategy(surface, strategy, seed)?);
        }
    }
    Ok(reports)
}

/// Re-run one arm for focused tests and deterministic analysis.
pub fn run_spatial_design_arm(
    surface: SyntheticSpatialSurface,
    strategy: SamplingStrategy,
    seed: u64,
) -> Result<SpatialDesignReport, SpatialDesignError> {
    run_one_strategy(surface, strategy, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_panel_has_equal_budgets_and_distinct_report_digests() {
        let reports = run_spatial_experiment_design_benchmark(42).unwrap();
        assert_eq!(reports.len(), 7 * 4);
        assert!(reports.iter().all(|report| {
            report.evidence_class == "synthetic"
                && report.metrics.observation_count == TOTAL_OBSERVATION_BUDGET
                && report.metrics.additional_observation_budget == ADDITIONAL_OBSERVATION_BUDGET
                && report.observations.len() == TOTAL_OBSERVATION_BUDGET
                && report.report_digest_sha256.len() == 64
        }));
        let distinct_digests: std::collections::HashSet<_> = reports.iter()
            .map(|report| report.report_digest_sha256.as_str()).collect();
        assert_eq!(distinct_digests.len(), reports.len());
    }

    #[test]
    fn identical_inputs_replay_identically() {
        let first = run_spatial_experiment_design_benchmark(17).unwrap();
        let second = run_spatial_experiment_design_benchmark(17).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn seed_changes_stochastic_arm_and_measurement_noise() {
        let first = run_spatial_design_arm(
            SyntheticSpatialSurface::HeteroskedasticNoise,
            SamplingStrategy::SeededRandom,
            42,
        ).unwrap();
        let second = run_spatial_design_arm(
            SyntheticSpatialSurface::HeteroskedasticNoise,
            SamplingStrategy::SeededRandom,
            43,
        ).unwrap();
        assert_ne!(first.observations, second.observations);
    }

    #[test]
    fn hotspot_and_discontinuous_boundary_are_frozen_adversarial_cases() {
        let hotspot = run_spatial_design_arm(
            SyntheticSpatialSurface::LocalizedHotspot,
            SamplingStrategy::FixedStratified,
            42,
        ).unwrap();
        let boundary = run_spatial_design_arm(
            SyntheticSpatialSurface::PiecewiseBoundary,
            SamplingStrategy::FixedStratified,
            42,
        ).unwrap();
        assert!(hotspot.metrics.critical_region_evaluation_points > 0);
        assert!(boundary.metrics.critical_region_evaluation_points > 0);
    }

    #[test]
    fn null_effect_does_not_invent_true_critical_region() {
        let report = run_spatial_design_arm(
            SyntheticSpatialSurface::NullSpatialEffect,
            SamplingStrategy::ModelDisagreement,
            42,
        ).unwrap();
        assert_eq!(report.metrics.critical_region_evaluation_points, 0);
        assert_eq!(report.metrics.critical_region_recall, None);
    }

    #[test]
    fn selectors_never_revisit_initial_or_prior_locations() {
        for report in run_spatial_experiment_design_benchmark(9).unwrap() {
            let mut positions = std::collections::HashSet::new();
            for observation in &report.observations {
                let key = ((observation.position.x * 10.0).round() as u32,
                    (observation.position.y * 10.0).round() as u32);
                assert!(positions.insert(key));
                assert!(observation.position.x >= 0.1 && observation.position.x <= 0.9);
                assert!(observation.position.y >= 0.1 && observation.position.y <= 0.9);
            }
        }
    }
}
