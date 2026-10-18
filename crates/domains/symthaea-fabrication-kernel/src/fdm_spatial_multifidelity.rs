// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic multi-fidelity spatial measurement experiment design.
//!
//! The selectable action is (location, measurement method, repeat count).
//! Fast-screen and reference-metrology observations have different declared
//! costs and noise models. Surface values are accessed only by the measurement
//! simulator and post-hoc evaluator; selectors receive recorded observations,
//! candidate actions and declared costs, never the latent oracle.
//!
//! All values are synthetic. This module has no instrument or printer I/O.

use crate::crypto_digest::sha256;
use crate::fdm_spatial_benchmark::NormalizedPosition;
use crate::fdm_spatial_experiment_design::{
    ADDITIONAL_OBSERVATION_BUDGET, INITIAL_OBSERVATION_COUNT, SamplingStrategy,
    SpatialDesignObservation, SyntheticSpatialSurface, candidate_positions, choose_next_location,
    distance, fixed_stratified_priority, initial_positions, latent_surface, same_position,
};
use serde::{Deserialize, Serialize};

pub const FDM_MULTIFIDELITY_SCHEMA: &str = "symthaea.fdm-multifidelity-synthetic.v1";
pub const MULTIFIDELITY_BUDGET_LEVELS: [f64; 6] = [33.0, 40.0, 48.0, 56.0, 64.0, 72.0];
pub const MAX_ADDITIONAL_ACTIONS: usize = 10;
const EVALUATION_GRID_DIVISIONS: usize = 20;
const CRITICAL_ERROR_THRESHOLD_MM: f64 = 0.10;
const CRITICAL_REGION_DETECTION_RADIUS: f64 = 0.08;
const EPSILON: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetrologyMethod {
    FastScreen,
    ReferenceMetrology,
}

impl MetrologyMethod {
    pub fn name(self) -> &'static str {
        match self {
            Self::FastScreen => "fast_screen",
            Self::ReferenceMetrology => "reference_metrology",
        }
    }

    fn per_reading_cost(self, schedule: &MultiFidelityCostSchedule) -> f64 {
        match self {
            Self::FastScreen => schedule.fast_screen_cost_per_reading,
            Self::ReferenceMetrology => schedule.reference_cost_per_reading,
        }
    }

    fn nominal_sigma_mm(self, position: NormalizedPosition, surface: SyntheticSpatialSurface) -> f64 {
        let base = match self {
            Self::FastScreen => 0.025,
            Self::ReferenceMetrology => 0.004,
        };
        if surface == SyntheticSpatialSurface::HeteroskedasticNoise
            && self == Self::FastScreen
            && position.x > 0.55
        {
            base * 2.5
        } else {
            base
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetrologyStressCase {
    UnbiasedIndependent,
    BiasedFastScreen,
    CorrelatedRepeats,
    StaleReferenceCalibration,
}

impl MetrologyStressCase {
    pub fn name(self) -> &'static str {
        match self {
            Self::UnbiasedIndependent => "unbiased_independent",
            Self::BiasedFastScreen => "biased_fast_screen",
            Self::CorrelatedRepeats => "correlated_repeats",
            Self::StaleReferenceCalibration => "stale_reference_calibration",
        }
    }

    pub fn all() -> [Self; 4] {
        [
            Self::UnbiasedIndependent,
            Self::BiasedFastScreen,
            Self::CorrelatedRepeats,
            Self::StaleReferenceCalibration,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiFidelityStrategy {
    ReferenceOnly,
    FastScreenOnly,
    FixedStaged,
    CoverageGuardedStaged,
    CostNormalizedDecision,
    SeededRandom,
}

impl MultiFidelityStrategy {
    pub fn name(self) -> &'static str {
        match self {
            Self::ReferenceOnly => "reference_only",
            Self::FastScreenOnly => "fast_screen_only",
            Self::FixedStaged => "fixed_staged",
            Self::CoverageGuardedStaged => "coverage_guarded_staged",
            Self::CostNormalizedDecision => "cost_normalized_decision",
            Self::SeededRandom => "seeded_random",
        }
    }

    pub fn all() -> [Self; 6] {
        [
            Self::ReferenceOnly,
            Self::FastScreenOnly,
            Self::FixedStaged,
            Self::CoverageGuardedStaged,
            Self::CostNormalizedDecision,
            Self::SeededRandom,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiFidelityCostSchedule {
    pub schema_version: String,
    pub schedule_name: String,
    pub budget_units: f64,
    pub one_time_setup_units: f64,
    pub fast_screen_cost_per_reading: f64,
    pub reference_cost_per_reading: f64,
    pub travel_units_per_normalized_distance: f64,
    pub edge_zone_width_normalized: f64,
    pub edge_measurement_surcharge_units: f64,
}

impl MultiFidelityCostSchedule {
    /// Abstract, inspectable costs. These are not seconds, currency, or measured costs.
    pub fn demo(budget_units: f64) -> Self {
        Self {
            schema_version: FDM_MULTIFIDELITY_SCHEMA.to_owned(),
            schedule_name: "synthetic-fast-screen-reference-travel-v1".to_owned(),
            budget_units,
            one_time_setup_units: 2.0,
            fast_screen_cost_per_reading: 1.0,
            reference_cost_per_reading: 4.0,
            travel_units_per_normalized_distance: 2.0,
            edge_zone_width_normalized: 0.15,
            edge_measurement_surcharge_units: 0.35,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementAction {
    pub position: NormalizedPosition,
    pub method: MetrologyMethod,
    /// Supported repeat plans are exactly one or three readings per action.
    pub repeat_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiFidelityObservation {
    pub observation_id: String,
    pub acquisition_step: usize,
    pub position: NormalizedPosition,
    pub method: MetrologyMethod,
    pub readings_mm: Vec<f64>,
    pub mean_mm: f64,
    pub sample_standard_deviation_mm: f64,
    /// Declared random-measurement uncertainty; it intentionally excludes
    /// unknown systematic bias in the biased-screen and stale-calibration arms.
    pub reported_standard_uncertainty_mm: f64,
    pub calibration_state: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiFidelityCostLedger {
    pub setup_units: f64,
    pub acquisition_units: f64,
    pub travel_units: f64,
    pub edge_surcharge_units: f64,
    pub incremental_total_units: f64,
    pub cumulative_total_units: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostedMultiFidelityObservation {
    pub observation: MultiFidelityObservation,
    pub cost: MultiFidelityCostLedger,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiFidelityStopReason {
    ObservationActionCapReached,
    BudgetConstraintReached,
    StrategyExhausted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiFidelityMetrics {
    pub planned_budget_units: f64,
    pub spent_budget_units: f64,
    pub unused_budget_units: f64,
    pub setup_cost_units: f64,
    pub fast_screen_cost_units: f64,
    pub reference_metrology_cost_units: f64,
    pub travel_cost_units: f64,
    pub edge_surcharge_cost_units: f64,
    pub normalized_route_distance: f64,
    pub action_count: usize,
    pub reading_count: usize,
    pub fast_screen_action_count: usize,
    pub reference_action_count: usize,
    pub repeated_action_count: usize,
    pub evaluation_grid_point_count: usize,
    pub covered_grid_point_count: usize,
    pub unknown_grid_point_count: usize,
    pub spatial_coverage: f64,
    pub mean_absolute_error_mm: Option<f64>,
    pub worst_covered_location_error_mm: Option<f64>,
    pub critical_evaluation_point_count: usize,
    pub covered_critical_point_count: usize,
    pub false_safe_point_count: usize,
    pub false_safe_rate: Option<f64>,
    pub critical_region_recall: Option<f64>,
    pub false_alarm_action_count: usize,
    pub mean_reported_uncertainty_mm: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiFidelityReport {
    pub schema_version: String,
    pub evidence_class: String,
    pub surface: SyntheticSpatialSurface,
    pub stress_case: MetrologyStressCase,
    pub strategy: MultiFidelityStrategy,
    pub seed: u64,
    pub cost_schedule: MultiFidelityCostSchedule,
    pub cost_schedule_digest_sha256: String,
    pub stop_reason: MultiFidelityStopReason,
    pub metrics: MultiFidelityMetrics,
    pub observations: Vec<CostedMultiFidelityObservation>,
    pub report_digest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiFidelityError {
    InvalidSchedule(&'static str),
    InitialDesignExceedsBudget,
    InvalidAction(&'static str),
    InvalidReport(String),
    SerializationFailure,
}

impl std::fmt::Display for MultiFidelityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSchedule(reason) => write!(formatter, "invalid multi-fidelity cost schedule: {reason}"),
            Self::InitialDesignExceedsBudget => formatter.write_str("common initial design exceeds the declared cost budget"),
            Self::InvalidAction(reason) => write!(formatter, "invalid metrology action: {reason}"),
            Self::InvalidReport(reason) => write!(formatter, "invalid multi-fidelity report: {reason}"),
            Self::SerializationFailure => formatter.write_str("could not serialize multi-fidelity evidence"),
        }
    }
}

impl std::error::Error for MultiFidelityError {}

#[derive(Debug, Clone, Copy)]
struct IncrementalCost {
    setup: f64,
    acquisition: f64,
    travel: f64,
    edge: f64,
    total: f64,
}

#[derive(Debug, Clone, Copy)]
struct PointEstimate {
    position: NormalizedPosition,
    value_mm: f64,
    uncertainty_mm: f64,
}

fn all_lattice_positions() -> Vec<NormalizedPosition> {
    let mut positions = Vec::with_capacity(81);
    for y_tick in 1..=9 {
        for x_tick in 1..=9 {
            positions.push(NormalizedPosition {
                x: f64::from(x_tick) / 10.0,
                y: f64::from(y_tick) / 10.0,
            });
        }
    }
    positions
}

fn validate_schedule(schedule: &MultiFidelityCostSchedule) -> Result<(), MultiFidelityError> {
    if schedule.schema_version != FDM_MULTIFIDELITY_SCHEMA {
        return Err(MultiFidelityError::InvalidSchedule("unsupported schema version"));
    }
    if schedule.schedule_name.trim().is_empty() {
        return Err(MultiFidelityError::InvalidSchedule("empty schedule name"));
    }
    if !schedule.budget_units.is_finite() || schedule.budget_units <= 0.0 {
        return Err(MultiFidelityError::InvalidSchedule("budget must be finite and positive"));
    }
    for (value, field) in [
        (schedule.one_time_setup_units, "setup cost"),
        (schedule.fast_screen_cost_per_reading, "fast-screen cost"),
        (schedule.reference_cost_per_reading, "reference cost"),
        (schedule.travel_units_per_normalized_distance, "travel cost"),
        (schedule.edge_measurement_surcharge_units, "edge surcharge"),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(MultiFidelityError::InvalidSchedule(field));
        }
    }
    if schedule.fast_screen_cost_per_reading <= 0.0
        || schedule.reference_cost_per_reading < schedule.fast_screen_cost_per_reading
    {
        return Err(MultiFidelityError::InvalidSchedule("reference metrology must not be cheaper than fast screening"));
    }
    if !schedule.edge_zone_width_normalized.is_finite()
        || !(0.0..=0.5).contains(&schedule.edge_zone_width_normalized)
    {
        return Err(MultiFidelityError::InvalidSchedule("edge-zone width must be in [0, 0.5]"));
    }
    Ok(())
}

fn near_edge(position: NormalizedPosition, width: f64) -> bool {
    position.x.min(1.0 - position.x).min(position.y).min(1.0 - position.y) <= width
}

fn incremental_cost(
    previous: NormalizedPosition,
    action: &MeasurementAction,
    schedule: &MultiFidelityCostSchedule,
    is_first: bool,
) -> Result<IncrementalCost, MultiFidelityError> {
    if !matches!(action.repeat_count, 1 | 3) {
        return Err(MultiFidelityError::InvalidAction("repeat count must be one or three"));
    }
    let setup = if is_first { schedule.one_time_setup_units } else { 0.0 };
    let acquisition = action.method.per_reading_cost(schedule) * action.repeat_count as f64;
    let travel = distance(previous, action.position) * schedule.travel_units_per_normalized_distance;
    let edge = if near_edge(action.position, schedule.edge_zone_width_normalized) {
        schedule.edge_measurement_surcharge_units
    } else {
        0.0
    };
    Ok(IncrementalCost {
        setup,
        acquisition,
        travel,
        edge,
        total: setup + acquisition + travel + edge,
    })
}

fn schedule_digest(schedule: &MultiFidelityCostSchedule) -> Result<String, MultiFidelityError> {
    let bytes = serde_json::to_vec(schedule).map_err(|_| MultiFidelityError::SerializationFailure)?;
    let domain = b"symthaea.fdm-multifidelity-schedule.v1\0";
    let mut input = Vec::with_capacity(domain.len() + bytes.len());
    input.extend_from_slice(domain);
    input.extend_from_slice(&bytes);
    Ok(sha256(&input).to_hex())
}

fn report_digest(report: &MultiFidelityReport) -> Result<String, MultiFidelityError> {
    let bytes = serde_json::to_vec(&(
        &report.schema_version,
        &report.evidence_class,
        report.surface,
        report.stress_case,
        report.strategy,
        report.seed,
        &report.cost_schedule,
        &report.cost_schedule_digest_sha256,
        report.stop_reason,
        &report.metrics,
        &report.observations,
    )).map_err(|_| MultiFidelityError::SerializationFailure)?;
    let domain = b"symthaea.fdm-multifidelity-report.v1\0";
    let mut input = Vec::with_capacity(domain.len() + bytes.len());
    input.extend_from_slice(domain);
    input.extend_from_slice(&bytes);
    Ok(sha256(&input).to_hex())
}

fn deterministic_unit_noise(
    seed: u64,
    position: NormalizedPosition,
    method: MetrologyMethod,
    replicate: usize,
    salt: u64,
) -> f64 {
    let x = (position.x * 1000.0).round() as u64;
    let y = (position.y * 1000.0).round() as u64;
    let method_code = match method {
        MetrologyMethod::FastScreen => 0xA24B_AED4_963E_E407,
        MetrologyMethod::ReferenceMetrology => 0x9FB2_1C65_1E98_DF25,
    };
    let mut value = seed
        ^ x.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ y.wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ method_code
        ^ (replicate as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
        ^ salt;
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    let unit = (value >> 11) as f64 / 9_007_199_254_740_992.0;
    2.0 * unit - 1.0
}

fn correlation_for(stress: MetrologyStressCase, method: MetrologyMethod) -> f64 {
    if stress != MetrologyStressCase::CorrelatedRepeats {
        return 0.0;
    }
    match method {
        MetrologyMethod::FastScreen => 0.85,
        MetrologyMethod::ReferenceMetrology => 0.70,
    }
}

fn systematic_bias(
    stress: MetrologyStressCase,
    method: MetrologyMethod,
    position: NormalizedPosition,
) -> f64 {
    match (stress, method) {
        (MetrologyStressCase::BiasedFastScreen, MetrologyMethod::FastScreen) => {
            if position.x > 0.55 { -0.035 } else { 0.015 }
        }
        (MetrologyStressCase::StaleReferenceCalibration, MetrologyMethod::ReferenceMetrology) => -0.040,
        _ => 0.0,
    }
}

fn measure_action(
    surface: SyntheticSpatialSurface,
    stress: MetrologyStressCase,
    seed: u64,
    action: &MeasurementAction,
    index: usize,
) -> MultiFidelityObservation {
    let nominal_sigma = action.method.nominal_sigma_mm(action.position, surface);
    let rho = correlation_for(stress, action.method);
    let common = deterministic_unit_noise(seed, action.position, action.method, 0, 0x312A_5F18);
    let mut readings = Vec::with_capacity(action.repeat_count);
    for replicate in 0..action.repeat_count {
        let independent = deterministic_unit_noise(
            seed, action.position, action.method, replicate, 0xC7D1_3E92,
        );
        let standardized_noise = rho.sqrt() * common + (1.0 - rho).sqrt() * independent;
        readings.push(
            latent_surface(surface, action.position)
                + systematic_bias(stress, action.method, action.position)
                + nominal_sigma * standardized_noise,
        );
    }
    let mean = readings.iter().sum::<f64>() / readings.len() as f64;
    let sample_variance = if readings.len() > 1 {
        readings.iter().map(|reading| (reading - mean).powi(2)).sum::<f64>()
            / (readings.len() - 1) as f64
    } else {
        0.0
    };
    let rho = correlation_for(stress, action.method);
    let uncertainty = nominal_sigma * (rho + (1.0 - rho) / readings.len() as f64).sqrt();
    MultiFidelityObservation {
        observation_id: format!("mf-obs-{index:03}"),
        acquisition_step: index,
        position: action.position,
        method: action.method,
        readings_mm: readings,
        mean_mm: mean,
        sample_standard_deviation_mm: sample_variance.sqrt(),
        reported_standard_uncertainty_mm: uncertainty,
        calibration_state: match stress {
            MetrologyStressCase::StaleReferenceCalibration
                if action.method == MetrologyMethod::ReferenceMetrology => "stale_unrecognized".to_owned(),
            _ => "nominal_synthetic".to_owned(),
        },
    }
}

fn estimate_at_position(
    position: NormalizedPosition,
    observations: &[CostedMultiFidelityObservation],
) -> Option<PointEstimate> {
    let relevant: Vec<_> = observations.iter()
        .filter(|item| same_position(item.observation.position, position))
        .map(|item| &item.observation)
        .collect();
    if relevant.is_empty() {
        return None;
    }
    let mut precision_sum = 0.0;
    let mut weighted_sum = 0.0;
    for observation in &relevant {
        let variance = observation.reported_standard_uncertainty_mm.powi(2);
        if variance > 0.0 && variance.is_finite() {
            let precision = 1.0 / variance;
            precision_sum += precision;
            weighted_sum += precision * observation.mean_mm;
        }
    }
    if !precision_sum.is_finite() || precision_sum <= 0.0 {
        return None;
    }
    Some(PointEstimate {
        position,
        value_mm: weighted_sum / precision_sum,
        uncertainty_mm: (1.0 / precision_sum).sqrt(),
    })
}

fn point_estimates(observations: &[CostedMultiFidelityObservation]) -> Vec<PointEstimate> {
    all_lattice_positions().into_iter()
        .filter_map(|position| estimate_at_position(position, observations))
        .collect()
}

fn selector_observations(observations: &[CostedMultiFidelityObservation]) -> Vec<SpatialDesignObservation> {
    point_estimates(observations).into_iter().enumerate().map(|(index, estimate)| {
        SpatialDesignObservation {
            observation_id: format!("combined-{index:03}"),
            acquisition_step: index,
            position: estimate.position,
            signed_error_mm: estimate.value_mm,
        }
    }).collect()
}

fn estimate_prediction(
    estimates: &[PointEstimate],
    target: NormalizedPosition,
    inverse_distance: bool,
) -> Option<f64> {
    if estimates.is_empty() {
        return None;
    }
    if !inverse_distance {
        return estimates.iter().min_by(|left, right| {
            distance(left.position, target).total_cmp(&distance(right.position, target))
                .then_with(|| left.position.y.total_cmp(&right.position.y))
                .then_with(|| left.position.x.total_cmp(&right.position.x))
        }).map(|estimate| estimate.value_mm);
    }
    let mut ranked = estimates.to_vec();
    ranked.sort_by(|left, right| {
        distance(left.position, target).total_cmp(&distance(right.position, target))
            .then_with(|| left.position.y.total_cmp(&right.position.y))
            .then_with(|| left.position.x.total_cmp(&right.position.x))
    });
    ranked.truncate(4);
    if let Some(first) = ranked.first() {
        if distance(first.position, target) == 0.0 {
            return Some(first.value_mm);
        }
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for estimate in ranked {
        let d = distance(estimate.position, target);
        let weight = 1.0 / (d * d);
        numerator += weight * estimate.value_mm;
        denominator += weight;
    }
    (denominator.is_finite() && denominator > 0.0).then_some(numerator / denominator)
}

fn action_available(
    position: NormalizedPosition,
    method: MetrologyMethod,
    observations: &[CostedMultiFidelityObservation],
) -> bool {
    if observations.iter().any(|item| {
        same_position(item.observation.position, position)
            && item.observation.method == method
    }) {
        return false;
    }
    // Once reference metrology has been performed, another fast screen is
    // dominated for this benchmark. Repeats belong inside a single action.
    if method == MetrologyMethod::FastScreen && observations.iter().any(|item| {
        same_position(item.observation.position, position)
            && item.observation.method == MetrologyMethod::ReferenceMetrology
    }) {
        return false;
    }
    true
}

fn borderline_score(value_mm: f64, uncertainty_mm: f64) -> f64 {
    let distance_to_threshold = (value_mm.abs() - CRITICAL_ERROR_THRESHOLD_MM).abs();
    1.0 / (0.015 + distance_to_threshold + uncertainty_mm)
}

fn repeat_plan(
    position: NormalizedPosition,
    observations: &[CostedMultiFidelityObservation],
) -> usize {
    let estimate = estimate_at_position(position, observations);
    match estimate {
        Some(estimate)
            if (estimate.value_mm.abs() - CRITICAL_ERROR_THRESHOLD_MM).abs()
                <= (2.0 * estimate.uncertainty_mm).max(0.025) => 3,
        _ => 1,
    }
}

fn choose_reference_promotion(
    observations: &[CostedMultiFidelityObservation],
) -> Option<MeasurementAction> {
    observations.iter()
        .filter(|item| item.observation.method == MetrologyMethod::FastScreen)
        .filter(|item| action_available(
            item.observation.position,
            MetrologyMethod::ReferenceMetrology,
            observations,
        ))
        .map(|item| {
            let score = borderline_score(
                item.observation.mean_mm,
                item.observation.reported_standard_uncertainty_mm,
            );
            (item.observation.position, score)
        })
        .max_by(|(left_position, left_score), (right_position, right_score)| {
            left_score.total_cmp(right_score)
                .then_with(|| right_position.y.total_cmp(&left_position.y))
                .then_with(|| right_position.x.total_cmp(&left_position.x))
        })
        .map(|(position, _)| MeasurementAction {
            position,
            method: MetrologyMethod::ReferenceMetrology,
            repeat_count: repeat_plan(position, observations),
        })
}

fn select_action(
    strategy: MultiFidelityStrategy,
    observations: &[CostedMultiFidelityObservation],
    schedule: &MultiFidelityCostSchedule,
    seed: u64,
    step: usize,
) -> Option<MeasurementAction> {
    let estimates = point_estimates(observations);
    let observed_positions: Vec<_> = estimates.iter().map(|estimate| estimate.position).collect();
    let all_positions = all_lattice_positions();

    let unmeasured_fast: Vec<_> = all_positions.iter().copied()
        .filter(|position| action_available(*position, MetrologyMethod::FastScreen, observations))
        .collect();
    let unmeasured_reference: Vec<_> = all_positions.iter().copied()
        .filter(|position| action_available(*position, MetrologyMethod::ReferenceMetrology, observations))
        .collect();

    match strategy {
        MultiFidelityStrategy::ReferenceOnly => {
            let candidates: Vec<_> = unmeasured_reference.into_iter()
                .filter(|position| !observations.iter().any(|item| same_position(item.observation.position, *position)))
                .collect();
            choose_next_location(SamplingStrategy::FarthestPoint, &candidates, &selector_observations(observations), seed, step)
                .map(|position| MeasurementAction { position, method: MetrologyMethod::ReferenceMetrology, repeat_count: 1 })
        }
        MultiFidelityStrategy::FastScreenOnly => {
            let candidates: Vec<_> = unmeasured_fast.into_iter()
                .filter(|position| !observations.iter().any(|item| same_position(item.observation.position, *position)))
                .collect();
            choose_next_location(SamplingStrategy::FarthestPoint, &candidates, &selector_observations(observations), seed, step)
                .map(|position| MeasurementAction { position, method: MetrologyMethod::FastScreen, repeat_count: 1 })
        }
        MultiFidelityStrategy::FixedStaged => {
            if let Some(promotion) = choose_reference_promotion(observations) {
                let is_urgent = observations.iter().any(|item| {
                    item.observation.method == MetrologyMethod::FastScreen
                        && same_position(item.observation.position, promotion.position)
                        && item.observation.mean_mm.abs() >= CRITICAL_ERROR_THRESHOLD_MM
                });
                if is_urgent
                    || (estimate_at_position(promotion.position, observations).is_some_and(|estimate| {
                        (estimate.value_mm.abs() - CRITICAL_ERROR_THRESHOLD_MM).abs()
                            <= (2.0 * estimate.uncertainty_mm).max(0.025)
                    }))
                {
                    return Some(promotion);
                }
            }
            fixed_stratified_priority().into_iter()
                .find(|position| action_available(*position, MetrologyMethod::FastScreen, observations))
                .map(|position| MeasurementAction { position, method: MetrologyMethod::FastScreen, repeat_count: 1 })
        }
        MultiFidelityStrategy::CoverageGuardedStaged => {
            if step % 2 == 1 {
                if let Some(promotion) = choose_reference_promotion(observations) {
                    return Some(promotion);
                }
            }
            let candidates: Vec<_> = unmeasured_fast.into_iter()
                .filter(|position| !observations.iter().any(|item| same_position(item.observation.position, *position)))
                .collect();
            choose_next_location(SamplingStrategy::FarthestPoint, &candidates, &selector_observations(observations), seed, step)
                .map(|position| MeasurementAction { position, method: MetrologyMethod::FastScreen, repeat_count: 1 })
                .or_else(|| choose_reference_promotion(observations))
        }
        MultiFidelityStrategy::SeededRandom => {
            let candidates: Vec<_> = unmeasured_fast.into_iter()
                .filter(|position| !observations.iter().any(|item| same_position(item.observation.position, *position)))
                .collect();
            choose_next_location(SamplingStrategy::SeededRandom, &candidates, &selector_observations(observations), seed, step)
                .map(|position| MeasurementAction { position, method: MetrologyMethod::FastScreen, repeat_count: 1 })
        }
        MultiFidelityStrategy::CostNormalizedDecision => {
            let mut actions = Vec::new();
            for position in all_positions {
                if action_available(position, MetrologyMethod::FastScreen, observations)
                    && !observations.iter().any(|item| same_position(item.observation.position, position))
                {
                    actions.push(MeasurementAction {
                        position,
                        method: MetrologyMethod::FastScreen,
                        repeat_count: 1,
                    });
                }
                if action_available(position, MetrologyMethod::ReferenceMetrology, observations) {
                    actions.push(MeasurementAction {
                        position,
                        method: MetrologyMethod::ReferenceMetrology,
                        repeat_count: repeat_plan(position, observations),
                    });
                }
            }
            actions.into_iter().max_by(|left, right| {
                let score = |action: &MeasurementAction| {
                    let cost = incremental_cost(
                        observations.last().map(|item| item.observation.position)
                            .unwrap_or(NormalizedPosition { x: 0.5, y: 0.5 }),
                        action,
                        schedule,
                        false,
                    ).map(|cost| cost.total).unwrap_or(f64::INFINITY);
                    let nearest_distance = estimates.iter()
                        .map(|estimate| distance(estimate.position, action.position))
                        .reduce(f64::min).unwrap_or(1.0);
                    let nearest = estimate_prediction(&estimates, action.position, false).unwrap_or(0.0);
                    let idw = estimate_prediction(&estimates, action.position, true).unwrap_or(0.0);
                    let disagreement = (nearest - idw).abs();
                    let risk = 1.0 / (0.02 + (nearest.abs() - CRITICAL_ERROR_THRESHOLD_MM).abs());
                    let fidelity_gain = match action.method {
                        MetrologyMethod::FastScreen => 1.0,
                        MetrologyMethod::ReferenceMetrology => 1.5,
                    };
                    let exploration = if observed_positions.iter().any(|p| same_position(*p, action.position)) {
                        0.0
                    } else {
                        nearest_distance
                    };
                    if cost <= 0.0 || !cost.is_finite() {
                        0.0
                    } else {
                        (0.03 + disagreement + 0.04 * risk + 0.25 * exploration)
                            * fidelity_gain
                            / cost
                    }
                };
                score(left).total_cmp(&score(right))
                    .then_with(|| right.position.y.total_cmp(&left.position.y))
                    .then_with(|| right.position.x.total_cmp(&left.position.x))
                    .then_with(|| match (left.method, right.method) {
                        (MetrologyMethod::FastScreen, MetrologyMethod::ReferenceMetrology) => std::cmp::Ordering::Greater,
                        (MetrologyMethod::ReferenceMetrology, MetrologyMethod::FastScreen) => std::cmp::Ordering::Less,
                        _ => std::cmp::Ordering::Equal,
                    })
            })
        }
    }
}

fn component_cost(
    previous: NormalizedPosition,
    action: &MeasurementAction,
    schedule: &MultiFidelityCostSchedule,
    first: bool,
) -> Result<IncrementalCost, MultiFidelityError> {
    incremental_cost(previous, action, schedule, first)
}

fn point_in_frozen_lattice(position: NormalizedPosition) -> bool {
    (1..=9).any(|x| (position.x - f64::from(x) / 10.0).abs() <= EPSILON)
        && (1..=9).any(|y| (position.y - f64::from(y) / 10.0).abs() <= EPSILON)
}

fn close(left: f64, right: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= 1.0e-9
}

fn evaluate_metrics(
    surface: SyntheticSpatialSurface,
    stress: MetrologyStressCase,
    observations: &[CostedMultiFidelityObservation],
    schedule: &MultiFidelityCostSchedule,
) -> MultiFidelityMetrics {
    let estimates = point_estimates(observations);
    let reading_count = observations.iter().map(|item| item.observation.readings_mm.len()).sum::<usize>();
    let fast_actions = observations.iter().filter(|item| item.observation.method == MetrologyMethod::FastScreen).count();
    let reference_actions = observations.len() - fast_actions;
    let repeated_actions = observations.iter().filter(|item| item.observation.readings_mm.len() > 1).count();
    let mut absolute_errors = Vec::new();
    let mut max_error: Option<f64> = None;
    let mut evaluation_count = 0usize;
    let mut covered_count = 0usize;
    let mut unknown_count = 0usize;
    let mut critical_count = 0usize;
    let mut covered_critical_count = 0usize;
    let mut false_safe_count = 0usize;
    let mut detected_critical_count = 0usize;

    let alerted_positions: Vec<_> = estimates.iter()
        .filter(|estimate| estimate.value_mm.abs() - 2.0 * estimate.uncertainty_mm >= CRITICAL_ERROR_THRESHOLD_MM)
        .map(|estimate| estimate.position)
        .collect();

    let mut false_alarm_actions = 0usize;
    for item in observations {
        let measured_alert = item.observation.mean_mm.abs()
            - 2.0 * item.observation.reported_standard_uncertainty_mm
            >= CRITICAL_ERROR_THRESHOLD_MM;
        if measured_alert
            && latent_surface(surface, item.observation.position).abs() < CRITICAL_ERROR_THRESHOLD_MM
        {
            false_alarm_actions += 1;
        }
    }

    let mut uncertainty_sum = 0.0;
    for estimate in &estimates {
        uncertainty_sum += estimate.uncertainty_mm;
    }
    let mean_uncertainty = (!estimates.is_empty()).then_some(uncertainty_sum / estimates.len() as f64);

    for y_index in 0..=EVALUATION_GRID_DIVISIONS {
        for x_index in 0..=EVALUATION_GRID_DIVISIONS {
            evaluation_count += 1;
            let target = NormalizedPosition {
                x: x_index as f64 / EVALUATION_GRID_DIVISIONS as f64,
                y: y_index as f64 / EVALUATION_GRID_DIVISIONS as f64,
            };
            let oracle = latent_surface(surface, target);
            let is_critical = oracle.abs() >= CRITICAL_ERROR_THRESHOLD_MM;
            if is_critical {
                critical_count += 1;
                if alerted_positions.iter().any(|position| distance(*position, target) <= CRITICAL_REGION_DETECTION_RADIUS) {
                    detected_critical_count += 1;
                }
            }
            let nearest = estimates.iter().min_by(|left, right| {
                distance(left.position, target).total_cmp(&distance(right.position, target))
                    .then_with(|| left.position.y.total_cmp(&right.position.y))
                    .then_with(|| left.position.x.total_cmp(&right.position.x))
            });
            match nearest {
                Some(estimate) if distance(estimate.position, target) <= CRITICAL_REGION_DETECTION_RADIUS => {
                    covered_count += 1;
                    let error = (estimate.value_mm - oracle).abs();
                    absolute_errors.push(error);
                    max_error = Some(max_error.map_or(error, |previous| previous.max(error)));
                    if is_critical {
                        covered_critical_count += 1;
                        if estimate.value_mm.abs() + 2.0 * estimate.uncertainty_mm
                            < CRITICAL_ERROR_THRESHOLD_MM
                        {
                            false_safe_count += 1;
                        }
                    }
                }
                _ => {
                    unknown_count += 1;
                }
            }
        }
    }

    let sum_component = |select: fn(&MultiFidelityCostLedger) -> f64| {
        observations.iter().map(|item| select(&item.cost)).sum::<f64>()
    };
    let spent = observations.last().map(|item| item.cost.cumulative_total_units).unwrap_or(0.0);
    let route_distance = observations.iter().map(|item| {
        if item.observation.acquisition_step == 0 {
            distance(NormalizedPosition { x: 0.5, y: 0.5 }, item.observation.position)
        } else {
            distance(
                observations[item.observation.acquisition_step - 1].observation.position,
                item.observation.position,
            )
        }
    }).sum();
    let _ = stress;
    MultiFidelityMetrics {
        planned_budget_units: schedule.budget_units,
        spent_budget_units: spent,
        unused_budget_units: (schedule.budget_units - spent).max(0.0),
        setup_cost_units: sum_component(|cost| cost.setup_units),
        fast_screen_cost_units: sum_component(|cost| {
            // Component totals include method acquisition separately below; return zero here.
            let _ = cost;
            0.0
        }) + observations.iter().filter(|item| item.observation.method == MetrologyMethod::FastScreen)
            .map(|item| item.cost.acquisition_units).sum::<f64>(),
        reference_metrology_cost_units: observations.iter()
            .filter(|item| item.observation.method == MetrologyMethod::ReferenceMetrology)
            .map(|item| item.cost.acquisition_units).sum::<f64>(),
        travel_cost_units: sum_component(|cost| cost.travel_units),
        edge_surcharge_cost_units: sum_component(|cost| cost.edge_surcharge_units),
        normalized_route_distance: route_distance,
        action_count: observations.len(),
        reading_count,
        fast_screen_action_count: fast_actions,
        reference_action_count: reference_actions,
        repeated_action_count: repeated_actions,
        evaluation_grid_point_count: evaluation_count,
        covered_grid_point_count: covered_count,
        unknown_grid_point_count: unknown_count,
        spatial_coverage: covered_count as f64 / evaluation_count as f64,
        mean_absolute_error_mm: (!absolute_errors.is_empty()).then(|| {
            absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64
        }),
        worst_covered_location_error_mm: max_error,
        critical_evaluation_point_count: critical_count,
        covered_critical_point_count: covered_critical_count,
        false_safe_point_count: false_safe_count,
        false_safe_rate: (covered_critical_count > 0).then_some(
            false_safe_count as f64 / covered_critical_count as f64
        ),
        critical_region_recall: (critical_count > 0).then_some(
            detected_critical_count as f64 / critical_count as f64
        ),
        false_alarm_action_count: false_alarm_actions,
        mean_reported_uncertainty_mm: mean_uncertainty,
    }
}

fn validate_observation_values(
    surface: SyntheticSpatialSurface,
    stress: MetrologyStressCase,
    seed: u64,
    observation: &MultiFidelityObservation,
) -> Result<(), MultiFidelityError> {
    if observation.readings_mm.len() != 1 && observation.readings_mm.len() != 3 {
        return Err(MultiFidelityError::InvalidReport("measurement repeat count is not one or three".to_owned()));
    }
    if observation.readings_mm.iter().any(|value| !value.is_finite()) {
        return Err(MultiFidelityError::InvalidReport("non-finite measurement reading".to_owned()));
    }
    let expected_action = MeasurementAction {
        position: observation.position,
        method: observation.method,
        repeat_count: observation.readings_mm.len(),
    };
    let expected = measure_action(surface, stress, seed, &expected_action, observation.acquisition_step);
    if expected.readings_mm != observation.readings_mm
        || expected.calibration_state != observation.calibration_state
        || !close(expected.mean_mm, observation.mean_mm)
        || !close(expected.sample_standard_deviation_mm, observation.sample_standard_deviation_mm)
        || !close(expected.reported_standard_uncertainty_mm, observation.reported_standard_uncertainty_mm)
    {
        return Err(MultiFidelityError::InvalidReport("measurement values/uncertainty do not match the frozen synthetic oracle".to_owned()));
    }
    Ok(())
}

fn costed_observation(
    surface: SyntheticSpatialSurface,
    stress: MetrologyStressCase,
    seed: u64,
    index: usize,
    action: &MeasurementAction,
    previous: NormalizedPosition,
    previous_cumulative: f64,
    schedule: &MultiFidelityCostSchedule,
) -> Result<CostedMultiFidelityObservation, MultiFidelityError> {
    let cost = component_cost(previous, action, schedule, index == 0)?;
    Ok(CostedMultiFidelityObservation {
        observation: measure_action(surface, stress, seed, action, index),
        cost: MultiFidelityCostLedger {
            setup_units: cost.setup,
            acquisition_units: cost.acquisition,
            travel_units: cost.travel,
            edge_surcharge_units: cost.edge,
            incremental_total_units: cost.total,
            cumulative_total_units: previous_cumulative + cost.total,
        },
    })
}

/// Choose a policy's preferred action without considering the budget field.
/// The budget runner takes a prefix of this deterministic action sequence.
pub fn choose_multifidelity_action(
    strategy: MultiFidelityStrategy,
    observations: &[CostedMultiFidelityObservation],
    schedule: &MultiFidelityCostSchedule,
    seed: u64,
    step: usize,
) -> Option<MeasurementAction> {
    select_action(strategy, observations, schedule, seed, step)
}

/// Run a single multi-fidelity stress/surface/strategy/budget arm.
pub fn run_multifidelity_arm(
    surface: SyntheticSpatialSurface,
    stress: MetrologyStressCase,
    strategy: MultiFidelityStrategy,
    seed: u64,
    schedule: MultiFidelityCostSchedule,
) -> Result<MultiFidelityReport, MultiFidelityError> {
    validate_schedule(&schedule)?;
    let schedule_digest_sha256 = schedule_digest(&schedule)?;
    let initial_method = if strategy == MultiFidelityStrategy::ReferenceOnly {
        MetrologyMethod::ReferenceMetrology
    } else {
        MetrologyMethod::FastScreen
    };
    let mut observations = Vec::with_capacity(INITIAL_OBSERVATION_COUNT + MAX_ADDITIONAL_ACTIONS);
    let mut cumulative = 0.0;
    let mut previous = NormalizedPosition { x: 0.5, y: 0.5 };
    for (index, position) in initial_positions().into_iter().enumerate() {
        let action = MeasurementAction { position, method: initial_method, repeat_count: 1 };
        let item = costed_observation(
            surface, stress, seed, index, &action, previous, cumulative, &schedule,
        )?;
        cumulative = item.cost.cumulative_total_units;
        previous = position;
        observations.push(item);
    }
    if cumulative > schedule.budget_units + EPSILON {
        return Err(MultiFidelityError::InitialDesignExceedsBudget);
    }

    let stop_reason = loop {
        if observations.len() >= INITIAL_OBSERVATION_COUNT + MAX_ADDITIONAL_ACTIONS {
            break MultiFidelityStopReason::ObservationActionCapReached;
        }
        let step = observations.len() - INITIAL_OBSERVATION_COUNT;
        let Some(action) = choose_multifidelity_action(strategy, &observations, &schedule, seed, step) else {
            break MultiFidelityStopReason::StrategyExhausted;
        };
        let previous = observations.last().map(|item| item.observation.position)
            .unwrap_or(NormalizedPosition { x: 0.5, y: 0.5 });
        let cost = incremental_cost(previous, &action, &schedule, false)?;
        let remaining = schedule.budget_units - cumulative;
        if cost.total > remaining + EPSILON {
            break MultiFidelityStopReason::BudgetConstraintReached;
        }
        let index = observations.len();
        let item = costed_observation(
            surface, stress, seed, index, &action, previous, cumulative, &schedule,
        )?;
        cumulative = item.cost.cumulative_total_units;
        observations.push(item);
    };

    let metrics = evaluate_metrics(surface, stress, &observations, &schedule);
    let mut report = MultiFidelityReport {
        schema_version: FDM_MULTIFIDELITY_SCHEMA.to_owned(),
        evidence_class: "synthetic".to_owned(),
        surface,
        stress_case: stress,
        strategy,
        seed,
        cost_schedule: schedule,
        cost_schedule_digest_sha256,
        stop_reason,
        metrics,
        observations,
        report_digest_sha256: String::new(),
    };
    report.report_digest_sha256 = report_digest(&report)?;
    verify_multifidelity_report(&report)?;
    Ok(report)
}

/// Run the fixed surface × stress × policy × budget panel for one seed.
pub fn run_multifidelity_benchmark(seed: u64) -> Result<Vec<MultiFidelityReport>, MultiFidelityError> {
    let capacity = SyntheticSpatialSurface::all().len()
        * MetrologyStressCase::all().len()
        * MultiFidelityStrategy::all().len()
        * MULTIFIDELITY_BUDGET_LEVELS.len();
    let mut reports = Vec::with_capacity(capacity);
    for surface in SyntheticSpatialSurface::all() {
        for stress in MetrologyStressCase::all() {
            for strategy in MultiFidelityStrategy::all() {
                for budget in MULTIFIDELITY_BUDGET_LEVELS {
                    reports.push(run_multifidelity_arm(
                        surface, stress, strategy, seed, MultiFidelityCostSchedule::demo(budget),
                    )?);
                }
            }
        }
    }
    Ok(reports)
}

/// Recompute observations, cost components, method counts, metrics, budget, and identity.
/// This proves consistency with the frozen synthetic schema, not trusted physical origin.
pub fn verify_multifidelity_report(report: &MultiFidelityReport) -> Result<(), MultiFidelityError> {
    validate_schedule(&report.cost_schedule)?;
    if report.schema_version != FDM_MULTIFIDELITY_SCHEMA || report.evidence_class != "synthetic" {
        return Err(MultiFidelityError::InvalidReport("unsupported schema/evidence class".to_owned()));
    }
    if report.cost_schedule_digest_sha256 != schedule_digest(&report.cost_schedule)? {
        return Err(MultiFidelityError::InvalidReport("cost schedule digest mismatch".to_owned()));
    }
    if !(INITIAL_OBSERVATION_COUNT..=INITIAL_OBSERVATION_COUNT + MAX_ADDITIONAL_ACTIONS)
        .contains(&report.observations.len())
    {
        return Err(MultiFidelityError::InvalidReport("action count outside the frozen cap".to_owned()));
    }

    let expected_initial = initial_positions();
    let expected_initial_method = if report.strategy == MultiFidelityStrategy::ReferenceOnly {
        MetrologyMethod::ReferenceMetrology
    } else {
        MetrologyMethod::FastScreen
    };
    let mut cumulative = 0.0;
    let mut previous = NormalizedPosition { x: 0.5, y: 0.5 };
    for (index, item) in report.observations.iter().enumerate() {
        let observation = &item.observation;
        if observation.observation_id != format!("mf-obs-{index:03}")
            || observation.acquisition_step != index
        {
            return Err(MultiFidelityError::InvalidReport("observation identity/order mismatch".to_owned()));
        }
        if index < INITIAL_OBSERVATION_COUNT {
            if !same_position(observation.position, expected_initial[index])
                || observation.method != expected_initial_method
                || observation.readings_mm.len() != 1
            {
                return Err(MultiFidelityError::InvalidReport("common initial design or fidelity changed".to_owned()));
            }
        }
        if !point_in_frozen_lattice(observation.position) {
            return Err(MultiFidelityError::InvalidReport("position outside frozen lattice".to_owned()));
        }
        if index >= INITIAL_OBSERVATION_COUNT && !all_lattice_positions().iter()
            .any(|position| same_position(*position, observation.position))
        {
            return Err(MultiFidelityError::InvalidReport("action position not in candidate pool".to_owned()));
        }
        if report.observations[..index].iter().any(|previous_item| {
            same_position(previous_item.observation.position, observation.position)
                && previous_item.observation.method == observation.method
        }) {
            return Err(MultiFidelityError::InvalidReport("duplicate location/method; use a repeat plan".to_owned()));
        }
        validate_observation_values(report.surface, report.stress_case, report.seed, observation)?;
        let action = MeasurementAction {
            position: observation.position,
            method: observation.method,
            repeat_count: observation.readings_mm.len(),
        };
        let expected_cost = incremental_cost(previous, &action, &report.cost_schedule, index == 0)?;
        if !close(item.cost.setup_units, expected_cost.setup)
            || !close(item.cost.acquisition_units, expected_cost.acquisition)
            || !close(item.cost.travel_units, expected_cost.travel)
            || !close(item.cost.edge_surcharge_units, expected_cost.edge)
            || !close(item.cost.incremental_total_units, expected_cost.total)
        {
            return Err(MultiFidelityError::InvalidReport("component cost ledger mismatch".to_owned()));
        }
        cumulative += expected_cost.total;
        if !close(item.cost.cumulative_total_units, cumulative) {
            return Err(MultiFidelityError::InvalidReport("cumulative cost mismatch".to_owned()));
        }
        previous = observation.position;
    }
    if cumulative > report.cost_schedule.budget_units + EPSILON {
        return Err(MultiFidelityError::InvalidReport("budget overspent".to_owned()));
    }
    let recomputed = evaluate_metrics(
        report.surface, report.stress_case, &report.observations, &report.cost_schedule,
    );
    if recomputed != report.metrics {
        return Err(MultiFidelityError::InvalidReport("metrics do not recompute from the trace".to_owned()));
    }

    let action_cap_reached = report.observations.len()
        == INITIAL_OBSERVATION_COUNT + MAX_ADDITIONAL_ACTIONS;
    let expected_stop = if action_cap_reached {
        MultiFidelityStopReason::ObservationActionCapReached
    } else {
        let step = report.observations.len() - INITIAL_OBSERVATION_COUNT;
        match choose_multifidelity_action(report.strategy, &report.observations, &report.cost_schedule, report.seed, step) {
            None => MultiFidelityStopReason::StrategyExhausted,
            Some(action) => {
                let last = report.observations.last()
                    .ok_or_else(|| MultiFidelityError::InvalidReport("empty trace".to_owned()))?
                    .observation.position;
                let next_cost = incremental_cost(last, &action, &report.cost_schedule, false)?.total;
                if next_cost > report.cost_schedule.budget_units - cumulative + EPSILON {
                    MultiFidelityStopReason::BudgetConstraintReached
                } else {
                    return Err(MultiFidelityError::InvalidReport("trace stopped despite an affordable next action".to_owned()));
                }
            }
        }
    };
    if report.stop_reason != expected_stop {
        return Err(MultiFidelityError::InvalidReport("stopping reason does not match trace/budget".to_owned()));
    }
    if report.report_digest_sha256 != report_digest(report)? {
        return Err(MultiFidelityError::InvalidReport("report digest mismatch".to_owned()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_dimensions_and_all_cost_ledgers_verify() {
        let reports = run_multifidelity_benchmark(42).unwrap();
        assert_eq!(
            reports.len(),
            SyntheticSpatialSurface::all().len()
                * MetrologyStressCase::all().len()
                * MultiFidelityStrategy::all().len()
                * MULTIFIDELITY_BUDGET_LEVELS.len()
        );
        for report in &reports {
            assert_eq!(report.evidence_class, "synthetic");
            assert!(report.metrics.spent_budget_units <= report.metrics.planned_budget_units + EPSILON);
            assert!(report.metrics.unused_budget_units >= 0.0);
            assert_eq!(report.metrics.action_count, report.observations.len());
            verify_multifidelity_report(report).unwrap();
        }
    }

    #[test]
    fn exact_replay_and_budget_traces_are_nested() {
        for strategy in MultiFidelityStrategy::all() {
            let mut previous_positions = Vec::new();
            let mut previous_spend = 0.0;
            for budget in MULTIFIDELITY_BUDGET_LEVELS {
                let report = run_multifidelity_arm(
                    SyntheticSpatialSurface::LocalizedHotspot,
                    MetrologyStressCase::UnbiasedIndependent,
                    strategy,
                    22,
                    MultiFidelityCostSchedule::demo(budget),
                ).unwrap();
                let positions: Vec<_> = report.observations.iter().map(|item| (
                    (item.observation.position.x * 10.0).round() as u32,
                    (item.observation.position.y * 10.0).round() as u32,
                    item.observation.method,
                    item.observation.readings_mm.len(),
                )).collect();
                assert!(positions.len() >= previous_positions.len());
                assert_eq!(&positions[..previous_positions.len()], previous_positions.as_slice());
                assert!(report.metrics.spent_budget_units + EPSILON >= previous_spend);
                previous_positions = positions;
                previous_spend = report.metrics.spent_budget_units;
            }
        }
        let schedule = MultiFidelityCostSchedule::demo(48.0);
        let first = run_multifidelity_arm(
            SyntheticSpatialSurface::PiecewiseBoundary,
            MetrologyStressCase::CorrelatedRepeats,
            MultiFidelityStrategy::CostNormalizedDecision,
            12,
            schedule.clone(),
        ).unwrap();
        let second = run_multifidelity_arm(
            SyntheticSpatialSurface::PiecewiseBoundary,
            MetrologyStressCase::CorrelatedRepeats,
            MultiFidelityStrategy::CostNormalizedDecision,
            12,
            schedule,
        ).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn reference_method_costs_more_and_has_lower_declared_noise_floor() {
        let point = NormalizedPosition { x: 0.5, y: 0.5 };
        let fast = MetrologyMethod::FastScreen.nominal_sigma_mm(point, SyntheticSpatialSurface::SmoothQuadratic);
        let reference = MetrologyMethod::ReferenceMetrology.nominal_sigma_mm(point, SyntheticSpatialSurface::SmoothQuadratic);
        assert!(reference < fast);
        let schedule = MultiFidelityCostSchedule::demo(40.0);
        assert!(MetrologyMethod::ReferenceMetrology.per_reading_cost(&schedule)
            > MetrologyMethod::FastScreen.per_reading_cost(&schedule));
    }

    #[test]
    fn three_correlated_repeats_do_not_claim_independent_sqrt_n_gain() {
        let position = NormalizedPosition { x: 0.5, y: 0.5 };
        let independent = correlation_for(MetrologyStressCase::UnbiasedIndependent, MetrologyMethod::FastScreen);
        let correlated = correlation_for(MetrologyStressCase::CorrelatedRepeats, MetrologyMethod::FastScreen);
        assert_eq!(independent, 0.0);
        assert!(correlated > 0.0);
        let sigma = MetrologyMethod::FastScreen.nominal_sigma_mm(position, SyntheticSpatialSurface::SmoothQuadratic);
        let independent_three = sigma * (independent + (1.0 - independent) / 3.0).sqrt();
        let correlated_three = sigma * (correlated + (1.0 - correlated) / 3.0).sqrt();
        assert!(correlated_three > independent_three);
    }

    #[test]
    fn bias_and_stale_calibration_are_visible_as_negative_controls() {
        let position = NormalizedPosition { x: 0.8, y: 0.3 };
        let action = MeasurementAction { position, method: MetrologyMethod::FastScreen, repeat_count: 1 };
        let clean = measure_action(
            SyntheticSpatialSurface::SmoothQuadratic, MetrologyStressCase::UnbiasedIndependent, 7, &action, 0,
        );
        let biased = measure_action(
            SyntheticSpatialSurface::SmoothQuadratic, MetrologyStressCase::BiasedFastScreen, 7, &action, 0,
        );
        assert_ne!(clean.mean_mm, biased.mean_mm);
        let reference_action = MeasurementAction {
            position, method: MetrologyMethod::ReferenceMetrology, repeat_count: 1,
        };
        let current = measure_action(
            SyntheticSpatialSurface::SmoothQuadratic, MetrologyStressCase::UnbiasedIndependent, 7, &reference_action, 0,
        );
        let stale = measure_action(
            SyntheticSpatialSurface::SmoothQuadratic, MetrologyStressCase::StaleReferenceCalibration, 7, &reference_action, 0,
        );
        assert_ne!(current.mean_mm, stale.mean_mm);
        assert_eq!(stale.calibration_state, "stale_unrecognized");
    }

    #[test]
    fn tampered_reading_cost_metric_or_stop_reason_fails_closed() {
        let mut report = run_multifidelity_arm(
            SyntheticSpatialSurface::EdgeRollOff,
            MetrologyStressCase::UnbiasedIndependent,
            MultiFidelityStrategy::FixedStaged,
            3,
            MultiFidelityCostSchedule::demo(48.0),
        ).unwrap();
        report.observations[0].observation.mean_mm += 0.01;
        assert!(verify_multifidelity_report(&report).is_err());

        let mut cost_changed = run_multifidelity_arm(
            SyntheticSpatialSurface::SmoothQuadratic,
            MetrologyStressCase::UnbiasedIndependent,
            MultiFidelityStrategy::FastScreenOnly,
            3,
            MultiFidelityCostSchedule::demo(40.0),
        ).unwrap();
        cost_changed.observations[0].cost.acquisition_units += 2.0;
        assert!(verify_multifidelity_report(&cost_changed).is_err());

        let mut metric_changed = run_multifidelity_arm(
            SyntheticSpatialSurface::NullSpatialEffect,
            MetrologyStressCase::UnbiasedIndependent,
            MultiFidelityStrategy::SeededRandom,
            3,
            MultiFidelityCostSchedule::demo(40.0),
        ).unwrap();
        metric_changed.metrics.false_safe_point_count += 1;
        assert!(verify_multifidelity_report(&metric_changed).is_err());
    }

    #[test]
    fn invalid_schedule_and_repeat_count_fail_closed() {
        let mut invalid = MultiFidelityCostSchedule::demo(40.0);
        invalid.reference_cost_per_reading = -1.0;
        assert!(run_multifidelity_arm(
            SyntheticSpatialSurface::SmoothQuadratic,
            MetrologyStressCase::UnbiasedIndependent,
            MultiFidelityStrategy::ReferenceOnly,
            1, invalid,
        ).is_err());
        let bad_action = MeasurementAction {
            position: NormalizedPosition { x: 0.5, y: 0.5 },
            method: MetrologyMethod::FastScreen,
            repeat_count: 2,
        };
        assert!(incremental_cost(
            NormalizedPosition { x: 0.5, y: 0.5 },
            &bad_action,
            &MultiFidelityCostSchedule::demo(40.0),
            false,
        ).is_err());
    }
}
