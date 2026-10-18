// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cost-constrained spatial calibration experiment design.
//!
//! Candidate selection receives only observations acquired so far, available
//! locations, and a declared synthetic cost schedule. Latent surface values are
//! reserved for post-hoc evaluation. Cost units are abstract benchmark units;
//! they are not seconds, currency, or an estimate of physical machine cost.

use crate::crypto_digest::sha256;
use crate::fdm_spatial_benchmark::NormalizedPosition;
use crate::fdm_spatial_experiment_design::{
    ADDITIONAL_OBSERVATION_BUDGET, INITIAL_OBSERVATION_COUNT, SamplingStrategy,
    SpatialDesignObservation, SyntheticSpatialSurface, candidate_positions, choose_next_location,
    distance, fixed_stratified_priority, initial_positions, latent_surface, measure,
    predict_idw, predict_nearest, same_position,
};
use serde::{Deserialize, Serialize};

pub const FDM_COST_BUDGET_SCHEMA: &str = "symthaea.fdm-cost-budget-synthetic.v1";
pub const COST_BUDGET_LEVELS: [f64; 6] = [17.0, 20.0, 23.0, 26.0, 29.0, 32.0];
const EVALUATION_GRID_DIVISIONS: usize = 20;
const CRITICAL_ERROR_THRESHOLD_MM: f64 = 0.10;
const CRITICAL_REGION_DETECTION_RADIUS: f64 = 0.08;
const EPSILON: f64 = 1.0e-9;

/// How a measurement-cost schedule prices movement and acquisition.
/// Values are synthetic, versioned assumptions, not observed machine costs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticCostSchedule {
    pub schema_version: String,
    pub schedule_name: String,
    pub budget_units: f64,
    pub one_time_setup_units: f64,
    pub acquisition_units_per_measurement: f64,
    pub travel_units_per_normalized_distance: f64,
    pub edge_zone_width_normalized: f64,
    pub edge_measurement_surcharge_units: f64,
}

impl SyntheticCostSchedule {
    /// A deliberately simple schedule whose assumptions remain inspectable.
    pub fn demo(budget_units: f64) -> Self {
        Self {
            schema_version: FDM_COST_BUDGET_SCHEMA.to_owned(),
            schedule_name: "synthetic-travel-acquisition-edge-v1".to_owned(),
            budget_units,
            one_time_setup_units: 2.0,
            acquisition_units_per_measurement: 1.0,
            travel_units_per_normalized_distance: 2.0,
            edge_zone_width_normalized: 0.15,
            edge_measurement_surcharge_units: 0.35,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostBudgetStrategy {
    FixedStratified,
    FarthestPoint,
    ModelDisagreement,
    CoverageGuardedDisagreement,
    CostNormalizedDisagreement,
    SeededRandom,
}

impl CostBudgetStrategy {
    pub fn name(self) -> &'static str {
        match self {
            Self::FixedStratified => "fixed_stratified",
            Self::FarthestPoint => "farthest_point",
            Self::ModelDisagreement => "model_disagreement",
            Self::CoverageGuardedDisagreement => "coverage_guarded_disagreement",
            Self::CostNormalizedDisagreement => "cost_normalized_disagreement",
            Self::SeededRandom => "seeded_random",
        }
    }

    pub fn all() -> [Self; 6] {
        [
            Self::FixedStratified,
            Self::FarthestPoint,
            Self::ModelDisagreement,
            Self::CoverageGuardedDisagreement,
            Self::CostNormalizedDisagreement,
            Self::SeededRandom,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostBudgetStopReason {
    BudgetConstraintReached,
    FixedPlanNextPointUnaffordable,
    ObservationCapReached,
    CandidatePoolExhausted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementCostLedger {
    pub setup_units: f64,
    pub acquisition_units: f64,
    pub travel_units: f64,
    pub edge_surcharge_units: f64,
    pub incremental_total_units: f64,
    pub cumulative_total_units: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostedSpatialObservation {
    pub observation: SpatialDesignObservation,
    pub cost: MeasurementCostLedger,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostBudgetMetrics {
    pub planned_budget_units: f64,
    pub spent_budget_units: f64,
    pub unused_budget_units: f64,
    pub setup_cost_units: f64,
    pub acquisition_cost_units: f64,
    pub travel_cost_units: f64,
    pub edge_surcharge_cost_units: f64,
    pub normalized_route_distance: f64,
    pub observation_count: usize,
    pub additional_observation_count: usize,
    pub mean_nearest_sample_distance: f64,
    pub maximum_nearest_sample_distance: f64,
    pub nearest_sample_estimator_mae_mm: f64,
    pub nearest_sample_estimator_max_absolute_error_mm: f64,
    pub critical_region_evaluation_points: usize,
    /// Recall is undefined (None), not zero, if the surface has no critical grid points.
    pub critical_region_recall: Option<f64>,
    pub false_alarm_observation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostBudgetReport {
    pub schema_version: String,
    pub evidence_class: String,
    pub surface: SyntheticSpatialSurface,
    pub strategy: CostBudgetStrategy,
    pub seed: u64,
    pub cost_schedule: SyntheticCostSchedule,
    pub cost_schedule_digest_sha256: String,
    pub stop_reason: CostBudgetStopReason,
    pub metrics: CostBudgetMetrics,
    /// Ordered measurements with component costs and cumulative spend.
    pub observations: Vec<CostedSpatialObservation>,
    pub report_digest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostBudgetError {
    InvalidSchedule(&'static str),
    InitialDesignExceedsBudget { required_units_bits: u64, budget_units_bits: u64 },
    InvalidTrace(String),
    SerializationFailure,
}

impl std::fmt::Display for CostBudgetError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSchedule(reason) => write!(formatter, "invalid synthetic cost schedule: {reason}"),
            Self::InitialDesignExceedsBudget { .. } => {
                formatter.write_str("the common initial design exceeds the declared cost budget")
            }
            Self::InvalidTrace(reason) => write!(formatter, "invalid cost-budget trace: {reason}"),
            Self::SerializationFailure => formatter.write_str("could not serialize cost-budget report"),
        }
    }
}

impl std::error::Error for CostBudgetError {}

#[derive(Debug, Clone, Copy)]
struct IncrementalCost {
    setup: f64,
    acquisition: f64,
    travel: f64,
    edge: f64,
    total: f64,
}

fn validate_schedule(schedule: &SyntheticCostSchedule) -> Result<(), CostBudgetError> {
    if schedule.schema_version != FDM_COST_BUDGET_SCHEMA {
        return Err(CostBudgetError::InvalidSchedule("unsupported schema version"));
    }
    if schedule.schedule_name.trim().is_empty() {
        return Err(CostBudgetError::InvalidSchedule("schedule name is empty"));
    }
    if !schedule.budget_units.is_finite() || schedule.budget_units <= 0.0 {
        return Err(CostBudgetError::InvalidSchedule("budget must be finite and positive"));
    }
    for (value, field) in [
        (schedule.one_time_setup_units, "one-time setup cost"),
        (schedule.acquisition_units_per_measurement, "acquisition cost"),
        (schedule.travel_units_per_normalized_distance, "travel cost"),
        (schedule.edge_measurement_surcharge_units, "edge surcharge"),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(CostBudgetError::InvalidSchedule(field));
        }
    }
    if schedule.acquisition_units_per_measurement <= 0.0 {
        return Err(CostBudgetError::InvalidSchedule("acquisition cost must be positive"));
    }
    if !schedule.edge_zone_width_normalized.is_finite()
        || !(0.0..=0.5).contains(&schedule.edge_zone_width_normalized)
    {
        return Err(CostBudgetError::InvalidSchedule("edge zone width must be in [0, 0.5]"));
    }
    Ok(())
}

fn near_edge(point: NormalizedPosition, width: f64) -> bool {
    point.x.min(1.0 - point.x).min(point.y).min(1.0 - point.y) <= width
}

fn incremental_cost(
    previous: NormalizedPosition,
    target: NormalizedPosition,
    schedule: &SyntheticCostSchedule,
    is_first: bool,
) -> IncrementalCost {
    let setup = if is_first { schedule.one_time_setup_units } else { 0.0 };
    let acquisition = schedule.acquisition_units_per_measurement;
    let travel = distance(previous, target) * schedule.travel_units_per_normalized_distance;
    let edge = if near_edge(target, schedule.edge_zone_width_normalized) {
        schedule.edge_measurement_surcharge_units
    } else {
        0.0
    };
    IncrementalCost {
        setup,
        acquisition,
        travel,
        edge,
        total: setup + acquisition + travel + edge,
    }
}

fn schedule_digest(schedule: &SyntheticCostSchedule) -> Result<String, CostBudgetError> {
    let bytes = serde_json::to_vec(schedule).map_err(|_| CostBudgetError::SerializationFailure)?;
    let domain = b"symthaea.fdm-cost-budget-schedule.v1\0";
    let mut input = Vec::with_capacity(domain.len() + bytes.len());
    input.extend_from_slice(domain);
    input.extend_from_slice(&bytes);
    Ok(sha256(&input).to_hex())
}

fn report_digest(report: &CostBudgetReport) -> Result<String, CostBudgetError> {
    let bytes = serde_json::to_vec(&(
        &report.schema_version,
        &report.evidence_class,
        report.surface,
        report.strategy,
        report.seed,
        &report.cost_schedule,
        &report.cost_schedule_digest_sha256,
        report.stop_reason,
        &report.metrics,
        &report.observations,
    )).map_err(|_| CostBudgetError::SerializationFailure)?;
    let domain = b"symthaea.fdm-cost-budget-report.v1\0";
    let mut input = Vec::with_capacity(domain.len() + bytes.len());
    input.extend_from_slice(domain);
    input.extend_from_slice(&bytes);
    Ok(sha256(&input).to_hex())
}

fn costed_observation(
    index: usize,
    position: NormalizedPosition,
    measured_error_mm: f64,
    previous_position: NormalizedPosition,
    previous_cumulative: f64,
    schedule: &SyntheticCostSchedule,
) -> CostedSpatialObservation {
    let parts = incremental_cost(previous_position, position, schedule, index == 0);
    CostedSpatialObservation {
        observation: SpatialDesignObservation {
            observation_id: format!("cost-obs-{index:03}"),
            acquisition_step: index,
            position,
            measured_error_mm,
        },
        cost: MeasurementCostLedger {
            setup_units: parts.setup,
            acquisition_units: parts.acquisition,
            travel_units: parts.travel,
            edge_surcharge_units: parts.edge,
            incremental_total_units: parts.total,
            cumulative_total_units: previous_cumulative + parts.total,
        },
    }
}

fn choose_cost_normalized_disagreement(
    candidates: &[NormalizedPosition],
    observations: &[SpatialDesignObservation],
    last_position: NormalizedPosition,
    schedule: &SyntheticCostSchedule,
    seed: u64,
    step: usize,
) -> Option<NormalizedPosition> {
    candidates.iter().copied().max_by(|left, right| {
        let score = |target: NormalizedPosition| {
            let disagreement = match (
                predict_nearest(observations, target),
                predict_idw(observations, target),
            ) {
                (Some(nearest), Some(weighted)) => (nearest - weighted).abs(),
                _ => 0.0,
            };
            let cost = incremental_cost(last_position, target, schedule, false).total;
            if cost > 0.0 { disagreement / cost } else { 0.0 }
        };
        score(*left)
            .total_cmp(&score(*right))
            .then_with(|| {
                // Prefer better spatial coverage when disagreement/cost ties.
                let left_distance = observations.iter()
                    .map(|obs| distance(obs.position, *left))
                    .reduce(f64::min).unwrap_or(0.0);
                let right_distance = observations.iter()
                    .map(|obs| distance(obs.position, *right))
                    .reduce(f64::min).unwrap_or(0.0);
                left_distance.total_cmp(&right_distance)
            })
            .then_with(|| right.y.total_cmp(&left.y))
            .then_with(|| right.x.total_cmp(&left.x))
    }).or_else(|| {
        let _ = (seed, step);
        None
    })
}

fn select_candidate(
    strategy: CostBudgetStrategy,
    candidates: &[NormalizedPosition],
    observations: &[SpatialDesignObservation],
    last_position: NormalizedPosition,
    schedule: &SyntheticCostSchedule,
    seed: u64,
    step: usize,
    remaining_budget: f64,
) -> Option<NormalizedPosition> {
    if candidates.is_empty() {
        return None;
    }
    if strategy == CostBudgetStrategy::FixedStratified {
        return fixed_stratified_priority().into_iter()
            .find(|preferred| candidates.iter().any(|candidate| same_position(*candidate, *preferred)));
    }
    let affordable: Vec<_> = candidates.iter().copied().filter(|candidate| {
        incremental_cost(last_position, *candidate, schedule, false).total
            <= remaining_budget + EPSILON
    }).collect();
    if affordable.is_empty() {
        return None;
    }
    match strategy {
        CostBudgetStrategy::FixedStratified => unreachable!("handled above"),
        CostBudgetStrategy::FarthestPoint => choose_next_location(
            SamplingStrategy::FarthestPoint, &affordable, observations, seed, step,
        ),
        CostBudgetStrategy::ModelDisagreement => choose_next_location(
            SamplingStrategy::ModelDisagreement, &affordable, observations, seed, step,
        ),
        CostBudgetStrategy::CoverageGuardedDisagreement => choose_next_location(
            SamplingStrategy::CoverageGuardedDisagreement, &affordable, observations, seed, step,
        ),
        CostBudgetStrategy::CostNormalizedDisagreement => choose_cost_normalized_disagreement(
            &affordable, observations, last_position, schedule, seed, step,
        ),
        CostBudgetStrategy::SeededRandom => choose_next_location(
            SamplingStrategy::SeededRandom, &affordable, observations, seed, step,
        ),
        CostBudgetStrategy::FixedStratified => unreachable!("handled above"),
    }
}

fn evaluate_costed_design(
    surface: SyntheticSpatialSurface,
    observations: &[CostedSpatialObservation],
    schedule: &SyntheticCostSchedule,
) -> CostBudgetMetrics {
    let raw_observations: Vec<_> = observations.iter().map(|item| item.observation.clone()).collect();
    let mut distance_sum = 0.0;
    let mut maximum_distance: f64 = 0.0;
    let mut absolute_error_sum = 0.0;
    let mut maximum_absolute_error: f64 = 0.0;
    let mut evaluation_count = 0usize;
    let mut critical_points = 0usize;
    let mut detected_critical_points = 0usize;
    let risky_samples: Vec<_> = raw_observations.iter()
        .filter(|observation| observation.measured_error_mm.abs() >= CRITICAL_ERROR_THRESHOLD_MM)
        .map(|observation| observation.position)
        .collect();
    let mut false_alarm_count = 0usize;

    for observation in &raw_observations {
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
            let nearest_distance = raw_observations.iter()
                .map(|observation| distance(observation.position, target))
                .reduce(f64::min).unwrap_or(f64::INFINITY);
            distance_sum += nearest_distance;
            maximum_distance = maximum_distance.max(nearest_distance);

            let oracle = latent_surface(surface, target);
            let prediction = predict_nearest(&raw_observations, target).unwrap_or(0.0);
            let absolute_error = (prediction - oracle).abs();
            absolute_error_sum += absolute_error;
            maximum_absolute_error = maximum_absolute_error.max(absolute_error);
            evaluation_count += 1;

            if oracle.abs() >= CRITICAL_ERROR_THRESHOLD_MM {
                critical_points += 1;
                if risky_samples.iter().any(|sample| {
                    distance(*sample, target) <= CRITICAL_REGION_DETECTION_RADIUS
                }) {
                    detected_critical_points += 1;
                }
            }
        }
    }

    let sum_component = |select: fn(&MeasurementCostLedger) -> f64| {
        observations.iter().map(|observation| select(&observation.cost)).sum::<f64>()
    };
    let used = observations.last().map(|observation| observation.cost.cumulative_total_units).unwrap_or(0.0);
    CostBudgetMetrics {
        planned_budget_units: schedule.budget_units,
        spent_budget_units: used,
        unused_budget_units: (schedule.budget_units - used).max(0.0),
        setup_cost_units: sum_component(|cost| cost.setup_units),
        acquisition_cost_units: sum_component(|cost| cost.acquisition_units),
        travel_cost_units: sum_component(|cost| cost.travel_units),
        edge_surcharge_cost_units: sum_component(|cost| cost.edge_surcharge_units),
        normalized_route_distance: observations.iter().map(|observation| {
            if observation.observation.acquisition_step == 0 {
                distance(NormalizedPosition { x: 0.5, y: 0.5 }, observation.observation.position)
            } else {
                distance(
                    observations[observation.observation.acquisition_step - 1].observation.position,
                    observation.observation.position,
                )
            }
        }).sum(),
        observation_count: observations.len(),
        additional_observation_count: observations.len().saturating_sub(INITIAL_OBSERVATION_COUNT),
        mean_nearest_sample_distance: distance_sum / evaluation_count as f64,
        maximum_nearest_sample_distance: maximum_distance,
        nearest_sample_estimator_mae_mm: absolute_error_sum / evaluation_count as f64,
        nearest_sample_estimator_max_absolute_error_mm: maximum_absolute_error,
        critical_region_evaluation_points: critical_points,
        critical_region_recall: (critical_points > 0)
            .then_some(detected_critical_points as f64 / critical_points as f64),
        false_alarm_observation_count: false_alarm_count,
    }
}

/// Run a single synthetic surface/strategy/budget arm. Initial measurements are
/// identical across strategies; incremental actions must fit the remaining budget.
pub fn run_cost_budget_arm(
    surface: SyntheticSpatialSurface,
    strategy: CostBudgetStrategy,
    seed: u64,
    schedule: SyntheticCostSchedule,
) -> Result<CostBudgetReport, CostBudgetError> {
    validate_schedule(&schedule)?;
    let schedule_digest = schedule_digest(&schedule)?;
    let mut trace = Vec::with_capacity(INITIAL_OBSERVATION_COUNT + ADDITIONAL_OBSERVATION_BUDGET);
    let mut cumulative = 0.0;
    let mut last_position = NormalizedPosition { x: 0.5, y: 0.5 };

    for (index, position) in initial_positions().into_iter().enumerate() {
        let item = costed_observation(
            index, position, measure(surface, position, seed), last_position, cumulative, &schedule,
        );
        cumulative = item.cost.cumulative_total_units;
        last_position = position;
        trace.push(item);
    }
    if cumulative > schedule.budget_units + EPSILON {
        return Err(CostBudgetError::InitialDesignExceedsBudget {
            required_units_bits: cumulative.to_bits(),
            budget_units_bits: schedule.budget_units.to_bits(),
        });
    }

    let mut candidates = candidate_positions();
    let mut stop_reason = CostBudgetStopReason::ObservationCapReached;
    loop {
        if trace.len() >= INITIAL_OBSERVATION_COUNT + ADDITIONAL_OBSERVATION_BUDGET {
            stop_reason = CostBudgetStopReason::ObservationCapReached;
            break;
        }
        if candidates.is_empty() {
            stop_reason = CostBudgetStopReason::CandidatePoolExhausted;
            break;
        }
        let remaining = schedule.budget_units - cumulative;
        let selected = if strategy == CostBudgetStrategy::FixedStratified {
            let next_fixed = fixed_stratified_priority().into_iter()
                .find(|preferred| candidates.iter().any(|candidate| same_position(*candidate, *preferred)));
            match next_fixed {
                Some(position) => {
                    let required = incremental_cost(last_position, position, &schedule, false).total;
                    if required > remaining + EPSILON {
                        stop_reason = CostBudgetStopReason::FixedPlanNextPointUnaffordable;
                        break;
                    }
                    Some(position)
                }
                None => {
                    stop_reason = CostBudgetStopReason::CandidatePoolExhausted;
                    break;
                }
            }
        } else {
            let raw: Vec<_> = trace.iter().map(|item| item.observation.clone()).collect();
            select_candidate(
                strategy, &candidates, &raw, last_position, &schedule, seed,
                trace.len().saturating_sub(INITIAL_OBSERVATION_COUNT), remaining,
            )
        };

        let Some(position) = selected else {
            stop_reason = CostBudgetStopReason::BudgetConstraintReached;
            break;
        };
        let costs = incremental_cost(last_position, position, &schedule, false);
        if costs.total > remaining + EPSILON {
            return Err(CostBudgetError::InvalidTrace(
                "selector returned a measurement that exceeds the remaining budget".to_owned(),
            ));
        }
        let index = trace.len();
        let item = costed_observation(
            index, position, measure(surface, position, seed), last_position, cumulative, &schedule,
        );
        cumulative = item.cost.cumulative_total_units;
        last_position = position;
        trace.push(item);
        candidates.retain(|candidate| !same_position(*candidate, position));
    }

    let metrics = evaluate_costed_design(surface, &trace, &schedule);
    let mut report = CostBudgetReport {
        schema_version: FDM_COST_BUDGET_SCHEMA.to_owned(),
        evidence_class: "synthetic".to_owned(),
        surface,
        strategy,
        seed,
        cost_schedule: schedule,
        cost_schedule_digest_sha256: schedule_digest,
        stop_reason,
        metrics,
        observations: trace,
        report_digest_sha256: String::new(),
    };
    report.report_digest_sha256 = report_digest(&report)?;
    verify_cost_budget_report(&report)?;
    Ok(report)
}

/// Run the predeclared budget curve on each synthetic surface and each strategy.
pub fn run_cost_budget_benchmark(seed: u64) -> Result<Vec<CostBudgetReport>, CostBudgetError> {
    let mut reports = Vec::with_capacity(
        SyntheticSpatialSurface::all().len() * CostBudgetStrategy::all().len() * COST_BUDGET_LEVELS.len()
    );
    for surface in SyntheticSpatialSurface::all() {
        for strategy in CostBudgetStrategy::all() {
            for budget in COST_BUDGET_LEVELS {
                reports.push(run_cost_budget_arm(surface, strategy, seed, SyntheticCostSchedule::demo(budget))?);
            }
        }
    }
    Ok(reports)
}

/// Independently re-check budget arithmetic, trace order, input bounds, and report digest.
/// It does not establish that the synthetic surface is a physical printer model.
pub fn verify_cost_budget_report(report: &CostBudgetReport) -> Result<(), CostBudgetError> {
    validate_schedule(&report.cost_schedule)?;
    if report.schema_version != FDM_COST_BUDGET_SCHEMA || report.evidence_class != "synthetic" {
        return Err(CostBudgetError::InvalidTrace("unsupported schema or evidence class".to_owned()));
    }
    if report.cost_schedule_digest_sha256 != schedule_digest(&report.cost_schedule)? {
        return Err(CostBudgetError::InvalidTrace("cost schedule digest mismatch".to_owned()));
    }
    if !(INITIAL_OBSERVATION_COUNT..=INITIAL_OBSERVATION_COUNT + ADDITIONAL_OBSERVATION_BUDGET)
        .contains(&report.observations.len())
    {
        return Err(CostBudgetError::InvalidTrace("observation count is outside the frozen budget".to_owned()));
    }
    let expected_initial = initial_positions();
    let mut unique = std::collections::HashSet::new();
    let mut cumulative = 0.0;
    let mut previous = NormalizedPosition { x: 0.5, y: 0.5 };
    for (index, item) in report.observations.iter().enumerate() {
        let observation = &item.observation;
        if observation.acquisition_step != index || observation.observation_id != format!("cost-obs-{index:03}") {
            return Err(CostBudgetError::InvalidTrace("observation identity/order mismatch".to_owned()));
        }
        if index < expected_initial.len() && !same_position(observation.position, expected_initial[index]) {
            return Err(CostBudgetError::InvalidTrace("common initial-design position changed".to_owned()));
        }
        if !observation.position.x.is_finite()
            || !observation.position.y.is_finite()
            || !(0.0..=1.0).contains(&observation.position.x)
            || !(0.0..=1.0).contains(&observation.position.y)
            || !observation.measured_error_mm.is_finite()
        {
            return Err(CostBudgetError::InvalidTrace("non-finite or out-of-domain observation".to_owned()));
        }
        let key = ((observation.position.x * 10.0).round() as u32, (observation.position.y * 10.0).round() as u32);
        if !unique.insert(key) {
            return Err(CostBudgetError::InvalidTrace("duplicate observation position".to_owned()));
        }
        let expected = incremental_cost(previous, observation.position, &report.cost_schedule, index == 0);
        let close = |actual: f64, wanted: f64| actual.is_finite() && (actual - wanted).abs() <= 1.0e-9;
        if !close(item.cost.setup_units, expected.setup)
            || !close(item.cost.acquisition_units, expected.acquisition)
            || !close(item.cost.travel_units, expected.travel)
            || !close(item.cost.edge_surcharge_units, expected.edge)
            || !close(item.cost.incremental_total_units, expected.total)
        {
            return Err(CostBudgetError::InvalidTrace("per-measurement cost ledger mismatch".to_owned()));
        }
        cumulative += expected.total;
        if !close(item.cost.cumulative_total_units, cumulative) {
            return Err(CostBudgetError::InvalidTrace("cumulative cost ledger mismatch".to_owned()));
        }
        previous = observation.position;
    }
    if cumulative > report.cost_schedule.budget_units + EPSILON {
        return Err(CostBudgetError::InvalidTrace("reported trace exceeds its cost budget".to_owned()));
    }
    if !close_f64(report.metrics.spent_budget_units, cumulative)
        || !close_f64(report.metrics.unused_budget_units, (report.cost_schedule.budget_units - cumulative).max(0.0))
        || report.metrics.observation_count != report.observations.len()
        || report.metrics.additional_observation_count != report.observations.len() - INITIAL_OBSERVATION_COUNT
    {
        return Err(CostBudgetError::InvalidTrace("summary totals do not match cost ledger".to_owned()));
    }
    if report.report_digest_sha256 != report_digest(report)? {
        return Err(CostBudgetError::InvalidTrace("report digest mismatch".to_owned()));
    }
    Ok(())
}

fn close_f64(actual: f64, expected: f64) -> bool {
    actual.is_finite() && (actual - expected).abs() <= 1.0e-9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_cost_curve_has_equal_initial_design_and_all_declared_arms() {
        let reports = run_cost_budget_benchmark(42).unwrap();
        assert_eq!(
            reports.len(),
            SyntheticSpatialSurface::all().len() * CostBudgetStrategy::all().len() * COST_BUDGET_LEVELS.len()
        );
        for report in &reports {
            assert_eq!(report.evidence_class, "synthetic");
            assert!(report.observations.len() >= INITIAL_OBSERVATION_COUNT);
            assert!(report.observations.len() <= INITIAL_OBSERVATION_COUNT + ADDITIONAL_OBSERVATION_BUDGET);
            assert!(report.metrics.spent_budget_units <= report.metrics.planned_budget_units + EPSILON);
            assert!(report.metrics.unused_budget_units >= 0.0);
            assert_eq!(report.observations.len(), report.metrics.observation_count);
            verify_cost_budget_report(report).unwrap();
        }
    }

    #[test]
    fn identical_inputs_replay_identically() {
        let schedule = SyntheticCostSchedule::demo(23.0);
        let first = run_cost_budget_arm(
            SyntheticSpatialSurface::LocalizedHotspot,
            CostBudgetStrategy::CostNormalizedDisagreement,
            42,
            schedule.clone(),
        ).unwrap();
        let second = run_cost_budget_arm(
            SyntheticSpatialSurface::LocalizedHotspot,
            CostBudgetStrategy::CostNormalizedDisagreement,
            42,
            schedule,
        ).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn lower_budgets_never_spend_more_than_higher_budgets() {
        for strategy in CostBudgetStrategy::all() {
            let mut previous_budget = 0.0;
            for budget in COST_BUDGET_LEVELS {
                let report = run_cost_budget_arm(
                    SyntheticSpatialSurface::Anisotropic,
                    strategy,
                    9,
                    SyntheticCostSchedule::demo(budget),
                ).unwrap();
                assert!(report.metrics.spent_budget_units <= budget + EPSILON);
                assert!(report.metrics.spent_budget_units + EPSILON >= previous_budget);
                previous_budget = report.metrics.spent_budget_units;
            }
        }
    }

    #[test]
    fn report_verifier_rejects_cost_and_digest_tampering() {
        let mut report = run_cost_budget_arm(
            SyntheticSpatialSurface::EdgeRollOff,
            CostBudgetStrategy::FarthestPoint,
            12,
            SyntheticCostSchedule::demo(26.0),
        ).unwrap();
        report.observations[5].cost.travel_units += 0.5;
        assert!(verify_cost_budget_report(&report).is_err());

        let mut digest_tampered = run_cost_budget_arm(
            SyntheticSpatialSurface::SmoothQuadratic,
            CostBudgetStrategy::SeededRandom,
            12,
            SyntheticCostSchedule::demo(26.0),
        ).unwrap();
        digest_tampered.metrics.spent_budget_units += 0.01;
        assert!(verify_cost_budget_report(&digest_tampered).is_err());
    }

    #[test]
    fn invalid_cost_schedule_and_undersized_initial_budget_fail_closed() {
        let mut invalid = SyntheticCostSchedule::demo(20.0);
        invalid.travel_units_per_normalized_distance = f64::NAN;
        assert!(run_cost_budget_arm(
            SyntheticSpatialSurface::SmoothQuadratic,
            CostBudgetStrategy::FarthestPoint,
            1,
            invalid,
        ).is_err());
        assert!(matches!(
            run_cost_budget_arm(
                SyntheticSpatialSurface::SmoothQuadratic,
                CostBudgetStrategy::FarthestPoint,
                1,
                SyntheticCostSchedule::demo(10.0),
            ),
            Err(CostBudgetError::InitialDesignExceedsBudget { .. })
        ));
    }
}
