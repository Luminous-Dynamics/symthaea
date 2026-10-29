// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded nonanticipative rolling-horizon policy for the synthetic airport energy hub.
//! At stage t, it observes the actual current interval, uses only supplied forecasts for
//! later intervals, commits the first action, verifies that transition, and carries SOC.
//! This is a replay/reference simulator, not a live airport controller.

use std::fmt;

use sha2::{Digest, Sha256};

use crate::aviation_energy_hub::{
    canonical_energy_hub_digest, seal_energy_hub_scenario, solve_exact_energy_hub_pareto,
    verify_energy_hub_plan, EnergyHubError, EnergyHubPlan, EnergyHubStepAction,
    EvaluatedEnergyHubPlan, TimeIndexedEnergyHubScenario, EnergyHubInterval,
};

const MAX_NODES_PER_STAGE: u64 = 250_000;

#[derive(Debug, Clone, PartialEq)]
pub struct ForecastWindow {
    pub issued_at_epoch: i64,
    pub available_at_epoch: i64,
    /// Forecast values for stages t+1 through the horizon end; current-stage conditions
    /// are taken from the actual observation and cannot be supplied by this vector.
    pub future_intervals: Vec<EnergyHubInterval>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RollingHorizonEpisode {
    pub realized_scenario: TimeIndexedEnergyHubScenario,
    pub decision_times_epoch: Vec<i64>,
    pub observation_available_at_epoch: Vec<i64>,
    /// One forecast snapshot per decision stage.
    pub forecasts_by_stage: Vec<ForecastWindow>,
}

/// Interval-local physical and source-flow accounting. Unlike the whole-horizon
/// service ledger, this record does not divide one interval's costs/emissions by the
/// full episode passenger-km denominator.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutedIntervalLedger {
    pub source_cost_amount: f64,
    pub accounted_lifecycle_co2e_kg: f64,
    pub primary_source_energy_kwh: f64,
    pub load_delivered_energy_kwh: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RollingHorizonStep {
    pub stage_index: usize,
    pub decision_at_epoch: i64,
    pub observation_available_at_epoch: i64,
    pub forecast_issued_at_epoch: i64,
    pub forecast_available_at_epoch: i64,
    pub initial_soc_ledger_ref: String,
    pub forecast_digest: String,
    pub planning_scenario_digest: String,
    pub action: EnergyHubStepAction,
    pub nodes_visited: u64,
    pub feasible_plans_seen: u64,
    pub realized_step_ledger: ExecutedIntervalLedger,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollingHorizonStatus {
    Completed,
    SearchBudgetExceeded { stage_index: usize, nodes_visited: u64 },
    NoFeasiblePlan { stage_index: usize },
    SolverFailure { stage_index: usize, source: EnergyHubError },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RollingHorizonRun {
    pub episode_digest: String,
    pub steps: Vec<RollingHorizonStep>,
    pub status: RollingHorizonStatus,
    pub final_soc_quanta: u32,
    /// Present only when every stage executed and the entire realized plan independently verifies.
    pub evaluated_realized_plan: Option<EvaluatedEnergyHubPlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollingHorizonError {
    InvalidScenarioDigest,
    InvalidStageCount,
    InvalidDecisionTime { stage_index: usize },
    ObservationUnavailable { stage_index: usize },
    ForecastUnavailable { stage_index: usize },
    InvalidForecastTime { stage_index: usize },
    ForecastWindowLength { stage_index: usize },
    ForecastIntervalIdentityMismatch { stage_index: usize, interval_index: usize },
    InvalidSearchBudget,
    InvalidForecastValues { stage_index: usize, source: EnergyHubError },
    ArithmeticOverflow,
}

impl fmt::Display for RollingHorizonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScenarioDigest => write!(f, "realized episode scenario digest is invalid"),
            Self::InvalidStageCount => write!(f, "decision, observation and forecast counts must match the horizon"),
            Self::InvalidDecisionTime { stage_index } => write!(f, "decision timestamps are invalid at stage {stage_index}"),
            Self::ObservationUnavailable { stage_index } => write!(f, "current observation is not available at stage {stage_index}"),
            Self::ForecastUnavailable { stage_index } => write!(f, "future forecast is not available at stage {stage_index}"),
            Self::InvalidForecastTime { stage_index } => write!(f, "forecast timestamps are inconsistent at stage {stage_index}"),
            Self::ForecastWindowLength { stage_index } => write!(f, "future forecast horizon length is wrong at stage {stage_index}"),
            Self::ForecastIntervalIdentityMismatch { stage_index, interval_index } => write!(f, "forecast interval identity mismatch at stage {stage_index}, offset {interval_index}"),
            Self::InvalidSearchBudget => write!(f, "node budget must be between 1 and the configured maximum"),
            Self::InvalidForecastValues { stage_index, source } => write!(f, "forecast model inputs are invalid at stage {stage_index}: {source}"),
            Self::ArithmeticOverflow => write!(f, "rolling-horizon episode accounting overflowed"),
        }
    }
}

impl std::error::Error for RollingHorizonError {}

/// Calculate a digest for a forecast snapshot, including issue/availability times and
/// all future interval fields. This is content identity, not publisher authentication.
pub fn canonical_forecast_digest(window: &ForecastWindow) -> Result<String, RollingHorizonError> {
    if window.issued_at_epoch < 0 || window.available_at_epoch < window.issued_at_epoch {
        return Err(RollingHorizonError::InvalidForecastTime { stage_index: 0 });
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"symthaea.clean-airline.forecast-window.v1\0");
    put_i64(&mut bytes, window.issued_at_epoch);
    put_i64(&mut bytes, window.available_at_epoch);
    put_u32(&mut bytes, usize_to_u32(window.future_intervals.len())?);
    for interval in &window.future_intervals {
        encode_interval(&mut bytes, interval);
    }
    Ok(sha256_hex(&bytes))
}

/// Validate all temporal inputs, then execute a rolling-horizon policy. A solver timeout
/// or infeasibility returns a partial trace with a distinct status; it is never converted
/// into a successful or zero-cost episode.
pub fn run_rolling_horizon(
    episode: &RollingHorizonEpisode,
    max_nodes_per_stage: u64,
) -> Result<RollingHorizonRun, RollingHorizonError> {
    if max_nodes_per_stage == 0 || max_nodes_per_stage > MAX_NODES_PER_STAGE {
        return Err(RollingHorizonError::InvalidSearchBudget);
    }
    let realized = &episode.realized_scenario;
    let realized_digest = canonical_energy_hub_digest(realized)
        .map_err(|_| RollingHorizonError::InvalidScenarioDigest)?;
    if realized.input_digest != realized_digest {
        return Err(RollingHorizonError::InvalidScenarioDigest);
    }

    let horizon = realized.intervals.len();
    if horizon == 0
        || episode.decision_times_epoch.len() != horizon
        || episode.observation_available_at_epoch.len() != horizon
        || episode.forecasts_by_stage.len() != horizon
    {
        return Err(RollingHorizonError::InvalidStageCount);
    }
    for stage in 0..horizon {
        let decision_at = episode.decision_times_epoch[stage];
        let observation_available = episode.observation_available_at_epoch[stage];
        if decision_at < 0
            || (stage > 0 && decision_at <= episode.decision_times_epoch[stage - 1])
        {
            return Err(RollingHorizonError::InvalidDecisionTime { stage_index: stage });
        }
        if observation_available < 0 || observation_available > decision_at {
            return Err(RollingHorizonError::ObservationUnavailable { stage_index: stage });
        }
        let forecast = &episode.forecasts_by_stage[stage];
        if forecast.issued_at_epoch < 0 || forecast.available_at_epoch < forecast.issued_at_epoch {
            return Err(RollingHorizonError::InvalidForecastTime { stage_index: stage });
        }
        if forecast.available_at_epoch > decision_at {
            return Err(RollingHorizonError::ForecastUnavailable { stage_index: stage });
        }
        if forecast.future_intervals.len() != horizon - stage - 1 {
            return Err(RollingHorizonError::ForecastWindowLength { stage_index: stage });
        }
        for (offset, predicted) in forecast.future_intervals.iter().enumerate() {
            let actual_index = stage + 1 + offset;
            if predicted.id != realized.intervals[actual_index].id {
                return Err(RollingHorizonError::ForecastIntervalIdentityMismatch {
                    stage_index: stage,
                    interval_index: actual_index,
                });
            }
        }
        // Validate the input values without consulting future realized values. The
        // forecast is combined with the observed current interval below.
        let _ = canonical_forecast_digest(forecast)?;
    }

    let episode_digest = canonical_episode_digest(episode)?;
    let mut soc = realized.storage.initial_soc_quanta;
    let mut actions = Vec::with_capacity(horizon);
    let mut steps = Vec::with_capacity(horizon);

    for stage in 0..horizon {
        let forecast = &episode.forecasts_by_stage[stage];
        let mut planning_scenario = realized.clone();
        planning_scenario.scenario_id = format!("{}-rolling-stage-{}", realized.scenario_id, stage);
        planning_scenario.input_digest = "pending".to_owned();
        planning_scenario.storage.initial_soc_quanta = soc;
        let initial_soc_ledger_ref =
            prefix_inventory_ledger_ref(&realized.storage.initial_soc_ledger_ref, &steps)?;
        planning_scenario.storage.initial_soc_ledger_ref = initial_soc_ledger_ref.clone();
        planning_scenario.intervals.clear();
        planning_scenario.intervals.push(realized.intervals[stage].clone());
        planning_scenario.intervals.extend(forecast.future_intervals.iter().cloned());
        seal_energy_hub_scenario(&mut planning_scenario)
            .map_err(|source| RollingHorizonError::InvalidForecastValues {
                stage_index: stage,
                source,
            })?;
        let planning_digest = planning_scenario.input_digest.clone();
        let forecast_digest = canonical_forecast_digest(forecast)?;

        let exact = match solve_exact_energy_hub_pareto(&planning_scenario, max_nodes_per_stage) {
            Ok(result) => result,
            Err(EnergyHubError::SearchBudgetExceeded { nodes_visited }) => {
                return Ok(RollingHorizonRun {
                    episode_digest,
                    steps,
                    status: RollingHorizonStatus::SearchBudgetExceeded {
                        stage_index: stage,
                        nodes_visited,
                    },
                    final_soc_quanta: soc,
                    evaluated_realized_plan: None,
                });
            }
            Err(EnergyHubError::NoFeasiblePlan) => {
                return Ok(RollingHorizonRun {
                    episode_digest,
                    steps,
                    status: RollingHorizonStatus::NoFeasiblePlan { stage_index: stage },
                    final_soc_quanta: soc,
                    evaluated_realized_plan: None,
                });
            }
            Err(source) => {
                return Ok(RollingHorizonRun {
                    episode_digest,
                    steps,
                    status: RollingHorizonStatus::SolverFailure {
                        stage_index: stage,
                        source,
                    },
                    final_soc_quanta: soc,
                    evaluated_realized_plan: None,
                });
            }
        };

        let selected = exact.frontier.first().ok_or(RollingHorizonError::ArithmeticOverflow)?;
        let action = selected.plan.actions.first().ok_or(RollingHorizonError::ArithmeticOverflow)?.clone();

        // The chosen action is the first action of a complete, independently
        // verified plan returned by the exact solver. A per-interval P0 service ledger
        // is intentionally not built here: a valid zero-demand interval may charge
        // storage and deliver no load until a later interval.
        let realized_interval = &realized.intervals[stage];
        let source_cost_amount =
            f64::from(realized_interval.renewable_generation_quanta)
                * realized_interval.renewable_cost_per_quantum
            + f64::from(action.grid_import_quanta) * realized_interval.grid_cost_per_quantum;
        let accounted_lifecycle_co2e_kg =
            f64::from(realized_interval.renewable_generation_quanta)
                * realized_interval.renewable_lifecycle_co2e_kg_per_quantum
            + f64::from(action.grid_import_quanta)
                * realized_interval.grid_lifecycle_co2e_kg_per_quantum;
        let primary_source_energy_kwh =
            f64::from(realized_interval.renewable_generation_quanta
                .checked_add(action.grid_import_quanta)
                .ok_or(RollingHorizonError::ArithmeticOverflow)?)
                * realized.quantum_kwh;
        let load_delivered_energy_kwh =
            f64::from(realized_interval.demand_quanta) * realized.quantum_kwh;
        if !source_cost_amount.is_finite()
            || !accounted_lifecycle_co2e_kg.is_finite()
            || !primary_source_energy_kwh.is_finite()
            || !load_delivered_energy_kwh.is_finite()
        {
            return Ok(RollingHorizonRun {
                episode_digest,
                steps,
                status: RollingHorizonStatus::SolverFailure {
                    stage_index: stage,
                    source: EnergyHubError::ArithmeticOverflow,
                },
                final_soc_quanta: soc,
                evaluated_realized_plan: None,
            });
        }
        let realized_step_ledger = ExecutedIntervalLedger {
            source_cost_amount,
            accounted_lifecycle_co2e_kg,
            primary_source_energy_kwh,
            load_delivered_energy_kwh,
        };
        soc = action.soc_after_quanta;
        actions.push(action.clone());
        steps.push(RollingHorizonStep {
            stage_index: stage,
            decision_at_epoch: episode.decision_times_epoch[stage],
            observation_available_at_epoch: episode.observation_available_at_epoch[stage],
            forecast_issued_at_epoch: forecast.issued_at_epoch,
            forecast_available_at_epoch: forecast.available_at_epoch,
            initial_soc_ledger_ref,
            forecast_digest,
            planning_scenario_digest: planning_digest,
            action,
            nodes_visited: exact.nodes_visited,
            feasible_plans_seen: exact.feasible_plans_seen,
            realized_step_ledger,
        });
    }

    let realized_plan = EnergyHubPlan {
        scenario_id: realized.scenario_id.clone(),
        input_digest: realized.input_digest.clone(),
        actions,
    };
    let evaluated_realized_plan = match verify_energy_hub_plan(realized, &realized_plan) {
        Ok(evaluated) => evaluated,
        Err(source) => {
            return Ok(RollingHorizonRun {
                episode_digest,
                steps,
                status: RollingHorizonStatus::SolverFailure {
                    stage_index: horizon,
                    source,
                },
                final_soc_quanta: soc,
                evaluated_realized_plan: None,
            });
        }
    };

    Ok(RollingHorizonRun {
        episode_digest,
        steps,
        status: RollingHorizonStatus::Completed,
        final_soc_quanta: soc,
        evaluated_realized_plan: Some(evaluated_realized_plan),
    })
}

fn canonical_episode_digest(
    episode: &RollingHorizonEpisode,
) -> Result<String, RollingHorizonError> {
    let realized_digest = canonical_energy_hub_digest(&episode.realized_scenario)
        .map_err(|_| RollingHorizonError::InvalidScenarioDigest)?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"symthaea.clean-airline.rolling-horizon-episode.v1\0");
    put_string(&mut bytes, &episode.realized_scenario.scenario_id);
    put_string(&mut bytes, &realized_digest);
    put_u32(&mut bytes, usize_to_u32(episode.decision_times_epoch.len())?);
    for stage in 0..episode.decision_times_epoch.len() {
        put_i64(&mut bytes, episode.decision_times_epoch[stage]);
        put_i64(&mut bytes, episode.observation_available_at_epoch[stage]);
        let forecast_digest = canonical_forecast_digest(&episode.forecasts_by_stage[stage])?;
        put_string(&mut bytes, &forecast_digest);
    }
    Ok(sha256_hex(&bytes))
}

/// Bind carried inventory to the already verified prior episode prefix. This digest
/// contains only committed actions and realized current-stage accounting, never future
/// realized values. At stage zero the external opening-inventory ledger reference is used.
fn prefix_inventory_ledger_ref(
    opening_ledger_ref: &str,
    prior_steps: &[RollingHorizonStep],
) -> Result<String, RollingHorizonError> {
    if prior_steps.is_empty() {
        return Ok(opening_ledger_ref.to_owned());
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"symthaea.clean-airline.storage-prefix-ledger.v1\0");
    put_string(&mut bytes, opening_ledger_ref);
    put_u32(&mut bytes, usize_to_u32(prior_steps.len())?);
    for step in prior_steps {
        put_u32(&mut bytes, usize_to_u32(step.stage_index)?);
        encode_action(&mut bytes, &step.action);
        put_f64(&mut bytes, step.realized_step_ledger.source_cost_amount);
        put_f64(
            &mut bytes,
            step.realized_step_ledger.accounted_lifecycle_co2e_kg,
        );
        put_f64(
            &mut bytes,
            step.realized_step_ledger.primary_source_energy_kwh,
        );
        put_f64(
            &mut bytes,
            step.realized_step_ledger.load_delivered_energy_kwh,
        );
        put_string(&mut bytes, &step.planning_scenario_digest);
    }
    Ok(sha256_hex(&bytes))
}

fn encode_action(bytes: &mut Vec<u8>, action: &EnergyHubStepAction) {
    put_string(bytes, &action.interval_id);
    for value in [
        action.renewable_direct_quanta,
        action.grid_import_quanta,
        action.battery_discharge_output_quanta,
        action.renewable_charge_input_quanta,
        action.renewable_curtailed_quanta,
        action.soc_before_quanta,
        action.soc_after_quanta,
    ] {
        put_u32(bytes, value);
    }
}

fn encode_interval(bytes: &mut Vec<u8>, interval: &EnergyHubInterval) {
    put_string(bytes, &interval.id);
    put_u32(bytes, interval.demand_quanta);
    put_u32(bytes, interval.charger_capacity_quanta);
    put_u32(bytes, interval.renewable_generation_quanta);
    put_u32(bytes, interval.grid_import_capacity_quanta);
    put_f64(bytes, interval.renewable_cost_per_quantum);
    put_f64(bytes, interval.renewable_lifecycle_co2e_kg_per_quantum);
    put_string(bytes, &interval.renewable_evidence_ref);
    bytes.push(encode_evidence_class(interval.renewable_evidence_class));
    put_f64(bytes, interval.grid_cost_per_quantum);
    put_f64(bytes, interval.grid_lifecycle_co2e_kg_per_quantum);
    put_string(bytes, &interval.grid_evidence_ref);
    bytes.push(encode_evidence_class(interval.grid_evidence_class));
}

fn encode_evidence_class(class: crate::aviation_energy_accounting::EvidenceClass) -> u8 {
    use crate::aviation_energy_accounting::EvidenceClass;
    match class {
        EvidenceClass::Measured => 0,
        EvidenceClass::SupplierReported => 1,
        EvidenceClass::LiteratureDerived => 2,
        EvidenceClass::CalibratedModel => 3,
        EvidenceClass::UncalibratedModel => 4,
        EvidenceClass::Synthetic => 5,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(7 + digest.len() * 2);
    result.push_str("sha256:");
    for byte in digest.iter() {
        use std::fmt::Write as _;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn usize_to_u32(value: usize) -> Result<u32, RollingHorizonError> {
    u32::try_from(value).map_err(|_| RollingHorizonError::ArithmeticOverflow)
}

fn put_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_string(bytes: &mut Vec<u8>, value: &str) {
    put_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn put_f64(bytes: &mut Vec<u8>, value: f64) {
    let normalized = if value == 0.0 { 0.0 } else { value };
    put_u64(bytes, normalized.to_bits());
}
