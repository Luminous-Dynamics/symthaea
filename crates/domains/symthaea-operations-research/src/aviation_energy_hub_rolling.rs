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
    verify_energy_hub_plan, EnergyHubError, EnergyHubMetrics, EnergyHubPlan, EnergyHubStepAction,
    EnergyStorageSpec, EvaluatedEnergyHubPlan, TimeIndexedEnergyHubScenario, EnergyHubInterval,
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

#[derive(Debug, Clone, PartialEq)]
pub struct RollingHorizonStep {
    pub stage_index: usize,
    pub decision_at_epoch: i64,
    pub observation_available_at_epoch: i64,
    pub forecast_digest: String,
    pub planning_scenario_digest: String,
    pub action: EnergyHubStepAction,
    pub nodes_visited: u64,
    pub feasible_plans_seen: u64,
    pub realized_step_metrics: EnergyHubMetrics,
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

        let mut step_scenario = planning_scenario.clone();
        step_scenario.scenario_id = format!("{}-committed-step-{}", realized.scenario_id, stage);
        step_scenario.input_digest = "pending".to_owned();
        step_scenario.intervals = vec![realized.intervals[stage].clone()];
        if stage + 1 < horizon {
            step_scenario.storage.minimum_terminal_soc_quanta = 0;
        }
        seal_energy_hub_scenario(&mut step_scenario)
            .map_err(|source| RollingHorizonError::InvalidForecastValues {
                stage_index: stage,
                source,
            })?;
        let step_plan = EnergyHubPlan {
            scenario_id: step_scenario.scenario_id.clone(),
            input_digest: step_scenario.input_digest.clone(),
            actions: vec![action.clone()],
        };
        let evaluated_step = match verify_energy_hub_plan(&step_scenario, &step_plan) {
            Ok(evaluated) => evaluated,
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
        soc = action.soc_after_quanta;
        actions.push(action.clone());
        steps.push(RollingHorizonStep {
            stage_index: stage,
            decision_at_epoch: episode.decision_times_epoch[stage],
            observation_available_at_epoch: episode.observation_available_at_epoch[stage],
            forecast_digest,
            planning_scenario_digest: planning_digest,
            action,
            nodes_visited: exact.nodes_visited,
            feasible_plans_seen: exact.feasible_plans_seen,
            realized_step_metrics: evaluated_step.metrics,
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
