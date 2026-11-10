// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Minimal synthetic mechanical environment for HLB-VSSI-001.
//!
//! The fixture is deliberately a small, deterministic two-state thermal/vibration
//! system. Its purpose is to qualify the benchmark interface and fault taxonomy,
//! not to represent a physical machine or establish industrial validity.
//!
//! The evaluated investigator receives only VssiPublicTask and observations
//! returned for selected experiments. The hidden case and disposition oracle
//! remain inside VssiMechanicalEnvironment. The benchmark runner must keep the
//! environment handle and its evaluator-only methods outside the investigator
//! process; this module does not itself provide a process sandbox.
//!
//! State, parameter, structural, observation-model and out-of-support faults are
//! deliberately disjoint. In particular, the out-of-support case adds an
//! unregistered nonlinear mechanism and must be met with abstention rather than
//! forced classification or a parameter patch.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Fault family selected by the benchmark generator, never included in the
/// public task envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VssiFixtureCase {
    HealthyControl,
    InitialStateFault,
    ParameterFault,
    StructuralFault,
    ObservationModelFault,
    OpenWorldFault,
}

/// Operation the investigator proposes. It is a proposal only; this fixture
/// never promotes a candidate or mutates a production model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VssiDisposition {
    NoChange,
    StateUpdate,
    ParameterRepair,
    StructuralRepair,
    ObservationModelRepair,
    Abstain,
}

/// Registered observation or intervention protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VssiExperimentKind {
    PassiveObservation,
    ThermalImpulse,
    AlternatingLoad,
}

/// Nominal two-state mechanical model supplied to the investigator.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VssiMechanicalParameters {
    pub temperature_retention: f64,
    pub temperature_from_vibration: f64,
    pub vibration_from_temperature: f64,
    pub vibration_retention: f64,
    pub load_to_temperature: f64,
    pub load_to_vibration: f64,
}

impl Default for VssiMechanicalParameters {
    fn default() -> Self {
        Self {
            temperature_retention: 0.82,
            temperature_from_vibration: 0.12,
            vibration_from_temperature: 0.18,
            vibration_retention: 0.72,
            load_to_temperature: 0.20,
            load_to_vibration: 0.35,
        }
    }
}

/// One scalar sensor reading. Event time and acquisition time are kept
/// separate so the fixture can expose delayed and asynchronous observations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VssiSensorReading {
    pub value: f64,
    pub acquired_at: f64,
}

/// One event-time row. Either sensor may be absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VssiObservation {
    pub event_time: f64,
    pub temperature: Option<VssiSensorReading>,
    pub vibration: Option<VssiSensorReading>,
}

/// Public metadata for a registered experiment. Costs are abstract fixture
/// units, not money, energy measurements or calibrated physical risk.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VssiExperimentSpec {
    pub kind: VssiExperimentKind,
    pub observation_cost: f64,
    pub minimum_load: f64,
    pub maximum_load: f64,
    pub duration_ticks: u8,
}

/// Investigator-visible task. It intentionally contains no fault-family or
/// expected-disposition field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VssiPublicTask {
    pub scenario_id: String,
    pub nominal_parameters: VssiMechanicalParameters,
    pub initial_observations: Vec<VssiObservation>,
    pub available_experiments: Vec<VssiExperimentSpec>,
    /// Maximum cumulative cost of investigations in abstract fixture units.
    pub observation_budget: f64,
    /// Declared correlated sensor-noise covariance for the synthetic fixture.
    /// It is a model input, not an empirical calibration claim.
    pub observation_covariance: [[f64; 2]; 2],
}

/// Evaluator-only hidden mechanism. Its fields are private by design.
#[derive(Debug, Clone)]
struct HiddenMechanism {
    case: VssiFixtureCase,
    actual_parameters: VssiMechanicalParameters,
    initial_temperature_offset: f64,
    temperature_sensor_bias: f64,
    vibration_sensor_gain: f64,
    unregistered_temperature_nonlinearity: f64,
}

/// Environment handle. Pass the public task and selected experiment
/// observations to the investigator, but keep this handle in the evaluator.
#[derive(Debug, Clone)]
pub struct VssiMechanicalEnvironment {
    task: VssiPublicTask,
    hidden: HiddenMechanism,
    spent_observation_cost: f64,
}

/// A disposition score intended only for the independent benchmark evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VssiDispositionScore {
    pub correct_disposition: bool,
    pub unnecessary_edit: bool,
    pub correct_abstention: bool,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum VssiFixtureError {
    #[error("scenario identifier must not be empty")]
    EmptyScenarioId,
    #[error("synthetic trajectory became non-finite at tick {tick}")]
    NonFiniteTrajectory { tick: u8 },
    #[error("experiment {kind:?} is not registered for this task")]
    UnregisteredExperiment { kind: VssiExperimentKind },
    #[error("experiment cost {requested} exceeds remaining budget {remaining}")]
    BudgetExceeded { remaining: f64, requested: f64 },
}

impl VssiMechanicalEnvironment {
    /// Create one deterministic task from a hidden benchmark case.
    ///
    /// The case is a generator/evaluator input. Do not forward it to the
    /// evaluated investigator or serialize it into the public task manifest.
    pub fn new(
        scenario_id: impl Into<String>,
        case: VssiFixtureCase,
    ) -> Result<Self, VssiFixtureError> {
        let scenario_id = scenario_id.into();
        if scenario_id.trim().is_empty() {
            return Err(VssiFixtureError::EmptyScenarioId);
        }

        let nominal_parameters = VssiMechanicalParameters::default();
        let hidden = hidden_mechanism(case, nominal_parameters);
        let initial_observations = simulate(&hidden, VssiExperimentKind::PassiveObservation)?
            .into_iter()
            .filter(|observation| observation.event_time <= 3.0)
            .collect();

        let task = VssiPublicTask {
            scenario_id,
            nominal_parameters,
            initial_observations,
            available_experiments: vec![
                VssiExperimentSpec {
                    kind: VssiExperimentKind::PassiveObservation,
                    observation_cost: 0.25,
                    minimum_load: 0.15,
                    maximum_load: 0.15,
                    duration_ticks: 16,
                },
                VssiExperimentSpec {
                    kind: VssiExperimentKind::ThermalImpulse,
                    observation_cost: 0.7,
                    minimum_load: 0.15,
                    maximum_load: 0.95,
                    duration_ticks: 16,
                },
                VssiExperimentSpec {
                    kind: VssiExperimentKind::AlternatingLoad,
                    observation_cost: 1.0,
                    minimum_load: -0.35,
                    maximum_load: 0.75,
                    duration_ticks: 16,
                },
            ],
            observation_budget: 1.2,
            observation_covariance: [[0.04, 0.028], [0.028, 0.05]],
        };

        Ok(Self { task, hidden, spent_observation_cost: 0.0 })
    }

    /// Return the investigator-visible task envelope.
    pub fn public_task(&self) -> &VssiPublicTask {
        &self.task
    }

    /// Release observations for one selected, registered experiment.
    ///
    /// This method models the environment boundary: the investigator chooses
    /// a protocol identifier and receives observations, not hidden state,
    /// fault labels or simulator parameters.
    pub fn observe_experiment(
        &mut self,
        kind: VssiExperimentKind,
    ) -> Result<Vec<VssiObservation>, VssiFixtureError> {
        let spec = self.task.available_experiments.iter()
            .find(|spec| spec.kind == kind)
            .copied()
            .ok_or(VssiFixtureError::UnregisteredExperiment { kind })?;
        let remaining = self.remaining_observation_budget();
        if spec.observation_cost > remaining + 1e-12 {
            return Err(VssiFixtureError::BudgetExceeded {
                remaining,
                requested: spec.observation_cost,
            });
        }
        let observations = simulate(&self.hidden, kind)?;
        self.spent_observation_cost += spec.observation_cost;
        Ok(observations)
    }

    /// Cumulative abstract experiment cost consumed by this environment.
    pub fn spent_observation_cost(&self) -> f64 {
        self.spent_observation_cost
    }

    /// Remaining abstract experiment budget.
    pub fn remaining_observation_budget(&self) -> f64 {
        (self.task.observation_budget - self.spent_observation_cost).max(0.0)
    }

    /// Unit-test oracle only. The independent campaign evaluator must live
    /// outside the investigator-facing crate/process boundary.
    #[cfg(test)]
    pub(crate) fn score_disposition(
        &self,
        proposed: VssiDisposition,
    ) -> VssiDispositionScore {
        let expected = match self.hidden.case {
            VssiFixtureCase::HealthyControl => VssiDisposition::NoChange,
            VssiFixtureCase::InitialStateFault => VssiDisposition::StateUpdate,
            VssiFixtureCase::ParameterFault => VssiDisposition::ParameterRepair,
            VssiFixtureCase::StructuralFault => VssiDisposition::StructuralRepair,
            VssiFixtureCase::ObservationModelFault => VssiDisposition::ObservationModelRepair,
            VssiFixtureCase::OpenWorldFault => VssiDisposition::Abstain,
        };
        VssiDispositionScore {
            correct_disposition: proposed == expected,
            unnecessary_edit: self.hidden.case == VssiFixtureCase::HealthyControl
                && matches!(
                    proposed,
                    VssiDisposition::StateUpdate
                        | VssiDisposition::ParameterRepair
                        | VssiDisposition::StructuralRepair
                        | VssiDisposition::ObservationModelRepair
                ),
            correct_abstention: self.hidden.case == VssiFixtureCase::OpenWorldFault
                && proposed == VssiDisposition::Abstain,
        }
    }
}

fn hidden_mechanism(
    case: VssiFixtureCase,
    nominal: VssiMechanicalParameters,
) -> HiddenMechanism {
    let mut actual_parameters = nominal;
    let mut initial_temperature_offset = 0.0;
    let mut temperature_sensor_bias = 0.0;
    let mut vibration_sensor_gain = 1.0;
    let mut unregistered_temperature_nonlinearity = 0.0;

    match case {
        VssiFixtureCase::HealthyControl => {}
        VssiFixtureCase::InitialStateFault => {
            initial_temperature_offset = 1.15;
        }
        VssiFixtureCase::ParameterFault => {
            actual_parameters.temperature_retention = 0.57;
        }
        VssiFixtureCase::StructuralFault => {
            // The nominal temperature <- vibration causal edge is absent.
            actual_parameters.temperature_from_vibration = 0.0;
        }
        VssiFixtureCase::ObservationModelFault => {
            temperature_sensor_bias = 0.65;
            vibration_sensor_gain = 1.22;
        }
        VssiFixtureCase::OpenWorldFault => {
            // This nonlinear mechanism is deliberately absent from the
            // registered nominal/parameter/structural repair vocabulary.
            unregistered_temperature_nonlinearity = 0.16;
        }
    }

    HiddenMechanism {
        case,
        actual_parameters,
        initial_temperature_offset,
        temperature_sensor_bias,
        vibration_sensor_gain,
        unregistered_temperature_nonlinearity,
    }
}

fn experiment_load(kind: VssiExperimentKind, tick: u8) -> f64 {
    match kind {
        VssiExperimentKind::PassiveObservation => 0.15,
        VssiExperimentKind::ThermalImpulse => {
            if (4..=6).contains(&tick) { 0.95 } else { 0.15 }
        }
        VssiExperimentKind::AlternatingLoad => {
            if tick % 4 < 2 { 0.75 } else { -0.35 }
        }
    }
}

fn simulate(
    hidden: &HiddenMechanism,
    experiment: VssiExperimentKind,
) -> Result<Vec<VssiObservation>, VssiFixtureError> {
    const TEMPERATURE_TICKS: [u8; 6] = [0, 1, 4, 7, 11, 15];
    const VIBRATION_TICKS: [u8; 6] = [0, 2, 4, 8, 11, 15];
    const COMMON_MODE: [f64; 8] = [0.0, 0.02, -0.015, 0.03, -0.02, 0.012, -0.01, 0.018];
    const TEMPERATURE_NOISE: [f64; 8] = [0.01, -0.01, 0.005, -0.006, 0.008, -0.004, 0.003, -0.007];
    const VIBRATION_NOISE: [f64; 8] = [-0.006, 0.004, -0.003, 0.007, -0.005, 0.002, -0.004, 0.006];

    let parameters = hidden.actual_parameters;
    let mut temperature = 0.25 + hidden.initial_temperature_offset;
    let mut vibration = 0.08;
    let mut observations = Vec::new();

    for tick in 0..16_u8 {
        let common = COMMON_MODE[(tick as usize) % COMMON_MODE.len()];
        let temperature_value = temperature
            + hidden.temperature_sensor_bias
            + common
            + TEMPERATURE_NOISE[(tick as usize) % TEMPERATURE_NOISE.len()];
        let vibration_value = hidden.vibration_sensor_gain * vibration
            + 0.9 * common
            + VIBRATION_NOISE[(tick as usize) % VIBRATION_NOISE.len()];

        let temperature_reading = TEMPERATURE_TICKS.contains(&tick).then_some(
            VssiSensorReading {
                value: temperature_value,
                acquired_at: tick as f64 + 0.12,
            },
        );
        let vibration_reading = VIBRATION_TICKS.contains(&tick).then_some(
            VssiSensorReading {
                value: vibration_value,
                acquired_at: tick as f64 + 0.31,
            },
        );
        if temperature_reading.is_some() || vibration_reading.is_some() {
            observations.push(VssiObservation {
                event_time: tick as f64,
                temperature: temperature_reading,
                vibration: vibration_reading,
            });
        }

        if !temperature_value.is_finite() || !vibration_value.is_finite() {
            return Err(VssiFixtureError::NonFiniteTrajectory { tick });
        }

        let load = experiment_load(experiment, tick);
        let next_temperature = parameters.temperature_retention * temperature
            + parameters.temperature_from_vibration * vibration
            + parameters.load_to_temperature * load
            + hidden.unregistered_temperature_nonlinearity * temperature.powi(2);
        let next_vibration = parameters.vibration_from_temperature * temperature
            + parameters.vibration_retention * vibration
            + parameters.load_to_vibration * load;

        if !next_temperature.is_finite() || !next_vibration.is_finite() {
            return Err(VssiFixtureError::NonFiniteTrajectory { tick });
        }
        temperature = next_temperature;
        vibration = next_vibration;
    }

    Ok(observations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_case_has_an_explicit_expected_disposition() {
        let cases = [
            (VssiFixtureCase::HealthyControl, VssiDisposition::NoChange),
            (VssiFixtureCase::InitialStateFault, VssiDisposition::StateUpdate),
            (VssiFixtureCase::ParameterFault, VssiDisposition::ParameterRepair),
            (VssiFixtureCase::StructuralFault, VssiDisposition::StructuralRepair),
            (
                VssiFixtureCase::ObservationModelFault,
                VssiDisposition::ObservationModelRepair,
            ),
            (VssiFixtureCase::OpenWorldFault, VssiDisposition::Abstain),
        ];

        for (index, (case, expected)) in cases.into_iter().enumerate() {
            let environment =
                VssiMechanicalEnvironment::new(format!("vssi-v0-1-{index}"), case).unwrap();
            assert!(environment.score_disposition(expected).correct_disposition);
        }
    }

    #[test]
    fn healthy_control_penalizes_unnecessary_model_edits() {
        let environment = VssiMechanicalEnvironment::new(
            "vssi-healthy-negative-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();

        let score = environment.score_disposition(VssiDisposition::ParameterRepair);
        assert!(!score.correct_disposition);
        assert!(score.unnecessary_edit);
        assert!(!score.correct_abstention);
    }

    #[test]
    fn open_world_fault_requires_abstention() {
        let environment = VssiMechanicalEnvironment::new(
            "vssi-open-world-negative-control",
            VssiFixtureCase::OpenWorldFault,
        )
        .unwrap();

        let abstention = environment.score_disposition(VssiDisposition::Abstain);
        let forced_repair = environment.score_disposition(VssiDisposition::ParameterRepair);
        assert!(abstention.correct_abstention);
        assert!(!forced_repair.correct_disposition);
    }

    #[test]
    fn task_envelope_does_not_serialize_the_hidden_fault_label() {
        let environment = VssiMechanicalEnvironment::new(
            "vssi-hidden-label-control",
            VssiFixtureCase::StructuralFault,
        )
        .unwrap();

        let serialized = serde_json::to_string(environment.public_task()).unwrap();
        assert!(!serialized.contains("StructuralFault"));
        assert!(!serialized.contains("expected_disposition"));
        assert!(!serialized.contains("fault_family"));
    }

    #[test]
    fn investigation_protocols_produce_distinct_reproducible_traces() {
        let mut pulse_environment = VssiMechanicalEnvironment::new(
            "vssi-investigation-pulse-control",
            VssiFixtureCase::ParameterFault,
        )
        .unwrap();
        let pulse = pulse_environment
            .observe_experiment(VssiExperimentKind::ThermalImpulse)
            .unwrap();

        let mut pulse_replay_environment = VssiMechanicalEnvironment::new(
            "vssi-investigation-pulse-replay-control",
            VssiFixtureCase::ParameterFault,
        )
        .unwrap();
        let pulse_replay = pulse_replay_environment
            .observe_experiment(VssiExperimentKind::ThermalImpulse)
            .unwrap();

        let mut alternating_environment = VssiMechanicalEnvironment::new(
            "vssi-investigation-alternating-control",
            VssiFixtureCase::ParameterFault,
        )
        .unwrap();
        let alternating = alternating_environment
            .observe_experiment(VssiExperimentKind::AlternatingLoad)
            .unwrap();

        assert_eq!(pulse, pulse_replay);
        assert_ne!(pulse, alternating);
    }

    #[test]
    fn observations_keep_event_time_acquisition_time_and_missingness_separate() {
        let mut environment = VssiMechanicalEnvironment::new(
            "vssi-irregular-observation-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();

        let trace = environment
            .observe_experiment(VssiExperimentKind::PassiveObservation)
            .unwrap();
        assert!(trace
            .iter()
            .any(|row| row.temperature.is_none() && row.vibration.is_some()));
        assert!(trace
            .iter()
            .any(|row| row.temperature.is_some() && row.vibration.is_none()));
        for row in trace {
            if let Some(reading) = row.temperature {
                assert!(reading.acquired_at > row.event_time);
                assert!(reading.value.is_finite());
            }
            if let Some(reading) = row.vibration {
                assert!(reading.acquired_at > row.event_time);
                assert!(reading.value.is_finite());
            }
        }
    }

    #[test]
    fn experiment_catalog_declares_bounded_cost_and_input_envelopes() {
        let environment = VssiMechanicalEnvironment::new(
            "vssi-budget-contract-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();

        let task = environment.public_task();
        assert_eq!(task.available_experiments.len(), 3);
        assert_eq!(task.observation_budget, 1.2);
        assert_eq!(task.available_experiments[0].observation_cost, 0.25);
        assert!(task.available_experiments.iter().all(|spec| {
            spec.observation_cost.is_finite()
                && spec.observation_cost >= 0.0
                && spec.minimum_load <= spec.maximum_load
                && spec.duration_ticks == 16
        }));
        assert!(task.observation_covariance[0][1] > 0.0);
        assert_eq!(
            task.observation_covariance[0][1],
            task.observation_covariance[1][0]
        );
    }

    #[test]
    fn experiment_budget_is_enforced_before_observations_are_released() {
        let mut environment = VssiMechanicalEnvironment::new(
            "vssi-budget-enforcement-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();

        environment
            .observe_experiment(VssiExperimentKind::ThermalImpulse)
            .unwrap();
        assert!((environment.spent_observation_cost() - 0.7).abs() < 1e-12);
        assert!((environment.remaining_observation_budget() - 0.5).abs() < 1e-12);

        let rejected = environment.observe_experiment(VssiExperimentKind::AlternatingLoad);
        assert!(matches!(rejected, Err(VssiFixtureError::BudgetExceeded { .. })));
        assert!((environment.spent_observation_cost() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn repeated_passive_acquisition_also_consumes_the_budget() {
        let mut environment = VssiMechanicalEnvironment::new(
            "vssi-passive-budget-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();

        for _ in 0..4 {
            environment
                .observe_experiment(VssiExperimentKind::PassiveObservation)
                .unwrap();
        }
        assert!((environment.spent_observation_cost() - 1.0).abs() < 1e-12);

        let rejected = environment.observe_experiment(VssiExperimentKind::PassiveObservation);
        assert!(matches!(rejected, Err(VssiFixtureError::BudgetExceeded { .. })));
        assert!((environment.spent_observation_cost() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn distinct_hidden_cases_are_observably_distinguishable_by_at_least_one_protocol() {
        let cases = [
            VssiFixtureCase::HealthyControl,
            VssiFixtureCase::InitialStateFault,
            VssiFixtureCase::ParameterFault,
            VssiFixtureCase::StructuralFault,
            VssiFixtureCase::ObservationModelFault,
            VssiFixtureCase::OpenWorldFault,
        ];
        let protocols = [
            VssiExperimentKind::PassiveObservation,
            VssiExperimentKind::ThermalImpulse,
            VssiExperimentKind::AlternatingLoad,
        ];

        for left_index in 0..cases.len() {
            for right_index in (left_index + 1)..cases.len() {
                let mut distinguishable = false;
                for protocol in protocols {
                    let mut left = VssiMechanicalEnvironment::new(
                        format!("vssi-separability-left-{left_index}-{right_index}"),
                        cases[left_index],
                    )
                    .unwrap();
                    let mut right = VssiMechanicalEnvironment::new(
                        format!("vssi-separability-right-{left_index}-{right_index}"),
                        cases[right_index],
                    )
                    .unwrap();
                    let left_trace = left.observe_experiment(protocol).unwrap();
                    let right_trace = right.observe_experiment(protocol).unwrap();
                    if left_trace != right_trace {
                        distinguishable = true;
                        break;
                    }
                }
                assert!(
                    distinguishable,
                    "hidden cases {left_index} and {right_index} have identical registered traces"
                );
            }
        }
    }

    #[test]
    fn empty_scenario_identifier_is_rejected() {
        assert_eq!(
            VssiMechanicalEnvironment::new("  ", VssiFixtureCase::HealthyControl).unwrap_err(),
            VssiFixtureError::EmptyScenarioId
        );
    }
}
