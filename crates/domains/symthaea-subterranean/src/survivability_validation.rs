// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic field-survivability acceptance contracts.

use crate::actuator_isolation::{
    ActuatorIsolationPolicy, ActuatorIsolationSupervisor, PhysicalActuator,
};
use crate::capability_profile::{CapabilityDisposition, CapabilityProfile};
use crate::embodiment::SubterraneanEmbodiment;
use crate::field_envelope::{FieldEnvelopeAssessment, FieldEnvelopeSupervisor};
use crate::maintenance::MaintenanceAssessment;
use crate::partition_recovery::{
    PartitionObservation, PartitionRecoveryMode, PartitionRecoveryPolicy,
    PartitionRecoverySupervisor,
};
use crate::sensor_redundancy::{
    RedundantSensorFrame, SensorFusionReport, SensorFusionSupervisor, SensorSourceId,
    SensorSourceObservation,
};
use crate::types::{
    BATTERY_RATIO, CUTTER_TEMP_C, GAS_RISK, SubterraneanCommand, SubterraneanState,
};
use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurvivabilityContract {
    CriticalSensorQuorum,
    RobustMedianFusion,
    PersistentActuatorIsolation,
    ThermalPowerPrioritization,
    GracefulReturnDisposition,
    PartitionReconciliation,
    CheckpointReplayContinuity,
}

impl SurvivabilityContract {
    pub const ALL: [Self; 7] = [
        Self::CriticalSensorQuorum,
        Self::RobustMedianFusion,
        Self::PersistentActuatorIsolation,
        Self::ThermalPowerPrioritization,
        Self::GracefulReturnDisposition,
        Self::PartitionReconciliation,
        Self::CheckpointReplayContinuity,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::CriticalSensorQuorum => "critical_sensor_quorum",
            Self::RobustMedianFusion => "robust_median_fusion",
            Self::PersistentActuatorIsolation => "persistent_actuator_isolation",
            Self::ThermalPowerPrioritization => "thermal_power_prioritization",
            Self::GracefulReturnDisposition => "graceful_return_disposition",
            Self::PartitionReconciliation => "partition_reconciliation",
            Self::CheckpointReplayContinuity => "checkpoint_replay_continuity",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SurvivabilityGateFailure {
    pub contract: SurvivabilityContract,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SurvivabilityValidationReport {
    pub passed: Vec<SurvivabilityContract>,
    pub failures: Vec<SurvivabilityGateFailure>,
}

impl SurvivabilityValidationReport {
    pub fn is_success(&self) -> bool {
        self.failures.is_empty() && self.passed.len() == SurvivabilityContract::ALL.len()
    }

    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SurvivabilityValidator;

impl SurvivabilityValidator {
    fn redundant_frame(state: &SubterraneanState, sequence: u64) -> RedundantSensorFrame {
        RedundantSensorFrame {
            observations: vec![
                SensorSourceObservation::from_state(SensorSourceId(0), sequence, state),
                SensorSourceObservation::from_state(SensorSourceId(1), sequence, state),
            ],
        }
    }

    fn evaluate(contract: SurvivabilityContract) -> Result<(), String> {
        match contract {
            SurvivabilityContract::CriticalSensorQuorum => {
                let state = SubterraneanState::home();
                let mut second = SensorSourceObservation::from_state(SensorSourceId(1), 1, &state);
                second.valid[BATTERY_RATIO] = false;
                let frame = RedundantSensorFrame {
                    observations: vec![
                        SensorSourceObservation::from_state(SensorSourceId(0), 1, &state),
                        second,
                    ],
                };
                let (_, report) = SensorFusionSupervisor::default().fuse(&frame, &state);
                if !report.requires_fail_closed() {
                    return Err("critical battery channel lost quorum without fail-close".into());
                }
                Ok(())
            }
            SurvivabilityContract::RobustMedianFusion => {
                let fallback = SubterraneanState::home();
                let mut a = fallback.clone();
                let mut b = fallback.clone();
                let mut c = fallback.clone();
                a.channels[GAS_RISK] = 0.2;
                b.channels[GAS_RISK] = 0.21;
                c.channels[GAS_RISK] = 0.95;
                let frame = RedundantSensorFrame {
                    observations: vec![
                        SensorSourceObservation::from_state(SensorSourceId(0), 1, &a),
                        SensorSourceObservation::from_state(SensorSourceId(1), 1, &b),
                        SensorSourceObservation::from_state(SensorSourceId(2), 1, &c),
                    ],
                };
                let (fused, _) = SensorFusionSupervisor::default().fuse(&frame, &fallback);
                if (fused.channels[GAS_RISK] - 0.21).abs() > 1e-9 {
                    return Err("single gas outlier dominated fused observation".into());
                }
                Ok(())
            }
            SurvivabilityContract::PersistentActuatorIsolation => {
                let mut monitor = ActuatorIsolationSupervisor::new(ActuatorIsolationPolicy {
                    mismatch_penalty: 0.25,
                    isolation_threshold: 0.2,
                    mismatch_streak_limit: 4,
                    ..Default::default()
                });
                let state = SubterraneanState::home();
                let mut command = SubterraneanCommand::zero();
                command.set_left_track(1.0);
                for _ in 0..4 {
                    monitor.observe(&command, &state, &state);
                }
                let constrained = monitor.constrain(command);
                if !monitor.report().is_isolated(PhysicalActuator::LeftTrack)
                    || constrained.left_track() != 0.0
                    || constrained.right_track() != 0.0
                {
                    return Err("failed track retained authority or affected healthy axis".into());
                }
                Ok(())
            }
            SurvivabilityContract::ThermalPowerPrioritization => {
                let mut state = SubterraneanState::home();
                state.channels[CUTTER_TEMP_C] = 150.0;
                state.channels[BATTERY_RATIO] = 0.18;
                let assessment = FieldEnvelopeSupervisor::default().assess(
                    &state,
                    1.0,
                    MaintenanceAssessment::nominal(),
                );
                let mut command = SubterraneanCommand::zero();
                command.set_cutter_head(1.0);
                let command = assessment.constrain(command);
                if command.cutter_head() != 0.0 || command.thermal_pump() < 0.85 {
                    return Err("thermal protection did not prefer cooling over cutting".into());
                }
                Ok(())
            }
            SurvivabilityContract::GracefulReturnDisposition => {
                let mut actuators = crate::ActuatorIsolationReport::nominal();
                actuators.isolated[PhysicalActuator::LeftTrack.index()] = true;
                actuators.isolated_count = 1;
                let profile = CapabilityProfile::assess(
                    SensorFusionReport::nominal(),
                    actuators,
                    FieldEnvelopeAssessment::nominal(),
                    MaintenanceAssessment::nominal(),
                );
                if profile.disposition != CapabilityDisposition::ReturnOnly {
                    return Err("single mobility-axis loss did not degrade to return-only".into());
                }
                Ok(())
            }
            SurvivabilityContract::PartitionReconciliation => {
                let mut supervisor = PartitionRecoverySupervisor::new(PartitionRecoveryPolicy {
                    grace_steps: 0,
                    local_autonomy_steps: 1,
                    reconciliation_dwell_steps: 2,
                    ..Default::default()
                });
                let disconnected = PartitionObservation {
                    surface_reachable: false,
                    fresh_peers: 1,
                    battery_ratio: 0.8,
                    return_feasible: true,
                    local_map_revision: 7,
                    highest_peer_map_revision: 7,
                };
                supervisor.update(disconnected);
                let mut restored = disconnected;
                restored.surface_reachable = true;
                let first = supervisor.update(restored);
                if first.mode != PartitionRecoveryMode::Reconciling || first.motion_permitted {
                    return Err("restored link bypassed reconciliation hold".into());
                }
                let second = supervisor.update(restored);
                if second.mode != PartitionRecoveryMode::Connected {
                    return Err("bounded healthy dwell did not complete reconciliation".into());
                }
                Ok(())
            }
            SurvivabilityContract::CheckpointReplayContinuity => {
                let genesis = GenesisSeed::from_phrase("survivability checkpoint validation");
                let thought = ContinuousHV::random(HDC_DIMENSION, 7711);
                let state = SubterraneanState::home();
                let frame = Self::redundant_frame(&state, 9);
                let mut source = SubterraneanEmbodiment::new(&genesis);
                source.ingest_redundant_sensor_frame(frame.clone());
                source.step(&thought, 0.005, 0.9);
                let checkpoint = source.operational_checkpoint();
                let mut restored = SubterraneanEmbodiment::new(&genesis);
                restored
                    .load_operational_checkpoint(&checkpoint)
                    .map_err(|error| format!("checkpoint rejected: {error:?}"))?;
                restored.ingest_redundant_sensor_frame(frame);
                restored.step(&thought, 0.005, 0.9);
                if restored.sensor_fusion_report().replay_rejections == 0 {
                    return Err("checkpoint forgot sensor replay state".into());
                }
                Ok(())
            }
        }
    }

    pub fn run(&self) -> SurvivabilityValidationReport {
        let mut passed = Vec::new();
        let mut failures = Vec::new();
        for contract in SurvivabilityContract::ALL {
            match Self::evaluate(contract) {
                Ok(()) => passed.push(contract),
                Err(detail) => failures.push(SurvivabilityGateFailure { contract, detail }),
            }
        }
        SurvivabilityValidationReport { passed, failures }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_survivability_acceptance_contracts_pass() {
        let report = SurvivabilityValidator.run();
        assert!(report.is_success(), "{:#?}", report.failures);
        assert!(report.to_pretty_json().is_ok());
    }
}
