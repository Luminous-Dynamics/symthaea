// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Combined learned-controller, mission, operator and recovery-state checkpoint.

use crate::actuator_isolation::ActuatorIsolationSupervisor;
use crate::controller::{CheckpointError, ControllerCheckpoint};
use crate::degraded_operations::DegradedOperationsSupervisor;
use crate::field_envelope::FieldEnvelopeSupervisor;
use crate::mission_executive::{MissionCheckpointError, MissionExecutiveCheckpoint};
use crate::operator_authority::OperatorAuthority;
use crate::partition_recovery::PartitionRecoverySupervisor;
use crate::sensor_redundancy::SensorFusionSupervisor;
use crate::temporal_assurance::TemporalAssuranceSupervisor;
use crate::update_control::UpdateManager;
use serde::{Deserialize, Serialize};

pub const OPERATIONAL_CHECKPOINT_SCHEMA_VERSION: u32 = 3;
pub const MIN_SUPPORTED_OPERATIONAL_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubterraneanOperationalCheckpoint {
    pub schema_version: u32,
    pub controller: ControllerCheckpoint,
    pub mission: MissionExecutiveCheckpoint,
    #[serde(default)]
    pub operator_authority: OperatorAuthority,
    #[serde(default)]
    pub degraded_supervisor: DegradedOperationsSupervisor,
    #[serde(default)]
    pub update_manager: Option<UpdateManager>,
    #[serde(default)]
    pub sensor_fusion: SensorFusionSupervisor,
    #[serde(default)]
    pub actuator_isolation: ActuatorIsolationSupervisor,
    #[serde(default)]
    pub field_envelope: FieldEnvelopeSupervisor,
    #[serde(default)]
    pub partition_recovery: PartitionRecoverySupervisor,
    #[serde(default)]
    pub temporal: TemporalAssuranceSupervisor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationalCheckpointError {
    UnsupportedSchema { found: u32, expected: u32 },
    Controller(CheckpointError),
    Mission(MissionCheckpointError),
    InvalidOperatorState,
    InvalidDegradedState,
    InvalidUpdateState,
    InvalidSensorFusionState,
    InvalidActuatorIsolationState,
    InvalidFieldEnvelopeState,
    InvalidPartitionRecoveryState,
    InvalidTemporalState,
}

impl From<CheckpointError> for OperationalCheckpointError {
    fn from(value: CheckpointError) -> Self {
        Self::Controller(value)
    }
}

impl From<MissionCheckpointError> for OperationalCheckpointError {
    fn from(value: MissionCheckpointError) -> Self {
        Self::Mission(value)
    }
}
