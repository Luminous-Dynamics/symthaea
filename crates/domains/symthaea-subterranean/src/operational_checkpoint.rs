// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Combined learned-controller, mission, operator and recovery-state checkpoint.

#[path = "restore_actions.rs"]
pub(crate) mod restore_actions;
#[path = "restore_admission.rs"]
pub(crate) mod restore_admission;
#[path = "restore_execution.rs"]
pub(crate) mod restore_execution;
#[path = "restore_merge.rs"]
pub(crate) mod restore_merge;
#[path = "restore_operator.rs"]
pub(crate) mod restore_operator;
#[path = "restore_semantics.rs"]
pub mod restore_semantics;
#[path = "operational_checkpoint_validation.rs"]
mod validation;
#[cfg(test)]
#[path = "restore_source_adversarial.rs"]
mod restore_source_adversarial;

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
    /// Operator authority is security-critical restore state and must be present
    /// explicitly. Missing historical authority is unknown, not nominal, so it
    /// must not deserialize through `OperatorAuthority::default()` into a wider
    /// `None` constraint.
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
