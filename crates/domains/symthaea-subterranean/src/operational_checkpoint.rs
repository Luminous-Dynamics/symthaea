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
#[path = "operational_checkpoint_wire.rs"]
mod wire;
#[cfg(test)]
#[path = "restore_source_adversarial.rs"]
mod restore_source_adversarial;

pub use wire::OperationalCheckpointWireError;

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
use serde::{Deserialize, Deserializer, Serialize};

/// Schema v4 is the first operational checkpoint schema whose version denotes a
/// complete authority-bearing field contract rather than only a parseable shape.
///
/// Older schema numbers are intentionally not accepted directly. Legacy data
/// requires an explicit conservative migration before it can become current
/// restore source state.
pub const OPERATIONAL_CHECKPOINT_SCHEMA_VERSION: u32 = 4;
pub const MIN_SUPPORTED_OPERATIONAL_CHECKPOINT_SCHEMA_VERSION: u32 = 4;

/// Deserialize an explicitly present optional field.
///
/// `Option<T>` normally permits a missing map key to deserialize as `None`.
/// That is unsafe for authority-bearing checkpoint state because an absent field
/// means the source never recorded whether the state existed. Applying a custom
/// deserializer removes that special missing-field behavior: explicit JSON
/// `null` is still `None`, but an absent `update_manager` key is an error.
fn deserialize_required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

/// Portable multi-domain checkpoint data.
///
/// This type is intentionally public because checkpoint data must be exportable,
/// inspectable and transportable. Possessing or deserializing it does **not**
/// grant authority to replace a live embodiment's operational state.
///
/// In particular, downstream crates cannot invoke the crate-internal legacy
/// whole-checkpoint loader. Production restore must eventually enter through the
/// affine admission/execution/activation boundary instead.
///
/// Every authority-bearing v4 field is required to be present. Missing state is
/// unknown, not nominal. For `update_manager`, explicit `null` is the only
/// representation of a known absence; an omitted key is rejected. Unknown
/// top-level fields are also rejected so one schema number cannot silently
/// describe multiple authority contracts.
///
/// For externally supplied JSON intended for restore, use
/// [`SubterraneanOperationalCheckpoint::from_strict_v4_json`]. The ordinary
/// Serde representation remains useful for inspection/transport, but strict
/// ingress additionally proves that no nested wire state was silently ignored
/// or default-synthesized before source commitment.
///
/// ```compile_fail,E0624
/// use symthaea_subterranean::{
///     SubterraneanOperationalCheckpoint,
///     embodiment::SubterraneanEmbodiment,
/// };
///
/// fn bypass_restore(
///     live: &mut SubterraneanEmbodiment,
///     checkpoint: &SubterraneanOperationalCheckpoint,
/// ) {
///     // This must remain inaccessible to downstream crates. If this snippet
///     // ever compiles, the RA-33 whole-checkpoint restore bypass has reopened.
///     live.load_operational_checkpoint(checkpoint).unwrap();
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubterraneanOperationalCheckpoint {
    pub schema_version: u32,
    pub controller: ControllerCheckpoint,
    pub mission: MissionExecutiveCheckpoint,
    /// Operator authority is security-critical restore state and must be present
    /// explicitly. Missing historical authority is unknown, not nominal, so it
    /// must not deserialize through `OperatorAuthority::default()` into a wider
    /// `None` constraint.
    pub operator_authority: OperatorAuthority,
    /// Degraded authority state is required. Missing historical degraded state
    /// must not synthesize `Normal` authority.
    pub degraded_supervisor: DegradedOperationsSupervisor,
    /// Update lifecycle state is required even though the value itself is
    /// optional. `null` means explicitly no manager; an absent key is unknown.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub update_manager: Option<UpdateManager>,
    /// Sensor reliability and replay barriers are durable restore evidence.
    pub sensor_fusion: SensorFusionSupervisor,
    /// Isolation latches remove actuator authority and therefore cannot default
    /// to an unisolated supervisor when historical state is missing.
    pub actuator_isolation: ActuatorIsolationSupervisor,
    /// Field-envelope restrictions cannot default to nominal on missing history.
    pub field_envelope: FieldEnvelopeSupervisor,
    /// Partition/reconciliation state cannot default to Connected/authoritative.
    pub partition_recovery: PartitionRecoverySupervisor,
    /// Temporal holds and causal history cannot default to nominal authority.
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
