// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure validation for portable operational checkpoint source state.
//!
//! Restore preparation must be able to reject malformed checkpoint data before
//! mutating a live embodiment or creating affine execution authority. These
//! validators deliberately inspect portable source state only; they do not
//! perform restore, reconciliation, requalification, or authority widening.

use super::{
    MIN_SUPPORTED_OPERATIONAL_CHECKPOINT_SCHEMA_VERSION, OPERATIONAL_CHECKPOINT_SCHEMA_VERSION,
    OperationalCheckpointError, SubterraneanOperationalCheckpoint,
};
use crate::mission_executive::{
    MISSION_CHECKPOINT_SCHEMA_VERSION, MissionCheckpointError, MissionExecutiveCheckpoint,
};

impl MissionExecutiveCheckpoint {
    /// Validate portable mission-executive state without mutating a live
    /// `MissionExecutive`.
    pub fn validate(&self) -> Result<(), MissionCheckpointError> {
        if self.schema_version != MISSION_CHECKPOINT_SCHEMA_VERSION {
            return Err(MissionCheckpointError::UnsupportedSchema {
                found: self.schema_version,
                expected: MISSION_CHECKPOINT_SCHEMA_VERSION,
            });
        }
        self.graph
            .validate()
            .map_err(MissionCheckpointError::InvalidGraph)?;
        self.scheduler
            .validate()
            .map_err(MissionCheckpointError::InvalidScheduler)?;
        if !self.logistics.validate() {
            return Err(MissionCheckpointError::InvalidLogistics);
        }
        if !self.maintenance.validate() {
            return Err(MissionCheckpointError::InvalidMaintenance);
        }
        if self.graph.node(self.current_node).is_none() {
            return Err(MissionCheckpointError::InvalidCurrentNode);
        }
        if self.graph.node(self.surface_node).is_none() {
            return Err(MissionCheckpointError::InvalidSurfaceNode);
        }
        self.route_policy
            .validate()
            .map_err(|_| MissionCheckpointError::InvalidRoutePolicy)?;
        Ok(())
    }
}

impl SubterraneanOperationalCheckpoint {
    /// Validate the complete portable checkpoint source without touching a live
    /// embodiment.
    ///
    /// Passing this function means only that the source is structurally valid.
    /// It does **not** mean the checkpoint is admissible for live restore. The
    /// RA restore transaction must still compare authority/evidence semantics,
    /// bind the exact source, recheck the live generation fence, execute every
    /// canonical action, and reconcile before activation.
    pub fn validate_source(&self) -> Result<(), OperationalCheckpointError> {
        if self.schema_version < MIN_SUPPORTED_OPERATIONAL_CHECKPOINT_SCHEMA_VERSION
            || self.schema_version > OPERATIONAL_CHECKPOINT_SCHEMA_VERSION
        {
            return Err(OperationalCheckpointError::UnsupportedSchema {
                found: self.schema_version,
                expected: OPERATIONAL_CHECKPOINT_SCHEMA_VERSION,
            });
        }
        self.controller.validate()?;
        self.mission.validate()?;
        if !self.operator_authority.validate() {
            return Err(OperationalCheckpointError::InvalidOperatorState);
        }
        if !self.degraded_supervisor.validate() {
            return Err(OperationalCheckpointError::InvalidDegradedState);
        }
        if self
            .update_manager
            .as_ref()
            .is_some_and(|manager| !manager.validate())
        {
            return Err(OperationalCheckpointError::InvalidUpdateState);
        }
        if !self.sensor_fusion.validate() {
            return Err(OperationalCheckpointError::InvalidSensorFusionState);
        }
        if !self.actuator_isolation.validate() {
            return Err(OperationalCheckpointError::InvalidActuatorIsolationState);
        }
        if !self.field_envelope.validate() {
            return Err(OperationalCheckpointError::InvalidFieldEnvelopeState);
        }
        if !self.partition_recovery.validate() {
            return Err(OperationalCheckpointError::InvalidPartitionRecoveryState);
        }
        if !self.temporal.validate() {
            return Err(OperationalCheckpointError::InvalidTemporalState);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment::SubterraneanEmbodiment;
    use crate::operator_authority::OperatorConstraint;
    use crate::operator_protocol::{
        AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
    };
    use symthaea_core::genesis::GenesisSeed;

    fn checkpoint() -> SubterraneanOperationalCheckpoint {
        SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("pure-checkpoint-validation"))
            .operational_checkpoint()
    }

    #[test]
    fn nominal_operational_checkpoint_passes_pure_source_validation() {
        assert_eq!(checkpoint().validate_source(), Ok(()));
    }

    #[test]
    fn malformed_controller_is_rejected_without_live_owner() {
        let mut value = checkpoint();
        value.controller.hdc_dimension = value.controller.hdc_dimension.saturating_add(1);
        assert!(matches!(
            value.validate_source(),
            Err(OperationalCheckpointError::Controller(_))
        ));
    }

    #[test]
    fn malformed_mission_is_rejected_without_live_owner() {
        let mut value = checkpoint();
        value.mission.schema_version = value.mission.schema_version.saturating_add(1);
        assert!(matches!(
            value.validate_source(),
            Err(OperationalCheckpointError::Mission(_))
        ));
    }

    #[test]
    fn structurally_valid_authority_is_not_mistaken_for_restore_admission() {
        let mut value = checkpoint();
        let decision = value
            .operator_authority
            .ingest(
                OperatorCommandEnvelope {
                    operator: OperatorId(7),
                    role: OperatorRole::SafetyOfficer,
                    authentication: AuthenticationLevel::HardwareBacked,
                    epoch: 1,
                    sequence: 1,
                    proposal_id: 11,
                    issued_step: 0,
                    expires_step: 100,
                    command: OperatorCommand::HoldPosition,
                },
                0,
                true,
            )
            .expect("structurally valid restrictive operator state");
        assert!(matches!(
            decision,
            crate::operator_authority::OperatorDecision::Applied(OperatorConstraint::HoldPosition)
        ));
        assert_eq!(
            value.operator_authority.constraint(),
            OperatorConstraint::HoldPosition
        );
        // Structural validity says only that the source is well-formed. Whether
        // this restriction must be joined/preserved against live authority is a
        // separate restore-admission/execution question.
        assert_eq!(value.validate_source(), Ok(()));
    }
}
