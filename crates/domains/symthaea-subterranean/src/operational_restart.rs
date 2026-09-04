// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit separation between simulation episode recreation and operational restart.
//!
//! `EmbodimentBridge::reset()` is a fleet-wide simulation/episode contract: it
//! recreates default body state and intentionally clears episode-local moral state.
//! That behavior is useful for simulation, but it is not a production recovery
//! primitive. A real asset continues to have physical history, latched faults,
//! authority restrictions and replay evidence across a process/runtime restart.
//!
//! This module therefore provides a separate conservative restart path. It never
//! resets the physical simulator. Instead it portable-normalizes the current
//! operational checkpoint (discarding host-local positive recovery progress while
//! retaining serialized adverse/replay evidence), forces the restart-sensitive
//! authority domains into requalification/reconciliation, then reloads the result
//! through the existing validated crate-internal checkpoint hydration boundary.

use crate::degraded_operations::{DegradedMode, DegradedObservation};
use crate::embodiment::{MotorSafetyLevel, SubterraneanEmbodiment};
use crate::field_envelope::FieldEnvelopeMode;
use crate::operational_checkpoint::{OperationalCheckpointError, SubterraneanOperationalCheckpoint};
use crate::operator_authority::OperatorConstraint;
use crate::partition_recovery::PartitionRecoveryMode;
use crate::plan_freshness::RuntimeRevisions;
use crate::sensor_redundancy::RedundantSensorFrame;
use crate::temporal_assurance::{
    TEMPORAL_RUNTIME_FRAME_SCHEMA_VERSION, TemporalAuthority, TemporalRuntimeFrame,
};
use crate::update_control::UpdateState;

/// Observable postcondition of an operational restart boundary transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationalRestartReport {
    pub operator_constraint: OperatorConstraint,
    pub degraded_mode: DegradedMode,
    pub partition_mode: PartitionRecoveryMode,
    pub partition_motion_permitted: bool,
    pub team_state_authoritative: bool,
    pub temporal_authority: TemporalAuthority,
    pub temporal_hold_latched: bool,
    pub isolated_actuators: usize,
    pub field_envelope_mode: FieldEnvelopeMode,
    pub update_state: Option<UpdateState>,
}

#[derive(Debug)]
pub enum OperationalRestartError {
    Encoding(serde_json::Error),
    Checkpoint(OperationalCheckpointError),
}

impl std::fmt::Display for OperationalRestartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encoding(error) => write!(f, "operational restart checkpoint encoding failed: {error}"),
            Self::Checkpoint(error) => write!(f, "operational restart checkpoint rejected: {error:?}"),
        }
    }
}

impl std::error::Error for OperationalRestartError {}

impl From<OperationalCheckpointError> for OperationalRestartError {
    fn from(value: OperationalCheckpointError) -> Self {
        Self::Checkpoint(value)
    }
}

fn portable_normalize(
    checkpoint: &SubterraneanOperationalCheckpoint,
) -> Result<SubterraneanOperationalCheckpoint, OperationalRestartError> {
    let encoded = serde_json::to_vec(checkpoint).map_err(OperationalRestartError::Encoding)?;
    let normalized =
        serde_json::from_slice(&encoded).map_err(OperationalRestartError::Encoding)?;
    Ok(normalized)
}

/// Explicit name for the existing synthetic episode reset semantics.
///
/// This intentionally delegates to `SubterraneanEmbodiment::reset()`: it may
/// recreate nominal/default world state and erase episode-local authority history.
/// Do not use this API for process restart, watchdog recovery, reconnection, or a
/// continuing physical asset.
pub trait SimulationEpisodeReset {
    fn reset_simulation_episode(&mut self);
}

impl SimulationEpisodeReset for SubterraneanEmbodiment {
    fn reset_simulation_episode(&mut self) {
        SubterraneanEmbodiment::reset(self);
    }
}

/// Conservative operational restart for a continuing physical asset.
///
/// The transition is intentionally one-way restrictive:
///
/// - physical simulator state is not reset;
/// - serialized operator restriction and replay barriers are preserved;
/// - host-local recovery issuance/partial quorum progress is dropped by the
///   portable checkpoint round-trip;
/// - degraded authority is forced to `RecoveryRequired`;
/// - partition/team authority is forced to `Reconciling` and non-authoritative;
/// - temporal authority is latched at `HoldForReview` and positive clean-dwell
///   recovery credit is discarded;
/// - any queued pre-restart redundant-sensor input is invalidated by replacing it
///   with an empty frame, which the next control cycle treats as fail-closed
///   critical-channel no-quorum without advancing source replay state;
/// - actuator isolation, field envelope, maintenance/mission state and update
///   lifecycle state are preserved by the checkpoint;
/// - a Red safety override is installed as an additional runtime floor.
///
/// This first tranche does not yet own the RA-34 live owner/boot generation
/// fence. Restart-generation rotation therefore remains an explicit follow-up;
/// the runtime domains here remain restrictive until requalification.
pub trait OperationalRestart {
    fn restart_operational_runtime(
        &mut self,
    ) -> Result<OperationalRestartReport, OperationalRestartError>;
}

impl OperationalRestart for SubterraneanEmbodiment {
    fn restart_operational_runtime(
        &mut self,
    ) -> Result<OperationalRestartReport, OperationalRestartError> {
        // Portable normalization is security-relevant. `OperatorAuthority` marks
        // issued recovery proposals and partial quorum progress `serde(skip)`, so
        // a restart cannot carry positive widening progress across the boundary.
        // Its replay barriers and active restriction are serialized and survive.
        let mut checkpoint = portable_normalize(&self.operational_checkpoint())?;
        checkpoint.validate_source()?;

        // Enter RecoveryRequired without fabricating watchdog/link failures. The
        // invalid checkpoint flag represents the continuity break itself.
        checkpoint.degraded_supervisor.update(DegradedObservation {
            operator_link_fresh: true,
            control_loop_healthy: true,
            checkpoint_valid: false,
            reboot_count_in_window: 0,
            battery_ratio: 1.0,
            return_feasible: false,
            at_surface_or_service_bay: false,
        });

        checkpoint
            .partition_recovery
            .enter_operational_restart_reconciliation();

        // A runtime restart breaks temporal continuity. Feed a deliberately
        // invalid control interval with an otherwise current-schema empty frame;
        // this preserves existing ledgers while latching HoldForReview and
        // discarding positive clean-dwell recovery credit.
        let mut restart_frame = TemporalRuntimeFrame::default();
        restart_frame.schema_version = TEMPORAL_RUNTIME_FRAME_SCHEMA_VERSION;
        checkpoint.temporal.assess(
            0.0,
            0,
            RuntimeRevisions::default(),
            &restart_frame,
            false,
            false,
        );

        checkpoint.validate_source()?;
        self.load_operational_checkpoint(&checkpoint)?;

        // Do not let a pre-restart queued sensor frame cross the continuity
        // boundary. An empty frame is intentionally fail-closed on the next
        // cycle: zero accepted sources and critical-channel no-quorum. Because
        // it declares no source, it does not mutate source replay sequences or
        // reliability history.
        self.ingest_redundant_sensor_frame(RedundantSensorFrame::default());

        // This is an additional floor, not the primary authority mechanism.
        // Degraded/partition/temporal supervisors remain restrictive even if a
        // caller later clears the generic safety override.
        self.set_safety_override(MotorSafetyLevel::Red);

        let current = self.operational_checkpoint();
        let partition = current.partition_recovery.assessment();
        Ok(OperationalRestartReport {
            operator_constraint: current.operator_authority.constraint(),
            degraded_mode: current.degraded_supervisor.mode(),
            partition_mode: partition.mode,
            partition_motion_permitted: partition.motion_permitted,
            team_state_authoritative: partition.team_state_authoritative,
            temporal_authority: current.temporal.last().authority,
            temporal_hold_latched: current.temporal.hold_latched(),
            isolated_actuators: current.actuator_isolation.report().isolated_count,
            field_envelope_mode: current.field_envelope.last_assessment().mode,
            update_state: current.update_manager.as_ref().map(|manager| manager.state()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actuator_isolation::{
        ActuatorIsolationPolicy, ActuatorIsolationSupervisor, PhysicalActuator,
    };
    use crate::field_envelope::FieldEnvelopeMode;
    use crate::maintenance::MaintenanceAssessment;
    use crate::operator_authority::recovery_authority::{
        RecoveryApprovalEnvelopeV1, RecoveryDigest, RecoveryProposalV1,
    };
    use crate::operator_authority::{OperatorAuthorityRejection, OperatorDecision};
    use crate::operator_protocol::{
        AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
    };
    use crate::sensor_redundancy::{
        RedundantSensorFrame, SensorSourceId, SensorSourceObservation,
    };
    use crate::types::{BATTERY_RATIO, SubterraneanCommand, SubterraneanState};
    use crate::update_control::{ArtifactDigest, UpdateManager};
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::ContinuousHV;

    fn command(operator: u64, sequence: u64, proposal_id: u64, command: OperatorCommand) -> OperatorCommandEnvelope {
        OperatorCommandEnvelope {
            operator: OperatorId(operator),
            role: OperatorRole::SafetyOfficer,
            authentication: AuthenticationLevel::HardwareBacked,
            epoch: 1,
            sequence,
            proposal_id,
            issued_step: 10,
            expires_step: 100,
            command,
        }
    }

    fn recovery_proposal(id: u64, active: OperatorConstraint) -> RecoveryProposalV1 {
        RecoveryProposalV1::new(
            id,
            active,
            RecoveryDigest([1; 32]),
            RecoveryDigest([2; 32]),
            RecoveryDigest([3; 32]),
            1,
            1,
            10,
            100,
        )
    }

    fn recovery_approval(
        operator: u64,
        sequence: u64,
        proposal: RecoveryProposalV1,
    ) -> RecoveryApprovalEnvelopeV1 {
        RecoveryApprovalEnvelopeV1 {
            operator: OperatorId(operator),
            role: OperatorRole::SafetyOfficer,
            authentication: AuthenticationLevel::HardwareBacked,
            epoch: 1,
            sequence,
            approval_issued_step: 20,
            proposal,
        }
    }

    #[test]
    fn simulation_episode_reset_is_explicitly_a_new_synthetic_world() {
        let mut live = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("ra35-simulation-reset"));
        live.ingest_operator_command(command(1, 1, 1, OperatorCommand::EmergencyStop))
            .expect("restrictive command");
        assert_eq!(
            live.operational_checkpoint().operator_authority.constraint(),
            OperatorConstraint::EmergencyStop
        );

        live.reset_simulation_episode();
        assert_eq!(
            live.operational_checkpoint().operator_authority.constraint(),
            OperatorConstraint::None
        );
    }

    #[test]
    fn operational_restart_preserves_each_operator_restriction() {
        let cases = [
            (OperatorCommand::EmergencyStop, OperatorConstraint::EmergencyStop),
            (OperatorCommand::EnterMaintenance, OperatorConstraint::MaintenanceLock),
            (OperatorCommand::HoldPosition, OperatorConstraint::HoldPosition),
            (OperatorCommand::ReturnHome, OperatorConstraint::ReturnHome),
        ];

        for (index, (requested, expected)) in cases.into_iter().enumerate() {
            let mut live = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase(&format!(
                "ra35-operator-restriction-{index}"
            )));
            live.ingest_operator_command(command(1, 1, index as u64 + 1, requested))
                .expect("restrictive command");
            let report = live
                .restart_operational_runtime()
                .expect("conservative restart");
            assert_eq!(report.operator_constraint, expected);
            assert_eq!(
                live.operational_checkpoint().operator_authority.constraint(),
                expected
            );
        }
    }

    #[test]
    fn operational_restart_preserves_restriction_and_replay_but_drops_recovery_progress() {
        let mut live = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("ra35-operator"));
        live.ingest_operator_command(command(1, 1, 1, OperatorCommand::EmergencyStop))
            .expect("emergency stop");

        let mut checkpoint = live.operational_checkpoint();
        let proposal = recovery_proposal(77, OperatorConstraint::EmergencyStop);
        checkpoint
            .operator_authority
            .issue_recovery_proposal(proposal, 20)
            .expect("issue recovery");
        assert!(matches!(
            checkpoint
                .operator_authority
                .approve_recovery(recovery_approval(2, 1, proposal), 20)
                .expect("first recovery approval"),
            OperatorDecision::PendingQuorum {
                approvals: 1,
                required: 2
            }
        ));
        live.load_operational_checkpoint(&checkpoint)
            .expect("install pre-restart authority state");
        assert_eq!(
            live.operational_checkpoint()
                .operator_authority
                .pending_approvals(77),
            1
        );

        let report = live
            .restart_operational_runtime()
            .expect("conservative restart");
        assert_eq!(report.operator_constraint, OperatorConstraint::EmergencyStop);

        let after = live.operational_checkpoint();
        assert_eq!(
            after.operator_authority.constraint(),
            OperatorConstraint::EmergencyStop
        );
        assert_eq!(after.operator_authority.pending_approvals(77), 0);
        assert_eq!(after.operator_authority.issued_recovery_proposal(77), None);

        // The recovery approval's consumed sequence is durable adverse/replay
        // evidence even though its positive quorum progress is gone.
        assert_eq!(
            live.ingest_operator_command(command(2, 1, 999, OperatorCommand::EmergencyStop)),
            Err(OperatorAuthorityRejection::Replay)
        );
    }

    #[test]
    fn operational_restart_invalidates_queued_pre_restart_sensor_frame() {
        let mut live = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("ra35-sensor-cache"));
        let state = SubterraneanState::home();
        live.ingest_redundant_sensor_frame(RedundantSensorFrame {
            observations: vec![SensorSourceObservation::from_state(
                SensorSourceId(0),
                1,
                &state,
            )],
        });

        live.restart_operational_runtime()
            .expect("conservative restart");
        let thought = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 3501);
        live.step(&thought, 0.005, 0.9);
        let report = live.sensor_fusion_report();

        // The queued valid pre-restart frame must have been overwritten by the
        // empty restart sentinel rather than consumed after the continuity break.
        assert_eq!(report.accepted_sources, 0);
        assert!(report.requires_fail_closed());
        assert!(report.critical_channels_without_quorum > 0);
    }

    #[test]
    fn operational_restart_forces_requalification_and_preserves_fault_restrictions() {
        let mut live = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("ra35-faults"));
        let mut checkpoint = live.operational_checkpoint();

        checkpoint.actuator_isolation = ActuatorIsolationSupervisor::new(ActuatorIsolationPolicy {
            energized_threshold: 0.1,
            mismatch_penalty: 1.0,
            recovery_rate: 0.0,
            isolation_threshold: 1.0,
            mismatch_streak_limit: 1,
        });
        let mut cutter_command = SubterraneanCommand::zero();
        cutter_command.set_cutter_head(1.0);
        let state = SubterraneanState::home();
        checkpoint
            .actuator_isolation
            .observe(&cutter_command, &state, &state);
        assert!(
            checkpoint
                .actuator_isolation
                .report()
                .is_isolated(PhysicalActuator::Cutter)
        );

        let mut low_battery = SubterraneanState::home();
        low_battery.channels[BATTERY_RATIO] = 0.15;
        let field_before = checkpoint.field_envelope.assess(
            &low_battery,
            1.0,
            MaintenanceAssessment::nominal(),
        );
        assert_eq!(field_before.mode, FieldEnvelopeMode::CriticalPower);

        checkpoint.update_manager = Some(
            UpdateManager::new(ArtifactDigest([9; 32]), 1).expect("valid update manager"),
        );
        live.load_operational_checkpoint(&checkpoint)
            .expect("install fault state");

        let report = live
            .restart_operational_runtime()
            .expect("conservative restart");
        assert_eq!(report.degraded_mode, DegradedMode::RecoveryRequired);
        assert_eq!(report.partition_mode, PartitionRecoveryMode::Reconciling);
        assert!(!report.partition_motion_permitted);
        assert!(!report.team_state_authoritative);
        assert_eq!(report.temporal_authority, TemporalAuthority::HoldForReview);
        assert!(report.temporal_hold_latched);
        assert_eq!(report.isolated_actuators, 1);
        assert_eq!(report.field_envelope_mode, FieldEnvelopeMode::CriticalPower);
        assert_eq!(report.update_state, Some(UpdateState::Idle));

        let after = live.operational_checkpoint();
        assert!(
            after
                .actuator_isolation
                .report()
                .is_isolated(PhysicalActuator::Cutter)
        );
        assert_eq!(
            after.field_envelope.last_assessment().mode,
            FieldEnvelopeMode::CriticalPower
        );
        assert_eq!(
            after.update_manager.as_ref().map(|m| m.current_digest()),
            Some(ArtifactDigest([9; 32]))
        );
    }
}
