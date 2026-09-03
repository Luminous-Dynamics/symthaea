// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public-surface characterization for authority-monotone operational restore.
//!
//! A structurally valid historical checkpoint is evidence about past state. It
//! must not replace newer, more restrictive live authority without a separately
//! verified recovery transition.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};
use symthaea_subterranean::actuator_isolation::{
    ActuatorIsolationPolicy, ActuatorIsolationSupervisor, PhysicalActuator,
};
use symthaea_subterranean::degraded_operations::DegradedMode;
use symthaea_subterranean::embodiment::SubterraneanEmbodiment;
use symthaea_subterranean::field_envelope::{
    FieldEnvelopeMode, FieldEnvelopeSupervisor,
};
use symthaea_subterranean::maintenance::MaintenanceAssessment;
use symthaea_subterranean::operator_authority::OperatorConstraint;
use symthaea_subterranean::operator_protocol::{
    AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
};
use symthaea_subterranean::partition_recovery::{
    PartitionObservation, PartitionRecoveryMode, PartitionRecoveryPolicy,
    PartitionRecoverySupervisor,
};
use symthaea_subterranean::sensor_redundancy::{
    RedundantSensorFrame, SensorSourceId, SensorSourceObservation,
};
use symthaea_subterranean::temporal_assurance::{TemporalAuthority, TemporalRuntimeFrame};
use symthaea_subterranean::types::{
    SubterraneanCommand, SubterraneanState, BATTERY_RATIO,
};
use symthaea_subterranean::update_control::{
    ArtifactDigest, UpdateManifest, UpdateState, UPDATE_MANIFEST_SCHEMA_VERSION,
};

fn thought(seed: u64) -> ContinuousHV {
    ContinuousHV::random(HDC_DIMENSION, seed)
}

fn operator_command(
    sequence: u64,
    proposal_id: u64,
    command: OperatorCommand,
) -> OperatorCommandEnvelope {
    OperatorCommandEnvelope {
        operator: OperatorId(77),
        role: OperatorRole::SafetyOfficer,
        authentication: AuthenticationLevel::HardwareBacked,
        epoch: 1,
        sequence,
        proposal_id,
        issued_step: 0,
        expires_step: 10_000,
        command,
    }
}

#[test]
fn stale_checkpoint_must_not_clear_degraded_recovery_required() {
    let genesis = GenesisSeed::from_phrase("restore monotonic degraded");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);
    let older = embodiment.operational_checkpoint();

    embodiment.set_runtime_health(true, true, false, 0);
    embodiment.step(&thought(91_001), 0.005, 0.9);
    assert_eq!(embodiment.degraded_mode(), DegradedMode::RecoveryRequired);

    let restore = embodiment.load_operational_checkpoint(&older);
    assert!(
        restore.is_err() || embodiment.degraded_mode() == DegradedMode::RecoveryRequired,
        "stale checkpoint restored degraded authority from RecoveryRequired to a wider mode"
    );
    assert_eq!(embodiment.degraded_mode(), DegradedMode::RecoveryRequired);
}

#[test]
fn stale_checkpoint_must_not_clear_temporal_hold_for_review() {
    let genesis = GenesisSeed::from_phrase("restore monotonic temporal");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);
    let older = embodiment.operational_checkpoint();

    // The default frame has an invalid schema and no clock/observation basis,
    // which must produce a latched HoldForReview.
    embodiment.ingest_temporal_frame(TemporalRuntimeFrame::default());
    embodiment.step(&thought(91_002), 0.005, 0.9);
    assert_eq!(
        embodiment.temporal_assessment().authority,
        TemporalAuthority::HoldForReview
    );

    let restore = embodiment.load_operational_checkpoint(&older);
    assert!(
        restore.is_err()
            || embodiment.temporal_assessment().authority == TemporalAuthority::HoldForReview,
        "stale checkpoint erased a latched temporal review hold"
    );
    assert_eq!(
        embodiment.temporal_assessment().authority,
        TemporalAuthority::HoldForReview
    );
}

#[test]
fn stale_checkpoint_must_not_erase_sensor_fail_closed_state() {
    let genesis = GenesisSeed::from_phrase("restore monotonic sensor");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);
    let older = embodiment.operational_checkpoint();

    let state = SubterraneanState::home();
    let mut invalid = SensorSourceObservation::from_state(SensorSourceId(1), 1, &state);
    invalid.valid[BATTERY_RATIO] = false;
    embodiment.ingest_redundant_sensor_frame(RedundantSensorFrame {
        observations: vec![
            SensorSourceObservation::from_state(SensorSourceId(0), 1, &state),
            invalid,
        ],
    });
    embodiment.step(&thought(91_003), 0.005, 0.9);
    assert!(embodiment.sensor_fusion_report().requires_fail_closed());

    let restore = embodiment.load_operational_checkpoint(&older);
    assert!(
        restore.is_err() || embodiment.sensor_fusion_report().requires_fail_closed(),
        "stale checkpoint replaced fail-closed sensor evidence with an older nominal report"
    );
    assert!(embodiment.sensor_fusion_report().requires_fail_closed());
}

#[test]
fn stale_checkpoint_must_not_erase_update_rollback_required() {
    let genesis = GenesisSeed::from_phrase("restore monotonic update");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);
    embodiment
        .ingest_operator_command(operator_command(
            1,
            1,
            OperatorCommand::EnterMaintenance,
        ))
        .expect("maintenance lock should be accepted");
    assert_eq!(
        embodiment.operator_constraint(),
        OperatorConstraint::MaintenanceLock
    );

    // Capture a checkpoint with the same operator authority but before an
    // update transition exists, isolating update-state restore semantics.
    let older = embodiment.operational_checkpoint();

    embodiment
        .initialize_update_control(ArtifactDigest([1; 32]), 1)
        .expect("update manager should initialize");
    embodiment
        .stage_update(UpdateManifest {
            schema_version: UPDATE_MANIFEST_SCHEMA_VERSION,
            release_id: 2,
            artifact_digest: ArtifactDigest([2; 32]),
            configuration_digest: ArtifactDigest([3; 32]),
            rollback_digest: ArtifactDigest([1; 32]),
            minimum_checkpoint_schema: 1,
            issued_epoch: 2,
            expires_step: 10_000,
        })
        .expect("update should stage at home under maintenance lock");
    embodiment
        .activate_staged_update(50)
        .expect("staged update should activate into health probation");
    embodiment
        .observe_update_health(false)
        .expect("failed health observation should require rollback");
    assert_eq!(embodiment.update_state(), Some(UpdateState::RollbackRequired));

    let restore = embodiment.load_operational_checkpoint(&older);
    assert!(
        restore.is_err() || embodiment.update_state() == Some(UpdateState::RollbackRequired),
        "stale checkpoint erased a live rollback obligation"
    );
    assert_eq!(embodiment.update_state(), Some(UpdateState::RollbackRequired));
}

#[test]
fn stale_checkpoint_must_not_clear_partition_reconciliation_authority() {
    let genesis = GenesisSeed::from_phrase("restore monotonic partition");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);
    let older = embodiment.operational_checkpoint();
    let mut restrictive = older.clone();

    let mut partition = PartitionRecoverySupervisor::new(PartitionRecoveryPolicy {
        grace_steps: 0,
        local_autonomy_steps: 0,
        reconciliation_dwell_steps: 20,
        minimum_battery_for_local_autonomy: 0.35,
    });
    let assessment = partition.update(PartitionObservation {
        surface_reachable: false,
        fresh_peers: 0,
        battery_ratio: 0.8,
        return_feasible: false,
        local_map_revision: 1,
        highest_peer_map_revision: 1,
    });
    assert_eq!(assessment.mode, PartitionRecoveryMode::HoldAndBeacon);
    assert!(!assessment.motion_permitted);
    restrictive.partition_recovery = partition;

    embodiment
        .load_operational_checkpoint(&restrictive)
        .expect("restrictive checkpoint should establish the test state");
    assert!(!embodiment.partition_recovery_assessment().motion_permitted);

    let restore = embodiment.load_operational_checkpoint(&older);
    assert!(
        restore.is_err() || !embodiment.partition_recovery_assessment().motion_permitted,
        "stale checkpoint restored motion before partition reconciliation"
    );
    assert!(!embodiment.partition_recovery_assessment().motion_permitted);
}

#[test]
fn stale_checkpoint_must_not_clear_actuator_isolation() {
    let genesis = GenesisSeed::from_phrase("restore monotonic actuator isolation");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);
    let older = embodiment.operational_checkpoint();
    let mut restrictive = older.clone();

    let mut isolation = ActuatorIsolationSupervisor::new(ActuatorIsolationPolicy {
        mismatch_penalty: 0.25,
        mismatch_streak_limit: 4,
        isolation_threshold: 0.2,
        ..Default::default()
    });
    let state = SubterraneanState::home();
    let mut command = SubterraneanCommand::zero();
    command.set_left_track(1.0);
    for _ in 0..4 {
        isolation.observe(&command, &state, &state);
    }
    assert!(isolation.report().is_isolated(PhysicalActuator::LeftTrack));
    restrictive.actuator_isolation = isolation;

    embodiment
        .load_operational_checkpoint(&restrictive)
        .expect("restrictive checkpoint should establish isolated actuator");
    assert!(
        embodiment
            .actuator_isolation_report()
            .is_isolated(PhysicalActuator::LeftTrack)
    );

    let restore = embodiment.load_operational_checkpoint(&older);
    assert!(
        restore.is_err()
            || embodiment
                .actuator_isolation_report()
                .is_isolated(PhysicalActuator::LeftTrack),
        "stale checkpoint reauthorized a newer isolated actuator"
    );
    assert!(
        embodiment
            .actuator_isolation_report()
            .is_isolated(PhysicalActuator::LeftTrack)
    );
}

#[test]
fn stale_checkpoint_must_not_restore_nominal_field_envelope() {
    let genesis = GenesisSeed::from_phrase("restore monotonic field envelope");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);
    let older = embodiment.operational_checkpoint();
    let mut restrictive = older.clone();

    let mut state = SubterraneanState::home();
    state.channels[BATTERY_RATIO] = 0.05;
    let mut envelope = FieldEnvelopeSupervisor::default();
    let assessment = envelope.assess(&state, 1.0, MaintenanceAssessment::nominal());
    assert_eq!(assessment.mode, FieldEnvelopeMode::SurvivalHold);
    assert!(!assessment.mission_work_allowed);
    restrictive.field_envelope = envelope;

    embodiment
        .load_operational_checkpoint(&restrictive)
        .expect("restrictive checkpoint should establish survival envelope");
    assert_eq!(
        embodiment.field_envelope_assessment().mode,
        FieldEnvelopeMode::SurvivalHold
    );

    let restore = embodiment.load_operational_checkpoint(&older);
    assert!(
        restore.is_err()
            || embodiment.field_envelope_assessment().mode == FieldEnvelopeMode::SurvivalHold,
        "stale checkpoint restored a nominal field envelope over a newer survival restriction"
    );
    assert_eq!(
        embodiment.field_envelope_assessment().mode,
        FieldEnvelopeMode::SurvivalHold
    );
}
