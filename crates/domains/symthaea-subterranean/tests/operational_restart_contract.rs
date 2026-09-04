// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public API qualification for RA-35 operational restart semantics.

use symthaea_core::genesis::GenesisSeed;
use symthaea_subterranean::embodiment::SubterraneanEmbodiment;
use symthaea_subterranean::{
    ArtifactDigest, AuthenticationLevel, DegradedMode, OPERATIONAL_CHECKPOINT_SCHEMA_VERSION,
    OperationalRestart, OperatorCommand, OperatorCommandEnvelope, OperatorConstraint, OperatorId,
    OperatorRole, PartitionRecoveryMode, TemporalAuthority, UPDATE_MANIFEST_SCHEMA_VERSION,
    UpdateManifest, UpdateState,
};

fn maintenance_lock() -> OperatorCommandEnvelope {
    OperatorCommandEnvelope {
        operator: OperatorId(71),
        role: OperatorRole::SafetyOfficer,
        authentication: AuthenticationLevel::HardwareBacked,
        epoch: 1,
        sequence: 1,
        proposal_id: 7101,
        issued_step: 0,
        expires_step: 100,
        command: OperatorCommand::EnterMaintenance,
    }
}

#[test]
fn rollback_required_survives_public_operational_restart() {
    let mut live = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase(
        "ra35-public-rollback-required",
    ));
    live.ingest_operator_command(maintenance_lock())
        .expect("maintenance lock");

    let old_digest = ArtifactDigest([1; 32]);
    let new_digest = ArtifactDigest([2; 32]);
    live.initialize_update_control(old_digest, 1)
        .expect("initialize update control");
    live.stage_update(UpdateManifest {
        schema_version: UPDATE_MANIFEST_SCHEMA_VERSION,
        release_id: 2,
        artifact_digest: new_digest,
        configuration_digest: ArtifactDigest([3; 32]),
        rollback_digest: old_digest,
        minimum_checkpoint_schema: OPERATIONAL_CHECKPOINT_SCHEMA_VERSION,
        issued_epoch: 2,
        expires_step: 100,
    })
    .expect("stage update");
    assert_eq!(
        live.activate_staged_update(50).expect("activate update"),
        new_digest
    );
    assert_eq!(live.update_state(), Some(UpdateState::PendingHealth));
    assert_eq!(
        live.observe_update_health(false),
        Ok(UpdateState::RollbackRequired)
    );

    let before = live.operational_checkpoint();
    assert_eq!(
        before.update_manager.as_ref().map(|manager| manager.state()),
        Some(UpdateState::RollbackRequired)
    );
    assert_eq!(
        before
            .update_manager
            .as_ref()
            .map(|manager| manager.current_digest()),
        Some(new_digest)
    );

    let report = live
        .restart_operational_runtime()
        .expect("conservative operational restart");
    assert_eq!(report.update_state, Some(UpdateState::RollbackRequired));
    assert_eq!(live.update_state(), Some(UpdateState::RollbackRequired));

    let after = live.operational_checkpoint();
    assert_eq!(
        after.update_manager.as_ref().map(|manager| manager.state()),
        Some(UpdateState::RollbackRequired)
    );
    assert_eq!(
        after
            .update_manager
            .as_ref()
            .map(|manager| manager.current_digest()),
        Some(new_digest)
    );

    // Restart preserves the rollback obligation; it does not perform or waive
    // rollback automatically. The existing explicit rollback path remains the
    // only operation that can consume the obligation.
    assert_eq!(live.rollback_update().expect("explicit rollback"), old_digest);
    assert_eq!(live.update_state(), Some(UpdateState::RolledBack));
}

#[test]
fn repeated_operational_restart_is_authority_idempotent() {
    let mut live = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase(
        "ra35-public-restart-idempotence",
    ));
    live.ingest_operator_command(maintenance_lock())
        .expect("maintenance lock");

    let first = live
        .restart_operational_runtime()
        .expect("first conservative restart");
    let second = live
        .restart_operational_runtime()
        .expect("second conservative restart");

    // A repeated restart may represent another continuity break, but it cannot
    // earn recovery progress or weaken any observable authority dimension.
    assert_eq!(second, first);
    assert_eq!(second.operator_constraint, OperatorConstraint::MaintenanceLock);
    assert_eq!(second.degraded_mode, DegradedMode::RecoveryRequired);
    assert_eq!(second.partition_mode, PartitionRecoveryMode::Reconciling);
    assert!(!second.partition_motion_permitted);
    assert!(!second.team_state_authoritative);
    assert_eq!(second.temporal_authority, TemporalAuthority::HoldForReview);
    assert!(second.temporal_hold_latched);
}
