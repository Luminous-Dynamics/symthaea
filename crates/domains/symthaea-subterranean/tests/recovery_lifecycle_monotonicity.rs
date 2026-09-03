// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public-surface regression gates for recovery-authority lifecycle monotonicity.
//!
//! These tests intentionally exercise only downstream-callable APIs. A reset or
//! checkpoint restore is a lifecycle operation, not an implicit recovery grant.

use symthaea_core::genesis::GenesisSeed;
use symthaea_subterranean::embodiment::SubterraneanEmbodiment;
use symthaea_subterranean::{
    AuthenticationLevel, MIN_SUPPORTED_OPERATIONAL_CHECKPOINT_SCHEMA_VERSION, OperatorCommand,
    OperatorCommandEnvelope, OperatorConstraint, OperatorId, OperatorRole,
    SubterraneanOperationalCheckpoint,
};

fn command(sequence: u64, proposal_id: u64, command: OperatorCommand) -> OperatorCommandEnvelope {
    OperatorCommandEnvelope {
        operator: OperatorId(7),
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
fn public_reset_must_not_clear_active_operator_restriction() {
    let genesis = GenesisSeed::from_phrase("recovery lifecycle reset monotonicity");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);

    embodiment
        .ingest_operator_command(command(1, 1, OperatorCommand::EmergencyStop))
        .expect("emergency stop should be accepted");
    assert_eq!(
        embodiment.operator_constraint(),
        OperatorConstraint::EmergencyStop
    );

    embodiment.reset();

    assert_eq!(
        embodiment.operator_constraint(),
        OperatorConstraint::EmergencyStop,
        "public reset widened authority by clearing an active operator restriction"
    );
}

#[test]
fn stale_checkpoint_must_not_clear_a_newer_operator_restriction() {
    let genesis = GenesisSeed::from_phrase("recovery lifecycle stale checkpoint");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);

    // Capture a valid historical state before the later restriction exists.
    let older_checkpoint = embodiment.operational_checkpoint();

    embodiment
        .ingest_operator_command(command(1, 1, OperatorCommand::HoldPosition))
        .expect("hold should be accepted");
    assert_eq!(embodiment.operator_constraint(), OperatorConstraint::HoldPosition);

    let restore = embodiment.load_operational_checkpoint(&older_checkpoint);

    assert!(
        restore.is_err() || embodiment.operator_constraint() == OperatorConstraint::HoldPosition,
        "restoring a valid older checkpoint widened current operator authority"
    );
    assert_eq!(
        embodiment.operator_constraint(),
        OperatorConstraint::HoldPosition,
        "historical checkpoint authority must not replace a newer live restriction"
    );
}

#[test]
fn legacy_checkpoint_missing_operator_authority_must_not_widen_live_authority() {
    let genesis = GenesisSeed::from_phrase("recovery lifecycle legacy checkpoint");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);

    // Model a checkpoint produced by a supported older schema before the
    // operator-authority field existed. Current serde compatibility defaults a
    // missing authority field, so the restore boundary must still fail closed.
    let checkpoint = embodiment.operational_checkpoint();
    let mut encoded = serde_json::to_value(checkpoint).expect("checkpoint should serialize");
    let object = encoded
        .as_object_mut()
        .expect("checkpoint serialization should be an object");
    object.remove("operator_authority");
    object.insert(
        "schema_version".to_string(),
        serde_json::json!(MIN_SUPPORTED_OPERATIONAL_CHECKPOINT_SCHEMA_VERSION),
    );

    // Rejecting such a legacy checkpoint at deserialization is an acceptable
    // future policy. If compatibility keeps accepting it, loading it must not
    // weaken a newer live restriction.
    let legacy = match serde_json::from_value::<SubterraneanOperationalCheckpoint>(encoded) {
        Ok(value) => value,
        Err(_) => return,
    };

    embodiment
        .ingest_operator_command(command(1, 1, OperatorCommand::EmergencyStop))
        .expect("emergency stop should be accepted");
    assert_eq!(
        embodiment.operator_constraint(),
        OperatorConstraint::EmergencyStop
    );

    let restore = embodiment.load_operational_checkpoint(&legacy);
    assert!(
        restore.is_err() || embodiment.operator_constraint() == OperatorConstraint::EmergencyStop,
        "legacy checkpoint compatibility synthesized wider operator authority"
    );
    assert_eq!(
        embodiment.operator_constraint(),
        OperatorConstraint::EmergencyStop,
        "missing historical authority state must not become current permission"
    );
}

#[test]
fn checkpoint_with_equal_operator_restriction_remains_restorable() {
    let genesis = GenesisSeed::from_phrase("recovery lifecycle equal checkpoint");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);

    embodiment
        .ingest_operator_command(command(1, 1, OperatorCommand::HoldPosition))
        .expect("hold should be accepted");
    let checkpoint = embodiment.operational_checkpoint();

    embodiment
        .load_operational_checkpoint(&checkpoint)
        .expect("equal-authority checkpoint should remain restorable");

    assert_eq!(embodiment.operator_constraint(), OperatorConstraint::HoldPosition);
}
