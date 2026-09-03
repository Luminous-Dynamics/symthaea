// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public-surface regression gates for recovery-authority lifecycle monotonicity.
//!
//! These tests intentionally exercise only downstream-callable APIs. A reset or
//! checkpoint restore is a lifecycle operation, not an implicit recovery grant.

use symthaea_core::genesis::GenesisSeed;
use symthaea_subterranean::embodiment::SubterraneanEmbodiment;
use symthaea_subterranean::{
    AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorConstraint, OperatorId,
    OperatorRole,
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
