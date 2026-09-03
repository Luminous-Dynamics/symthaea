// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public-surface red gate for operator replay evidence across checkpoint restore.
//!
//! Restoring an older checkpoint must not move the accepted operator sequence
//! backward and make an already-consumed command admissible again.

use symthaea_core::genesis::GenesisSeed;
use symthaea_subterranean::embodiment::SubterraneanEmbodiment;
use symthaea_subterranean::operator_authority::{
    OperatorAuthorityRejection, OperatorConstraint,
};
use symthaea_subterranean::operator_protocol::{
    AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
};

fn command(sequence: u64, proposal_id: u64) -> OperatorCommandEnvelope {
    OperatorCommandEnvelope {
        operator: OperatorId(77),
        role: OperatorRole::SafetyOfficer,
        authentication: AuthenticationLevel::HardwareBacked,
        epoch: 1,
        sequence,
        proposal_id,
        issued_step: 0,
        expires_step: 10_000,
        command: OperatorCommand::EmergencyStop,
    }
}

#[test]
fn stale_checkpoint_must_not_reopen_consumed_operator_sequence() {
    let genesis = GenesisSeed::from_phrase("restore operator replay monotonicity");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);

    embodiment
        .ingest_operator_command(command(1, 1))
        .expect("first emergency stop should be accepted");
    assert_eq!(
        embodiment.operator_constraint(),
        OperatorConstraint::EmergencyStop
    );

    // Capture the same authority constraint before sequence 2 is consumed. This
    // isolates replay evidence from authority-latch monotonicity: old and live
    // checkpoints both remain EmergencyStop.
    let older = embodiment.operational_checkpoint();

    let consumed = command(2, 2);
    embodiment
        .ingest_operator_command(consumed)
        .expect("newer sequence should be accepted once");
    assert_eq!(
        embodiment.ingest_operator_command(consumed),
        Err(OperatorAuthorityRejection::Replay),
        "live replay resistance must reject sequence 2 before restore"
    );

    let restore = embodiment.load_operational_checkpoint(&older);
    if restore.is_err() {
        return;
    }

    assert_eq!(
        embodiment.operator_constraint(),
        OperatorConstraint::EmergencyStop,
        "the test requires authority constraint to remain unchanged"
    );
    assert_eq!(
        embodiment.ingest_operator_command(consumed),
        Err(OperatorAuthorityRejection::Replay),
        "stale restore moved operator replay history backward and reopened an already-consumed command"
    );
}
