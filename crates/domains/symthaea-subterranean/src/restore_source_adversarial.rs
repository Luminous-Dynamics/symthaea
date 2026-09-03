// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adversarial qualification for restore-source portable normalization.
//!
//! These tests target a subtle RA-32 boundary: host-local state marked with
//! `serde(skip)` must never survive into the normalized object owned by an
//! `OperationalRestoreSource`, while durable replay evidence must survive.

use super::restore_admission::OperationalRestoreSource;
use crate::embodiment::SubterraneanEmbodiment;
use crate::operator_authority::{OperatorAuthorityRejection, OperatorConstraint, OperatorDecision};
use crate::operator_authority::recovery_authority::{
    RecoveryApprovalEnvelopeV1, RecoveryDigest, RecoveryProposalV1,
};
use crate::operator_protocol::{
    AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
};
use symthaea_core::genesis::GenesisSeed;

fn hold_command(sequence: u64) -> OperatorCommandEnvelope {
    OperatorCommandEnvelope {
        operator: OperatorId(41),
        role: OperatorRole::SafetyOfficer,
        authentication: AuthenticationLevel::HardwareBacked,
        epoch: 1,
        sequence,
        proposal_id: 4100 + sequence,
        issued_step: 20,
        expires_step: 100,
        command: OperatorCommand::HoldPosition,
    }
}

fn recovery_proposal(id: u64) -> RecoveryProposalV1 {
    RecoveryProposalV1::new(
        id,
        OperatorConstraint::HoldPosition,
        RecoveryDigest([1; 32]),
        RecoveryDigest([2; 32]),
        RecoveryDigest([3; 32]),
        7,
        11,
        20,
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
        approval_issued_step: 21,
        proposal,
    }
}

fn checkpoint_with_hold(phrase: &str) -> super::SubterraneanOperationalCheckpoint {
    let mut checkpoint =
        SubterraneanEmbodiment::new(&GenesisSeed::from_phrase(phrase)).operational_checkpoint();
    checkpoint
        .operator_authority
        .ingest(hold_command(1), 20, true)
        .expect("hold must be accepted");
    assert_eq!(
        checkpoint.operator_authority.constraint(),
        OperatorConstraint::HoldPosition
    );
    checkpoint
}

#[test]
fn host_local_recovery_issuance_cannot_change_committed_source_identity() {
    let baseline = OperationalRestoreSource::capture(checkpoint_with_hold(
        "restore-source-skipped-issuance",
    ))
    .expect("baseline source");

    let mut candidate = checkpoint_with_hold("restore-source-skipped-issuance");
    let proposal = recovery_proposal(9001);
    candidate
        .operator_authority
        .issue_recovery_proposal(proposal, 20)
        .expect("host-local issuance");
    assert_eq!(
        candidate
            .operator_authority
            .issued_recovery_proposal(proposal.proposal_id()),
        Some(proposal)
    );

    let normalized = OperationalRestoreSource::capture(candidate).expect("normalized source");

    // `issued_recovery` is host-local and skipped by portable serialization, so
    // it cannot alter either the committed identity or the object executors see.
    assert_eq!(normalized.digest(), baseline.digest());
    assert_eq!(
        normalized
            .checkpoint()
            .operator_authority
            .issued_recovery_proposal(proposal.proposal_id()),
        None
    );
    assert_eq!(
        normalized.checkpoint().operator_authority.constraint(),
        OperatorConstraint::HoldPosition
    );
}

#[test]
fn partial_recovery_quorum_is_dropped_but_consumed_replay_evidence_survives() {
    let mut candidate = checkpoint_with_hold("restore-source-skipped-quorum");
    let proposal = recovery_proposal(9002);
    candidate
        .operator_authority
        .issue_recovery_proposal(proposal, 20)
        .expect("host-local issuance");

    let approval = recovery_approval(52, 1, proposal);
    assert!(matches!(
        candidate
            .operator_authority
            .approve_recovery(approval, 21)
            .expect("first approval"),
        OperatorDecision::PendingQuorum {
            approvals: 1,
            required: 2
        }
    ));
    assert_eq!(candidate.operator_authority.pending_approvals(9002), 1);

    let normalized = OperationalRestoreSource::capture(candidate).expect("normalized source");
    assert_eq!(
        normalized
            .checkpoint()
            .operator_authority
            .issued_recovery_proposal(9002),
        None
    );
    assert_eq!(
        normalized
            .checkpoint()
            .operator_authority
            .pending_approvals(9002),
        0
    );

    // The approval's positive widening progress is ephemeral, but its consumed
    // replay sequence is durable adverse/replay evidence and must remain.
    let mut restored = normalized.checkpoint().operator_authority.clone();
    assert_eq!(
        restored.approve_recovery(approval, 21),
        Err(OperatorAuthorityRejection::RecoveryProposalNotIssued)
    );

    // The same operator sequence cannot be reused as a normal command either.
    let replay = OperatorCommandEnvelope {
        operator: OperatorId(52),
        role: OperatorRole::SafetyOfficer,
        authentication: AuthenticationLevel::HardwareBacked,
        epoch: 1,
        sequence: 1,
        proposal_id: 9999,
        issued_step: 20,
        expires_step: 100,
        command: OperatorCommand::EmergencyStop,
    };
    assert_eq!(
        restored.ingest(replay, 21, true),
        Err(OperatorAuthorityRejection::Replay)
    );
}
