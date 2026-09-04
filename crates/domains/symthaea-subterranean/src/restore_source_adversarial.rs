// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adversarial qualification for restore-source portable normalization.
//!
//! These tests target subtle RA-32 boundaries:
//! - host-local state marked with `serde(skip)` must never survive into the
//!   normalized object owned by an `OperationalRestoreSource`;
//! - durable adverse/replay evidence must survive normalization;
//! - semantically equivalent bounded mission state must have one deterministic
//!   portable representation regardless of caller insertion order;
//! - skipped diagnostic cache state may disappear only while the durable state
//!   that actually constrains future scheduling remains intact.

use super::restore_admission::OperationalRestoreSource;
use crate::embodiment::SubterraneanEmbodiment;
use crate::operator_authority::recovery_authority::{
    RecoveryApprovalEnvelopeV1, RecoveryDigest, RecoveryProposalV1,
};
use crate::operator_authority::{OperatorAuthorityRejection, OperatorConstraint, OperatorDecision};
use crate::operator_protocol::{
    AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
};
use crate::tunnel_graph::{TunnelEdge, TunnelNode, TunnelNodeId, TunnelNodeKind};
use crate::work_orders::{
    WorkKind, WorkOrder, WorkOrderId, WorkPreemptionReason, WorkPriority, WorkResourceEstimate,
    WorkStatus,
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

fn mission_node(id: u32, kind: TunnelNodeKind, depth_m: f64) -> TunnelNode {
    TunnelNode {
        id: TunnelNodeId(id),
        kind,
        depth_m,
        survey_confidence: 0.95,
    }
}

fn mission_edge(from: u32, to: u32, revision: u64) -> TunnelEdge {
    TunnelEdge {
        from: TunnelNodeId(from),
        to: TunnelNodeId(to),
        length_m: 10.0 + f64::from(to),
        energy_per_m: 0.001,
        obstruction_risk: 0.05,
        water_risk: 0.04,
        roof_risk: 0.03,
        confidence: 0.96,
        traversable: true,
        bidirectional: true,
        revision,
    }
}

fn mission_work(id: u64, target: u32, kind: WorkKind) -> WorkOrder {
    WorkOrder {
        id: WorkOrderId(id),
        kind,
        target: TunnelNodeId(target),
        priority: WorkPriority::Routine,
        prerequisites: [None; 4],
        estimated_steps: 10 + id,
        deadline_step: None,
        resources: WorkResourceEstimate {
            battery_fraction: 0.01,
            sealant_fraction: 0.0,
            relay_units: 0,
            roof_support_units: 0,
            sample_capacity: 0.0,
            spoil_capacity: 0.02,
        },
        status: WorkStatus::Pending,
        completed_steps: 0,
    }
}

fn mission_source(reverse: bool) -> OperationalRestoreSource {
    let genesis = GenesisSeed::from_phrase("restore-source-canonical-mission");
    let mut embodiment = SubterraneanEmbodiment::new(&genesis);

    let mut nodes = [
        mission_node(1, TunnelNodeKind::Junction, 8.0),
        mission_node(2, TunnelNodeKind::Workface, 16.0),
    ];
    if reverse {
        nodes.reverse();
    }
    for node in nodes {
        embodiment.add_tunnel_node(node).expect("mission node");
    }

    let mut edges = [mission_edge(0, 1, 1), mission_edge(1, 2, 1)];
    if reverse {
        edges.reverse();
    }
    for edge in edges {
        embodiment.upsert_tunnel_edge(edge).expect("mission edge");
    }

    let mut work = [
        mission_work(10, 1, WorkKind::Survey),
        mission_work(20, 2, WorkKind::Bore),
    ];
    if reverse {
        work.reverse();
    }
    for order in work {
        embodiment.submit_work_order(order).expect("work order");
    }

    OperationalRestoreSource::capture(embodiment.operational_checkpoint())
        .expect("canonical mission source")
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

#[test]
fn equivalent_mission_insertion_orders_have_one_portable_source_identity() {
    let forward = mission_source(false);
    let reverse = mission_source(true);

    // Graph nodes/edges and work orders are canonicalized by their domain
    // owners. Caller insertion order therefore cannot become hidden restore
    // identity or make equivalent portable state commit differently.
    assert_eq!(forward.digest(), reverse.digest());

    let forward_bytes =
        serde_json::to_vec(forward.checkpoint()).expect("forward normalized encoding");
    let reverse_bytes =
        serde_json::to_vec(reverse.checkpoint()).expect("reverse normalized encoding");
    assert_eq!(forward_bytes, reverse_bytes);
}

#[test]
fn skipped_scheduler_preemption_diagnostic_drops_but_suspended_work_survives() {
    let genesis = GenesisSeed::from_phrase("restore-source-skipped-preemption-diagnostic");
    let mut checkpoint = SubterraneanEmbodiment::new(&genesis).operational_checkpoint();
    let order = mission_work(77, 0, WorkKind::Survey);
    checkpoint
        .mission
        .scheduler
        .submit(order)
        .expect("submit work");
    assert_eq!(
        checkpoint.mission.scheduler.select_next(0),
        Some(WorkOrderId(77))
    );
    checkpoint
        .mission
        .scheduler
        .preempt(WorkPreemptionReason::PhysicalHazard)
        .expect("preempt active work");
    let before = checkpoint.mission.scheduler.snapshot();
    assert_eq!(
        before.last_preemption,
        Some(WorkPreemptionReason::PhysicalHazard)
    );
    assert_eq!(before.active, None);
    assert_eq!(
        checkpoint
            .mission
            .scheduler
            .order(WorkOrderId(77))
            .expect("durable work")
            .status,
        WorkStatus::Suspended
    );

    let portable_before = serde_json::to_vec(&checkpoint).expect("portable checkpoint");
    let normalized = OperationalRestoreSource::capture(checkpoint).expect("normalized source");
    let portable_after =
        serde_json::to_vec(normalized.checkpoint()).expect("normalized checkpoint encoding");

    // The skipped diagnostic must not create hidden source identity.
    assert_eq!(portable_after, portable_before);
    assert_eq!(
        normalized
            .checkpoint()
            .mission
            .scheduler
            .snapshot()
            .last_preemption,
        None
    );

    // But the durable scheduling consequence remains: the work is still
    // suspended and no active order was invented by normalization.
    assert_eq!(
        normalized
            .checkpoint()
            .mission
            .scheduler
            .order(WorkOrderId(77))
            .expect("normalized work")
            .status,
        WorkStatus::Suspended
    );
    assert_eq!(
        normalized.checkpoint().mission.scheduler.snapshot().active,
        None
    );
}
