// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Black-box tests for separating quorum evidence from quorum policy.
//!
//! This file exercises abstract authority policy only. Approval counts here are
//! reference semantics, not authenticated signer/failure-domain evidence.

use symthaea_replicator_safety::{
    CapabilitySet, ContainmentStatus, DenialReason, EvidenceDigest, GrantId, GrantIssuerClass,
    LineageId, LineageStatus, MonitoringStatus, QuorumEvidence, QuorumPolicy,
    ReplicationAuthorityRequest, ReplicationBudgetSnapshot, ReplicationGrant, RiskClass,
    SafetyCaseStatus, SubjectId, evaluate_replication_authority,
    evaluate_replication_authority_with_quorum_policy,
};

const A: CapabilitySet = CapabilitySet::from_bits(0b01);

fn sid(b: u8) -> SubjectId { SubjectId::new([b; 32]) }
fn lid(b: u8) -> LineageId { LineageId::new([b; 32]) }
fn dig(b: u8) -> EvidenceDigest { EvidenceDigest::new([b; 32]) }

fn grant(subject: SubjectId, lineage: LineageId) -> ReplicationGrant {
    ReplicationGrant {
        grant_id: GrantId::new([9; 32]),
        subject,
        lineage,
        issuer_class: GrantIssuerClass::ExternalIndependent,
        allowed_capabilities: A,
        not_before_unix_secs: 1,
        expires_at_unix_secs: 1_000,
        generation: 1,
        max_direct_children: 4,
        max_total_descendants: 8,
        max_lineage_depth: 4,
        max_resource_units: 100,
        safety_case_digest: dig(4),
        containment_envelope_digest: dig(5),
    }
}

fn request(class: RiskClass, approvals: u16, requester_requirement: u16) -> ReplicationAuthorityRequest {
    ReplicationAuthorityRequest {
        subject: sid(1),
        lineage: lid(1),
        risk_class: class,
        requested_capabilities: A,
        parent_capability_ceiling: A,
        now_unix_secs: 100,
        expected_grant_generation: 1,
        quarantined: false,
        revoked: false,
        lineage_status: LineageStatus { known: true, acyclic: true, parent_binding_valid: true },
        monitoring: MonitoringStatus { healthy: true, fresh: true },
        safety_case: SafetyCaseStatus {
            design_qualified: true,
            policy_current: true,
            evidence_current: true,
            current_safety_case_digest: dig(4),
        },
        containment: ContainmentStatus {
            environment_matches: true,
            current_envelope_digest: dig(5),
        },
        budget: ReplicationBudgetSnapshot {
            direct_children_consumed: 0,
            total_descendants_consumed: 0,
            current_lineage_depth: 0,
            resource_units_consumed: 0,
            requested_resource_units: 1,
        },
        quorum: QuorumEvidence {
            independent_approvals: approvals,
            required_independent_approvals: requester_requirement,
        },
    }
}

fn denied_for_quorum(decision: symthaea_replicator_safety::ReplicationDecision) -> bool {
    decision
        .denial_reasons()
        .contains(&DenialReason::InsufficientIndependentQuorum)
}

#[test]
fn requester_zero_cannot_weaken_r3_reference_floor() {
    let g = grant(sid(1), lid(1));

    let zero = request(RiskClass::R3, 0, 0);
    assert!(denied_for_quorum(evaluate_replication_authority(&zero, Some(&g))));

    let one = request(RiskClass::R3, 1, 0);
    assert!(evaluate_replication_authority(&one, Some(&g)).is_allowed());
}

#[test]
fn weak_policy_and_request_cannot_weaken_r4_r5_floor() {
    let g = grant(sid(1), lid(1));
    let weak = QuorumPolicy::new(0, 0, 0, 0, 0);

    for class in [RiskClass::R4, RiskClass::R5] {
        let one = request(class, 1, 0);
        assert!(denied_for_quorum(
            evaluate_replication_authority_with_quorum_policy(&one, Some(&g), weak)
        ));

        let two = request(class, 2, 0);
        assert!(evaluate_replication_authority_with_quorum_policy(&two, Some(&g), weak)
            .is_allowed());
    }
}

#[test]
fn stronger_policy_dominates_weaker_requester_requirement() {
    let g = grant(sid(1), lid(1));
    let stronger = QuorumPolicy::new(0, 0, 3, 4, 4);

    let insufficient = request(RiskClass::R3, 2, 0);
    assert!(denied_for_quorum(
        evaluate_replication_authority_with_quorum_policy(&insufficient, Some(&g), stronger)
    ));

    let enough = request(RiskClass::R3, 3, 0);
    assert!(evaluate_replication_authority_with_quorum_policy(&enough, Some(&g), stronger)
        .is_allowed());
}

#[test]
fn requester_can_only_make_its_own_requirement_stricter() {
    let g = grant(sid(1), lid(1));
    let baseline = QuorumPolicy::REFERENCE_BASELINE;

    let requester_demands_three = request(RiskClass::R3, 2, 3);
    assert!(denied_for_quorum(
        evaluate_replication_authority_with_quorum_policy(
            &requester_demands_three,
            Some(&g),
            baseline,
        )
    ));

    let satisfies_self_restriction = request(RiskClass::R3, 3, 3);
    assert!(evaluate_replication_authority_with_quorum_policy(
        &satisfies_self_restriction,
        Some(&g),
        baseline,
    )
    .is_allowed());
}
