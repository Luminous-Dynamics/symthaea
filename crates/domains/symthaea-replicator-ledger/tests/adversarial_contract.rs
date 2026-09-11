// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Black-box adversarial tests for the Replicator Safety Kernel ledger.
//!
//! These tests intentionally use only public APIs. They verify containment and
//! authority invariants from the perspective of an external caller and contain
//! no physical replication mechanism.

use symthaea_replicator_ledger::{
    BoundReplicationDecision, BoundedReplicationAuthorization, LedgerAuthorityContext,
    LedgerEpochId, LedgerError, LineagePolicy, MutationId, ReplicationLedger,
    RuntimeSafetyWitness, evaluate_bound_replication_authority,
};
use symthaea_replicator_safety::{
    CapabilitySet, ContainmentStatus, DenialReason, EvidenceDigest, GrantId, GrantIssuerClass,
    LineageId, MonitoringStatus, QuorumEvidence, ReplicationAuthorityRequest, ReplicationGrant,
    RiskClass, SafetyCaseStatus, SubjectId,
};

const A: CapabilitySet = CapabilitySet::from_bits(0b0001);
const B: CapabilitySet = CapabilitySet::from_bits(0b0010);
const AB: CapabilitySet = CapabilitySet::from_bits(0b0011);

fn sid(tag: u8) -> SubjectId {
    SubjectId::new([tag; 32])
}

fn lid(tag: u8) -> LineageId {
    LineageId::new([tag; 32])
}

fn gid(tag: u8) -> GrantId {
    GrantId::new([tag; 32])
}

fn dig(tag: u8) -> EvidenceDigest {
    EvidenceDigest::new([tag; 32])
}

fn mid(tag: u8) -> MutationId {
    MutationId::new([tag; 32])
}

fn root_policy() -> LineagePolicy {
    LineagePolicy {
        risk_class: RiskClass::R3,
        capability_ceiling: AB,
        max_direct_children_per_subject: 4,
        max_total_descendants: 8,
        max_lineage_depth: 4,
        max_resource_units: 100,
    }
}

fn ledger() -> ReplicationLedger {
    ReplicationLedger::new(LedgerEpochId::new([0xA7; 32]), sid(1), lid(1), root_policy())
}

fn grant(
    subject: SubjectId,
    lineage: LineageId,
    grant_tag: u8,
    generation: u64,
    capabilities: CapabilitySet,
    max_total_descendants: u64,
    max_resource_units: u64,
) -> ReplicationGrant {
    ReplicationGrant {
        grant_id: gid(grant_tag),
        subject,
        lineage,
        issuer_class: GrantIssuerClass::ExternalIndependent,
        allowed_capabilities: capabilities,
        not_before_unix_secs: 1,
        expires_at_unix_secs: 1_000,
        generation,
        max_direct_children: 4,
        max_total_descendants,
        max_lineage_depth: 4,
        max_resource_units,
        safety_case_digest: dig(4),
        containment_envelope_digest: dig(5),
    }
}

fn request(
    subject: SubjectId,
    lineage: LineageId,
    context: LedgerAuthorityContext,
    generation: u64,
    requested_capabilities: CapabilitySet,
    requested_resource_units: u64,
) -> ReplicationAuthorityRequest {
    let mut budget = context.budget;
    budget.requested_resource_units = requested_resource_units;
    let independent_approvals = if context.risk_class.requires_high_consequence_quorum() {
        2
    } else {
        1
    };

    ReplicationAuthorityRequest {
        subject,
        lineage,
        risk_class: context.risk_class,
        requested_capabilities,
        parent_capability_ceiling: context.parent_capability_ceiling,
        now_unix_secs: 100,
        expected_grant_generation: generation,
        quarantined: context.quarantined,
        revoked: context.revoked,
        lineage_status: context.lineage_status,
        monitoring: MonitoringStatus {
            healthy: true,
            fresh: true,
        },
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
        budget,
        quorum: QuorumEvidence {
            independent_approvals,
            required_independent_approvals: 1,
        },
    }
}

fn runtime() -> RuntimeSafetyWitness {
    RuntimeSafetyWitness {
        monitoring_healthy: true,
        monitoring_fresh: true,
        policy_current: true,
        evidence_current: true,
        containment_matches: true,
        safety_case_digest: dig(4),
        containment_envelope_digest: dig(5),
    }
}

fn authorize(
    ledger: &ReplicationLedger,
    subject: SubjectId,
    lineage: LineageId,
    grant: &ReplicationGrant,
    requested_capabilities: CapabilitySet,
    requested_resource_units: u64,
) -> BoundedReplicationAuthorization {
    let context = ledger
        .authority_context(subject, lineage, requested_resource_units)
        .expect("ledger context should be available");
    let request = request(
        subject,
        lineage,
        context,
        grant.generation,
        requested_capabilities,
        requested_resource_units,
    );
    match evaluate_bound_replication_authority(context.cursor, &request, Some(grant)) {
        BoundReplicationDecision::Allow(authorization) => authorization,
        decision => panic!("expected authority, got {decision:?}"),
    }
}

#[test]
fn competing_allows_from_one_snapshot_cannot_double_commit() {
    let mut ledger = ledger();
    let root = sid(1);
    let lineage = lid(1);
    let grant = grant(root, lineage, 10, 1, A, 8, 100);

    let first = authorize(&ledger, root, lineage, &grant, A, 1);
    let second = authorize(&ledger, root, lineage, &grant, A, 1);
    let shared_cursor = first.cursor();
    assert_eq!(shared_cursor, second.cursor());

    ledger
        .commit_descendant(mid(10), &first, sid(2), 1, 100, &runtime())
        .expect("first commit should consume the snapshot");

    let error = ledger
        .commit_descendant(mid(11), &second, sid(3), 1, 100, &runtime())
        .expect_err("second authorization from the stale snapshot must fail");
    assert!(matches!(error, LedgerError::CursorMismatch { .. }));
    assert_eq!(ledger.subject_snapshot(sid(3)), Err(LedgerError::UnknownSubject(sid(3))));
}

#[test]
fn runtime_monitor_failure_after_evaluation_fails_without_mutation() {
    let mut ledger = ledger();
    let root = sid(1);
    let lineage = lid(1);
    let grant = grant(root, lineage, 12, 1, A, 8, 100);
    let authorization = authorize(&ledger, root, lineage, &grant, A, 1);
    let cursor_before = ledger.cursor();
    let events_before = ledger.events().len();

    let mut failed_runtime = runtime();
    failed_runtime.monitoring_healthy = false;
    let error = ledger
        .commit_descendant(mid(12), &authorization, sid(2), 1, 100, &failed_runtime)
        .expect_err("runtime monitor loss must veto an otherwise valid allow");

    assert_eq!(error, LedgerError::RuntimeMonitorUnhealthy);
    assert_eq!(ledger.cursor(), cursor_before);
    assert_eq!(ledger.events().len(), events_before);
    assert_eq!(ledger.subject_snapshot(sid(2)), Err(LedgerError::UnknownSubject(sid(2))));
}

#[test]
fn descendant_capability_ceiling_cannot_be_reexpanded_by_a_fresh_grant() {
    let mut ledger = ledger();
    let root = sid(1);
    let child = sid(2);
    let lineage = lid(1);

    let root_grant = grant(root, lineage, 20, 1, A, 8, 100);
    let root_authorization = authorize(&ledger, root, lineage, &root_grant, A, 1);
    ledger
        .commit_descendant(mid(20), &root_authorization, child, 1, 100, &runtime())
        .expect("child creation should succeed");

    let child_snapshot = ledger.subject_snapshot(child).expect("child should exist");
    assert_eq!(child_snapshot.capability_ceiling, A);

    let fresh_child_grant = grant(child, lineage, 21, 1, AB, 8, 100);
    let context = ledger
        .authority_context(child, lineage, 1)
        .expect("child context should be available");
    let request = request(child, lineage, context, 1, B, 1);
    let decision = evaluate_bound_replication_authority(context.cursor, &request, Some(&fresh_child_grant));

    assert!(!decision.is_allowed());
    assert!(decision
        .denial_reasons()
        .contains(&DenialReason::RequestedCapabilitiesExceedParent));
}

#[test]
fn ancestor_population_budget_remains_binding_on_grandchildren() {
    let mut ledger = ledger();
    let root = sid(1);
    let child = sid(2);
    let grandchild = sid(3);
    let lineage = lid(1);

    let root_grant = grant(root, lineage, 30, 1, A, 1, 100);
    let root_authorization = authorize(&ledger, root, lineage, &root_grant, A, 1);
    ledger
        .commit_descendant(mid(30), &root_authorization, child, 1, 100, &runtime())
        .expect("root should be able to consume its one descendant slot");

    let child_grant = grant(child, lineage, 31, 1, A, 8, 100);
    let child_authorization = authorize(&ledger, child, lineage, &child_grant, A, 1);
    let error = ledger
        .commit_descendant(mid(31), &child_authorization, grandchild, 1, 100, &runtime())
        .expect_err("a fresh child grant must not erase an ancestor population ceiling");

    assert_eq!(error, LedgerError::SubjectDescendantBudgetExhausted(root));
    assert_eq!(ledger.subject_snapshot(grandchild), Err(LedgerError::UnknownSubject(grandchild)));
}

#[test]
fn ancestor_lineage_quarantine_becomes_a_constitutional_denial() {
    let mut ledger = ledger();
    let root = sid(1);
    let root_lineage = lid(1);
    let child_lineage = lid(2);

    let child_policy = LineagePolicy {
        risk_class: RiskClass::R3,
        capability_ceiling: A,
        max_direct_children_per_subject: 2,
        max_total_descendants: 4,
        max_lineage_depth: 3,
        max_resource_units: 50,
    };
    let cursor = ledger.cursor();
    ledger
        .register_lineage_branch(cursor, mid(40), child_lineage, root_lineage, child_policy)
        .expect("narrowing lineage branch should register");

    let cursor = ledger.cursor();
    ledger
        .quarantine_lineage(cursor, mid(41), root_lineage)
        .expect("root lineage quarantine should commit");

    let context = ledger
        .authority_context(root, child_lineage, 1)
        .expect("context construction should preserve negative facts for audit");
    assert!(context.quarantined);

    let branch_grant = grant(root, child_lineage, 42, 1, A, 4, 50);
    let request = request(root, child_lineage, context, 1, A, 1);
    let decision = evaluate_bound_replication_authority(context.cursor, &request, Some(&branch_grant));
    assert!(!decision.is_allowed());
    assert!(decision.denial_reasons().contains(&DenialReason::Quarantined));
}

#[test]
fn lineage_branch_cannot_lower_risk_or_widen_budget() {
    let mut ledger = ledger();
    let root_lineage = lid(1);

    let lower_risk = LineagePolicy {
        risk_class: RiskClass::R2,
        ..root_policy()
    };
    let error = ledger
        .register_lineage_branch(ledger.cursor(), mid(50), lid(2), root_lineage, lower_risk)
        .expect_err("descendant lineage must not lower ancestor risk class");
    assert_eq!(error, LedgerError::BranchPolicyEscalation);

    let wider_budget = LineagePolicy {
        max_total_descendants: root_policy().max_total_descendants + 1,
        ..root_policy()
    };
    let error = ledger
        .register_lineage_branch(ledger.cursor(), mid(51), lid(3), root_lineage, wider_budget)
        .expect_err("descendant lineage must not widen ancestor population budget");
    assert_eq!(error, LedgerError::BranchPolicyEscalation);
}

#[test]
fn failed_replay_attempt_does_not_append_a_second_event() {
    let mut ledger = ledger();
    let root = sid(1);
    let lineage = lid(1);
    let grant = grant(root, lineage, 60, 1, A, 8, 100);
    let authorization = authorize(&ledger, root, lineage, &grant, A, 1);
    let mutation = mid(60);

    ledger
        .commit_descendant(mutation, &authorization, sid(2), 1, 100, &runtime())
        .expect("first delivery should commit");
    let events_after_first = ledger.events().len();
    let cursor_after_first = ledger.cursor();

    assert!(ledger
        .commit_descendant(mutation, &authorization, sid(2), 1, 100, &runtime())
        .is_err());
    assert_eq!(ledger.events().len(), events_after_first);
    assert_eq!(ledger.cursor(), cursor_after_first);
    assert!(ledger.verify_append_only_structure());
}
