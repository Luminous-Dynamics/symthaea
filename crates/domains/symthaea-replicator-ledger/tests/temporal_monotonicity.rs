// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Black-box temporal ordering checks for the RSK ledger.
//!
//! These tests exercise abstract authorization-time semantics only. They do not
//! establish trusted clock provenance and contain no physical replication mechanism.

use symthaea_replicator_ledger::{
    BoundReplicationDecision, BoundedReplicationAuthorization, LedgerEpochId, LedgerError,
    LineagePolicy, MutationId, ReplicationLedger, RuntimeSafetyWitness,
    evaluate_bound_replication_authority,
};
use symthaea_replicator_safety::{
    CapabilitySet, ContainmentStatus, EvidenceDigest, GrantId, GrantIssuerClass, LineageId,
    MonitoringStatus, QuorumEvidence, ReplicationAuthorityRequest, ReplicationGrant, RiskClass,
    SafetyCaseStatus, SubjectId,
};

const A: CapabilitySet = CapabilitySet::from_bits(0b01);

fn sid(b: u8) -> SubjectId { SubjectId::new([b; 32]) }
fn lid(b: u8) -> LineageId { LineageId::new([b; 32]) }
fn gid(b: u8) -> GrantId { GrantId::new([b; 32]) }
fn dig(b: u8) -> EvidenceDigest { EvidenceDigest::new([b; 32]) }
fn mid(b: u8) -> MutationId { MutationId::new([b; 32]) }

fn policy() -> LineagePolicy {
    LineagePolicy {
        risk_class: RiskClass::R3,
        capability_ceiling: A,
        max_direct_children_per_subject: 4,
        max_total_descendants: 8,
        max_lineage_depth: 4,
        max_resource_units: 100,
    }
}

fn ledger() -> ReplicationLedger {
    ReplicationLedger::new(LedgerEpochId::new([7; 32]), sid(1), lid(1), policy())
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

fn authorize(ledger: &ReplicationLedger) -> BoundedReplicationAuthorization {
    let context = ledger.authority_context(sid(1), lid(1), 10).unwrap();
    let request = ReplicationAuthorityRequest {
        subject: sid(1),
        lineage: lid(1),
        risk_class: context.risk_class,
        requested_capabilities: A,
        parent_capability_ceiling: context.parent_capability_ceiling,
        now_unix_secs: 100,
        expected_grant_generation: 1,
        quarantined: context.quarantined,
        revoked: context.revoked,
        lineage_status: context.lineage_status,
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
        budget: context.budget,
        quorum: QuorumEvidence {
            independent_approvals: 1,
            required_independent_approvals: 1,
        },
    };
    let grant = ReplicationGrant {
        grant_id: gid(1),
        subject: sid(1),
        lineage: lid(1),
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
    };
    let BoundReplicationDecision::Allow(authorization) =
        evaluate_bound_replication_authority(context.cursor, &request, Some(&grant))
    else {
        panic!("expected authorization")
    };
    authorization
}

#[test]
fn pre_evaluation_time_fails_without_mutation_and_without_consuming_mutation_id() {
    let mut ledger = ledger();
    let authorization = authorize(&ledger);
    assert_eq!(authorization.evaluated_at_unix_secs(), 100);

    let cursor_before = ledger.cursor();
    let event_count_before = ledger.events().len();
    let root_before = ledger.subject_snapshot(sid(1)).unwrap();
    let lineage_before = ledger.lineage_snapshot(lid(1)).unwrap();

    assert_eq!(
        ledger.commit_descendant(mid(1), &authorization, sid(2), 10, 99, &runtime()),
        Err(LedgerError::AuthorityTimeBeforeEvaluation {
            evaluated_at: 100,
            committed_at: 99,
        })
    );

    assert_eq!(ledger.cursor(), cursor_before);
    assert_eq!(ledger.events().len(), event_count_before);
    assert_eq!(ledger.subject_snapshot(sid(1)).unwrap(), root_before);
    assert_eq!(ledger.lineage_snapshot(lid(1)).unwrap(), lineage_before);
    assert_eq!(
        ledger.subject_snapshot(sid(2)),
        Err(LedgerError::UnknownSubject(sid(2)))
    );

    // Temporal denial must not consume the mutation ID. The lower endpoint is
    // inclusive when all other predicates remain current.
    ledger
        .commit_descendant(mid(1), &authorization, sid(2), 10, 100, &runtime())
        .unwrap();
}

#[test]
fn validity_interval_is_half_open_and_expiry_denials_do_not_mutate() {
    let mut at_expiry = ledger();
    let authorization = authorize(&at_expiry);

    for committed_at in [1_000_u64, 1_001_u64] {
        let cursor_before = at_expiry.cursor();
        let events_before = at_expiry.events().len();
        assert_eq!(
            at_expiry.commit_descendant(
                mid(1),
                &authorization,
                sid(2),
                10,
                committed_at,
                &runtime(),
            ),
            Err(LedgerError::AuthorityExpired)
        );
        assert_eq!(at_expiry.cursor(), cursor_before);
        assert_eq!(at_expiry.events().len(), events_before);
        assert_eq!(
            at_expiry.subject_snapshot(sid(2)),
            Err(LedgerError::UnknownSubject(sid(2)))
        );
    }
}
