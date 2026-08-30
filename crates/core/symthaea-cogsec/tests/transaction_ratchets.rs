// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transactional ratchets for the monitor-domain-sealed CogSec facade.
//!
//! These tests intentionally model a non-zero protected sink. The important
//! property is not merely that denial leaves a default state empty; denial,
//! stale authorization, and foreign-monitor authority must preserve already
//! meaningful state exactly.

use std::collections::BTreeSet;
use symthaea_cogsec::{
    ArtifactIntegrity, AuthorityError, CapabilityFact, CognitiveSecurityLabel, CommitPermit,
    Confidentiality, Consequence, ControlIntegrity, Digest32, MutationKind, MutationRequest,
    OriginState, PolicyRule, PolicySnapshot, PrincipalId, ReferenceMonitor, ResourceId,
    ResourceScope, TaintLevel, TrustedFactAuthority, TrustedFacts,
};

fn d(byte: u8) -> Digest32 {
    Digest32([byte; 32])
}

fn authenticated_label() -> CognitiveSecurityLabel {
    let mut provenance_roots = BTreeSet::new();
    provenance_roots.insert(d(3));
    CognitiveSecurityLabel {
        control_integrity: ControlIntegrity::Authenticated,
        confidentiality: Confidentiality::Local,
        origin: OriginState::Authenticated,
        artifact_integrity: ArtifactIntegrity::Authenticated,
        taint: TaintLevel::Clean,
        provenance_roots,
    }
}

struct Fixture {
    monitor: ReferenceMonitor,
    authority: TrustedFactAuthority,
    capability: CapabilityFact,
    request: MutationRequest,
    policy: PolicySnapshot,
}

impl Fixture {
    fn new() -> Self {
        let (monitor, authority) = ReferenceMonitor::bootstrap();
        let subject = PrincipalId("local-user".into());
        let resource = ResourceId("mind/goals".into());
        let capability = authority.issue_capability(
            d(4),
            subject.clone(),
            MutationKind::GoalActivation,
            ResourceScope::Exact(resource.clone()),
            Consequence::High,
            11,
            13,
            40,
            Some(50),
            false,
        );
        let request = MutationRequest {
            request_id: d(1),
            kind: MutationKind::GoalActivation,
            subject,
            resource,
            mutation_digest: d(2),
            expected_resource_state_root: d(8),
            expected_policy_root: d(9),
            input_label: authenticated_label(),
            consequence: Consequence::High,
            sequence: 42,
        };
        let policy = PolicySnapshot {
            root: d(9),
            epoch: 7,
            rules: vec![PolicyRule {
                kind: MutationKind::GoalActivation,
                minimum_control_integrity: ControlIntegrity::Authenticated,
                maximum_taint: TaintLevel::Clean,
                capability_required: true,
            }],
        };
        Self {
            monitor,
            authority,
            capability,
            request,
            policy,
        }
    }

    fn facts(
        &self,
        state_root: Digest32,
        authorization_epoch: u64,
        revocation_epoch: u64,
    ) -> TrustedFacts {
        self.authority
            .snapshot(
                state_root,
                self.policy.root,
                self.policy.epoch,
                authorization_epoch,
                revocation_epoch,
                &[&self.capability],
            )
            .expect("fixture facts must belong to monitor domain")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProtectedGoalState {
    state_root: Digest32,
    active_goal_count: u64,
    accepted_mutations: Vec<Digest32>,
}

impl ProtectedGoalState {
    fn nonzero() -> Self {
        Self {
            state_root: d(8),
            active_goal_count: 7,
            accepted_mutations: vec![d(31), d(32)],
        }
    }

    /// Reference P0 sink contract for future runtime integration.
    ///
    /// The sink both consumes post-revalidation typestate and checks that the
    /// permit belongs to the monitor domain owned by this protected state.
    fn commit(
        &mut self,
        monitor: &ReferenceMonitor,
        permit: CommitPermit,
    ) -> Result<(), &'static str> {
        if !monitor.accepts_commit_permit(&permit) {
            return Err("foreign monitor domain");
        }
        if permit.kind() != MutationKind::GoalActivation {
            return Err("wrong mutation kind");
        }
        if permit.resource() != &ResourceId("mind/goals".into()) {
            return Err("wrong resource");
        }
        if permit.resource_state_root() != self.state_root {
            return Err("stale resource state");
        }

        self.active_goal_count += 1;
        self.accepted_mutations.push(permit.mutation_digest());
        self.state_root = d(10);
        Ok(())
    }
}

#[test]
fn denied_request_preserves_nonzero_state_exactly() {
    let mut fixture = Fixture::new();
    fixture.request.input_label.taint = TaintLevel::Tainted;
    let facts = fixture.facts(d(8), 11, 13);

    let state = ProtectedGoalState::nonzero();
    let before = state.clone();

    let authorization = fixture
        .monitor
        .authorize(&fixture.request, &facts, &fixture.policy);
    assert!(authorization.is_err());
    assert_eq!(state, before);
}

#[test]
fn stale_authorization_cannot_reach_commit_typestate_or_mutate_state() {
    let fixture = Fixture::new();
    let initial_facts = fixture.facts(d(8), 11, 13);
    let permit = fixture
        .monitor
        .authorize(&fixture.request, &initial_facts, &fixture.policy)
        .expect("fixture must authorize");

    let mut state = ProtectedGoalState::nonzero();
    state.state_root = d(77);
    state.active_goal_count = 8;
    state.accepted_mutations.push(d(76));
    let before = state.clone();

    let fresh_facts = fixture.facts(state.state_root, 11, 13);
    let precommit = fixture
        .monitor
        .precommit(permit, &fresh_facts, &fixture.policy);
    assert!(precommit.is_err());
    assert_eq!(state, before);
}

#[test]
fn fresh_commit_permit_commits_once_against_exact_state_and_domain() {
    let fixture = Fixture::new();
    let facts = fixture.facts(d(8), 11, 13);
    let permit = fixture
        .monitor
        .authorize(&fixture.request, &facts, &fixture.policy)
        .expect("fixture must authorize");
    let commit_permit = fixture
        .monitor
        .precommit(permit, &facts, &fixture.policy)
        .expect("unchanged security context must precommit");

    let mut state = ProtectedGoalState::nonzero();
    state
        .commit(&fixture.monitor, commit_permit)
        .expect("fresh same-domain commit permit must commit");

    assert_eq!(state.active_goal_count, 8);
    assert_eq!(state.accepted_mutations.last(), Some(&d(2)));
    assert_eq!(state.state_root, d(10));
}

#[test]
fn revocation_between_authorize_and_precommit_preserves_nonzero_state() {
    let fixture = Fixture::new();
    let facts = fixture.facts(d(8), 11, 13);
    let permit = fixture
        .monitor
        .authorize(&fixture.request, &facts, &fixture.policy)
        .expect("fixture must authorize");

    let state = ProtectedGoalState::nonzero();
    let before = state.clone();
    let revoked_context = fixture.facts(d(8), 11, 14);

    assert!(
        fixture
            .monitor
            .precommit(permit, &revoked_context, &fixture.policy)
            .is_err()
    );
    assert_eq!(state, before);
}

#[test]
fn authorization_epoch_change_invalidates_reauthorization() {
    let fixture = Fixture::new();
    let changed = fixture.facts(d(8), 12, 13);

    let state = ProtectedGoalState::nonzero();
    let before = state.clone();
    assert!(
        fixture
            .monitor
            .authorize(&fixture.request, &changed, &fixture.policy)
            .is_err()
    );
    assert_eq!(state, before);
}

#[test]
fn foreign_monitor_commit_permit_is_rejected_without_state_change() {
    let fixture_a = Fixture::new();
    let fixture_b = Fixture::new();
    let facts_b = fixture_b.facts(d(8), 11, 13);
    let permit_b = fixture_b
        .monitor
        .authorize(&fixture_b.request, &facts_b, &fixture_b.policy)
        .expect("domain B must authorize its own request");
    let commit_b = fixture_b
        .monitor
        .precommit(permit_b, &facts_b, &fixture_b.policy)
        .expect("domain B must precommit its own permit");

    let mut state = ProtectedGoalState::nonzero();
    let before = state.clone();
    let result = state.commit(&fixture_a.monitor, commit_b);
    assert_eq!(result, Err("foreign monitor domain"));
    assert_eq!(state, before);
}

#[test]
fn facts_from_foreign_domain_never_reach_inner_monitor() {
    let fixture_a = Fixture::new();
    let fixture_b = Fixture::new();
    let facts_b = fixture_b.facts(d(8), 11, 13);

    let result = fixture_a
        .monitor
        .authorize(&fixture_a.request, &facts_b, &fixture_a.policy);
    assert!(matches!(
        result,
        Err(AuthorityError::MonitorDomainMismatch)
    ));
}
