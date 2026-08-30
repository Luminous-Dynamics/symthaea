// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transactional ratchets for the CogSec constitutional core.
//!
//! These tests intentionally model a non-zero protected sink. The important
//! property is not merely that denial leaves a default state empty; denial and
//! stale authorization must preserve already-meaningful state exactly.

use std::collections::BTreeSet;
use symthaea_cogsec::{
    ArtifactIntegrity, CapabilityFact, CognitiveSecurityLabel, Confidentiality, Consequence,
    ControlIntegrity, Digest32, MutationKind, MutationPermit, MutationRequest, OriginState,
    PolicyRule, PolicySnapshot, PrincipalId, ReferenceMonitor, ResourceId, ResourceScope,
    TaintLevel, TrustedFacts,
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

fn fixture() -> (MutationRequest, TrustedFacts, PolicySnapshot) {
    let subject = PrincipalId("local-user".into());
    let resource = ResourceId("mind/goals".into());
    let policy_root = d(9);
    let state_root = d(8);

    let request = MutationRequest {
        request_id: d(1),
        kind: MutationKind::GoalActivation,
        subject: subject.clone(),
        resource: resource.clone(),
        mutation_digest: d(2),
        expected_resource_state_root: state_root,
        expected_policy_root: policy_root,
        input_label: authenticated_label(),
        consequence: Consequence::High,
        sequence: 42,
    };

    let facts = TrustedFacts {
        resource_state_root: state_root,
        policy_root,
        policy_epoch: 7,
        authorization_epoch: 11,
        revocation_epoch: 13,
        capabilities: vec![CapabilityFact {
            capability_id: d(4),
            subject,
            mutation: MutationKind::GoalActivation,
            resource_scope: ResourceScope::Exact(resource),
            max_consequence: Consequence::High,
            authorization_epoch: 11,
            revocation_epoch: 13,
            valid_from_sequence: 40,
            valid_until_sequence: Some(50),
            revoked: false,
        }],
    };

    let policy = PolicySnapshot {
        root: policy_root,
        epoch: 7,
        rules: vec![PolicyRule {
            kind: MutationKind::GoalActivation,
            minimum_control_integrity: ControlIntegrity::Authenticated,
            maximum_taint: TaintLevel::Clean,
            capability_required: true,
        }],
    };

    (request, facts, policy)
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

    /// Reference sink contract for the future runtime integration.
    ///
    /// The sink independently checks the state binding immediately before
    /// consuming the non-cloneable permit. This is intentionally redundant with
    /// authorization-time checking: the resource can change after authorization.
    fn commit(&mut self, permit: MutationPermit) -> Result<(), &'static str> {
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
    let (mut request, facts, policy) = fixture();
    request.input_label.taint = TaintLevel::Tainted;

    let mut state = ProtectedGoalState::nonzero();
    let before = state.clone();

    let authorization = ReferenceMonitor.authorize(&request, &facts, &policy);
    assert!(authorization.is_err());
    assert_eq!(state, before);
}

#[test]
fn stale_permit_cannot_mutate_nonzero_state() {
    let (request, facts, policy) = fixture();
    let permit = ReferenceMonitor
        .authorize(&request, &facts, &policy)
        .expect("fixture must authorize");

    let mut state = ProtectedGoalState::nonzero();
    // Another valid transaction wins the race after authorization.
    state.state_root = d(77);
    state.active_goal_count = 8;
    state.accepted_mutations.push(d(76));
    let before = state.clone();

    assert_eq!(state.commit(permit), Err("stale resource state"));
    assert_eq!(state, before);
}

#[test]
fn fresh_permit_commits_once_against_exact_state() {
    let (request, facts, policy) = fixture();
    let permit = ReferenceMonitor
        .authorize(&request, &facts, &policy)
        .expect("fixture must authorize");

    let mut state = ProtectedGoalState::nonzero();
    state.commit(permit).expect("fresh permit must commit");

    assert_eq!(state.active_goal_count, 8);
    assert_eq!(state.accepted_mutations.last(), Some(&d(2)));
    assert_eq!(state.state_root, d(10));
    // MutationPermit is consumed by commit(), so the same permit cannot be
    // replayed in safe Rust without manufacturing a new authorization object.
}

#[test]
fn authorization_epoch_change_invalidates_reauthorization() {
    let (request, mut facts, policy) = fixture();
    facts.authorization_epoch += 1;

    let mut state = ProtectedGoalState::nonzero();
    let before = state.clone();
    assert!(ReferenceMonitor.authorize(&request, &facts, &policy).is_err());
    assert_eq!(state, before);
}
