// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::time::SystemTime;
use symthaea_ai_assurance::{
    ActionRisk, AuthorityDomain, Observation, Observe, ObservedOutcome, PrincipalId, Proposed,
    ResolutionDecision, Scope, TrustError, TrustedAction, Write,
};

fn scope() -> Scope {
    Scope::new("workspace", ["symthaea", "src"]).unwrap()
}

#[test]
fn model_substituted_verifier_cannot_cross_execution_boundary() {
    let host = AuthorityDomain::new(PrincipalId::new());
    let attacker = AuthorityDomain::new(PrincipalId::new());
    let host_verifier = host.verifier();
    let attacker_verifier = attacker.verifier();
    let actor = PrincipalId::new();

    let action = TrustedAction::<Write, Proposed>::propose(
        &host_verifier,
        actor,
        "edit-source",
        scope(),
        b"patch-v1",
    )
    .assess(ActionRisk::Reversible);
    let grant =
        host.issue_bound_one_shot::<Write>(actor, scope(), None, action.authorization_binding());
    let authorized = action
        .authorize(grant, &host_verifier, SystemTime::now())
        .unwrap();

    let result = authorized.record_execution(&attacker_verifier, [1; 32], SystemTime::now());
    assert!(matches!(result, Err(TrustError::WrongDomain { .. })));
}

#[test]
fn pre_revocation_proposal_cannot_consume_post_revocation_grant() {
    let host = AuthorityDomain::new(PrincipalId::new());
    let verifier = host.verifier();
    let actor = PrincipalId::new();

    let action = TrustedAction::<Write, Proposed>::propose(
        &verifier,
        actor,
        "edit-source",
        scope(),
        b"patch-v1",
    )
    .assess(ActionRisk::Reversible);
    host.revoke_all().unwrap();
    let fresh_grant =
        host.issue_bound_one_shot::<Write>(actor, scope(), None, action.authorization_binding());

    let result = action.authorize(fresh_grant, &verifier, SystemTime::now());
    assert!(matches!(result, Err(TrustError::RevokedEpoch { .. })));
}

#[test]
fn observer_epoch_rotation_revokes_pending_observation_authority() {
    let execution = AuthorityDomain::new(PrincipalId::new());
    let observation = AuthorityDomain::new(PrincipalId::new());
    let exec_verifier = execution.verifier();
    let obs_verifier = observation.verifier();
    let actor = PrincipalId::new();
    let observer_principal = PrincipalId::new();

    let action = TrustedAction::<Write, Proposed>::propose(
        &exec_verifier,
        actor,
        "edit-source",
        scope(),
        b"patch-v1",
    )
    .assess(ActionRisk::Reversible);
    let grant = execution.issue_bound_one_shot::<Write>(
        actor,
        scope(),
        None,
        action.authorization_binding(),
    );
    let executed = action
        .authorize(grant, &exec_verifier, SystemTime::now())
        .unwrap()
        .record_execution(&exec_verifier, [2; 32], SystemTime::now())
        .unwrap();
    let observer = observation.issue_bound_one_shot::<Observe>(
        observer_principal,
        scope(),
        None,
        executed.observation_binding(),
    );
    observation.revoke_all().unwrap();

    let result = executed.observe(
        observer,
        &obs_verifier,
        Observation::new(ObservedOutcome::Success, [3; 32]),
        SystemTime::now(),
    );
    assert!(matches!(result, Err(TrustError::RevokedEpoch { .. })));
}

#[test]
fn execution_revocation_does_not_destroy_post_hoc_evidence_collection() {
    let execution = AuthorityDomain::new(PrincipalId::new());
    let observation = AuthorityDomain::new(PrincipalId::new());
    let exec_verifier = execution.verifier();
    let obs_verifier = observation.verifier();
    let actor = PrincipalId::new();
    let observer_principal = PrincipalId::new();

    let action = TrustedAction::<Write, Proposed>::propose(
        &exec_verifier,
        actor,
        "edit-source",
        scope(),
        b"patch-v1",
    )
    .assess(ActionRisk::Reversible);
    let grant = execution.issue_bound_one_shot::<Write>(
        actor,
        scope(),
        None,
        action.authorization_binding(),
    );
    let executed = action
        .authorize(grant, &exec_verifier, SystemTime::now())
        .unwrap()
        .record_execution(&exec_verifier, [4; 32], SystemTime::now())
        .unwrap();

    // Revoke future execution authority after the side effect has already happened.
    execution.revoke_all().unwrap();

    let observer = observation.issue_bound_one_shot::<Observe>(
        observer_principal,
        scope(),
        None,
        executed.observation_binding(),
    );
    let observed = executed
        .observe(
            observer,
            &obs_verifier,
            Observation::new(ObservedOutcome::Success, [5; 32]),
            SystemTime::now(),
        )
        .unwrap();
    let (_, receipt) = observed.resolve(ResolutionDecision::Confirmed);

    assert_eq!(receipt.execution_domain(), execution.domain_id());
    assert_eq!(receipt.observer_domain(), observation.domain_id());
    assert_eq!(receipt.receipt().output_digest(), [4; 32]);
}

#[test]
fn observer_verifier_substitution_is_rejected() {
    let execution = AuthorityDomain::new(PrincipalId::new());
    let observation = AuthorityDomain::new(PrincipalId::new());
    let attacker = AuthorityDomain::new(PrincipalId::new());
    let exec_verifier = execution.verifier();
    let attacker_verifier = attacker.verifier();
    let actor = PrincipalId::new();
    let observer_principal = PrincipalId::new();

    let action = TrustedAction::<Write, Proposed>::propose(
        &exec_verifier,
        actor,
        "edit-source",
        scope(),
        b"patch-v1",
    )
    .assess(ActionRisk::Reversible);
    let grant = execution.issue_bound_one_shot::<Write>(
        actor,
        scope(),
        None,
        action.authorization_binding(),
    );
    let executed = action
        .authorize(grant, &exec_verifier, SystemTime::now())
        .unwrap()
        .record_execution(&exec_verifier, [6; 32], SystemTime::now())
        .unwrap();
    let observer = observation.issue_bound_one_shot::<Observe>(
        observer_principal,
        scope(),
        None,
        executed.observation_binding(),
    );

    let result = executed.observe(
        observer,
        &attacker_verifier,
        Observation::new(ObservedOutcome::Success, [7; 32]),
        SystemTime::now(),
    );
    assert!(matches!(result, Err(TrustError::WrongDomain { .. })));
}
