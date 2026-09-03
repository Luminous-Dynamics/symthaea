// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use ed25519_dalek::{Signer, SigningKey};
use symthaea_authority::{
    AuthorityContext, AuthorityDecision, AuthorityEpoch, CapabilityGrant, DenyReason, Digest32,
    GrantUseState, NegativeAuthorityFact, Operation, PrincipalId, ResourceRef, evaluate_authority,
};
use symthaea_authority_state::{
    AUTHORITY_STATE_SCHEMA_VERSION, AuthorityStateChallengeV1, AuthorityStateError,
    AuthorityStatePolicyV1, AuthorityStateStatementV1, AuthorityStateWitnessId,
    PendingAuthorityStateChallenge, TrustedAuthorityStateWitnessV1, verify_authority_state_v1,
};
use symthaea_authority_time::{
    AUTHORITY_TIME_SCHEMA_VERSION, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
    TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1, VerifiedAuthorityTime,
    verify_authority_time_v1,
};

fn grant() -> CapabilityGrant {
    let mut grant = CapabilityGrant::new(
        "authority-state-test-grant",
        PrincipalId("xenia://operator/alice".into()),
        PrincipalId("symthaea://agent/system-recovery".into()),
        AuthorityEpoch(7),
    );
    grant.audience = Some(PrincipalId(
        "spiffe://luminous.local/symthaea/system-broker".into(),
    ));
    grant.resources = BTreeSet::from([
        ResourceRef("host:alpha/service:postgresql".into()),
        ResourceRef("host:alpha/service:nginx".into()),
    ]);
    grant.operations = BTreeSet::from([Operation("service.restart".into())]);
    grant.expires_at_unix_s = Some(2_000);
    grant.max_uses = 1;
    grant
}

fn authority_time(grant: &CapabilityGrant, witnessed: u64) -> VerifiedAuthorityTime {
    let key_a = SigningKey::from_bytes(&[71; 32]);
    let key_b = SigningKey::from_bytes(&[72; 32]);
    let policy = TrustedTimePolicyV1 {
        schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
        policy_id: [73; 16],
        authorities: vec![
            TrustedTimeAuthorityV1 {
                authority_id: TimeAuthorityId([1; 16]),
                verifying_key: key_a.verifying_key().to_bytes(),
                organization_binding: [81; 32],
                service_binding: [91; 32],
            },
            TrustedTimeAuthorityV1 {
                authority_id: TimeAuthorityId([2; 16]),
                verifying_key: key_b.verifying_key().to_bytes(),
                organization_binding: [82; 32],
                service_binding: [92; 32],
            },
        ],
        threshold: 2,
        minimum_organizations: 2,
        maximum_uncertainty_s: 1,
        maximum_challenge_age_ns: 5_000_000_000,
        maximum_post_verification_age_ns: 5_000_000_000,
    };
    let pending = PendingAuthorityTimeChallenge::new(&policy, grant.digest().0).unwrap();
    let challenge = pending.wire();
    let sign = |authority_id: TimeAuthorityId, key: &SigningKey| {
        let mut statement = AuthorityTimeStatementV1 {
            schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
            authority_id,
            policy_digest: challenge.policy_digest,
            subject_digest: challenge.subject_digest,
            challenge_nonce: challenge.nonce,
            witnessed_unix_s: witnessed,
            uncertainty_s: 1,
            signature: Vec::new(),
        };
        statement.signature = key
            .sign(&statement.canonical_message().unwrap())
            .to_bytes()
            .to_vec();
        statement
    };
    verify_authority_time_v1(
        &policy,
        pending,
        &[
            sign(TimeAuthorityId([1; 16]), &key_a),
            sign(TimeAuthorityId([2; 16]), &key_b),
        ],
    )
    .unwrap()
}

fn state_policy() -> (AuthorityStatePolicyV1, Vec<SigningKey>) {
    let key_a = SigningKey::from_bytes(&[11; 32]);
    let key_b = SigningKey::from_bytes(&[12; 32]);
    let key_c = SigningKey::from_bytes(&[13; 32]);
    (
        AuthorityStatePolicyV1 {
            schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
            policy_id: [14; 16],
            witnesses: vec![
                TrustedAuthorityStateWitnessV1 {
                    witness_id: AuthorityStateWitnessId([1; 16]),
                    verifying_key: key_a.verifying_key().to_bytes(),
                    organization_binding: [21; 32],
                    service_binding: [31; 32],
                },
                TrustedAuthorityStateWitnessV1 {
                    witness_id: AuthorityStateWitnessId([2; 16]),
                    verifying_key: key_b.verifying_key().to_bytes(),
                    organization_binding: [22; 32],
                    service_binding: [32; 32],
                },
                TrustedAuthorityStateWitnessV1 {
                    witness_id: AuthorityStateWitnessId([3; 16]),
                    verifying_key: key_c.verifying_key().to_bytes(),
                    organization_binding: [21; 32],
                    service_binding: [33; 32],
                },
            ],
            threshold: 2,
            minimum_organizations: 2,
            maximum_challenge_age_s: 10,
            maximum_post_verification_age_s: 10,
        },
        vec![key_a, key_b, key_c],
    )
}

#[allow(clippy::too_many_arguments)]
fn statement(
    challenge: AuthorityStateChallengeV1,
    key: &SigningKey,
    witness_id: AuthorityStateWitnessId,
    source_sequence: u64,
    source_digest: Digest32,
    state_sequence: u64,
    epoch: AuthorityEpoch,
    facts: Vec<NegativeAuthorityFact>,
    generation: u64,
) -> AuthorityStateStatementV1 {
    let mut statement = AuthorityStateStatementV1 {
        schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
        witness_id,
        challenge_nonce: challenge.nonce,
        grant_digest: challenge.grant_digest,
        state_policy_digest: challenge.state_policy_digest,
        time_policy_digest: challenge.time_policy_digest,
        source_frontier_sequence: source_sequence,
        source_frontier_digest: source_digest,
        state_sequence,
        authority_epoch: epoch,
        negative_facts: facts,
        witness_generation: generation,
        signature: Vec::new(),
    };
    statement.signature = key
        .sign(&statement.canonical_message().unwrap())
        .to_bytes()
        .to_vec();
    statement
}

fn relevant_facts(grant: &CapabilityGrant) -> Vec<NegativeAuthorityFact> {
    vec![
        NegativeAuthorityFact::FreezeResource {
            resource: ResourceRef("host:alpha/service:postgresql".into()),
        },
        NegativeAuthorityFact::TombstonePrincipal {
            principal: grant.audience.clone().unwrap(),
        },
    ]
}

#[test]
fn independent_witnesses_produce_one_indivisible_state_snapshot() {
    let grant = grant();
    let time = authority_time(&grant, 1_000);
    let (policy, keys) = state_policy();
    let pending = PendingAuthorityStateChallenge::new(&policy, &grant, &time).unwrap();
    let challenge = pending.wire();
    let facts = relevant_facts(&grant);

    // Reverse fact ordering in the second witness. Canonical snapshot semantics
    // must agree independently of transport ordering.
    let mut reversed = facts.clone();
    reversed.reverse();
    let verified = verify_authority_state_v1(
        &policy,
        &grant,
        pending,
        &time,
        &[
            statement(
                challenge,
                &keys[0],
                AuthorityStateWitnessId([1; 16]),
                20,
                Digest32([41; 32]),
                9,
                AuthorityEpoch(7),
                facts,
                100,
            ),
            statement(
                challenge,
                &keys[1],
                AuthorityStateWitnessId([2; 16]),
                20,
                Digest32([41; 32]),
                9,
                AuthorityEpoch(7),
                reversed,
                205,
            ),
        ],
    )
    .unwrap();

    assert_eq!(verified.grant_digest(), grant.digest());
    assert_eq!(verified.source_frontier(), (20, Digest32([41; 32])));
    assert_eq!(verified.state_sequence(), 9);
    assert_eq!(verified.authority_epoch(), AuthorityEpoch(7));
    assert_eq!(verified.negative_facts().len(), 2);
    assert_eq!(verified.witness_count(), 2);
    verified.ensure_fresh(&grant, &time).unwrap();
}

#[test]
fn authenticated_revocation_dominates_positive_grant_in_reference_evaluator() {
    let grant = grant();
    let time = authority_time(&grant, 1_000);
    let (policy, keys) = state_policy();
    let pending = PendingAuthorityStateChallenge::new(&policy, &grant, &time).unwrap();
    let challenge = pending.wire();
    let facts = vec![NegativeAuthorityFact::RevokeGrant {
        grant_digest: grant.digest(),
    }];
    let verified = verify_authority_state_v1(
        &policy,
        &grant,
        pending,
        &time,
        &[
            statement(
                challenge,
                &keys[0],
                AuthorityStateWitnessId([1; 16]),
                20,
                Digest32([42; 32]),
                10,
                AuthorityEpoch(7),
                facts.clone(),
                100,
            ),
            statement(
                challenge,
                &keys[1],
                AuthorityStateWitnessId([2; 16]),
                20,
                Digest32([42; 32]),
                10,
                AuthorityEpoch(7),
                facts,
                101,
            ),
        ],
    )
    .unwrap();

    let decision = evaluate_authority(
        &grant,
        AuthorityContext {
            now_unix_s: 1_000,
            current_epoch: verified.authority_epoch(),
            use_state: GrantUseState::default(),
        },
        verified.negative_facts(),
    );
    assert_eq!(
        decision,
        AuthorityDecision::Deny(DenyReason::ExplicitlyRevoked)
    );
}

#[test]
fn newer_current_epoch_is_valid_state_evidence_and_makes_old_grant_stale() {
    let grant = grant();
    let time = authority_time(&grant, 1_000);
    let (policy, keys) = state_policy();
    let pending = PendingAuthorityStateChallenge::new(&policy, &grant, &time).unwrap();
    let challenge = pending.wire();
    let verified = verify_authority_state_v1(
        &policy,
        &grant,
        pending,
        &time,
        &[
            statement(
                challenge,
                &keys[0],
                AuthorityStateWitnessId([1; 16]),
                21,
                Digest32([43; 32]),
                11,
                AuthorityEpoch(8),
                Vec::new(),
                102,
            ),
            statement(
                challenge,
                &keys[1],
                AuthorityStateWitnessId([2; 16]),
                21,
                Digest32([43; 32]),
                11,
                AuthorityEpoch(8),
                Vec::new(),
                103,
            ),
        ],
    )
    .unwrap();

    assert_eq!(verified.authority_epoch(), AuthorityEpoch(8));
    assert_eq!(
        evaluate_authority(
            &grant,
            AuthorityContext {
                now_unix_s: 1_000,
                current_epoch: verified.authority_epoch(),
                use_state: GrantUseState::default(),
            },
            verified.negative_facts(),
        ),
        AuthorityDecision::Deny(DenyReason::EpochStale)
    );
}

#[test]
fn old_signed_state_cannot_cross_a_new_challenge_nonce() {
    let grant = grant();
    let time = authority_time(&grant, 1_000);
    let (policy, keys) = state_policy();
    let first = PendingAuthorityStateChallenge::new(&policy, &grant, &time).unwrap();
    let first_wire = first.wire();
    let old_a = statement(
        first_wire,
        &keys[0],
        AuthorityStateWitnessId([1; 16]),
        20,
        Digest32([44; 32]),
        9,
        AuthorityEpoch(7),
        Vec::new(),
        100,
    );
    let old_b = statement(
        first_wire,
        &keys[1],
        AuthorityStateWitnessId([2; 16]),
        20,
        Digest32([44; 32]),
        9,
        AuthorityEpoch(7),
        Vec::new(),
        101,
    );

    let second = PendingAuthorityStateChallenge::new(&policy, &grant, &time).unwrap();
    assert!(matches!(
        verify_authority_state_v1(&policy, &grant, second, &time, &[old_a, old_b]),
        Err(AuthorityStateError::InvalidStatement)
    ));
}

#[test]
fn source_frontier_or_revocation_disagreement_fails_closed() {
    let grant = grant();
    let time = authority_time(&grant, 1_000);
    let (policy, keys) = state_policy();
    let pending = PendingAuthorityStateChallenge::new(&policy, &grant, &time).unwrap();
    let challenge = pending.wire();

    assert!(matches!(
        verify_authority_state_v1(
            &policy,
            &grant,
            pending,
            &time,
            &[
                statement(
                    challenge,
                    &keys[0],
                    AuthorityStateWitnessId([1; 16]),
                    20,
                    Digest32([45; 32]),
                    9,
                    AuthorityEpoch(7),
                    Vec::new(),
                    100,
                ),
                statement(
                    challenge,
                    &keys[1],
                    AuthorityStateWitnessId([2; 16]),
                    21,
                    Digest32([46; 32]),
                    10,
                    AuthorityEpoch(8),
                    vec![NegativeAuthorityFact::RevokeGrant {
                        grant_digest: grant.digest(),
                    }],
                    101,
                ),
            ],
        ),
        Err(AuthorityStateError::StateDisagreement)
    ));
}

#[test]
fn irrelevant_negative_fact_is_rejected_instead_of_silently_carried() {
    let grant = grant();
    let time = authority_time(&grant, 1_000);
    let (policy, keys) = state_policy();
    let pending = PendingAuthorityStateChallenge::new(&policy, &grant, &time).unwrap();
    let challenge = pending.wire();
    let irrelevant = vec![NegativeAuthorityFact::FreezeResource {
        resource: ResourceRef("host:beta/service:sshd".into()),
    }];

    assert!(matches!(
        verify_authority_state_v1(
            &policy,
            &grant,
            pending,
            &time,
            &[
                statement(
                    challenge,
                    &keys[0],
                    AuthorityStateWitnessId([1; 16]),
                    20,
                    Digest32([47; 32]),
                    9,
                    AuthorityEpoch(7),
                    irrelevant.clone(),
                    100,
                ),
                statement(
                    challenge,
                    &keys[1],
                    AuthorityStateWitnessId([2; 16]),
                    20,
                    Digest32([47; 32]),
                    9,
                    AuthorityEpoch(7),
                    irrelevant,
                    101,
                ),
            ],
        ),
        Err(AuthorityStateError::IrrelevantNegativeFact)
    ));
}

#[test]
fn two_witnesses_from_same_organization_do_not_satisfy_diversity() {
    let grant = grant();
    let time = authority_time(&grant, 1_000);
    let (policy, keys) = state_policy();
    let pending = PendingAuthorityStateChallenge::new(&policy, &grant, &time).unwrap();
    let challenge = pending.wire();

    // Witnesses 1 and 3 deliberately share organization_binding [21; 32].
    assert!(matches!(
        verify_authority_state_v1(
            &policy,
            &grant,
            pending,
            &time,
            &[
                statement(
                    challenge,
                    &keys[0],
                    AuthorityStateWitnessId([1; 16]),
                    20,
                    Digest32([48; 32]),
                    9,
                    AuthorityEpoch(7),
                    Vec::new(),
                    100,
                ),
                statement(
                    challenge,
                    &keys[2],
                    AuthorityStateWitnessId([3; 16]),
                    20,
                    Digest32([48; 32]),
                    9,
                    AuthorityEpoch(7),
                    Vec::new(),
                    102,
                ),
            ],
        ),
        Err(AuthorityStateError::InsufficientDiversity)
    ));
}
