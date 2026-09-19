use super::*;
use ed25519_dalek::{Signer, SigningKey};
use symthaea_authority::{
    evaluate_authority, AuthorityContextRef, AuthorityDecision, AuthorityEpoch, DenyReason,
    Operation, PrincipalId, PurposeId, ResourceRef,
};
use symthaea_authority_time::{
    verify_authority_time_v1, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
    TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1, AUTHORITY_TIME_SCHEMA_VERSION,
};

fn digest(byte: u8) -> Digest32 {
    Digest32([byte; 32])
}

fn root_grant() -> CapabilityGrant {
    let mut grant = CapabilityGrant::new(
        "grant-1",
        PrincipalId("issuer".into()),
        PrincipalId("robot-1".into()),
        PurposeId("goal-directed-actuation".into()),
        AuthorityEpoch(7),
        AuthorityContextRef::new("swarm-controller", digest(9)),
    );
    grant.resources.insert(ResourceRef("robot-1".into()));
    grant.operations.insert(Operation("move".into()));
    grant
}

fn verified_time(subject: [u8; 32]) -> VerifiedAuthorityTime {
    let key1 = SigningKey::from_bytes(&[1u8; 32]);
    let key2 = SigningKey::from_bytes(&[2u8; 32]);
    let policy = TrustedTimePolicyV1 {
        schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
        policy_id: [7; 16],
        authorities: vec![
            TrustedTimeAuthorityV1 {
                authority_id: TimeAuthorityId([1; 16]),
                verifying_key: key1.verifying_key().to_bytes(),
                organization_binding: [11; 32],
                service_binding: [21; 32],
            },
            TrustedTimeAuthorityV1 {
                authority_id: TimeAuthorityId([2; 16]),
                verifying_key: key2.verifying_key().to_bytes(),
                organization_binding: [12; 32],
                service_binding: [22; 32],
            },
        ],
        threshold: 2,
        minimum_organizations: 2,
        maximum_uncertainty_s: 2,
        maximum_challenge_age_ns: 60_000_000_000,
        maximum_post_verification_age_ns: 60_000_000_000,
    };
    let challenge = PendingAuthorityTimeChallenge::new(&policy, subject).unwrap();
    let wire = challenge.wire();

    let mut first = AuthorityTimeStatementV1 {
        schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
        authority_id: TimeAuthorityId([1; 16]),
        policy_digest: wire.policy_digest,
        subject_digest: wire.subject_digest,
        challenge_nonce: wire.nonce,
        witnessed_unix_s: 1_800_000_000,
        uncertainty_s: 1,
        signature: Vec::new(),
    };
    first.signature = key1
        .sign(&first.canonical_message().unwrap())
        .to_bytes()
        .to_vec();

    let mut second = AuthorityTimeStatementV1 {
        authority_id: TimeAuthorityId([2; 16]),
        ..first.clone()
    };
    second.signature = key2
        .sign(&second.canonical_message().unwrap())
        .to_bytes()
        .to_vec();

    verify_authority_time_v1(&policy, challenge, &[first, second]).unwrap()
}

fn state_policy() -> (AuthorityStatePolicyV2, SigningKey, SigningKey) {
    let key1 = SigningKey::from_bytes(&[3u8; 32]);
    let key2 = SigningKey::from_bytes(&[4u8; 32]);
    (
        AuthorityStatePolicyV2 {
            schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
            policy_id: [9; 16],
            witnesses: vec![
                TrustedAuthorityStateWitnessV2 {
                    witness_id: AuthorityStateWitnessId([3; 16]),
                    verifying_key: key1.verifying_key().to_bytes(),
                    organization_binding: [31; 32],
                    service_binding: [41; 32],
                },
                TrustedAuthorityStateWitnessV2 {
                    witness_id: AuthorityStateWitnessId([4; 16]),
                    verifying_key: key2.verifying_key().to_bytes(),
                    organization_binding: [32; 32],
                    service_binding: [42; 32],
                },
            ],
            threshold: 2,
            minimum_organizations: 2,
            maximum_challenge_age_s: 60,
            maximum_post_verification_age_s: 60,
        },
        key1,
        key2,
    )
}

fn signed_statement(
    grant: &CapabilityGrant,
    wire: AuthorityStateChallengeV2,
    key: &SigningKey,
    witness_id: [u8; 16],
    context: Option<AuthorityContextRef>,
    negative_facts: Vec<NegativeAuthorityFact>,
) -> AuthorityStateStatementV2 {
    let mut statement = AuthorityStateStatementV2 {
        schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
        witness_id: AuthorityStateWitnessId(witness_id),
        challenge_nonce: wire.nonce,
        grant_digest: grant.digest(),
        state_policy_digest: wire.state_policy_digest,
        time_policy_digest: wire.time_policy_digest,
        source_frontier_sequence: 11,
        source_frontier_digest: digest(55),
        state_sequence: 12,
        authority_epoch: AuthorityEpoch(7),
        authority_context: context,
        negative_facts,
        witness_generation: 1,
        signature: Vec::new(),
    };
    statement.signature = key
        .sign(&statement.canonical_message().unwrap())
        .to_bytes()
        .to_vec();
    statement
}

#[test]
fn snapshot_identity_changes_with_current_context() {
    let grant = root_grant();
    let context_a = AuthorityContextRef::new("swarm-controller", digest(2));
    let context_b = AuthorityContextRef::new("swarm-controller", digest(3));
    let a = snapshot_digest_v2(
        grant.digest(),
        1,
        digest(1),
        1,
        AuthorityEpoch(7),
        Some(&context_a),
        &[],
    )
    .unwrap();
    let b = snapshot_digest_v2(
        grant.digest(),
        1,
        digest(1),
        1,
        AuthorityEpoch(7),
        Some(&context_b),
        &[],
    )
    .unwrap();
    assert_ne!(a, b);
}

#[test]
fn snapshot_identity_distinguishes_absent_context() {
    let grant = root_grant();
    let context = AuthorityContextRef::new("swarm-controller", digest(2));
    let present = snapshot_digest_v2(
        grant.digest(),
        1,
        digest(1),
        1,
        AuthorityEpoch(7),
        Some(&context),
        &[],
    )
    .unwrap();
    let absent = snapshot_digest_v2(
        grant.digest(),
        1,
        digest(1),
        1,
        AuthorityEpoch(7),
        None,
        &[],
    )
    .unwrap();
    assert_ne!(present, absent);
}

#[test]
fn matching_verified_state_feeds_pure_evaluator() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0);
    let (policy, key1, key2) = state_policy();
    let challenge = PendingAuthorityStateChallengeV2::new(&policy, &grant, &time).unwrap();
    let wire = challenge.wire();
    let context = grant.authority_context.clone();
    let statements = vec![
        signed_statement(
            &grant,
            wire,
            &key1,
            [3; 16],
            Some(context.clone()),
            vec![],
        ),
        signed_statement(&grant, wire, &key2, [4; 16], Some(context), vec![]),
    ];

    let state = verify_authority_state_v2(&policy, &grant, challenge, &time, &statements).unwrap();
    let input = state
        .evaluation_input(&grant, &time, GrantUseState::default())
        .unwrap();
    assert_eq!(
        evaluate_authority(&grant, &input, state.negative_facts()),
        AuthorityDecision::Allow
    );
}

#[test]
fn verified_context_rotation_denies_old_grant() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0);
    let (policy, key1, key2) = state_policy();
    let challenge = PendingAuthorityStateChallengeV2::new(&policy, &grant, &time).unwrap();
    let wire = challenge.wire();
    let rotated = AuthorityContextRef::new("swarm-controller", digest(99));
    let statements = vec![
        signed_statement(
            &grant,
            wire,
            &key1,
            [3; 16],
            Some(rotated.clone()),
            vec![],
        ),
        signed_statement(&grant, wire, &key2, [4; 16], Some(rotated), vec![]),
    ];

    let state = verify_authority_state_v2(&policy, &grant, challenge, &time, &statements).unwrap();
    let input = state
        .evaluation_input(&grant, &time, GrantUseState::default())
        .unwrap();
    assert_eq!(
        evaluate_authority(&grant, &input, state.negative_facts()),
        AuthorityDecision::Deny(DenyReason::ContextMismatch)
    );
}

#[test]
fn verified_absence_of_active_context_fails_closed() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0);
    let (policy, key1, key2) = state_policy();
    let challenge = PendingAuthorityStateChallengeV2::new(&policy, &grant, &time).unwrap();
    let wire = challenge.wire();
    let statements = vec![
        signed_statement(&grant, wire, &key1, [3; 16], None, vec![]),
        signed_statement(&grant, wire, &key2, [4; 16], None, vec![]),
    ];

    let state = verify_authority_state_v2(&policy, &grant, challenge, &time, &statements).unwrap();
    assert!(state.authority_context().is_none());
    assert!(matches!(
        state.evaluation_input(&grant, &time, GrantUseState::default()),
        Err(AuthorityStateError::NoCurrentAuthorityContext)
    ));
}

#[test]
fn witnesses_must_agree_on_current_context_state() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0);
    let (policy, key1, key2) = state_policy();
    let challenge = PendingAuthorityStateChallengeV2::new(&policy, &grant, &time).unwrap();
    let wire = challenge.wire();
    let statements = vec![
        signed_statement(
            &grant,
            wire,
            &key1,
            [3; 16],
            Some(AuthorityContextRef::new("swarm-controller", digest(9))),
            vec![],
        ),
        signed_statement(&grant, wire, &key2, [4; 16], None, vec![]),
    ];

    assert!(matches!(
        verify_authority_state_v2(&policy, &grant, challenge, &time, &statements),
        Err(AuthorityStateError::StateDisagreement)
    ));
}

#[test]
fn verified_context_revocation_stays_negative_authority() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0);
    let (policy, key1, key2) = state_policy();
    let challenge = PendingAuthorityStateChallengeV2::new(&policy, &grant, &time).unwrap();
    let wire = challenge.wire();
    let fact = NegativeAuthorityFact::RevokeContext {
        context: grant.authority_context.clone(),
    };
    let statements = vec![
        signed_statement(
            &grant,
            wire,
            &key1,
            [3; 16],
            Some(grant.authority_context.clone()),
            vec![fact.clone()],
        ),
        signed_statement(
            &grant,
            wire,
            &key2,
            [4; 16],
            Some(grant.authority_context.clone()),
            vec![fact],
        ),
    ];

    let state = verify_authority_state_v2(&policy, &grant, challenge, &time, &statements).unwrap();
    let input = state
        .evaluation_input(&grant, &time, GrantUseState::default())
        .unwrap();
    assert_eq!(
        evaluate_authority(&grant, &input, state.negative_facts()),
        AuthorityDecision::Deny(DenyReason::ContextRevoked)
    );
}

#[test]
fn v2_negative_fact_digest_is_context_sensitive() {
    let a = negative_fact_digest_v2(&NegativeAuthorityFact::RevokeContext {
        context: AuthorityContextRef::new("swarm-controller", digest(1)),
    })
    .unwrap();
    let b = negative_fact_digest_v2(&NegativeAuthorityFact::RevokeContext {
        context: AuthorityContextRef::new("swarm-controller", digest(2)),
    })
    .unwrap();
    assert_ne!(a, b);
}
