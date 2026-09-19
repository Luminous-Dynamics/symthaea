use std::fmt;
use std::sync::{Arc, Mutex};

use super::*;
use ed25519_dalek::{Signer, SigningKey};
use symthaea_action_checkpoint::{CheckpointHeadV2, GrantAccountCheckpointV2};
use symthaea_action_frontier::{establish_grant_frontier_v2, CheckpointCasStoreV2};
use symthaea_action_runtime::{
    EffectBindingDigest, EffectIntentId, GrantAccountV2, ReservationState,
};
use symthaea_authority::{
    evaluate_authority, AuthorityEvaluationInput, GrantUseState, Operation, PrincipalId, PurposeId,
    ResourceRef,
};
use symthaea_authority_state::{
    verify_authority_state_v2, AuthorityStateChallengeV2, AuthorityStatePolicyV2,
    AuthorityStateStatementV2, AuthorityStateWitnessId, PendingAuthorityStateChallengeV2,
    TrustedAuthorityStateWitnessV2, AUTHORITY_STATE_SCHEMA_VERSION,
};
use symthaea_authority_time::{
    verify_authority_time_v1, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
    TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1, AUTHORITY_TIME_SCHEMA_VERSION,
};

fn digest(byte: u8) -> Digest32 {
    Digest32([byte; 32])
}

fn risk(units: u64) -> RiskBudget {
    RiskBudget {
        mutation_units: units,
        ..RiskBudget::default()
    }
}

fn root_grant(max_uses: u32) -> CapabilityGrant {
    let mut grant = CapabilityGrant::new(
        "reserved-use-v2",
        PrincipalId("issuer".into()),
        PrincipalId("robot-1".into()),
        PurposeId("goal-directed-actuation".into()),
        AuthorityEpoch(7),
        AuthorityContextRef::new("swarm-controller", digest(9)),
    );
    grant.resources.insert(ResourceRef("robot-1".into()));
    grant.operations.insert(Operation("move".into()));
    grant.max_uses = max_uses;
    grant.risk_budget = risk(u64::from(max_uses));
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

fn signed_state_statement(
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
        authority_epoch: grant.authority_epoch,
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

fn verified_state(
    grant: &CapabilityGrant,
    time: &VerifiedAuthorityTime,
    context: Option<AuthorityContextRef>,
    negative_facts: Vec<NegativeAuthorityFact>,
) -> VerifiedAuthorityStateV2 {
    let (policy, key1, key2) = state_policy();
    let challenge = PendingAuthorityStateChallengeV2::new(&policy, grant, time).unwrap();
    let wire = challenge.wire();
    let statements = vec![
        signed_state_statement(
            grant,
            wire,
            &key1,
            [3; 16],
            context.clone(),
            negative_facts.clone(),
        ),
        signed_state_statement(grant, wire, &key2, [4; 16], context, negative_facts),
    ];
    verify_authority_state_v2(&policy, grant, challenge, time, &statements).unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CasConflict;

impl fmt::Display for CasConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CAS conflict")
    }
}

impl std::error::Error for CasConflict {}

#[derive(Clone, Default)]
struct SharedCasStore {
    state: Arc<Mutex<Option<CheckpointHeadV2>>>,
}

impl CheckpointCasStoreV2 for SharedCasStore {
    type Error = CasConflict;

    fn current_head(&mut self) -> Result<Option<CheckpointHeadV2>, Self::Error> {
        self.state.lock().map(|state| *state).map_err(|_| CasConflict)
    }

    fn compare_and_swap(
        &mut self,
        expected_previous: Option<CheckpointHeadV2>,
        checkpoint: &GrantAccountCheckpointV2,
    ) -> Result<CheckpointHeadV2, Self::Error> {
        let mut state = self.state.lock().map_err(|_| CasConflict)?;
        if *state != expected_previous {
            return Err(CasConflict);
        }
        let next = checkpoint.head().map_err(|_| CasConflict)?;
        *state = Some(next);
        Ok(next)
    }
}

fn reserve_one(
    grant: &CapabilityGrant,
) -> (
    GrantAccountV2,
    CasFrontierV2<SharedCasStore>,
    PersistedReservationV2,
) {
    let mut account = GrantAccountV2::new_root(grant).unwrap();
    let (_, mut frontier) =
        establish_grant_frontier_v2(grant, &account, SharedCasStore::default()).unwrap();
    account
        .reserve_execution(
            EffectIntentId(digest(61)),
            AttemptId(digest(62)),
            EffectBindingDigest(digest(63)),
            risk(1),
        )
        .unwrap();
    let reserved =
        GrantAccountCheckpointV2::successor(frontier.expected_checkpoint(), grant, &account)
            .unwrap();
    let persisted = frontier.persist_new_reservation(reserved).unwrap();
    (account, frontier, persisted)
}

fn core_input(
    grant: &CapabilityGrant,
    now_unix_s: u64,
    context: AuthorityContextRef,
) -> AuthorityEvaluationInput {
    AuthorityEvaluationInput {
        now_unix_s,
        current_epoch: grant.authority_epoch,
        current_authority_context: context,
        use_state: GrantUseState::default(),
    }
}

#[test]
fn nonbudget_predicate_matches_core_when_use_is_available() {
    let grant = root_grant(2);
    let now = 1_800_000_000;
    let context = grant.authority_context.clone();

    let cases = vec![
        vec![],
        vec![NegativeAuthorityFact::RevokeGrant {
            grant_digest: grant.digest(),
        }],
        vec![NegativeAuthorityFact::RevokeContext {
            context: context.clone(),
        }],
        vec![NegativeAuthorityFact::TombstonePrincipal {
            principal: grant.subject.clone(),
        }],
        vec![NegativeAuthorityFact::FreezeResource {
            resource: ResourceRef("robot-1".into()),
        }],
        vec![NegativeAuthorityFact::MinimumResourceEpoch {
            resource: ResourceRef("robot-1".into()),
            minimum_epoch: AuthorityEpoch(grant.authority_epoch.0 + 1),
        }],
    ];

    for facts in cases {
        assert_eq!(
            evaluate_reserved_use_nonbudget(
                &grant,
                now,
                grant.authority_epoch,
                &context,
                &facts,
            ),
            evaluate_authority(&grant, &core_input(&grant, now, context.clone()), &facts)
        );
    }

    let rotated = AuthorityContextRef::new("swarm-controller", digest(99));
    assert_eq!(
        evaluate_reserved_use_nonbudget(&grant, now, grant.authority_epoch, &rotated, &[]),
        evaluate_authority(&grant, &core_input(&grant, now, rotated), &[])
    );

    assert_eq!(
        evaluate_reserved_use_nonbudget(
            &grant,
            now,
            AuthorityEpoch(grant.authority_epoch.0 + 1),
            &context,
            &[],
        ),
        AuthorityDecision::Deny(DenyReason::EpochStale)
    );

    let mut expired = grant.clone();
    expired.expires_at_unix_s = Some(now - 1);
    assert_eq!(
        evaluate_reserved_use_nonbudget(
            &expired,
            now,
            expired.authority_epoch,
            &expired.authority_context,
            &[],
        ),
        evaluate_authority(
            &expired,
            &core_input(&expired, now, expired.authority_context.clone()),
            &[],
        )
    );

    let mut delegated = grant.clone();
    delegated.parent_digest = Some(digest(77));
    assert_eq!(
        evaluate_reserved_use_nonbudget(
            &delegated,
            now,
            delegated.authority_epoch,
            &delegated.authority_context,
            &[],
        ),
        AuthorityDecision::Deny(DenyReason::DelegationChainRequired)
    );
}

#[test]
fn one_use_reservation_is_not_mistaken_for_a_second_use() {
    let grant = root_grant(1);
    let (account, _frontier, _persisted) = reserve_one(&grant);
    let input = AuthorityEvaluationInput {
        now_unix_s: 1_800_000_000,
        current_epoch: grant.authority_epoch,
        current_authority_context: grant.authority_context.clone(),
        use_state: account.authority_use_state().unwrap(),
    };
    assert_eq!(
        evaluate_authority(&grant, &input, &[]),
        AuthorityDecision::Deny(DenyReason::UseBudgetExhausted)
    );
    assert_eq!(
        evaluate_reserved_use_nonbudget(
            &grant,
            input.now_unix_s,
            input.current_epoch,
            &input.current_authority_context,
            &[],
        ),
        AuthorityDecision::Allow
    );
}

#[test]
fn real_verified_state_binds_to_exact_persisted_reservation() {
    let grant = root_grant(1);
    let (_account, frontier, persisted) = reserve_one(&grant);
    let time = verified_time(grant.digest().0);
    let state = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![],
    );

    let currentness =
        verify_reserved_use_currentness_v2(&frontier, &grant, persisted, state, &time).unwrap();
    assert_eq!(currentness.grant_digest(), grant.digest());
    assert_eq!(currentness.persisted_head(), frontier.expected_head());
    assert_eq!(currentness.authority_state_sequence(), 12);
    assert_eq!(currentness.authority_source_frontier(), (11, digest(55)));
}

#[test]
fn context_rotation_denies_reserved_use() {
    let grant = root_grant(1);
    let (_account, frontier, persisted) = reserve_one(&grant);
    let time = verified_time(grant.digest().0);
    let rotated = AuthorityContextRef::new("swarm-controller", digest(99));
    let state = verified_state(&grant, &time, Some(rotated), vec![]);

    assert!(matches!(
        verify_reserved_use_currentness_v2(&frontier, &grant, persisted, state, &time),
        Err(ReservedUseCurrentnessError::Denied(DenyReason::ContextMismatch))
    ));
}

#[test]
fn verified_revocation_denies_reserved_use() {
    let grant = root_grant(1);
    let (_account, frontier, persisted) = reserve_one(&grant);
    let time = verified_time(grant.digest().0);
    let state = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![NegativeAuthorityFact::RevokeGrant {
            grant_digest: grant.digest(),
        }],
    );

    assert!(matches!(
        verify_reserved_use_currentness_v2(&frontier, &grant, persisted, state, &time),
        Err(ReservedUseCurrentnessError::Denied(DenyReason::ExplicitlyRevoked))
    ));
}

#[test]
fn absent_current_context_fails_closed() {
    let grant = root_grant(1);
    let (_account, frontier, persisted) = reserve_one(&grant);
    let time = verified_time(grant.digest().0);
    let state = verified_state(&grant, &time, None, vec![]);

    assert!(matches!(
        verify_reserved_use_currentness_v2(&frontier, &grant, persisted, state, &time),
        Err(ReservedUseCurrentnessError::NoCurrentAuthorityContext)
    ));
}

#[test]
fn currentness_must_cross_durable_arming_before_stronger_object_exists() {
    let grant = root_grant(1);
    let (mut account, mut frontier, persisted) = reserve_one(&grant);
    let time = verified_time(grant.digest().0);
    let state = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![],
    );
    let currentness =
        verify_reserved_use_currentness_v2(&frontier, &grant, persisted, state, &time).unwrap();
    let reservation = currentness.reservation_id();

    account.mark_outcome_unknown(reservation).unwrap();
    let successor =
        GrantAccountCheckpointV2::successor(frontier.expected_checkpoint(), &grant, &account)
            .unwrap();
    let armed =
        arm_current_reserved_use_v2(&mut frontier, &grant, currentness, &time, successor).unwrap();

    assert_eq!(armed.reservation_id(), reservation);
    assert_eq!(armed.armed_head(), frontier.expected_head());
    assert_ne!(armed.reserved_head(), armed.armed_head());
    assert_eq!(
        frontier
            .expected_checkpoint()
            .snapshot()
            .reservations
            .get(&reservation)
            .unwrap()
            .state,
        ReservationState::OutcomeUnknown
    );
}

#[test]
fn frontier_advance_after_currentness_blocks_durable_arming() {
    let grant = root_grant(2);
    let (mut account, mut frontier, persisted) = reserve_one(&grant);
    let time = verified_time(grant.digest().0);
    let state = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![],
    );
    let currentness =
        verify_reserved_use_currentness_v2(&frontier, &grant, persisted, state, &time).unwrap();
    let reservation = currentness.reservation_id();

    let no_op =
        GrantAccountCheckpointV2::successor(frontier.expected_checkpoint(), &grant, &account)
            .unwrap();
    frontier.persist_successor(no_op).unwrap();

    account.mark_outcome_unknown(reservation).unwrap();
    let successor =
        GrantAccountCheckpointV2::successor(frontier.expected_checkpoint(), &grant, &account)
            .unwrap();

    assert!(matches!(
        arm_current_reserved_use_v2(&mut frontier, &grant, currentness, &time, successor),
        Err(ArmCurrentReservedUseError::Currentness(
            ReservedUseCurrentnessError::PersistedHeadMismatch
        ))
    ));
}
