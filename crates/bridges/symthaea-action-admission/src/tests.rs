use super::*;

use std::fmt;
use std::sync::{Arc, Mutex};

use ed25519_dalek::{Signer, SigningKey};
use symthaea_action_checkpoint::{CheckpointHeadV2, GrantAccountCheckpointV2};
use symthaea_action_frontier::{establish_grant_frontier_v2, CheckpointCasStoreV2};
use symthaea_action_runtime::{EffectIntentId, AttemptId, ReservationState};
use symthaea_authority::{
    AuthorityContextRef, AuthorityEpoch, NegativeAuthorityFact, Operation, PrincipalId, PurposeId,
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

fn root_grant() -> CapabilityGrant {
    let mut grant = CapabilityGrant::new(
        "grant-action-admission-v2",
        PrincipalId("issuer".into()),
        PrincipalId("robot-1".into()),
        PurposeId("goal-directed-actuation".into()),
        AuthorityEpoch(7),
        AuthorityContextRef::new("swarm-controller", digest(9)),
    );
    grant.audience = Some(PrincipalId("hal-1".into()));
    grant.resources.insert(ResourceRef("robot-1".into()));
    grant.operations.insert(Operation("move".into()));
    grant.max_uses = 1;
    grant.risk_budget = RiskBudget {
        mutation_units: 1,
        ..RiskBudget::default()
    };
    grant
}

fn binding(grant: &CapabilityGrant) -> EffectAuthorityBindingV2 {
    EffectAuthorityBindingV2 {
        subject: grant.subject.clone(),
        executor: PrincipalId("hal-1".into()),
        purpose: grant.purpose.clone(),
        task: None,
        resource: ResourceRef("robot-1".into()),
        operation: Operation("move".into()),
        plan_digest: None,
        world_digest: None,
        parameters_digest: digest(77),
        risk_charge: RiskBudget {
            mutation_units: 1,
            ..RiskBudget::default()
        },
    }
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
        signed_statement(
            grant,
            wire,
            &key1,
            [3; 16],
            context.clone(),
            negative_facts.clone(),
        ),
        signed_statement(grant, wire, &key2, [4; 16], context, negative_facts),
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

#[test]
fn max_use_one_dispatches_the_owned_reservation_without_second_use_allocation() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0);
    let state = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![],
    );
    let mut account = GrantAccountV2::new_root(&grant).unwrap();
    let (_, mut frontier) =
        establish_grant_frontier_v2(&grant, &account, SharedCasStore::default()).unwrap();

    let authorized = authorize_and_persist_reservation(
        &mut frontier,
        &grant,
        &state,
        &time,
        &mut account,
        EffectIntentId(digest(80)),
        AttemptId(digest(81)),
        binding(&grant),
    )
    .unwrap();

    assert_eq!(authorized.allocation_use_state(), GrantUseState::default());
    assert_eq!(account.authority_use_state().unwrap().charged(), 1);
    let reservation_id = authorized.reservation_id();
    assert_eq!(
        account.reservation(reservation_id).unwrap().state,
        ReservationState::Reserved
    );

    let armed = arm_authorized_reservation(
        &mut frontier,
        &grant,
        &state,
        &time,
        &mut account,
        authorized,
    )
    .unwrap();
    assert_eq!(
        account.reservation(reservation_id).unwrap().state,
        ReservationState::OutcomeUnknown
    );

    let permit = mint_dispatch_permit(&grant, &state, &time, armed).unwrap();
    assert_eq!(permit.grant_digest(), grant.digest());
    assert_eq!(permit.reservation_id(), reservation_id);
    assert_eq!(permit.effect_binding().resource, ResourceRef("robot-1".into()));
    assert_ne!((permit.permit_id().0).0, [0; 32]);
}

#[test]
fn revocation_after_reservation_blocks_arming_before_local_state_transition() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0);
    let initial_state = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![],
    );
    let mut account = GrantAccountV2::new_root(&grant).unwrap();
    let (_, mut frontier) =
        establish_grant_frontier_v2(&grant, &account, SharedCasStore::default()).unwrap();
    let authorized = authorize_and_persist_reservation(
        &mut frontier,
        &grant,
        &initial_state,
        &time,
        &mut account,
        EffectIntentId(digest(82)),
        AttemptId(digest(83)),
        binding(&grant),
    )
    .unwrap();
    let reservation_id = authorized.reservation_id();

    let revoked = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![NegativeAuthorityFact::RevokeGrant {
            grant_digest: grant.digest(),
        }],
    );
    let error = arm_authorized_reservation(
        &mut frontier,
        &grant,
        &revoked,
        &time,
        &mut account,
        authorized,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        AdmissionFrontierV2Error::Admission(AdmissionV2Error::AuthorityDenied(
            DenyReason::ExplicitlyRevoked
        ))
    ));
    assert_eq!(
        account.reservation(reservation_id).unwrap().state,
        ReservationState::Reserved
    );
}

#[test]
fn absent_current_authority_context_cannot_allocate_or_persist_a_use() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0);
    let no_context = verified_state(&grant, &time, None, vec![]);
    let mut account = GrantAccountV2::new_root(&grant).unwrap();
    let (_, mut frontier) =
        establish_grant_frontier_v2(&grant, &account, SharedCasStore::default()).unwrap();
    let genesis_head = frontier.current_head();

    let error = authorize_and_persist_reservation(
        &mut frontier,
        &grant,
        &no_context,
        &time,
        &mut account,
        EffectIntentId(digest(84)),
        AttemptId(digest(85)),
        binding(&grant),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        AdmissionFrontierV2Error::Admission(AdmissionV2Error::NoCurrentAuthorityContext)
    ));
    assert_eq!(account.authority_use_state().unwrap(), GrantUseState::default());
    assert_eq!(frontier.current_head(), genesis_head);
}

#[test]
fn semantic_effect_mismatch_fails_before_reservation() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0);
    let state = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![],
    );
    let mut account = GrantAccountV2::new_root(&grant).unwrap();
    let (_, mut frontier) =
        establish_grant_frontier_v2(&grant, &account, SharedCasStore::default()).unwrap();
    let mut wrong = binding(&grant);
    wrong.operation = Operation("delete".into());

    let error = authorize_and_persist_reservation(
        &mut frontier,
        &grant,
        &state,
        &time,
        &mut account,
        EffectIntentId(digest(86)),
        AttemptId(digest(87)),
        wrong,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        AdmissionFrontierV2Error::Admission(AdmissionV2Error::EffectBindingMismatch(
            "operation"
        ))
    ));
    assert_eq!(account.authority_use_state().unwrap(), GrantUseState::default());
}
