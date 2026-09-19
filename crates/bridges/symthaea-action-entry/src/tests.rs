use super::*;

use std::fmt;
use std::sync::{Arc, Mutex};

use ed25519_dalek::{Signer, SigningKey};
use symthaea_action_admission::{
    arm_authorized_reservation, authorize_and_persist_reservation, mint_dispatch_permit,
    EffectAuthorityBindingV2,
};
use symthaea_action_checkpoint::{CheckpointHeadV2, GrantAccountCheckpointV2};
use symthaea_action_frontier::{establish_grant_frontier_v2, CheckpointCasStoreV2};
use symthaea_action_runtime::{AttemptId, EffectIntentId, GrantAccountV2, ReservationState};
use symthaea_authority::{
    AuthorityContextRef, AuthorityEpoch, NegativeAuthorityFact, Operation, PurposeId, ResourceRef,
    RiskBudget,
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
        "grant-action-entry-v2",
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

fn verified_time(subject: [u8; 32], policy_byte: u8) -> VerifiedAuthorityTime {
    let key1 = SigningKey::from_bytes(&[1u8; 32]);
    let key2 = SigningKey::from_bytes(&[2u8; 32]);
    let policy = TrustedTimePolicyV1 {
        schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
        policy_id: [policy_byte; 16],
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

fn state_policy(policy_byte: u8) -> (AuthorityStatePolicyV2, SigningKey, SigningKey) {
    let key1 = SigningKey::from_bytes(&[3u8; 32]);
    let key2 = SigningKey::from_bytes(&[4u8; 32]);
    (
        AuthorityStatePolicyV2 {
            schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
            policy_id: [policy_byte; 16],
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

#[allow(clippy::too_many_arguments)]
fn signed_statement(
    grant: &CapabilityGrant,
    wire: AuthorityStateChallengeV2,
    key: &SigningKey,
    witness_id: [u8; 16],
    context: Option<AuthorityContextRef>,
    negative_facts: Vec<NegativeAuthorityFact>,
    source_frontier_sequence: u64,
    source_frontier_digest: Digest32,
    state_sequence: u64,
) -> AuthorityStateStatementV2 {
    let mut statement = AuthorityStateStatementV2 {
        schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
        witness_id: AuthorityStateWitnessId(witness_id),
        challenge_nonce: wire.nonce,
        grant_digest: grant.digest(),
        state_policy_digest: wire.state_policy_digest,
        time_policy_digest: wire.time_policy_digest,
        source_frontier_sequence,
        source_frontier_digest,
        state_sequence,
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

#[allow(clippy::too_many_arguments)]
fn verified_state(
    grant: &CapabilityGrant,
    time: &VerifiedAuthorityTime,
    context: Option<AuthorityContextRef>,
    negative_facts: Vec<NegativeAuthorityFact>,
    policy_byte: u8,
    source_frontier_sequence: u64,
    source_frontier_digest: Digest32,
    state_sequence: u64,
) -> VerifiedAuthorityStateV2 {
    let (policy, key1, key2) = state_policy(policy_byte);
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
            source_frontier_sequence,
            source_frontier_digest,
            state_sequence,
        ),
        signed_statement(
            grant,
            wire,
            &key2,
            [4; 16],
            context,
            negative_facts,
            source_frontier_sequence,
            source_frontier_digest,
            state_sequence,
        ),
    ];
    verify_authority_state_v2(&policy, grant, challenge, time, &statements).unwrap()
}

fn initial_state(grant: &CapabilityGrant, time: &VerifiedAuthorityTime) -> VerifiedAuthorityStateV2 {
    verified_state(
        grant,
        time,
        Some(grant.authority_context.clone()),
        vec![],
        9,
        11,
        digest(55),
        12,
    )
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

fn mint_permit(
    grant: &CapabilityGrant,
    time: &VerifiedAuthorityTime,
    state: &VerifiedAuthorityStateV2,
) -> (DispatchPermitV2, GrantAccountV2, ReservationId) {
    let mut account = GrantAccountV2::new_root(grant).unwrap();
    let (_, mut frontier) =
        establish_grant_frontier_v2(grant, &account, SharedCasStore::default()).unwrap();
    let authorized = authorize_and_persist_reservation(
        &mut frontier,
        grant,
        state,
        time,
        &mut account,
        EffectIntentId(digest(80)),
        AttemptId(digest(81)),
        binding(grant),
    )
    .unwrap();
    let reservation_id = authorized.reservation_id();
    let armed = arm_authorized_reservation(
        &mut frontier,
        grant,
        state,
        time,
        &mut account,
        authorized,
    )
    .unwrap();
    let permit = mint_dispatch_permit(grant, state, time, armed).unwrap();
    (permit, account, reservation_id)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExecutorFailure;

impl fmt::Display for ExecutorFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("executor failure")
    }
}

impl std::error::Error for ExecutorFailure {}

struct CountingExecutor {
    principal: PrincipalId,
    calls: usize,
    fail: bool,
}

impl CountingExecutor {
    fn good() -> Self {
        Self {
            principal: PrincipalId("hal-1".into()),
            calls: 0,
            fail: false,
        }
    }
}

impl BoundEffectExecutorV2 for CountingExecutor {
    type Output = DispatchPermitId;
    type Error = ExecutorFailure;

    fn executor_principal(&self) -> &PrincipalId {
        &self.principal
    }

    fn enter_effect(
        &mut self,
        context: &EffectEntryContextV2<'_>,
    ) -> Result<Self::Output, Self::Error> {
        self.calls += 1;
        if self.fail {
            Err(ExecutorFailure)
        } else {
            Ok(context.permit_id())
        }
    }
}

#[test]
fn unchanged_fresh_state_invokes_bound_executor_once_for_max_use_one() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0, 7);
    let state = initial_state(&grant, &time);
    let (permit, account, reservation_id) = mint_permit(&grant, &time, &state);
    assert_eq!(account.authority_use_state().unwrap().charged(), 1);
    assert_eq!(
        account.reservation(reservation_id).unwrap().state,
        ReservationState::OutcomeUnknown
    );

    let mut executor = CountingExecutor::good();
    let permit_id = dispatch_once_v2(&mut executor, &grant, &state, &time, permit).unwrap();
    assert_eq!(executor.calls, 1);
    assert_ne!((permit_id.0).0, [0; 32]);
}

#[test]
fn revocation_after_mint_blocks_executor_entry() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0, 7);
    let mint_state = initial_state(&grant, &time);
    let (permit, _, _) = mint_permit(&grant, &time, &mint_state);
    let revoked = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![NegativeAuthorityFact::RevokeGrant {
            grant_digest: grant.digest(),
        }],
        9,
        12,
        digest(56),
        13,
    );

    let mut executor = CountingExecutor::good();
    let error = dispatch_once_v2(&mut executor, &grant, &revoked, &time, permit).unwrap_err();
    assert!(matches!(
        error,
        EffectEntryV2Error::Admission(EntryAdmissionV2Error::AuthorityDenied(
            DenyReason::ExplicitlyRevoked
        ))
    ));
    assert_eq!(executor.calls, 0);
}

#[test]
fn rotated_context_after_mint_blocks_executor_entry() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0, 7);
    let mint_state = initial_state(&grant, &time);
    let (permit, _, _) = mint_permit(&grant, &time, &mint_state);
    let rotated = verified_state(
        &grant,
        &time,
        Some(AuthorityContextRef::new("swarm-controller", digest(99))),
        vec![],
        9,
        12,
        digest(56),
        13,
    );

    let mut executor = CountingExecutor::good();
    let error = dispatch_once_v2(&mut executor, &grant, &rotated, &time, permit).unwrap_err();
    assert!(matches!(
        error,
        EffectEntryV2Error::Admission(EntryAdmissionV2Error::AuthorityDenied(
            DenyReason::ContextMismatch
        ))
    ));
    assert_eq!(executor.calls, 0);
}

#[test]
fn wrong_executor_principal_is_rejected_before_call() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0, 7);
    let state = initial_state(&grant, &time);
    let (permit, _, _) = mint_permit(&grant, &time, &state);
    let mut executor = CountingExecutor {
        principal: PrincipalId("other-hal".into()),
        calls: 0,
        fail: false,
    };

    let error = dispatch_once_v2(&mut executor, &grant, &state, &time, permit).unwrap_err();
    assert!(matches!(
        error,
        EffectEntryV2Error::Admission(EntryAdmissionV2Error::ExecutorPrincipalMismatch)
    ));
    assert_eq!(executor.calls, 0);
}

#[test]
fn freshly_signed_older_state_is_still_rollback() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0, 7);
    let mint_state = initial_state(&grant, &time);
    let (permit, _, _) = mint_permit(&grant, &time, &mint_state);
    let rollback = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![],
        9,
        10,
        digest(54),
        11,
    );

    let mut executor = CountingExecutor::good();
    let error = dispatch_once_v2(&mut executor, &grant, &rollback, &time, permit).unwrap_err();
    assert!(matches!(
        error,
        EffectEntryV2Error::Admission(EntryAdmissionV2Error::StateSequenceRollback { .. })
    ));
    assert_eq!(executor.calls, 0);
}

#[test]
fn same_sequence_conflicting_snapshot_fails_closed() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0, 7);
    let mint_state = initial_state(&grant, &time);
    let (permit, _, _) = mint_permit(&grant, &time, &mint_state);
    let contradiction = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![],
        9,
        11,
        digest(56),
        12,
    );

    let mut executor = CountingExecutor::good();
    let error = dispatch_once_v2(&mut executor, &grant, &contradiction, &time, permit).unwrap_err();
    assert!(matches!(
        error,
        EffectEntryV2Error::Admission(EntryAdmissionV2Error::StateSequenceContradiction)
    ));
    assert_eq!(executor.calls, 0);
}

#[test]
fn newer_state_with_older_source_frontier_is_rejected() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0, 7);
    let mint_state = initial_state(&grant, &time);
    let (permit, _, _) = mint_permit(&grant, &time, &mint_state);
    let rollback = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![],
        9,
        10,
        digest(54),
        13,
    );

    let mut executor = CountingExecutor::good();
    let error = dispatch_once_v2(&mut executor, &grant, &rollback, &time, permit).unwrap_err();
    assert!(matches!(
        error,
        EffectEntryV2Error::Admission(EntryAdmissionV2Error::SourceFrontierRollback { .. })
    ));
    assert_eq!(executor.calls, 0);
}

#[test]
fn newer_state_with_same_frontier_sequence_conflicting_digest_is_rejected() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0, 7);
    let mint_state = initial_state(&grant, &time);
    let (permit, _, _) = mint_permit(&grant, &time, &mint_state);
    let contradiction = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![],
        9,
        11,
        digest(56),
        13,
    );

    let mut executor = CountingExecutor::good();
    let error = dispatch_once_v2(&mut executor, &grant, &contradiction, &time, permit).unwrap_err();
    assert!(matches!(
        error,
        EffectEntryV2Error::Admission(EntryAdmissionV2Error::SourceFrontierContradiction)
    ));
    assert_eq!(executor.calls, 0);
}

#[test]
fn changed_state_policy_cannot_reauthorize_old_permit() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0, 7);
    let mint_state = initial_state(&grant, &time);
    let (permit, _, _) = mint_permit(&grant, &time, &mint_state);
    let changed_policy = verified_state(
        &grant,
        &time,
        Some(grant.authority_context.clone()),
        vec![],
        8,
        12,
        digest(56),
        13,
    );

    let mut executor = CountingExecutor::good();
    let error = dispatch_once_v2(&mut executor, &grant, &changed_policy, &time, permit).unwrap_err();
    assert!(matches!(
        error,
        EffectEntryV2Error::Admission(EntryAdmissionV2Error::StatePolicyChanged)
    ));
    assert_eq!(executor.calls, 0);
}

#[test]
fn changed_time_policy_cannot_reauthorize_old_permit() {
    let grant = root_grant();
    let mint_time = verified_time(grant.digest().0, 7);
    let mint_state = initial_state(&grant, &mint_time);
    let (permit, _, _) = mint_permit(&grant, &mint_time, &mint_state);

    let new_time = verified_time(grant.digest().0, 8);
    let new_state = verified_state(
        &grant,
        &new_time,
        Some(grant.authority_context.clone()),
        vec![],
        9,
        12,
        digest(56),
        13,
    );
    let mut executor = CountingExecutor::good();
    let error = dispatch_once_v2(&mut executor, &grant, &new_state, &new_time, permit).unwrap_err();
    assert!(matches!(
        error,
        EffectEntryV2Error::Admission(EntryAdmissionV2Error::TimePolicyChanged)
    ));
    assert_eq!(executor.calls, 0);
}

#[test]
fn executor_error_does_not_refund_outcome_unknown() {
    let grant = root_grant();
    let time = verified_time(grant.digest().0, 7);
    let state = initial_state(&grant, &time);
    let (permit, account, reservation_id) = mint_permit(&grant, &time, &state);
    let mut executor = CountingExecutor {
        fail: true,
        ..CountingExecutor::good()
    };

    let error = dispatch_once_v2(&mut executor, &grant, &state, &time, permit).unwrap_err();
    assert!(matches!(error, EffectEntryV2Error::Executor(ExecutorFailure)));
    assert_eq!(executor.calls, 1);
    assert_eq!(
        account.reservation(reservation_id).unwrap().state,
        ReservationState::OutcomeUnknown
    );
    assert_eq!(account.authority_use_state().unwrap().charged(), 1);
}
