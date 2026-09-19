// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Same-object current-authority admission for crash-conservative effects.
//!
//! This crate is the v0.2 composition waist between verified current authority
//! and the repaired durable action frontier. It deliberately does not execute
//! effects.
//!
//! ```text
//! CapabilityGrant v2
//! + VerifiedAuthorityStateV2
//! + VerifiedAuthorityTime
//! + exact current GrantAccountV2 / grant-owning CasFrontierV2
//! + exact semantic effect binding
//!     -> authorized + durably persisted reservation
//!     -> fresh reserved-use currentness
//!     -> durable OutcomeUnknown arming
//!     -> fresh currentness
//!     -> affine DispatchPermitV2
//! ```
//!
//! The permit is not an effect receipt and does not prove that an executor
//! accepted, applied, or completed the effect. Phi, confidence, simulation
//! success, utility, and reputation are intentionally absent from authority.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_action_checkpoint::{CheckpointHeadV2, CheckpointV2Error, GrantAccountCheckpointV2};
use symthaea_action_frontier::{
    CasFrontierV2, CheckpointCasStoreV2, DurablyArmedReservationV2, FrontierV2Error,
    PersistedReservationV2,
};
use symthaea_action_runtime::{
    AttemptId, EffectBindingDigest, EffectIntentId, GrantAccountV2, ReservationId, RuntimeV2Error,
};
use symthaea_authority::{
    evaluate_authority, AuthorityDecision, AuthorityEvaluationInput, CapabilityGrant, DenyReason,
    Digest32, GrantUseState, GrantValidationError, Operation, PrincipalId, PurposeId, ResourceRef,
    RiskBudget, TaskId,
};
use symthaea_authority_state::{AuthorityStateError, VerifiedAuthorityStateV2};
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use thiserror::Error;

const EFFECT_BINDING_DOMAIN: &[u8] = b"symthaea.action-admission.effect-binding.v2\0";
const DISPATCH_PERMIT_DOMAIN: &[u8] = b"symthaea.action-admission.dispatch-permit.v2\0";
const MAX_BINDING_IDENTIFIER_BYTES: usize = 1024;

/// Exact authority-relevant meaning of one domain effect.
///
/// This is ordinary semantic data, not authority. A domain adapter may create
/// it, but only this crate's same-object composition may turn a matching binding
/// into an affine dispatch permit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectAuthorityBindingV2 {
    pub subject: PrincipalId,
    pub executor: PrincipalId,
    pub purpose: PurposeId,
    pub task: Option<TaskId>,
    pub resource: ResourceRef,
    pub operation: Operation,
    pub plan_digest: Option<Digest32>,
    pub world_digest: Option<Digest32>,
    pub parameters_digest: Digest32,
    pub risk_charge: RiskBudget,
}

impl EffectAuthorityBindingV2 {
    pub fn validate_against(&self, grant: &CapabilityGrant) -> Result<(), AdmissionV2Error> {
        grant.validate()?;
        self.validate_shape()?;

        if self.subject != grant.subject {
            return Err(AdmissionV2Error::EffectBindingMismatch("subject"));
        }
        if grant
            .audience
            .as_ref()
            .is_some_and(|audience| audience != &self.executor)
        {
            return Err(AdmissionV2Error::EffectBindingMismatch("executor/audience"));
        }
        if self.purpose != grant.purpose {
            return Err(AdmissionV2Error::EffectBindingMismatch("purpose"));
        }
        if grant
            .task
            .as_ref()
            .is_some_and(|task| self.task.as_ref() != Some(task))
        {
            return Err(AdmissionV2Error::EffectBindingMismatch("task"));
        }
        if !grant.resources.contains(&self.resource) {
            return Err(AdmissionV2Error::EffectBindingMismatch("resource"));
        }
        if !grant.operations.contains(&self.operation) {
            return Err(AdmissionV2Error::EffectBindingMismatch("operation"));
        }
        if grant
            .plan_digest
            .is_some_and(|required| self.plan_digest != Some(required))
        {
            return Err(AdmissionV2Error::EffectBindingMismatch("plan"));
        }
        if grant
            .world_digest
            .is_some_and(|required| self.world_digest != Some(required))
        {
            return Err(AdmissionV2Error::EffectBindingMismatch("world"));
        }
        if !self.risk_charge.attenuates(grant.risk_budget) {
            return Err(AdmissionV2Error::EffectBindingMismatch("risk charge"));
        }
        Ok(())
    }

    /// Domain-separated semantic commitment independent of storage serialization.
    pub fn digest(&self) -> Result<EffectBindingDigest, AdmissionV2Error> {
        self.validate_shape()?;
        let mut transcript = Vec::with_capacity(512);
        push_bytes(&mut transcript, EFFECT_BINDING_DOMAIN)?;
        push_string(&mut transcript, &self.subject.0)?;
        push_string(&mut transcript, &self.executor.0)?;
        push_string(&mut transcript, &self.purpose.0)?;
        push_optional_string(&mut transcript, self.task.as_ref().map(|task| task.0.as_str()))?;
        push_string(&mut transcript, &self.resource.0)?;
        push_string(&mut transcript, &self.operation.0)?;
        push_optional_digest(&mut transcript, self.plan_digest);
        push_optional_digest(&mut transcript, self.world_digest);
        transcript.extend_from_slice(&self.parameters_digest.0);
        push_risk(&mut transcript, self.risk_charge);
        Ok(EffectBindingDigest(Digest32(*blake3::hash(&transcript).as_bytes())))
    }

    fn validate_shape(&self) -> Result<(), AdmissionV2Error> {
        for value in [
            self.subject.0.as_str(),
            self.executor.0.as_str(),
            self.purpose.0.as_str(),
            self.resource.0.as_str(),
            self.operation.0.as_str(),
        ] {
            validate_identifier(value)?;
        }
        if let Some(task) = self.task.as_ref() {
            validate_identifier(&task.0)?;
        }
        if self.parameters_digest.0 == [0; 32] {
            return Err(AdmissionV2Error::ZeroParametersDigest);
        }
        Ok(())
    }
}

/// Affine proof that one exact use was admitted under verified authority and
/// durably persisted in `Reserved` state.
#[derive(Debug)]
pub struct AuthorizedPersistedReservationV2 {
    grant_digest: Digest32,
    effect_intent_id: EffectIntentId,
    attempt_id: AttemptId,
    reservation_id: ReservationId,
    binding: EffectAuthorityBindingV2,
    binding_digest: EffectBindingDigest,
    allocation_use_state: GrantUseState,
    allocation_authority_snapshot_digest: Digest32,
    persisted: PersistedReservationV2,
}

impl AuthorizedPersistedReservationV2 {
    pub fn grant_digest(&self) -> Digest32 {
        self.grant_digest
    }

    pub fn reservation_id(&self) -> ReservationId {
        self.reservation_id
    }

    pub fn effect_intent_id(&self) -> EffectIntentId {
        self.effect_intent_id
    }

    pub fn attempt_id(&self) -> AttemptId {
        self.attempt_id
    }

    pub fn effect_binding(&self) -> &EffectAuthorityBindingV2 {
        &self.binding
    }

    pub fn effect_binding_digest(&self) -> EffectBindingDigest {
        self.binding_digest
    }

    pub fn allocation_use_state(&self) -> GrantUseState {
        self.allocation_use_state
    }

    pub fn allocation_authority_snapshot_digest(&self) -> Digest32 {
        self.allocation_authority_snapshot_digest
    }

    pub fn persisted_head(&self) -> CheckpointHeadV2 {
        self.persisted.persisted_head()
    }
}

/// Affine proof that an authority-admitted reservation was durably moved to
/// `OutcomeUnknown` only after another fresh reserved-use currentness check.
#[derive(Debug)]
pub struct AuthorizedDurablyArmedReservationV2 {
    grant_digest: Digest32,
    effect_intent_id: EffectIntentId,
    attempt_id: AttemptId,
    reservation_id: ReservationId,
    binding: EffectAuthorityBindingV2,
    binding_digest: EffectBindingDigest,
    allocation_use_state: GrantUseState,
    allocation_authority_snapshot_digest: Digest32,
    prearm_authority_snapshot_digest: Digest32,
    armed: DurablyArmedReservationV2,
}

impl AuthorizedDurablyArmedReservationV2 {
    pub fn grant_digest(&self) -> Digest32 {
        self.grant_digest
    }

    pub fn reservation_id(&self) -> ReservationId {
        self.reservation_id
    }

    pub fn effect_binding_digest(&self) -> EffectBindingDigest {
        self.binding_digest
    }

    pub fn allocation_use_state(&self) -> GrantUseState {
        self.allocation_use_state
    }

    pub fn armed_head(&self) -> CheckpointHeadV2 {
        self.armed.armed_head()
    }
}

/// Ordinary stable identity of one affine dispatch permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DispatchPermitId(pub Digest32);

/// Affine same-object authority to cross one future effect-entry boundary.
///
/// Deliberately neither Clone nor Serde. A downstream executor must consume this
/// object by value. Its existence proves admission at mint time only; it does
/// not prove effect dispatch, acceptance, application, observation, or success.
#[derive(Debug)]
pub struct DispatchPermitV2 {
    permit_id: DispatchPermitId,
    grant_digest: Digest32,
    effect_intent_id: EffectIntentId,
    attempt_id: AttemptId,
    reservation_id: ReservationId,
    binding: EffectAuthorityBindingV2,
    binding_digest: EffectBindingDigest,
    allocation_use_state: GrantUseState,
    reserved_head: CheckpointHeadV2,
    armed_head: CheckpointHeadV2,
    allocation_authority_snapshot_digest: Digest32,
    prearm_authority_snapshot_digest: Digest32,
    dispatch_authority_snapshot_digest: Digest32,
    dispatch_authority_state_sequence: u64,
    dispatch_authority_source_frontier: (u64, Digest32),
    dispatch_state_policy_digest: [u8; 32],
    dispatch_time_policy_digest: [u8; 32],
}

impl DispatchPermitV2 {
    pub fn permit_id(&self) -> DispatchPermitId {
        self.permit_id
    }

    pub fn grant_digest(&self) -> Digest32 {
        self.grant_digest
    }

    pub fn effect_intent_id(&self) -> EffectIntentId {
        self.effect_intent_id
    }

    pub fn attempt_id(&self) -> AttemptId {
        self.attempt_id
    }

    pub fn reservation_id(&self) -> ReservationId {
        self.reservation_id
    }

    pub fn effect_binding(&self) -> &EffectAuthorityBindingV2 {
        &self.binding
    }

    pub fn effect_binding_digest(&self) -> EffectBindingDigest {
        self.binding_digest
    }

    pub fn allocation_use_state(&self) -> GrantUseState {
        self.allocation_use_state
    }

    pub fn reserved_head(&self) -> CheckpointHeadV2 {
        self.reserved_head
    }

    pub fn armed_head(&self) -> CheckpointHeadV2 {
        self.armed_head
    }

    pub fn allocation_authority_snapshot_digest(&self) -> Digest32 {
        self.allocation_authority_snapshot_digest
    }

    pub fn prearm_authority_snapshot_digest(&self) -> Digest32 {
        self.prearm_authority_snapshot_digest
    }

    pub fn dispatch_authority_snapshot_digest(&self) -> Digest32 {
        self.dispatch_authority_snapshot_digest
    }

    pub fn dispatch_authority_state_sequence(&self) -> u64 {
        self.dispatch_authority_state_sequence
    }

    pub fn dispatch_authority_source_frontier(&self) -> (u64, Digest32) {
        self.dispatch_authority_source_frontier
    }

    pub fn dispatch_state_policy_digest(&self) -> [u8; 32] {
        self.dispatch_state_policy_digest
    }

    pub fn dispatch_time_policy_digest(&self) -> [u8; 32] {
        self.dispatch_time_policy_digest
    }
}

/// Phase A: verify current authority for a *new* use, allocate exactly that use
/// in the exact account, and durably CAS the resulting single `Reserved` record.
///
/// The use counters are derived from `account`; callers never provide them.
/// On any error after account mutation the caller must discard/recover the local
/// account from a separately trusted/authenticated frontier before retrying.
pub fn authorize_and_persist_reservation<S>(
    frontier: &mut CasFrontierV2<S>,
    grant: &CapabilityGrant,
    state: &VerifiedAuthorityStateV2,
    time: &VerifiedAuthorityTime,
    account: &mut GrantAccountV2,
    effect_intent_id: EffectIntentId,
    attempt_id: AttemptId,
    binding: EffectAuthorityBindingV2,
) -> Result<AuthorizedPersistedReservationV2, AdmissionFrontierV2Error<S::Error>>
where
    S: CheckpointCasStoreV2,
{
    binding.validate_against(grant)?;
    ensure_frontier_grant(grant, frontier)?;
    ensure_account_frontier_sync(account, frontier)?;
    if account.grant_digest() != grant.digest() {
        return Err(AdmissionV2Error::AccountGrantMismatch.into());
    }

    let allocation_use_state = account.authority_use_state()?;
    require_current_authority(grant, state, time, allocation_use_state)?;
    let binding_digest = binding.digest()?;
    let allocation_authority_snapshot_digest = state.snapshot_digest();

    let reservation_id = account.reserve_execution(
        effect_intent_id,
        attempt_id,
        binding_digest,
        binding.risk_charge,
    )?;
    let successor =
        GrantAccountCheckpointV2::successor(frontier.expected_checkpoint(), grant, account)?;
    let persisted = frontier.persist_new_reservation(successor)?;

    if persisted.grant_digest() != grant.digest()
        || persisted.reservation_id() != reservation_id
        || persisted.effect_intent_id() != effect_intent_id
        || persisted.attempt_id() != attempt_id
        || persisted.effect_binding_digest() != binding_digest
        || persisted.risk_charge() != binding.risk_charge
    {
        return Err(AdmissionV2Error::InternalBindingMismatch.into());
    }

    Ok(AuthorizedPersistedReservationV2 {
        grant_digest: grant.digest(),
        effect_intent_id,
        attempt_id,
        reservation_id,
        binding,
        binding_digest,
        allocation_use_state,
        allocation_authority_snapshot_digest,
        persisted,
    })
}

/// Phase B1: re-establish current authority for the **already allocated** use,
/// then durably CAS only its `Reserved -> OutcomeUnknown` transition.
///
/// The evaluator receives the exact pre-allocation use state captured in Phase A.
/// This is deliberate: the owned reservation is not a request to allocate a
/// second use. No counter subtraction or caller-selected replacement occurs.
pub fn arm_authorized_reservation<S>(
    frontier: &mut CasFrontierV2<S>,
    grant: &CapabilityGrant,
    state: &VerifiedAuthorityStateV2,
    time: &VerifiedAuthorityTime,
    account: &mut GrantAccountV2,
    authorized: AuthorizedPersistedReservationV2,
) -> Result<AuthorizedDurablyArmedReservationV2, AdmissionFrontierV2Error<S::Error>>
where
    S: CheckpointCasStoreV2,
{
    ensure_frontier_grant(grant, frontier)?;
    ensure_account_frontier_sync(account, frontier)?;
    verify_authorized_persisted(grant, &authorized)?;
    require_current_authority(grant, state, time, authorized.allocation_use_state)?;

    let prearm_authority_snapshot_digest = state.snapshot_digest();
    account.mark_outcome_unknown(authorized.reservation_id)?;
    let successor =
        GrantAccountCheckpointV2::successor(frontier.expected_checkpoint(), grant, account)?;

    let AuthorizedPersistedReservationV2 {
        grant_digest,
        effect_intent_id,
        attempt_id,
        reservation_id,
        binding,
        binding_digest,
        allocation_use_state,
        allocation_authority_snapshot_digest,
        persisted,
    } = authorized;

    let armed = frontier.arm_outcome_unknown(persisted, successor)?;
    if armed.grant_digest() != grant_digest
        || armed.reservation_id() != reservation_id
        || armed.effect_intent_id() != effect_intent_id
        || armed.attempt_id() != attempt_id
        || armed.effect_binding_digest() != binding_digest
        || armed.risk_charge() != binding.risk_charge
    {
        return Err(AdmissionV2Error::InternalBindingMismatch.into());
    }

    Ok(AuthorizedDurablyArmedReservationV2 {
        grant_digest,
        effect_intent_id,
        attempt_id,
        reservation_id,
        binding,
        binding_digest,
        allocation_use_state,
        allocation_authority_snapshot_digest,
        prearm_authority_snapshot_digest,
        armed,
    })
}

/// Phase B2: perform one final fresh current-authority check after durable
/// conservative arming and mint the only dispatch-capable object in this crate.
///
/// Failure consumes the affine armed token and leaves durable accounting in
/// `OutcomeUnknown`; recovery/reconciliation must then prove what is safe to do.
pub fn mint_dispatch_permit(
    grant: &CapabilityGrant,
    state: &VerifiedAuthorityStateV2,
    time: &VerifiedAuthorityTime,
    armed: AuthorizedDurablyArmedReservationV2,
) -> Result<DispatchPermitV2, AdmissionV2Error> {
    if armed.grant_digest != grant.digest() {
        return Err(AdmissionV2Error::GrantMismatch);
    }
    armed.binding.validate_against(grant)?;
    if armed.armed.grant_digest() != armed.grant_digest
        || armed.armed.reservation_id() != armed.reservation_id
        || armed.armed.effect_intent_id() != armed.effect_intent_id
        || armed.armed.attempt_id() != armed.attempt_id
        || armed.armed.effect_binding_digest() != armed.binding_digest
        || armed.armed.risk_charge() != armed.binding.risk_charge
    {
        return Err(AdmissionV2Error::InternalBindingMismatch);
    }

    require_current_authority(grant, state, time, armed.allocation_use_state)?;
    let dispatch_authority_snapshot_digest = state.snapshot_digest();
    let dispatch_authority_source_frontier = state.source_frontier();
    let reserved_head = armed.armed.reserved_head();
    let armed_head = armed.armed.armed_head();
    let permit_id = derive_dispatch_permit_id(
        armed.grant_digest,
        armed.effect_intent_id,
        armed.attempt_id,
        armed.reservation_id,
        armed.binding_digest,
        armed.allocation_use_state,
        reserved_head,
        armed_head,
        armed.allocation_authority_snapshot_digest,
        armed.prearm_authority_snapshot_digest,
        dispatch_authority_snapshot_digest,
        state.state_sequence(),
        dispatch_authority_source_frontier,
        state.state_policy_digest(),
        state.time_policy_digest(),
    );

    Ok(DispatchPermitV2 {
        permit_id,
        grant_digest: armed.grant_digest,
        effect_intent_id: armed.effect_intent_id,
        attempt_id: armed.attempt_id,
        reservation_id: armed.reservation_id,
        binding: armed.binding,
        binding_digest: armed.binding_digest,
        allocation_use_state: armed.allocation_use_state,
        reserved_head,
        armed_head,
        allocation_authority_snapshot_digest: armed.allocation_authority_snapshot_digest,
        prearm_authority_snapshot_digest: armed.prearm_authority_snapshot_digest,
        dispatch_authority_snapshot_digest,
        dispatch_authority_state_sequence: state.state_sequence(),
        dispatch_authority_source_frontier,
        dispatch_state_policy_digest: state.state_policy_digest(),
        dispatch_time_policy_digest: state.time_policy_digest(),
    })
}

fn ensure_frontier_grant<S>(
    grant: &CapabilityGrant,
    frontier: &CasFrontierV2<S>,
) -> Result<(), AdmissionV2Error>
where
    S: CheckpointCasStoreV2,
{
    if frontier.grant_digest() != grant.digest() {
        return Err(AdmissionV2Error::FrontierGrantMismatch);
    }
    Ok(())
}

fn ensure_account_frontier_sync<S>(
    account: &GrantAccountV2,
    frontier: &CasFrontierV2<S>,
) -> Result<(), AdmissionV2Error>
where
    S: CheckpointCasStoreV2,
{
    if account.snapshot() != frontier.expected_checkpoint().snapshot().clone() {
        return Err(AdmissionV2Error::AccountFrontierDiverged);
    }
    Ok(())
}

fn verify_authorized_persisted(
    grant: &CapabilityGrant,
    authorized: &AuthorizedPersistedReservationV2,
) -> Result<(), AdmissionV2Error> {
    if authorized.grant_digest != grant.digest()
        || authorized.persisted.grant_digest() != authorized.grant_digest
        || authorized.persisted.reservation_id() != authorized.reservation_id
        || authorized.persisted.effect_intent_id() != authorized.effect_intent_id
        || authorized.persisted.attempt_id() != authorized.attempt_id
        || authorized.persisted.effect_binding_digest() != authorized.binding_digest
        || authorized.persisted.risk_charge() != authorized.binding.risk_charge
    {
        return Err(AdmissionV2Error::InternalBindingMismatch);
    }
    authorized.binding.validate_against(grant)
}

fn require_current_authority(
    grant: &CapabilityGrant,
    state: &VerifiedAuthorityStateV2,
    time: &VerifiedAuthorityTime,
    use_state: GrantUseState,
) -> Result<(), AdmissionV2Error> {
    state.ensure_fresh(grant, time)?;
    let current_authority_context = state
        .authority_context()
        .cloned()
        .ok_or(AdmissionV2Error::NoCurrentAuthorityContext)?;
    let input = AuthorityEvaluationInput {
        now_unix_s: time.conservative_now_unix_s()?,
        current_epoch: state.authority_epoch(),
        current_authority_context,
        use_state,
    };
    match evaluate_authority(grant, &input, state.negative_facts()) {
        AuthorityDecision::Allow => Ok(()),
        AuthorityDecision::Deny(reason) => Err(AdmissionV2Error::AuthorityDenied(reason)),
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_dispatch_permit_id(
    grant_digest: Digest32,
    effect_intent_id: EffectIntentId,
    attempt_id: AttemptId,
    reservation_id: ReservationId,
    binding_digest: EffectBindingDigest,
    allocation_use_state: GrantUseState,
    reserved_head: CheckpointHeadV2,
    armed_head: CheckpointHeadV2,
    allocation_authority_snapshot_digest: Digest32,
    prearm_authority_snapshot_digest: Digest32,
    dispatch_authority_snapshot_digest: Digest32,
    dispatch_authority_state_sequence: u64,
    dispatch_authority_source_frontier: (u64, Digest32),
    dispatch_state_policy_digest: [u8; 32],
    dispatch_time_policy_digest: [u8; 32],
) -> DispatchPermitId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DISPATCH_PERMIT_DOMAIN);
    hasher.update(&grant_digest.0);
    hasher.update(&(effect_intent_id.0).0);
    hasher.update(&(attempt_id.0).0);
    hasher.update(&(reservation_id.0).0);
    hasher.update(&(binding_digest.0).0);
    hasher.update(&allocation_use_state.committed.to_be_bytes());
    hasher.update(&allocation_use_state.reserved.to_be_bytes());
    hasher.update(&reserved_head.sequence.to_be_bytes());
    hasher.update(&reserved_head.digest.0);
    hasher.update(&armed_head.sequence.to_be_bytes());
    hasher.update(&armed_head.digest.0);
    hasher.update(&allocation_authority_snapshot_digest.0);
    hasher.update(&prearm_authority_snapshot_digest.0);
    hasher.update(&dispatch_authority_snapshot_digest.0);
    hasher.update(&dispatch_authority_state_sequence.to_be_bytes());
    hasher.update(&dispatch_authority_source_frontier.0.to_be_bytes());
    hasher.update(&(dispatch_authority_source_frontier.1).0);
    hasher.update(&dispatch_state_policy_digest);
    hasher.update(&dispatch_time_policy_digest);
    DispatchPermitId(Digest32(*hasher.finalize().as_bytes()))
}

fn validate_identifier(value: &str) -> Result<(), AdmissionV2Error> {
    if value.is_empty() || value.len() > MAX_BINDING_IDENTIFIER_BYTES {
        return Err(AdmissionV2Error::InvalidEffectIdentifier);
    }
    Ok(())
}

fn push_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<(), AdmissionV2Error> {
    let len = u32::try_from(value.len()).map_err(|_| AdmissionV2Error::CanonicalEncoding)?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value);
    Ok(())
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), AdmissionV2Error> {
    validate_identifier(value)?;
    push_bytes(out, value.as_bytes())
}

fn push_optional_string(out: &mut Vec<u8>, value: Option<&str>) -> Result<(), AdmissionV2Error> {
    match value {
        Some(value) => {
            out.push(1);
            push_string(out, value)
        }
        None => {
            out.push(0);
            Ok(())
        }
    }
}

fn push_optional_digest(out: &mut Vec<u8>, value: Option<Digest32>) {
    match value {
        Some(value) => {
            out.push(1);
            out.extend_from_slice(&value.0);
        }
        None => out.push(0),
    }
}

fn push_risk(out: &mut Vec<u8>, risk: RiskBudget) {
    out.extend_from_slice(&risk.mutation_units.to_be_bytes());
    out.extend_from_slice(&risk.irreversible_units.to_be_bytes());
    out.extend_from_slice(&risk.external_disclosure_bytes.to_be_bytes());
    out.extend_from_slice(&risk.monetary_microunits.to_be_bytes());
}

#[derive(Debug, Error)]
pub enum AdmissionV2Error {
    #[error("capability grant is invalid: {0}")]
    InvalidGrant(#[from] GrantValidationError),
    #[error("verified authority state failed: {0}")]
    AuthorityState(#[from] AuthorityStateError),
    #[error("verified authority time failed: {0}")]
    AuthorityTime(#[from] AuthorityTimeError),
    #[error("runtime accounting failed: {0}")]
    Runtime(#[from] RuntimeV2Error),
    #[error("checkpoint construction failed: {0}")]
    Checkpoint(#[from] CheckpointV2Error),
    #[error("pure authority evaluation denied the use: {0:?}")]
    AuthorityDenied(DenyReason),
    #[error("verified current authority state reports no active context")]
    NoCurrentAuthorityContext,
    #[error("frontier owns a different capability grant")]
    FrontierGrantMismatch,
    #[error("grant account belongs to a different capability grant")]
    AccountGrantMismatch,
    #[error("in-memory grant account does not equal the exact adapter-local expected frontier snapshot")]
    AccountFrontierDiverged,
    #[error("effect authority binding does not match grant field: {0}")]
    EffectBindingMismatch(&'static str),
    #[error("effect binding contains an empty or oversized exact identifier")]
    InvalidEffectIdentifier,
    #[error("effect parameter commitment must not be all-zero")]
    ZeroParametersDigest,
    #[error("effect-binding canonical encoding failed")]
    CanonicalEncoding,
    #[error("authority/persistence objects do not name the same exact effect")]
    InternalBindingMismatch,
    #[error("dispatch grant does not match the durably armed reservation")]
    GrantMismatch,
}

#[derive(Debug, Error)]
pub enum AdmissionFrontierV2Error<E>
where
    E: StdError + 'static,
{
    #[error("action admission failed: {0}")]
    Admission(#[source] AdmissionV2Error),
    #[error("action frontier failed: {0}")]
    Frontier(#[source] FrontierV2Error<E>),
}

impl<E> From<AdmissionV2Error> for AdmissionFrontierV2Error<E>
where
    E: StdError + 'static,
{
    fn from(value: AdmissionV2Error) -> Self {
        Self::Admission(value)
    }
}

impl<E> From<RuntimeV2Error> for AdmissionFrontierV2Error<E>
where
    E: StdError + 'static,
{
    fn from(value: RuntimeV2Error) -> Self {
        Self::Admission(AdmissionV2Error::Runtime(value))
    }
}

impl<E> From<CheckpointV2Error> for AdmissionFrontierV2Error<E>
where
    E: StdError + 'static,
{
    fn from(value: CheckpointV2Error) -> Self {
        Self::Admission(AdmissionV2Error::Checkpoint(value))
    }
}

impl<E> From<FrontierV2Error<E>> for AdmissionFrontierV2Error<E>
where
    E: StdError + 'static,
{
    fn from(value: FrontierV2Error<E>) -> Self {
        Self::Frontier(value)
    }
}

#[cfg(test)]
mod tests;
