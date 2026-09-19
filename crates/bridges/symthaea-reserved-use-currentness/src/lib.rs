// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reserved-use currentness for Symthaea action authority v0.2.
//!
//! This crate composes two independently developed facts without turning either
//! into execution authority:
//!
//! ```text
//! affine PersistedReservationV2
//! + fresh VerifiedAuthorityStateV2
//! + VerifiedAuthorityTime for the exact grant
//! + unchanged non-use authority predicates
//!     -> ReservedUseCurrentnessV2
//! ```
//!
//! `ReservedUseCurrentnessV2` deliberately does **not** claim that the local
//! durable frontier was freshly observed at construction time and is not a
//! dispatch permit. The stronger transition consumes it through the existing
//! CAS frontier:
//!
//! ```text
//! ReservedUseCurrentnessV2
//! + exact Reserved -> OutcomeUnknown successor
//! + successful exact-head CAS
//! + fresh post-CAS durable head observation
//!     -> CurrentnessBoundArmedReservationV2
//! ```
//!
//! Even `CurrentnessBoundArmedReservationV2` is not a dispatch permit. A later
//! point-of-effect tranche must close the remaining authority-state race before
//! an irreversible external effect can begin.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_action_checkpoint::GrantAccountCheckpointV2;
use symthaea_action_frontier::{
    CasFrontierV2, CheckpointCasStoreV2, DurablyArmedReservationV2, FrontierV2Error,
    PersistedReservationV2,
};
use symthaea_action_runtime::{
    AttemptId, EffectBindingDigest, EffectIntentId, ReservationId, ReservationState,
};
use symthaea_authority::{
    AuthorityContextRef, AuthorityDecision, AuthorityEpoch, CapabilityGrant, DenyReason, Digest32,
    NegativeAuthorityFact, RiskBudget,
};
use symthaea_authority_state::{AuthorityStateError, VerifiedAuthorityStateV2};
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use thiserror::Error;

/// Fresh verified authority facts bound to one already-persisted reservation.
///
/// This type owns both opaque proof objects. Neither can be detached through the
/// public API, so later code cannot accidentally use the verified state for a
/// different reservation or use the reservation without the state that was
/// checked for it.
///
/// Deliberately neither `Clone` nor Serde.
#[derive(Debug)]
pub struct ReservedUseCurrentnessV2 {
    persisted: PersistedReservationV2,
    authority_state: VerifiedAuthorityStateV2,
}

impl ReservedUseCurrentnessV2 {
    pub fn grant_digest(&self) -> Digest32 {
        self.persisted.grant_digest()
    }

    pub fn reservation_id(&self) -> ReservationId {
        self.persisted.reservation_id()
    }

    pub fn effect_intent_id(&self) -> EffectIntentId {
        self.persisted.effect_intent_id()
    }

    pub fn attempt_id(&self) -> AttemptId {
        self.persisted.attempt_id()
    }

    pub fn effect_binding_digest(&self) -> EffectBindingDigest {
        self.persisted.effect_binding_digest()
    }

    pub fn risk_charge(&self) -> RiskBudget {
        self.persisted.risk_charge()
    }

    pub fn persisted_head(&self) -> symthaea_action_checkpoint::CheckpointHeadV2 {
        self.persisted.persisted_head()
    }

    pub fn authority_snapshot_digest(&self) -> Digest32 {
        self.authority_state.snapshot_digest()
    }

    pub fn authority_state_sequence(&self) -> u64 {
        self.authority_state.state_sequence()
    }

    pub fn authority_source_frontier(&self) -> (u64, Digest32) {
        self.authority_state.source_frontier()
    }

    pub fn authority_epoch(&self) -> AuthorityEpoch {
        self.authority_state.authority_epoch()
    }

    /// Borrow the exact verified state retained by this currentness proof.
    ///
    /// The state cannot be moved out independently of the reservation binding.
    pub fn authority_state(&self) -> &VerifiedAuthorityStateV2 {
        &self.authority_state
    }
}

/// Durable conservative arming that retains the exact verified authority state
/// used immediately before the transition.
///
/// This remains persistence/currentness evidence only. It cannot enter an
/// external effect by itself.
#[derive(Debug)]
pub struct CurrentnessBoundArmedReservationV2 {
    armed: DurablyArmedReservationV2,
    authority_state: VerifiedAuthorityStateV2,
}

impl CurrentnessBoundArmedReservationV2 {
    pub fn grant_digest(&self) -> Digest32 {
        self.armed.grant_digest()
    }

    pub fn reservation_id(&self) -> ReservationId {
        self.armed.reservation_id()
    }

    pub fn effect_intent_id(&self) -> EffectIntentId {
        self.armed.effect_intent_id()
    }

    pub fn attempt_id(&self) -> AttemptId {
        self.armed.attempt_id()
    }

    pub fn effect_binding_digest(&self) -> EffectBindingDigest {
        self.armed.effect_binding_digest()
    }

    pub fn risk_charge(&self) -> RiskBudget {
        self.armed.risk_charge()
    }

    pub fn reserved_head(&self) -> symthaea_action_checkpoint::CheckpointHeadV2 {
        self.armed.reserved_head()
    }

    pub fn armed_head(&self) -> symthaea_action_checkpoint::CheckpointHeadV2 {
        self.armed.armed_head()
    }

    pub fn authority_snapshot_digest(&self) -> Digest32 {
        self.authority_state.snapshot_digest()
    }

    pub fn authority_state_sequence(&self) -> u64 {
        self.authority_state.state_sequence()
    }

    pub fn authority_state(&self) -> &VerifiedAuthorityStateV2 {
        &self.authority_state
    }
}

/// Bind fresh verifier-owned authority facts to one exact affine persisted
/// reservation without allocating or evaluating another use.
///
/// The local frontier check here is intentionally an adapter-local identity
/// check, not a fresh durable read. A stale local writer cannot reach the
/// stronger armed object because [`arm_current_reserved_use_v2`] must traverse
/// the frontier's exact-head CAS and fresh post-CAS observation.
pub fn verify_reserved_use_currentness_v2<S>(
    frontier: &CasFrontierV2<S>,
    grant: &CapabilityGrant,
    persisted: PersistedReservationV2,
    authority_state: VerifiedAuthorityStateV2,
    time: &VerifiedAuthorityTime,
) -> Result<ReservedUseCurrentnessV2, ReservedUseCurrentnessError>
where
    S: CheckpointCasStoreV2,
{
    validate_persisted_binding(frontier, grant, &persisted)?;
    validate_current_authority(grant, &persisted, &authority_state, time)?;

    Ok(ReservedUseCurrentnessV2 {
        persisted,
        authority_state,
    })
}

/// Consume reserved-use currentness into the exact durable
/// `Reserved -> OutcomeUnknown` transition.
///
/// Freshness and every non-use authority predicate are checked again immediately
/// before entering the frontier CAS path. The resulting object still grants no
/// effect-entry authority.
pub fn arm_current_reserved_use_v2<S>(
    frontier: &mut CasFrontierV2<S>,
    grant: &CapabilityGrant,
    currentness: ReservedUseCurrentnessV2,
    time: &VerifiedAuthorityTime,
    successor: GrantAccountCheckpointV2,
) -> Result<CurrentnessBoundArmedReservationV2, ArmCurrentReservedUseError<S::Error>>
where
    S: CheckpointCasStoreV2,
{
    let ReservedUseCurrentnessV2 {
        persisted,
        authority_state,
    } = currentness;

    validate_persisted_binding(frontier, grant, &persisted)
        .map_err(ArmCurrentReservedUseError::Currentness)?;
    validate_current_authority(grant, &persisted, &authority_state, time)
        .map_err(ArmCurrentReservedUseError::Currentness)?;

    let armed = frontier
        .arm_outcome_unknown(persisted, successor)
        .map_err(ArmCurrentReservedUseError::Frontier)?;

    Ok(CurrentnessBoundArmedReservationV2 {
        armed,
        authority_state,
    })
}

fn validate_persisted_binding<S>(
    frontier: &CasFrontierV2<S>,
    grant: &CapabilityGrant,
    persisted: &PersistedReservationV2,
) -> Result<(), ReservedUseCurrentnessError>
where
    S: CheckpointCasStoreV2,
{
    if frontier.is_contained() {
        return Err(ReservedUseCurrentnessError::FrontierContained);
    }

    let grant_digest = grant.digest();
    if frontier.grant_digest() != grant_digest || persisted.grant_digest() != grant_digest {
        return Err(ReservedUseCurrentnessError::GrantMismatch);
    }
    if persisted.persisted_head() != frontier.expected_head() {
        return Err(ReservedUseCurrentnessError::PersistedHeadMismatch);
    }

    let record = frontier
        .expected_checkpoint()
        .snapshot()
        .reservations
        .get(&persisted.reservation_id())
        .ok_or(ReservedUseCurrentnessError::PersistedRecordMismatch)?;

    if record.state != ReservationState::Reserved
        || record.reservation_id != persisted.reservation_id()
        || record.effect_intent_id != persisted.effect_intent_id()
        || record.attempt_id != persisted.attempt_id()
        || record.effect_binding_digest != persisted.effect_binding_digest()
        || record.risk_charge != persisted.risk_charge()
    {
        return Err(ReservedUseCurrentnessError::PersistedRecordMismatch);
    }

    Ok(())
}

fn validate_current_authority(
    grant: &CapabilityGrant,
    persisted: &PersistedReservationV2,
    authority_state: &VerifiedAuthorityStateV2,
    time: &VerifiedAuthorityTime,
) -> Result<(), ReservedUseCurrentnessError> {
    let grant_digest = grant.digest();
    if persisted.grant_digest() != grant_digest || authority_state.grant_digest() != grant_digest {
        return Err(ReservedUseCurrentnessError::GrantMismatch);
    }

    authority_state.ensure_fresh(grant, time)?;
    let current_context = authority_state
        .authority_context()
        .ok_or(ReservedUseCurrentnessError::NoCurrentAuthorityContext)?;
    let now_unix_s = time.conservative_now_unix_s()?;

    match evaluate_reserved_use_nonbudget(
        grant,
        now_unix_s,
        authority_state.authority_epoch(),
        current_context,
        authority_state.negative_facts(),
    ) {
        AuthorityDecision::Allow => Ok(()),
        AuthorityDecision::Deny(reason) => Err(ReservedUseCurrentnessError::Denied(reason)),
    }
}

/// Evaluate every v0.2 positive/negative authority predicate except allocation
/// of an additional use.
///
/// This is intentionally a separate predicate rather than hidden subtraction
/// from `GrantUseState`: the use represented by `PersistedReservationV2` was
/// already allocated. The caller cannot supply counters to this function.
fn evaluate_reserved_use_nonbudget(
    grant: &CapabilityGrant,
    now_unix_s: u64,
    current_epoch: AuthorityEpoch,
    current_authority_context: &AuthorityContextRef,
    negative_facts: &[NegativeAuthorityFact],
) -> AuthorityDecision {
    if let Err(error) = grant.validate() {
        return AuthorityDecision::Deny(DenyReason::InvalidGrant(error));
    }
    if grant.parent_digest.is_some() {
        return AuthorityDecision::Deny(DenyReason::DelegationChainRequired);
    }
    if grant.authority_epoch != current_epoch {
        return AuthorityDecision::Deny(DenyReason::EpochStale);
    }
    if &grant.authority_context != current_authority_context {
        return AuthorityDecision::Deny(DenyReason::ContextMismatch);
    }
    if grant
        .expires_at_unix_s
        .is_some_and(|expiry| now_unix_s > expiry)
    {
        return AuthorityDecision::Deny(DenyReason::Expired);
    }

    let grant_digest = grant.digest();
    for fact in negative_facts {
        match fact {
            NegativeAuthorityFact::RevokeGrant { grant_digest: revoked }
                if *revoked == grant_digest =>
            {
                return AuthorityDecision::Deny(DenyReason::ExplicitlyRevoked);
            }
            NegativeAuthorityFact::RevokeContext { context }
                if context == &grant.authority_context =>
            {
                return AuthorityDecision::Deny(DenyReason::ContextRevoked);
            }
            NegativeAuthorityFact::TombstonePrincipal { principal }
                if principal == &grant.subject
                    || principal == &grant.issuer
                    || grant.audience.as_ref() == Some(principal) =>
            {
                return AuthorityDecision::Deny(DenyReason::PrincipalTombstoned);
            }
            NegativeAuthorityFact::FreezeResource { resource }
                if grant.resources.contains(resource) =>
            {
                return AuthorityDecision::Deny(DenyReason::ResourceFrozen);
            }
            NegativeAuthorityFact::MinimumResourceEpoch {
                resource,
                minimum_epoch,
            } if grant.resources.contains(resource) && grant.authority_epoch < *minimum_epoch => {
                return AuthorityDecision::Deny(DenyReason::ResourceEpochStale);
            }
            _ => {}
        }
    }

    AuthorityDecision::Allow
}

#[derive(Debug, Error)]
pub enum ReservedUseCurrentnessError {
    #[error("verified authority state failed freshness/currentness validation: {0}")]
    AuthorityState(#[from] AuthorityStateError),
    #[error("verified authority time failed: {0}")]
    AuthorityTime(#[from] AuthorityTimeError),
    #[error("grant identity differs across frontier, reservation, or verified authority state")]
    GrantMismatch,
    #[error("frontier is already contained")]
    FrontierContained,
    #[error("persisted reservation no longer names the adapter's expected frontier head")]
    PersistedHeadMismatch,
    #[error("persisted reservation token does not match the exact expected Reserved record")]
    PersistedRecordMismatch,
    #[error("verified authority source reports no active authority context")]
    NoCurrentAuthorityContext,
    #[error("reserved use is not current under non-use authority predicates: {0:?}")]
    Denied(DenyReason),
}

#[derive(Debug, Error)]
pub enum ArmCurrentReservedUseError<E>
where
    E: StdError + 'static,
{
    #[error("reserved-use currentness failed before durable arming: {0}")]
    Currentness(#[source] ReservedUseCurrentnessError),
    #[error("durable frontier arming failed: {0}")]
    Frontier(#[source] FrontierV2Error<E>),
}

#[cfg(test)]
mod tests;
