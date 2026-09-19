// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-shot point-of-entry authority firewall for Symthaea effects.
//!
//! V2D creates an affine `DispatchPermitV2` after same-object current-authority,
//! accounting, and durable-arming checks. This crate closes the remaining
//! post-mint holding window by consuming that permit and re-establishing current
//! authority immediately before invoking one provisioned executor adapter.
//!
//! ```text
//! DispatchPermitV2
//! + exact CapabilityGrant
//! + fresh VerifiedAuthorityStateV2
//! + fresh VerifiedAuthorityTime
//! + matching provisioned executor-principal assertion
//! + anti-rollback state/policy checks
//!     -> invoke one adapter once
//! ```
//!
//! The adapter principal in this v0.2 tranche is an in-process provisioning /
//! configuration assertion, not cryptographically verified workload, software,
//! device, or embodiment identity. Consequential deployments require a stronger
//! verifier-owned executor binding before making that claim.
//!
//! Executor invocation is not an effect receipt. Success from this crate does
//! not prove remote acceptance, physical/digital state change, observation, or
//! task success.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_action_admission::{
    AdmissionV2Error, DispatchPermitId, DispatchPermitV2, EffectAuthorityBindingV2,
};
use symthaea_action_runtime::{AttemptId, EffectIntentId, ReservationId};
use symthaea_authority::{
    evaluate_authority, AuthorityDecision, AuthorityEvaluationInput, CapabilityGrant, DenyReason,
    Digest32, PrincipalId,
};
use symthaea_authority_state::{AuthorityStateError, VerifiedAuthorityStateV2};
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use thiserror::Error;

/// Exact non-authoritative context passed by reference to one executor call.
///
/// This object is intentionally borrowed and contains no constructor that can
/// create authority. The adapter receives it only after the firewall consumes a
/// real `DispatchPermitV2` and completes point-of-entry revalidation.
#[derive(Debug)]
pub struct EffectEntryContextV2<'a> {
    permit_id: DispatchPermitId,
    grant_digest: Digest32,
    effect_intent_id: EffectIntentId,
    attempt_id: AttemptId,
    reservation_id: ReservationId,
    binding: &'a EffectAuthorityBindingV2,
    authority_snapshot_digest: Digest32,
    authority_state_sequence: u64,
    authority_source_frontier: (u64, Digest32),
}

impl<'a> EffectEntryContextV2<'a> {
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

    pub fn binding(&self) -> &EffectAuthorityBindingV2 {
        self.binding
    }

    pub fn authority_snapshot_digest(&self) -> Digest32 {
        self.authority_snapshot_digest
    }

    pub fn authority_state_sequence(&self) -> u64 {
        self.authority_state_sequence
    }

    pub fn authority_source_frontier(&self) -> (u64, Digest32) {
        self.authority_source_frontier
    }
}

/// Executor adapter whose provisioned principal assertion participates in the
/// point-of-entry configuration check.
///
/// This is deliberately **not** an executor-attestation theorem: a public trait
/// implementation can self-report a principal. The comparison prevents ordinary
/// in-process misrouting only inside the trusted adapter/process boundary.
/// High-consequence callers must additionally require a verifier-owned workload /
/// software / device / embodiment binding before treating executor identity as
/// authenticated.
pub trait BoundEffectExecutorV2 {
    type Output;
    type Error: StdError + 'static;

    fn executor_principal(&self) -> &PrincipalId;

    /// Enter the domain-specific effect boundary.
    ///
    /// Returning `Ok` means only that this adapter invocation returned `Ok`.
    /// Domain-specific acceptance, application, observation, and success claims
    /// require stronger downstream receipts.
    fn enter_effect(
        &mut self,
        context: &EffectEntryContextV2<'_>,
    ) -> Result<Self::Output, Self::Error>;
}

/// Consume one V2D permit and invoke one provisioned executor exactly once after
/// fresh point-of-entry authority validation.
///
/// The permit is moved into this function. If admission fails or the executor
/// returns an error, no permit is returned to the caller. Durable accounting
/// remains conservatively `OutcomeUnknown`; there is no automatic refund.
pub fn dispatch_once_v2<E>(
    executor: &mut E,
    grant: &CapabilityGrant,
    state: &VerifiedAuthorityStateV2,
    time: &VerifiedAuthorityTime,
    permit: DispatchPermitV2,
) -> Result<E::Output, EffectEntryV2Error<E::Error>>
where
    E: BoundEffectExecutorV2,
{
    validate_point_of_entry(executor.executor_principal(), grant, state, time, &permit)?;

    let context = EffectEntryContextV2 {
        permit_id: permit.permit_id(),
        grant_digest: permit.grant_digest(),
        effect_intent_id: permit.effect_intent_id(),
        attempt_id: permit.attempt_id(),
        reservation_id: permit.reservation_id(),
        binding: permit.effect_binding(),
        authority_snapshot_digest: state.snapshot_digest(),
        authority_state_sequence: state.state_sequence(),
        authority_source_frontier: state.source_frontier(),
    };

    executor
        .enter_effect(&context)
        .map_err(EffectEntryV2Error::Executor)
}

fn validate_point_of_entry(
    executor_principal: &PrincipalId,
    grant: &CapabilityGrant,
    state: &VerifiedAuthorityStateV2,
    time: &VerifiedAuthorityTime,
    permit: &DispatchPermitV2,
) -> Result<(), EntryAdmissionV2Error> {
    if permit.grant_digest() != grant.digest() {
        return Err(EntryAdmissionV2Error::GrantMismatch);
    }
    permit.effect_binding().validate_against(grant)?;
    if &permit.effect_binding().executor != executor_principal {
        return Err(EntryAdmissionV2Error::ExecutorPrincipalMismatch);
    }

    state.ensure_fresh(grant, time)?;

    if state.state_policy_digest() != permit.dispatch_state_policy_digest() {
        return Err(EntryAdmissionV2Error::StatePolicyChanged);
    }
    if state.time_policy_digest() != permit.dispatch_time_policy_digest()
        || time.policy_digest() != permit.dispatch_time_policy_digest()
    {
        return Err(EntryAdmissionV2Error::TimePolicyChanged);
    }

    let permit_state_sequence = permit.dispatch_authority_state_sequence();
    let current_state_sequence = state.state_sequence();
    if current_state_sequence < permit_state_sequence {
        return Err(EntryAdmissionV2Error::StateSequenceRollback {
            permit: permit_state_sequence,
            current: current_state_sequence,
        });
    }
    if current_state_sequence == permit_state_sequence
        && state.snapshot_digest() != permit.dispatch_authority_snapshot_digest()
    {
        return Err(EntryAdmissionV2Error::StateSequenceContradiction);
    }

    let (permit_frontier_sequence, permit_frontier_digest) =
        permit.dispatch_authority_source_frontier();
    let (current_frontier_sequence, current_frontier_digest) = state.source_frontier();
    if current_frontier_sequence < permit_frontier_sequence {
        return Err(EntryAdmissionV2Error::SourceFrontierRollback {
            permit: permit_frontier_sequence,
            current: current_frontier_sequence,
        });
    }
    if current_frontier_sequence == permit_frontier_sequence
        && current_frontier_digest != permit_frontier_digest
    {
        return Err(EntryAdmissionV2Error::SourceFrontierContradiction);
    }

    let current_authority_context = state
        .authority_context()
        .cloned()
        .ok_or(EntryAdmissionV2Error::NoCurrentAuthorityContext)?;
    let input = AuthorityEvaluationInput {
        now_unix_s: time.conservative_now_unix_s()?,
        current_epoch: state.authority_epoch(),
        current_authority_context,
        use_state: permit.allocation_use_state(),
    };
    match evaluate_authority(grant, &input, state.negative_facts()) {
        AuthorityDecision::Allow => Ok(()),
        AuthorityDecision::Deny(reason) => Err(EntryAdmissionV2Error::AuthorityDenied(reason)),
    }
}

#[derive(Debug, Error)]
pub enum EntryAdmissionV2Error {
    #[error("dispatch permit belongs to a different capability grant")]
    GrantMismatch,
    #[error("V2D effect binding no longer validates against the exact grant: {0}")]
    EffectBinding(#[from] AdmissionV2Error),
    #[error("provisioned executor principal does not match the permit executor binding")]
    ExecutorPrincipalMismatch,
    #[error("verified authority state failed point-of-entry freshness: {0}")]
    AuthorityState(#[from] AuthorityStateError),
    #[error("verified authority time failed point-of-entry freshness: {0}")]
    AuthorityTime(#[from] AuthorityTimeError),
    #[error("verified authority state reports no active authority context")]
    NoCurrentAuthorityContext,
    #[error("authority-state policy identity changed since permit mint")]
    StatePolicyChanged,
    #[error("authority-time policy identity changed since permit mint")]
    TimePolicyChanged,
    #[error("authority state sequence rolled back from {permit} to {current}")]
    StateSequenceRollback { permit: u64, current: u64 },
    #[error("same authority state sequence now names a different snapshot")]
    StateSequenceContradiction,
    #[error("authority source frontier rolled back from {permit} to {current}")]
    SourceFrontierRollback { permit: u64, current: u64 },
    #[error("same authority source-frontier sequence now names a different digest")]
    SourceFrontierContradiction,
    #[error("point-of-entry authority evaluation denied the effect: {0:?}")]
    AuthorityDenied(DenyReason),
}

#[derive(Debug, Error)]
pub enum EffectEntryV2Error<E>
where
    E: StdError + 'static,
{
    #[error("effect-entry admission failed: {0}")]
    Admission(#[from] EntryAdmissionV2Error),
    #[error("provisioned executor returned an error after point-of-entry admission: {0}")]
    Executor(#[source] E),
}

#[cfg(test)]
mod tests;
