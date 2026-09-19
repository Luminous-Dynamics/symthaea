// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-conservative grant-use accounting for Symthaea authority v0.2.
//!
//! This crate owns deterministic accounting mechanics only. It does not
//! authenticate current authority state, persist checkpoints, provide a CAS
//! store, mint dispatch authority, execute effects, or authenticate authority
//! refunds.
//!
//! Core separation:
//!
//! ```text
//! CapabilityGrant v2
//!     != GrantAccountSnapshotV2
//!     != GrantAccountV2
//!     != reserved use
//!     != durable reserved use
//!     != verified reconciliation
//!     != dispatch permit
//!     != effect
//! ```
//!
//! Persisted snapshots are ordinary data. Restoring an account always requires
//! the externally supplied exact `CapabilityGrant` v2 and full invariant
//! revalidation. The account is still not execution authority by itself.
//!
//! `ReservationState::Released` remains part of the frozen V2 representation so
//! a future verifier-owned reconciliation protocol can represent a proven
//! not-applied attempt. This crate intentionally exposes no ordinary transition
//! that creates `Released`: returning capacity is an authority-increasing act
//! and requires proof outside this deterministic accounting layer.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_authority::{
    CapabilityGrant, Digest32, GrantUseState, GrantValidationError, RiskBudget,
};
use thiserror::Error;

/// Schema version for the deterministic account representation.
pub const ACTION_RUNTIME_SCHEMA_VERSION: u16 = 2;
const RESERVATION_ID_DOMAIN: &[u8] = b"symthaea.action-runtime.reservation.v2\0";

/// Semantic consequence intended by the caller/domain adapter.
///
/// This identity may remain stable across a retry only after an earlier attempt
/// is independently proven not dispatched/applied by a future reconciliation
/// layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EffectIntentId(pub Digest32);

/// One concrete attempt to dispatch an effect intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AttemptId(pub Digest32);

/// Commitment to the exact domain-specific effect binding.
///
/// The domain adapter owns the canonical preimage (resource, operation,
/// purpose, task, plan/world bindings, parameters, etc.). This runtime does not
/// reinterpret those semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EffectBindingDigest(pub Digest32);

/// Deterministically derived identity of one grant-use reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ReservationId(pub Digest32);

/// Lifecycle of one exact use/risk reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReservationState {
    /// Capacity is allocated but no dispatch permit may yet exist.
    Reserved,
    /// Dispatch may have happened; the reservation stays fully charged.
    OutcomeUnknown,
    /// The effect is independently known to have been applied.
    Committed,
    /// Capacity returned only after a future verifier-owned exact-attempt
    /// reconciliation proves the effect was not applied. Ordinary runtime APIs
    /// in this schema cannot create this state.
    Released,
}

/// Persistable record for one exact attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionReservationV2 {
    pub reservation_id: ReservationId,
    pub effect_intent_id: EffectIntentId,
    pub attempt_id: AttemptId,
    pub effect_binding_digest: EffectBindingDigest,
    pub risk_charge: RiskBudget,
    pub state: ReservationState,
}

/// Persistable deterministic account state.
///
/// This is audit/recovery data only. It does not recreate a live/current
/// authority fact by itself; [`GrantAccountV2::restore`] requires the exact
/// external grant again, while trusted checkpoint/frontier lineage is separate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantAccountSnapshotV2 {
    pub schema_version: u16,
    pub grant_digest: Digest32,
    pub max_uses: u32,
    pub risk_budget: RiskBudget,
    pub reservations: BTreeMap<ReservationId, ExecutionReservationV2>,
}

/// In-process deterministic account for one exact root capability grant.
///
/// Deliberately not `Clone`, `Serialize`, or `Deserialize`: duplicating an
/// in-process account must not become a way to multiply authority state. Cross-
/// process durability/concurrency requires checkpoint/CAS tranches.
#[derive(Debug)]
pub struct GrantAccountV2 {
    snapshot: GrantAccountSnapshotV2,
}

impl GrantAccountV2 {
    /// Create an empty account for one exact authority-v0.2 root grant.
    ///
    /// Delegated records remain blocked until a separate verified delegation-
    /// chain theorem exists.
    pub fn new_root(grant: &CapabilityGrant) -> Result<Self, RuntimeV2Error> {
        validate_root_grant(grant)?;
        Ok(Self {
            snapshot: GrantAccountSnapshotV2 {
                schema_version: ACTION_RUNTIME_SCHEMA_VERSION,
                grant_digest: grant.digest(),
                max_uses: grant.max_uses,
                risk_budget: grant.risk_budget,
                reservations: BTreeMap::new(),
            },
        })
    }

    /// Restore ordinary persisted data only after rebinding it to the exact
    /// externally supplied root grant and re-deriving every invariant.
    ///
    /// A restored account remains ordinary deterministic state. In particular,
    /// this function does not prove checkpoint currentness or authorize a
    /// `Released` transition that was not already part of an independently
    /// qualified persisted lineage.
    pub fn restore(
        grant: &CapabilityGrant,
        snapshot: GrantAccountSnapshotV2,
    ) -> Result<Self, RuntimeV2Error> {
        validate_snapshot_against_grant(grant, &snapshot)?;
        Ok(Self { snapshot })
    }

    /// Return ordinary persistable evidence/state.
    pub fn snapshot(&self) -> GrantAccountSnapshotV2 {
        self.snapshot.clone()
    }

    pub fn grant_digest(&self) -> Digest32 {
        self.snapshot.grant_digest
    }

    /// Derive the pure authority-core use state from the exact account state.
    ///
    /// Higher verified-authority composition may derive from this exact account;
    /// callers must not substitute independently authored counters.
    pub fn authority_use_state(&self) -> Result<GrantUseState, RuntimeV2Error> {
        let mut committed = 0u32;
        let mut reserved = 0u32;
        for reservation in self.snapshot.reservations.values() {
            match reservation.state {
                ReservationState::Committed => {
                    committed = committed
                        .checked_add(1)
                        .ok_or(RuntimeV2Error::ArithmeticOverflow)?;
                }
                ReservationState::Reserved | ReservationState::OutcomeUnknown => {
                    reserved = reserved
                        .checked_add(1)
                        .ok_or(RuntimeV2Error::ArithmeticOverflow)?;
                }
                ReservationState::Released => {}
            }
        }
        Ok(GrantUseState {
            committed,
            reserved,
        })
    }

    /// Reserve exactly one use/risk allocation for one exact effect attempt.
    ///
    /// `ReservationId` is derived by this crate rather than caller-selected.
    pub fn reserve_execution(
        &mut self,
        effect_intent_id: EffectIntentId,
        attempt_id: AttemptId,
        effect_binding_digest: EffectBindingDigest,
        risk_charge: RiskBudget,
    ) -> Result<ReservationId, RuntimeV2Error> {
        validate_nonzero_id(effect_intent_id.0, RuntimeV2Error::ZeroEffectIntentId)?;
        validate_nonzero_id(attempt_id.0, RuntimeV2Error::ZeroAttemptId)?;
        validate_nonzero_id(
            effect_binding_digest.0,
            RuntimeV2Error::ZeroEffectBindingDigest,
        )?;

        if self
            .snapshot
            .reservations
            .values()
            .any(|reservation| reservation.attempt_id == attempt_id)
        {
            return Err(RuntimeV2Error::AttemptIdAlreadyUsed);
        }

        if self.snapshot.reservations.values().any(|reservation| {
            reservation.effect_intent_id == effect_intent_id
                && reservation.state != ReservationState::Released
        }) {
            return Err(RuntimeV2Error::EffectIntentAlreadyCharged);
        }

        self.ensure_capacity(1, risk_charge)?;

        let reservation_id = derive_reservation_id(
            self.snapshot.grant_digest,
            effect_intent_id,
            attempt_id,
            effect_binding_digest,
            risk_charge,
        );
        if self.snapshot.reservations.contains_key(&reservation_id) {
            return Err(RuntimeV2Error::DuplicateReservation);
        }

        self.snapshot.reservations.insert(
            reservation_id,
            ExecutionReservationV2 {
                reservation_id,
                effect_intent_id,
                attempt_id,
                effect_binding_digest,
                risk_charge,
                state: ReservationState::Reserved,
            },
        );
        self.validate_invariants()?;
        Ok(reservation_id)
    }

    /// Arm the conservative uncertainty state before any future dispatch permit
    /// is minted. `OutcomeUnknown` remains fully charged across crashes/restart.
    pub fn mark_outcome_unknown(
        &mut self,
        reservation_id: ReservationId,
    ) -> Result<(), RuntimeV2Error> {
        let reservation = self.reservation_mut(reservation_id)?;
        require_state(reservation.state, ReservationState::Reserved)?;
        reservation.state = ReservationState::OutcomeUnknown;
        self.validate_invariants()
    }

    /// Reconcile an uncertain attempt that is independently known to have been
    /// applied. A `Reserved` record cannot jump directly to `Committed`.
    ///
    /// This transition never returns capacity, so it is safe for this
    /// deterministic accounting layer to expose directly.
    pub fn reconcile_applied(
        &mut self,
        reservation_id: ReservationId,
    ) -> Result<(), RuntimeV2Error> {
        let reservation = self.reservation_mut(reservation_id)?;
        require_state(reservation.state, ReservationState::OutcomeUnknown)?;
        reservation.state = ReservationState::Committed;
        self.validate_invariants()
    }

    pub fn reservation(
        &self,
        reservation_id: ReservationId,
    ) -> Option<&ExecutionReservationV2> {
        self.snapshot.reservations.get(&reservation_id)
    }

    pub fn remaining_use_capacity(&self) -> Result<u32, RuntimeV2Error> {
        let charged = self.authority_use_state()?.charged();
        self.snapshot
            .max_uses
            .checked_sub(charged)
            .ok_or(RuntimeV2Error::InvariantViolation)
    }

    pub fn remaining_risk_capacity(&self) -> Result<RiskBudget, RuntimeV2Error> {
        risk_checked_sub(self.snapshot.risk_budget, self.total_charged_risk()?)
            .ok_or(RuntimeV2Error::InvariantViolation)
    }

    fn reservation_mut(
        &mut self,
        reservation_id: ReservationId,
    ) -> Result<&mut ExecutionReservationV2, RuntimeV2Error> {
        self.snapshot
            .reservations
            .get_mut(&reservation_id)
            .ok_or(RuntimeV2Error::UnknownReservation)
    }

    fn ensure_capacity(
        &self,
        additional_uses: u32,
        additional_risk: RiskBudget,
    ) -> Result<(), RuntimeV2Error> {
        let charged_uses = self.authority_use_state()?.charged();
        let requested_uses = charged_uses
            .checked_add(additional_uses)
            .ok_or(RuntimeV2Error::ArithmeticOverflow)?;
        if requested_uses > self.snapshot.max_uses {
            return Err(RuntimeV2Error::UseCapacityExceeded);
        }

        let requested_risk = risk_checked_add(self.total_charged_risk()?, additional_risk)
            .ok_or(RuntimeV2Error::ArithmeticOverflow)?;
        if !requested_risk.attenuates(self.snapshot.risk_budget) {
            return Err(RuntimeV2Error::RiskCapacityExceeded);
        }
        Ok(())
    }

    fn total_charged_risk(&self) -> Result<RiskBudget, RuntimeV2Error> {
        let mut total = RiskBudget::default();
        for reservation in self.snapshot.reservations.values() {
            if reservation.state != ReservationState::Released {
                total = risk_checked_add(total, reservation.risk_charge)
                    .ok_or(RuntimeV2Error::ArithmeticOverflow)?;
            }
        }
        Ok(total)
    }

    fn validate_invariants(&self) -> Result<(), RuntimeV2Error> {
        validate_snapshot_shape(&self.snapshot)
    }
}

fn validate_root_grant(grant: &CapabilityGrant) -> Result<(), RuntimeV2Error> {
    grant.validate().map_err(RuntimeV2Error::InvalidGrant)?;
    if grant.parent_digest.is_some() {
        return Err(RuntimeV2Error::DelegatedGrantRequiresVerifiedChain);
    }
    Ok(())
}

fn validate_snapshot_against_grant(
    grant: &CapabilityGrant,
    snapshot: &GrantAccountSnapshotV2,
) -> Result<(), RuntimeV2Error> {
    validate_root_grant(grant)?;
    if snapshot.schema_version != ACTION_RUNTIME_SCHEMA_VERSION {
        return Err(RuntimeV2Error::UnsupportedSnapshotSchema);
    }
    if snapshot.grant_digest != grant.digest() {
        return Err(RuntimeV2Error::GrantDigestMismatch);
    }
    if snapshot.max_uses != grant.max_uses || snapshot.risk_budget != grant.risk_budget {
        return Err(RuntimeV2Error::GrantCeilingMismatch);
    }
    validate_snapshot_shape(snapshot)
}

fn validate_snapshot_shape(snapshot: &GrantAccountSnapshotV2) -> Result<(), RuntimeV2Error> {
    if snapshot.schema_version != ACTION_RUNTIME_SCHEMA_VERSION {
        return Err(RuntimeV2Error::UnsupportedSnapshotSchema);
    }

    let mut attempts = BTreeSet::new();
    let mut charged_effects = BTreeSet::new();
    let mut charged_uses = 0u32;
    let mut charged_risk = RiskBudget::default();

    for (map_key, reservation) in &snapshot.reservations {
        if map_key != &reservation.reservation_id {
            return Err(RuntimeV2Error::ReservationKeyMismatch);
        }
        validate_nonzero_id(reservation.effect_intent_id.0, RuntimeV2Error::ZeroEffectIntentId)?;
        validate_nonzero_id(reservation.attempt_id.0, RuntimeV2Error::ZeroAttemptId)?;
        validate_nonzero_id(
            reservation.effect_binding_digest.0,
            RuntimeV2Error::ZeroEffectBindingDigest,
        )?;

        let expected_id = derive_reservation_id(
            snapshot.grant_digest,
            reservation.effect_intent_id,
            reservation.attempt_id,
            reservation.effect_binding_digest,
            reservation.risk_charge,
        );
        if reservation.reservation_id != expected_id {
            return Err(RuntimeV2Error::ReservationIdentityMismatch);
        }
        if !attempts.insert(reservation.attempt_id) {
            return Err(RuntimeV2Error::AttemptIdAlreadyUsed);
        }

        if reservation.state != ReservationState::Released {
            if !charged_effects.insert(reservation.effect_intent_id) {
                return Err(RuntimeV2Error::EffectIntentAlreadyCharged);
            }
            charged_uses = charged_uses
                .checked_add(1)
                .ok_or(RuntimeV2Error::ArithmeticOverflow)?;
            charged_risk = risk_checked_add(charged_risk, reservation.risk_charge)
                .ok_or(RuntimeV2Error::ArithmeticOverflow)?;
        }
    }

    if charged_uses > snapshot.max_uses || !charged_risk.attenuates(snapshot.risk_budget) {
        return Err(RuntimeV2Error::InvariantViolation);
    }
    Ok(())
}

fn validate_nonzero_id(
    value: Digest32,
    error: RuntimeV2Error,
) -> Result<(), RuntimeV2Error> {
    if value.0 == [0; 32] {
        Err(error)
    } else {
        Ok(())
    }
}

fn derive_reservation_id(
    grant_digest: Digest32,
    effect_intent_id: EffectIntentId,
    attempt_id: AttemptId,
    effect_binding_digest: EffectBindingDigest,
    risk_charge: RiskBudget,
) -> ReservationId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RESERVATION_ID_DOMAIN);
    hasher.update(&grant_digest.0);
    hasher.update(&(effect_intent_id.0).0);
    hasher.update(&(attempt_id.0).0);
    hasher.update(&(effect_binding_digest.0).0);
    hasher.update(&risk_charge.mutation_units.to_be_bytes());
    hasher.update(&risk_charge.irreversible_units.to_be_bytes());
    hasher.update(&risk_charge.external_disclosure_bytes.to_be_bytes());
    hasher.update(&risk_charge.monetary_microunits.to_be_bytes());
    ReservationId(Digest32(*hasher.finalize().as_bytes()))
}

fn require_state(
    actual: ReservationState,
    expected: ReservationState,
) -> Result<(), RuntimeV2Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(RuntimeV2Error::InvalidReservationTransition {
            from: actual,
            expected,
        })
    }
}

fn risk_checked_add(left: RiskBudget, right: RiskBudget) -> Option<RiskBudget> {
    Some(RiskBudget {
        mutation_units: left.mutation_units.checked_add(right.mutation_units)?,
        irreversible_units: left.irreversible_units.checked_add(right.irreversible_units)?,
        external_disclosure_bytes: left
            .external_disclosure_bytes
            .checked_add(right.external_disclosure_bytes)?,
        monetary_microunits: left
            .monetary_microunits
            .checked_add(right.monetary_microunits)?,
    })
}

fn risk_checked_sub(total: RiskBudget, charged: RiskBudget) -> Option<RiskBudget> {
    Some(RiskBudget {
        mutation_units: total.mutation_units.checked_sub(charged.mutation_units)?,
        irreversible_units: total
            .irreversible_units
            .checked_sub(charged.irreversible_units)?,
        external_disclosure_bytes: total
            .external_disclosure_bytes
            .checked_sub(charged.external_disclosure_bytes)?,
        monetary_microunits: total
            .monetary_microunits
            .checked_sub(charged.monetary_microunits)?,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RuntimeV2Error {
    #[error("invalid authority-v0.2 grant: {0}")]
    InvalidGrant(GrantValidationError),
    #[error("delegated grants require a separately verified delegation chain")]
    DelegatedGrantRequiresVerifiedChain,
    #[error("unsupported action-runtime snapshot schema")]
    UnsupportedSnapshotSchema,
    #[error("persisted account does not bind the externally supplied exact grant")]
    GrantDigestMismatch,
    #[error("persisted account ceiling does not match the externally supplied exact grant")]
    GrantCeilingMismatch,
    #[error("effect intent identity must not be the all-zero placeholder")]
    ZeroEffectIntentId,
    #[error("attempt identity must not be the all-zero placeholder")]
    ZeroAttemptId,
    #[error("effect binding identity must not be the all-zero placeholder")]
    ZeroEffectBindingDigest,
    #[error("attempt identity has already been used")]
    AttemptIdAlreadyUsed,
    #[error("the same semantic effect already has a charged attempt")]
    EffectIntentAlreadyCharged,
    #[error("derived reservation identity is already present")]
    DuplicateReservation,
    #[error("reservation map key does not match embedded identity")]
    ReservationKeyMismatch,
    #[error("reservation identity does not match its exact grant/effect/attempt/risk preimage")]
    ReservationIdentityMismatch,
    #[error("unknown reservation")]
    UnknownReservation,
    #[error("invalid reservation transition from {from:?}; expected {expected:?}")]
    InvalidReservationTransition {
        from: ReservationState,
        expected: ReservationState,
    },
    #[error("grant use capacity exceeded")]
    UseCapacityExceeded,
    #[error("grant risk capacity exceeded")]
    RiskCapacityExceeded,
    #[error("integer accounting overflow")]
    ArithmeticOverflow,
    #[error("account invariant violated")]
    InvariantViolation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use symthaea_authority::{
        AuthorityContextRef, AuthorityEpoch, Operation, PrincipalId, PurposeId, ResourceRef,
    };

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn intent(byte: u8) -> EffectIntentId {
        EffectIntentId(digest(byte))
    }

    fn attempt(byte: u8) -> AttemptId {
        AttemptId(digest(byte))
    }

    fn binding(byte: u8) -> EffectBindingDigest {
        EffectBindingDigest(digest(byte))
    }

    fn risk(units: u64) -> RiskBudget {
        RiskBudget {
            mutation_units: units,
            ..RiskBudget::default()
        }
    }

    fn grant(max_uses: u32, risk_budget: RiskBudget) -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "grant-v2",
            PrincipalId("issuer".into()),
            PrincipalId("subject".into()),
            PurposeId("goal-directed-actuation".into()),
            AuthorityEpoch(7),
            AuthorityContextRef::new("symthaea.test.authority-context.v1", digest(33)),
        );
        grant
            .resources
            .insert(ResourceRef("robot-1/joint-3".into()));
        grant
            .operations
            .insert(Operation("robot.servo.write".into()));
        grant.max_uses = max_uses;
        grant.risk_budget = risk_budget;
        grant
    }

    #[test]
    fn account_is_bound_to_exact_root_grant() {
        let grant = grant(2, risk(2));
        let account = GrantAccountV2::new_root(&grant).unwrap();
        assert_eq!(account.grant_digest(), grant.digest());
        assert_eq!(account.authority_use_state().unwrap(), GrantUseState::default());
    }

    #[test]
    fn delegated_grant_is_blocked_without_verified_chain() {
        let mut delegated = grant(2, risk(2));
        delegated.parent_digest = Some(digest(99));
        assert_eq!(
            GrantAccountV2::new_root(&delegated).unwrap_err(),
            RuntimeV2Error::DelegatedGrantRequiresVerifiedChain
        );
    }

    #[test]
    fn restore_requires_external_exact_grant() {
        let grant_a = grant(2, risk(2));
        let account = GrantAccountV2::new_root(&grant_a).unwrap();
        let snapshot = account.snapshot();

        let mut grant_b = grant_a.clone();
        grant_b.grant_id = "different-grant".into();
        assert_eq!(
            GrantAccountV2::restore(&grant_b, snapshot).unwrap_err(),
            RuntimeV2Error::GrantDigestMismatch
        );
    }

    #[test]
    fn snapshot_cannot_rewrite_its_own_ceiling() {
        let grant = grant(2, risk(2));
        let account = GrantAccountV2::new_root(&grant).unwrap();
        let mut snapshot = account.snapshot();
        snapshot.max_uses = 20;
        assert_eq!(
            GrantAccountV2::restore(&grant, snapshot).unwrap_err(),
            RuntimeV2Error::GrantCeilingMismatch
        );
    }

    #[test]
    fn one_use_grant_reserves_exactly_one_use() {
        let grant = grant(1, risk(1));
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        account
            .reserve_execution(intent(1), attempt(11), binding(21), risk(1))
            .unwrap();
        assert_eq!(
            account.authority_use_state().unwrap(),
            GrantUseState {
                committed: 0,
                reserved: 1,
            }
        );
        assert_eq!(
            account.reserve_execution(intent(2), attempt(12), binding(22), RiskBudget::default()),
            Err(RuntimeV2Error::UseCapacityExceeded)
        );
    }

    #[test]
    fn unknown_attempt_blocks_retry_of_same_semantic_effect() {
        let grant = grant(2, risk(2));
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let first = account
            .reserve_execution(intent(1), attempt(11), binding(21), risk(1))
            .unwrap();
        account.mark_outcome_unknown(first).unwrap();
        assert_eq!(
            account.reserve_execution(intent(1), attempt(12), binding(21), risk(1)),
            Err(RuntimeV2Error::EffectIntentAlreadyCharged)
        );
    }

    #[test]
    fn attempt_identity_cannot_be_reused_while_original_attempt_is_charged() {
        let grant = grant(2, risk(2));
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        account
            .reserve_execution(intent(1), attempt(11), binding(21), risk(1))
            .unwrap();
        assert_eq!(
            account.reserve_execution(intent(2), attempt(11), binding(22), risk(1)),
            Err(RuntimeV2Error::AttemptIdAlreadyUsed)
        );
    }

    #[test]
    fn reserved_cannot_jump_directly_to_committed() {
        let grant = grant(1, risk(1));
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let id = account
            .reserve_execution(intent(1), attempt(11), binding(21), risk(1))
            .unwrap();
        assert_eq!(
            account.reconcile_applied(id),
            Err(RuntimeV2Error::InvalidReservationTransition {
                from: ReservationState::Reserved,
                expected: ReservationState::OutcomeUnknown,
            })
        );
    }

    #[test]
    fn outcome_unknown_stays_charged_then_reconciles_to_committed() {
        let grant = grant(1, risk(1));
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let id = account
            .reserve_execution(intent(1), attempt(11), binding(21), risk(1))
            .unwrap();
        account.mark_outcome_unknown(id).unwrap();
        assert_eq!(account.authority_use_state().unwrap().reserved, 1);
        account.reconcile_applied(id).unwrap();
        assert_eq!(
            account.authority_use_state().unwrap(),
            GrantUseState {
                committed: 1,
                reserved: 0,
            }
        );
        assert_eq!(account.remaining_risk_capacity().unwrap(), RiskBudget::default());
    }

    #[test]
    fn tampered_reservation_map_key_fails_restore() {
        let grant = grant(1, risk(1));
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let id = account
            .reserve_execution(intent(1), attempt(11), binding(21), risk(1))
            .unwrap();
        let mut snapshot = account.snapshot();
        let reservation = snapshot.reservations.remove(&id).unwrap();
        snapshot
            .reservations
            .insert(ReservationId(digest(77)), reservation);
        assert_eq!(
            GrantAccountV2::restore(&grant, snapshot).unwrap_err(),
            RuntimeV2Error::ReservationKeyMismatch
        );
    }

    #[test]
    fn tampered_effect_binding_breaks_reservation_identity() {
        let grant = grant(1, risk(1));
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let id = account
            .reserve_execution(intent(1), attempt(11), binding(21), risk(1))
            .unwrap();
        let mut snapshot = account.snapshot();
        snapshot
            .reservations
            .get_mut(&id)
            .unwrap()
            .effect_binding_digest = binding(88);
        assert_eq!(
            GrantAccountV2::restore(&grant, snapshot).unwrap_err(),
            RuntimeV2Error::ReservationIdentityMismatch
        );
    }

    #[test]
    fn released_snapshot_is_data_not_a_runtime_refund_proof() {
        let grant = grant(1, risk(1));
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let id = account
            .reserve_execution(intent(1), attempt(11), binding(21), risk(1))
            .unwrap();
        account.mark_outcome_unknown(id).unwrap();

        let mut snapshot = account.snapshot();
        snapshot.reservations.get_mut(&id).unwrap().state = ReservationState::Released;
        let restored = GrantAccountV2::restore(&grant, snapshot).unwrap();

        // Structural restore deliberately does not authenticate how Released was
        // earned. The strict checkpoint/frontier layer must reject an ordinary
        // OutcomeUnknown -> Released successor; future verified reconciliation
        // will own that transition.
        assert_eq!(restored.authority_use_state().unwrap(), GrantUseState::default());
    }

    proptest! {
        #[test]
        fn repeated_distinct_reservations_never_exceed_use_ceiling(
            max_uses in 1u32..24,
            attempts_count in 1u32..48,
        ) {
            let grant = grant(max_uses, RiskBudget {
                mutation_units: 1_000,
                ..RiskBudget::default()
            });
            let mut account = GrantAccountV2::new_root(&grant).unwrap();
            for index in 0..attempts_count {
                let byte = (index % 240 + 1) as u8;
                let _ = account.reserve_execution(
                    intent(byte),
                    attempt(byte.wrapping_add(1).max(1)),
                    binding(byte.wrapping_add(2).max(1)),
                    RiskBudget::default(),
                );
                prop_assert!(account.authority_use_state().unwrap().charged() <= max_uses);
            }
        }
    }
}
