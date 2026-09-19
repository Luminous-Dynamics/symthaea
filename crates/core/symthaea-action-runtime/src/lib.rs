// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-conservative accounting for one exact bounded-authority grant.
//!
//! This crate accounts for authority; it does not dispatch effects. The central
//! invariant is:
//!
//! ```text
//! committed + in-flight reservations + delegation escrow <= grant ceiling
//! ```
//!
//! Unknown external outcomes remain charged. v0.2 deliberately exposes no
//! transition that releases an `OutcomeUnknown` execution and no partial child
//! escrow refund. Those authority-increasing transitions require separately
//! verified reconciliation evidence in a later layer.

#![deny(unsafe_code)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use symthaea_authority::{
    AttenuationError, CapabilityGrant, Digest32, GrantUseState, GrantValidationError, RiskBudget,
};
use thiserror::Error;

pub const ACTION_RUNTIME_SCHEMA_VERSION: u16 = 2;
pub const MAX_ACCOUNT_IDENTIFIER_BYTES: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExecutionId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ReservationId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EscrowId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReservationState {
    Reserved,
    OutcomeUnknown,
    Committed,
    ReleasedBeforeDispatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionReservation {
    pub reservation_id: ReservationId,
    pub execution_id: ExecutionId,
    /// Exact immutable effect/admission intent this reservation was created for.
    pub effect_digest: Digest32,
    pub risk_charge: RiskBudget,
    pub state: ReservationState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscrowState {
    Open,
    OutcomeUnknown,
    ClosedFullyCharged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegationEscrow {
    pub escrow_id: EscrowId,
    /// Exact delegated child record whose full ceiling is reserved.
    pub child_grant_digest: Digest32,
    pub allocated_uses: u32,
    pub allocated_risk: RiskBudget,
    pub state: EscrowState,
    pub committed_uses: u32,
    pub committed_risk: RiskBudget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantAccountSnapshot {
    pub schema_version: u16,
    pub grant_digest: Digest32,
    pub max_uses: u32,
    pub risk_budget: RiskBudget,
    pub committed_uses: u32,
    pub committed_risk: RiskBudget,
    pub reservations: BTreeMap<ReservationId, ExecutionReservation>,
    pub escrows: BTreeMap<EscrowId, DelegationEscrow>,
}

#[derive(Debug)]
pub struct GrantAccount {
    snapshot: GrantAccountSnapshot,
}

impl GrantAccount {
    pub fn new(grant: &CapabilityGrant) -> Result<Self, RuntimeAccountingError> {
        grant.validate()?;
        let account = Self {
            snapshot: GrantAccountSnapshot {
                schema_version: ACTION_RUNTIME_SCHEMA_VERSION,
                grant_digest: grant.digest(),
                max_uses: grant.max_uses,
                risk_budget: grant.risk_budget,
                committed_uses: 0,
                committed_risk: RiskBudget::default(),
                reservations: BTreeMap::new(),
                escrows: BTreeMap::new(),
            },
        };
        account.validate_invariants_against_grant(grant)?;
        Ok(account)
    }

    pub fn from_snapshot(
        grant: &CapabilityGrant,
        snapshot: GrantAccountSnapshot,
    ) -> Result<Self, RuntimeAccountingError> {
        let account = Self { snapshot };
        account.validate_invariants_against_grant(grant)?;
        Ok(account)
    }

    pub fn snapshot(&self) -> GrantAccountSnapshot {
        self.snapshot.clone()
    }

    pub fn grant_digest(&self) -> Digest32 {
        self.snapshot.grant_digest
    }

    /// Diagnostic/unverified counters. These are not a proof of current durable
    /// accounting and must never be accepted as authority evidence by themselves.
    pub fn unverified_use_state(&self) -> Result<GrantUseState, RuntimeAccountingError> {
        Ok(GrantUseState {
            committed: self.snapshot.committed_uses,
            reserved: self.active_reserved_uses()?,
        })
    }

    pub fn unverified_charged_risk(&self) -> Result<RiskBudget, RuntimeAccountingError> {
        self.total_charged_risk()
    }

    pub fn remaining_use_capacity(&self) -> Result<u32, RuntimeAccountingError> {
        self.snapshot
            .max_uses
            .checked_sub(self.total_charged_uses()?)
            .ok_or(RuntimeAccountingError::InvariantViolation)
    }

    pub fn remaining_risk_capacity(&self) -> Result<RiskBudget, RuntimeAccountingError> {
        risk_checked_sub(self.snapshot.risk_budget, self.total_charged_risk()?)
            .ok_or(RuntimeAccountingError::InvariantViolation)
    }

    pub fn reserve_execution(
        &mut self,
        reservation_id: ReservationId,
        execution_id: ExecutionId,
        effect_digest: Digest32,
        risk_charge: RiskBudget,
    ) -> Result<(), RuntimeAccountingError> {
        validate_id(&reservation_id.0)?;
        validate_id(&execution_id.0)?;
        if effect_digest.0 == [0; 32] {
            return Err(RuntimeAccountingError::ZeroEffectDigest);
        }
        if self.snapshot.reservations.contains_key(&reservation_id) {
            return Err(RuntimeAccountingError::DuplicateReservation);
        }
        if self
            .snapshot
            .reservations
            .values()
            .any(|reservation| reservation.execution_id == execution_id)
        {
            return Err(RuntimeAccountingError::DuplicateExecutionId);
        }
        self.ensure_capacity(1, risk_charge)?;
        self.snapshot.reservations.insert(
            reservation_id.clone(),
            ExecutionReservation {
                reservation_id,
                execution_id,
                effect_digest,
                risk_charge,
                state: ReservationState::Reserved,
            },
        );
        self.validate_internal_invariants()
    }

    pub fn mark_outcome_unknown(
        &mut self,
        reservation_id: &ReservationId,
    ) -> Result<(), RuntimeAccountingError> {
        let reservation = self.reservation_mut(reservation_id)?;
        require_reservation_state(reservation.state, ReservationState::Reserved)?;
        reservation.state = ReservationState::OutcomeUnknown;
        self.validate_internal_invariants()
    }

    pub fn commit_observed(
        &mut self,
        reservation_id: &ReservationId,
    ) -> Result<(), RuntimeAccountingError> {
        self.commit_reservation_from(reservation_id, ReservationState::Reserved)
    }

    /// Conservative recovery: if an uncertain effect is established as applied,
    /// charge it permanently. No symmetric release exists in this schema.
    pub fn reconcile_applied(
        &mut self,
        reservation_id: &ReservationId,
    ) -> Result<(), RuntimeAccountingError> {
        self.commit_reservation_from(reservation_id, ReservationState::OutcomeUnknown)
    }

    /// Release is permitted only before dispatch/uncertainty is recorded.
    pub fn cancel_before_dispatch(
        &mut self,
        reservation_id: &ReservationId,
    ) -> Result<(), RuntimeAccountingError> {
        let reservation = self.reservation_mut(reservation_id)?;
        require_reservation_state(reservation.state, ReservationState::Reserved)?;
        reservation.state = ReservationState::ReleasedBeforeDispatch;
        self.validate_internal_invariants()
    }

    /// Reserve the exact full ceiling of an attenuated child grant.
    pub fn reserve_delegation_escrow(
        &mut self,
        parent: &CapabilityGrant,
        child: &CapabilityGrant,
        escrow_id: EscrowId,
    ) -> Result<(), RuntimeAccountingError> {
        self.require_grant(parent)?;
        child.validate_attenuation(parent)?;
        validate_id(&escrow_id.0)?;
        if self.snapshot.escrows.contains_key(&escrow_id) {
            return Err(RuntimeAccountingError::DuplicateEscrow);
        }
        let child_grant_digest = child.digest();
        if self
            .snapshot
            .escrows
            .values()
            .any(|escrow| escrow.child_grant_digest == child_grant_digest)
        {
            return Err(RuntimeAccountingError::DuplicateChildEscrow);
        }
        self.ensure_capacity(child.max_uses, child.risk_budget)?;
        self.snapshot.escrows.insert(
            escrow_id.clone(),
            DelegationEscrow {
                escrow_id,
                child_grant_digest,
                allocated_uses: child.max_uses,
                allocated_risk: child.risk_budget,
                state: EscrowState::Open,
                committed_uses: 0,
                committed_risk: RiskBudget::default(),
            },
        );
        self.validate_internal_invariants()
    }

    pub fn mark_escrow_outcome_unknown(
        &mut self,
        escrow_id: &EscrowId,
    ) -> Result<(), RuntimeAccountingError> {
        let escrow = self.escrow_mut(escrow_id)?;
        require_escrow_state(escrow.state, EscrowState::Open)?;
        escrow.state = EscrowState::OutcomeUnknown;
        self.validate_internal_invariants()
    }

    /// Close an ordinary child conservatively by charging its entire allocation.
    /// Returning unused child capacity requires a future verified reconciliation proof.
    pub fn close_escrow_fully_charged(
        &mut self,
        escrow_id: &EscrowId,
        child: &CapabilityGrant,
    ) -> Result<(), RuntimeAccountingError> {
        self.commit_escrow_from(escrow_id, child, EscrowState::Open)
    }

    /// Conservative recovery for an uncertain child: charge the entire allocation.
    pub fn reconcile_unknown_escrow_fully_applied(
        &mut self,
        escrow_id: &EscrowId,
        child: &CapabilityGrant,
    ) -> Result<(), RuntimeAccountingError> {
        self.commit_escrow_from(escrow_id, child, EscrowState::OutcomeUnknown)
    }

    pub fn validate_invariants_against_grant(
        &self,
        grant: &CapabilityGrant,
    ) -> Result<(), RuntimeAccountingError> {
        self.require_grant(grant)?;
        self.validate_internal_invariants()
    }

    fn require_grant(&self, grant: &CapabilityGrant) -> Result<(), RuntimeAccountingError> {
        grant.validate()?;
        if self.snapshot.schema_version != ACTION_RUNTIME_SCHEMA_VERSION {
            return Err(RuntimeAccountingError::UnsupportedSnapshotSchema);
        }
        if self.snapshot.grant_digest != grant.digest()
            || self.snapshot.max_uses != grant.max_uses
            || self.snapshot.risk_budget != grant.risk_budget
        {
            return Err(RuntimeAccountingError::SnapshotGrantMismatch);
        }
        Ok(())
    }

    fn commit_reservation_from(
        &mut self,
        reservation_id: &ReservationId,
        required_state: ReservationState,
    ) -> Result<(), RuntimeAccountingError> {
        self.validate_internal_invariants()?;
        let (risk_charge, state) = {
            let reservation = self.reservation_mut(reservation_id)?;
            (reservation.risk_charge, reservation.state)
        };
        require_reservation_state(state, required_state)?;

        let new_committed_uses = self
            .snapshot
            .committed_uses
            .checked_add(1)
            .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
        let new_committed_risk = risk_checked_add(self.snapshot.committed_risk, risk_charge)
            .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;

        self.snapshot.committed_uses = new_committed_uses;
        self.snapshot.committed_risk = new_committed_risk;
        self.reservation_mut(reservation_id)?.state = ReservationState::Committed;
        self.validate_internal_invariants()
    }

    fn commit_escrow_from(
        &mut self,
        escrow_id: &EscrowId,
        child: &CapabilityGrant,
        required_state: EscrowState,
    ) -> Result<(), RuntimeAccountingError> {
        self.validate_internal_invariants()?;
        child.validate()?;
        let (child_digest, allocated_uses, allocated_risk, state) = {
            let escrow = self.escrow_mut(escrow_id)?;
            (
                escrow.child_grant_digest,
                escrow.allocated_uses,
                escrow.allocated_risk,
                escrow.state,
            )
        };
        require_escrow_state(state, required_state)?;
        if child.digest() != child_digest
            || child.max_uses != allocated_uses
            || child.risk_budget != allocated_risk
        {
            return Err(RuntimeAccountingError::ChildGrantMismatch);
        }

        let new_committed_uses = self
            .snapshot
            .committed_uses
            .checked_add(allocated_uses)
            .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
        let new_committed_risk = risk_checked_add(self.snapshot.committed_risk, allocated_risk)
            .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;

        self.snapshot.committed_uses = new_committed_uses;
        self.snapshot.committed_risk = new_committed_risk;
        let escrow = self.escrow_mut(escrow_id)?;
        escrow.committed_uses = allocated_uses;
        escrow.committed_risk = allocated_risk;
        escrow.state = EscrowState::ClosedFullyCharged;
        self.validate_internal_invariants()
    }

    fn reservation_mut(
        &mut self,
        reservation_id: &ReservationId,
    ) -> Result<&mut ExecutionReservation, RuntimeAccountingError> {
        self.snapshot
            .reservations
            .get_mut(reservation_id)
            .ok_or(RuntimeAccountingError::ReservationNotFound)
    }

    fn escrow_mut(
        &mut self,
        escrow_id: &EscrowId,
    ) -> Result<&mut DelegationEscrow, RuntimeAccountingError> {
        self.snapshot
            .escrows
            .get_mut(escrow_id)
            .ok_or(RuntimeAccountingError::EscrowNotFound)
    }

    fn ensure_capacity(
        &self,
        additional_uses: u32,
        additional_risk: RiskBudget,
    ) -> Result<(), RuntimeAccountingError> {
        let uses = self
            .total_charged_uses()?
            .checked_add(additional_uses)
            .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
        if uses > self.snapshot.max_uses {
            return Err(RuntimeAccountingError::UseCapacityExceeded);
        }
        let risk = risk_checked_add(self.total_charged_risk()?, additional_risk)
            .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
        if !risk.attenuates(self.snapshot.risk_budget) {
            return Err(RuntimeAccountingError::RiskCapacityExceeded);
        }
        Ok(())
    }

    fn active_reserved_uses(&self) -> Result<u32, RuntimeAccountingError> {
        let mut total = 0u32;
        for reservation in self.snapshot.reservations.values() {
            if matches!(
                reservation.state,
                ReservationState::Reserved | ReservationState::OutcomeUnknown
            ) {
                total = total
                    .checked_add(1)
                    .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
            }
        }
        for escrow in self.snapshot.escrows.values() {
            if matches!(escrow.state, EscrowState::Open | EscrowState::OutcomeUnknown) {
                total = total
                    .checked_add(escrow.allocated_uses)
                    .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
            }
        }
        Ok(total)
    }

    fn total_charged_uses(&self) -> Result<u32, RuntimeAccountingError> {
        self.snapshot
            .committed_uses
            .checked_add(self.active_reserved_uses()?)
            .ok_or(RuntimeAccountingError::ArithmeticOverflow)
    }

    fn total_charged_risk(&self) -> Result<RiskBudget, RuntimeAccountingError> {
        let mut total = self.snapshot.committed_risk;
        for reservation in self.snapshot.reservations.values() {
            if matches!(
                reservation.state,
                ReservationState::Reserved | ReservationState::OutcomeUnknown
            ) {
                total = risk_checked_add(total, reservation.risk_charge)
                    .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
            }
        }
        for escrow in self.snapshot.escrows.values() {
            if matches!(escrow.state, EscrowState::Open | EscrowState::OutcomeUnknown) {
                total = risk_checked_add(total, escrow.allocated_risk)
                    .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
            }
        }
        Ok(total)
    }

    fn validate_internal_invariants(&self) -> Result<(), RuntimeAccountingError> {
        if self.snapshot.schema_version != ACTION_RUNTIME_SCHEMA_VERSION
            || self.snapshot.grant_digest.0 == [0; 32]
            || self.snapshot.max_uses == 0
        {
            return Err(RuntimeAccountingError::InvariantViolation);
        }

        let mut derived_uses = 0u32;
        let mut derived_risk = RiskBudget::default();
        let mut execution_ids = std::collections::BTreeSet::new();
        let mut child_grant_digests = std::collections::BTreeSet::new();

        for (key, reservation) in &self.snapshot.reservations {
            if key != &reservation.reservation_id {
                return Err(RuntimeAccountingError::ReservationKeyMismatch);
            }
            validate_id(&reservation.reservation_id.0)?;
            validate_id(&reservation.execution_id.0)?;
            if !execution_ids.insert(reservation.execution_id.clone()) {
                return Err(RuntimeAccountingError::DuplicateExecutionId);
            }
            if reservation.effect_digest.0 == [0; 32] {
                return Err(RuntimeAccountingError::ZeroEffectDigest);
            }
            if reservation.state == ReservationState::Committed {
                derived_uses = derived_uses
                    .checked_add(1)
                    .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
                derived_risk = risk_checked_add(derived_risk, reservation.risk_charge)
                    .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
            }
        }

        for (key, escrow) in &self.snapshot.escrows {
            if key != &escrow.escrow_id {
                return Err(RuntimeAccountingError::EscrowKeyMismatch);
            }
            validate_id(&escrow.escrow_id.0)?;
            if escrow.child_grant_digest.0 == [0; 32] || escrow.allocated_uses == 0 {
                return Err(RuntimeAccountingError::InvariantViolation);
            }
            if !child_grant_digests.insert(escrow.child_grant_digest) {
                return Err(RuntimeAccountingError::DuplicateChildEscrow);
            }
            match escrow.state {
                EscrowState::Open | EscrowState::OutcomeUnknown => {
                    if escrow.committed_uses != 0 || escrow.committed_risk != RiskBudget::default() {
                        return Err(RuntimeAccountingError::PrematureEscrowCommitment);
                    }
                }
                EscrowState::ClosedFullyCharged => {
                    if escrow.committed_uses != escrow.allocated_uses
                        || escrow.committed_risk != escrow.allocated_risk
                    {
                        return Err(RuntimeAccountingError::PartialEscrowRefundUnsupported);
                    }
                    derived_uses = derived_uses
                        .checked_add(escrow.committed_uses)
                        .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
                    derived_risk = risk_checked_add(derived_risk, escrow.committed_risk)
                        .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
                }
            }
        }

        if derived_uses != self.snapshot.committed_uses
            || derived_risk != self.snapshot.committed_risk
        {
            return Err(RuntimeAccountingError::CommittedAccountingMismatch);
        }
        if self.total_charged_uses()? > self.snapshot.max_uses
            || !self.total_charged_risk()?.attenuates(self.snapshot.risk_budget)
        {
            return Err(RuntimeAccountingError::InvariantViolation);
        }
        Ok(())
    }
}

fn validate_id(value: &str) -> Result<(), RuntimeAccountingError> {
    if value.is_empty() || value.len() > MAX_ACCOUNT_IDENTIFIER_BYTES {
        return Err(RuntimeAccountingError::InvalidIdentifier);
    }
    Ok(())
}

fn require_reservation_state(
    actual: ReservationState,
    required: ReservationState,
) -> Result<(), RuntimeAccountingError> {
    if actual == required {
        Ok(())
    } else {
        Err(RuntimeAccountingError::InvalidReservationTransition)
    }
}

fn require_escrow_state(
    actual: EscrowState,
    required: EscrowState,
) -> Result<(), RuntimeAccountingError> {
    if actual == required {
        Ok(())
    } else {
        Err(RuntimeAccountingError::InvalidEscrowTransition)
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

fn risk_checked_sub(left: RiskBudget, right: RiskBudget) -> Option<RiskBudget> {
    Some(RiskBudget {
        mutation_units: left.mutation_units.checked_sub(right.mutation_units)?,
        irreversible_units: left.irreversible_units.checked_sub(right.irreversible_units)?,
        external_disclosure_bytes: left
            .external_disclosure_bytes
            .checked_sub(right.external_disclosure_bytes)?,
        monetary_microunits: left
            .monetary_microunits
            .checked_sub(right.monetary_microunits)?,
    })
}

#[derive(Debug, Error)]
pub enum RuntimeAccountingError {
    #[error("capability grant is invalid: {0}")]
    InvalidGrant(#[from] GrantValidationError),
    #[error("delegated child is not a valid attenuation of its parent: {0}")]
    InvalidDelegation(#[from] AttenuationError),
    #[error("runtime-accounting snapshot schema is unsupported")]
    UnsupportedSnapshotSchema,
    #[error("runtime-accounting snapshot does not bind the supplied grant")]
    SnapshotGrantMismatch,
    #[error("accounting identifier is empty or exceeds the canonical bound")]
    InvalidIdentifier,
    #[error("execution reservation effect digest must not be zero")]
    ZeroEffectDigest,
    #[error("reservation id already exists")]
    DuplicateReservation,
    #[error("execution id already exists in this grant account")]
    DuplicateExecutionId,
    #[error("reservation id was not found")]
    ReservationNotFound,
    #[error("reservation transition is not allowed from the current state")]
    InvalidReservationTransition,
    #[error("delegation escrow id already exists")]
    DuplicateEscrow,
    #[error("this exact child grant already has delegation escrow in this parent account")]
    DuplicateChildEscrow,
    #[error("delegation escrow id was not found")]
    EscrowNotFound,
    #[error("delegation escrow transition is not allowed from the current state")]
    InvalidEscrowTransition,
    #[error("delegation escrow does not bind this exact child grant")]
    ChildGrantMismatch,
    #[error("requested use capacity exceeds the grant ceiling")]
    UseCapacityExceeded,
    #[error("requested risk capacity exceeds the grant ceiling")]
    RiskCapacityExceeded,
    #[error("accounting arithmetic overflow")]
    ArithmeticOverflow,
    #[error("reservation map key does not match its embedded id")]
    ReservationKeyMismatch,
    #[error("escrow map key does not match its embedded id")]
    EscrowKeyMismatch,
    #[error("open/uncertain escrow contains committed accounting")]
    PrematureEscrowCommitment,
    #[error("partial escrow refund is unsupported without verified reconciliation")]
    PartialEscrowRefundUnsupported,
    #[error("committed counters do not reconstruct from durable records")]
    CommittedAccountingMismatch,
    #[error("runtime-accounting invariant failed")]
    InvariantViolation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_authority::{
        AuthorityContextRef, AuthorityEpoch, Operation, PrincipalId, PurposeId, ResourceRef,
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

    fn parent_grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "parent",
            PrincipalId("issuer".into()),
            PrincipalId("robot".into()),
            PurposeId("goal-directed-actuation".into()),
            AuthorityEpoch(7),
            AuthorityContextRef::new("swarm-controller", digest(9)),
        );
        grant.resources.insert(ResourceRef("robot".into()));
        grant.operations.insert(Operation("move".into()));
        grant.max_uses = 4;
        grant.delegation_depth_remaining = 2;
        grant.risk_budget = risk(4);
        grant
    }

    fn child_grant(parent: &CapabilityGrant, id: &str) -> CapabilityGrant {
        let mut child = parent.clone();
        child.grant_id = id.into();
        child.issuer = parent.subject.clone();
        child.subject = PrincipalId(format!("worker-{id}"));
        child.parent_digest = Some(parent.digest());
        child.max_uses = 2;
        child.delegation_depth_remaining = 1;
        child.risk_budget = risk(2);
        child
    }

    #[test]
    fn unknown_execution_remains_fully_charged() {
        let grant = parent_grant();
        let mut account = GrantAccount::new(&grant).unwrap();
        let id = ReservationId("r1".into());
        account
            .reserve_execution(id.clone(), ExecutionId("e1".into()), digest(1), risk(1))
            .unwrap();
        assert_eq!(account.remaining_use_capacity().unwrap(), 3);
        account.mark_outcome_unknown(&id).unwrap();
        assert_eq!(account.remaining_use_capacity().unwrap(), 3);
        account.reconcile_applied(&id).unwrap();
        assert_eq!(account.remaining_use_capacity().unwrap(), 3);
    }

    #[test]
    fn cancellation_before_dispatch_releases_capacity() {
        let grant = parent_grant();
        let mut account = GrantAccount::new(&grant).unwrap();
        let id = ReservationId("r1".into());
        account
            .reserve_execution(id.clone(), ExecutionId("e1".into()), digest(1), risk(1))
            .unwrap();
        account.cancel_before_dispatch(&id).unwrap();
        assert_eq!(account.remaining_use_capacity().unwrap(), 4);
    }

    #[test]
    fn execution_ids_are_unique_within_a_grant_account() {
        let grant = parent_grant();
        let mut account = GrantAccount::new(&grant).unwrap();
        account
            .reserve_execution(
                ReservationId("r1".into()),
                ExecutionId("e1".into()),
                digest(1),
                risk(1),
            )
            .unwrap();
        assert!(matches!(
            account.reserve_execution(
                ReservationId("r2".into()),
                ExecutionId("e1".into()),
                digest(2),
                risk(1),
            ),
            Err(RuntimeAccountingError::DuplicateExecutionId)
        ));
    }

    #[test]
    fn delegation_escrow_binds_exact_child_digest() {
        let parent = parent_grant();
        let child_a = child_grant(&parent, "child-a");
        let child_b = child_grant(&parent, "child-b");
        assert_eq!(child_a.max_uses, child_b.max_uses);
        assert_eq!(child_a.risk_budget, child_b.risk_budget);
        assert_ne!(child_a.digest(), child_b.digest());

        let mut account = GrantAccount::new(&parent).unwrap();
        let escrow = EscrowId("escrow-1".into());
        account
            .reserve_delegation_escrow(&parent, &child_a, escrow.clone())
            .unwrap();
        assert!(matches!(
            account.close_escrow_fully_charged(&escrow, &child_b),
            Err(RuntimeAccountingError::ChildGrantMismatch)
        ));
        account
            .close_escrow_fully_charged(&escrow, &child_a)
            .unwrap();
    }

    #[test]
    fn exact_child_grant_can_only_have_one_escrow() {
        let parent = parent_grant();
        let child = child_grant(&parent, "child");
        let mut account = GrantAccount::new(&parent).unwrap();
        account
            .reserve_delegation_escrow(&parent, &child, EscrowId("e1".into()))
            .unwrap();
        assert!(matches!(
            account.reserve_delegation_escrow(&parent, &child, EscrowId("e2".into())),
            Err(RuntimeAccountingError::DuplicateChildEscrow)
        ));
    }

    #[test]
    fn broader_child_cannot_reserve_escrow() {
        let parent = parent_grant();
        let mut child = child_grant(&parent, "child");
        child.max_uses = parent.max_uses + 1;
        let mut account = GrantAccount::new(&parent).unwrap();
        assert!(matches!(
            account.reserve_delegation_escrow(&parent, &child, EscrowId("e".into())),
            Err(RuntimeAccountingError::InvalidDelegation(_))
        ));
    }

    #[test]
    fn unknown_child_can_only_be_reconciled_as_fully_charged_here() {
        let parent = parent_grant();
        let child = child_grant(&parent, "child");
        let mut account = GrantAccount::new(&parent).unwrap();
        let escrow = EscrowId("escrow".into());
        account
            .reserve_delegation_escrow(&parent, &child, escrow.clone())
            .unwrap();
        assert_eq!(account.remaining_use_capacity().unwrap(), 2);
        account.mark_escrow_outcome_unknown(&escrow).unwrap();
        account
            .reconcile_unknown_escrow_fully_applied(&escrow, &child)
            .unwrap();
        assert_eq!(account.remaining_use_capacity().unwrap(), 2);
    }

    #[test]
    fn snapshot_cannot_be_rehydrated_under_another_grant() {
        let grant = parent_grant();
        let snapshot = GrantAccount::new(&grant).unwrap().snapshot();
        let mut other = grant.clone();
        other.grant_id = "other".into();
        assert!(matches!(
            GrantAccount::from_snapshot(&other, snapshot),
            Err(RuntimeAccountingError::SnapshotGrantMismatch)
        ));
    }

    #[test]
    fn tampered_committed_counter_fails_reconstruction() {
        let grant = parent_grant();
        let mut snapshot = GrantAccount::new(&grant).unwrap().snapshot();
        snapshot.committed_uses = 1;
        assert!(matches!(
            GrantAccount::from_snapshot(&grant, snapshot),
            Err(RuntimeAccountingError::CommittedAccountingMismatch)
        ));
    }

    #[test]
    fn failed_commit_does_not_partially_mutate_account() {
        let grant = parent_grant();
        let mut account = GrantAccount::new(&grant).unwrap();
        let id = ReservationId("r1".into());
        account
            .reserve_execution(id.clone(), ExecutionId("e1".into()), digest(1), risk(1))
            .unwrap();

        // Corrupt the private in-memory state to force the checked-add failure
        // path and prove that the transition performs no partial successor write.
        account.snapshot.committed_risk.mutation_units = u64::MAX;
        let before = account.snapshot.clone();
        assert!(matches!(
            account.commit_observed(&id),
            Err(RuntimeAccountingError::ArithmeticOverflow)
                | Err(RuntimeAccountingError::CommittedAccountingMismatch)
                | Err(RuntimeAccountingError::InvariantViolation)
        ));
        assert_eq!(account.snapshot, before);
    }

    #[test]
    fn execution_reservation_binds_effect_digest() {
        let grant = parent_grant();
        let mut account = GrantAccount::new(&grant).unwrap();
        account
            .reserve_execution(
                ReservationId("r1".into()),
                ExecutionId("e1".into()),
                digest(77),
                risk(1),
            )
            .unwrap();
        assert_eq!(
            account
                .snapshot()
                .reservations
                .get(&ReservationId("r1".into()))
                .unwrap()
                .effect_digest,
            digest(77)
        );
    }
}
