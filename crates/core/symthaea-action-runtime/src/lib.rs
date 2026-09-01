// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-conservative action accounting for bounded Symthaea agency.
//!
//! `symthaea-authority` defines what a grant means. This crate defines how a
//! runtime consumes that grant without multiplying authority under concurrency,
//! delegation, crashes, or uncertain external outcomes.
//!
//! The central accounting invariant is:
//!
//! ```text
//! committed + in-flight reservations + delegation escrow <= grant ceiling
//! ```
//!
//! Unknown outcomes remain charged until an explicit reconciliation transition
//! proves whether the external effect occurred. Delegation escrow reserves the
//! full child ceiling up front, so spawning workers cannot multiply a parent's
//! remaining authority.

#![deny(unsafe_code)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use symthaea_authority::{CapabilityGrant, Digest32, GrantUseState, RiskBudget};
use thiserror::Error;

/// Stable caller-supplied execution identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExecutionId(pub String);

/// Stable caller-supplied reservation identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ReservationId(pub String);

/// Stable caller-supplied delegation escrow identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EscrowId(pub String);

/// Lifecycle of one execution reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReservationState {
    /// Capacity reserved before dispatch.
    Reserved,
    /// Dispatch may have taken effect, but the runtime cannot yet prove the result.
    OutcomeUnknown,
    /// The external effect is known to have occurred and is durably charged.
    Committed,
    /// The reservation is known not to have produced an external effect.
    Released,
}

/// One use plus its cumulative consequence charge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionReservation {
    pub reservation_id: ReservationId,
    pub execution_id: ExecutionId,
    pub risk_charge: RiskBudget,
    pub state: ReservationState,
}

/// Lifecycle of a child-agent delegation escrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscrowState {
    /// Full child ceiling is reserved from the parent.
    Open,
    /// Child state is uncertain; the full ceiling remains charged.
    OutcomeUnknown,
    /// Child execution has been reconciled and unused capacity returned.
    Closed,
}

/// Parent-side reservation of authority for a delegated worker/task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegationEscrow {
    pub escrow_id: EscrowId,
    pub allocated_uses: u32,
    pub allocated_risk: RiskBudget,
    pub state: EscrowState,
    /// Known child uses once closed.
    pub committed_uses: u32,
    /// Known child risk once closed.
    pub committed_risk: RiskBudget,
}

/// Deterministic account snapshot suitable for durable persistence by a higher layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantAccountSnapshot {
    pub grant_digest: Digest32,
    pub max_uses: u32,
    pub risk_budget: RiskBudget,
    pub committed_uses: u32,
    pub committed_risk: RiskBudget,
    pub reservations: BTreeMap<ReservationId, ExecutionReservation>,
    pub escrows: BTreeMap<EscrowId, DelegationEscrow>,
}

/// In-memory reference state machine for one capability grant.
///
/// Persistence, locking across processes, signatures, and distributed consensus
/// belong above this crate. Callers must durably persist reservation transitions
/// before dispatch if they want crash safety across process/host failure.
#[derive(Debug, Clone)]
pub struct GrantAccount {
    snapshot: GrantAccountSnapshot,
}

impl GrantAccount {
    pub fn new(grant: &CapabilityGrant) -> Self {
        Self {
            snapshot: GrantAccountSnapshot {
                grant_digest: grant.digest(),
                max_uses: grant.max_uses,
                risk_budget: grant.risk_budget,
                committed_uses: 0,
                committed_risk: RiskBudget::default(),
                reservations: BTreeMap::new(),
                escrows: BTreeMap::new(),
            },
        }
    }

    /// Rehydrate a previously persisted account after validating its invariants.
    pub fn from_snapshot(snapshot: GrantAccountSnapshot) -> Result<Self, RuntimeAccountingError> {
        let account = Self { snapshot };
        account.validate_invariants()?;
        Ok(account)
    }

    pub fn snapshot(&self) -> GrantAccountSnapshot {
        self.snapshot.clone()
    }

    /// Authority-kernel compatible use accounting.
    ///
    /// Open/unknown delegation escrow is charged at its *allocated* use ceiling,
    /// not its current observed usage, which prevents authority multiplication.
    pub fn authority_use_state(&self) -> GrantUseState {
        GrantUseState {
            committed: self.snapshot.committed_uses,
            reserved: self.active_reserved_uses(),
        }
    }

    /// Reserve one execution before dispatch.
    pub fn reserve_execution(
        &mut self,
        reservation_id: ReservationId,
        execution_id: ExecutionId,
        risk_charge: RiskBudget,
    ) -> Result<(), RuntimeAccountingError> {
        if self.snapshot.reservations.contains_key(&reservation_id) {
            return Err(RuntimeAccountingError::DuplicateReservation);
        }

        self.ensure_capacity(1, risk_charge)?;
        self.snapshot.reservations.insert(
            reservation_id.clone(),
            ExecutionReservation {
                reservation_id,
                execution_id,
                risk_charge,
                state: ReservationState::Reserved,
            },
        );
        self.validate_invariants()
    }

    /// Mark a dispatched execution as uncertain. It remains fully charged.
    pub fn mark_outcome_unknown(
        &mut self,
        reservation_id: &ReservationId,
    ) -> Result<(), RuntimeAccountingError> {
        let reservation = self.reservation_mut(reservation_id)?;
        require_reservation_state(reservation.state, ReservationState::Reserved)?;
        reservation.state = ReservationState::OutcomeUnknown;
        self.validate_invariants()
    }

    /// Normal success path: the external effect is known to have occurred.
    pub fn commit_observed(
        &mut self,
        reservation_id: &ReservationId,
    ) -> Result<(), RuntimeAccountingError> {
        self.commit_reservation_from(reservation_id, ReservationState::Reserved)
    }

    /// Recovery path: an unknown execution is proven to have taken effect.
    pub fn reconcile_applied(
        &mut self,
        reservation_id: &ReservationId,
    ) -> Result<(), RuntimeAccountingError> {
        self.commit_reservation_from(reservation_id, ReservationState::OutcomeUnknown)
    }

    /// Cancel work that is known not to have been dispatched.
    pub fn cancel_before_dispatch(
        &mut self,
        reservation_id: &ReservationId,
    ) -> Result<(), RuntimeAccountingError> {
        let reservation = self.reservation_mut(reservation_id)?;
        require_reservation_state(reservation.state, ReservationState::Reserved)?;
        reservation.state = ReservationState::Released;
        self.validate_invariants()
    }

    /// Recovery path: an unknown execution is independently proven not applied.
    pub fn reconcile_not_applied(
        &mut self,
        reservation_id: &ReservationId,
    ) -> Result<(), RuntimeAccountingError> {
        let reservation = self.reservation_mut(reservation_id)?;
        require_reservation_state(reservation.state, ReservationState::OutcomeUnknown)?;
        reservation.state = ReservationState::Released;
        self.validate_invariants()
    }

    /// Reserve a fixed child ceiling before delegating work.
    ///
    /// The full allocation stays charged while the escrow is open or uncertain.
    pub fn reserve_delegation_escrow(
        &mut self,
        escrow_id: EscrowId,
        allocated_uses: u32,
        allocated_risk: RiskBudget,
    ) -> Result<(), RuntimeAccountingError> {
        if allocated_uses == 0 {
            return Err(RuntimeAccountingError::ZeroEscrowUses);
        }
        if self.snapshot.escrows.contains_key(&escrow_id) {
            return Err(RuntimeAccountingError::DuplicateEscrow);
        }
        self.ensure_capacity(allocated_uses, allocated_risk)?;
        self.snapshot.escrows.insert(
            escrow_id.clone(),
            DelegationEscrow {
                escrow_id,
                allocated_uses,
                allocated_risk,
                state: EscrowState::Open,
                committed_uses: 0,
                committed_risk: RiskBudget::default(),
            },
        );
        self.validate_invariants()
    }

    /// Mark child state uncertain. Full allocated authority remains charged.
    pub fn mark_escrow_outcome_unknown(
        &mut self,
        escrow_id: &EscrowId,
    ) -> Result<(), RuntimeAccountingError> {
        let escrow = self.escrow_mut(escrow_id)?;
        require_escrow_state(escrow.state, EscrowState::Open)?;
        escrow.state = EscrowState::OutcomeUnknown;
        self.validate_invariants()
    }

    /// Close a normally completed child escrow and return unused authority.
    pub fn close_escrow(
        &mut self,
        escrow_id: &EscrowId,
        committed_uses: u32,
        committed_risk: RiskBudget,
    ) -> Result<(), RuntimeAccountingError> {
        self.close_escrow_from(
            escrow_id,
            EscrowState::Open,
            committed_uses,
            committed_risk,
        )
    }

    /// Reconcile an uncertain child and return only capacity proven unused.
    pub fn reconcile_escrow(
        &mut self,
        escrow_id: &EscrowId,
        committed_uses: u32,
        committed_risk: RiskBudget,
    ) -> Result<(), RuntimeAccountingError> {
        self.close_escrow_from(
            escrow_id,
            EscrowState::OutcomeUnknown,
            committed_uses,
            committed_risk,
        )
    }

    pub fn remaining_use_capacity(&self) -> u32 {
        self.snapshot
            .max_uses
            .saturating_sub(self.total_charged_uses())
    }

    pub fn remaining_risk_capacity(&self) -> Result<RiskBudget, RuntimeAccountingError> {
        risk_checked_sub(self.snapshot.risk_budget, self.total_charged_risk()?)
            .ok_or(RuntimeAccountingError::InvariantViolation)
    }

    fn commit_reservation_from(
        &mut self,
        reservation_id: &ReservationId,
        required_state: ReservationState,
    ) -> Result<(), RuntimeAccountingError> {
        let (risk_charge, state) = {
            let reservation = self.reservation_mut(reservation_id)?;
            (reservation.risk_charge, reservation.state)
        };
        require_reservation_state(state, required_state)?;

        self.snapshot.committed_uses = self
            .snapshot
            .committed_uses
            .checked_add(1)
            .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
        self.snapshot.committed_risk = risk_checked_add(self.snapshot.committed_risk, risk_charge)
            .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
        self.reservation_mut(reservation_id)?.state = ReservationState::Committed;
        self.validate_invariants()
    }

    fn close_escrow_from(
        &mut self,
        escrow_id: &EscrowId,
        required_state: EscrowState,
        committed_uses: u32,
        committed_risk: RiskBudget,
    ) -> Result<(), RuntimeAccountingError> {
        let (allocated_uses, allocated_risk, state) = {
            let escrow = self.escrow_mut(escrow_id)?;
            (escrow.allocated_uses, escrow.allocated_risk, escrow.state)
        };
        require_escrow_state(state, required_state)?;
        if committed_uses > allocated_uses {
            return Err(RuntimeAccountingError::EscrowUsesExceeded);
        }
        if !committed_risk.attenuates(allocated_risk) {
            return Err(RuntimeAccountingError::EscrowRiskExceeded);
        }

        self.snapshot.committed_uses = self
            .snapshot
            .committed_uses
            .checked_add(committed_uses)
            .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
        self.snapshot.committed_risk =
            risk_checked_add(self.snapshot.committed_risk, committed_risk)
                .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;

        let escrow = self.escrow_mut(escrow_id)?;
        escrow.committed_uses = committed_uses;
        escrow.committed_risk = committed_risk;
        escrow.state = EscrowState::Closed;
        self.validate_invariants()
    }

    fn ensure_capacity(
        &self,
        additional_uses: u32,
        additional_risk: RiskBudget,
    ) -> Result<(), RuntimeAccountingError> {
        let total_uses = self
            .total_charged_uses()
            .checked_add(additional_uses)
            .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
        if total_uses > self.snapshot.max_uses {
            return Err(RuntimeAccountingError::UseCapacityExceeded);
        }

        let charged_risk = self.total_charged_risk()?;
        let requested_total = risk_checked_add(charged_risk, additional_risk)
            .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
        if !requested_total.attenuates(self.snapshot.risk_budget) {
            return Err(RuntimeAccountingError::RiskCapacityExceeded);
        }
        Ok(())
    }

    fn active_reserved_uses(&self) -> u32 {
        let execution_uses = self
            .snapshot
            .reservations
            .values()
            .filter(|reservation| {
                matches!(
                    reservation.state,
                    ReservationState::Reserved | ReservationState::OutcomeUnknown
                )
            })
            .count() as u32;
        let escrow_uses = self
            .snapshot
            .escrows
            .values()
            .filter(|escrow| matches!(escrow.state, EscrowState::Open | EscrowState::OutcomeUnknown))
            .fold(0u32, |sum, escrow| sum.saturating_add(escrow.allocated_uses));
        execution_uses.saturating_add(escrow_uses)
    }

    fn total_charged_uses(&self) -> u32 {
        self.snapshot
            .committed_uses
            .saturating_add(self.active_reserved_uses())
    }

    fn total_charged_risk(&self) -> Result<RiskBudget, RuntimeAccountingError> {
        let mut total = self.snapshot.committed_risk;
        for reservation in self.snapshot.reservations.values().filter(|reservation| {
            matches!(
                reservation.state,
                ReservationState::Reserved | ReservationState::OutcomeUnknown
            )
        }) {
            total = risk_checked_add(total, reservation.risk_charge)
                .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
        }
        for escrow in self.snapshot.escrows.values().filter(|escrow| {
            matches!(escrow.state, EscrowState::Open | EscrowState::OutcomeUnknown)
        }) {
            total = risk_checked_add(total, escrow.allocated_risk)
                .ok_or(RuntimeAccountingError::ArithmeticOverflow)?;
        }
        Ok(total)
    }

    fn validate_invariants(&self) -> Result<(), RuntimeAccountingError> {
        if self.total_charged_uses() > self.snapshot.max_uses {
            return Err(RuntimeAccountingError::InvariantViolation);
        }
        let risk = self.total_charged_risk()?;
        if !risk.attenuates(self.snapshot.risk_budget) {
            return Err(RuntimeAccountingError::InvariantViolation);
        }
        Ok(())
    }

    fn reservation_mut(
        &mut self,
        reservation_id: &ReservationId,
    ) -> Result<&mut ExecutionReservation, RuntimeAccountingError> {
        self.snapshot
            .reservations
            .get_mut(reservation_id)
            .ok_or(RuntimeAccountingError::UnknownReservation)
    }

    fn escrow_mut(
        &mut self,
        escrow_id: &EscrowId,
    ) -> Result<&mut DelegationEscrow, RuntimeAccountingError> {
        self.snapshot
            .escrows
            .get_mut(escrow_id)
            .ok_or(RuntimeAccountingError::UnknownEscrow)
    }
}

fn require_reservation_state(
    actual: ReservationState,
    expected: ReservationState,
) -> Result<(), RuntimeAccountingError> {
    if actual == expected {
        Ok(())
    } else {
        Err(RuntimeAccountingError::InvalidReservationTransition {
            from: actual,
            expected,
        })
    }
}

fn require_escrow_state(
    actual: EscrowState,
    expected: EscrowState,
) -> Result<(), RuntimeAccountingError> {
    if actual == expected {
        Ok(())
    } else {
        Err(RuntimeAccountingError::InvalidEscrowTransition {
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
pub enum RuntimeAccountingError {
    #[error("duplicate execution reservation id")]
    DuplicateReservation,
    #[error("unknown execution reservation")]
    UnknownReservation,
    #[error("invalid execution reservation transition from {from:?}; expected {expected:?}")]
    InvalidReservationTransition {
        from: ReservationState,
        expected: ReservationState,
    },
    #[error("use capacity exceeded")]
    UseCapacityExceeded,
    #[error("risk capacity exceeded")]
    RiskCapacityExceeded,
    #[error("delegation escrow must reserve at least one use")]
    ZeroEscrowUses,
    #[error("duplicate delegation escrow id")]
    DuplicateEscrow,
    #[error("unknown delegation escrow")]
    UnknownEscrow,
    #[error("invalid delegation escrow transition from {from:?}; expected {expected:?}")]
    InvalidEscrowTransition {
        from: EscrowState,
        expected: EscrowState,
    },
    #[error("child committed more uses than its escrow allocation")]
    EscrowUsesExceeded,
    #[error("child committed more risk than its escrow allocation")]
    EscrowRiskExceeded,
    #[error("integer accounting overflow")]
    ArithmeticOverflow,
    #[error("persisted accounting state violates the grant ceiling")]
    InvariantViolation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use symthaea_authority::{AuthorityEpoch, PrincipalId};

    fn grant(max_uses: u32, risk: RiskBudget) -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "grant-1",
            PrincipalId("issuer".into()),
            PrincipalId("subject".into()),
            AuthorityEpoch(7),
        );
        grant.max_uses = max_uses;
        grant.risk_budget = risk;
        grant
    }

    fn risk(mutation_units: u64) -> RiskBudget {
        RiskBudget {
            mutation_units,
            ..RiskBudget::default()
        }
    }

    #[test]
    fn unknown_outcome_remains_charged_until_reconciled() {
        let grant = grant(1, risk(1));
        let mut account = GrantAccount::new(&grant);
        let id = ReservationId("r1".into());
        account
            .reserve_execution(id.clone(), ExecutionId("e1".into()), risk(1))
            .unwrap();
        account.mark_outcome_unknown(&id).unwrap();
        assert_eq!(account.authority_use_state().reserved, 1);
        assert_eq!(
            account.reserve_execution(
                ReservationId("r2".into()),
                ExecutionId("e2".into()),
                RiskBudget::default(),
            ),
            Err(RuntimeAccountingError::UseCapacityExceeded)
        );
        account.reconcile_not_applied(&id).unwrap();
        assert_eq!(account.remaining_use_capacity(), 1);
    }

    #[test]
    fn applied_unknown_moves_from_reserved_to_committed_without_double_charge() {
        let grant = grant(1, risk(1));
        let mut account = GrantAccount::new(&grant);
        let id = ReservationId("r1".into());
        account
            .reserve_execution(id.clone(), ExecutionId("e1".into()), risk(1))
            .unwrap();
        account.mark_outcome_unknown(&id).unwrap();
        account.reconcile_applied(&id).unwrap();
        assert_eq!(account.authority_use_state(), GrantUseState { committed: 1, reserved: 0 });
        assert_eq!(account.remaining_risk_capacity().unwrap(), RiskBudget::default());
    }

    #[test]
    fn delegation_escrow_prevents_multi_agent_authority_multiplication() {
        let grant = grant(6, risk(6));
        let mut account = GrantAccount::new(&grant);
        account
            .reserve_delegation_escrow(EscrowId("child-a".into()), 2, risk(2))
            .unwrap();
        account
            .reserve_delegation_escrow(EscrowId("child-b".into()), 3, risk(3))
            .unwrap();
        assert_eq!(account.remaining_use_capacity(), 1);
        assert_eq!(account.remaining_risk_capacity().unwrap(), risk(1));
        assert_eq!(
            account.reserve_delegation_escrow(EscrowId("child-c".into()), 2, risk(2)),
            Err(RuntimeAccountingError::UseCapacityExceeded)
        );
    }

    #[test]
    fn closing_escrow_commits_used_capacity_and_returns_unused_capacity() {
        let grant = grant(5, risk(5));
        let mut account = GrantAccount::new(&grant);
        let escrow = EscrowId("child".into());
        account
            .reserve_delegation_escrow(escrow.clone(), 4, risk(4))
            .unwrap();
        account.close_escrow(&escrow, 2, risk(2)).unwrap();
        assert_eq!(account.authority_use_state(), GrantUseState { committed: 2, reserved: 0 });
        assert_eq!(account.remaining_use_capacity(), 3);
        assert_eq!(account.remaining_risk_capacity().unwrap(), risk(3));
    }

    #[test]
    fn uncertain_escrow_keeps_full_allocation_charged() {
        let grant = grant(4, risk(4));
        let mut account = GrantAccount::new(&grant);
        let escrow = EscrowId("child".into());
        account
            .reserve_delegation_escrow(escrow.clone(), 3, risk(3))
            .unwrap();
        account.mark_escrow_outcome_unknown(&escrow).unwrap();
        assert_eq!(account.remaining_use_capacity(), 1);
        assert_eq!(account.remaining_risk_capacity().unwrap(), risk(1));
        account.reconcile_escrow(&escrow, 1, risk(1)).unwrap();
        assert_eq!(account.remaining_use_capacity(), 3);
    }

    #[test]
    fn corrupted_snapshot_fails_closed() {
        let grant = grant(1, risk(1));
        let account = GrantAccount::new(&grant);
        let mut snapshot = account.snapshot();
        snapshot.committed_uses = 2;
        assert_eq!(
            GrantAccount::from_snapshot(snapshot).unwrap_err(),
            RuntimeAccountingError::InvariantViolation
        );
    }

    proptest! {
        #[test]
        fn repeated_reservations_never_exceed_use_ceiling(max_uses in 1u32..32, attempts in 1u32..64) {
            let grant = grant(max_uses, RiskBudget {
                mutation_units: 1_000,
                ..RiskBudget::default()
            });
            let mut account = GrantAccount::new(&grant);
            for index in 0..attempts {
                let _ = account.reserve_execution(
                    ReservationId(format!("r-{index}")),
                    ExecutionId(format!("e-{index}")),
                    RiskBudget::default(),
                );
                prop_assert!(account.authority_use_state().charged() <= max_uses);
            }
        }
    }
}
