// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Atomic admission of already-authorized effects against concurrent revocation.
//!
//! The wider assurance stack performs exact point-in-time checks immediately
//! before state-changing adapter entry. Those checks alone cannot decide the
//! race where revocation occurs after the final validation read but before the
//! adapter crosses its point of no return.
//!
//! This module adds a deliberately small host-owned linearization primitive:
//!
//! 1. the host issues an affine [`EffectEntryTicket`] for one exact action;
//! 2. [`EffectEntryDomain::acquire`] and [`EffectEntryDomain::revoke_all`]
//!    serialize on the same short critical section;
//! 3. successful acquisition returns an affine [`EffectEntryPermit`] and records
//!    a monotonic linearization sequence;
//! 4. a completed revocation invalidates every still-unacquired ticket from an
//!    earlier epoch;
//! 5. a permit acquired before revocation remains admission for that one effect.
//!
//! The domain lock is **not** held while the effect callback runs. Acquisition is
//! the point of no return for admission semantics; post-entry cancellation is a
//! separate adapter-specific guarantee.
//!
//! This primitive does not itself prove that arbitrary code used the permit. A
//! concrete production adapter must make [`EffectEntryPermit::enter`] (or an
//! equivalent permit-consuming boundary) structurally necessary before its first
//! external side effect.

use std::fmt;
use std::sync::Mutex;

use uuid::Uuid;

/// Stable identity of one effect-entry linearization domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct EffectEntryDomainId(Uuid);

impl EffectEntryDomainId {
    /// Borrow the underlying UUID for evidence serialization.
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

/// Monotonic revocation epoch for one effect-entry domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct EffectEntryEpoch(u64);

impl EffectEntryEpoch {
    /// Numeric epoch value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Total order of successful permit acquisitions and revocations in one domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct EffectEntrySequence(u64);

impl EffectEntrySequence {
    /// Numeric linearization sequence.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Identity of one pre-entry ticket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct EffectEntryTicketId(Uuid);

impl EffectEntryTicketId {
    /// Borrow the underlying UUID for evidence serialization.
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

/// Identity of one successfully acquired effect-entry permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct EffectEntryPermitId(Uuid);

impl EffectEntryPermitId {
    /// Borrow the underlying UUID for evidence serialization.
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

/// Host-owned domain that linearizes revocation against effect admission.
///
/// Model/planner code should never receive this minting/acquisition root. A
/// trusted adapter host retains the domain and only passes affine tickets or
/// permits across narrower boundaries.
#[derive(Debug)]
pub struct EffectEntryDomain {
    domain_id: EffectEntryDomainId,
    state: Mutex<EffectEntryState>,
}

impl EffectEntryDomain {
    /// Construct a fresh effect-entry domain at epoch zero.
    pub fn new() -> Self {
        Self {
            domain_id: EffectEntryDomainId(Uuid::new_v4()),
            state: Mutex::new(EffectEntryState {
                epoch: 0,
                sequence: 0,
            }),
        }
    }

    /// Stable domain identity.
    pub const fn domain_id(&self) -> EffectEntryDomainId {
        self.domain_id
    }

    /// Current revocation epoch.
    pub fn current_epoch(&self) -> EffectEntryEpoch {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        EffectEntryEpoch(state.epoch)
    }

    /// Current linearization sequence.
    pub fn current_sequence(&self) -> EffectEntrySequence {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        EffectEntrySequence(state.sequence)
    }

    /// Issue an affine ticket for one exact action binding in the current epoch.
    ///
    /// Ticket issuance is not the effect-entry linearization point. A ticket from
    /// an epoch revoked before [`Self::acquire`] runs will fail closed.
    pub fn issue_ticket(&self, action_binding: [u8; 32]) -> EffectEntryTicket {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        EffectEntryTicket {
            ticket_id: EffectEntryTicketId(Uuid::new_v4()),
            domain_id: self.domain_id,
            epoch: EffectEntryEpoch(state.epoch),
            action_binding,
        }
    }

    /// Atomically acquire admission for one exact effect.
    ///
    /// Successful acquisition is the linearization point. If revocation already
    /// completed, an earlier-epoch ticket is rejected. If acquisition wins the
    /// race, the returned permit remains valid for that one already-admitted
    /// effect even if revocation completes immediately afterward.
    pub fn acquire(
        &self,
        ticket: EffectEntryTicket,
        expected_action_binding: [u8; 32],
    ) -> Result<EffectEntryPermit, EffectEntryError> {
        if ticket.domain_id != self.domain_id {
            return Err(EffectEntryError::WrongDomain {
                expected: self.domain_id,
                actual: ticket.domain_id,
            });
        }
        if ticket.action_binding != expected_action_binding {
            return Err(EffectEntryError::ActionBindingMismatch);
        }

        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if ticket.epoch.0 != state.epoch {
            return Err(EffectEntryError::Revoked {
                ticket_epoch: ticket.epoch,
                current_epoch: EffectEntryEpoch(state.epoch),
                current_sequence: EffectEntrySequence(state.sequence),
            });
        }

        let next_sequence = state
            .sequence
            .checked_add(1)
            .ok_or(EffectEntryError::SequenceExhausted)?;
        state.sequence = next_sequence;

        Ok(EffectEntryPermit {
            permit_id: EffectEntryPermitId(Uuid::new_v4()),
            ticket_id: ticket.ticket_id,
            domain_id: self.domain_id,
            epoch: ticket.epoch,
            action_binding: ticket.action_binding,
            acquisition_sequence: EffectEntrySequence(next_sequence),
        })
    }

    /// Rotate the admission epoch and linearize revocation against acquisition.
    ///
    /// The method holds the same short critical section as [`Self::acquire`]. A
    /// returned receipt therefore proves that every ticket from `previous_epoch`
    /// which had not already acquired a permit will fail on later acquisition.
    pub fn revoke_all(&self) -> Result<EffectRevocationReceipt, EffectEntryError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous_epoch = state.epoch;
        let next_epoch = state
            .epoch
            .checked_add(1)
            .ok_or(EffectEntryError::EpochExhausted)?;
        let next_sequence = state
            .sequence
            .checked_add(1)
            .ok_or(EffectEntryError::SequenceExhausted)?;

        state.epoch = next_epoch;
        state.sequence = next_sequence;

        Ok(EffectRevocationReceipt {
            domain_id: self.domain_id,
            previous_epoch: EffectEntryEpoch(previous_epoch),
            current_epoch: EffectEntryEpoch(next_epoch),
            revocation_sequence: EffectEntrySequence(next_sequence),
        })
    }
}

impl Default for EffectEntryDomain {
    fn default() -> Self {
        Self::new()
    }
}

/// Affine pre-entry request bound to one domain, epoch, and exact action.
///
/// The type is intentionally neither `Clone` nor `Copy`.
#[derive(Debug)]
pub struct EffectEntryTicket {
    ticket_id: EffectEntryTicketId,
    domain_id: EffectEntryDomainId,
    epoch: EffectEntryEpoch,
    action_binding: [u8; 32],
}

impl EffectEntryTicket {
    /// Ticket identity.
    pub const fn ticket_id(&self) -> EffectEntryTicketId {
        self.ticket_id
    }

    /// Effect-entry domain that issued the ticket.
    pub const fn domain_id(&self) -> EffectEntryDomainId {
        self.domain_id
    }

    /// Epoch captured when the ticket was issued.
    pub const fn epoch(&self) -> EffectEntryEpoch {
        self.epoch
    }

    /// Exact action authorization binding.
    pub const fn action_binding(&self) -> [u8; 32] {
        self.action_binding
    }
}

/// Affine one-effect admission whose acquisition already won the revocation race.
///
/// The permit is intentionally neither `Clone` nor `Copy`. Consuming
/// [`Self::enter`] makes one effect callback structurally follow successful
/// admission without holding the domain lock during the callback.
///
/// ```compile_fail
/// use symthaea_ai_assurance::EffectEntryDomain;
///
/// let domain = EffectEntryDomain::new();
/// let binding = [7; 32];
/// let ticket = domain.issue_ticket(binding);
/// let permit = domain.acquire(ticket, binding).unwrap();
/// let _ = permit.enter(|| ());
/// let _ = permit.enter(|| ());
/// ```
#[derive(Debug)]
pub struct EffectEntryPermit {
    permit_id: EffectEntryPermitId,
    ticket_id: EffectEntryTicketId,
    domain_id: EffectEntryDomainId,
    epoch: EffectEntryEpoch,
    action_binding: [u8; 32],
    acquisition_sequence: EffectEntrySequence,
}

impl EffectEntryPermit {
    /// Permit identity.
    pub const fn permit_id(&self) -> EffectEntryPermitId {
        self.permit_id
    }

    /// Ticket consumed to acquire this permit.
    pub const fn ticket_id(&self) -> EffectEntryTicketId {
        self.ticket_id
    }

    /// Linearization domain.
    pub const fn domain_id(&self) -> EffectEntryDomainId {
        self.domain_id
    }

    /// Epoch in which acquisition succeeded.
    pub const fn epoch(&self) -> EffectEntryEpoch {
        self.epoch
    }

    /// Exact action binding admitted by this permit.
    pub const fn action_binding(&self) -> [u8; 32] {
        self.action_binding
    }

    /// Total-order position at which acquisition won or preceded revocation.
    pub const fn acquisition_sequence(&self) -> EffectEntrySequence {
        self.acquisition_sequence
    }

    /// Consume this permit and invoke one already-admitted effect callback.
    ///
    /// No domain lock is held while `effect` runs. A later revocation may support
    /// adapter-specific cancellation, but it does not retroactively invalidate
    /// this admission.
    pub fn enter<F, R>(self, effect: F) -> (EffectEntryReceipt, R)
    where
        F: FnOnce() -> R,
    {
        let receipt = EffectEntryReceipt {
            permit_id: self.permit_id,
            ticket_id: self.ticket_id,
            domain_id: self.domain_id,
            epoch: self.epoch,
            action_binding: self.action_binding,
            acquisition_sequence: self.acquisition_sequence,
        };
        let result = effect();
        (receipt, result)
    }
}

/// Immutable evidence that one effect admission linearized successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectEntryReceipt {
    permit_id: EffectEntryPermitId,
    ticket_id: EffectEntryTicketId,
    domain_id: EffectEntryDomainId,
    epoch: EffectEntryEpoch,
    action_binding: [u8; 32],
    acquisition_sequence: EffectEntrySequence,
}

impl EffectEntryReceipt {
    /// Permit that admitted the effect.
    pub const fn permit_id(self) -> EffectEntryPermitId {
        self.permit_id
    }

    /// Ticket consumed by acquisition.
    pub const fn ticket_id(self) -> EffectEntryTicketId {
        self.ticket_id
    }

    /// Linearization domain.
    pub const fn domain_id(self) -> EffectEntryDomainId {
        self.domain_id
    }

    /// Admission epoch.
    pub const fn epoch(self) -> EffectEntryEpoch {
        self.epoch
    }

    /// Exact admitted action binding.
    pub const fn action_binding(self) -> [u8; 32] {
        self.action_binding
    }

    /// Acquisition's total-order position within the domain.
    pub const fn acquisition_sequence(self) -> EffectEntrySequence {
        self.acquisition_sequence
    }
}

/// Immutable evidence that one revocation epoch rotation linearized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectRevocationReceipt {
    domain_id: EffectEntryDomainId,
    previous_epoch: EffectEntryEpoch,
    current_epoch: EffectEntryEpoch,
    revocation_sequence: EffectEntrySequence,
}

impl EffectRevocationReceipt {
    /// Linearization domain.
    pub const fn domain_id(self) -> EffectEntryDomainId {
        self.domain_id
    }

    /// Epoch invalidated by this rotation.
    pub const fn previous_epoch(self) -> EffectEntryEpoch {
        self.previous_epoch
    }

    /// New epoch after rotation.
    pub const fn current_epoch(self) -> EffectEntryEpoch {
        self.current_epoch
    }

    /// Revocation's total-order position within the domain.
    pub const fn revocation_sequence(self) -> EffectEntrySequence {
        self.revocation_sequence
    }
}

/// Failure while acquiring or rotating effect-entry authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectEntryError {
    /// Ticket belongs to another admission domain.
    WrongDomain {
        /// Domain retained by the host.
        expected: EffectEntryDomainId,
        /// Domain carried by the supplied ticket.
        actual: EffectEntryDomainId,
    },
    /// Ticket targets another exact action.
    ActionBindingMismatch,
    /// Ticket epoch was rotated before acquisition won the race.
    Revoked {
        /// Epoch captured by the ticket.
        ticket_epoch: EffectEntryEpoch,
        /// Current host epoch when acquisition was attempted.
        current_epoch: EffectEntryEpoch,
        /// Most recent successful acquisition/revocation sequence at rejection.
        current_sequence: EffectEntrySequence,
    },
    /// Revocation epoch counter cannot advance further.
    EpochExhausted,
    /// Linearization sequence counter cannot advance further.
    SequenceExhausted,
}

impl fmt::Display for EffectEntryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongDomain { .. } => write!(f, "effect-entry ticket belongs to another domain"),
            Self::ActionBindingMismatch => {
                write!(f, "effect-entry ticket targets another exact action")
            }
            Self::Revoked { .. } => write!(f, "effect-entry ticket was revoked before acquisition"),
            Self::EpochExhausted => write!(f, "effect-entry epoch counter exhausted"),
            Self::SequenceExhausted => write!(f, "effect-entry sequence counter exhausted"),
        }
    }
}

impl std::error::Error for EffectEntryError {}

#[derive(Debug)]
struct EffectEntryState {
    epoch: u64,
    sequence: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn revocation_before_acquisition_rejects_without_entry() {
        let domain = EffectEntryDomain::new();
        let binding = [1; 32];
        let ticket = domain.issue_ticket(binding);
        let revocation = domain.revoke_all().unwrap();

        let result = domain.acquire(ticket, binding);
        assert!(matches!(result, Err(EffectEntryError::Revoked { .. })));
        assert_eq!(revocation.previous_epoch().get(), 0);
        assert_eq!(revocation.current_epoch().get(), 1);
    }

    #[test]
    fn acquired_permit_survives_later_revocation_for_one_effect() {
        let domain = EffectEntryDomain::new();
        let binding = [2; 32];
        let ticket = domain.issue_ticket(binding);
        let permit = domain.acquire(ticket, binding).unwrap();
        let acquisition = permit.acquisition_sequence();
        let revocation = domain.revoke_all().unwrap();
        let (receipt, value) = permit.enter(|| 42_u64);

        assert_eq!(value, 42);
        assert_eq!(receipt.action_binding(), binding);
        assert_eq!(receipt.acquisition_sequence(), acquisition);
        assert!(acquisition < revocation.revocation_sequence());
    }

    #[test]
    fn old_ticket_fails_but_new_epoch_ticket_can_be_admitted() {
        let domain = EffectEntryDomain::new();
        let binding = [3; 32];
        let old_ticket = domain.issue_ticket(binding);
        domain.revoke_all().unwrap();
        assert!(matches!(
            domain.acquire(old_ticket, binding),
            Err(EffectEntryError::Revoked { .. })
        ));

        let new_ticket = domain.issue_ticket(binding);
        assert_eq!(new_ticket.epoch(), domain.current_epoch());
        assert!(domain.acquire(new_ticket, binding).is_ok());
    }

    #[test]
    fn wrong_domain_and_action_binding_fail_closed() {
        let first = EffectEntryDomain::new();
        let second = EffectEntryDomain::new();
        let binding = [4; 32];
        let ticket = first.issue_ticket(binding);
        assert!(matches!(
            second.acquire(ticket, binding),
            Err(EffectEntryError::WrongDomain { .. })
        ));

        let ticket = first.issue_ticket(binding);
        assert!(matches!(
            first.acquire(ticket, [5; 32]),
            Err(EffectEntryError::ActionBindingMismatch)
        ));
    }

    #[test]
    fn repeated_acquire_revoke_race_has_only_linearized_outcomes() {
        for iteration in 0_u8..64 {
            let domain = Arc::new(EffectEntryDomain::new());
            let binding = [iteration; 32];
            let ticket = domain.issue_ticket(binding);
            let barrier = Arc::new(Barrier::new(3));

            let acquire_domain = Arc::clone(&domain);
            let acquire_barrier = Arc::clone(&barrier);
            let acquire_thread = thread::spawn(move || {
                acquire_barrier.wait();
                acquire_domain.acquire(ticket, binding)
            });

            let revoke_domain = Arc::clone(&domain);
            let revoke_barrier = Arc::clone(&barrier);
            let revoke_thread = thread::spawn(move || {
                revoke_barrier.wait();
                revoke_domain.revoke_all().unwrap()
            });

            barrier.wait();
            let acquire = acquire_thread.join().unwrap();
            let revocation = revoke_thread.join().unwrap();

            match acquire {
                Ok(permit) => {
                    assert!(permit.acquisition_sequence() < revocation.revocation_sequence());
                    let (receipt, entered) = permit.enter(|| true);
                    assert!(entered);
                    assert!(receipt.acquisition_sequence() < revocation.revocation_sequence());
                }
                Err(EffectEntryError::Revoked {
                    ticket_epoch,
                    current_epoch,
                    current_sequence,
                }) => {
                    assert!(ticket_epoch < current_epoch);
                    assert_eq!(current_sequence, revocation.revocation_sequence());
                }
                Err(other) => panic!("unexpected race outcome: {other}"),
            }
        }
    }
}
