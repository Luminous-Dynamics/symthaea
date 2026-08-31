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
//! 4. revocation rotates the epoch **and latches admission closed**;
//! 5. while stopped, new tickets cannot be issued and old tickets cannot acquire;
//! 6. [`EffectEntryDomain::resume`] reopens admission only after already-admitted
//!    work has become quiescent;
//! 7. a permit acquired before revocation remains admission for that one effect;
//! 8. revocation evidence records already-acquired and currently in-flight work
//!    instead of pretending those effects disappeared.
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
use std::sync::{Arc, Mutex};

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

/// Total order of successful acquisitions, revocations, and resumptions.
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

/// Snapshot of already-admitted work in one effect-entry domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EffectEntryActivity {
    outstanding_permits: u64,
    in_flight_effects: u64,
}

impl EffectEntryActivity {
    /// Acquired permits that have not yet entered their effect callback.
    pub const fn outstanding_permits(self) -> u64 {
        self.outstanding_permits
    }

    /// Effect callbacks that have begun and not yet returned/unwound.
    pub const fn in_flight_effects(self) -> u64 {
        self.in_flight_effects
    }

    /// Whether the domain currently has no already-admitted work.
    pub const fn is_quiescent(self) -> bool {
        self.outstanding_permits == 0 && self.in_flight_effects == 0
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
    state: Arc<Mutex<EffectEntryState>>,
}

impl EffectEntryDomain {
    /// Construct a fresh effect-entry domain at epoch zero with admission open.
    pub fn new() -> Self {
        Self {
            domain_id: EffectEntryDomainId(Uuid::new_v4()),
            state: Arc::new(Mutex::new(EffectEntryState {
                epoch: 0,
                sequence: 0,
                admission_open: true,
                outstanding_permits: 0,
                in_flight_effects: 0,
            })),
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

    /// Whether admission is latched closed after revocation.
    pub fn is_stopped(&self) -> bool {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        !state.admission_open
    }

    /// Current already-admitted work counters.
    pub fn activity(&self) -> EffectEntryActivity {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.activity()
    }

    /// Issue an affine ticket for one exact action binding in the current epoch.
    ///
    /// Ticket issuance is not the effect-entry linearization point. A stopped
    /// domain refuses new tickets until [`Self::resume`] succeeds.
    pub fn issue_ticket(
        &self,
        action_binding: [u8; 32],
    ) -> Result<EffectEntryTicket, EffectEntryError> {
        let ticket_id = EffectEntryTicketId(Uuid::new_v4());
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.admission_open {
            return Err(EffectEntryError::AdmissionStopped {
                current_epoch: EffectEntryEpoch(state.epoch),
                current_sequence: EffectEntrySequence(state.sequence),
            });
        }
        Ok(EffectEntryTicket {
            ticket_id,
            domain_id: self.domain_id,
            epoch: EffectEntryEpoch(state.epoch),
            action_binding,
        })
    }

    /// Atomically acquire admission for one exact effect.
    ///
    /// Successful acquisition is the linearization point. If revocation already
    /// completed, admission is latched closed and acquisition fails. After an
    /// explicit resume, an earlier-epoch ticket is still rejected as stale.
    ///
    /// If acquisition wins the race, the returned permit remains valid for that
    /// one already-admitted effect even if revocation completes immediately
    /// afterward.
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

        let permit_id = EffectEntryPermitId(Uuid::new_v4());
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.admission_open {
            return Err(EffectEntryError::AdmissionStopped {
                current_epoch: EffectEntryEpoch(state.epoch),
                current_sequence: EffectEntrySequence(state.sequence),
            });
        }
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
        let next_outstanding = state
            .outstanding_permits
            .checked_add(1)
            .ok_or(EffectEntryError::PermitCounterExhausted)?;
        state.sequence = next_sequence;
        state.outstanding_permits = next_outstanding;

        Ok(EffectEntryPermit {
            permit_id,
            ticket_id: ticket.ticket_id,
            domain_id: self.domain_id,
            epoch: ticket.epoch,
            action_binding: ticket.action_binding,
            acquisition_sequence: EffectEntrySequence(next_sequence),
            state: Some(Arc::clone(&self.state)),
        })
    }

    /// Rotate the admission epoch and latch new effect admission closed.
    ///
    /// The method holds the same short critical section as [`Self::acquire`]. A
    /// returned receipt proves that no later acquisition can succeed until the
    /// host explicitly resumes the domain. Already-acquired or in-flight work is
    /// counted explicitly in the receipt and is not retroactively invalidated.
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
        state.admission_open = false;
        let admitted_activity = state.activity();

        Ok(EffectRevocationReceipt {
            domain_id: self.domain_id,
            previous_epoch: EffectEntryEpoch(previous_epoch),
            current_epoch: EffectEntryEpoch(next_epoch),
            revocation_sequence: EffectEntrySequence(next_sequence),
            admitted_activity,
        })
    }

    /// Explicitly reopen a stopped domain after already-admitted work is quiescent.
    ///
    /// Resume is fail-closed while an acquired permit is outstanding or an effect
    /// callback is still in flight. This prevents a stopped domain from admitting
    /// a new epoch of work while pre-stop admitted work is still unresolved.
    pub fn resume(&self) -> Result<EffectResumeReceipt, EffectEntryError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.admission_open {
            return Err(EffectEntryError::AlreadyRunning);
        }
        let activity = state.activity();
        if !activity.is_quiescent() {
            return Err(EffectEntryError::ResumeWhileActive { activity });
        }

        let next_sequence = state
            .sequence
            .checked_add(1)
            .ok_or(EffectEntryError::SequenceExhausted)?;
        state.sequence = next_sequence;
        state.admission_open = true;

        Ok(EffectResumeReceipt {
            domain_id: self.domain_id,
            epoch: EffectEntryEpoch(state.epoch),
            resume_sequence: EffectEntrySequence(next_sequence),
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
/// let ticket = domain.issue_ticket(binding).unwrap();
/// let permit = domain.acquire(ticket, binding).unwrap();
/// let _ = permit.enter(|| ()).unwrap();
/// let _ = permit.enter(|| ()).unwrap();
/// ```
#[derive(Debug)]
pub struct EffectEntryPermit {
    permit_id: EffectEntryPermitId,
    ticket_id: EffectEntryTicketId,
    domain_id: EffectEntryDomainId,
    epoch: EffectEntryEpoch,
    action_binding: [u8; 32],
    acquisition_sequence: EffectEntrySequence,
    state: Option<Arc<Mutex<EffectEntryState>>>,
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
    /// No domain lock is held while `effect` runs. Immediately before invoking
    /// the callback, this method moves the permit from the outstanding count to
    /// the in-flight count. A private drop guard removes the in-flight count even
    /// if the callback unwinds.
    pub fn enter<F, R>(mut self, effect: F) -> Result<(EffectEntryReceipt, R), EffectEntryError>
    where
        F: FnOnce() -> R,
    {
        let state = Arc::clone(
            self.state
                .as_ref()
                .expect("live effect-entry permit always retains domain state"),
        );
        let activity_at_entry = {
            let mut locked = state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if locked.outstanding_permits == 0 {
                return Err(EffectEntryError::ActivityInvariant);
            }
            let next_in_flight = locked
                .in_flight_effects
                .checked_add(1)
                .ok_or(EffectEntryError::InFlightCounterExhausted)?;
            locked.outstanding_permits -= 1;
            locked.in_flight_effects = next_in_flight;
            locked.activity()
        };

        self.state = None;
        let in_flight = EffectInFlightGuard {
            state: Arc::clone(&state),
        };
        let receipt = EffectEntryReceipt {
            permit_id: self.permit_id,
            ticket_id: self.ticket_id,
            domain_id: self.domain_id,
            epoch: self.epoch,
            action_binding: self.action_binding,
            acquisition_sequence: self.acquisition_sequence,
            activity_at_entry,
        };
        let result = effect();
        drop(in_flight);
        Ok((receipt, result))
    }
}

impl Drop for EffectEntryPermit {
    fn drop(&mut self) {
        let Some(state) = self.state.take() else {
            return;
        };
        let mut locked = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(locked.outstanding_permits > 0);
        if locked.outstanding_permits > 0 {
            locked.outstanding_permits -= 1;
        }
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
    activity_at_entry: EffectEntryActivity,
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

    /// Already-admitted work snapshot immediately before the callback began.
    pub const fn activity_at_entry(self) -> EffectEntryActivity {
        self.activity_at_entry
    }
}

/// Immutable evidence that one revocation epoch rotation linearized and stopped admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectRevocationReceipt {
    domain_id: EffectEntryDomainId,
    previous_epoch: EffectEntryEpoch,
    current_epoch: EffectEntryEpoch,
    revocation_sequence: EffectEntrySequence,
    admitted_activity: EffectEntryActivity,
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

    /// New stopped epoch after rotation.
    pub const fn current_epoch(self) -> EffectEntryEpoch {
        self.current_epoch
    }

    /// Revocation's total-order position within the domain.
    pub const fn revocation_sequence(self) -> EffectEntrySequence {
        self.revocation_sequence
    }

    /// Already-admitted work that revocation did not retroactively invalidate.
    pub const fn admitted_activity(self) -> EffectEntryActivity {
        self.admitted_activity
    }
}

/// Immutable evidence that a quiescent stopped domain was explicitly resumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectResumeReceipt {
    domain_id: EffectEntryDomainId,
    epoch: EffectEntryEpoch,
    resume_sequence: EffectEntrySequence,
}

impl EffectResumeReceipt {
    /// Linearization domain.
    pub const fn domain_id(self) -> EffectEntryDomainId {
        self.domain_id
    }

    /// Epoch reopened by the host.
    pub const fn epoch(self) -> EffectEntryEpoch {
        self.epoch
    }

    /// Resume's total-order position within the domain.
    pub const fn resume_sequence(self) -> EffectEntrySequence {
        self.resume_sequence
    }
}

/// Failure while issuing, acquiring, entering, revoking, or resuming effect entry.
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
    /// Effect admission is latched closed after revocation.
    AdmissionStopped {
        /// Current stopped epoch.
        current_epoch: EffectEntryEpoch,
        /// Most recent successful linearization sequence.
        current_sequence: EffectEntrySequence,
    },
    /// Ticket epoch was rotated before acquisition won the race.
    Revoked {
        /// Epoch captured by the ticket.
        ticket_epoch: EffectEntryEpoch,
        /// Current host epoch when acquisition was attempted.
        current_epoch: EffectEntryEpoch,
        /// Most recent successful acquisition/revocation/resume sequence.
        current_sequence: EffectEntrySequence,
    },
    /// Resume was requested while the domain was already admitting work.
    AlreadyRunning,
    /// Resume was requested while pre-stop admitted work was still active.
    ResumeWhileActive {
        /// Outstanding/in-flight activity preventing safe resume.
        activity: EffectEntryActivity,
    },
    /// Revocation epoch counter cannot advance further.
    EpochExhausted,
    /// Linearization sequence counter cannot advance further.
    SequenceExhausted,
    /// Outstanding-permit accounting counter cannot advance further.
    PermitCounterExhausted,
    /// In-flight effect accounting counter cannot advance further.
    InFlightCounterExhausted,
    /// Internal permit/activity accounting invariant was violated.
    ActivityInvariant,
}

impl fmt::Display for EffectEntryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongDomain { .. } => write!(f, "effect-entry ticket belongs to another domain"),
            Self::ActionBindingMismatch => {
                write!(f, "effect-entry ticket targets another exact action")
            }
            Self::AdmissionStopped { .. } => write!(f, "effect admission is stopped"),
            Self::Revoked { .. } => {
                write!(f, "effect-entry ticket belongs to an earlier revoked epoch")
            }
            Self::AlreadyRunning => write!(f, "effect admission is already running"),
            Self::ResumeWhileActive { .. } => {
                write!(f, "effect admission cannot resume while admitted work remains active")
            }
            Self::EpochExhausted => write!(f, "effect-entry epoch counter exhausted"),
            Self::SequenceExhausted => write!(f, "effect-entry sequence counter exhausted"),
            Self::PermitCounterExhausted => {
                write!(f, "effect-entry outstanding permit counter exhausted")
            }
            Self::InFlightCounterExhausted => {
                write!(f, "effect-entry in-flight counter exhausted")
            }
            Self::ActivityInvariant => write!(f, "effect-entry activity invariant failed"),
        }
    }
}

impl std::error::Error for EffectEntryError {}

#[derive(Debug)]
struct EffectEntryState {
    epoch: u64,
    sequence: u64,
    admission_open: bool,
    outstanding_permits: u64,
    in_flight_effects: u64,
}

impl EffectEntryState {
    fn activity(&self) -> EffectEntryActivity {
        EffectEntryActivity {
            outstanding_permits: self.outstanding_permits,
            in_flight_effects: self.in_flight_effects,
        }
    }
}

#[derive(Debug)]
struct EffectInFlightGuard {
    state: Arc<Mutex<EffectEntryState>>,
}

impl Drop for EffectInFlightGuard {
    fn drop(&mut self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(state.in_flight_effects > 0);
        if state.in_flight_effects > 0 {
            state.in_flight_effects -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::{mpsc, Arc, Barrier};
    use std::thread;

    #[test]
    fn revocation_latches_admission_until_explicit_resume() {
        let domain = EffectEntryDomain::new();
        let binding = [1; 32];
        let ticket_while_running = domain.issue_ticket(binding).unwrap();
        let stale_after_resume = domain.issue_ticket(binding).unwrap();
        let revocation = domain.revoke_all().unwrap();

        assert!(domain.is_stopped());
        assert_eq!(revocation.previous_epoch().get(), 0);
        assert_eq!(revocation.current_epoch().get(), 1);
        assert!(revocation.admitted_activity().is_quiescent());
        assert!(matches!(
            domain.issue_ticket(binding),
            Err(EffectEntryError::AdmissionStopped { .. })
        ));
        assert!(matches!(
            domain.acquire(ticket_while_running, binding),
            Err(EffectEntryError::AdmissionStopped { .. })
        ));

        let resume = domain.resume().unwrap();
        assert!(!domain.is_stopped());
        assert_eq!(resume.epoch(), revocation.current_epoch());
        assert!(revocation.revocation_sequence() < resume.resume_sequence());
        assert!(matches!(
            domain.acquire(stale_after_resume, binding),
            Err(EffectEntryError::Revoked { .. })
        ));
    }

    #[test]
    fn acquired_permit_survives_stop_but_blocks_resume_until_quiescent() {
        let domain = EffectEntryDomain::new();
        let binding = [2; 32];
        let ticket = domain.issue_ticket(binding).unwrap();
        let permit = domain.acquire(ticket, binding).unwrap();
        let acquisition = permit.acquisition_sequence();
        assert_eq!(domain.activity().outstanding_permits(), 1);

        let revocation = domain.revoke_all().unwrap();
        assert!(domain.is_stopped());
        assert_eq!(revocation.admitted_activity().outstanding_permits(), 1);
        assert_eq!(revocation.admitted_activity().in_flight_effects(), 0);
        assert!(matches!(
            domain.resume(),
            Err(EffectEntryError::ResumeWhileActive { .. })
        ));

        let (receipt, value) = permit.enter(|| 42_u64).unwrap();
        assert_eq!(value, 42);
        assert_eq!(receipt.action_binding(), binding);
        assert_eq!(receipt.acquisition_sequence(), acquisition);
        assert!(acquisition < revocation.revocation_sequence());
        assert!(domain.activity().is_quiescent());

        domain.resume().unwrap();
        assert!(!domain.is_stopped());
    }

    #[test]
    fn revocation_during_callback_reports_in_flight_work_and_does_not_block() {
        let domain = Arc::new(EffectEntryDomain::new());
        let binding = [3; 32];
        let ticket = domain.issue_ticket(binding).unwrap();
        let permit = domain.acquire(ticket, binding).unwrap();

        let (entered_tx, entered_rx) = mpsc::channel();
        let (continue_tx, continue_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            permit.enter(|| {
                entered_tx.send(()).unwrap();
                continue_rx.recv().unwrap();
                3_u64
            })
        });

        entered_rx.recv().unwrap();
        let revocation = domain.revoke_all().unwrap();
        assert_eq!(revocation.admitted_activity().outstanding_permits(), 0);
        assert_eq!(revocation.admitted_activity().in_flight_effects(), 1);
        assert!(matches!(
            domain.resume(),
            Err(EffectEntryError::ResumeWhileActive { .. })
        ));

        continue_tx.send(()).unwrap();
        let (_, value) = worker.join().unwrap().unwrap();
        assert_eq!(value, 3);
        assert!(domain.activity().is_quiescent());
        domain.resume().unwrap();
    }

    #[test]
    fn dropping_unused_permit_repairs_outstanding_count() {
        let domain = EffectEntryDomain::new();
        let binding = [8; 32];
        let ticket = domain.issue_ticket(binding).unwrap();
        let permit = domain.acquire(ticket, binding).unwrap();
        assert_eq!(domain.activity().outstanding_permits(), 1);
        drop(permit);
        assert!(domain.activity().is_quiescent());
    }

    #[test]
    fn callback_unwind_repairs_in_flight_count() {
        let domain = EffectEntryDomain::new();
        let binding = [9; 32];
        let ticket = domain.issue_ticket(binding).unwrap();
        let permit = domain.acquire(ticket, binding).unwrap();

        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = permit.enter(|| panic!("synthetic adapter unwind"));
        }));
        assert!(result.is_err());
        assert!(domain.activity().is_quiescent());
    }

    #[test]
    fn new_epoch_ticket_can_be_admitted_only_after_resume() {
        let domain = EffectEntryDomain::new();
        let binding = [4; 32];
        let old_ticket = domain.issue_ticket(binding).unwrap();
        domain.revoke_all().unwrap();
        assert!(matches!(
            domain.issue_ticket(binding),
            Err(EffectEntryError::AdmissionStopped { .. })
        ));
        domain.resume().unwrap();
        assert!(matches!(
            domain.acquire(old_ticket, binding),
            Err(EffectEntryError::Revoked { .. })
        ));

        let new_ticket = domain.issue_ticket(binding).unwrap();
        assert_eq!(new_ticket.epoch(), domain.current_epoch());
        let permit = domain.acquire(new_ticket, binding).unwrap();
        drop(permit);
        assert!(domain.activity().is_quiescent());
    }

    #[test]
    fn wrong_domain_and_action_binding_fail_closed() {
        let first = EffectEntryDomain::new();
        let second = EffectEntryDomain::new();
        let binding = [5; 32];
        let ticket = first.issue_ticket(binding).unwrap();
        assert!(matches!(
            second.acquire(ticket, binding),
            Err(EffectEntryError::WrongDomain { .. })
        ));

        let ticket = first.issue_ticket(binding).unwrap();
        assert!(matches!(
            first.acquire(ticket, [6; 32]),
            Err(EffectEntryError::ActionBindingMismatch)
        ));
        assert!(first.activity().is_quiescent());
    }

    #[test]
    fn resume_while_running_fails_without_changing_sequence() {
        let domain = EffectEntryDomain::new();
        let before = domain.current_sequence();
        assert!(matches!(domain.resume(), Err(EffectEntryError::AlreadyRunning)));
        assert_eq!(domain.current_sequence(), before);
    }

    #[test]
    fn repeated_acquire_revoke_race_has_only_linearized_outcomes() {
        for iteration in 0_u8..64 {
            let domain = Arc::new(EffectEntryDomain::new());
            let binding = [iteration; 32];
            let ticket = domain.issue_ticket(binding).unwrap();
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
            assert!(domain.is_stopped());

            match acquire {
                Ok(permit) => {
                    assert!(permit.acquisition_sequence() < revocation.revocation_sequence());
                    assert_eq!(revocation.admitted_activity().outstanding_permits(), 1);
                    assert!(matches!(
                        domain.resume(),
                        Err(EffectEntryError::ResumeWhileActive { .. })
                    ));
                    let (receipt, entered) = permit.enter(|| true).unwrap();
                    assert!(entered);
                    assert!(receipt.acquisition_sequence() < revocation.revocation_sequence());
                    assert!(domain.activity().is_quiescent());
                }
                Err(EffectEntryError::AdmissionStopped {
                    current_epoch,
                    current_sequence,
                }) => {
                    assert_eq!(current_epoch, revocation.current_epoch());
                    assert_eq!(current_sequence, revocation.revocation_sequence());
                    assert!(revocation.admitted_activity().is_quiescent());
                }
                Err(other) => panic!("unexpected race outcome: {other}"),
            }
        }
    }
}
