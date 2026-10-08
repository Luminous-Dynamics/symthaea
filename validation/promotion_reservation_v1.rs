#![forbid(unsafe_code)]

//! Standalone reference transaction for Promotion Reservation v1.
//! This file intentionally has no provider or repository credentials.

use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromotionState {
    Reserved,
    DispatchPrepared,
    ReconciliationRequired,
    Completed,
    Rejected,
    Superseded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionReservationV1 {
    pub reservation_id: String,
    pub promotion_operation_id: String,
    pub lease_id: String,
    pub predecessor_head: String,
    pub reservation_head: String,
    pub fencing_token: u64,
    pub subject_repository: String,
    pub subject_pr: u64,
    pub expected_pr_head_sha: String,
    pub trust_root_generation: u64,
    pub governance_snapshot: String,
    pub provider_profile: String,
    pub state: PromotionState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionDispatchIntentV1 {
    pub reservation_id: String,
    pub promotion_operation_id: String,
    pub predecessor_head: String,
    pub reservation_head: String,
    pub fencing_token: u64,
    pub expected_pr_head_sha: String,
    pub provider_profile: String,
    pub trust_root_generation: u64,
    pub attempt_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReservationError {
    StaleLedgerHead,
    StaleDispatchFence,
    LeaseUnavailable,
    LeaseAlreadyConsumed,
    ReservationNotDispatchable,
    DispatchAlreadyPrepared,
    OperationIdentityMismatch,
}

#[derive(Debug, Default)]
pub struct PromotionReservationLedgerV1 {
    current_head: String,
    current_trust_root_generation: u64,
    current_fencing_token: u64,
    active_lease: Option<String>,
    reservation: Option<PromotionReservationV1>,
    dispatch_intent: Option<PromotionDispatchIntentV1>,
    consumed_leases: HashSet<String>,
    next_reservation: u64,
}

impl PromotionReservationLedgerV1 {
    pub fn new(initial_head: impl Into<String>, lease_id: impl Into<String>) -> Self {
        Self {
            current_head: initial_head.into(),
            current_trust_root_generation: 1,
            current_fencing_token: 0,
            active_lease: Some(lease_id.into()),
            ..Self::default()
        }
    }

    pub fn current_head(&self) -> &str {
        &self.current_head
    }

    pub fn reservation(&self) -> Option<&PromotionReservationV1> {
        self.reservation.as_ref()
    }

    pub fn dispatch_intent(&self) -> Option<&PromotionDispatchIntentV1> {
        self.dispatch_intent.as_ref()
    }

    pub fn reserve(
        &mut self,
        observed_head: &str,
        lease_id: &str,
        next_head: impl Into<String>,
        subject_repository: impl Into<String>,
        subject_pr: u64,
        expected_pr_head_sha: impl Into<String>,
        trust_root_generation: u64,
        governance_snapshot: impl Into<String>,
        provider_profile: impl Into<String>,
    ) -> Result<&PromotionReservationV1, ReservationError> {
        if observed_head != self.current_head {
            return Err(ReservationError::StaleLedgerHead);
        }
        if self.active_lease.as_deref() != Some(lease_id) {
            if self.consumed_leases.contains(lease_id) {
                return Err(ReservationError::LeaseAlreadyConsumed);
            }
            return Err(ReservationError::LeaseUnavailable);
        }

        let reservation_head = next_head.into();
        self.next_reservation += 1;
        self.current_fencing_token += 1;
        let reservation = PromotionReservationV1 {
            reservation_id: format!("promotion-reservation-{}", self.next_reservation),
            promotion_operation_id: format!("promotion-operation-{}", self.next_reservation),
            lease_id: lease_id.to_owned(),
            predecessor_head: observed_head.to_owned(),
            reservation_head: reservation_head.clone(),
            fencing_token: self.current_fencing_token,
            subject_repository: subject_repository.into(),
            subject_pr,
            expected_pr_head_sha: expected_pr_head_sha.into(),
            trust_root_generation,
            governance_snapshot: governance_snapshot.into(),
            provider_profile: provider_profile.into(),
            state: PromotionState::Reserved,
        };

        self.active_lease = None;
        self.consumed_leases.insert(lease_id.to_owned());
        self.current_head = reservation_head;
        self.reservation = Some(reservation);

        Ok(self.reservation.as_ref().expect("reservation installed"))
    }

    pub fn invalidate(
        &mut self,
        observed_head: &str,
        next_head: impl Into<String>,
        trust_root_generation: u64,
    ) -> Result<(), ReservationError> {
        if observed_head != self.current_head {
            return Err(ReservationError::StaleLedgerHead);
        }
        self.current_fencing_token += 1;
        self.current_trust_root_generation = trust_root_generation;
        self.current_head = next_head.into();
        if let Some(reservation) = self.reservation.as_mut() {
            if reservation.state == PromotionState::Reserved {
                reservation.state = PromotionState::Superseded;
            }
        }
        Ok(())
    }

    pub fn prepare_dispatch(
        &mut self,
        observed_operation_id: &str,
        observed_head: &str,
        observed_trust_root_generation: u64,
        observed_fencing_token: u64,
        attempt_sequence: u64,
    ) -> Result<&PromotionDispatchIntentV1, ReservationError> {
        let current_head = self.current_head.clone();
        let current_root = self.current_trust_root_generation;
        let current_fence = self.current_fencing_token;
        let reservation = self
            .reservation
            .as_mut()
            .ok_or(ReservationError::ReservationNotDispatchable)?;

        if reservation.promotion_operation_id != observed_operation_id {
            return Err(ReservationError::OperationIdentityMismatch);
        }
        if observed_head != current_head {
            return Err(ReservationError::StaleLedgerHead);
        }
        if observed_trust_root_generation != current_root
            || observed_fencing_token != current_fence
            || reservation.fencing_token != current_fence
            || reservation.reservation_head != current_head
        {
            return Err(ReservationError::StaleDispatchFence);
        }
        if reservation.state != PromotionState::Reserved {
            if reservation.state == PromotionState::DispatchPrepared {
                return Err(ReservationError::DispatchAlreadyPrepared);
            }
            return Err(ReservationError::ReservationNotDispatchable);
        }

        let intent = PromotionDispatchIntentV1 {
            reservation_id: reservation.reservation_id.clone(),
            promotion_operation_id: reservation.promotion_operation_id.clone(),
            predecessor_head: reservation.predecessor_head.clone(),
            reservation_head: reservation.reservation_head.clone(),
            fencing_token: reservation.fencing_token,
            expected_pr_head_sha: reservation.expected_pr_head_sha.clone(),
            provider_profile: reservation.provider_profile.clone(),
            trust_root_generation: reservation.trust_root_generation,
            attempt_sequence,
        };

        reservation.state = PromotionState::DispatchPrepared;
        self.dispatch_intent = Some(intent);
        Ok(self.dispatch_intent.as_ref().expect("intent installed"))
    }

    pub fn record_unknown(&mut self, operation_id: &str) -> Result<(), ReservationError> {
        let reservation = self
            .reservation
            .as_mut()
            .ok_or(ReservationError::ReservationNotDispatchable)?;

        if reservation.promotion_operation_id != operation_id {
            return Err(ReservationError::OperationIdentityMismatch);
        }
        if self.dispatch_intent.is_none() || reservation.state != PromotionState::DispatchPrepared {
            return Err(ReservationError::ReservationNotDispatchable);
        }

        reservation.state = PromotionState::ReconciliationRequired;
        Ok(())
    }

    pub fn reconcile_complete(&mut self, operation_id: &str) -> Result<(), ReservationError> {
        let reservation = self
            .reservation
            .as_mut()
            .ok_or(ReservationError::ReservationNotDispatchable)?;

        if reservation.promotion_operation_id != operation_id {
            return Err(ReservationError::OperationIdentityMismatch);
        }
        if reservation.state != PromotionState::ReconciliationRequired {
            return Err(ReservationError::ReservationNotDispatchable);
        }

        reservation.state = PromotionState::Completed;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reserve_one() -> PromotionReservationLedgerV1 {
        let mut ledger = PromotionReservationLedgerV1::new("L0", "LEASE-1");
        ledger
            .reserve(
                "L0",
                "LEASE-1",
                "L1",
                "Luminous-Dynamics/symthaea",
                7085,
                "H1",
                7,
                "gov-1",
                "github-async-merge",
            )
            .expect("reservation should succeed");
        ledger
    }

    #[test]
    fn reservation_is_single_use() {
        let mut ledger = PromotionReservationLedgerV1::new("L0", "LEASE-1");
        let reservation = ledger
            .reserve("L0", "LEASE-1", "L1", "repo", 1, "H1", 1, "gov", "github")
            .expect("first reservation")
            .clone();

        assert_eq!(reservation.predecessor_head, "L0");
        assert_eq!(ledger.current_head(), "L1");

        assert_eq!(
            ledger.reserve("L0", "LEASE-1", "L2", "repo", 1, "H1", 1, "gov", "github"),
            Err(ReservationError::StaleLedgerHead)
        );
        assert_eq!(
            ledger.reserve("L1", "LEASE-1", "L2", "repo", 1, "H1", 1, "gov", "github"),
            Err(ReservationError::LeaseAlreadyConsumed)
        );
    }

    #[test]
    fn dispatch_intent_exists_before_unknown() {
        let mut ledger = reserve_one();
        let op = ledger.reservation().unwrap().promotion_operation_id.clone();

        let intent = ledger.prepare_dispatch(&op, "L1", 7, 1, 1).unwrap().clone();
        assert_eq!(intent.promotion_operation_id, op);
        assert_eq!(ledger.reservation().unwrap().state, PromotionState::DispatchPrepared);

        ledger.record_unknown(&op).unwrap();
        assert_eq!(ledger.reservation().unwrap().state, PromotionState::ReconciliationRequired);
        assert_eq!(
            ledger.prepare_dispatch(&op, "L1", 7, 1, 2),
            Err(ReservationError::ReservationNotDispatchable)
        );
    }

    #[test]
    fn unknown_outcome_requires_same_operation_identity() {
        let mut ledger = reserve_one();
        let op = ledger.reservation().unwrap().promotion_operation_id.clone();
        ledger.prepare_dispatch(&op, "L1", 7, 1, 1).unwrap();

        assert_eq!(
            ledger.record_unknown("different-operation"),
            Err(ReservationError::OperationIdentityMismatch)
        );

        ledger.record_unknown(&op).unwrap();
        ledger.reconcile_complete(&op).unwrap();
        assert_eq!(ledger.reservation().unwrap().state, PromotionState::Completed);
    }

    #[test]
    fn root_change_after_reservation_supersedes_before_dispatch() {
        let mut ledger = reserve_one();
        let op = ledger.reservation().unwrap().promotion_operation_id.clone();

        ledger.invalidate("L1", "I1", 8).unwrap();
        assert_eq!(ledger.current_head(), "I1");
        assert_eq!(ledger.reservation().unwrap().state, PromotionState::Superseded);
        assert_eq!(
            ledger.prepare_dispatch(&op, 1),
            Err(ReservationError::ReservationNotDispatchable)
        );
    }


    #[test]
    fn unrelated_ledger_transition_rejects_stale_dispatch_fence() {
        let mut ledger = reserve_one();
        let op = ledger.reservation().unwrap().promotion_operation_id.clone();

        // A different domain transition advances the shared ledger without
        // explicitly touching this reservation record.
        ledger.current_fencing_token += 1;
        ledger.current_head = "L2".to_owned();

        assert_eq!(
            ledger.prepare_dispatch(&op, "L1", 7, 1, 1),
            Err(ReservationError::StaleDispatchFence)
        );
    }

    #[test]
    fn stale_writer_cannot_publish_from_old_predecessor() {
        let mut ledger = PromotionReservationLedgerV1::new("L0", "LEASE-1");
        ledger
            .reserve("L0", "LEASE-1", "L1", "repo", 1, "H1", 1, "gov", "github")
            .unwrap();

        assert_eq!(
            ledger.invalidate("L0", "I-stale", 2),
            Err(ReservationError::StaleLedgerHead)
        );
        assert_eq!(ledger.current_head(), "L1");
    }
}
