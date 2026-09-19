// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Linearizable CAS frontier for action authority v0.2.
//!
//! This crate composes the strict checkpoint state machine with an external
//! compare-and-swap persistence contract. It still does not authenticate a
//! concrete store implementation, mint current authority, or authorize effect
//! dispatch.
//!
//! Core separation:
//!
//! ```text
//! strict checkpoint successor
//!     != durable/linearizable persistence
//!     != trusted store provisioning
//!     != authenticated restart anchor
//!     != fresh current authority
//!     != dispatch permit
//!     != effect
//! ```
//!
//! V2D convergence deliberately exposes no production restart constructor. A
//! serialized checkpoint/head cannot recreate a writable frontier after process
//! loss. Restart activation belongs above this crate and must consume a future
//! verifier-owned authenticated anchor/currentness proof.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_action_checkpoint::{
    CheckpointHeadV2, CheckpointV2Error, GrantAccountCheckpointV2,
};
use symthaea_action_runtime::{
    AttemptId, EffectBindingDigest, EffectIntentId, ExecutionReservationV2, GrantAccountV2,
    ReservationId, ReservationState,
};
use symthaea_authority::{CapabilityGrant, Digest32, RiskBudget};
use thiserror::Error;

/// Atomic external checkpoint frontier.
///
/// `current_head` must be a linearizable observation of the same durable state
/// used by `compare_and_swap`. An implementation satisfies this trait only if
/// comparison with `expected_previous` and installation of `checkpoint` are one
/// atomic/linearizable state transition. A read-then-write implementation
/// without equivalent exclusion does not satisfy the contract.
///
/// The trait itself does not prove that a concrete provider is durable,
/// provisioned by the deployment, rollback-resistant across machine loss, or
/// otherwise trustworthy.
pub trait CheckpointCasStoreV2 {
    type Error: StdError + 'static;

    fn current_head(&mut self) -> Result<Option<CheckpointHeadV2>, Self::Error>;

    fn compare_and_swap(
        &mut self,
        expected_previous: Option<CheckpointHeadV2>,
        checkpoint: &GrantAccountCheckpointV2,
    ) -> Result<CheckpointHeadV2, Self::Error>;
}

/// Ordinary evidence of a successfully acknowledged generation-zero install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EstablishedGrantFrontierV2 {
    pub checkpoint: GrantAccountCheckpointV2,
    pub head: CheckpointHeadV2,
}

/// Affine in-process proof that one exact reservation was persisted in
/// `Reserved` state at one exact CAS frontier.
///
/// Deliberately neither Clone nor Serde. Persisted checkpoint bytes remain
/// evidence only and do not recreate this token by themselves.
#[derive(Debug)]
pub struct PersistedReservationV2 {
    grant_digest: Digest32,
    reservation_id: ReservationId,
    effect_intent_id: EffectIntentId,
    attempt_id: AttemptId,
    effect_binding_digest: EffectBindingDigest,
    risk_charge: RiskBudget,
    persisted_head: CheckpointHeadV2,
}

impl PersistedReservationV2 {
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

    pub fn effect_binding_digest(&self) -> EffectBindingDigest {
        self.effect_binding_digest
    }

    pub fn risk_charge(&self) -> RiskBudget {
        self.risk_charge
    }

    pub fn persisted_head(&self) -> CheckpointHeadV2 {
        self.persisted_head
    }
}

/// Affine in-process proof that the exact persisted reservation advanced from
/// `Reserved` to durable `OutcomeUnknown` under one exact CAS successor.
///
/// This is a persistence/ordering capability only. It is not a dispatch permit
/// and contains no cognition- or simulation-derived authority.
#[derive(Debug)]
pub struct DurablyArmedReservationV2 {
    grant_digest: Digest32,
    reservation_id: ReservationId,
    effect_intent_id: EffectIntentId,
    attempt_id: AttemptId,
    effect_binding_digest: EffectBindingDigest,
    risk_charge: RiskBudget,
    reserved_head: CheckpointHeadV2,
    armed_head: CheckpointHeadV2,
}

impl DurablyArmedReservationV2 {
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

    pub fn effect_binding_digest(&self) -> EffectBindingDigest {
        self.effect_binding_digest
    }

    pub fn risk_charge(&self) -> RiskBudget {
        self.risk_charge
    }

    pub fn reserved_head(&self) -> CheckpointHeadV2 {
        self.reserved_head
    }

    pub fn armed_head(&self) -> CheckpointHeadV2 {
        self.armed_head
    }
}

/// One in-process writer view of an external CAS frontier.
///
/// The object is intentionally not Clone and owns the exact grant whose
/// checkpoint lineage it may progress. A store failure, CAS conflict, malformed
/// successor, unexpected store advancement, or wrong acknowledgement latches
/// the writer into containment because its local view can no longer prove which
/// durable head won.
///
/// There is intentionally no production constructor that resumes this object
/// from caller-supplied persisted bytes/head and no method that extracts the
/// underlying store after containment.
pub struct CasFrontierV2<S> {
    store: S,
    grant: CapabilityGrant,
    expected_checkpoint: GrantAccountCheckpointV2,
    expected_head: CheckpointHeadV2,
    contained: bool,
}

impl<S> CasFrontierV2<S>
where
    S: CheckpointCasStoreV2,
{
    /// Adapter-local expected checkpoint. This is not a fresh store read.
    pub fn expected_checkpoint(&self) -> &GrantAccountCheckpointV2 {
        &self.expected_checkpoint
    }

    /// Adapter-local expected head. This is not a fresh store read.
    pub fn expected_head(&self) -> CheckpointHeadV2 {
        self.expected_head
    }

    pub fn grant_digest(&self) -> Digest32 {
        self.grant.digest()
    }

    pub fn is_contained(&self) -> bool {
        self.contained
    }

    /// Persist any strict legal semantic successor.
    ///
    /// This returns only the acknowledged-and-reobserved head. Stronger typed
    /// transitions such as reservation persistence and durable arming use
    /// dedicated methods below.
    pub fn persist_successor(
        &mut self,
        successor: GrantAccountCheckpointV2,
    ) -> Result<CheckpointHeadV2, FrontierV2Error<S::Error>> {
        self.cas_successor(successor)
    }

    /// Persist exactly one newly allocated reservation while every previously
    /// persisted reservation remains byte-semantically unchanged.
    ///
    /// Only an exact one-record addition in `Reserved` state mints the affine
    /// `PersistedReservationV2` token. The token is not returned unless the CAS
    /// succeeded and a fresh store read still observes the exact installed head.
    pub fn persist_new_reservation(
        &mut self,
        successor: GrantAccountCheckpointV2,
    ) -> Result<PersistedReservationV2, FrontierV2Error<S::Error>> {
        self.ensure_not_contained()?;
        successor
            .verify_successor_of(&self.expected_checkpoint, &self.grant)
            .map_err(FrontierV2Error::Checkpoint)?;
        let reservation = exact_single_new_reserved(
            self.expected_checkpoint.snapshot(),
            successor.snapshot(),
        )?;
        let head = self.cas_successor(successor)?;
        Ok(PersistedReservationV2 {
            grant_digest: self.grant.digest(),
            reservation_id: reservation.reservation_id,
            effect_intent_id: reservation.effect_intent_id,
            attempt_id: reservation.attempt_id,
            effect_binding_digest: reservation.effect_binding_digest,
            risk_charge: reservation.risk_charge,
            persisted_head: head,
        })
    }

    /// Persist the exact `Reserved -> OutcomeUnknown` transition for one affine
    /// previously persisted reservation.
    ///
    /// The token is consumed. The transition must change no other reservation
    /// and may introduce no new record. Success proves durable conservative
    /// arming only; it still grants no dispatch authority. The armed token is
    /// not returned unless a fresh store read still observes the exact installed
    /// head after CAS.
    pub fn arm_outcome_unknown(
        &mut self,
        persisted: PersistedReservationV2,
        successor: GrantAccountCheckpointV2,
    ) -> Result<DurablyArmedReservationV2, FrontierV2Error<S::Error>> {
        self.ensure_not_contained()?;
        if persisted.grant_digest != self.grant.digest() {
            return Err(FrontierV2Error::PersistedReservationGrantMismatch);
        }
        if persisted.persisted_head != self.expected_head {
            return Err(FrontierV2Error::PersistedReservationHeadMismatch);
        }

        let current = self
            .expected_checkpoint
            .snapshot()
            .reservations
            .get(&persisted.reservation_id)
            .ok_or(FrontierV2Error::PersistedReservationRecordMismatch)?;
        if current.state != ReservationState::Reserved || !token_matches(current, &persisted) {
            return Err(FrontierV2Error::PersistedReservationRecordMismatch);
        }

        successor
            .verify_successor_of(&self.expected_checkpoint, &self.grant)
            .map_err(FrontierV2Error::Checkpoint)?;
        exact_single_arm(
            self.expected_checkpoint.snapshot(),
            successor.snapshot(),
            persisted.reservation_id,
        )?;

        let reserved_head = self.expected_head;
        let armed_head = self.cas_successor(successor)?;
        Ok(DurablyArmedReservationV2 {
            grant_digest: persisted.grant_digest,
            reservation_id: persisted.reservation_id,
            effect_intent_id: persisted.effect_intent_id,
            attempt_id: persisted.attempt_id,
            effect_binding_digest: persisted.effect_binding_digest,
            risk_charge: persisted.risk_charge,
            reserved_head,
            armed_head,
        })
    }

    fn cas_successor(
        &mut self,
        successor: GrantAccountCheckpointV2,
    ) -> Result<CheckpointHeadV2, FrontierV2Error<S::Error>> {
        self.ensure_not_contained()?;
        if let Err(error) = successor.verify_successor_of(&self.expected_checkpoint, &self.grant) {
            self.contained = true;
            return Err(FrontierV2Error::Checkpoint(error));
        }
        let next_head = match successor.head() {
            Ok(head) => head,
            Err(error) => {
                self.contained = true;
                return Err(FrontierV2Error::Checkpoint(error));
            }
        };

        let acknowledged = match self
            .store
            .compare_and_swap(Some(self.expected_head), &successor)
        {
            Ok(head) => head,
            Err(error) => {
                self.contained = true;
                return Err(FrontierV2Error::Store(error));
            }
        };
        if acknowledged != next_head {
            self.contained = true;
            return Err(FrontierV2Error::AcknowledgedWrongHead);
        }

        // Do not mint/return a positive persistence fact when the durable store
        // is already known to have advanced again before this operation returns.
        let observed = match self.store.current_head() {
            Ok(head) => head,
            Err(error) => {
                self.contained = true;
                return Err(FrontierV2Error::Store(error));
            }
        };
        if observed != Some(next_head) {
            self.contained = true;
            return Err(FrontierV2Error::StoreFrontierChanged);
        }

        self.expected_checkpoint = successor;
        self.expected_head = next_head;
        Ok(next_head)
    }

    fn ensure_not_contained(&self) -> Result<(), FrontierV2Error<S::Error>> {
        if self.contained {
            Err(FrontierV2Error::Contained)
        } else {
            Ok(())
        }
    }
}

/// Atomically establish the empty generation-zero frontier before any use is
/// allocated.
///
/// This is the only production constructor for a writable frontier in this
/// tranche. Existing durable state cannot be reactivated without a future
/// verifier-owned authenticated restart anchor.
///
/// A store error is outcome-uncertain from this layer's perspective: no live
/// frontier object is returned and callers must reconcile externally rather than
/// assuming the genesis write did or did not occur.
pub fn establish_grant_frontier_v2<S>(
    grant: &CapabilityGrant,
    account: &GrantAccountV2,
    mut store: S,
) -> Result<(EstablishedGrantFrontierV2, CasFrontierV2<S>), FrontierV2Error<S::Error>>
where
    S: CheckpointCasStoreV2,
{
    let existing = store.current_head().map_err(FrontierV2Error::Store)?;
    if existing.is_some() {
        return Err(FrontierV2Error::StoreNotEmpty);
    }

    let checkpoint =
        GrantAccountCheckpointV2::first(grant, account).map_err(FrontierV2Error::Checkpoint)?;
    let head = checkpoint.head().map_err(FrontierV2Error::Checkpoint)?;
    let acknowledged = store
        .compare_and_swap(None, &checkpoint)
        .map_err(FrontierV2Error::Store)?;
    if acknowledged != head {
        return Err(FrontierV2Error::AcknowledgedWrongHead);
    }

    // If another writer advances the frontier before this function returns, do
    // not hand out a writer already known to be stale.
    let observed = store.current_head().map_err(FrontierV2Error::Store)?;
    if observed != Some(head) {
        return Err(FrontierV2Error::StoreFrontierChanged);
    }

    let evidence = EstablishedGrantFrontierV2 {
        checkpoint: checkpoint.clone(),
        head,
    };
    let frontier = CasFrontierV2 {
        store,
        grant: grant.clone(),
        expected_checkpoint: checkpoint,
        expected_head: head,
        contained: false,
    };
    Ok((evidence, frontier))
}

fn exact_single_new_reserved<E>(
    previous: &symthaea_action_runtime::GrantAccountSnapshotV2,
    next: &symthaea_action_runtime::GrantAccountSnapshotV2,
) -> Result<ExecutionReservationV2, FrontierV2Error<E>>
where
    E: StdError + 'static,
{
    if next.reservations.len() != previous.reservations.len().saturating_add(1) {
        return Err(FrontierV2Error::ReservationPersistenceNotSingleAddition);
    }

    for (id, old) in &previous.reservations {
        if next.reservations.get(id) != Some(old) {
            return Err(FrontierV2Error::ReservationPersistenceChangedPriorState);
        }
    }

    let mut new_records = next
        .reservations
        .iter()
        .filter(|(id, _)| !previous.reservations.contains_key(id));
    let (_, record) = new_records
        .next()
        .ok_or(FrontierV2Error::ReservationPersistenceNotSingleAddition)?;
    if new_records.next().is_some() || record.state != ReservationState::Reserved {
        return Err(FrontierV2Error::ReservationPersistenceNotSingleAddition);
    }
    Ok(record.clone())
}

fn exact_single_arm<E>(
    previous: &symthaea_action_runtime::GrantAccountSnapshotV2,
    next: &symthaea_action_runtime::GrantAccountSnapshotV2,
    target: ReservationId,
) -> Result<(), FrontierV2Error<E>>
where
    E: StdError + 'static,
{
    if previous.reservations.len() != next.reservations.len() {
        return Err(FrontierV2Error::ArmChangedOtherState);
    }

    for (id, old) in &previous.reservations {
        let new = next
            .reservations
            .get(id)
            .ok_or(FrontierV2Error::ArmChangedOtherState)?;
        if *id == target {
            if old.state != ReservationState::Reserved
                || new.state != ReservationState::OutcomeUnknown
                || !same_reservation_identity(old, new)
            {
                return Err(FrontierV2Error::ArmTargetMismatch);
            }
        } else if new != old {
            return Err(FrontierV2Error::ArmChangedOtherState);
        }
    }
    Ok(())
}

fn token_matches(record: &ExecutionReservationV2, token: &PersistedReservationV2) -> bool {
    record.reservation_id == token.reservation_id
        && record.effect_intent_id == token.effect_intent_id
        && record.attempt_id == token.attempt_id
        && record.effect_binding_digest == token.effect_binding_digest
        && record.risk_charge == token.risk_charge
}

fn same_reservation_identity(old: &ExecutionReservationV2, new: &ExecutionReservationV2) -> bool {
    old.reservation_id == new.reservation_id
        && old.effect_intent_id == new.effect_intent_id
        && old.attempt_id == new.attempt_id
        && old.effect_binding_digest == new.effect_binding_digest
        && old.risk_charge == new.risk_charge
}

#[derive(Debug, Error)]
pub enum FrontierV2Error<E>
where
    E: StdError + 'static,
{
    #[error("checkpoint validation failed: {0}")]
    Checkpoint(#[source] CheckpointV2Error),
    #[error("checkpoint CAS store failed or outcome is uncertain: {0}")]
    Store(#[source] E),
    #[error("frontier store must be empty for generation-zero establishment")]
    StoreNotEmpty,
    #[error("durable store frontier changed outside this writer")]
    StoreFrontierChanged,
    #[error("checkpoint store acknowledged a different head than the submitted successor")]
    AcknowledgedWrongHead,
    #[error("frontier writer is contained after persistence uncertainty/conflict")]
    Contained,
    #[error("typed reservation persistence requires exactly one new Reserved record")]
    ReservationPersistenceNotSingleAddition,
    #[error("typed reservation persistence changed a previously persisted reservation")]
    ReservationPersistenceChangedPriorState,
    #[error("persisted reservation token belongs to another grant")]
    PersistedReservationGrantMismatch,
    #[error("persisted reservation token no longer names this frontier head")]
    PersistedReservationHeadMismatch,
    #[error("persisted reservation token does not match the exact current Reserved record")]
    PersistedReservationRecordMismatch,
    #[error("durable arming target is not the exact Reserved -> OutcomeUnknown transition")]
    ArmTargetMismatch,
    #[error("durable arming changed another reservation or reservation set")]
    ArmChangedOtherState,
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::sync::{Arc, Mutex};

    use super::*;
    use symthaea_authority::{
        AuthorityContextRef, AuthorityEpoch, Operation, PrincipalId, PurposeId, ResourceRef,
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct CasConflict;

    impl fmt::Display for CasConflict {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("CAS conflict")
        }
    }

    impl StdError for CasConflict {}

    #[derive(Clone, Default)]
    struct SharedCasStore {
        state: Arc<Mutex<Option<CheckpointHeadV2>>>,
    }

    impl CheckpointCasStoreV2 for SharedCasStore {
        type Error = CasConflict;

        fn current_head(&mut self) -> Result<Option<CheckpointHeadV2>, Self::Error> {
            self.state.lock().map(|state| *state).map_err(|_| CasConflict)
        }

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

    struct WrongAckStore {
        state: Option<CheckpointHeadV2>,
    }

    impl CheckpointCasStoreV2 for WrongAckStore {
        type Error = CasConflict;

        fn current_head(&mut self) -> Result<Option<CheckpointHeadV2>, Self::Error> {
            Ok(self.state)
        }

        fn compare_and_swap(
            &mut self,
            expected_previous: Option<CheckpointHeadV2>,
            checkpoint: &GrantAccountCheckpointV2,
        ) -> Result<CheckpointHeadV2, Self::Error> {
            if self.state != expected_previous {
                return Err(CasConflict);
            }
            let actual = checkpoint.head().map_err(|_| CasConflict)?;
            self.state = Some(actual);
            if actual.sequence == 0 {
                Ok(actual)
            } else {
                Ok(CheckpointHeadV2 {
                    sequence: actual.sequence,
                    digest: Digest32([0xee; 32]),
                })
            }
        }
    }

    struct AdvanceAfterCasStore {
        state: Option<CheckpointHeadV2>,
    }

    impl CheckpointCasStoreV2 for AdvanceAfterCasStore {
        type Error = CasConflict;

        fn current_head(&mut self) -> Result<Option<CheckpointHeadV2>, Self::Error> {
            Ok(self.state)
        }

        fn compare_and_swap(
            &mut self,
            expected_previous: Option<CheckpointHeadV2>,
            checkpoint: &GrantAccountCheckpointV2,
        ) -> Result<CheckpointHeadV2, Self::Error> {
            if self.state != expected_previous {
                return Err(CasConflict);
            }
            let actual = checkpoint.head().map_err(|_| CasConflict)?;
            if actual.sequence == 0 {
                self.state = Some(actual);
            } else {
                self.state = Some(CheckpointHeadV2 {
                    sequence: actual.sequence.saturating_add(1),
                    digest: Digest32([0xab; 32]),
                });
            }
            Ok(actual)
        }
    }

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn risk(units: u64) -> RiskBudget {
        RiskBudget {
            mutation_units: units,
            ..RiskBudget::default()
        }
    }

    fn grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "grant-frontier-v2",
            PrincipalId("issuer".into()),
            PrincipalId("subject".into()),
            PurposeId("test-effect".into()),
            AuthorityEpoch(11),
            AuthorityContextRef::new("symthaea.test.context.v1", digest(1)),
        );
        grant.resources.insert(ResourceRef("resource-1".into()));
        grant.operations.insert(Operation("operate".into()));
        grant.max_uses = 4;
        grant.risk_budget = risk(4);
        grant
    }

    fn competing_writer_for_test(
        store: SharedCasStore,
        grant: CapabilityGrant,
        checkpoint: GrantAccountCheckpointV2,
        head: CheckpointHeadV2,
    ) -> CasFrontierV2<SharedCasStore> {
        assert_eq!(*store.state.lock().unwrap(), Some(head));
        CasFrontierV2 {
            store,
            grant,
            expected_checkpoint: checkpoint,
            expected_head: head,
            contained: false,
        }
    }

    #[test]
    fn establishes_empty_generation_zero_before_any_use() {
        let grant = grant();
        let account = GrantAccountV2::new_root(&grant).unwrap();
        let (established, frontier) =
            establish_grant_frontier_v2(&grant, &account, SharedCasStore::default()).unwrap();
        assert_eq!(established.head.sequence, 0);
        assert!(established.checkpoint.snapshot().reservations.is_empty());
        assert_eq!(frontier.expected_head(), established.head);
        assert_eq!(frontier.grant_digest(), grant.digest());
    }

    #[test]
    fn existing_store_cannot_be_reactivated_without_future_authenticator() {
        let grant = grant();
        let shared = SharedCasStore::default();
        let account = GrantAccountV2::new_root(&grant).unwrap();
        let (_established, _frontier) =
            establish_grant_frontier_v2(&grant, &account, shared.clone()).unwrap();
        assert!(matches!(
            establish_grant_frontier_v2(&grant, &account, shared),
            Err(FrontierV2Error::StoreNotEmpty)
        ));
    }

    #[test]
    fn two_stale_writers_cannot_both_publish_successors() {
        let grant = grant();
        let shared = SharedCasStore::default();
        let account = GrantAccountV2::new_root(&grant).unwrap();
        let (established, mut writer_a) =
            establish_grant_frontier_v2(&grant, &account, shared.clone()).unwrap();
        let mut writer_b = competing_writer_for_test(
            shared,
            grant.clone(),
            established.checkpoint.clone(),
            established.head,
        );

        let mut account_a = established.checkpoint.verify_payload(&grant).unwrap();
        account_a
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                risk(1),
            )
            .unwrap();
        let successor_a = GrantAccountCheckpointV2::successor(
            &established.checkpoint,
            &grant,
            &account_a,
        )
        .unwrap();

        let mut account_b = established.checkpoint.verify_payload(&grant).unwrap();
        account_b
            .reserve_execution(
                EffectIntentId(digest(5)),
                AttemptId(digest(6)),
                EffectBindingDigest(digest(7)),
                risk(1),
            )
            .unwrap();
        let successor_b = GrantAccountCheckpointV2::successor(
            &established.checkpoint,
            &grant,
            &account_b,
        )
        .unwrap();

        writer_a.persist_successor(successor_a).unwrap();
        assert!(matches!(
            writer_b.persist_successor(successor_b),
            Err(FrontierV2Error::Store(CasConflict))
        ));
        assert!(writer_b.is_contained());
    }

    #[test]
    fn exact_reserved_then_outcome_unknown_mints_typed_persistence_tokens() {
        let grant = grant();
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let (_, mut frontier) =
            establish_grant_frontier_v2(&grant, &account, SharedCasStore::default()).unwrap();

        let reservation = account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                risk(1),
            )
            .unwrap();
        let reserved_checkpoint = GrantAccountCheckpointV2::successor(
            frontier.expected_checkpoint(),
            &grant,
            &account,
        )
        .unwrap();
        let persisted = frontier.persist_new_reservation(reserved_checkpoint).unwrap();
        assert_eq!(persisted.reservation_id(), reservation);
        assert_eq!(persisted.persisted_head(), frontier.expected_head());

        account.mark_outcome_unknown(reservation).unwrap();
        let unknown_checkpoint = GrantAccountCheckpointV2::successor(
            frontier.expected_checkpoint(),
            &grant,
            &account,
        )
        .unwrap();
        let armed = frontier
            .arm_outcome_unknown(persisted, unknown_checkpoint)
            .unwrap();
        assert_eq!(armed.reservation_id(), reservation);
        assert_eq!(armed.armed_head(), frontier.expected_head());
        assert_ne!(armed.reserved_head(), armed.armed_head());
    }

    #[test]
    fn persisted_reservation_token_is_head_bound() {
        let grant = grant();
        let mut account = GrantAccountV2::new_root(&grant).unwrap();
        let (_, mut frontier) =
            establish_grant_frontier_v2(&grant, &account, SharedCasStore::default()).unwrap();
        let reservation = account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                risk(1),
            )
            .unwrap();
        let reserved_checkpoint = GrantAccountCheckpointV2::successor(
            frontier.expected_checkpoint(),
            &grant,
            &account,
        )
        .unwrap();
        let persisted = frontier.persist_new_reservation(reserved_checkpoint).unwrap();

        let no_op = GrantAccountCheckpointV2::successor(
            frontier.expected_checkpoint(),
            &grant,
            &account,
        )
        .unwrap();
        frontier.persist_successor(no_op).unwrap();

        account.mark_outcome_unknown(reservation).unwrap();
        let unknown = GrantAccountCheckpointV2::successor(
            frontier.expected_checkpoint(),
            &grant,
            &account,
        )
        .unwrap();
        assert!(matches!(
            frontier.arm_outcome_unknown(persisted, unknown),
            Err(FrontierV2Error::PersistedReservationHeadMismatch)
        ));
    }

    #[test]
    fn wrong_acknowledgement_contains_writer() {
        let grant = grant();
        let account = GrantAccountV2::new_root(&grant).unwrap();
        let (_, mut frontier) = establish_grant_frontier_v2(
            &grant,
            &account,
            WrongAckStore { state: None },
        )
        .unwrap();

        let mut next_account = GrantAccountV2::new_root(&grant).unwrap();
        next_account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                risk(1),
            )
            .unwrap();
        let successor = GrantAccountCheckpointV2::successor(
            frontier.expected_checkpoint(),
            &grant,
            &next_account,
        )
        .unwrap();
        assert!(matches!(
            frontier.persist_successor(successor),
            Err(FrontierV2Error::AcknowledgedWrongHead)
        ));
        assert!(frontier.is_contained());
    }

    #[test]
    fn post_cas_external_advance_prevents_positive_persistence_fact() {
        let grant = grant();
        let account = GrantAccountV2::new_root(&grant).unwrap();
        let (_, mut frontier) = establish_grant_frontier_v2(
            &grant,
            &account,
            AdvanceAfterCasStore { state: None },
        )
        .unwrap();

        let mut next_account = GrantAccountV2::new_root(&grant).unwrap();
        next_account
            .reserve_execution(
                EffectIntentId(digest(2)),
                AttemptId(digest(3)),
                EffectBindingDigest(digest(4)),
                risk(1),
            )
            .unwrap();
        let successor = GrantAccountCheckpointV2::successor(
            frontier.expected_checkpoint(),
            &grant,
            &next_account,
        )
        .unwrap();
        assert!(matches!(
            frontier.persist_new_reservation(successor),
            Err(FrontierV2Error::StoreFrontierChanged)
        ));
        assert!(frontier.is_contained());
    }
}
