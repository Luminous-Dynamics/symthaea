// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Linearizable checkpoint-frontier binding for bounded authority accounting.
//!
//! This crate proves a deliberately narrow property for a frontier established
//! from an empty durable store:
//!
//! ```text
//! exact CapabilityGrant
//! + empty linearizable store
//! + exact generation-zero checkpoint
//! + every successor payload valid under the exact grant
//! + every successor installed by full-head compare-and-swap
//! + fresh store observation still equals adapter expected head
//! + accounting bound to that same head/grant
//!     -> FrontierBoundGrantAccounting
//! ```
//!
//! There is deliberately no production restart/reopen constructor in v0.2.
//! Re-activating a persisted frontier after process loss requires a future
//! externally authenticated checkpoint-anchor proof (Xenia/TPM/append-only log
//! or equivalent). A caller-provided `CheckpointHead` cannot clear containment
//! or recreate writable authority progression in this crate.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_action_checkpoint::{
    CheckpointError, CheckpointHead, GrantAccountCheckpoint, HeadBoundGrantAccounting,
};
use symthaea_action_runtime::{GrantAccount, RuntimeAccountingError};
use symthaea_authority::{
    CapabilityGrant, Digest32, GrantUseState, GrantValidationError, RiskBudget,
};
use thiserror::Error;

/// Linearizable durable frontier store.
///
/// `current_head` must be a linearizable read of the same durable frontier used
/// by `compare_and_swap`. `compare_and_swap` must atomically compare the full
/// current head, durably install the supplied checkpoint, and return the exact
/// full head of the installed checkpoint on success.
pub trait CheckpointCasStore {
    type Error: StdError + 'static;

    fn current_head(&mut self) -> Result<Option<CheckpointHead>, Self::Error>;

    fn compare_and_swap(
        &mut self,
        expected_previous: Option<CheckpointHead>,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<CheckpointHead, Self::Error>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EstablishedGrantFrontier {
    pub checkpoint: GrantAccountCheckpoint,
    pub head: CheckpointHead,
}

/// Opaque accounting state bound to one freshly observed durable-store frontier.
/// Not Clone and not Serde.
///
/// This remains a point-in-time fact. Another writer may advance the store after
/// this object is created; any later admission layer must consume it together
/// with a fresh/atomic frontier transition rather than treating it as an
/// indefinitely current capability.
#[derive(Debug)]
pub struct FrontierBoundGrantAccounting {
    establishment_head: CheckpointHead,
    accounting: HeadBoundGrantAccounting,
}

impl FrontierBoundGrantAccounting {
    pub fn grant_digest(&self) -> Digest32 {
        self.accounting.grant_digest()
    }

    pub fn establishment_head(&self) -> CheckpointHead {
        self.establishment_head
    }

    /// Store frontier observed at the binding operation.
    pub fn observed_head(&self) -> CheckpointHead {
        self.accounting.checkpoint_head()
    }

    pub fn use_state(&self) -> &GrantUseState {
        self.accounting.use_state()
    }

    pub fn charged_risk(&self) -> RiskBudget {
        self.accounting.charged_risk()
    }
}

/// In-process writable frontier for one exact grant, created only by successful
/// generation-zero establishment in this schema.
///
/// Stale-writer conflicts, observed external advancement, malformed successor
/// payloads, or persistence uncertainty latch the frontier into containment.
/// The production API exposes no method that clears containment and no method
/// that reopens a frontier from caller-provided persisted identity.
pub struct CasCheckpointFrontier<S>
where
    S: CheckpointCasStore,
{
    inner: S,
    grant: CapabilityGrant,
    establishment_head: CheckpointHead,
    expected_head: CheckpointHead,
    contained: bool,
}

impl<S> CasCheckpointFrontier<S>
where
    S: CheckpointCasStore,
{
    pub fn grant_digest(&self) -> Digest32 {
        self.grant.digest()
    }

    pub fn establishment_head(&self) -> CheckpointHead {
        self.establishment_head
    }

    /// Adapter-local expected frontier. This is not a fresh store read.
    pub fn expected_head(&self) -> CheckpointHead {
        self.expected_head
    }

    pub fn is_contained(&self) -> bool {
        self.contained
    }

    /// Persist one exact, internally valid successor of the current expected
    /// frontier.
    ///
    /// A matching sequence/predecessor/grant digest is not enough: the entire
    /// checkpoint accounting payload must first verify under the exact grant.
    pub fn persist_successor(
        &mut self,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<CheckpointHead, FrontierError<S::Error>> {
        if self.contained {
            return Err(FrontierError::Contained);
        }

        if let Err(error) = checkpoint.verify_payload(&self.grant) {
            self.contained = true;
            return Err(FrontierError::Checkpoint(error));
        }
        if let Err(error) = self.validate_successor_shape(checkpoint) {
            self.contained = true;
            return Err(error);
        }

        let expected_next = match checkpoint.head() {
            Ok(head) => head,
            Err(error) => {
                self.contained = true;
                return Err(FrontierError::Checkpoint(error));
            }
        };

        let acknowledged = match self
            .inner
            .compare_and_swap(Some(self.expected_head), checkpoint)
        {
            Ok(head) => head,
            Err(error) => {
                self.contained = true;
                return Err(FrontierError::Store(error));
            }
        };
        if acknowledged != expected_next {
            self.contained = true;
            return Err(FrontierError::AcknowledgedWrongHead);
        }

        // Do not return a newly advanced adapter if another writer already moved
        // the durable frontier again before this operation completes.
        let observed = match self.inner.current_head() {
            Ok(head) => head,
            Err(error) => {
                self.contained = true;
                return Err(FrontierError::Store(error));
            }
        };
        if observed != Some(expected_next) {
            self.contained = true;
            return Err(FrontierError::StoreFrontierChanged);
        }

        self.expected_head = expected_next;
        Ok(expected_next)
    }

    /// Bind head-bound accounting only after a fresh linearizable store read
    /// confirms the durable frontier still equals this adapter's expected head.
    ///
    /// The input is consumed so the resulting object owns the exact accounting
    /// state that matched that freshly observed point-in-time frontier.
    pub fn bind_current_accounting(
        &mut self,
        accounting: HeadBoundGrantAccounting,
    ) -> Result<FrontierBoundGrantAccounting, FrontierError<S::Error>> {
        if self.contained {
            return Err(FrontierError::Contained);
        }

        let actual = match self.inner.current_head() {
            Ok(head) => head,
            Err(error) => {
                self.contained = true;
                return Err(FrontierError::Store(error));
            }
        };
        if actual != Some(self.expected_head) {
            self.contained = true;
            return Err(FrontierError::StoreFrontierChanged);
        }
        if accounting.checkpoint_head() != self.expected_head {
            return Err(FrontierError::AccountingHeadMismatch);
        }
        if accounting.grant_digest() != self.grant.digest() {
            return Err(FrontierError::AccountingGrantMismatch);
        }

        Ok(FrontierBoundGrantAccounting {
            establishment_head: self.establishment_head,
            accounting,
        })
    }

    fn validate_successor_shape(
        &self,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<(), FrontierError<S::Error>> {
        if checkpoint.grant_digest != self.grant.digest()
            || checkpoint.grant_digest != self.expected_head.grant_digest
        {
            return Err(FrontierError::GrantChanged);
        }
        let expected_sequence = self
            .expected_head
            .sequence
            .checked_add(1)
            .ok_or(FrontierError::SequenceOverflow)?;
        if checkpoint.sequence != expected_sequence {
            return Err(FrontierError::SequenceMismatch);
        }
        if checkpoint.previous_checkpoint_digest != Some(self.expected_head.digest) {
            return Err(FrontierError::PreviousDigestMismatch);
        }
        Ok(())
    }
}

/// Establish generation zero in an empty linearizable store for one exact grant.
///
/// This is the only production constructor for a writable frontier in v0.2.
/// Restart/recovery activation is intentionally absent until an external anchor
/// authenticator exists.
pub fn establish_grant_frontier<S>(
    grant: &CapabilityGrant,
    mut store: S,
) -> Result<
    (EstablishedGrantFrontier, CasCheckpointFrontier<S>),
    FrontierError<S::Error>,
>
where
    S: CheckpointCasStore,
{
    grant.validate().map_err(FrontierError::Grant)?;
    let existing = store.current_head().map_err(FrontierError::Store)?;
    if existing.is_some() {
        return Err(FrontierError::StoreNotEmpty);
    }

    let account = GrantAccount::new(grant).map_err(FrontierError::Runtime)?;
    let checkpoint = GrantAccountCheckpoint::first(grant, account.snapshot())
        .map_err(FrontierError::Checkpoint)?;
    let head = checkpoint.head().map_err(FrontierError::Checkpoint)?;
    let acknowledged = store
        .compare_and_swap(None, &checkpoint)
        .map_err(FrontierError::Store)?;
    if acknowledged != head {
        return Err(FrontierError::AcknowledgedWrongHead);
    }

    // A concurrent successor immediately after genesis is safe globally but
    // would make the returned adapter stale. Refuse to return that stale view.
    let observed = store.current_head().map_err(FrontierError::Store)?;
    if observed != Some(head) {
        return Err(FrontierError::StoreFrontierChanged);
    }

    Ok((
        EstablishedGrantFrontier {
            checkpoint,
            head,
        },
        CasCheckpointFrontier {
            inner: store,
            grant: grant.clone(),
            establishment_head: head,
            expected_head: head,
            contained: false,
        },
    ))
}

#[derive(Debug, Error)]
pub enum FrontierError<E>
where
    E: StdError + 'static,
{
    #[error("invalid capability grant: {0}")]
    Grant(#[source] GrantValidationError),
    #[error("checkpoint validation failed: {0}")]
    Checkpoint(#[source] CheckpointError),
    #[error("runtime accounting failed: {0}")]
    Runtime(#[source] RuntimeAccountingError),
    #[error("frontier store failed: {0}")]
    Store(#[source] E),
    #[error("frontier store is not empty during generation-zero establishment")]
    StoreNotEmpty,
    #[error("frontier is contained after prior persistence uncertainty or conflict")]
    Contained,
    #[error("checkpoint store acknowledged a different successor head")]
    AcknowledgedWrongHead,
    #[error("durable store frontier changed outside this adapter")]
    StoreFrontierChanged,
    #[error("successor attempted to change the exact grant bound to this frontier")]
    GrantChanged,
    #[error("checkpoint sequence overflow")]
    SequenceOverflow,
    #[error("checkpoint sequence does not follow the expected frontier")]
    SequenceMismatch,
    #[error("checkpoint predecessor digest does not equal the expected frontier")]
    PreviousDigestMismatch,
    #[error("head-bound accounting does not match the freshly observed frontier")]
    AccountingHeadMismatch,
    #[error("head-bound accounting grant does not match the frontier grant")]
    AccountingGrantMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;
    use std::sync::{Arc, Mutex};
    use symthaea_action_checkpoint::bind_checkpoint_to_head;
    use symthaea_action_runtime::{ExecutionId, ReservationId};
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
        state: Arc<Mutex<Option<CheckpointHead>>>,
    }

    impl CheckpointCasStore for SharedCasStore {
        type Error = CasConflict;

        fn current_head(&mut self) -> Result<Option<CheckpointHead>, Self::Error> {
            self.state.lock().map(|state| *state).map_err(|_| CasConflict)
        }

        fn compare_and_swap(
            &mut self,
            expected_previous: Option<CheckpointHead>,
            checkpoint: &GrantAccountCheckpoint,
        ) -> Result<CheckpointHead, Self::Error> {
            let mut state = self.state.lock().map_err(|_| CasConflict)?;
            if *state != expected_previous {
                return Err(CasConflict);
            }
            let next = checkpoint.head().map_err(|_| CasConflict)?;
            *state = Some(next);
            Ok(next)
        }
    }

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn grant(id: &str) -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            id,
            PrincipalId("issuer".into()),
            PrincipalId("robot".into()),
            PurposeId("goal-directed-actuation".into()),
            AuthorityEpoch(7),
            AuthorityContextRef::new("swarm-controller", digest(9)),
        );
        grant.resources.insert(ResourceRef("robot".into()));
        grant.operations.insert(Operation("move".into()));
        grant.max_uses = 3;
        grant.risk_budget.mutation_units = 3;
        grant
    }

    fn reserved_successor(
        established: &EstablishedGrantFrontier,
        grant: &CapabilityGrant,
    ) -> GrantAccountCheckpoint {
        let mut account = GrantAccount::new(grant).unwrap();
        account
            .reserve_execution(
                ReservationId("r1".into()),
                ExecutionId("e1".into()),
                digest(1),
                RiskBudget {
                    mutation_units: 1,
                    ..RiskBudget::default()
                },
            )
            .unwrap();
        GrantAccountCheckpoint::successor(
            &established.checkpoint,
            grant,
            account.snapshot(),
        )
        .unwrap()
    }

    /// Test-only constructor for competing-writer simulations. Production code
    /// intentionally has no restart/reopen constructor in v0.2.
    fn competing_frontier_for_test(
        store: SharedCasStore,
        grant: CapabilityGrant,
        establishment_head: CheckpointHead,
    ) -> CasCheckpointFrontier<SharedCasStore> {
        assert_eq!(*store.state.lock().unwrap(), Some(establishment_head));
        CasCheckpointFrontier {
            inner: store,
            grant,
            establishment_head,
            expected_head: establishment_head,
            contained: false,
        }
    }

    #[test]
    fn bootstrap_establishes_exact_grant_bound_generation_zero() {
        let grant = grant("g1");
        let (established, frontier) =
            establish_grant_frontier(&grant, SharedCasStore::default()).unwrap();
        assert_eq!(established.head.grant_digest, grant.digest());
        assert_eq!(established.head.sequence, 0);
        assert_eq!(frontier.grant_digest(), grant.digest());
        assert_eq!(frontier.expected_head(), established.head);
    }

    #[test]
    fn existing_store_cannot_be_reactivated_without_future_authenticator() {
        let grant = grant("g1");
        let shared = SharedCasStore::default();
        let (_established, _frontier) =
            establish_grant_frontier(&grant, shared.clone()).unwrap();
        assert!(matches!(
            establish_grant_frontier(&grant, shared),
            Err(FrontierError::StoreNotEmpty)
        ));
    }

    #[test]
    fn malformed_checkpoint_payload_cannot_advance_frontier() {
        let grant = grant("g1");
        let shared = SharedCasStore::default();
        let (established, mut frontier) =
            establish_grant_frontier(&grant, shared.clone()).unwrap();
        let mut successor = reserved_successor(&established, &grant);
        successor.snapshot.max_uses = grant.max_uses + 100;

        assert!(matches!(
            frontier.persist_successor(&successor),
            Err(FrontierError::Checkpoint(_))
        ));
        assert!(frontier.is_contained());
        assert_eq!(*shared.state.lock().unwrap(), Some(established.head));
    }

    #[test]
    fn two_stale_writers_cannot_both_publish_successors() {
        let grant = grant("g1");
        let shared = SharedCasStore::default();
        let (established, mut frontier_a) =
            establish_grant_frontier(&grant, shared.clone()).unwrap();
        let mut frontier_b = competing_frontier_for_test(
            shared.clone(),
            grant.clone(),
            established.head,
        );
        let successor = reserved_successor(&established, &grant);

        let first = frontier_a.persist_successor(&successor).unwrap();
        assert_eq!(frontier_a.expected_head(), first);
        assert!(matches!(
            frontier_b.persist_successor(&successor),
            Err(FrontierError::Store(CasConflict))
        ));
        assert!(frontier_b.is_contained());
    }

    #[test]
    fn binding_detects_external_frontier_advance() {
        let grant = grant("g1");
        let shared = SharedCasStore::default();
        let (established, mut frontier_a) =
            establish_grant_frontier(&grant, shared.clone()).unwrap();
        let mut frontier_b = competing_frontier_for_test(
            shared,
            grant.clone(),
            established.head,
        );

        let stale = bind_checkpoint_to_head(&grant, &established.checkpoint, established.head)
            .unwrap();
        let successor = reserved_successor(&established, &grant);
        frontier_a.persist_successor(&successor).unwrap();

        assert!(matches!(
            frontier_b.bind_current_accounting(stale),
            Err(FrontierError::StoreFrontierChanged)
        ));
        assert!(frontier_b.is_contained());
    }

    #[test]
    fn current_accounting_must_match_freshly_observed_frontier_exactly() {
        let grant = grant("g1");
        let (established, mut frontier) =
            establish_grant_frontier(&grant, SharedCasStore::default()).unwrap();

        let genesis_accounting =
            bind_checkpoint_to_head(&grant, &established.checkpoint, established.head).unwrap();
        let current = frontier.bind_current_accounting(genesis_accounting).unwrap();
        assert_eq!(current.observed_head(), established.head);
        assert_eq!(current.grant_digest(), grant.digest());

        let successor = reserved_successor(&established, &grant);
        let successor_head = frontier.persist_successor(&successor).unwrap();

        let stale =
            bind_checkpoint_to_head(&grant, &established.checkpoint, established.head).unwrap();
        assert!(matches!(
            frontier.bind_current_accounting(stale),
            Err(FrontierError::AccountingHeadMismatch)
        ));

        let fresh = bind_checkpoint_to_head(&grant, &successor, successor_head).unwrap();
        assert_eq!(
            frontier
                .bind_current_accounting(fresh)
                .unwrap()
                .observed_head(),
            successor_head
        );
    }
}
