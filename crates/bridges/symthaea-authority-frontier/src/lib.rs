// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Linearizable checkpoint-frontier binding for bounded authority accounting.
//!
//! This crate proves a deliberately narrow property:
//!
//! ```text
//! supplied bootstrap anchor
//! + store currently equals that anchor
//! + every successor installed by exact compare-and-swap
//! + fresh store observation still equals the adapter's expected head
//! + checkpoint/accounting bound to that same head
//!     -> FrontierBoundGrantAccounting
//! ```
//!
//! The bootstrap anchor is *not* authenticated here. A Xenia/TPM/append-only-log
//! verifier must separately authenticate it before any stronger currentness claim.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_action_checkpoint::{
    CheckpointError, CheckpointHead, GrantAccountCheckpoint, HeadBoundGrantAccounting,
};
use symthaea_action_runtime::{GrantAccount, RuntimeAccountingError};
use symthaea_authority::{CapabilityGrant, Digest32, GrantUseState, RiskBudget};
use thiserror::Error;

/// Linearizable durable frontier store.
///
/// `compare_and_swap` must compare the full current head and install the new
/// checkpoint atomically. `current_head` must return the store's current durable
/// frontier as observed by the same storage authority.
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

/// Opaque accounting state bound to one freshly observed store frontier derived
/// from one supplied bootstrap anchor. Not Clone and not Serde.
#[derive(Debug)]
pub struct FrontierBoundGrantAccounting {
    bootstrap_anchor: CheckpointHead,
    accounting: HeadBoundGrantAccounting,
}

impl FrontierBoundGrantAccounting {
    pub fn grant_digest(&self) -> Digest32 {
        self.accounting.grant_digest()
    }

    pub fn bootstrap_anchor(&self) -> CheckpointHead {
        self.bootstrap_anchor
    }

    /// Store frontier observed at the binding operation. This is a point-in-time
    /// fact; another writer may advance the durable store afterward.
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

/// In-process frontier state relative to one supplied bootstrap anchor.
///
/// The anchor is explicitly a claim, not an authenticated fact. After creation,
/// stale-writer conflicts, observed external advancement, or persistence
/// uncertainty latch the frontier into containment.
pub struct CasCheckpointFrontier<S>
where
    S: CheckpointCasStore,
{
    inner: S,
    bootstrap_anchor: CheckpointHead,
    expected_head: CheckpointHead,
    contained: bool,
}

impl<S> CasCheckpointFrontier<S>
where
    S: CheckpointCasStore,
{
    /// Reopen at one supplied bootstrap anchor only if the store currently
    /// reports the exact same full head.
    ///
    /// This proves equality with the store; it does not authenticate the anchor.
    pub fn reopen_from_anchor_claim(
        mut inner: S,
        bootstrap_anchor: CheckpointHead,
    ) -> Result<Self, FrontierError<S::Error>> {
        validate_head(bootstrap_anchor)?;
        let actual = inner.current_head().map_err(FrontierError::Store)?;
        if actual != Some(bootstrap_anchor) {
            return Err(FrontierError::BootstrapAnchorMismatch);
        }
        Ok(Self {
            inner,
            bootstrap_anchor,
            expected_head: bootstrap_anchor,
            contained: false,
        })
    }

    pub fn bootstrap_anchor(&self) -> CheckpointHead {
        self.bootstrap_anchor
    }

    /// Adapter-local expected frontier. This is not a fresh store read.
    pub fn expected_head(&self) -> CheckpointHead {
        self.expected_head
    }

    pub fn is_contained(&self) -> bool {
        self.contained
    }

    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Persist one exact successor of the current expected frontier.
    pub fn persist_successor(
        &mut self,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<CheckpointHead, FrontierError<S::Error>> {
        if self.contained {
            return Err(FrontierError::Contained);
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

        self.expected_head = expected_next;
        Ok(expected_next)
    }

    /// Bind head-bound accounting only after a fresh store read confirms the
    /// durable store still equals this adapter's expected head.
    ///
    /// The input is consumed so the resulting object owns the exact accounting
    /// state that matched that freshly observed head.
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
        if accounting.grant_digest() != self.expected_head.grant_digest {
            return Err(FrontierError::AccountingGrantMismatch);
        }
        Ok(FrontierBoundGrantAccounting {
            bootstrap_anchor: self.bootstrap_anchor,
            accounting,
        })
    }

    fn validate_successor_shape(
        &self,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<(), FrontierError<S::Error>> {
        if checkpoint.grant_digest != self.expected_head.grant_digest {
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

/// Establish generation zero in an empty linearizable store.
///
/// The returned head is current relative to this store at establishment, but
/// still requires an external authenticator before it can become a trusted
/// restart anchor.
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

    Ok((
        EstablishedGrantFrontier {
            checkpoint,
            head,
        },
        CasCheckpointFrontier {
            inner: store,
            bootstrap_anchor: head,
            expected_head: head,
            contained: false,
        },
    ))
}

fn validate_head<E>(head: CheckpointHead) -> Result<(), FrontierError<E>>
where
    E: StdError + 'static,
{
    if head.grant_digest.0 == [0; 32] || head.digest.0 == [0; 32] {
        Err(FrontierError::InvalidHead)
    } else {
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum FrontierError<E>
where
    E: StdError + 'static,
{
    #[error("checkpoint validation failed: {0}")]
    Checkpoint(#[source] CheckpointError),
    #[error("runtime accounting failed: {0}")]
    Runtime(#[source] RuntimeAccountingError),
    #[error("frontier store failed: {0}")]
    Store(#[source] E),
    #[error("bootstrap anchor is malformed")]
    InvalidHead,
    #[error("store current head does not equal the supplied bootstrap anchor")]
    BootstrapAnchorMismatch,
    #[error("frontier store is not empty during generation-zero establishment")]
    StoreNotEmpty,
    #[error("frontier is contained after prior persistence uncertainty or conflict")]
    Contained,
    #[error("checkpoint store acknowledged a different successor head")]
    AcknowledgedWrongHead,
    #[error("durable store frontier changed outside this adapter")]
    StoreFrontierChanged,
    #[error("successor attempted to change the grant bound to this frontier")]
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

    #[test]
    fn bootstrap_establishes_exact_grant_bound_generation_zero() {
        let grant = grant("g1");
        let (established, frontier) =
            establish_grant_frontier(&grant, SharedCasStore::default()).unwrap();
        assert_eq!(established.head.grant_digest, grant.digest());
        assert_eq!(established.head.sequence, 0);
        assert_eq!(frontier.expected_head(), established.head);
    }

    #[test]
    fn reopen_requires_store_to_equal_supplied_anchor() {
        let grant = grant("g1");
        let (established, frontier) =
            establish_grant_frontier(&grant, SharedCasStore::default()).unwrap();
        let store = frontier.into_inner();
        let mut wrong = established.head;
        wrong.digest = digest(99);
        assert!(matches!(
            CasCheckpointFrontier::reopen_from_anchor_claim(store.clone(), wrong),
            Err(FrontierError::BootstrapAnchorMismatch)
        ));
        assert!(CasCheckpointFrontier::reopen_from_anchor_claim(store, established.head).is_ok());
    }

    #[test]
    fn two_stale_writers_cannot_both_publish_successors() {
        let grant = grant("g1");
        let shared = SharedCasStore::default();
        let (established, frontier_a) = establish_grant_frontier(&grant, shared.clone()).unwrap();
        let mut frontier_a = frontier_a;
        let mut frontier_b =
            CasCheckpointFrontier::reopen_from_anchor_claim(shared.clone(), established.head)
                .unwrap();

        let mut account = GrantAccount::new(&grant).unwrap();
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
        let successor = GrantAccountCheckpoint::successor(
            &established.checkpoint,
            &grant,
            account.snapshot(),
        )
        .unwrap();

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
        let (established, mut frontier_a) = establish_grant_frontier(&grant, shared.clone()).unwrap();
        let mut frontier_b =
            CasCheckpointFrontier::reopen_from_anchor_claim(shared, established.head).unwrap();

        let stale = bind_checkpoint_to_head(
            &grant,
            &established.checkpoint,
            established.head,
        )
        .unwrap();

        let mut account = GrantAccount::new(&grant).unwrap();
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
        let successor = GrantAccountCheckpoint::successor(
            &established.checkpoint,
            &grant,
            account.snapshot(),
        )
        .unwrap();
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

        let genesis_accounting = bind_checkpoint_to_head(
            &grant,
            &established.checkpoint,
            established.head,
        )
        .unwrap();
        let current = frontier.bind_current_accounting(genesis_accounting).unwrap();
        assert_eq!(current.observed_head(), established.head);
        assert_eq!(current.grant_digest(), grant.digest());

        let mut account = GrantAccount::new(&grant).unwrap();
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
        let successor = GrantAccountCheckpoint::successor(
            &established.checkpoint,
            &grant,
            account.snapshot(),
        )
        .unwrap();
        let successor_head = frontier.persist_successor(&successor).unwrap();

        let stale = bind_checkpoint_to_head(
            &grant,
            &established.checkpoint,
            established.head,
        )
        .unwrap();
        assert!(matches!(
            frontier.bind_current_accounting(stale),
            Err(FrontierError::AccountingHeadMismatch)
        ));

        let fresh = bind_checkpoint_to_head(&grant, &successor, successor_head).unwrap();
        assert_eq!(
            frontier.bind_current_accounting(fresh).unwrap().observed_head(),
            successor_head
        );
    }

    #[test]
    fn successor_cannot_change_grant() {
        let grant_a = grant("a");
        let grant_b = grant("b");
        let (established, frontier) =
            establish_grant_frontier(&grant_a, SharedCasStore::default()).unwrap();
        let account_b = GrantAccount::new(&grant_b).unwrap();
        let foreign = GrantAccountCheckpoint::successor(
            &established.checkpoint,
            &grant_b,
            account_b.snapshot(),
        );
        assert!(foreign.is_err());

        // Constructing a foreign successor through the safe checkpoint API is
        // already impossible because predecessor verification is grant-bound.
        assert!(!frontier.is_contained());
    }
}
