// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compare-and-swap authority frontier for the Symthaea Agency Kernel.
//!
//! A local hash chain prevents undetected rewriting only relative to a trusted
//! head. This crate adds a stronger persistence contract: every new checkpoint
//! must atomically replace one exact expected head. A stale writer therefore
//! cannot create a second accepted successor from an already-advanced frontier.
//!
//! It also establishes a generation-zero checkpoint before consequential
//! authority is requested, giving Xenia or another authority service a concrete
//! anti-rollback anchor to bind before any effect-entry reservation exists.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_action_checkpoint::{
    CheckpointError, CheckpointHead, GrantAccountCheckpoint,
};
use symthaea_action_runtime::GrantAccount;
use symthaea_authority::CapabilityGrant;
use symthaea_system_broker::CheckpointStore;
use thiserror::Error;

/// Atomic external frontier store.
///
/// Implementations must compare the current durable head with
/// `expected_previous` and install `checkpoint` as the new durable frontier in
/// one atomic/linearizable operation. Returning success after a non-atomic
/// read-then-write does not satisfy this contract.
pub trait CheckpointCasStore {
    type Error: StdError + 'static;

    fn compare_and_swap(
        &mut self,
        expected_previous: Option<CheckpointHead>,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<CheckpointHead, Self::Error>;
}

/// Exact generation-zero authority frontier established before delegation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EstablishedGrantFrontier {
    pub checkpoint: GrantAccountCheckpoint,
    pub head: CheckpointHead,
}

/// Adapter that makes a CAS store usable by the existing typed system broker.
///
/// The adapter tracks one exact expected head. Any persistence uncertainty or
/// stale-writer rejection latches it into containment. The enclosing broker has
/// its own independent containment latch as well.
pub struct CasCheckpointStoreAdapter<S> {
    inner: S,
    expected_head: Option<CheckpointHead>,
    contained: bool,
}

impl<S> CasCheckpointStoreAdapter<S>
where
    S: CheckpointCasStore,
{
    /// Restore an adapter at one externally trusted current frontier.
    pub fn from_trusted_head(inner: S, trusted_head: CheckpointHead) -> Self {
        Self {
            inner,
            expected_head: Some(trusted_head),
            contained: false,
        }
    }

    pub fn expected_head(&self) -> Option<CheckpointHead> {
        self.expected_head
    }

    pub fn is_contained(&self) -> bool {
        self.contained
    }

    pub fn into_inner(self) -> S {
        self.inner
    }

    fn validate_successor_shape(
        &self,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<(), FrontierError<S::Error>> {
        match self.expected_head {
            None => {
                if checkpoint.sequence != 0 || checkpoint.previous_checkpoint_digest.is_some() {
                    return Err(FrontierError::UnexpectedGenesisShape);
                }
            }
            Some(previous) => {
                let expected_sequence = previous
                    .sequence
                    .checked_add(1)
                    .ok_or(FrontierError::SequenceOverflow)?;
                if checkpoint.sequence != expected_sequence {
                    return Err(FrontierError::SequenceMismatch);
                }
                if checkpoint.previous_checkpoint_digest != Some(previous.digest) {
                    return Err(FrontierError::PreviousDigestMismatch);
                }
            }
        }
        Ok(())
    }
}

impl<S> CheckpointStore for CasCheckpointStoreAdapter<S>
where
    S: CheckpointCasStore,
{
    type Error = FrontierError<S::Error>;

    fn persist(
        &mut self,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<CheckpointHead, Self::Error> {
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
            .compare_and_swap(self.expected_head, checkpoint)
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
        self.expected_head = Some(expected_next);
        Ok(expected_next)
    }
}

/// Establish a durable generation-zero checkpoint and return both its exact
/// frontier and a CAS adapter already positioned to persist its successors.
///
/// This should happen **before** Xenia or another authority service signs a
/// consequential capability. The returned `head` is the anti-rollback anchor
/// that the authority statement should bind.
pub fn establish_grant_frontier<S>(
    grant: &CapabilityGrant,
    mut store: S,
) -> Result<(EstablishedGrantFrontier, CasCheckpointStoreAdapter<S>), FrontierError<S::Error>>
where
    S: CheckpointCasStore,
{
    let account = GrantAccount::new(grant);
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
        CasCheckpointStoreAdapter {
            inner: store,
            expected_head: Some(head),
            contained: false,
        },
    ))
}

/// CAS/frontier failures. Any error after a consequential session starts should
/// be treated as a containment condition until trusted state is re-established.
#[derive(Debug, Error)]
pub enum FrontierError<E>
where
    E: StdError + 'static,
{
    #[error("checkpoint validation failed: {0}")]
    Checkpoint(#[source] CheckpointError),
    #[error("CAS checkpoint store failed: {0}")]
    Store(#[source] E),
    #[error("checkpoint store acknowledged a different head")]
    AcknowledgedWrongHead,
    #[error("CAS adapter is contained after prior persistence uncertainty")]
    Contained,
    #[error("checkpoint did not have the required generation-zero shape")]
    UnexpectedGenesisShape,
    #[error("checkpoint sequence overflow")]
    SequenceOverflow,
    #[error("checkpoint sequence does not follow the trusted frontier")]
    SequenceMismatch,
    #[error("checkpoint predecessor does not equal the trusted frontier")]
    PreviousDigestMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;
    use std::sync::{Arc, Mutex};
    use symthaea_action_checkpoint::GrantAccountCheckpoint;
    use symthaea_authority::{AuthorityEpoch, PrincipalId, RiskBudget};

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

    fn grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "g1",
            PrincipalId("issuer".into()),
            PrincipalId("subject".into()),
            AuthorityEpoch(3),
        );
        grant.max_uses = 2;
        grant.risk_budget = RiskBudget {
            mutation_units: 2,
            ..RiskBudget::default()
        };
        grant
    }

    #[test]
    fn bootstrap_establishes_generation_zero_before_any_use() {
        let grant = grant();
        let (frontier, adapter) = establish_grant_frontier(&grant, SharedCasStore::default()).unwrap();
        assert_eq!(frontier.checkpoint.sequence, 0);
        assert!(frontier.checkpoint.previous_checkpoint_digest.is_none());
        assert_eq!(frontier.checkpoint.snapshot.committed_uses, 0);
        assert!(frontier.checkpoint.snapshot.reservations.is_empty());
        assert_eq!(adapter.expected_head(), Some(frontier.head));
    }

    #[test]
    fn two_stale_writers_cannot_both_publish_successors() {
        let grant = grant();
        let shared = SharedCasStore::default();
        let (frontier, adapter_a) = establish_grant_frontier(&grant, shared.clone()).unwrap();
        let mut adapter_b = CasCheckpointStoreAdapter::from_trusted_head(shared.clone(), frontier.head);
        let mut adapter_a = adapter_a;

        let account = GrantAccount::new(&grant);
        let successor = GrantAccountCheckpoint::successor(
            &frontier.checkpoint,
            &grant,
            account.snapshot(),
        )
        .unwrap();

        let first = adapter_a.persist(&successor).unwrap();
        assert_eq!(adapter_a.expected_head(), Some(first));
        assert!(matches!(
            adapter_b.persist(&successor),
            Err(FrontierError::Store(CasConflict))
        ));
        assert!(adapter_b.is_contained());
    }

    #[test]
    fn adapter_rejects_wrong_predecessor_before_store_call() {
        let grant = grant();
        let shared = SharedCasStore::default();
        let (frontier, mut adapter) = establish_grant_frontier(&grant, shared).unwrap();
        let account = GrantAccount::new(&grant);
        let mut successor = GrantAccountCheckpoint::successor(
            &frontier.checkpoint,
            &grant,
            account.snapshot(),
        )
        .unwrap();
        successor.previous_checkpoint_digest = Some(symthaea_authority::Digest32([99; 32]));
        assert!(matches!(
            adapter.persist(&successor),
            Err(FrontierError::PreviousDigestMismatch)
        ));
        assert!(adapter.is_contained());
    }
}
