// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error as StdError;

use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_actuation_effect_dispatch::{
    DurablePhysicalEffectAttemptJournal, DurablePreparedPhysicalEffectAttempt,
    PhysicalEffectAttemptCorrelation, PhysicalEffectAttemptJournalHead,
    RollbackProtectedPhysicalEffectAttemptJournal,
};
use thiserror::Error;

use crate::{
    DurableEffectAttemptJournalCheckpointV1, DurableEffectAttemptJournalHeadV1,
    DurableEffectAttemptJournalStore, EffectAttemptJournalError, PreparedDurableEffectAttemptV1,
};

impl DurableEffectAttemptJournalHeadV1 {
    /// Reconstruct an authority-free journal head read from an independent durable anchor.
    pub fn from_anchor_parts(
        generation: u64,
        digest: Digest32,
    ) -> Result<Self, EffectAttemptJournalError> {
        let dispatch = PhysicalEffectAttemptJournalHead::new(generation, digest)
            .map_err(|_| EffectAttemptJournalError::InvalidJournalHead)?;
        Ok(Self::from_dispatch_head(dispatch))
    }
}

/// Independent anti-rollback retention boundary for one device's attempt journal.
///
/// Implementations are expected to use a failure domain independent from the rollbackable local
/// journal storage (for example TPM/NVRAM, a remote quorum, or another deployment-specific
/// monotonic anchor). `compare_and_swap` must durably retain `next` before returning it.
pub trait IndependentEffectAttemptHeadAnchor {
    type Error: StdError + Send + Sync + 'static;

    /// Return the independently retained current head.
    fn current_head(&mut self) -> Result<DurableEffectAttemptJournalHeadV1, Self::Error>;

    /// Atomically advance from `expected` to `next` and return the durably retained head.
    ///
    /// Implementations must fail closed if `expected` is no longer current and must never move the
    /// anchor backwards.
    fn compare_and_swap(
        &mut self,
        expected: DurableEffectAttemptJournalHeadV1,
        next: DurableEffectAttemptJournalHeadV1,
    ) -> Result<DurableEffectAttemptJournalHeadV1, Self::Error>;
}

/// Opaque proof that local `Prepared` state and its independent anti-rollback anchor agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RollbackProtectedPreparedEffectAttemptV1 {
    local: PreparedDurableEffectAttemptV1,
    anchored_head: PhysicalEffectAttemptJournalHead,
}

impl DurablePreparedPhysicalEffectAttempt for RollbackProtectedPreparedEffectAttemptV1 {
    fn journal_head(&self) -> PhysicalEffectAttemptJournalHead {
        self.anchored_head
    }
}

/// Device-scoped attempt journal whose every successful transition is independently anchored.
///
/// The wrapper deliberately becomes permanently poisoned after any uncertain local mutation or
/// anchor operation. Reopening from the independently retained anchor is then required before any
/// further physical attempt can be considered.
pub struct RollbackProtectedEffectAttemptJournal<A>
where
    A: IndependentEffectAttemptHeadAnchor,
{
    local: DurableEffectAttemptJournalStore,
    anchor: A,
    anchored_head: DurableEffectAttemptJournalHeadV1,
    poisoned: bool,
}

impl<A> RollbackProtectedEffectAttemptJournal<A>
where
    A: IndependentEffectAttemptHeadAnchor,
{
    /// Open the rollbackable local journal against the independently retained anchor head.
    ///
    /// If local bytes were advanced without the anchor (or rolled back behind it), local open fails
    /// closed because the exact retained head no longer matches the checkpoint on disk.
    pub fn open(
        root: impl Into<std::path::PathBuf>,
        device: &ResourceRef,
        mut anchor: A,
    ) -> Result<Self, RollbackProtectedEffectAttemptJournalError<A::Error>> {
        let anchored_head = anchor
            .current_head()
            .map_err(RollbackProtectedEffectAttemptJournalError::AnchorRead)?;
        let local = DurableEffectAttemptJournalStore::open(root, device, anchored_head)
            .map_err(RollbackProtectedEffectAttemptJournalError::Local)?;
        Ok(Self {
            local,
            anchor,
            anchored_head,
            poisoned: false,
        })
    }

    pub const fn anchored_head(&self) -> DurableEffectAttemptJournalHeadV1 {
        self.anchored_head
    }

    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Re-read the local checkpoint and require that the independent anchor still names it.
    pub fn current_checkpoint(
        &mut self,
    ) -> Result<
        DurableEffectAttemptJournalCheckpointV1,
        RollbackProtectedEffectAttemptJournalError<A::Error>,
    > {
        self.ensure_usable()?;
        let anchor_head = match self.anchor.current_head() {
            Ok(head) => head,
            Err(source) => return Err(self.poison_anchor_read(source)),
        };
        if anchor_head != self.anchored_head {
            self.poisoned = true;
            return Err(RollbackProtectedEffectAttemptJournalError::AnchorMoved {
                expected: self.anchored_head,
                observed: anchor_head,
            });
        }
        match self.local.current_checkpoint() {
            Ok(checkpoint) => Ok(checkpoint),
            Err(source) => Err(self.poison_local(source)),
        }
    }

    fn ensure_usable(
        &self,
    ) -> Result<(), RollbackProtectedEffectAttemptJournalError<A::Error>> {
        if self.poisoned {
            return Err(RollbackProtectedEffectAttemptJournalError::Poisoned);
        }
        Ok(())
    }

    fn protect_transition(
        &mut self,
        expected: DurableEffectAttemptJournalHeadV1,
        local_head: PhysicalEffectAttemptJournalHead,
    ) -> Result<PhysicalEffectAttemptJournalHead, RollbackProtectedEffectAttemptJournalError<A::Error>>
    {
        let next = DurableEffectAttemptJournalHeadV1::from_dispatch_head(local_head);
        if expected != self.anchored_head {
            self.poisoned = true;
            return Err(RollbackProtectedEffectAttemptJournalError::InternalHeadMismatch {
                expected: self.anchored_head,
                supplied: expected,
            });
        }
        let expected_generation = match expected.generation().checked_add(1) {
            Some(generation) => generation,
            None => {
                self.poisoned = true;
                return Err(RollbackProtectedEffectAttemptJournalError::GenerationOverflow);
            }
        };
        if next.generation() != expected_generation || next.digest() == Digest32([0; 32]) {
            self.poisoned = true;
            return Err(RollbackProtectedEffectAttemptJournalError::NonSequentialLocalHead {
                previous: expected,
                next,
            });
        }

        let confirmed = match self.anchor.compare_and_swap(expected, next) {
            Ok(head) => head,
            Err(source) => {
                self.poisoned = true;
                return Err(RollbackProtectedEffectAttemptJournalError::AnchorAdvance {
                    local_head: next,
                    source,
                });
            }
        };
        if confirmed != next {
            self.poisoned = true;
            return Err(
                RollbackProtectedEffectAttemptJournalError::AnchorConfirmationMismatch {
                    local_head: next,
                    confirmed,
                },
            );
        }

        let observed = match self.anchor.current_head() {
            Ok(head) => head,
            Err(source) => {
                self.poisoned = true;
                return Err(
                    RollbackProtectedEffectAttemptJournalError::AnchorPostAdvanceRead {
                        local_head: next,
                        source,
                    },
                );
            }
        };
        if observed != next {
            self.poisoned = true;
            return Err(
                RollbackProtectedEffectAttemptJournalError::AnchorPostAdvanceMismatch {
                    local_head: next,
                    observed,
                },
            );
        }

        self.anchored_head = observed;
        Ok(local_head)
    }

    fn prepared_matches_anchor(&self, prepared: &RollbackProtectedPreparedEffectAttemptV1) -> bool {
        prepared.anchored_head.generation() == self.anchored_head.generation()
            && prepared.anchored_head.digest() == self.anchored_head.digest()
    }

    fn poison_local(
        &mut self,
        source: EffectAttemptJournalError,
    ) -> RollbackProtectedEffectAttemptJournalError<A::Error> {
        self.poisoned = true;
        RollbackProtectedEffectAttemptJournalError::Local(source)
    }

    fn poison_anchor_read(
        &mut self,
        source: A::Error,
    ) -> RollbackProtectedEffectAttemptJournalError<A::Error> {
        self.poisoned = true;
        RollbackProtectedEffectAttemptJournalError::AnchorRead(source)
    }
}

impl<A> RollbackProtectedPhysicalEffectAttemptJournal for RollbackProtectedEffectAttemptJournal<A>
where
    A: IndependentEffectAttemptHeadAnchor,
{
    type Error = RollbackProtectedEffectAttemptJournalError<A::Error>;
    type Prepared = RollbackProtectedPreparedEffectAttemptV1;

    fn persist_prepared_anchored(
        &mut self,
        correlation: &PhysicalEffectAttemptCorrelation,
    ) -> Result<Self::Prepared, Self::Error> {
        self.ensure_usable()?;
        let expected = self.anchored_head;
        let local = match self.local.persist_prepared(correlation) {
            Ok(prepared) => prepared,
            Err(source) => return Err(self.poison_local(source)),
        };
        let local_head = local.journal_head();
        let anchored_head = self.protect_transition(expected, local_head)?;
        Ok(RollbackProtectedPreparedEffectAttemptV1 {
            local,
            anchored_head,
        })
    }

    fn persist_abandoned_before_port_anchored(
        &mut self,
        prepared: &Self::Prepared,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error> {
        self.ensure_usable()?;
        if !self.prepared_matches_anchor(prepared) {
            self.poisoned = true;
            return Err(RollbackProtectedEffectAttemptJournalError::PreparedAnchorMismatch);
        }
        let expected = self.anchored_head;
        let local_head = match self.local.persist_abandoned_before_port(&prepared.local) {
            Ok(head) => head,
            Err(source) => return Err(self.poison_local(source)),
        };
        self.protect_transition(expected, local_head)
    }

    fn persist_adapter_acknowledged_anchored(
        &mut self,
        prepared: &Self::Prepared,
        adapter_evidence_digest: Digest32,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error> {
        self.ensure_usable()?;
        if !self.prepared_matches_anchor(prepared) {
            self.poisoned = true;
            return Err(RollbackProtectedEffectAttemptJournalError::PreparedAnchorMismatch);
        }
        let expected = self.anchored_head;
        let local_head = match self
            .local
            .persist_adapter_acknowledged(&prepared.local, adapter_evidence_digest)
        {
            Ok(head) => head,
            Err(source) => return Err(self.poison_local(source)),
        };
        self.protect_transition(expected, local_head)
    }

    fn persist_adapter_indeterminate_anchored(
        &mut self,
        prepared: &Self::Prepared,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error> {
        self.ensure_usable()?;
        if !self.prepared_matches_anchor(prepared) {
            self.poisoned = true;
            return Err(RollbackProtectedEffectAttemptJournalError::PreparedAnchorMismatch);
        }
        let expected = self.anchored_head;
        let local_head = match self.local.persist_adapter_indeterminate(&prepared.local) {
            Ok(head) => head,
            Err(source) => return Err(self.poison_local(source)),
        };
        self.protect_transition(expected, local_head)
    }
}

#[derive(Debug, Error)]
pub enum RollbackProtectedEffectAttemptJournalError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("rollback-protected attempt journal is poisoned and must be reopened")]
    Poisoned,
    #[error("local crash-durable attempt journal failed: {0}")]
    Local(#[source] EffectAttemptJournalError),
    #[error("independent anti-rollback anchor could not be read: {0}")]
    AnchorRead(#[source] E),
    #[error("independent anti-rollback anchor changed outside this wrapper: expected={expected:?}, observed={observed:?}")]
    AnchorMoved {
        expected: DurableEffectAttemptJournalHeadV1,
        observed: DurableEffectAttemptJournalHeadV1,
    },
    #[error("wrapper head bookkeeping diverged: current={expected:?}, supplied={supplied:?}")]
    InternalHeadMismatch {
        expected: DurableEffectAttemptJournalHeadV1,
        supplied: DurableEffectAttemptJournalHeadV1,
    },
    #[error("rollback-protected journal generation overflow")]
    GenerationOverflow,
    #[error("local attempt journal produced a non-sequential successor head: previous={previous:?}, next={next:?}")]
    NonSequentialLocalHead {
        previous: DurableEffectAttemptJournalHeadV1,
        next: DurableEffectAttemptJournalHeadV1,
    },
    #[error("independent anti-rollback anchor advance became uncertain after local head {local_head:?}: {source}")]
    AnchorAdvance {
        local_head: DurableEffectAttemptJournalHeadV1,
        #[source]
        source: E,
    },
    #[error("independent anti-rollback anchor confirmed a different head after local transition: local={local_head:?}, confirmed={confirmed:?}")]
    AnchorConfirmationMismatch {
        local_head: DurableEffectAttemptJournalHeadV1,
        confirmed: DurableEffectAttemptJournalHeadV1,
    },
    #[error("independent anti-rollback anchor could not be re-read after confirming local head {local_head:?}: {source}")]
    AnchorPostAdvanceRead {
        local_head: DurableEffectAttemptJournalHeadV1,
        #[source]
        source: E,
    },
    #[error("independent anti-rollback anchor re-read disagreed with the confirmed local head: local={local_head:?}, observed={observed:?}")]
    AnchorPostAdvanceMismatch {
        local_head: DurableEffectAttemptJournalHeadV1,
        observed: DurableEffectAttemptJournalHeadV1,
    },
    #[error("protected Prepared proof no longer matches the wrapper's independently anchored head")]
    PreparedAnchorMismatch,
}
