// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Concrete post-restore promotion barrier backed by the episodic continuity anchor protocol.
//!
//! The welfare layer intentionally knows nothing about TPMs, TEEs, Xenia or Mycelix. It asks only
//! for independently verified promotion evidence after local `Restored` durability and before live
//! heap activation. This crate satisfies that narrow contract by reopening the durable SQLite
//! continuity database, advancing #2098's complete manifest through compare-and-swap, and then
//! performing an anchored recovery against the newly accepted snapshot.

#![deny(unsafe_code)]

use std::error::Error as StdError;
use std::path::{Path, PathBuf};

use symthaea_episodic_continuity::{ContinuityStoreError, SqliteEpisodicContinuityStore};
use symthaea_episodic_continuity_anchor::{
    AnchorProtocolError, ContinuityAnchorError, ContinuityAnchorSnapshot, ContinuityHeadAnchor,
    advance_anchor_after_durable_store, recover_with_anchor,
};
use symthaea_welfare_assurance::restored_continuity_promotion::{
    RestoredContinuityPromotionBarrier, RestoredContinuityPromotionFailure,
    RestoredContinuityPromotionRequest, VerifiedRestoredContinuityPromotion,
};
use thiserror::Error;

/// #2098-backed implementation of the welfare-side restored-continuity promotion barrier.
///
/// `database_path` is intentionally reopened for every promotion. The proof therefore observes the
/// durable state a fresh process would recover rather than trusting a writer connection's transient
/// state. `previous` is advanced only after the external CAS succeeds.
pub struct SqliteRestoredContinuityAnchorBarrier<'a, A>
where
    A: ContinuityHeadAnchor,
{
    database_path: PathBuf,
    anchor: &'a mut A,
    previous: ContinuityAnchorSnapshot,
    intent_head: symthaea_fabrication_kernel::crypto_digest::Sha256Digest,
    committed_at_unix_s: u64,
}

impl<'a, A> SqliteRestoredContinuityAnchorBarrier<'a, A>
where
    A: ContinuityHeadAnchor,
{
    pub fn new(
        database_path: impl Into<PathBuf>,
        anchor: &'a mut A,
        previous: ContinuityAnchorSnapshot,
        intent_head: symthaea_fabrication_kernel::crypto_digest::Sha256Digest,
        committed_at_unix_s: u64,
    ) -> Result<Self, RestoredAnchorBarrierConfigurationError> {
        previous
            .validate()
            .map_err(RestoredAnchorBarrierConfigurationError::PreviousAnchor)?;
        if intent_head.0 == [0; 32] {
            return Err(RestoredAnchorBarrierConfigurationError::ZeroIntentHead);
        }
        if previous.intent_head != intent_head {
            return Err(RestoredAnchorBarrierConfigurationError::IntentHeadNotAnchored);
        }
        if committed_at_unix_s < previous.committed_at_unix_s {
            return Err(RestoredAnchorBarrierConfigurationError::CommitTimeRegression {
                previous: previous.committed_at_unix_s,
                next: committed_at_unix_s,
            });
        }
        Ok(Self {
            database_path: database_path.into(),
            anchor,
            previous,
            intent_head,
            committed_at_unix_s,
        })
    }

    pub fn previous_snapshot(&self) -> &ContinuityAnchorSnapshot {
        &self.previous
    }

    pub fn database_path(&self) -> &Path {
        &self.database_path
    }

    pub fn commit_typed(
        &mut self,
        request: &RestoredContinuityPromotionRequest,
    ) -> Result<VerifiedRestoredContinuityPromotion, RestoredAnchorBarrierError<A::Error>> {
        if request.store_target_id() != self.previous.store_target_id {
            return Err(RestoredAnchorBarrierError::StoreTargetMismatch);
        }
        if request.restored_quarantine_head() == self.previous.quarantine_head {
            return Err(RestoredAnchorBarrierError::QuarantineHeadDidNotAdvance);
        }

        // Reopen the database after local Restored persistence. This intentionally proves durable
        // bytes, not merely the state visible through the writer that performed the transition.
        let store = SqliteEpisodicContinuityStore::open(&self.database_path)
            .map_err(RestoredAnchorBarrierError::StoreOpen)?;
        let previous_commitment = self
            .previous
            .commitment()
            .map_err(RestoredAnchorBarrierError::AnchorShape)?;

        let (next, anchor_reference) = advance_anchor_after_durable_store(
            &store,
            self.anchor,
            &self.previous,
            self.intent_head,
            request.restored_quarantine_head(),
            self.committed_at_unix_s,
        )
        .map_err(RestoredAnchorBarrierError::Protocol)?;

        // CAS has succeeded. From this point onward, keep the in-process predecessor aligned with
        // the independently accepted state even if later evidence shaping/revalidation fails. A
        // failure after CAS leaves live memory inactive and execution in-doubt, which is safe.
        self.previous = next.clone();

        if next.store_target_id != request.store_target_id() {
            return Err(RestoredAnchorBarrierError::ReturnedStoreTargetMismatch);
        }
        if next.intent_head != self.intent_head {
            return Err(RestoredAnchorBarrierError::ReturnedIntentHeadMismatch);
        }
        if next.quarantine_head != request.restored_quarantine_head() {
            return Err(RestoredAnchorBarrierError::ReturnedQuarantineHeadMismatch);
        }

        let next_commitment = next
            .commitment()
            .map_err(RestoredAnchorBarrierError::AnchorShape)?;

        // Re-read the exact newly anchored state through the normal anchored recovery theorem and
        // prove that the restored occurrence is now an active exact UUID/content pair.
        let anchored = recover_with_anchor(&store, self.anchor, request.store_target_id())
            .map_err(RestoredAnchorBarrierError::PostCasRecovery)?;
        if anchored.anchor != next || anchored.anchor_commitment != next_commitment {
            return Err(RestoredAnchorBarrierError::PostCasSnapshotMismatch);
        }
        let active = anchored
            .recovered
            .activation_plan
            .active
            .iter()
            .find(|entry| entry.instance_id == request.instance_id())
            .ok_or(RestoredAnchorBarrierError::RestoredOccurrenceNotActive)?;
        if active.content_id != request.content_id() {
            return Err(RestoredAnchorBarrierError::RestoredContentMismatch);
        }
        if anchored
            .recovered
            .activation_plan
            .inactive
            .iter()
            .any(|entry| entry.instance_id == request.instance_id())
        {
            return Err(RestoredAnchorBarrierError::RestoredOccurrenceStillInactive);
        }

        let evidence = VerifiedRestoredContinuityPromotion::try_new(
            next.store_target_id.clone(),
            next.quarantine_head,
            previous_commitment,
            next_commitment,
            next.revision,
            next.continuity_manifest_digest,
            anchor_reference,
        )
        .map_err(|error| RestoredAnchorBarrierError::EvidenceShape(error.to_string()))?;
        evidence
            .validate_for(request)
            .map_err(|error| RestoredAnchorBarrierError::EvidenceShape(error.to_string()))?;
        Ok(evidence)
    }
}

impl<A> RestoredContinuityPromotionBarrier for SqliteRestoredContinuityAnchorBarrier<'_, A>
where
    A: ContinuityHeadAnchor,
{
    fn commit_restored_continuity(
        &mut self,
        request: &RestoredContinuityPromotionRequest,
    ) -> Result<VerifiedRestoredContinuityPromotion, RestoredContinuityPromotionFailure> {
        self.commit_typed(request)
            .map_err(|error| RestoredContinuityPromotionFailure::new(error.to_string()))
    }
}

#[derive(Debug, Error)]
pub enum RestoredAnchorBarrierConfigurationError {
    #[error("previous continuity anchor is invalid: {0}")]
    PreviousAnchor(#[source] ContinuityAnchorError),
    #[error("restored-continuity barrier intent head must not be zero")]
    ZeroIntentHead,
    #[error("restored-continuity barrier intent head is not the one bound by the previous anchor")]
    IntentHeadNotAnchored,
    #[error("restored-continuity barrier commit time regressed: previous={previous}, next={next}")]
    CommitTimeRegression { previous: u64, next: u64 },
}

#[derive(Debug, Error)]
pub enum RestoredAnchorBarrierError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("restored-continuity request store target does not match previous anchor")]
    StoreTargetMismatch,
    #[error("restored quarantine head did not advance from the previous anchor")]
    QuarantineHeadDidNotAdvance,
    #[error("could not reopen durable continuity database: {0}")]
    StoreOpen(#[source] ContinuityStoreError),
    #[error("continuity anchor snapshot shape failed: {0}")]
    AnchorShape(#[source] ContinuityAnchorError),
    #[error("continuity anchor advancement failed: {0}")]
    Protocol(#[source] AnchorProtocolError<E>),
    #[error("returned continuity anchor store target mismatch")]
    ReturnedStoreTargetMismatch,
    #[error("returned continuity anchor intent head mismatch")]
    ReturnedIntentHeadMismatch,
    #[error("returned continuity anchor quarantine head mismatch")]
    ReturnedQuarantineHeadMismatch,
    #[error("anchored post-CAS recovery failed: {0}")]
    PostCasRecovery(#[source] AnchorProtocolError<E>),
    #[error("anchored post-CAS recovery returned a different snapshot/commitment")]
    PostCasSnapshotMismatch,
    #[error("newly anchored recovery does not make the restored occurrence active")]
    RestoredOccurrenceNotActive,
    #[error("newly anchored recovery content identity does not match restored occurrence")]
    RestoredContentMismatch,
    #[error("newly anchored recovery still classifies the restored occurrence inactive")]
    RestoredOccurrenceStillInactive,
    #[error("restored-continuity promotion evidence shape failed: {0}")]
    EvidenceShape(String),
}
