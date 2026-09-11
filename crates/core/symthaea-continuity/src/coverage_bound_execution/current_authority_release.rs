// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public physical release facade for fresh owner/operator authority.
//!
//! The underlying transition-authority currentness implementation stays crate-private
//! so external adapters cannot route around the durable current-authority commitment.
//! This module deliberately exposes proof/configuration types while retaining the
//! pending/ready execution state machine behind the strongest release theorem.

use thiserror::Error;

pub use super::current_transition_authority::{
    TRANSITION_AUTHORITY_CURRENTNESS_AUTH_PURPOSE,
    TRANSITION_AUTHORITY_CURRENTNESS_CLAIM_SCHEMA_V1,
    AuthenticatedTransitionAuthorityCurrentnessId, CurrentAuthorizedTransitionAuthorityId,
    CurrentAuthorizedTransitionAuthorityV1, CurrentAuthorityEffectExecutionEligibilityV1,
    CurrentAuthorityExecutionError, CurrentAuthorityNoEffectsExecutionEligibilityV1,
    QualifiedTransitionAuthorityCurrentnessId, QualifiedTransitionAuthorityCurrentnessV1,
    TransitionAuthorityCurrentnessClaimId, TransitionAuthorityCurrentnessClaimV1,
    TransitionAuthorityCurrentnessError, TransitionAuthorityDispositionV1,
    TrustedTransitionAuthorityCurrentnessRootId, TrustedTransitionAuthorityCurrentnessRootV1,
    canonical_transition_authority_currentness_claim_bytes,
    canonical_transition_authority_currentness_claim_digest,
};

use super::current_transition_authority::{
    PendingCurrentAuthorityEffectExecutionV1, PendingCurrentAuthorityNoEffectsExecutionV1,
    ReadyCurrentAuthorityEffectExecutionV1, ReadyCurrentAuthorityNoEffectsExecutionV1,
    prepare_current_authority_effect_execution, prepare_current_authority_no_effects_execution,
};
use super::current_transition_authority_commitment::{
    CurrentTransitionAuthorityExecutionBindingV1,
    QualifiedCurrentTransitionAuthorityExecutionCommitmentId,
    QualifiedCurrentTransitionAuthorityExecutionCommitmentV1,
};
use super::current_verifier_commitment::{
    QualifiedCurrentVerifierExecutionCommitmentId,
    QualifiedCurrentVerifierExecutionCommitmentV1,
};
use super::{QualifiedEffectCoverageCommitmentV1};
use crate::backend_bound_execution::QualifiedBackendEffectCommitmentV1;
use crate::effect_scoped_execution::QualifiedEffectScopeCommitmentV1;
use crate::execution_capability::{
    ExecutionAttemptId, ExecutionAttemptOutcomeV1, ExecutionAttemptReceiptV1,
    ExecutionBackendId, ExecutionEpochAnchorModeV1,
};
use crate::execution_journal::ReconstructedExecutionJournalV1;
use crate::execution_journal_anchor::{
    QualifiedExecutionJournalAnchorId, QualifiedExecutionJournalAnchorV1,
};
use crate::external_effects::ExternalEffectContractV1;
use crate::no_effects_execution::QualifiedNoEffectsExecutionCommitmentV1;
use crate::scope::ContinuitySubjectId;
use crate::trusted_commit_epoch::QualifiedTrustedCommitEpochV1;
use crate::witness::TargetRealizationId;

#[derive(Debug)]
pub struct PendingDurableCurrentAuthorityEffectExecutionV1 {
    inner: PendingCurrentAuthorityEffectExecutionV1,
}

impl PendingDurableCurrentAuthorityEffectExecutionV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn release_after_durable_current_authority(
        self,
        journal: &ReconstructedExecutionJournalV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        effect_scope: &QualifiedEffectScopeCommitmentV1,
        backend_commitment: &QualifiedBackendEffectCommitmentV1,
        coverage_commitment: &QualifiedEffectCoverageCommitmentV1,
        current_verifier_commitment: &QualifiedCurrentVerifierExecutionCommitmentV1,
        current_authority_commitment: &QualifiedCurrentTransitionAuthorityExecutionCommitmentV1,
    ) -> Result<ReadyDurableCurrentAuthorityEffectExecutionV1, DurableCurrentAuthorityReleaseError> {
        let expected_binding = CurrentTransitionAuthorityExecutionBindingV1::Effectful {
            historical_commitment_id: coverage_commitment.id(),
            attempt_id: coverage_commitment.attempt_id(),
        };
        require_commitment_world(
            current_authority_commitment,
            journal_anchor.id(),
            expected_binding,
            current_verifier_commitment.id(),
        )?;
        let ready = self.inner.release_after_durable_current_authority(
            journal,
            journal_anchor,
            effect_scope,
            backend_commitment,
            coverage_commitment,
            current_verifier_commitment,
        )?;
        require_ready_matches_commitment_effectful(&ready, current_authority_commitment)?;
        Ok(ReadyDurableCurrentAuthorityEffectExecutionV1 {
            inner: ready,
            current_authority_commitment_id: current_authority_commitment.id(),
        })
    }
}

#[derive(Debug)]
pub struct ReadyDurableCurrentAuthorityEffectExecutionV1 {
    inner: ReadyCurrentAuthorityEffectExecutionV1,
    current_authority_commitment_id: QualifiedCurrentTransitionAuthorityExecutionCommitmentId,
}

impl ReadyDurableCurrentAuthorityEffectExecutionV1 {
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.inner.attempt_id() }
    pub fn backend_id(&self) -> ExecutionBackendId { self.inner.backend_id() }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.inner.subject_id() }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.inner.source_realization_id() }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.inner.target_realization_id() }
    pub fn effect_contract(&self) -> &ExternalEffectContractV1 { self.inner.effect_contract() }
    pub fn current_authority_id(&self) -> CurrentAuthorizedTransitionAuthorityId {
        self.inner.current_authority_id()
    }
    pub fn authority_currentness_id(&self) -> QualifiedTransitionAuthorityCurrentnessId {
        self.inner.authority_currentness_id()
    }
    pub fn current_authority_commitment_id(
        &self,
    ) -> QualifiedCurrentTransitionAuthorityExecutionCommitmentId {
        self.current_authority_commitment_id
    }

    pub fn finish(
        self,
        outcome: ExecutionAttemptOutcomeV1,
        backend_evidence_digest: [u8; 32],
        result_digest: [u8; 32],
    ) -> Result<ExecutionAttemptReceiptV1, DurableCurrentAuthorityReleaseError> {
        Ok(self.inner.finish(outcome, backend_evidence_digest, result_digest)?)
    }
}

#[derive(Debug)]
pub struct PendingDurableCurrentAuthorityNoEffectsExecutionV1 {
    inner: PendingCurrentAuthorityNoEffectsExecutionV1,
}

impl PendingDurableCurrentAuthorityNoEffectsExecutionV1 {
    pub fn release_after_durable_current_authority(
        self,
        journal: &ReconstructedExecutionJournalV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        commitment: &QualifiedNoEffectsExecutionCommitmentV1,
        current_verifier_commitment: &QualifiedCurrentVerifierExecutionCommitmentV1,
        current_authority_commitment: &QualifiedCurrentTransitionAuthorityExecutionCommitmentV1,
    ) -> Result<ReadyDurableCurrentAuthorityNoEffectsExecutionV1, DurableCurrentAuthorityReleaseError> {
        let expected_binding = CurrentTransitionAuthorityExecutionBindingV1::NoEffects {
            historical_commitment_id: commitment.id(),
            attempt_id: commitment.attempt_id(),
        };
        require_commitment_world(
            current_authority_commitment,
            journal_anchor.id(),
            expected_binding,
            current_verifier_commitment.id(),
        )?;
        let ready = self.inner.release_after_durable_current_authority(
            journal,
            journal_anchor,
            commitment,
            current_verifier_commitment,
        )?;
        require_ready_matches_commitment_no_effects(&ready, current_authority_commitment)?;
        Ok(ReadyDurableCurrentAuthorityNoEffectsExecutionV1 {
            inner: ready,
            current_authority_commitment_id: current_authority_commitment.id(),
        })
    }
}

#[derive(Debug)]
pub struct ReadyDurableCurrentAuthorityNoEffectsExecutionV1 {
    inner: ReadyCurrentAuthorityNoEffectsExecutionV1,
    current_authority_commitment_id: QualifiedCurrentTransitionAuthorityExecutionCommitmentId,
}

impl ReadyDurableCurrentAuthorityNoEffectsExecutionV1 {
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.inner.attempt_id() }
    pub fn backend_id(&self) -> ExecutionBackendId { self.inner.backend_id() }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.inner.subject_id() }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.inner.source_realization_id() }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.inner.target_realization_id() }
    pub fn current_authority_id(&self) -> CurrentAuthorizedTransitionAuthorityId {
        self.inner.current_authority_id()
    }
    pub fn authority_currentness_id(&self) -> QualifiedTransitionAuthorityCurrentnessId {
        self.inner.authority_currentness_id()
    }
    pub fn current_authority_commitment_id(
        &self,
    ) -> QualifiedCurrentTransitionAuthorityExecutionCommitmentId {
        self.current_authority_commitment_id
    }

    pub fn finish(
        self,
        outcome: ExecutionAttemptOutcomeV1,
        backend_evidence_digest: [u8; 32],
        result_digest: [u8; 32],
    ) -> Result<ExecutionAttemptReceiptV1, DurableCurrentAuthorityReleaseError> {
        Ok(self.inner.finish(outcome, backend_evidence_digest, result_digest)?)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_durable_current_authority_effect_execution(
    bound: CurrentAuthorityEffectExecutionEligibilityV1,
    current_epoch: &QualifiedTrustedCommitEpochV1,
    previous_epoch: Option<&QualifiedTrustedCommitEpochV1>,
    epoch_anchor_mode: ExecutionEpochAnchorModeV1,
    predecessor_journal_anchor: &QualifiedExecutionJournalAnchorV1,
    session_generation: u64,
    session_nonce: [u8; 32],
) -> Result<PendingDurableCurrentAuthorityEffectExecutionV1, DurableCurrentAuthorityReleaseError> {
    let inner = prepare_current_authority_effect_execution(
        bound,
        current_epoch,
        previous_epoch,
        epoch_anchor_mode,
        predecessor_journal_anchor,
        session_generation,
        session_nonce,
    )?;
    Ok(PendingDurableCurrentAuthorityEffectExecutionV1 { inner })
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_durable_current_authority_no_effects_execution(
    bound: CurrentAuthorityNoEffectsExecutionEligibilityV1,
    current_epoch: &QualifiedTrustedCommitEpochV1,
    previous_epoch: Option<&QualifiedTrustedCommitEpochV1>,
    epoch_anchor_mode: ExecutionEpochAnchorModeV1,
    predecessor_journal_anchor: &QualifiedExecutionJournalAnchorV1,
    session_generation: u64,
    session_nonce: [u8; 32],
) -> Result<PendingDurableCurrentAuthorityNoEffectsExecutionV1, DurableCurrentAuthorityReleaseError> {
    let inner = prepare_current_authority_no_effects_execution(
        bound,
        current_epoch,
        previous_epoch,
        epoch_anchor_mode,
        predecessor_journal_anchor,
        session_generation,
        session_nonce,
    )?;
    Ok(PendingDurableCurrentAuthorityNoEffectsExecutionV1 { inner })
}

fn require_commitment_world(
    commitment: &QualifiedCurrentTransitionAuthorityExecutionCommitmentV1,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    expected_binding: CurrentTransitionAuthorityExecutionBindingV1,
    current_verifier_commitment_id: QualifiedCurrentVerifierExecutionCommitmentId,
) -> Result<(), DurableCurrentAuthorityReleaseError> {
    if commitment.journal_anchor_id() != journal_anchor_id
        || commitment.binding() != expected_binding
        || commitment.current_verifier_commitment_id() != current_verifier_commitment_id
    {
        return Err(DurableCurrentAuthorityReleaseError::CurrentAuthorityCommitmentMismatch);
    }
    Ok(())
}

fn require_ready_matches_commitment_effectful(
    ready: &ReadyCurrentAuthorityEffectExecutionV1,
    commitment: &QualifiedCurrentTransitionAuthorityExecutionCommitmentV1,
) -> Result<(), DurableCurrentAuthorityReleaseError> {
    if commitment.binding().attempt_id() != ready.attempt_id()
        || commitment.current_authority_id() != ready.current_authority_id()
        || commitment.authority_currentness_id() != ready.authority_currentness_id()
        || commitment.subject_id() != ready.subject_id()
        || commitment.source_realization_id() != ready.source_realization_id()
        || commitment.target_realization_id() != ready.target_realization_id()
    {
        return Err(DurableCurrentAuthorityReleaseError::ReadyCommitmentMismatch);
    }
    Ok(())
}

fn require_ready_matches_commitment_no_effects(
    ready: &ReadyCurrentAuthorityNoEffectsExecutionV1,
    commitment: &QualifiedCurrentTransitionAuthorityExecutionCommitmentV1,
) -> Result<(), DurableCurrentAuthorityReleaseError> {
    if commitment.binding().attempt_id() != ready.attempt_id()
        || commitment.current_authority_id() != ready.current_authority_id()
        || commitment.authority_currentness_id() != ready.authority_currentness_id()
        || commitment.subject_id() != ready.subject_id()
        || commitment.source_realization_id() != ready.source_realization_id()
        || commitment.target_realization_id() != ready.target_realization_id()
    {
        return Err(DurableCurrentAuthorityReleaseError::ReadyCommitmentMismatch);
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum DurableCurrentAuthorityReleaseError {
    #[error(transparent)]
    Inner(#[from] CurrentAuthorityExecutionError),
    #[error("durable current-authority commitment differs from exact release world")]
    CurrentAuthorityCommitmentMismatch,
    #[error("ready physical transition differs from durable current-authority commitment")]
    ReadyCommitmentMismatch,
}
