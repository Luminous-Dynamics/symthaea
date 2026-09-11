// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh-current verifier authority at the final physical coverage boundary.
//!
//! Historical coverage qualification is immutable evidence. It is not, by itself,
//! proof that the verifier which produced it is still the currently authorized
//! verifier when physical authority is released.
//!
//! V1 intentionally uses one exact decision boundary rather than a freshness TTL:
//!
//! `coverage analysis time == verifier currentness time == transition commit time`.
//!
//! The wrappers in this module are non-Serde and retain the fresh current-verifier
//! proof lineage until the physical ready token is produced. The historical effect
//! and no-effects coverage IDs remain unchanged.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    CoverageBoundExecutionError, CoverageQualifiedBackendEffectEligibilityV1,
    PendingCoverageQualifiedEffectExecutionV1, QualifiedEffectCoverageCommitmentV1,
    ReadyCoverageQualifiedEffectExecutionV1, prepare_coverage_qualified_effect_execution,
};
use super::current_verifier_commitment::{
    CurrentVerifierExecutionBindingV1, QualifiedCurrentVerifierExecutionCommitmentId,
    QualifiedCurrentVerifierExecutionCommitmentV1,
};
use crate::backend_bound_execution::QualifiedBackendEffectCommitmentV1;
use crate::effect_coverage::{QualifiedExternalEffectCoverageId, QualifiedExternalEffectCoverageV1};
use crate::effect_scoped_execution::QualifiedEffectScopeCommitmentV1;
use crate::execution_capability::{
    ExecutionAttemptId, ExecutionAttemptOutcomeV1, ExecutionAttemptReceiptV1,
    ExecutionBackendId, ExecutionEpochAnchorModeV1,
};
use crate::execution_journal::ReconstructedExecutionJournalV1;
use crate::execution_journal_anchor::QualifiedExecutionJournalAnchorV1;
use crate::external_effects::ExternalEffectContractV1;
use crate::no_effects_execution::{
    NoEffectsExecutionError, PendingProvenNoEffectsExecutionV1,
    QualifiedNoEffectsExecutionCommitmentV1, ReadyProvenNoEffectsExecutionV1,
    prepare_proven_no_effects_execution,
};
use crate::no_external_effects::{
    NoEffectsKnownGoodBoundEligibilityV1, QualifiedNoExternalEffectsCoverageId,
    QualifiedNoExternalEffectsCoverageV1,
};
use crate::scope::ContinuitySubjectId;
use crate::trusted_commit_epoch::QualifiedTrustedCommitEpochV1;
use crate::verifier::{VerifierProfileId, VerifierProfileV1};
use crate::verifier_adoption_authority::{
    CurrentAuthorizedVerifierProfileId, CurrentAuthorizedVerifierProfileV1,
    QualifiedVerifierProfileAdoptionId, QualifiedVerifierProfileAdoptionV1,
};
use crate::witness::TargetRealizationId;

const EFFECTFUL_CURRENT_COVERAGE_DOMAIN: &[u8] =
    b"symthaea.continuity.current-authorized-effect-coverage.v1\0";
const NO_EFFECTS_CURRENT_COVERAGE_DOMAIN: &[u8] =
    b"symthaea.continuity.current-authorized-no-effects-coverage.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name {
            pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
        }
    };
}

digest_id!(CurrentAuthorizedExternalEffectCoverageId);
digest_id!(CurrentAuthorizedNoExternalEffectsCoverageId);

/// Non-Serde proof that one exact complete effect-coverage result was produced by
/// the verifier adoption which is freshly attested as the current authorized head at
/// this exact decision boundary.
#[derive(Debug)]
pub struct CurrentAuthorizedExternalEffectCoverageV1 {
    current_coverage_id: CurrentAuthorizedExternalEffectCoverageId,
    coverage: QualifiedExternalEffectCoverageV1,
    adoption_id: QualifiedVerifierProfileAdoptionId,
    current_verifier_id: CurrentAuthorizedVerifierProfileId,
    decision_time_unix_ms: u64,
}

impl CurrentAuthorizedExternalEffectCoverageV1 {
    pub fn bind_at_decision(
        coverage: QualifiedExternalEffectCoverageV1,
        adoption: &QualifiedVerifierProfileAdoptionV1,
        current: &CurrentAuthorizedVerifierProfileV1,
        decision_time_unix_ms: u64,
    ) -> Result<Self, CurrentAuthorizedCoverageError> {
        require_current_verifier_at_decision(
            coverage.verifier_profile_id(),
            coverage.verifier_root_epoch(),
            coverage.analyzed_at_unix_ms(),
            adoption,
            current,
            decision_time_unix_ms,
        )?;
        let decision_time = decision_time_unix_ms.to_le_bytes();
        let current_coverage_id = CurrentAuthorizedExternalEffectCoverageId(domain_hash_parts(
            EFFECTFUL_CURRENT_COVERAGE_DOMAIN,
            &[
                coverage.id().as_bytes(), adoption.id().as_bytes(), current.id().as_bytes(),
                coverage.profile_id().as_bytes(), coverage.verifier_profile_id().as_bytes(),
                &coverage.verifier_root_epoch().to_le_bytes(), &decision_time,
            ],
        ));
        Ok(Self {
            current_coverage_id,
            coverage,
            adoption_id: adoption.id(),
            current_verifier_id: current.id(),
            decision_time_unix_ms,
        })
    }

    pub fn id(&self) -> CurrentAuthorizedExternalEffectCoverageId { self.current_coverage_id }
    pub fn coverage(&self) -> &QualifiedExternalEffectCoverageV1 { &self.coverage }
    pub fn adoption_id(&self) -> QualifiedVerifierProfileAdoptionId { self.adoption_id }
    pub fn current_verifier_id(&self) -> CurrentAuthorizedVerifierProfileId { self.current_verifier_id }
    pub fn decision_time_unix_ms(&self) -> u64 { self.decision_time_unix_ms }
}

/// Non-Serde sibling proof for the independently qualified canonical-empty effect set.
#[derive(Debug)]
pub struct CurrentAuthorizedNoExternalEffectsCoverageV1 {
    current_coverage_id: CurrentAuthorizedNoExternalEffectsCoverageId,
    coverage: QualifiedNoExternalEffectsCoverageV1,
    adoption_id: QualifiedVerifierProfileAdoptionId,
    current_verifier_id: CurrentAuthorizedVerifierProfileId,
    decision_time_unix_ms: u64,
}

impl CurrentAuthorizedNoExternalEffectsCoverageV1 {
    pub fn bind_at_decision(
        coverage: QualifiedNoExternalEffectsCoverageV1,
        adoption: &QualifiedVerifierProfileAdoptionV1,
        current: &CurrentAuthorizedVerifierProfileV1,
        decision_time_unix_ms: u64,
    ) -> Result<Self, CurrentAuthorizedCoverageError> {
        require_current_verifier_at_decision(
            coverage.verifier_profile_id(),
            coverage.verifier_root_epoch(),
            coverage.analyzed_at_unix_ms(),
            adoption,
            current,
            decision_time_unix_ms,
        )?;
        let decision_time = decision_time_unix_ms.to_le_bytes();
        let current_coverage_id = CurrentAuthorizedNoExternalEffectsCoverageId(domain_hash_parts(
            NO_EFFECTS_CURRENT_COVERAGE_DOMAIN,
            &[
                coverage.id().as_bytes(), adoption.id().as_bytes(), current.id().as_bytes(),
                coverage.coverage_profile_id().as_bytes(), coverage.verifier_profile_id().as_bytes(),
                &coverage.verifier_root_epoch().to_le_bytes(), &decision_time,
            ],
        ));
        Ok(Self {
            current_coverage_id,
            coverage,
            adoption_id: adoption.id(),
            current_verifier_id: current.id(),
            decision_time_unix_ms,
        })
    }

    pub fn id(&self) -> CurrentAuthorizedNoExternalEffectsCoverageId { self.current_coverage_id }
    pub fn coverage(&self) -> &QualifiedNoExternalEffectsCoverageV1 { &self.coverage }
    pub fn adoption_id(&self) -> QualifiedVerifierProfileAdoptionId { self.adoption_id }
    pub fn current_verifier_id(&self) -> CurrentAuthorizedVerifierProfileId { self.current_verifier_id }
    pub fn decision_time_unix_ms(&self) -> u64 { self.decision_time_unix_ms }
}

/// Final non-Clone effectful eligibility. Historical complete coverage is retained,
/// but physical preparation additionally requires the fresh current-authority wrapper.
#[derive(Debug)]
pub struct CurrentAuthorizedEffectExecutionEligibilityV1 {
    inner: CoverageQualifiedBackendEffectEligibilityV1,
    current_coverage: CurrentAuthorizedExternalEffectCoverageV1,
}

impl CurrentAuthorizedEffectExecutionEligibilityV1 {
    pub fn bind(
        inner: CoverageQualifiedBackendEffectEligibilityV1,
        current_coverage: CurrentAuthorizedExternalEffectCoverageV1,
    ) -> Result<Self, CurrentAuthorizedCoverageError> {
        let transition_time = inner.bound().scoped().plan().commit_time_unix_ms();
        if inner.coverage().id() != current_coverage.coverage().id() {
            return Err(CurrentAuthorizedCoverageError::EffectCoverageMismatch);
        }
        if current_coverage.decision_time_unix_ms() != transition_time {
            return Err(CurrentAuthorizedCoverageError::TransitionDecisionTimeMismatch);
        }
        Ok(Self { inner, current_coverage })
    }

    pub fn current_coverage(&self) -> &CurrentAuthorizedExternalEffectCoverageV1 {
        &self.current_coverage
    }
}

/// Final non-Clone no-effects eligibility. The exact canonical-empty proof must be
/// backed by the same freshly current verifier at the exact A -> B decision boundary.
#[derive(Debug)]
pub struct CurrentAuthorizedNoEffectsExecutionEligibilityV1 {
    inner: NoEffectsKnownGoodBoundEligibilityV1,
    current_coverage: CurrentAuthorizedNoExternalEffectsCoverageV1,
}

impl CurrentAuthorizedNoEffectsExecutionEligibilityV1 {
    pub fn bind(
        inner: NoEffectsKnownGoodBoundEligibilityV1,
        current_coverage: CurrentAuthorizedNoExternalEffectsCoverageV1,
    ) -> Result<Self, CurrentAuthorizedCoverageError> {
        if inner.coverage().id() != current_coverage.coverage().id() {
            return Err(CurrentAuthorizedCoverageError::NoEffectsCoverageMismatch);
        }
        if inner.declaration().commit_time_unix_ms() != current_coverage.decision_time_unix_ms() {
            return Err(CurrentAuthorizedCoverageError::TransitionDecisionTimeMismatch);
        }
        Ok(Self { inner, current_coverage })
    }

    pub fn current_coverage(&self) -> &CurrentAuthorizedNoExternalEffectsCoverageV1 {
        &self.current_coverage
    }
}

/// Pending effectful state which deliberately retains current-verifier provenance
/// until the existing protected effect/backend/coverage world releases a ready token.
#[derive(Debug)]
pub struct PendingCurrentAuthorizedEffectExecutionV1 {
    inner: PendingCoverageQualifiedEffectExecutionV1,
    current_coverage: CurrentAuthorizedExternalEffectCoverageV1,
}

impl PendingCurrentAuthorizedEffectExecutionV1 {
    pub fn inner(&self) -> &PendingCoverageQualifiedEffectExecutionV1 { &self.inner }
    pub fn current_coverage(&self) -> &CurrentAuthorizedExternalEffectCoverageV1 {
        &self.current_coverage
    }

    pub fn release_after_durable_current_coverage(
        self,
        journal: &ReconstructedExecutionJournalV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        effect_scope: &QualifiedEffectScopeCommitmentV1,
        backend_commitment: &QualifiedBackendEffectCommitmentV1,
        coverage_commitment: &QualifiedEffectCoverageCommitmentV1,
        current_verifier_commitment: &QualifiedCurrentVerifierExecutionCommitmentV1,
    ) -> Result<ReadyCurrentAuthorizedEffectExecutionV1, CurrentAuthorizedCoverageError> {
        let expected_binding = CurrentVerifierExecutionBindingV1::Effectful {
            historical_commitment_id: coverage_commitment.id(),
            historical_coverage_id: self.current_coverage.coverage().id(),
            current_coverage_id: self.current_coverage.id(),
        };
        if self.current_coverage.decision_time_unix_ms() != journal_anchor.anchored_at_unix_ms()
            || coverage_commitment.coverage_id() != self.current_coverage.coverage().id()
            || current_verifier_commitment.journal_anchor_id() != journal_anchor.id()
            || current_verifier_commitment.binding() != expected_binding
            || current_verifier_commitment.adoption_id() != self.current_coverage.adoption_id()
            || current_verifier_commitment.current_verifier_id()
                != self.current_coverage.current_verifier_id()
            || current_verifier_commitment.decision_time_unix_ms()
                != self.current_coverage.decision_time_unix_ms()
        {
            return Err(CurrentAuthorizedCoverageError::CurrentVerifierCommitmentMismatch);
        }
        let ready = self.inner.release_after_durable_coverage(
            journal, journal_anchor, effect_scope, backend_commitment, coverage_commitment,
        )?;
        if ready.coverage().id() != self.current_coverage.coverage().id() {
            return Err(CurrentAuthorizedCoverageError::EffectCoverageMismatch);
        }
        Ok(ReadyCurrentAuthorizedEffectExecutionV1 {
            inner: ready,
            current_coverage_id: self.current_coverage.id(),
            adoption_id: self.current_coverage.adoption_id(),
            current_verifier_id: self.current_coverage.current_verifier_id(),
            current_verifier_commitment_id: current_verifier_commitment.id(),
        })
    }
}

#[derive(Debug)]
pub struct ReadyCurrentAuthorizedEffectExecutionV1 {
    inner: ReadyCoverageQualifiedEffectExecutionV1,
    current_coverage_id: CurrentAuthorizedExternalEffectCoverageId,
    adoption_id: QualifiedVerifierProfileAdoptionId,
    current_verifier_id: CurrentAuthorizedVerifierProfileId,
    current_verifier_commitment_id: QualifiedCurrentVerifierExecutionCommitmentId,
}

impl ReadyCurrentAuthorizedEffectExecutionV1 {
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.inner.attempt_id() }
    pub fn backend_id(&self) -> ExecutionBackendId { self.inner.backend_id() }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.inner.subject_id() }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.inner.source_realization_id() }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.inner.target_realization_id() }
    pub fn effect_contract(&self) -> &ExternalEffectContractV1 { self.inner.effect_contract() }
    pub fn current_coverage_id(&self) -> CurrentAuthorizedExternalEffectCoverageId {
        self.current_coverage_id
    }
    pub fn adoption_id(&self) -> QualifiedVerifierProfileAdoptionId { self.adoption_id }
    pub fn current_verifier_id(&self) -> CurrentAuthorizedVerifierProfileId { self.current_verifier_id }
    pub fn current_verifier_commitment_id(&self) -> QualifiedCurrentVerifierExecutionCommitmentId {
        self.current_verifier_commitment_id
    }

    pub fn finish(
        self,
        outcome: ExecutionAttemptOutcomeV1,
        backend_evidence_digest: [u8; 32],
        result_digest: [u8; 32],
    ) -> Result<ExecutionAttemptReceiptV1, CurrentAuthorizedCoverageError> {
        Ok(self.inner.finish(outcome, backend_evidence_digest, result_digest)?)
    }
}

#[derive(Debug)]
pub struct PendingCurrentAuthorizedNoEffectsExecutionV1 {
    inner: PendingProvenNoEffectsExecutionV1,
    current_coverage: CurrentAuthorizedNoExternalEffectsCoverageV1,
}

impl PendingCurrentAuthorizedNoEffectsExecutionV1 {
    pub fn current_coverage(&self) -> &CurrentAuthorizedNoExternalEffectsCoverageV1 {
        &self.current_coverage
    }

    pub fn release_after_durable_current_no_effects(
        self,
        journal: &ReconstructedExecutionJournalV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        commitment: &QualifiedNoEffectsExecutionCommitmentV1,
        current_verifier_commitment: &QualifiedCurrentVerifierExecutionCommitmentV1,
    ) -> Result<ReadyCurrentAuthorizedNoEffectsExecutionV1, CurrentAuthorizedCoverageError> {
        let expected_binding = CurrentVerifierExecutionBindingV1::NoEffects {
            historical_commitment_id: commitment.id(),
            historical_coverage_id: self.current_coverage.coverage().id(),
            current_coverage_id: self.current_coverage.id(),
        };
        if self.current_coverage.decision_time_unix_ms() != journal_anchor.anchored_at_unix_ms()
            || commitment.coverage_id() != self.current_coverage.coverage().id()
            || current_verifier_commitment.journal_anchor_id() != journal_anchor.id()
            || current_verifier_commitment.binding() != expected_binding
            || current_verifier_commitment.adoption_id() != self.current_coverage.adoption_id()
            || current_verifier_commitment.current_verifier_id()
                != self.current_coverage.current_verifier_id()
            || current_verifier_commitment.decision_time_unix_ms()
                != self.current_coverage.decision_time_unix_ms()
        {
            return Err(CurrentAuthorizedCoverageError::CurrentVerifierCommitmentMismatch);
        }
        let ready = self.inner.release_after_protected_no_effects(journal, journal_anchor, commitment)?;
        Ok(ReadyCurrentAuthorizedNoEffectsExecutionV1 {
            inner: ready,
            current_coverage_id: self.current_coverage.id(),
            adoption_id: self.current_coverage.adoption_id(),
            current_verifier_id: self.current_coverage.current_verifier_id(),
            current_verifier_commitment_id: current_verifier_commitment.id(),
        })
    }
}

#[derive(Debug)]
pub struct ReadyCurrentAuthorizedNoEffectsExecutionV1 {
    inner: ReadyProvenNoEffectsExecutionV1,
    current_coverage_id: CurrentAuthorizedNoExternalEffectsCoverageId,
    adoption_id: QualifiedVerifierProfileAdoptionId,
    current_verifier_id: CurrentAuthorizedVerifierProfileId,
    current_verifier_commitment_id: QualifiedCurrentVerifierExecutionCommitmentId,
}

impl ReadyCurrentAuthorizedNoEffectsExecutionV1 {
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.inner.attempt_id() }
    pub fn backend_id(&self) -> ExecutionBackendId { self.inner.backend_id() }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.inner.subject_id() }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.inner.source_realization_id() }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.inner.target_realization_id() }
    pub fn current_coverage_id(&self) -> CurrentAuthorizedNoExternalEffectsCoverageId {
        self.current_coverage_id
    }
    pub fn adoption_id(&self) -> QualifiedVerifierProfileAdoptionId { self.adoption_id }
    pub fn current_verifier_id(&self) -> CurrentAuthorizedVerifierProfileId { self.current_verifier_id }
    pub fn current_verifier_commitment_id(&self) -> QualifiedCurrentVerifierExecutionCommitmentId {
        self.current_verifier_commitment_id
    }

    pub fn finish(
        self,
        outcome: ExecutionAttemptOutcomeV1,
        backend_evidence_digest: [u8; 32],
        result_digest: [u8; 32],
    ) -> Result<ExecutionAttemptReceiptV1, CurrentAuthorizedCoverageError> {
        Ok(self.inner.finish(outcome, backend_evidence_digest, result_digest)?)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_current_authorized_effect_execution(
    bound: CurrentAuthorizedEffectExecutionEligibilityV1,
    current_epoch: &QualifiedTrustedCommitEpochV1,
    previous_epoch: Option<&QualifiedTrustedCommitEpochV1>,
    epoch_anchor_mode: ExecutionEpochAnchorModeV1,
    predecessor_journal_anchor: &QualifiedExecutionJournalAnchorV1,
    session_generation: u64,
    session_nonce: [u8; 32],
) -> Result<PendingCurrentAuthorizedEffectExecutionV1, CurrentAuthorizedCoverageError> {
    let CurrentAuthorizedEffectExecutionEligibilityV1 { inner, current_coverage } = bound;
    let pending = prepare_coverage_qualified_effect_execution(
        inner, current_epoch, previous_epoch, epoch_anchor_mode,
        predecessor_journal_anchor, session_generation, session_nonce,
    )?;
    Ok(PendingCurrentAuthorizedEffectExecutionV1 { inner: pending, current_coverage })
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_current_authorized_no_effects_execution(
    bound: CurrentAuthorizedNoEffectsExecutionEligibilityV1,
    current_epoch: &QualifiedTrustedCommitEpochV1,
    previous_epoch: Option<&QualifiedTrustedCommitEpochV1>,
    epoch_anchor_mode: ExecutionEpochAnchorModeV1,
    predecessor_journal_anchor: &QualifiedExecutionJournalAnchorV1,
    session_generation: u64,
    session_nonce: [u8; 32],
) -> Result<PendingCurrentAuthorizedNoEffectsExecutionV1, CurrentAuthorizedCoverageError> {
    let CurrentAuthorizedNoEffectsExecutionEligibilityV1 { inner, current_coverage } = bound;
    let pending = prepare_proven_no_effects_execution(
        inner, current_epoch, previous_epoch, epoch_anchor_mode,
        predecessor_journal_anchor, session_generation, session_nonce,
    )?;
    Ok(PendingCurrentAuthorizedNoEffectsExecutionV1 { inner: pending, current_coverage })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CurrentAuthorizedCoverageError {
    #[error(transparent)]
    EffectExecution(#[from] CoverageBoundExecutionError),
    #[error(transparent)]
    NoEffectsExecution(#[from] NoEffectsExecutionError),
    #[error("current-authorized coverage decision time must be non-zero")]
    ZeroDecisionTime,
    #[error("fresh current verifier proof does not descend from the exact supplied adoption")]
    AdoptionCurrentnessMismatch,
    #[error("coverage verifier profile is not the exact freshly current authorized verifier")]
    VerifierProfileMismatch,
    #[error("coverage verifier root epoch differs from the freshly current verifier profile")]
    VerifierRootEpochMismatch,
    #[error("coverage analysis occurred before exact verifier adoption was admitted")]
    CoverageBeforeAdoptionAdmission,
    #[error("coverage analysis lies outside the exact verifier adoption validity interval")]
    CoverageOutsideAdoptionValidity,
    #[error("coverage analysis was not performed at the exact physical decision boundary")]
    CoverageNotAtDecisionBoundary,
    #[error("fresh verifier-currentness attestation was not anchored at the exact physical decision boundary")]
    CurrentnessNotAtDecisionBoundary,
    #[error("effectful eligibility carries a different historical coverage proof")]
    EffectCoverageMismatch,
    #[error("no-effects eligibility carries a different historical coverage proof")]
    NoEffectsCoverageMismatch,
    #[error("fresh verifier decision boundary differs from exact A -> B commit time")]
    TransitionDecisionTimeMismatch,
    #[error("protected post-intent journal world differs from fresh verifier decision boundary")]
    DurableDecisionBoundaryMismatch,
    #[error("protected current-verifier execution commitment differs from exact release world")]
    CurrentVerifierCommitmentMismatch,
}

fn require_current_verifier_at_decision(
    coverage_verifier_profile_id: VerifierProfileId,
    coverage_verifier_root_epoch: u64,
    analyzed_at_unix_ms: u64,
    adoption: &QualifiedVerifierProfileAdoptionV1,
    current: &CurrentAuthorizedVerifierProfileV1,
    decision_time_unix_ms: u64,
) -> Result<(), CurrentAuthorizedCoverageError> {
    if decision_time_unix_ms == 0 {
        return Err(CurrentAuthorizedCoverageError::ZeroDecisionTime);
    }
    if current.adoption_id() != adoption.id()
        || current.transition_digest() != adoption.transition_digest()
        || current.subject().id() != adoption.subject().id()
        || current.generation() != adoption.generation()
    {
        return Err(CurrentAuthorizedCoverageError::AdoptionCurrentnessMismatch);
    }
    if coverage_verifier_profile_id != adoption.verifier_profile_id()
        || coverage_verifier_profile_id != current.verifier_profile_id()
    {
        return Err(CurrentAuthorizedCoverageError::VerifierProfileMismatch);
    }
    let current_profile: &VerifierProfileV1 = current.verifier_profile();
    if coverage_verifier_root_epoch != adoption.verifier_profile().root_epoch()
        || coverage_verifier_root_epoch != current_profile.root_epoch()
    {
        return Err(CurrentAuthorizedCoverageError::VerifierRootEpochMismatch);
    }
    if analyzed_at_unix_ms < adoption.admitted_at_unix_ms() {
        return Err(CurrentAuthorizedCoverageError::CoverageBeforeAdoptionAdmission);
    }
    if analyzed_at_unix_ms < adoption.subject().valid_from_unix_ms()
        || analyzed_at_unix_ms >= adoption.subject().valid_until_unix_ms()
    {
        return Err(CurrentAuthorizedCoverageError::CoverageOutsideAdoptionValidity);
    }
    if analyzed_at_unix_ms != decision_time_unix_ms {
        return Err(CurrentAuthorizedCoverageError::CoverageNotAtDecisionBoundary);
    }
    if current.anchored_at_unix_ms() != decision_time_unix_ms {
        return Err(CurrentAuthorizedCoverageError::CurrentnessNotAtDecisionBoundary);
    }
    Ok(())
}

fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    for part in parts {
        h.update(&((*part).len() as u64).to_le_bytes());
        h.update(part);
    }
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_authorized_coverage_domains_are_distinct() {
        assert_ne!(EFFECTFUL_CURRENT_COVERAGE_DOMAIN, NO_EFFECTS_CURRENT_COVERAGE_DOMAIN);
    }
}
