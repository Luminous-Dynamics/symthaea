// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rollback-resistant commitment that a fresh current-verifier proof gated one
//! exact physical execution decision.
//!
//! `CurrentAuthorized*CoverageV1` is deliberately non-Serde type-level authority.
//! That is sufficient to prevent an in-process bypass, but a crash-time auditor also
//! needs durable proof that this exact current-verifier decision was consumed before
//! physical mutation. This module commits that decision under the execution journal's
//! rollback-resistant root while preserving the independent platform root used by the
//! verifier-adoption currentness theorem.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::current_authorized::{
    CurrentAuthorizedExternalEffectCoverageId, CurrentAuthorizedExternalEffectCoverageV1,
    CurrentAuthorizedNoExternalEffectsCoverageId, CurrentAuthorizedNoExternalEffectsCoverageV1,
};
use super::{QualifiedEffectCoverageCommitmentId, QualifiedEffectCoverageCommitmentV1};
use crate::effect_coverage::QualifiedExternalEffectCoverageId;
use crate::execution_journal_anchor::{
    ExecutionJournalAnchorError, ExecutionJournalAnchorProfileId,
    ExecutionJournalAnchorProfileV1, QualifiedExecutionJournalAnchorId,
    QualifiedExecutionJournalAnchorV1,
};
use crate::no_effects_execution::{
    QualifiedNoEffectsExecutionCommitmentId, QualifiedNoEffectsExecutionCommitmentV1,
};
use crate::no_external_effects::QualifiedNoExternalEffectsCoverageId;
use crate::trusted_commit_epoch::QualifiedTrustedCommitEpochId;
use crate::verifier::VerifierProfileId;
use crate::verifier_adoption_authority::{
    CurrentAuthorizedVerifierProfileId, CurrentAuthorizedVerifierProfileV1,
    QualifiedVerifierProfileAdoptionId,
};

pub const CURRENT_VERIFIER_EXECUTION_COMMITMENT_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-current-verifier-execution-commitment-claim-v1";
pub const CURRENT_VERIFIER_EXECUTION_COMMITMENT_AUTH_PURPOSE: &str =
    "symthaea.continuity.current-verifier-execution-commitment.v1";

const CLAIM_DOMAIN: &[u8] =
    b"symthaea.continuity.current-verifier-execution-commitment-claim.v1\0";
const WIRE_DOMAIN: &[u8] =
    b"symthaea.continuity.current-verifier-execution-commitment-wire.v1\0";
const AUTH_DOMAIN: &[u8] =
    b"symthaea.continuity.authenticated-current-verifier-execution-commitment.v1\0";
const QUALIFIED_DOMAIN: &[u8] =
    b"symthaea.continuity.qualified-current-verifier-execution-commitment.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name {
            pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
        }
    };
}

digest_id!(CurrentVerifierExecutionCommitmentClaimId);
digest_id!(AuthenticatedCurrentVerifierExecutionCommitmentId);
digest_id!(QualifiedCurrentVerifierExecutionCommitmentId);

/// Typed lane-specific material retained in the durable commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurrentVerifierExecutionBindingV1 {
    Effectful {
        historical_commitment_id: QualifiedEffectCoverageCommitmentId,
        historical_coverage_id: QualifiedExternalEffectCoverageId,
        current_coverage_id: CurrentAuthorizedExternalEffectCoverageId,
    },
    NoEffects {
        historical_commitment_id: QualifiedNoEffectsExecutionCommitmentId,
        historical_coverage_id: QualifiedNoExternalEffectsCoverageId,
        current_coverage_id: CurrentAuthorizedNoExternalEffectsCoverageId,
    },
}

impl CurrentVerifierExecutionBindingV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Effectful { .. } => 1,
            Self::NoEffects { .. } => 2,
        }
    }
}

/// Serializable platform claim over the exact current-verifier decision consumed by
/// one already-protected historical coverage world.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentVerifierExecutionCommitmentClaimV1 {
    schema_version: String,
    execution_profile_id: ExecutionJournalAnchorProfileId,
    execution_root_epoch: u64,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    binding: CurrentVerifierExecutionBindingV1,
    adoption_id: QualifiedVerifierProfileAdoptionId,
    current_verifier_id: CurrentAuthorizedVerifierProfileId,
    verifier_profile_id: VerifierProfileId,
    verifier_platform_profile_id: ExecutionJournalAnchorProfileId,
    verifier_platform_root_epoch: u64,
    verifier_currentness_sequence: u64,
    verifier_freshness_challenge_digest: [u8; 32],
    decision_time_unix_ms: u64,
    raw_commitment_evidence_digest: [u8; 32],
    claim_id: CurrentVerifierExecutionCommitmentClaimId,
}

impl CurrentVerifierExecutionCommitmentClaimV1 {
    pub fn for_effectful(
        profile: &ExecutionJournalAnchorProfileV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        historical: &QualifiedEffectCoverageCommitmentV1,
        current_coverage: &CurrentAuthorizedExternalEffectCoverageV1,
        current_verifier: &CurrentAuthorizedVerifierProfileV1,
        raw_commitment_evidence_digest: [u8; 32],
    ) -> Result<Self, CurrentVerifierExecutionCommitmentError> {
        let binding = CurrentVerifierExecutionBindingV1::Effectful {
            historical_commitment_id: historical.id(),
            historical_coverage_id: historical.coverage_id(),
            current_coverage_id: current_coverage.id(),
        };
        Self::new_common(
            profile,
            journal_anchor,
            binding,
            historical.journal_anchor_id(),
            historical.coverage_id() == current_coverage.coverage().id(),
            current_coverage.adoption_id(),
            current_coverage.current_verifier_id(),
            current_coverage.coverage().verifier_profile_id(),
            current_coverage.decision_time_unix_ms(),
            current_verifier,
            raw_commitment_evidence_digest,
        )
    }

    pub fn for_no_effects(
        profile: &ExecutionJournalAnchorProfileV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        historical: &QualifiedNoEffectsExecutionCommitmentV1,
        current_coverage: &CurrentAuthorizedNoExternalEffectsCoverageV1,
        current_verifier: &CurrentAuthorizedVerifierProfileV1,
        raw_commitment_evidence_digest: [u8; 32],
    ) -> Result<Self, CurrentVerifierExecutionCommitmentError> {
        let binding = CurrentVerifierExecutionBindingV1::NoEffects {
            historical_commitment_id: historical.id(),
            historical_coverage_id: historical.coverage_id(),
            current_coverage_id: current_coverage.id(),
        };
        Self::new_common(
            profile,
            journal_anchor,
            binding,
            historical.journal_anchor_id(),
            historical.coverage_id() == current_coverage.coverage().id(),
            current_coverage.adoption_id(),
            current_coverage.current_verifier_id(),
            current_coverage.coverage().verifier_profile_id(),
            current_coverage.decision_time_unix_ms(),
            current_verifier,
            raw_commitment_evidence_digest,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_common(
        profile: &ExecutionJournalAnchorProfileV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        binding: CurrentVerifierExecutionBindingV1,
        historical_journal_anchor_id: QualifiedExecutionJournalAnchorId,
        historical_coverage_matches: bool,
        adoption_id: QualifiedVerifierProfileAdoptionId,
        current_verifier_id: CurrentAuthorizedVerifierProfileId,
        verifier_profile_id: VerifierProfileId,
        decision_time_unix_ms: u64,
        current_verifier: &CurrentAuthorizedVerifierProfileV1,
        raw_commitment_evidence_digest: [u8; 32],
    ) -> Result<Self, CurrentVerifierExecutionCommitmentError> {
        profile.validate()?;
        if raw_commitment_evidence_digest == [0; 32]
            || current_verifier.freshness_challenge_digest() == [0; 32]
        {
            return Err(CurrentVerifierExecutionCommitmentError::ZeroDigest);
        }
        if profile.id() != journal_anchor.profile_id()
            || profile.root_epoch() != journal_anchor.root_epoch()
            || historical_journal_anchor_id != journal_anchor.id()
            || !historical_coverage_matches
        {
            return Err(CurrentVerifierExecutionCommitmentError::HistoricalWorldMismatch);
        }
        if adoption_id != current_verifier.adoption_id()
            || current_verifier_id != current_verifier.id()
            || verifier_profile_id != current_verifier.verifier_profile_id()
        {
            return Err(CurrentVerifierExecutionCommitmentError::CurrentVerifierMismatch);
        }
        if decision_time_unix_ms == 0
            || decision_time_unix_ms != journal_anchor.anchored_at_unix_ms()
            || decision_time_unix_ms != current_verifier.anchored_at_unix_ms()
        {
            return Err(CurrentVerifierExecutionCommitmentError::DecisionBoundaryMismatch);
        }
        if current_verifier.platform_root_epoch() == 0 || current_verifier.anchor_sequence() == 0 {
            return Err(CurrentVerifierExecutionCommitmentError::ZeroGenerationOrSequence);
        }

        let claim_id = CurrentVerifierExecutionCommitmentClaimId(hash_claim(
            profile.id(),
            profile.root_epoch(),
            journal_anchor.id(),
            journal_anchor.trusted_epoch_id(),
            binding,
            adoption_id,
            current_verifier_id,
            verifier_profile_id,
            current_verifier.platform_profile_id(),
            current_verifier.platform_root_epoch(),
            current_verifier.anchor_sequence(),
            current_verifier.freshness_challenge_digest(),
            decision_time_unix_ms,
            raw_commitment_evidence_digest,
        ));
        Ok(Self {
            schema_version: CURRENT_VERIFIER_EXECUTION_COMMITMENT_CLAIM_SCHEMA_V1.to_owned(),
            execution_profile_id: profile.id(),
            execution_root_epoch: profile.root_epoch(),
            journal_anchor_id: journal_anchor.id(),
            trusted_epoch_id: journal_anchor.trusted_epoch_id(),
            binding,
            adoption_id,
            current_verifier_id,
            verifier_profile_id,
            verifier_platform_profile_id: current_verifier.platform_profile_id(),
            verifier_platform_root_epoch: current_verifier.platform_root_epoch(),
            verifier_currentness_sequence: current_verifier.anchor_sequence(),
            verifier_freshness_challenge_digest: current_verifier.freshness_challenge_digest(),
            decision_time_unix_ms,
            raw_commitment_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), CurrentVerifierExecutionCommitmentError> {
        if self.schema_version != CURRENT_VERIFIER_EXECUTION_COMMITMENT_CLAIM_SCHEMA_V1 {
            return Err(CurrentVerifierExecutionCommitmentError::UnsupportedClaimSchema(
                self.schema_version.clone(),
            ));
        }
        if self.execution_root_epoch == 0
            || self.verifier_platform_root_epoch == 0
            || self.verifier_currentness_sequence == 0
            || self.decision_time_unix_ms == 0
        {
            return Err(CurrentVerifierExecutionCommitmentError::ZeroGenerationOrSequence);
        }
        if self.verifier_freshness_challenge_digest == [0; 32]
            || self.raw_commitment_evidence_digest == [0; 32]
        {
            return Err(CurrentVerifierExecutionCommitmentError::ZeroDigest);
        }
        let expected = CurrentVerifierExecutionCommitmentClaimId(hash_claim(
            self.execution_profile_id,
            self.execution_root_epoch,
            self.journal_anchor_id,
            self.trusted_epoch_id,
            self.binding,
            self.adoption_id,
            self.current_verifier_id,
            self.verifier_profile_id,
            self.verifier_platform_profile_id,
            self.verifier_platform_root_epoch,
            self.verifier_currentness_sequence,
            self.verifier_freshness_challenge_digest,
            self.decision_time_unix_ms,
            self.raw_commitment_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(CurrentVerifierExecutionCommitmentError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> CurrentVerifierExecutionCommitmentClaimId { self.claim_id }
    pub fn binding(&self) -> CurrentVerifierExecutionBindingV1 { self.binding }
    pub fn journal_anchor_id(&self) -> QualifiedExecutionJournalAnchorId { self.journal_anchor_id }
    pub fn adoption_id(&self) -> QualifiedVerifierProfileAdoptionId { self.adoption_id }
    pub fn current_verifier_id(&self) -> CurrentAuthorizedVerifierProfileId { self.current_verifier_id }
    pub fn decision_time_unix_ms(&self) -> u64 { self.decision_time_unix_ms }
}

pub fn canonical_current_verifier_execution_commitment_claim_bytes(
    claim: &CurrentVerifierExecutionCommitmentClaimV1,
) -> Result<Vec<u8>, CurrentVerifierExecutionCommitmentError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(768);
    out.extend_from_slice(WIRE_DOMAIN);
    out.extend_from_slice(claim.execution_profile_id.as_bytes());
    out.extend_from_slice(&claim.execution_root_epoch.to_le_bytes());
    out.extend_from_slice(claim.journal_anchor_id.as_bytes());
    out.extend_from_slice(claim.trusted_epoch_id.as_bytes());
    encode_binding(&mut out, claim.binding);
    out.extend_from_slice(claim.adoption_id.as_bytes());
    out.extend_from_slice(claim.current_verifier_id.as_bytes());
    out.extend_from_slice(claim.verifier_profile_id.as_bytes());
    out.extend_from_slice(claim.verifier_platform_profile_id.as_bytes());
    out.extend_from_slice(&claim.verifier_platform_root_epoch.to_le_bytes());
    out.extend_from_slice(&claim.verifier_currentness_sequence.to_le_bytes());
    out.extend_from_slice(&claim.verifier_freshness_challenge_digest);
    out.extend_from_slice(&claim.decision_time_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.raw_commitment_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_current_verifier_execution_commitment_claim_digest(
    claim: &CurrentVerifierExecutionCommitmentClaimV1,
) -> Result<[u8; 32], CurrentVerifierExecutionCommitmentError> {
    Ok(*blake3::hash(&canonical_current_verifier_execution_commitment_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedCurrentVerifierExecutionCommitmentV1 {
    claim: CurrentVerifierExecutionCommitmentClaimV1,
    profile: ExecutionJournalAnchorProfileV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedCurrentVerifierExecutionCommitmentId,
}

impl AuthenticatedCurrentVerifierExecutionCommitmentV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: CurrentVerifierExecutionCommitmentClaimV1,
        profile: ExecutionJournalAnchorProfileV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, CurrentVerifierExecutionCommitmentError> {
        claim.validate()?;
        profile.validate()?;
        if authentication_evidence_digest == [0; 32] {
            return Err(CurrentVerifierExecutionCommitmentError::ZeroDigest);
        }
        if claim.execution_profile_id != profile.id()
            || claim.execution_root_epoch != profile.root_epoch()
        {
            return Err(CurrentVerifierExecutionCommitmentError::CommitmentRootMismatch);
        }
        let evidence_id = AuthenticatedCurrentVerifierExecutionCommitmentId(domain_hash_parts(
            AUTH_DOMAIN,
            &[
                claim.id().as_bytes(),
                profile.id().as_bytes(),
                &profile.root_epoch().to_le_bytes(),
                &authentication_evidence_digest,
            ],
        ));
        Ok(Self { claim, profile, authentication_evidence_digest, evidence_id })
    }
}

/// Non-Serde proof that the exact fresh-current verifier decision was durably bound
/// to the same exact post-intent journal world as the historical coverage commitment.
#[derive(Debug, Clone)]
pub struct QualifiedCurrentVerifierExecutionCommitmentV1 {
    commitment_id: QualifiedCurrentVerifierExecutionCommitmentId,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    binding: CurrentVerifierExecutionBindingV1,
    adoption_id: QualifiedVerifierProfileAdoptionId,
    current_verifier_id: CurrentAuthorizedVerifierProfileId,
    decision_time_unix_ms: u64,
}

impl QualifiedCurrentVerifierExecutionCommitmentV1 {
    pub(crate) fn qualify_effectful(
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        historical: &QualifiedEffectCoverageCommitmentV1,
        current_coverage: &CurrentAuthorizedExternalEffectCoverageV1,
        current_verifier: &CurrentAuthorizedVerifierProfileV1,
        authenticated: &AuthenticatedCurrentVerifierExecutionCommitmentV1,
    ) -> Result<Self, CurrentVerifierExecutionCommitmentError> {
        let expected_binding = CurrentVerifierExecutionBindingV1::Effectful {
            historical_commitment_id: historical.id(),
            historical_coverage_id: historical.coverage_id(),
            current_coverage_id: current_coverage.id(),
        };
        Self::qualify_common(
            journal_anchor,
            historical.journal_anchor_id(),
            historical.coverage_id() == current_coverage.coverage().id(),
            expected_binding,
            current_coverage.adoption_id(),
            current_coverage.current_verifier_id(),
            current_coverage.decision_time_unix_ms(),
            current_verifier,
            authenticated,
        )
    }

    pub(crate) fn qualify_no_effects(
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        historical: &QualifiedNoEffectsExecutionCommitmentV1,
        current_coverage: &CurrentAuthorizedNoExternalEffectsCoverageV1,
        current_verifier: &CurrentAuthorizedVerifierProfileV1,
        authenticated: &AuthenticatedCurrentVerifierExecutionCommitmentV1,
    ) -> Result<Self, CurrentVerifierExecutionCommitmentError> {
        let expected_binding = CurrentVerifierExecutionBindingV1::NoEffects {
            historical_commitment_id: historical.id(),
            historical_coverage_id: historical.coverage_id(),
            current_coverage_id: current_coverage.id(),
        };
        Self::qualify_common(
            journal_anchor,
            historical.journal_anchor_id(),
            historical.coverage_id() == current_coverage.coverage().id(),
            expected_binding,
            current_coverage.adoption_id(),
            current_coverage.current_verifier_id(),
            current_coverage.decision_time_unix_ms(),
            current_verifier,
            authenticated,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn qualify_common(
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        historical_journal_anchor_id: QualifiedExecutionJournalAnchorId,
        historical_coverage_matches: bool,
        expected_binding: CurrentVerifierExecutionBindingV1,
        expected_adoption_id: QualifiedVerifierProfileAdoptionId,
        expected_current_verifier_id: CurrentAuthorizedVerifierProfileId,
        expected_decision_time_unix_ms: u64,
        current_verifier: &CurrentAuthorizedVerifierProfileV1,
        authenticated: &AuthenticatedCurrentVerifierExecutionCommitmentV1,
    ) -> Result<Self, CurrentVerifierExecutionCommitmentError> {
        authenticated.claim.validate()?;
        authenticated.profile.validate()?;
        let claim = &authenticated.claim;
        if authenticated.profile.id() != journal_anchor.profile_id()
            || authenticated.profile.root_epoch() != journal_anchor.root_epoch()
            || claim.execution_profile_id != journal_anchor.profile_id()
            || claim.execution_root_epoch != journal_anchor.root_epoch()
            || claim.journal_anchor_id != journal_anchor.id()
            || claim.trusted_epoch_id != journal_anchor.trusted_epoch_id()
            || historical_journal_anchor_id != journal_anchor.id()
            || !historical_coverage_matches
        {
            return Err(CurrentVerifierExecutionCommitmentError::HistoricalWorldMismatch);
        }
        if claim.binding != expected_binding
            || claim.adoption_id != expected_adoption_id
            || claim.current_verifier_id != expected_current_verifier_id
            || claim.current_verifier_id != current_verifier.id()
            || claim.adoption_id != current_verifier.adoption_id()
            || claim.verifier_profile_id != current_verifier.verifier_profile_id()
            || claim.verifier_platform_profile_id != current_verifier.platform_profile_id()
            || claim.verifier_platform_root_epoch != current_verifier.platform_root_epoch()
            || claim.verifier_currentness_sequence != current_verifier.anchor_sequence()
            || claim.verifier_freshness_challenge_digest != current_verifier.freshness_challenge_digest()
        {
            return Err(CurrentVerifierExecutionCommitmentError::CurrentVerifierMismatch);
        }
        if claim.decision_time_unix_ms != expected_decision_time_unix_ms
            || claim.decision_time_unix_ms != journal_anchor.anchored_at_unix_ms()
            || claim.decision_time_unix_ms != current_verifier.anchored_at_unix_ms()
        {
            return Err(CurrentVerifierExecutionCommitmentError::DecisionBoundaryMismatch);
        }
        let commitment_id = QualifiedCurrentVerifierExecutionCommitmentId(domain_hash_parts(
            QUALIFIED_DOMAIN,
            &[
                claim.id().as_bytes(),
                journal_anchor.id().as_bytes(),
                authenticated.evidence_id.as_bytes(),
                expected_current_verifier_id.as_bytes(),
                &expected_decision_time_unix_ms.to_le_bytes(),
            ],
        ));
        Ok(Self {
            commitment_id,
            journal_anchor_id: journal_anchor.id(),
            binding: expected_binding,
            adoption_id: expected_adoption_id,
            current_verifier_id: expected_current_verifier_id,
            decision_time_unix_ms: expected_decision_time_unix_ms,
        })
    }

    pub fn id(&self) -> QualifiedCurrentVerifierExecutionCommitmentId { self.commitment_id }
    pub fn journal_anchor_id(&self) -> QualifiedExecutionJournalAnchorId { self.journal_anchor_id }
    pub fn binding(&self) -> CurrentVerifierExecutionBindingV1 { self.binding }
    pub fn adoption_id(&self) -> QualifiedVerifierProfileAdoptionId { self.adoption_id }
    pub fn current_verifier_id(&self) -> CurrentAuthorizedVerifierProfileId { self.current_verifier_id }
    pub fn decision_time_unix_ms(&self) -> u64 { self.decision_time_unix_ms }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CurrentVerifierExecutionCommitmentError {
    #[error(transparent)]
    JournalAnchor(#[from] ExecutionJournalAnchorError),
    #[error("unsupported current-verifier execution commitment schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("current-verifier execution commitment digest must be non-zero")]
    ZeroDigest,
    #[error("current-verifier execution commitment generation/sequence must be non-zero")]
    ZeroGenerationOrSequence,
    #[error("historical coverage commitment differs from exact post-intent journal world")]
    HistoricalWorldMismatch,
    #[error("current verifier/adoption proof differs from current-authorized coverage wrapper")]
    CurrentVerifierMismatch,
    #[error("current-verifier execution decision boundary is not exact")]
    DecisionBoundaryMismatch,
    #[error("current-verifier execution commitment identity mismatch")]
    ClaimIdentityMismatch,
    #[error("current-verifier execution commitment changed execution rollback-resistant root")]
    CommitmentRootMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    execution_profile_id: ExecutionJournalAnchorProfileId,
    execution_root_epoch: u64,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    binding: CurrentVerifierExecutionBindingV1,
    adoption_id: QualifiedVerifierProfileAdoptionId,
    current_verifier_id: CurrentAuthorizedVerifierProfileId,
    verifier_profile_id: VerifierProfileId,
    verifier_platform_profile_id: ExecutionJournalAnchorProfileId,
    verifier_platform_root_epoch: u64,
    verifier_currentness_sequence: u64,
    verifier_freshness_challenge_digest: [u8; 32],
    decision_time_unix_ms: u64,
    raw_commitment_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(CLAIM_DOMAIN);
    h.update(execution_profile_id.as_bytes());
    h.update(&execution_root_epoch.to_le_bytes());
    h.update(journal_anchor_id.as_bytes());
    h.update(trusted_epoch_id.as_bytes());
    hash_binding(&mut h, binding);
    h.update(adoption_id.as_bytes());
    h.update(current_verifier_id.as_bytes());
    h.update(verifier_profile_id.as_bytes());
    h.update(verifier_platform_profile_id.as_bytes());
    h.update(&verifier_platform_root_epoch.to_le_bytes());
    h.update(&verifier_currentness_sequence.to_le_bytes());
    h.update(&verifier_freshness_challenge_digest);
    h.update(&decision_time_unix_ms.to_le_bytes());
    h.update(&raw_commitment_evidence_digest);
    *h.finalize().as_bytes()
}

fn hash_binding(h: &mut blake3::Hasher, binding: CurrentVerifierExecutionBindingV1) {
    h.update(&[binding.tag()]);
    match binding {
        CurrentVerifierExecutionBindingV1::Effectful {
            historical_commitment_id,
            historical_coverage_id,
            current_coverage_id,
        } => {
            h.update(historical_commitment_id.as_bytes());
            h.update(historical_coverage_id.as_bytes());
            h.update(current_coverage_id.as_bytes());
        }
        CurrentVerifierExecutionBindingV1::NoEffects {
            historical_commitment_id,
            historical_coverage_id,
            current_coverage_id,
        } => {
            h.update(historical_commitment_id.as_bytes());
            h.update(historical_coverage_id.as_bytes());
            h.update(current_coverage_id.as_bytes());
        }
    }
}

fn encode_binding(out: &mut Vec<u8>, binding: CurrentVerifierExecutionBindingV1) {
    out.push(binding.tag());
    match binding {
        CurrentVerifierExecutionBindingV1::Effectful {
            historical_commitment_id,
            historical_coverage_id,
            current_coverage_id,
        } => {
            out.extend_from_slice(historical_commitment_id.as_bytes());
            out.extend_from_slice(historical_coverage_id.as_bytes());
            out.extend_from_slice(current_coverage_id.as_bytes());
        }
        CurrentVerifierExecutionBindingV1::NoEffects {
            historical_commitment_id,
            historical_coverage_id,
            current_coverage_id,
        } => {
            out.extend_from_slice(historical_commitment_id.as_bytes());
            out.extend_from_slice(historical_coverage_id.as_bytes());
            out.extend_from_slice(current_coverage_id.as_bytes());
        }
    }
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
    fn current_verifier_commitment_domains_are_distinct() {
        assert_ne!(CLAIM_DOMAIN, WIRE_DOMAIN);
        assert_ne!(WIRE_DOMAIN, AUTH_DOMAIN);
        assert_ne!(AUTH_DOMAIN, QUALIFIED_DOMAIN);
    }
}
