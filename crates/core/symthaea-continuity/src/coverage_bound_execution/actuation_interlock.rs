// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Protocol-only actuation interlock and fencing contract.
//!
//! This module deliberately grants **no physical authority**. It defines the exact
//! subject and monotonic decision chain a future Spore/NETCONF/Redfish/gNOI/storage
//! adapter must authenticate and enforce at the same boundary that performs the
//! irreversible mutation.
//!
//! Core theorem:
//!
//! `DurableExecutionAuthority != ActuationDecision != BackendEnforcement`.
//!
//! A wall-clock TTL is not the source of stale-holder safety. V1 uses a monotonically
//! increasing fencing generation plus an exact predecessor chain. A later generation
//! must make earlier generations stale at the resource boundary; this crate does not
//! claim that property until a backend-specific adapter proves it.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::current_authority_release::{
    ReadyDurableCurrentAuthorityEffectExecutionV1,
    ReadyDurableCurrentAuthorityNoEffectsExecutionV1,
};
use super::current_transition_authority_commitment::{
    QualifiedCurrentTransitionAuthorityExecutionCommitmentId,
    QualifiedCurrentTransitionAuthorityExecutionCommitmentV1,
};
use super::current_verifier_commitment::QualifiedCurrentVerifierExecutionCommitmentId;
use crate::execution_capability::{ExecutionAttemptId, ExecutionBackendId, ExecutionBackendProfileV1};
use crate::execution_journal_anchor::{
    QualifiedExecutionJournalAnchorId, QualifiedExecutionJournalAnchorV1,
};
use crate::scope::ContinuitySubjectId;
use crate::trusted_commit_epoch::QualifiedTrustedCommitEpochId;
use crate::witness::TargetRealizationId;

pub const ACTUATION_ENFORCEMENT_BOUNDARY_PROFILE_SCHEMA_V1: &str =
    "symthaea-continuity-actuation-enforcement-boundary-profile-v1";
pub const ACTUATION_INTERLOCK_SUBJECT_SCHEMA_V1: &str =
    "symthaea-continuity-actuation-interlock-subject-v1";
pub const ACTUATION_INTERLOCK_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-actuation-interlock-claim-v1";
pub const ACTUATION_INTERLOCK_AUTH_PURPOSE: &str =
    "symthaea.continuity.actuation-interlock.v1";

const PROFILE_DOMAIN: &[u8] = b"symthaea.continuity.actuation-enforcement-boundary-profile.v1\0";
const SUBJECT_DOMAIN: &[u8] = b"symthaea.continuity.actuation-interlock-subject.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.actuation-interlock-claim.v1\0";
const WIRE_DOMAIN: &[u8] = b"symthaea.continuity.actuation-interlock-wire.v1\0";
const BOUND_DOMAIN: &[u8] = b"symthaea.continuity.bound-actuation-interlock-decision.v1\0";
const MAX_TEXT_BYTES: usize = 1024;

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name {
            pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
        }
    };
}

digest_id!(ActuationEnforcementBoundaryProfileId);
digest_id!(ActuationInterlockSubjectId);
digest_id!(ActuationInterlockClaimId);
digest_id!(BoundActuationInterlockDecisionId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActuationExecutionLaneV1 {
    Effectful,
    ProvenNoEffects,
}

impl ActuationExecutionLaneV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Effectful => 1,
            Self::ProvenNoEffects => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActuationDenyReasonV1 {
    EmergencyStop,
    CurrentAuthorityRevoked,
    CurrentAuthoritySuperseded,
    VerifierSuperseded,
    BackendSuperseded,
    ResourceUnavailable,
    OperatorDeny,
    PolicyDeny,
}

impl ActuationDenyReasonV1 {
    fn tag(self) -> u8 {
        match self {
            Self::EmergencyStop => 1,
            Self::CurrentAuthorityRevoked => 2,
            Self::CurrentAuthoritySuperseded => 3,
            Self::VerifierSuperseded => 4,
            Self::BackendSuperseded => 5,
            Self::ResourceUnavailable => 6,
            Self::OperatorDeny => 7,
            Self::PolicyDeny => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActuationInterlockDispositionV1 {
    Permit,
    Deny { reason: ActuationDenyReasonV1 },
}

impl ActuationInterlockDispositionV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Permit => 1,
            Self::Deny { .. } => 2,
        }
    }

    fn is_deny(self) -> bool {
        matches!(self, Self::Deny { .. })
    }
}

/// Descriptive identity of the boundary that is expected to check a fencing
/// generation and perform/authorize the irreversible mutation.
///
/// This profile is configuration, not proof that the boundary actually enforces the
/// contract. Backend-specific qualification must establish that later.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationEnforcementBoundaryProfileV1 {
    schema_version: String,
    boundary_name: String,
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    boundary_implementation_digest: [u8; 32],
    one_use_mechanism_digest: [u8; 32],
    profile_generation: u64,
    profile_id: ActuationEnforcementBoundaryProfileId,
}

impl ActuationEnforcementBoundaryProfileV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        boundary_name: impl Into<String>,
        backend: &ExecutionBackendProfileV1,
        boundary_implementation_digest: [u8; 32],
        one_use_mechanism_digest: [u8; 32],
        profile_generation: u64,
    ) -> Result<Self, ActuationInterlockError> {
        backend.validate()?;
        let boundary_name = checked_text("actuation enforcement boundary name", boundary_name.into())?;
        if boundary_implementation_digest == [0; 32] || one_use_mechanism_digest == [0; 32] {
            return Err(ActuationInterlockError::ZeroDigest);
        }
        if profile_generation == 0 {
            return Err(ActuationInterlockError::ZeroGenerationOrTime);
        }
        let profile_id = ActuationEnforcementBoundaryProfileId(hash_profile(
            &boundary_name,
            backend.id(),
            backend.implementation_digest(),
            backend.backend_generation(),
            boundary_implementation_digest,
            one_use_mechanism_digest,
            profile_generation,
        ));
        Ok(Self {
            schema_version: ACTUATION_ENFORCEMENT_BOUNDARY_PROFILE_SCHEMA_V1.to_owned(),
            boundary_name,
            backend_id: backend.id(),
            backend_implementation_digest: backend.implementation_digest(),
            backend_generation: backend.backend_generation(),
            boundary_implementation_digest,
            one_use_mechanism_digest,
            profile_generation,
            profile_id,
        })
    }

    pub fn validate(&self) -> Result<(), ActuationInterlockError> {
        if self.schema_version != ACTUATION_ENFORCEMENT_BOUNDARY_PROFILE_SCHEMA_V1 {
            return Err(ActuationInterlockError::UnsupportedProfileSchema(
                self.schema_version.clone(),
            ));
        }
        let canonical = checked_text("actuation enforcement boundary name", self.boundary_name.clone())?;
        if canonical != self.boundary_name {
            return Err(ActuationInterlockError::NonCanonicalText);
        }
        if self.backend_implementation_digest == [0; 32]
            || self.boundary_implementation_digest == [0; 32]
            || self.one_use_mechanism_digest == [0; 32]
        {
            return Err(ActuationInterlockError::ZeroDigest);
        }
        if self.backend_generation == 0 || self.profile_generation == 0 {
            return Err(ActuationInterlockError::ZeroGenerationOrTime);
        }
        let expected = ActuationEnforcementBoundaryProfileId(hash_profile(
            &self.boundary_name,
            self.backend_id,
            self.backend_implementation_digest,
            self.backend_generation,
            self.boundary_implementation_digest,
            self.one_use_mechanism_digest,
            self.profile_generation,
        ));
        if expected != self.profile_id {
            return Err(ActuationInterlockError::ProfileIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ActuationEnforcementBoundaryProfileId { self.profile_id }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn backend_implementation_digest(&self) -> [u8; 32] { self.backend_implementation_digest }
    pub fn backend_generation(&self) -> u64 { self.backend_generation }
    pub fn boundary_implementation_digest(&self) -> [u8; 32] { self.boundary_implementation_digest }
    pub fn one_use_mechanism_digest(&self) -> [u8; 32] { self.one_use_mechanism_digest }
    pub fn profile_generation(&self) -> u64 { self.profile_generation }
}

/// Exact execution world for which a backend/resource may later issue a fencing
/// decision. Serializable and descriptive only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationInterlockSubjectV1 {
    schema_version: String,
    lane: ActuationExecutionLaneV1,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    current_authority_commitment_id: QualifiedCurrentTransitionAuthorityExecutionCommitmentId,
    current_verifier_commitment_id: QualifiedCurrentVerifierExecutionCommitmentId,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    enforcement_profile_id: ActuationEnforcementBoundaryProfileId,
    enforcement_boundary_digest: [u8; 32],
    one_use_mechanism_digest: [u8; 32],
    enforcement_profile_generation: u64,
    subject_record_id: ActuationInterlockSubjectId,
}

impl ActuationInterlockSubjectV1 {
    pub fn for_effectful(
        ready: &ReadyDurableCurrentAuthorityEffectExecutionV1,
        current_authority_commitment: &QualifiedCurrentTransitionAuthorityExecutionCommitmentV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        backend: &ExecutionBackendProfileV1,
        enforcement: &ActuationEnforcementBoundaryProfileV1,
    ) -> Result<Self, ActuationInterlockError> {
        Self::new_common(
            ActuationExecutionLaneV1::Effectful,
            ready.attempt_id(),
            ready.subject_id(),
            ready.source_realization_id(),
            ready.target_realization_id(),
            ready.backend_id(),
            ready.current_authority_commitment_id(),
            current_authority_commitment,
            journal_anchor,
            backend,
            enforcement,
        )
    }

    pub fn for_no_effects(
        ready: &ReadyDurableCurrentAuthorityNoEffectsExecutionV1,
        current_authority_commitment: &QualifiedCurrentTransitionAuthorityExecutionCommitmentV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        backend: &ExecutionBackendProfileV1,
        enforcement: &ActuationEnforcementBoundaryProfileV1,
    ) -> Result<Self, ActuationInterlockError> {
        Self::new_common(
            ActuationExecutionLaneV1::ProvenNoEffects,
            ready.attempt_id(),
            ready.subject_id(),
            ready.source_realization_id(),
            ready.target_realization_id(),
            ready.backend_id(),
            ready.current_authority_commitment_id(),
            current_authority_commitment,
            journal_anchor,
            backend,
            enforcement,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_common(
        lane: ActuationExecutionLaneV1,
        attempt_id: ExecutionAttemptId,
        subject_id: ContinuitySubjectId,
        source_realization_id: TargetRealizationId,
        target_realization_id: TargetRealizationId,
        ready_backend_id: ExecutionBackendId,
        ready_authority_commitment_id: QualifiedCurrentTransitionAuthorityExecutionCommitmentId,
        current_authority_commitment: &QualifiedCurrentTransitionAuthorityExecutionCommitmentV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        backend: &ExecutionBackendProfileV1,
        enforcement: &ActuationEnforcementBoundaryProfileV1,
    ) -> Result<Self, ActuationInterlockError> {
        backend.validate()?;
        enforcement.validate()?;
        if ready_authority_commitment_id != current_authority_commitment.id()
            || current_authority_commitment.binding().attempt_id() != attempt_id
            || current_authority_commitment.subject_id() != subject_id
            || current_authority_commitment.source_realization_id() != source_realization_id
            || current_authority_commitment.target_realization_id() != target_realization_id
            || current_authority_commitment.journal_anchor_id() != journal_anchor.id()
        {
            return Err(ActuationInterlockError::ExecutionWorldMismatch);
        }
        if ready_backend_id != backend.id()
            || enforcement.backend_id() != backend.id()
            || enforcement.backend_implementation_digest() != backend.implementation_digest()
            || enforcement.backend_generation() != backend.backend_generation()
        {
            return Err(ActuationInterlockError::BackendBoundaryMismatch);
        }
        if source_realization_id == target_realization_id {
            return Err(ActuationInterlockError::SourceEqualsTarget);
        }
        let subject_record_id = ActuationInterlockSubjectId(hash_subject(
            lane,
            attempt_id,
            subject_id,
            source_realization_id,
            target_realization_id,
            backend.id(),
            backend.implementation_digest(),
            backend.backend_generation(),
            current_authority_commitment.id(),
            current_authority_commitment.current_verifier_commitment_id(),
            journal_anchor.id(),
            journal_anchor.trusted_epoch_id(),
            enforcement.id(),
            enforcement.boundary_implementation_digest(),
            enforcement.one_use_mechanism_digest(),
            enforcement.profile_generation(),
        ));
        Ok(Self {
            schema_version: ACTUATION_INTERLOCK_SUBJECT_SCHEMA_V1.to_owned(),
            lane,
            attempt_id,
            subject_id,
            source_realization_id,
            target_realization_id,
            backend_id: backend.id(),
            backend_implementation_digest: backend.implementation_digest(),
            backend_generation: backend.backend_generation(),
            current_authority_commitment_id: current_authority_commitment.id(),
            current_verifier_commitment_id: current_authority_commitment.current_verifier_commitment_id(),
            journal_anchor_id: journal_anchor.id(),
            trusted_epoch_id: journal_anchor.trusted_epoch_id(),
            enforcement_profile_id: enforcement.id(),
            enforcement_boundary_digest: enforcement.boundary_implementation_digest(),
            one_use_mechanism_digest: enforcement.one_use_mechanism_digest(),
            enforcement_profile_generation: enforcement.profile_generation(),
            subject_record_id,
        })
    }

    pub fn validate(&self) -> Result<(), ActuationInterlockError> {
        if self.schema_version != ACTUATION_INTERLOCK_SUBJECT_SCHEMA_V1 {
            return Err(ActuationInterlockError::UnsupportedSubjectSchema(
                self.schema_version.clone(),
            ));
        }
        if self.backend_generation == 0 || self.enforcement_profile_generation == 0 {
            return Err(ActuationInterlockError::ZeroGenerationOrTime);
        }
        if self.backend_implementation_digest == [0; 32]
            || self.enforcement_boundary_digest == [0; 32]
            || self.one_use_mechanism_digest == [0; 32]
        {
            return Err(ActuationInterlockError::ZeroDigest);
        }
        if self.source_realization_id == self.target_realization_id {
            return Err(ActuationInterlockError::SourceEqualsTarget);
        }
        let expected = ActuationInterlockSubjectId(hash_subject(
            self.lane,
            self.attempt_id,
            self.subject_id,
            self.source_realization_id,
            self.target_realization_id,
            self.backend_id,
            self.backend_implementation_digest,
            self.backend_generation,
            self.current_authority_commitment_id,
            self.current_verifier_commitment_id,
            self.journal_anchor_id,
            self.trusted_epoch_id,
            self.enforcement_profile_id,
            self.enforcement_boundary_digest,
            self.one_use_mechanism_digest,
            self.enforcement_profile_generation,
        ));
        if expected != self.subject_record_id {
            return Err(ActuationInterlockError::SubjectIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ActuationInterlockSubjectId { self.subject_record_id }
    pub fn lane(&self) -> ActuationExecutionLaneV1 { self.lane }
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.attempt_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.source_realization_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn current_authority_commitment_id(&self) -> QualifiedCurrentTransitionAuthorityExecutionCommitmentId {
        self.current_authority_commitment_id
    }
    pub fn current_verifier_commitment_id(&self) -> QualifiedCurrentVerifierExecutionCommitmentId {
        self.current_verifier_commitment_id
    }
    pub fn journal_anchor_id(&self) -> QualifiedExecutionJournalAnchorId { self.journal_anchor_id }
    pub fn trusted_epoch_id(&self) -> QualifiedTrustedCommitEpochId { self.trusted_epoch_id }
    pub fn enforcement_profile_id(&self) -> ActuationEnforcementBoundaryProfileId {
        self.enforcement_profile_id
    }
}

/// Transportable backend/resource decision over one exact actuation subject. This is
/// not a physical permit. A future adapter must authenticate this canonical payload
/// and prove resource-bound fencing enforcement before the decision can authorize I/O.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationInterlockClaimV1 {
    schema_version: String,
    subject_id: ActuationInterlockSubjectId,
    fencing_generation: u64,
    predecessor_claim_id: Option<ActuationInterlockClaimId>,
    disposition: ActuationInterlockDispositionV1,
    freshness_challenge_digest: [u8; 32],
    observed_at_unix_ms: u64,
    raw_evidence_digest: [u8; 32],
    claim_id: ActuationInterlockClaimId,
}

impl ActuationInterlockClaimV1 {
    pub fn initial(
        subject: &ActuationInterlockSubjectV1,
        disposition: ActuationInterlockDispositionV1,
        freshness_challenge_digest: [u8; 32],
        observed_at_unix_ms: u64,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, ActuationInterlockError> {
        subject.validate()?;
        Self::new_common(
            subject.id(),
            1,
            None,
            disposition,
            freshness_challenge_digest,
            observed_at_unix_ms,
            raw_evidence_digest,
        )
    }

    pub fn successor(
        subject: &ActuationInterlockSubjectV1,
        previous: &ActuationInterlockClaimV1,
        disposition: ActuationInterlockDispositionV1,
        freshness_challenge_digest: [u8; 32],
        observed_at_unix_ms: u64,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, ActuationInterlockError> {
        subject.validate()?;
        previous.validate()?;
        if previous.subject_id != subject.id() {
            return Err(ActuationInterlockError::SubjectMismatch);
        }
        if previous.disposition.is_deny() && disposition != previous.disposition {
            return Err(ActuationInterlockError::TerminalDenyChanged);
        }
        if previous.freshness_challenge_digest == freshness_challenge_digest {
            return Err(ActuationInterlockError::ChallengeReplay);
        }
        if observed_at_unix_ms < previous.observed_at_unix_ms {
            return Err(ActuationInterlockError::ObservationTimeRollback);
        }
        let fencing_generation = previous
            .fencing_generation
            .checked_add(1)
            .ok_or(ActuationInterlockError::FencingGenerationOverflow)?;
        Self::new_common(
            subject.id(),
            fencing_generation,
            Some(previous.id()),
            disposition,
            freshness_challenge_digest,
            observed_at_unix_ms,
            raw_evidence_digest,
        )
    }

    fn new_common(
        subject_id: ActuationInterlockSubjectId,
        fencing_generation: u64,
        predecessor_claim_id: Option<ActuationInterlockClaimId>,
        disposition: ActuationInterlockDispositionV1,
        freshness_challenge_digest: [u8; 32],
        observed_at_unix_ms: u64,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, ActuationInterlockError> {
        if fencing_generation == 0 || observed_at_unix_ms == 0 {
            return Err(ActuationInterlockError::ZeroGenerationOrTime);
        }
        if freshness_challenge_digest == [0; 32] || raw_evidence_digest == [0; 32] {
            return Err(ActuationInterlockError::ZeroDigest);
        }
        if fencing_generation == 1 && predecessor_claim_id.is_some() {
            return Err(ActuationInterlockError::InvalidInitialPredecessor);
        }
        if fencing_generation > 1 && predecessor_claim_id.is_none() {
            return Err(ActuationInterlockError::MissingPredecessor);
        }
        let claim_id = ActuationInterlockClaimId(hash_claim(
            subject_id,
            fencing_generation,
            predecessor_claim_id,
            disposition,
            freshness_challenge_digest,
            observed_at_unix_ms,
            raw_evidence_digest,
        ));
        Ok(Self {
            schema_version: ACTUATION_INTERLOCK_CLAIM_SCHEMA_V1.to_owned(),
            subject_id,
            fencing_generation,
            predecessor_claim_id,
            disposition,
            freshness_challenge_digest,
            observed_at_unix_ms,
            raw_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), ActuationInterlockError> {
        if self.schema_version != ACTUATION_INTERLOCK_CLAIM_SCHEMA_V1 {
            return Err(ActuationInterlockError::UnsupportedClaimSchema(
                self.schema_version.clone(),
            ));
        }
        if self.fencing_generation == 0 || self.observed_at_unix_ms == 0 {
            return Err(ActuationInterlockError::ZeroGenerationOrTime);
        }
        if self.freshness_challenge_digest == [0; 32] || self.raw_evidence_digest == [0; 32] {
            return Err(ActuationInterlockError::ZeroDigest);
        }
        if self.fencing_generation == 1 && self.predecessor_claim_id.is_some() {
            return Err(ActuationInterlockError::InvalidInitialPredecessor);
        }
        if self.fencing_generation > 1 && self.predecessor_claim_id.is_none() {
            return Err(ActuationInterlockError::MissingPredecessor);
        }
        let expected = ActuationInterlockClaimId(hash_claim(
            self.subject_id,
            self.fencing_generation,
            self.predecessor_claim_id,
            self.disposition,
            self.freshness_challenge_digest,
            self.observed_at_unix_ms,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(ActuationInterlockError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ActuationInterlockClaimId { self.claim_id }
    pub fn subject_id(&self) -> ActuationInterlockSubjectId { self.subject_id }
    pub fn fencing_generation(&self) -> u64 { self.fencing_generation }
    pub fn predecessor_claim_id(&self) -> Option<ActuationInterlockClaimId> {
        self.predecessor_claim_id
    }
    pub fn disposition(&self) -> ActuationInterlockDispositionV1 { self.disposition }
    pub fn freshness_challenge_digest(&self) -> [u8; 32] { self.freshness_challenge_digest }
    pub fn observed_at_unix_ms(&self) -> u64 { self.observed_at_unix_ms }
}

/// Canonical bytes for a future backend/resource attestation adapter. No Serde
/// representation is trusted as an authentication payload.
pub fn canonical_actuation_interlock_claim_bytes(
    claim: &ActuationInterlockClaimV1,
) -> Result<Vec<u8>, ActuationInterlockError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(384);
    out.extend_from_slice(WIRE_DOMAIN);
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(&claim.fencing_generation.to_le_bytes());
    encode_optional_id(&mut out, claim.predecessor_claim_id.map(|id| *id.as_bytes()));
    encode_disposition(&mut out, claim.disposition);
    out.extend_from_slice(&claim.freshness_challenge_digest);
    out.extend_from_slice(&claim.observed_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.raw_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_actuation_interlock_claim_digest(
    claim: &ActuationInterlockClaimV1,
) -> Result<[u8; 32], ActuationInterlockError> {
    Ok(*blake3::hash(&canonical_actuation_interlock_claim_bytes(claim)?).as_bytes())
}

/// Non-Serde, non-authoritative proof that one claim is canonically bound to one
/// exact subject, challenge and predecessor chain. It proves protocol coherence only.
#[derive(Debug, Clone)]
pub struct BoundActuationInterlockDecisionV1 {
    bound_id: BoundActuationInterlockDecisionId,
    subject_id: ActuationInterlockSubjectId,
    claim_id: ActuationInterlockClaimId,
    fencing_generation: u64,
    disposition: ActuationInterlockDispositionV1,
}

impl BoundActuationInterlockDecisionV1 {
    pub fn bind(
        subject: &ActuationInterlockSubjectV1,
        claim: &ActuationInterlockClaimV1,
        expected_freshness_challenge: [u8; 32],
        previous: Option<&ActuationInterlockClaimV1>,
    ) -> Result<Self, ActuationInterlockError> {
        subject.validate()?;
        claim.validate()?;
        if expected_freshness_challenge == [0; 32]
            || claim.freshness_challenge_digest != expected_freshness_challenge
        {
            return Err(ActuationInterlockError::ChallengeMismatch);
        }
        if claim.subject_id != subject.id() {
            return Err(ActuationInterlockError::SubjectMismatch);
        }
        validate_progression(previous, claim)?;
        let bound_id = BoundActuationInterlockDecisionId(domain_hash_parts(
            BOUND_DOMAIN,
            &[
                subject.id().as_bytes(),
                claim.id().as_bytes(),
                &expected_freshness_challenge,
            ],
        ));
        Ok(Self {
            bound_id,
            subject_id: subject.id(),
            claim_id: claim.id(),
            fencing_generation: claim.fencing_generation,
            disposition: claim.disposition,
        })
    }

    pub fn id(&self) -> BoundActuationInterlockDecisionId { self.bound_id }
    pub fn subject_id(&self) -> ActuationInterlockSubjectId { self.subject_id }
    pub fn claim_id(&self) -> ActuationInterlockClaimId { self.claim_id }
    pub fn fencing_generation(&self) -> u64 { self.fencing_generation }
    pub fn disposition(&self) -> ActuationInterlockDispositionV1 { self.disposition }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ActuationInterlockError {
    #[error(transparent)]
    Execution(#[from] crate::execution_capability::ExecutionCapabilityError),
    #[error("unsupported actuation enforcement profile schema: {0}")]
    UnsupportedProfileSchema(String),
    #[error("unsupported actuation interlock subject schema: {0}")]
    UnsupportedSubjectSchema(String),
    #[error("unsupported actuation interlock claim schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds the text bound")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("text is not canonical")]
    NonCanonicalText,
    #[error("actuation interlock digest must be non-zero")]
    ZeroDigest,
    #[error("actuation interlock generation/time must be non-zero")]
    ZeroGenerationOrTime,
    #[error("actuation enforcement profile identity mismatch")]
    ProfileIdentityMismatch,
    #[error("durable execution world differs from actuation subject")]
    ExecutionWorldMismatch,
    #[error("backend implementation differs from actuation enforcement boundary")]
    BackendBoundaryMismatch,
    #[error("actuation source realization equals target realization")]
    SourceEqualsTarget,
    #[error("actuation interlock subject identity mismatch")]
    SubjectIdentityMismatch,
    #[error("actuation interlock claim identity mismatch")]
    ClaimIdentityMismatch,
    #[error("actuation interlock claim belongs to a different subject")]
    SubjectMismatch,
    #[error("actuation interlock freshness challenge mismatch")]
    ChallengeMismatch,
    #[error("actuation interlock freshness challenge was replayed")]
    ChallengeReplay,
    #[error("initial actuation fence must not name a predecessor")]
    InvalidInitialPredecessor,
    #[error("non-initial actuation fence must name its predecessor")]
    MissingPredecessor,
    #[error("actuation fencing generation overflow")]
    FencingGenerationOverflow,
    #[error("actuation fencing generation/predecessor is not the exact successor")]
    FencingProgressionMismatch,
    #[error("actuation observation time rolled backward")]
    ObservationTimeRollback,
    #[error("a denied actuation attempt cannot change disposition")]
    TerminalDenyChanged,
}

fn validate_progression(
    previous: Option<&ActuationInterlockClaimV1>,
    next: &ActuationInterlockClaimV1,
) -> Result<(), ActuationInterlockError> {
    match previous {
        None => {
            if next.fencing_generation != 1 || next.predecessor_claim_id.is_some() {
                return Err(ActuationInterlockError::FencingProgressionMismatch);
            }
        }
        Some(previous) => {
            previous.validate()?;
            if previous.subject_id != next.subject_id {
                return Err(ActuationInterlockError::SubjectMismatch);
            }
            let expected = previous
                .fencing_generation
                .checked_add(1)
                .ok_or(ActuationInterlockError::FencingGenerationOverflow)?;
            if next.fencing_generation != expected
                || next.predecessor_claim_id != Some(previous.id())
            {
                return Err(ActuationInterlockError::FencingProgressionMismatch);
            }
            if next.freshness_challenge_digest == previous.freshness_challenge_digest {
                return Err(ActuationInterlockError::ChallengeReplay);
            }
            if next.observed_at_unix_ms < previous.observed_at_unix_ms {
                return Err(ActuationInterlockError::ObservationTimeRollback);
            }
            if previous.disposition.is_deny() && next.disposition != previous.disposition {
                return Err(ActuationInterlockError::TerminalDenyChanged);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn hash_profile(
    boundary_name: &str,
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    boundary_implementation_digest: [u8; 32],
    one_use_mechanism_digest: [u8; 32],
    profile_generation: u64,
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(PROFILE_DOMAIN);
    hash_len_prefixed(&mut h, boundary_name.as_bytes());
    h.update(backend_id.as_bytes());
    h.update(&backend_implementation_digest);
    h.update(&backend_generation.to_le_bytes());
    h.update(&boundary_implementation_digest);
    h.update(&one_use_mechanism_digest);
    h.update(&profile_generation.to_le_bytes());
    *h.finalize().as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn hash_subject(
    lane: ActuationExecutionLaneV1,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    current_authority_commitment_id: QualifiedCurrentTransitionAuthorityExecutionCommitmentId,
    current_verifier_commitment_id: QualifiedCurrentVerifierExecutionCommitmentId,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    enforcement_profile_id: ActuationEnforcementBoundaryProfileId,
    enforcement_boundary_digest: [u8; 32],
    one_use_mechanism_digest: [u8; 32],
    enforcement_profile_generation: u64,
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(SUBJECT_DOMAIN);
    h.update(&[lane.tag()]);
    h.update(attempt_id.as_bytes());
    h.update(subject_id.as_bytes());
    h.update(source_realization_id.as_bytes());
    h.update(target_realization_id.as_bytes());
    h.update(backend_id.as_bytes());
    h.update(&backend_implementation_digest);
    h.update(&backend_generation.to_le_bytes());
    h.update(current_authority_commitment_id.as_bytes());
    h.update(current_verifier_commitment_id.as_bytes());
    h.update(journal_anchor_id.as_bytes());
    h.update(trusted_epoch_id.as_bytes());
    h.update(enforcement_profile_id.as_bytes());
    h.update(&enforcement_boundary_digest);
    h.update(&one_use_mechanism_digest);
    h.update(&enforcement_profile_generation.to_le_bytes());
    *h.finalize().as_bytes()
}

fn hash_claim(
    subject_id: ActuationInterlockSubjectId,
    fencing_generation: u64,
    predecessor_claim_id: Option<ActuationInterlockClaimId>,
    disposition: ActuationInterlockDispositionV1,
    freshness_challenge_digest: [u8; 32],
    observed_at_unix_ms: u64,
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(CLAIM_DOMAIN);
    h.update(subject_id.as_bytes());
    h.update(&fencing_generation.to_le_bytes());
    hash_optional_id(&mut h, predecessor_claim_id.map(|id| *id.as_bytes()));
    hash_disposition(&mut h, disposition);
    h.update(&freshness_challenge_digest);
    h.update(&observed_at_unix_ms.to_le_bytes());
    h.update(&raw_evidence_digest);
    *h.finalize().as_bytes()
}

fn hash_disposition(h: &mut blake3::Hasher, disposition: ActuationInterlockDispositionV1) {
    h.update(&[disposition.tag()]);
    if let ActuationInterlockDispositionV1::Deny { reason } = disposition {
        h.update(&[reason.tag()]);
    }
}

fn encode_disposition(out: &mut Vec<u8>, disposition: ActuationInterlockDispositionV1) {
    out.push(disposition.tag());
    if let ActuationInterlockDispositionV1::Deny { reason } = disposition {
        out.push(reason.tag());
    }
}

fn hash_optional_id(h: &mut blake3::Hasher, id: Option<[u8; 32]>) {
    match id {
        Some(id) => { h.update(&[1]); h.update(&id); }
        None => { h.update(&[0]); }
    }
}

fn encode_optional_id(out: &mut Vec<u8>, id: Option<[u8; 32]>) {
    match id {
        Some(id) => { out.push(1); out.extend_from_slice(&id); }
        None => out.push(0),
    }
}

fn hash_len_prefixed(h: &mut blake3::Hasher, bytes: &[u8]) {
    h.update(&(bytes.len() as u64).to_le_bytes());
    h.update(bytes);
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

fn checked_text(field: &'static str, value: String) -> Result<String, ActuationInterlockError> {
    let value = value.trim().to_owned();
    if value.is_empty() {
        return Err(ActuationInterlockError::BlankText { field });
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(ActuationInterlockError::TextTooLong { field });
    }
    if value.chars().any(char::is_control) {
        return Err(ActuationInterlockError::ControlCharacters { field });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_domains_are_distinct() {
        assert_ne!(PROFILE_DOMAIN, SUBJECT_DOMAIN);
        assert_ne!(SUBJECT_DOMAIN, CLAIM_DOMAIN);
        assert_ne!(CLAIM_DOMAIN, WIRE_DOMAIN);
        assert_ne!(WIRE_DOMAIN, BOUND_DOMAIN);
    }

    #[test]
    fn deny_disposition_is_terminal_for_one_attempt() {
        assert!(ActuationInterlockDispositionV1::Deny {
            reason: ActuationDenyReasonV1::EmergencyStop,
        }
        .is_deny());
        assert!(!ActuationInterlockDispositionV1::Permit.is_deny());
    }
}
