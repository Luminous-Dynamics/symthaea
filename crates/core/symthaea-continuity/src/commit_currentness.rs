// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact local currentness immediately before a destructive continuity commit.
//!
//! A qualified continuity witness proves that one exact subject/contract/target world
//! satisfied its verification obligations. It does not prove that the local machine
//! or device still matches that target when a later distributed transition is about
//! to commit.
//!
//! Core theorem:
//!
//! `QualifiedContinuityWitness != CurrentLocalState != CommitEligibility != ExecutionAuthority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed_qualification::{
    DistributedCurrentStateDigest, QualifiedDistributedTransitionWitnessId,
    QualifiedDistributedTransitionWitnessV1,
};
use crate::distributed_state::DistributedStateContextId;
use crate::scope::ContinuitySubjectId;
use crate::subject_witness::{
    SubjectBoundQualifiedContinuityWitnessId, SubjectBoundQualifiedContinuityWitnessV1,
    SubjectWitnessBindingError,
};
use crate::verifier::{VerificationAdmissionError, VerifierProfileId, VerifierProfileV1};
use crate::witness::TargetRealizationId;

pub const LOCAL_COMMIT_CURRENTNESS_POLICY_SCHEMA_V1: &str =
    "symthaea-continuity-local-commit-currentness-policy-v1";
pub const LOCAL_COMMIT_STATE_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-local-commit-state-claim-v1";

const POLICY_DOMAIN: &[u8] = b"symthaea.continuity.local-commit-currentness-policy.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.local-commit-state-claim.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-local-commit-state.v1\0";
const QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-local-commit-currentness.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LocalCommitCurrentnessPolicyId([u8; 32]);

impl LocalCommitCurrentnessPolicyId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LocalCommitStateClaimId([u8; 32]);

impl LocalCommitStateClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedLocalCommitStateEvidenceId([u8; 32]);

impl AuthenticatedLocalCommitStateEvidenceId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedLocalCommitCurrentnessId([u8; 32]);

impl QualifiedLocalCommitCurrentnessId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Which exact verifier snapshot may establish local state immediately before commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalCommitCurrentnessPolicyV1 {
    schema_version: String,
    policy_generation: u64,
    verifier_profile_id: VerifierProfileId,
    max_age_ms: u64,
    max_future_skew_ms: u64,
    policy_id: LocalCommitCurrentnessPolicyId,
}

impl LocalCommitCurrentnessPolicyV1 {
    pub fn new(
        policy_generation: u64,
        verifier_profile: &VerifierProfileV1,
        max_age_ms: u64,
        max_future_skew_ms: u64,
    ) -> Result<Self, LocalCommitCurrentnessError> {
        verifier_profile.validate()?;
        if policy_generation == 0 {
            return Err(LocalCommitCurrentnessError::ZeroPolicyGeneration);
        }
        if max_age_ms == 0 {
            return Err(LocalCommitCurrentnessError::ZeroMaxAge);
        }
        let verifier_profile_id = verifier_profile.id();
        let policy_id = LocalCommitCurrentnessPolicyId(hash_policy(
            policy_generation,
            verifier_profile_id,
            max_age_ms,
            max_future_skew_ms,
        ));
        Ok(Self {
            schema_version: LOCAL_COMMIT_CURRENTNESS_POLICY_SCHEMA_V1.to_owned(),
            policy_generation,
            verifier_profile_id,
            max_age_ms,
            max_future_skew_ms,
            policy_id,
        })
    }

    pub fn validate(&self) -> Result<(), LocalCommitCurrentnessError> {
        if self.schema_version != LOCAL_COMMIT_CURRENTNESS_POLICY_SCHEMA_V1 {
            return Err(LocalCommitCurrentnessError::UnsupportedPolicySchema(
                self.schema_version.clone(),
            ));
        }
        if self.policy_generation == 0 {
            return Err(LocalCommitCurrentnessError::ZeroPolicyGeneration);
        }
        if self.max_age_ms == 0 {
            return Err(LocalCommitCurrentnessError::ZeroMaxAge);
        }
        let expected = LocalCommitCurrentnessPolicyId(hash_policy(
            self.policy_generation,
            self.verifier_profile_id,
            self.max_age_ms,
            self.max_future_skew_ms,
        ));
        if expected != self.policy_id {
            return Err(LocalCommitCurrentnessError::PolicyIdentityMismatch);
        }
        Ok(())
    }

    pub fn validate_against_profile(
        &self,
        profile: &VerifierProfileV1,
    ) -> Result<ValidatedLocalCommitCurrentnessPolicyV1, LocalCommitCurrentnessError> {
        self.validate()?;
        profile.validate()?;
        if profile.id() != self.verifier_profile_id {
            return Err(LocalCommitCurrentnessError::VerifierProfileMismatch);
        }
        Ok(ValidatedLocalCommitCurrentnessPolicyV1 {
            inner: self.clone(),
            profile: profile.clone(),
        })
    }

    pub fn id(&self) -> LocalCommitCurrentnessPolicyId {
        self.policy_id
    }

    pub fn policy_generation(&self) -> u64 {
        self.policy_generation
    }

    pub fn verifier_profile_id(&self) -> VerifierProfileId {
        self.verifier_profile_id
    }

    pub fn max_age_ms(&self) -> u64 {
        self.max_age_ms
    }

    pub fn max_future_skew_ms(&self) -> u64 {
        self.max_future_skew_ms
    }
}

/// Non-Serde policy rebound to one exact provisioned verifier profile.
#[derive(Debug, Clone)]
pub struct ValidatedLocalCommitCurrentnessPolicyV1 {
    inner: LocalCommitCurrentnessPolicyV1,
    profile: VerifierProfileV1,
}

impl ValidatedLocalCommitCurrentnessPolicyV1 {
    pub fn id(&self) -> LocalCommitCurrentnessPolicyId {
        self.inner.id()
    }

    pub fn policy_generation(&self) -> u64 {
        self.inner.policy_generation()
    }

    pub fn verifier_profile_id(&self) -> VerifierProfileId {
        self.inner.verifier_profile_id()
    }

    pub fn max_age_ms(&self) -> u64 {
        self.inner.max_age_ms()
    }

    pub fn max_future_skew_ms(&self) -> u64 {
        self.inner.max_future_skew_ms()
    }

    pub fn verifier_root_epoch(&self) -> u64 {
        self.profile.root_epoch()
    }

    pub fn as_raw(&self) -> &LocalCommitCurrentnessPolicyV1 {
        &self.inner
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalCommitObservationOutcomeV1 {
    MatchesQualifiedTarget,
    Drifted,
    Unknown,
}

impl LocalCommitObservationOutcomeV1 {
    fn tag(self) -> u8 {
        match self {
            Self::MatchesQualifiedTarget => 1,
            Self::Drifted => 2,
            Self::Unknown => 3,
        }
    }
}

/// Transportable local state claim. It is not trusted merely because it says
/// `MatchesQualifiedTarget`, and it deliberately cannot choose an evidence class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalCommitStateClaimV1 {
    schema_version: String,
    subject_witness_id: SubjectBoundQualifiedContinuityWitnessId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_witness_id: QualifiedDistributedTransitionWitnessId,
    distributed_context_id: DistributedStateContextId,
    distributed_current_state_digest: DistributedCurrentStateDigest,
    verifier_profile_id: VerifierProfileId,
    observed_at_unix_ms: u64,
    outcome: LocalCommitObservationOutcomeV1,
    observed_local_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
    claim_id: LocalCommitStateClaimId,
}

impl LocalCommitStateClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        local_witness: &SubjectBoundQualifiedContinuityWitnessV1,
        distributed_witness: &QualifiedDistributedTransitionWitnessV1,
        verifier_profile_id: VerifierProfileId,
        observed_at_unix_ms: u64,
        outcome: LocalCommitObservationOutcomeV1,
        observed_local_state_digest: Option<[u8; 32]>,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, LocalCommitCurrentnessError> {
        local_witness.validate()?;
        require_local_candidate(local_witness, distributed_witness)?;
        validate_observation_material(
            observed_at_unix_ms,
            outcome,
            observed_local_state_digest,
            raw_evidence_digest,
        )?;

        let subject_witness_id = local_witness.id();
        let subject_id = local_witness.subject_id();
        let target_realization_id = local_witness.target_realization_id();
        let distributed_witness_id = distributed_witness.id();
        let distributed_context_id = distributed_witness.context_id();
        let distributed_current_state_digest = distributed_witness.current_state_digest();
        let claim_id = LocalCommitStateClaimId(hash_claim(
            subject_witness_id,
            subject_id,
            target_realization_id,
            distributed_witness_id,
            distributed_context_id,
            distributed_current_state_digest,
            verifier_profile_id,
            observed_at_unix_ms,
            outcome,
            observed_local_state_digest,
            raw_evidence_digest,
        ));

        Ok(Self {
            schema_version: LOCAL_COMMIT_STATE_CLAIM_SCHEMA_V1.to_owned(),
            subject_witness_id,
            subject_id,
            target_realization_id,
            distributed_witness_id,
            distributed_context_id,
            distributed_current_state_digest,
            verifier_profile_id,
            observed_at_unix_ms,
            outcome,
            observed_local_state_digest,
            raw_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), LocalCommitCurrentnessError> {
        if self.schema_version != LOCAL_COMMIT_STATE_CLAIM_SCHEMA_V1 {
            return Err(LocalCommitCurrentnessError::UnsupportedClaimSchema(
                self.schema_version.clone(),
            ));
        }
        validate_observation_material(
            self.observed_at_unix_ms,
            self.outcome,
            self.observed_local_state_digest,
            self.raw_evidence_digest,
        )?;
        let expected = LocalCommitStateClaimId(hash_claim(
            self.subject_witness_id,
            self.subject_id,
            self.target_realization_id,
            self.distributed_witness_id,
            self.distributed_context_id,
            self.distributed_current_state_digest,
            self.verifier_profile_id,
            self.observed_at_unix_ms,
            self.outcome,
            self.observed_local_state_digest,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(LocalCommitCurrentnessError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> LocalCommitStateClaimId {
        self.claim_id
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyCheckedLocalCommitStateEvidenceV1 {
    claim: LocalCommitStateClaimV1,
    profile: VerifierProfileV1,
}

pub(crate) fn policy_check_local_commit_state_claim(
    local_witness: &SubjectBoundQualifiedContinuityWitnessV1,
    distributed_witness: &QualifiedDistributedTransitionWitnessV1,
    policy: &ValidatedLocalCommitCurrentnessPolicyV1,
    profile: &VerifierProfileV1,
    claim: LocalCommitStateClaimV1,
) -> Result<PolicyCheckedLocalCommitStateEvidenceV1, LocalCommitCurrentnessError> {
    local_witness.validate()?;
    profile.validate()?;
    claim.validate()?;
    require_local_candidate(local_witness, distributed_witness)?;
    if policy.verifier_profile_id() != profile.id() || claim.verifier_profile_id != profile.id() {
        return Err(LocalCommitCurrentnessError::VerifierProfileMismatch);
    }
    if claim.subject_witness_id != local_witness.id()
        || claim.subject_id != local_witness.subject_id()
        || claim.target_realization_id != local_witness.target_realization_id()
    {
        return Err(LocalCommitCurrentnessError::LocalWitnessMismatch);
    }
    if claim.distributed_witness_id != distributed_witness.id()
        || claim.distributed_context_id != distributed_witness.context_id()
        || claim.distributed_current_state_digest != distributed_witness.current_state_digest()
    {
        return Err(LocalCommitCurrentnessError::DistributedWitnessMismatch);
    }
    Ok(PolicyCheckedLocalCommitStateEvidenceV1 {
        claim,
        profile: profile.clone(),
    })
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedLocalCommitStateEvidenceV1 {
    checked: PolicyCheckedLocalCommitStateEvidenceV1,
    evidence_id: AuthenticatedLocalCommitStateEvidenceId,
}

impl AuthenticatedLocalCommitStateEvidenceV1 {
    fn id(&self) -> AuthenticatedLocalCommitStateEvidenceId {
        self.evidence_id
    }

    fn profile_id(&self) -> VerifierProfileId {
        self.checked.profile.id()
    }

    fn root_epoch(&self) -> u64 {
        self.checked.profile.root_epoch()
    }

    fn observed_at_unix_ms(&self) -> u64 {
        self.checked.claim.observed_at_unix_ms
    }

    fn outcome(&self) -> LocalCommitObservationOutcomeV1 {
        self.checked.claim.outcome
    }

    fn observed_local_state_digest(&self) -> Option<[u8; 32]> {
        self.checked.claim.observed_local_state_digest
    }

    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        checked: PolicyCheckedLocalCommitStateEvidenceV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, LocalCommitCurrentnessError> {
        if authentication_evidence_digest == [0; 32] {
            return Err(LocalCommitCurrentnessError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedLocalCommitStateEvidenceId(domain_hash_parts(
            AUTH_DOMAIN,
            &[
                checked.claim.id().as_bytes(),
                checked.profile.id().as_bytes(),
                &authentication_evidence_digest,
            ],
        ));
        Ok(Self { checked, evidence_id })
    }
}

/// Non-Serde proof that the exact local subject still matches the already-qualified
/// target in the exact distributed world at one exact commit-time evaluation.
#[derive(Debug, Clone)]
pub struct QualifiedLocalCommitCurrentnessV1 {
    currentness_id: QualifiedLocalCommitCurrentnessId,
    subject_witness_id: SubjectBoundQualifiedContinuityWitnessId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_witness_id: QualifiedDistributedTransitionWitnessId,
    distributed_context_id: DistributedStateContextId,
    distributed_current_state_digest: DistributedCurrentStateDigest,
    policy_id: LocalCommitCurrentnessPolicyId,
    policy_generation: u64,
    verifier_profile_id: VerifierProfileId,
    verifier_root_epoch: u64,
    evidence_id: AuthenticatedLocalCommitStateEvidenceId,
    observed_local_state_digest: [u8; 32],
    qualified_at_unix_ms: u64,
}

impl QualifiedLocalCommitCurrentnessV1 {
    pub fn id(&self) -> QualifiedLocalCommitCurrentnessId {
        self.currentness_id
    }

    pub fn subject_witness_id(&self) -> SubjectBoundQualifiedContinuityWitnessId {
        self.subject_witness_id
    }

    pub fn subject_id(&self) -> ContinuitySubjectId {
        self.subject_id
    }

    pub fn target_realization_id(&self) -> TargetRealizationId {
        self.target_realization_id
    }

    pub fn distributed_witness_id(&self) -> QualifiedDistributedTransitionWitnessId {
        self.distributed_witness_id
    }

    pub fn distributed_context_id(&self) -> DistributedStateContextId {
        self.distributed_context_id
    }

    pub fn distributed_current_state_digest(&self) -> DistributedCurrentStateDigest {
        self.distributed_current_state_digest
    }

    pub fn policy_id(&self) -> LocalCommitCurrentnessPolicyId {
        self.policy_id
    }

    pub fn policy_generation(&self) -> u64 {
        self.policy_generation
    }

    pub fn verifier_profile_id(&self) -> VerifierProfileId {
        self.verifier_profile_id
    }

    pub fn verifier_root_epoch(&self) -> u64 {
        self.verifier_root_epoch
    }

    pub fn evidence_id(&self) -> AuthenticatedLocalCommitStateEvidenceId {
        self.evidence_id
    }

    pub fn observed_local_state_digest(&self) -> [u8; 32] {
        self.observed_local_state_digest
    }

    pub fn qualified_at_unix_ms(&self) -> u64 {
        self.qualified_at_unix_ms
    }
}

pub(crate) fn qualify_local_commit_currentness(
    local_witness: &SubjectBoundQualifiedContinuityWitnessV1,
    distributed_witness: &QualifiedDistributedTransitionWitnessV1,
    policy: &ValidatedLocalCommitCurrentnessPolicyV1,
    evidence: &AuthenticatedLocalCommitStateEvidenceV1,
    qualified_at_unix_ms: u64,
) -> Result<QualifiedLocalCommitCurrentnessV1, LocalCommitCurrentnessError> {
    if qualified_at_unix_ms == 0 {
        return Err(LocalCommitCurrentnessError::ZeroQualificationTime);
    }
    local_witness.validate()?;
    require_local_candidate(local_witness, distributed_witness)?;
    if evidence.checked.claim.subject_witness_id != local_witness.id()
        || evidence.checked.claim.subject_id != local_witness.subject_id()
        || evidence.checked.claim.target_realization_id != local_witness.target_realization_id()
    {
        return Err(LocalCommitCurrentnessError::LocalWitnessMismatch);
    }
    if evidence.checked.claim.distributed_witness_id != distributed_witness.id()
        || evidence.checked.claim.distributed_context_id != distributed_witness.context_id()
        || evidence.checked.claim.distributed_current_state_digest != distributed_witness.current_state_digest()
    {
        return Err(LocalCommitCurrentnessError::DistributedWitnessMismatch);
    }
    if evidence.profile_id() != policy.verifier_profile_id() {
        return Err(LocalCommitCurrentnessError::VerifierProfileMismatch);
    }
    if evidence.outcome() != LocalCommitObservationOutcomeV1::MatchesQualifiedTarget {
        return Err(LocalCommitCurrentnessError::LocalStateNotCurrent);
    }
    check_freshness(
        evidence.observed_at_unix_ms(),
        qualified_at_unix_ms,
        policy.max_age_ms(),
        policy.max_future_skew_ms(),
    )?;
    let observed_local_state_digest = evidence
        .observed_local_state_digest()
        .ok_or(LocalCommitCurrentnessError::MissingObservedStateDigest)?;

    let currentness_id = QualifiedLocalCommitCurrentnessId(domain_hash_parts(
        QUALIFIED_DOMAIN,
        &[
            local_witness.id().as_bytes(),
            distributed_witness.id().as_bytes(),
            distributed_witness.current_state_digest().as_bytes(),
            policy.id().as_bytes(),
            evidence.id().as_bytes(),
            evidence.profile_id().as_bytes(),
            &evidence.root_epoch().to_le_bytes(),
            &observed_local_state_digest,
            &qualified_at_unix_ms.to_le_bytes(),
        ],
    ));

    Ok(QualifiedLocalCommitCurrentnessV1 {
        currentness_id,
        subject_witness_id: local_witness.id(),
        subject_id: local_witness.subject_id(),
        target_realization_id: local_witness.target_realization_id(),
        distributed_witness_id: distributed_witness.id(),
        distributed_context_id: distributed_witness.context_id(),
        distributed_current_state_digest: distributed_witness.current_state_digest(),
        policy_id: policy.id(),
        policy_generation: policy.policy_generation(),
        verifier_profile_id: evidence.profile_id(),
        verifier_root_epoch: evidence.root_epoch(),
        evidence_id: evidence.id(),
        observed_local_state_digest,
        qualified_at_unix_ms,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LocalCommitCurrentnessError {
    #[error(transparent)]
    SubjectWitness(#[from] SubjectWitnessBindingError),
    #[error(transparent)]
    Verification(#[from] VerificationAdmissionError),
    #[error("unsupported local commit currentness policy schema: {0}")]
    UnsupportedPolicySchema(String),
    #[error("unsupported local commit state claim schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("local commit currentness policy generation must be non-zero")]
    ZeroPolicyGeneration,
    #[error("local commit currentness max age must be non-zero")]
    ZeroMaxAge,
    #[error("local commit currentness policy identity mismatch")]
    PolicyIdentityMismatch,
    #[error("local commit verifier profile does not match the pinned policy profile")]
    VerifierProfileMismatch,
    #[error("local continuity subject is not in the exact distributed transition candidate set")]
    LocalSubjectNotCandidate,
    #[error("local commit observation time must be non-zero")]
    ZeroObservationTime,
    #[error("local commit raw evidence digest must be non-zero")]
    ZeroRawEvidenceDigest,
    #[error("matching/drifted local state requires a non-zero observed state digest")]
    MissingObservedStateDigest,
    #[error("UNKNOWN local state must not fabricate an observed state digest")]
    UnknownCarriesObservedState,
    #[error("stored local commit state claim identity mismatch")]
    ClaimIdentityMismatch,
    #[error("local commit claim does not bind the exact local qualified witness")]
    LocalWitnessMismatch,
    #[error("local commit claim does not bind the exact distributed qualified witness/current state")]
    DistributedWitnessMismatch,
    #[error("local commit authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("local commit qualification time must be non-zero")]
    ZeroQualificationTime,
    #[error("local state is not currently observed to match the qualified target")]
    LocalStateNotCurrent,
    #[error("local commit observation is stale: age {age_ms} ms > {allowed_ms} ms")]
    StaleObservation { age_ms: u64, allowed_ms: u64 },
    #[error("local commit observation is too far in the future: skew {skew_ms} ms > {allowed_ms} ms")]
    ObservationFromFuture { skew_ms: u64, allowed_ms: u64 },
}

fn require_local_candidate(
    local_witness: &SubjectBoundQualifiedContinuityWitnessV1,
    distributed_witness: &QualifiedDistributedTransitionWitnessV1,
) -> Result<(), LocalCommitCurrentnessError> {
    if distributed_witness
        .candidate_subject_ids()
        .binary_search(&local_witness.subject_id())
        .is_err()
    {
        return Err(LocalCommitCurrentnessError::LocalSubjectNotCandidate);
    }
    Ok(())
}

fn validate_observation_material(
    observed_at_unix_ms: u64,
    outcome: LocalCommitObservationOutcomeV1,
    observed_local_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> Result<(), LocalCommitCurrentnessError> {
    if observed_at_unix_ms == 0 {
        return Err(LocalCommitCurrentnessError::ZeroObservationTime);
    }
    if raw_evidence_digest == [0; 32] {
        return Err(LocalCommitCurrentnessError::ZeroRawEvidenceDigest);
    }
    match outcome {
        LocalCommitObservationOutcomeV1::MatchesQualifiedTarget
        | LocalCommitObservationOutcomeV1::Drifted => {
            if !matches!(observed_local_state_digest, Some(digest) if digest != [0; 32]) {
                return Err(LocalCommitCurrentnessError::MissingObservedStateDigest);
            }
        }
        LocalCommitObservationOutcomeV1::Unknown => {
            if observed_local_state_digest.is_some() {
                return Err(LocalCommitCurrentnessError::UnknownCarriesObservedState);
            }
        }
    }
    Ok(())
}

fn check_freshness(
    observed_at_unix_ms: u64,
    qualified_at_unix_ms: u64,
    max_age_ms: u64,
    max_future_skew_ms: u64,
) -> Result<(), LocalCommitCurrentnessError> {
    if observed_at_unix_ms > qualified_at_unix_ms {
        let skew = observed_at_unix_ms - qualified_at_unix_ms;
        if skew > max_future_skew_ms {
            return Err(LocalCommitCurrentnessError::ObservationFromFuture {
                skew_ms: skew,
                allowed_ms: max_future_skew_ms,
            });
        }
        return Ok(());
    }
    let age = qualified_at_unix_ms - observed_at_unix_ms;
    if age > max_age_ms {
        return Err(LocalCommitCurrentnessError::StaleObservation {
            age_ms: age,
            allowed_ms: max_age_ms,
        });
    }
    Ok(())
}

fn hash_policy(
    generation: u64,
    verifier_profile_id: VerifierProfileId,
    max_age_ms: u64,
    max_future_skew_ms: u64,
) -> [u8; 32] {
    domain_hash_parts(
        POLICY_DOMAIN,
        &[
            &generation.to_le_bytes(),
            verifier_profile_id.as_bytes(),
            &max_age_ms.to_le_bytes(),
            &max_future_skew_ms.to_le_bytes(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    subject_witness_id: SubjectBoundQualifiedContinuityWitnessId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_witness_id: QualifiedDistributedTransitionWitnessId,
    distributed_context_id: DistributedStateContextId,
    distributed_current_state_digest: DistributedCurrentStateDigest,
    verifier_profile_id: VerifierProfileId,
    observed_at_unix_ms: u64,
    outcome: LocalCommitObservationOutcomeV1,
    observed_local_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let presence = [u8::from(observed_local_state_digest.is_some())];
    let observed_state = observed_local_state_digest.unwrap_or([0; 32]);
    let outcome_tag = [outcome.tag()];
    domain_hash_parts(
        CLAIM_DOMAIN,
        &[
            subject_witness_id.as_bytes(),
            subject_id.as_bytes(),
            target_realization_id.as_bytes(),
            distributed_witness_id.as_bytes(),
            distributed_context_id.as_bytes(),
            distributed_current_state_digest.as_bytes(),
            verifier_profile_id.as_bytes(),
            &observed_at_unix_ms.to_le_bytes(),
            &outcome_tag,
            &presence,
            &observed_state,
            &raw_evidence_digest,
        ],
    )
}

fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    for part in parts {
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::witness::EvidenceClass;

    fn profile(seed: u8) -> VerifierProfileV1 {
        VerifierProfileV1::new(
            format!("local-commit-{seed}"),
            [seed; 32],
            seed as u64,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    #[test]
    fn policy_identity_binds_profile_generation_and_freshness() {
        let p = profile(1);
        let a = LocalCommitCurrentnessPolicyV1::new(1, &p, 500, 25).unwrap();
        let b = LocalCommitCurrentnessPolicyV1::new(2, &p, 500, 25).unwrap();
        let c = LocalCommitCurrentnessPolicyV1::new(1, &p, 501, 25).unwrap();
        assert_ne!(a.id(), b.id());
        assert_ne!(a.id(), c.id());
        a.validate_against_profile(&p).unwrap();
    }

    #[test]
    fn same_name_different_verifier_root_cannot_rebind_policy() {
        let trusted = VerifierProfileV1::new(
            "local-commit",
            [1; 32],
            1,
            EvidenceClass::HardwareVerified,
        )
        .unwrap();
        let substituted = VerifierProfileV1::new(
            "local-commit",
            [2; 32],
            1,
            EvidenceClass::HardwareVerified,
        )
        .unwrap();
        let policy = LocalCommitCurrentnessPolicyV1::new(1, &trusted, 500, 25).unwrap();
        assert_eq!(
            policy.validate_against_profile(&substituted).unwrap_err(),
            LocalCommitCurrentnessError::VerifierProfileMismatch
        );
    }

    #[test]
    fn stale_and_future_observations_fail_closed() {
        assert_eq!(
            check_freshness(1_000, 2_001, 1_000, 10).unwrap_err(),
            LocalCommitCurrentnessError::StaleObservation {
                age_ms: 1_001,
                allowed_ms: 1_000,
            }
        );
        assert_eq!(
            check_freshness(2_011, 2_000, 1_000, 10).unwrap_err(),
            LocalCommitCurrentnessError::ObservationFromFuture {
                skew_ms: 11,
                allowed_ms: 10,
            }
        );
    }

    #[test]
    fn unknown_cannot_carry_fake_state_digest() {
        assert_eq!(
            validate_observation_material(
                1,
                LocalCommitObservationOutcomeV1::Unknown,
                Some([1; 32]),
                [2; 32],
            )
            .unwrap_err(),
            LocalCommitCurrentnessError::UnknownCarriesObservedState
        );
    }
}
