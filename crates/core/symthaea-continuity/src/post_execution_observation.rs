// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent physical-state observation after an execution attempt.
//!
//! Backend success is not physical truth. This layer lets a verifier independently
//! establish what exact realization is actually observable after an attempt, including
//! intent-only crash recovery where no backend receipt survived.
//!
//! `ExecutionReceipt != PostExecutionObservation != LastKnownGood`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed_state::DistributedStateContextId;
use crate::execution_capability::{ExecutionAttemptId, ExecutionAttemptIntentV1};
use crate::scope::ContinuitySubjectId;
use crate::verifier::{VerificationAdmissionError, VerifierProfileId, VerifierProfileV1};
use crate::witness::TargetRealizationId;

pub const POST_EXECUTION_OBSERVATION_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-post-execution-observation-claim-v1";

const POLICY_DOMAIN: &[u8] = b"symthaea.continuity.post-execution-policy.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.post-execution-observation.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-post-execution-observation.v1\0";
const QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-post-execution-observation.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PostExecutionObservationPolicyId([u8; 32]);
impl PostExecutionObservationPolicyId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PostExecutionObservationClaimId([u8; 32]);
impl PostExecutionObservationClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedPostExecutionObservationId([u8; 32]);
impl AuthenticatedPostExecutionObservationId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedPostExecutionObservationId([u8; 32]);
impl QualifiedPostExecutionObservationId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Exact post-execution verifier profile and freshness rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostExecutionObservationPolicyV1 {
    policy_id: PostExecutionObservationPolicyId,
    policy_generation: u64,
    verifier_profile_id: VerifierProfileId,
    verifier_root_epoch: u64,
    maximum_observation_age_ms: u64,
    maximum_future_skew_ms: u64,
}

impl PostExecutionObservationPolicyV1 {
    pub fn new(
        profile: &VerifierProfileV1,
        policy_generation: u64,
        maximum_observation_age_ms: u64,
        maximum_future_skew_ms: u64,
    ) -> Result<Self, PostExecutionObservationError> {
        profile.validate()?;
        if policy_generation == 0 {
            return Err(PostExecutionObservationError::ZeroPolicyGeneration);
        }
        if maximum_observation_age_ms == 0 {
            return Err(PostExecutionObservationError::ZeroMaximumObservationAge);
        }
        let policy_id = PostExecutionObservationPolicyId(hash_policy(
            profile.id(),
            profile.root_epoch(),
            policy_generation,
            maximum_observation_age_ms,
            maximum_future_skew_ms,
        ));
        Ok(Self {
            policy_id,
            policy_generation,
            verifier_profile_id: profile.id(),
            verifier_root_epoch: profile.root_epoch(),
            maximum_observation_age_ms,
            maximum_future_skew_ms,
        })
    }

    pub fn validate_against_profile(
        &self,
        profile: &VerifierProfileV1,
    ) -> Result<(), PostExecutionObservationError> {
        profile.validate()?;
        if profile.id() != self.verifier_profile_id || profile.root_epoch() != self.verifier_root_epoch {
            return Err(PostExecutionObservationError::VerifierProfileMismatch);
        }
        let expected = PostExecutionObservationPolicyId(hash_policy(
            self.verifier_profile_id,
            self.verifier_root_epoch,
            self.policy_generation,
            self.maximum_observation_age_ms,
            self.maximum_future_skew_ms,
        ));
        if expected != self.policy_id {
            return Err(PostExecutionObservationError::PolicyIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> PostExecutionObservationPolicyId {
        self.policy_id
    }

    pub fn policy_generation(&self) -> u64 {
        self.policy_generation
    }

    pub fn verifier_profile_id(&self) -> VerifierProfileId {
        self.verifier_profile_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PostExecutionObservedStateV1 {
    ExpectedTargetObserved,
    DifferentKnownRealization,
    Unreachable,
    Unknown,
}

impl PostExecutionObservedStateV1 {
    fn tag(self) -> u8 {
        match self {
            Self::ExpectedTargetObserved => 1,
            Self::DifferentKnownRealization => 2,
            Self::Unreachable => 3,
            Self::Unknown => 4,
        }
    }
}

/// Serializable, untrusted physical observation claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostExecutionObservationClaimV1 {
    schema_version: String,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    expected_target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    verifier_profile_id: VerifierProfileId,
    observed_at_unix_ms: u64,
    observed_state: PostExecutionObservedStateV1,
    observed_realization_id: Option<TargetRealizationId>,
    observed_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
    claim_id: PostExecutionObservationClaimId,
}

impl PostExecutionObservationClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        intent: &ExecutionAttemptIntentV1,
        verifier_profile_id: VerifierProfileId,
        observed_at_unix_ms: u64,
        observed_state: PostExecutionObservedStateV1,
        observed_realization_id: Option<TargetRealizationId>,
        observed_state_digest: Option<[u8; 32]>,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, PostExecutionObservationError> {
        intent.validate()?;
        validate_observation_material(
            intent.target_realization_id(),
            observed_at_unix_ms,
            observed_state,
            observed_realization_id,
            observed_state_digest,
            raw_evidence_digest,
        )?;
        let claim_id = PostExecutionObservationClaimId(hash_claim(
            intent.id(),
            intent.subject_id(),
            intent.target_realization_id(),
            intent.distributed_context_id(),
            verifier_profile_id,
            observed_at_unix_ms,
            observed_state,
            observed_realization_id,
            observed_state_digest,
            raw_evidence_digest,
        ));
        Ok(Self {
            schema_version: POST_EXECUTION_OBSERVATION_CLAIM_SCHEMA_V1.to_owned(),
            attempt_id: intent.id(),
            subject_id: intent.subject_id(),
            expected_target_realization_id: intent.target_realization_id(),
            distributed_context_id: intent.distributed_context_id(),
            verifier_profile_id,
            observed_at_unix_ms,
            observed_state,
            observed_realization_id,
            observed_state_digest,
            raw_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), PostExecutionObservationError> {
        if self.schema_version != POST_EXECUTION_OBSERVATION_CLAIM_SCHEMA_V1 {
            return Err(PostExecutionObservationError::UnsupportedClaimSchema(
                self.schema_version.clone(),
            ));
        }
        validate_observation_material(
            self.expected_target_realization_id,
            self.observed_at_unix_ms,
            self.observed_state,
            self.observed_realization_id,
            self.observed_state_digest,
            self.raw_evidence_digest,
        )?;
        let expected = PostExecutionObservationClaimId(hash_claim(
            self.attempt_id,
            self.subject_id,
            self.expected_target_realization_id,
            self.distributed_context_id,
            self.verifier_profile_id,
            self.observed_at_unix_ms,
            self.observed_state,
            self.observed_realization_id,
            self.observed_state_digest,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(PostExecutionObservationError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> PostExecutionObservationClaimId {
        self.claim_id
    }

    pub fn observed_state(&self) -> PostExecutionObservedStateV1 {
        self.observed_state
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyCheckedPostExecutionObservationV1 {
    claim: PostExecutionObservationClaimV1,
    policy: PostExecutionObservationPolicyV1,
}

pub(crate) fn policy_check_post_execution_observation(
    intent: &ExecutionAttemptIntentV1,
    profile: &VerifierProfileV1,
    policy: &PostExecutionObservationPolicyV1,
    claim: PostExecutionObservationClaimV1,
) -> Result<PolicyCheckedPostExecutionObservationV1, PostExecutionObservationError> {
    intent.validate()?;
    policy.validate_against_profile(profile)?;
    claim.validate()?;
    if claim.attempt_id != intent.id()
        || claim.subject_id != intent.subject_id()
        || claim.expected_target_realization_id != intent.target_realization_id()
        || claim.distributed_context_id != intent.distributed_context_id()
    {
        return Err(PostExecutionObservationError::IntentContextMismatch);
    }
    if claim.verifier_profile_id != profile.id()
        || claim.verifier_profile_id != policy.verifier_profile_id()
    {
        return Err(PostExecutionObservationError::VerifierProfileMismatch);
    }
    Ok(PolicyCheckedPostExecutionObservationV1 {
        claim,
        policy: policy.clone(),
    })
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedPostExecutionObservationV1 {
    checked: PolicyCheckedPostExecutionObservationV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedPostExecutionObservationId,
}

impl AuthenticatedPostExecutionObservationV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        checked: PolicyCheckedPostExecutionObservationV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, PostExecutionObservationError> {
        if authentication_evidence_digest == [0; 32] {
            return Err(PostExecutionObservationError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedPostExecutionObservationId(domain_hash_parts(
            AUTH_DOMAIN,
            &[
                checked.claim.id().as_bytes(),
                checked.policy.id().as_bytes(),
                checked.policy.verifier_profile_id().as_bytes(),
                &authentication_evidence_digest,
            ],
        ));
        Ok(Self {
            checked,
            authentication_evidence_digest,
            evidence_id,
        })
    }
}

/// Non-Serde exact physical-state observation accepted under one exact verifier root.
/// This is still not a health verdict, LKG promotion, or execution authority.
#[derive(Debug, Clone)]
pub struct QualifiedPostExecutionObservationV1 {
    qualified_id: QualifiedPostExecutionObservationId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    expected_target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    policy_id: PostExecutionObservationPolicyId,
    policy_generation: u64,
    verifier_profile_id: VerifierProfileId,
    verifier_root_epoch: u64,
    authenticated_evidence_id: AuthenticatedPostExecutionObservationId,
    observed_at_unix_ms: u64,
    qualified_at_unix_ms: u64,
    observed_state: PostExecutionObservedStateV1,
    observed_realization_id: Option<TargetRealizationId>,
    observed_state_digest: Option<[u8; 32]>,
}

impl QualifiedPostExecutionObservationV1 {
    pub fn id(&self) -> QualifiedPostExecutionObservationId {
        self.qualified_id
    }

    pub fn attempt_id(&self) -> ExecutionAttemptId {
        self.attempt_id
    }

    pub fn subject_id(&self) -> ContinuitySubjectId {
        self.subject_id
    }

    pub fn expected_target_realization_id(&self) -> TargetRealizationId {
        self.expected_target_realization_id
    }

    pub fn distributed_context_id(&self) -> DistributedStateContextId {
        self.distributed_context_id
    }

    pub fn observed_state(&self) -> PostExecutionObservedStateV1 {
        self.observed_state
    }

    pub fn observed_realization_id(&self) -> Option<TargetRealizationId> {
        self.observed_realization_id
    }

    pub fn observed_state_digest(&self) -> Option<[u8; 32]> {
        self.observed_state_digest
    }

    pub fn qualified_at_unix_ms(&self) -> u64 {
        self.qualified_at_unix_ms
    }
}

pub(crate) fn qualify_post_execution_observation(
    evidence: &AuthenticatedPostExecutionObservationV1,
    qualified_at_unix_ms: u64,
) -> Result<QualifiedPostExecutionObservationV1, PostExecutionObservationError> {
    if qualified_at_unix_ms == 0 {
        return Err(PostExecutionObservationError::ZeroQualificationTime);
    }
    let claim = &evidence.checked.claim;
    let policy = &evidence.checked.policy;
    check_freshness(
        claim.observed_at_unix_ms,
        qualified_at_unix_ms,
        policy.maximum_observation_age_ms,
        policy.maximum_future_skew_ms,
    )?;
    let qualified_id = QualifiedPostExecutionObservationId(domain_hash_parts(
        QUALIFIED_DOMAIN,
        &[
            claim.id().as_bytes(),
            policy.id().as_bytes(),
            policy.verifier_profile_id().as_bytes(),
            &policy.verifier_root_epoch.to_le_bytes(),
            evidence.evidence_id.as_bytes(),
            &qualified_at_unix_ms.to_le_bytes(),
        ],
    ));
    Ok(QualifiedPostExecutionObservationV1 {
        qualified_id,
        attempt_id: claim.attempt_id,
        subject_id: claim.subject_id,
        expected_target_realization_id: claim.expected_target_realization_id,
        distributed_context_id: claim.distributed_context_id,
        policy_id: policy.id(),
        policy_generation: policy.policy_generation,
        verifier_profile_id: policy.verifier_profile_id,
        verifier_root_epoch: policy.verifier_root_epoch,
        authenticated_evidence_id: evidence.evidence_id,
        observed_at_unix_ms: claim.observed_at_unix_ms,
        qualified_at_unix_ms,
        observed_state: claim.observed_state,
        observed_realization_id: claim.observed_realization_id,
        observed_state_digest: claim.observed_state_digest,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PostExecutionObservationError {
    #[error(transparent)]
    Verification(#[from] VerificationAdmissionError),
    #[error(transparent)]
    Execution(#[from] ExecutionCapabilityError),
    #[error("unsupported post-execution observation claim schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("post-execution observation policy generation must be non-zero")]
    ZeroPolicyGeneration,
    #[error("post-execution observation maximum age must be non-zero")]
    ZeroMaximumObservationAge,
    #[error("post-execution observation policy identity mismatch")]
    PolicyIdentityMismatch,
    #[error("post-execution verifier profile does not match exact policy root")]
    VerifierProfileMismatch,
    #[error("post-execution observation time must be non-zero")]
    ZeroObservationTime,
    #[error("post-execution raw evidence digest must be non-zero")]
    ZeroRawEvidenceDigest,
    #[error("known post-execution physical state requires a non-zero state digest")]
    MissingObservedStateDigest,
    #[error("unknown/unreachable state must not fabricate a physical state digest")]
    UnexpectedObservedStateDigest,
    #[error("expected-target observation must identify the exact expected realization")]
    ExpectedTargetIdentityMismatch,
    #[error("different-realization observation must name a realization different from the expected target")]
    DifferentRealizationIdentityInvalid,
    #[error("unknown/unreachable state must not claim a known realization")]
    UnexpectedObservedRealization,
    #[error("stored post-execution observation claim identity mismatch")]
    ClaimIdentityMismatch,
    #[error("post-execution observation belongs to a different execution intent")]
    IntentContextMismatch,
    #[error("post-execution authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("post-execution qualification time must be non-zero")]
    ZeroQualificationTime,
    #[error("post-execution evidence is stale: age {age_ms} ms > {allowed_ms} ms")]
    StaleObservation { age_ms: u64, allowed_ms: u64 },
    #[error("post-execution evidence is too far in the future: skew {skew_ms} ms > {allowed_ms} ms")]
    ObservationFromFuture { skew_ms: u64, allowed_ms: u64 },
}

fn validate_observation_material(
    expected_target: TargetRealizationId,
    observed_at_unix_ms: u64,
    observed_state: PostExecutionObservedStateV1,
    observed_realization_id: Option<TargetRealizationId>,
    observed_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> Result<(), PostExecutionObservationError> {
    if observed_at_unix_ms == 0 {
        return Err(PostExecutionObservationError::ZeroObservationTime);
    }
    if raw_evidence_digest == [0; 32] {
        return Err(PostExecutionObservationError::ZeroRawEvidenceDigest);
    }
    match observed_state {
        PostExecutionObservedStateV1::ExpectedTargetObserved => {
            if observed_realization_id != Some(expected_target) {
                return Err(PostExecutionObservationError::ExpectedTargetIdentityMismatch);
            }
            if observed_state_digest.is_none() || observed_state_digest == Some([0; 32]) {
                return Err(PostExecutionObservationError::MissingObservedStateDigest);
            }
        }
        PostExecutionObservedStateV1::DifferentKnownRealization => {
            let Some(realization) = observed_realization_id else {
                return Err(PostExecutionObservationError::DifferentRealizationIdentityInvalid);
            };
            if realization == expected_target {
                return Err(PostExecutionObservationError::DifferentRealizationIdentityInvalid);
            }
            if observed_state_digest.is_none() || observed_state_digest == Some([0; 32]) {
                return Err(PostExecutionObservationError::MissingObservedStateDigest);
            }
        }
        PostExecutionObservedStateV1::Unreachable | PostExecutionObservedStateV1::Unknown => {
            if observed_realization_id.is_some() {
                return Err(PostExecutionObservationError::UnexpectedObservedRealization);
            }
            if observed_state_digest.is_some() {
                return Err(PostExecutionObservationError::UnexpectedObservedStateDigest);
            }
        }
    }
    Ok(())
}

fn check_freshness(
    observed_at_unix_ms: u64,
    qualified_at_unix_ms: u64,
    maximum_age_ms: u64,
    maximum_future_skew_ms: u64,
) -> Result<(), PostExecutionObservationError> {
    if observed_at_unix_ms > qualified_at_unix_ms {
        let skew = observed_at_unix_ms - qualified_at_unix_ms;
        if skew > maximum_future_skew_ms {
            return Err(PostExecutionObservationError::ObservationFromFuture {
                skew_ms: skew,
                allowed_ms: maximum_future_skew_ms,
            });
        }
        return Ok(());
    }
    let age = qualified_at_unix_ms - observed_at_unix_ms;
    if age > maximum_age_ms {
        return Err(PostExecutionObservationError::StaleObservation {
            age_ms: age,
            allowed_ms: maximum_age_ms,
        });
    }
    Ok(())
}

fn hash_policy(
    profile_id: VerifierProfileId,
    root_epoch: u64,
    policy_generation: u64,
    maximum_age_ms: u64,
    maximum_future_skew_ms: u64,
) -> [u8; 32] {
    domain_hash_parts(
        POLICY_DOMAIN,
        &[
            profile_id.as_bytes(),
            &root_epoch.to_le_bytes(),
            &policy_generation.to_le_bytes(),
            &maximum_age_ms.to_le_bytes(),
            &maximum_future_skew_ms.to_le_bytes(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    expected_target: TargetRealizationId,
    context_id: DistributedStateContextId,
    profile_id: VerifierProfileId,
    observed_at_unix_ms: u64,
    observed_state: PostExecutionObservedStateV1,
    observed_realization_id: Option<TargetRealizationId>,
    observed_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let state_tag = [observed_state.tag()];
    let realization_present = [u8::from(observed_realization_id.is_some())];
    let state_digest_present = [u8::from(observed_state_digest.is_some())];
    let realization_bytes = observed_realization_id
        .map(|id| *id.as_bytes())
        .unwrap_or([0; 32]);
    let state_digest = observed_state_digest.unwrap_or([0; 32]);
    domain_hash_parts(
        CLAIM_DOMAIN,
        &[
            attempt_id.as_bytes(),
            subject_id.as_bytes(),
            expected_target.as_bytes(),
            context_id.as_bytes(),
            profile_id.as_bytes(),
            &observed_at_unix_ms.to_le_bytes(),
            &state_tag,
            &realization_present,
            &realization_bytes,
            &state_digest_present,
            &state_digest,
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

    #[test]
    fn verifier_root_rotation_changes_post_execution_policy_identity() {
        let a = VerifierProfileV1::new("post-exec", [1; 32], 1, EvidenceClass::HardwareVerified)
            .unwrap();
        let b = VerifierProfileV1::new("post-exec", [2; 32], 2, EvidenceClass::HardwareVerified)
            .unwrap();
        let pa = PostExecutionObservationPolicyV1::new(&a, 1, 5_000, 100).unwrap();
        let pb = PostExecutionObservationPolicyV1::new(&b, 1, 5_000, 100).unwrap();
        assert_ne!(pa.id(), pb.id());
    }

    #[test]
    fn unreachable_state_cannot_fabricate_observed_state_digest() {
        let target = TargetRealizationId::from_digest([9; 32]).unwrap();
        assert_eq!(
            validate_observation_material(
                target,
                1,
                PostExecutionObservedStateV1::Unreachable,
                None,
                Some([8; 32]),
                [7; 32],
            )
            .unwrap_err(),
            PostExecutionObservationError::UnexpectedObservedStateDigest
        );
    }

    #[test]
    fn different_realization_must_actually_differ() {
        let target = TargetRealizationId::from_digest([9; 32]).unwrap();
        assert_eq!(
            validate_observation_material(
                target,
                1,
                PostExecutionObservedStateV1::DifferentKnownRealization,
                Some(target),
                Some([8; 32]),
                [7; 32],
            )
            .unwrap_err(),
            PostExecutionObservationError::DifferentRealizationIdentityInvalid
        );
    }
}
