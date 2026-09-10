// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Post-execution health observation after exact target identity is established.
//!
//! `ExpectedTargetObserved != Healthy != LastKnownGood`.
//!
//! Health remains adapter/profile-defined so the continuity kernel does not pretend a
//! database, hypervisor, switch, storage array, and workstation share one universal
//! health test. The kernel binds the exact approved health-profile digest and verifier.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed_state::DistributedStateContextId;
use crate::execution_capability::ExecutionAttemptId;
use crate::post_execution_observation::{
    PostExecutionObservedStateV1, QualifiedPostExecutionObservationId,
    QualifiedPostExecutionObservationV1,
};
use crate::scope::ContinuitySubjectId;
use crate::verifier::{VerificationAdmissionError, VerifierProfileId, VerifierProfileV1};
use crate::witness::TargetRealizationId;

pub const POST_EXECUTION_HEALTH_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-post-execution-health-claim-v1";
pub const POST_EXECUTION_HEALTH_AUTH_PURPOSE: &str =
    "symthaea.continuity.post-execution-health.v1";

const POLICY_DOMAIN: &[u8] = b"symthaea.continuity.post-execution-health-policy.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.post-execution-health-claim.v1\0";
const CLAIM_WIRE_DOMAIN: &[u8] = b"symthaea.continuity.post-execution-health-wire.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-post-execution-health.v1\0";
const QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-post-execution-health.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PostExecutionHealthPolicyId([u8; 32]);
impl PostExecutionHealthPolicyId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PostExecutionHealthClaimId([u8; 32]);
impl PostExecutionHealthClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedPostExecutionHealthId([u8; 32]);
impl AuthenticatedPostExecutionHealthId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedPostExecutionHealthId([u8; 32]);
impl QualifiedPostExecutionHealthId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Constructor-owned health policy bound to one exact verifier root and one exact
/// adapter-defined health profile digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostExecutionHealthPolicyV1 {
    policy_id: PostExecutionHealthPolicyId,
    policy_generation: u64,
    verifier_profile_id: VerifierProfileId,
    verifier_root_epoch: u64,
    health_profile_digest: [u8; 32],
    maximum_observation_age_ms: u64,
    maximum_future_skew_ms: u64,
}

impl PostExecutionHealthPolicyV1 {
    pub fn new(
        profile: &VerifierProfileV1,
        policy_generation: u64,
        health_profile_digest: [u8; 32],
        maximum_observation_age_ms: u64,
        maximum_future_skew_ms: u64,
    ) -> Result<Self, PostExecutionHealthError> {
        profile.validate()?;
        if policy_generation == 0 {
            return Err(PostExecutionHealthError::ZeroPolicyGeneration);
        }
        if health_profile_digest == [0; 32] {
            return Err(PostExecutionHealthError::ZeroHealthProfileDigest);
        }
        if maximum_observation_age_ms == 0 {
            return Err(PostExecutionHealthError::ZeroMaximumObservationAge);
        }
        let policy_id = PostExecutionHealthPolicyId(hash_policy(
            profile.id(),
            profile.root_epoch(),
            policy_generation,
            health_profile_digest,
            maximum_observation_age_ms,
            maximum_future_skew_ms,
        ));
        Ok(Self {
            policy_id,
            policy_generation,
            verifier_profile_id: profile.id(),
            verifier_root_epoch: profile.root_epoch(),
            health_profile_digest,
            maximum_observation_age_ms,
            maximum_future_skew_ms,
        })
    }

    pub fn validate_against_profile(
        &self,
        profile: &VerifierProfileV1,
    ) -> Result<(), PostExecutionHealthError> {
        profile.validate()?;
        if profile.id() != self.verifier_profile_id || profile.root_epoch() != self.verifier_root_epoch {
            return Err(PostExecutionHealthError::VerifierProfileMismatch);
        }
        let expected = PostExecutionHealthPolicyId(hash_policy(
            self.verifier_profile_id,
            self.verifier_root_epoch,
            self.policy_generation,
            self.health_profile_digest,
            self.maximum_observation_age_ms,
            self.maximum_future_skew_ms,
        ));
        if expected != self.policy_id {
            return Err(PostExecutionHealthError::PolicyIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> PostExecutionHealthPolicyId {
        self.policy_id
    }

    pub fn health_profile_digest(&self) -> [u8; 32] {
        self.health_profile_digest
    }

    pub fn verifier_profile_id(&self) -> VerifierProfileId {
        self.verifier_profile_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PostExecutionHealthOutcomeV1 {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl PostExecutionHealthOutcomeV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Healthy => 1,
            Self::Degraded => 2,
            Self::Unhealthy => 3,
            Self::Unknown => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostExecutionHealthClaimV1 {
    schema_version: String,
    post_observation_id: QualifiedPostExecutionObservationId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    verifier_profile_id: VerifierProfileId,
    health_profile_digest: [u8; 32],
    observed_at_unix_ms: u64,
    outcome: PostExecutionHealthOutcomeV1,
    health_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
    claim_id: PostExecutionHealthClaimId,
}

impl PostExecutionHealthClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        post_observation: &QualifiedPostExecutionObservationV1,
        verifier_profile_id: VerifierProfileId,
        health_profile_digest: [u8; 32],
        observed_at_unix_ms: u64,
        outcome: PostExecutionHealthOutcomeV1,
        health_state_digest: Option<[u8; 32]>,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, PostExecutionHealthError> {
        require_expected_target(post_observation)?;
        validate_health_material(
            health_profile_digest,
            observed_at_unix_ms,
            outcome,
            health_state_digest,
            raw_evidence_digest,
        )?;
        let claim_id = PostExecutionHealthClaimId(hash_claim(
            post_observation.id(),
            post_observation.attempt_id(),
            post_observation.subject_id(),
            post_observation.expected_target_realization_id(),
            post_observation.distributed_context_id(),
            verifier_profile_id,
            health_profile_digest,
            observed_at_unix_ms,
            outcome,
            health_state_digest,
            raw_evidence_digest,
        ));
        Ok(Self {
            schema_version: POST_EXECUTION_HEALTH_CLAIM_SCHEMA_V1.to_owned(),
            post_observation_id: post_observation.id(),
            attempt_id: post_observation.attempt_id(),
            subject_id: post_observation.subject_id(),
            target_realization_id: post_observation.expected_target_realization_id(),
            distributed_context_id: post_observation.distributed_context_id(),
            verifier_profile_id,
            health_profile_digest,
            observed_at_unix_ms,
            outcome,
            health_state_digest,
            raw_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), PostExecutionHealthError> {
        if self.schema_version != POST_EXECUTION_HEALTH_CLAIM_SCHEMA_V1 {
            return Err(PostExecutionHealthError::UnsupportedClaimSchema(
                self.schema_version.clone(),
            ));
        }
        validate_health_material(
            self.health_profile_digest,
            self.observed_at_unix_ms,
            self.outcome,
            self.health_state_digest,
            self.raw_evidence_digest,
        )?;
        let expected = PostExecutionHealthClaimId(hash_claim(
            self.post_observation_id,
            self.attempt_id,
            self.subject_id,
            self.target_realization_id,
            self.distributed_context_id,
            self.verifier_profile_id,
            self.health_profile_digest,
            self.observed_at_unix_ms,
            self.outcome,
            self.health_state_digest,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(PostExecutionHealthError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> PostExecutionHealthClaimId {
        self.claim_id
    }
}

/// Stable authentication bytes for a Xenia/signature/attestation adapter.
/// Serde encoding is deliberately not part of the trust contract.
pub fn canonical_post_execution_health_claim_bytes(
    claim: &PostExecutionHealthClaimV1,
) -> Result<Vec<u8>, PostExecutionHealthError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(448);
    out.extend_from_slice(CLAIM_WIRE_DOMAIN);
    out.extend_from_slice(claim.post_observation_id.as_bytes());
    out.extend_from_slice(claim.attempt_id.as_bytes());
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.target_realization_id.as_bytes());
    out.extend_from_slice(claim.distributed_context_id.as_bytes());
    out.extend_from_slice(claim.verifier_profile_id.as_bytes());
    out.extend_from_slice(&claim.health_profile_digest);
    out.extend_from_slice(&claim.observed_at_unix_ms.to_le_bytes());
    out.push(claim.outcome.tag());
    match claim.health_state_digest {
        Some(digest) => {
            out.push(1);
            out.extend_from_slice(&digest);
        }
        None => out.push(0),
    }
    out.extend_from_slice(&claim.raw_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_post_execution_health_claim_digest(
    claim: &PostExecutionHealthClaimV1,
) -> Result<[u8; 32], PostExecutionHealthError> {
    Ok(*blake3::hash(&canonical_post_execution_health_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyCheckedPostExecutionHealthV1 {
    claim: PostExecutionHealthClaimV1,
    policy: PostExecutionHealthPolicyV1,
}

pub(crate) fn policy_check_post_execution_health(
    post_observation: &QualifiedPostExecutionObservationV1,
    profile: &VerifierProfileV1,
    policy: &PostExecutionHealthPolicyV1,
    claim: PostExecutionHealthClaimV1,
) -> Result<PolicyCheckedPostExecutionHealthV1, PostExecutionHealthError> {
    require_expected_target(post_observation)?;
    policy.validate_against_profile(profile)?;
    claim.validate()?;
    if claim.post_observation_id != post_observation.id()
        || claim.attempt_id != post_observation.attempt_id()
        || claim.subject_id != post_observation.subject_id()
        || claim.target_realization_id != post_observation.expected_target_realization_id()
        || claim.distributed_context_id != post_observation.distributed_context_id()
    {
        return Err(PostExecutionHealthError::PostObservationContextMismatch);
    }
    if claim.verifier_profile_id != profile.id()
        || claim.verifier_profile_id != policy.verifier_profile_id()
    {
        return Err(PostExecutionHealthError::VerifierProfileMismatch);
    }
    if claim.health_profile_digest != policy.health_profile_digest() {
        return Err(PostExecutionHealthError::HealthProfileMismatch);
    }
    Ok(PolicyCheckedPostExecutionHealthV1 {
        claim,
        policy: policy.clone(),
    })
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedPostExecutionHealthV1 {
    checked: PolicyCheckedPostExecutionHealthV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedPostExecutionHealthId,
}

impl AuthenticatedPostExecutionHealthV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        checked: PolicyCheckedPostExecutionHealthV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, PostExecutionHealthError> {
        if authentication_evidence_digest == [0; 32] {
            return Err(PostExecutionHealthError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedPostExecutionHealthId(domain_hash_parts(
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

/// Non-Serde exact health observation. `Healthy` here still does not imply LKG.
#[derive(Debug, Clone)]
pub struct QualifiedPostExecutionHealthV1 {
    qualified_id: QualifiedPostExecutionHealthId,
    post_observation_id: QualifiedPostExecutionObservationId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    policy_id: PostExecutionHealthPolicyId,
    policy_generation: u64,
    verifier_profile_id: VerifierProfileId,
    verifier_root_epoch: u64,
    authenticated_evidence_id: AuthenticatedPostExecutionHealthId,
    health_profile_digest: [u8; 32],
    observed_at_unix_ms: u64,
    qualified_at_unix_ms: u64,
    outcome: PostExecutionHealthOutcomeV1,
    health_state_digest: Option<[u8; 32]>,
}

impl QualifiedPostExecutionHealthV1 {
    pub fn id(&self) -> QualifiedPostExecutionHealthId {
        self.qualified_id
    }

    pub fn post_observation_id(&self) -> QualifiedPostExecutionObservationId {
        self.post_observation_id
    }

    pub fn attempt_id(&self) -> ExecutionAttemptId {
        self.attempt_id
    }

    pub fn subject_id(&self) -> ContinuitySubjectId {
        self.subject_id
    }

    pub fn target_realization_id(&self) -> TargetRealizationId {
        self.target_realization_id
    }

    pub fn distributed_context_id(&self) -> DistributedStateContextId {
        self.distributed_context_id
    }

    pub fn health_profile_digest(&self) -> [u8; 32] {
        self.health_profile_digest
    }

    pub fn outcome(&self) -> PostExecutionHealthOutcomeV1 {
        self.outcome
    }

    pub fn qualified_at_unix_ms(&self) -> u64 {
        self.qualified_at_unix_ms
    }
}

pub(crate) fn qualify_post_execution_health(
    evidence: &AuthenticatedPostExecutionHealthV1,
    qualified_at_unix_ms: u64,
) -> Result<QualifiedPostExecutionHealthV1, PostExecutionHealthError> {
    if qualified_at_unix_ms == 0 {
        return Err(PostExecutionHealthError::ZeroQualificationTime);
    }
    let claim = &evidence.checked.claim;
    let policy = &evidence.checked.policy;
    check_freshness(
        claim.observed_at_unix_ms,
        qualified_at_unix_ms,
        policy.maximum_observation_age_ms,
        policy.maximum_future_skew_ms,
    )?;
    let qualified_id = QualifiedPostExecutionHealthId(domain_hash_parts(
        QUALIFIED_DOMAIN,
        &[
            claim.id().as_bytes(),
            policy.id().as_bytes(),
            evidence.evidence_id.as_bytes(),
            &qualified_at_unix_ms.to_le_bytes(),
        ],
    ));
    Ok(QualifiedPostExecutionHealthV1 {
        qualified_id,
        post_observation_id: claim.post_observation_id,
        attempt_id: claim.attempt_id,
        subject_id: claim.subject_id,
        target_realization_id: claim.target_realization_id,
        distributed_context_id: claim.distributed_context_id,
        policy_id: policy.id(),
        policy_generation: policy.policy_generation,
        verifier_profile_id: policy.verifier_profile_id,
        verifier_root_epoch: policy.verifier_root_epoch,
        authenticated_evidence_id: evidence.evidence_id,
        health_profile_digest: claim.health_profile_digest,
        observed_at_unix_ms: claim.observed_at_unix_ms,
        qualified_at_unix_ms,
        outcome: claim.outcome,
        health_state_digest: claim.health_state_digest,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PostExecutionHealthError {
    #[error(transparent)]
    Verification(#[from] VerificationAdmissionError),
    #[error("post-execution health policy generation must be non-zero")]
    ZeroPolicyGeneration,
    #[error("post-execution health profile digest must be non-zero")]
    ZeroHealthProfileDigest,
    #[error("post-execution health maximum observation age must be non-zero")]
    ZeroMaximumObservationAge,
    #[error("post-execution health verifier profile mismatch")]
    VerifierProfileMismatch,
    #[error("post-execution health policy identity mismatch")]
    PolicyIdentityMismatch,
    #[error("unsupported post-execution health claim schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("health observation may only follow an independently observed exact expected target")]
    ExpectedTargetNotEstablished,
    #[error("health observation time must be non-zero")]
    ZeroObservationTime,
    #[error("post-execution health raw evidence digest must be non-zero")]
    ZeroRawEvidenceDigest,
    #[error("known health outcome requires a non-zero health-state digest")]
    MissingHealthStateDigest,
    #[error("UNKNOWN health must not fabricate a health-state digest")]
    UnexpectedHealthStateDigest,
    #[error("stored post-execution health claim identity mismatch")]
    ClaimIdentityMismatch,
    #[error("post-execution health claim belongs to a different physical observation")]
    PostObservationContextMismatch,
    #[error("post-execution health claim uses a different health profile than policy")]
    HealthProfileMismatch,
    #[error("post-execution health authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("post-execution health qualification time must be non-zero")]
    ZeroQualificationTime,
    #[error("post-execution health evidence is stale: age {age_ms} ms > {allowed_ms} ms")]
    StaleObservation { age_ms: u64, allowed_ms: u64 },
    #[error("post-execution health evidence is too far in the future: skew {skew_ms} ms > {allowed_ms} ms")]
    ObservationFromFuture { skew_ms: u64, allowed_ms: u64 },
}

fn require_expected_target(
    post_observation: &QualifiedPostExecutionObservationV1,
) -> Result<(), PostExecutionHealthError> {
    if post_observation.observed_state() != PostExecutionObservedStateV1::ExpectedTargetObserved
        || post_observation.observed_realization_id()
            != Some(post_observation.expected_target_realization_id())
    {
        return Err(PostExecutionHealthError::ExpectedTargetNotEstablished);
    }
    Ok(())
}

fn validate_health_material(
    health_profile_digest: [u8; 32],
    observed_at_unix_ms: u64,
    outcome: PostExecutionHealthOutcomeV1,
    health_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> Result<(), PostExecutionHealthError> {
    if health_profile_digest == [0; 32] {
        return Err(PostExecutionHealthError::ZeroHealthProfileDigest);
    }
    if observed_at_unix_ms == 0 {
        return Err(PostExecutionHealthError::ZeroObservationTime);
    }
    if raw_evidence_digest == [0; 32] {
        return Err(PostExecutionHealthError::ZeroRawEvidenceDigest);
    }
    match outcome {
        PostExecutionHealthOutcomeV1::Unknown => {
            if health_state_digest.is_some() {
                return Err(PostExecutionHealthError::UnexpectedHealthStateDigest);
            }
        }
        PostExecutionHealthOutcomeV1::Healthy
        | PostExecutionHealthOutcomeV1::Degraded
        | PostExecutionHealthOutcomeV1::Unhealthy => {
            if health_state_digest.is_none() || health_state_digest == Some([0; 32]) {
                return Err(PostExecutionHealthError::MissingHealthStateDigest);
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
) -> Result<(), PostExecutionHealthError> {
    if observed_at_unix_ms > qualified_at_unix_ms {
        let skew = observed_at_unix_ms - qualified_at_unix_ms;
        if skew > maximum_future_skew_ms {
            return Err(PostExecutionHealthError::ObservationFromFuture {
                skew_ms: skew,
                allowed_ms: maximum_future_skew_ms,
            });
        }
        return Ok(());
    }
    let age = qualified_at_unix_ms - observed_at_unix_ms;
    if age > maximum_age_ms {
        return Err(PostExecutionHealthError::StaleObservation {
            age_ms: age,
            allowed_ms: maximum_age_ms,
        });
    }
    Ok(())
}

fn hash_policy(
    profile_id: VerifierProfileId,
    root_epoch: u64,
    generation: u64,
    health_profile_digest: [u8; 32],
    maximum_age_ms: u64,
    maximum_future_skew_ms: u64,
) -> [u8; 32] {
    domain_hash_parts(
        POLICY_DOMAIN,
        &[
            profile_id.as_bytes(),
            &root_epoch.to_le_bytes(),
            &generation.to_le_bytes(),
            &health_profile_digest,
            &maximum_age_ms.to_le_bytes(),
            &maximum_future_skew_ms.to_le_bytes(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    post_observation_id: QualifiedPostExecutionObservationId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    target_id: TargetRealizationId,
    context_id: DistributedStateContextId,
    profile_id: VerifierProfileId,
    health_profile_digest: [u8; 32],
    observed_at_unix_ms: u64,
    outcome: PostExecutionHealthOutcomeV1,
    health_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let outcome_tag = [outcome.tag()];
    let state_present = [u8::from(health_state_digest.is_some())];
    let state_digest = health_state_digest.unwrap_or([0; 32]);
    domain_hash_parts(
        CLAIM_DOMAIN,
        &[
            post_observation_id.as_bytes(),
            attempt_id.as_bytes(),
            subject_id.as_bytes(),
            target_id.as_bytes(),
            context_id.as_bytes(),
            profile_id.as_bytes(),
            &health_profile_digest,
            &observed_at_unix_ms.to_le_bytes(),
            &outcome_tag,
            &state_present,
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
    fn health_profile_is_identity_material() {
        let profile = VerifierProfileV1::new("health", [1; 32], 1, EvidenceClass::HardwareVerified)
            .unwrap();
        let a = PostExecutionHealthPolicyV1::new(&profile, 1, [2; 32], 5_000, 100).unwrap();
        let b = PostExecutionHealthPolicyV1::new(&profile, 1, [3; 32], 5_000, 100).unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn unknown_health_cannot_fabricate_state_digest() {
        assert_eq!(
            validate_health_material(
                [1; 32],
                1,
                PostExecutionHealthOutcomeV1::Unknown,
                Some([2; 32]),
                [3; 32],
            )
            .unwrap_err(),
            PostExecutionHealthError::UnexpectedHealthStateDigest
        );
    }

    #[test]
    fn known_health_requires_state_digest() {
        assert_eq!(
            validate_health_material(
                [1; 32],
                1,
                PostExecutionHealthOutcomeV1::Healthy,
                None,
                [3; 32],
            )
            .unwrap_err(),
            PostExecutionHealthError::MissingHealthStateDigest
        );
    }

    #[test]
    fn authentication_purpose_and_wire_domain_are_explicit() {
        assert_eq!(
            POST_EXECUTION_HEALTH_AUTH_PURPOSE,
            "symthaea.continuity.post-execution-health.v1"
        );
        assert_ne!(CLAIM_WIRE_DOMAIN, CLAIM_DOMAIN);
    }
}
