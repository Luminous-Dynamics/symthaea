// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact local healthy-state snapshots with explicit identity lineage.
//!
//! Normal post-execution target health and crash-reconciled source health answer the
//! same downstream question only after different identity proofs. This module keeps
//! those identity bases explicit while giving distributed-health composition one
//! common non-Serde healthy-local-state shape.
//!
//! `ObservedIdentity != HealthEvidence != QualifiedHealthyLocalSnapshot`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::crash_reconciliation::{
    CrashReconciliationClassificationV1, CrashReconciliationId,
    QualifiedCrashReconciliationV1,
};
use crate::distributed_state::DistributedStateContextId;
use crate::execution_capability::ExecutionAttemptId;
use crate::post_execution_health::{
    PostExecutionHealthOutcomeV1, QualifiedPostExecutionHealthId,
    QualifiedPostExecutionHealthV1,
};
use crate::scope::ContinuitySubjectId;
use crate::transition_lineage::KnownGoodBoundExecutionAttemptIntentV1;
use crate::verifier::{VerificationAdmissionError, VerifierProfileId, VerifierProfileV1};
use crate::witness::TargetRealizationId;

pub const CRASH_SOURCE_HEALTH_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-crash-source-health-claim-v1";
pub const CRASH_SOURCE_HEALTH_AUTH_PURPOSE: &str =
    "symthaea.continuity.crash-source-health.v1";

const SOURCE_POLICY_DOMAIN: &[u8] = b"symthaea.continuity.crash-source-health-policy.v1\0";
const SOURCE_CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.crash-source-health-claim.v1\0";
const SOURCE_CLAIM_WIRE_DOMAIN: &[u8] = b"symthaea.continuity.crash-source-health-wire.v1\0";
const SOURCE_AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-crash-source-health.v1\0";
const SOURCE_QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-crash-source-health.v1\0";
const SNAPSHOT_DOMAIN: &[u8] = b"symthaea.continuity.qualified-healthy-local-snapshot.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CrashSourceHealthPolicyId([u8; 32]);
impl CrashSourceHealthPolicyId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CrashSourceHealthClaimId([u8; 32]);
impl CrashSourceHealthClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthenticatedCrashSourceHealthId([u8; 32]);
impl AuthenticatedCrashSourceHealthId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedCrashSourceHealthId([u8; 32]);
impl QualifiedCrashSourceHealthId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedHealthyLocalSnapshotId([u8; 32]);
impl QualifiedHealthyLocalSnapshotId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Exact verifier root and adapter-defined health profile allowed to establish
/// health for a crash-reconciled source realization A.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrashSourceHealthPolicyV1 {
    policy_id: CrashSourceHealthPolicyId,
    policy_generation: u64,
    verifier_profile_id: VerifierProfileId,
    verifier_root_epoch: u64,
    health_profile_digest: [u8; 32],
    maximum_observation_age_ms: u64,
    maximum_future_skew_ms: u64,
}

impl CrashSourceHealthPolicyV1 {
    pub fn new(
        profile: &VerifierProfileV1,
        policy_generation: u64,
        health_profile_digest: [u8; 32],
        maximum_observation_age_ms: u64,
        maximum_future_skew_ms: u64,
    ) -> Result<Self, ExactLocalHealthError> {
        profile.validate()?;
        if policy_generation == 0 {
            return Err(ExactLocalHealthError::ZeroPolicyGeneration);
        }
        if health_profile_digest == [0; 32] {
            return Err(ExactLocalHealthError::ZeroHealthProfileDigest);
        }
        if maximum_observation_age_ms == 0 {
            return Err(ExactLocalHealthError::ZeroMaximumObservationAge);
        }
        let policy_id = CrashSourceHealthPolicyId(hash_policy(
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
    ) -> Result<(), ExactLocalHealthError> {
        profile.validate()?;
        if profile.id() != self.verifier_profile_id
            || profile.root_epoch() != self.verifier_root_epoch
        {
            return Err(ExactLocalHealthError::VerifierProfileMismatch);
        }
        let expected = CrashSourceHealthPolicyId(hash_policy(
            self.verifier_profile_id,
            self.verifier_root_epoch,
            self.policy_generation,
            self.health_profile_digest,
            self.maximum_observation_age_ms,
            self.maximum_future_skew_ms,
        ));
        if expected != self.policy_id {
            return Err(ExactLocalHealthError::PolicyIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> CrashSourceHealthPolicyId { self.policy_id }
    pub fn verifier_profile_id(&self) -> VerifierProfileId { self.verifier_profile_id }
    pub fn health_profile_digest(&self) -> [u8; 32] { self.health_profile_digest }
}

/// Transportable but untrusted health claim for exact source A after crash
/// reconciliation established A's physical identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrashSourceHealthClaimV1 {
    schema_version: String,
    reconciliation_id: CrashReconciliationId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    verifier_profile_id: VerifierProfileId,
    health_profile_digest: [u8; 32],
    observed_at_unix_ms: u64,
    outcome: PostExecutionHealthOutcomeV1,
    health_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
    claim_id: CrashSourceHealthClaimId,
}

impl CrashSourceHealthClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        reconciliation: &QualifiedCrashReconciliationV1,
        intent: &KnownGoodBoundExecutionAttemptIntentV1,
        verifier_profile_id: VerifierProfileId,
        health_profile_digest: [u8; 32],
        observed_at_unix_ms: u64,
        outcome: PostExecutionHealthOutcomeV1,
        health_state_digest: Option<[u8; 32]>,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, ExactLocalHealthError> {
        require_source_reconciliation(reconciliation, intent)?;
        if observed_at_unix_ms < reconciliation.record().reconciled_at_unix_ms() {
            return Err(ExactLocalHealthError::HealthPredatesIdentity);
        }
        validate_health_material(
            health_profile_digest,
            observed_at_unix_ms,
            outcome,
            health_state_digest,
            raw_evidence_digest,
        )?;
        let lineage = intent.lineage();
        let reconciliation_id = reconciliation.id();
        let attempt_id = intent.attempt_id();
        let subject_id = lineage.subject_id();
        let source_realization_id = lineage.source_realization_id();
        let distributed_context_id = lineage.distributed_context_id();
        let claim_id = CrashSourceHealthClaimId(hash_claim(
            reconciliation_id,
            attempt_id,
            subject_id,
            source_realization_id,
            distributed_context_id,
            verifier_profile_id,
            health_profile_digest,
            observed_at_unix_ms,
            outcome,
            health_state_digest,
            raw_evidence_digest,
        ));
        Ok(Self {
            schema_version: CRASH_SOURCE_HEALTH_CLAIM_SCHEMA_V1.to_owned(),
            reconciliation_id,
            attempt_id,
            subject_id,
            source_realization_id,
            distributed_context_id,
            verifier_profile_id,
            health_profile_digest,
            observed_at_unix_ms,
            outcome,
            health_state_digest,
            raw_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), ExactLocalHealthError> {
        if self.schema_version != CRASH_SOURCE_HEALTH_CLAIM_SCHEMA_V1 {
            return Err(ExactLocalHealthError::UnsupportedClaimSchema(
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
        let expected = CrashSourceHealthClaimId(hash_claim(
            self.reconciliation_id,
            self.attempt_id,
            self.subject_id,
            self.source_realization_id,
            self.distributed_context_id,
            self.verifier_profile_id,
            self.health_profile_digest,
            self.observed_at_unix_ms,
            self.outcome,
            self.health_state_digest,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(ExactLocalHealthError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> CrashSourceHealthClaimId { self.claim_id }
}

/// Stable authentication bytes for a Xenia/signature/attestation adapter.
/// Serde encoding is deliberately not part of the trust contract.
pub fn canonical_crash_source_health_claim_bytes(
    claim: &CrashSourceHealthClaimV1,
) -> Result<Vec<u8>, ExactLocalHealthError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(448);
    out.extend_from_slice(SOURCE_CLAIM_WIRE_DOMAIN);
    out.extend_from_slice(claim.reconciliation_id.as_bytes());
    out.extend_from_slice(claim.attempt_id.as_bytes());
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.source_realization_id.as_bytes());
    out.extend_from_slice(claim.distributed_context_id.as_bytes());
    out.extend_from_slice(claim.verifier_profile_id.as_bytes());
    out.extend_from_slice(&claim.health_profile_digest);
    out.extend_from_slice(&claim.observed_at_unix_ms.to_le_bytes());
    out.push(health_outcome_tag(claim.outcome));
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

pub fn canonical_crash_source_health_claim_digest(
    claim: &CrashSourceHealthClaimV1,
) -> Result<[u8; 32], ExactLocalHealthError> {
    Ok(*blake3::hash(&canonical_crash_source_health_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyCheckedCrashSourceHealthV1 {
    claim: CrashSourceHealthClaimV1,
    policy: CrashSourceHealthPolicyV1,
}

pub(crate) fn policy_check_crash_source_health(
    reconciliation: &QualifiedCrashReconciliationV1,
    intent: &KnownGoodBoundExecutionAttemptIntentV1,
    profile: &VerifierProfileV1,
    policy: &CrashSourceHealthPolicyV1,
    claim: CrashSourceHealthClaimV1,
) -> Result<PolicyCheckedCrashSourceHealthV1, ExactLocalHealthError> {
    require_source_reconciliation(reconciliation, intent)?;
    policy.validate_against_profile(profile)?;
    claim.validate()?;
    let lineage = intent.lineage();
    if claim.reconciliation_id != reconciliation.id()
        || claim.attempt_id != intent.attempt_id()
        || claim.subject_id != lineage.subject_id()
        || claim.source_realization_id != lineage.source_realization_id()
        || claim.distributed_context_id != lineage.distributed_context_id()
    {
        return Err(ExactLocalHealthError::ReconciliationContextMismatch);
    }
    if claim.verifier_profile_id != profile.id()
        || claim.verifier_profile_id != policy.verifier_profile_id()
    {
        return Err(ExactLocalHealthError::VerifierProfileMismatch);
    }
    if claim.health_profile_digest != policy.health_profile_digest() {
        return Err(ExactLocalHealthError::HealthProfileMismatch);
    }
    if claim.observed_at_unix_ms < reconciliation.record().reconciled_at_unix_ms() {
        return Err(ExactLocalHealthError::HealthPredatesIdentity);
    }
    Ok(PolicyCheckedCrashSourceHealthV1 {
        claim,
        policy: policy.clone(),
    })
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedCrashSourceHealthV1 {
    checked: PolicyCheckedCrashSourceHealthV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedCrashSourceHealthId,
}

impl AuthenticatedCrashSourceHealthV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        checked: PolicyCheckedCrashSourceHealthV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, ExactLocalHealthError> {
        if authentication_evidence_digest == [0; 32] {
            return Err(ExactLocalHealthError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedCrashSourceHealthId(domain_hash_parts(
            SOURCE_AUTH_DOMAIN,
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

/// Non-Serde exact health proof for crash-reconciled source A. This is still not a
/// recovery proof and grants no physical execution authority.
#[derive(Debug, Clone)]
pub struct QualifiedCrashSourceHealthV1 {
    qualified_id: QualifiedCrashSourceHealthId,
    reconciliation_id: CrashReconciliationId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    policy_id: CrashSourceHealthPolicyId,
    policy_generation: u64,
    verifier_profile_id: VerifierProfileId,
    verifier_root_epoch: u64,
    authenticated_evidence_id: AuthenticatedCrashSourceHealthId,
    health_profile_digest: [u8; 32],
    qualified_at_unix_ms: u64,
    outcome: PostExecutionHealthOutcomeV1,
    health_state_digest: Option<[u8; 32]>,
}

impl QualifiedCrashSourceHealthV1 {
    pub fn id(&self) -> QualifiedCrashSourceHealthId { self.qualified_id }
    pub fn reconciliation_id(&self) -> CrashReconciliationId { self.reconciliation_id }
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.attempt_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.source_realization_id }
    pub fn distributed_context_id(&self) -> DistributedStateContextId { self.distributed_context_id }
    pub fn health_profile_digest(&self) -> [u8; 32] { self.health_profile_digest }
    pub fn outcome(&self) -> PostExecutionHealthOutcomeV1 { self.outcome }
    pub fn qualified_at_unix_ms(&self) -> u64 { self.qualified_at_unix_ms }
}

pub(crate) fn qualify_crash_source_health(
    reconciliation: &QualifiedCrashReconciliationV1,
    intent: &KnownGoodBoundExecutionAttemptIntentV1,
    evidence: &AuthenticatedCrashSourceHealthV1,
    qualified_at_unix_ms: u64,
) -> Result<QualifiedCrashSourceHealthV1, ExactLocalHealthError> {
    require_source_reconciliation(reconciliation, intent)?;
    if qualified_at_unix_ms == 0 {
        return Err(ExactLocalHealthError::ZeroQualificationTime);
    }
    let claim = &evidence.checked.claim;
    let policy = &evidence.checked.policy;
    let lineage = intent.lineage();
    if claim.reconciliation_id != reconciliation.id()
        || claim.attempt_id != intent.attempt_id()
        || claim.subject_id != lineage.subject_id()
        || claim.source_realization_id != lineage.source_realization_id()
        || claim.distributed_context_id != lineage.distributed_context_id()
    {
        return Err(ExactLocalHealthError::ReconciliationContextMismatch);
    }
    if claim.observed_at_unix_ms < reconciliation.record().reconciled_at_unix_ms() {
        return Err(ExactLocalHealthError::HealthPredatesIdentity);
    }
    check_freshness(
        claim.observed_at_unix_ms,
        qualified_at_unix_ms,
        policy.maximum_observation_age_ms,
        policy.maximum_future_skew_ms,
    )?;
    let qualified_id = QualifiedCrashSourceHealthId(domain_hash_parts(
        SOURCE_QUALIFIED_DOMAIN,
        &[
            reconciliation.id().as_bytes(),
            intent.id().as_bytes(),
            claim.id().as_bytes(),
            policy.id().as_bytes(),
            evidence.evidence_id.as_bytes(),
            &qualified_at_unix_ms.to_le_bytes(),
        ],
    ));
    Ok(QualifiedCrashSourceHealthV1 {
        qualified_id,
        reconciliation_id: reconciliation.id(),
        attempt_id: intent.attempt_id(),
        subject_id: lineage.subject_id(),
        source_realization_id: lineage.source_realization_id(),
        distributed_context_id: lineage.distributed_context_id(),
        policy_id: policy.id(),
        policy_generation: policy.policy_generation,
        verifier_profile_id: policy.verifier_profile_id,
        verifier_root_epoch: policy.verifier_root_epoch,
        authenticated_evidence_id: evidence.evidence_id,
        health_profile_digest: claim.health_profile_digest,
        qualified_at_unix_ms,
        outcome: claim.outcome,
        health_state_digest: claim.health_state_digest,
    })
}

/// The identity lineage that made an exact local Healthy state admissible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthyLocalSnapshotBasisV1 {
    PostExecutionTarget {
        health_id: QualifiedPostExecutionHealthId,
        attempt_id: ExecutionAttemptId,
    },
    CrashReconciledSource {
        health_id: QualifiedCrashSourceHealthId,
        reconciliation_id: CrashReconciliationId,
        attempt_id: ExecutionAttemptId,
    },
}

/// Common downstream proof that one exact local realization is independently known
/// Healthy. Identity provenance remains explicit and domain-separated.
#[derive(Debug, Clone)]
pub struct QualifiedHealthyLocalSnapshotV1 {
    snapshot_id: QualifiedHealthyLocalSnapshotId,
    basis: HealthyLocalSnapshotBasisV1,
    subject_id: ContinuitySubjectId,
    realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    health_profile_digest: [u8; 32],
    qualified_at_unix_ms: u64,
}

impl QualifiedHealthyLocalSnapshotV1 {
    pub fn from_post_execution_target(
        health: &QualifiedPostExecutionHealthV1,
    ) -> Result<Self, ExactLocalHealthError> {
        if health.outcome() != PostExecutionHealthOutcomeV1::Healthy {
            return Err(ExactLocalHealthError::HealthNotHealthy);
        }
        let basis = HealthyLocalSnapshotBasisV1::PostExecutionTarget {
            health_id: health.id(),
            attempt_id: health.attempt_id(),
        };
        Ok(Self::new_snapshot(
            basis,
            health.subject_id(),
            health.target_realization_id(),
            health.distributed_context_id(),
            health.health_profile_digest(),
            health.qualified_at_unix_ms(),
        ))
    }

    pub fn from_crash_reconciled_source(
        health: &QualifiedCrashSourceHealthV1,
    ) -> Result<Self, ExactLocalHealthError> {
        if health.outcome() != PostExecutionHealthOutcomeV1::Healthy {
            return Err(ExactLocalHealthError::HealthNotHealthy);
        }
        let basis = HealthyLocalSnapshotBasisV1::CrashReconciledSource {
            health_id: health.id(),
            reconciliation_id: health.reconciliation_id(),
            attempt_id: health.attempt_id(),
        };
        Ok(Self::new_snapshot(
            basis,
            health.subject_id(),
            health.source_realization_id(),
            health.distributed_context_id(),
            health.health_profile_digest(),
            health.qualified_at_unix_ms(),
        ))
    }

    fn new_snapshot(
        basis: HealthyLocalSnapshotBasisV1,
        subject_id: ContinuitySubjectId,
        realization_id: TargetRealizationId,
        distributed_context_id: DistributedStateContextId,
        health_profile_digest: [u8; 32],
        qualified_at_unix_ms: u64,
    ) -> Self {
        let snapshot_id = QualifiedHealthyLocalSnapshotId(hash_snapshot(
            basis,
            subject_id,
            realization_id,
            distributed_context_id,
            health_profile_digest,
            qualified_at_unix_ms,
        ));
        Self {
            snapshot_id,
            basis,
            subject_id,
            realization_id,
            distributed_context_id,
            health_profile_digest,
            qualified_at_unix_ms,
        }
    }

    pub fn id(&self) -> QualifiedHealthyLocalSnapshotId { self.snapshot_id }
    pub fn basis(&self) -> HealthyLocalSnapshotBasisV1 { self.basis }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn realization_id(&self) -> TargetRealizationId { self.realization_id }
    pub fn distributed_context_id(&self) -> DistributedStateContextId { self.distributed_context_id }
    pub fn health_profile_digest(&self) -> [u8; 32] { self.health_profile_digest }
    pub fn qualified_at_unix_ms(&self) -> u64 { self.qualified_at_unix_ms }
}

fn require_source_reconciliation(
    reconciliation: &QualifiedCrashReconciliationV1,
    intent: &KnownGoodBoundExecutionAttemptIntentV1,
) -> Result<(), ExactLocalHealthError> {
    if reconciliation.classification()
        != CrashReconciliationClassificationV1::SourceKnownGoodObserved
        || reconciliation.record().attempt_id() != intent.attempt_id()
        || reconciliation.record().observed_realization_id()
            != Some(intent.lineage().source_realization_id())
    {
        return Err(ExactLocalHealthError::SourceIdentityNotEstablished);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExactLocalHealthError {
    #[error(transparent)]
    Verification(#[from] VerificationAdmissionError),
    #[error("unsupported crash-source health claim schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("crash-source health policy generation must be non-zero")]
    ZeroPolicyGeneration,
    #[error("crash-source health profile digest must be non-zero")]
    ZeroHealthProfileDigest,
    #[error("crash-source health maximum observation age must be non-zero")]
    ZeroMaximumObservationAge,
    #[error("crash-source health verifier profile does not match exact policy root")]
    VerifierProfileMismatch,
    #[error("crash-source health policy identity mismatch")]
    PolicyIdentityMismatch,
    #[error("crash reconciliation does not establish the exact source known-good realization")]
    SourceIdentityNotEstablished,
    #[error("crash-source health evidence belongs to another reconciliation/subject/realization")]
    ReconciliationContextMismatch,
    #[error("health observation predates exact source identity reconciliation")]
    HealthPredatesIdentity,
    #[error("health observation time must be non-zero")]
    ZeroObservationTime,
    #[error("health raw evidence digest must be non-zero")]
    ZeroRawEvidenceDigest,
    #[error("known health outcome requires a non-zero health-state digest")]
    MissingHealthStateDigest,
    #[error("UNKNOWN health must not fabricate a health-state digest")]
    UnexpectedHealthStateDigest,
    #[error("crash-source health claim identity mismatch")]
    ClaimIdentityMismatch,
    #[error("crash-source health claim uses a different health profile than policy")]
    HealthProfileMismatch,
    #[error("crash-source health authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("crash-source health qualification time must be non-zero")]
    ZeroQualificationTime,
    #[error("crash-source health evidence is stale: age {age_ms} ms > {allowed_ms} ms")]
    StaleObservation { age_ms: u64, allowed_ms: u64 },
    #[error("crash-source health evidence is too far in the future: skew {skew_ms} ms > {allowed_ms} ms")]
    ObservationFromFuture { skew_ms: u64, allowed_ms: u64 },
    #[error("local health snapshot requires an independently qualified Healthy outcome")]
    HealthNotHealthy,
}

fn validate_health_material(
    health_profile_digest: [u8; 32],
    observed_at_unix_ms: u64,
    outcome: PostExecutionHealthOutcomeV1,
    health_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> Result<(), ExactLocalHealthError> {
    if health_profile_digest == [0; 32] {
        return Err(ExactLocalHealthError::ZeroHealthProfileDigest);
    }
    if observed_at_unix_ms == 0 {
        return Err(ExactLocalHealthError::ZeroObservationTime);
    }
    if raw_evidence_digest == [0; 32] {
        return Err(ExactLocalHealthError::ZeroRawEvidenceDigest);
    }
    match outcome {
        PostExecutionHealthOutcomeV1::Unknown => {
            if health_state_digest.is_some() {
                return Err(ExactLocalHealthError::UnexpectedHealthStateDigest);
            }
        }
        PostExecutionHealthOutcomeV1::Healthy
        | PostExecutionHealthOutcomeV1::Degraded
        | PostExecutionHealthOutcomeV1::Unhealthy => {
            if health_state_digest.is_none() || health_state_digest == Some([0; 32]) {
                return Err(ExactLocalHealthError::MissingHealthStateDigest);
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
) -> Result<(), ExactLocalHealthError> {
    if observed_at_unix_ms > qualified_at_unix_ms {
        let skew = observed_at_unix_ms - qualified_at_unix_ms;
        if skew > maximum_future_skew_ms {
            return Err(ExactLocalHealthError::ObservationFromFuture {
                skew_ms: skew,
                allowed_ms: maximum_future_skew_ms,
            });
        }
        return Ok(());
    }
    let age = qualified_at_unix_ms - observed_at_unix_ms;
    if age > maximum_age_ms {
        return Err(ExactLocalHealthError::StaleObservation {
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
    domain_hash_parts(SOURCE_POLICY_DOMAIN, &[
        profile_id.as_bytes(),
        &root_epoch.to_le_bytes(),
        &generation.to_le_bytes(),
        &health_profile_digest,
        &maximum_age_ms.to_le_bytes(),
        &maximum_future_skew_ms.to_le_bytes(),
    ])
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    reconciliation_id: CrashReconciliationId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    verifier_profile_id: VerifierProfileId,
    health_profile_digest: [u8; 32],
    observed_at_unix_ms: u64,
    outcome: PostExecutionHealthOutcomeV1,
    health_state_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SOURCE_CLAIM_DOMAIN);
    hasher.update(reconciliation_id.as_bytes());
    hasher.update(attempt_id.as_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(source_realization_id.as_bytes());
    hasher.update(distributed_context_id.as_bytes());
    hasher.update(verifier_profile_id.as_bytes());
    hasher.update(&health_profile_digest);
    hasher.update(&observed_at_unix_ms.to_le_bytes());
    hasher.update(&[health_outcome_tag(outcome)]);
    match health_state_digest {
        Some(digest) => {
            hasher.update(&[1]);
            hasher.update(&digest);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&raw_evidence_digest);
    *hasher.finalize().as_bytes()
}

fn hash_snapshot(
    basis: HealthyLocalSnapshotBasisV1,
    subject_id: ContinuitySubjectId,
    realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    health_profile_digest: [u8; 32],
    qualified_at_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SNAPSHOT_DOMAIN);
    match basis {
        HealthyLocalSnapshotBasisV1::PostExecutionTarget { health_id, attempt_id } => {
            hasher.update(&[1]);
            hasher.update(health_id.as_bytes());
            hasher.update(attempt_id.as_bytes());
        }
        HealthyLocalSnapshotBasisV1::CrashReconciledSource {
            health_id,
            reconciliation_id,
            attempt_id,
        } => {
            hasher.update(&[2]);
            hasher.update(health_id.as_bytes());
            hasher.update(reconciliation_id.as_bytes());
            hasher.update(attempt_id.as_bytes());
        }
    }
    hasher.update(subject_id.as_bytes());
    hasher.update(realization_id.as_bytes());
    hasher.update(distributed_context_id.as_bytes());
    hasher.update(&health_profile_digest);
    hasher.update(&qualified_at_unix_ms.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn health_outcome_tag(outcome: PostExecutionHealthOutcomeV1) -> u8 {
    match outcome {
        PostExecutionHealthOutcomeV1::Healthy => 1,
        PostExecutionHealthOutcomeV1::Degraded => 2,
        PostExecutionHealthOutcomeV1::Unhealthy => 3,
        PostExecutionHealthOutcomeV1::Unknown => 4,
    }
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

    #[test]
    fn target_and_source_snapshot_domains_are_distinct() {
        assert_ne!(SOURCE_QUALIFIED_DOMAIN, SNAPSHOT_DOMAIN);
    }

    #[test]
    fn unknown_health_cannot_be_a_healthy_snapshot_semantically() {
        assert_ne!(
            health_outcome_tag(PostExecutionHealthOutcomeV1::Unknown),
            health_outcome_tag(PostExecutionHealthOutcomeV1::Healthy)
        );
    }

    #[test]
    fn source_authentication_purpose_and_wire_domain_are_explicit() {
        assert_eq!(
            CRASH_SOURCE_HEALTH_AUTH_PURPOSE,
            "symthaea.continuity.crash-source-health.v1"
        );
        assert_ne!(SOURCE_CLAIM_WIRE_DOMAIN, SOURCE_CLAIM_DOMAIN);
    }
}
