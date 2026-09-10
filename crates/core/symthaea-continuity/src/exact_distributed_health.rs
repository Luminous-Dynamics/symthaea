// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Additive distributed-health V2 bound to a generic exact Healthy local snapshot.
//!
//! V1 remains the post-execution-target proof and retains its original hash contract.
//! V2 uses the same shared closed-world distributed evaluator but can qualify either
//! exact healthy target B or exact healthy crash-reconciled source A without erasing
//! their identity provenance.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed::{DistributedChangeBudgetId, RecoveryPathClassV1};
use crate::distributed_currentness::{
    DistributedCurrentnessPolicyId, ValidatedDistributedCurrentnessPolicyV1,
};
use crate::distributed_evidence::{
    AuthenticatedFailureDomainStateEvidenceId, AuthenticatedFailureDomainStateEvidenceV1,
    AuthenticatedRecoveryPathStateEvidenceId, AuthenticatedRecoveryPathStateEvidenceV1,
};
use crate::distributed_health_common::evaluate_distributed_health;
use crate::distributed_state::{
    AuthenticatedParticipantStateEvidenceId, AuthenticatedParticipantStateEvidenceV1,
    DistributedStateContextId, ParticipantSetDigest, ValidatedDistributedStateContextV1,
};
use crate::exact_local_health::{
    HealthyLocalSnapshotBasisV1, QualifiedHealthyLocalSnapshotId,
    QualifiedHealthyLocalSnapshotV1,
};
use crate::failure_domain::{FailureDomainPolicyId, ValidatedFailureDomainPolicyV1};
use crate::post_transition_distributed_health::PostTransitionDistributedHealthError;
use crate::scope::ContinuitySubjectId;
use crate::verifier::VerifierProfileId;
use crate::witness::TargetRealizationId;

const CURRENT_STATE_DOMAIN: &[u8] =
    b"symthaea.continuity.exact-distributed-current-state.v2\0";
const QUALIFIED_DOMAIN: &[u8] =
    b"symthaea.continuity.qualified-exact-distributed-health.v2\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExactDistributedStateDigestV2([u8; 32]);
impl ExactDistributedStateDigestV2 {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedExactDistributedHealthIdV2([u8; 32]);
impl QualifiedExactDistributedHealthIdV2 {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExactDistributedVerifierSnapshotV2 {
    profile_id: VerifierProfileId,
    root_epoch: u64,
}

impl ExactDistributedVerifierSnapshotV2 {
    pub fn profile_id(&self) -> VerifierProfileId { self.profile_id }
    pub fn root_epoch(&self) -> u64 { self.root_epoch }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactDistributedRecoveryPathV2 {
    recovery_path_class: RecoveryPathClassV1,
    recovery_path_identity_digest: [u8; 32],
}

impl ExactDistributedRecoveryPathV2 {
    pub fn recovery_path_class(&self) -> &RecoveryPathClassV1 {
        &self.recovery_path_class
    }

    pub fn recovery_path_identity_digest(&self) -> [u8; 32] {
        self.recovery_path_identity_digest
    }
}

/// Non-Serde exact distributed-health proof around one exact independently Healthy
/// local realization. The local snapshot basis remains visible so a caller can tell
/// whether the local identity came from normal target observation or crash-source
/// reconciliation.
#[derive(Debug, Clone)]
pub struct QualifiedExactDistributedHealthV2 {
    qualified_id: QualifiedExactDistributedHealthIdV2,
    local_snapshot_id: QualifiedHealthyLocalSnapshotId,
    local_snapshot_basis: HealthyLocalSnapshotBasisV1,
    subject_id: ContinuitySubjectId,
    realization_id: TargetRealizationId,
    health_profile_digest: [u8; 32],
    context_id: DistributedStateContextId,
    aggregate_subject_id: ContinuitySubjectId,
    budget_id: DistributedChangeBudgetId,
    budget_generation: u64,
    participant_set_digest: ParticipantSetDigest,
    currentness_policy_id: DistributedCurrentnessPolicyId,
    currentness_policy_generation: u64,
    current_state_digest: ExactDistributedStateDigestV2,
    verifier_snapshots: Vec<ExactDistributedVerifierSnapshotV2>,
    recovery_paths: Vec<ExactDistributedRecoveryPathV2>,
    unavailable_count: u32,
    healthy_count: u32,
    healthy_domains: Vec<(FailureDomainPolicyId, u32)>,
    evaluated_at_unix_ms: u64,
}

impl QualifiedExactDistributedHealthV2 {
    pub fn id(&self) -> QualifiedExactDistributedHealthIdV2 { self.qualified_id }
    pub fn local_snapshot_id(&self) -> QualifiedHealthyLocalSnapshotId { self.local_snapshot_id }
    pub fn local_snapshot_basis(&self) -> HealthyLocalSnapshotBasisV1 { self.local_snapshot_basis }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn realization_id(&self) -> TargetRealizationId { self.realization_id }
    pub fn health_profile_digest(&self) -> [u8; 32] { self.health_profile_digest }
    pub fn context_id(&self) -> DistributedStateContextId { self.context_id }
    pub fn aggregate_subject_id(&self) -> ContinuitySubjectId { self.aggregate_subject_id }
    pub fn current_state_digest(&self) -> ExactDistributedStateDigestV2 { self.current_state_digest }
    pub fn verifier_snapshots(&self) -> &[ExactDistributedVerifierSnapshotV2] { &self.verifier_snapshots }
    pub fn recovery_paths(&self) -> &[ExactDistributedRecoveryPathV2] { &self.recovery_paths }
    pub fn unavailable_count(&self) -> u32 { self.unavailable_count }
    pub fn healthy_count(&self) -> u32 { self.healthy_count }
    pub fn healthy_domains(&self) -> &[(FailureDomainPolicyId, u32)] { &self.healthy_domains }
    pub fn evaluated_at_unix_ms(&self) -> u64 { self.evaluated_at_unix_ms }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compose_exact_distributed_health_v2(
    local_snapshot: &QualifiedHealthyLocalSnapshotV1,
    context: &ValidatedDistributedStateContextV1,
    currentness: &ValidatedDistributedCurrentnessPolicyV1,
    failure_domain_policies: &[ValidatedFailureDomainPolicyV1],
    participant_evidence: &[AuthenticatedParticipantStateEvidenceV1],
    failure_domain_evidence: &[AuthenticatedFailureDomainStateEvidenceV1],
    recovery_evidence: &[AuthenticatedRecoveryPathStateEvidenceV1],
    evaluated_at_unix_ms: u64,
) -> Result<QualifiedExactDistributedHealthV2, ExactDistributedHealthError> {
    if evaluated_at_unix_ms == 0 {
        return Err(ExactDistributedHealthError::ZeroEvaluationTime);
    }
    if local_snapshot.distributed_context_id() != context.id() {
        return Err(ExactDistributedHealthError::LocalSnapshotContextMismatch);
    }
    if local_snapshot.qualified_at_unix_ms() != evaluated_at_unix_ms {
        return Err(ExactDistributedHealthError::LocalSnapshotEvaluationTimeMismatch);
    }
    if context
        .candidate_subject_ids()
        .binary_search(&local_snapshot.subject_id())
        .is_err()
    {
        return Err(ExactDistributedHealthError::LocalSubjectOutsideCandidateSet);
    }

    let evaluated = evaluate_distributed_health(
        local_snapshot.subject_id(),
        context,
        currentness,
        failure_domain_policies,
        participant_evidence,
        failure_domain_evidence,
        recovery_evidence,
        evaluated_at_unix_ms,
    )?;

    let verifier_snapshots = evaluated
        .verifier_snapshots
        .iter()
        .map(|(profile_id, root_epoch)| ExactDistributedVerifierSnapshotV2 {
            profile_id: *profile_id,
            root_epoch: *root_epoch,
        })
        .collect::<Vec<_>>();
    let recovery_paths = evaluated
        .recovery_paths
        .iter()
        .map(|(recovery_path_class, recovery_path_identity_digest)| {
            ExactDistributedRecoveryPathV2 {
                recovery_path_class: recovery_path_class.clone(),
                recovery_path_identity_digest: *recovery_path_identity_digest,
            }
        })
        .collect::<Vec<_>>();

    let current_state_digest = ExactDistributedStateDigestV2(hash_current_state(
        local_snapshot.id(),
        context.id(),
        currentness.id(),
        &evaluated.participant_evidence_ids,
        &evaluated.failure_evidence_ids,
        &evaluated.recovery_evidence_ids,
    ));
    let qualified_id = QualifiedExactDistributedHealthIdV2(hash_qualified(
        current_state_digest,
        local_snapshot,
        context,
        currentness,
        &verifier_snapshots,
        &recovery_paths,
        evaluated.unavailable_count,
        evaluated.healthy_count,
        &evaluated.healthy_domains,
        evaluated_at_unix_ms,
    ));

    Ok(QualifiedExactDistributedHealthV2 {
        qualified_id,
        local_snapshot_id: local_snapshot.id(),
        local_snapshot_basis: local_snapshot.basis(),
        subject_id: local_snapshot.subject_id(),
        realization_id: local_snapshot.realization_id(),
        health_profile_digest: local_snapshot.health_profile_digest(),
        context_id: context.id(),
        aggregate_subject_id: context.aggregate_subject_id(),
        budget_id: context.budget_id(),
        budget_generation: context.budget_generation(),
        participant_set_digest: context.participant_set_digest(),
        currentness_policy_id: currentness.id(),
        currentness_policy_generation: currentness.policy_generation(),
        current_state_digest,
        verifier_snapshots,
        recovery_paths,
        unavailable_count: evaluated.unavailable_count,
        healthy_count: evaluated.healthy_count,
        healthy_domains: evaluated.healthy_domains,
        evaluated_at_unix_ms,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExactDistributedHealthError {
    #[error(transparent)]
    DistributedEvaluation(#[from] PostTransitionDistributedHealthError),
    #[error("exact distributed-health evaluation time must be non-zero")]
    ZeroEvaluationTime,
    #[error("healthy local snapshot belongs to another distributed context")]
    LocalSnapshotContextMismatch,
    #[error("healthy local snapshot and distributed evaluation must share one exact logical time")]
    LocalSnapshotEvaluationTimeMismatch,
    #[error("healthy local snapshot subject is outside the exact distributed candidate set")]
    LocalSubjectOutsideCandidateSet,
}

fn hash_current_state(
    local_snapshot_id: QualifiedHealthyLocalSnapshotId,
    context_id: DistributedStateContextId,
    currentness_policy_id: DistributedCurrentnessPolicyId,
    participant_ids: &[AuthenticatedParticipantStateEvidenceId],
    failure_ids: &[AuthenticatedFailureDomainStateEvidenceId],
    recovery_ids: &[AuthenticatedRecoveryPathStateEvidenceId],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CURRENT_STATE_DOMAIN);
    hasher.update(local_snapshot_id.as_bytes());
    hasher.update(context_id.as_bytes());
    hasher.update(currentness_policy_id.as_bytes());
    hash_ids(&mut hasher, participant_ids.iter().map(|id| id.as_bytes()));
    hash_ids(&mut hasher, failure_ids.iter().map(|id| id.as_bytes()));
    hash_ids(&mut hasher, recovery_ids.iter().map(|id| id.as_bytes()));
    *hasher.finalize().as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn hash_qualified(
    current_state_digest: ExactDistributedStateDigestV2,
    local_snapshot: &QualifiedHealthyLocalSnapshotV1,
    context: &ValidatedDistributedStateContextV1,
    currentness: &ValidatedDistributedCurrentnessPolicyV1,
    verifier_snapshots: &[ExactDistributedVerifierSnapshotV2],
    recovery_paths: &[ExactDistributedRecoveryPathV2],
    unavailable_count: u32,
    healthy_count: u32,
    healthy_domains: &[(FailureDomainPolicyId, u32)],
    evaluated_at_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFIED_DOMAIN);
    hasher.update(current_state_digest.as_bytes());
    hasher.update(local_snapshot.id().as_bytes());
    hasher.update(local_snapshot.subject_id().as_bytes());
    hasher.update(local_snapshot.realization_id().as_bytes());
    hasher.update(&local_snapshot.health_profile_digest());
    hasher.update(context.id().as_bytes());
    hasher.update(context.aggregate_subject_id().as_bytes());
    hasher.update(context.budget_id().as_bytes());
    hasher.update(&context.budget_generation().to_le_bytes());
    hasher.update(context.participant_set_digest().as_bytes());
    hasher.update(currentness.id().as_bytes());
    hasher.update(&currentness.policy_generation().to_le_bytes());
    hasher.update(&unavailable_count.to_le_bytes());
    hasher.update(&healthy_count.to_le_bytes());
    hasher.update(&evaluated_at_unix_ms.to_le_bytes());
    hasher.update(&(verifier_snapshots.len() as u64).to_le_bytes());
    for snapshot in verifier_snapshots {
        hasher.update(snapshot.profile_id.as_bytes());
        hasher.update(&snapshot.root_epoch.to_le_bytes());
    }
    hasher.update(&(recovery_paths.len() as u64).to_le_bytes());
    for path in recovery_paths {
        encode_recovery_class(&mut hasher, &path.recovery_path_class);
        hasher.update(&path.recovery_path_identity_digest);
    }
    hasher.update(&(healthy_domains.len() as u64).to_le_bytes());
    for (policy_id, count) in healthy_domains {
        hasher.update(policy_id.as_bytes());
        hasher.update(&count.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn hash_ids<'a>(hasher: &mut blake3::Hasher, ids: impl Iterator<Item = &'a [u8; 32]>) {
    let ids = ids.collect::<Vec<_>>();
    hasher.update(&(ids.len() as u64).to_le_bytes());
    for id in ids {
        hasher.update(id);
    }
}

fn encode_recovery_class(hasher: &mut blake3::Hasher, class: &RecoveryPathClassV1) {
    match class {
        RecoveryPathClassV1::DeviceLocalAutomaticRollback => {
            hasher.update(&[1]);
        }
        RecoveryPathClassV1::OutOfBandManagement => {
            hasher.update(&[2]);
        }
        RecoveryPathClassV1::IndependentNetworkPath => {
            hasher.update(&[3]);
        }
        RecoveryPathClassV1::LocalPhysicalIntervention => {
            hasher.update(&[4]);
        }
        RecoveryPathClassV1::Custom { kind_id } => {
            hasher.update(&[255]);
            hasher.update(&(kind_id.len() as u64).to_le_bytes());
            hasher.update(kind_id.as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_domains_are_distinct_from_v1_domains() {
        assert_ne!(
            CURRENT_STATE_DOMAIN,
            b"symthaea.continuity.post-transition-distributed-current-state.v1\0"
        );
        assert_ne!(
            QUALIFIED_DOMAIN,
            b"symthaea.continuity.qualified-post-transition-distributed-health.v1\0"
        );
    }
}
