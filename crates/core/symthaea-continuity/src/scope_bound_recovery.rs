// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scope-bounded continuity recovery after crash + external-effect reconciliation.
//!
//! Returning the local/distributed system to exact source A is not sufficient when
//! an attempted A -> B transition may have caused externally visible effects. This
//! module composes the exact current crash-recovery theorem with the exact pre-mutation
//! protected effect scope and complete reconciliation of every declared obligation.
//!
//! The result is deliberately scope-bounded: it proves recovery only inside the exact
//! declared coverage manifest. It does not claim knowledge of undeclared effects.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::crash_recovery_qualification::{
    CrashRecoveryQualificationError, QualifiedCrashRecoveryToActiveKnownGoodId,
    QualifiedCrashRecoveryToActiveKnownGoodV1,
};
use crate::current_crash_recovery::{
    CurrentCrashRecoveryError, QualifiedCurrentCrashRecoveryId,
    QualifiedCurrentCrashRecoveryV1,
};
use crate::effect_scope_rebind::{
    EffectScopeRebindError, ReboundEffectScopeId, ReboundEffectScopedExecutionV1,
};
use crate::effect_scoped_execution::{
    EffectScopedExecutionError, QualifiedEffectScopeCommitmentId,
    QualifiedEffectScopeCommitmentV1,
};
use crate::execution_capability::ExecutionAttemptId;
use crate::external_effect_authority::{
    ExternalEffectPlanId, QualifiedExternalEffectAuthorizationId,
};
use crate::external_effects::{
    ExternalEffectContractId, ExternalEffectError,
    QualifiedExternalEffectReconciliationId, QualifiedExternalEffectReconciliationV1,
};
use crate::scope::ContinuitySubjectId;
use crate::witness::TargetRealizationId;

pub const SCOPE_BOUND_RECOVERY_RECORD_SCHEMA_V1: &str =
    "symthaea-continuity-scope-bound-recovery-record-v1";

const RECOVERY_DOMAIN: &[u8] = b"symthaea.continuity.scope-bound-recovery.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedScopeBoundContinuityRecoveryId([u8; 32]);
impl QualifiedScopeBoundContinuityRecoveryId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Serializable audit record. It cannot recreate any parent non-Serde proof after
/// restart; `rebind()` requires every exact live parent again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeBoundContinuityRecoveryRecordV1 {
    schema_version: String,
    current_crash_recovery_id: QualifiedCurrentCrashRecoveryId,
    crash_recovery_id: QualifiedCrashRecoveryToActiveKnownGoodId,
    rebound_effect_scope_id: ReboundEffectScopeId,
    scope_commitment_id: QualifiedEffectScopeCommitmentId,
    effect_reconciliation_id: QualifiedExternalEffectReconciliationId,
    original_attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    recovered_realization_id: TargetRealizationId,
    abandoned_target_id: TargetRealizationId,
    effect_plan_id: ExternalEffectPlanId,
    effect_authorization_id: QualifiedExternalEffectAuthorizationId,
    effect_contract_id: ExternalEffectContractId,
    coverage_manifest_digest: [u8; 32],
    protected_scope_committed_at_unix_ms: u64,
    recovered_at_unix_ms: u64,
    decision_time_unix_ms: u64,
    recovery_id: QualifiedScopeBoundContinuityRecoveryId,
}

impl ScopeBoundContinuityRecoveryRecordV1 {
    pub fn validate(&self) -> Result<(), ScopeBoundRecoveryError> {
        if self.schema_version != SCOPE_BOUND_RECOVERY_RECORD_SCHEMA_V1 {
            return Err(ScopeBoundRecoveryError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.coverage_manifest_digest == [0; 32] {
            return Err(ScopeBoundRecoveryError::ZeroCoverageManifestDigest);
        }
        if self.recovered_realization_id == self.abandoned_target_id {
            return Err(ScopeBoundRecoveryError::SourceEqualsTarget);
        }
        if self.protected_scope_committed_at_unix_ms == 0
            || self.recovered_at_unix_ms == 0
            || self.decision_time_unix_ms == 0
        {
            return Err(ScopeBoundRecoveryError::ZeroTime);
        }
        if self.protected_scope_committed_at_unix_ms >= self.recovered_at_unix_ms {
            return Err(ScopeBoundRecoveryError::ScopeCommitmentNotBeforeRecovery);
        }
        if self.decision_time_unix_ms < self.recovered_at_unix_ms {
            return Err(ScopeBoundRecoveryError::DecisionPredatesRecovery);
        }
        let expected = QualifiedScopeBoundContinuityRecoveryId(hash_recovery(
            self.current_crash_recovery_id,
            self.crash_recovery_id,
            self.rebound_effect_scope_id,
            self.scope_commitment_id,
            self.effect_reconciliation_id,
            self.original_attempt_id,
            self.subject_id,
            self.recovered_realization_id,
            self.abandoned_target_id,
            self.effect_plan_id,
            self.effect_authorization_id,
            self.effect_contract_id,
            self.coverage_manifest_digest,
            self.protected_scope_committed_at_unix_ms,
            self.recovered_at_unix_ms,
            self.decision_time_unix_ms,
        ));
        if expected != self.recovery_id {
            return Err(ScopeBoundRecoveryError::RecoveryIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> QualifiedScopeBoundContinuityRecoveryId { self.recovery_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn recovered_realization_id(&self) -> TargetRealizationId { self.recovered_realization_id }
    pub fn abandoned_target_id(&self) -> TargetRealizationId { self.abandoned_target_id }
    pub fn effect_contract_id(&self) -> ExternalEffectContractId { self.effect_contract_id }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
    pub fn decision_time_unix_ms(&self) -> u64 { self.decision_time_unix_ms }
}

/// Non-Serde proof that exact A is recovered/current and every obligation inside the
/// exact pre-authorized, pre-mutation protected external-effect scope is reconciled
/// at the same logical decision boundary.
#[derive(Debug, Clone)]
pub struct QualifiedScopeBoundContinuityRecoveryV1 {
    record: ScopeBoundContinuityRecoveryRecordV1,
}

impl QualifiedScopeBoundContinuityRecoveryV1 {
    pub fn qualify(
        current_recovery: &QualifiedCurrentCrashRecoveryV1,
        crash_recovery: &QualifiedCrashRecoveryToActiveKnownGoodV1,
        rebound_scope: &ReboundEffectScopedExecutionV1,
        scope_commitment: &QualifiedEffectScopeCommitmentV1,
        effect_reconciliation: &QualifiedExternalEffectReconciliationV1,
    ) -> Result<Self, ScopeBoundRecoveryError> {
        current_recovery.record().validate()?;
        crash_recovery.record().validate()?;
        effect_reconciliation.record().validate()?;

        let current = current_recovery.record();
        let crash = crash_recovery.record();
        let effects = effect_reconciliation.record();

        if current.crash_recovery_id() != crash_recovery.id() {
            return Err(ScopeBoundRecoveryError::CurrentRecoveryParentMismatch);
        }
        if current.subject_id() != crash.subject_id()
            || current.recovered_realization_id() != crash.recovered_realization_id()
        {
            return Err(ScopeBoundRecoveryError::CurrentRecoveryStateMismatch);
        }

        if rebound_scope.attempt_id() != crash.original_attempt_id()
            || rebound_scope.subject_id() != crash.subject_id()
            || rebound_scope.source_realization_id() != crash.recovered_realization_id()
            || rebound_scope.target_realization_id() != crash.failed_or_abandoned_target_id()
        {
            return Err(ScopeBoundRecoveryError::RecoveredAttemptLineageMismatch);
        }

        if scope_commitment.envelope_id() != rebound_scope.envelope_id()
            || scope_commitment.attempt_id() != rebound_scope.attempt_id()
            || scope_commitment.subject_id() != rebound_scope.subject_id()
            || scope_commitment.effect_plan_id() != rebound_scope.plan_id()
            || scope_commitment.effect_authorization_id() != rebound_scope.authorization_id()
            || scope_commitment.effect_contract_id() != rebound_scope.contract_id()
            || scope_commitment.coverage_manifest_digest() != rebound_scope.coverage_manifest_digest()
        {
            return Err(ScopeBoundRecoveryError::ProtectedScopeMismatch);
        }

        if effects.contract_id() != rebound_scope.contract_id()
            || effects.coverage_manifest_digest() != rebound_scope.coverage_manifest_digest()
        {
            return Err(ScopeBoundRecoveryError::EffectReconciliationScopeMismatch);
        }

        let decision_time_unix_ms = current.currentness_anchored_at_unix_ms();
        if effects.reconciled_at_unix_ms() != decision_time_unix_ms {
            return Err(ScopeBoundRecoveryError::DecisionTimeMismatch {
                currentness_time_unix_ms: decision_time_unix_ms,
                effect_reconciliation_time_unix_ms: effects.reconciled_at_unix_ms(),
            });
        }
        if scope_commitment.committed_at_unix_ms() >= crash.recovered_at_unix_ms() {
            return Err(ScopeBoundRecoveryError::ScopeCommitmentNotBeforeRecovery);
        }
        if decision_time_unix_ms < crash.recovered_at_unix_ms() {
            return Err(ScopeBoundRecoveryError::DecisionPredatesRecovery);
        }

        let recovery_id = QualifiedScopeBoundContinuityRecoveryId(hash_recovery(
            current_recovery.id(),
            crash_recovery.id(),
            rebound_scope.id(),
            scope_commitment.id(),
            effect_reconciliation.id(),
            crash.original_attempt_id(),
            crash.subject_id(),
            crash.recovered_realization_id(),
            crash.failed_or_abandoned_target_id(),
            rebound_scope.plan_id(),
            rebound_scope.authorization_id(),
            rebound_scope.contract_id(),
            rebound_scope.coverage_manifest_digest(),
            scope_commitment.committed_at_unix_ms(),
            crash.recovered_at_unix_ms(),
            decision_time_unix_ms,
        ));
        let record = ScopeBoundContinuityRecoveryRecordV1 {
            schema_version: SCOPE_BOUND_RECOVERY_RECORD_SCHEMA_V1.to_owned(),
            current_crash_recovery_id: current_recovery.id(),
            crash_recovery_id: crash_recovery.id(),
            rebound_effect_scope_id: rebound_scope.id(),
            scope_commitment_id: scope_commitment.id(),
            effect_reconciliation_id: effect_reconciliation.id(),
            original_attempt_id: crash.original_attempt_id(),
            subject_id: crash.subject_id(),
            recovered_realization_id: crash.recovered_realization_id(),
            abandoned_target_id: crash.failed_or_abandoned_target_id(),
            effect_plan_id: rebound_scope.plan_id(),
            effect_authorization_id: rebound_scope.authorization_id(),
            effect_contract_id: rebound_scope.contract_id(),
            coverage_manifest_digest: rebound_scope.coverage_manifest_digest(),
            protected_scope_committed_at_unix_ms: scope_commitment.committed_at_unix_ms(),
            recovered_at_unix_ms: crash.recovered_at_unix_ms(),
            decision_time_unix_ms,
            recovery_id,
        };
        record.validate()?;
        Ok(Self { record })
    }

    pub fn rebind(
        record: ScopeBoundContinuityRecoveryRecordV1,
        current_recovery: &QualifiedCurrentCrashRecoveryV1,
        crash_recovery: &QualifiedCrashRecoveryToActiveKnownGoodV1,
        rebound_scope: &ReboundEffectScopedExecutionV1,
        scope_commitment: &QualifiedEffectScopeCommitmentV1,
        effect_reconciliation: &QualifiedExternalEffectReconciliationV1,
    ) -> Result<Self, ScopeBoundRecoveryError> {
        record.validate()?;
        let fresh = Self::qualify(
            current_recovery,
            crash_recovery,
            rebound_scope,
            scope_commitment,
            effect_reconciliation,
        )?;
        if fresh.record != record {
            return Err(ScopeBoundRecoveryError::RecoveryLineageMismatch);
        }
        Ok(fresh)
    }

    pub fn id(&self) -> QualifiedScopeBoundContinuityRecoveryId { self.record.id() }
    pub fn record(&self) -> &ScopeBoundContinuityRecoveryRecordV1 { &self.record }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ScopeBoundRecoveryError {
    #[error(transparent)]
    CurrentRecovery(#[from] CurrentCrashRecoveryError),
    #[error(transparent)]
    CrashRecovery(#[from] CrashRecoveryQualificationError),
    #[error(transparent)]
    Rebind(#[from] EffectScopeRebindError),
    #[error(transparent)]
    ScopeCommitment(#[from] EffectScopedExecutionError),
    #[error(transparent)]
    ExternalEffect(#[from] ExternalEffectError),
    #[error("unsupported scope-bounded recovery schema: {0}")]
    UnsupportedSchema(String),
    #[error("scope-bounded recovery coverage manifest digest must be non-zero")]
    ZeroCoverageManifestDigest,
    #[error("scope-bounded recovery source and target realizations must differ")]
    SourceEqualsTarget,
    #[error("scope-bounded recovery times must be non-zero")]
    ZeroTime,
    #[error("protected effect scope was not committed before recovered state was established")]
    ScopeCommitmentNotBeforeRecovery,
    #[error("scope-bounded recovery decision predates recovered state")]
    DecisionPredatesRecovery,
    #[error("current crash recovery does not name the exact supplied crash-recovery parent")]
    CurrentRecoveryParentMismatch,
    #[error("current crash recovery and crash-recovery parent disagree on exact A/subject")]
    CurrentRecoveryStateMismatch,
    #[error("rebound protected effect scope belongs to another A -> B attempt")]
    RecoveredAttemptLineageMismatch,
    #[error("qualified pre-mutation effect-scope commitment differs from rebound scope")]
    ProtectedScopeMismatch,
    #[error("external-effect reconciliation differs from exact protected attempt contract")]
    EffectReconciliationScopeMismatch,
    #[error("recovery decision time mismatch: active-LKG currentness={currentness_time_unix_ms}, external-effect reconciliation={effect_reconciliation_time_unix_ms}")]
    DecisionTimeMismatch {
        currentness_time_unix_ms: u64,
        effect_reconciliation_time_unix_ms: u64,
    },
    #[error("scope-bounded recovery identity does not match canonical fields")]
    RecoveryIdentityMismatch,
    #[error("persisted scope-bounded recovery record does not match exact live parent proofs")]
    RecoveryLineageMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_recovery(
    current_recovery_id: QualifiedCurrentCrashRecoveryId,
    crash_recovery_id: QualifiedCrashRecoveryToActiveKnownGoodId,
    rebound_scope_id: ReboundEffectScopeId,
    scope_commitment_id: QualifiedEffectScopeCommitmentId,
    effect_reconciliation_id: QualifiedExternalEffectReconciliationId,
    original_attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    recovered_realization_id: TargetRealizationId,
    abandoned_target_id: TargetRealizationId,
    effect_plan_id: ExternalEffectPlanId,
    effect_authorization_id: QualifiedExternalEffectAuthorizationId,
    effect_contract_id: ExternalEffectContractId,
    coverage_manifest_digest: [u8; 32],
    protected_scope_committed_at_unix_ms: u64,
    recovered_at_unix_ms: u64,
    decision_time_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECOVERY_DOMAIN);
    hasher.update(current_recovery_id.as_bytes());
    hasher.update(crash_recovery_id.as_bytes());
    hasher.update(rebound_scope_id.as_bytes());
    hasher.update(scope_commitment_id.as_bytes());
    hasher.update(effect_reconciliation_id.as_bytes());
    hasher.update(original_attempt_id.as_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(recovered_realization_id.as_bytes());
    hasher.update(abandoned_target_id.as_bytes());
    hasher.update(effect_plan_id.as_bytes());
    hasher.update(effect_authorization_id.as_bytes());
    hasher.update(effect_contract_id.as_bytes());
    hasher.update(&coverage_manifest_digest);
    hasher.update(&protected_scope_committed_at_unix_ms.to_le_bytes());
    hasher.update(&recovered_at_unix_ms.to_le_bytes());
    hasher.update(&decision_time_unix_ms.to_le_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_bounded_domain_is_distinct_from_crash_recovery_domain() {
        assert_ne!(
            RECOVERY_DOMAIN,
            b"symthaea.continuity.crash-recovered-active-known-good.v1\0"
        );
    }
}
