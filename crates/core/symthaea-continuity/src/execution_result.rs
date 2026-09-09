// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical execution result reconstructed from durable intent + backend receipt.
//!
//! A backend receipt is intentionally not the authority for transition lineage.
//! The pre-mutation durable intent owns eligibility/backend/session/subject/target/
//! context/epoch lineage; the receipt contributes only the reported outcome and
//! result evidence for that exact attempt.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed_state::DistributedStateContextId;
use crate::execution_capability::{
    ExecutionAttemptId, ExecutionAttemptIntentV1, ExecutionAttemptOutcomeV1,
    ExecutionAttemptReceiptId, ExecutionAttemptReceiptV1, ExecutionBackendId,
    ExecutionCapabilityError, ExecutionSessionId, OneUseExecutionCapabilityId,
};
use crate::scope::ContinuitySubjectId;
use crate::trusted_commit_epoch::{QualifiedTrustedCommitEpochId, TrustedCommitEligibilityId};
use crate::witness::TargetRealizationId;

const RESULT_DOMAIN: &[u8] = b"symthaea.continuity.canonical-execution-result.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CanonicalExecutionAttemptResultId([u8; 32]);

impl CanonicalExecutionAttemptResultId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Non-Serde canonical join of one validated durable intent and one validated
/// backend-reported receipt.
///
/// Lineage is copied only from the intent. The receipt is not permitted to redefine
/// what subject, backend, session, or trusted epoch was actually authorized.
#[derive(Debug, Clone)]
pub struct CanonicalExecutionAttemptResultV1 {
    result_id: CanonicalExecutionAttemptResultId,
    attempt_id: ExecutionAttemptId,
    intent_capability_id: OneUseExecutionCapabilityId,
    trusted_eligibility_id: TrustedCommitEligibilityId,
    backend_id: ExecutionBackendId,
    session_id: ExecutionSessionId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    receipt_id: ExecutionAttemptReceiptId,
    outcome: ExecutionAttemptOutcomeV1,
    backend_evidence_digest: [u8; 32],
    result_digest: [u8; 32],
}

impl CanonicalExecutionAttemptResultV1 {
    pub fn bind(
        intent: &ExecutionAttemptIntentV1,
        receipt: &ExecutionAttemptReceiptV1,
    ) -> Result<Self, ExecutionResultBindingError> {
        intent.validate()?;
        receipt.validate()?;
        if intent.id() != receipt.attempt_id() {
            return Err(ExecutionResultBindingError::ReceiptAttemptMismatch);
        }

        let result_id = CanonicalExecutionAttemptResultId(hash_result(
            intent,
            receipt.id(),
            receipt.outcome(),
            receipt.backend_evidence_digest(),
            receipt.result_digest(),
        ));

        Ok(Self {
            result_id,
            attempt_id: intent.id(),
            intent_capability_id: intent.capability_id(),
            trusted_eligibility_id: intent.trusted_eligibility_id(),
            backend_id: intent.backend_id(),
            session_id: intent.session_id(),
            subject_id: intent.subject_id(),
            target_realization_id: intent.target_realization_id(),
            distributed_context_id: intent.distributed_context_id(),
            trusted_epoch_id: intent.trusted_epoch_id(),
            receipt_id: receipt.id(),
            outcome: receipt.outcome(),
            backend_evidence_digest: receipt.backend_evidence_digest(),
            result_digest: receipt.result_digest(),
        })
    }

    pub fn id(&self) -> CanonicalExecutionAttemptResultId {
        self.result_id
    }

    pub fn attempt_id(&self) -> ExecutionAttemptId {
        self.attempt_id
    }

    pub fn intent_capability_id(&self) -> OneUseExecutionCapabilityId {
        self.intent_capability_id
    }

    pub fn trusted_eligibility_id(&self) -> TrustedCommitEligibilityId {
        self.trusted_eligibility_id
    }

    pub fn backend_id(&self) -> ExecutionBackendId {
        self.backend_id
    }

    pub fn session_id(&self) -> ExecutionSessionId {
        self.session_id
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

    pub fn trusted_epoch_id(&self) -> QualifiedTrustedCommitEpochId {
        self.trusted_epoch_id
    }

    pub fn receipt_id(&self) -> ExecutionAttemptReceiptId {
        self.receipt_id
    }

    pub fn outcome(&self) -> ExecutionAttemptOutcomeV1 {
        self.outcome
    }

    pub fn backend_evidence_digest(&self) -> [u8; 32] {
        self.backend_evidence_digest
    }

    pub fn result_digest(&self) -> [u8; 32] {
        self.result_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExecutionResultBindingError {
    #[error(transparent)]
    Artifact(#[from] ExecutionCapabilityError),
    #[error("execution receipt belongs to a different durable attempt intent")]
    ReceiptAttemptMismatch,
}

fn hash_result(
    intent: &ExecutionAttemptIntentV1,
    receipt_id: ExecutionAttemptReceiptId,
    outcome: ExecutionAttemptOutcomeV1,
    backend_evidence_digest: [u8; 32],
    result_digest: [u8; 32],
) -> [u8; 32] {
    let outcome_tag = [match outcome {
        ExecutionAttemptOutcomeV1::Succeeded => 1,
        ExecutionAttemptOutcomeV1::Failed => 2,
        ExecutionAttemptOutcomeV1::RolledBack => 3,
        ExecutionAttemptOutcomeV1::Indeterminate => 4,
    }];

    let mut hasher = blake3::Hasher::new();
    hasher.update(RESULT_DOMAIN);
    hasher.update(intent.id().as_bytes());
    hasher.update(intent.capability_id().as_bytes());
    hasher.update(intent.trusted_eligibility_id().as_bytes());
    hasher.update(intent.backend_id().as_bytes());
    hasher.update(intent.session_id().as_bytes());
    hasher.update(intent.subject_id().as_bytes());
    hasher.update(intent.target_realization_id().as_bytes());
    hasher.update(intent.distributed_context_id().as_bytes());
    hasher.update(intent.trusted_epoch_id().as_bytes());
    hasher.update(receipt_id.as_bytes());
    hasher.update(&outcome_tag);
    hasher.update(&backend_evidence_digest);
    hasher.update(&result_digest);
    *hasher.finalize().as_bytes()
}
