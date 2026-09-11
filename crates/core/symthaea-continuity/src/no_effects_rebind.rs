// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Restart rebind for the serialized no-effects execution envelope.
//!
//! A self-consistent persisted envelope is audit material, not authority. Rebind
//! requires the exact durable A -> B intent, exact owner-authorized no-effects
//! declaration, exact reconstructed no-effects authorization, exact independent
//! empty-effect coverage proof, and exact backend implementation. The canonical
//! envelope identity is then rederived from those live proof objects.
//!
//! `SerializedEnvelope != ReboundNoEffectsWorld != ProtectedCommitment`.

use thiserror::Error;

use crate::execution_capability::{ExecutionBackendId, ExecutionBackendProfileV1, ExecutionCapabilityError};
use crate::no_effects_execution::{
    NoEffectsExecutionEnvelopeId, NoEffectsExecutionEnvelopeV1, NoEffectsExecutionError,
};
use crate::no_external_effects::{
    NoExternalEffectsDeclarationId, NoExternalEffectsDeclarationV1, NoExternalEffectsError,
    QualifiedNoExternalEffectsAuthorizationId, QualifiedNoExternalEffectsAuthorizationV1,
    QualifiedNoExternalEffectsCoverageId, QualifiedNoExternalEffectsCoverageV1,
};
use crate::scope::ContinuitySubjectId;
use crate::transition_lineage::{
    KnownGoodBoundExecutionAttemptIntentV1, KnownGoodExecutionIntentId,
    KnownGoodTransitionLineageError,
};
use crate::witness::TargetRealizationId;

const ENVELOPE_DOMAIN: &[u8] = b"symthaea.continuity.no-effects-execution-envelope.v1\0";
const REBIND_DOMAIN: &[u8] = b"symthaea.continuity.rebound-no-effects-execution.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReboundNoEffectsExecutionId([u8; 32]);
impl ReboundNoEffectsExecutionId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Non-Serde proof that one persisted no-effects envelope has been reconstructed
/// from the exact live semantic/authority/verifier/backend parents after restart.
///
/// This grants no execution or recovery authority. The exact protected pre-mutation
/// commitment remains a separate required proof downstream.
#[derive(Debug, Clone)]
pub struct ReboundNoEffectsExecutionV1 {
    rebind_id: ReboundNoEffectsExecutionId,
    envelope: NoEffectsExecutionEnvelopeV1,
    known_good_intent_id: KnownGoodExecutionIntentId,
    declaration_id: NoExternalEffectsDeclarationId,
    authorization_id: QualifiedNoExternalEffectsAuthorizationId,
    coverage_id: QualifiedNoExternalEffectsCoverageId,
    backend_id: ExecutionBackendId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
}

impl ReboundNoEffectsExecutionV1 {
    pub fn rebind(
        envelope: NoEffectsExecutionEnvelopeV1,
        intent: &KnownGoodBoundExecutionAttemptIntentV1,
        declaration: &NoExternalEffectsDeclarationV1,
        authorization: &QualifiedNoExternalEffectsAuthorizationV1,
        coverage: &QualifiedNoExternalEffectsCoverageV1,
        backend: &ExecutionBackendProfileV1,
    ) -> Result<Self, NoEffectsRebindError> {
        envelope.validate()?;
        intent.validate()?;
        declaration.validate()?;
        backend.validate()?;

        let lineage = intent.lineage();
        let execution_intent = intent.execution_intent();
        if declaration.subject_id() != lineage.subject_id()
            || declaration.target_realization_id() != lineage.target_realization_id()
            || declaration.distributed_context_id() != lineage.distributed_context_id()
            || declaration.commit_time_unix_ms() != lineage.commit_time_unix_ms()
            || declaration.backend_id() != backend.id()
            || declaration.backend_implementation_digest() != backend.implementation_digest()
            || declaration.backend_generation() != backend.backend_generation()
            || execution_intent.backend_id() != backend.id()
            || authorization.declaration_id() != declaration.id()
            || coverage.declaration_id() != declaration.id()
            || coverage.analyzed_at_unix_ms() != declaration.commit_time_unix_ms()
            || coverage.transaction_challenge() == [0; 32]
        {
            return Err(NoEffectsRebindError::LiveProofContextMismatch);
        }

        if envelope.known_good_intent_id() != intent.id()
            || envelope.attempt_id() != intent.attempt_id()
            || envelope.subject_id() != lineage.subject_id()
            || envelope.source_realization_id() != lineage.source_realization_id()
            || envelope.target_realization_id() != lineage.target_realization_id()
            || envelope.declaration_id() != declaration.id()
            || envelope.authorization_id() != authorization.id()
            || envelope.coverage_id() != coverage.id()
            || envelope.coverage_manifest_digest() != declaration.coverage_manifest_digest()
            || envelope.coverage_analyzed_at_unix_ms() != coverage.analyzed_at_unix_ms()
            || envelope.coverage_challenge() != coverage.transaction_challenge()
            || envelope.backend_id() != backend.id()
        {
            return Err(NoEffectsRebindError::EnvelopeLineageMismatch);
        }

        // Recompute the exact V1 envelope identity from live parents rather than
        // trusting duplicated serialized backend/material fields in the envelope.
        let expected_envelope = hash_envelope_from_live(
            intent,
            declaration,
            authorization,
            coverage,
            backend,
        );
        if envelope.id().as_bytes() != &expected_envelope {
            return Err(NoEffectsRebindError::EnvelopeCanonicalReconstructionMismatch);
        }

        let rebind_id = ReboundNoEffectsExecutionId(domain_hash_parts(
            REBIND_DOMAIN,
            &[
                envelope.id().as_bytes(), intent.id().as_bytes(), declaration.id().as_bytes(),
                authorization.id().as_bytes(), coverage.id().as_bytes(), backend.id().as_bytes(),
                &backend.implementation_digest(), &backend.backend_generation().to_le_bytes(),
            ],
        ));
        Ok(Self {
            rebind_id,
            envelope,
            known_good_intent_id: intent.id(),
            declaration_id: declaration.id(),
            authorization_id: authorization.id(),
            coverage_id: coverage.id(),
            backend_id: backend.id(),
            subject_id: lineage.subject_id(),
            source_realization_id: lineage.source_realization_id(),
            target_realization_id: lineage.target_realization_id(),
        })
    }

    pub fn id(&self) -> ReboundNoEffectsExecutionId { self.rebind_id }
    pub fn envelope(&self) -> &NoEffectsExecutionEnvelopeV1 { &self.envelope }
    pub fn envelope_id(&self) -> NoEffectsExecutionEnvelopeId { self.envelope.id() }
    pub fn known_good_intent_id(&self) -> KnownGoodExecutionIntentId { self.known_good_intent_id }
    pub fn declaration_id(&self) -> NoExternalEffectsDeclarationId { self.declaration_id }
    pub fn authorization_id(&self) -> QualifiedNoExternalEffectsAuthorizationId { self.authorization_id }
    pub fn coverage_id(&self) -> QualifiedNoExternalEffectsCoverageId { self.coverage_id }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.source_realization_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NoEffectsRebindError {
    #[error(transparent)]
    Execution(#[from] NoEffectsExecutionError),
    #[error(transparent)]
    NoEffects(#[from] NoExternalEffectsError),
    #[error(transparent)]
    Backend(#[from] ExecutionCapabilityError),
    #[error(transparent)]
    Lineage(#[from] KnownGoodTransitionLineageError),
    #[error("live no-effects authority/verifier/backend proofs do not describe the exact A -> B world")]
    LiveProofContextMismatch,
    #[error("persisted no-effects envelope does not name the exact reconstructed live parents")]
    EnvelopeLineageMismatch,
    #[error("persisted no-effects envelope identity cannot be canonically reconstructed from live parents")]
    EnvelopeCanonicalReconstructionMismatch,
}

fn hash_envelope_from_live(
    intent: &KnownGoodBoundExecutionAttemptIntentV1,
    declaration: &NoExternalEffectsDeclarationV1,
    authorization: &QualifiedNoExternalEffectsAuthorizationV1,
    coverage: &QualifiedNoExternalEffectsCoverageV1,
    backend: &ExecutionBackendProfileV1,
) -> [u8; 32] {
    let lineage = intent.lineage();
    domain_hash_parts(
        ENVELOPE_DOMAIN,
        &[
            intent.id().as_bytes(), intent.attempt_id().as_bytes(), lineage.subject_id().as_bytes(),
            lineage.source_realization_id().as_bytes(), lineage.target_realization_id().as_bytes(),
            declaration.id().as_bytes(), authorization.id().as_bytes(), coverage.id().as_bytes(),
            &declaration.coverage_manifest_digest(), &coverage.analyzed_at_unix_ms().to_le_bytes(),
            &coverage.transaction_challenge(), backend.id().as_bytes(),
            &backend.implementation_digest(), &backend.backend_generation().to_le_bytes(),
        ],
    )
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
    fn rebind_domain_is_distinct_from_persisted_envelope_domain() {
        assert_ne!(REBIND_DOMAIN, ENVELOPE_DOMAIN);
    }
}
