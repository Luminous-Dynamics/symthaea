// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact restart rebind for persisted effect-scoped execution metadata.
//!
//! A serialized `EffectScopedExecutionEnvelopeV1` is audit/recovery material only.
//! After restart it must be rebound to the exact durable A -> B intent, exact
//! pre-authorized effect plan, and exact reconstructed non-Serde effect authorization.
//! The attempt-specific contract is then rederived from those live inputs.
//!
//! `SerializedEnvelope != ReboundEffectScope != ExecutionAuthority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::effect_scoped_execution::{
    EffectScopedExecutionEnvelopeId, EffectScopedExecutionEnvelopeV1, EffectScopedExecutionError,
};
use crate::execution_capability::ExecutionAttemptId;
use crate::external_effect_authority::{
    ExternalEffectAuthorityError, ExternalEffectPlanId, ExternalEffectPlanV1,
    QualifiedExternalEffectAuthorizationId, QualifiedExternalEffectAuthorizationV1,
};
use crate::external_effects::{
    ExternalEffectContractId, ExternalEffectContractV1, ExternalEffectError,
};
use crate::scope::ContinuitySubjectId;
use crate::transition_lineage::{
    KnownGoodBoundExecutionAttemptIntentV1, KnownGoodExecutionIntentId,
    KnownGoodTransitionLineageError,
};
use crate::witness::TargetRealizationId;

const REBOUND_DOMAIN: &[u8] = b"symthaea.continuity.rebound-effect-scope.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ReboundEffectScopeId([u8; 32]);
impl ReboundEffectScopeId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Non-Serde proof that one persisted envelope has been rebound to the exact live
/// lineage and authorization objects required by the pre-mutation theorem.
#[derive(Debug, Clone)]
pub struct ReboundEffectScopedExecutionV1 {
    rebound_id: ReboundEffectScopeId,
    envelope: EffectScopedExecutionEnvelopeV1,
    contract: ExternalEffectContractV1,
    known_good_intent_id: KnownGoodExecutionIntentId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    plan_id: ExternalEffectPlanId,
    authorization_id: QualifiedExternalEffectAuthorizationId,
    contract_id: ExternalEffectContractId,
    coverage_manifest_digest: [u8; 32],
}

impl ReboundEffectScopedExecutionV1 {
    pub fn rebind(
        envelope: EffectScopedExecutionEnvelopeV1,
        intent: &KnownGoodBoundExecutionAttemptIntentV1,
        plan: &ExternalEffectPlanV1,
        authorization: &QualifiedExternalEffectAuthorizationV1,
    ) -> Result<Self, EffectScopeRebindError> {
        envelope.validate()?;
        intent.validate()?;
        plan.validate()?;
        let lineage = intent.lineage();

        if envelope.known_good_intent_id() != intent.id()
            || envelope.attempt_id() != intent.attempt_id()
            || envelope.subject_id() != lineage.subject_id()
            || envelope.source_realization_id() != lineage.source_realization_id()
            || envelope.target_realization_id() != lineage.target_realization_id()
            || envelope.distributed_context_id() != lineage.distributed_context_id()
            || envelope.effect_plan_id() != plan.id()
            || envelope.effect_authorization_id() != authorization.id()
            || envelope.coverage_manifest_digest() != plan.coverage_manifest_digest()
        {
            return Err(EffectScopeRebindError::EnvelopeLiveContextMismatch);
        }

        if plan.subject_id() != lineage.subject_id()
            || plan.target_realization_id() != lineage.target_realization_id()
            || plan.distributed_context_id() != lineage.distributed_context_id()
            || plan.commit_time_unix_ms() != lineage.commit_time_unix_ms()
            || authorization.plan_id() != plan.id()
            || authorization.subject_id() != lineage.subject_id()
            || authorization.target_realization_id() != lineage.target_realization_id()
            || authorization.distributed_context_id() != lineage.distributed_context_id()
            || authorization.coverage_manifest_digest() != plan.coverage_manifest_digest()
            || authorization.authorized_at_unix_ms() != lineage.commit_time_unix_ms()
        {
            return Err(EffectScopeRebindError::AuthorizationLiveContextMismatch);
        }

        let contract = ExternalEffectContractV1::new(
            intent,
            plan.coverage_manifest_digest(),
            plan.obligations().to_vec(),
        )?;
        if contract.id() != envelope.effect_contract_id() {
            return Err(EffectScopeRebindError::DerivedContractMismatch);
        }

        let rebound_id = ReboundEffectScopeId(hash_rebound(
            envelope.id(),
            intent.id(),
            intent.attempt_id(),
            lineage.subject_id(),
            lineage.source_realization_id(),
            lineage.target_realization_id(),
            plan.id(),
            authorization.id(),
            contract.id(),
            plan.coverage_manifest_digest(),
        ));
        Ok(Self {
            rebound_id,
            envelope,
            contract,
            known_good_intent_id: intent.id(),
            attempt_id: intent.attempt_id(),
            subject_id: lineage.subject_id(),
            source_realization_id: lineage.source_realization_id(),
            target_realization_id: lineage.target_realization_id(),
            plan_id: plan.id(),
            authorization_id: authorization.id(),
            contract_id: contract.id(),
            coverage_manifest_digest: plan.coverage_manifest_digest(),
        })
    }

    pub fn id(&self) -> ReboundEffectScopeId { self.rebound_id }
    pub fn envelope(&self) -> &EffectScopedExecutionEnvelopeV1 { &self.envelope }
    pub fn envelope_id(&self) -> EffectScopedExecutionEnvelopeId { self.envelope.id() }
    pub fn contract(&self) -> &ExternalEffectContractV1 { &self.contract }
    pub fn contract_id(&self) -> ExternalEffectContractId { self.contract_id }
    pub fn known_good_intent_id(&self) -> KnownGoodExecutionIntentId { self.known_good_intent_id }
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.attempt_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.source_realization_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn plan_id(&self) -> ExternalEffectPlanId { self.plan_id }
    pub fn authorization_id(&self) -> QualifiedExternalEffectAuthorizationId { self.authorization_id }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EffectScopeRebindError {
    #[error(transparent)]
    Envelope(#[from] EffectScopedExecutionError),
    #[error(transparent)]
    EffectAuthority(#[from] ExternalEffectAuthorityError),
    #[error(transparent)]
    ExternalEffect(#[from] ExternalEffectError),
    #[error(transparent)]
    TransitionLineage(#[from] KnownGoodTransitionLineageError),
    #[error("persisted effect envelope does not match exact live A -> B intent/plan/authorization")]
    EnvelopeLiveContextMismatch,
    #[error("reconstructed effect authorization does not match exact live A -> B transaction")]
    AuthorizationLiveContextMismatch,
    #[error("freshly derived attempt-specific effect contract differs from persisted envelope")]
    DerivedContractMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_rebound(
    envelope_id: EffectScopedExecutionEnvelopeId,
    known_good_intent_id: KnownGoodExecutionIntentId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    source_id: TargetRealizationId,
    target_id: TargetRealizationId,
    plan_id: ExternalEffectPlanId,
    authorization_id: QualifiedExternalEffectAuthorizationId,
    contract_id: ExternalEffectContractId,
    coverage_manifest_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(REBOUND_DOMAIN);
    hasher.update(envelope_id.as_bytes());
    hasher.update(known_good_intent_id.as_bytes());
    hasher.update(attempt_id.as_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(source_id.as_bytes());
    hasher.update(target_id.as_bytes());
    hasher.update(plan_id.as_bytes());
    hasher.update(authorization_id.as_bytes());
    hasher.update(contract_id.as_bytes());
    hasher.update(&coverage_manifest_digest);
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebound_domain_is_distinct_from_envelope_domain() {
        assert_ne!(
            REBOUND_DOMAIN,
            b"symthaea.continuity.effect-scoped-execution-envelope.v1\0"
        );
    }
}
