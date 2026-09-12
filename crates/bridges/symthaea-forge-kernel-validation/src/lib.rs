// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic Forge validation bound to parent-observed kernel isolation.
//!
//! This bridge does not implement another scorer. It composes two already-separate propositions:
//!
//! 1. the exact frozen model produced the exact label-blind response only after the parent observed
//!    and accepted the pre-exec kernel isolation state, and the exact sandbox process later exited;
//! 2. the existing evaluator-bound validation path reconstructs deterministic Brier from that exact
//!    request/response under the frozen study/support/coverage protocol.
//!
//! The result is a stronger validation/holdout proposition without changing model fitting, scoring,
//! holdout access, search policy, or promotion authority.

use serde::Serialize;
use symthaea_algorithms::ContentId;
use symthaea_forge::{
    ForgeProposalCorpusManifest, ForgeProposalEvaluationRequest, ForgeProposalEvaluationResponse,
    ForgeProposalEvaluatorHoldoutPermit, ForgeProposalEvaluatorValidationError,
    ForgeProposalEvaluatorValidationReceipt, ForgeProposalExecutableModelBinding,
    ForgeProposalFitPermit, ForgeProposalFrozenModel, ForgeProposalHoldoutSeal,
    ForgeProposalMetricScore, ForgeProposalStudySpec, ForgeProposalValidationCoverageReceipt,
    ForgeProposalValidationCoverageSpec, ForgeProposalValidationGateSpec,
    ForgeProposalValidationScorePermit, ForgeProposalValidationSet,
    ForgeProposalValidationTargetSet,
};
use symthaea_forge_linux_kernel_attestation::{
    KernelIsolationGate, KernelSandboxObservation,
};
use symthaea_forge_linux_observed_exec::{
    KernelGatedEvaluatorPolicy, KernelGatedExecutionReceipt, ObservedEvaluatorError,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KernelGatedValidationError {
    #[error(transparent)]
    Execution(#[from] ObservedEvaluatorError),
    #[error(transparent)]
    Validation(#[from] ForgeProposalEvaluatorValidationError),
    #[error("kernel-gated validation receipt does not bind the supplied execution/validation evidence")]
    ValidationScopeMismatch,
    #[error("kernel-gated validation receipt identity does not match canonical fields")]
    ValidationIdentityMismatch,
    #[error("kernel-gated holdout permit does not bind the supplied execution/validation evidence")]
    HoldoutScopeMismatch,
    #[error("kernel-gated holdout permit identity does not match canonical fields")]
    HoldoutIdentityMismatch,
}

/// Strong validation proposition: the deterministic Brier result is bound to the exact response
/// produced by an evaluator that remained blocked until the parent-side kernel gate passed.
#[derive(Debug, Clone, Serialize)]
pub struct KernelGatedModelValidationReceipt {
    id: ContentId,
    kernel_policy_id: ContentId,
    kernel_execution_receipt_id: ContentId,
    kernel_observation_id: ContentId,
    kernel_gate_id: ContentId,
    executable_model_binding_id: ContentId,
    evaluator_validation_receipt: ForgeProposalEvaluatorValidationReceipt,
}

impl KernelGatedModelValidationReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn record_brier(
        kernel_policy: &KernelGatedEvaluatorPolicy,
        kernel_execution: &KernelGatedExecutionReceipt,
        kernel_observation: &KernelSandboxObservation,
        kernel_gate: &KernelIsolationGate,
        binding: &ForgeProposalExecutableModelBinding,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        validation_gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
    ) -> Result<Self, KernelGatedValidationError> {
        kernel_execution.validate_for(
            kernel_policy,
            binding,
            model,
            request,
            response,
            kernel_observation,
            kernel_gate,
        )?;
        let evaluator_validation_receipt = ForgeProposalEvaluatorValidationReceipt::record_brier(
            manifest,
            study,
            validation_gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
            response,
        )?;
        if kernel_execution.kernel_observation_id() != kernel_observation.id()
            || kernel_execution.kernel_gate_id() != kernel_gate.id()
            || evaluator_validation_receipt.request_id() != request.id()
            || evaluator_validation_receipt.response_id() != response.id()
            || evaluator_validation_receipt.execution_context_id()
                != response.execution_context_id()
        {
            return Err(KernelGatedValidationError::ValidationScopeMismatch);
        }
        let id = derive_validation_id(
            kernel_policy.id(),
            kernel_execution.id(),
            kernel_observation.id(),
            kernel_gate.id(),
            binding.id(),
            evaluator_validation_receipt.id(),
        );
        Ok(Self {
            id,
            kernel_policy_id: kernel_policy.id().clone(),
            kernel_execution_receipt_id: kernel_execution.id().clone(),
            kernel_observation_id: kernel_observation.id().clone(),
            kernel_gate_id: kernel_gate.id().clone(),
            executable_model_binding_id: binding.id().clone(),
            evaluator_validation_receipt,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn kernel_policy_id(&self) -> &ContentId { &self.kernel_policy_id }
    pub fn kernel_execution_receipt_id(&self) -> &ContentId {
        &self.kernel_execution_receipt_id
    }
    pub fn kernel_observation_id(&self) -> &ContentId { &self.kernel_observation_id }
    pub fn kernel_gate_id(&self) -> &ContentId { &self.kernel_gate_id }
    pub fn executable_model_binding_id(&self) -> &ContentId {
        &self.executable_model_binding_id
    }
    pub fn evaluator_validation_receipt_id(&self) -> &ContentId {
        self.evaluator_validation_receipt.id()
    }
    pub fn model_id(&self) -> &ContentId { self.evaluator_validation_receipt.model_id() }
    pub fn primary_score(&self) -> &ForgeProposalMetricScore {
        self.evaluator_validation_receipt.primary_score()
    }
    pub fn passed(&self) -> bool { self.evaluator_validation_receipt.passed() }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        kernel_policy: &KernelGatedEvaluatorPolicy,
        kernel_execution: &KernelGatedExecutionReceipt,
        kernel_observation: &KernelSandboxObservation,
        kernel_gate: &KernelIsolationGate,
        binding: &ForgeProposalExecutableModelBinding,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        validation_gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
    ) -> Result<(), KernelGatedValidationError> {
        kernel_execution.validate_for(
            kernel_policy,
            binding,
            model,
            request,
            response,
            kernel_observation,
            kernel_gate,
        )?;
        self.evaluator_validation_receipt.validate_for(
            manifest,
            study,
            validation_gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
            response,
        )?;
        if self.kernel_policy_id != *kernel_policy.id()
            || self.kernel_execution_receipt_id != *kernel_execution.id()
            || self.kernel_observation_id != *kernel_observation.id()
            || self.kernel_gate_id != *kernel_gate.id()
            || self.executable_model_binding_id != *binding.id()
            || kernel_execution.kernel_observation_id() != kernel_observation.id()
            || kernel_execution.kernel_gate_id() != kernel_gate.id()
            || self.evaluator_validation_receipt.request_id() != request.id()
            || self.evaluator_validation_receipt.response_id() != response.id()
            || self.evaluator_validation_receipt.execution_context_id()
                != response.execution_context_id()
        {
            return Err(KernelGatedValidationError::ValidationScopeMismatch);
        }
        let expected = derive_validation_id(
            &self.kernel_policy_id,
            &self.kernel_execution_receipt_id,
            &self.kernel_observation_id,
            &self.kernel_gate_id,
            &self.executable_model_binding_id,
            self.evaluator_validation_receipt.id(),
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(KernelGatedValidationError::ValidationIdentityMismatch)
        }
    }

    fn inner(&self) -> &ForgeProposalEvaluatorValidationReceipt {
        &self.evaluator_validation_receipt
    }
}

fn derive_validation_id(
    policy_id: &ContentId,
    execution_id: &ContentId,
    observation_id: &ContentId,
    gate_id: &ContentId,
    binding_id: &ContentId,
    evaluator_validation_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-kernel-gated-model-validation-receipt.v1",
        [
            policy_id.as_str().as_bytes(),
            execution_id.as_str().as_bytes(),
            observation_id.as_str().as_bytes(),
            gate_id.as_str().as_bytes(),
            binding_id.as_str().as_bytes(),
            evaluator_validation_id.as_str().as_bytes(),
        ],
    )
}

/// Identity-only holdout prerequisite carrying the exact kernel-gated validation lineage.
#[derive(Debug, Clone, Serialize)]
pub struct KernelGatedModelHoldoutPermit {
    id: ContentId,
    kernel_validation_receipt_id: ContentId,
    kernel_policy_id: ContentId,
    kernel_execution_receipt_id: ContentId,
    kernel_gate_id: ContentId,
    inner: ForgeProposalEvaluatorHoldoutPermit,
}

impl KernelGatedModelHoldoutPermit {
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        kernel_policy: &KernelGatedEvaluatorPolicy,
        kernel_execution: &KernelGatedExecutionReceipt,
        kernel_observation: &KernelSandboxObservation,
        kernel_gate: &KernelIsolationGate,
        binding: &ForgeProposalExecutableModelBinding,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        validation_gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
        receipt: &KernelGatedModelValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<Self, KernelGatedValidationError> {
        receipt.validate_for(
            kernel_policy,
            kernel_execution,
            kernel_observation,
            kernel_gate,
            binding,
            manifest,
            study,
            validation_gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
            response,
        )?;
        let inner = ForgeProposalEvaluatorHoldoutPermit::issue(
            manifest,
            study,
            validation_gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
            response,
            receipt.inner(),
            holdout,
        )?;
        let id = derive_holdout_id(
            inner.id(),
            receipt.id(),
            kernel_policy.id(),
            kernel_execution.id(),
            kernel_gate.id(),
        );
        Ok(Self {
            id,
            kernel_validation_receipt_id: receipt.id().clone(),
            kernel_policy_id: kernel_policy.id().clone(),
            kernel_execution_receipt_id: kernel_execution.id().clone(),
            kernel_gate_id: kernel_gate.id().clone(),
            inner,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn kernel_validation_receipt_id(&self) -> &ContentId {
        &self.kernel_validation_receipt_id
    }
    pub fn kernel_policy_id(&self) -> &ContentId { &self.kernel_policy_id }
    pub fn kernel_execution_receipt_id(&self) -> &ContentId {
        &self.kernel_execution_receipt_id
    }
    pub fn kernel_gate_id(&self) -> &ContentId { &self.kernel_gate_id }
    pub fn model_id(&self) -> &ContentId { self.inner.model_id() }
    pub fn holdout_seal_id(&self) -> &ContentId { self.inner.holdout_seal_id() }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        kernel_policy: &KernelGatedEvaluatorPolicy,
        kernel_execution: &KernelGatedExecutionReceipt,
        kernel_observation: &KernelSandboxObservation,
        kernel_gate: &KernelIsolationGate,
        binding: &ForgeProposalExecutableModelBinding,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        validation_gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
        receipt: &KernelGatedModelValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<(), KernelGatedValidationError> {
        receipt.validate_for(
            kernel_policy,
            kernel_execution,
            kernel_observation,
            kernel_gate,
            binding,
            manifest,
            study,
            validation_gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
            response,
        )?;
        self.inner.validate_for(
            manifest,
            study,
            validation_gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
            response,
            receipt.inner(),
            holdout,
        )?;
        if self.kernel_validation_receipt_id != *receipt.id()
            || self.kernel_policy_id != *kernel_policy.id()
            || self.kernel_execution_receipt_id != *kernel_execution.id()
            || self.kernel_gate_id != *kernel_gate.id()
            || self.inner.model_id() != model.id()
            || self.inner.holdout_seal_id() != holdout.id()
        {
            return Err(KernelGatedValidationError::HoldoutScopeMismatch);
        }
        let expected = derive_holdout_id(
            self.inner.id(),
            &self.kernel_validation_receipt_id,
            &self.kernel_policy_id,
            &self.kernel_execution_receipt_id,
            &self.kernel_gate_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(KernelGatedValidationError::HoldoutIdentityMismatch)
        }
    }
}

fn derive_holdout_id(
    inner_id: &ContentId,
    validation_receipt_id: &ContentId,
    policy_id: &ContentId,
    execution_id: &ContentId,
    gate_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-kernel-gated-model-holdout-permit.v1",
        [
            inner_id.as_str().as_bytes(),
            validation_receipt_id.as_str().as_bytes(),
            policy_id.as_str().as_bytes(),
            execution_id.as_str().as_bytes(),
            gate_id.as_str().as_bytes(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_identity_changes_with_kernel_gate() {
        let policy = ContentId::derive("test", [b"policy".as_slice()]);
        let execution = ContentId::derive("test", [b"execution".as_slice()]);
        let observation = ContentId::derive("test", [b"observation".as_slice()]);
        let gate_a = ContentId::derive("test", [b"gate-a".as_slice()]);
        let gate_b = ContentId::derive("test", [b"gate-b".as_slice()]);
        let binding = ContentId::derive("test", [b"binding".as_slice()]);
        let inner = ContentId::derive("test", [b"inner".as_slice()]);
        assert_ne!(
            derive_validation_id(&policy, &execution, &observation, &gate_a, &binding, &inner),
            derive_validation_id(&policy, &execution, &observation, &gate_b, &binding, &inner)
        );
    }
}
