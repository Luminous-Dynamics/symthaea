// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic validation and holdout prerequisites bound to an exact Linux-isolated evaluator run.
//!
//! This composes the bubblewrap execution theorem with the existing evaluator-bound deterministic
//! Brier theorem. It does not change scoring semantics; it strengthens runtime provenance so the
//! validation receipt proves that the response being scored came from the exact isolated execution
//! whose model payload, namespaces, filesystem view, transport and limits are content-addressed.

use crate::proposal_corpus::{
    ForgeProposalCorpusManifest, ForgeProposalHoldoutSeal, ForgeProposalValidationSet,
};
use crate::proposal_evaluator_protocol::{
    ForgeProposalEvaluationRequest, ForgeProposalEvaluationResponse,
};
use crate::proposal_evaluator_validation::{
    ForgeProposalEvaluatorHoldoutPermit, ForgeProposalEvaluatorValidationError,
    ForgeProposalEvaluatorValidationReceipt,
};
use crate::proposal_executable_model::{
    ForgeProposalExecutableModelBinding, ForgeProposalExecutableModelError,
};
use crate::proposal_linux_isolation::{
    ForgeProposalBubblewrapExecutionReceipt, ForgeProposalBubblewrapIsolationPolicy,
    ForgeProposalLinuxIsolationError,
};
use crate::proposal_model::{
    ForgeProposalFrozenModel, ForgeProposalMetricScore, ForgeProposalValidationGateSpec,
};
use crate::proposal_study::ForgeProposalStudySpec;
use crate::proposal_support::ForgeProposalFitPermit;
use crate::proposal_validation_blind::ForgeProposalValidationTargetSet;
use crate::proposal_validation_coverage::{
    ForgeProposalValidationCoverageReceipt, ForgeProposalValidationCoverageSpec,
    ForgeProposalValidationScorePermit,
};
use serde::Serialize;
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalIsolatedValidationError {
    #[error(transparent)]
    Isolation(#[from] ForgeProposalLinuxIsolationError),
    #[error(transparent)]
    ExecutableModel(#[from] ForgeProposalExecutableModelError),
    #[error(transparent)]
    Validation(#[from] ForgeProposalEvaluatorValidationError),
    #[error("isolated validation receipt does not bind the supplied model/isolation evidence")]
    ValidationScopeMismatch,
    #[error("isolated validation receipt identity does not match canonical fields")]
    ValidationIdentityMismatch,
    #[error("isolated holdout permit does not bind the supplied model/isolation evidence")]
    HoldoutScopeMismatch,
    #[error("isolated holdout permit identity does not match canonical fields")]
    HoldoutIdentityMismatch,
}

/// Strongest Linux v1 validation proposition: this deterministic Brier result is the score of the
/// exact response produced by the exact bubblewrap-isolated execution of the frozen model payload.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalIsolatedModelValidationReceipt {
    id: ContentId,
    isolation_policy_id: ContentId,
    isolation_execution_receipt_id: ContentId,
    executable_model_binding_id: ContentId,
    evaluator_validation_receipt: ForgeProposalEvaluatorValidationReceipt,
}

impl ForgeProposalIsolatedModelValidationReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn record_brier(
        isolation_policy: &ForgeProposalBubblewrapIsolationPolicy,
        isolation_receipt: &ForgeProposalBubblewrapExecutionReceipt,
        binding: &ForgeProposalExecutableModelBinding,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
    ) -> Result<Self, ForgeProposalIsolatedValidationError> {
        binding.validate_for(model, request.protocol())?;
        isolation_receipt.validate_for(
            isolation_policy,
            binding,
            model,
            request,
            response,
        )?;
        let evaluator_validation_receipt = ForgeProposalEvaluatorValidationReceipt::record_brier(
            manifest,
            study,
            gate,
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
        if isolation_receipt.model_id() != model.id()
            || isolation_receipt.model_payload_id() != model.model_payload_id()
            || isolation_receipt.executable_model_binding_id() != binding.id()
            || isolation_receipt.request_id() != request.id()
            || isolation_receipt.response_id() != response.id()
            || isolation_receipt.execution_context_id() != response.execution_context_id()
            || evaluator_validation_receipt.request_id() != request.id()
            || evaluator_validation_receipt.response_id() != response.id()
            || evaluator_validation_receipt.execution_context_id() != response.execution_context_id()
        {
            return Err(ForgeProposalIsolatedValidationError::ValidationScopeMismatch);
        }
        let id = derive_validation_receipt_id(
            isolation_policy.id(),
            isolation_receipt.id(),
            binding.id(),
            evaluator_validation_receipt.id(),
        );
        Ok(Self {
            id,
            isolation_policy_id: isolation_policy.id().clone(),
            isolation_execution_receipt_id: isolation_receipt.id().clone(),
            executable_model_binding_id: binding.id().clone(),
            evaluator_validation_receipt,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn isolation_policy_id(&self) -> &ContentId { &self.isolation_policy_id }
    pub fn isolation_execution_receipt_id(&self) -> &ContentId {
        &self.isolation_execution_receipt_id
    }
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
        isolation_policy: &ForgeProposalBubblewrapIsolationPolicy,
        isolation_receipt: &ForgeProposalBubblewrapExecutionReceipt,
        binding: &ForgeProposalExecutableModelBinding,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
    ) -> Result<(), ForgeProposalIsolatedValidationError> {
        binding.validate_for(model, request.protocol())?;
        isolation_receipt.validate_for(
            isolation_policy,
            binding,
            model,
            request,
            response,
        )?;
        self.evaluator_validation_receipt.validate_for(
            manifest,
            study,
            gate,
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
        if self.isolation_policy_id != *isolation_policy.id()
            || self.isolation_execution_receipt_id != *isolation_receipt.id()
            || self.executable_model_binding_id != *binding.id()
            || isolation_receipt.model_id() != model.id()
            || isolation_receipt.model_payload_id() != model.model_payload_id()
            || isolation_receipt.request_id() != request.id()
            || isolation_receipt.response_id() != response.id()
            || isolation_receipt.execution_context_id() != response.execution_context_id()
            || self.evaluator_validation_receipt.request_id() != request.id()
            || self.evaluator_validation_receipt.response_id() != response.id()
            || self.evaluator_validation_receipt.execution_context_id()
                != response.execution_context_id()
        {
            return Err(ForgeProposalIsolatedValidationError::ValidationScopeMismatch);
        }
        let expected = derive_validation_receipt_id(
            &self.isolation_policy_id,
            &self.isolation_execution_receipt_id,
            &self.executable_model_binding_id,
            self.evaluator_validation_receipt.id(),
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalIsolatedValidationError::ValidationIdentityMismatch)
        }
    }

    pub(crate) fn evaluator_validation_receipt(&self) -> &ForgeProposalEvaluatorValidationReceipt {
        &self.evaluator_validation_receipt
    }
}

fn derive_validation_receipt_id(
    isolation_policy_id: &ContentId,
    isolation_execution_receipt_id: &ContentId,
    binding_id: &ContentId,
    evaluator_validation_receipt_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-isolated-model-validation-receipt.v1",
        [
            isolation_policy_id.as_str().as_bytes(),
            isolation_execution_receipt_id.as_str().as_bytes(),
            binding_id.as_str().as_bytes(),
            evaluator_validation_receipt_id.as_str().as_bytes(),
        ],
    )
}

/// Identity-only holdout prerequisite whose validation proof is bound to the exact isolated model
/// execution. This does not expose or score holdout observations.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalIsolatedModelHoldoutPermit {
    id: ContentId,
    isolated_validation_receipt_id: ContentId,
    isolation_policy_id: ContentId,
    isolation_execution_receipt_id: ContentId,
    inner: ForgeProposalEvaluatorHoldoutPermit,
}

impl ForgeProposalIsolatedModelHoldoutPermit {
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        isolation_policy: &ForgeProposalBubblewrapIsolationPolicy,
        isolation_receipt: &ForgeProposalBubblewrapExecutionReceipt,
        binding: &ForgeProposalExecutableModelBinding,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
        receipt: &ForgeProposalIsolatedModelValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<Self, ForgeProposalIsolatedValidationError> {
        receipt.validate_for(
            isolation_policy,
            isolation_receipt,
            binding,
            manifest,
            study,
            gate,
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
            gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
            response,
            receipt.evaluator_validation_receipt(),
            holdout,
        )?;
        let id = derive_holdout_permit_id(
            inner.id(),
            receipt.id(),
            isolation_policy.id(),
            isolation_receipt.id(),
        );
        Ok(Self {
            id,
            isolated_validation_receipt_id: receipt.id().clone(),
            isolation_policy_id: isolation_policy.id().clone(),
            isolation_execution_receipt_id: isolation_receipt.id().clone(),
            inner,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn isolated_validation_receipt_id(&self) -> &ContentId {
        &self.isolated_validation_receipt_id
    }
    pub fn isolation_policy_id(&self) -> &ContentId { &self.isolation_policy_id }
    pub fn isolation_execution_receipt_id(&self) -> &ContentId {
        &self.isolation_execution_receipt_id
    }
    pub fn model_id(&self) -> &ContentId { self.inner.model_id() }
    pub fn holdout_seal_id(&self) -> &ContentId { self.inner.holdout_seal_id() }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        isolation_policy: &ForgeProposalBubblewrapIsolationPolicy,
        isolation_receipt: &ForgeProposalBubblewrapExecutionReceipt,
        binding: &ForgeProposalExecutableModelBinding,
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
        receipt: &ForgeProposalIsolatedModelValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<(), ForgeProposalIsolatedValidationError> {
        receipt.validate_for(
            isolation_policy,
            isolation_receipt,
            binding,
            manifest,
            study,
            gate,
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
            gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            targets,
            request,
            response,
            receipt.evaluator_validation_receipt(),
            holdout,
        )?;
        if self.isolated_validation_receipt_id != *receipt.id()
            || self.isolation_policy_id != *isolation_policy.id()
            || self.isolation_execution_receipt_id != *isolation_receipt.id()
            || self.inner.model_id() != model.id()
            || self.inner.holdout_seal_id() != holdout.id()
        {
            return Err(ForgeProposalIsolatedValidationError::HoldoutScopeMismatch);
        }
        let expected = derive_holdout_permit_id(
            self.inner.id(),
            &self.isolated_validation_receipt_id,
            &self.isolation_policy_id,
            &self.isolation_execution_receipt_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalIsolatedValidationError::HoldoutIdentityMismatch)
        }
    }
}

fn derive_holdout_permit_id(
    inner_permit_id: &ContentId,
    isolated_validation_receipt_id: &ContentId,
    isolation_policy_id: &ContentId,
    isolation_execution_receipt_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-isolated-model-holdout-permit.v1",
        [
            inner_permit_id.as_str().as_bytes(),
            isolated_validation_receipt_id.as_str().as_bytes(),
            isolation_policy_id.as_str().as_bytes(),
            isolation_execution_receipt_id.as_str().as_bytes(),
        ],
    )
}
