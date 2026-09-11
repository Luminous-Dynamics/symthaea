// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation/holdout evidence bound to the exact executable payload of the frozen model.
//!
//! This composes the executable-model theorem with launch-bound deterministic validation so the
//! strongest v1 statement becomes:
//! frozen model payload == exact executable bytes launched -> exact blind response -> exact Brier.

use crate::proposal_corpus::{
    ForgeProposalCorpusManifest, ForgeProposalHoldoutSeal, ForgeProposalValidationSet,
};
use crate::proposal_evaluator_execution::ForgeProposalEvaluatorLaunchPolicy;
use crate::proposal_evaluator_protocol::{
    ForgeProposalEvaluationRequest, ForgeProposalEvaluationResponse,
};
use crate::proposal_executable_model::{
    ForgeProposalExecutableModelBinding, ForgeProposalExecutableModelError,
    ForgeProposalExecutableModelLaunchReceipt,
};
use crate::proposal_launch_validation::{
    ForgeProposalLaunchValidationError, ForgeProposalLaunchedEvaluatorHoldoutPermit,
    ForgeProposalLaunchedEvaluatorValidationReceipt,
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
pub enum ForgeProposalModelValidationError {
    #[error(transparent)]
    ExecutableModel(#[from] ForgeProposalExecutableModelError),
    #[error(transparent)]
    LaunchValidation(#[from] ForgeProposalLaunchValidationError),
    #[error("model-bound validation receipt does not bind the supplied executable-model evidence")]
    ValidationScopeMismatch,
    #[error("model-bound validation receipt identity does not match canonical fields")]
    ValidationIdentityMismatch,
    #[error("model-bound holdout permit does not bind the supplied executable-model evidence")]
    HoldoutScopeMismatch,
    #[error("model-bound holdout permit identity does not match canonical fields")]
    HoldoutIdentityMismatch,
}

/// Strongest v1 validation receipt: the deterministic validation score is bound to the launch whose
/// exact executable bytes are the frozen model payload.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalExecutableModelValidationReceipt {
    id: ContentId,
    executable_model_binding_id: ContentId,
    executable_model_launch_receipt_id: ContentId,
    launched_validation_receipt: ForgeProposalLaunchedEvaluatorValidationReceipt,
}

impl ForgeProposalExecutableModelValidationReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn record_brier(
        binding: &ForgeProposalExecutableModelBinding,
        model_launch_receipt: &ForgeProposalExecutableModelLaunchReceipt,
        launch_policy: &ForgeProposalEvaluatorLaunchPolicy,
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
    ) -> Result<Self, ForgeProposalModelValidationError> {
        binding.validate_for(model, request.protocol())?;
        model_launch_receipt.validate_for(binding, model, request, response)?;
        let launched_validation_receipt = ForgeProposalLaunchedEvaluatorValidationReceipt::record_brier(
            launch_policy,
            model_launch_receipt.direct_launch_receipt(),
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
        if model_launch_receipt.model_id() != model.id()
            || model_launch_receipt.model_payload_id() != model.model_payload_id()
            || model_launch_receipt.binding_id() != binding.id()
            || launched_validation_receipt.direct_launch_receipt_id()
                != model_launch_receipt.direct_launch_receipt().id()
        {
            return Err(ForgeProposalModelValidationError::ValidationScopeMismatch);
        }
        let id = derive_validation_receipt_id(
            binding.id(),
            model_launch_receipt.id(),
            launched_validation_receipt.id(),
        );
        Ok(Self {
            id,
            executable_model_binding_id: binding.id().clone(),
            executable_model_launch_receipt_id: model_launch_receipt.id().clone(),
            launched_validation_receipt,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn executable_model_binding_id(&self) -> &ContentId {
        &self.executable_model_binding_id
    }
    pub fn executable_model_launch_receipt_id(&self) -> &ContentId {
        &self.executable_model_launch_receipt_id
    }
    pub fn launched_validation_receipt_id(&self) -> &ContentId {
        self.launched_validation_receipt.id()
    }
    pub fn model_id(&self) -> &ContentId { self.launched_validation_receipt.model_id() }
    pub fn primary_score(&self) -> &ForgeProposalMetricScore {
        self.launched_validation_receipt.primary_score()
    }
    pub fn passed(&self) -> bool { self.launched_validation_receipt.passed() }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        binding: &ForgeProposalExecutableModelBinding,
        model_launch_receipt: &ForgeProposalExecutableModelLaunchReceipt,
        launch_policy: &ForgeProposalEvaluatorLaunchPolicy,
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
    ) -> Result<(), ForgeProposalModelValidationError> {
        binding.validate_for(model, request.protocol())?;
        model_launch_receipt.validate_for(binding, model, request, response)?;
        self.launched_validation_receipt.validate_for(
            launch_policy,
            model_launch_receipt.direct_launch_receipt(),
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
        if self.executable_model_binding_id != *binding.id()
            || self.executable_model_launch_receipt_id != *model_launch_receipt.id()
            || self.launched_validation_receipt.direct_launch_receipt_id()
                != model_launch_receipt.direct_launch_receipt().id()
        {
            return Err(ForgeProposalModelValidationError::ValidationScopeMismatch);
        }
        let expected = derive_validation_receipt_id(
            &self.executable_model_binding_id,
            &self.executable_model_launch_receipt_id,
            self.launched_validation_receipt.id(),
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalModelValidationError::ValidationIdentityMismatch)
        }
    }

    pub(crate) fn launched_validation_receipt(
        &self,
    ) -> &ForgeProposalLaunchedEvaluatorValidationReceipt {
        &self.launched_validation_receipt
    }
}

fn derive_validation_receipt_id(
    binding_id: &ContentId,
    launch_receipt_id: &ContentId,
    launched_validation_receipt_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-executable-model-validation-receipt.v1",
        [
            binding_id.as_str().as_bytes(),
            launch_receipt_id.as_str().as_bytes(),
            launched_validation_receipt_id.as_str().as_bytes(),
        ],
    )
}

/// Identity-only holdout permit whose validation prerequisite is bound all the way back to the exact
/// executable payload of the frozen model.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalExecutableModelHoldoutPermit {
    id: ContentId,
    executable_model_validation_receipt_id: ContentId,
    executable_model_binding_id: ContentId,
    executable_model_launch_receipt_id: ContentId,
    inner: ForgeProposalLaunchedEvaluatorHoldoutPermit,
}

impl ForgeProposalExecutableModelHoldoutPermit {
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        binding: &ForgeProposalExecutableModelBinding,
        model_launch_receipt: &ForgeProposalExecutableModelLaunchReceipt,
        launch_policy: &ForgeProposalEvaluatorLaunchPolicy,
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
        receipt: &ForgeProposalExecutableModelValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<Self, ForgeProposalModelValidationError> {
        receipt.validate_for(
            binding,
            model_launch_receipt,
            launch_policy,
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
        let inner = ForgeProposalLaunchedEvaluatorHoldoutPermit::issue(
            launch_policy,
            model_launch_receipt.direct_launch_receipt(),
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
            receipt.launched_validation_receipt(),
            holdout,
        )?;
        let id = derive_holdout_permit_id(
            inner.id(),
            receipt.id(),
            binding.id(),
            model_launch_receipt.id(),
        );
        Ok(Self {
            id,
            executable_model_validation_receipt_id: receipt.id().clone(),
            executable_model_binding_id: binding.id().clone(),
            executable_model_launch_receipt_id: model_launch_receipt.id().clone(),
            inner,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn executable_model_validation_receipt_id(&self) -> &ContentId {
        &self.executable_model_validation_receipt_id
    }
    pub fn executable_model_binding_id(&self) -> &ContentId {
        &self.executable_model_binding_id
    }
    pub fn executable_model_launch_receipt_id(&self) -> &ContentId {
        &self.executable_model_launch_receipt_id
    }
    pub fn model_id(&self) -> &ContentId { self.inner.model_id() }
    pub fn holdout_seal_id(&self) -> &ContentId { self.inner.holdout_seal_id() }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        binding: &ForgeProposalExecutableModelBinding,
        model_launch_receipt: &ForgeProposalExecutableModelLaunchReceipt,
        launch_policy: &ForgeProposalEvaluatorLaunchPolicy,
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
        receipt: &ForgeProposalExecutableModelValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<(), ForgeProposalModelValidationError> {
        receipt.validate_for(
            binding,
            model_launch_receipt,
            launch_policy,
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
            launch_policy,
            model_launch_receipt.direct_launch_receipt(),
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
            receipt.launched_validation_receipt(),
            holdout,
        )?;
        if self.executable_model_validation_receipt_id != *receipt.id()
            || self.executable_model_binding_id != *binding.id()
            || self.executable_model_launch_receipt_id != *model_launch_receipt.id()
        {
            return Err(ForgeProposalModelValidationError::HoldoutScopeMismatch);
        }
        let expected = derive_holdout_permit_id(
            self.inner.id(),
            &self.executable_model_validation_receipt_id,
            &self.executable_model_binding_id,
            &self.executable_model_launch_receipt_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalModelValidationError::HoldoutIdentityMismatch)
        }
    }
}

fn derive_holdout_permit_id(
    inner_permit_id: &ContentId,
    validation_receipt_id: &ContentId,
    binding_id: &ContentId,
    launch_receipt_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-executable-model-holdout-permit.v1",
        [
            inner_permit_id.as_str().as_bytes(),
            validation_receipt_id.as_str().as_bytes(),
            binding_id.as_str().as_bytes(),
            launch_receipt_id.as_str().as_bytes(),
        ],
    )
}
