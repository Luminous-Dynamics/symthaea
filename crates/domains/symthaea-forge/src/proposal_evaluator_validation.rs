// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public validation/holdout chain bound to the label-blind evaluator protocol.
//!
//! This module makes the evaluator request/response boundary mandatory for public validation
//! evidence. The response is converted to the crate-internal blind prediction set, deterministic
//! Brier is reconstructed from exact validation labels only after that response is frozen, and the
//! existing crate-internal deterministic receipt machinery is then reused.
//!
//! This still proves no OS/process isolation by itself; it proves only that the public validation
//! proposition names the exact evaluator request and response artifacts.

use crate::proposal_corpus::{
    ForgeProposalCorpusManifest, ForgeProposalHoldoutSeal, ForgeProposalValidationSet,
};
use crate::proposal_evaluator_protocol::{
    ForgeProposalEvaluationRequest, ForgeProposalEvaluationResponse,
    ForgeProposalEvaluatorProtocolError,
};
use crate::proposal_model::{
    ForgeProposalFrozenModel, ForgeProposalMetricScore, ForgeProposalValidationGateSpec,
};
use crate::proposal_study::ForgeProposalStudySpec;
use crate::proposal_support::ForgeProposalFitPermit;
use crate::proposal_validation_blind::{
    ForgeProposalBlindDeterministicBrierReceipt, ForgeProposalBlindValidationError,
    ForgeProposalValidationTargetSet,
};
use crate::proposal_validation_chain::{
    ForgeProposalDeterministicHoldoutPermit, ForgeProposalDeterministicValidationError,
    ForgeProposalDeterministicValidationReceipt,
};
use crate::proposal_validation_coverage::{
    ForgeProposalValidationCoverageReceipt, ForgeProposalValidationCoverageSpec,
    ForgeProposalValidationScorePermit,
};
use serde::Serialize;
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalEvaluatorValidationError {
    #[error(transparent)]
    Protocol(#[from] ForgeProposalEvaluatorProtocolError),
    #[error(transparent)]
    Blind(#[from] ForgeProposalBlindValidationError),
    #[error(transparent)]
    Deterministic(#[from] ForgeProposalDeterministicValidationError),
    #[error("evaluator-bound validation evidence does not match the supplied study/model/gates")]
    ScopeMismatch,
    #[error("evaluator-bound validation receipt identity does not match canonical fields")]
    ReceiptIdentityMismatch,
    #[error("evaluator-bound holdout permit identity does not match canonical fields")]
    HoldoutIdentityMismatch,
}

/// Public validation receipt that binds the exact evaluator request/response artifacts.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalEvaluatorValidationReceipt {
    id: ContentId,
    request_id: ContentId,
    response_id: ContentId,
    execution_context_id: ContentId,
    deterministic_receipt: ForgeProposalDeterministicValidationReceipt,
}

impl ForgeProposalEvaluatorValidationReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn record_brier(
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
    ) -> Result<Self, ForgeProposalEvaluatorValidationError> {
        response.validate_for(request)?;
        if request.study_id() != study.id()
            || request.validation_gate_id() != gate.id()
            || request.validation_coverage_spec_id() != coverage_spec.id()
            || request.validation_score_permit_id() != score_permit.id()
            || request.fit_permit_id() != fit_permit.id()
            || request.model_id() != model.id()
            || request.target_set_id() != targets.id()
            || response.request_id() != request.id()
            || response.model_id() != model.id()
            || response.target_set_id() != targets.id()
        {
            return Err(ForgeProposalEvaluatorValidationError::ScopeMismatch);
        }

        let blind_predictions = response.to_blind_prediction_set(request, model, gate, targets)?;
        let blind_brier = ForgeProposalBlindDeterministicBrierReceipt::score(
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
            &blind_predictions,
        )?;
        let deterministic_receipt = ForgeProposalDeterministicValidationReceipt::record_brier(
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
            &blind_predictions,
            &blind_brier,
        )?;

        let id = derive_receipt_id(
            request.id(),
            response.id(),
            response.execution_context_id(),
            deterministic_receipt.id(),
        );
        Ok(Self {
            id,
            request_id: request.id().clone(),
            response_id: response.id().clone(),
            execution_context_id: response.execution_context_id().clone(),
            deterministic_receipt,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn request_id(&self) -> &ContentId { &self.request_id }
    pub fn response_id(&self) -> &ContentId { &self.response_id }
    pub fn execution_context_id(&self) -> &ContentId { &self.execution_context_id }
    pub fn model_id(&self) -> &ContentId { self.deterministic_receipt.model_id() }
    pub fn primary_score(&self) -> &ForgeProposalMetricScore {
        self.deterministic_receipt.primary_score()
    }
    pub fn passed(&self) -> bool { self.deterministic_receipt.passed() }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
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
    ) -> Result<(), ForgeProposalEvaluatorValidationError> {
        response.validate_for(request)?;
        if self.request_id != *request.id()
            || self.response_id != *response.id()
            || self.execution_context_id != *response.execution_context_id()
            || request.study_id() != study.id()
            || request.validation_gate_id() != gate.id()
            || request.validation_coverage_spec_id() != coverage_spec.id()
            || request.validation_score_permit_id() != score_permit.id()
            || request.fit_permit_id() != fit_permit.id()
            || request.model_id() != model.id()
            || request.target_set_id() != targets.id()
        {
            return Err(ForgeProposalEvaluatorValidationError::ScopeMismatch);
        }
        let blind_predictions = response.to_blind_prediction_set(request, model, gate, targets)?;
        let blind_brier = ForgeProposalBlindDeterministicBrierReceipt::score(
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
            &blind_predictions,
        )?;
        self.deterministic_receipt.validate_for(
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
            &blind_predictions,
            &blind_brier,
        )?;
        let expected = derive_receipt_id(
            &self.request_id,
            &self.response_id,
            &self.execution_context_id,
            self.deterministic_receipt.id(),
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalEvaluatorValidationError::ReceiptIdentityMismatch)
        }
    }

    pub(crate) fn deterministic_receipt(&self) -> &ForgeProposalDeterministicValidationReceipt {
        &self.deterministic_receipt
    }
}

fn derive_receipt_id(
    request_id: &ContentId,
    response_id: &ContentId,
    execution_context_id: &ContentId,
    deterministic_receipt_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-evaluator-validation-receipt.v1",
        [
            request_id.as_str().as_bytes(),
            response_id.as_str().as_bytes(),
            execution_context_id.as_str().as_bytes(),
            deterministic_receipt_id.as_str().as_bytes(),
        ],
    )
}

/// Identity-only holdout permit whose validation prerequisite names the exact evaluator artifacts.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalEvaluatorHoldoutPermit {
    id: ContentId,
    evaluator_validation_receipt_id: ContentId,
    request_id: ContentId,
    response_id: ContentId,
    execution_context_id: ContentId,
    inner: ForgeProposalDeterministicHoldoutPermit,
}

impl ForgeProposalEvaluatorHoldoutPermit {
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
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
        receipt: &ForgeProposalEvaluatorValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<Self, ForgeProposalEvaluatorValidationError> {
        receipt.validate_for(
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
        let blind_predictions = response.to_blind_prediction_set(request, model, gate, targets)?;
        let blind_brier = ForgeProposalBlindDeterministicBrierReceipt::score(
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
            &blind_predictions,
        )?;
        let inner = ForgeProposalDeterministicHoldoutPermit::issue(
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
            &blind_predictions,
            &blind_brier,
            receipt.deterministic_receipt(),
            holdout,
        )?;
        let id = derive_holdout_id(
            inner.id(),
            receipt.id(),
            request.id(),
            response.id(),
            response.execution_context_id(),
        );
        Ok(Self {
            id,
            evaluator_validation_receipt_id: receipt.id().clone(),
            request_id: request.id().clone(),
            response_id: response.id().clone(),
            execution_context_id: response.execution_context_id().clone(),
            inner,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn evaluator_validation_receipt_id(&self) -> &ContentId {
        &self.evaluator_validation_receipt_id
    }
    pub fn request_id(&self) -> &ContentId { &self.request_id }
    pub fn response_id(&self) -> &ContentId { &self.response_id }
    pub fn execution_context_id(&self) -> &ContentId { &self.execution_context_id }
    pub fn model_id(&self) -> &ContentId { self.inner.model_id() }
    pub fn holdout_seal_id(&self) -> &ContentId { self.inner.holdout_seal_id() }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
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
        receipt: &ForgeProposalEvaluatorValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<(), ForgeProposalEvaluatorValidationError> {
        receipt.validate_for(
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
        let blind_predictions = response.to_blind_prediction_set(request, model, gate, targets)?;
        let blind_brier = ForgeProposalBlindDeterministicBrierReceipt::score(
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
            &blind_predictions,
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
            &blind_predictions,
            &blind_brier,
            receipt.deterministic_receipt(),
            holdout,
        )?;
        if self.evaluator_validation_receipt_id != *receipt.id()
            || self.request_id != *request.id()
            || self.response_id != *response.id()
            || self.execution_context_id != *response.execution_context_id()
        {
            return Err(ForgeProposalEvaluatorValidationError::ScopeMismatch);
        }
        let expected = derive_holdout_id(
            self.inner.id(),
            &self.evaluator_validation_receipt_id,
            &self.request_id,
            &self.response_id,
            &self.execution_context_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalEvaluatorValidationError::HoldoutIdentityMismatch)
        }
    }
}

fn derive_holdout_id(
    inner_permit_id: &ContentId,
    evaluator_validation_receipt_id: &ContentId,
    request_id: &ContentId,
    response_id: &ContentId,
    execution_context_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-evaluator-holdout-permit.v1",
        [
            inner_permit_id.as_str().as_bytes(),
            evaluator_validation_receipt_id.as_str().as_bytes(),
            request_id.as_str().as_bytes(),
            response_id.as_str().as_bytes(),
            execution_context_id.as_str().as_bytes(),
        ],
    )
}
