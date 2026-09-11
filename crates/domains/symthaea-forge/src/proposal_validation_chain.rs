// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public deterministic validation/holdout chain over crate-internal generic receipt machinery.
//!
//! External callers cannot mint a passing validation result from an independently supplied
//! aggregate score. The public path requires a complete label-blind target/prediction set and the
//! deterministic Brier receipt reconstructed from it. Report-only metrics are intentionally
//! unsupported in v1 until they also have deterministic row-reconstructable scorers.

use crate::proposal_corpus::{
    ForgeProposalCorpusManifest, ForgeProposalHoldoutSeal, ForgeProposalValidationSet,
};
use crate::proposal_model::{
    ForgeProposalFrozenModel, ForgeProposalHoldoutEvaluationPermit, ForgeProposalMetricScore,
    ForgeProposalModelError, ForgeProposalValidationGateSpec, ForgeProposalValidationReceipt,
};
use crate::proposal_study::ForgeProposalStudySpec;
use crate::proposal_support::ForgeProposalFitPermit;
use crate::proposal_validation_blind::{
    ForgeProposalBlindDeterministicBrierReceipt, ForgeProposalBlindValidationError,
    ForgeProposalBlindValidationPredictionSet, ForgeProposalValidationTargetSet,
};
use crate::proposal_validation_coverage::{
    ForgeProposalValidationCoverageReceipt, ForgeProposalValidationCoverageSpec,
    ForgeProposalValidationScorePermit,
};
use serde::Serialize;
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalDeterministicValidationError {
    #[error(transparent)]
    Model(#[from] ForgeProposalModelError),
    #[error(transparent)]
    Blind(#[from] ForgeProposalBlindValidationError),
    #[error("deterministic validation v1 forbids report-only metrics until deterministic scorers exist")]
    UnsupportedReportMetrics,
    #[error("deterministic validation receipt does not bind the supplied blind row-level evidence")]
    ReceiptScopeMismatch,
    #[error("deterministic validation receipt identity does not match canonical fields")]
    ReceiptIdentityMismatch,
    #[error("deterministic holdout permit does not bind the supplied validation evidence")]
    HoldoutScopeMismatch,
    #[error("deterministic holdout permit identity does not match canonical fields")]
    HoldoutIdentityMismatch,
}

#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalDeterministicValidationReceipt {
    id: ContentId,
    target_set_id: ContentId,
    blind_prediction_set_id: ContentId,
    blind_brier_receipt_id: ContentId,
    inner: ForgeProposalValidationReceipt,
}

impl ForgeProposalDeterministicValidationReceipt {
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
        blind_predictions: &ForgeProposalBlindValidationPredictionSet,
        blind_brier: &ForgeProposalBlindDeterministicBrierReceipt,
    ) -> Result<Self, ForgeProposalDeterministicValidationError> {
        if !study.evaluation().report_only().is_empty() {
            return Err(ForgeProposalDeterministicValidationError::UnsupportedReportMetrics);
        }
        blind_brier.validate_for(
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
            blind_predictions,
        )?;

        let inner = ForgeProposalValidationReceipt::record(
            manifest,
            study,
            gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            blind_brier.score_value().clone(),
            Vec::new(),
            blind_brier.id().clone(),
        )?;
        if inner.primary_score() != blind_brier.score_value()
            || inner.evaluation_evidence_id() != blind_brier.id()
            || inner.model_id() != model.id()
        {
            return Err(ForgeProposalDeterministicValidationError::ReceiptScopeMismatch);
        }
        let id = derive_deterministic_validation_receipt_id(
            inner.id(),
            targets.id(),
            blind_predictions.id(),
            blind_brier.id(),
        );
        Ok(Self {
            id,
            target_set_id: targets.id().clone(),
            blind_prediction_set_id: blind_predictions.id().clone(),
            blind_brier_receipt_id: blind_brier.id().clone(),
            inner,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn target_set_id(&self) -> &ContentId { &self.target_set_id }
    pub fn blind_prediction_set_id(&self) -> &ContentId { &self.blind_prediction_set_id }
    pub fn blind_brier_receipt_id(&self) -> &ContentId { &self.blind_brier_receipt_id }
    pub fn model_id(&self) -> &ContentId { self.inner.model_id() }
    pub fn primary_score(&self) -> &ForgeProposalMetricScore { self.inner.primary_score() }
    pub fn passed(&self) -> bool { self.inner.passed() }

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
        blind_predictions: &ForgeProposalBlindValidationPredictionSet,
        blind_brier: &ForgeProposalBlindDeterministicBrierReceipt,
    ) -> Result<(), ForgeProposalDeterministicValidationError> {
        if !study.evaluation().report_only().is_empty() {
            return Err(ForgeProposalDeterministicValidationError::UnsupportedReportMetrics);
        }
        blind_brier.validate_for(
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
            blind_predictions,
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
        )?;
        if self.target_set_id != *targets.id()
            || self.blind_prediction_set_id != *blind_predictions.id()
            || self.blind_brier_receipt_id != *blind_brier.id()
            || self.inner.primary_score() != blind_brier.score_value()
            || self.inner.evaluation_evidence_id() != blind_brier.id()
            || self.inner.model_id() != model.id()
        {
            return Err(ForgeProposalDeterministicValidationError::ReceiptScopeMismatch);
        }
        let expected = derive_deterministic_validation_receipt_id(
            self.inner.id(),
            &self.target_set_id,
            &self.blind_prediction_set_id,
            &self.blind_brier_receipt_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalDeterministicValidationError::ReceiptIdentityMismatch)
        }
    }

    pub(crate) fn inner(&self) -> &ForgeProposalValidationReceipt { &self.inner }
}

fn derive_deterministic_validation_receipt_id(
    inner_receipt_id: &ContentId,
    target_set_id: &ContentId,
    blind_prediction_set_id: &ContentId,
    blind_brier_receipt_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-blind-deterministic-validation-receipt.v1",
        [
            inner_receipt_id.as_str().as_bytes(),
            target_set_id.as_str().as_bytes(),
            blind_prediction_set_id.as_str().as_bytes(),
            blind_brier_receipt_id.as_str().as_bytes(),
        ],
    )
}

#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalDeterministicHoldoutPermit {
    id: ContentId,
    deterministic_validation_receipt_id: ContentId,
    target_set_id: ContentId,
    blind_prediction_set_id: ContentId,
    blind_brier_receipt_id: ContentId,
    inner: ForgeProposalHoldoutEvaluationPermit,
}

impl ForgeProposalDeterministicHoldoutPermit {
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
        blind_predictions: &ForgeProposalBlindValidationPredictionSet,
        blind_brier: &ForgeProposalBlindDeterministicBrierReceipt,
        receipt: &ForgeProposalDeterministicValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<Self, ForgeProposalDeterministicValidationError> {
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
            blind_predictions,
            blind_brier,
        )?;
        let inner = ForgeProposalHoldoutEvaluationPermit::issue(
            manifest,
            study,
            gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            receipt.inner(),
            validation,
            holdout,
        )?;
        let id = derive_deterministic_holdout_permit_id(
            inner.id(),
            receipt.id(),
            targets.id(),
            blind_predictions.id(),
            blind_brier.id(),
        );
        Ok(Self {
            id,
            deterministic_validation_receipt_id: receipt.id().clone(),
            target_set_id: targets.id().clone(),
            blind_prediction_set_id: blind_predictions.id().clone(),
            blind_brier_receipt_id: blind_brier.id().clone(),
            inner,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn deterministic_validation_receipt_id(&self) -> &ContentId {
        &self.deterministic_validation_receipt_id
    }
    pub fn target_set_id(&self) -> &ContentId { &self.target_set_id }
    pub fn blind_prediction_set_id(&self) -> &ContentId { &self.blind_prediction_set_id }
    pub fn blind_brier_receipt_id(&self) -> &ContentId { &self.blind_brier_receipt_id }
    pub fn holdout_seal_id(&self) -> &ContentId { self.inner.holdout_seal_id() }
    pub fn model_id(&self) -> &ContentId { self.inner.model_id() }

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
        blind_predictions: &ForgeProposalBlindValidationPredictionSet,
        blind_brier: &ForgeProposalBlindDeterministicBrierReceipt,
        receipt: &ForgeProposalDeterministicValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<(), ForgeProposalDeterministicValidationError> {
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
            blind_predictions,
            blind_brier,
        )?;
        self.inner.validate_for(
            study,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            receipt.inner(),
            holdout,
        )?;
        if self.deterministic_validation_receipt_id != *receipt.id()
            || self.target_set_id != *targets.id()
            || self.blind_prediction_set_id != *blind_predictions.id()
            || self.blind_brier_receipt_id != *blind_brier.id()
            || self.inner.model_id() != model.id()
            || self.inner.holdout_seal_id() != holdout.id()
        {
            return Err(ForgeProposalDeterministicValidationError::HoldoutScopeMismatch);
        }
        let expected = derive_deterministic_holdout_permit_id(
            self.inner.id(),
            &self.deterministic_validation_receipt_id,
            &self.target_set_id,
            &self.blind_prediction_set_id,
            &self.blind_brier_receipt_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalDeterministicValidationError::HoldoutIdentityMismatch)
        }
    }
}

fn derive_deterministic_holdout_permit_id(
    inner_permit_id: &ContentId,
    deterministic_validation_receipt_id: &ContentId,
    target_set_id: &ContentId,
    blind_prediction_set_id: &ContentId,
    blind_brier_receipt_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-blind-deterministic-holdout-permit.v1",
        [
            inner_permit_id.as_str().as_bytes(),
            deterministic_validation_receipt_id.as_str().as_bytes(),
            target_set_id.as_str().as_bytes(),
            blind_prediction_set_id.as_str().as_bytes(),
            blind_brier_receipt_id.as_str().as_bytes(),
        ],
    )
}
