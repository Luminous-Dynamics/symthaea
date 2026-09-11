// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen-model and validation-receipt boundary for Forge proposal studies.
//!
//! This module still does not train or score a model. It makes the intended order explicit:
//! precommit the study -> satisfy frozen support -> mint fit permit bound to validation coverage ->
//! precommit validation metric gate -> freeze one trained model identity -> prove validation
//! coverage -> record validation for that exact model -> only a passing receipt may mint an
//! identity-only holdout-evaluation permit. Holdout rows remain inaccessible.

use crate::proposal_corpus::{
    ForgeProposalCorpusError, ForgeProposalCorpusManifest, ForgeProposalHoldoutSeal,
    ForgeProposalValidationSet,
};
use crate::proposal_study::{ForgeProposalMetric, ForgeProposalStudySpec};
use crate::proposal_support::ForgeProposalFitPermit;
use crate::proposal_validation_coverage::{
    ForgeProposalValidationCoverageError, ForgeProposalValidationCoverageReceipt,
    ForgeProposalValidationCoverageSpec, ForgeProposalValidationScorePermit,
};
use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalModelError {
    #[error(transparent)]
    Corpus(#[from] ForgeProposalCorpusError),
    #[error(transparent)]
    ValidationCoverage(#[from] ForgeProposalValidationCoverageError),
    #[error("validation score scale must be greater than zero")]
    ZeroScoreScale,
    #[error("Brier score or Brier acceptance threshold exceeds the exact [0,1] score range")]
    InvalidBrierRange,
    #[error("validation gate does not bind the supplied frozen study")]
    GateStudyMismatch,
    #[error("validation gate identity does not match canonical fields")]
    GateIdentityMismatch,
    #[error("fit permit does not bind the supplied study/training/validation-coverage identity")]
    FitPermitMismatch,
    #[error("metric score identity does not match canonical fields")]
    MetricScoreIdentityMismatch,
    #[error("frozen model does not bind the supplied study/gate/fit permit/validation coverage spec")]
    FrozenModelScopeMismatch,
    #[error("frozen model identity does not match canonical fields")]
    FrozenModelIdentityMismatch,
    #[error("validation dataset does not match the study's exact validation identity")]
    ValidationDatasetMismatch,
    #[error("validation metric set does not exactly match the precommitted evaluation specification")]
    ValidationMetricMismatch,
    #[error("validation receipt identity does not match canonical fields")]
    ValidationReceiptIdentityMismatch,
    #[error("holdout evaluation permit requires a passing validation receipt")]
    ValidationDidNotPass,
    #[error("holdout permit scope does not match the supplied study/model/receipt/seal/support and validation permits")]
    HoldoutScopeMismatch,
    #[error("holdout permit identity does not match canonical fields")]
    HoldoutPermitIdentityMismatch,
}

fn metric_tag(metric: ForgeProposalMetric) -> &'static [u8] {
    match metric {
        ForgeProposalMetric::BrierScore => b"brier-score",
        ForgeProposalMetric::LogLoss => b"log-loss",
    }
}

/// Precommitted validation acceptance gate. It must be bound by the frozen model itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalValidationGateSpec {
    id: ContentId,
    study_id: ContentId,
    primary_metric: ForgeProposalMetric,
    score_scale: u64,
    max_primary_score: u64,
    scorer_implementation_id: ContentId,
    scoring_configuration_id: ContentId,
}

impl ForgeProposalValidationGateSpec {
    pub fn precommit(
        study: &ForgeProposalStudySpec,
        score_scale: u64,
        max_primary_score: u64,
        scorer_implementation_id: ContentId,
        scoring_configuration_id: ContentId,
    ) -> Result<Self, ForgeProposalModelError> {
        if score_scale == 0 {
            return Err(ForgeProposalModelError::ZeroScoreScale);
        }
        let primary_metric = study.evaluation().primary();
        if primary_metric == ForgeProposalMetric::BrierScore && max_primary_score > score_scale {
            return Err(ForgeProposalModelError::InvalidBrierRange);
        }
        let id = derive_gate_id(
            study.id(),
            primary_metric,
            score_scale,
            max_primary_score,
            &scorer_implementation_id,
            &scoring_configuration_id,
        );
        Ok(Self {
            id,
            study_id: study.id().clone(),
            primary_metric,
            score_scale,
            max_primary_score,
            scorer_implementation_id,
            scoring_configuration_id,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn study_id(&self) -> &ContentId { &self.study_id }
    pub fn primary_metric(&self) -> ForgeProposalMetric { self.primary_metric }
    pub fn score_scale(&self) -> u64 { self.score_scale }
    pub fn max_primary_score(&self) -> u64 { self.max_primary_score }
    pub fn scorer_implementation_id(&self) -> &ContentId { &self.scorer_implementation_id }
    pub fn scoring_configuration_id(&self) -> &ContentId { &self.scoring_configuration_id }

    pub fn validate_for(
        &self,
        study: &ForgeProposalStudySpec,
    ) -> Result<(), ForgeProposalModelError> {
        if self.study_id != *study.id()
            || self.primary_metric != study.evaluation().primary()
            || self.score_scale == 0
            || (self.primary_metric == ForgeProposalMetric::BrierScore
                && self.max_primary_score > self.score_scale)
        {
            return Err(ForgeProposalModelError::GateStudyMismatch);
        }
        let expected = derive_gate_id(
            &self.study_id,
            self.primary_metric,
            self.score_scale,
            self.max_primary_score,
            &self.scorer_implementation_id,
            &self.scoring_configuration_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalModelError::GateIdentityMismatch)
        }
    }
}

fn derive_gate_id(
    study_id: &ContentId,
    metric: ForgeProposalMetric,
    score_scale: u64,
    max_score: u64,
    scorer_implementation_id: &ContentId,
    scoring_configuration_id: &ContentId,
) -> ContentId {
    let scale = score_scale.to_be_bytes();
    let max = max_score.to_be_bytes();
    ContentId::derive(
        "symthaea.forge-proposal-validation-gate.v1",
        [
            study_id.as_str().as_bytes(),
            metric_tag(metric),
            scale.as_slice(),
            max.as_slice(),
            scorer_implementation_id.as_str().as_bytes(),
            scoring_configuration_id.as_str().as_bytes(),
        ],
    )
}

/// Deterministic fixed-point metric value. `scaled_value / scale` is the recorded score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalMetricScore {
    id: ContentId,
    metric: ForgeProposalMetric,
    scale: u64,
    scaled_value: u64,
}

impl ForgeProposalMetricScore {
    pub fn new(
        metric: ForgeProposalMetric,
        scale: u64,
        scaled_value: u64,
    ) -> Result<Self, ForgeProposalModelError> {
        if scale == 0 {
            return Err(ForgeProposalModelError::ZeroScoreScale);
        }
        if metric == ForgeProposalMetric::BrierScore && scaled_value > scale {
            return Err(ForgeProposalModelError::InvalidBrierRange);
        }
        let id = derive_metric_score_id(metric, scale, scaled_value);
        Ok(Self { id, metric, scale, scaled_value })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn metric(&self) -> ForgeProposalMetric { self.metric }
    pub fn scale(&self) -> u64 { self.scale }
    pub fn scaled_value(&self) -> u64 { self.scaled_value }

    pub fn validate(&self) -> Result<(), ForgeProposalModelError> {
        let rebuilt = Self::new(self.metric, self.scale, self.scaled_value)?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeProposalModelError::MetricScoreIdentityMismatch)
        }
    }
}

fn derive_metric_score_id(metric: ForgeProposalMetric, scale: u64, value: u64) -> ContentId {
    let scale = scale.to_be_bytes();
    let value = value.to_be_bytes();
    ContentId::derive(
        "symthaea.forge-proposal-metric-score.v1",
        [metric_tag(metric), scale.as_slice(), value.as_slice()],
    )
}

/// Immutable identity of one trained model payload under one already-frozen study, fit permit,
/// validation gate, and validation-coverage specification.
///
/// The constructor accepts no validation rows and no holdout rows. The model payload and external
/// training evidence remain content-addressed inputs supplied by a future trainer implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalFrozenModel {
    id: ContentId,
    study_id: ContentId,
    validation_gate_id: ContentId,
    validation_coverage_spec_id: ContentId,
    fit_permit_id: ContentId,
    training_set_id: ContentId,
    estimator_spec_id: ContentId,
    training_seed: u64,
    model_payload_id: ContentId,
    trainer_context_id: ContentId,
    training_evidence_id: ContentId,
}

impl ForgeProposalFrozenModel {
    pub fn freeze(
        study: &ForgeProposalStudySpec,
        gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        fit_permit: &ForgeProposalFitPermit,
        model_payload_id: ContentId,
        trainer_context_id: ContentId,
        training_evidence_id: ContentId,
    ) -> Result<Self, ForgeProposalModelError> {
        gate.validate_for(study)?;
        coverage_spec.validate_for(study)?;
        validate_fit_permit_scope(study, coverage_spec, fit_permit)?;
        let id = derive_model_id(
            study.id(),
            gate.id(),
            coverage_spec.id(),
            fit_permit.id(),
            study.training_set_id(),
            study.estimator().id(),
            study.training_seed(),
            &model_payload_id,
            &trainer_context_id,
            &training_evidence_id,
        );
        Ok(Self {
            id,
            study_id: study.id().clone(),
            validation_gate_id: gate.id().clone(),
            validation_coverage_spec_id: coverage_spec.id().clone(),
            fit_permit_id: fit_permit.id().clone(),
            training_set_id: study.training_set_id().clone(),
            estimator_spec_id: study.estimator().id().clone(),
            training_seed: study.training_seed(),
            model_payload_id,
            trainer_context_id,
            training_evidence_id,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn study_id(&self) -> &ContentId { &self.study_id }
    pub fn validation_gate_id(&self) -> &ContentId { &self.validation_gate_id }
    pub fn validation_coverage_spec_id(&self) -> &ContentId { &self.validation_coverage_spec_id }
    pub fn fit_permit_id(&self) -> &ContentId { &self.fit_permit_id }
    pub fn model_payload_id(&self) -> &ContentId { &self.model_payload_id }
    pub fn trainer_context_id(&self) -> &ContentId { &self.trainer_context_id }
    pub fn training_evidence_id(&self) -> &ContentId { &self.training_evidence_id }

    pub fn validate_for(
        &self,
        study: &ForgeProposalStudySpec,
        gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        fit_permit: &ForgeProposalFitPermit,
    ) -> Result<(), ForgeProposalModelError> {
        gate.validate_for(study)?;
        coverage_spec.validate_for(study)?;
        validate_fit_permit_scope(study, coverage_spec, fit_permit)?;
        if self.study_id != *study.id()
            || self.validation_gate_id != *gate.id()
            || self.validation_coverage_spec_id != *coverage_spec.id()
            || self.fit_permit_id != *fit_permit.id()
            || self.training_set_id != *study.training_set_id()
            || self.estimator_spec_id != *study.estimator().id()
            || self.training_seed != study.training_seed()
        {
            return Err(ForgeProposalModelError::FrozenModelScopeMismatch);
        }
        let expected = derive_model_id(
            &self.study_id,
            &self.validation_gate_id,
            &self.validation_coverage_spec_id,
            &self.fit_permit_id,
            &self.training_set_id,
            &self.estimator_spec_id,
            self.training_seed,
            &self.model_payload_id,
            &self.trainer_context_id,
            &self.training_evidence_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalModelError::FrozenModelIdentityMismatch)
        }
    }
}

fn validate_fit_permit_scope(
    study: &ForgeProposalStudySpec,
    coverage_spec: &ForgeProposalValidationCoverageSpec,
    permit: &ForgeProposalFitPermit,
) -> Result<(), ForgeProposalModelError> {
    if permit.study_id() != study.id()
        || permit.training_set_id() != study.training_set_id()
        || permit.validation_coverage_spec_id() != coverage_spec.id()
    {
        Err(ForgeProposalModelError::FitPermitMismatch)
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_model_id(
    study_id: &ContentId,
    gate_id: &ContentId,
    coverage_spec_id: &ContentId,
    fit_permit_id: &ContentId,
    training_set_id: &ContentId,
    estimator_spec_id: &ContentId,
    training_seed: u64,
    model_payload_id: &ContentId,
    trainer_context_id: &ContentId,
    training_evidence_id: &ContentId,
) -> ContentId {
    let seed = training_seed.to_be_bytes();
    ContentId::derive(
        "symthaea.forge-proposal-frozen-model.v3",
        [
            study_id.as_str().as_bytes(),
            gate_id.as_str().as_bytes(),
            coverage_spec_id.as_str().as_bytes(),
            fit_permit_id.as_str().as_bytes(),
            training_set_id.as_str().as_bytes(),
            estimator_spec_id.as_str().as_bytes(),
            seed.as_slice(),
            model_payload_id.as_str().as_bytes(),
            trainer_context_id.as_str().as_bytes(),
            training_evidence_id.as_str().as_bytes(),
        ],
    )
}

/// Validation result for one exact frozen model. Pass/fail is derived only from the precommitted
/// metric gate, and scoring is admissible only after the precommitted coverage gate has passed.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalValidationReceipt {
    id: ContentId,
    study_id: ContentId,
    gate_id: ContentId,
    validation_coverage_spec_id: ContentId,
    validation_coverage_receipt_id: ContentId,
    validation_score_permit_id: ContentId,
    fit_permit_id: ContentId,
    model_id: ContentId,
    validation_set_id: ContentId,
    context_id: ContentId,
    primary_score: ForgeProposalMetricScore,
    report_scores: Vec<ForgeProposalMetricScore>,
    evaluation_evidence_id: ContentId,
    passed: bool,
}

impl ForgeProposalValidationReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        primary_score: ForgeProposalMetricScore,
        mut report_scores: Vec<ForgeProposalMetricScore>,
        evaluation_evidence_id: ContentId,
    ) -> Result<Self, ForgeProposalModelError> {
        manifest.validate()?;
        validation.validate_for(manifest)?;
        gate.validate_for(study)?;
        coverage_spec.validate_for(study)?;
        coverage_receipt.validate_for(coverage_spec, manifest, study, validation)?;
        score_permit.validate_for(coverage_spec, coverage_receipt)?;
        validate_fit_permit_scope(study, coverage_spec, fit_permit)?;
        model.validate_for(study, gate, coverage_spec, fit_permit)?;
        if manifest.id() != study.corpus_manifest_id()
            || validation.id() != study.validation_set_id()
            || validation.manifest_id() != manifest.id()
            || score_permit.validation_set_id() != validation.id()
        {
            return Err(ForgeProposalModelError::ValidationDatasetMismatch);
        }

        validate_metric_set(study, gate, &primary_score, &mut report_scores)?;
        let passed = primary_score.scaled_value() <= gate.max_primary_score();
        let context_id = manifest.context_id().clone();
        let id = derive_validation_receipt_id(
            study.id(),
            gate.id(),
            coverage_spec.id(),
            coverage_receipt.id(),
            score_permit.id(),
            fit_permit.id(),
            model.id(),
            validation.id(),
            &context_id,
            primary_score.id(),
            &report_scores,
            &evaluation_evidence_id,
            passed,
        );
        Ok(Self {
            id,
            study_id: study.id().clone(),
            gate_id: gate.id().clone(),
            validation_coverage_spec_id: coverage_spec.id().clone(),
            validation_coverage_receipt_id: coverage_receipt.id().clone(),
            validation_score_permit_id: score_permit.id().clone(),
            fit_permit_id: fit_permit.id().clone(),
            model_id: model.id().clone(),
            validation_set_id: validation.id().clone(),
            context_id,
            primary_score,
            report_scores,
            evaluation_evidence_id,
            passed,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn model_id(&self) -> &ContentId { &self.model_id }
    pub fn fit_permit_id(&self) -> &ContentId { &self.fit_permit_id }
    pub fn validation_coverage_spec_id(&self) -> &ContentId { &self.validation_coverage_spec_id }
    pub fn validation_coverage_receipt_id(&self) -> &ContentId {
        &self.validation_coverage_receipt_id
    }
    pub fn validation_score_permit_id(&self) -> &ContentId { &self.validation_score_permit_id }
    pub fn primary_score(&self) -> &ForgeProposalMetricScore { &self.primary_score }
    pub fn report_scores(&self) -> &[ForgeProposalMetricScore] { &self.report_scores }
    pub fn evaluation_evidence_id(&self) -> &ContentId { &self.evaluation_evidence_id }
    pub fn passed(&self) -> bool { self.passed }

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
    ) -> Result<(), ForgeProposalModelError> {
        manifest.validate()?;
        validation.validate_for(manifest)?;
        gate.validate_for(study)?;
        coverage_spec.validate_for(study)?;
        coverage_receipt.validate_for(coverage_spec, manifest, study, validation)?;
        score_permit.validate_for(coverage_spec, coverage_receipt)?;
        validate_fit_permit_scope(study, coverage_spec, fit_permit)?;
        model.validate_for(study, gate, coverage_spec, fit_permit)?;
        if self.study_id != *study.id()
            || self.gate_id != *gate.id()
            || self.validation_coverage_spec_id != *coverage_spec.id()
            || self.validation_coverage_receipt_id != *coverage_receipt.id()
            || self.validation_score_permit_id != *score_permit.id()
            || self.fit_permit_id != *fit_permit.id()
            || self.model_id != *model.id()
            || self.validation_set_id != *validation.id()
            || self.context_id != *manifest.context_id()
            || validation.id() != study.validation_set_id()
            || score_permit.validation_set_id() != validation.id()
        {
            return Err(ForgeProposalModelError::ValidationDatasetMismatch);
        }
        let mut reports = self.report_scores.clone();
        validate_metric_set(study, gate, &self.primary_score, &mut reports)?;
        if reports != self.report_scores
            || self.passed != (self.primary_score.scaled_value() <= gate.max_primary_score())
        {
            return Err(ForgeProposalModelError::ValidationMetricMismatch);
        }
        let expected = derive_validation_receipt_id(
            &self.study_id,
            &self.gate_id,
            &self.validation_coverage_spec_id,
            &self.validation_coverage_receipt_id,
            &self.validation_score_permit_id,
            &self.fit_permit_id,
            &self.model_id,
            &self.validation_set_id,
            &self.context_id,
            self.primary_score.id(),
            &self.report_scores,
            &self.evaluation_evidence_id,
            self.passed,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalModelError::ValidationReceiptIdentityMismatch)
        }
    }
}

fn validate_metric_set(
    study: &ForgeProposalStudySpec,
    gate: &ForgeProposalValidationGateSpec,
    primary: &ForgeProposalMetricScore,
    report_scores: &mut Vec<ForgeProposalMetricScore>,
) -> Result<(), ForgeProposalModelError> {
    primary.validate()?;
    if primary.metric() != study.evaluation().primary()
        || primary.metric() != gate.primary_metric()
        || primary.scale() != gate.score_scale()
    {
        return Err(ForgeProposalModelError::ValidationMetricMismatch);
    }
    for score in report_scores.iter() {
        score.validate()?;
        if score.scale() != gate.score_scale() {
            return Err(ForgeProposalModelError::ValidationMetricMismatch);
        }
    }
    report_scores.sort_by_key(|score| score.metric());
    let mut seen = BTreeSet::new();
    if report_scores.iter().any(|score| !seen.insert(score.metric())) {
        return Err(ForgeProposalModelError::ValidationMetricMismatch);
    }
    let observed = report_scores.iter().map(|score| score.metric()).collect::<Vec<_>>();
    if observed != study.evaluation().report_only() {
        return Err(ForgeProposalModelError::ValidationMetricMismatch);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn derive_validation_receipt_id(
    study_id: &ContentId,
    gate_id: &ContentId,
    coverage_spec_id: &ContentId,
    coverage_receipt_id: &ContentId,
    score_permit_id: &ContentId,
    fit_permit_id: &ContentId,
    model_id: &ContentId,
    validation_set_id: &ContentId,
    context_id: &ContentId,
    primary_score_id: &ContentId,
    report_scores: &[ForgeProposalMetricScore],
    evidence_id: &ContentId,
    passed: bool,
) -> ContentId {
    let count = (report_scores.len() as u64).to_be_bytes();
    let mut parts = vec![
        study_id.as_str().as_bytes().to_vec(),
        gate_id.as_str().as_bytes().to_vec(),
        coverage_spec_id.as_str().as_bytes().to_vec(),
        coverage_receipt_id.as_str().as_bytes().to_vec(),
        score_permit_id.as_str().as_bytes().to_vec(),
        fit_permit_id.as_str().as_bytes().to_vec(),
        model_id.as_str().as_bytes().to_vec(),
        validation_set_id.as_str().as_bytes().to_vec(),
        context_id.as_str().as_bytes().to_vec(),
        primary_score_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        report_scores
            .iter()
            .map(|score| score.id().as_str().as_bytes().to_vec()),
    );
    parts.push(evidence_id.as_str().as_bytes().to_vec());
    parts.push(if passed { b"pass".to_vec() } else { b"fail".to_vec() });
    ContentId::derive(
        "symthaea.forge-proposal-validation-receipt.v3",
        parts.iter().map(Vec::as_slice),
    )
}

/// Identity-only permit proving an exact frozen model passed its precommitted validation gate after
/// the exact role-isolated validation corpus also passed its precommitted observed-coverage gate.
/// It still does not expose holdout rows or perform holdout scoring.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalHoldoutEvaluationPermit {
    id: ContentId,
    study_id: ContentId,
    validation_coverage_spec_id: ContentId,
    validation_score_permit_id: ContentId,
    fit_permit_id: ContentId,
    model_id: ContentId,
    validation_receipt_id: ContentId,
    holdout_seal_id: ContentId,
}

impl ForgeProposalHoldoutEvaluationPermit {
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
        receipt: &ForgeProposalValidationReceipt,
        validation: &ForgeProposalValidationSet,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<Self, ForgeProposalModelError> {
        holdout.validate_for(manifest)?;
        coverage_receipt.validate_for(coverage_spec, manifest, study, validation)?;
        score_permit.validate_for(coverage_spec, coverage_receipt)?;
        validate_fit_permit_scope(study, coverage_spec, fit_permit)?;
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
        )?;
        if !receipt.passed() {
            return Err(ForgeProposalModelError::ValidationDidNotPass);
        }
        if holdout.id() != study.holdout_seal_id() || holdout.manifest_id() != manifest.id() {
            return Err(ForgeProposalModelError::HoldoutScopeMismatch);
        }
        let id = derive_holdout_permit_id(
            study.id(),
            coverage_spec.id(),
            score_permit.id(),
            fit_permit.id(),
            model.id(),
            receipt.id(),
            holdout.id(),
        );
        Ok(Self {
            id,
            study_id: study.id().clone(),
            validation_coverage_spec_id: coverage_spec.id().clone(),
            validation_score_permit_id: score_permit.id().clone(),
            fit_permit_id: fit_permit.id().clone(),
            model_id: model.id().clone(),
            validation_receipt_id: receipt.id().clone(),
            holdout_seal_id: holdout.id().clone(),
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn study_id(&self) -> &ContentId { &self.study_id }
    pub fn validation_coverage_spec_id(&self) -> &ContentId { &self.validation_coverage_spec_id }
    pub fn validation_score_permit_id(&self) -> &ContentId { &self.validation_score_permit_id }
    pub fn fit_permit_id(&self) -> &ContentId { &self.fit_permit_id }
    pub fn model_id(&self) -> &ContentId { &self.model_id }
    pub fn validation_receipt_id(&self) -> &ContentId { &self.validation_receipt_id }
    pub fn holdout_seal_id(&self) -> &ContentId { &self.holdout_seal_id }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        study: &ForgeProposalStudySpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        receipt: &ForgeProposalValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<(), ForgeProposalModelError> {
        coverage_spec.validate_for(study)?;
        score_permit.validate_for(coverage_spec, coverage_receipt)?;
        validate_fit_permit_scope(study, coverage_spec, fit_permit)?;
        if !receipt.passed()
            || self.study_id != *study.id()
            || self.validation_coverage_spec_id != *coverage_spec.id()
            || self.validation_score_permit_id != *score_permit.id()
            || self.fit_permit_id != *fit_permit.id()
            || self.model_id != *model.id()
            || self.validation_receipt_id != *receipt.id()
            || self.holdout_seal_id != *holdout.id()
            || holdout.id() != study.holdout_seal_id()
            || model.validation_coverage_spec_id() != coverage_spec.id()
            || model.fit_permit_id() != fit_permit.id()
            || receipt.validation_coverage_spec_id() != coverage_spec.id()
            || receipt.validation_score_permit_id() != score_permit.id()
            || receipt.fit_permit_id() != fit_permit.id()
        {
            return Err(ForgeProposalModelError::HoldoutScopeMismatch);
        }
        let expected = derive_holdout_permit_id(
            &self.study_id,
            &self.validation_coverage_spec_id,
            &self.validation_score_permit_id,
            &self.fit_permit_id,
            &self.model_id,
            &self.validation_receipt_id,
            &self.holdout_seal_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalModelError::HoldoutPermitIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_holdout_permit_id(
    study_id: &ContentId,
    coverage_spec_id: &ContentId,
    score_permit_id: &ContentId,
    fit_permit_id: &ContentId,
    model_id: &ContentId,
    receipt_id: &ContentId,
    holdout_seal_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-holdout-evaluation-permit.v3",
        [
            study_id.as_str().as_bytes(),
            coverage_spec_id.as_str().as_bytes(),
            score_permit_id.as_str().as_bytes(),
            fit_permit_id.as_str().as_bytes(),
            model_id.as_str().as_bytes(),
            receipt_id.as_str().as_bytes(),
            holdout_seal_id.as_str().as_bytes(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_scores_are_fixed_point_and_brier_is_range_checked() {
        let brier = ForgeProposalMetricScore::new(ForgeProposalMetric::BrierScore, 1_000_000, 123_456)
            .unwrap();
        assert_eq!(brier.scaled_value(), 123_456);
        assert!(brier.validate().is_ok());
        assert!(ForgeProposalMetricScore::new(
            ForgeProposalMetric::BrierScore,
            1_000_000,
            1_000_001,
        )
        .is_err());
        assert!(ForgeProposalMetricScore::new(ForgeProposalMetric::LogLoss, 0, 1).is_err());
    }

    #[test]
    fn metric_score_identity_distinguishes_scale_and_value() {
        let a = ForgeProposalMetricScore::new(ForgeProposalMetric::LogLoss, 1_000_000, 10).unwrap();
        let b = ForgeProposalMetricScore::new(ForgeProposalMetric::LogLoss, 1_000_000, 11).unwrap();
        let c = ForgeProposalMetricScore::new(ForgeProposalMetric::LogLoss, 1_000, 10).unwrap();
        assert_ne!(a.id(), b.id());
        assert_ne!(a.id(), c.id());
    }
}
