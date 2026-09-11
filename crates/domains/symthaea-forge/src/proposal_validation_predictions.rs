// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Row-level immutable validation predictions and deterministic Brier scoring.
//!
//! A validation metric is only reconstructable if the exact predictions that produced it survive.
//! This module therefore binds one fixed-point probability to every *observed* validation endpoint
//! record, proves exact one-to-one coverage, and computes Brier score using checked integer
//! arithmetic. Censored and counterfactual-unobserved rows remain part of validation-coverage
//! evidence but are never silently converted into labels for scoring.
//!
//! This module performs no fitting, opens no holdout rows, and grants no search/runtime authority.

use crate::proposal_corpus::{
    ForgeProposalCorpusError, ForgeProposalCorpusManifest, ForgeProposalValidationSet,
};
use crate::proposal_endpoints::{
    ForgeProposalEndpointError, ForgeProposalEndpointRecord, ForgeProposalEndpointValue,
};
use crate::proposal_model::{
    ForgeProposalFrozenModel, ForgeProposalMetricScore, ForgeProposalModelError,
    ForgeProposalValidationGateSpec,
};
use crate::proposal_study::{ForgeProposalMetric, ForgeProposalStudySpec};
use crate::proposal_support::ForgeProposalFitPermit;
use crate::proposal_validation_coverage::{
    ForgeProposalValidationCoverageError, ForgeProposalValidationCoverageReceipt,
    ForgeProposalValidationCoverageSpec, ForgeProposalValidationScorePermit,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalValidationPredictionError {
    #[error(transparent)]
    Corpus(#[from] ForgeProposalCorpusError),
    #[error(transparent)]
    Endpoint(#[from] ForgeProposalEndpointError),
    #[error(transparent)]
    Model(#[from] ForgeProposalModelError),
    #[error(transparent)]
    Coverage(#[from] ForgeProposalValidationCoverageError),
    #[error("validation prediction probability scale must be greater than zero")]
    ZeroProbabilityScale,
    #[error("validation prediction probability exceeds its fixed-point scale")]
    ProbabilityOutOfRange,
    #[error("validation predictions may target only actually observed endpoint records")]
    NonObservedTarget,
    #[error("validation prediction does not bind the supplied model/endpoint record/scale")]
    PredictionScopeMismatch,
    #[error("validation prediction identity does not match canonical fields")]
    PredictionIdentityMismatch,
    #[error("validation prediction set contains duplicate endpoint predictions")]
    DuplicatePrediction,
    #[error("validation prediction set does not cover every and only observed validation endpoint")]
    PredictionCoverageMismatch,
    #[error("validation prediction set identity does not match canonical fields")]
    PredictionSetIdentityMismatch,
    #[error("deterministic v1 validation scoring supports only the Brier primary metric")]
    UnsupportedMetric,
    #[error("validation gate scorer/configuration does not name the deterministic Brier v1 scorer")]
    ScorerIdentityMismatch,
    #[error("deterministic validation scoring arithmetic overflow")]
    ArithmeticOverflow,
    #[error("deterministic Brier score identity does not match canonical fields")]
    ScoreIdentityMismatch,
}

/// Exact implementation identity for the integer-only deterministic Brier scorer in this module.
pub fn forge_deterministic_brier_scorer_id() -> ContentId {
    ContentId::derive(
        "symthaea.forge-deterministic-brier-scorer.v1",
        [b"observed-endpoints-only;checked-u128;round-half-up".as_slice()],
    )
}

/// Exact configuration identity for v1 Brier scoring semantics.
///
/// The probability/score scale itself is already frozen by `ForgeProposalValidationGateSpec`.
pub fn forge_deterministic_brier_configuration_id() -> ContentId {
    ContentId::derive(
        "symthaea.forge-deterministic-brier-config.v1",
        [b"score=sum((p-y)^2)/(n*scale);output-scale=scale".as_slice()],
    )
}

/// One immutable model probability for one exact *observed* validation endpoint record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalValidationPrediction {
    id: ContentId,
    model_id: ContentId,
    source_row_id: ContentId,
    endpoint_record_id: ContentId,
    scale: u64,
    probability_scaled: u64,
}

impl ForgeProposalValidationPrediction {
    pub fn for_record(
        model: &ForgeProposalFrozenModel,
        record: &ForgeProposalEndpointRecord,
        scale: u64,
        probability_scaled: u64,
    ) -> Result<Self, ForgeProposalValidationPredictionError> {
        if scale == 0 {
            return Err(ForgeProposalValidationPredictionError::ZeroProbabilityScale);
        }
        if probability_scaled > scale {
            return Err(ForgeProposalValidationPredictionError::ProbabilityOutOfRange);
        }
        if !matches!(record.value(), ForgeProposalEndpointValue::Observed(_)) {
            return Err(ForgeProposalValidationPredictionError::NonObservedTarget);
        }
        let id = derive_prediction_id(
            model.id(),
            record.source_row_id(),
            record.id(),
            scale,
            probability_scaled,
        );
        Ok(Self {
            id,
            model_id: model.id().clone(),
            source_row_id: record.source_row_id().clone(),
            endpoint_record_id: record.id().clone(),
            scale,
            probability_scaled,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn model_id(&self) -> &ContentId { &self.model_id }
    pub fn source_row_id(&self) -> &ContentId { &self.source_row_id }
    pub fn endpoint_record_id(&self) -> &ContentId { &self.endpoint_record_id }
    pub fn scale(&self) -> u64 { self.scale }
    pub fn probability_scaled(&self) -> u64 { self.probability_scaled }

    pub fn validate_for(
        &self,
        model: &ForgeProposalFrozenModel,
        record: &ForgeProposalEndpointRecord,
        scale: u64,
    ) -> Result<(), ForgeProposalValidationPredictionError> {
        if scale == 0 {
            return Err(ForgeProposalValidationPredictionError::ZeroProbabilityScale);
        }
        if self.probability_scaled > scale {
            return Err(ForgeProposalValidationPredictionError::ProbabilityOutOfRange);
        }
        if !matches!(record.value(), ForgeProposalEndpointValue::Observed(_)) {
            return Err(ForgeProposalValidationPredictionError::NonObservedTarget);
        }
        if self.model_id != *model.id()
            || self.source_row_id != *record.source_row_id()
            || self.endpoint_record_id != *record.id()
            || self.scale != scale
        {
            return Err(ForgeProposalValidationPredictionError::PredictionScopeMismatch);
        }
        let expected = derive_prediction_id(
            &self.model_id,
            &self.source_row_id,
            &self.endpoint_record_id,
            self.scale,
            self.probability_scaled,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalValidationPredictionError::PredictionIdentityMismatch)
        }
    }
}

fn derive_prediction_id(
    model_id: &ContentId,
    source_row_id: &ContentId,
    endpoint_record_id: &ContentId,
    scale: u64,
    probability_scaled: u64,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-validation-prediction.v1",
        [
            model_id.as_str().as_bytes(),
            source_row_id.as_str().as_bytes(),
            endpoint_record_id.as_str().as_bytes(),
            scale.to_be_bytes().as_slice(),
            probability_scaled.to_be_bytes().as_slice(),
        ],
    )
}

/// Canonical set containing exactly one prediction for every observed validation endpoint label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalValidationPredictionSet {
    id: ContentId,
    manifest_id: ContentId,
    study_id: ContentId,
    validation_gate_id: ContentId,
    coverage_spec_id: ContentId,
    coverage_receipt_id: ContentId,
    score_permit_id: ContentId,
    fit_permit_id: ContentId,
    model_id: ContentId,
    validation_set_id: ContentId,
    scale: u64,
    predictions: Vec<ForgeProposalValidationPrediction>,
}

impl ForgeProposalValidationPredictionSet {
    #[allow(clippy::too_many_arguments)]
    pub fn freeze(
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        mut predictions: Vec<ForgeProposalValidationPrediction>,
    ) -> Result<Self, ForgeProposalValidationPredictionError> {
        manifest.validate()?;
        validation.validate_for(manifest)?;
        coverage_spec.validate_for(study)?;
        coverage_receipt.validate_for(coverage_spec, manifest, study, validation)?;
        score_permit.validate_for(coverage_spec, coverage_receipt)?;
        model.validate_for(study, gate, coverage_spec, fit_permit)?;
        if manifest.id() != study.corpus_manifest_id()
            || validation.id() != study.validation_set_id()
            || validation.manifest_id() != manifest.id()
            || model.validation_coverage_spec_id() != coverage_spec.id()
        {
            return Err(ForgeProposalValidationPredictionError::PredictionCoverageMismatch);
        }

        let scale = gate.score_scale();
        if scale == 0 {
            return Err(ForgeProposalValidationPredictionError::ZeroProbabilityScale);
        }
        let expected = observed_records(validation, study)?;
        predictions.sort_by(|left, right| {
            left.endpoint_record_id().as_str().cmp(right.endpoint_record_id().as_str())
        });
        if predictions
            .windows(2)
            .any(|pair| pair[0].endpoint_record_id() == pair[1].endpoint_record_id())
        {
            return Err(ForgeProposalValidationPredictionError::DuplicatePrediction);
        }
        if predictions.len() != expected.len() {
            return Err(ForgeProposalValidationPredictionError::PredictionCoverageMismatch);
        }
        for (prediction, (record, _label)) in predictions.iter().zip(expected.iter()) {
            if prediction.endpoint_record_id() != record.id() {
                return Err(ForgeProposalValidationPredictionError::PredictionCoverageMismatch);
            }
            prediction.validate_for(model, record, scale)?;
        }

        let id = derive_prediction_set_id(
            manifest.id(),
            study.id(),
            gate.id(),
            coverage_spec.id(),
            coverage_receipt.id(),
            score_permit.id(),
            fit_permit.id(),
            model.id(),
            validation.id(),
            scale,
            &predictions,
        );
        Ok(Self {
            id,
            manifest_id: manifest.id().clone(),
            study_id: study.id().clone(),
            validation_gate_id: gate.id().clone(),
            coverage_spec_id: coverage_spec.id().clone(),
            coverage_receipt_id: coverage_receipt.id().clone(),
            score_permit_id: score_permit.id().clone(),
            fit_permit_id: fit_permit.id().clone(),
            model_id: model.id().clone(),
            validation_set_id: validation.id().clone(),
            scale,
            predictions,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn model_id(&self) -> &ContentId { &self.model_id }
    pub fn validation_set_id(&self) -> &ContentId { &self.validation_set_id }
    pub fn score_permit_id(&self) -> &ContentId { &self.score_permit_id }
    pub fn scale(&self) -> u64 { self.scale }
    pub fn predictions(&self) -> &[ForgeProposalValidationPrediction] { &self.predictions }

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
    ) -> Result<(), ForgeProposalValidationPredictionError> {
        let rebuilt = Self::freeze(
            manifest,
            study,
            gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            self.predictions.clone(),
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeProposalValidationPredictionError::PredictionSetIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_prediction_set_id(
    manifest_id: &ContentId,
    study_id: &ContentId,
    gate_id: &ContentId,
    coverage_spec_id: &ContentId,
    coverage_receipt_id: &ContentId,
    score_permit_id: &ContentId,
    fit_permit_id: &ContentId,
    model_id: &ContentId,
    validation_set_id: &ContentId,
    scale: u64,
    predictions: &[ForgeProposalValidationPrediction],
) -> ContentId {
    let count = (predictions.len() as u64).to_be_bytes();
    let mut parts = vec![
        manifest_id.as_str().as_bytes().to_vec(),
        study_id.as_str().as_bytes().to_vec(),
        gate_id.as_str().as_bytes().to_vec(),
        coverage_spec_id.as_str().as_bytes().to_vec(),
        coverage_receipt_id.as_str().as_bytes().to_vec(),
        score_permit_id.as_str().as_bytes().to_vec(),
        fit_permit_id.as_str().as_bytes().to_vec(),
        model_id.as_str().as_bytes().to_vec(),
        validation_set_id.as_str().as_bytes().to_vec(),
        scale.to_be_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        predictions
            .iter()
            .map(|prediction| prediction.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-proposal-validation-prediction-set.v1",
        parts.iter().map(Vec::as_slice),
    )
}

fn observed_records(
    validation: &ForgeProposalValidationSet,
    study: &ForgeProposalStudySpec,
) -> Result<Vec<(ForgeProposalEndpointRecord, bool)>, ForgeProposalValidationPredictionError> {
    let mut records = Vec::new();
    for table in validation.tables() {
        for row in table.rows() {
            let record = ForgeProposalEndpointRecord::from_row(row, study.endpoint())?;
            if let ForgeProposalEndpointValue::Observed(label) = record.value() {
                records.push((record, label));
            }
        }
    }
    records.sort_by(|left, right| left.0.id().as_str().cmp(right.0.id().as_str()));
    let mut seen = BTreeSet::new();
    if records
        .iter()
        .any(|(record, _)| !seen.insert(record.id().as_str().to_string()))
    {
        return Err(ForgeProposalValidationPredictionError::PredictionCoverageMismatch);
    }
    Ok(records)
}

/// Deterministically reconstructed Brier score for one exact prediction set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalDeterministicBrierReceipt {
    id: ContentId,
    prediction_set_id: ContentId,
    model_id: ContentId,
    validation_set_id: ContentId,
    scoring_implementation_id: ContentId,
    scoring_configuration_id: ContentId,
    observed_prediction_count: u64,
    sum_squared_error: u128,
    score: ForgeProposalMetricScore,
}

impl ForgeProposalDeterministicBrierReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn score(
        manifest: &ForgeProposalCorpusManifest,
        study: &ForgeProposalStudySpec,
        gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        predictions: &ForgeProposalValidationPredictionSet,
    ) -> Result<Self, ForgeProposalValidationPredictionError> {
        if study.evaluation().primary() != ForgeProposalMetric::BrierScore
            || gate.primary_metric() != ForgeProposalMetric::BrierScore
        {
            return Err(ForgeProposalValidationPredictionError::UnsupportedMetric);
        }
        let scoring_implementation_id = forge_deterministic_brier_scorer_id();
        let scoring_configuration_id = forge_deterministic_brier_configuration_id();
        if gate.scorer_implementation_id() != &scoring_implementation_id
            || gate.scoring_configuration_id() != &scoring_configuration_id
        {
            return Err(ForgeProposalValidationPredictionError::ScorerIdentityMismatch);
        }
        predictions.validate_for(
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

        let expected = observed_records(validation, study)?;
        let labels = expected
            .iter()
            .map(|(record, label)| (record.id().as_str().to_string(), *label))
            .collect::<BTreeMap<_, _>>();
        if labels.len() != predictions.predictions().len() || labels.is_empty() {
            return Err(ForgeProposalValidationPredictionError::PredictionCoverageMismatch);
        }

        let scale = predictions.scale();
        let mut sum_squared_error = 0u128;
        for prediction in predictions.predictions() {
            let label = *labels
                .get(prediction.endpoint_record_id().as_str())
                .ok_or(ForgeProposalValidationPredictionError::PredictionCoverageMismatch)?;
            let target = if label { scale } else { 0 };
            let difference = prediction.probability_scaled().abs_diff(target) as u128;
            let squared = difference
                .checked_mul(difference)
                .ok_or(ForgeProposalValidationPredictionError::ArithmeticOverflow)?;
            sum_squared_error = sum_squared_error
                .checked_add(squared)
                .ok_or(ForgeProposalValidationPredictionError::ArithmeticOverflow)?;
        }

        let observed_prediction_count = u64::try_from(predictions.predictions().len())
            .map_err(|_| ForgeProposalValidationPredictionError::ArithmeticOverflow)?;
        let score_scaled = brier_scaled_from_sum(sum_squared_error, observed_prediction_count, scale)?;
        let score = ForgeProposalMetricScore::new(
            ForgeProposalMetric::BrierScore,
            scale,
            score_scaled,
        )?;
        let id = derive_brier_receipt_id(
            predictions.id(),
            model.id(),
            validation.id(),
            &scoring_implementation_id,
            &scoring_configuration_id,
            observed_prediction_count,
            sum_squared_error,
            score.id(),
        );
        Ok(Self {
            id,
            prediction_set_id: predictions.id().clone(),
            model_id: model.id().clone(),
            validation_set_id: validation.id().clone(),
            scoring_implementation_id,
            scoring_configuration_id,
            observed_prediction_count,
            sum_squared_error,
            score,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn prediction_set_id(&self) -> &ContentId { &self.prediction_set_id }
    pub fn model_id(&self) -> &ContentId { &self.model_id }
    pub fn validation_set_id(&self) -> &ContentId { &self.validation_set_id }
    pub fn observed_prediction_count(&self) -> u64 { self.observed_prediction_count }
    pub fn sum_squared_error(&self) -> u128 { self.sum_squared_error }
    pub fn score_value(&self) -> &ForgeProposalMetricScore { &self.score }

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
        predictions: &ForgeProposalValidationPredictionSet,
    ) -> Result<(), ForgeProposalValidationPredictionError> {
        let rebuilt = Self::score(
            manifest,
            study,
            gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            predictions,
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeProposalValidationPredictionError::ScoreIdentityMismatch)
        }
    }
}

fn derive_brier_receipt_id(
    prediction_set_id: &ContentId,
    model_id: &ContentId,
    validation_set_id: &ContentId,
    scoring_implementation_id: &ContentId,
    scoring_configuration_id: &ContentId,
    observed_prediction_count: u64,
    sum_squared_error: u128,
    score_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-deterministic-brier-receipt.v1",
        [
            prediction_set_id.as_str().as_bytes(),
            model_id.as_str().as_bytes(),
            validation_set_id.as_str().as_bytes(),
            scoring_implementation_id.as_str().as_bytes(),
            scoring_configuration_id.as_str().as_bytes(),
            observed_prediction_count.to_be_bytes().as_slice(),
            sum_squared_error.to_be_bytes().as_slice(),
            score_id.as_str().as_bytes(),
        ],
    )
}

fn brier_scaled_from_sum(
    sum_squared_error: u128,
    observed_prediction_count: u64,
    scale: u64,
) -> Result<u64, ForgeProposalValidationPredictionError> {
    if scale == 0 {
        return Err(ForgeProposalValidationPredictionError::ZeroProbabilityScale);
    }
    if observed_prediction_count == 0 {
        return Err(ForgeProposalValidationPredictionError::PredictionCoverageMismatch);
    }
    let denominator = u128::from(observed_prediction_count)
        .checked_mul(u128::from(scale))
        .ok_or(ForgeProposalValidationPredictionError::ArithmeticOverflow)?;
    let quotient = sum_squared_error / denominator;
    let remainder = sum_squared_error % denominator;
    let half_up_threshold = denominator / 2 + denominator % 2;
    let rounded = quotient
        .checked_add(u128::from(remainder >= half_up_threshold))
        .ok_or(ForgeProposalValidationPredictionError::ArithmeticOverflow)?;
    let result = u64::try_from(rounded)
        .map_err(|_| ForgeProposalValidationPredictionError::ArithmeticOverflow)?;
    if result > scale {
        return Err(ForgeProposalValidationPredictionError::ArithmeticOverflow);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_brier_exact_extremes_score_zero() {
        let scale = 1_000u64;
        let sum = 0u128;
        assert_eq!(brier_scaled_from_sum(sum, 2, scale).unwrap(), 0);
    }

    #[test]
    fn deterministic_brier_half_probabilities_score_quarter() {
        let scale = 1_000u64;
        let difference = 500u128;
        let sum = difference * difference * 2;
        assert_eq!(brier_scaled_from_sum(sum, 2, scale).unwrap(), 250);
    }

    #[test]
    fn deterministic_brier_uses_round_half_up() {
        // score = 1 / (2 * 3) = 1/6 on scale 3 => 0.5, rounded half-up to 1.
        assert_eq!(brier_scaled_from_sum(1, 2, 3).unwrap(), 0);
        // Explicit tie: numerator 3 / denominator 6 = 0.5 -> 1.
        assert_eq!(brier_scaled_from_sum(3, 2, 3).unwrap(), 1);
    }

    #[test]
    fn scorer_and_configuration_identities_are_distinct() {
        assert_ne!(
            forge_deterministic_brier_scorer_id(),
            forge_deterministic_brier_configuration_id()
        );
    }
}
