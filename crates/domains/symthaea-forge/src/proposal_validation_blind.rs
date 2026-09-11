// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Label-blind public validation prediction surface.
//!
//! Prediction generation must not receive the endpoint label it is supposed to predict. This module
//! therefore exposes one target for every validation `(attempt, family)` row using only pre-decision
//! state. A model must freeze probabilities for the complete target set before scoring joins those
//! targets back to observed/censored/counterfactual endpoint values internally.

use crate::family_learning::ForgeTransformationFamilyId;
use crate::proposal_corpus::{
    ForgeProposalCorpusError, ForgeProposalCorpusManifest, ForgeProposalValidationSet,
};
use crate::proposal_endpoints::{ForgeProposalEndpoint, ForgeProposalEndpointRecord, ForgeProposalEndpointValue};
use crate::proposal_model::{
    ForgeProposalFrozenModel, ForgeProposalMetricScore, ForgeProposalValidationGateSpec,
};
use crate::proposal_study::ForgeProposalStudySpec;
use crate::proposal_support::ForgeProposalFitPermit;
use crate::proposal_validation_coverage::{
    ForgeProposalValidationCoverageReceipt, ForgeProposalValidationCoverageSpec,
    ForgeProposalValidationScorePermit,
};
use crate::proposal_validation_predictions::{
    ForgeProposalDeterministicBrierReceipt, ForgeProposalValidationPrediction,
    ForgeProposalValidationPredictionError, ForgeProposalValidationPredictionSet,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalBlindValidationError {
    #[error(transparent)]
    Corpus(#[from] ForgeProposalCorpusError),
    #[error(transparent)]
    Internal(#[from] ForgeProposalValidationPredictionError),
    #[error("validation target set is empty or contains duplicate/non-canonical targets")]
    InvalidTargetSet,
    #[error("validation target does not match the supplied pre-decision proposal row")]
    TargetScopeMismatch,
    #[error("validation target identity does not match canonical pre-decision fields")]
    TargetIdentityMismatch,
    #[error("blind prediction probability exceeds its fixed-point scale")]
    ProbabilityOutOfRange,
    #[error("blind prediction does not bind the supplied model/target/scale")]
    PredictionScopeMismatch,
    #[error("blind prediction identity does not match canonical fields")]
    PredictionIdentityMismatch,
    #[error("blind prediction set does not cover every validation target exactly once")]
    PredictionCoverageMismatch,
    #[error("blind prediction set identity does not match canonical fields")]
    PredictionSetIdentityMismatch,
    #[error("blind deterministic Brier receipt identity does not match canonical fields")]
    BrierIdentityMismatch,
}

fn endpoint_tag(endpoint: ForgeProposalEndpoint) -> &'static [u8] {
    match endpoint {
        ForgeProposalEndpoint::DistinctCandidate => b"distinct-candidate",
        ForgeProposalEndpoint::CompilePassed => b"compile-passed",
        ForgeProposalEndpoint::CorrectnessPassed => b"correctness-passed",
        ForgeProposalEndpoint::EvaluationValid => b"evaluation-valid",
        ForgeProposalEndpoint::SelectedForContinuation => b"selected-for-continuation",
    }
}

/// One label-blind validation prediction target.
///
/// The target ID commits exact run/attempt/parent identity internally, but those high-cardinality
/// identifiers are intentionally not exposed as model features here. Public feature values are
/// limited to the current study's pre-decision v1 vocabulary subset available directly on a proposal
/// row: family, generation, and opportunity counts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalValidationTarget {
    id: ContentId,
    family_id: ForgeTransformationFamilyId,
    generation: u64,
    family_eligible_sites: u64,
    total_eligible_sites: u64,
}

impl ForgeProposalValidationTarget {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn family_id(&self) -> &ForgeTransformationFamilyId { &self.family_id }
    pub fn generation(&self) -> u64 { self.generation }
    pub fn family_eligible_sites(&self) -> u64 { self.family_eligible_sites }
    pub fn total_eligible_sites(&self) -> u64 { self.total_eligible_sites }
}

fn target_from_row(
    run_id: &ContentId,
    row: &crate::proposal_dataset::ForgeProposalObservationRow,
    endpoint: ForgeProposalEndpoint,
) -> ForgeProposalValidationTarget {
    let id = derive_target_id(
        run_id,
        row.attempt_id().as_content_id(),
        row.parent_artifact_id(),
        row.family_id(),
        row.generation(),
        row.eligible_sites(),
        row.total_eligible_sites(),
        endpoint,
    );
    ForgeProposalValidationTarget {
        id,
        family_id: row.family_id().clone(),
        generation: row.generation(),
        family_eligible_sites: row.eligible_sites(),
        total_eligible_sites: row.total_eligible_sites(),
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_target_id(
    run_id: &ContentId,
    attempt_id: &ContentId,
    parent_artifact_id: &ContentId,
    family_id: &ForgeTransformationFamilyId,
    generation: u64,
    eligible_sites: u64,
    total_eligible_sites: u64,
    endpoint: ForgeProposalEndpoint,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-validation-target.v1",
        [
            run_id.as_str().as_bytes(),
            attempt_id.as_str().as_bytes(),
            parent_artifact_id.as_str().as_bytes(),
            family_id.as_content_id().as_str().as_bytes(),
            generation.to_be_bytes().as_slice(),
            eligible_sites.to_be_bytes().as_slice(),
            total_eligible_sites.to_be_bytes().as_slice(),
            endpoint_tag(endpoint),
        ],
    )
}

/// Canonical label-blind target set for every validation proposal row, including rows whose endpoint
/// is later censored or counterfactual-unobserved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalValidationTargetSet {
    id: ContentId,
    endpoint: ForgeProposalEndpoint,
    targets: Vec<ForgeProposalValidationTarget>,
}

impl ForgeProposalValidationTargetSet {
    pub fn from_validation(
        manifest: &ForgeProposalCorpusManifest,
        validation: &ForgeProposalValidationSet,
        endpoint: ForgeProposalEndpoint,
    ) -> Result<Self, ForgeProposalBlindValidationError> {
        manifest.validate()?;
        validation.validate_for(manifest)?;
        let mut targets = Vec::new();
        for table in validation.tables() {
            for row in table.rows() {
                targets.push(target_from_row(table.run_id(), row, endpoint));
            }
        }
        targets.sort_by(|left, right| left.id().as_str().cmp(right.id().as_str()));
        if targets.is_empty()
            || targets.windows(2).any(|pair| pair[0].id() == pair[1].id())
        {
            return Err(ForgeProposalBlindValidationError::InvalidTargetSet);
        }
        let id = derive_target_set_id(endpoint, &targets);
        Ok(Self { id, endpoint, targets })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn endpoint(&self) -> ForgeProposalEndpoint { self.endpoint }
    pub fn targets(&self) -> &[ForgeProposalValidationTarget] { &self.targets }

    pub fn validate_for(
        &self,
        manifest: &ForgeProposalCorpusManifest,
        validation: &ForgeProposalValidationSet,
    ) -> Result<(), ForgeProposalBlindValidationError> {
        let rebuilt = Self::from_validation(manifest, validation, self.endpoint)?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeProposalBlindValidationError::TargetIdentityMismatch)
        }
    }
}

fn derive_target_set_id(
    endpoint: ForgeProposalEndpoint,
    targets: &[ForgeProposalValidationTarget],
) -> ContentId {
    let count = (targets.len() as u64).to_be_bytes();
    let mut parts = vec![endpoint_tag(endpoint).to_vec(), count.to_vec()];
    parts.extend(targets.iter().map(|target| target.id().as_str().as_bytes().to_vec()));
    ContentId::derive(
        "symthaea.forge-proposal-validation-target-set.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// One model probability attached to a label-blind target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalBlindValidationPrediction {
    id: ContentId,
    model_id: ContentId,
    target_id: ContentId,
    scale: u64,
    probability_scaled: u64,
}

impl ForgeProposalBlindValidationPrediction {
    pub fn for_target(
        model: &ForgeProposalFrozenModel,
        target: &ForgeProposalValidationTarget,
        scale: u64,
        probability_scaled: u64,
    ) -> Result<Self, ForgeProposalBlindValidationError> {
        if scale == 0 || probability_scaled > scale {
            return Err(ForgeProposalBlindValidationError::ProbabilityOutOfRange);
        }
        let id = derive_blind_prediction_id(model.id(), target.id(), scale, probability_scaled);
        Ok(Self {
            id,
            model_id: model.id().clone(),
            target_id: target.id().clone(),
            scale,
            probability_scaled,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn model_id(&self) -> &ContentId { &self.model_id }
    pub fn target_id(&self) -> &ContentId { &self.target_id }
    pub fn scale(&self) -> u64 { self.scale }
    pub fn probability_scaled(&self) -> u64 { self.probability_scaled }

    pub fn validate_for(
        &self,
        model: &ForgeProposalFrozenModel,
        target: &ForgeProposalValidationTarget,
        scale: u64,
    ) -> Result<(), ForgeProposalBlindValidationError> {
        if scale == 0 || self.probability_scaled > scale {
            return Err(ForgeProposalBlindValidationError::ProbabilityOutOfRange);
        }
        if self.model_id != *model.id() || self.target_id != *target.id() || self.scale != scale {
            return Err(ForgeProposalBlindValidationError::PredictionScopeMismatch);
        }
        let expected = derive_blind_prediction_id(
            &self.model_id,
            &self.target_id,
            self.scale,
            self.probability_scaled,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalBlindValidationError::PredictionIdentityMismatch)
        }
    }
}

fn derive_blind_prediction_id(
    model_id: &ContentId,
    target_id: &ContentId,
    scale: u64,
    probability_scaled: u64,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-blind-validation-prediction.v1",
        [
            model_id.as_str().as_bytes(),
            target_id.as_str().as_bytes(),
            scale.to_be_bytes().as_slice(),
            probability_scaled.to_be_bytes().as_slice(),
        ],
    )
}

/// Exact prediction coverage for every label-blind validation target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalBlindValidationPredictionSet {
    id: ContentId,
    model_id: ContentId,
    target_set_id: ContentId,
    scale: u64,
    predictions: Vec<ForgeProposalBlindValidationPrediction>,
}

impl ForgeProposalBlindValidationPredictionSet {
    pub fn freeze(
        model: &ForgeProposalFrozenModel,
        gate: &ForgeProposalValidationGateSpec,
        targets: &ForgeProposalValidationTargetSet,
        mut predictions: Vec<ForgeProposalBlindValidationPrediction>,
    ) -> Result<Self, ForgeProposalBlindValidationError> {
        let scale = gate.score_scale();
        if scale == 0 {
            return Err(ForgeProposalBlindValidationError::ProbabilityOutOfRange);
        }
        predictions.sort_by(|left, right| left.target_id().as_str().cmp(right.target_id().as_str()));
        if predictions.len() != targets.targets().len()
            || predictions.windows(2).any(|pair| pair[0].target_id() == pair[1].target_id())
        {
            return Err(ForgeProposalBlindValidationError::PredictionCoverageMismatch);
        }
        for (prediction, target) in predictions.iter().zip(targets.targets()) {
            if prediction.target_id() != target.id() {
                return Err(ForgeProposalBlindValidationError::PredictionCoverageMismatch);
            }
            prediction.validate_for(model, target, scale)?;
        }
        let id = derive_blind_prediction_set_id(model.id(), targets.id(), scale, &predictions);
        Ok(Self {
            id,
            model_id: model.id().clone(),
            target_set_id: targets.id().clone(),
            scale,
            predictions,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn model_id(&self) -> &ContentId { &self.model_id }
    pub fn target_set_id(&self) -> &ContentId { &self.target_set_id }
    pub fn scale(&self) -> u64 { self.scale }
    pub fn predictions(&self) -> &[ForgeProposalBlindValidationPrediction] { &self.predictions }

    pub fn validate_for(
        &self,
        model: &ForgeProposalFrozenModel,
        gate: &ForgeProposalValidationGateSpec,
        targets: &ForgeProposalValidationTargetSet,
    ) -> Result<(), ForgeProposalBlindValidationError> {
        let rebuilt = Self::freeze(model, gate, targets, self.predictions.clone())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeProposalBlindValidationError::PredictionSetIdentityMismatch)
        }
    }
}

fn derive_blind_prediction_set_id(
    model_id: &ContentId,
    target_set_id: &ContentId,
    scale: u64,
    predictions: &[ForgeProposalBlindValidationPrediction],
) -> ContentId {
    let count = (predictions.len() as u64).to_be_bytes();
    let mut parts = vec![
        model_id.as_str().as_bytes().to_vec(),
        target_set_id.as_str().as_bytes().to_vec(),
        scale.to_be_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        predictions
            .iter()
            .map(|prediction| prediction.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-proposal-blind-validation-prediction-set.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Public deterministic Brier evidence built only after a complete label-blind prediction set has
/// been frozen. Endpoint labels are joined internally at this stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalBlindDeterministicBrierReceipt {
    id: ContentId,
    target_set_id: ContentId,
    blind_prediction_set_id: ContentId,
    inner: ForgeProposalDeterministicBrierReceipt,
}

impl ForgeProposalBlindDeterministicBrierReceipt {
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
        targets: &ForgeProposalValidationTargetSet,
        blind_predictions: &ForgeProposalBlindValidationPredictionSet,
    ) -> Result<Self, ForgeProposalBlindValidationError> {
        if targets.endpoint() != study.endpoint() {
            return Err(ForgeProposalBlindValidationError::TargetScopeMismatch);
        }
        targets.validate_for(manifest, validation)?;
        blind_predictions.validate_for(model, gate, targets)?;

        let by_target = blind_predictions
            .predictions()
            .iter()
            .map(|prediction| (prediction.target_id().as_str().to_string(), prediction))
            .collect::<BTreeMap<_, _>>();
        let mut observed_predictions = Vec::new();
        let mut seen_targets = BTreeSet::new();
        for table in validation.tables() {
            for row in table.rows() {
                let target = target_from_row(table.run_id(), row, study.endpoint());
                let prediction = by_target
                    .get(target.id().as_str())
                    .ok_or(ForgeProposalBlindValidationError::PredictionCoverageMismatch)?;
                seen_targets.insert(target.id().as_str().to_string());
                let record = ForgeProposalEndpointRecord::from_row(row, study.endpoint())
                    .map_err(ForgeProposalValidationPredictionError::from)?;
                if matches!(record.value(), ForgeProposalEndpointValue::Observed(_)) {
                    observed_predictions.push(ForgeProposalValidationPrediction::for_record(
                        model,
                        &record,
                        blind_predictions.scale(),
                        prediction.probability_scaled(),
                    )?);
                }
            }
        }
        if seen_targets.len() != targets.targets().len() {
            return Err(ForgeProposalBlindValidationError::PredictionCoverageMismatch);
        }

        let internal_set = ForgeProposalValidationPredictionSet::freeze(
            manifest,
            study,
            gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            observed_predictions,
        )?;
        let inner = ForgeProposalDeterministicBrierReceipt::score(
            manifest,
            study,
            gate,
            coverage_spec,
            coverage_receipt,
            score_permit,
            fit_permit,
            model,
            validation,
            &internal_set,
        )?;
        let id = derive_blind_brier_receipt_id(targets.id(), blind_predictions.id(), inner.id());
        Ok(Self {
            id,
            target_set_id: targets.id().clone(),
            blind_prediction_set_id: blind_predictions.id().clone(),
            inner,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn target_set_id(&self) -> &ContentId { &self.target_set_id }
    pub fn blind_prediction_set_id(&self) -> &ContentId { &self.blind_prediction_set_id }
    pub fn score_value(&self) -> &ForgeProposalMetricScore { self.inner.score_value() }
    pub fn observed_prediction_count(&self) -> u64 { self.inner.observed_prediction_count() }
    pub fn sum_squared_error(&self) -> u128 { self.inner.sum_squared_error() }

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
    ) -> Result<(), ForgeProposalBlindValidationError> {
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
            targets,
            blind_predictions,
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeProposalBlindValidationError::BrierIdentityMismatch)
        }
    }

    pub(crate) fn inner(&self) -> &ForgeProposalDeterministicBrierReceipt { &self.inner }
}

fn derive_blind_brier_receipt_id(
    target_set_id: &ContentId,
    blind_prediction_set_id: &ContentId,
    inner_brier_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-blind-deterministic-brier-receipt.v1",
        [
            target_set_id.as_str().as_bytes(),
            blind_prediction_set_id.as_str().as_bytes(),
            inner_brier_id.as_str().as_bytes(),
        ],
    )
}
