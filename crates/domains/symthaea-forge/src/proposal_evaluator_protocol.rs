// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Label-blind evaluator transport protocol for Forge proposal studies.
//!
//! This module defines the artifact boundary between validation-data preparation and model
//! prediction. The request contains only the frozen model/protocol identities plus the exact
//! schema-derived, label-blind feature table. It deliberately does not contain validation proposal
//! tables, endpoint records, endpoint observability, family coverage counts, or labels.
//!
//! The response contains only one fixed-point probability per request feature row plus an opaque
//! execution-context identity. Labels are joined later by the deterministic scoring boundary.
//!
//! This is a protocol theorem, not an OS isolation theorem. A later worker/sandbox implementation
//! must prove that the prediction process actually receives only this request artifact.

use crate::proposal_corpus::{
    ForgeProposalCorpusManifest, ForgeProposalValidationSet,
};
use crate::proposal_model::{
    ForgeProposalFrozenModel, ForgeProposalModelError, ForgeProposalValidationGateSpec,
};
use crate::proposal_study::{ForgeProposalStudyError, ForgeProposalStudySpec};
use crate::proposal_support::ForgeProposalFitPermit;
use crate::proposal_validation_blind::{
    ForgeProposalBlindValidationError, ForgeProposalBlindValidationPrediction,
    ForgeProposalBlindValidationPredictionSet, ForgeProposalValidationTargetSet,
};
use crate::proposal_validation_coverage::{
    ForgeProposalValidationCoverageError, ForgeProposalValidationCoverageReceipt,
    ForgeProposalValidationCoverageSpec, ForgeProposalValidationScorePermit,
};
use crate::proposal_validation_features::{
    ForgeProposalValidationFeatureError, ForgeProposalValidationFeatureRow,
    ForgeProposalValidationFeatureTable,
};
use crate::sequence_stats::ExactContextForgeSequenceCohort;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalEvaluatorProtocolError {
    #[error(transparent)]
    Study(#[from] ForgeProposalStudyError),
    #[error(transparent)]
    Model(#[from] ForgeProposalModelError),
    #[error(transparent)]
    Coverage(#[from] ForgeProposalValidationCoverageError),
    #[error(transparent)]
    Blind(#[from] ForgeProposalBlindValidationError),
    #[error(transparent)]
    Features(#[from] ForgeProposalValidationFeatureError),
    #[error("evaluator protocol implementation/configuration/transport identities must be distinct non-empty content identities")]
    InvalidProtocol,
    #[error("evaluator protocol identity does not match canonical fields")]
    ProtocolIdentityMismatch,
    #[error("evaluator request does not bind the supplied study/model/features/permits")]
    RequestScopeMismatch,
    #[error("evaluator request identity does not match canonical fields")]
    RequestIdentityMismatch,
    #[error("evaluator prediction probability exceeds the frozen fixed-point scale")]
    ProbabilityOutOfRange,
    #[error("evaluator prediction does not bind the supplied request feature row")]
    PredictionScopeMismatch,
    #[error("evaluator prediction identity does not match canonical fields")]
    PredictionIdentityMismatch,
    #[error("evaluator response does not cover every request feature row exactly once")]
    ResponseCoverageMismatch,
    #[error("evaluator response identity does not match canonical fields")]
    ResponseIdentityMismatch,
    #[error("evaluator response cannot be converted to the supplied blind target set/model/gate")]
    BlindConversionMismatch,
}

/// Frozen identity of the evaluator implementation and its wire-format/configuration contract.
///
/// This object intentionally does not claim process, VM, container, seccomp, or network isolation.
/// Those properties belong in a later execution receipt produced by an actual worker launcher.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalEvaluatorProtocolSpec {
    id: ContentId,
    runner_implementation_id: ContentId,
    runner_configuration_id: ContentId,
    transport_schema_id: ContentId,
}

impl ForgeProposalEvaluatorProtocolSpec {
    pub fn new(
        runner_implementation_id: ContentId,
        runner_configuration_id: ContentId,
        transport_schema_id: ContentId,
    ) -> Result<Self, ForgeProposalEvaluatorProtocolError> {
        let ids = [
            runner_implementation_id.as_str(),
            runner_configuration_id.as_str(),
            transport_schema_id.as_str(),
        ];
        if ids.iter().any(|value| value.is_empty())
            || ids[0] == ids[1]
            || ids[0] == ids[2]
            || ids[1] == ids[2]
        {
            return Err(ForgeProposalEvaluatorProtocolError::InvalidProtocol);
        }
        let id = derive_protocol_id(
            &runner_implementation_id,
            &runner_configuration_id,
            &transport_schema_id,
        );
        Ok(Self {
            id,
            runner_implementation_id,
            runner_configuration_id,
            transport_schema_id,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn runner_implementation_id(&self) -> &ContentId { &self.runner_implementation_id }
    pub fn runner_configuration_id(&self) -> &ContentId { &self.runner_configuration_id }
    pub fn transport_schema_id(&self) -> &ContentId { &self.transport_schema_id }

    pub fn validate(&self) -> Result<(), ForgeProposalEvaluatorProtocolError> {
        let rebuilt = Self::new(
            self.runner_implementation_id.clone(),
            self.runner_configuration_id.clone(),
            self.transport_schema_id.clone(),
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeProposalEvaluatorProtocolError::ProtocolIdentityMismatch)
        }
    }
}

fn derive_protocol_id(
    implementation_id: &ContentId,
    configuration_id: &ContentId,
    transport_schema_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-evaluator-protocol.v1",
        [
            implementation_id.as_str().as_bytes(),
            configuration_id.as_str().as_bytes(),
            transport_schema_id.as_str().as_bytes(),
        ],
    )
}

/// Serialization-safe request that is sufficient for prediction but carries no validation labels.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalEvaluationRequest {
    id: ContentId,
    protocol: ForgeProposalEvaluatorProtocolSpec,
    study_id: ContentId,
    validation_gate_id: ContentId,
    validation_coverage_spec_id: ContentId,
    validation_score_permit_id: ContentId,
    fit_permit_id: ContentId,
    model_id: ContentId,
    target_set_id: ContentId,
    feature_table: ForgeProposalValidationFeatureTable,
    score_scale: u64,
}

impl ForgeProposalEvaluationRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn freeze(
        protocol: ForgeProposalEvaluatorProtocolSpec,
        manifest: &ForgeProposalCorpusManifest,
        cohort: &ExactContextForgeSequenceCohort,
        study: &ForgeProposalStudySpec,
        gate: &ForgeProposalValidationGateSpec,
        coverage_spec: &ForgeProposalValidationCoverageSpec,
        coverage_receipt: &ForgeProposalValidationCoverageReceipt,
        score_permit: &ForgeProposalValidationScorePermit,
        fit_permit: &ForgeProposalFitPermit,
        model: &ForgeProposalFrozenModel,
        validation: &ForgeProposalValidationSet,
        targets: &ForgeProposalValidationTargetSet,
        features: ForgeProposalValidationFeatureTable,
    ) -> Result<Self, ForgeProposalEvaluatorProtocolError> {
        protocol.validate()?;
        gate.validate_for(study)?;
        coverage_spec.validate_for(study)?;
        coverage_receipt.validate_for(coverage_spec, manifest, study, validation)?;
        score_permit.validate_for(coverage_spec, coverage_receipt)?;
        model.validate_for(study, gate, coverage_spec, fit_permit)?;
        targets.validate_for(manifest, validation)?;
        features.validate_for(manifest, cohort, study, validation, targets)?;

        if study.validation_set_id() != validation.id()
            || targets.endpoint() != study.endpoint()
            || features.study_id() != study.id()
            || features.target_set_id() != targets.id()
            || features.feature_schema_id() != study.feature_schema().id()
            || score_permit.validation_set_id() != validation.id()
            || model.fit_permit_id() != fit_permit.id()
            || model.validation_coverage_spec_id() != coverage_spec.id()
        {
            return Err(ForgeProposalEvaluatorProtocolError::RequestScopeMismatch);
        }
        let score_scale = gate.score_scale();
        if score_scale == 0 {
            return Err(ForgeProposalEvaluatorProtocolError::RequestScopeMismatch);
        }
        let id = derive_request_id(
            protocol.id(),
            study.id(),
            gate.id(),
            coverage_spec.id(),
            score_permit.id(),
            fit_permit.id(),
            model.id(),
            targets.id(),
            features.id(),
            study.feature_schema().id(),
            score_scale,
        );
        Ok(Self {
            id,
            protocol,
            study_id: study.id().clone(),
            validation_gate_id: gate.id().clone(),
            validation_coverage_spec_id: coverage_spec.id().clone(),
            validation_score_permit_id: score_permit.id().clone(),
            fit_permit_id: fit_permit.id().clone(),
            model_id: model.id().clone(),
            target_set_id: targets.id().clone(),
            feature_table: features,
            score_scale,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn protocol(&self) -> &ForgeProposalEvaluatorProtocolSpec { &self.protocol }
    pub fn study_id(&self) -> &ContentId { &self.study_id }
    pub fn validation_gate_id(&self) -> &ContentId { &self.validation_gate_id }
    pub fn validation_coverage_spec_id(&self) -> &ContentId { &self.validation_coverage_spec_id }
    pub fn validation_score_permit_id(&self) -> &ContentId { &self.validation_score_permit_id }
    pub fn fit_permit_id(&self) -> &ContentId { &self.fit_permit_id }
    pub fn model_id(&self) -> &ContentId { &self.model_id }
    pub fn target_set_id(&self) -> &ContentId { &self.target_set_id }
    pub fn feature_table(&self) -> &ForgeProposalValidationFeatureTable { &self.feature_table }
    pub fn feature_rows(&self) -> &[ForgeProposalValidationFeatureRow] { self.feature_table.rows() }
    pub fn score_scale(&self) -> u64 { self.score_scale }

    pub fn validate_identity(&self) -> Result<(), ForgeProposalEvaluatorProtocolError> {
        self.protocol.validate()?;
        if self.study_id != *self.feature_table.study_id()
            || self.target_set_id != *self.feature_table.target_set_id()
            || self.feature_table.feature_schema_id().as_str().is_empty()
            || self.score_scale == 0
            || self.feature_table.rows().is_empty()
        {
            return Err(ForgeProposalEvaluatorProtocolError::RequestScopeMismatch);
        }
        let expected = derive_request_id(
            self.protocol.id(),
            &self.study_id,
            &self.validation_gate_id,
            &self.validation_coverage_spec_id,
            &self.validation_score_permit_id,
            &self.fit_permit_id,
            &self.model_id,
            &self.target_set_id,
            self.feature_table.id(),
            self.feature_table.feature_schema_id(),
            self.score_scale,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalEvaluatorProtocolError::RequestIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_request_id(
    protocol_id: &ContentId,
    study_id: &ContentId,
    gate_id: &ContentId,
    coverage_spec_id: &ContentId,
    score_permit_id: &ContentId,
    fit_permit_id: &ContentId,
    model_id: &ContentId,
    target_set_id: &ContentId,
    feature_table_id: &ContentId,
    feature_schema_id: &ContentId,
    score_scale: u64,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-evaluation-request.v1",
        [
            protocol_id.as_str().as_bytes(),
            study_id.as_str().as_bytes(),
            gate_id.as_str().as_bytes(),
            coverage_spec_id.as_str().as_bytes(),
            score_permit_id.as_str().as_bytes(),
            fit_permit_id.as_str().as_bytes(),
            model_id.as_str().as_bytes(),
            target_set_id.as_str().as_bytes(),
            feature_table_id.as_str().as_bytes(),
            feature_schema_id.as_str().as_bytes(),
            score_scale.to_be_bytes().as_slice(),
        ],
    )
}

/// One probability emitted by the evaluator for one exact request feature row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalEvaluatorPrediction {
    id: ContentId,
    request_id: ContentId,
    target_id: ContentId,
    scale: u64,
    probability_scaled: u64,
}

impl ForgeProposalEvaluatorPrediction {
    pub fn for_feature_row(
        request: &ForgeProposalEvaluationRequest,
        row: &ForgeProposalValidationFeatureRow,
        probability_scaled: u64,
    ) -> Result<Self, ForgeProposalEvaluatorProtocolError> {
        request.validate_identity()?;
        if probability_scaled > request.score_scale() {
            return Err(ForgeProposalEvaluatorProtocolError::ProbabilityOutOfRange);
        }
        if row.feature_schema_id() != request.feature_table().feature_schema_id()
            || !request
                .feature_rows()
                .iter()
                .any(|candidate| candidate.id() == row.id() && candidate.target_id() == row.target_id())
        {
            return Err(ForgeProposalEvaluatorProtocolError::PredictionScopeMismatch);
        }
        let id = derive_prediction_id(
            request.id(),
            row.target_id(),
            request.score_scale(),
            probability_scaled,
        );
        Ok(Self {
            id,
            request_id: request.id().clone(),
            target_id: row.target_id().clone(),
            scale: request.score_scale(),
            probability_scaled,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn request_id(&self) -> &ContentId { &self.request_id }
    pub fn target_id(&self) -> &ContentId { &self.target_id }
    pub fn scale(&self) -> u64 { self.scale }
    pub fn probability_scaled(&self) -> u64 { self.probability_scaled }

    pub fn validate_for(
        &self,
        request: &ForgeProposalEvaluationRequest,
        row: &ForgeProposalValidationFeatureRow,
    ) -> Result<(), ForgeProposalEvaluatorProtocolError> {
        if self.request_id != *request.id()
            || self.target_id != *row.target_id()
            || self.scale != request.score_scale()
            || self.probability_scaled > self.scale
        {
            return Err(ForgeProposalEvaluatorProtocolError::PredictionScopeMismatch);
        }
        let expected = derive_prediction_id(
            &self.request_id,
            &self.target_id,
            self.scale,
            self.probability_scaled,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalEvaluatorProtocolError::PredictionIdentityMismatch)
        }
    }
}

fn derive_prediction_id(
    request_id: &ContentId,
    target_id: &ContentId,
    scale: u64,
    probability_scaled: u64,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-evaluator-prediction.v1",
        [
            request_id.as_str().as_bytes(),
            target_id.as_str().as_bytes(),
            scale.to_be_bytes().as_slice(),
            probability_scaled.to_be_bytes().as_slice(),
        ],
    )
}

/// Label-blind evaluator output. It cannot contain labels, endpoint records, or validation tables.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalEvaluationResponse {
    id: ContentId,
    request_id: ContentId,
    protocol_id: ContentId,
    model_id: ContentId,
    target_set_id: ContentId,
    execution_context_id: ContentId,
    predictions: Vec<ForgeProposalEvaluatorPrediction>,
}

impl ForgeProposalEvaluationResponse {
    pub fn freeze(
        request: &ForgeProposalEvaluationRequest,
        execution_context_id: ContentId,
        mut predictions: Vec<ForgeProposalEvaluatorPrediction>,
    ) -> Result<Self, ForgeProposalEvaluatorProtocolError> {
        request.validate_identity()?;
        predictions.sort_by(|left, right| left.target_id().cmp(right.target_id()));
        if predictions.len() != request.feature_rows().len()
            || predictions
                .windows(2)
                .any(|pair| pair[0].target_id() == pair[1].target_id())
        {
            return Err(ForgeProposalEvaluatorProtocolError::ResponseCoverageMismatch);
        }
        let rows = request
            .feature_rows()
            .iter()
            .map(|row| (row.target_id().as_str().to_string(), row))
            .collect::<BTreeMap<_, _>>();
        if rows.len() != request.feature_rows().len() {
            return Err(ForgeProposalEvaluatorProtocolError::ResponseCoverageMismatch);
        }
        for prediction in &predictions {
            let row = rows
                .get(prediction.target_id().as_str())
                .ok_or(ForgeProposalEvaluatorProtocolError::ResponseCoverageMismatch)?;
            prediction.validate_for(request, row)?;
        }
        let predicted_targets = predictions
            .iter()
            .map(|prediction| prediction.target_id().as_str().to_string())
            .collect::<BTreeSet<_>>();
        if predicted_targets.len() != rows.len()
            || !rows.keys().all(|target| predicted_targets.contains(target))
        {
            return Err(ForgeProposalEvaluatorProtocolError::ResponseCoverageMismatch);
        }
        let id = derive_response_id(
            request.id(),
            request.protocol().id(),
            request.model_id(),
            request.target_set_id(),
            &execution_context_id,
            &predictions,
        );
        Ok(Self {
            id,
            request_id: request.id().clone(),
            protocol_id: request.protocol().id().clone(),
            model_id: request.model_id().clone(),
            target_set_id: request.target_set_id().clone(),
            execution_context_id,
            predictions,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn request_id(&self) -> &ContentId { &self.request_id }
    pub fn model_id(&self) -> &ContentId { &self.model_id }
    pub fn target_set_id(&self) -> &ContentId { &self.target_set_id }
    pub fn execution_context_id(&self) -> &ContentId { &self.execution_context_id }
    pub fn predictions(&self) -> &[ForgeProposalEvaluatorPrediction] { &self.predictions }

    pub fn validate_for(
        &self,
        request: &ForgeProposalEvaluationRequest,
    ) -> Result<(), ForgeProposalEvaluatorProtocolError> {
        let rebuilt = Self::freeze(
            request,
            self.execution_context_id.clone(),
            self.predictions.clone(),
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeProposalEvaluatorProtocolError::ResponseIdentityMismatch)
        }
    }

    /// Convert the protocol response into the existing blind prediction-set type without touching
    /// labels. This is the only bridge needed by the downstream deterministic scorer.
    pub fn to_blind_prediction_set(
        &self,
        request: &ForgeProposalEvaluationRequest,
        model: &ForgeProposalFrozenModel,
        gate: &ForgeProposalValidationGateSpec,
        targets: &ForgeProposalValidationTargetSet,
    ) -> Result<ForgeProposalBlindValidationPredictionSet, ForgeProposalEvaluatorProtocolError> {
        self.validate_for(request)?;
        if self.model_id != *model.id()
            || self.target_set_id != *targets.id()
            || request.model_id() != model.id()
            || request.target_set_id() != targets.id()
            || request.score_scale() != gate.score_scale()
        {
            return Err(ForgeProposalEvaluatorProtocolError::BlindConversionMismatch);
        }
        let target_map = targets
            .targets()
            .iter()
            .map(|target| (target.id().as_str().to_string(), target))
            .collect::<BTreeMap<_, _>>();
        let mut blind = Vec::with_capacity(self.predictions.len());
        for prediction in &self.predictions {
            let target = target_map
                .get(prediction.target_id().as_str())
                .ok_or(ForgeProposalEvaluatorProtocolError::BlindConversionMismatch)?;
            blind.push(ForgeProposalBlindValidationPrediction::for_target(
                model,
                target,
                request.score_scale(),
                prediction.probability_scaled(),
            )?);
        }
        ForgeProposalBlindValidationPredictionSet::freeze(model, gate, targets, blind)
            .map_err(ForgeProposalEvaluatorProtocolError::from)
    }
}

fn derive_response_id(
    request_id: &ContentId,
    protocol_id: &ContentId,
    model_id: &ContentId,
    target_set_id: &ContentId,
    execution_context_id: &ContentId,
    predictions: &[ForgeProposalEvaluatorPrediction],
) -> ContentId {
    let count = (predictions.len() as u64).to_be_bytes();
    let mut parts = vec![
        request_id.as_str().as_bytes().to_vec(),
        protocol_id.as_str().as_bytes().to_vec(),
        model_id.as_str().as_bytes().to_vec(),
        target_set_id.as_str().as_bytes().to_vec(),
        execution_context_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        predictions
            .iter()
            .map(|prediction| prediction.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-proposal-evaluation-response.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    #[test]
    fn protocol_identity_binds_runner_configuration_and_transport() {
        let base = ForgeProposalEvaluatorProtocolSpec::new(
            cid("runner", "implementation-a"),
            cid("runner-config", "config-a"),
            cid("transport", "schema-a"),
        )
        .unwrap();
        let changed_config = ForgeProposalEvaluatorProtocolSpec::new(
            cid("runner", "implementation-a"),
            cid("runner-config", "config-b"),
            cid("transport", "schema-a"),
        )
        .unwrap();
        let changed_transport = ForgeProposalEvaluatorProtocolSpec::new(
            cid("runner", "implementation-a"),
            cid("runner-config", "config-a"),
            cid("transport", "schema-b"),
        )
        .unwrap();
        assert_ne!(base.id(), changed_config.id());
        assert_ne!(base.id(), changed_transport.id());
        base.validate().unwrap();
    }
}
