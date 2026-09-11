// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation/holdout evidence bound to one exact bounded direct evaluator launch.
//!
//! `proposal_evaluator_validation` proves that deterministic validation used one exact label-blind
//! evaluator request/response pair. This module strengthens that proposition by requiring the exact
//! `ForgeProposalDirectLaunchReceipt` that produced the response and by revalidating its launch
//! policy, process identity, transport commitments, and execution record.
//!
//! This still does not upgrade runtime isolation: the embedded v1 execution record remains
//! `NotEstablishedV1` for filesystem/network/namespace/seccomp/VM containment.

use crate::proposal_corpus::{
    ForgeProposalCorpusManifest, ForgeProposalHoldoutSeal, ForgeProposalValidationSet,
};
use crate::proposal_evaluator_execution::{
    ForgeProposalEvaluatorExecutionError, ForgeProposalEvaluatorLaunchPolicy,
};
use crate::proposal_evaluator_launcher::{
    forge_direct_evaluator_launcher_configuration_id,
    forge_direct_evaluator_launcher_implementation_id, forge_direct_evaluator_transport_schema_id,
    ForgeProposalDirectLaunchReceipt,
};
use crate::proposal_evaluator_protocol::{
    ForgeProposalEvaluationRequest, ForgeProposalEvaluationResponse,
    ForgeProposalEvaluatorProtocolError,
};
use crate::proposal_evaluator_validation::{
    ForgeProposalEvaluatorHoldoutPermit, ForgeProposalEvaluatorValidationError,
    ForgeProposalEvaluatorValidationReceipt,
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
pub enum ForgeProposalLaunchValidationError {
    #[error(transparent)]
    Protocol(#[from] ForgeProposalEvaluatorProtocolError),
    #[error(transparent)]
    Execution(#[from] ForgeProposalEvaluatorExecutionError),
    #[error(transparent)]
    Validation(#[from] ForgeProposalEvaluatorValidationError),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
    #[error("direct launch receipt does not bind the supplied policy/request/response")]
    LaunchScopeMismatch,
    #[error("direct launch receipt identity does not match canonical fields")]
    LaunchIdentityMismatch,
    #[error("launch-bound validation receipt does not bind the supplied launch/evaluator evidence")]
    ValidationScopeMismatch,
    #[error("launch-bound validation receipt identity does not match canonical fields")]
    ValidationIdentityMismatch,
    #[error("launch-bound holdout permit does not bind the supplied launch/validation evidence")]
    HoldoutScopeMismatch,
    #[error("launch-bound holdout permit identity does not match canonical fields")]
    HoldoutIdentityMismatch,
}

/// Revalidate the public fields of one direct-launch receipt against its exact semantic request,
/// response and precommitted launch policy.
///
/// The raw runner stdout bytes are represented by `stdout_wire_id`; v1 does not retain those bytes
/// inside the receipt. This theorem therefore verifies the opaque commitment and all construction
/// relationships, not independent re-observation of already-discarded stdout bytes.
pub fn validate_direct_launch_receipt(
    policy: &ForgeProposalEvaluatorLaunchPolicy,
    request: &ForgeProposalEvaluationRequest,
    response: &ForgeProposalEvaluationResponse,
    receipt: &ForgeProposalDirectLaunchReceipt,
) -> Result<(), ForgeProposalLaunchValidationError> {
    policy.validate_for(request.protocol())?;
    response.validate_for(request)?;
    receipt
        .execution_record()
        .validate_for(policy, request, response)?;

    let protocol = request.protocol();
    if policy.launcher_implementation_id()
        != &forge_direct_evaluator_launcher_implementation_id()
        || policy.launcher_configuration_id()
            != &forge_direct_evaluator_launcher_configuration_id()
        || protocol.transport_schema_id() != &forge_direct_evaluator_transport_schema_id()
        || receipt.runner_artifact_id() != protocol.runner_implementation_id()
        || receipt.runner_argv_id() != protocol.runner_configuration_id()
        || receipt.transport_schema_id() != protocol.transport_schema_id()
        || receipt.exit_code() != 0
        || receipt.wall_time_ms() != receipt.execution_record().wall_time_ms()
        || receipt.execution_record().request_id() != request.id()
        || receipt.execution_record().response_id() != response.id()
        || receipt.execution_record().execution_context_id() != response.execution_context_id()
    {
        return Err(ForgeProposalLaunchValidationError::LaunchScopeMismatch);
    }

    let request_wire = serde_json::to_vec(request)?;
    let expected_stdin_wire_id = ContentId::derive(
        "symthaea.forge-direct-evaluator-observed-stdin.v1",
        [request_wire.as_slice()],
    );
    if receipt.stdin_wire_id() != &expected_stdin_wire_id {
        return Err(ForgeProposalLaunchValidationError::LaunchScopeMismatch);
    }

    let expected_launcher_evidence_id = ContentId::derive(
        "symthaea.forge-direct-evaluator-launch-evidence.v1",
        [
            policy.id().as_str().as_bytes(),
            receipt.runner_artifact_id().as_str().as_bytes(),
            receipt.runner_argv_id().as_str().as_bytes(),
            receipt.stdin_wire_id().as_str().as_bytes(),
            receipt.stdout_wire_id().as_str().as_bytes(),
            receipt
                .execution_record()
                .stderr_artifact_id()
                .as_str()
                .as_bytes(),
            receipt.wall_time_ms().to_be_bytes().as_slice(),
        ],
    );
    if receipt.execution_record().launcher_evidence_id() != &expected_launcher_evidence_id {
        return Err(ForgeProposalLaunchValidationError::LaunchScopeMismatch);
    }

    let expected_id = derive_direct_launch_receipt_id(receipt);
    if receipt.id() == &expected_id {
        Ok(())
    } else {
        Err(ForgeProposalLaunchValidationError::LaunchIdentityMismatch)
    }
}

fn derive_direct_launch_receipt_id(receipt: &ForgeProposalDirectLaunchReceipt) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-direct-launch-receipt.v1",
        [
            receipt.execution_record().id().as_str().as_bytes(),
            receipt.runner_artifact_id().as_str().as_bytes(),
            receipt.runner_argv_id().as_str().as_bytes(),
            receipt.transport_schema_id().as_str().as_bytes(),
            receipt.stdin_wire_id().as_str().as_bytes(),
            receipt.stdout_wire_id().as_str().as_bytes(),
            receipt.exit_code().to_be_bytes().as_slice(),
            receipt.wall_time_ms().to_be_bytes().as_slice(),
        ],
    )
}

/// Validation receipt proving that the exact response scored by the deterministic validation chain
/// is the response bound by one exact bounded direct-launch receipt.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalLaunchedEvaluatorValidationReceipt {
    id: ContentId,
    launch_policy_id: ContentId,
    direct_launch_receipt_id: ContentId,
    evaluator_validation_receipt: ForgeProposalEvaluatorValidationReceipt,
}

impl ForgeProposalLaunchedEvaluatorValidationReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn record_brier(
        launch_policy: &ForgeProposalEvaluatorLaunchPolicy,
        launch_receipt: &ForgeProposalDirectLaunchReceipt,
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
    ) -> Result<Self, ForgeProposalLaunchValidationError> {
        validate_direct_launch_receipt(launch_policy, request, response, launch_receipt)?;
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
        if evaluator_validation_receipt.request_id() != request.id()
            || evaluator_validation_receipt.response_id() != response.id()
            || evaluator_validation_receipt.execution_context_id()
                != response.execution_context_id()
            || launch_receipt.execution_record().execution_context_id()
                != response.execution_context_id()
        {
            return Err(ForgeProposalLaunchValidationError::ValidationScopeMismatch);
        }
        let id = derive_validation_receipt_id(
            launch_policy.id(),
            launch_receipt.id(),
            evaluator_validation_receipt.id(),
        );
        Ok(Self {
            id,
            launch_policy_id: launch_policy.id().clone(),
            direct_launch_receipt_id: launch_receipt.id().clone(),
            evaluator_validation_receipt,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn launch_policy_id(&self) -> &ContentId { &self.launch_policy_id }
    pub fn direct_launch_receipt_id(&self) -> &ContentId { &self.direct_launch_receipt_id }
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
        launch_policy: &ForgeProposalEvaluatorLaunchPolicy,
        launch_receipt: &ForgeProposalDirectLaunchReceipt,
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
    ) -> Result<(), ForgeProposalLaunchValidationError> {
        validate_direct_launch_receipt(launch_policy, request, response, launch_receipt)?;
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
        if self.launch_policy_id != *launch_policy.id()
            || self.direct_launch_receipt_id != *launch_receipt.id()
            || self.evaluator_validation_receipt.request_id() != request.id()
            || self.evaluator_validation_receipt.response_id() != response.id()
            || self.evaluator_validation_receipt.execution_context_id()
                != response.execution_context_id()
        {
            return Err(ForgeProposalLaunchValidationError::ValidationScopeMismatch);
        }
        let expected = derive_validation_receipt_id(
            &self.launch_policy_id,
            &self.direct_launch_receipt_id,
            self.evaluator_validation_receipt.id(),
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalLaunchValidationError::ValidationIdentityMismatch)
        }
    }

    pub(crate) fn evaluator_validation_receipt(&self) -> &ForgeProposalEvaluatorValidationReceipt {
        &self.evaluator_validation_receipt
    }
}

fn derive_validation_receipt_id(
    launch_policy_id: &ContentId,
    launch_receipt_id: &ContentId,
    evaluator_validation_receipt_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-launched-evaluator-validation-receipt.v1",
        [
            launch_policy_id.as_str().as_bytes(),
            launch_receipt_id.as_str().as_bytes(),
            evaluator_validation_receipt_id.as_str().as_bytes(),
        ],
    )
}

/// Identity-only holdout permit whose validation prerequisite is bound to one exact direct launch.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalLaunchedEvaluatorHoldoutPermit {
    id: ContentId,
    launched_validation_receipt_id: ContentId,
    launch_policy_id: ContentId,
    direct_launch_receipt_id: ContentId,
    inner: ForgeProposalEvaluatorHoldoutPermit,
}

impl ForgeProposalLaunchedEvaluatorHoldoutPermit {
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        launch_policy: &ForgeProposalEvaluatorLaunchPolicy,
        launch_receipt: &ForgeProposalDirectLaunchReceipt,
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
        receipt: &ForgeProposalLaunchedEvaluatorValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<Self, ForgeProposalLaunchValidationError> {
        receipt.validate_for(
            launch_policy,
            launch_receipt,
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
            launch_policy.id(),
            launch_receipt.id(),
        );
        Ok(Self {
            id,
            launched_validation_receipt_id: receipt.id().clone(),
            launch_policy_id: launch_policy.id().clone(),
            direct_launch_receipt_id: launch_receipt.id().clone(),
            inner,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn launched_validation_receipt_id(&self) -> &ContentId {
        &self.launched_validation_receipt_id
    }
    pub fn launch_policy_id(&self) -> &ContentId { &self.launch_policy_id }
    pub fn direct_launch_receipt_id(&self) -> &ContentId { &self.direct_launch_receipt_id }
    pub fn model_id(&self) -> &ContentId { self.inner.model_id() }
    pub fn holdout_seal_id(&self) -> &ContentId { self.inner.holdout_seal_id() }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        launch_policy: &ForgeProposalEvaluatorLaunchPolicy,
        launch_receipt: &ForgeProposalDirectLaunchReceipt,
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
        receipt: &ForgeProposalLaunchedEvaluatorValidationReceipt,
        holdout: &ForgeProposalHoldoutSeal,
    ) -> Result<(), ForgeProposalLaunchValidationError> {
        receipt.validate_for(
            launch_policy,
            launch_receipt,
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
        if self.launched_validation_receipt_id != *receipt.id()
            || self.launch_policy_id != *launch_policy.id()
            || self.direct_launch_receipt_id != *launch_receipt.id()
        {
            return Err(ForgeProposalLaunchValidationError::HoldoutScopeMismatch);
        }
        let expected = derive_holdout_permit_id(
            self.inner.id(),
            &self.launched_validation_receipt_id,
            &self.launch_policy_id,
            &self.direct_launch_receipt_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalLaunchValidationError::HoldoutIdentityMismatch)
        }
    }
}

fn derive_holdout_permit_id(
    inner_permit_id: &ContentId,
    launched_validation_receipt_id: &ContentId,
    launch_policy_id: &ContentId,
    launch_receipt_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-launched-evaluator-holdout-permit.v1",
        [
            inner_permit_id.as_str().as_bytes(),
            launched_validation_receipt_id.as_str().as_bytes(),
            launch_policy_id.as_str().as_bytes(),
            launch_receipt_id.as_str().as_bytes(),
        ],
    )
}
