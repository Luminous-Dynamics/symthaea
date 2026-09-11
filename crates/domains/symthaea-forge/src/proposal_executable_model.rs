// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Executable-model binding for the first Forge evaluator implementation.
//!
//! V1 deliberately chooses a narrow model format: the frozen model payload is the exact evaluator
//! executable artifact. The evaluator protocol names that same artifact as its runner implementation,
//! and the direct launcher independently hashes the executable bytes before spawning them.
//!
//! Therefore the composed v1 proposition is:
//! `frozen model payload ID == protocol runner artifact ID == executable bytes actually launched`.
//!
//! A future generic evaluator that transports separate weights/model bytes should introduce a new
//! version rather than weakening this identity relationship by interpretation.

use crate::proposal_evaluator_execution::ForgeProposalEvaluatorLaunchPolicy;
use crate::proposal_evaluator_launcher::{
    run_direct_evaluator, ForgeProposalDirectLaunchReceipt, ForgeProposalEvaluatorLauncherError,
};
use crate::proposal_evaluator_protocol::{
    ForgeProposalEvaluationRequest, ForgeProposalEvaluationResponse,
    ForgeProposalEvaluatorProtocolError, ForgeProposalEvaluatorProtocolSpec,
};
use crate::proposal_model::ForgeProposalFrozenModel;
use serde::Serialize;
use std::path::Path;
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalExecutableModelError {
    #[error(transparent)]
    Protocol(#[from] ForgeProposalEvaluatorProtocolError),
    #[error(transparent)]
    Launcher(#[from] ForgeProposalEvaluatorLauncherError),
    #[error("v1 executable-model binding requires the frozen model payload to equal the evaluator runner artifact identity")]
    ModelPayloadMismatch,
    #[error("executable-model binding does not match the supplied frozen model/protocol")]
    BindingScopeMismatch,
    #[error("executable-model binding identity does not match canonical fields")]
    BindingIdentityMismatch,
    #[error("model-bound launch receipt does not match the supplied binding/model/request/response")]
    LaunchScopeMismatch,
    #[error("model-bound launch receipt identity does not match canonical fields")]
    LaunchIdentityMismatch,
}

/// Immutable proof that one frozen model is represented by one exact evaluator executable artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalExecutableModelBinding {
    id: ContentId,
    model_id: ContentId,
    model_payload_id: ContentId,
    protocol_id: ContentId,
    runner_artifact_id: ContentId,
}

impl ForgeProposalExecutableModelBinding {
    pub fn bind(
        model: &ForgeProposalFrozenModel,
        protocol: &ForgeProposalEvaluatorProtocolSpec,
    ) -> Result<Self, ForgeProposalExecutableModelError> {
        protocol.validate()?;
        if model.model_payload_id() != protocol.runner_implementation_id() {
            return Err(ForgeProposalExecutableModelError::ModelPayloadMismatch);
        }
        let id = derive_binding_id(
            model.id(),
            model.model_payload_id(),
            protocol.id(),
            protocol.runner_implementation_id(),
        );
        Ok(Self {
            id,
            model_id: model.id().clone(),
            model_payload_id: model.model_payload_id().clone(),
            protocol_id: protocol.id().clone(),
            runner_artifact_id: protocol.runner_implementation_id().clone(),
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn model_id(&self) -> &ContentId { &self.model_id }
    pub fn model_payload_id(&self) -> &ContentId { &self.model_payload_id }
    pub fn protocol_id(&self) -> &ContentId { &self.protocol_id }
    pub fn runner_artifact_id(&self) -> &ContentId { &self.runner_artifact_id }

    pub fn validate_for(
        &self,
        model: &ForgeProposalFrozenModel,
        protocol: &ForgeProposalEvaluatorProtocolSpec,
    ) -> Result<(), ForgeProposalExecutableModelError> {
        protocol.validate()?;
        if self.model_id != *model.id()
            || self.model_payload_id != *model.model_payload_id()
            || self.protocol_id != *protocol.id()
            || self.runner_artifact_id != *protocol.runner_implementation_id()
            || self.model_payload_id != self.runner_artifact_id
        {
            return Err(ForgeProposalExecutableModelError::BindingScopeMismatch);
        }
        let expected = derive_binding_id(
            &self.model_id,
            &self.model_payload_id,
            &self.protocol_id,
            &self.runner_artifact_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalExecutableModelError::BindingIdentityMismatch)
        }
    }
}

fn derive_binding_id(
    model_id: &ContentId,
    model_payload_id: &ContentId,
    protocol_id: &ContentId,
    runner_artifact_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-executable-model-binding.v1",
        [
            model_id.as_str().as_bytes(),
            model_payload_id.as_str().as_bytes(),
            protocol_id.as_str().as_bytes(),
            runner_artifact_id.as_str().as_bytes(),
        ],
    )
}

/// Launch receipt proving that the executable bytes observed by the direct launcher are the exact
/// payload of the frozen model named by the request.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalExecutableModelLaunchReceipt {
    id: ContentId,
    binding_id: ContentId,
    model_id: ContentId,
    model_payload_id: ContentId,
    direct_launch_receipt: ForgeProposalDirectLaunchReceipt,
}

impl ForgeProposalExecutableModelLaunchReceipt {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn binding_id(&self) -> &ContentId { &self.binding_id }
    pub fn model_id(&self) -> &ContentId { &self.model_id }
    pub fn model_payload_id(&self) -> &ContentId { &self.model_payload_id }
    pub fn direct_launch_receipt(&self) -> &ForgeProposalDirectLaunchReceipt {
        &self.direct_launch_receipt
    }

    pub fn validate_for(
        &self,
        binding: &ForgeProposalExecutableModelBinding,
        model: &ForgeProposalFrozenModel,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
    ) -> Result<(), ForgeProposalExecutableModelError> {
        binding.validate_for(model, request.protocol())?;
        response.validate_for(request)?;
        if request.model_id() != model.id()
            || self.binding_id != *binding.id()
            || self.model_id != *model.id()
            || self.model_payload_id != *model.model_payload_id()
            || self.direct_launch_receipt.runner_artifact_id() != model.model_payload_id()
            || self.direct_launch_receipt.execution_record().request_id() != request.id()
            || self.direct_launch_receipt.execution_record().response_id() != response.id()
            || self.direct_launch_receipt.execution_record().execution_context_id()
                != response.execution_context_id()
        {
            return Err(ForgeProposalExecutableModelError::LaunchScopeMismatch);
        }
        let expected = derive_launch_receipt_id(
            &self.binding_id,
            &self.model_id,
            &self.model_payload_id,
            self.direct_launch_receipt.id(),
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalExecutableModelError::LaunchIdentityMismatch)
        }
    }
}

fn derive_launch_receipt_id(
    binding_id: &ContentId,
    model_id: &ContentId,
    model_payload_id: &ContentId,
    direct_launch_receipt_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-executable-model-launch-receipt.v1",
        [
            binding_id.as_str().as_bytes(),
            model_id.as_str().as_bytes(),
            model_payload_id.as_str().as_bytes(),
            direct_launch_receipt_id.as_str().as_bytes(),
        ],
    )
}

/// Execute a v1 model-specific evaluator and prove that the exact bytes launched are the exact
/// frozen model payload committed by the request.
pub fn run_executable_model_evaluator(
    binding: &ForgeProposalExecutableModelBinding,
    policy: &ForgeProposalEvaluatorLaunchPolicy,
    request: &ForgeProposalEvaluationRequest,
    model: &ForgeProposalFrozenModel,
    executable: impl AsRef<Path>,
    args: &[String],
) -> Result<
    (
        ForgeProposalEvaluationResponse,
        ForgeProposalExecutableModelLaunchReceipt,
    ),
    ForgeProposalExecutableModelError,
> {
    binding.validate_for(model, request.protocol())?;
    if request.model_id() != model.id() {
        return Err(ForgeProposalExecutableModelError::BindingScopeMismatch);
    }
    let (response, direct_launch_receipt) =
        run_direct_evaluator(policy, request, executable, args)?;
    if direct_launch_receipt.runner_artifact_id() != model.model_payload_id() {
        return Err(ForgeProposalExecutableModelError::LaunchScopeMismatch);
    }
    let id = derive_launch_receipt_id(
        binding.id(),
        model.id(),
        model.model_payload_id(),
        direct_launch_receipt.id(),
    );
    let receipt = ForgeProposalExecutableModelLaunchReceipt {
        id,
        binding_id: binding.id().clone(),
        model_id: model.id().clone(),
        model_payload_id: model.model_payload_id().clone(),
        direct_launch_receipt,
    };
    receipt.validate_for(binding, model, request, &response)?;
    Ok((response, receipt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_identity_commits_model_and_runner_artifact() {
        let model_id = ContentId::derive("model", [b"m".as_slice()]);
        let payload = ContentId::derive("runner", [b"payload".as_slice()]);
        let protocol_id = ContentId::derive("protocol", [b"p".as_slice()]);
        let a = derive_binding_id(&model_id, &payload, &protocol_id, &payload);
        let other = ContentId::derive("runner", [b"other".as_slice()]);
        let b = derive_binding_id(&model_id, &payload, &protocol_id, &other);
        assert_ne!(a, b);
    }
}
