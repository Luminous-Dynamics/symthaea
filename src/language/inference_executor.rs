// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IF-4 permit-gated OpenAI-compatible executor.
//!
//! The raw transport is private. Callers must transfer a `PreparedInferenceExecution`
//! and the executor re-derives the complete v2 binding immediately before I/O.

#[cfg(not(test))]
use super::inference_binding::{CredentialStateBinding, QuotaStateBinding};
#[cfg(not(test))]
use super::inference_contract::{InferenceCandidate, InferencePolicy, InferenceRequest};
#[cfg(not(test))]
use super::inference_execution_envelope::{
    EndpointCredentialMode, EndpointStateBinding, InferenceExecutionEnvelopeError,
    InferenceGenerationControls, admit_and_bind_execution,
};
#[cfg(not(test))]
use super::inference_permit::PreparedInferenceExecution;
#[cfg(not(test))]
use super::inference_receipt::{
    InferenceFailureClass, InferenceReceipt, InferenceReceiptError,
};
#[cfg(not(test))]
use super::openai_compatible_transport::{
    OpenAiCompatibleConfig, OpenAiCompatibleTransport, TransportConfigError, TransportCredential,
    TransportError, TransportGenerationRequest,
};

#[cfg(test)]
use crate::inference_binding::{CredentialStateBinding, QuotaStateBinding};
#[cfg(test)]
use crate::inference_contract::{InferenceCandidate, InferencePolicy, InferenceRequest};
#[cfg(test)]
use crate::inference_execution_envelope::{
    EndpointCredentialMode, EndpointStateBinding, InferenceExecutionEnvelopeError,
    InferenceGenerationControls, admit_and_bind_execution,
};
#[cfg(test)]
use crate::inference_permit::PreparedInferenceExecution;
#[cfg(test)]
use crate::inference_receipt::{InferenceFailureClass, InferenceReceipt, InferenceReceiptError};
#[cfg(test)]
use crate::openai_compatible_transport::{
    OpenAiCompatibleConfig, OpenAiCompatibleTransport, TransportConfigError, TransportCredential,
    TransportError, TransportGenerationRequest,
};

use std::fmt;
use std::time::Duration;

/// Qualified monotonic/trusted tick source supplied by the embedding runtime.
pub trait InferenceTickSource: Send + Sync {
    fn now_tick(&self) -> u64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceExecutionFailure {
    BindingRejected,
    BoundStateChanged,
    UnsupportedTools,
    UnsupportedStructuredOutput,
    WrongStreamingEntryPoint,
    OutputTokenLimitTooLarge,
    Transport,
    ProviderHttp,
    ResponseDecode,
    EmptyResponse,
}

/// Terminal execution result. Operational failures still carry a receipt.
#[derive(Debug)]
pub struct InferenceExecutionOutcome {
    pub response_text: Option<String>,
    pub receipt: InferenceReceipt,
    pub failure: Option<InferenceExecutionFailure>,
}

/// OpenAI-compatible executor whose underlying transport cannot be accessed directly.
pub struct OpenAiInferenceExecutor<C> {
    transport: OpenAiCompatibleTransport,
    endpoint: EndpointStateBinding,
    clock: C,
}

impl<C: InferenceTickSource> fmt::Debug for OpenAiInferenceExecutor<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiInferenceExecutor")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

impl<C: InferenceTickSource> OpenAiInferenceExecutor<C> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider_id: impl Into<String>,
        base_url: &str,
        wire_model: impl Into<String>,
        credential: TransportCredential,
        timeout: Duration,
        config_epoch: u64,
        clock: C,
    ) -> Result<Self, InferenceExecutorConfigError> {
        let provider_id = provider_id.into();
        let wire_model = wire_model.into();
        let credential_mode = match &credential {
            TransportCredential::None => EndpointCredentialMode::None,
            TransportCredential::Bearer(_) => EndpointCredentialMode::Bearer,
        };
        let timeout_millis = u64::try_from(timeout.as_millis())
            .map_err(|_| InferenceExecutorConfigError::TimeoutTooLarge)?;

        let config = OpenAiCompatibleConfig::new(
            provider_id,
            base_url,
            wire_model,
            credential,
        )?
        .with_timeout(timeout)?;

        // Build endpoint evidence from the already-normalized transport config.
        let endpoint = EndpointStateBinding::openai_compatible(
            config.provider_id(),
            config.base_url().as_str(),
            config.model(),
            timeout_millis,
            credential_mode,
            config_epoch,
        )?;
        let transport = OpenAiCompatibleTransport::new(config)?;
        Ok(Self {
            transport,
            endpoint,
            clock,
        })
    }

    /// Non-authoritative metadata needed to create the earlier admission binding.
    pub fn endpoint_binding(&self) -> &EndpointStateBinding {
        &self.endpoint
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn execute(
        &self,
        prepared: PreparedInferenceExecution,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        candidate: &InferenceCandidate,
        credential: &CredentialStateBinding,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
    ) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
        if controls.streaming() {
            return self.failure_outcome(
                prepared,
                InferenceExecutionFailure::WrongStreamingEntryPoint,
                InferenceFailureClass::VerificationRejected,
            );
        }

        let wire_request = match self.verify_and_derive(
            &prepared,
            policy,
            request,
            candidate,
            credential,
            quota,
            controls,
        ) {
            Ok(request) => request,
            Err(failure) => {
                return self.failure_outcome(
                    prepared,
                    failure,
                    InferenceFailureClass::VerificationRejected,
                );
            }
        };

        match self.transport.generate(&wire_request).await {
            Ok(response) => {
                let completed_at_tick = self.clock.now_tick();
                let receipt = InferenceReceipt::success(
                    prepared,
                    &response.text,
                    completed_at_tick,
                )?;
                Ok(InferenceExecutionOutcome {
                    response_text: Some(response.text),
                    receipt,
                    failure: None,
                })
            }
            Err(error) => {
                let (failure, receipt_class) = classify_transport_error(&error);
                self.failure_outcome(prepared, failure, receipt_class)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn execute_streaming(
        &self,
        prepared: PreparedInferenceExecution,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        candidate: &InferenceCandidate,
        credential: &CredentialStateBinding,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
        if !controls.streaming() {
            return self.failure_outcome(
                prepared,
                InferenceExecutionFailure::WrongStreamingEntryPoint,
                InferenceFailureClass::VerificationRejected,
            );
        }

        let wire_request = match self.verify_and_derive(
            &prepared,
            policy,
            request,
            candidate,
            credential,
            quota,
            controls,
        ) {
            Ok(request) => request,
            Err(failure) => {
                return self.failure_outcome(
                    prepared,
                    failure,
                    InferenceFailureClass::VerificationRejected,
                );
            }
        };

        match self
            .transport
            .generate_streaming(&wire_request, on_token)
            .await
        {
            Ok(response) => {
                let completed_at_tick = self.clock.now_tick();
                let receipt = InferenceReceipt::success(
                    prepared,
                    &response.text,
                    completed_at_tick,
                )?;
                Ok(InferenceExecutionOutcome {
                    response_text: Some(response.text),
                    receipt,
                    failure: None,
                })
            }
            Err(error) => {
                let (failure, receipt_class) = classify_transport_error(&error);
                self.failure_outcome(prepared, failure, receipt_class)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_and_derive(
        &self,
        prepared: &PreparedInferenceExecution,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        candidate: &InferenceCandidate,
        credential: &CredentialStateBinding,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
    ) -> Result<TransportGenerationRequest, InferenceExecutionFailure> {
        if request.requirements.require_tools {
            return Err(InferenceExecutionFailure::UnsupportedTools);
        }
        if request.requirements.require_structured_output {
            return Err(InferenceExecutionFailure::UnsupportedStructuredOutput);
        }

        let rebound = admit_and_bind_execution(
            policy,
            request,
            candidate,
            credential,
            quota,
            &self.endpoint,
            controls,
        )
        .map_err(|_| InferenceExecutionFailure::BindingRejected)?;
        if rebound.binding() != prepared.binding() {
            return Err(InferenceExecutionFailure::BoundStateChanged);
        }

        let max_tokens = usize::try_from(request.requirements.max_output_tokens)
            .map_err(|_| InferenceExecutionFailure::OutputTokenLimitTooLarge)?;
        Ok(TransportGenerationRequest {
            prompt: request.prompt.clone(),
            system_prompt: request.system_prompt.clone(),
            temperature: controls.temperature_f32(),
            max_tokens,
        })
    }

    fn failure_outcome(
        &self,
        prepared: PreparedInferenceExecution,
        failure: InferenceExecutionFailure,
        receipt_class: InferenceFailureClass,
    ) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
        let completed_at_tick = self.clock.now_tick();
        let receipt = InferenceReceipt::failure(prepared, receipt_class, completed_at_tick)?;
        Ok(InferenceExecutionOutcome {
            response_text: None,
            receipt,
            failure: Some(failure),
        })
    }
}

fn classify_transport_error(
    error: &TransportError,
) -> (InferenceExecutionFailure, InferenceFailureClass) {
    match error {
        TransportError::HttpStatus(_) => (
            InferenceExecutionFailure::ProviderHttp,
            InferenceFailureClass::ProviderHttp,
        ),
        TransportError::ResponseDecode => (
            InferenceExecutionFailure::ResponseDecode,
            InferenceFailureClass::ResponseDecode,
        ),
        TransportError::EmptyResponse => (
            InferenceExecutionFailure::EmptyResponse,
            InferenceFailureClass::EmptyResponse,
        ),
        TransportError::Config(_)
        | TransportError::ClientBuild
        | TransportError::Request
        | TransportError::StreamRead => (
            InferenceExecutionFailure::Transport,
            InferenceFailureClass::Transport,
        ),
    }
}

#[derive(Debug)]
pub enum InferenceExecutorConfigError {
    TransportConfig(TransportConfigError),
    Transport(TransportError),
    Envelope(InferenceExecutionEnvelopeError),
    TimeoutTooLarge,
}

impl From<TransportConfigError> for InferenceExecutorConfigError {
    fn from(value: TransportConfigError) -> Self {
        Self::TransportConfig(value)
    }
}

impl From<TransportError> for InferenceExecutorConfigError {
    fn from(value: TransportError) -> Self {
        Self::Transport(value)
    }
}

impl From<InferenceExecutionEnvelopeError> for InferenceExecutorConfigError {
    fn from(value: InferenceExecutionEnvelopeError) -> Self {
        Self::Envelope(value)
    }
}

impl fmt::Display for InferenceExecutorConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TransportConfig(error) => write!(f, "transport config rejected: {error}"),
            Self::Transport(error) => write!(f, "transport construction failed: {error}"),
            Self::Envelope(error) => write!(f, "endpoint binding rejected: {error}"),
            Self::TimeoutTooLarge => write!(f, "transport timeout is too large to bind"),
        }
    }
}

impl std::error::Error for InferenceExecutorConfigError {}

#[derive(Debug)]
pub enum InferenceExecutorFatalError {
    Receipt(InferenceReceiptError),
}

impl From<InferenceReceiptError> for InferenceExecutorFatalError {
    fn from(value: InferenceReceiptError) -> Self {
        Self::Receipt(value)
    }
}

impl fmt::Display for InferenceExecutorFatalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Receipt(error) => write!(f, "terminal inference receipt failed: {error}"),
        }
    }
}

impl std::error::Error for InferenceExecutorFatalError {}
