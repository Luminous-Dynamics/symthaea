// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Permit-gated OpenAI-compatible executor.
//!
//! The raw executor understands the IF-4/v2 semantic execution envelope plus a
//! crate-internal exact-binding dispatch seam. Higher authority layers (current
//! provider profile, resource scope, future manifests) re-derive their stronger
//! bindings above this file and pass only the final exact binding downward.

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
use super::inference_permit::{InferenceExecutionBinding, PreparedInferenceExecution};
#[cfg(not(test))]
use super::inference_receipt::{
    InferenceFailureClass, InferenceRateLimitEvidence, InferenceReceipt, InferenceReceiptError,
    InferenceTokenUsageEvidence, InferenceWireEvidence,
};
#[cfg(not(test))]
use super::openai_compatible_transport::{
    OpenAiCompatibleConfig, OpenAiCompatibleTransport, ObservedTransportFailure,
    ObservedTransportGeneration, TransportConfigError, TransportCredential, TransportError,
    TransportGenerationRequest, TransportWireObservation,
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
use crate::inference_permit::{InferenceExecutionBinding, PreparedInferenceExecution};
#[cfg(test)]
use crate::inference_receipt::{
    InferenceFailureClass, InferenceRateLimitEvidence, InferenceReceipt, InferenceReceiptError,
    InferenceTokenUsageEvidence, InferenceWireEvidence,
};
#[cfg(test)]
use crate::openai_compatible_transport::{
    OpenAiCompatibleConfig, OpenAiCompatibleTransport, ObservedTransportFailure,
    ObservedTransportGeneration, TransportConfigError, TransportCredential, TransportError,
    TransportGenerationRequest, TransportWireObservation,
};

use std::fmt;
use std::time::Duration;

pub trait InferenceTickSource: Send + Sync {
    fn now_tick(&self) -> u64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceExecutionFailure {
    BindingRejected,
    ProviderProfileRejected,
    ResourceScopeRejected,
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

pub struct InferenceExecutionOutcome {
    pub response_text: Option<String>,
    pub receipt: InferenceReceipt,
    pub failure: Option<InferenceExecutionFailure>,
    pub wire_observation: Option<TransportWireObservation>,
}

impl fmt::Debug for InferenceExecutionOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceExecutionOutcome")
            .field("response_text_present", &self.response_text.is_some())
            .field("receipt", &self.receipt)
            .field("failure", &self.failure)
            .field("wire_observation_present", &self.wire_observation.is_some())
            .finish()
    }
}

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
        let config = OpenAiCompatibleConfig::new(provider_id, base_url, wire_model, credential)?
            .with_timeout(timeout)?;
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
            &prepared, policy, request, candidate, credential, quota, controls,
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
        self.dispatch_non_streaming(prepared, wire_request).await
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
            &prepared, policy, request, candidate, credential, quota, controls,
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
        self.dispatch_streaming(prepared, wire_request, on_token).await
    }

    /// Crate-internal seam for a stricter child layer that has already re-derived
    /// its complete execution binding. The exact expected binding is compared here
    /// immediately before wire request derivation and private transport dispatch.
    pub(crate) async fn execute_preverified_binding(
        &self,
        prepared: PreparedInferenceExecution,
        expected_binding: InferenceExecutionBinding,
        request: &InferenceRequest,
        controls: InferenceGenerationControls,
    ) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
        if controls.streaming() {
            return self.failure_outcome(
                prepared,
                InferenceExecutionFailure::WrongStreamingEntryPoint,
                InferenceFailureClass::VerificationRejected,
            );
        }
        let wire_request = match self.verify_exact_binding_and_derive(
            &prepared,
            expected_binding,
            request,
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
        self.dispatch_non_streaming(prepared, wire_request).await
    }

    pub(crate) async fn execute_streaming_preverified_binding(
        &self,
        prepared: PreparedInferenceExecution,
        expected_binding: InferenceExecutionBinding,
        request: &InferenceRequest,
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
        let wire_request = match self.verify_exact_binding_and_derive(
            &prepared,
            expected_binding,
            request,
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
        self.dispatch_streaming(prepared, wire_request, on_token).await
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
        self.validate_supported_wire_contract(request)?;
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
        self.derive_wire_request(request, controls)
    }

    fn verify_exact_binding_and_derive(
        &self,
        prepared: &PreparedInferenceExecution,
        expected_binding: InferenceExecutionBinding,
        request: &InferenceRequest,
        controls: InferenceGenerationControls,
    ) -> Result<TransportGenerationRequest, InferenceExecutionFailure> {
        self.validate_supported_wire_contract(request)?;
        if &expected_binding != prepared.binding() {
            return Err(InferenceExecutionFailure::BoundStateChanged);
        }
        self.derive_wire_request(request, controls)
    }

    fn validate_supported_wire_contract(
        &self,
        request: &InferenceRequest,
    ) -> Result<(), InferenceExecutionFailure> {
        if request.requirements.require_tools {
            return Err(InferenceExecutionFailure::UnsupportedTools);
        }
        if request.requirements.require_structured_output {
            return Err(InferenceExecutionFailure::UnsupportedStructuredOutput);
        }
        Ok(())
    }

    fn derive_wire_request(
        &self,
        request: &InferenceRequest,
        controls: InferenceGenerationControls,
    ) -> Result<TransportGenerationRequest, InferenceExecutionFailure> {
        let max_tokens = usize::try_from(request.requirements.max_output_tokens)
            .map_err(|_| InferenceExecutionFailure::OutputTokenLimitTooLarge)?;
        Ok(TransportGenerationRequest {
            prompt: request.prompt.clone(),
            system_prompt: request.system_prompt.clone(),
            temperature: controls.temperature_f32(),
            max_tokens,
        })
    }

    async fn dispatch_non_streaming(
        &self,
        prepared: PreparedInferenceExecution,
        wire_request: TransportGenerationRequest,
    ) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
        match self.transport.generate_observed(&wire_request).await {
            Ok(observed) => self.success_outcome(prepared, observed),
            Err(failure) => self.transport_failure_outcome(prepared, failure),
        }
    }

    async fn dispatch_streaming(
        &self,
        prepared: PreparedInferenceExecution,
        wire_request: TransportGenerationRequest,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
        match self
            .transport
            .generate_streaming_observed(&wire_request, on_token)
            .await
        {
            Ok(observed) => self.success_outcome(prepared, observed),
            Err(failure) => self.transport_failure_outcome(prepared, failure),
        }
    }

    fn success_outcome(
        &self,
        prepared: PreparedInferenceExecution,
        observed: ObservedTransportGeneration,
    ) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
        let completed_at_tick = self.clock.now_tick();
        let wire_evidence = receipt_wire_evidence(&observed.observation)?;
        let receipt = InferenceReceipt::success_observed(
            prepared,
            &observed.response.text,
            completed_at_tick,
            wire_evidence,
        )?;
        Ok(InferenceExecutionOutcome {
            response_text: Some(observed.response.text),
            receipt,
            failure: None,
            wire_observation: Some(observed.observation),
        })
    }

    fn transport_failure_outcome(
        &self,
        prepared: PreparedInferenceExecution,
        observed_failure: ObservedTransportFailure,
    ) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
        let (failure, receipt_class) = classify_transport_error(&observed_failure.error);
        let completed_at_tick = self.clock.now_tick();
        let wire_evidence = receipt_wire_evidence(&observed_failure.observation)?;
        let receipt = InferenceReceipt::failure_observed(
            prepared,
            receipt_class,
            completed_at_tick,
            wire_evidence,
        )?;
        Ok(InferenceExecutionOutcome {
            response_text: None,
            receipt,
            failure: Some(failure),
            wire_observation: Some(observed_failure.observation),
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
            wire_observation: None,
        })
    }
}

fn receipt_wire_evidence(
    observation: &TransportWireObservation,
) -> Result<InferenceWireEvidence, InferenceReceiptError> {
    let usage = observation.usage.map(|usage| InferenceTokenUsageEvidence {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    });
    let limits = observation.rate_limits;
    let rate_limits = InferenceRateLimitEvidence {
        retry_after_millis: limits.retry_after_millis,
        request_limit: limits.request_limit,
        request_remaining: limits.request_remaining,
        request_reset_after_millis: limits.request_reset_after_millis,
        token_limit: limits.token_limit,
        token_remaining: limits.token_remaining,
        token_reset_after_millis: limits.token_reset_after_millis,
    };
    InferenceWireEvidence::new(
        observation.response_id.as_deref(),
        observation.request_id_header.as_deref(),
        observation.provider_model.as_deref(),
        observation.system_fingerprint.as_deref(),
        observation.finish_reason.as_deref(),
        usage,
        rate_limits,
        observation.latency_millis,
        observation.metadata_conflict,
        observation.metadata_rejected,
    )
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
