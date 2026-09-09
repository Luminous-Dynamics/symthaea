// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IF-4 executable inference envelope and final v2 permit binding.
//!
//! IF-3 binds semantic request/provider state. IF-4 composes those digests with
//! execution-only facts that also affect what is actually sent on the wire:
//! sampling controls and the exact endpoint/protocol configuration.

#[cfg(not(test))]
use super::inference_binding::{
    CredentialStateBinding, InferenceBindingError, QuotaStateBinding, digest_candidate,
    digest_credential_state, digest_policy, digest_quota_state, digest_request, digest_route,
};
#[cfg(not(test))]
use super::inference_contract::{
    AdmittedInferenceRoute, ExecutionLocation, InferenceAdmissionError, InferenceCandidate,
    InferencePolicy, InferenceRequest, ModelIdentity,
};
#[cfg(not(test))]
use super::inference_permit::{BindingDigest, InferenceExecutionBinding, InferencePermitError};

#[cfg(test)]
use crate::inference_binding::{
    CredentialStateBinding, InferenceBindingError, QuotaStateBinding, digest_candidate,
    digest_credential_state, digest_policy, digest_quota_state, digest_request, digest_route,
};
#[cfg(test)]
use crate::inference_contract::{
    AdmittedInferenceRoute, ExecutionLocation, InferenceAdmissionError, InferenceCandidate,
    InferencePolicy, InferenceRequest, ModelIdentity,
};
#[cfg(test)]
use crate::inference_permit::{BindingDigest, InferenceExecutionBinding, InferencePermitError};

use std::fmt;

const V2_ROOT_DOMAIN: &[u8] = b"symthaea.inference.execution-binding.v2";
const REQUEST_DOMAIN: &[u8] = b"request-plus-generation-controls";
const PROVIDER_DOMAIN: &[u8] = b"candidate-plus-endpoint-state";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointProtocol {
    OpenAiCompatibleV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointCredentialMode {
    None,
    Bearer,
}

/// Immutable, non-secret description of the exact execution endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointStateBinding {
    protocol: EndpointProtocol,
    provider_id: String,
    base_url: String,
    wire_model: String,
    timeout_millis: u64,
    credential_mode: EndpointCredentialMode,
    config_epoch: u64,
}

impl EndpointStateBinding {
    pub fn openai_compatible(
        provider_id: impl Into<String>,
        base_url: impl Into<String>,
        wire_model: impl Into<String>,
        timeout_millis: u64,
        credential_mode: EndpointCredentialMode,
        config_epoch: u64,
    ) -> Result<Self, InferenceExecutionEnvelopeError> {
        let provider_id = provider_id.into();
        let base_url = base_url.into();
        let wire_model = wire_model.into();
        if provider_id.trim().is_empty() {
            return Err(InferenceExecutionEnvelopeError::EmptyEndpointProviderId);
        }
        if base_url.trim().is_empty() {
            return Err(InferenceExecutionEnvelopeError::EmptyEndpointUrl);
        }
        if wire_model.trim().is_empty() {
            return Err(InferenceExecutionEnvelopeError::EmptyWireModel);
        }
        if timeout_millis == 0 {
            return Err(InferenceExecutionEnvelopeError::ZeroEndpointTimeout);
        }
        Ok(Self {
            protocol: EndpointProtocol::OpenAiCompatibleV1,
            provider_id,
            base_url,
            wire_model,
            timeout_millis,
            credential_mode,
            config_epoch,
        })
    }

    pub const fn protocol(&self) -> EndpointProtocol {
        self.protocol
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn wire_model(&self) -> &str {
        &self.wire_model
    }

    pub const fn timeout_millis(&self) -> u64 {
        self.timeout_millis
    }

    pub const fn credential_mode(&self) -> EndpointCredentialMode {
        self.credential_mode
    }

    pub const fn config_epoch(&self) -> u64 {
        self.config_epoch
    }
}

/// Wire-affecting generation controls represented without floating-point ambiguity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceGenerationControls {
    temperature_milli: u16,
    streaming: bool,
}

impl InferenceGenerationControls {
    /// Temperature is thousandths: 700 means 0.700. v1 supports 0.000..=2.000.
    pub fn new(
        temperature_milli: u16,
        streaming: bool,
    ) -> Result<Self, InferenceExecutionEnvelopeError> {
        if temperature_milli > 2_000 {
            return Err(InferenceExecutionEnvelopeError::TemperatureOutOfRange);
        }
        Ok(Self {
            temperature_milli,
            streaming,
        })
    }

    pub const fn temperature_milli(&self) -> u16 {
        self.temperature_milli
    }

    pub const fn streaming(&self) -> bool {
        self.streaming
    }

    pub fn temperature_f32(&self) -> f32 {
        f32::from(self.temperature_milli) / 1_000.0
    }
}

impl Default for InferenceGenerationControls {
    fn default() -> Self {
        Self {
            temperature_milli: 700,
            streaming: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundExecutableAdmission {
    route: AdmittedInferenceRoute,
    binding: InferenceExecutionBinding,
}

impl BoundExecutableAdmission {
    pub fn route(&self) -> &AdmittedInferenceRoute {
        &self.route
    }

    pub const fn binding(&self) -> &InferenceExecutionBinding {
        &self.binding
    }
}

/// Admit and bind every fact required for an OpenAI-compatible execution.
pub fn admit_and_bind_execution(
    policy: &InferencePolicy,
    request: &InferenceRequest,
    candidate: &InferenceCandidate,
    credential: &CredentialStateBinding,
    quota: &QuotaStateBinding,
    endpoint: &EndpointStateBinding,
    controls: InferenceGenerationControls,
) -> Result<BoundExecutableAdmission, InferenceExecutionEnvelopeError> {
    validate_endpoint_candidate(candidate, endpoint)?;
    if request.requirements.require_streaming && !controls.streaming {
        return Err(InferenceExecutionEnvelopeError::RequiredStreamingDisabled);
    }
    if controls.streaming && !candidate.supports_streaming {
        return Err(InferenceExecutionEnvelopeError::CandidateCannotStream);
    }

    let route = policy
        .admit(request, candidate)
        .map_err(InferenceExecutionEnvelopeError::Admission)?;

    let semantic_request = digest_request(request).map_err(InferenceExecutionEnvelopeError::Binding)?;
    let candidate_state = digest_candidate(candidate).map_err(InferenceExecutionEnvelopeError::Binding)?;

    let binding = InferenceExecutionBinding {
        request_digest: digest_execution_request(semantic_request, controls)?,
        route_digest: digest_route(&route).map_err(InferenceExecutionEnvelopeError::Binding)?,
        policy_digest: digest_policy(policy).map_err(InferenceExecutionEnvelopeError::Binding)?,
        provider_state_digest: digest_execution_provider_state(candidate_state, endpoint)?,
        credential_state_digest: digest_credential_state(credential)
            .map_err(InferenceExecutionEnvelopeError::Binding)?,
        quota_state_digest: digest_quota_state(quota)
            .map_err(InferenceExecutionEnvelopeError::Binding)?,
    };

    Ok(BoundExecutableAdmission { route, binding })
}

pub fn digest_execution_request(
    semantic_request_digest: BindingDigest,
    controls: InferenceGenerationControls,
) -> Result<BindingDigest, InferenceExecutionEnvelopeError> {
    let mut h = V2Hasher::new(REQUEST_DOMAIN);
    h.digest(semantic_request_digest);
    h.u16(controls.temperature_milli);
    h.bool(controls.streaming);
    h.finish()
}

pub fn digest_execution_provider_state(
    candidate_digest: BindingDigest,
    endpoint: &EndpointStateBinding,
) -> Result<BindingDigest, InferenceExecutionEnvelopeError> {
    let mut h = V2Hasher::new(PROVIDER_DOMAIN);
    h.digest(candidate_digest);
    h.u8(match endpoint.protocol {
        EndpointProtocol::OpenAiCompatibleV1 => 1,
    });
    h.string(&endpoint.provider_id);
    h.string(&endpoint.base_url);
    h.string(&endpoint.wire_model);
    h.u64(endpoint.timeout_millis);
    h.u8(match endpoint.credential_mode {
        EndpointCredentialMode::None => 1,
        EndpointCredentialMode::Bearer => 2,
    });
    h.u64(endpoint.config_epoch);
    h.finish()
}

fn validate_endpoint_candidate(
    candidate: &InferenceCandidate,
    endpoint: &EndpointStateBinding,
) -> Result<(), InferenceExecutionEnvelopeError> {
    if candidate.provider_id != endpoint.provider_id {
        return Err(InferenceExecutionEnvelopeError::EndpointProviderMismatch);
    }

    // IF-4 only qualifies provider-attested OpenAI-compatible wire identity.
    // Content-verified local and opaque endpoints need their own adapters.
    match &candidate.model {
        ModelIdentity::ProviderAttested {
            provider,
            declared_model,
        } if provider == &endpoint.provider_id && declared_model == &endpoint.wire_model => {}
        ModelIdentity::ProviderAttested { .. } => {
            return Err(InferenceExecutionEnvelopeError::EndpointModelMismatch);
        }
        ModelIdentity::ContentVerified { .. } | ModelIdentity::OpaqueEndpoint { .. } => {
            return Err(InferenceExecutionEnvelopeError::ModelIdentityNotWireVerifiable);
        }
    }

    // This executor is a remote provider adapter. Local/native execution remains
    // on existing sovereign paths until a separately qualified local adapter exists.
    if candidate.location != ExecutionLocation::RemoteProvider {
        return Err(InferenceExecutionEnvelopeError::UnsupportedExecutionLocation);
    }
    Ok(())
}

struct V2Hasher {
    hasher: blake3::Hasher,
}

impl V2Hasher {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(V2_ROOT_DOMAIN.len() as u64).to_le_bytes());
        hasher.update(V2_ROOT_DOMAIN);
        hasher.update(&(domain.len() as u64).to_le_bytes());
        hasher.update(domain);
        Self { hasher }
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u8(&mut self, value: u8) {
        self.hasher.update(&[value]);
    }

    fn u16(&mut self, value: u16) {
        self.hasher.update(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.hasher.update(&value.to_le_bytes());
    }

    fn string(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.hasher.update(value.as_bytes());
    }

    fn digest(&mut self, value: BindingDigest) {
        self.hasher.update(value.as_bytes());
    }

    fn finish(self) -> Result<BindingDigest, InferenceExecutionEnvelopeError> {
        BindingDigest::new(*self.hasher.finalize().as_bytes())
            .map_err(InferenceExecutionEnvelopeError::Permit)
    }
}

#[derive(Debug)]
pub enum InferenceExecutionEnvelopeError {
    Admission(InferenceAdmissionError),
    Binding(InferenceBindingError),
    Permit(InferencePermitError),
    EmptyEndpointProviderId,
    EmptyEndpointUrl,
    EmptyWireModel,
    ZeroEndpointTimeout,
    TemperatureOutOfRange,
    RequiredStreamingDisabled,
    CandidateCannotStream,
    EndpointProviderMismatch,
    EndpointModelMismatch,
    ModelIdentityNotWireVerifiable,
    UnsupportedExecutionLocation,
}

impl fmt::Display for InferenceExecutionEnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => write!(f, "inference admission failed: {error}"),
            Self::Binding(error) => write!(f, "inference semantic binding failed: {error}"),
            Self::Permit(error) => write!(f, "inference execution digest failed: {error}"),
            Self::EmptyEndpointProviderId => write!(f, "endpoint provider id must not be empty"),
            Self::EmptyEndpointUrl => write!(f, "endpoint URL must not be empty"),
            Self::EmptyWireModel => write!(f, "endpoint wire model must not be empty"),
            Self::ZeroEndpointTimeout => write!(f, "endpoint timeout must be non-zero"),
            Self::TemperatureOutOfRange => write!(f, "temperature must be between 0.000 and 2.000"),
            Self::RequiredStreamingDisabled => write!(f, "request requires streaming but execution controls disable it"),
            Self::CandidateCannotStream => write!(f, "execution requests streaming from a candidate that cannot stream"),
            Self::EndpointProviderMismatch => write!(f, "endpoint provider does not match admitted candidate"),
            Self::EndpointModelMismatch => write!(f, "endpoint wire model does not match provider-attested model"),
            Self::ModelIdentityNotWireVerifiable => write!(f, "IF-4 OpenAI executor requires provider-attested model identity"),
            Self::UnsupportedExecutionLocation => write!(f, "IF-4 OpenAI executor requires RemoteProvider execution location"),
        }
    }
}

impl std::error::Error for InferenceExecutionEnvelopeError {}
