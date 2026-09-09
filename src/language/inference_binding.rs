// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical IF-3 inference binding construction.
//!
//! Binding digests are BLAKE3 over explicit, versioned, domain-separated field
//! encodings. They are deliberately NOT hashes of Rust memory layout, Debug text,
//! Serde JSON, or bincode. That keeps evidence stable across irrelevant compiler
//! and serialization changes and avoids creating a generic persistence surface for
//! sensitive prompts.

#[cfg(not(test))]
use super::inference_contract::{
    AdmittedInferenceRoute, ExecutionLocation, InferenceAdmissionError, InferenceCandidate,
    InferencePolicy, InferencePurpose, InferenceRequest, InformationClass, ModelIdentity,
    ProviderRetentionPolicy, ProviderRoutingPolicy, ProviderTrainingPolicy,
};
#[cfg(not(test))]
use super::inference_permit::{BindingDigest, InferenceExecutionBinding, InferencePermitError};

#[cfg(test)]
use crate::inference_contract::{
    AdmittedInferenceRoute, ExecutionLocation, InferenceAdmissionError, InferenceCandidate,
    InferencePolicy, InferencePurpose, InferenceRequest, InformationClass, ModelIdentity,
    ProviderRetentionPolicy, ProviderRoutingPolicy, ProviderTrainingPolicy,
};
#[cfg(test)]
use crate::inference_permit::{BindingDigest, InferenceExecutionBinding, InferencePermitError};

use std::fmt;

const ROOT_DOMAIN: &[u8] = b"symthaea.inference.binding.v1";
const REQUEST_DOMAIN: &[u8] = b"request";
const ROUTE_DOMAIN: &[u8] = b"route";
const POLICY_DOMAIN: &[u8] = b"policy";
const PROVIDER_DOMAIN: &[u8] = b"provider-state";
const CREDENTIAL_DOMAIN: &[u8] = b"credential-state";
const QUOTA_DOMAIN: &[u8] = b"quota-state";
const RESPONSE_DOMAIN: &[u8] = b"response-text";

/// Non-secret identity/epoch of the credential used for one provider account.
///
/// This intentionally contains no bearer token or secret key material.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CredentialStateBinding {
    credential_id: String,
    epoch: u64,
}

impl CredentialStateBinding {
    pub fn new(credential_id: impl Into<String>, epoch: u64) -> Result<Self, InferenceBindingError> {
        let credential_id = credential_id.into();
        if credential_id.trim().is_empty() {
            return Err(InferenceBindingError::EmptyCredentialId);
        }
        Ok(Self {
            credential_id,
            epoch,
        })
    }

    pub fn credential_id(&self) -> &str {
        &self.credential_id
    }

    pub const fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// Exact quota observation used when admission was made.
///
/// The scope id is an opaque non-secret identifier for the provider account,
/// project, organisation, peer, or other quota domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuotaStateBinding {
    scope_id: String,
    epoch: u64,
    remaining_requests: Option<u64>,
    remaining_tokens: Option<u64>,
    reset_at_tick: Option<u64>,
}

impl QuotaStateBinding {
    pub fn new(scope_id: impl Into<String>, epoch: u64) -> Result<Self, InferenceBindingError> {
        let scope_id = scope_id.into();
        if scope_id.trim().is_empty() {
            return Err(InferenceBindingError::EmptyQuotaScopeId);
        }
        Ok(Self {
            scope_id,
            epoch,
            remaining_requests: None,
            remaining_tokens: None,
            reset_at_tick: None,
        })
    }

    pub fn with_remaining_requests(mut self, remaining: Option<u64>) -> Self {
        self.remaining_requests = remaining;
        self
    }

    pub fn with_remaining_tokens(mut self, remaining: Option<u64>) -> Self {
        self.remaining_tokens = remaining;
        self
    }

    pub fn with_reset_at_tick(mut self, reset_at_tick: Option<u64>) -> Self {
        self.reset_at_tick = reset_at_tick;
        self
    }

    pub fn scope_id(&self) -> &str {
        &self.scope_id
    }

    pub const fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// Route admission plus the exact six-way binding required by IF-2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundInferenceAdmission {
    route: AdmittedInferenceRoute,
    binding: InferenceExecutionBinding,
}

impl BoundInferenceAdmission {
    pub fn route(&self) -> &AdmittedInferenceRoute {
        &self.route
    }

    pub const fn binding(&self) -> &InferenceExecutionBinding {
        &self.binding
    }
}

/// Admit a candidate and immediately bind the exact semantic/provider state.
///
/// The route is created inside this function rather than accepted from the caller,
/// preventing a route admitted for one candidate from being rebound to another.
pub fn admit_and_bind(
    policy: &InferencePolicy,
    request: &InferenceRequest,
    candidate: &InferenceCandidate,
    credential: &CredentialStateBinding,
    quota: &QuotaStateBinding,
) -> Result<BoundInferenceAdmission, InferenceBindingError> {
    let route = policy
        .admit(request, candidate)
        .map_err(InferenceBindingError::Admission)?;

    let binding = InferenceExecutionBinding {
        request_digest: digest_request(request)?,
        route_digest: digest_route(&route)?,
        policy_digest: digest_policy(policy)?,
        provider_state_digest: digest_candidate(candidate)?,
        credential_state_digest: digest_credential_state(credential)?,
        quota_state_digest: digest_quota_state(quota)?,
    };

    Ok(BoundInferenceAdmission { route, binding })
}

/// BLAKE3 digest of the exact request, including prompt content, without exposing
/// canonical prompt bytes to callers.
pub fn digest_request(request: &InferenceRequest) -> Result<BindingDigest, InferenceBindingError> {
    let mut h = CanonicalHasher::new(REQUEST_DOMAIN);
    h.string(&request.prompt);
    h.option_string(request.system_prompt.as_deref());
    h.u8(purpose_tag(request.requirements.purpose));
    h.u8(information_class_tag(request.requirements.information_class));
    h.u64(request.requirements.estimated_input_tokens);
    h.u64(request.requirements.max_output_tokens);
    h.bool(request.requirements.require_streaming);
    h.bool(request.requirements.require_tools);
    h.bool(request.requirements.require_structured_output);
    h.finish()
}

pub fn digest_route(route: &AdmittedInferenceRoute) -> Result<BindingDigest, InferenceBindingError> {
    let mut h = CanonicalHasher::new(ROUTE_DOMAIN);
    h.string(route.provider_id());
    h.model_identity(route.model());
    h.u8(execution_location_tag(route.location()));
    h.u8(information_class_tag(route.information_class()));
    h.u8(purpose_tag(route.purpose()));
    h.u64(route.admitted_max_charge_microusd());
    h.finish()
}

pub fn digest_policy(policy: &InferencePolicy) -> Result<BindingDigest, InferenceBindingError> {
    let mut h = CanonicalHasher::new(POLICY_DOMAIN);
    h.bool(policy.allow_remote);
    h.bool(policy.allow_provider_training);
    h.bool(policy.allow_provider_retention);
    h.bool(policy.allow_third_party_routing);
    h.u64(policy.max_charge_microusd);
    h.finish()
}

/// Bind the full candidate snapshot that admission depended on, not merely the
/// selected provider/model names. This catches privacy/cost/capability drift.
pub fn digest_candidate(
    candidate: &InferenceCandidate,
) -> Result<BindingDigest, InferenceBindingError> {
    let mut h = CanonicalHasher::new(PROVIDER_DOMAIN);
    h.string(&candidate.provider_id);
    h.model_identity(&candidate.model);
    h.u8(execution_location_tag(candidate.location));

    // Supported purposes are semantically a set. Canonicalize order and duplicates.
    let mut purposes: Vec<u8> = candidate
        .supported_purposes
        .iter()
        .copied()
        .map(purpose_tag)
        .collect();
    purposes.sort_unstable();
    purposes.dedup();
    h.u64(purposes.len() as u64);
    for purpose in purposes {
        h.u8(purpose);
    }

    h.u64(candidate.context_window_tokens);
    h.bool(candidate.supports_streaming);
    h.bool(candidate.supports_tools);
    h.bool(candidate.supports_structured_output);
    h.u8(training_policy_tag(candidate.training_policy));
    h.u8(retention_policy_tag(candidate.retention_policy));
    h.u8(routing_policy_tag(candidate.routing_policy));
    h.option_u64(candidate.max_charge_microusd);
    h.finish()
}

pub fn digest_credential_state(
    credential: &CredentialStateBinding,
) -> Result<BindingDigest, InferenceBindingError> {
    let mut h = CanonicalHasher::new(CREDENTIAL_DOMAIN);
    h.string(&credential.credential_id);
    h.u64(credential.epoch);
    h.finish()
}

pub fn digest_quota_state(quota: &QuotaStateBinding) -> Result<BindingDigest, InferenceBindingError> {
    let mut h = CanonicalHasher::new(QUOTA_DOMAIN);
    h.string(&quota.scope_id);
    h.u64(quota.epoch);
    h.option_u64(quota.remaining_requests);
    h.option_u64(quota.remaining_tokens);
    h.option_u64(quota.reset_at_tick);
    h.finish()
}

/// Digest successful model output without retaining the raw text in a receipt.
pub fn digest_response_text(text: &str) -> Result<BindingDigest, InferenceBindingError> {
    let mut h = CanonicalHasher::new(RESPONSE_DOMAIN);
    h.string(text);
    h.finish()
}

struct CanonicalHasher {
    hasher: blake3::Hasher,
}

impl CanonicalHasher {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(ROOT_DOMAIN.len() as u64).to_le_bytes());
        hasher.update(ROOT_DOMAIN);
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

    fn u64(&mut self, value: u64) {
        self.hasher.update(&value.to_le_bytes());
    }

    fn bytes(&mut self, value: &[u8]) {
        self.u64(value.len() as u64);
        self.hasher.update(value);
    }

    fn string(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    fn option_string(&mut self, value: Option<&str>) {
        match value {
            None => self.u8(0),
            Some(value) => {
                self.u8(1);
                self.string(value);
            }
        }
    }

    fn option_u64(&mut self, value: Option<u64>) {
        match value {
            None => self.u8(0),
            Some(value) => {
                self.u8(1);
                self.u64(value);
            }
        }
    }

    fn model_identity(&mut self, model: &ModelIdentity) {
        match model {
            ModelIdentity::ContentVerified { digest } => {
                self.u8(1);
                self.string(digest);
            }
            ModelIdentity::ProviderAttested {
                provider,
                declared_model,
            } => {
                self.u8(2);
                self.string(provider);
                self.string(declared_model);
            }
            ModelIdentity::OpaqueEndpoint { endpoint_id } => {
                self.u8(3);
                self.string(endpoint_id);
            }
        }
    }

    fn finish(self) -> Result<BindingDigest, InferenceBindingError> {
        BindingDigest::new(*self.hasher.finalize().as_bytes()).map_err(InferenceBindingError::Permit)
    }
}

fn purpose_tag(value: InferencePurpose) -> u8 {
    match value {
        InferencePurpose::Translation => 1,
        InferencePurpose::Summarization => 2,
        InferencePurpose::Extraction => 3,
        InferencePurpose::Classification => 4,
        InferencePurpose::GeneralReasoning => 5,
        InferencePurpose::MathematicalReasoning => 6,
        InferencePurpose::ScientificReasoning => 7,
        InferencePurpose::CodeGeneration => 8,
        InferencePurpose::CodeRepair => 9,
        InferencePurpose::CodeReview => 10,
        InferencePurpose::CreativeWriting => 11,
        InferencePurpose::Dialogue => 12,
        InferencePurpose::ToolProposal => 13,
        InferencePurpose::Embedding => 14,
        InferencePurpose::Reranking => 15,
    }
}

fn information_class_tag(value: InformationClass) -> u8 {
    match value {
        InformationClass::Public => 1,
        InformationClass::UserProvided => 2,
        InformationClass::Personal => 3,
        InformationClass::PrivateMemory => 4,
        InformationClass::EpisodicMemory => 5,
        InformationClass::CognitiveState => 6,
        InformationClass::IdentitySecret => 7,
        InformationClass::AuthenticationSecret => 8,
        InformationClass::RemoteSafeDerived => 9,
    }
}

fn execution_location_tag(value: ExecutionLocation) -> u8 {
    match value {
        ExecutionLocation::LocalProcess => 1,
        ExecutionLocation::LocalDevice => 2,
        ExecutionLocation::LocalNetwork => 3,
        ExecutionLocation::RemoteProvider => 4,
        ExecutionLocation::CommunityPeer => 5,
    }
}

fn training_policy_tag(value: ProviderTrainingPolicy) -> u8 {
    match value {
        ProviderTrainingPolicy::Never => 1,
        ProviderTrainingPolicy::MayTrain => 2,
        ProviderTrainingPolicy::Unknown => 3,
    }
}

fn retention_policy_tag(value: ProviderRetentionPolicy) -> u8 {
    match value {
        ProviderRetentionPolicy::ZeroRetention => 1,
        ProviderRetentionPolicy::MayRetain => 2,
        ProviderRetentionPolicy::Unknown => 3,
    }
}

fn routing_policy_tag(value: ProviderRoutingPolicy) -> u8 {
    match value {
        ProviderRoutingPolicy::DirectOnly => 1,
        ProviderRoutingPolicy::MayRouteThirdParty => 2,
        ProviderRoutingPolicy::Unknown => 3,
    }
}

#[derive(Debug)]
pub enum InferenceBindingError {
    Admission(InferenceAdmissionError),
    Permit(InferencePermitError),
    EmptyCredentialId,
    EmptyQuotaScopeId,
}

impl fmt::Display for InferenceBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => write!(f, "inference admission failed: {error}"),
            Self::Permit(error) => write!(f, "inference binding digest failed: {error}"),
            Self::EmptyCredentialId => write!(f, "credential state id must not be empty"),
            Self::EmptyQuotaScopeId => write!(f, "quota scope id must not be empty"),
        }
    }
}

impl std::error::Error for InferenceBindingError {}
