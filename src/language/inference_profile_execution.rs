// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! IF-11 current-provider-profile execution binding.
//!
//! A qualified profile is evidence, not timeless execution authority. IF-11 composes
//! IF-4's v2 execution binding with the exact IF-10 deployment/profile identity and
//! requires the current registry profile to be resolved again adjacent to execution.

#[cfg(not(test))]
use super::inference_binding::{CredentialStateBinding, QuotaStateBinding};
#[cfg(not(test))]
use super::inference_contract::{
    AdmittedInferenceRoute, InferencePolicy, InferenceRequest, ModelIdentity,
};
#[cfg(not(test))]
use super::inference_execution_envelope::{
    BoundExecutableAdmission, EndpointProtocol, EndpointStateBinding,
    InferenceExecutionEnvelopeError, InferenceGenerationControls, admit_and_bind_execution,
};
#[cfg(not(test))]
use super::inference_permit::{BindingDigest, InferenceExecutionBinding, InferencePermitError};
#[cfg(not(test))]
use super::inference_provider_registry::{
    ProviderProfileKey, ProviderQualificationError, ProviderQualificationPolicy, ProviderRegistry,
    ProviderProtocol, QualifiedProviderCandidate,
};

#[cfg(test)]
use crate::inference_binding::{CredentialStateBinding, QuotaStateBinding};
#[cfg(test)]
use crate::inference_contract::{
    AdmittedInferenceRoute, InferencePolicy, InferenceRequest, ModelIdentity,
};
#[cfg(test)]
use crate::inference_execution_envelope::{
    BoundExecutableAdmission, EndpointProtocol, EndpointStateBinding,
    InferenceExecutionEnvelopeError, InferenceGenerationControls, admit_and_bind_execution,
};
#[cfg(test)]
use crate::inference_permit::{BindingDigest, InferenceExecutionBinding, InferencePermitError};
#[cfg(test)]
use crate::inference_provider_registry::{
    ProviderProfileKey, ProviderQualificationError, ProviderQualificationPolicy, ProviderRegistry,
    ProviderProtocol, QualifiedProviderCandidate,
};

use std::fmt;

const V3_ROOT_DOMAIN: &[u8] = b"symthaea.inference.provider-execution-binding.v3";
const PROVIDER_PROFILE_DOMAIN: &[u8] = b"v2-provider-state-plus-current-profile";

pub trait ProviderProfileResolver {
    fn resolve_current(
        &self,
        key: &ProviderProfileKey,
        now_tick: u64,
        policy: ProviderQualificationPolicy,
    ) -> Result<QualifiedProviderCandidate, ProviderProfileResolveError>;
}

impl ProviderProfileResolver for ProviderRegistry {
    fn resolve_current(
        &self,
        key: &ProviderProfileKey,
        now_tick: u64,
        policy: ProviderQualificationPolicy,
    ) -> Result<QualifiedProviderCandidate, ProviderProfileResolveError> {
        self.get(key)
            .ok_or(ProviderProfileResolveError::ProfileNotFound)?
            .qualify(now_tick, policy)
            .map_err(ProviderProfileResolveError::Qualification)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileBoundExecutableAdmission {
    route: AdmittedInferenceRoute,
    binding: InferenceExecutionBinding,
}

impl ProfileBoundExecutableAdmission {
    pub fn route(&self) -> &AdmittedInferenceRoute {
        &self.route
    }

    pub const fn binding(&self) -> &InferenceExecutionBinding {
        &self.binding
    }
}

#[allow(clippy::too_many_arguments)]
pub fn admit_and_bind_profile_execution(
    policy: &InferencePolicy,
    request: &InferenceRequest,
    profile: &QualifiedProviderCandidate,
    credential: &CredentialStateBinding,
    quota: &QuotaStateBinding,
    endpoint: &EndpointStateBinding,
    controls: InferenceGenerationControls,
    now_tick: u64,
) -> Result<ProfileBoundExecutableAdmission, InferenceProfileExecutionError> {
    if !profile.is_fresh_at(now_tick) {
        return Err(InferenceProfileExecutionError::ProfileExpiredOrNotYetValid);
    }
    validate_profile_endpoint(profile, endpoint)?;

    let base = admit_and_bind_execution(
        policy,
        request,
        profile.candidate(),
        credential,
        quota,
        endpoint,
        controls,
    )
    .map_err(InferenceProfileExecutionError::Envelope)?;

    profile_bind(base, profile)
}

fn profile_bind(
    base: BoundExecutableAdmission,
    profile: &QualifiedProviderCandidate,
) -> Result<ProfileBoundExecutableAdmission, InferenceProfileExecutionError> {
    let mut binding = *base.binding();
    binding.provider_state_digest =
        digest_profile_provider_state(binding.provider_state_digest, profile)?;
    Ok(ProfileBoundExecutableAdmission {
        route: base.route().clone(),
        binding,
    })
}

/// Compose the stable current-profile identity into provider state.
///
/// `qualified_at_tick` is deliberately NOT included: re-resolving the same claim
/// bundle later must reproduce the same binding. Freshness is checked separately
/// against `valid_until_tick` on every bind/rebind.
pub fn digest_profile_provider_state(
    v2_provider_state: BindingDigest,
    profile: &QualifiedProviderCandidate,
) -> Result<BindingDigest, InferenceProfileExecutionError> {
    let mut hasher = blake3::Hasher::new();
    put_bytes(&mut hasher, V3_ROOT_DOMAIN);
    put_bytes(&mut hasher, PROVIDER_PROFILE_DOMAIN);
    hasher.update(v2_provider_state.as_bytes());
    put_bytes(&mut hasher, profile.profile_digest().as_bytes());
    put_string(&mut hasher, profile.deployment_id());
    put_string(&mut hasher, profile.account_scope_id());
    hasher.update(&profile.profile_epoch().to_le_bytes());
    hasher.update(&profile.valid_until_tick().to_le_bytes());
    BindingDigest::new(*hasher.finalize().as_bytes()).map_err(InferenceProfileExecutionError::Permit)
}

fn validate_profile_endpoint(
    profile: &QualifiedProviderCandidate,
    endpoint: &EndpointStateBinding,
) -> Result<(), InferenceProfileExecutionError> {
    let candidate = profile.candidate();
    if candidate.provider_id != endpoint.provider_id() {
        return Err(InferenceProfileExecutionError::EndpointProviderMismatch);
    }

    let declared_model = match &candidate.model {
        ModelIdentity::ProviderAttested {
            provider,
            declared_model,
        } if provider == &candidate.provider_id => declared_model,
        _ => return Err(InferenceProfileExecutionError::ProfileModelIdentityMismatch),
    };
    if declared_model != endpoint.wire_model() {
        return Err(InferenceProfileExecutionError::EndpointModelMismatch);
    }

    match (profile.endpoint().protocol(), endpoint.protocol()) {
        (ProviderProtocol::OpenAiCompatibleV1, EndpointProtocol::OpenAiCompatibleV1) => {}
    }
    if profile.endpoint().base_url() != endpoint.base_url() {
        return Err(InferenceProfileExecutionError::EndpointUrlMismatch);
    }
    Ok(())
}

fn put_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn put_string(hasher: &mut blake3::Hasher, value: &str) {
    put_bytes(hasher, value.as_bytes());
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderProfileResolveError {
    ProfileNotFound,
    Qualification(ProviderQualificationError),
}

impl fmt::Display for ProviderProfileResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProfileNotFound => write!(f, "current provider profile was not found"),
            Self::Qualification(error) => write!(f, "current provider profile rejected: {error}"),
        }
    }
}

impl std::error::Error for ProviderProfileResolveError {}

#[derive(Debug)]
pub enum InferenceProfileExecutionError {
    Envelope(InferenceExecutionEnvelopeError),
    Permit(InferencePermitError),
    ProfileExpiredOrNotYetValid,
    EndpointProviderMismatch,
    EndpointModelMismatch,
    EndpointUrlMismatch,
    ProfileModelIdentityMismatch,
}

impl fmt::Display for InferenceProfileExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Envelope(error) => write!(f, "base inference execution binding failed: {error}"),
            Self::Permit(error) => write!(f, "provider-profile digest failed: {error}"),
            Self::ProfileExpiredOrNotYetValid => {
                write!(f, "qualified provider profile is not fresh")
            }
            Self::EndpointProviderMismatch => {
                write!(f, "profile provider does not match executor endpoint")
            }
            Self::EndpointModelMismatch => {
                write!(f, "profile model does not match executor wire model")
            }
            Self::EndpointUrlMismatch => {
                write!(f, "profile endpoint URL does not match executor endpoint")
            }
            Self::ProfileModelIdentityMismatch => write!(
                f,
                "qualified profile has incoherent provider-attested model identity"
            ),
        }
    }
}

impl std::error::Error for InferenceProfileExecutionError {}
