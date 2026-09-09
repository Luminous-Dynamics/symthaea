// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! IF-13 resource-scoped execution binding.
//!
//! IF-12 scope verification is evidence, not execution authority. IF-13 folds the
//! freshly verified scope proof into the IF-11 v3 provider-state binding and owns
//! re-verification adjacent to the private I/O boundary.

#[cfg(not(test))]
use super::inference_binding::{CredentialStateBinding, QuotaStateBinding};
#[cfg(not(test))]
use super::inference_contract::{AdmittedInferenceRoute, InferencePolicy, InferenceRequest};
#[cfg(not(test))]
use super::inference_current_profile_executor::CurrentProfileCredentialExecutor;
#[cfg(not(test))]
use super::inference_execution_envelope::{EndpointStateBinding, InferenceGenerationControls};
#[cfg(not(test))]
use super::inference_executor::{
    InferenceExecutionFailure, InferenceExecutionOutcome, InferenceExecutorFatalError,
    InferenceTickSource,
};
#[cfg(not(test))]
use super::inference_permit::{
    BindingDigest, InferenceExecutionBinding, InferencePermitError, PreparedInferenceExecution,
};
#[cfg(not(test))]
use super::inference_profile_execution::{
    InferenceProfileExecutionError, ProviderProfileResolveError, ProviderProfileResolver,
    admit_and_bind_profile_execution,
};
#[cfg(not(test))]
use super::inference_provider_registry::{ProviderProfileKey, QualifiedProviderCandidate};
#[cfg(not(test))]
use super::inference_receipt::{InferenceFailureClass, InferenceReceipt};
#[cfg(not(test))]
use super::inference_resource_scope::{
    InferenceResourceScopeError, InferenceResourceScopeRegistry, VerifiedInferenceResourceScope,
};

#[cfg(test)]
use crate::inference_binding::{CredentialStateBinding, QuotaStateBinding};
#[cfg(test)]
use crate::inference_contract::{AdmittedInferenceRoute, InferencePolicy, InferenceRequest};
#[cfg(test)]
use crate::inference_current_profile_executor::CurrentProfileCredentialExecutor;
#[cfg(test)]
use crate::inference_execution_envelope::{EndpointStateBinding, InferenceGenerationControls};
#[cfg(test)]
use crate::inference_executor::{
    InferenceExecutionFailure, InferenceExecutionOutcome, InferenceExecutorFatalError,
    InferenceTickSource,
};
#[cfg(test)]
use crate::inference_permit::{
    BindingDigest, InferenceExecutionBinding, InferencePermitError, PreparedInferenceExecution,
};
#[cfg(test)]
use crate::inference_profile_execution::{
    InferenceProfileExecutionError, ProviderProfileResolveError, ProviderProfileResolver,
    admit_and_bind_profile_execution,
};
#[cfg(test)]
use crate::inference_provider_registry::{ProviderProfileKey, QualifiedProviderCandidate};
#[cfg(test)]
use crate::inference_receipt::{InferenceFailureClass, InferenceReceipt};
#[cfg(test)]
use crate::inference_resource_scope::{
    InferenceResourceScopeError, InferenceResourceScopeRegistry, VerifiedInferenceResourceScope,
};

use std::fmt;

const V4_ROOT_DOMAIN: &[u8] = b"symthaea.inference.resource-execution-binding.v4";
const RESOURCE_SCOPE_DOMAIN: &[u8] = b"v3-provider-state-plus-resource-scope-proof";

pub trait InferenceResourceScopeResolver {
    fn verify_current_scope(
        &self,
        profile: &QualifiedProviderCandidate,
        credential: &CredentialStateBinding,
        quota: &QuotaStateBinding,
    ) -> Result<VerifiedInferenceResourceScope, InferenceResourceScopeError>;
}

impl InferenceResourceScopeResolver for InferenceResourceScopeRegistry {
    fn verify_current_scope(
        &self,
        profile: &QualifiedProviderCandidate,
        credential: &CredentialStateBinding,
        quota: &QuotaStateBinding,
    ) -> Result<VerifiedInferenceResourceScope, InferenceResourceScopeError> {
        self.verify(profile, credential, quota)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceScopedExecutableAdmission {
    route: AdmittedInferenceRoute,
    binding: InferenceExecutionBinding,
}

impl ResourceScopedExecutableAdmission {
    pub fn route(&self) -> &AdmittedInferenceRoute {
        &self.route
    }

    pub const fn binding(&self) -> &InferenceExecutionBinding {
        &self.binding
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn admit_and_bind_resource_execution(
    policy: &InferencePolicy,
    request: &InferenceRequest,
    profile: &QualifiedProviderCandidate,
    credential: &CredentialStateBinding,
    quota: &QuotaStateBinding,
    endpoint: &EndpointStateBinding,
    controls: InferenceGenerationControls,
    now_tick: u64,
    scope_proof: &VerifiedInferenceResourceScope,
) -> Result<ResourceScopedExecutableAdmission, InferenceResourceExecutionError> {
    validate_scope_proof(profile, credential, quota, scope_proof)?;
    let base = admit_and_bind_profile_execution(
        policy,
        request,
        profile,
        credential,
        quota,
        endpoint,
        controls,
        now_tick,
    )
    .map_err(InferenceResourceExecutionError::Profile)?;

    let mut binding = *base.binding();
    binding.provider_state_digest =
        digest_resource_provider_state(binding.provider_state_digest, scope_proof)?;
    Ok(ResourceScopedExecutableAdmission {
        route: base.route().clone(),
        binding,
    })
}

fn validate_scope_proof(
    profile: &QualifiedProviderCandidate,
    credential: &CredentialStateBinding,
    quota: &QuotaStateBinding,
    scope_proof: &VerifiedInferenceResourceScope,
) -> Result<(), InferenceResourceExecutionError> {
    let scope = scope_proof.scope();
    if scope.provider_id() != profile.candidate().provider_id
        || scope.deployment_id() != profile.deployment_id()
        || scope.account_scope_id() != profile.account_scope_id()
    {
        return Err(InferenceResourceExecutionError::ScopeProofProfileMismatch);
    }
    if scope_proof.credential_id() != credential.credential_id()
        || scope_proof.credential_epoch() != credential.epoch()
    {
        return Err(InferenceResourceExecutionError::ScopeProofCredentialMismatch);
    }
    if scope_proof.quota_scope_id() != quota.scope_id()
        || scope_proof.quota_epoch() != quota.epoch()
    {
        return Err(InferenceResourceExecutionError::ScopeProofQuotaMismatch);
    }
    Ok(())
}

pub(crate) fn digest_resource_provider_state(
    v3_provider_state: BindingDigest,
    scope_proof: &VerifiedInferenceResourceScope,
) -> Result<BindingDigest, InferenceResourceExecutionError> {
    let mut hasher = blake3::Hasher::new();
    put_bytes(&mut hasher, V4_ROOT_DOMAIN);
    put_bytes(&mut hasher, RESOURCE_SCOPE_DOMAIN);
    hasher.update(v3_provider_state.as_bytes());
    put_bytes(&mut hasher, scope_proof.proof_digest().as_bytes());
    BindingDigest::new(*hasher.finalize().as_bytes())
        .map_err(InferenceResourceExecutionError::Permit)
}

fn put_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

pub struct ResourceScopedCurrentProfileExecutor<C: ?Sized, R, S> {
    current: CurrentProfileCredentialExecutor<C, R>,
    scope_resolver: S,
}

impl<C: InferenceTickSource + ?Sized, R, S> fmt::Debug
    for ResourceScopedCurrentProfileExecutor<C, R, S>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceScopedCurrentProfileExecutor")
            .field("current", &self.current)
            .field("scope_resolver_present", &true)
            .finish()
    }
}

impl<
        C: InferenceTickSource + ?Sized,
        R: ProviderProfileResolver,
        S: InferenceResourceScopeResolver,
    > ResourceScopedCurrentProfileExecutor<C, R, S>
{
    pub fn new(current: CurrentProfileCredentialExecutor<C, R>, scope_resolver: S) -> Self {
        Self {
            current,
            scope_resolver,
        }
    }

    pub fn profile_key(&self) -> &ProviderProfileKey {
        self.current.profile_key()
    }

    pub fn endpoint_binding(&self) -> &EndpointStateBinding {
        self.current.endpoint_binding()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_current(
        &self,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
    ) -> Result<ResourceScopedExecutableAdmission, InferenceResourceExecutionError> {
        let now_tick = self.current.trusted_now_tick();
        let profile = self
            .current
            .resolve_current_profile()
            .map_err(InferenceResourceExecutionError::ProfileResolve)?;
        let proof = self
            .scope_resolver
            .verify_current_scope(&profile, self.current.credential_binding(), quota)
            .map_err(InferenceResourceExecutionError::Scope)?;
        admit_and_bind_resource_execution(
            policy,
            request,
            &profile,
            self.current.credential_binding(),
            quota,
            self.current.endpoint_binding(),
            controls,
            now_tick,
            &proof,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn execute_current(
        &self,
        prepared: PreparedInferenceExecution,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
    ) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
        let now_tick = self.current.trusted_now_tick();
        let profile = match self.current.resolve_current_profile() {
            Ok(profile) => profile,
            Err(_) => {
                return local_verification_failure(
                    prepared,
                    now_tick,
                    InferenceExecutionFailure::ProviderProfileRejected,
                )
            }
        };
        let proof = match self.scope_resolver.verify_current_scope(
            &profile,
            self.current.credential_binding(),
            quota,
        ) {
            Ok(proof) => proof,
            Err(_) => {
                return local_verification_failure(
                    prepared,
                    now_tick,
                    InferenceExecutionFailure::ResourceScopeRejected,
                )
            }
        };
        let rebound = match admit_and_bind_resource_execution(
            policy,
            request,
            &profile,
            self.current.credential_binding(),
            quota,
            self.current.endpoint_binding(),
            controls,
            now_tick,
            &proof,
        ) {
            Ok(bound) => bound,
            Err(_) => {
                return local_verification_failure(
                    prepared,
                    now_tick,
                    InferenceExecutionFailure::ResourceScopeRejected,
                )
            }
        };

        self.current
            .inner()
            .execute_preverified_binding(prepared, *rebound.binding(), request, controls)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn execute_streaming_current(
        &self,
        prepared: PreparedInferenceExecution,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
        let now_tick = self.current.trusted_now_tick();
        let profile = match self.current.resolve_current_profile() {
            Ok(profile) => profile,
            Err(_) => {
                return local_verification_failure(
                    prepared,
                    now_tick,
                    InferenceExecutionFailure::ProviderProfileRejected,
                )
            }
        };
        let proof = match self.scope_resolver.verify_current_scope(
            &profile,
            self.current.credential_binding(),
            quota,
        ) {
            Ok(proof) => proof,
            Err(_) => {
                return local_verification_failure(
                    prepared,
                    now_tick,
                    InferenceExecutionFailure::ResourceScopeRejected,
                )
            }
        };
        let rebound = match admit_and_bind_resource_execution(
            policy,
            request,
            &profile,
            self.current.credential_binding(),
            quota,
            self.current.endpoint_binding(),
            controls,
            now_tick,
            &proof,
        ) {
            Ok(bound) => bound,
            Err(_) => {
                return local_verification_failure(
                    prepared,
                    now_tick,
                    InferenceExecutionFailure::ResourceScopeRejected,
                )
            }
        };

        self.current
            .inner()
            .execute_streaming_preverified_binding(
                prepared,
                *rebound.binding(),
                request,
                controls,
                on_token,
            )
            .await
    }
}

fn local_verification_failure(
    prepared: PreparedInferenceExecution,
    completed_at_tick: u64,
    failure: InferenceExecutionFailure,
) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
    let receipt = InferenceReceipt::failure(
        prepared,
        InferenceFailureClass::VerificationRejected,
        completed_at_tick,
    )
    .map_err(InferenceExecutorFatalError::Receipt)?;
    Ok(InferenceExecutionOutcome {
        response_text: None,
        receipt,
        failure: Some(failure),
        wire_observation: None,
    })
}

#[derive(Debug)]
pub enum InferenceResourceExecutionError {
    ProfileResolve(ProviderProfileResolveError),
    Scope(InferenceResourceScopeError),
    Profile(InferenceProfileExecutionError),
    Permit(InferencePermitError),
    ScopeProofProfileMismatch,
    ScopeProofCredentialMismatch,
    ScopeProofQuotaMismatch,
}

impl fmt::Display for InferenceResourceExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProfileResolve(error) => write!(f, "current provider profile rejected: {error}"),
            Self::Scope(error) => write!(f, "resource scope verification failed: {error}"),
            Self::Profile(error) => write!(f, "profile execution binding failed: {error}"),
            Self::Permit(error) => write!(f, "resource execution digest failed: {error}"),
            Self::ScopeProofProfileMismatch => {
                write!(f, "resource scope proof does not match provider profile")
            }
            Self::ScopeProofCredentialMismatch => {
                write!(f, "resource scope proof does not match credential identity")
            }
            Self::ScopeProofQuotaMismatch => {
                write!(f, "resource scope proof does not match quota identity")
            }
        }
    }
}

impl std::error::Error for InferenceResourceExecutionError {}
