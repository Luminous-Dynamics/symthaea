// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! IF-11 safe executor that owns current-provider-profile resolution.
//!
//! Callers cannot supply a qualified profile, endpoint, provider/model tuple, or
//! freshness timestamp at execution. Construction resolves the current profile,
//! derives the private HTTP executor from that profile, and shares one exact Arc
//! clock between profile freshness checks and the underlying inference executor.
//! The v3 profile theorem is owned here; lower credential/HTTP layers only receive
//! an exact preverified binding.

#[cfg(not(test))]
use super::inference_binding::{CredentialStateBinding, QuotaStateBinding};
#[cfg(not(test))]
use super::inference_contract::{InferencePolicy, InferenceRequest, ModelIdentity};
#[cfg(not(test))]
use super::inference_credential::{
    CredentialBoundOpenAiExecutor, InferenceCredentialExecutorError, InferenceCredentialLease,
};
#[cfg(not(test))]
use super::inference_execution_envelope::{EndpointStateBinding, InferenceGenerationControls};
#[cfg(not(test))]
use super::inference_executor::{
    InferenceExecutionFailure, InferenceExecutionOutcome, InferenceExecutorFatalError,
    InferenceTickSource,
};
#[cfg(not(test))]
use super::inference_permit::PreparedInferenceExecution;
#[cfg(not(test))]
use super::inference_profile_execution::{
    InferenceProfileExecutionError, ProfileBoundExecutableAdmission, ProviderProfileResolveError,
    ProviderProfileResolver, admit_and_bind_profile_execution,
};
#[cfg(not(test))]
use super::inference_provider_registry::{
    ProviderProfileKey, ProviderQualificationPolicy, QualifiedProviderCandidate,
};
#[cfg(not(test))]
use super::inference_receipt::{InferenceFailureClass, InferenceReceipt};

#[cfg(test)]
use crate::inference_binding::{CredentialStateBinding, QuotaStateBinding};
#[cfg(test)]
use crate::inference_contract::{InferencePolicy, InferenceRequest, ModelIdentity};
#[cfg(test)]
use crate::inference_credential::{
    CredentialBoundOpenAiExecutor, InferenceCredentialExecutorError, InferenceCredentialLease,
};
#[cfg(test)]
use crate::inference_execution_envelope::{EndpointStateBinding, InferenceGenerationControls};
#[cfg(test)]
use crate::inference_executor::{
    InferenceExecutionFailure, InferenceExecutionOutcome, InferenceExecutorFatalError,
    InferenceTickSource,
};
#[cfg(test)]
use crate::inference_permit::PreparedInferenceExecution;
#[cfg(test)]
use crate::inference_profile_execution::{
    InferenceProfileExecutionError, ProfileBoundExecutableAdmission, ProviderProfileResolveError,
    ProviderProfileResolver, admit_and_bind_profile_execution,
};
#[cfg(test)]
use crate::inference_provider_registry::{
    ProviderProfileKey, ProviderQualificationPolicy, QualifiedProviderCandidate,
};
#[cfg(test)]
use crate::inference_receipt::{InferenceFailureClass, InferenceReceipt};

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

impl<T: InferenceTickSource + ?Sized> InferenceTickSource for Arc<T> {
    fn now_tick(&self) -> u64 {
        (**self).now_tick()
    }
}

pub struct CurrentProfileCredentialExecutor<C: ?Sized, R> {
    executor: CredentialBoundOpenAiExecutor<Arc<C>>,
    resolver: R,
    profile_key: ProviderProfileKey,
    qualification_policy: ProviderQualificationPolicy,
    clock: Arc<C>,
}

impl<C: InferenceTickSource + ?Sized, R> fmt::Debug for CurrentProfileCredentialExecutor<C, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CurrentProfileCredentialExecutor")
            .field("executor", &self.executor)
            .field("profile_key", &self.profile_key)
            .field("qualification_policy", &self.qualification_policy)
            .field("resolver_present", &true)
            .field("shared_trusted_clock_present", &true)
            .finish()
    }
}

impl<C: InferenceTickSource + ?Sized, R: ProviderProfileResolver>
    CurrentProfileCredentialExecutor<C, R>
{
    #[allow(clippy::too_many_arguments)]
    pub fn from_current_profile(
        resolver: R,
        profile_key: ProviderProfileKey,
        qualification_policy: ProviderQualificationPolicy,
        lease: InferenceCredentialLease,
        timeout: Duration,
        config_epoch: u64,
        clock: Arc<C>,
    ) -> Result<Self, CurrentProfileExecutorConfigError> {
        let now_tick = clock.now_tick();
        let profile = resolver
            .resolve_current(&profile_key, now_tick, qualification_policy)
            .map_err(CurrentProfileExecutorConfigError::Resolve)?;

        let candidate = profile.candidate();
        let declared_model = match &candidate.model {
            ModelIdentity::ProviderAttested {
                provider,
                declared_model,
            } if provider == &candidate.provider_id => declared_model.clone(),
            _ => return Err(CurrentProfileExecutorConfigError::ModelIdentityMismatch),
        };

        let executor = CredentialBoundOpenAiExecutor::from_lease(
            candidate.provider_id.clone(),
            profile.endpoint().base_url(),
            declared_model,
            lease,
            timeout,
            config_epoch,
            clock.clone(),
        )
        .map_err(CurrentProfileExecutorConfigError::CredentialExecutor)?;

        Ok(Self {
            executor,
            resolver,
            profile_key,
            qualification_policy,
            clock,
        })
    }

    pub fn profile_key(&self) -> &ProviderProfileKey {
        &self.profile_key
    }

    /// Read-only endpoint metadata for diagnostics/tests. No lower executor handle
    /// is exposed publicly, preventing callers from bypassing current-profile checks.
    pub fn endpoint_binding(&self) -> &EndpointStateBinding {
        self.executor.endpoint_binding()
    }

    /// Non-secret credential identity used in authority bindings.
    pub fn credential_binding(&self) -> &CredentialStateBinding {
        self.executor.credential_binding()
    }

    /// Crate-internal access for stricter child wrappers only. External callers
    /// cannot obtain the lower credential/transport executor.
    pub(crate) fn inner(&self) -> &CredentialBoundOpenAiExecutor<Arc<C>> {
        &self.executor
    }

    pub(crate) fn trusted_now_tick(&self) -> u64 {
        self.clock.now_tick()
    }

    pub(crate) fn resolve_current_profile(
        &self,
    ) -> Result<QualifiedProviderCandidate, ProviderProfileResolveError> {
        self.resolver.resolve_current(
            &self.profile_key,
            self.clock.now_tick(),
            self.qualification_policy,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_current(
        &self,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
    ) -> Result<ProfileBoundExecutableAdmission, CurrentProfileBindingError> {
        let now_tick = self.clock.now_tick();
        let profile = self
            .resolver
            .resolve_current(&self.profile_key, now_tick, self.qualification_policy)
            .map_err(CurrentProfileBindingError::Resolve)?;
        admit_and_bind_profile_execution(
            policy,
            request,
            &profile,
            self.executor.credential_binding(),
            quota,
            self.executor.endpoint_binding(),
            controls,
            now_tick,
        )
        .map_err(CurrentProfileBindingError::Profile)
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
        let now_tick = self.clock.now_tick();
        let profile = match self
            .resolver
            .resolve_current(&self.profile_key, now_tick, self.qualification_policy)
        {
            Ok(profile) => profile,
            Err(_) => return profile_verification_failure(prepared, now_tick),
        };
        let rebound = match admit_and_bind_profile_execution(
            policy,
            request,
            &profile,
            self.executor.credential_binding(),
            quota,
            self.executor.endpoint_binding(),
            controls,
            now_tick,
        ) {
            Ok(bound) => bound,
            Err(_) => return profile_verification_failure(prepared, now_tick),
        };
        self.executor
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
        let now_tick = self.clock.now_tick();
        let profile = match self
            .resolver
            .resolve_current(&self.profile_key, now_tick, self.qualification_policy)
        {
            Ok(profile) => profile,
            Err(_) => return profile_verification_failure(prepared, now_tick),
        };
        let rebound = match admit_and_bind_profile_execution(
            policy,
            request,
            &profile,
            self.executor.credential_binding(),
            quota,
            self.executor.endpoint_binding(),
            controls,
            now_tick,
        ) {
            Ok(bound) => bound,
            Err(_) => return profile_verification_failure(prepared, now_tick),
        };
        self.executor
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

fn profile_verification_failure(
    prepared: PreparedInferenceExecution,
    completed_at_tick: u64,
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
        failure: Some(InferenceExecutionFailure::ProviderProfileRejected),
        wire_observation: None,
    })
}

#[derive(Debug)]
pub enum CurrentProfileExecutorConfigError {
    Resolve(ProviderProfileResolveError),
    ModelIdentityMismatch,
    CredentialExecutor(InferenceCredentialExecutorError),
}

impl fmt::Display for CurrentProfileExecutorConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resolve(error) => {
                write!(f, "current provider profile construction failed: {error}")
            }
            Self::ModelIdentityMismatch => write!(
                f,
                "current provider profile has incoherent provider-attested model identity"
            ),
            Self::CredentialExecutor(error) => {
                write!(f, "current provider executor construction failed: {error}")
            }
        }
    }
}

impl std::error::Error for CurrentProfileExecutorConfigError {}

#[derive(Debug)]
pub enum CurrentProfileBindingError {
    Resolve(ProviderProfileResolveError),
    Profile(InferenceProfileExecutionError),
}

impl fmt::Display for CurrentProfileBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resolve(error) => {
                write!(f, "current provider profile resolution failed: {error}")
            }
            Self::Profile(error) => write!(f, "current provider profile binding failed: {error}"),
        }
    }
}

impl std::error::Error for CurrentProfileBindingError {}
