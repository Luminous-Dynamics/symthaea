// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! IF-9 credential-bound inference executor.
//!
//! A credential lease carries both the secret material used on the wire and the
//! exact non-secret credential identity/epoch used in inference bindings.

#[cfg(not(test))]
use super::inference_binding::{CredentialStateBinding, InferenceBindingError, QuotaStateBinding};
#[cfg(not(test))]
use super::inference_contract::{InferenceCandidate, InferencePolicy, InferenceRequest};
#[cfg(not(test))]
use super::inference_execution_envelope::{EndpointStateBinding, InferenceGenerationControls};
#[cfg(not(test))]
use super::inference_executor::{
    InferenceExecutionOutcome, InferenceExecutorConfigError, InferenceExecutorFatalError,
    InferenceTickSource, OpenAiInferenceExecutor,
};
#[cfg(not(test))]
use super::inference_permit::PreparedInferenceExecution;
#[cfg(not(test))]
use super::openai_compatible_transport::{BearerCredential, TransportConfigError, TransportCredential};

#[cfg(test)]
use crate::inference_binding::{CredentialStateBinding, InferenceBindingError, QuotaStateBinding};
#[cfg(test)]
use crate::inference_contract::{InferenceCandidate, InferencePolicy, InferenceRequest};
#[cfg(test)]
use crate::inference_execution_envelope::{EndpointStateBinding, InferenceGenerationControls};
#[cfg(test)]
use crate::inference_executor::{
    InferenceExecutionOutcome, InferenceExecutorConfigError, InferenceExecutorFatalError,
    InferenceTickSource, OpenAiInferenceExecutor,
};
#[cfg(test)]
use crate::inference_permit::PreparedInferenceExecution;
#[cfg(test)]
use crate::openai_compatible_transport::{BearerCredential, TransportConfigError, TransportCredential};

use std::fmt;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceCredentialMode {
    Anonymous,
    Bearer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceCredentialHandle {
    credential_id: String,
    expected_epoch: u64,
    mode: InferenceCredentialMode,
}

impl InferenceCredentialHandle {
    pub fn new(
        credential_id: impl Into<String>,
        expected_epoch: u64,
        mode: InferenceCredentialMode,
    ) -> Result<Self, InferenceCredentialError> {
        let credential_id = credential_id.into();
        if credential_id.trim().is_empty() {
            return Err(InferenceCredentialError::EmptyCredentialId);
        }
        Ok(Self { credential_id, expected_epoch, mode })
    }

    pub fn credential_id(&self) -> &str { &self.credential_id }
    pub const fn expected_epoch(&self) -> u64 { self.expected_epoch }
    pub const fn mode(&self) -> InferenceCredentialMode { self.mode }
}

/// Non-clone credential capability. Secret material is deliberately absent from Debug.
pub struct InferenceCredentialLease {
    binding: CredentialStateBinding,
    mode: InferenceCredentialMode,
    transport: TransportCredential,
}

impl fmt::Debug for InferenceCredentialLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceCredentialLease")
            .field("credential_id", &self.binding.credential_id())
            .field("epoch", &self.binding.epoch())
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl InferenceCredentialLease {
    pub fn bearer(
        credential_id: impl Into<String>,
        epoch: u64,
        secret: impl Into<String>,
    ) -> Result<Self, InferenceCredentialError> {
        let credential_id = credential_id.into();
        let binding = CredentialStateBinding::new(credential_id, epoch)?;
        let bearer = BearerCredential::new(secret.into())?;
        Ok(Self {
            binding,
            mode: InferenceCredentialMode::Bearer,
            transport: TransportCredential::Bearer(bearer),
        })
    }

    pub fn anonymous(
        credential_id: impl Into<String>,
        epoch: u64,
    ) -> Result<Self, InferenceCredentialError> {
        Ok(Self {
            binding: CredentialStateBinding::new(credential_id, epoch)?,
            mode: InferenceCredentialMode::Anonymous,
            transport: TransportCredential::None,
        })
    }

    pub fn binding(&self) -> &CredentialStateBinding { &self.binding }
    pub const fn mode(&self) -> InferenceCredentialMode { self.mode }

    fn into_parts(self) -> (CredentialStateBinding, TransportCredential) {
        (self.binding, self.transport)
    }
}

/// Resolver boundary for future OS-keyring/Xenia-backed credential stores.
pub trait InferenceCredentialResolver {
    fn resolve(
        &self,
        handle: &InferenceCredentialHandle,
    ) -> Result<InferenceCredentialLease, InferenceCredentialResolveError>;
}

/// Validate that a resolver returned exactly the handle the caller requested.
pub fn resolve_credential<R: InferenceCredentialResolver>(
    resolver: &R,
    handle: &InferenceCredentialHandle,
) -> Result<InferenceCredentialLease, InferenceCredentialResolveError> {
    let lease = resolver.resolve(handle)?;
    if lease.binding().credential_id() != handle.credential_id()
        || lease.binding().epoch() != handle.expected_epoch()
        || lease.mode() != handle.mode()
    {
        return Err(InferenceCredentialResolveError::BindingMismatch);
    }
    Ok(lease)
}

/// Safe wrapper: callers cannot supply a different credential binding at execution.
pub struct CredentialBoundOpenAiExecutor<C> {
    inner: OpenAiInferenceExecutor<C>,
    binding: CredentialStateBinding,
}

impl<C: InferenceTickSource> fmt::Debug for CredentialBoundOpenAiExecutor<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CredentialBoundOpenAiExecutor")
            .field("credential_id", &self.binding.credential_id())
            .field("credential_epoch", &self.binding.epoch())
            .field("endpoint", self.inner.endpoint_binding())
            .finish_non_exhaustive()
    }
}

impl<C: InferenceTickSource> CredentialBoundOpenAiExecutor<C> {
    #[allow(clippy::too_many_arguments)]
    pub fn from_lease(
        provider_id: impl Into<String>,
        base_url: &str,
        wire_model: impl Into<String>,
        lease: InferenceCredentialLease,
        timeout: Duration,
        config_epoch: u64,
        clock: C,
    ) -> Result<Self, InferenceCredentialExecutorError> {
        let (binding, transport) = lease.into_parts();
        let inner = OpenAiInferenceExecutor::new(
            provider_id,
            base_url,
            wire_model,
            transport,
            timeout,
            config_epoch,
            clock,
        )?;
        Ok(Self { inner, binding })
    }

    pub fn credential_binding(&self) -> &CredentialStateBinding { &self.binding }
    pub fn endpoint_binding(&self) -> &EndpointStateBinding { self.inner.endpoint_binding() }

    #[allow(clippy::too_many_arguments)]
    pub async fn execute(
        &self,
        prepared: PreparedInferenceExecution,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        candidate: &InferenceCandidate,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
    ) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
        self.inner
            .execute(
                prepared,
                policy,
                request,
                candidate,
                &self.binding,
                quota,
                controls,
            )
            .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn execute_streaming(
        &self,
        prepared: PreparedInferenceExecution,
        policy: &InferencePolicy,
        request: &InferenceRequest,
        candidate: &InferenceCandidate,
        quota: &QuotaStateBinding,
        controls: InferenceGenerationControls,
        on_token: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<InferenceExecutionOutcome, InferenceExecutorFatalError> {
        self.inner
            .execute_streaming(
                prepared,
                policy,
                request,
                candidate,
                &self.binding,
                quota,
                controls,
                on_token,
            )
            .await
    }
}

#[derive(Debug)]
pub enum InferenceCredentialError {
    EmptyCredentialId,
    Binding(InferenceBindingError),
    Transport(TransportConfigError),
}

impl From<InferenceBindingError> for InferenceCredentialError {
    fn from(value: InferenceBindingError) -> Self { Self::Binding(value) }
}
impl From<TransportConfigError> for InferenceCredentialError {
    fn from(value: TransportConfigError) -> Self { Self::Transport(value) }
}

impl fmt::Display for InferenceCredentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCredentialId => write!(f, "credential id must not be empty"),
            Self::Binding(error) => write!(f, "credential binding rejected: {error}"),
            Self::Transport(error) => write!(f, "credential material rejected: {error}"),
        }
    }
}
impl std::error::Error for InferenceCredentialError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceCredentialResolveError {
    NotFound,
    BackendUnavailable,
    BindingMismatch,
}

impl fmt::Display for InferenceCredentialResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "credential handle was not found"),
            Self::BackendUnavailable => write!(f, "credential resolver is unavailable"),
            Self::BindingMismatch => write!(f, "resolved credential does not match requested handle"),
        }
    }
}
impl std::error::Error for InferenceCredentialResolveError {}

#[derive(Debug)]
pub enum InferenceCredentialExecutorError {
    Executor(InferenceExecutorConfigError),
}

impl From<InferenceExecutorConfigError> for InferenceCredentialExecutorError {
    fn from(value: InferenceExecutorConfigError) -> Self { Self::Executor(value) }
}

impl fmt::Display for InferenceCredentialExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Executor(error) => write!(f, "credential-bound executor rejected: {error}"),
        }
    }
}
impl std::error::Error for InferenceCredentialExecutorError {}
