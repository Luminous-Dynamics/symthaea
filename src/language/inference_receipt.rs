// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IF-3/IF-6 inference execution receipts.
//!
//! Receipts are evidence, not authority. Successful receipts retain only a
//! domain-separated digest of model output, never the raw response text.
//!
//! IF-6 may additionally retain a privacy-minimized projection of provider wire
//! observations. Raw provider request/response ids are digested before entering
//! the receipt. Later settlement layers may also inspect the terminal HTTP status
//! of a failed provider request; provider response bodies remain excluded.

#[cfg(not(test))]
use super::inference_binding::{InferenceBindingError, digest_response_text};
#[cfg(not(test))]
use super::inference_permit::{
    BindingDigest, InferenceExecutionBinding, InferencePermitError, PreparedInferenceExecution,
};

#[cfg(test)]
use crate::inference_binding::{InferenceBindingError, digest_response_text};
#[cfg(test)]
use crate::inference_permit::{
    BindingDigest, InferenceExecutionBinding, InferencePermitError, PreparedInferenceExecution,
};

use std::fmt;

const WIRE_ID_ROOT_DOMAIN: &[u8] = b"symthaea.inference.receipt.wire-id.v1";
const RESPONSE_ID_DOMAIN: &[u8] = b"provider-response-id";
const REQUEST_ID_DOMAIN: &[u8] = b"provider-request-id";
const MAX_RECEIPT_METADATA_BYTES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceFailureClass {
    Transport,
    ProviderHttp,
    ResponseDecode,
    EmptyResponse,
    Cancelled,
    VerificationRejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceCompletion {
    Success { response_digest: BindingDigest },
    Failure { class: InferenceFailureClass },
}

/// Provider-declared usage evidence. These counts are observations rather than
/// independently verified tokenizer measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InferenceTokenUsageEvidence {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// Rate-limit evidence normalized by the transport. Reset values are relative
/// durations, not trusted wall-clock timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InferenceRateLimitEvidence {
    pub retry_after_millis: Option<u64>,
    pub request_limit: Option<u64>,
    pub request_remaining: Option<u64>,
    pub request_reset_after_millis: Option<u64>,
    pub token_limit: Option<u64>,
    pub token_remaining: Option<u64>,
    pub token_reset_after_millis: Option<u64>,
}

/// Privacy-minimized provider wire evidence retained by an inference receipt.
///
/// Raw provider request/response ids are never stored. Their domain-separated
/// BLAKE3 digests are sufficient for later equality/correlation checks against an
/// explicitly supplied external provider record. `provider_http_status` is only a
/// numeric terminal status observed for a failed HTTP request; no remote body is
/// retained with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceWireEvidence {
    provider_response_id_digest: Option<BindingDigest>,
    provider_request_id_digest: Option<BindingDigest>,
    provider_declared_model: Option<String>,
    system_fingerprint: Option<String>,
    finish_reason: Option<String>,
    usage: Option<InferenceTokenUsageEvidence>,
    rate_limits: InferenceRateLimitEvidence,
    latency_millis: u64,
    metadata_conflict: bool,
    metadata_rejected: bool,
    provider_http_status: Option<u16>,
}

impl InferenceWireEvidence {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider_response_id: Option<&str>,
        provider_request_id: Option<&str>,
        provider_declared_model: Option<&str>,
        system_fingerprint: Option<&str>,
        finish_reason: Option<&str>,
        usage: Option<InferenceTokenUsageEvidence>,
        rate_limits: InferenceRateLimitEvidence,
        latency_millis: u64,
        metadata_conflict: bool,
        metadata_rejected: bool,
    ) -> Result<Self, InferenceReceiptError> {
        Ok(Self {
            provider_response_id_digest: provider_response_id
                .map(|value| digest_wire_identifier(RESPONSE_ID_DOMAIN, value))
                .transpose()?,
            provider_request_id_digest: provider_request_id
                .map(|value| digest_wire_identifier(REQUEST_ID_DOMAIN, value))
                .transpose()?,
            provider_declared_model: bounded_metadata(provider_declared_model)?,
            system_fingerprint: bounded_metadata(system_fingerprint)?,
            finish_reason: bounded_metadata(finish_reason)?,
            usage,
            rate_limits,
            latency_millis,
            metadata_conflict,
            metadata_rejected,
            provider_http_status: None,
        })
    }

    /// Attach the terminal HTTP status observed by the transport. Kept crate-only
    /// so ordinary callers cannot manufacture settlement evidence on a receipt.
    pub(crate) fn with_provider_http_status(mut self, status: Option<u16>) -> Self {
        self.provider_http_status = status;
        self
    }

    pub fn provider_response_id_digest(&self) -> Option<&BindingDigest> {
        self.provider_response_id_digest.as_ref()
    }

    pub fn provider_request_id_digest(&self) -> Option<&BindingDigest> {
        self.provider_request_id_digest.as_ref()
    }

    pub fn provider_declared_model(&self) -> Option<&str> {
        self.provider_declared_model.as_deref()
    }

    pub fn system_fingerprint(&self) -> Option<&str> {
        self.system_fingerprint.as_deref()
    }

    pub fn finish_reason(&self) -> Option<&str> {
        self.finish_reason.as_deref()
    }

    pub const fn usage(&self) -> Option<InferenceTokenUsageEvidence> {
        self.usage
    }

    pub const fn rate_limits(&self) -> InferenceRateLimitEvidence {
        self.rate_limits
    }

    pub const fn latency_millis(&self) -> u64 {
        self.latency_millis
    }

    pub const fn metadata_conflict(&self) -> bool {
        self.metadata_conflict
    }

    pub const fn metadata_rejected(&self) -> bool {
        self.metadata_rejected
    }

    pub const fn provider_http_status(&self) -> Option<u16> {
        self.provider_http_status
    }
}

/// Non-authoritative evidence that a prepared inference attempt reached a terminal state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceReceipt {
    binding: InferenceExecutionBinding,
    permit_generation: u64,
    prepared_at_tick: u64,
    completed_at_tick: u64,
    completion: InferenceCompletion,
    wire_evidence: Option<InferenceWireEvidence>,
}

impl InferenceReceipt {
    /// Consume the prepared authority object and produce success evidence.
    /// Raw response text is hashed and immediately discarded by this API.
    pub fn success(
        prepared: PreparedInferenceExecution,
        response_text: &str,
        completed_at_tick: u64,
    ) -> Result<Self, InferenceReceiptError> {
        Self::success_with_wire(prepared, response_text, completed_at_tick, None)
    }

    /// Success receipt with an optional privacy-minimized provider observation.
    pub fn success_observed(
        prepared: PreparedInferenceExecution,
        response_text: &str,
        completed_at_tick: u64,
        wire_evidence: InferenceWireEvidence,
    ) -> Result<Self, InferenceReceiptError> {
        Self::success_with_wire(
            prepared,
            response_text,
            completed_at_tick,
            Some(wire_evidence),
        )
    }

    fn success_with_wire(
        prepared: PreparedInferenceExecution,
        response_text: &str,
        completed_at_tick: u64,
        wire_evidence: Option<InferenceWireEvidence>,
    ) -> Result<Self, InferenceReceiptError> {
        validate_completion_time(&prepared, completed_at_tick)?;
        let response_digest =
            digest_response_text(response_text).map_err(InferenceReceiptError::Binding)?;
        Ok(Self {
            binding: *prepared.binding(),
            permit_generation: prepared.generation(),
            prepared_at_tick: prepared.prepared_at_tick(),
            completed_at_tick,
            completion: InferenceCompletion::Success { response_digest },
            wire_evidence,
        })
    }

    /// Consume the prepared authority object and produce failure evidence.
    pub fn failure(
        prepared: PreparedInferenceExecution,
        class: InferenceFailureClass,
        completed_at_tick: u64,
    ) -> Result<Self, InferenceReceiptError> {
        Self::failure_with_wire(prepared, class, completed_at_tick, None)
    }

    /// Failure receipt retaining only safe wire evidence such as rate-limit hints.
    pub fn failure_observed(
        prepared: PreparedInferenceExecution,
        class: InferenceFailureClass,
        completed_at_tick: u64,
        wire_evidence: InferenceWireEvidence,
    ) -> Result<Self, InferenceReceiptError> {
        Self::failure_with_wire(prepared, class, completed_at_tick, Some(wire_evidence))
    }

    fn failure_with_wire(
        prepared: PreparedInferenceExecution,
        class: InferenceFailureClass,
        completed_at_tick: u64,
        wire_evidence: Option<InferenceWireEvidence>,
    ) -> Result<Self, InferenceReceiptError> {
        validate_completion_time(&prepared, completed_at_tick)?;
        Ok(Self {
            binding: *prepared.binding(),
            permit_generation: prepared.generation(),
            prepared_at_tick: prepared.prepared_at_tick(),
            completed_at_tick,
            completion: InferenceCompletion::Failure { class },
            wire_evidence,
        })
    }

    pub const fn binding(&self) -> &InferenceExecutionBinding {
        &self.binding
    }

    pub const fn permit_generation(&self) -> u64 {
        self.permit_generation
    }

    pub const fn prepared_at_tick(&self) -> u64 {
        self.prepared_at_tick
    }

    pub const fn completed_at_tick(&self) -> u64 {
        self.completed_at_tick
    }

    pub const fn completion(&self) -> &InferenceCompletion {
        &self.completion
    }

    pub const fn response_digest(&self) -> Option<&BindingDigest> {
        match &self.completion {
            InferenceCompletion::Success { response_digest } => Some(response_digest),
            InferenceCompletion::Failure { .. } => None,
        }
    }

    pub fn wire_evidence(&self) -> Option<&InferenceWireEvidence> {
        self.wire_evidence.as_ref()
    }
}

fn validate_completion_time(
    prepared: &PreparedInferenceExecution,
    completed_at_tick: u64,
) -> Result<(), InferenceReceiptError> {
    if completed_at_tick < prepared.prepared_at_tick() {
        return Err(InferenceReceiptError::CompletionBeforePreparation);
    }
    Ok(())
}

fn bounded_metadata(value: Option<&str>) -> Result<Option<String>, InferenceReceiptError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_empty()
        || value.len() > MAX_RECEIPT_METADATA_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(InferenceReceiptError::InvalidWireMetadata);
    }
    Ok(Some(value.to_string()))
}

fn digest_wire_identifier(
    domain: &[u8],
    value: &str,
) -> Result<BindingDigest, InferenceReceiptError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(WIRE_ID_ROOT_DOMAIN.len() as u64).to_le_bytes());
    hasher.update(WIRE_ID_ROOT_DOMAIN);
    hasher.update(&(domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
    BindingDigest::new(*hasher.finalize().as_bytes())
        .map_err(InferenceReceiptError::WireIdentifierDigest)
}

#[derive(Debug)]
pub enum InferenceReceiptError {
    Binding(InferenceBindingError),
    WireIdentifierDigest(InferencePermitError),
    InvalidWireMetadata,
    CompletionBeforePreparation,
}

impl fmt::Display for InferenceReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binding(error) => write!(f, "failed to bind inference receipt: {error}"),
            Self::WireIdentifierDigest(error) => {
                write!(f, "failed to digest provider wire identifier: {error}")
            }
            Self::InvalidWireMetadata => write!(f, "provider wire metadata is invalid"),
            Self::CompletionBeforePreparation => {
                write!(f, "receipt completion tick precedes execution preparation")
            }
        }
    }
}

impl std::error::Error for InferenceReceiptError {}
