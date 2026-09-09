// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IF-3 inference execution receipts.
//!
//! Receipts are evidence, not authority. Successful receipts retain only a
//! domain-separated digest of model output, never the raw response text.

#[cfg(not(test))]
use super::inference_binding::{InferenceBindingError, digest_response_text};
#[cfg(not(test))]
use super::inference_permit::{BindingDigest, InferenceExecutionBinding, PreparedInferenceExecution};

#[cfg(test)]
use crate::inference_binding::{InferenceBindingError, digest_response_text};
#[cfg(test)]
use crate::inference_permit::{BindingDigest, InferenceExecutionBinding, PreparedInferenceExecution};

use std::fmt;

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

/// Non-authoritative evidence that a prepared inference attempt reached a terminal state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceReceipt {
    binding: InferenceExecutionBinding,
    permit_generation: u64,
    prepared_at_tick: u64,
    completed_at_tick: u64,
    completion: InferenceCompletion,
}

impl InferenceReceipt {
    /// Consume the prepared authority object and produce success evidence.
    /// Raw response text is hashed and immediately discarded by this API.
    pub fn success(
        prepared: PreparedInferenceExecution,
        response_text: &str,
        completed_at_tick: u64,
    ) -> Result<Self, InferenceReceiptError> {
        validate_completion_time(&prepared, completed_at_tick)?;
        let response_digest = digest_response_text(response_text)
            .map_err(InferenceReceiptError::Binding)?;
        Ok(Self {
            binding: *prepared.binding(),
            permit_generation: prepared.generation(),
            prepared_at_tick: prepared.prepared_at_tick(),
            completed_at_tick,
            completion: InferenceCompletion::Success { response_digest },
        })
    }

    /// Consume the prepared authority object and produce failure evidence.
    pub fn failure(
        prepared: PreparedInferenceExecution,
        class: InferenceFailureClass,
        completed_at_tick: u64,
    ) -> Result<Self, InferenceReceiptError> {
        validate_completion_time(&prepared, completed_at_tick)?;
        Ok(Self {
            binding: *prepared.binding(),
            permit_generation: prepared.generation(),
            prepared_at_tick: prepared.prepared_at_tick(),
            completed_at_tick,
            completion: InferenceCompletion::Failure { class },
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

#[derive(Debug)]
pub enum InferenceReceiptError {
    Binding(InferenceBindingError),
    CompletionBeforePreparation,
}

impl fmt::Display for InferenceReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binding(error) => write!(f, "failed to bind inference receipt: {error}"),
            Self::CompletionBeforePreparation => {
                write!(f, "receipt completion tick precedes execution preparation")
            }
        }
    }
}

impl std::error::Error for InferenceReceiptError {}
