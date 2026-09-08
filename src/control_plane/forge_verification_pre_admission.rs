// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Non-authoritative Forge verification pre-admission binding.
//!
//! This module joins the exact current trusted compilation chain to the exact
//! verification request and resource scope without granting execution authority.
//!
//! Forge v1 defines `ForgeVerificationRequestV1::artifact` as the exact source
//! Wasm artifact identity that was compiled by the authenticated compilation
//! receipt. The precompiled output remains a distinct identity retained by the
//! current compilation-attestation witness.

use super::forge::{
    ForgeCapabilityScopeV1, ForgeExecutionModeV1, ForgeVerificationRequestV1,
};
use super::forge_compilation::ForgeCompilationReceiptV1;
use super::forge_current_compilation_attestation::CurrentForgeCompilationAttestationV1;
use super::forge_profile::ForgeExecutionProfileDefinitionV1;
use std::error::Error;
use std::fmt;

/// Private in-process witness that exact trusted compilation identity, exact
/// verification intent, exact execution profile and exact non-authoritative
/// capability scope all refer to the same candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgeVerificationPreAdmissionV1 {
    current_compilation: CurrentForgeCompilationAttestationV1,
    request: ForgeVerificationRequestV1,
    scope: ForgeCapabilityScopeV1,
    mode: ForgeExecutionModeV1,
    decision_time_unix_s: u64,
    request_commitment: [u8; 32],
    scope_commitment: [u8; 32],
}

impl ForgeVerificationPreAdmissionV1 {
    pub fn current_compilation(&self) -> &CurrentForgeCompilationAttestationV1 {
        &self.current_compilation
    }

    pub fn request(&self) -> &ForgeVerificationRequestV1 {
        &self.request
    }

    pub fn scope(&self) -> &ForgeCapabilityScopeV1 {
        &self.scope
    }

    pub fn mode(&self) -> ForgeExecutionModeV1 {
        self.mode
    }

    pub fn decision_time_unix_s(&self) -> u64 {
        self.decision_time_unix_s
    }

    pub fn request_commitment(&self) -> [u8; 32] {
        self.request_commitment
    }

    pub fn scope_commitment(&self) -> [u8; 32] {
        self.scope_commitment
    }
}

/// Bind one exact current trusted compilation to one exact Forge verification
/// request and one exact non-authoritative capability scope.
pub fn bind_forge_verification_pre_admission_v1(
    current_compilation: &CurrentForgeCompilationAttestationV1,
    receipt: &ForgeCompilationReceiptV1,
    execution_profile: &ForgeExecutionProfileDefinitionV1,
    request: &ForgeVerificationRequestV1,
    scope: &ForgeCapabilityScopeV1,
    mode: ForgeExecutionModeV1,
    decision_time_unix_s: u64,
) -> Result<ForgeVerificationPreAdmissionV1, ForgeVerificationPreAdmissionErrorV1> {
    if decision_time_unix_s != current_compilation.point_of_use_unix_s() {
        return Err(ForgeVerificationPreAdmissionErrorV1::DecisionTimeMismatch {
            current_compilation_at_unix_s: current_compilation.point_of_use_unix_s(),
            requested_at_unix_s: decision_time_unix_s,
        });
    }

    receipt
        .validate()
        .map_err(|error| ForgeVerificationPreAdmissionErrorV1::CompilationReceipt(error.to_string()))?;
    let receipt_commitment = receipt
        .canonical_commitment_v1()
        .map_err(|error| ForgeVerificationPreAdmissionErrorV1::CompilationReceipt(error.to_string()))?;
    if receipt_commitment != current_compilation.compilation_receipt_commitment() {
        return Err(ForgeVerificationPreAdmissionErrorV1::CompilationReceiptMismatch);
    }

    request
        .validate()
        .map_err(|error| ForgeVerificationPreAdmissionErrorV1::VerificationRequest(error.to_string()))?;
    if request.artifact != receipt.request.source_artifact {
        return Err(ForgeVerificationPreAdmissionErrorV1::SourceArtifactMismatch);
    }

    execution_profile
        .validate()
        .map_err(|error| ForgeVerificationPreAdmissionErrorV1::ExecutionProfile(error.to_string()))?;
    let execution_profile_commitment = execution_profile
        .canonical_commitment_v1()
        .map_err(|error| ForgeVerificationPreAdmissionErrorV1::ExecutionProfile(error.to_string()))?;
    if execution_profile_commitment != current_compilation.execution_profile_commitment() {
        return Err(ForgeVerificationPreAdmissionErrorV1::ExecutionProfileMismatch);
    }
    if request.execution_profile.profile_id != execution_profile.profile_id
        || request.execution_profile.profile_commitment != execution_profile_commitment
    {
        return Err(ForgeVerificationPreAdmissionErrorV1::RequestedExecutionProfileMismatch);
    }

    let scope_matches = scope
        .matches_request(request, mode)
        .map_err(|error| ForgeVerificationPreAdmissionErrorV1::CapabilityScope(error.to_string()))?;
    if !scope_matches {
        return Err(ForgeVerificationPreAdmissionErrorV1::CapabilityScopeMismatch);
    }

    let request_commitment = request
        .canonical_commitment_v1()
        .map_err(|error| ForgeVerificationPreAdmissionErrorV1::VerificationRequest(error.to_string()))?;
    let scope_commitment = scope
        .canonical_commitment_v1()
        .map_err(|error| ForgeVerificationPreAdmissionErrorV1::CapabilityScope(error.to_string()))?;

    Ok(ForgeVerificationPreAdmissionV1 {
        current_compilation: current_compilation.clone(),
        request: request.clone(),
        scope: scope.clone(),
        mode,
        decision_time_unix_s,
        request_commitment,
        scope_commitment,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeVerificationPreAdmissionErrorV1 {
    DecisionTimeMismatch {
        current_compilation_at_unix_s: u64,
        requested_at_unix_s: u64,
    },
    CompilationReceipt(String),
    CompilationReceiptMismatch,
    VerificationRequest(String),
    SourceArtifactMismatch,
    ExecutionProfile(String),
    ExecutionProfileMismatch,
    RequestedExecutionProfileMismatch,
    CapabilityScope(String),
    CapabilityScopeMismatch,
}

impl fmt::Display for ForgeVerificationPreAdmissionErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DecisionTimeMismatch {
                current_compilation_at_unix_s,
                requested_at_unix_s,
            } => write!(
                f,
                "Forge pre-admission time {requested_at_unix_s} differs from current compilation-attestation time {current_compilation_at_unix_s}"
            ),
            Self::CompilationReceipt(error) => write!(f, "invalid Forge compilation receipt: {error}"),
            Self::CompilationReceiptMismatch => write!(f, "Forge compilation receipt does not match the authenticated current receipt"),
            Self::VerificationRequest(error) => write!(f, "invalid Forge verification request: {error}"),
            Self::SourceArtifactMismatch => write!(f, "Forge verification request source artifact does not match the authenticated compilation request"),
            Self::ExecutionProfile(error) => write!(f, "invalid Forge execution profile: {error}"),
            Self::ExecutionProfileMismatch => write!(f, "Forge execution profile does not match the profile retained by current compiled-artifact preflight"),
            Self::RequestedExecutionProfileMismatch => write!(f, "Forge verification request does not name the exact selected execution-profile definition"),
            Self::CapabilityScope(error) => write!(f, "invalid Forge capability scope: {error}"),
            Self::CapabilityScopeMismatch => write!(f, "Forge capability scope does not exactly match the verification request and requested execution mode"),
        }
    }
}

impl Error for ForgeVerificationPreAdmissionErrorV1 {}

#[cfg(test)]
mod tests;
