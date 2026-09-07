// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed, representation-independent Forge compilation request/receipt content.
//!
//! These objects describe what an isolated compilation worker was asked to
//! compile and what exact compiled bytes it produced. They are evidence content,
//! not trust by themselves: there is no issuer, signature, authority generation,
//! expiry, revocation, or proof that a receipt came from an approved worker.

use super::forge::{ForgeArtifactIdentityV1, ForgeDigestAlgorithmV1, FORGE_PROTOCOL_VERSION};
use super::forge_profile::ForgeExecutionProfileDefinitionV1;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

pub const MAX_FORGE_COMPILATION_PROFILE_ID_BYTES_V1: usize = 128;
pub const FORGE_COMPILATION_REQUEST_DOMAIN_V1: &[u8] =
    b"symthaea.forge.compilation-request.v1\0";
pub const FORGE_COMPILATION_RECEIPT_DOMAIN_V1: &[u8] =
    b"symthaea.forge.compilation-receipt.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeCompilationProfileIdentityV1 {
    pub profile_id: String,
    pub profile_commitment: [u8; 32],
}

impl ForgeCompilationProfileIdentityV1 {
    pub fn new(
        profile_id: impl Into<String>,
        profile_commitment: [u8; 32],
    ) -> Result<Self, ForgeCompilationProtocolError> {
        let value = Self { profile_id: profile_id.into(), profile_commitment };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ForgeCompilationProtocolError> {
        validate_bounded_nonempty(
            "compilation_profile.profile_id",
            &self.profile_id,
            MAX_FORGE_COMPILATION_PROFILE_ID_BYTES_V1,
        )?;
        if self.profile_commitment == [0; 32] {
            return Err(ForgeCompilationProtocolError::ZeroCommitment(
                "compilation_profile.profile_commitment",
            ));
        }
        Ok(())
    }
}

/// Content identity of exact Wasmtime precompiled bytes.
///
/// This is a distinct type from source-Wasm identity so source and compiled
/// bytes cannot be substituted merely because both are byte strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeCompiledArtifactIdentityV1 {
    pub digest_algorithm: ForgeDigestAlgorithmV1,
    pub digest: [u8; 32],
    pub byte_len: u64,
}

impl ForgeCompiledArtifactIdentityV1 {
    pub fn new_blake3_256(
        digest: [u8; 32],
        byte_len: u64,
    ) -> Result<Self, ForgeCompilationProtocolError> {
        if byte_len == 0 {
            return Err(ForgeCompilationProtocolError::EmptyCompiledArtifact);
        }
        Ok(Self { digest_algorithm: ForgeDigestAlgorithmV1::Blake3_256, digest, byte_len })
    }

    pub fn validate(&self) -> Result<(), ForgeCompilationProtocolError> {
        if self.byte_len == 0 {
            return Err(ForgeCompilationProtocolError::EmptyCompiledArtifact);
        }
        Ok(())
    }
}

/// Non-authoritative request sent to an isolated compilation plane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeCompilationRequestV1 {
    pub protocol_version: u16,
    pub source_artifact: ForgeArtifactIdentityV1,
    pub compilation_profile: ForgeCompilationProfileIdentityV1,
    pub target_runtime_lineage_commitment: [u8; 32],
    pub wasm_feature_policy_commitment: [u8; 32],
}

impl ForgeCompilationRequestV1 {
    pub fn new(
        source_artifact: ForgeArtifactIdentityV1,
        compilation_profile: ForgeCompilationProfileIdentityV1,
        target_runtime_lineage_commitment: [u8; 32],
        wasm_feature_policy_commitment: [u8; 32],
    ) -> Result<Self, ForgeCompilationProtocolError> {
        let request = Self {
            protocol_version: FORGE_PROTOCOL_VERSION,
            source_artifact,
            compilation_profile,
            target_runtime_lineage_commitment,
            wasm_feature_policy_commitment,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), ForgeCompilationProtocolError> {
        if self.protocol_version != FORGE_PROTOCOL_VERSION {
            return Err(ForgeCompilationProtocolError::UnsupportedProtocolVersion(self.protocol_version));
        }
        self.source_artifact
            .validate()
            .map_err(|error| ForgeCompilationProtocolError::SourceArtifact(error.to_string()))?;
        self.compilation_profile.validate()?;
        if self.target_runtime_lineage_commitment == [0; 32] {
            return Err(ForgeCompilationProtocolError::ZeroCommitment("target_runtime_lineage_commitment"));
        }
        if self.wasm_feature_policy_commitment == [0; 32] {
            return Err(ForgeCompilationProtocolError::ZeroCommitment("wasm_feature_policy_commitment"));
        }
        Ok(())
    }

    pub fn canonical_bytes_v1(&self) -> Result<Vec<u8>, ForgeCompilationProtocolError> {
        self.validate()?;
        let mut out = Vec::with_capacity(
            FORGE_COMPILATION_REQUEST_DOMAIN_V1.len()
                + 2 + 1 + 32 + 8 + 2 + MAX_FORGE_COMPILATION_PROFILE_ID_BYTES_V1 + 32 + 32 + 32,
        );
        out.extend_from_slice(FORGE_COMPILATION_REQUEST_DOMAIN_V1);
        out.extend_from_slice(&self.protocol_version.to_be_bytes());
        append_source_artifact(&mut out, &self.source_artifact);
        append_bounded_utf8(
            &mut out,
            "compilation_profile.profile_id",
            &self.compilation_profile.profile_id,
            MAX_FORGE_COMPILATION_PROFILE_ID_BYTES_V1,
        )?;
        out.extend_from_slice(&self.compilation_profile.profile_commitment);
        out.extend_from_slice(&self.target_runtime_lineage_commitment);
        out.extend_from_slice(&self.wasm_feature_policy_commitment);
        Ok(out)
    }

    pub fn canonical_commitment_v1(&self) -> Result<[u8; 32], ForgeCompilationProtocolError> {
        Ok(*blake3::hash(&self.canonical_bytes_v1()?).as_bytes())
    }

    /// Match only compilation-relevant execution identity.
    /// Fuel/memory execution limits are intentionally not compilation identity.
    pub fn matches_execution_profile(
        &self,
        execution_profile: &ForgeExecutionProfileDefinitionV1,
    ) -> Result<bool, ForgeCompilationProtocolError> {
        self.validate()?;
        execution_profile
            .validate()
            .map_err(|error| ForgeCompilationProtocolError::ExecutionProfile(error.to_string()))?;
        Ok(self.target_runtime_lineage_commitment == execution_profile.runtime.lineage_commitment
            && self.wasm_feature_policy_commitment == execution_profile.wasm_feature_policy_commitment)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeCompilationFailureClassV1 {
    DecodeOrValidationRejected,
    FeaturePolicyRejected,
    ResourceBoundaryExceeded,
    CompilerRejected,
    OutputRejected,
    WorkerInternalFailure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForgeCompilationOutcomeV1 {
    Succeeded { compiled_artifact: ForgeCompiledArtifactIdentityV1 },
    Rejected { failure_class: ForgeCompilationFailureClassV1 },
}

/// Evidence content emitted by an isolated compilation worker.
///
/// A raw receipt is not trusted merely because it validates structurally. The
/// future unsafe loader must require a separately authenticated/current receipt
/// wrapper or equivalent provider-verified authority before accepting bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeCompilationReceiptV1 {
    pub request: ForgeCompilationRequestV1,
    pub outcome: ForgeCompilationOutcomeV1,
}

impl ForgeCompilationReceiptV1 {
    pub fn succeeded(
        request: ForgeCompilationRequestV1,
        compiled_artifact: ForgeCompiledArtifactIdentityV1,
    ) -> Result<Self, ForgeCompilationProtocolError> {
        let receipt = Self { request, outcome: ForgeCompilationOutcomeV1::Succeeded { compiled_artifact } };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn rejected(
        request: ForgeCompilationRequestV1,
        failure_class: ForgeCompilationFailureClassV1,
    ) -> Result<Self, ForgeCompilationProtocolError> {
        let receipt = Self { request, outcome: ForgeCompilationOutcomeV1::Rejected { failure_class } };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> Result<(), ForgeCompilationProtocolError> {
        self.request.validate()?;
        if let ForgeCompilationOutcomeV1::Succeeded { compiled_artifact } = &self.outcome {
            compiled_artifact.validate()?;
        }
        Ok(())
    }

    pub fn compiled_artifact(&self) -> Option<&ForgeCompiledArtifactIdentityV1> {
        match &self.outcome {
            ForgeCompilationOutcomeV1::Succeeded { compiled_artifact } => Some(compiled_artifact),
            ForgeCompilationOutcomeV1::Rejected { .. } => None,
        }
    }

    pub fn canonical_bytes_v1(&self) -> Result<Vec<u8>, ForgeCompilationProtocolError> {
        self.validate()?;
        let request_commitment = self.request.canonical_commitment_v1()?;
        let mut out = Vec::with_capacity(FORGE_COMPILATION_RECEIPT_DOMAIN_V1.len() + 32 + 1 + 41);
        out.extend_from_slice(FORGE_COMPILATION_RECEIPT_DOMAIN_V1);
        out.extend_from_slice(&request_commitment);
        match &self.outcome {
            ForgeCompilationOutcomeV1::Succeeded { compiled_artifact } => {
                out.push(1);
                append_compiled_artifact(&mut out, compiled_artifact);
            }
            ForgeCompilationOutcomeV1::Rejected { failure_class } => {
                out.push(2);
                out.extend_from_slice(&failure_class_tag(*failure_class).to_be_bytes());
            }
        }
        Ok(out)
    }

    pub fn canonical_commitment_v1(&self) -> Result<[u8; 32], ForgeCompilationProtocolError> {
        Ok(*blake3::hash(&self.canonical_bytes_v1()?).as_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeCompilationProtocolError {
    UnsupportedProtocolVersion(u16),
    EmptyField(&'static str),
    FieldTooLong { field: &'static str, max_bytes: usize, actual_bytes: usize },
    ZeroCommitment(&'static str),
    EmptyCompiledArtifact,
    SourceArtifact(String),
    ExecutionProfile(String),
}

impl fmt::Display for ForgeCompilationProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedProtocolVersion(version) => write!(f, "unsupported Forge compilation protocol version {version}"),
            Self::EmptyField(field) => write!(f, "Forge compilation field {field} must be non-empty"),
            Self::FieldTooLong { field, max_bytes, actual_bytes } => write!(f, "Forge compilation field {field} is {actual_bytes} bytes; v1 maximum is {max_bytes} bytes"),
            Self::ZeroCommitment(field) => write!(f, "Forge compilation commitment field {field} may not be all-zero"),
            Self::EmptyCompiledArtifact => write!(f, "compiled Forge artifact must contain at least one byte"),
            Self::SourceArtifact(error) => write!(f, "invalid source artifact identity: {error}"),
            Self::ExecutionProfile(error) => write!(f, "invalid execution profile: {error}"),
        }
    }
}

impl Error for ForgeCompilationProtocolError {}

fn validate_bounded_nonempty(field: &'static str, value: &str, max_bytes: usize) -> Result<(), ForgeCompilationProtocolError> {
    if value.trim().is_empty() {
        return Err(ForgeCompilationProtocolError::EmptyField(field));
    }
    if value.len() > max_bytes {
        return Err(ForgeCompilationProtocolError::FieldTooLong { field, max_bytes, actual_bytes: value.len() });
    }
    Ok(())
}

fn append_bounded_utf8(out: &mut Vec<u8>, field: &'static str, value: &str, max_bytes: usize) -> Result<(), ForgeCompilationProtocolError> {
    validate_bounded_nonempty(field, value, max_bytes)?;
    let len = u16::try_from(value.len()).map_err(|_| ForgeCompilationProtocolError::FieldTooLong {
        field,
        max_bytes: max_bytes.min(u16::MAX as usize),
        actual_bytes: value.len(),
    })?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn digest_algorithm_tag(value: ForgeDigestAlgorithmV1) -> u8 {
    match value { ForgeDigestAlgorithmV1::Blake3_256 => 1 }
}

fn append_source_artifact(out: &mut Vec<u8>, value: &ForgeArtifactIdentityV1) {
    out.push(digest_algorithm_tag(value.digest_algorithm));
    out.extend_from_slice(&value.digest);
    out.extend_from_slice(&value.byte_len.to_be_bytes());
}

fn append_compiled_artifact(out: &mut Vec<u8>, value: &ForgeCompiledArtifactIdentityV1) {
    out.push(digest_algorithm_tag(value.digest_algorithm));
    out.extend_from_slice(&value.digest);
    out.extend_from_slice(&value.byte_len.to_be_bytes());
}

const fn failure_class_tag(value: ForgeCompilationFailureClassV1) -> u16 {
    match value {
        ForgeCompilationFailureClassV1::DecodeOrValidationRejected => 1,
        ForgeCompilationFailureClassV1::FeaturePolicyRejected => 2,
        ForgeCompilationFailureClassV1::ResourceBoundaryExceeded => 3,
        ForgeCompilationFailureClassV1::CompilerRejected => 4,
        ForgeCompilationFailureClassV1::OutputRejected => 5,
        ForgeCompilationFailureClassV1::WorkerInternalFailure => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_plane::forge_profile::{
        ForgeDeterminismPolicyV1, ForgeExecutionProfileDefinitionV1, ForgeImportPolicyV1,
        ForgeInterruptionPolicyV1, ForgeRuntimeIdentityV1, ForgeVerifierAbiV1,
    };

    fn source() -> ForgeArtifactIdentityV1 {
        ForgeArtifactIdentityV1::new_blake3_256([0x11; 32], 100).unwrap()
    }
    fn compilation_profile() -> ForgeCompilationProfileIdentityV1 {
        ForgeCompilationProfileIdentityV1::new("compile-v1", [0x22; 32]).unwrap()
    }
    fn request() -> ForgeCompilationRequestV1 {
        ForgeCompilationRequestV1::new(source(), compilation_profile(), [0x33; 32], [0x44; 32]).unwrap()
    }
    fn compiled() -> ForgeCompiledArtifactIdentityV1 {
        ForgeCompiledArtifactIdentityV1::new_blake3_256([0x55; 32], 200).unwrap()
    }
    fn execution_profile() -> ForgeExecutionProfileDefinitionV1 {
        ForgeExecutionProfileDefinitionV1 {
            protocol_version: FORGE_PROTOCOL_VERSION,
            profile_id: "run-v1".into(),
            runtime: ForgeRuntimeIdentityV1::new_wasmtime("44.0.1", [0x33; 32]).unwrap(),
            wasm_feature_policy_commitment: [0x44; 32],
            abi: ForgeVerifierAbiV1::NoArgsI32,
            imports: ForgeImportPolicyV1::NoImports,
            determinism: ForgeDeterminismPolicyV1::Strict,
            interruption: ForgeInterruptionPolicyV1::DeterministicFuel,
            fuel: 1,
            max_precompiled_artifact_bytes: 1,
            max_linear_memory_bytes: 0,
            max_memories: 0,
            max_table_elements: 0,
            max_tables: 0,
            max_instances: 1,
            max_wasm_stack_bytes: 1,
            trap_on_grow_failure: true,
        }
    }

    #[test]
    fn compilation_profile_identity_rejects_placeholders() {
        assert!(ForgeCompilationProfileIdentityV1::new("", [1; 32]).is_err());
        assert!(ForgeCompilationProfileIdentityV1::new("x", [0; 32]).is_err());
    }

    #[test]
    fn request_requires_runtime_and_feature_commitments() {
        assert!(ForgeCompilationRequestV1::new(source(), compilation_profile(), [0; 32], [1; 32]).is_err());
        assert!(ForgeCompilationRequestV1::new(source(), compilation_profile(), [1; 32], [0; 32]).is_err());
    }

    #[test]
    fn request_commitment_binds_every_compilation_identity_dimension() {
        let base = request();
        let commitment = base.canonical_commitment_v1().unwrap();
        let mut changed = base.clone();
        changed.source_artifact.digest[0] ^= 1;
        assert_ne!(changed.canonical_commitment_v1().unwrap(), commitment);
        let mut changed = base.clone();
        changed.compilation_profile.profile_commitment[0] ^= 1;
        assert_ne!(changed.canonical_commitment_v1().unwrap(), commitment);
        let mut changed = base.clone();
        changed.target_runtime_lineage_commitment[0] ^= 1;
        assert_ne!(changed.canonical_commitment_v1().unwrap(), commitment);
        let mut changed = base;
        changed.wasm_feature_policy_commitment[0] ^= 1;
        assert_ne!(changed.canonical_commitment_v1().unwrap(), commitment);
    }

    #[test]
    fn execution_profile_compatibility_is_exact_on_runtime_and_features() {
        let req = request();
        let profile = execution_profile();
        assert!(req.matches_execution_profile(&profile).unwrap());
        let mut different_runtime = profile.clone();
        different_runtime.runtime.lineage_commitment[0] ^= 1;
        assert!(!req.matches_execution_profile(&different_runtime).unwrap());
        let mut different_features = profile;
        different_features.wasm_feature_policy_commitment[0] ^= 1;
        assert!(!req.matches_execution_profile(&different_features).unwrap());
    }

    #[test]
    fn successful_receipt_binds_exact_compiled_artifact() {
        let a = ForgeCompilationReceiptV1::succeeded(request(), compiled()).unwrap();
        let mut different_artifact = compiled();
        different_artifact.digest[0] ^= 1;
        let b = ForgeCompilationReceiptV1::succeeded(request(), different_artifact).unwrap();
        assert_eq!(a.compiled_artifact(), Some(&compiled()));
        assert_ne!(a.canonical_commitment_v1().unwrap(), b.canonical_commitment_v1().unwrap());
    }

    #[test]
    fn source_substitution_changes_receipt_identity() {
        let a = ForgeCompilationReceiptV1::succeeded(request(), compiled()).unwrap();
        let mut req_b = request();
        req_b.source_artifact.digest[0] ^= 1;
        let b = ForgeCompilationReceiptV1::succeeded(req_b, compiled()).unwrap();
        assert_ne!(a.canonical_commitment_v1().unwrap(), b.canonical_commitment_v1().unwrap());
    }

    #[test]
    fn rejection_carries_no_compiled_artifact_and_is_identity_bearing() {
        let a = ForgeCompilationReceiptV1::rejected(request(), ForgeCompilationFailureClassV1::FeaturePolicyRejected).unwrap();
        let b = ForgeCompilationReceiptV1::rejected(request(), ForgeCompilationFailureClassV1::ResourceBoundaryExceeded).unwrap();
        assert!(a.compiled_artifact().is_none());
        assert_ne!(a.canonical_commitment_v1().unwrap(), b.canonical_commitment_v1().unwrap());
    }
}
