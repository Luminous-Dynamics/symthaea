// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Structural preflight for trusted-compiled Forge artifacts.
//!
//! This module joins compilation evidence to an execution-profile identity and
//! exact candidate bytes without assigning trust or crossing Wasmtime's unsafe
//! precompiled-module deserialization boundary.
//!
//! A successful [`ForgeCompiledArtifactPreflightV1`] means only:
//!
//! - the compilation receipt is structurally valid and successful;
//! - its requested runtime lineage and Wasm-feature policy match the selected
//!   execution profile;
//! - the compiled-artifact identity fits the execution profile's byte ceiling;
//! - the exact candidate bytes match the receipt's compiled digest and length.
//!
//! It does **not** prove that the receipt came from an approved/current worker,
//! that execution is authorized, or that the bytes are safe to deserialize.

use super::forge::ForgeDigestAlgorithmV1;
use super::forge_compilation::{
    ForgeCompilationReceiptV1, ForgeCompiledArtifactIdentityV1,
};
use super::forge_profile::ForgeExecutionProfileDefinitionV1;
use std::error::Error;
use std::fmt;

/// Non-serializable structural join between exact compiled bytes, one successful
/// compilation receipt, and one exact execution profile.
///
/// Fields are private and this type intentionally has no serde derives. It is an
/// in-process result of revalidation, not a portable authority/proof token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgeCompiledArtifactPreflightV1 {
    compiled_artifact: ForgeCompiledArtifactIdentityV1,
    compilation_receipt_commitment: [u8; 32],
    execution_profile_commitment: [u8; 32],
}

impl ForgeCompiledArtifactPreflightV1 {
    /// Revalidate exact compilation/effect identity before any later trusted
    /// deserialization gate.
    pub fn verify(
        receipt: &ForgeCompilationReceiptV1,
        execution_profile: &ForgeExecutionProfileDefinitionV1,
        candidate_compiled_bytes: &[u8],
    ) -> Result<Self, ForgeCompiledArtifactPreflightError> {
        receipt
            .validate()
            .map_err(|error| ForgeCompiledArtifactPreflightError::CompilationReceipt(
                error.to_string(),
            ))?;
        execution_profile
            .validate()
            .map_err(|error| ForgeCompiledArtifactPreflightError::ExecutionProfile(
                error.to_string(),
            ))?;

        let compiled_artifact = receipt
            .compiled_artifact()
            .ok_or(ForgeCompiledArtifactPreflightError::CompilationRejected)?;

        if receipt.request.target_runtime_lineage_commitment
            != execution_profile.runtime.lineage_commitment
        {
            return Err(ForgeCompiledArtifactPreflightError::RuntimeLineageMismatch);
        }
        if receipt.request.wasm_feature_policy_commitment
            != execution_profile.wasm_feature_policy_commitment
        {
            return Err(ForgeCompiledArtifactPreflightError::WasmFeaturePolicyMismatch);
        }

        let execution_limit = u64::from(execution_profile.max_precompiled_artifact_bytes);
        if compiled_artifact.byte_len > execution_limit {
            return Err(
                ForgeCompiledArtifactPreflightError::ReceiptArtifactExceedsExecutionLimit {
                    receipt_bytes: compiled_artifact.byte_len,
                    execution_limit_bytes: execution_limit,
                },
            );
        }

        let candidate_len = u64::try_from(candidate_compiled_bytes.len()).map_err(|_| {
            ForgeCompiledArtifactPreflightError::CandidateLengthNotRepresentable
        })?;
        if candidate_len > execution_limit {
            return Err(
                ForgeCompiledArtifactPreflightError::CandidateExceedsExecutionLimit {
                    candidate_bytes: candidate_len,
                    execution_limit_bytes: execution_limit,
                },
            );
        }
        if candidate_len != compiled_artifact.byte_len {
            return Err(ForgeCompiledArtifactPreflightError::CandidateLengthMismatch {
                expected_bytes: compiled_artifact.byte_len,
                actual_bytes: candidate_len,
            });
        }

        let candidate_digest = match compiled_artifact.digest_algorithm {
            ForgeDigestAlgorithmV1::Blake3_256 => *blake3::hash(candidate_compiled_bytes).as_bytes(),
        };
        if candidate_digest != compiled_artifact.digest {
            return Err(ForgeCompiledArtifactPreflightError::CandidateDigestMismatch);
        }

        let compilation_receipt_commitment = receipt
            .canonical_commitment_v1()
            .map_err(|error| ForgeCompiledArtifactPreflightError::CompilationReceipt(
                error.to_string(),
            ))?;
        let execution_profile_commitment = execution_profile
            .canonical_commitment_v1()
            .map_err(|error| ForgeCompiledArtifactPreflightError::ExecutionProfile(
                error.to_string(),
            ))?;

        Ok(Self {
            compiled_artifact: compiled_artifact.clone(),
            compilation_receipt_commitment,
            execution_profile_commitment,
        })
    }

    /// Exact compiled-artifact identity revalidated against candidate bytes.
    pub fn compiled_artifact(&self) -> &ForgeCompiledArtifactIdentityV1 {
        &self.compiled_artifact
    }

    /// Commitment to the exact compilation receipt used in this preflight.
    pub fn compilation_receipt_commitment(&self) -> [u8; 32] {
        self.compilation_receipt_commitment
    }

    /// Commitment to the exact execution profile used in this preflight.
    pub fn execution_profile_commitment(&self) -> [u8; 32] {
        self.execution_profile_commitment
    }
}

/// Fail-closed structural preflight failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeCompiledArtifactPreflightError {
    CompilationReceipt(String),
    ExecutionProfile(String),
    CompilationRejected,
    RuntimeLineageMismatch,
    WasmFeaturePolicyMismatch,
    ReceiptArtifactExceedsExecutionLimit {
        receipt_bytes: u64,
        execution_limit_bytes: u64,
    },
    CandidateLengthNotRepresentable,
    CandidateExceedsExecutionLimit {
        candidate_bytes: u64,
        execution_limit_bytes: u64,
    },
    CandidateLengthMismatch {
        expected_bytes: u64,
        actual_bytes: u64,
    },
    CandidateDigestMismatch,
}

impl fmt::Display for ForgeCompiledArtifactPreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompilationReceipt(error) => {
                write!(f, "invalid Forge compilation receipt: {error}")
            }
            Self::ExecutionProfile(error) => {
                write!(f, "invalid Forge execution profile: {error}")
            }
            Self::CompilationRejected => {
                write!(f, "Forge compilation receipt does not contain a successful artifact")
            }
            Self::RuntimeLineageMismatch => write!(
                f,
                "Forge compilation target runtime lineage does not match execution profile"
            ),
            Self::WasmFeaturePolicyMismatch => write!(
                f,
                "Forge compilation Wasm-feature policy does not match execution profile"
            ),
            Self::ReceiptArtifactExceedsExecutionLimit {
                receipt_bytes,
                execution_limit_bytes,
            } => write!(
                f,
                "Forge compiled receipt artifact is {receipt_bytes} bytes; execution profile maximum is {execution_limit_bytes} bytes"
            ),
            Self::CandidateLengthNotRepresentable => write!(
                f,
                "candidate compiled artifact length cannot be represented by Forge v1"
            ),
            Self::CandidateExceedsExecutionLimit {
                candidate_bytes,
                execution_limit_bytes,
            } => write!(
                f,
                "candidate compiled artifact is {candidate_bytes} bytes; execution profile maximum is {execution_limit_bytes} bytes"
            ),
            Self::CandidateLengthMismatch {
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "candidate compiled artifact length mismatch: receipt expects {expected_bytes} bytes, candidate has {actual_bytes} bytes"
            ),
            Self::CandidateDigestMismatch => {
                write!(f, "candidate compiled artifact digest does not match compilation receipt")
            }
        }
    }
}

impl Error for ForgeCompiledArtifactPreflightError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_plane::forge::{
        ForgeArtifactIdentityV1, FORGE_PROTOCOL_VERSION,
    };
    use crate::control_plane::forge_compilation::{
        ForgeCompilationFailureClassV1, ForgeCompilationProfileIdentityV1,
        ForgeCompilationRequestV1,
    };
    use crate::control_plane::forge_profile::{
        ForgeDeterminismPolicyV1, ForgeImportPolicyV1,
        ForgeInterruptionPolicyV1, ForgeRuntimeIdentityV1, ForgeVerifierAbiV1,
    };

    const RUNTIME_LINEAGE: [u8; 32] = [0x33; 32];
    const FEATURE_POLICY: [u8; 32] = [0x44; 32];
    const CANDIDATE: &[u8] = b"forge-precompiled-artifact-v1";

    fn request() -> ForgeCompilationRequestV1 {
        ForgeCompilationRequestV1::new(
            ForgeArtifactIdentityV1::new_blake3_256([0x11; 32], 100).unwrap(),
            ForgeCompilationProfileIdentityV1::new("compile-v1", [0x22; 32]).unwrap(),
            RUNTIME_LINEAGE,
            FEATURE_POLICY,
        )
        .unwrap()
    }

    fn successful_receipt(bytes: &[u8]) -> ForgeCompilationReceiptV1 {
        let identity = ForgeCompiledArtifactIdentityV1::new_blake3_256(
            *blake3::hash(bytes).as_bytes(),
            u64::try_from(bytes.len()).unwrap(),
        )
        .unwrap();
        ForgeCompilationReceiptV1::succeeded(request(), identity).unwrap()
    }

    fn execution_profile() -> ForgeExecutionProfileDefinitionV1 {
        ForgeExecutionProfileDefinitionV1 {
            protocol_version: FORGE_PROTOCOL_VERSION,
            profile_id: "run-v1".into(),
            runtime: ForgeRuntimeIdentityV1::new_wasmtime("44.0.1", RUNTIME_LINEAGE).unwrap(),
            wasm_feature_policy_commitment: FEATURE_POLICY,
            abi: ForgeVerifierAbiV1::NoArgsI32,
            imports: ForgeImportPolicyV1::NoImports,
            determinism: ForgeDeterminismPolicyV1::Strict,
            interruption: ForgeInterruptionPolicyV1::DeterministicFuel,
            fuel: 1_000,
            max_precompiled_artifact_bytes: 4_096,
            max_linear_memory_bytes: 0,
            max_memories: 0,
            max_table_elements: 0,
            max_tables: 0,
            max_instances: 1,
            max_wasm_stack_bytes: 64 * 1024,
            trap_on_grow_failure: true,
        }
    }

    #[test]
    fn exact_successful_receipt_and_candidate_produce_preflight() {
        let receipt = successful_receipt(CANDIDATE);
        let profile = execution_profile();
        let preflight = ForgeCompiledArtifactPreflightV1::verify(
            &receipt,
            &profile,
            CANDIDATE,
        )
        .unwrap();

        assert_eq!(preflight.compiled_artifact(), receipt.compiled_artifact().unwrap());
        assert_eq!(
            preflight.compilation_receipt_commitment(),
            receipt.canonical_commitment_v1().unwrap()
        );
        assert_eq!(
            preflight.execution_profile_commitment(),
            profile.canonical_commitment_v1().unwrap()
        );
    }

    #[test]
    fn rejected_compilation_receipt_never_preflights() {
        let receipt = ForgeCompilationReceiptV1::rejected(
            request(),
            ForgeCompilationFailureClassV1::CompilerRejected,
        )
        .unwrap();
        assert!(matches!(
            ForgeCompiledArtifactPreflightV1::verify(
                &receipt,
                &execution_profile(),
                CANDIDATE,
            ),
            Err(ForgeCompiledArtifactPreflightError::CompilationRejected)
        ));
    }

    #[test]
    fn runtime_or_feature_policy_substitution_fails_closed() {
        let receipt = successful_receipt(CANDIDATE);

        let mut runtime_drift = execution_profile();
        runtime_drift.runtime.lineage_commitment[0] ^= 0xFF;
        assert!(matches!(
            ForgeCompiledArtifactPreflightV1::verify(
                &receipt,
                &runtime_drift,
                CANDIDATE,
            ),
            Err(ForgeCompiledArtifactPreflightError::RuntimeLineageMismatch)
        ));

        let mut feature_drift = execution_profile();
        feature_drift.wasm_feature_policy_commitment[0] ^= 0xFF;
        assert!(matches!(
            ForgeCompiledArtifactPreflightV1::verify(
                &receipt,
                &feature_drift,
                CANDIDATE,
            ),
            Err(ForgeCompiledArtifactPreflightError::WasmFeaturePolicyMismatch)
        ));
    }

    #[test]
    fn candidate_length_and_digest_are_both_exact() {
        let receipt = successful_receipt(CANDIDATE);
        let profile = execution_profile();

        let shorter = &CANDIDATE[..CANDIDATE.len() - 1];
        assert!(matches!(
            ForgeCompiledArtifactPreflightV1::verify(&receipt, &profile, shorter),
            Err(ForgeCompiledArtifactPreflightError::CandidateLengthMismatch { .. })
        ));

        let mut same_len_different_bytes = CANDIDATE.to_vec();
        same_len_different_bytes[0] ^= 0x01;
        assert!(matches!(
            ForgeCompiledArtifactPreflightV1::verify(
                &receipt,
                &profile,
                &same_len_different_bytes,
            ),
            Err(ForgeCompiledArtifactPreflightError::CandidateDigestMismatch)
        ));
    }

    #[test]
    fn execution_profile_byte_ceiling_applies_to_receipt_and_candidate() {
        let receipt = successful_receipt(CANDIDATE);
        let mut profile = execution_profile();
        profile.max_precompiled_artifact_bytes = 1;

        assert!(matches!(
            ForgeCompiledArtifactPreflightV1::verify(&receipt, &profile, CANDIDATE),
            Err(
                ForgeCompiledArtifactPreflightError::ReceiptArtifactExceedsExecutionLimit { .. }
            )
        ));
    }

    #[test]
    fn execution_only_limit_changes_do_not_require_recompilation_but_change_preflight_identity() {
        let receipt = successful_receipt(CANDIDATE);
        let profile_a = execution_profile();
        let mut profile_b = execution_profile();
        profile_b.fuel += 1;

        let a = ForgeCompiledArtifactPreflightV1::verify(&receipt, &profile_a, CANDIDATE).unwrap();
        let b = ForgeCompiledArtifactPreflightV1::verify(&receipt, &profile_b, CANDIDATE).unwrap();

        assert_eq!(a.compiled_artifact(), b.compiled_artifact());
        assert_eq!(
            a.compilation_receipt_commitment(),
            b.compilation_receipt_commitment()
        );
        assert_ne!(
            a.execution_profile_commitment(),
            b.execution_profile_commitment()
        );
    }
}
