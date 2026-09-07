// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical Forge v1 compilation-profile definition.
//!
//! This module defines *what* a compilation profile means, not which numeric
//! process/resource values are safe. Operational limits must come from #684's
//! isolated compiler-worker qualification lineage.

use super::forge::FORGE_PROTOCOL_VERSION;
use super::forge_compilation::ForgeCompilationProfileIdentityV1;
use super::forge_profile::ForgeImportPolicyV1;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

pub const MAX_FORGE_COMPILATION_PROFILE_ID_BYTES_V1: usize = 128;
pub const MAX_FORGE_COMPILER_VERSION_BYTES_V1: usize = 64;
pub const MAX_FORGE_TARGET_TRIPLE_BYTES_V1: usize = 128;
pub const FORGE_COMPILATION_PROFILE_DOMAIN_V1: &[u8] =
    b"symthaea.forge.compilation-profile-definition.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeCompilationRuntimeFamilyV1 {
    Wasmtime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeCompilerBackendV1 {
    Cranelift,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeCompilationInputFormatV1 {
    /// Raw WebAssembly binary only; no WAT/text parsing.
    WasmBinary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeCompilationOutputFormatV1 {
    /// Wasmtime serialized/precompiled core module.
    WasmtimeSerializedModule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeOptimizationLevelV1 {
    None,
    Speed,
    SpeedAndSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeCompilationParallelismV1 {
    SingleThread,
    Parallel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeModuleVersionPolicyV1 {
    /// Preserve Wasmtime's exact package-version compatibility check.
    WasmtimeVersion,
}

/// Canonical definition of an isolated Forge compiler-worker profile.
///
/// Commitments here are identity only. They do not prove that the referenced
/// compiler worker, target CPU policy, or host isolation profile is trusted or
/// current.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeCompilationProfileDefinitionV1 {
    pub protocol_version: u16,
    pub profile_id: String,
    pub runtime_family: ForgeCompilationRuntimeFamilyV1,
    /// Exact Wasmtime compiler/runtime package version string.
    pub runtime_version: String,
    pub compiler_backend: ForgeCompilerBackendV1,
    /// Exact compiler/package/toolchain source lineage.
    pub compiler_lineage_commitment: [u8; 32],
    /// Explicit target triple. The worker must not silently use host inference.
    pub target_triple: String,
    /// Commitment to exact target CPU features/flags selected for codegen.
    pub target_cpu_policy_commitment: [u8; 32],
    /// Shared source-language feature policy from #692.
    pub wasm_feature_policy_commitment: [u8; 32],
    /// Commitment to remaining exact Wasmtime/Cranelift compiler settings not
    /// individually represented by this v1 schema.
    pub compiler_settings_commitment: [u8; 32],
    /// Commitment to external compiler-worker process/container/VM isolation.
    pub isolation_profile_commitment: [u8; 32],
    pub input_format: ForgeCompilationInputFormatV1,
    pub output_format: ForgeCompilationOutputFormatV1,
    pub optimization: ForgeOptimizationLevelV1,
    /// Require Cranelift NaN canonicalization for deterministic floating output.
    pub nan_canonicalization: bool,
    pub parallelism: ForgeCompilationParallelismV1,
    pub module_version_policy: ForgeModuleVersionPolicyV1,
    /// Source module import policy enforced before accepting compilation output.
    pub imports: ForgeImportPolicyV1,
    /// Whether DWARF/native debug information may be emitted into compiled output.
    pub debug_info: bool,
    /// Whether native-address to Wasm-address maps are emitted.
    pub generate_address_map: bool,
    /// Admission ceiling on raw source bytes. This is not a CPU/RSS safety proof.
    pub max_source_module_bytes: u32,
    /// Ceiling on serialized compiler output before receipt acceptance.
    pub max_compiled_artifact_bytes: u32,
}

impl ForgeCompilationProfileDefinitionV1 {
    pub fn validate(&self) -> Result<(), ForgeCompilationProfileError> {
        if self.protocol_version != FORGE_PROTOCOL_VERSION {
            return Err(ForgeCompilationProfileError::UnsupportedProtocolVersion(
                self.protocol_version,
            ));
        }
        validate_bounded_nonempty(
            "profile_id",
            &self.profile_id,
            MAX_FORGE_COMPILATION_PROFILE_ID_BYTES_V1,
        )?;
        validate_bounded_nonempty(
            "runtime_version",
            &self.runtime_version,
            MAX_FORGE_COMPILER_VERSION_BYTES_V1,
        )?;
        validate_bounded_nonempty(
            "target_triple",
            &self.target_triple,
            MAX_FORGE_TARGET_TRIPLE_BYTES_V1,
        )?;
        for (field, commitment) in [
            ("compiler_lineage_commitment", self.compiler_lineage_commitment),
            ("target_cpu_policy_commitment", self.target_cpu_policy_commitment),
            ("wasm_feature_policy_commitment", self.wasm_feature_policy_commitment),
            ("compiler_settings_commitment", self.compiler_settings_commitment),
            ("isolation_profile_commitment", self.isolation_profile_commitment),
        ] {
            if commitment == [0; 32] {
                return Err(ForgeCompilationProfileError::ZeroCommitment(field));
            }
        }
        if self.max_source_module_bytes == 0 {
            return Err(ForgeCompilationProfileError::ZeroLimit(
                "max_source_module_bytes",
            ));
        }
        if self.max_compiled_artifact_bytes == 0 {
            return Err(ForgeCompilationProfileError::ZeroLimit(
                "max_compiled_artifact_bytes",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes_v1(&self) -> Result<Vec<u8>, ForgeCompilationProfileError> {
        self.validate()?;
        let mut out = Vec::with_capacity(
            FORGE_COMPILATION_PROFILE_DOMAIN_V1.len()
                + 2
                + 2 + MAX_FORGE_COMPILATION_PROFILE_ID_BYTES_V1
                + 1
                + 2 + MAX_FORGE_COMPILER_VERSION_BYTES_V1
                + 1
                + 32
                + 2 + MAX_FORGE_TARGET_TRIPLE_BYTES_V1
                + (4 * 32)
                + 10
                + 8,
        );
        out.extend_from_slice(FORGE_COMPILATION_PROFILE_DOMAIN_V1);
        out.extend_from_slice(&self.protocol_version.to_be_bytes());
        append_bounded_utf8(
            &mut out,
            "profile_id",
            &self.profile_id,
            MAX_FORGE_COMPILATION_PROFILE_ID_BYTES_V1,
        )?;
        out.push(runtime_family_tag(self.runtime_family));
        append_bounded_utf8(
            &mut out,
            "runtime_version",
            &self.runtime_version,
            MAX_FORGE_COMPILER_VERSION_BYTES_V1,
        )?;
        out.push(compiler_backend_tag(self.compiler_backend));
        out.extend_from_slice(&self.compiler_lineage_commitment);
        append_bounded_utf8(
            &mut out,
            "target_triple",
            &self.target_triple,
            MAX_FORGE_TARGET_TRIPLE_BYTES_V1,
        )?;
        out.extend_from_slice(&self.target_cpu_policy_commitment);
        out.extend_from_slice(&self.wasm_feature_policy_commitment);
        out.extend_from_slice(&self.compiler_settings_commitment);
        out.extend_from_slice(&self.isolation_profile_commitment);
        out.push(input_format_tag(self.input_format));
        out.push(output_format_tag(self.output_format));
        out.push(optimization_tag(self.optimization));
        out.push(u8::from(self.nan_canonicalization));
        out.push(parallelism_tag(self.parallelism));
        out.push(module_version_policy_tag(self.module_version_policy));
        out.push(import_policy_tag(self.imports));
        out.push(u8::from(self.debug_info));
        out.push(u8::from(self.generate_address_map));
        out.extend_from_slice(&self.max_source_module_bytes.to_be_bytes());
        out.extend_from_slice(&self.max_compiled_artifact_bytes.to_be_bytes());
        Ok(out)
    }

    pub fn canonical_commitment_v1(&self) -> Result<[u8; 32], ForgeCompilationProfileError> {
        Ok(*blake3::hash(&self.canonical_bytes_v1()?).as_bytes())
    }

    pub fn identity_v1(&self) -> Result<ForgeCompilationProfileIdentityV1, ForgeCompilationProfileError> {
        let commitment = self.canonical_commitment_v1()?;
        ForgeCompilationProfileIdentityV1::new(self.profile_id.clone(), commitment)
            .map_err(|error| ForgeCompilationProfileError::Identity(error.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeCompilationProfileError {
    UnsupportedProtocolVersion(u16),
    EmptyField(&'static str),
    FieldTooLong {
        field: &'static str,
        max_bytes: usize,
        actual_bytes: usize,
    },
    ZeroCommitment(&'static str),
    ZeroLimit(&'static str),
    Identity(String),
}

impl fmt::Display for ForgeCompilationProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedProtocolVersion(version) => {
                write!(f, "unsupported Forge compilation-profile version {version}")
            }
            Self::EmptyField(field) => write!(f, "Forge compilation-profile field {field} must be non-empty"),
            Self::FieldTooLong { field, max_bytes, actual_bytes } => write!(
                f,
                "Forge compilation-profile field {field} is {actual_bytes} bytes; v1 maximum is {max_bytes} bytes"
            ),
            Self::ZeroCommitment(field) => write!(f, "Forge compilation-profile commitment {field} may not be all-zero"),
            Self::ZeroLimit(field) => write!(f, "Forge compilation-profile limit {field} must be greater than zero"),
            Self::Identity(error) => write!(f, "invalid compilation-profile identity: {error}"),
        }
    }
}

impl Error for ForgeCompilationProfileError {}

fn validate_bounded_nonempty(
    field: &'static str,
    value: &str,
    max_bytes: usize,
) -> Result<(), ForgeCompilationProfileError> {
    if value.trim().is_empty() {
        return Err(ForgeCompilationProfileError::EmptyField(field));
    }
    if value.len() > max_bytes {
        return Err(ForgeCompilationProfileError::FieldTooLong {
            field,
            max_bytes,
            actual_bytes: value.len(),
        });
    }
    Ok(())
}

fn append_bounded_utf8(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
    max_bytes: usize,
) -> Result<(), ForgeCompilationProfileError> {
    validate_bounded_nonempty(field, value, max_bytes)?;
    let len = u16::try_from(value.len()).map_err(|_| ForgeCompilationProfileError::FieldTooLong {
        field,
        max_bytes: max_bytes.min(u16::MAX as usize),
        actual_bytes: value.len(),
    })?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

const fn runtime_family_tag(value: ForgeCompilationRuntimeFamilyV1) -> u8 {
    match value { ForgeCompilationRuntimeFamilyV1::Wasmtime => 1 }
}
const fn compiler_backend_tag(value: ForgeCompilerBackendV1) -> u8 {
    match value { ForgeCompilerBackendV1::Cranelift => 1 }
}
const fn input_format_tag(value: ForgeCompilationInputFormatV1) -> u8 {
    match value { ForgeCompilationInputFormatV1::WasmBinary => 1 }
}
const fn output_format_tag(value: ForgeCompilationOutputFormatV1) -> u8 {
    match value { ForgeCompilationOutputFormatV1::WasmtimeSerializedModule => 1 }
}
const fn optimization_tag(value: ForgeOptimizationLevelV1) -> u8 {
    match value {
        ForgeOptimizationLevelV1::None => 1,
        ForgeOptimizationLevelV1::Speed => 2,
        ForgeOptimizationLevelV1::SpeedAndSize => 3,
    }
}
const fn parallelism_tag(value: ForgeCompilationParallelismV1) -> u8 {
    match value {
        ForgeCompilationParallelismV1::SingleThread => 1,
        ForgeCompilationParallelismV1::Parallel => 2,
    }
}
const fn module_version_policy_tag(value: ForgeModuleVersionPolicyV1) -> u8 {
    match value { ForgeModuleVersionPolicyV1::WasmtimeVersion => 1 }
}
const fn import_policy_tag(value: ForgeImportPolicyV1) -> u8 {
    match value { ForgeImportPolicyV1::NoImports => 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Arbitrary frozen-vector values. These are NOT compiler/resource recommendations.
    fn profile() -> ForgeCompilationProfileDefinitionV1 {
        ForgeCompilationProfileDefinitionV1 {
            protocol_version: FORGE_PROTOCOL_VERSION,
            profile_id: "compile-v1".into(),
            runtime_family: ForgeCompilationRuntimeFamilyV1::Wasmtime,
            runtime_version: "44.0.1".into(),
            compiler_backend: ForgeCompilerBackendV1::Cranelift,
            compiler_lineage_commitment: [0x11; 32],
            target_triple: "x86_64-unknown-linux-gnu".into(),
            target_cpu_policy_commitment: [0x22; 32],
            wasm_feature_policy_commitment: [0x33; 32],
            compiler_settings_commitment: [0x44; 32],
            isolation_profile_commitment: [0x55; 32],
            input_format: ForgeCompilationInputFormatV1::WasmBinary,
            output_format: ForgeCompilationOutputFormatV1::WasmtimeSerializedModule,
            optimization: ForgeOptimizationLevelV1::Speed,
            nan_canonicalization: true,
            parallelism: ForgeCompilationParallelismV1::SingleThread,
            module_version_policy: ForgeModuleVersionPolicyV1::WasmtimeVersion,
            imports: ForgeImportPolicyV1::NoImports,
            debug_info: false,
            generate_address_map: false,
            max_source_module_bytes: 123_456,
            max_compiled_artifact_bytes: 654_321,
        }
    }

    #[test]
    fn profile_identity_is_exact_definition_commitment() {
        let value = profile();
        let identity = value.identity_v1().unwrap();
        assert_eq!(identity.profile_id, value.profile_id);
        assert_eq!(identity.profile_commitment, value.canonical_commitment_v1().unwrap());
    }

    #[test]
    fn every_lineage_commitment_is_identity_bearing() {
        let base = profile().canonical_commitment_v1().unwrap();
        for field in 0..5 {
            let mut value = profile();
            match field {
                0 => value.compiler_lineage_commitment[0] ^= 1,
                1 => value.target_cpu_policy_commitment[0] ^= 1,
                2 => value.wasm_feature_policy_commitment[0] ^= 1,
                3 => value.compiler_settings_commitment[0] ^= 1,
                4 => value.isolation_profile_commitment[0] ^= 1,
                _ => return,
            }
            assert_ne!(value.canonical_commitment_v1().unwrap(), base);
        }
    }

    #[test]
    fn explicit_codegen_semantics_are_identity_bearing() {
        let base = profile().canonical_commitment_v1().unwrap();
        let mut value = profile();
        value.optimization = ForgeOptimizationLevelV1::SpeedAndSize;
        assert_ne!(value.canonical_commitment_v1().unwrap(), base);
        let mut value = profile();
        value.nan_canonicalization = false;
        assert_ne!(value.canonical_commitment_v1().unwrap(), base);
        let mut value = profile();
        value.parallelism = ForgeCompilationParallelismV1::Parallel;
        assert_ne!(value.canonical_commitment_v1().unwrap(), base);
        let mut value = profile();
        value.debug_info = true;
        assert_ne!(value.canonical_commitment_v1().unwrap(), base);
        let mut value = profile();
        value.generate_address_map = true;
        assert_ne!(value.canonical_commitment_v1().unwrap(), base);
    }

    #[test]
    fn target_and_byte_ceilings_are_identity_bearing() {
        let base = profile().canonical_commitment_v1().unwrap();
        let mut value = profile();
        value.target_triple = "aarch64-unknown-linux-gnu".into();
        assert_ne!(value.canonical_commitment_v1().unwrap(), base);
        let mut value = profile();
        value.max_source_module_bytes += 1;
        assert_ne!(value.canonical_commitment_v1().unwrap(), base);
        let mut value = profile();
        value.max_compiled_artifact_bytes += 1;
        assert_ne!(value.canonical_commitment_v1().unwrap(), base);
    }

    #[test]
    fn zero_placeholders_and_limits_fail_closed() {
        let mut value = profile();
        value.compiler_lineage_commitment = [0; 32];
        assert!(value.validate().is_err());
        let mut value = profile();
        value.max_source_module_bytes = 0;
        assert!(value.validate().is_err());
        let mut value = profile();
        value.max_compiled_artifact_bytes = 0;
        assert!(value.validate().is_err());
    }

    #[test]
    fn overlong_or_empty_identifiers_fail_closed() {
        let mut value = profile();
        value.profile_id.clear();
        assert!(value.validate().is_err());
        let mut value = profile();
        value.target_triple = "x".repeat(MAX_FORGE_TARGET_TRIPLE_BYTES_V1 + 1);
        assert!(value.validate().is_err());
    }
}
