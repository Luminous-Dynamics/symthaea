// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical definition of one Forge execution-plane profile.
//!
//! This module defines *execution* semantics only. Raw WebAssembly compilation
//! belongs to a separate, isolated compilation plane. Numeric values here have
//! no recommended defaults: operational limits must come from a separately
//! qualified measurement/benchmark lineage.

use super::forge::{ForgeExecutionProfileIdentityV1, FORGE_PROTOCOL_VERSION};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Maximum UTF-8 byte length of a human-readable profile id.
pub const MAX_FORGE_PROFILE_ID_BYTES_V1: usize = 128;
/// Maximum UTF-8 byte length of an exact runtime version string.
pub const MAX_FORGE_RUNTIME_VERSION_BYTES_V1: usize = 64;
/// Domain separator for canonical execution-profile bytes.
pub const FORGE_PROFILE_DEFINITION_DOMAIN_V1: &[u8] =
    b"symthaea.forge.execution-profile-definition.v1\0";

/// Runtime family used by the v1 Forge execution plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeRuntimeFamilyV1 {
    /// Bytecode Alliance Wasmtime runtime.
    Wasmtime,
}

/// Exact run-time lineage used to interpret a Forge execution profile.
///
/// This does not identify a compiler. A hardened execution plane may use a
/// Wasmtime build with runtime support but no Cranelift/Winch compiler features.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeRuntimeIdentityV1 {
    /// Runtime family.
    pub family: ForgeRuntimeFamilyV1,
    /// Exact runtime version string, e.g. `44.0.1`.
    pub version: String,
    /// Commitment to the exact runtime package/build/toolchain lineage selected
    /// by the qualification capsule. This is identity, not trust/authority.
    pub lineage_commitment: [u8; 32],
}

impl ForgeRuntimeIdentityV1 {
    /// Construct a structurally valid Wasmtime runtime identity.
    pub fn new_wasmtime(
        version: impl Into<String>,
        lineage_commitment: [u8; 32],
    ) -> Result<Self, ForgeProfileError> {
        let value = Self {
            family: ForgeRuntimeFamilyV1::Wasmtime,
            version: version.into(),
            lineage_commitment,
        };
        value.validate()?;
        Ok(value)
    }

    /// Validate the runtime identity without assigning trust to it.
    pub fn validate(&self) -> Result<(), ForgeProfileError> {
        validate_bounded_nonempty(
            "runtime.version",
            &self.version,
            MAX_FORGE_RUNTIME_VERSION_BYTES_V1,
        )?;
        if self.lineage_commitment == [0; 32] {
            return Err(ForgeProfileError::ZeroCommitment("runtime.lineage_commitment"));
        }
        Ok(())
    }
}

/// Verifier ABI supported by this first profile schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeVerifierAbiV1 {
    /// Call exactly one exported `() -> i32` function.
    NoArgsI32,
}

/// Host-import policy for the verifier sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeImportPolicyV1 {
    /// Instantiate with no host imports and no WASI surface.
    NoImports,
}

/// Determinism contract for the verifier runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeDeterminismPolicyV1 {
    /// Deterministic profile: deterministic fuel exhaustion, canonical NaNs,
    /// no nondeterministic imports, no threads/shared-memory effects, and no
    /// nondeterministic relaxed-SIMD semantics. The exact feature configuration
    /// is additionally bound by `wasm_feature_policy_commitment`.
    Strict,
}

/// Execution interruption semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ForgeInterruptionPolicyV1 {
    /// Deterministic Wasmtime fuel exhaustion only. Epoch/wall-clock interruption
    /// is intentionally excluded from v1 and must use a separately versioned profile.
    DeterministicFuel,
}

/// Canonical definition of a Forge v1 execution profile.
///
/// Numeric values are explicit *inputs*. This type has no default constructor so
/// an unqualified value cannot silently become a security policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForgeExecutionProfileDefinitionV1 {
    /// Forge protocol version.
    pub protocol_version: u16,
    /// Human-readable stable profile id.
    pub profile_id: String,
    /// Exact execution-runtime lineage.
    pub runtime: ForgeRuntimeIdentityV1,
    /// Commitment to the exact Wasmtime feature-policy/configuration allowlist.
    /// This prevents newly enabled or otherwise unenumerated features from
    /// preserving profile identity accidentally.
    pub wasm_feature_policy_commitment: [u8; 32],
    /// Verifier ABI.
    pub abi: ForgeVerifierAbiV1,
    /// Host import/WASI policy.
    pub imports: ForgeImportPolicyV1,
    /// Determinism policy.
    pub determinism: ForgeDeterminismPolicyV1,
    /// Deterministic interruption policy.
    pub interruption: ForgeInterruptionPolicyV1,
    /// Fuel granted to one verification execution.
    pub fuel: u64,
    /// Maximum bytes accepted for the exact trusted precompiled Wasmtime artifact
    /// before deserialization. This is NOT a bound on raw-Wasm compilation cost.
    pub max_precompiled_artifact_bytes: u32,
    /// Maximum bytes for each guest linear memory when memories are permitted.
    pub max_linear_memory_bytes: u32,
    /// Maximum number of guest linear memories in the store.
    pub max_memories: u32,
    /// Maximum elements for each guest table when tables are permitted.
    pub max_table_elements: u32,
    /// Maximum number of guest tables in the store.
    pub max_tables: u32,
    /// Maximum number of guest instances in the store.
    pub max_instances: u32,
    /// Maximum Wasm stack bytes configured for the engine.
    pub max_wasm_stack_bytes: u32,
    /// Whether failed memory/table growth traps instead of returning a grow failure.
    pub trap_on_grow_failure: bool,
}

impl ForgeExecutionProfileDefinitionV1 {
    /// Validate structural/profile invariants.
    ///
    /// This proves only that a profile is well-formed. It does not prove that the
    /// numeric values are safe, performant, sufficient, or qualified.
    pub fn validate(&self) -> Result<(), ForgeProfileError> {
        if self.protocol_version != FORGE_PROTOCOL_VERSION {
            return Err(ForgeProfileError::UnsupportedProtocolVersion(
                self.protocol_version,
            ));
        }
        validate_bounded_nonempty(
            "profile_id",
            &self.profile_id,
            MAX_FORGE_PROFILE_ID_BYTES_V1,
        )?;
        self.runtime.validate()?;
        if self.wasm_feature_policy_commitment == [0; 32] {
            return Err(ForgeProfileError::ZeroCommitment(
                "wasm_feature_policy_commitment",
            ));
        }
        require_nonzero_u64("fuel", self.fuel)?;
        require_nonzero_u32(
            "max_precompiled_artifact_bytes",
            self.max_precompiled_artifact_bytes,
        )?;
        require_nonzero_u32("max_instances", self.max_instances)?;
        require_nonzero_u32("max_wasm_stack_bytes", self.max_wasm_stack_bytes)?;

        validate_count_size_pair(
            "max_memories",
            self.max_memories,
            "max_linear_memory_bytes",
            self.max_linear_memory_bytes,
        )?;
        validate_count_size_pair(
            "max_tables",
            self.max_tables,
            "max_table_elements",
            self.max_table_elements,
        )?;
        Ok(())
    }

    /// Frozen representation-independent v1 encoding.
    pub fn canonical_bytes_v1(&self) -> Result<Vec<u8>, ForgeProfileError> {
        self.validate()?;

        let mut out = Vec::with_capacity(
            FORGE_PROFILE_DEFINITION_DOMAIN_V1.len()
                + 2
                + 2
                + MAX_FORGE_PROFILE_ID_BYTES_V1
                + 1
                + 2
                + MAX_FORGE_RUNTIME_VERSION_BYTES_V1
                + 32
                + 32
                + 4
                + 8
                + (7 * 4)
                + 1,
        );
        out.extend_from_slice(FORGE_PROFILE_DEFINITION_DOMAIN_V1);
        out.extend_from_slice(&self.protocol_version.to_be_bytes());
        append_bounded_utf8(
            &mut out,
            "profile_id",
            &self.profile_id,
            MAX_FORGE_PROFILE_ID_BYTES_V1,
        )?;
        out.push(runtime_family_tag(self.runtime.family));
        append_bounded_utf8(
            &mut out,
            "runtime.version",
            &self.runtime.version,
            MAX_FORGE_RUNTIME_VERSION_BYTES_V1,
        )?;
        out.extend_from_slice(&self.runtime.lineage_commitment);
        out.extend_from_slice(&self.wasm_feature_policy_commitment);
        out.push(verifier_abi_tag(self.abi));
        out.push(import_policy_tag(self.imports));
        out.push(determinism_policy_tag(self.determinism));
        out.push(interruption_policy_tag(self.interruption));
        out.extend_from_slice(&self.fuel.to_be_bytes());
        out.extend_from_slice(&self.max_precompiled_artifact_bytes.to_be_bytes());
        out.extend_from_slice(&self.max_linear_memory_bytes.to_be_bytes());
        out.extend_from_slice(&self.max_memories.to_be_bytes());
        out.extend_from_slice(&self.max_table_elements.to_be_bytes());
        out.extend_from_slice(&self.max_tables.to_be_bytes());
        out.extend_from_slice(&self.max_instances.to_be_bytes());
        out.extend_from_slice(&self.max_wasm_stack_bytes.to_be_bytes());
        out.push(u8::from(self.trap_on_grow_failure));
        Ok(out)
    }

    /// BLAKE3-256 commitment to the exact canonical execution-profile definition.
    pub fn canonical_commitment_v1(&self) -> Result<[u8; 32], ForgeProfileError> {
        Ok(*blake3::hash(&self.canonical_bytes_v1()?).as_bytes())
    }

    /// Convert this exact definition into the profile identity carried by Forge
    /// requests/scopes/evidence.
    pub fn identity_v1(&self) -> Result<ForgeExecutionProfileIdentityV1, ForgeProfileError> {
        let commitment = self.canonical_commitment_v1()?;
        ForgeExecutionProfileIdentityV1::new(self.profile_id.clone(), commitment)
            .map_err(|error| ForgeProfileError::Identity(error.to_string()))
    }
}

/// Structural/canonicalization errors for Forge execution profiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeProfileError {
    UnsupportedProtocolVersion(u16),
    EmptyField(&'static str),
    FieldTooLong {
        field: &'static str,
        max_bytes: usize,
        actual_bytes: usize,
    },
    ZeroCommitment(&'static str),
    ZeroLimit(&'static str),
    ZeroWideLimit(&'static str),
    InconsistentResourceLimit {
        count_field: &'static str,
        size_field: &'static str,
    },
    Identity(String),
}

impl fmt::Display for ForgeProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedProtocolVersion(version) => {
                write!(f, "unsupported Forge profile protocol version {version}")
            }
            Self::EmptyField(field) => write!(f, "Forge profile field {field} must be non-empty"),
            Self::FieldTooLong {
                field,
                max_bytes,
                actual_bytes,
            } => write!(
                f,
                "Forge profile field {field} is {actual_bytes} bytes; v1 maximum is {max_bytes}"
            ),
            Self::ZeroCommitment(field) => {
                write!(f, "Forge profile commitment field {field} may not be all-zero")
            }
            Self::ZeroLimit(field) | Self::ZeroWideLimit(field) => {
                write!(f, "Forge profile limit {field} must be greater than zero")
            }
            Self::InconsistentResourceLimit {
                count_field,
                size_field,
            } => write!(
                f,
                "Forge profile resource limits {count_field} and {size_field} are inconsistent"
            ),
            Self::Identity(message) => write!(f, "invalid Forge profile identity: {message}"),
        }
    }
}

impl Error for ForgeProfileError {}

fn validate_bounded_nonempty(
    field: &'static str,
    value: &str,
    max_bytes: usize,
) -> Result<(), ForgeProfileError> {
    if value.trim().is_empty() {
        return Err(ForgeProfileError::EmptyField(field));
    }
    if value.len() > max_bytes {
        return Err(ForgeProfileError::FieldTooLong {
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
) -> Result<(), ForgeProfileError> {
    validate_bounded_nonempty(field, value, max_bytes)?;
    let len = u16::try_from(value.len()).map_err(|_| ForgeProfileError::FieldTooLong {
        field,
        max_bytes: max_bytes.min(u16::MAX as usize),
        actual_bytes: value.len(),
    })?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn require_nonzero_u32(field: &'static str, value: u32) -> Result<(), ForgeProfileError> {
    if value == 0 {
        Err(ForgeProfileError::ZeroLimit(field))
    } else {
        Ok(())
    }
}

fn require_nonzero_u64(field: &'static str, value: u64) -> Result<(), ForgeProfileError> {
    if value == 0 {
        Err(ForgeProfileError::ZeroWideLimit(field))
    } else {
        Ok(())
    }
}

fn validate_count_size_pair(
    count_field: &'static str,
    count: u32,
    size_field: &'static str,
    size: u32,
) -> Result<(), ForgeProfileError> {
    if (count == 0) != (size == 0) {
        Err(ForgeProfileError::InconsistentResourceLimit {
            count_field,
            size_field,
        })
    } else {
        Ok(())
    }
}

fn runtime_family_tag(value: ForgeRuntimeFamilyV1) -> u8 {
    match value {
        ForgeRuntimeFamilyV1::Wasmtime => 1,
    }
}

fn verifier_abi_tag(value: ForgeVerifierAbiV1) -> u8 {
    match value {
        ForgeVerifierAbiV1::NoArgsI32 => 1,
    }
}

fn import_policy_tag(value: ForgeImportPolicyV1) -> u8 {
    match value {
        ForgeImportPolicyV1::NoImports => 1,
    }
}

fn determinism_policy_tag(value: ForgeDeterminismPolicyV1) -> u8 {
    match value {
        ForgeDeterminismPolicyV1::Strict => 1,
    }
}

fn interruption_policy_tag(value: ForgeInterruptionPolicyV1) -> u8 {
    match value {
        ForgeInterruptionPolicyV1::DeterministicFuel => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Arbitrary frozen test-vector values. These are NOT operational/security recommendations.
    fn profile() -> ForgeExecutionProfileDefinitionV1 {
        ForgeExecutionProfileDefinitionV1 {
            protocol_version: FORGE_PROTOCOL_VERSION,
            profile_id: "wasmtime-fuel-v1".into(),
            runtime: ForgeRuntimeIdentityV1::new_wasmtime("44.0.1", [0xA1; 32]).unwrap(),
            wasm_feature_policy_commitment: [0xB2; 32],
            abi: ForgeVerifierAbiV1::NoArgsI32,
            imports: ForgeImportPolicyV1::NoImports,
            determinism: ForgeDeterminismPolicyV1::Strict,
            interruption: ForgeInterruptionPolicyV1::DeterministicFuel,
            fuel: 123_456,
            max_precompiled_artifact_bytes: 1_234_567,
            max_linear_memory_bytes: 2_000_000,
            max_memories: 1,
            max_table_elements: 512,
            max_tables: 1,
            max_instances: 1,
            max_wasm_stack_bytes: 262_144,
            trap_on_grow_failure: true,
        }
    }

    fn frozen_bytes() -> Vec<u8> {
        let mut expected = FORGE_PROFILE_DEFINITION_DOMAIN_V1.to_vec();
        expected.extend_from_slice(&FORGE_PROTOCOL_VERSION.to_be_bytes());
        expected.extend_from_slice(&16u16.to_be_bytes());
        expected.extend_from_slice(b"wasmtime-fuel-v1");
        expected.push(1); // Wasmtime
        expected.extend_from_slice(&6u16.to_be_bytes());
        expected.extend_from_slice(b"44.0.1");
        expected.extend_from_slice(&[0xA1; 32]);
        expected.extend_from_slice(&[0xB2; 32]);
        expected.push(1); // NoArgsI32
        expected.push(1); // NoImports
        expected.push(1); // Strict determinism
        expected.push(1); // DeterministicFuel
        expected.extend_from_slice(&123_456u64.to_be_bytes());
        expected.extend_from_slice(&1_234_567u32.to_be_bytes());
        expected.extend_from_slice(&2_000_000u32.to_be_bytes());
        expected.extend_from_slice(&1u32.to_be_bytes());
        expected.extend_from_slice(&512u32.to_be_bytes());
        expected.extend_from_slice(&1u32.to_be_bytes());
        expected.extend_from_slice(&1u32.to_be_bytes());
        expected.extend_from_slice(&262_144u32.to_be_bytes());
        expected.push(1);
        expected
    }

    #[test]
    fn profile_vector_is_frozen_and_representation_independent() {
        let value = profile();
        assert_eq!(value.canonical_bytes_v1().unwrap(), frozen_bytes());
        assert_eq!(
            value.canonical_commitment_v1().unwrap(),
            *blake3::hash(&frozen_bytes()).as_bytes()
        );
    }

    #[test]
    fn profile_identity_uses_exact_definition_commitment() {
        let value = profile();
        let identity = value.identity_v1().unwrap();
        assert_eq!(identity.profile_id, value.profile_id);
        assert_eq!(
            identity.profile_commitment,
            value.canonical_commitment_v1().unwrap()
        );
    }

    #[test]
    fn runtime_and_feature_lineage_change_profile_identity() {
        let base = profile().canonical_commitment_v1().unwrap();

        let mut runtime = profile();
        runtime.runtime.lineage_commitment[0] ^= 0x01;
        assert_ne!(runtime.canonical_commitment_v1().unwrap(), base);

        let mut features = profile();
        features.wasm_feature_policy_commitment[0] ^= 0x01;
        assert_ne!(features.canonical_commitment_v1().unwrap(), base);

        let mut version = profile();
        version.runtime.version = "44.0.2".into();
        assert_ne!(version.canonical_commitment_v1().unwrap(), base);
    }

    #[test]
    fn every_explicit_numeric_limit_is_identity_bearing() {
        let base = profile().canonical_commitment_v1().unwrap();

        let mut fuel = profile();
        fuel.fuel += 1;
        assert_ne!(fuel.canonical_commitment_v1().unwrap(), base);

        let mut artifact = profile();
        artifact.max_precompiled_artifact_bytes += 1;
        assert_ne!(artifact.canonical_commitment_v1().unwrap(), base);

        let mut memory = profile();
        memory.max_linear_memory_bytes += 1;
        assert_ne!(memory.canonical_commitment_v1().unwrap(), base);

        let mut memories = profile();
        memories.max_memories += 1;
        assert_ne!(memories.canonical_commitment_v1().unwrap(), base);

        let mut elements = profile();
        elements.max_table_elements += 1;
        assert_ne!(elements.canonical_commitment_v1().unwrap(), base);

        let mut tables = profile();
        tables.max_tables += 1;
        assert_ne!(tables.canonical_commitment_v1().unwrap(), base);

        let mut instances = profile();
        instances.max_instances += 1;
        assert_ne!(instances.canonical_commitment_v1().unwrap(), base);

        let mut stack = profile();
        stack.max_wasm_stack_bytes += 1;
        assert_ne!(stack.canonical_commitment_v1().unwrap(), base);

        let mut grow = profile();
        grow.trap_on_grow_failure = false;
        assert_ne!(grow.canonical_commitment_v1().unwrap(), base);
    }

    #[test]
    fn placeholder_commitments_fail_closed() {
        let mut runtime = profile();
        runtime.runtime.lineage_commitment = [0; 32];
        assert!(matches!(
            runtime.validate(),
            Err(ForgeProfileError::ZeroCommitment("runtime.lineage_commitment"))
        ));

        let mut features = profile();
        features.wasm_feature_policy_commitment = [0; 32];
        assert!(matches!(
            features.validate(),
            Err(ForgeProfileError::ZeroCommitment(
                "wasm_feature_policy_commitment"
            ))
        ));
    }

    #[test]
    fn resource_count_and_size_must_agree() {
        let mut no_memories = profile();
        no_memories.max_memories = 0;
        assert!(matches!(
            no_memories.validate(),
            Err(ForgeProfileError::InconsistentResourceLimit {
                count_field: "max_memories",
                ..
            })
        ));

        let mut no_tables = profile();
        no_tables.max_tables = 0;
        assert!(matches!(
            no_tables.validate(),
            Err(ForgeProfileError::InconsistentResourceLimit {
                count_field: "max_tables",
                ..
            })
        ));

        let mut memory_free = profile();
        memory_free.max_memories = 0;
        memory_free.max_linear_memory_bytes = 0;
        assert!(memory_free.validate().is_ok());

        let mut table_free = profile();
        table_free.max_tables = 0;
        table_free.max_table_elements = 0;
        assert!(table_free.validate().is_ok());
    }

    #[test]
    fn required_zero_budgets_fail_closed() {
        let mut fuel = profile();
        fuel.fuel = 0;
        assert!(fuel.validate().is_err());

        let mut artifact = profile();
        artifact.max_precompiled_artifact_bytes = 0;
        assert!(artifact.validate().is_err());

        let mut instances = profile();
        instances.max_instances = 0;
        assert!(instances.validate().is_err());

        let mut stack = profile();
        stack.max_wasm_stack_bytes = 0;
        assert!(stack.validate().is_err());
    }
}
